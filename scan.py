from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import requests
from datetime import datetime, timedelta
from pytz import timezone
import os
import pandas as pd
from datetime import timedelta
from indector import add_indicators
from config import symbol_scan
from config import  PCT_CHANGE_THRESHOLD, FILTER_MODE
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from config import SCAN_FROM, SCAN_TO, INTERVAL, VOLATILITY_THRESHOLD, MIN_VOLUME
print("Start Scan ..")
tz = timezone("Asia/Jerusalem")
BINANCE_API = "https://api.binance.com"
MIN_CANDLES = 30

def to_ms(dt_str): return int(datetime.strptime(dt_str, "%Y-%m-%d %H:%M").timestamp() * 1000)
def fetch_candles(symbol, interval, start, end=None, limit=1000):
    # תחזיר את כל הדאטה בין start ל-end, גם אם יש יותר מ-1000 נרות
    all_data = []
    start_time = to_ms(start)
    end_time = to_ms(end) if end else None

    while True:
        params = {
            "symbol": symbol,
            "interval": interval,
            "startTime": start_time,
            "limit": limit
        }
        if end_time:
            params["endTime"] = end_time

        r = requests.get(f"{BINANCE_API}/api/v3/klines", params=params).json()
        if not isinstance(r, list) or len(r) == 0:
            break

        all_data.extend(r)
        # אם קיבלנו פחות מה-limit, נגמר
        if len(r) < limit:
            break

        # עדכן את זמן ההתחלה לנר הבא
        last_time = r[-1][0]
        # תזוזה של דקה קדימה (או אינטרוול רלוונטי)
        start_time = last_time + 60*1000  # 60 שניות * 1000 = מ"ש

        # בטיחות: אם עברנו את end_time — עצור
        if end_time and start_time >= end_time:
            break

    if not all_data:
        return None

    df = pd.DataFrame(all_data)[[0,1,2,3,4,5]]
    df.columns = ["time", "open", "high", "low", "close", "volume"]
    df["time"] = pd.to_datetime(df["time"], unit="ms").dt.tz_localize("UTC").dt.tz_convert(tz).dt.tz_localize(None)
    df = df.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})
    return df

def get_binance_symbols():
    data = requests.get(f"{BINANCE_API}/api/v3/exchangeInfo").json()
    quotes = ("USDT", "USD", "BUSD", "TUSD", "FDUSD", "BTC", "ETH")
    return [
        s["symbol"] for s in data["symbols"]
        if s["isSpotTradingAllowed"] and s["status"] == "TRADING"
        and any(s["symbol"].endswith(q) for q in quotes)
        and not any(x in s["baseAsset"] for x in ["UP", "DOWN", "BEAR", "BULL"])
    ]

def is_volatile(symbol):
    try:
        df = fetch_candles(symbol, INTERVAL, SCAN_FROM, SCAN_TO)
        if df is None or df.empty:
            return None

        # תנודתיות
        if VOLATILITY_THRESHOLD and VOLATILITY_THRESHOLD > 0.0:
            df["volatility"] = (df["high"] - df["low"]) / df["open"]
            filter_volatility = (df["volatility"] >= VOLATILITY_THRESHOLD)
        else:
            filter_volatility = pd.Series([False]*len(df))

        # אחוז שינוי
        if PCT_CHANGE_THRESHOLD and PCT_CHANGE_THRESHOLD > 0.0:
            df["pct_change"] = (df["close"] - df["open"]) / df["open"]
            filter_pct_change = (df["pct_change"] >= PCT_CHANGE_THRESHOLD)
        else:
            filter_pct_change = pd.Series([False]*len(df))

        filter_volume = (df["volume"] >= MIN_VOLUME)

        conditions = []
        if VOLATILITY_THRESHOLD and VOLATILITY_THRESHOLD > 0.0:
            conditions.append(filter_volatility & filter_volume)
        if PCT_CHANGE_THRESHOLD and PCT_CHANGE_THRESHOLD > 0.0:
            conditions.append(filter_pct_change & filter_volume)

        if not conditions:
            return symbol

        if FILTER_MODE.upper() == "AND" and len(conditions) > 1:
            condition = conditions[0]
            for cond in conditions[1:]:
                condition &= cond
        else:
            condition = conditions[0]
            for cond in conditions[1:]:
                condition |= cond

        if condition.any():
            return symbol
        return None

    except Exception as e:
        print(f"⚠️ {symbol} is_volatile error: {e}")
        return None


def get_expected_candles(scan_from, scan_to, interval):
    delta = pd.to_datetime(scan_to) - pd.to_datetime(scan_from)
    interval_map = {
        "1m": 60,
        "3m": 3 * 60,
        "5m": 5 * 60,
        "15m": 15 * 60,
        "30m": 30 * 60,
        "1h": 60 * 60,
        "2h": 2 * 60 * 60,
        "4h": 4 * 60 * 60,
        "6h": 6 * 60 * 60,
        "8h": 8 * 60 * 60,
        "12h": 12 * 60 * 60,
        "1d": 24 * 60 * 60,
    }
    seconds = delta.total_seconds()
    interval_sec = interval_map.get(interval)
    if not interval_sec:
        raise ValueError(f"אינטרוול לא נתמך: {interval}")
    # +1 כדי לכלול גם את הקצה הימני (למשל 00:00 עד 23:59 אמור להיות 24 נרות של שעה)
    expected = int(seconds // interval_sec) + 1
    return expected


def ensure_minimum_candles(df, symbol, min_bars_needed):
    # אם יש פחות מהנדרש — תמשוך מההתחלה יותר אחורה
    if df is None or len(df) < min_bars_needed:
        # חישוב כמות דקות/שעות/ימים אחורה לפי האינטרוול
        interval_map = {
            "1m": 1,
            "3m": 3,
            "5m": 5,
            "15m": 15,
            "30m": 30,
            "1h": 60,
            "2h": 120,
            "4h": 240,
            "6h": 360,
            "8h": 480,
            "12h": 720,
            "1d": 1440,
            "1w": 10080,  # 7*24*60
            "1y": 525600
        }
        interval_minutes = interval_map.get(INTERVAL, 1)
        interval_key = INTERVAL.strip().lower()
        if interval_key not in interval_map:
            raise ValueError(f"INTERVAL לא נתמך: {INTERVAL}")
        interval_minutes = interval_map[interval_key]
        total_minutes = min_bars_needed * interval_minutes

        new_start = (pd.to_datetime(SCAN_FROM) - timedelta(minutes=total_minutes)).strftime("%Y-%m-%d %H:%M")
        print(f"🔄 מושך עוד דאטה: {symbol} מ־{new_start} עד {SCAN_TO} כדי להגיע לפחות {min_bars_needed} נרות")
        return fetch_candles(symbol, INTERVAL, new_start, SCAN_TO)
    return df


def download_data(symbol):
    # --- הגדרת חלונות לאינדיקטורים ---
    MACD_SLOW = 26
    MACD_SIGNAL = 9
    ADX_WINDOW = 14
    EMA_LONG = 25
    BB_WINDOW = 20
    TRIX_WINDOW = 15
    CCI_WINDOW = 9
    WMA_WINDOW = 14

    max_window = max(
        MACD_SLOW + MACD_SIGNAL,   # MACD slow+signal
        ADX_WINDOW * 2,            # ADX בד"כ צריך פעמיים
        EMA_LONG,
        BB_WINDOW,
        TRIX_WINDOW,
        CCI_WINDOW,
        WMA_WINDOW
    )
    interval_map = {
        "1m": 1,
        "3m": 3,
        "5m": 5,
        "15m": 15,
        "30m": 30,
        "1h": 60,
        "2h": 120,
        "4h": 240,
        "6h": 360,
        "8h": 480,
        "12h": 720,
        "1d": 1440,
        "1w": 10080,
    }

    interval_minutes = interval_map[INTERVAL]
    back_minutes = (max_window - 1) * interval_minutes

    scan_from_dt = pd.to_datetime(SCAN_FROM)
    true_start = (scan_from_dt - timedelta(minutes=back_minutes)).strftime("%Y-%m-%d %H:%M")

    # --- משיכת נתונים ---
    df = fetch_candles(symbol, INTERVAL, true_start, SCAN_TO)
    if df is None or df.empty:
        print(f"⚠️ {symbol}: אין נתונים בכלל")
        return None

    # --- חישוב אינדיקטורים ---
    df = add_indicators(df)

    # --- חיתוך רק לטווח המבוקש ---
    df = df[(df["time"] >= pd.to_datetime(SCAN_FROM)) & (df["time"] <= pd.to_datetime(SCAN_TO))]
    if df.empty:
        print(f"⚠️ {symbol}: אין נתונים בטווח המבוקש — לא יישמר קובץ")
        return None

    # --- שמירה לפי ימים, רק אם day_df לא ריק ---
    for single_date in df["time"].dt.date.unique():
        day_df = df[df["time"].dt.date == single_date]
        if day_df.empty:
            continue  # דלג על יום בלי נתונים!
        year = single_date.year
        month = f"{single_date.month:02d}"
        day = f"{single_date.day:02d}"
        base_dir = f"C:\\Users\\LiorSw\\data\\historical_kline\\{symbol}\\{year}\\{month}\\{day}\\{INTERVAL}"
        os.makedirs(base_dir, exist_ok=True)
        filename = f"{symbol}_{year}-{month}-{day}_{INTERVAL}.csv"
        full_path = os.path.join(base_dir, filename)

        # שמור רק אם יש שורות רלוונטיות!
        if os.path.exists(full_path):
            old_df = pd.read_csv(full_path, parse_dates=["time"])
            merged = pd.concat([old_df, day_df])
            merged = merged.drop_duplicates(subset=["time"]).sort_values(by="time")
            if not merged.empty:
                merged.to_csv(full_path, index=False, encoding="utf-8-sig")
        else:
            if not day_df.empty:
                day_df.to_csv(full_path, index=False, encoding="utf-8-sig")

    return symbol

def main():
    if symbol_scan:
        symbols = symbol_scan
    else:
        symbols = get_binance_symbols()
    print(f"🔍 סורק {len(symbols)} מטבעות...")

    with ThreadPoolExecutor(max_workers=30) as ex:
        volatiles = list(filter(None, tqdm(
            [f.result() for f in as_completed({ex.submit(is_volatile, s): s for s in symbols})],
            desc="📊 תנודתיים")))

    print(f"\n✅ נמצאו {len(volatiles)} מטבעות תנודתיים.")
    # ... ושאר הקוד שלך כרגיל

    print("🪙", ", ".join(volatiles))



    with ThreadPoolExecutor(max_workers=10) as ex:
        list(tqdm(
            as_completed([ex.submit(download_data, s) for s in volatiles]),
            total=len(volatiles),
            desc="⬇️ שמירה"))

    print("🎯 הסתיים בהצלחה.")
if __name__ == "__main__":
    main()