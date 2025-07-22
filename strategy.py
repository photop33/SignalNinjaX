import os
import pandas as pd
import numpy as np
import ta
from itertools import product
from tqdm import tqdm
from config import INDICATOR_CONDITIONS, RUN_PARAMS, interval, symbol, start_time_str, end_time_str

ind_params = INDICATOR_CONDITIONS["strategy_adx_ema_breakout_atr"]
symbols = symbol if isinstance(symbol, list) else [symbol]
interval = interval
start_time = pd.to_datetime(start_time_str)
end_time = pd.to_datetime(end_time_str)


# === טעינה ===
def load_all_candles(symbol, interval, start, end):
    base_dir = r"C:\Users\LiorSw\data\historical_kline"
    dfs = []
    current = start.date()
    last = end.date()
    while current <= last:
        year = str(current.year)
        month = f"{current.month:02d}"
        day = f"{current.day:02d}"
        path = os.path.join(base_dir, symbol, year, month, day, interval)
        fname = f"{symbol}_{year}-{month}-{day}_{interval}.csv"
        full_path = os.path.join(path, fname)
        if os.path.exists(full_path):
            df = pd.read_csv(full_path)
            if 'time' in df.columns:
                df["time"] = pd.to_datetime(df["time"], errors='coerce')
            dfs.append(df)
        current += pd.Timedelta(days=1)
    if dfs:
        df_all = pd.concat(dfs, ignore_index=True)
        df_all = df_all[(df_all["time"] >= start) & (df_all["time"] <= end)]
        return df_all
    else:
        return pd.DataFrame()


df = load_all_candles(symbols[0], interval, start_time, end_time)
if df.empty:
    print("אין דאטה!")
    exit()

# === חישוב אינדיקטורים מראש ===
for w in ind_params["ema_short_window"] + ind_params["ema_long_window"]:
    df[f'ema_{w}'] = df['close'].ewm(span=w).mean()
for w in ind_params["atr_window"]:
    df[f'atr_{w}'] = ta.volatility.AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=w
    ).average_true_range()
for w in ind_params["ADX_min"]:
    df[f'adx_{w}'] = ta.trend.ADXIndicator(
        high=df["high"], low=df["low"], close=df["close"], window=w
    ).adx()
for w in ind_params["volume_window"]:
    df[f'vol_ma_{w}'] = df["volume"].rolling(window=w).mean()
for w in ind_params["lookback_breakout"]:
    df[f'high_prev_{w}'] = df['high'].shift(1).rolling(window=w).max()  # כפי שבפונקציה

# === יצירת כל הקומבינציות (כולל RUN_PARAMS) ===
param_names = list(ind_params.keys())
param_values = [ind_params[k] for k in param_names]
combos = list(product(*param_values))

all_run_combos = []
for run_cfg in RUN_PARAMS:
    for combo in combos:
        all_run_combos.append({
            **dict(zip(param_names, combo)),
            **run_cfg
        })

# === הפעלת האסטרטגיה על כל נר-קומבינציה (וקטוריזציה מלאה) ===
result_rows = []

for combo in tqdm(all_run_combos, desc="וקטוריזציה"):
    ema_short = combo["ema_short_window"]
    ema_long = combo["ema_long_window"]
    adx_min = combo["ADX_min"]
    atr_window = combo["atr_window"]
    atr_mult = combo["atr_mult"]
    rr_ratio = combo["rr_ratio"]
    volume_window = combo["volume_window"]
    lookback_breakout = combo["lookback_breakout"]

    # עמודות מוכנות
    ema_short_col = f'ema_{ema_short}'
    ema_long_col = f'ema_{ema_long}'
    adx_col = f'adx_{adx_min}'
    atr_col = f'atr_{atr_window}'
    vol_ma_col = f'vol_ma_{volume_window}'
    high_prev_col = f'high_prev_{lookback_breakout}'

    dft = df.copy()
    dft["score"] = 0
    dft["reasons"] = ""
    dft["ema_short"] = dft[ema_short_col]
    dft["ema_long"] = dft[ema_long_col]
    dft["ADX"] = dft[adx_col]
    dft["atr"] = dft[atr_col]
    dft["volume_ma"] = dft[vol_ma_col]
    dft["high_prev"] = dft[high_prev_col]
    dft["SL"] = np.nan
    dft["TP"] = np.nan

    # תנאים (וקטורית):
    # 1. מחיר מעל EMA-long
    cond1 = dft["close"] > dft["ema_long"]
    dft.loc[cond1, "score"] += 1
    dft.loc[cond1, "reasons"] += "מחיר מעל EMA-long;"

    # 2. EMA-short > EMA-long
    cond2 = dft["ema_short"] > dft["ema_long"]
    dft.loc[cond2, "score"] += 1
    dft.loc[cond2, "reasons"] += "EMA-short > EMA-long (מגמת עלייה);"

    # 3. ADX > ADX_min
    cond3 = dft["ADX"] > adx_min
    dft.loc[cond3, "score"] += 1
    dft.loc[cond3, "reasons"] += f"ADX > {adx_min};"

    # 4. volume > avg_volume
    cond4 = dft["volume"] > dft["volume_ma"]
    dft.loc[cond4, "score"] += 1
    dft.loc[cond4, "reasons"] += "נפח מסחר גבוה מהממוצע;"

    # 5. Breakout: close > high_prev
    cond5 = dft["close"] > dft["high_prev"]
    dft.loc[cond5, "score"] += 2
    dft.loc[cond5, "reasons"] += f"Breakout: סגירה מעל High-{lookback_breakout};"

    # 6. ATR עבור SL/TP
    cond6 = dft["atr"] > 0
    dft.loc[cond6, "SL"] = dft["close"] - atr_mult * dft["atr"]
    dft.loc[cond6, "TP"] = dft["close"] + atr_mult * rr_ratio * dft["atr"]
    dft.loc[cond6, "reasons"] += f"סטופ וטייק חושבו;"

    # הוספת פרמטרי הקומבו
    for k, v in combo.items():
        dft[k] = v

    # רק נרות עם מספיק דאטה
    vals = [ema_long, atr_window, volume_window, lookback_breakout]
    min_len = max(vals) + 2
    dft = dft.iloc[min_len:]

    # ציון מעל הסף (אפשר להוסיף פה STRATEGY_THRESHOLDS)
    dft_signals = dft[dft["score"] >= 6]  # סף סיגנל, ניתן להחליף
    result_rows.append(dft_signals)

df_results = pd.concat(result_rows, ignore_index=True)
df_results.to_csv("vectorized_signals_full.csv", index=False, encoding="utf-8-sig")
print("✅ וקטוריזציה מלאה הסתיימה! נשמר: vectorized_signals_full.csv")
