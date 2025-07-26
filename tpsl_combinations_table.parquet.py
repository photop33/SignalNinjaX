import pandas as pd
import os
from tqdm import tqdm

COMMON_FILE = "common_table.parquet"

def load_candles_1h(symbol, start_time, end_time):
    base_dir = r"C:\Users\LiorSw\data\historical_kline"
    interval = "1h"
    dfs = []
    current = start_time.date()
    last = end_time.date()
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
        df_all = df_all.sort_values('time').reset_index(drop=True)
        return df_all
    else:
        return pd.DataFrame()

def find_tp_sl(row, candles):
    TP, SL = row["TP"], row["SL"]
    entry_price = row["close"]
    entry_time = row["time"]
    start_check_time = entry_time + pd.Timedelta(hours=4)
    relevant = candles[candles["time"] >= start_check_time]

    if relevant.empty or pd.isna(TP) or pd.isna(SL) or pd.isna(entry_price):
        return "No Data", pd.NaT, None

    for _, candle in relevant.iterrows():
        hit_tp = candle['high'] >= TP
        hit_sl = candle['low'] <= SL
        if hit_tp and hit_sl:
            if abs(TP - entry_price) < abs(entry_price - SL):
                return "TP Hit", candle["time"], 100 * (TP - entry_price) / entry_price
            else:
                return "SL Hit", candle["time"], 100 * (SL - entry_price) / entry_price
        elif hit_tp:
            return "TP Hit", candle["time"], 100 * (TP - entry_price) / entry_price
        elif hit_sl:
            return "SL Hit", candle["time"], 100 * (SL - entry_price) / entry_price

    return "Still Open", pd.NaT, None

def main():
    print("📥 טוען common_table.parquet...")
    df = pd.read_parquet(COMMON_FILE)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # ✨ ודא שהעמודות הדרושות קיימות
    for col in ["result", "cross_line_time", "PnL_%"]:
        if col not in df.columns:
            df[col] = pd.NA

    print("🔍 סינון רק שורות שעדיין לא חושבו...")
    df_to_process = df[df["cross_line_time"].isna()].copy()
    if df_to_process.empty:
        print("⛔ אין שורות חדשות לחישוב.")
        return

    symbols = df_to_process["symbol"].dropna().unique()
    min_time = df_to_process["time"].min()
    max_time = df_to_process["time"].max()

    print("📦 טוען קנדלים 1H...")
    candles_dict = {
        sym: load_candles_1h(sym, min_time - pd.Timedelta(days=1), max_time + pd.Timedelta(days=10))
        for sym in tqdm(symbols, desc="טעינת נרות")
    }

    print("🚀 מחשב TP/SL לכל שורה...")
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="חישוב"):
        symbol = row["symbol"]
        candles = candles_dict.get(symbol)
        if candles is None or candles.empty:
            continue

        result, cross_time, pnl = find_tp_sl(row, candles)
        df.loc[df["common_id"] == row["common_id"], ["result", "cross_line_time", "PnL_%"]] = result, cross_time, pnl

    print("💾 שומר קובץ מעודכן...")
    df.to_parquet(COMMON_FILE, index=False)
    print("✅ סיום.")

if __name__ == "__main__":
    main()
