import pandas as pd
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import multiprocessing

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

def find_tp_sl_row(row, candles_1h_dict, tp_sl_cache):
    symbol = row["symbol"] if "symbol" in row else "BTCUSDT"
    entry_time = row['time']
    TP = row['TP']
    SL = row['SL']
    entry_price = row['close']
    cache_key = (str(entry_time), symbol)
    if cache_key in tp_sl_cache:
        return tp_sl_cache[cache_key]

    if pd.isna(TP) or pd.isna(SL) or pd.isna(entry_time):
        result = {"result": "No TP/SL", "cross_line_time": pd.NaT, "PnL_%": None}
        tp_sl_cache[cache_key] = result
        return result
    candles = candles_1h_dict[symbol]
    if candles.empty:
        result = {"result": "No Data", "cross_line_time": pd.NaT, "PnL_%": None}
        tp_sl_cache[cache_key] = result
        return result

    relevant = candles[candles['time'] > entry_time]
    for idx, candle in relevant.iterrows():
        hit_tp = candle['high'] >= TP
        hit_sl = candle['low'] <= SL
        if hit_tp and hit_sl:
            if abs(TP - entry_price) < abs(entry_price - SL):
                res = {"result": "TP Hit", "cross_line_time": candle['time'], "PnL_%": 100 * (TP - entry_price) / entry_price}
            else:
                res = {"result": "SL Hit", "cross_line_time": candle['time'], "PnL_%": 100 * (SL - entry_price) / entry_price}
            tp_sl_cache[cache_key] = res
            return res
        elif hit_tp:
            res = {"result": "TP Hit", "cross_line_time": candle['time'], "PnL_%": 100 * (TP - entry_price) / entry_price}
            tp_sl_cache[cache_key] = res
            return res
        elif hit_sl:
            res = {"result": "SL Hit", "cross_line_time": candle['time'], "PnL_%": 100 * (SL - entry_price) / entry_price}
            tp_sl_cache[cache_key] = res
            return res
    res = {"result": "Still Open", "cross_line_time": pd.NaT, "PnL_%": None}
    tp_sl_cache[cache_key] = res
    return res

def process_chunk(chunk, candles_1h_dict, tp_sl_cache):
    # חישוב התוצאות לכל שורה בצ'אנק עם cache גלובלי
    results = []
    for _, row in chunk.iterrows():
        out = find_tp_sl_row(row, candles_1h_dict, tp_sl_cache)
        results.append(out)
    res_df = pd.DataFrame(results)
    chunk = chunk.reset_index(drop=True)
    for col in ['result', 'cross_line_time', 'PnL_%']:
        chunk[col] = res_df[col]
    return chunk

def main():
    # === שלב 1: טען את תוצאות הסיגנלים מהוקטוריזציה כ־stream בצ'אנקים ===
    chunk_size = 50000
    input_file = "vectorized_signals_full.csv"
    output_file = "vectorized_signals_with_tp_sl.csv"
    symbol_col = "symbol"

    # שלב 2: טעינת טווחי זמן וסימבולים
    first_chunk = pd.read_csv(input_file, nrows=50000, parse_dates=['time'])
    all_symbols = first_chunk[symbol_col].unique().tolist() if symbol_col in first_chunk.columns else ["BTCUSDT"]
    # מציאת טווחי זמן (אפשר לשפר עם קריאת צ'אנק ראשון ואחרון)
    min_time = first_chunk["time"].min()
    max_time = first_chunk["time"].max()

    # שלב 3: טען קנדלים פעם אחת לכל סימבול
    print("📥 טוען קנדלים 1H לכל סימבול...")
    candles_1h_dict = {}
    for sym in tqdm(all_symbols, desc="טעינת נרות 1H"):
        candles_1h_dict[sym] = load_candles_1h(sym, min_time, max_time + pd.Timedelta(days=5))
    print("✅ כל הקנדלים נטענו לזיכרון")

    # שלב 4: עיבוד בצ'אנקים + multiprocessing
    tp_sl_cache = {}
    cpu_count = min(12, multiprocessing.cpu_count())

    reader = pd.read_csv(input_file, parse_dates=['time'], chunksize=chunk_size)
    with open(output_file, "w", encoding="utf-8-sig") as f_out:
        header_written = False
        chunk_num = 0
        for chunk in tqdm(reader, desc="חישוב TP/SL לכל צ'אנק"):
            # רשימת שורות למקביליות
            chunk_result = process_chunk(chunk, candles_1h_dict, tp_sl_cache)
            if not header_written:
                chunk_result.to_csv(f_out, index=False, header=True)
                header_written = True
            else:
                chunk_result.to_csv(f_out, index=False, header=False)
            chunk_num += 1
    print("✅ בוצע חישוב TP/SL קדימה — נשמר קובץ:", output_file)

if __name__ == "__main__":
    main()
