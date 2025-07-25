import os
import pandas as pd
import numpy as np
import polars as pl
from itertools import product
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from time import perf_counter
from config import INDICATOR_CONDITIONS, RUN_PARAMS, symbol, start_time_str, end_time_str

# ==== הגדרות כלליות ====
ind_params = INDICATOR_CONDITIONS["strategy_adx_ema_breakout_atr"]
symbols = symbol if isinstance(symbol, list) else [symbol]
start_time = pd.to_datetime(start_time_str)
end_time = pd.to_datetime(end_time_str)
TEMP_DIR = r"C:\Users\LiorSw\data\strategy_chunks"
os.makedirs(TEMP_DIR, exist_ok=True)

# ==== מדידת זמן כולל ====
total_start = perf_counter()

print("📥 שלב 1: טוען קובץ Parquet עם אינדיקטורים...")
df_full = pd.read_parquet("candles_with_indicators.parquet")
df_full = df_full[(df_full["time"] >= start_time) & (df_full["time"] <= end_time)]
if df_full.empty:
    print("❌ אין נתונים בטווח שבחרת")
    exit()

print("⚙️ שלב 2: יוצר קומבינציות...")
param_names = list(ind_params.keys())
param_values = [ind_params[k] for k in param_names]
combos = list(product(*param_values))
all_run_combos = [
    {**dict(zip(param_names, combo)), **run_cfg}
    for run_cfg in RUN_PARAMS
    for combo in combos
]

def process_combo(i_combo):
    i, combo = i_combo
    try:
        ema_short = combo["ema_short_window"]
        ema_long = combo["ema_long_window"]
        adx_min = combo["ADX_min"]
        atr_window = combo["atr_window"]
        atr_mult = combo["atr_mult"]
        rr_ratio = combo["rr_ratio"]
        volume_window = combo["volume_window"]
        lookback_breakout = combo["lookback_breakout"]
        high_prev_col = f"high_prev_{lookback_breakout}"

        cols = [
            "time", "close", "volume",
            f"ema_{ema_short}", f"ema_{ema_long}",
            f"adx_{atr_window}_{adx_min}", f"atr_{atr_window}",
            f"vol_ma_{volume_window}", high_prev_col
        ]

        dft = df_full[cols].copy()
        dft["ema_short"] = dft[f"ema_{ema_short}"]
        dft["ema_long"] = dft[f"ema_{ema_long}"]
        dft["ADX"] = dft[f"adx_{atr_window}_{adx_min}"]
        dft["atr"] = dft[f"atr_{atr_window}"]
        dft["volume_ma"] = dft[f"vol_ma_{volume_window}"]
        dft["high_prev"] = dft[high_prev_col]  # ✅ כאן אתה יוצר את העמודה הקבועה

        dft.drop(columns=[
            f"ema_{ema_short}",
            f"ema_{ema_long}",
            f"adx_{atr_window}_{adx_min}",
            f"atr_{atr_window}",
            f"vol_ma_{volume_window}",
            high_prev_col
        ], inplace=True)

        dft["SL"] = np.where(dft["atr"] > 0, dft["close"] - atr_mult * dft["atr"], np.nan)
        dft["TP"] = np.where(dft["atr"] > 0, dft["close"] + atr_mult * rr_ratio * dft["atr"], np.nan)

        dft["score"] = (
            (dft["close"] > dft["ema_long"]).astype(int) +
            (dft["ema_short"] > dft["ema_long"]).astype(int) +
            (dft["ADX"] > adx_min).astype(int) +
            (dft["volume"] > dft["volume_ma"]).astype(int) +
            2 * (dft["close"] > dft["high_prev"]).astype(int)
        )

        min_len = max([ema_long, atr_window, volume_window, lookback_breakout]) + 2
        dft = dft.iloc[min_len:]
        dft = dft[dft["score"] >= 6]

        if dft.empty:
            return None

        for k, v in combo.items():
            dft[k] = v

        dft.to_csv(os.path.join(TEMP_DIR, f"chunk_{i}.csv"), index=False)
        return 1
    except Exception as e:
        print(f"❌ שגיאה בקומבו {i}: {e}")
        return 0

def main():
    print(f"🚀 שלב 3: מריץ {len(all_run_combos)} קומבינציות עם multiprocessing...")
    saved = 0
    with ProcessPoolExecutor() as executor:
        for res in tqdm(executor.map(process_combo, enumerate(all_run_combos)), total=len(all_run_combos), desc="🔍 וקטוריזציה"):
            saved += res

    print("🧠 שלב 4: איחוד כל הקבצים עם Polars...")
    chunk_files = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.endswith(".csv")]
    df_list = []
    for f in tqdm(chunk_files, desc="📄 טוען קבצים"):
        df_list.append(pl.read_csv(f))

    final_df = pl.concat(df_list)

    print("💾 שלב 5: כותב לקובץ startegy.parquet...")
    final_df.write_parquet("startegy.parquet")

    total_end = perf_counter()
    print(f"✅ הסתיים — startegy.parquet מוכן | ⏱️ זמן כולל: {total_end - total_start:.2f} שניות")
    print(f"📊 סך הכול קומבינציות שנשמרו: {saved}")

if __name__ == "__main__":
    main()
