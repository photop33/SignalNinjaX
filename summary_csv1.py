import os
import pandas as pd
import numpy as np
from itertools import product
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from time import perf_counter
from config import INDICATOR_CONDITIONS, RUN_PARAMS, symbol, start_time_str, end_time_str

# ==== הגדרות כלליות ====
ind_params = INDICATOR_CONDITIONS["strategy_adx_ema_breakout_atr"]
symbols = symbol if isinstance(symbol, list) else [symbol]
start_time = pd.to_datetime(start_time_str)
end_time = pd.to_datetime(end_time_str)
TEMP_DIR = r"C:\Users\LiorSw\data\strategy_chunks"
os.makedirs(TEMP_DIR, exist_ok=True)

# ==== הגדרות עמודות צפויות ====
expected_columns = [
    'symbol', 'time', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'macd',
    'macd_signal', 'macd_diff', 'ADX', 'ema_short', 'ema_long', 'bb_width', 'volatility', 'atr',
    'cci', 'parabolic_sar', 'vwap', 'wma', 'trix', 'bb_upper', 'bb_middle', 'bb_lower', 'score', 'reasons', 'volume_ma',
    'high_prev', 'SL', 'TP', 'sl_multiplier', 'rr_ratio', 'volatility_min', 'volatility_max',
    'cross_line_time', 'result', 'PnL_%'
]
import pyarrow.parquet as pq
print("📥 שלב 1: קורא את הסכימה של merged_signals.parquet...")

# ==== יצירת קומבינציות ====
print("⚙️ שלב 2: יוצר קומבינציות...")
param_names = list(ind_params.keys())
param_values = [ind_params[k] for k in param_names]
combos = list(product(*param_values))
all_run_combos = [
    {**dict(zip(param_names, combo)), **run_cfg}
    for run_cfg in RUN_PARAMS
    for combo in combos
]

# ==== פונקציה לעיבוד קומבו ====
def process_combo(i_combo):
    import pandas as pd
    import numpy as np
    import pyarrow.parquet as pq

    i, combo = i_combo
    try:
        all_columns = set(pq.read_schema("merged_signals.parquet").names)
        ema_short = combo["ema_short_window"]
        ema_long = combo["ema_long_window"]
        adx_min = combo["ADX_min"]
        atr_window = combo["atr_window"]
        atr_mult = combo["atr_mult"]
        rr_ratio = combo["rr_ratio"]
        volume_window = combo["volume_window"]
        lookback_breakout = combo["lookback_breakout"]
        high_prev_col = f"high_prev_{lookback_breakout}"

        requested_cols = [
            "time", "open", "high", "low", "close", "volume",
            "rsi", "macd", "macd_signal", "macd_diff", "ADX",

            "bb_width", "volatility", "cci",
            "parabolic_sar", "vwap", "wma", "trix", "bb_upper",
            "bb_middle", "bb_lower"


        ]

        cols = [col for col in requested_cols if col in all_columns]

        missing = [col for col in requested_cols if col not in all_columns]
        if missing:
            print(f"⚠️ קומבו {i}: עמודות חסרות: {missing}")
            return 0

        df = pd.read_parquet("merged_signals.parquet", columns=cols)
        df["time"] = pd.to_datetime(df["time"], errors="coerce")
        df = df[(df["time"] >= start_time) & (df["time"] <= end_time)]

        if df.empty:
            return 0
        df["ema_short"] = df.get(f"ema_{ema_short}")
        df["ema_long"] = df.get(f"ema_{ema_long}")
        df["ADX"] = df.get(f"adx_{atr_window}_{adx_min}")
        df["atr"] = df.get(f"atr_{atr_window}")
        df["volume_ma"] = df.get(f"vol_ma_{volume_window}")
        df["high_prev"] = df.get(high_prev_col)

        # טיפולים לערכים חסרים
        df["ema_short"] = pd.to_numeric(df["ema_short"], errors="coerce")
        df["ema_long"] = pd.to_numeric(df["ema_long"], errors="coerce")
        df["ADX"] = pd.to_numeric(df["ADX"], errors="coerce")
        df["volume_ma"] = pd.to_numeric(df["volume_ma"], errors="coerce")
        df["high_prev"] = pd.to_numeric(df["high_prev"], errors="coerce")
        df["volume"] = pd.to_numeric(df["volume"], errors="coerce")

        # סינון שורות עם ערכים חסרים לפני חישוב score
        df = df.dropna(subset=["ema_short", "ema_long", "ADX", "volume_ma", "high_prev", "close", "volume"])

        # חישוב ציון
        df["score"] = (
                (df["close"] > df["ema_long"]).astype(int) +
                (df["ema_short"] > df["ema_long"]).astype(int) +
                (df["ADX"] > adx_min).astype(int) +
                (df["volume"] > df["volume_ma"]).astype(int) +
                2 * (df["close"] > df["high_prev"]).astype(int)
        )

        min_len = max([ema_long, atr_window, volume_window, lookback_breakout]) + 2
        df = df.iloc[min_len:]
        df = df[df["score"] >= 6]

        if df.empty:
            return 0

        for k, v in combo.items():
            df[k] = v

        for col in expected_columns:
            if col not in df.columns:
                df[col] = pd.NA

        df["symbol"] = "BTCUSDT"
        df = df[sorted(df.columns)]
        df.to_parquet(os.path.join(TEMP_DIR, f"chunk_{i}.parquet"), index=False)
        print(f"✅ קומבו {i} נשמר כ- chunk_{i}.parquet")
        return 1
    except Exception as e:
        print(f"❌ שגיאה בקומבו {i}: {e}")
        return 0

# ==== הפונקציה הראשית ====
def main():
    total_start = perf_counter()
    print(f"🚀 שלב 3: מריץ {len(all_run_combos)} קומבינציות עם multiprocessing...")

    saved = 0
    with ProcessPoolExecutor(max_workers=6) as executor:
        futures = [executor.submit(process_combo, item) for item in enumerate(all_run_combos)]
        for f in tqdm(as_completed(futures), total=len(futures), desc="🔍 וקטוריזציה"):
            saved += f.result()

    print("🧠 שלב 4: איחוד כל הקבצים עם pandas...")
    chunk_files = [os.path.join(TEMP_DIR, f) for f in os.listdir(TEMP_DIR) if f.endswith(".parquet")]
    df_list = [pd.read_parquet(f) for f in tqdm(chunk_files, desc="📄 טוען קבצים")]
    final_df = pd.concat(df_list, ignore_index=True)

    print(f"💾 שלב 5: כותב לקובץ summary.csv...")
    final_df.to_csv("summary.csv", index=False, encoding="utf-8-sig")

    total_end = perf_counter()
    print(f"✅ הסתיים — summary.csv מוכן | ⏱️ זמן כולל: {total_end - total_start:.2f} שניות")
    print(f"📊 סך הכול קומבינציות שנשמרו: {saved}")

if __name__ == "__main__":
    main()
