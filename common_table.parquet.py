import pandas as pd
import hashlib
import time
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# ========= הגדרות =========
NUM_WORKERS = multiprocessing.cpu_count()

# ========= פונקציות =========
def log_time(stage_name, start_time):
    duration = time.perf_counter() - start_time
    print(f"⏱️ {stage_name} הסתיים תוך {duration:.2f} שניות.\n")

def hash_row_time(row_tuple):
    return hashlib.md5(str(row_tuple).encode()).hexdigest()[:10]

def hash_row_combo(row_tuple):
    return hashlib.md5(str(row_tuple).encode()).hexdigest()[:10]

# ========= שלב 1: טען קובץ =========
print("📥 שלב 1: טוען קובץ אסטרטגיה...")
start = time.perf_counter()
df = pd.read_parquet("strategy.parquet")
print("📋 עמודות שנמצאו:", df.columns.tolist())
log_time("טעינת הקובץ", start)

# ========= שלב 2: עמודות =========
wanted_common_columns = [
    'time', 'open', 'high', 'low', 'close', 'volume',
    'rsi', 'macd', 'macd_signal', 'macd_diff', 'ADX',
    'ema_short', 'ema_long', 'bb_width', 'volatility', 'atr',
    'cci', 'parabolic_sar', 'vwap', 'wma', 'trix',
    'bb_upper', 'bb_middle', 'bb_lower', 'ema_7', 'ema_25',
    'atr_14', 'adx_25', 'vol_ma_20', 'high_prev_5',
    'score', 'reasons', 'volume_ma', 'high_prev',
    'SL', 'TP', 'sl_multiplier', 'rr_ratio',
    'volatility_min', 'volatility_max'
]

combo_columns = [
    'ema_short_window', 'ema_long_window', 'ADX_min', 'atr_window',
    'atr_mult', 'volume_window', 'lookback_breakout'
]

# ========= שלב 3: ודא עמודות =========
print("🧩 שלב 2: בודק עמודות חסרות...")
start = time.perf_counter()
for col in tqdm(wanted_common_columns, desc="בודק עמודות"):
    if col not in df.columns:
        print(f"➕ מוסיף עמודה חסרה: {col}")
        df[col] = pd.NA
log_time("בדיקת עמודות חסרות", start)

# ========= שלב 4: טבלת Common =========
print("🔍 שלב 3: בונה טבלת Common...")
start = time.perf_counter()
df["symbol"] = "BTCUSDT"
common_df = df[wanted_common_columns + ["symbol"]].drop_duplicates().copy()

print("🧠 מחשב common_id עם ריבוי ליבות...")
rows = common_df[wanted_common_columns + ["symbol"]].itertuples(index=False, name=None)
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    common_ids = list(tqdm(executor.map(hash_row_time, rows), total=len(common_df), desc="🔢 common_id"))

common_df["common_id"] = common_ids
common_table = common_df[["common_id", "symbol"] + wanted_common_columns]
log_time("יצירת טבלת Common", start)

# ========= שלב 5: קישור common_id לפי time =========
print("🔗 שלב 4: קישור common_id לפי זמן...")
start = time.perf_counter()
time_to_common_id = dict(zip(common_df["time"], common_df["common_id"]))
df["common_id"] = df["time"].map(time_to_common_id)
log_time("קישור common_id", start)

# ========= שלב 6: טבלת קומבינציות =========
print("🧮 שלב 5: בונה טבלת קומבינציות...")
start = time.perf_counter()
combo_df = df[combo_columns + ["common_id"]].drop_duplicates().copy()

print("🧠 מחשב combo_id עם ריבוי ליבות...")
combo_rows = combo_df[combo_columns + ["common_id"]].itertuples(index=False, name=None)
with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
    combo_ids = list(tqdm(executor.map(hash_row_combo, combo_rows), total=len(combo_df), desc="🔐 combo_id"))

combo_df["combo_id"] = combo_ids
log_time("יצירת טבלת קומבינציות", start)

# ========= שלב 7: שמירה =========
print("💾 שלב 6: שומר קבצים...")
start = time.perf_counter()
common_table.to_parquet("common_table.parquet", index=False)
combo_df.to_parquet("combinations_table.parquet", index=False)
log_time("שמירת קבצים", start)

# ========= סיום =========
print("✅ הסתיים בהצלחה:")
print(" - common_table.parquet")
print(" - combinations_table.parquet")
