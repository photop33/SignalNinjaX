import pandas as pd
import hashlib

# ===== שלב 1: טען קובץ אסטרטגיה =====
print("📥 טוען קובץ אסטרטגיה...")
df = pd.read_parquet("strategy.parquet")  # או CSV

print("📋 עמודות שנמצאו:", df.columns.tolist())

# ===== שלב 2: הגדר עמודות מבוקשות =====
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

# ===== שלב 3: ודא שכל עמודות ה־common קיימות גם אם ריקות =====
for col in wanted_common_columns:
    if col not in df.columns:
        print(f"➕ מוסיף עמודה חסרה: {col}")
        df[col] = pd.NA

# ===== הוספת עמודת symbol עם ערך קבוע לכל שורה =====
df["symbol"] = "BTCUSDT"

# ===== שלב 4: יצירת טבלת COMMON עם מזהה ייחודי לכל time+symbol =====
def hash_time_row(row):
    return hashlib.md5(str(tuple(row)).encode()).hexdigest()[:10]

print("🔍 בונה טבלת Common לפי זמן...")
common_df = df[wanted_common_columns + ["symbol"]].drop_duplicates().copy()
common_df["common_id"] = common_df.apply(hash_time_row, axis=1)

# טבלת common מוכנה
common_table = common_df[["common_id", "symbol"] + wanted_common_columns]

# ===== שלב 5: קישור זמני סיגנל ל־common_id =====
print("🔗 קושר כל שורת אסטרטגיה לפי זמן ל־common_id...")
time_to_common_id = dict(zip(common_df["time"], common_df["common_id"]))
df["common_id"] = df["time"].map(time_to_common_id)

# ===== שלב 6: יצירת מזהה קומבינציה =====
def hash_combo_row(row):
    values = tuple(row[col] for col in combo_columns + ["common_id"])
    return hashlib.md5(str(values).encode()).hexdigest()[:10]

combo_df = df[combo_columns + ["common_id"]].copy()
combo_df["combo_id"] = combo_df.apply(hash_combo_row, axis=1)
combo_df = combo_df.drop_duplicates()

# ===== שלב 7: שמירת קבצים =====
print("💾 שומר קבצים...")
common_table.to_parquet("common_table.parquet", index=False)
combo_df.to_parquet("combinations_table.parquet", index=False)

print("✅ הסתיים בהצלחה:")
print(" - common_table.parquet")
print(" - combinations_table.parquet")
