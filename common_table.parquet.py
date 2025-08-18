import polars as pl
from tqdm import tqdm
from time import perf_counter

STRATEGY_FILE = "strategy.parquet"
COMMON_OUT = "common_table.parquet"
COMBO_OUT = "combinations_table.parquet"

# 🧠 העמודות שיישארו בקובץ הסופי
wanted_common_columns = [
    'time', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'macd',
    'macd_signal', 'macd_diff', 'ADX', 'ema_short', 'ema_long', 'bb_width', 'volatility', 'atr',
    'cci', 'parabolic_sar', 'vwap', 'wma', 'trix', 'bb_upper', 'bb_middle', 'bb_lower', 'score', 'reasons', 'volume_ma',
    'high_prev', 'SL', 'TP', 'sl_multiplier', 'rr_ratio', 'volatility_min', 'volatility_max','atr_mult'
    'cross_line_time', 'result', 'PnL_%']


def main():
    total_start = perf_counter()

    print("📥 טוען strategy.parquet...")
    df = pl.read_parquet(STRATEGY_FILE)

    print("📋 בודק עמודות חסרות...")
    for col in wanted_common_columns:
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).alias(col))

    print("🔍 בונה common_table ומזהה קומבינציות...")
    common_rows = []
    seen_keys = set()

    for row in tqdm(df.iter_rows(named=True), desc="⚙️ מעבד שורות"):
        key = row["time"]
        if key in seen_keys:
            continue

        # שורת Common חדשה
        common_rows.append({col: row.get(col, None) for col in wanted_common_columns})
        seen_keys.add(key)

    print("🧱 יוצר טבלאות Polars...")
    common_df = pl.DataFrame(common_rows)

    print("💾 שומר קבצים...")
    common_df.write_parquet(COMMON_OUT)

    print("✅ הסתיים בהצלחה:")
    print(f" - {COMMON_OUT} (עמודות: {len(common_df.columns)})")
    print(f"⏱️ זמן כולל: {perf_counter() - total_start:.2f} שניות")
    return 0
if __name__ == "__main__":
    main()
