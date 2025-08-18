import pandas as pd
from tqdm import tqdm

# 📥 טען את קובצי ה־Parquet
common_df = pd.read_parquet("common_table.parquet")
combo_df = pd.read_parquet("combinations_table.parquet")

# 🔄 מיזוג עם בר התקדמות (לא נדרש בפועל ב־merge, אבל מופיע ללוגיקה)
print("🔄 ממזג את הקבצים לפי time...")
tqdm.pandas(desc="מיזוג")
merged_df = combo_df.merge(common_df, on="time", how="left")

# 💾 שמירה לקובץ Parquet
print("💾 שומר את הקובץ לקובץ merged_signals.parquet...")
merged_df.to_parquet("merged_signals.parquet", index=False)

print("✅ קובץ מאוחד נשמר בשם merged_signals.parquet")
