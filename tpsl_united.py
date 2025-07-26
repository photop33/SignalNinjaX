import pandas as pd

# טען את קובצי ה־parquet
common_df = pd.read_parquet("common_table.parquet")
combo_df = pd.read_parquet("combinations_table.parquet")

# מיזוג לפי common_id
merged_df = combo_df.merge(common_df, on="common_id", how="left")

# שמירה ל־CSV
merged_df.to_csv("merged_signals.csv", index=False, encoding="utf-8-sig")

print("✅ קובץ מאוחד נשמר בשם merged_signals.csv")
