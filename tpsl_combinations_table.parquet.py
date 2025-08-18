import pandas as pd
from tqdm import tqdm

# === קבצים ===
COMMON_FILE = "common_table.parquet"
CANDLES_FILE = "candles_1h.parquet"


# === פונקציה לבדיקה אם TP או SL הושגו ===
def find_tp_sl(row, candles):
    TP, SL = row["TP"], row["SL"]
    entry_price = row["close"]
    entry_time = row["time"]

    if pd.isna(TP) or pd.isna(SL) or pd.isna(entry_price) or pd.isna(entry_time):
        return "No Data", pd.NaT, None

    start_check_time = entry_time + pd.Timedelta(hours=4)
    relevant = candles[candles["time"] >= start_check_time]

    for _, candle in relevant.iterrows():
        hit_tp = candle['high'] >= TP
        print(hit_tp)
        hit_sl = candle['low'] <= SL
        if hit_tp and hit_sl:
            # מה שבא ראשון
            if candle["low"] <= SL and candle["high"] >= TP:
                # סדר לפי מי קרוב יותר למחיר הכניסה
                if abs(TP - entry_price) < abs(entry_price - SL):
                    return "TP Hit", candle["time"], 100 * (TP - entry_price) / entry_price
                else:
                    return "SL Hit", candle["time"], 100 * (SL - entry_price) / entry_price
        elif hit_tp:
            return "TP Hit", candle["time"], 100 * (TP - entry_price) / entry_price
        elif hit_sl:
            return "SL Hit", candle["time"], 100 * (SL - entry_price) / entry_price

    return "Still Open", pd.NaT, None


# === פונקציית הרצה ראשית ===
def main():
    print("📥 טוען קובץ Common...")
    df = pd.read_parquet(COMMON_FILE)
    df["time"] = pd.to_datetime(df["time"], errors="coerce")

    # ודא שיש עמודות יעד
    for col in ["result", "cross_line_time", "PnL_%"]:
        if col not in df.columns:
            df[col] = pd.NA

    print("🔍 סינון רק שורות שעדיין לא חושבו...")
    df_to_process = df[df["cross_line_time"].isna()].copy()
    if df_to_process.empty:
        print("⛔ אין שורות חדשות לעיבוד.")
        return

    print("📦 טוען נרות פעם אחת בלבד...")
    candles = pd.read_parquet(CANDLES_FILE)
    candles["time"] = pd.to_datetime(candles["time"], errors="coerce")

    print("🚀 מחשב TP/SL לכל שורה...")
    for idx, row in tqdm(df_to_process.iterrows(), total=len(df_to_process), desc="חישוב"):
        result, cross_time, pnl = find_tp_sl(row, candles)
        df.loc[df["time"] == row["time"], ["result", "cross_line_time", "PnL_%"]] = result, cross_time, pnl

    print("💾 שומר תוצאה לקובץ...")
    df.to_parquet(COMMON_FILE, index=False)
    print("✅ סיום.")


if __name__ == "__main__":
    main()
