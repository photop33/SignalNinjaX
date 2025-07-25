import polars as pl
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import os
import time

# ---- הגדרות כלליות ----
condition_cols = [
    "ema_short_window", "ema_long_window", "ADX_min", "atr_window", "atr_mult",
    "volume_window", "lookback_breakout", "sl_multiplier", "rr_ratio",
    "volatility_min", "volatility_max"
]
indicators = [
    'volume', 'rsi', 'macd', 'macd_signal', 'macd_diff', 'ema_short', 'ema_long', 'bb_width',
    'atr', 'cci', 'parabolic_sar', 'vwap', 'wma', 'trix', 'bb_upper', 'bb_middle', 'bb_lower',
    'ema_7', 'ema_9', 'ema_10', 'ema_12', 'ema_13', 'ema_14', 'ema_21', 'ema_25', 'ema_26',
    'ema_30', 'ema_50', 'atr_14', 'atr_10', 'atr_20', 'atr_7', 'adx_18', 'adx_20', 'adx_22',
    'adx_25', 'adx_30', 'vol_ma_20', 'vol_ma_30', 'vol_ma_14', 'vol_ma_10', 'vol_ma_50',
    'high_prev_5', 'high_prev_10', 'high_prev_14', 'high_prev_20'
]


# ---- שלב 1: POLARS SUMMARY ----
def run_polars_summary():
    if os.path.exists("signals_summary_by_conditions.csv"):
        print("📄 קובץ summary כבר קיים – מדלג על שלב 1.")
        return

    start_polars = time.perf_counter()
    print("שלב 1: מריץ Polars Summary...")

    # קריאת שמות עמודות עם pandas
    with open("tpsl_result.csv", encoding="utf-8-sig") as f:
        first_line = f.readline().strip()
        columns = [col.strip() for col in first_line.split(",")]

    # קריאה עם polars
    df = pl.read_csv("tpsl_result.csv", has_header=True, new_columns=columns)
    df = df.rename({col: col.strip().replace('\r', '').replace('\n', '') for col in df.columns})

    # ניקוי ערכים ו-type casting
    for col in indicators + condition_cols + ["PnL_%"]:
        if col in df.columns and df[col].dtype == pl.Utf8:
            df = df.with_columns(
                pl.col(col).str.replace_all(r"\r|\n", "").cast(pl.Float64, strict=False)
            )

    # חישובי סיכום
    agg_exprs = []
    for ind in indicators:
        agg_exprs += [
            pl.col(ind).max().alias(f"{ind}_max"),
            pl.col(ind).min().alias(f"{ind}_min"),
            pl.col(ind).mean().alias(f"{ind}_avg"),
            pl.col(ind).var().alias(f"{ind}_var"),
        ]
    agg_exprs += [
        pl.col("PnL_%").count().alias("Num_Trades"),
        pl.col("PnL_%").sum().alias("Total_PnL_Percent"),
        pl.col("PnL_%").max().alias("Max_PnL_Percent"),
        pl.col("PnL_%").min().alias("Min_PnL_Percent"),
        (pl.col("PnL_%") > 0).sum().alias("TP_Hit_Count"),
        (pl.col("PnL_%") < 0).sum().alias("SL_Hit_Count"),
        ((pl.col("PnL_%") > 0).sum() / pl.col("PnL_%").count()).alias("TP_Hit_Percent"),
        ((pl.col("PnL_%") < 0).sum() / pl.col("PnL_%").count()).alias("SL_Hit_Percent"),
    ]

    summary = df.group_by(condition_cols).agg(agg_exprs)
    summary.write_csv("signals_summary_by_conditions.csv")
    end_polars = time.perf_counter()
    print("✅ סיכום ביניים נשמר ל-signals_summary_by_conditions.csv")
    print(f"⏱ סיום שלב 1: לקח {end_polars - start_polars:.1f} שניות")


# ---- שלב 2: MULTIPROCESSING CALCULATIONS ----
def profit_factor(pnl): return pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()) if any(pnl < 0) else np.nan
def sharpe_ratio(pnl): return pnl.mean() / pnl.std() if pnl.std() > 0 else np.nan
def max_drawdown(pnl): return (pd.Series(pnl).cumsum().cummax() - pd.Series(pnl).cumsum()).max()
def max_consecutive_losses(pnl):
    max_streak = streak = 0
    for val in pnl:
        if val < 0: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
    return max_streak
def win_streak(pnl):
    max_streak = streak = 0
    for val in pnl:
        if val > 0: streak += 1; max_streak = max(max_streak, streak)
        else: streak = 0
    return max_streak
def percent_outliers(pnl):
    std = pnl.std(); mean = pnl.mean()
    return np.sum(np.abs(pnl - mean) > 3 * std) / len(pnl) if len(pnl) else 0
def trade_distribution(times):
    hours = pd.to_datetime(times).dt.hour
    return ', '.join([f"{h}:{c}" for h, c in hours.value_counts().sort_index().items()])
def trade_distribution_days(times):
    days = pd.to_datetime(times).dt.dayofweek.map({0: 1, 1: 2, 2: 3, 3: 4, 4: 5, 5: 6, 6: 7})
    return ', '.join([f"{d}:{c}" for d, c in days.value_counts().sort_index().items()])
def max_min_exposure(entries, exits):
    events = pd.DataFrame({
        'time': pd.concat([pd.to_datetime(entries), pd.to_datetime(exits)]),
        'type': ['entry'] * len(entries) + ['exit'] * len(exits)
    }).sort_values('time')
    current = max_e = min_e = 0
    exposures = []
    for _, row in events.iterrows():
        current += 1 if row['type'] == 'entry' else -1
        exposures.append(current)
    return {'MAX_Exposure': max(exposures), 'MIN_Exposure': min(exposures)}

def process_group(idx_row, condition_cols, pnl_col, chunk_size=200000):
    idx, row = idx_row
    all_pnls, all_tp, all_sl, all_times, all_cross_times = [], [], [], [], []

    for chunk in pd.read_csv('tpsl_result.csv', chunksize=chunk_size, low_memory=False):
        chunk.columns = chunk.columns.str.strip().str.replace(r'[\r\n]', '', regex=True)
        for col in condition_cols + [pnl_col]:
            if col in chunk.columns:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce')

        filt = np.all([chunk[col] == row[col] for col in condition_cols if col in chunk.columns], axis=0)
        group = chunk[filt]
        if group.empty: continue

        all_pnls.extend(group[pnl_col].astype(float).tolist())
        all_tp.extend(group[group[pnl_col] > 0][pnl_col].astype(float).tolist())
        all_sl.extend(group[group[pnl_col] < 0][pnl_col].astype(float).tolist())
        all_times.extend(pd.to_datetime(group['time'], errors='coerce').dropna().tolist())
        all_cross_times.extend(pd.to_datetime(group['cross_line_time'], errors='coerce').dropna().tolist())

    if not all_pnls: return idx, {}

    pnl = np.array(all_pnls)
    result = {
        'Hour_Trade': trade_distribution(all_times),
        'Days_Trade': trade_distribution_days(all_times),
        'Pnl_TP_per_deal': np.mean(all_tp) if all_tp else np.nan,
        'Pnl_SL_per_deal': np.mean(all_sl) if all_sl else np.nan,
        'Max_Drawdown': max_drawdown(pnl),
        'Max_Consecutive_Losses': max_consecutive_losses(pnl),
        'Win_Streaks': win_streak(pnl),
        '%_Outliers': percent_outliers(pnl) * 100,
        'Expectancy': (np.mean(all_tp) if all_tp else 0) * (len(all_tp)/len(pnl)) +
                      (np.mean(all_sl) if all_sl else 0) * (len(all_sl)/len(pnl)) if len(pnl) else np.nan,
        'Profit_Factor': profit_factor(pnl),
        'Sharpe_Ratio': sharpe_ratio(pnl)
    }

    if all_times and all_cross_times:
        durations = (pd.Series(all_cross_times) - pd.Series(all_times)).dt.total_seconds() / 3600
        result['Avg_Trade_Duration_hours'] = durations.mean()
        result.update(max_min_exposure(all_times, all_cross_times))

    return idx, result


# ---- MAIN ----
if __name__ == "__main__":
    run_polars_summary()

    print("== שלב 2: חישובי summary עם צ'אנקים ומקביליות ==")
    start = time.perf_counter()

    summary = pd.read_csv('signals_summary_by_conditions.csv')
    for col in condition_cols + ['PnL_%']:
        if col in summary.columns:
            summary[col] = pd.to_numeric(summary[col], errors='coerce')

    stats_cols = [
        'Hour_Trade', 'Days_Trade', 'Avg_Trade_Duration_hours',
        'Pnl_TP_per_deal', 'Pnl_SL_per_deal', 'Max_Drawdown',
        'Max_Consecutive_Losses', 'Win_Streaks', '%_Outliers',
        'Expectancy', 'Profit_Factor', 'Sharpe_Ratio',
        'MAX_Exposure', 'MIN_Exposure'
    ]
    for col in stats_cols:
        summary[col] = pd.Series(dtype="object")

    with ProcessPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(process_group, (idx, row), condition_cols, "PnL_%")
            for idx, row in summary.iterrows()
        ]
        for f in tqdm(as_completed(futures), total=len(futures), desc="Processing summary"):
            try:
                idx, result = f.result()
                for key, val in result.items():
                    summary.at[idx, key] = val
            except Exception as e:
                print(f"⚠️ שגיאה בעיבוד קבוצה: {e}")

    summary.to_csv("signals_summary_by_conditions_full.csv", index=False)
    print(f"✅ סיים הכל! זמן כולל: {time.perf_counter() - start:.1f} שניות")
