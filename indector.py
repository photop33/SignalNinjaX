import pandas as pd
import ta

def add_indicators(df: pd.DataFrame, wma_window=14) -> pd.DataFrame:
    # ודא שיש מספיק שורות (הכי הרבה window שאתה משתמש בו זה 26 לפחות)
    if len(df) < 14:
        print(f"⚠️ מעט מדי נרות — חלק מהאינדיקטורים יהיו NaN")

    df[["open", "high", "low", "close", "volume"]] = df[
        ["open", "high", "low", "close", "volume"]
    ].astype(float)

    try:
        df["rsi"] = ta.momentum.RSIIndicator(close=df["close"], window=6).rsi()
    except Exception as e:
        df["rsi"] = None
        print(f"⚠️ RSI שגיאה: {e}")

    try:
        macd = ta.trend.MACD(close=df["close"], window_slow=26, window_fast=12, window_sign=9)
        df["macd"] = macd.macd()
        df["macd_signal"] = macd.macd_signal()
        df["macd_diff"] = macd.macd_diff()
    except Exception as e:
        df["macd"] = df["macd_signal"] = df["macd_diff"] = None
        print(f"⚠️ MACD שגיאה: {e}")

    try:
        df["adx"] = ta.trend.ADXIndicator(high=df["high"], low=df["low"], close=df["close"]).adx()
        df.rename(columns={"adx": ""}, inplace=True)
    except Exception as e:
        df["adx"] = None
        print(f"⚠️ ADX שגיאה: {e}")

    try:
        df["ema_short"] = ta.trend.EMAIndicator(close=df["close"], window=7).ema_indicator()
        df["ema_long"] = ta.trend.EMAIndicator(close=df["close"], window=25).ema_indicator()
    except Exception as e:
        df["ema_short"] = df["ema_long"] = None
        print(f"⚠️ EMA שגיאה: {e}")

    try:
        bb = ta.volatility.BollingerBands(close=df["close"], window=20, window_dev=2)
        df["bb_width"] = bb.bollinger_wband()
    except Exception as e:
        df["bb_width"] = None
        print(f"⚠️ Bollinger שגיאה: {e}")

    try:
        df["volatility"] = df["close"].rolling(window=10).std()
    except Exception as e:
        df["volatility"] = None
        print(f"⚠️ Volatility שגיאה: {e}")

    try:
        df["atr"] = ta.volatility.AverageTrueRange(
            high=df["high"], low=df["low"], close=df["close"], window=14
        ).average_true_range()
    except Exception as e:
        df["atr"] = None
        print(f"⚠️ ATR שגיאה: {e}")

    try:
        df["cci"] = ta.trend.CCIIndicator(
            high=df["high"], low=df["low"], close=df["close"], window=9
        ).cci()
    except Exception as e:
        df["cci"] = None
        print(f"⚠️ CCI שגיאה: {e}")

    try:
        df["parabolic_sar"] = ta.trend.PSARIndicator(
            high=df["high"], low=df["low"], close=df["close"]
        ).psar()
    except Exception as e:
        df["parabolic_sar"] = None
        print(f"⚠️ PSAR שגיאה: {e}")

    try:
        df["vwap"] = (df["close"] * df["volume"]).cumsum() / df["volume"].cumsum()
    except Exception as e:
        df["vwap"] = None
        print(f"⚠️ VWAP שגיאה: {e}")

    # --- כאן שינוי לחישוב WMA דינאמי ---
    try:
        df["wma"] = df["close"].rolling(window=wma_window).apply(
            lambda x: (x * range(1, len(x) + 1)).sum() / sum(range(1, len(x) + 1)),
            raw=True
        )
    except Exception as e:
        df["wma"] = None
        print(f"⚠️ WMA שגיאה: {e}")

    try:
        df["trix"] = ta.trend.TRIXIndicator(close=df["close"], window=15).trix()
    except Exception as e:
        df["trix"] = None
        print(f"⚠️ TRIX שגיאה: {e}")

    try:
        bb = ta.volatility.BollingerBands(close=df["close"], window=5, window_dev=2)
        df["bb_upper"] = bb.bollinger_hband()
        df["bb_middle"] = bb.bollinger_mavg()
        df["bb_lower"] = bb.bollinger_lband()
        df["bb_width"] = df["bb_upper"] - df["bb_lower"]
    except Exception as e:
        df["bb_upper"] = None
        df["bb_middle"] = None
        df["bb_lower"] = None
        df["bb_width"] = None
        print(f"⚠️ Bollinger שגיאה: {e}")

    return df
import numpy as np

def compute_adx_tradingview(df, di_len=14, adx_smooth=25):
    df = df.copy()

    df["tr"] = np.maximum(df["high"] - df["low"],
                          np.maximum(abs(df["high"] - df["close"].shift(1)),
                                     abs(df["low"] - df["close"].shift(1))))

    df["+dm"] = np.where((df["high"] - df["high"].shift(1) > df["low"].shift(1) - df["low"]) &
                         (df["high"] - df["high"].shift(1) > 0),
                         df["high"] - df["high"].shift(1), 0)

    df["-dm"] = np.where((df["low"].shift(1) - df["low"] > df["high"] - df["high"].shift(1)) &
                         (df["low"].shift(1) - df["low"] > 0),
                         df["low"].shift(1) - df["low"], 0)

    tr_smooth = df["tr"].rolling(window=di_len).sum()
    plus_dm_smooth = df["+dm"].rolling(window=di_len).sum()
    minus_dm_smooth = df["-dm"].rolling(window=di_len).sum()

    plus_di = 100 * plus_dm_smooth / tr_smooth
    minus_di = 100 * minus_dm_smooth / tr_smooth
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)

    adx = dx.rolling(window=adx_smooth).mean()

    return adx
