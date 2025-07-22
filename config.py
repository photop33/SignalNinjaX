from itertools import product

SCAN_FROM = "2025-01-01 00:00"
SCAN_TO = "2025-07-13 23:59"
INTERVAL = "4h"
symbol_scan = crypto_list = ["ETHUSDT"]


VOLATILITY_THRESHOLD = 0.0
PCT_CHANGE_THRESHOLD = 0.00
MIN_VOLUME = 0
FILTER_MODE = "OR"


symbol = ["BTCUSDT"]
interval = "4h"
start_time_str = "2017-07-1 00:00"
end_time_str = "2025-07-13 23:59"
INDICATOR_CONDITIONS = {
    "strategy_adx_ema_breakout_atr": {
        "ema_short_window": [7, 9,10,12,13,14],
        "ema_long_window": [21,25,26,30, 50],
        "ADX_min": [18,20,22, 25, 30],
        "atr_window": [14,10,20,7],
        "atr_mult": [1.0,1.2, 1.5,2.0],
        "volume_window": [20, 30,14, 10,50],
        "lookback_breakout": [5, 10,14,20]
    }
}
RUN_PARAMS = [
    #{"sl_multiplier": 2,   "rr_ratio": 1.1, "volatility_min": 600, "volatility_max": 650},# 100 אחוז
     {"sl_multiplier": 2,   "rr_ratio": 1.2, "volatility_min": 600, "volatility_max": 675},# 90 אחוז

]


ACTIVE_STRATEGIES = [
    "strategy_adx_ema_breakout_atr"
]
STRATEGY_THRESHOLDS = {
    "strategy_adx_ema_breakout_atr": 6
}
