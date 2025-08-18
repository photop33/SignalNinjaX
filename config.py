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
        "ema_short_window": [7],
        "ema_long_window": [25],
        "ADX_min": [25],
        "atr_window": [14],
        "volume_window": [20],
        "lookback_breakout": [5]

    }
}
RUN_PARAMS = [
    {"sl_multiplier": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,5],
     "rr_ratio": [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8, 1.9, 2.0,5],
     "volatility_min": [350,375,400,425,450,475,500,525,550,575],
     "volatility_max": [600,625,650,675,700,725,750],
     "atr_mult": [0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 2.75, 3.0,4, 5]

     }# 100 אחוז

     #{"sl_multiplier": 2,   "rr_ratio": 1.2, "volatility_min": 600, "volatility_max": 675},# 90 אחוז

]
