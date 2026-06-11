"""factor/v4/formulas/{} — V4 AST 语法"""

from typing import Dict

BASIC_FACTORS: Dict[str, str] = {
    "corr_close_vol_20d": "ts_corr(CLOSE, VOLUME, 20)",
    "cs_rank_mom_20d": "cs_rank((CLOSE - ts_delay(CLOSE, 20)))",
    "cs_rank_volume": "cs_rank(VOLUME)",
    "cs_zscore_mom_10d": "ts_zscore((CLOSE - ts_delay(CLOSE, 10)), 20)",
    "dollar_volume": "((CLOSE - OPEN) * VOLUME)",
    "ma_cross_10_60": "(ts_mean(CLOSE, 10) - ts_mean(CLOSE, 60))",
    "ma_cross_5_20": "(ts_mean(CLOSE, 5) - ts_mean(CLOSE, 20))",
    "mom_10d": "(CLOSE - ts_delay(CLOSE, 10))",
    "mom_20d": "(CLOSE - ts_delay(CLOSE, 20))",
    "mom_5d": "(CLOSE - ts_delay(CLOSE, 5))",
    "mom_60d": "(CLOSE - ts_delay(CLOSE, 60))",
    "range_pct": "((HIGH - LOW) / CLOSE)",
    "rev_20d": "(-(CLOSE - ts_delay(CLOSE, 20)))",
    "rev_5d": "(-(CLOSE - ts_delay(CLOSE, 5)))",
    "sma_10d": "ts_mean(CLOSE, 10)",
    "vol_10d": "ts_std(CLOSE, 10)",
    "vol_20d": "ts_std(CLOSE, 20)",
    "vol_60d": "ts_std(CLOSE, 60)",
    "vol_ratio_10d": "(VOLUME / ts_mean(VOLUME, 10))",
    "vol_ratio_20d": "(VOLUME / ts_mean(VOLUME, 20))",
}
