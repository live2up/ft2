"""factor/v4/formulas/{} — V4 AST 语法"""

from typing import Dict

INDUSTRY_ALPHA: Dict[str, str] = {
    "ind_amt_accel_20d": "cs_rank(ts_delta(ts_mean(AMOUNT, 10), 10))",
    "ind_argmax_cross_60d": "cs_rank((ts_argmax(CLOSE, 60) + ts_argmax(AMOUNT, 60)))",
    "ind_argmin_delta1_30d": "cs_rank(ts_argmin(ts_delta(CLOSE, 1), 30))",
    "ind_corr_bench_delta": "cs_rank(ts_delta(ts_corr(CLOSE, BENCH_CLOSE, 20), 20))",
    "ind_dual_sw_a1": "((0.9 * (cs_rank(ts_argmin(ts_delta(CLOSE, 1), 30)) if (ts_rank(BENCH_CLOSE, 60) > 0.7) else cs_rank(ts_argmax(CLOSE, 60)))) + (0.1 * cs_rank(ts_argmin(ts_delta(CLOSE, 1), 30))))",
    "ind_gap_range": "cs_rank(((OPEN - ts_delay(CLOSE, 1)) / ((HIGH - LOW) + 0.01)))",
    "ind_rank_stable_5d": "(-ts_std(cs_rank(CLOSE), 5))",
    "ind_relamt_rank_10d": "cs_rank(ts_rank(REL_AMOUNT, 10))",
    "ind_revert_30d": "cs_rank(ts_argmin(ts_delta(CLOSE, 1), 30))",
    "ind_revert_enhanced": "(cs_rank(ts_argmin(ts_delta(CLOSE, 1), 30)) + (((ts_delta(CLOSE, 5) > 0) and (ts_delta(AMOUNT, 5) > 0)) * cs_rank(ts_argmin(ts_delta(CLOSE, 1), 30))))",
    "ind_revert_vol_confirm": "cs_rank(((-ts_zscore(ts_rank(CLOSE, 10), 10)) * ts_zscore(AMOUNT, 20)))",
    "ind_share_residual_20d": "cs_rank(ts_regression_residual(SHARE, 20))",
    "ind_skew_relclose_60d": "cs_rank(ts_skew(REL_CLOSE, 60))",
    "ind_switch_bull60": "(cs_rank(ts_argmin(ts_delta(CLOSE, 1), 30)) if (ts_rank(BENCH_CLOSE, 60) > 0.7) else cs_rank(ts_argmax(CLOSE, 60)))",
    "ind_switch_volhi_ra_m6": "(cs_rank(ts_rank(REL_AMOUNT, 10)) if (ts_rank(ts_std(CLOSE, 20), 60) > 0.8) else cs_rank(ts_argmax(CLOSE, 60)))",
    "ind_trend_60d": "cs_rank(ts_argmax(CLOSE, 60))",
}
