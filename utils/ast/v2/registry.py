# -*- coding: utf-8 -*-
"""
utils/ast/registry.py — 兼容重导出枢纽

[重构] 2026-06-22 权威源已拆分为:
  - functions.py  → FUNC_REGISTRY, 72 函数原语, register_function
  - variables.py  → VALID_VAR_PREFIXES, is_valid_variable, register_variable

此文件保留向后兼容, 所有旧 import 路径仍生效:
  from utils.ast.registry import FUNC_REGISTRY, is_valid_variable

新代码建议直接导入:
  from utils.ast.functions import FUNC_REGISTRY
  from utils.ast.variables import is_valid_variable
"""
from .functions import *  # noqa: F401,F403
from .variables import *  # noqa: F401,F403

# 显式重导出 (避免 wildcard 导入的 lint 警告)
from .functions import (
    FUNC_REGISTRY, SAFE_CONSTANTS,
    register_function, unregister_function,
    _rolling, _expanding, _persist,
    ts_mean, ts_std, ts_sum, ts_max, ts_min, ts_median,
    ts_delta, ts_delay, ts_rank, ts_corr, ts_cov,
    ts_skew, ts_kurt, ts_argmax, ts_argmin, ts_roc, ts_zscore,
    ts_scale, ts_quantile, ts_av_diff, ts_decay_linear, ts_product,
    ts_resid,
    ts_slope, ts_rsq, ts_intercept, ts_predict,
    expanding_mean, expanding_median, expanding_std, expanding_percentile,
    cs_rank, cs_zscore, cs_scale, cs_winsorize, cs_quantile,
    safe_abs, safe_log, safe_sqrt, safe_sign, safe_exp, safe_tanh,
    safe_sigmoid, safe_relu, signed_power, safe_max, safe_min,
    persist,
    FUNC_CATEGORIES, get_func_category,
)
from .variables import (
    VALID_VAR_PREFIXES, is_valid_variable,
    register_variable, unregister_variable,
    VAR_CATEGORIES, get_var_category,
)
