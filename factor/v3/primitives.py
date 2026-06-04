"""
factor/v3/primitives.py — GP 符号回归原语集
=============================================================================

19 个时序/截面原语，全部纯 numpy 实现，无状态、无副作用。
从 v2/gp_primitives.py 直接移植。

原语分类：
  核心 7 个: ts_rank, ts_zscore, delay, correlation, decay_linear, cs_rank, signed_power
  191 扩展 12 个: ts_sum, ts_mean, ts_std, ts_max, ts_min, sma, covariance,
                   regbeta, ts_argmin, ts_argmax, ifelse, cs_zscore

使用方式：
>>> from factor.v3.primitives import ts_rank, delay
>>> result = ts_rank(factor_values, 20)  # factor_values: shape (T, N)

[移植] 2026-06-01 从 v2/gp_primitives.py 移植到 v3/primitives.py
=============================================================================
"""

import numpy as np
from typing import Optional


# ============================================================================
# 核心时序原语 (7 个)
# ============================================================================


def ts_rank(x: np.ndarray, period: int = 10) -> np.ndarray:
    """时序排名：值在过去 period 日的百分位排位"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    if x.ndim == 1:
        return _ts_rank_1d(x, period)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _ts_rank_1d(x[:, j], period)
        return result


def _ts_rank_1d(x: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[max(0, i - period + 1): i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            continue
        val = x[i]
        if np.isnan(val):
            continue
        result[i] = np.searchsorted(np.sort(valid), val, side='left') / len(valid)
    return result


def ts_zscore(x: np.ndarray, period: int = 20) -> np.ndarray:
    """时序标准化：过去 period 日的 Z-score"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    if x.ndim == 1:
        return _ts_zscore_1d(x, period)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _ts_zscore_1d(x[:, j], period)
        return result


def _ts_zscore_1d(x: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[max(0, i - period + 1): i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        mu = np.nanmean(valid)
        sigma = np.nanstd(valid, ddof=1)
        if sigma > 1e-10:
            result[i] = (x[i] - mu) / sigma
    return result


def delay(x: np.ndarray, period: int = 1) -> np.ndarray:
    """延迟：period 日前的值"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    if period < 1:
        raise ValueError(f"period 必须 ≥ 1，当前: {period}")
    result = np.full_like(x, np.nan)
    if period < len(x):
        result[period:] = x[:-period]
    return result


def delta(x: np.ndarray, period: int = 1) -> np.ndarray:
    """差分：x_t - x_{t-period}（语法糖：省 1 层 AST 深度）

    等价于 sub(x, delay(x, period))，但只占 1 层而非 2 层。
    292 个公式中出现 139 次，是最高频的隐式原语。

    [新增] 2026-06-01 v3 高频语法糖
    """
    x = np.asarray(x, dtype=float)
    period = int(period)
    if period < 1:
        raise ValueError(f"period 必须 >= 1，当前: {period}")
    shifted = delay(x, period)
    return x - shifted


def correlation(x: np.ndarray, y: np.ndarray, period: int = 20) -> np.ndarray:
    """滚动相关性：x 与 y 的 period 日滚动 Pearson 相关系数"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    period = int(period)
    if x.shape != y.shape:
        raise ValueError(f"x.shape={x.shape} 与 y.shape={y.shape} 不匹配")
    if x.ndim == 1:
        return _correlation_1d(x, y, period)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _correlation_1d(x[:, j], y[:, j], period)
        return result


def _correlation_1d(x: np.ndarray, y: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        xw = x[max(0, i - period + 1): i + 1]
        yw = y[max(0, i - period + 1): i + 1]
        mask = ~np.isnan(xw) & ~np.isnan(yw)
        if mask.sum() < 3:
            continue
        corr = np.corrcoef(xw[mask], yw[mask])[0, 1]
        if not np.isnan(corr):
            result[i] = corr
    return result


def decay_linear(x: np.ndarray, period: int = 10) -> np.ndarray:
    """线性衰减加权平均：最近值权重大"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    weights = np.arange(1, period + 1, dtype=float)
    weights = weights / weights.sum()
    if x.ndim == 1:
        return _decay_linear_1d(x, period, weights)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _decay_linear_1d(x[:, j], period, weights)
        return result


def _decay_linear_1d(x: np.ndarray, period: int, weights: np.ndarray) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[max(0, i - period + 1): i + 1]
        actual_len = min(len(window), period)
        w = weights[-actual_len:]
        val = np.nansum(window[-actual_len:] * w)
        result[i] = val
    return result


def cs_rank(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """横截面排名：在每行内计算百分位排名"""
    x = np.asarray(x, dtype=float)

    def _rank_1d(arr):
        valid = ~np.isnan(arr)
        result = np.full_like(arr, np.nan, dtype=float)
        if valid.sum() == 0:
            return result
        valid_vals = arr[valid]
        ranks = np.searchsorted(np.sort(valid_vals), valid_vals, side='left') / len(valid_vals)
        result[valid] = ranks
        return result

    if axis == -1 or axis == 1:
        result = np.full_like(x, np.nan)
        for i in range(x.shape[0]):
            result[i, :] = _rank_1d(x[i, :])
        return result
    else:
        raise ValueError(f"cs_rank 目前仅支持 axis=-1（横截面），当前 axis={axis}")


def cs_zscore(x: np.ndarray, period: int = 20) -> np.ndarray:
    """截面 Z-score 标准化（period 参数预留，当前未使用）"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        row = x[i, :]
        valid = row[~np.isnan(row)]
        if len(valid) < 2:
            continue
        mu = np.nanmean(valid)
        sigma = np.nanstd(valid, ddof=1)
        if sigma < 1e-10:
            result[i, :] = 0.0
        else:
            result[i, :] = (row - mu) / sigma
    return result


def signed_power(x: np.ndarray, exponent: float = 2.0) -> np.ndarray:
    """有符号幂：sign(x) * |x|^exponent"""
    x = np.asarray(x, dtype=float)
    exponent = float(exponent)
    return np.sign(x) * np.power(np.abs(x), exponent)


def winsorize(x: np.ndarray, n: float = 3.0) -> np.ndarray:
    """Winsorize 截尾：clamp 到 ±n 倍标准差

    抑制离群值（如行业单日暴涨 10%），提高因子稳健性。
    逐截面（逐行）计算阈值，避免全局统计导致的前瞻偏差。

    [新增] 2026-06-01 v3
    [修复] 2026-06-03 改为逐截面 winsorize，消除全局统计前瞻偏差
    """
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, np.nan)
    if x.ndim == 1:
        mu = float(np.nanmean(x))
        sig = float(np.nanstd(x))
        if sig < 1e-10:
            return np.full_like(x, mu)
        return np.clip(x, mu - n * sig, mu + n * sig)
    for i in range(x.shape[0]):
        row = x[i, :]
        valid = row[~np.isnan(row)]
        if len(valid) < 2:
            continue
        mu = float(np.nanmean(valid))
        sig = float(np.nanstd(valid))
        if sig < 1e-10:
            result[i, :] = mu
        else:
            result[i, :] = np.clip(row, mu - n * sig, mu + n * sig)
    return result


def cs_mean(x: np.ndarray) -> np.ndarray:
    """截面均值：每行中有值股票的均值，广播回 (T,N)

    [新增] 2026-06-01 v3 行业中性化
    """
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        row = x[i, :]
        valid = row[~np.isnan(row)]
        if len(valid) == 0:
            continue
        result[i, :] = np.nanmean(valid)
    return result


# ============================================================================
# 191 因子扩展：滚动统计 (5 个)
# ============================================================================


def ts_sum(x: np.ndarray, period: int = 10) -> np.ndarray:
    """滚动求和：过去 period 日的累加和"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    if x.ndim == 1:
        return _ts_sum_1d(x, period)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _ts_sum_1d(x[:, j], period)
        return result


def _ts_sum_1d(x: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    if n < period:
        return result
    cumsum = np.nancumsum(np.where(np.isnan(x), 0, x))
    result[period - 1:] = cumsum[period - 1:]
    if period > 1:
        result[period - 1:] -= cumsum[:n - period + 1]
    nan_count = np.zeros(n)
    nan_count[np.isnan(x)] = 1
    nan_cumsum = np.cumsum(nan_count)
    window_nan = nan_cumsum[period - 1:] - (nan_cumsum[:n - period + 1] if period > 1 else 0)
    result[period - 1:][window_nan > 0] = np.nan
    return result


def ts_mean(x: np.ndarray, period: int = 10) -> np.ndarray:
    """滚动均值：过去 period 日的算术平均"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    if x.ndim == 1:
        return _ts_mean_1d(x, period)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _ts_mean_1d(x[:, j], period)
        return result


def _ts_mean_1d(x: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[max(0, i - period + 1): i + 1]
        if np.all(np.isnan(window)):
            continue
        result[i] = np.nanmean(window)
    return result


def ts_std(x: np.ndarray, period: int = 20) -> np.ndarray:
    """滚动标准差：过去 period 日的标准差"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    if x.ndim == 1:
        return _ts_std_1d(x, period)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _ts_std_1d(x[:, j], period)
        return result


def _ts_std_1d(x: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[max(0, i - period + 1): i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 2:
            continue
        result[i] = np.nanstd(valid, ddof=1)
    return result


def ts_max(x: np.ndarray, period: int = 10) -> np.ndarray:
    """滚动最大值：过去 period 日的最大值"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    if x.ndim == 1:
        return _ts_extreme_1d(x, period, np.nanmax)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _ts_extreme_1d(x[:, j], period, np.nanmax)
        return result


def ts_min(x: np.ndarray, period: int = 10) -> np.ndarray:
    """滚动最小值：过去 period 日的最小值"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    if x.ndim == 1:
        return _ts_extreme_1d(x, period, np.nanmin)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _ts_extreme_1d(x[:, j], period, np.nanmin)
        return result


def _ts_extreme_1d(x: np.ndarray, period: int, fn) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[max(0, i - period + 1): i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            continue
        result[i] = fn(valid)
    return result


# ============================================================================
# 191 因子扩展：复合滚动 + 特殊 (7 个)
# ============================================================================


def sma(x: np.ndarray, period: int = 10, lag: int = 0) -> np.ndarray:
    """简单移动平均（含延迟）"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    lag = int(lag)
    if x.ndim == 1:
        return _sma_1d(x, period, lag)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _sma_1d(x[:, j], period, lag)
        return result


def _sma_1d(x: np.ndarray, period: int, lag: int) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    start = period - 1 + lag
    if start >= n:
        return result
    for i in range(start, n):
        window = x[max(0, i - lag - period + 1): i - lag + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            continue
        result[i] = np.nanmean(valid)
    return result


def covariance(x: np.ndarray, y: np.ndarray, period: int = 20) -> np.ndarray:
    """滚动协方差"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    period = int(period)
    if x.shape != y.shape:
        raise ValueError(f"x.shape={x.shape} 与 y.shape={y.shape} 不匹配")
    if x.ndim == 1:
        return _covariance_1d(x, y, period)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _covariance_1d(x[:, j], y[:, j], period)
        return result


def _covariance_1d(x: np.ndarray, y: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        xw = x[max(0, i - period + 1): i + 1]
        yw = y[max(0, i - period + 1): i + 1]
        mask = ~np.isnan(xw) & ~np.isnan(yw)
        if mask.sum() < 2:
            continue
        result[i] = np.cov(xw[mask], yw[mask], ddof=1)[0, 1]
    return result


def regbeta(x: np.ndarray, y: np.ndarray, period: int = 20) -> np.ndarray:
    """滚动回归 Beta"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    period = int(period)
    if x.shape != y.shape:
        raise ValueError(f"x.shape={x.shape} 与 y.shape={y.shape} 不匹配")
    if x.ndim == 1:
        return _regbeta_1d(x, y, period)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _regbeta_1d(x[:, j], y[:, j], period)
        return result


def _regbeta_1d(x: np.ndarray, y: np.ndarray, period: int) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        xw = x[max(0, i - period + 1): i + 1]
        yw = y[max(0, i - period + 1): i + 1]
        mask = ~np.isnan(xw) & ~np.isnan(yw)
        if mask.sum() < 3:
            continue
        cov_mat = np.cov(xw[mask], yw[mask], ddof=1)
        var_x = cov_mat[0, 0]
        if var_x < 1e-10:
            continue
        result[i] = cov_mat[0, 1] / var_x
    return result


def ts_argmin(x: np.ndarray, period: int = 10) -> np.ndarray:
    """距 N 日最低点的天数"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    if x.ndim == 1:
        return _ts_argextreme_1d(x, period, np.nanargmin)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _ts_argextreme_1d(x[:, j], period, np.nanargmin)
        return result


def ts_argmax(x: np.ndarray, period: int = 10) -> np.ndarray:
    """距 N 日最高点的天数"""
    x = np.asarray(x, dtype=float)
    period = int(period)
    if x.ndim == 1:
        return _ts_argextreme_1d(x, period, np.nanargmax)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _ts_argextreme_1d(x[:, j], period, np.nanargmax)
        return result


def _ts_argextreme_1d(x: np.ndarray, period: int, fn) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[max(0, i - period + 1): i + 1]
        valid_mask = ~np.isnan(window)
        if valid_mask.sum() == 0:
            continue
        arg_in_window = fn(window[valid_mask])
        valid_indices = np.where(valid_mask)[0]
        pos_in_window = valid_indices[arg_in_window]
        days_back = len(window) - 1 - pos_in_window
        result[i] = float(days_back)
    return result


def ifelse(cond: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """条件选择：cond > 0 时取 a，否则取 b"""
    cond = np.asarray(cond, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(cond > 0, a, b)


def ts_skew(x: np.ndarray, period: int = 20) -> np.ndarray:
    """滚动偏度：过去 period 日的分布偏度

    >0 = 右偏（涨多跌少，正收益为主导）
    <0 = 左偏（暴跌风险大）
    适合捕捉行业指数回报分布的不对称性。

    [新增] 2026-06-01 v3
    """
    x = np.asarray(x, dtype=float)
    period = int(period)
    if x.ndim == 1:
        return _ts_skew_1d(x, period)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _ts_skew_1d(x[:, j], period)
        return result


def _ts_skew_1d(x: np.ndarray, period: int) -> np.ndarray:
    """单列滚动偏度"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[max(0, i - period + 1):i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) < 3:
            continue
        mu = np.nanmean(valid)
        sigma = np.nanstd(valid, ddof=1)
        if sigma < 1e-10:
            result[i] = 0.0
        else:
            result[i] = float(np.nanmean(((valid - mu) / sigma) ** 3))
    return result


# ============================================================================
# 原语注册表
# ============================================================================

EXTENDED_PRIMITIVES = {
    'ts_rank':       (ts_rank, 1),
    'ts_zscore':     (ts_zscore, 1),
    'delay':         (delay, 1),
    'delta':         (delta, 1),           # [新增] v3 语法糖
    'correlation':   (correlation, 2),
    'decay_linear':  (decay_linear, 1),
    'cs_rank':       (cs_rank, 1),
    'cs_zscore':     (cs_zscore, 1),
    'cs_mean':       (cs_mean, 1),         # [新增] v3
    'signed_power':  (signed_power, 1),
    'winsorize':     (winsorize, 1),       # [新增] v3
    'ts_sum':        (ts_sum, 1),
    'ts_mean':       (ts_mean, 1),
    'ts_std':        (ts_std, 1),
    'ts_max':        (ts_max, 1),
    'ts_min':        (ts_min, 1),
    'sma':           (sma, 1),
    'covariance':    (covariance, 2),
    'regbeta':       (regbeta, 2),
    'ts_argmin':     (ts_argmin, 1),
    'ts_argmax':     (ts_argmax, 1),
    'ifelse':        (ifelse, 3),
    'ts_skew':       (ts_skew, 1),         # [新增] v3
}


def get_primitive(name: str):
    """按名称获取原语函数"""
    info = EXTENDED_PRIMITIVES.get(name)
    return info[0] if info else None


def list_primitives() -> list:
    """列出所有扩展原语"""
    return [
        {'name': name, 'arity': info[1]}
        for name, info in EXTENDED_PRIMITIVES.items()
    ]
