"""
GP 符号回归原语集——Phase C 核心组件

设计思路：
--------
1. 为因子符号回归（Phase C GP）提供扩展原语集
2. 基础原语集（V1 已有）：+, -, *, /, abs, sqrt, rank, clip, inv, neg
3. V2 扩展 7 个量化 Alpha 常用时序原语：
   - ts_rank:   时序排名（值在过去 N 日的百分位排名）
   - ts_zscore: 时序标准化（过去 N 日的 Z-score）
   - delay:     延迟（N 日前的值）
   - correlation: 滚动相关性
   - decay_linear: 线性衰减加权平均
   - cs_rank:   横截面排名（与 rank 语义一致，预留接口）
   - signed_power: 有符号幂 sign(x) * |x|^a

4. 所有函数为纯 numpy 实现，无状态、无副作用，适合 GP 安全求值
5. 函数返回与输入同 shape 的 ndarray，NaN 安全

使用方式：
--------
>>> from factor.v2.gp_primitives import ts_rank, ts_zscore, delay
>>> result = ts_rank(factor_values, 20)  # factor_values: shape (T, N) 的因子值

依赖说明：
--------
仅依赖 numpy，不依赖 ft2 其他模块，可独立用于任何 numpy 数组运算。
"""

import numpy as np
from typing import Optional


# ============================================================================
# 扩展时序原语
# ============================================================================


def ts_rank(x: np.ndarray, period: int = 10) -> np.ndarray:
    """时序排名：值在过去 period 日的百分位排位
    
    [新增] 2026-05-20 Phase C GP 原语扩展
    对每个位置，计算其值在过去 period 个观测中的排名（0~1 之间）。
    period 个观测均相同时，排名为 0.5（取中间值）。
    
    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 回看窗口
    
    返回：
        np.ndarray: 排名数组，与前 period-1 个位置为 NaN
    """
    x = np.asarray(x, dtype=float)
    # [修复] 2026-05-28 类型安全：确保 scalar
    period = int(period)
    if x.ndim == 1:
        return _ts_rank_1d(x, period)
    else:
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = _ts_rank_1d(x[:, j], period)
        return result


def _ts_rank_1d(x: np.ndarray, period: int) -> np.ndarray:
    """1D 时序排名"""
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
        # 计算在最完整窗口中的排名（0~1）
        result[i] = np.searchsorted(np.sort(valid), val, side='left') / len(valid)
    return result


def ts_zscore(x: np.ndarray, period: int = 20) -> np.ndarray:
    """时序标准化：过去 period 日的 Z-score
    
    [新增] 2026-05-20 Phase C GP 原语扩展
    z = (x - rolling_mean(x, period)) / rolling_std(x, period)
    使用 ddof=1 计算标准差，避免除零返回 NaN。
    
    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 回看窗口
    
    返回：
        np.ndarray: Z-score 数组，与前 period-1 个位置为 NaN
    """
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
    """1D 时序标准化"""
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
    """延迟：period 日前的值
    
    [新增] 2026-05-20 Phase C GP 原语扩展
    用法类似 pandas.shift(period)，用于构造"过去 N 日的因子值"。
    注意：无前瞻偏差，delay(x, 1) 返回的是前一日值。
    
    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 延迟天数（≥1）
    
    返回：
        np.ndarray: 延迟后的数组，前 period 个位置为 NaN
    """
    x = np.asarray(x, dtype=float)
    # [修复] 2026-05-28 类型安全：period 可能来自表达式 AST 评估，需确保标量
    period = int(period)
    if period < 1:
        raise ValueError(f"period 必须 ≥ 1，当前: {period}")
    result = np.full_like(x, np.nan)
    if period < len(x):
        result[period:] = x[:-period]
    return result


def correlation(x: np.ndarray, y: np.ndarray,
                period: int = 20) -> np.ndarray:
    """滚动相关性：x 与 y 的 period 日滚动 Pearson 相关系数
    
    [新增] 2026-05-20 Phase C GP 原语扩展
    对每个时间点，计算过去 period 日 x 和 y 的线性相关系数。
    
    参数说明：
        x, y: 输入数组，shape 相同，如 (T,) 或 (T, N)
        period: 回看窗口
    
    返回：
        np.ndarray: 相关系数数组，与前 period-1 个位置为 NaN
    """
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


def _correlation_1d(x: np.ndarray, y: np.ndarray,
                    period: int) -> np.ndarray:
    """1D 滚动相关性"""
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
    """线性衰减加权平均：最近值权重大，最远值权重小
    
    [新增] 2026-05-20 Phase C GP 原语扩展
    权重 = [1, 2, 3, ..., period] / sum(1..period)
    即越近的观测权重越大（线性递增）。
    
    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 回看窗口
    
    返回：
        np.ndarray: 加权平均数组
    """
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


def _decay_linear_1d(x: np.ndarray, period: int,
                     weights: np.ndarray) -> np.ndarray:
    """1D 线性衰减加权平均"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[max(0, i - period + 1): i + 1]
        actual_len = min(len(window), period)
        w = weights[-actual_len:]  # 取最后 actual_len 个权重
        val = np.nansum(window[-actual_len:] * w)
        result[i] = val
    return result


def cs_rank(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """横截面排名：在每列（或行）内计算值的百分位排名
    
    [新增] 2026-05-20 Phase C GP 原语扩展
    与 ts_rank（时序排名）相对，cs_rank 在每个截面上做排名。
    默认 axis=-1 表示沿 columns 方向排名（每行独立计算排名）。
    对 2D 数组 (T, N)，axis=-1 计算每 T 时刻各 N 标的的排名。
    
    参数说明：
        x: 输入数组
        axis: 排名方向，默认 -1（每行的 columns 之间排名）
    
    返回：
        np.ndarray: 排名数组，0~1 之间
    """
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
        # 沿 columns 方向排名
        result = np.full_like(x, np.nan)
        for i in range(x.shape[0]):
            result[i, :] = _rank_1d(x[i, :])
        return result
    else:
        raise ValueError(f"cs_rank 目前仅支持 axis=-1（横截面），当前 axis={axis}")


def signed_power(x: np.ndarray, exponent: float = 2.0) -> np.ndarray:
    """有符号幂：sign(x) * |x|^exponent
    
    [新增] 2026-05-20 Phase C GP 原语扩展
    保留 x 的符号方向，同时放大差异。
    例如 signed_power(x, 2) = sign(x) * x^2，比纯平方多保留了符号信息。
    
    参数说明：
        x: 输入数组
        exponent: 指数（>= 0）
    
    返回：
        np.ndarray: 有符号幂结果
    """
    x = np.asarray(x, dtype=float)
    exponent = float(exponent)
    return np.sign(x) * np.power(np.abs(x), exponent)


# ============================================================================
# 191 因子扩展原语（滚动统计 + 复合滚动 + 特殊）
# [新增] 2026-05-30 适配 GTJA 191 Alpha 因子库探索
# ============================================================================

# ---- 滚动统计（单变量） ----

def ts_sum(x: np.ndarray, period: int = 10) -> np.ndarray:
    """滚动求和：过去 period 日的累加和

    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 回看窗口

    返回：
        np.ndarray: 滚动和数组，前 period-1 个位置为 NaN
    """
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
    # 用累积和 O(n) 计算滚动窗口和
    cumsum = np.nancumsum(np.where(np.isnan(x), 0, x))
    result[period - 1:] = cumsum[period - 1:]
    if period > 1:
        result[period - 1:] -= cumsum[:n - period + 1]
    # 窗口内有 NaN 的位置置为 NaN
    nan_count = np.zeros(n)
    nan_count[np.isnan(x)] = 1
    nan_cumsum = np.cumsum(nan_count)
    window_nan = nan_cumsum[period - 1:] - (nan_cumsum[:n - period + 1] if period > 1 else 0)
    result[period - 1:][window_nan > 0] = np.nan
    return result


def ts_mean(x: np.ndarray, period: int = 10) -> np.ndarray:
    """滚动均值：过去 period 日的算术平均

    [新增] 2026-05-30 191 因子扩展
    等价于 ts_sum(x, period) / period，但 NaN 处理更精细。
    对应 GTJA191 的 MEAN(x, d)。

    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 回看窗口

    返回：
        np.ndarray: 滚动均值数组
    """
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
    """滚动标准差：过去 period 日的标准差

    [新增] 2026-05-30 191 因子扩展
    使用 ddof=1（样本标准差）。对应 GTJA191 的 STD(x, d)。

    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 回看窗口

    返回：
        np.ndarray: 滚动标准差数组
    """
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
    """滚动最大值：过去 period 日的最大值

    [新增] 2026-05-30 191 因子扩展
    对应 GTJA191 的 MAX(x, d)。

    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 回看窗口

    返回：
        np.ndarray: 滚动最大值数组
    """
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
    """滚动最小值：过去 period 日的最小值

    [新增] 2026-05-30 191 因子扩展
    对应 GTJA191 的 MIN(x, d)。

    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 回看窗口

    返回：
        np.ndarray: 滚动最小值数组
    """
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
    """1D 滚动极值（max/min 共用）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[max(0, i - period + 1): i + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            continue
        result[i] = fn(valid)
    return result


# ---- 复合滚动（双变量 / 多参数） ----

def sma(x: np.ndarray, period: int = 10, lag: int = 0) -> np.ndarray:
    """简单移动平均（含延迟）：period 日均线，滞后 lag 日

    [新增] 2026-05-30 191 因子扩展
    SME(x, period, lag) = ts_mean(x_{t - lag - period + 1 : t - lag}, period)
    即先对原始序列做 period 日简单移动平均，再延迟 lag 日。
    对应 GTJA191 的 SMA(x, n, m)。

    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 均线窗口
        lag: 延迟天数（≥ 0，默认 0 即不延迟）

    返回：
        np.ndarray: 延迟后的移动均值数组
    """
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
    """先 delay 再 rolling mean，或等价地，用位移后的窗口"""
    n = len(x)
    result = np.full(n, np.nan)
    start = period - 1 + lag
    if start >= n:
        return result
    for i in range(start, n):
        # 窗口: [i - lag - period + 1, i - lag]，不含 t > i-lag 的数据
        window = x[max(0, i - lag - period + 1): i - lag + 1]
        valid = window[~np.isnan(window)]
        if len(valid) == 0:
            continue
        result[i] = np.nanmean(valid)
    return result


def covariance(x: np.ndarray, y: np.ndarray,
               period: int = 20) -> np.ndarray:
    """滚动协方差：x 与 y 的 period 日滚动样本协方差

    [新增] 2026-05-30 191 因子扩展
    对应 GTJA191 的 COVIANCE(x, y, d)。

    参数说明：
        x, y: 输入数组，shape 相同
        period: 回看窗口

    返回：
        np.ndarray: 滚动协方差数组
    """
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


def regbeta(x: np.ndarray, y: np.ndarray,
            period: int = 20) -> np.ndarray:
    """滚动回归 Beta：y 对 x 的 period 日滚动线性回归斜率

    [新增] 2026-05-30 191 因子扩展
    β = Cov(x, y) / Var(x)。对应 GTJA191 的 REGBETA(x, y, d)。

    参数说明：
        x, y: 输入数组，shape 相同
        period: 回看窗口

    返回：
        np.ndarray: 滚动回归 Beta 数组
    """
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


# ---- 特殊原语 ----

def ts_argmin(x: np.ndarray, period: int = 10) -> np.ndarray:
    """距 N 日最低点的天数：返回过去 period 日中最小值距今的天数

    [新增] 2026-05-30 191 因子扩展
    对应 GTJA191 的 LOWDAY(x, d)。返回值为 0 ~ period-1。
    若当天为最低点则返回 0，若 period-1 天前为最低点则返回 period-1。

    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 回看窗口

    返回：
        np.ndarray: 距离最低点的天数
    """
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
    """距 N 日最高点的天数：返回过去 period 日中最大值距今的天数

    [新增] 2026-05-30 191 因子扩展
    对应 GTJA191 的 HIGHDAY(x, d)。返回值为 0 ~ period-1。
    若当天为最高点则返回 0，若 period-1 天前为最高点则返回 period-1。

    参数说明：
        x: 输入数组，shape (T,) 或 (T, N)
        period: 回看窗口

    返回：
        np.ndarray: 距离最高点的天数
    """
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
    """1D 滚动 argmin/argmax 共用"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(period - 1, n):
        window = x[max(0, i - period + 1): i + 1]
        valid_mask = ~np.isnan(window)
        if valid_mask.sum() == 0:
            continue
        arg_in_window = fn(window[valid_mask])
        # arg 相对于有效值序列的位置，还原回窗口内原始位置
        valid_indices = np.where(valid_mask)[0]
        pos_in_window = valid_indices[arg_in_window]
        # 距离当前的天数 = (窗口长度 - 1) - 位置
        days_back = len(window) - 1 - pos_in_window
        result[i] = float(days_back)
    return result


def ifelse(cond: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """条件选择：cond > 0 时取 a，否则取 b

    [新增] 2026-05-30 191 因子扩展
    对应 GTJA191 的 IFELSE(cond, a, b)。
    三个输入广播到相同 shape。

    参数说明：
        cond: 条件数组，>0 为 True
        a: 条件为 True 时的取值
        b: 条件为 False 时的取值

    返回：
        np.ndarray: 与输入广播一致的结果
    """
    cond = np.asarray(cond, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(cond > 0, a, b)


# ============================================================================
# 原语元信息（供 GP 引擎引用）
# ============================================================================

# 扩展原语注册表：name -> (function, arity, is_safe)
# arity: 参数数量（不含 period 等超参数）
# is_safe: 是否不会产生 NaN/inf（False 表示需要安全检查）
EXTENDED_PRIMITIVES = {
    'ts_rank':       (ts_rank, 1, False),
    'ts_zscore':     (ts_zscore, 1, False),
    'delay':         (delay, 1, False),
    'correlation':   (correlation, 2, False),
    'decay_linear':  (decay_linear, 1, False),
    'cs_rank':       (cs_rank, 1, False),
    'signed_power':  (signed_power, 1, False),
    # 191 因子扩展
    'ts_sum':        (ts_sum, 1, False),
    'ts_mean':       (ts_mean, 1, False),
    'ts_std':        (ts_std, 1, False),
    'ts_max':        (ts_max, 1, False),
    'ts_min':        (ts_min, 1, False),
    'sma':           (sma, 1, False),
    'covariance':    (covariance, 2, False),
    'regbeta':       (regbeta, 2, False),
    'ts_argmin':     (ts_argmin, 1, False),
    'ts_argmax':     (ts_argmax, 1, False),
    'ifelse':        (ifelse, 3, False),
}

# 所有可安全求值的原语名称集合
SAFE_PRIMITIVES = {
    name for name, (_, _, is_safe) in EXTENDED_PRIMITIVES.items() if is_safe
}


def get_primitive(name: str):
    """按名称获取原语函数
    
    Args:
        name: 原语名称，如 'ts_rank'
    
    Returns:
        callable or None: 原语函数
    """
    info = EXTENDED_PRIMITIVES.get(name)
    return info[0] if info else None


def list_primitives() -> list:
    """列出所有扩展原语"""
    return [
        {'name': name, 'arity': info[1], 'safe': info[2]}
        for name, info in EXTENDED_PRIMITIVES.items()
    ]
