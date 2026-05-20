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
        exponent: 指数（≥ 0）
    
    返回：
        np.ndarray: 有符号幂结果
    """
    x = np.asarray(x, dtype=float)
    return np.sign(x) * np.power(np.abs(x), exponent)


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
