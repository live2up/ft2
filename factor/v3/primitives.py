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
from functools import wraps


# ============================================================================
# 维度处理装饰器（统一 1D/2D 分发逻辑）
# ============================================================================

def handle_dimension(func):
    """
    时序原语维度分发装饰器

    自动处理 1D/2D numpy 数组的输入分发：
    - 1D: 直接调用被装饰函数
    - 2D: 沿 axis=1 逐列调用，结果沿原 shape 拼接

    此外自动处理 window/int 类型的标量参数转换，避免 range() 报错。

    消除 19 个时序函数中重复的 if/else 分支代码（约 120-150 行重复样板），
    让核心计算逻辑成为函数主体，提升可读性和可维护性。

    使用方式:
        @handle_dimension
        def ts_mean_1d(x: np.ndarray, window: int) -> np.ndarray:
            # 只关心 1D 计算逻辑，无需处理 2D 分支
            ...

        # 调用方仍按原签名使用，自动支持 1D/2D
        ts_mean_1d(x_1d, window=20)        # OK
        ts_mean_1d(x_2d, window=20)        # OK，自动逐列计算

    Args:
        func: 接收 1D ndarray 作为第一参数的时序计算函数

    Returns:
        wrapper: 与 func 输入/输出一致的包装函数
    """
    @wraps(func)
    def wrapper(x, *args, **kwargs):
        x = np.asarray(x, dtype=float)
        # 自动转换 window 等 int 标量参数（从 float/int 表达式参数）
        new_args = []
        for i, arg in enumerate(args):
            if isinstance(arg, (int, np.integer)) or (isinstance(arg, float) and arg.is_integer()):
                new_args.append(int(arg))
            else:
                new_args.append(arg)
        # 同时处理 kwargs 中常见的 window 参数
        new_kwargs = {}
        for k, v in kwargs.items():
            if isinstance(v, (int, np.integer)) or (isinstance(v, float) and v.is_integer()):
                new_kwargs[k] = int(v)
            else:
                new_kwargs[k] = v
        if x.ndim == 1:
            return func(x, *new_args, **new_kwargs)
        # 2D: 沿 axis=1 逐列计算
        result = np.full_like(x, np.nan)
        for j in range(x.shape[1]):
            result[:, j] = func(x[:, j], *new_args, **new_kwargs)
        return result
    return wrapper


# ============================================================================
# 核心时序原语 (7 个)
# ============================================================================


def ts_rank(x: np.ndarray, window: int = 10) -> np.ndarray:
    """时序排名：值在过去 window 日的百分位排位"""
    return _ts_rank_impl(x, window)


@handle_dimension
def _ts_rank_impl(x: np.ndarray, window: int) -> np.ndarray:
    """时序排名 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[max(0, i - window + 1): i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) == 0:
            continue
        val = x[i]
        if np.isnan(val):
            continue
        result[i] = np.searchsorted(np.sort(valid), val, side='left') / len(valid)
    return result


def ts_zscore(x: np.ndarray, window: int = 20) -> np.ndarray:
    """时序标准化：过去 window 日的 Z-score"""
    return _ts_zscore_impl(x, window)


@handle_dimension
def _ts_zscore_impl(x: np.ndarray, window: int) -> np.ndarray:
    """时序标准化 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[max(0, i - window + 1): i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) < 2:
            continue
        mu = np.nanmean(valid)
        sigma = np.nanstd(valid, ddof=1)
        if sigma > 1e-10:
            result[i] = (x[i] - mu) / sigma
    return result


def delay(x: np.ndarray, window: int = 1) -> np.ndarray:
    """延迟：window 日前的值"""
    x = np.asarray(x, dtype=float)
    window = int(window)
    if window < 1:
        raise ValueError(f"window 必须 ≥ 1，当前: {window}")
    result = np.full_like(x, np.nan)
    if window < len(x):
        result[window:] = x[:-window]
    return result


def delta(x: np.ndarray, window: int = 1) -> np.ndarray:
    """差分：x_t - x_{t-window}（语法糖：省 1 层 AST 深度）

    等价于 sub(x, delay(x, window))，但只占 1 层而非 2 层。
    292 个公式中出现 139 次，是最高频的隐式原语。

    [新增] 2026-06-01 v3 高频语法糖
    """
    x = np.asarray(x, dtype=float)
    window = int(window)
    if window < 1:
        raise ValueError(f"window 必须 >= 1，当前: {window}")
    shifted = delay(x, window)
    return x - shifted


def correlation(x: np.ndarray, y: np.ndarray, window: int = 20) -> np.ndarray:
    """滚动相关性：x 与 y 的 window 日滚动 Pearson 相关系数"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    window = int(window)
    if x.shape != y.shape:
        raise ValueError(f"x.shape={x.shape} 与 y.shape={y.shape} 不匹配")
    if x.ndim == 1:
        return _correlation_impl(x, y, window)
    result = np.full_like(x, np.nan)
    for j in range(x.shape[1]):
        result[:, j] = _correlation_impl(x[:, j], y[:, j], window)
    return result


@handle_dimension
def _correlation_impl(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """滚动相关性 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        xw = x[max(0, i - window + 1): i + 1]
        yw = y[max(0, i - window + 1): i + 1]
        mask = ~np.isnan(xw) & ~np.isnan(yw)
        if mask.sum() < 3:
            continue
        corr = np.corrcoef(xw[mask], yw[mask])[0, 1]
        if not np.isnan(corr):
            result[i] = corr
    return result


def decay_linear(x: np.ndarray, window: int = 10) -> np.ndarray:
    """线性衰减加权平均：最近值权重大"""
    return _decay_linear_impl(x, window)


@handle_dimension
def _decay_linear_impl(x: np.ndarray, window: int) -> np.ndarray:
    """线性衰减加权平均 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    weights = np.arange(1, window + 1, dtype=float)
    weights = weights / weights.sum()
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[max(0, i - window + 1): i + 1]
        actual_len = min(len(seg), window)
        w = weights[-actual_len:]
        val = np.nansum(seg[-actual_len:] * w)
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


def cs_zscore(x: np.ndarray, window: int = 20) -> np.ndarray:
    """截面 Z-score 标准化（window 参数预留，当前未使用）"""
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


def ts_sum(x: np.ndarray, window: int = 10) -> np.ndarray:
    """滚动求和：过去 window 日的累加和"""
    return _ts_sum_impl(x, window)


@handle_dimension
def _ts_sum_impl(x: np.ndarray, window: int) -> np.ndarray:
    """滚动求和 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    if n < window:
        return result
    cumsum = np.nancumsum(np.where(np.isnan(x), 0, x))
    result[window - 1:] = cumsum[window - 1:]
    if window > 1:
        result[window - 1:] -= cumsum[:n - window + 1]
    nan_count = np.zeros(n)
    nan_count[np.isnan(x)] = 1
    nan_cumsum = np.cumsum(nan_count)
    window_nan = nan_cumsum[window - 1:] - (nan_cumsum[:n - window + 1] if window > 1 else 0)
    result[window - 1:][window_nan > 0] = np.nan
    return result


def ts_mean(x: np.ndarray, window: int = 10) -> np.ndarray:
    """滚动均值：过去 window 日的算术平均"""
    return _ts_mean_impl(x, window)


@handle_dimension
def _ts_mean_impl(x: np.ndarray, window: int) -> np.ndarray:
    """滚动均值 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[max(0, i - window + 1): i + 1]
        if np.all(np.isnan(seg)):
            continue
        result[i] = np.nanmean(seg)
    return result


def ts_std(x: np.ndarray, window: int = 20) -> np.ndarray:
    """滚动标准差：过去 window 日的标准差"""
    return _ts_std_impl(x, window)


@handle_dimension
def _ts_std_impl(x: np.ndarray, window: int) -> np.ndarray:
    """滚动标准差 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[max(0, i - window + 1): i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) < 2:
            continue
        result[i] = np.nanstd(valid, ddof=1)
    return result


def ts_max(x: np.ndarray, window: int = 10) -> np.ndarray:
    """滚动最大值：过去 window 日的最大值"""
    return _ts_max_impl(x, window)


@handle_dimension
def _ts_max_impl(x: np.ndarray, window: int) -> np.ndarray:
    """滚动最大值 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[max(0, i - window + 1): i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) == 0:
            continue
        result[i] = np.nanmax(valid)
    return result


def ts_min(x: np.ndarray, window: int = 10) -> np.ndarray:
    """滚动最小值：过去 window 日的最小值"""
    return _ts_min_impl(x, window)


@handle_dimension
def _ts_min_impl(x: np.ndarray, window: int) -> np.ndarray:
    """滚动最小值 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[max(0, i - window + 1): i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) == 0:
            continue
        result[i] = np.nanmin(valid)
    return result


# ============================================================================
# 191 因子扩展：复合滚动 + 特殊 (7 个)
# ============================================================================


def sma(x: np.ndarray, window: int = 10, offset: int = 0) -> np.ndarray:
    """简单移动平均（含延迟）"""
    return _sma_impl(x, window, offset)


@handle_dimension
def _sma_impl(x: np.ndarray, window: int, offset: int) -> np.ndarray:
    """简单移动平均 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    start = window - 1 + offset
    if start >= n:
        return result
    for i in range(start, n):
        seg = x[max(0, i - offset - window + 1): i - offset + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) == 0:
            continue
        result[i] = np.nanmean(valid)
    return result


def covariance(x: np.ndarray, y: np.ndarray, window: int = 20) -> np.ndarray:
    """滚动协方差"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    window = int(window)
    if x.shape != y.shape:
        raise ValueError(f"x.shape={x.shape} 与 y.shape={y.shape} 不匹配")
    if x.ndim == 1:
        return _covariance_impl(x, y, window)
    result = np.full_like(x, np.nan)
    for j in range(x.shape[1]):
        result[:, j] = _covariance_impl(x[:, j], y[:, j], window)
    return result


@handle_dimension
def _covariance_impl(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """滚动协方差 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        xw = x[max(0, i - window + 1): i + 1]
        yw = y[max(0, i - window + 1): i + 1]
        mask = ~np.isnan(xw) & ~np.isnan(yw)
        if mask.sum() < 2:
            continue
        result[i] = np.cov(xw[mask], yw[mask], ddof=1)[0, 1]
    return result


def regbeta(x: np.ndarray, y: np.ndarray, window: int = 20) -> np.ndarray:
    """滚动回归 Beta"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    window = int(window)
    if x.shape != y.shape:
        raise ValueError(f"x.shape={x.shape} 与 y.shape={y.shape} 不匹配")
    if x.ndim == 1:
        return _regbeta_impl(x, y, window)
    result = np.full_like(x, np.nan)
    for j in range(x.shape[1]):
        result[:, j] = _regbeta_impl(x[:, j], y[:, j], window)
    return result


@handle_dimension
def _regbeta_impl(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """滚动回归 Beta 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        xw = x[max(0, i - window + 1): i + 1]
        yw = y[max(0, i - window + 1): i + 1]
        mask = ~np.isnan(xw) & ~np.isnan(yw)
        if mask.sum() < 3:
            continue
        cov_mat = np.cov(xw[mask], yw[mask], ddof=1)
        var_x = cov_mat[0, 0]
        if var_x < 1e-10:
            continue
        result[i] = cov_mat[0, 1] / var_x
    return result


def ts_argmin(x: np.ndarray, window: int = 10) -> np.ndarray:
    """距 N 日最低点的天数"""
    return _ts_argmin_impl(x, window)


@handle_dimension
def _ts_argmin_impl(x: np.ndarray, window: int) -> np.ndarray:
    """距 N 日最低点 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    return _ts_argextreme(x, window, np.nanargmin)


def ts_argmax(x: np.ndarray, window: int = 10) -> np.ndarray:
    """距 N 日最高点的天数"""
    return _ts_argmax_impl(x, window)


@handle_dimension
def _ts_argmax_impl(x: np.ndarray, window: int) -> np.ndarray:
    """距 N 日最高点 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    return _ts_argextreme(x, window, np.nanargmax)


def _ts_argextreme(x: np.ndarray, window: int, fn) -> np.ndarray:
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[max(0, i - window + 1): i + 1]
        valid_mask = ~np.isnan(seg)
        if valid_mask.sum() == 0:
            continue
        arg_in_window = fn(seg[valid_mask])
        valid_indices = np.where(valid_mask)[0]
        pos_in_window = valid_indices[arg_in_window]
        days_back = len(seg) - 1 - pos_in_window
        result[i] = float(days_back)
    return result


def ifelse(cond: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """条件选择：cond > 0 时取 a，否则取 b"""
    cond = np.asarray(cond, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.where(cond > 0, a, b)


def ts_skew(x: np.ndarray, window: int = 20) -> np.ndarray:
    """滚动偏度：过去 window 日的分布偏度

    >0 = 右偏（涨多跌少，正收益为主导）
    <0 = 左偏（暴跌风险大）
    适合捕捉行业指数回报分布的不对称性。

    [新增] 2026-06-01 v3
    """
    return _ts_skew_impl(x, window)


@handle_dimension
def _ts_skew_impl(x: np.ndarray, window: int) -> np.ndarray:
    """滚动偏度 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[max(0, i - window + 1):i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) < 3:
            continue
        mu = np.nanmean(valid)
        sigma = np.nanstd(valid, ddof=1)
        if sigma < 1e-10:
            result[i] = 0.0
        else:
            result[i] = float(np.nanmean(((valid - mu) / sigma) ** 3))
    return result


def ts_regression_residual(x: np.ndarray, window: int | float = 20) -> np.ndarray:
    """时序线性回归残差：对过去 window 日做线性拟合，返回残差

    对每个截面对象独立计算：
        1. 取过去 window 日的时序数据
        2. 拟合线性回归 x = a + b * t (t为时间索引0,1,2,...)
        3. 返回当前值 - 预测值 (残差)

    残差>0 = 实际值高于线性趋势 = 超预期强势
    残差<0 = 实际值低于线性趋势 = 超预期弱势

    比 delta(x, window) 的优势：
        - delta 只看首尾两个点，受噪声影响大
        - ts_regression_residual 看所有点，对噪声更鲁棒

    比 div(x, ts_mean(x, window)) 的优势：
        - div(x, mean) = 当前值相对于历史均值
        - ts_regression_residual = 当前值相对于线性趋势(含方向)
        - 趋势上升时残差为负=低于预期(但可能高于均值)，更精细

    比 decay_linear(x, window) 的优势：
        - decay_linear = 加权均值(给定权重)
        - ts_regression_residual = 真实线性回归(不预设权重)

    使用示例:
        ts_regression_residual(amount, 20)  → 成交额偏离20日趋势
        ts_regression_residual(close, 10)   → 价格偏离10日趋势

    Note:
        window < 3 时返回全0 (样本不足无法回归)。
        线性回归需要至少3个有效数据点，否则残差设为0。
        回归只用时序最后一个点做预测（当前期）。

    [新增] 2026-06-07
    """
    x = np.asarray(x, dtype=float)
    window = int(window)
    if window < 3:
        return np.zeros_like(x)
    if x.ndim == 1:
        return _ts_regression_residual_impl(x, window)
    result = np.full_like(x, np.nan)
    for j in range(x.shape[1]):
        result[:, j] = _ts_regression_residual_impl(x[:, j], window)
    return result


@handle_dimension
def _ts_regression_residual_impl(x: np.ndarray, window: int) -> np.ndarray:
    """滚动线性回归残差 1D 实现（由 handle_dimension 自动分发 1D/2D）"""
    n = len(x)
    result = np.full(n, np.nan)
    for i in range(window - 1, n):
        seg = x[max(0, i - window + 1): i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) < 3:
            continue
        t = np.arange(len(valid), dtype=float)
        # Linear regression: x = a + b*t
        cov = np.cov(t, valid)
        if cov[0, 0] < 1e-15:
            result[i] = 0.0
            continue
        b = cov[0, 1] / cov[0, 0]
        a = np.mean(valid) - b * np.mean(t)
        predicted = a + b * (len(valid) - 1)  # predict the last point
        result[i] = valid[-1] - predicted
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
    'ts_regression_residual': (ts_regression_residual, 1),  # [新增] 2026-06-07
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


# ============================================================================
# 向后兼容别名 (deprecation-safe migration)
# ============================================================================
# [重构] 2026-06-19 handle_dimension 装饰器引入后，旧的 _xxx_1d 私有函数
# 名称被重命名为 _xxx_impl。保留 _xxx_1d 别名指向新实现，避免破坏任何
# 直接引用旧名称的外部代码（虽然按 Python 约定 _ 前缀属模块私有）。
#
# 何时删除：下个大版本（v4.0）可移除这些别名。

_ts_rank_1d = _ts_rank_impl
_ts_zscore_1d = _ts_zscore_impl
_correlation_1d = _correlation_impl
_decay_linear_1d = _decay_linear_impl
_ts_sum_1d = _ts_sum_impl
_ts_mean_1d = _ts_mean_impl
_ts_std_1d = _ts_std_impl
_ts_max_1d = _ts_max_impl
_ts_min_1d = _ts_min_impl
_sma_1d = _sma_impl
_covariance_1d = _covariance_impl
_regbeta_1d = _regbeta_impl
_ts_argmin_1d = _ts_argmin_impl
_ts_argmax_1d = _ts_argmax_impl
_ts_argextreme_1d = _ts_argextreme
_ts_skew_1d = _ts_skew_impl
_ts_regression_residual_1d = _ts_regression_residual_impl
