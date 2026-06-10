"""
signals/v4/registry.py — 函数注册表
=============================================================================
Python AST DSL 的所有可用函数。每个函数接收 numpy 数组，返回 numpy 数组。

原则：
  - 所有滚动窗口只用历史数据（无前向偏差）
  - 签名统一：时序函数 (x, window)，双输入时序 (x, y, window)
  - 冷启动保护：前 window-1 天返回 NaN
=============================================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, Callable, Any


# ============================================================
# 工具函数
# ============================================================

def _rolling(x: np.ndarray, window: int, func, *args, **kwargs):
    """通用滚动窗口计算（仅历史数据）"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(window - 1, len(x)):
        result[i] = func(x[i - window + 1 : i + 1], *args, **kwargs)
    return result


def _expanding(x: np.ndarray, func, min_periods: int = 20, *args, **kwargs):
    """扩张窗口计算（仅历史数据）"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, np.nan, dtype=float)
    for i in range(min_periods - 1, len(x)):
        result[i] = func(x[: i + 1], *args, **kwargs)
    return result


def _rolling_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """滚动 Pearson 相关系数"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    result = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        xw, yw = x[i - window + 1 : i + 1], y[i - window + 1 : i + 1]
        std_x, std_y = np.std(xw, ddof=0), np.std(yw, ddof=0)
        if std_x < 1e-10 or std_y < 1e-10:
            result[i] = 0.0
        else:
            result[i] = np.corrcoef(xw, yw)[0, 1]
    return result


def _persist(x: np.ndarray, n: int) -> np.ndarray:
    """连续 n 日同向（>0）才输出 1.0，否则 0"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, 0.0)
    signal = x > 0
    for i in range(n - 1, len(x)):
        if np.all(signal[i - n + 1 : i + 1]):
            result[i] = 1.0
    return result


# ============================================================
# 时序函数
# ============================================================

def ts_mean(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling(x, window, np.mean)

def ts_std(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling(x, window, np.std, ddof=0)

def ts_sum(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling(x, window, np.sum)

def ts_max(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling(x, window, np.max)

def ts_min(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling(x, window, np.min)

def ts_median(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling(x, window, np.median)

def ts_delta(x: np.ndarray, window: int) -> np.ndarray:
    """x[t] - x[t-window]"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, np.nan)
    result[window:] = x[window:] - x[:-window]
    return result

def ts_delay(x: np.ndarray, window: int) -> np.ndarray:
    """滞后 window 天的值"""
    x = np.asarray(x, dtype=float)
    result = np.full_like(x, np.nan)
    result[window:] = x[:-window]
    return result

def ts_rank(x: np.ndarray, window: int) -> np.ndarray:
    """滚动窗口内的百分位排名 [0, 1]"""
    return _rolling(x, window, lambda w: (np.searchsorted(np.sort(w), w[-1]) + 1) / len(w))

def ts_corr(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    return _rolling_corr(x, y, window)

def ts_cov(x: np.ndarray, y: np.ndarray, window: int) -> np.ndarray:
    """滚动协方差"""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    result = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        result[i] = np.cov(x[i - window + 1 : i + 1], y[i - window + 1 : i + 1], ddof=0)[0, 1]
    return result

def ts_skew(x: np.ndarray, window: int) -> np.ndarray:
    """滚动偏度"""
    return _rolling(x, window, lambda w: _safe_skew(w))

def _safe_skew(w: np.ndarray) -> float:
    std = np.std(w, ddof=0)
    if std < 1e-10:
        return 0.0
    return np.mean(((w - np.mean(w)) / std) ** 3)

def ts_kurt(x: np.ndarray, window: int) -> np.ndarray:
    """滚动峰度（超额峰度，正态=0）"""
    return _rolling(x, window, lambda w: _safe_kurt(w))

def _safe_kurt(w: np.ndarray) -> float:
    std = np.std(w, ddof=0)
    if std < 1e-10:
        return 0.0
    return np.mean(((w - np.mean(w)) / std) ** 4) - 3.0

def ts_argmax(x: np.ndarray, window: int) -> np.ndarray:
    """窗口内最大值距今的天数（0=今天）"""
    return _rolling(x, window, lambda w: len(w) - 1 - np.argmax(w))

def ts_argmin(x: np.ndarray, window: int) -> np.ndarray:
    return _rolling(x, window, lambda w: len(w) - 1 - np.argmin(w))


# ============================================================
# 扩张统计函数（无前向偏差，用于阈值判断）
# ============================================================

def expanding_mean(x: np.ndarray, min_periods: int = 20) -> np.ndarray:
    return _expanding(x, np.mean, min_periods)

def expanding_median(x: np.ndarray, min_periods: int = 20) -> np.ndarray:
    return _expanding(x, np.median, min_periods)

def expanding_std(x: np.ndarray, min_periods: int = 20) -> np.ndarray:
    return _expanding(x, lambda w: np.std(w, ddof=0), min_periods)

def expanding_percentile(x: np.ndarray, p: float, min_periods: int = 20) -> np.ndarray:
    """扩张窗口 p 分位数（p 取值 0~1）"""
    return _expanding(x, lambda w: np.percentile(w, p * 100), min_periods)


# ============================================================
# 截面函数（单品种择时中意义有限，保留用于多品种扩展）
# ============================================================

def cs_rank(x: np.ndarray) -> np.ndarray:
    """截面排名 [0, 1]（2D 输入时按行排名；1D 返回 0.5）"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return np.full_like(x, 0.5)
    from scipy.stats import rankdata
    result = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        valid = ~np.isnan(x[i])
        if valid.sum() > 0:
            ranks = rankdata(x[i][valid]) / valid.sum()
            result[i][valid] = ranks
    return result

def cs_zscore(x: np.ndarray) -> np.ndarray:
    """截面 Z-score（2D 输入时按行标准化；1D 返回 0）"""
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        return np.zeros_like(x)
    result = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        valid = ~np.isnan(x[i])
        if valid.sum() > 1:
            mean = np.mean(x[i][valid])
            std = np.std(x[i][valid], ddof=0)
            if std > 1e-10:
                result[i][valid] = (x[i][valid] - mean) / std
            else:
                result[i][valid] = 0.0
    return result


# ============================================================
# 数学函数
# ============================================================

def safe_abs(x: np.ndarray) -> np.ndarray:
    return np.abs(x)

def safe_log(x: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(np.abs(x), 1e-10))

def safe_sqrt(x: np.ndarray) -> np.ndarray:
    return np.sqrt(np.maximum(x, 0.0))

def safe_sign(x: np.ndarray) -> np.ndarray:
    return np.sign(x)

def safe_exp(x: np.ndarray) -> np.ndarray:
    return np.exp(np.clip(x, -50, 50))

def safe_tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)

def safe_sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -500, 500)
    return 1.0 / (1.0 + np.exp(-clipped))

def safe_relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# ============================================================
# 信号函数
# ============================================================

def persist(x: np.ndarray, n: int = 3) -> np.ndarray:
    """连续 n 日同向（>0）才触发"""
    return _persist(x, n)


# ============================================================
# 函数注册表
# ============================================================

FUNC_REGISTRY: Dict[str, Callable] = {
    # 时序函数（按 symbol 滚动；单品种场景等同于全序列滚动）
    'ts_mean':   ts_mean,
    'ts_std':    ts_std,
    'ts_sum':    ts_sum,
    'ts_max':    ts_max,
    'ts_min':    ts_min,
    'ts_median': ts_median,
    'ts_delta':  ts_delta,
    'ts_delay':  ts_delay,
    'ts_rank':   ts_rank,
    'ts_corr':   ts_corr,
    'ts_cov':    ts_cov,
    'ts_skew':   ts_skew,
    'ts_kurt':   ts_kurt,
    'ts_argmax': ts_argmax,
    'ts_argmin': ts_argmin,
    
    # 扩张统计
    'expanding_mean':       expanding_mean,
    'expanding_median':     expanding_median,
    'expanding_std':        expanding_std,
    'expanding_percentile': expanding_percentile,
    
    # 截面函数
    'cs_rank':   cs_rank,
    'cs_zscore': cs_zscore,
    
    # 数学函数
    'abs':     safe_abs,
    'log':     safe_log,
    'sqrt':    safe_sqrt,
    'sign':    safe_sign,
    'exp':     safe_exp,
    'tanh':    safe_tanh,
    'sigmoid': safe_sigmoid,
    'relu':    safe_relu,
    
    # 信号确认
    'persist': persist,
}

# ============================================================
# 安全常量
# ============================================================

SAFE_CONSTANTS = {
    'True': 1.0,
    'False': 0.0,
    'None': 0.0,
    'pi': np.pi,
    'e': np.e,
}

# ============================================================
# 允许的变量名前缀（白名单）
# ============================================================

VALID_VAR_PREFIXES = [
    # 原始 OHLCV
    'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT', 'VWAP', 'RETURNS',
    # 特征前缀（通过 _ 拆分识别）
    'ATR', 'STDDEV', 'BBWIDTH', 'BBW', 'HV', 'NATR',
    'TRIMA', 'SMA', 'MA', 'EMA', 'TSF', 'WMA', 'DEMA', 'KAMA',
    'ADX', 'RSI', 'CCI', 'MACD', 'MFI', 'ULTOSC', 'ROC', 'MOM_RATIO',
    'TREND_STRENGTH', 'LINEARREG', 'VAR',
    'VOL_RATIO', 'VOL_CHG', 'VOL_REGIME', 'OBV',
    'AVGPRICE', 'WCLPRICE', 'CORREL', 'HT_SINE', 'HT_TRENDMODE',
    'MOM_CHG', 'UP_RATIO',
]

def is_valid_variable(name: str) -> bool:
    """检查变量名是否在白名单中"""
    upper = name.upper()
    # 精确匹配（单段名称，如 CLOSE, OPEN）
    if upper in VALID_VAR_PREFIXES:
        return True
    # 前缀匹配（如 RSI_14, ATR_7, EMA_20, VOL_RATIO_5_20）
    for prefix in VALID_VAR_PREFIXES:
        if upper.startswith(prefix + '_'):
            # 剩余部分必须全是数字或数字+下划线（如 5_20）
            rest = upper[len(prefix) + 1:]
            if rest and all(c.isdigit() or c == '_' for c in rest):
                return True
    return False
