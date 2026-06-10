"""
signals/v4/registry.py — 函数注册表
=============================================================================
Python AST DSL 的所有可用函数。每个函数接收 numpy 数组，返回 numpy 数组。

原则：
  - 所有滚动窗口只用历史数据（无前向偏差）
  - 冷启动保护：前 window-1 天返回 NaN
=============================================================================
"""
import numpy as np
from typing import Dict, Callable


def _rolling(x: np.ndarray, window: int, func, *a, **kw):
    x = np.asarray(x, dtype=float)
    r = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        r[i] = func(x[i - window + 1 : i + 1], *a, **kw)
    return r


def _expanding(x: np.ndarray, func, min_p: int = 20, *a, **kw):
    x = np.asarray(x, dtype=float)
    r = np.full_like(x, np.nan)
    for i in range(min_p - 1, len(x)):
        r[i] = func(x[: i + 1], *a, **kw)
    return r


def _persist(x: np.ndarray, n: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    r = np.full_like(x, 0.0)
    s = x > 0
    for i in range(n - 1, len(x)):
        if np.all(s[i - n + 1 : i + 1]):
            r[i] = 1.0
    return r


# ============================================================
# 时序函数
# ============================================================

def ts_mean(x, w):       return _rolling(x, w, np.mean)
def ts_std(x, w):        return _rolling(x, w, lambda a: np.std(a, ddof=0))
def ts_sum(x, w):        return _rolling(x, w, np.sum)
def ts_max(x, w):        return _rolling(x, w, np.max)
def ts_min(x, w):        return _rolling(x, w, np.min)
def ts_median(x, w):     return _rolling(x, w, np.median)

def ts_delta(x, w):
    x = np.asarray(x, float); r = np.full_like(x, np.nan)
    r[w:] = x[w:] - x[:-w]; return r

def ts_delay(x, w):
    x = np.asarray(x, float); r = np.full_like(x, np.nan)
    r[w:] = x[:-w]; return r

def ts_rank(x, w):
    return _rolling(x, w, lambda a: (np.searchsorted(np.sort(a), a[-1]) + 1) / len(a))

def ts_corr(x, y, w):
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = np.full_like(x, np.nan)
    for i in range(w - 1, len(x)):
        xw = x[i - w + 1 : i + 1]; yw = y[i - w + 1 : i + 1]
        sx, sy = np.std(xw, ddof=0), np.std(yw, ddof=0)
        r[i] = 0.0 if sx < 1e-10 or sy < 1e-10 else np.corrcoef(xw, yw)[0, 1]
    return r

def ts_skew(x, w):
    def f(a):
        s = np.std(a, ddof=0)
        return 0.0 if s < 1e-10 else np.mean(((a - np.mean(a)) / s) ** 3)
    return _rolling(x, w, f)

def ts_kurt(x, w):
    def f(a):
        s = np.std(a, ddof=0)
        return 0.0 if s < 1e-10 else np.mean(((a - np.mean(a)) / s) ** 4) - 3.0
    return _rolling(x, w, f)

def ts_argmax(x, w):
    return _rolling(x, w, lambda a: len(a) - 1 - np.argmax(a))

def ts_argmin(x, w):
    return _rolling(x, w, lambda a: len(a) - 1 - np.argmin(a))

def ts_roc(x, w):
    """变化率: (x[t] - x[t-w]) / x[t-w]"""
    x = np.asarray(x, float); r = np.full_like(x, np.nan)
    r[w:] = (x[w:] - x[:-w]) / np.where(np.abs(x[:-w]) > 1e-10, x[:-w], np.nan)
    return r


# ============================================================
# 扩张统计（无前向偏差）
# ============================================================

def expanding_mean(x, min_p=20):    return _expanding(x, np.mean, min_p)
def expanding_median(x, min_p=20):  return _expanding(x, np.median, min_p)
def expanding_std(x, min_p=20):     return _expanding(x, lambda a: np.std(a, ddof=0), min_p)
def expanding_percentile(x, p, min_p=20):
    return _expanding(x, lambda a: np.percentile(a, p * 100), min_p)


# ============================================================
# 截面函数
# ============================================================

def cs_rank(x):
    x = np.asarray(x, float)
    if x.ndim == 1: return np.full_like(x, 0.5)
    from scipy.stats import rankdata
    r = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        v = ~np.isnan(x[i])
        if v.sum() > 0:
            rk = rankdata(x[i][v]) / v.sum()
            r[i][v] = rk
    return r

def cs_zscore(x):
    x = np.asarray(x, float)
    if x.ndim == 1: return np.zeros_like(x)
    r = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        v = ~np.isnan(x[i])
        if v.sum() > 1:
            m, s = np.mean(x[i][v]), np.std(x[i][v], ddof=0)
            r[i][v] = (x[i][v] - m) / s if s > 1e-10 else 0.0
    return r


# ============================================================
# 数学函数
# ============================================================

def safe_abs(x):          return np.abs(x)
def safe_log(x):          return np.log(np.maximum(np.abs(x), 1e-10))
def safe_sqrt(x):         return np.sqrt(np.maximum(x, 0.0))
def safe_sign(x):         return np.sign(x)
def safe_exp(x):          return np.exp(np.clip(x, -50, 50))
def safe_tanh(x):         return np.tanh(x)
def safe_sigmoid(x):      return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def safe_relu(x):         return np.maximum(x, 0.0)


# ============================================================
# 信号函数
# ============================================================

def persist(x, n=3):
    return _persist(x, n)


# ============================================================
# 注册表
# ============================================================

FUNC_REGISTRY: Dict[str, Callable] = {
    # 时序
    'ts_mean':   ts_mean,   'ts_std':    ts_std,
    'ts_sum':    ts_sum,    'ts_max':    ts_max,
    'ts_min':    ts_min,    'ts_median': ts_median,
    'ts_delta':  ts_delta,  'ts_delay':  ts_delay,
    'ts_rank':   ts_rank,   'ts_corr':   ts_corr,
    'ts_skew':   ts_skew,   'ts_kurt':   ts_kurt,
    'ts_argmax': ts_argmax, 'ts_argmin': ts_argmin,
    'ts_roc':    ts_roc,
    # 扩张统计
    'expanding_mean': expanding_mean, 'expanding_median': expanding_median,
    'expanding_std':  expanding_std,  'expanding_percentile': expanding_percentile,
    # 截面
    'cs_rank': cs_rank, 'cs_zscore': cs_zscore,
    # 数学
    'abs': safe_abs, 'log': safe_log, 'sqrt': safe_sqrt,
    'sign': safe_sign, 'exp': safe_exp, 'tanh': safe_tanh,
    'sigmoid': safe_sigmoid, 'relu': safe_relu,
    # 信号
    'persist': persist,
}

SAFE_CONSTANTS = {'True': 1.0, 'False': 0.0, 'None': 0.0, 'pi': np.pi, 'e': np.e}

VALID_VAR_PREFIXES = [
    'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT', 'VWAP', 'RETURNS', 'RET',
    'ATR', 'STDDEV', 'BBWIDTH', 'HV', 'NATR',
    'TRIMA', 'SMA', 'MA', 'EMA', 'TSF', 'WMA', 'DEMA', 'KAMA',
    'ADX', 'RSI', 'CCI', 'MACD', 'MFI', 'ULTOSC', 'ROC', 'MOM_RATIO',
    'TREND_STRENGTH', 'LINEARREG', 'VAR',
    'VOL_RATIO', 'VOL_CHG', 'VOL_REGIME', 'OBV',
    'AVGPRICE', 'WCLPRICE', 'CORREL', 'MOM_CHG', 'UP_RATIO',
]

def is_valid_variable(name: str) -> bool:
    upper = name.upper()
    if upper in VALID_VAR_PREFIXES:
        return True
    for pfx in VALID_VAR_PREFIXES:
        if upper.startswith(pfx + '_'):
            rest = upper[len(pfx) + 1:]
            if rest and all(c.isdigit() or c == '_' for c in rest):
                return True
    return False
