# -*- coding: utf-8 -*-
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
import talib
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

def ts_cov(x, y, w):
    """Rolling covariance"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    r = np.full_like(x, np.nan)
    for i in range(w - 1, len(x)):
        xw = x[i - w + 1 : i + 1]; yw = y[i - w + 1 : i + 1]
        r[i] = np.cov(xw, yw, ddof=0)[0, 1]
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
    """Rate of change: (x[t]-x[t-w])/x[t-w]"""
    x = np.asarray(x, float); r = np.full_like(x, np.nan)
    r[w:] = (x[w:] - x[:-w]) / np.where(np.abs(x[:-w]) > 1e-10, x[:-w], np.nan)
    return r

def ts_zscore(x, d):
    """Rolling Z-score: (x-mu)/sigma"""
    mu = ts_mean(x, d); sg = ts_std(x, d)
    return np.where(sg > 1e-10, (np.asarray(x, float) - mu) / sg, 0)

def ts_scale(x, d):
    """Rolling scale: sum(abs(x)) in window -> 1"""
    x = np.asarray(x, float); r = np.full_like(x, np.nan)
    for i in range(d - 1, len(x)):
        seg = x[i - d + 1 : i + 1]; s = np.sum(np.abs(seg))
        r[i] = x[i] / s if s > 1e-10 else 0
    return r

def ts_quantile(x, d):
    """Rolling quantile rank 0~1"""
    return ts_rank(x, d)

def ts_av_diff(x, d):
    """Current minus rolling mean (deviation)"""
    return np.asarray(x, float) - ts_mean(x, d)

def ts_decay_linear(x, d):
    """Linear decay weighted mean"""
    x = np.asarray(x, float)
    r = np.full_like(x, np.nan)
    weights = np.arange(1, d + 1, dtype=float)
    w_sum = weights.sum()
    for i in range(d - 1, len(x)):
        r[i] = np.sum(x[i - d + 1 : i + 1] * weights) / w_sum
    return r

def ts_product(x, d):
    """Rolling product over d periods"""
    x = np.asarray(x, float)
    r = np.full_like(x, np.nan)
    for i in range(d - 1, len(x)):
        seg = x[i - d + 1 : i + 1]
        r[i] = np.prod(seg[~np.isnan(seg)]) if np.any(~np.isnan(seg)) else np.nan
    return r

def ts_regression(y, x, d, rettype=2):
    """Rolling linear regression: y = alpha + beta*x + eps
    """
    y, x = np.asarray(y, float), np.asarray(x, float)
    r = np.full_like(y, np.nan)
    for i in range(d - 1, len(y)):
        yw, xw = y[i - d + 1 : i + 1], x[i - d + 1 : i + 1]
        # 过滤 NaN
        valid = ~np.isnan(yw) & ~np.isnan(xw)
        if valid.sum() < 3:
            continue
        yv, xv = yw[valid], xw[valid]
        # 线性回归
        slope, intercept = np.polyfit(xv, yv, 1)
        y_pred = intercept + slope * xv
        ss_res = np.sum((yv - y_pred) ** 2)
        ss_tot = np.sum((yv - np.mean(yv)) ** 2)
        
        if rettype == 0: r[i] = slope
        elif rettype == 1: r[i] = intercept
        elif rettype == 2: r[i] = y[i] - (intercept + slope * x[i])  # 残差
        elif rettype == 3: r[i] = intercept + slope * x[i]            # 预测
        elif rettype == 4: r[i] = 1 - ss_res / ss_tot if ss_tot > 1e-10 else 0.0
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

def cs_scale(x, scale=1.0):
    """Cross-sectional scale: sum(abs(x)) = scale"""
    x = np.asarray(x, float)
    if x.ndim == 1: return x / (np.sum(np.abs(x)) + 1e-10) * scale
    r = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        v = ~np.isnan(x[i])
        if v.sum() > 0:
            s = np.sum(np.abs(x[i][v]))
            r[i][v] = x[i][v] / s * scale if s > 1e-10 else 0.0
    return r

def cs_winsorize(x, std=4.0):
    """Cross-sectional winsorize at +/-std"""
    x = np.asarray(x, float)
    if x.ndim == 1:
        m, s = np.mean(x), np.std(x)
        return np.clip(x, m - std * s, m + std * s)
    r = x.copy()
    for i in range(x.shape[0]):
        v = ~np.isnan(x[i])
        if v.sum() > 1:
            m, s = np.mean(x[i][v]), np.std(x[i][v], ddof=0)
            r[i][v] = np.clip(x[i][v], m - std * s, m + std * s)
    return r

def cs_normalize(x, use_std=False, limit=0.0):
    """Cross-sectional normalize"""
    if use_std:
        return cs_scale(cs_zscore(x))
    return cs_scale(x, 1.0)

def cs_quantile(x, driver='gaussian', sigma=1.0):
    """Cross-sectional quantile"""
    return cs_rank(x)


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
# 特征计算函数（从原始 OHLCV 数组实时算，无需 FeatureSpace）
# ============================================================

def _feature_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI centered: (RSI-50)/50, range [-1,1]. talib-aligned."""
    c = np.asarray(close, float)
    result = talib.RSI(c, timeperiod=period)
    result = np.nan_to_num(result, nan=50.0)
    return (result - 50) / 50


def _feature_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR - Wilder smoothed. talib-aligned."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.ATR(h, l, c, timeperiod=period)


def _feature_atr_sma(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR-SMA: simple rolling mean of TR (original V4, kept for compat)"""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    return _rolling(tr, period, np.mean)


def _wilder_smooth(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing: S[t]=(S[t-1]*(p-1)+x[t])/p, seed=SMA. talib-aligned."""
    x = np.asarray(x, float)
    n = len(x)
    r = np.full(n, np.nan)
    if n < period:
        return r
    r[period - 1] = np.mean(x[:period])
    for i in range(period, n):
        if np.isnan(r[i - 1]):
            r[i] = np.nan
        else:
            r[i] = (r[i - 1] * (period - 1) + x[i]) / period
    return r


def _feature_ema(x: np.ndarray, period: int = 20) -> np.ndarray:
    """EMA: recursive, talib-aligned. k=2/(p+1), seed=SMA."""
    x = np.asarray(x, float)
    n = len(x)
    k = 2.0 / (period + 1)
    r = np.full(n, np.nan)
    if n < period:
        return r
    r[period - 1] = np.mean(x[:period])
    for i in range(period, n):
        if np.isnan(r[i - 1]):
            r[i] = np.nan
        else:
            r[i] = (x[i] - r[i - 1]) * k + r[i - 1]
    return r


def _feature_bbwidth(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Bollinger Band Width = (upper-lower)/middle. talib-aligned."""
    c = np.asarray(close, float)
    upper, middle, lower = talib.BBANDS(c, timeperiod=period, nbdevup=2, nbdevdn=2, matype=0)
    return np.where(middle > 0, (upper - lower) / middle, 0)


def _feature_stddev(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Standard Deviation. talib-aligned."""
    c = np.asarray(close, float)
    return talib.STDDEV(c, timeperiod=period, nbdev=1)


def _feature_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX - Average Directional Index. talib-aligned."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.ADX(h, l, c, timeperiod=period)


def _feature_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """CCI - Commodity Channel Index. talib-aligned."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.CCI(h, l, c, timeperiod=period)


def _feature_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD histogram = 2*(DIF-DEA). talib-aligned."""
    c = np.asarray(close, float)
    dif, dea, hist = talib.MACD(c, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    return 2 * (dif - dea)


def _feature_trima(close: np.ndarray, period: int = 40) -> np.ndarray:
    """TRIMA - Triangular Moving Average. talib-aligned."""
    c = np.asarray(close, float)
    return talib.TRIMA(c, timeperiod=period)


def _feature_tsf(close: np.ndarray, period: int = 7) -> np.ndarray:
    """TSF - Time Series Forecast. talib-aligned."""
    c = np.asarray(close, float)
    return talib.TSF(c, timeperiod=period)


def _feature_kama(close: np.ndarray, period: int = 30) -> np.ndarray:
    """KAMA - Kaufman Adaptive Moving Average. talib-aligned."""
    c = np.asarray(close, float)
    return talib.KAMA(c, timeperiod=period)


def _feature_wma(close: np.ndarray, period: int = 20) -> np.ndarray:
    """WMA - Weighted Moving Average. talib-aligned."""
    c = np.asarray(close, float)
    return talib.WMA(c, timeperiod=period)


def _feature_dema(close: np.ndarray, period: int = 20) -> np.ndarray:
    """DEMA - Double Exponential Moving Average. talib-aligned."""
    c = np.asarray(close, float)
    return talib.DEMA(c, timeperiod=period)


def _feature_hv(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Historical Volatility: annualized std of daily returns x100."""
    c = np.asarray(close, float)
    rets = np.diff(c) / np.where(c[:-1] > 0, c[:-1], 1)
    rets = np.insert(rets, 0, 0)
    return _rolling(rets, period, lambda a: np.std(a, ddof=1) * np.sqrt(252) * 100)


def _feature_natr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """NATR - Normalized ATR. talib-aligned."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.NATR(h, l, c, timeperiod=period)


def _feature_var(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Variance. talib-aligned."""
    c = np.asarray(close, float)
    return talib.VAR(c, timeperiod=period)


def _feature_linearreg(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Linear Regression value. talib-aligned."""
    c = np.asarray(close, float)
    return talib.LINEARREG(c, timeperiod=period)


def _feature_vol_ratio(close: np.ndarray, volume: np.ndarray, short: int = 5, long: int = 20) -> np.ndarray:
    """Volume ratio = SMA(vol,short) / SMA(vol,long). No talib equivalent."""
    v = np.asarray(volume, float)
    vs = _rolling(v, short, np.mean)
    vl = _rolling(v, long, np.mean)
    return np.where(vl > 0, vs / vl, 0)


def _feature_amt_ratio(amount: np.ndarray, short: int = 5, long: int = 20) -> np.ndarray:
    # amt ratio = short_mean / long_mean
    a = np.asarray(amount, float)
    return np.where(_rolling(a, long, np.mean) > 0,
                    _rolling(a, short, np.mean) / _rolling(a, long, np.mean), 0)

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
    'ts_cov':      ts_cov,
    'ts_var':      lambda x, w: ts_std(x, w) ** 2,
    'ts_logret':   lambda x: safe_log(x / ts_delay(x, 1)),
    'ts_zscore':   ts_zscore,
    'ts_scale':    ts_scale,
    'ts_quantile': ts_quantile,
    'ts_av_diff':  ts_av_diff,
    'ts_decay_linear': ts_decay_linear,
    'ts_product':  ts_product,
    'ts_regression': ts_regression,
    # 扩张统计
    'expanding_mean': expanding_mean, 'expanding_median': expanding_median,
    'expanding_std':  expanding_std,  'expanding_percentile': expanding_percentile,
    # 截面
    'cs_rank':   cs_rank,   'cs_zscore':   cs_zscore,
    'cs_scale':  cs_scale,  'cs_winsorize': cs_winsorize,
    'cs_normalize': cs_normalize, 'cs_quantile': cs_quantile,
    # 数学
    'abs': safe_abs, 'log': safe_log, 'sqrt': safe_sqrt,
    'sign': safe_sign, 'exp': safe_exp, 'tanh': safe_tanh,
    'sigmoid': safe_sigmoid, 'relu': safe_relu,
    # 信号
    'persist': persist,
    # 特征计算（从 OHLCV 实时算，无需 FeatureSpace）
    'rsi':         _feature_rsi,
    'atr':         _feature_atr,
    'atr_sma':     _feature_atr_sma,
    
    'bb_width':    _feature_bbwidth,
    'stddev':      _feature_stddev,
    'adx':         _feature_adx,
    'cci':         _feature_cci,
    'macd':        _feature_macd,
    'trima':       _feature_trima,
    'ema':         _feature_ema,
    'wilder_smooth': _wilder_smooth,
    'tsf':         _feature_tsf,
    'kama':        _feature_kama,
    'wma':         _feature_wma,
    'dema':        _feature_dema,
    'hv':          _feature_hv,
    'natr':        _feature_natr,
    'var':         _feature_var,
    'linearreg':   _feature_linearreg,
    'vol_ratio':   _feature_vol_ratio,
    'amt_ratio':   _feature_amt_ratio,
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
