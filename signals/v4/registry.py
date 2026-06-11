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

def signed_power(x, exponent=2.0):
    """带符号幂变换: sign(x) * |x|^exponent
    保留方向，非线性放大/压缩幅度。
    exponent>1 放大极端值，exponent<1 压缩振幅。
    """
    return np.sign(x) * np.power(np.abs(x), float(exponent))

def safe_max(x, y):    return np.maximum(x, y)
def safe_min(x, y):    return np.minimum(x, y)

def ts_regression_residual(x, window):
    """时序回归残差: 对时间做线性回归 x=a+b*t，返回当前值-预测值"""
    x = np.asarray(x, float); window = int(window)
    if window < 3:
        return np.zeros_like(x)
    r = np.full_like(x, 0.0)
    for i in range(window - 1, len(x)):
        seg = x[i - window + 1: i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) < 3:
            continue
        t = np.arange(len(valid), dtype=float)
        cov = np.cov(t, valid)
        if cov[0, 0] < 1e-15:
            continue
        b = cov[0, 1] / cov[0, 0]
        a = np.mean(valid) - b * np.mean(t)
        predicted = a + b * (len(valid) - 1)
        r[i] = valid[-1] - predicted
    return r


# ============================================================
# 信号函数
# ============================================================
#
# ── 非对称买卖模式（单表达式即可实现，无需额外状态机） ──
#
# 方向非对称:  上涨追动量，下跌等反转
#   "ts_roc(CLOSE, 20) if SECTOR_UP > 0.5 else -ts_roc(CLOSE, 10)"
#   → 广度好时做多追涨，广度差时做多抄底
#
# 时长非对称:  买入需持续性确认，卖出单日即可
#   "persist(ts_roc(CLOSE, 5) > 0, 2) if BW_MA20 > 0.6 else -1 if CLOSE < ts_delay(CLOSE, 1) * 0.98 else 0"
#   → 买入需要连续2天确认，破位当日立即卖出
#
# 状态门控:    趋势/震荡两套逻辑
#   "(ts_roc(CLOSE, 20) if adx(HIGH, LOW, CLOSE, 14) > 30 else -ts_roc(CLOSE, 5) if adx(HIGH, LOW, CLOSE, 14) < 15 else 0)"
#   → 强趋势追涨，弱趋势反转，中间观望
#
# 量价背离:    价格高位但量不配合 → 卖出
#   "-1 if (CLOSE / ts_max(CLOSE, 20) > 0.98 and VOLUME < ts_mean(VOLUME, 5)) else 0"
#
# 所有模式输出为连续信号线，引擎层按 >0 做多 / <0 做空 解释。
# TODO: 滚动止损/止盈在引擎层实现，信号层专注买卖条件。

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
    'ts_regression_residual': ts_regression_residual,
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
    'signed_power': signed_power,
    'safe_max': safe_max, 'safe_min': safe_min,
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
    # 特征指标（通过 _feature_xxx 函数实时计算或预计算注入）
    'ATR',          # 平均真实波幅 (Average True Range)
    'STDDEV',       # 滚动标准差
    'BBWIDTH',      # 布林带宽度 (upper-lower)/middle
    'HV',           # 历史波动率 (年化)
    'NATR',         # 归一化 ATR (ATR/CLOSE)
    'TRIMA', 'SMA', 'MA', 'EMA', 'TSF', 'WMA', 'DEMA', 'KAMA',  # 各类均线
    'ADX',          # 平均趋向指数 (趋势强度)
    'RSI',          # 相对强弱指标
    'CCI',          # 商品通道指数
    'MACD',         # MACD 柱
    'MFI',          # 资金流量指数
    'ULTOSC',       # 终极摆动指标
    'ROC',          # 变化率 (Rate of Change)
    'MOM_RATIO',    # 动量比率
    'TREND_STRENGTH',  # 趋势强度
    'LINEARREG',    # 线性回归值
    'VAR',          # 滚动方差
    'VOL_RATIO',    # 量比 (短均量/长均量)
    'VOL_CHG',      # 成交量变化率
    'VOL_REGIME',   # 成交量状态(放量/缩量)
    'OBV',          # 能量潮 (On-Balance Volume)
    'AVGPRICE',     # 均价 (OPEN+HIGH+LOW+CLOSE)/4
    'WCLPRICE',     # 加权收盘价 (HIGH+LOW+2*CLOSE)/4
    'CORREL',       # 滚动相关系数
    'MOM_CHG',      # 动量变化
    'UP_RATIO',     # 上涨比例
    # ═══ 行业广度变量（预计算注入，测"市场共识"而非"价格偏离"） ═══
    # 价格宽度 — 多少行业在涨？
    'SECTOR_UP',           # 当日上涨行业比例 [0,1]
    'SECTOR_MOM20',        # 20日动量>0的行业比例
    'SECTOR_AD',           # 行业涨跌比 (adv-decl)/total
    # 价格宽度 Z-score 标准化
    'SECTOR_UP_Z',         # SECTOR_UP 的 Z-score
    'SECTOR_MOM20_Z',      # SECTOR_MOM20 的 Z-score
    # 均线宽度 — 均线上方行业密度
    'BW_MA5',              # 5日均线以上行业比例
    'BW_MA10',             # 10日均线以上行业比例
    'BW_MA20',             # 20日均线以上行业比例（经典背离信号源）
    'BW_MA20_Z',           # BW_MA20 的 Z-score
    'BW_MOM5',             # 5日宽度变化率
    # 离散度/轮动 — 行业有多一致？
    'DISP',                # 行业日收益截面标准差（高=分化，低=同涨同跌）
    'BW_MOM20',            # 20日宽度动量
    'ROTSPD',              # 行业轮动速度（领涨行业切换频率）
    'NHL20',               # 20日新高-新低行业净差
    'SKEW',                # 行业收益截面偏度（>0=少数极端领涨，<0=少数暴跌拖累）
    # 成交量结构 — 资金在往哪流？
    'VMED',                # 行业中位数成交量
    'VDISP',               # 行业成交量截面离散度（高=资金集中，低=均匀分布）
    'VSKEW',               # 行业成交量截面偏度（>0=少数行业吸金）
    # 相关性 — 市场是分散还是同步？（区别于单品种特征 CORREL）
    'CORR',                # 行业平均两两相关系数（>0.7=系统性风险模式，<0.3=分散模式）
    # 尾部风险 — 极端行情信号
    'TAILUP',              # 右尾强度（行业极端上涨密度）
    'TAILDOWN',            # 左尾强度（行业极端下跌密度）
    'TAILNET',             # 尾部净强度 TAILUP - TAILDOWN
    # 成交额
    'AMT5',                # 5日成交额指标
    'AMT10',               # 10日成交额指标
    # v4 因子模块自定义变量
    'BENCH', 'REL', 'SHARE', 'DOWNSIDE_VOL',
]

def is_valid_variable(name: str) -> bool:
    upper = name.upper()
    if upper in VALID_VAR_PREFIXES:
        return True
    for pfx in VALID_VAR_PREFIXES:
        if upper.startswith(pfx + '_'):
            rest = upper[len(pfx) + 1:]
            if rest and all(c.isascii() and (c.isalnum() or c == '_') for c in rest):
                return True
    return False


# ============================================================
# 临时自定义注册 (外部模板热添加，无需修改 registry.py)
# ============================================================
#
# 用法:
#   from signals.v4.registry import register_function, register_variable
#   register_function('my_indicator', lambda x, w: np.convolve(x, np.ones(w)/w, 'same'))
#   register_variable('MY_VAR')
#   expr = Expression("MY_VAR > 0 and my_indicator(CLOSE, 10) > 0")
#
# 注意: 注册是进程级全局操作，重复注册同名函数会覆盖。

def register_function(name: str, func: Callable) -> None:
    """临时注册自定义函数到表达式引擎。
    
    Args:
        name: 函数名 (表达式中的调用名)
        func: 函数实现，签名为 (*np.ndarray) -> np.ndarray
    """
    name_lower = name.lower()
    if name_lower in FUNC_REGISTRY:
        import warnings
        warnings.warn(
            f"register_function: '{name}' 已存在，将被覆盖。"
            f"原函数: {FUNC_REGISTRY[name_lower].__name__}"
        )
    FUNC_REGISTRY[name_lower] = func


def register_variable(prefix: str) -> None:
    """临时注册自定义变量前缀到表达式引擎。
    
    Args:
        prefix: 变量名前缀 (大小写不敏感)，支持 'prefix' 或 'prefix_suffix' 匹配
    """
    upper = prefix.upper()
    if upper not in VALID_VAR_PREFIXES:
        VALID_VAR_PREFIXES.append(upper)


def unregister_function(name: str) -> bool:
    """注销自定义函数，返回是否成功。内置函数不可注销。"""
    return FUNC_REGISTRY.pop(name.lower(), None) is not None


def unregister_variable(prefix: str) -> bool:
    """注销自定义变量前缀，返回是否成功。"""
    upper = prefix.upper()
    if upper in VALID_VAR_PREFIXES:
        VALID_VAR_PREFIXES.remove(upper)
        return True
    return False
