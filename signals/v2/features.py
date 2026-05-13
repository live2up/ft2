"""
signals/v2/features.py — 特征空间引擎
=============================================================================

核心抽象：纯函数计算特征 + 声明式配置 → 特征矩阵


============================================================================
                         架构层级（竖式）
============================================================================

 ┌─────────────────────────────────────────────────────────────────────┐
 │  输入：OHLCV DataFrame                                               │
 │  columns: open, high, low, close, volume                            │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  第一层  —  声明式配置 DEFAULT_CONFIG                                 │
 │                                                                      │
 │  DEFAULT_CONFIG = {                                                  │
 │      'features': {                ← 函数调用式语法                   │
 │          'volatility': ['ATR(7,14)', 'BBWIDTH(20,30)'],             │
 │          'trend':      ['EMA(5,10,15,20,30,40,60)'],                │
 │          'momentum':   ['RSI(5,7,10,14,20,30)'],                    │
 │      },                                                              │
 │      'regime': True,               ← 布尔开关 (硬编码 5 个状态特征)   │
 │      'differences': [              ← 列名引用式语法                  │
 │          ['BBWIDTH_30', 'TSF_7', 'sub'],                            │
 │      ],                                                              │
 │      'ma_windows': [10],           ← 为每个特征生成 MA 平滑版本      │
 │      'normalize': True,            ← 自动归一化                      │
 │  }                                                                   │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  第二层  —  函数注册表 _FEATURE_CALC_REGISTRY                         │
 │                                                                      │
 │  55+ 个 calc_xxx() 纯函数，统一签名:                                  │
 │    calc_xxx(data: pd.DataFrame, **params) → (raw_val, ma_val)        │
 │                                                                      │
 │  ┌────────────────┬────────────────────────────────────────────┐    │
 │  │   注册名        │   函数           │   底层引擎               │    │
 │  ├────────────────┼────────────────┼──────────────────────────┤    │
 │  │ ATR, EMA       │ calc_atr, ...   │ talib (优先) / 纯NumPy   │    │
 │  │ RSI, MACD      │ calc_rsi, ...   │ talib (优先) / 纯NumPy   │    │
 │  │ BBWIDTH, TSF   │ calc_bbwidth,...│ talib (优先) / 纯NumPy   │    │
 │  └────────────────┴────────────────┴────────────────────────────┘    │
 │                                                                      │
 │  可通过 register_feature(name, calc_fn) 注册自定义特征                │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  第三层  —  归一化策略（逐特征自动判断）                              │
 │                                                                      │
 │  ┌──────────────┬───────────────────────┬─────────────────────┐     │
 │  │   策略       │   公式                 │   适用特征           │     │
 │  ├──────────────┼───────────────────────┼─────────────────────┤     │
 │  │ close        │ raw_val / close       │ ATR,BBWIDTH,EMA,TSF │     │
 │  │ pct          │ raw_val / 100         │ HV, NATR            │     │
 │  │ raw          │ raw_val (不变)        │ RSI,ADX,MACD,CCI    │     │
 │  └──────────────┴───────────────────────┴─────────────────────┘     │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  第四层  —  FeatureSpace.fit().transform() → 特征矩阵                │
 │                                                                      │
 │  1. _build_configs()  — 解析配置，展开多参数，生成列名                │
 │     "ATR(7,14)" → ATR_7, ATR_14                                     │
 │     "EMA(5,10)" → EMA_5, EMA_10                                     │
 │                                                                      │
 │  2. regime=True       — 追加 5 个状态特征列                          │
 │     TREND_STRENGTH_20, VOL_REGIME_20, VOL_CHG_5, ...                │
 │                                                                      │
 │  3. differences       — 对已有列做数组运算                           │
 │     BBWIDTH_30 - TSF_7 → BBWIDTH_30_sub_TSF_7                       │
 │                                                                      │
 │  4. ma_windows        — 为每个特征生成 MA 平滑版本                   │
 │     ATR_7 + window=10 → ATR_7_MA                                    │
 │                                                                      │
 │  输出: pd.DataFrame (index=日期, columns=55+ 个特征)                  │
 └─────────────────────────────────────────────────────────────────────┘


============================================================================
                         参数规范对照
============================================================================

 ┌────────────────────┬──────────────────┬──────────────────────────────┐
 │       配置项         │      语法类型    │       示例 / 展开后           │
 ├────────────────────┼──────────────────┼──────────────────────────────┤
 │ features           │ 函数调用式        │ ATR(7,14) → ATR_7, ATR_14   │
 │                    │ (括号参数)        │ EMA(5,10) → EMA_5, EMA_10    │
 ├────────────────────┼──────────────────┼──────────────────────────────┤
 │ regime             │ 布尔开关          │ True → 自动启用5个状态特征    │
 ├────────────────────┼──────────────────┼──────────────────────────────┤
 │ differences        │ 列名引用式        │ [BBWIDTH_30, TSF_7, sub]     │
 │                    │ [左列, 右列, 运算符]│ → BBWIDTH_30_sub_TSF_7       │
 ├────────────────────┼──────────────────┼──────────────────────────────┤
 │ ma_windows         │ 整数列表          │ [10] → 每个特征 + _MA 后缀   │
 ├────────────────────┼──────────────────┼──────────────────────────────┤
 │ normalize          │ 布尔开关          │ True → 自动判断归一化策略     │
 └────────────────────┴──────────────────┴──────────────────────────────┘


============================================================================
                         架构原则（来自 AIdev 探索结论）
============================================================================

 1. 每个技术指标是纯函数，不是类 — 消除 100+ 个 SignalGenerator 子类错误模式
 2. 声明式配置驱动特征生成 — 新增特征只需添加配置项
 3. 特征空间是统一工作层级 — GP、传统信号、ML 共用同一特征矩阵
 4. 归一化内置：close/pct/raw 三种方式，每个特征独立配置
 5. MA 配对自动生成 — 原始值 + 多窗口 MA 值
 6. 衍生特征声明式生成 — 差异/比值/乘积等自动计算
 7. talib 优先 + 纯 NumPy 回退 — 无外部依赖也能运行

============================================================================
"""

import sys
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False


# ============================================================
# 纯函数式特征计算函数
# 统一签名: calc_xxx(data: pd.DataFrame, **params) -> Tuple[np.ndarray, np.ndarray]
# 返回: (原始值, MA平滑值)
# ============================================================

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        segment = arr[i - window + 1:i + 1]
        result[i] = np.nanmean(segment)
    return result


def calc_atr(data: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    h, l, c = data['high'].values, data['low'].values, data['close'].values
    if HAS_TALIB:
        raw = talib.ATR(h, l, c, timeperiod=period)
    else:
        tr = np.maximum(h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1)))
        raw = _rolling_mean(tr, period)
    return raw, _rolling_mean(raw, period)


def calc_stddev(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        raw = talib.STDDEV(c, timeperiod=period)
    else:
        raw = np.array([np.nanstd(c[max(0, i-period+1):i+1], ddof=1) for i in range(len(c))])
    return raw, _rolling_mean(raw, period)


def calc_bbwidth(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        upper, middle, lower = talib.BBANDS(c, timeperiod=period)
        raw = (upper - lower) / middle
    else:
        ma = _rolling_mean(c, period)
        std = np.array([np.nanstd(c[max(0, i-period+1):i+1], ddof=1) for i in range(len(c))])
        upper_b = ma + 2 * std
        lower_b = ma - 2 * std
        raw = np.where(ma > 0, (upper_b - lower_b) / ma, 0)
    return raw, _rolling_mean(raw, period)


def calc_hv(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    returns = np.diff(c) / c[:-1]
    returns = np.insert(returns, 0, np.nan)
    raw = np.full(len(c), np.nan)
    for i in range(period - 1, len(c)):
        seg = returns[i - period + 1:i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) > 1:
            raw[i] = np.nanstd(valid, ddof=1) * np.sqrt(252) * 100
    return raw, _rolling_mean(raw, period)


def calc_natr(data: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    h, l, c = data['high'].values, data['low'].values, data['close'].values
    if HAS_TALIB:
        raw = talib.NATR(h, l, c, timeperiod=period)
    else:
        tr = np.maximum(h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1)))
        atr = _rolling_mean(tr, period)
        raw = np.where(c > 0, atr / c * 100, 0)
    return raw, _rolling_mean(raw, period)


def calc_trima(data: pd.DataFrame, period: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        raw = talib.TRIMA(c, timeperiod=period)
    else:
        raw = _rolling_mean(_rolling_mean(c, period // 2 + 1), period // 2 + 1)
    return raw, _rolling_mean(raw, period)


def calc_sma_val(data: pd.DataFrame, period: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    raw = _rolling_mean(c, period)
    return raw, _rolling_mean(raw, period)


def calc_ma(data: pd.DataFrame, short: int = 5, long: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    ma_short = _rolling_mean(c, short)
    ma_long = _rolling_mean(c, long)
    raw = np.where(ma_long > 0, (ma_short - ma_long) / ma_long * 100, 0)
    return raw, _rolling_mean(raw, max(short, long))


def calc_ema(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        raw = talib.EMA(c, timeperiod=period)
    else:
        alpha = 2 / (period + 1)
        raw = np.full(len(c), np.nan)
        raw[period - 1] = np.nanmean(c[:period])
        for i in range(period, len(c)):
            raw[i] = alpha * c[i] + (1 - alpha) * raw[i - 1]
    return raw, _rolling_mean(raw, period)


def calc_tsf(data: pd.DataFrame, period: int = 7) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        raw = talib.TSF(c, timeperiod=period)
    else:
        raw = np.full(len(c), np.nan)
        for i in range(period - 1, len(c)):
            y = c[i - period + 1:i + 1]
            x = np.arange(period)
            slope, intercept = np.polyfit(x, y, 1)
            raw[i] = intercept + slope * period
    return raw, _rolling_mean(raw, period)


def calc_adx(data: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    h, l, c = data['high'].values, data['low'].values, data['close'].values
    if HAS_TALIB:
        raw = talib.ADX(h, l, c, timeperiod=period)
    else:
        raw = np.full(len(c), np.nan)
        plus_dm = np.maximum(h - np.roll(h, 1), 0)
        minus_dm = np.maximum(np.roll(l, 1) - l, 0)
        tr = np.maximum(h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1)))
        atr = _rolling_mean(tr, period)
        plus_di = np.where(atr > 0, _rolling_mean(plus_dm, period) / atr * 100, 0)
        minus_di = np.where(atr > 0, _rolling_mean(minus_dm, period) / atr * 100, 0)
        dx = np.where(plus_di + minus_di > 0, np.abs(plus_di - minus_di) / (plus_di + minus_di) * 100, 0)
        raw = _rolling_mean(dx, period)
    return raw, _rolling_mean(raw, period)


def calc_rsi(data: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        raw = talib.RSI(c, timeperiod=period)
    else:
        diff = np.diff(c)
        gain = np.where(diff > 0, diff, 0)
        loss = np.where(diff < 0, -diff, 0)
        raw = np.full(len(c), np.nan)
        for i in range(period, len(c)):
            avg_gain = np.nanmean(gain[i - period:i])
            avg_loss = np.nanmean(loss[i - period:i])
            if avg_loss > 0:
                raw[i + 1] = 100 - 100 / (1 + avg_gain / avg_loss)
            else:
                raw[i + 1] = 100
        raw[raw != raw] = 50
    raw = (raw - 50) / 50
    return raw, _rolling_mean(raw, period)


def calc_cci(data: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    h, l, c = data['high'].values, data['low'].values, data['close'].values
    if HAS_TALIB:
        raw = talib.CCI(h, l, c, timeperiod=period)
    else:
        tp = (h + l + c) / 3
        tp_ma = _rolling_mean(tp, period)
        md = np.full(len(c), np.nan)
        for i in range(period - 1, len(c)):
            md[i] = np.nanmean(np.abs(tp[i - period + 1:i + 1] - tp_ma[i]))
        raw = np.where(md > 0, (tp - tp_ma) / (0.015 * md), 0)
    return raw, _rolling_mean(raw, period)


def calc_macd(data: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        macd_line, signal_line, hist = talib.MACD(c, fastperiod=fast, slowperiod=slow, signalperiod=signal)
        raw = hist
    else:
        ema_fast = np.copy(c)
        alpha_f = 2 / (fast + 1)
        for i in range(1, len(c)):
            ema_fast[i] = alpha_f * c[i] + (1 - alpha_f) * ema_fast[i - 1]
        ema_slow = np.copy(c)
        alpha_s = 2 / (slow + 1)
        for i in range(1, len(c)):
            ema_slow[i] = alpha_s * c[i] + (1 - alpha_s) * ema_slow[i - 1]
        dif = ema_fast - ema_slow
        dea = np.full(len(c), np.nan)
        dea[slow - 1] = 0
        alpha_d = 2 / (signal + 1)
        for i in range(slow - 1, len(c)):
            if np.isnan(dea[i]):
                dea[i] = dif[i]
            else:
                dea[i] = alpha_d * dif[i] + (1 - alpha_d) * dea[i - 1]
        raw = 2 * (dif - dea)
    return raw, _rolling_mean(raw, signal)


def calc_var(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        raw = talib.VAR(c, timeperiod=period)
    else:
        raw = np.array([np.nanvar(c[max(0, i-period+1):i+1], ddof=1) for i in range(len(c))])
    return raw, _rolling_mean(raw, period)


def calc_linearreg(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        raw = talib.LINEARREG(c, timeperiod=period)
    else:
        raw = np.full(len(c), np.nan)
        for i in range(period - 1, len(c)):
            y = c[i - period + 1:i + 1]
            x = np.arange(period)
            slope, intercept = np.polyfit(x, y, 1)
            raw[i] = intercept + slope * (period - 1)
    return raw, _rolling_mean(raw, period)


def calc_vol_ratio(data: pd.DataFrame, short: int = 5, long: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    volume = data['volume'].values
    vol_short = _rolling_mean(volume, short)
    vol_long = _rolling_mean(volume, long)
    raw = np.where(vol_long > 0, vol_short / vol_long, 0)
    return raw, _rolling_mean(raw, short)


def calc_vol_chg(data: pd.DataFrame, period: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    volume = data['volume'].values
    vol_ma = _rolling_mean(volume, period)
    vol_long = _rolling_mean(volume, period * 3)
    ratio = np.where(vol_long > 0, vol_ma / vol_long, 0)
    ratio_shifted = np.roll(ratio, period)
    ratio_shifted[:period] = np.nan
    raw = np.where(ratio_shifted > 1e-10, ratio / ratio_shifted - 1.0, 0)
    return raw, _rolling_mean(raw, period)


def calc_trend_strength(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    returns = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        if c[i - 1] > 0:
            returns[i] = c[i] / c[i - 1] - 1
    cum_ret = np.full(len(c), np.nan)
    vol_arr = np.full(len(c), np.nan)
    for i in range(period - 1, len(c)):
        seg = returns[i - period + 1:i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) > 0:
            cum_ret[i] = np.nansum(valid)
            std_val = np.nanstd(valid, ddof=1) if len(valid) > 1 else 1e-10
            vol_arr[i] = std_val
    raw = np.where(vol_arr > 1e-10, cum_ret / vol_arr, 0)
    return raw, _rolling_mean(raw, period)


def calc_vol_regime(data: pd.DataFrame, short: int = 20, long: int = 60) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    returns = np.full(len(c), np.nan)
    for i in range(1, len(c)):
        if c[i - 1] > 0:
            returns[i] = c[i] / c[i - 1] - 1
    short_vol = np.full(len(c), np.nan)
    long_vol = np.full(len(c), np.nan)
    for i in range(long - 1, len(c)):
        s_seg = returns[max(0, i - short + 1):i + 1]
        s_valid = s_seg[~np.isnan(s_seg)]
        l_seg = returns[max(0, i - long + 1):i + 1]
        l_valid = l_seg[~np.isnan(l_seg)]
        if len(s_valid) > 1:
            short_vol[i] = np.nanstd(s_valid, ddof=1)
        if len(l_valid) > 1:
            long_vol[i] = np.nanstd(l_valid, ddof=1)
    raw = np.where(long_vol > 1e-10, short_vol / long_vol, 0)
    return raw, _rolling_mean(raw, short)


def calc_mom_chg(data: pd.DataFrame, period: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    momentum = np.full(len(c), np.nan)
    for i in range(period, len(c)):
        if c[i - period] > 0:
            momentum[i] = c[i] / c[i - period] - 1
    shifted = np.roll(momentum, period)
    shifted[:period] = np.nan
    raw = np.where(np.abs(shifted) > 1e-10, momentum - shifted, 0)
    return raw, _rolling_mean(raw, period)


def calc_up_ratio(data: pd.DataFrame, period: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    is_up = (c > np.roll(c, 1)).astype(float)
    is_up[0] = np.nan
    raw = np.full(len(c), np.nan)
    for i in range(period - 1, len(c)):
        seg = is_up[i - period + 1:i + 1]
        raw[i] = np.nanmean(seg)
    return raw, _rolling_mean(raw, period)


def calc_wma(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        wma = talib.WMA(c, timeperiod=period)
    else:
        wma = _rolling_mean(c, period)
    return wma, _rolling_mean(wma, period)


def calc_dema(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        dema = talib.DEMA(c, timeperiod=period)
    else:
        dema = _rolling_mean(c, period)
    return dema, _rolling_mean(dema, period)


def calc_kama(data: pd.DataFrame, period: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    c = data['close'].values
    if HAS_TALIB:
        kama = talib.KAMA(c, timeperiod=period)
    else:
        kama = _rolling_mean(c, period)
    return kama, _rolling_mean(kama, period)


def calc_mfi(data: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    if HAS_TALIB:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        volume = data['volume'].values.astype(float)
        mfi = talib.MFI(high, low, close, volume, timeperiod=period)
        raw = mfi - 50.0
    else:
        raw = np.zeros(len(data))
    return raw, _rolling_mean(raw, period)


def calc_ultosc(data: pd.DataFrame, period1: int = 7, period2: int = 14,
                period3: int = 28) -> Tuple[np.ndarray, np.ndarray]:
    if HAS_TALIB:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        ultosc = talib.ULTOSC(high, low, close,
                              timeperiod1=period1,
                              timeperiod2=period2,
                              timeperiod3=period3)
        raw = ultosc - 50.0
    else:
        raw = np.zeros(len(data))
    return raw, _rolling_mean(raw, period2)


def calc_obv(data: pd.DataFrame, signal_period: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    close = data['close'].values
    volume = data['volume'].values
    diff = np.diff(close, prepend=close[0])
    direction = np.sign(diff)
    obv = np.cumsum(volume * direction)
    raw = np.full(len(close), np.nan)
    for i in range(signal_period, len(close)):
        raw[i] = (obv[i] / obv[i - signal_period] - 1) * 100 if obv[i - signal_period] != 0 else 0
    raw[np.isinf(raw)] = 0
    return raw, _rolling_mean(np.nan_to_num(raw, nan=0), signal_period)


def calc_avgprice(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if HAS_TALIB:
        open_p = data['open'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        avgprice = talib.AVGPRICE(open_p, high, low, close)
        raw = np.where(avgprice > 0, (close - avgprice) / avgprice * 100, 0)
    else:
        raw = np.zeros(len(data))
    return raw, _rolling_mean(raw, 20)


def calc_wclprice(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if HAS_TALIB:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        wclprice = talib.WCLPRICE(high, low, close)
        raw = np.where(wclprice > 0, (close - wclprice) / wclprice * 100, 0)
    else:
        raw = np.zeros(len(data))
    return raw, _rolling_mean(raw, 20)


def calc_ht_sine(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if HAS_TALIB:
        close = data['close'].values.astype(float)
        sine, leadsine = talib.HT_SINE(close)
        raw = sine - leadsine
    else:
        raw = np.zeros(len(data))
    return raw, raw


def calc_ht_trendmode(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    if HAS_TALIB:
        close = data['close'].values.astype(float)
        ht_trendmode = talib.HT_TRENDMODE(close)
        raw = ht_trendmode - 0.5
    else:
        raw = np.zeros(len(data))
    return raw, raw


def calc_correl(data: pd.DataFrame, period: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    close = data['close'].values.astype(float)
    if HAS_TALIB:
        time_seq = np.arange(len(close), dtype=float)
        correl = talib.CORREL(close, time_seq, timeperiod=period)
    else:
        correl = np.zeros(len(close))
    return correl, _rolling_mean(correl, period)


# ============================================================
# 归一化处理
# ============================================================

def normalize_close(raw_val: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    close = data['close'].values
    return np.where(close > 0, raw_val / close, raw_val)


def normalize_pct(raw_val: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    return raw_val / 100.0


def normalize_raw(raw_val: np.ndarray, data: Optional[pd.DataFrame] = None) -> np.ndarray:
    return raw_val


NORMALIZE_MAP = {
    'close': normalize_close,
    'pct': normalize_pct,
    'raw': normalize_raw,
}


# ============================================================
# FeatureSpace — 特征空间引擎
# ============================================================

# 内置特征计算函数注册表
_FEATURE_CALC_REGISTRY: Dict[str, Callable] = {
    'ATR': calc_atr,
    'STDDEV': calc_stddev,
    'BBWIDTH': calc_bbwidth,
    'BBW': calc_bbwidth,
    'HV': calc_hv,
    'NATR': calc_natr,
    'TRIMA': calc_trima,
    'SMA': calc_sma_val,
    'MA': calc_ma,
    'EMA': calc_ema,
    'TSF': calc_tsf,
    'ADX': calc_adx,
    'RSI': calc_rsi,
    'CCI': calc_cci,
    'MACD': calc_macd,
    'VAR': calc_var,
    'LINEARREG': calc_linearreg,
    'VOL_RATIO': calc_vol_ratio,
    'VOL_CHG': calc_vol_chg,
    'TREND_STRENGTH': calc_trend_strength,
    'VOL_REGIME': calc_vol_regime,
    'MOM_CHG': calc_mom_chg,
    'UP_RATIO': calc_up_ratio,
    'WMA': calc_wma,
    'DEMA': calc_dema,
    'KAMA': calc_kama,
    'MFI': calc_mfi,
    'ULTOSC': calc_ultosc,
    'OBV': calc_obv,
    'AVGPRICE': calc_avgprice,
    'WCLPRICE': calc_wclprice,
    'HT_SINE': calc_ht_sine,
    'HT_TRENDMODE': calc_ht_trendmode,
    'CORREL': calc_correl,
}


def register_feature(name: str, calc_fn: Callable):
    """注册新的特征计算函数"""
    _FEATURE_CALC_REGISTRY[name.upper()] = calc_fn


# 默认配置：AIdev V5特征集
DEFAULT_CONFIG = {
    'features': {
        'volatility': ['ATR(7,14)', 'STDDEV(20,30)', 'BBWIDTH(20,30)',
                       'HV(20)', 'NATR(14)'],
        'trend': ['TRIMA(40,60)', 'SMA(50)',
                  'MA(short=5,long=20)',
                  'EMA(5,10,15,20,30,40,60)',
                  'WMA(20)', 'DEMA(20)', 'KAMA(10)'],
        'statistic': ['TSF(3,5,7,10,14,20)', 'VAR(20)', 'LINEARREG(20)', 'CORREL(10)'],
        'momentum': ['ADX(14)', 'RSI(5,7,10,14,20,30)', 'CCI(14,20)',
                     'MACD(fast=12,slow=26,signal=9)',
                     'MFI(14)', 'ULTOSC()'],
        'price_transform': ['AVGPRICE()', 'WCLPRICE()'],
        'cycle': ['HT_SINE()', 'HT_TRENDMODE()'],
        'volume': ['VOL_RATIO(5,10)', 'OBV(10)'],
    },
    'regime': True,
    'market_breadth': False,
    'differences': [
        ['BBWIDTH_30', 'TSF_7', 'sub'],
        ['ATR_7', 'TSF_7', 'sub'],
        ['STDDEV_20', 'TSF_7', 'sub'],
    ],
    'ma_windows': [10],
    'normalize': True,
}


class FeatureSpace:
    """特征空间引擎 — 声明式配置驱动，OHLCV → 特征矩阵"""

    def __init__(self, config: Optional[Dict] = None, data: Optional[pd.DataFrame] = None):
        """
        Args:
            config: 声明式配置字典，None则使用DEFAULT_CONFIG
            data: 可选，预先加载的OHLCV DataFrame
        """
        self.config = {**DEFAULT_CONFIG}
        if config:
            self._deep_merge(self.config, config)
        self._data = data
        self._feature_names = []
        self._feature_configs = {}
        self._built = False

    def _deep_merge(self, base: Dict, overrides: Dict):
        for key, val in overrides.items():
            if isinstance(val, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], val)
            else:
                base[key] = val

    def fit(self, data: pd.DataFrame) -> 'FeatureSpace':
        """绑定数据，解析配置，确定特征列名"""
        self._data = data
        self._feature_names = []
        self._feature_configs = {}
        self._build_configs()
        self._built = True
        return self

    def _parse_feature_str(self, feat_str: str) -> Tuple[str, Dict]:
        """解析 'ATR(7,14)' → ('ATR', {'period': 7}, ...)"""
        name_end = feat_str.find('(')
        if name_end == -1:
            return feat_str, {}
        name = feat_str[:name_end]
        params_str = feat_str[name_end + 1:-1]
        params = {}
        parts = [p.strip() for p in params_str.split(',') if p.strip()]
        key_params = []
        for p in parts:
            if '=' in p:
                k, v = p.split('=')
                params[k.strip()] = int(v.strip())
            else:
                key_params.append(int(p))
        if key_params:
            params['__default_params'] = key_params
        return name.upper(), params

    def _expand_params(self, name: str, params: Dict, calc_fn: Callable) -> List[Dict]:
        """展开多参数为多个特征配置"""
        import inspect
        sig = inspect.signature(calc_fn)
        param_names = [pn for pn in sig.parameters.keys() if pn != 'data']

        if '__default_params' in params:
            values = params['__default_params']
            result = []
            # 第一个默认参数如 period 取 values 中的值
            pos_param_idx = 0
            keyword_params = {k: v for k, v in params.items() if k != '__default_params'}
            for v in values:
                p = {
                    param_names[0 + pos_param_idx + len(keyword_params)]: v
                }
                p.update(keyword_params)
                result.append(p)
            return result
        else:
            return [params]

    def _build_configs(self):
        """解析配置，构建特征列表"""
        features_section = self.config.get('features', {})
        for category, feat_list in features_section.items():
            for feat_str in feat_list:
                name, params = self._parse_feature_str(feat_str)
                if name not in _FEATURE_CALC_REGISTRY:
                    print(f"    特征 '{name}' 未注册，跳过")
                    continue
                calc_fn = _FEATURE_CALC_REGISTRY[name]
                expanded = self._expand_params(name, params, calc_fn)
                for pcfg in expanded:
                    if pcfg:
                        parts = [str(v) for v in pcfg.values()]
                        col_name = "_".join([name] + parts)
                    else:
                        col_name = name
                    self._feature_names.append(col_name)
                    self._feature_configs[col_name] = {
                        'name': name,
                        'calc_fn': calc_fn,
                        'params': pcfg,
                        'category': category,
                    }

        # Regime特征
        if self.config.get('regime'):
            regime_feats = [
                ('TREND_STRENGTH_20', calc_trend_strength, {'period': 20}),
                ('VOL_REGIME_20', calc_vol_regime, {'short': 20, 'long': 60}),
                ('VOL_CHG_5', calc_vol_chg, {'period': 5}),
                ('MOM_CHG_5', calc_mom_chg, {'period': 5}),
                ('UP_RATIO_10', calc_up_ratio, {'period': 10}),
            ]
            for col_name, calc_fn, params in regime_feats:
                self._feature_names.append(col_name)
                self._feature_configs[col_name] = {
                    'name': col_name,
                    'calc_fn': calc_fn,
                    'params': params,
                    'category': 'regime',
                }

        self._original_names = list(self._feature_names)

    def transform(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """计算特征矩阵"""
        if data is not None:
            self.fit(data)
        if not self._built:
            raise RuntimeError("请先调用 fit(data) 或传入数据")
        if self._data is None:
            raise RuntimeError("无数据可用")

        df = self._data
        features = pd.DataFrame(index=df.index)

        for col_name in self._feature_names:
            cfg = self._feature_configs[col_name]
            try:
                raw_val, raw_ma = cfg['calc_fn'](df, **cfg['params'])
            except Exception:
                # 某些参数组合可能失败，尝试兼容
                try:
                    raw_val, raw_ma = cfg['calc_fn'](df, *list(cfg['params'].values())[:2])
                except Exception:
                    raw_val = np.full(len(df), np.nan)
                    raw_ma = raw_val

            if self.config.get('normalize', True):
                feat_name = cfg.get('name', '').upper()
                norm_method = self._guess_normalize(feat_name)
                if norm_method == 'close':
                    raw_val = normalize_close(raw_val, df)
                    raw_ma = normalize_close(raw_ma, df) if 'pct' not in feat_name.lower() else raw_ma
                elif norm_method == 'pct':
                    raw_val = normalize_pct(raw_val, df)
                    raw_ma = normalize_pct(raw_ma, df) if 'natr' not in feat_name.lower() and 'hv' not in feat_name.lower() else raw_ma / 100

            features[col_name] = raw_val

            # MA配对
            ma_windows = self.config.get('ma_windows', [])
            for mw in ma_windows:
                ma_col = f"{col_name}_MA_{mw}" if mw != 10 else f"{col_name}_MA"
                if col_name.endswith(f'_{mw}'):
                    continue
                features[ma_col] = _rolling_mean(raw_val, mw)

        # 衍生特征（差异特征等）
        differences = self.config.get('differences', [])
        for diff_config in differences:
            left_col, right_col, op = diff_config[0], diff_config[1], diff_config[2]
            derived_name = f"{left_col}_{op}_{right_col}"
            if left_col in features.columns and right_col in features.columns:
                lv = features[left_col].values
                rv = features[right_col].values
                if op == 'sub':
                    features[derived_name] = lv - rv
                elif op == 'div':
                    derived = np.where(np.abs(rv) > 1e-10, lv / rv, 0.0)
                    features[derived_name] = derived
                elif op == 'mul':
                    features[derived_name] = lv * rv
                elif op == 'add':
                    features[derived_name] = lv + rv
                self._feature_names.append(derived_name)

        features = features.replace([np.inf, -np.inf], np.nan)
        features = features.dropna()
        return features

    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """fit + transform 一步完成"""
        self.fit(data)
        return self.transform()

    def _guess_normalize(self, feat_name: str) -> str:
        """根据特征名推测归一化方式
        close_feats: 原始值为价格量级, 需除以close归一化为比例
        pct_feats: 原始值为百分比, 需除以100
        raw_feats: 原始值已归一化/无量纲/振荡类, 不做额外处理
        """
        # [修复] 从 close_feats 移除以下指标，原因：
        # BBWIDTH: calc_bbwidth 已计算 (upper-lower)/middle，本身已是比例值(0.05~0.3)，
        #          再除以close(如3000点)后值域变为~0.00001，双重归一化导致值域异常
        # MA: calc_ma 已计算 (short-long)/long*100，本身已是百分比变化，
        #     再除以close后值域异常，且破坏了thr_mean的比较关系
        # CCI: CCI本身是无量纲振荡指标(典型±300)，不是价格量级，不应除以close，
        #      除以close后值域随标的价格量级变化，thr_0行为不可预测
        close_feats = {'ATR', 'STDDEV', 'TRIMA', 'SMA', 'TSF', 'EMA',
                       'WMA', 'DEMA', 'KAMA'}
        pct_feats = {'HV', 'NATR'}
        raw_feats = {'ADX', 'RSI', 'CCI', 'BBW', 'BBWIDTH', 'MA',
                     'VOL_RATIO', 'VOL_CHG', 'TREND_STRENGTH',
                     'VOL_REGIME', 'MOM_CHG', 'UP_RATIO', 'MACD', 'VAR', 'LINEARREG',
                     'MFI', 'ULTOSC', 'OBV', 'AVGPRICE', 'WCLPRICE',
                     'HT_SINE', 'HT_TRENDMODE', 'CORREL'}

        name_upper = feat_name.upper()
        for cf in close_feats:
            if name_upper.startswith(cf):
                return 'close'
        for pf in pct_feats:
            if name_upper.startswith(pf):
                return 'pct'
        return 'raw'

    def get_feature_names(self) -> List[str]:
        return self._feature_names

    def get_feature_config(self) -> Dict:
        return self._feature_configs
