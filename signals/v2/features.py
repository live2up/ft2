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
 │     "ATR(period=[7,14])" → ATR{7}, ATR{14}                          │
 │     "EMA(period=[5,10])" → EMA{5}, EMA{10}                          │
 │                                                                      │
 │  2. regime=True       — 追加 5 个状态特征列                          │
 │     TREND_STRENGTH{20}, VOL_REGIME{20,60}, VOL_CHG{5}, ...          │
 │                                                                      │
 │  3. differences       — 对已有列做数组运算                           │
 │     BBWIDTH{30} - TSF{7} → BBWIDTH{30}_sub_TSF{7}                   │
 │                                                                      │
 │  4. ma_windows        — 为每个特征生成 MA 平滑版本                   │
 │     ATR{7} + window=10 → ATR{7}_MA                                  │
 │                                                                      │
 │  输出: pd.DataFrame (index=日期, columns=55+ 个特征)                  │
 └─────────────────────────────────────────────────────────────────────┘


============================================================================
                         参数规范对照
============================================================================

 ┌────────────────────┬──────────────────┬──────────────────────────────────┐
 │       配置项         │      语法类型    │       示例 / 展开后              │
 ├────────────────────┼──────────────────┼──────────────────────────────────┤
 │ features           │ 函数调用式        │ ATR(period=[7,14]) → ATR{7},     │
 │                    │ (关键字参数)      │ → ATR{14}                        │
 │                    │                  │ EMA(period=[5,10]) → EMA{5}       │
 ├────────────────────┼──────────────────┼──────────────────────────────────┤
 │ regime             │ 布尔开关          │ True → 5 个状态特征 ({} 列名)    │
 ├────────────────────┼──────────────────┼──────────────────────────────────┤
 │ differences        │ 列名引用式        │ ['BBWIDTH{30}', 'TSF{7}', 'sub'] │
 │                    │ [左列, 右列, 运算符]│ → BBWIDTH{30}_sub_TSF{7}        │
 ├────────────────────┼──────────────────┼──────────────────────────────────┤
 │ ma_windows         │ 整数列表          │ [10] → ATR{7} + MA → ATR{7}_MA  │
 ├────────────────────┼──────────────────┼──────────────────────────────────┤
 │ normalize          │ 布尔开关          │ True → 自动判断归一化策略         │
 └────────────────────┴──────────────────┴──────────────────────────────────┘


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
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable, Any

logger = logging.getLogger(__name__)

try:
    import talib
    HAS_TALIB = True
except ImportError:
    HAS_TALIB = False
    logger.warning("talib 未安装，部分特征（MFI/ULTOSC/AVGPRICE/WCLPRICE/HT_SINE/HT_TRENDMODE/CORREL）"
                   "将返回全零序列。建议安装 TA-Lib: pip install TA-Lib")


# ============================================================
# 纯函数式特征计算函数
# 统一签名: calc_xxx(data: pd.DataFrame, **params) -> Tuple[np.ndarray, np.ndarray]
# 返回: (原始值, MA平滑值)
# ============================================================

def _rolling_mean(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动窗口均值（尾部均值），无前瞻偏差

    使用 pandas rolling 实现，原生支持 NaN 值处理。
    性能比纯 Python 循环提升 20-50 倍。

    Args:
        arr: 输入序列
        window: 窗口大小

    Returns:
        滚动均值序列，前 window-1 天为 NaN
    """
    if len(arr) < window:
        return np.full_like(arr, np.nan, dtype=float)
    result = pd.Series(arr).rolling(window, min_periods=1).mean().values.copy()
    result[:window - 1] = np.nan
    return result


def calc_atr(data: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """平均真实波幅 (ATR)，衡量价格波动幅度的经典指标

    输出值域：归一化前为价格量级（与标的绝对价格正相关），归一化后为比例值
    """
    h, l, c = data['high'].values, data['low'].values, data['close'].values
    if HAS_TALIB:
        raw = talib.ATR(h, l, c, timeperiod=period)
    else:
        tr = np.maximum(h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1)))
        raw = _rolling_mean(tr, period)
    return raw, _rolling_mean(raw, period)


def calc_stddev(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """滚动标准差 (STDDEV)，衡量价格离散程度

    输出值域：与 ATR 类似，归一化前为价格量级，归一化后为比例值
    """
    c = data['close'].values
    if HAS_TALIB:
        raw = talib.STDDEV(c, timeperiod=period)
    else:
        raw = np.array([np.nanstd(c[max(0, i-period+1):i+1], ddof=1) for i in range(len(c))])
    return raw, _rolling_mean(raw, period)


def calc_bbwidth(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """布林带宽度 (BBWIDTH) = (上轨 - 下轨) / 中轨

    用途：衡量价格波动范围，宽度收缩预示可能趋势突破
    输出值域：已比例化（输出值约 0.02~0.30），无需再除以 close
    """
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
    """历史波动率 (HV)，年化日收益率标准差 × 100

    用途：衡量波动率绝对值，用于期权定价或波动率风格切换
    输出值域：年化百分比（如 25 表示年化波动 25%）
    """
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
    """归一化 ATR (NATR) = ATR / close × 100

    与 ATR 的区别：归一化消除了标的绝对价格的尺度影响，
    同一 NATR 值在不同标的间可比
    输出值域：百分比（如 2.5 表示 ATR = close 的 2.5%）
    """
    h, l, c = data['high'].values, data['low'].values, data['close'].values
    if HAS_TALIB:
        raw = talib.NATR(h, l, c, timeperiod=period)
    else:
        tr = np.maximum(h - l, np.abs(h - np.roll(c, 1)), np.abs(l - np.roll(c, 1)))
        atr = _rolling_mean(tr, period)
        raw = np.where(c > 0, atr / c * 100, 0)
    return raw, _rolling_mean(raw, period)


def calc_trima(data: pd.DataFrame, period: int = 40) -> Tuple[np.ndarray, np.ndarray]:
    """三角移动平均 (TRIMA)，比 SMA 更平滑，减少滞后

    通过两次移动平均实现：MA(MA(close, period/2+1), period/2+1)
    输出值域：价格量级，归一化后为比例值
    """
    c = data['close'].values
    if HAS_TALIB:
        raw = talib.TRIMA(c, timeperiod=period)
    else:
        raw = _rolling_mean(_rolling_mean(c, period // 2 + 1), period // 2 + 1)
    return raw, _rolling_mean(raw, period)


def calc_sma_val(data: pd.DataFrame, period: int = 50) -> Tuple[np.ndarray, np.ndarray]:
    """简单移动平均 (SMA)，基础趋势跟踪指标

    用途：获取价格的长期均线方向
    输出值域：价格量级，归一化后为比例值
    """
    c = data['close'].values
    raw = _rolling_mean(c, period)
    return raw, _rolling_mean(raw, period)


def calc_ma(data: pd.DataFrame, short: int = 5, long: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """均线乖离率 = (短期均线 - 长期均线) / 长期均线 × 100

    用途：衡量短期趋势强度，正值 = 短期强于长期 = 上升趋势
    输出值域：百分比（如 1.2 表示短均线高出长均线 1.2%）
    """
    c = data['close'].values
    ma_short = _rolling_mean(c, short)
    ma_long = _rolling_mean(c, long)
    raw = np.where(ma_long > 0, (ma_short - ma_long) / ma_long * 100, 0)
    return raw, _rolling_mean(raw, max(short, long))


def calc_ema(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """指数移动平均 (EMA)，对近期价格赋予更高权重

    与 SMA 的区别：EMA 对价格变化反应更快，滞后更小
    输出值域：价格量级，归一化后为比例值
    """
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
    """时间序列预测 (TSF)，线性回归外推预测

    对过去 period 个收盘价做线性回归，取回归线在最后一期的值作为预测
    输出值域：价格量级，归一化后为比例值
    """
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
    """平均趋向指数 (ADX)，衡量趋势强度（不区分方向）

    用途：ADX > 25 通常视为"趋势行情"，ADX < 20 视为"震荡行情"
    输出值域：[0, 100]，值越大趋势越强（归一化为 raw 策略，保持不变）
    """
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
    """相对强弱指标 (RSI)，衡量价格变化速度

    输出值域：**中心化版本** (raw - 50) / 50，值域 [-1, 1]
    因此 thr_0(RSI{14}) 的语义是"原始 RSI > 50" = 上涨阶段
    """
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
    """商品通道指标 (CCI)，衡量价格与统计均值的偏离程度

    用途：CCI > 100 超买，CCI < -100 超卖（经典解读）
    输出值域：无量纲（典型 ±300），不做 close 归一化
    """
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
    """MACD 柱状图 (MACD Histogram) = 2 × (DIF - DEA)

    MACD 线 = EMA(fast) - EMA(slow)，DEA 信号线 = EMA(MACD 线)
    MACD > 0 表示短期动能强于长期 = 上涨
    输出值域：价格量级，典型值约 -10~10（取决于标的）
    """
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
    """滚动方差 (VAR)，标准差平方，衡量价格波动

    输出值域：价格量级的平方，归一化后为比例
    """
    c = data['close'].values
    if HAS_TALIB:
        raw = talib.VAR(c, timeperiod=period)
    else:
        raw = np.array([np.nanvar(c[max(0, i-period+1):i+1], ddof=1) for i in range(len(c))])
    return raw, _rolling_mean(raw, period)


def calc_linearreg(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """线性回归值 (LINEARREG)，最小二乘法拟合的回归线末点值

    与 TSF 的区别：TSF 预测下一期，LINEARREG 输出当前期回归值
    输出值域：价格量级，归一化后为比例值
    """
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
    """量比 = 短期均量 / 长期均量

    用途：衡量成交量是否放量。> 1 = 放量，< 1 = 缩量
    输出值域：比例，无量纲。如 1.5 表示短期均量是长期均量的 1.5 倍
    """
    volume = data['volume'].values
    vol_short = _rolling_mean(volume, short)
    vol_long = _rolling_mean(volume, long)
    raw = np.where(vol_long > 0, vol_short / vol_long, 0)
    return raw, _rolling_mean(raw, short)


def calc_vol_chg(data: pd.DataFrame, period: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """量变率 = (当前量比 / 前期量比) - 1

    用途：衡量成交量的边际变化，正值 = 放量加速，负值 = 放量减速/缩量
    输出值域：百分比变化，无量纲。如 0.2 表示量比相比上升 20%
    """
    volume = data['volume'].values
    vol_ma = _rolling_mean(volume, period)
    vol_long = _rolling_mean(volume, period * 3)
    ratio = np.where(vol_long > 0, vol_ma / vol_long, 0)
    ratio_shifted = np.roll(ratio, period)
    ratio_shifted[:period] = np.nan
    raw = np.where(ratio_shifted > 1e-10, ratio / ratio_shifted - 1.0, 0)
    return raw, _rolling_mean(raw, period)


def calc_trend_strength(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """趋势强度 = 累计收益 / 波动率（类似 Sharpe Ratio 简化版）

    用途：衡量趋势的质量。正值 = 正收益 + 低回撤 = 强趋势；
          负值 = 负收益 + 高波动 = 弱趋势/震荡
    输出值域：无量纲比率
    """
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
    """波动率比值 = 短期波动率 / 长期波动率

    用途：识别波动率状态切换。> 1 = 近期波动增大（高波动 regime），
          < 1 = 近期波动减小（低波动 regime）
    输出值域：无量纲比例
    """
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
    """动量加速度 = 当前动量 - 前期动量

    动量 = close(t) / close(t-period) - 1（动量项的差价）
        正值 = 动量加速上升，负值 = 动量衰减
    输出值域：百分比变化，无量纲
    """
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
    """上涨比例 = 过去 period 日中日线收阳的比例

    用途：隐含市场状态。> 0.7 = 强势上涨，< 0.3 = 弱势下跌，0.5 附近 = 震荡
    输出值域：[0, 1]
    """
    c = data['close'].values
    is_up = (c > np.roll(c, 1)).astype(float)
    is_up[0] = np.nan
    raw = np.full(len(c), np.nan)
    for i in range(period - 1, len(c)):
        seg = is_up[i - period + 1:i + 1]
        raw[i] = np.nanmean(seg)
    return raw, _rolling_mean(raw, period)


def calc_wma(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """加权移动平均 (WMA)，近期数据权重线性递增

    用途：与 EMA 类似但权重模式不同，价格序列的平滑版本
    输出值域：价格量级
    """
    c = data['close'].values
    if HAS_TALIB:
        wma = talib.WMA(c, timeperiod=period)
    else:
        wma = _rolling_mean(c, period)
    return wma, _rolling_mean(wma, period)


def calc_dema(data: pd.DataFrame, period: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """双重指数移动平均 (DEMA)，减少 EMA 的滞后

    公式：DEMA = 2 × EMA - EMA(EMA)
    比普通 EMA 响应更快，适用于对延迟敏感的策略
    输出值域：价格量级
    """
    c = data['close'].values
    if HAS_TALIB:
        dema = talib.DEMA(c, timeperiod=period)
    else:
        dema = _rolling_mean(c, period)
    return dema, _rolling_mean(dema, period)


def calc_kama(data: pd.DataFrame, period: int = 30) -> Tuple[np.ndarray, np.ndarray]:
    """考夫曼自适应移动平均 (KAMA)，根据市场波动自动调整平滑速度

    用途：在趋势行情中快速跟进，在震荡行情中趋缓减少假突破
    输出值域：价格量级
    """
    c = data['close'].values
    if HAS_TALIB:
        kama = talib.KAMA(c, timeperiod=period)
    else:
        kama = _rolling_mean(c, period)
    return kama, _rolling_mean(kama, period)


def calc_mfi(data: pd.DataFrame, period: int = 14) -> Tuple[np.ndarray, np.ndarray]:
    """资金流量指标 (MFI)，带成交量的 RSI 版本

    用途：识别价量背离。中心化版本 (MFI - 50)，> 0 表示资金流入
    输出值域：中心化 [-50, 50]
    """
    if HAS_TALIB:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        volume = data['volume'].values.astype(float)
        mfi = talib.MFI(high, low, close, volume, timeperiod=period)
        raw = mfi - 50.0
    else:
        logger.warning("calc_mfi: 缺少 talib，返回全零序列")
        raw = np.zeros(len(data))
    return raw, _rolling_mean(raw, period)


def calc_ultosc(data: pd.DataFrame, period1: int = 7, period2: int = 14,
                period3: int = 28) -> Tuple[np.ndarray, np.ndarray]:
    """终极振荡器 (ULTOSC)，多周期加权衡量超买超卖

    用途：综合短/中/长三个周期的动量状态，减少单一周期假信号
    输出值域：中心化版本 (ULTOSC - 50)，> 0 = 上涨动能
    """
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
        logger.warning("calc_ultosc: 缺少 talib，返回全零序列")
        raw = np.zeros(len(data))
    return raw, _rolling_mean(raw, period2)


def calc_obv(data: pd.DataFrame, signal_period: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    """能量潮 (OBV) 变化率，基于量价关系的累积动量指标

    用途：OBV 上升 = 量价配合上涨，OBV 下降 = 量价背离
    输出值域：OBV 的百分比变化 × 100
    """
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
    """平均价格偏差 = (close - avgprice) / avgprice × 100

    avgprice = (open + high + low + close) / 4
    用途：衡量收盘价偏离日均价的程度
    输出值域：百分比，正值 = 收盘在日均价之上
    """
    if HAS_TALIB:
        open_p = data['open'].values.astype(float)
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        avgprice = talib.AVGPRICE(open_p, high, low, close)
        raw = np.where(avgprice > 0, (close - avgprice) / avgprice * 100, 0)
    else:
        logger.warning("calc_avgprice: 缺少 talib，返回全零序列")
        raw = np.zeros(len(data))
    return raw, _rolling_mean(raw, 20)


def calc_wclprice(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """加权收盘价偏差 = (close - wclprice) / wclprice × 100

    wclprice = (high + low + 2 × close) / 4（权重更多放在收盘价）
    用途：衡量收盘价相对于加权均价的位置
    输出值域：百分比，正值 = 收盘偏向上轨
    """
    if HAS_TALIB:
        high = data['high'].values.astype(float)
        low = data['low'].values.astype(float)
        close = data['close'].values.astype(float)
        wclprice = talib.WCLPRICE(high, low, close)
        raw = np.where(wclprice > 0, (close - wclprice) / wclprice * 100, 0)
    else:
        logger.warning("calc_wclprice: 缺少 talib，返回全零序列")
        raw = np.zeros(len(data))
    return raw, _rolling_mean(raw, 20)


def calc_ht_sine(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """希尔伯特变换正弦波差值 (HT_SINE)，识别市场周期相位

    用途：sine - leadsine 的差值，用于判断市场处于上涨/下跌周期哪个阶段
    输出值域：[-1, 1] 振荡
    """
    if HAS_TALIB:
        close = data['close'].values.astype(float)
        sine, leadsine = talib.HT_SINE(close)
        raw = sine - leadsine
    else:
        logger.warning("calc_ht_sine: 缺少 talib，返回全零序列")
        raw = np.zeros(len(data))
    return raw, raw


def calc_ht_trendmode(data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """希尔伯特变换趋势模式 (HT_TRENDMODE)

    用途：量化判断市场处于趋势模式 (> 0) 还是循环模式 (< 0)
    输出值域：中心化 {-0.5, 0.5}
    """
    if HAS_TALIB:
        close = data['close'].values.astype(float)
        ht_trendmode = talib.HT_TRENDMODE(close)
        raw = ht_trendmode - 0.5
    else:
        logger.warning("calc_ht_trendmode: 缺少 talib，返回全零序列")
        raw = np.zeros(len(data))
    return raw, raw


def calc_correl(data: pd.DataFrame, period: int = 10) -> Tuple[np.ndarray, np.ndarray]:
    close = data['close'].values.astype(float)
    if HAS_TALIB:
        time_seq = np.arange(len(close), dtype=float)
        correl = talib.CORREL(close, time_seq, timeperiod=period)
    else:
        logger.warning("calc_correl: 缺少 talib，返回全零序列")
        correl = np.zeros(len(close))
    return correl, _rolling_mean(correl, period)


def calc_roc(data: pd.DataFrame, period: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """Rate of Change: close(t)/close(t-period) - 1"""
    c = data['close'].values
    raw = np.full(len(c), np.nan)
    for i in range(period, len(c)):
        if c[i - period] > 0:
            raw[i] = c[i] / c[i - period] - 1
    return raw, _rolling_mean(raw, period)


def calc_mom_ratio(data: pd.DataFrame, short: int = 5, long: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """动量比: ROC_short / |ROC_long|，用于比较短/长期动量强弱"""
    c = data['close'].values
    roc_short = np.full(len(c), np.nan)
    roc_long = np.full(len(c), np.nan)
    for i in range(long, len(c)):
        if c[i - long] > 0:
            roc_long[i] = c[i] / c[i - long] - 1
        if i >= short and c[i - short] > 0:
            roc_short[i] = c[i] / c[i - short] - 1
    raw = np.where(np.abs(roc_long) > 1e-10, roc_short / np.abs(roc_long), 0)
    return raw, _rolling_mean(raw, short)


# ============================================================
# 归一化处理
# ============================================================

def normalize_close(raw_val: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    """close 归一化：原始值 / close，将价格量级转为比例

    适用：ATR, STDDEV, TRIMA, SMA, EMA, TSF 等价格量级特征
    效果：使不同价格的标的在统一尺度上可比
    """
    close = data['close'].values
    return np.where(close > 0, raw_val / close, raw_val)


def normalize_pct(raw_val: np.ndarray, data: pd.DataFrame) -> np.ndarray:
    """百分比归一化：原始值 / 100，将百分比转为小数

    适用：HV（历史波动率，如 25 → 0.25）、NATR
    """
    return raw_val / 100.0


def normalize_raw(raw_val: np.ndarray, data: Optional[pd.DataFrame] = None) -> np.ndarray:
    """原始值不变：不执行任何归一化

    适用：RSI(中心化[-1,1])、ADX([0,100])、MACD、BBWIDTH(已比例化) 等
    这些特征值域本身已经是无量纲的，不需要额外处理
    """
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
    'ROC': calc_roc,
    'MOM_RATIO': calc_mom_ratio,
}


def register_feature(name: str, calc_fn: Callable):
    """注册自定义特征计算函数到全局注册表

    用法:
        def calc_my_feature(data, period=10):
            ...
        register_feature("MY_FEATURE", calc_my_feature)

    注册后即可在 DEFAULT_CONFIG 中通过 'MY_FEATURE(period=10)' 调用，
    或直接在表达式中以 MY_FEATURE{10} 引用该特征列。
    """
    _FEATURE_CALC_REGISTRY[name.upper()] = calc_fn


# 默认配置：AIdev V5特征集
DEFAULT_CONFIG = {
    'features': {
        # [重构] 所有特征配置改用严格关键字参数 + 列表展开语法
        # 旧写法 'RSI(5,7,10,14,20,30)' 禁止: 纯位置参数, 不知5/7/10对应哪个形参
        # 新写法 'RSI(period=[5,7,10,14,20,30])' 明确: period参数展开为6列
        # 列表值做叉积展开, 标量值在所有实例中固定
        'volatility': [
            'ATR(period=[3,5,7,10,14,20,30])',
            'STDDEV(period=[10,15,20,25,30,40])',
            'BBWIDTH(period=[10,15,20,25,30,40])',
            'HV(period=[10,20,30,60])',
            'NATR(period=[7,10,14,20,30])',
        ],
        'trend': [
            'TRIMA(period=[20,30,40,50,60])', 'SMA(period=50)',
            'MA(short=5,long=20)',
            'EMA(period=[5,10,15,20,30,40,60])',
            'WMA(period=20)', 'DEMA(period=20)', 'KAMA(period=10)',
        ],
        'statistic': [
            'TSF(period=[3,5,7,10,14,20])',
            'VAR(period=[10,15,20,25,30])',
            'LINEARREG(period=[10,15,20,25,30])',
            'CORREL(period=[5,10,15,20,25])',
        ],
        'momentum': [
            'ADX(period=[7,10,14,20,25,30])',
            'RSI(period=[5,7,10,14,20,30])',
            'CCI(period=[7,10,14,20,30])',
            'MACD(fast=12,slow=26,signal=9)',
            'MFI(period=[7,10,14,20])', 'ULTOSC()',
            'ROC(period=[5,10,20,60])',
        ],
        'momentum_ratio': [
            'MOM_RATIO(short=[5,20],long=[20,60])',
        ],
        'price_transform': ['AVGPRICE()', 'WCLPRICE()'],
        'cycle': ['HT_SINE()', 'HT_TRENDMODE()'],
        # [修复] VOL_RATIO 旧写法 'VOL_RATIO(5,10)' 把5和10都映射到short, long默认20
        # 新写法明确指定: short=[5,10]展开, long=20固定 → VOL_RATIO_5_20, VOL_RATIO_10_20
        'volume': [
            'VOL_RATIO(short=[3,5,10,15,20],long=20)',
            'OBV(signal_period=[5,10,15,20])',
        ],
    },
    'regime': True,
    'market_breadth': False,
    'differences': [
        ['BBWIDTH{30}', 'TSF{7}', 'sub'],
        ['ATR{7}', 'TSF{7}', 'sub'],
        ['STDDEV{20}', 'TSF{7}', 'sub'],
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
        """递归合并配置字典，支持嵌套 dict 合并（非浅层替换）

        用途：允许用户自定义 config 覆盖 DEFAULT_CONFIG 的指定子项，
        而未被覆盖的项目保持默认值。例如只覆盖 'momentum' 配置，
        'volatility'/'trend' 等子部分不受影响。

        注意：列表按照整体替换策略（不是按元素合并）
        """
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
        """解析特征配置字符串为 (name, params) 对
        [重构] 严格关键字参数规范，禁止隐式位置参数展开

        支持的语法:
          'RSI(period=[5,7,10,14,20,30])'  — 关键字+列表展开 → 6列
          'ATR(period=[7,14])'              — 关键字+列表展开 → 2列
          'MACD(fast=12,slow=26,signal=9)'  — 关键字+标量 → 1列
          'ULTOSC()'                         — 无参数 → 1列

        禁止的语法:
          'RSI(5,7,10,14,20,30)'  — 纯位置参数: 不知道5/7/10各自对应哪个形参
          'VOL_RATIO(5,10)'       — 多位置参数: 5→short? 10→long? 语义歧义

        列表展开规则: 参数值为 [v1,v2,...] 时, 该参数做展开;
        多个列表参数做叉积展开, 标量参数在所有展开实例中保持固定。
        """
        name_end = feat_str.find('(')
        if name_end == -1:
            return feat_str, {}
        name = feat_str[:name_end]
        params_str = feat_str[name_end + 1:-1].strip()
        if not params_str:
            return name.upper(), {}

        params = {}
        positional_values = []
        # [修复] 不能简单 split(','), 因为列表值如 period=[7,14] 内部也有逗号
        # 改为逐字符扫描, 跟踪方括号深度, 只在顶层逗号处分割
        top_level_parts = []
        depth = 0
        current = []
        for ch in params_str:
            if ch == '[':
                depth += 1
                current.append(ch)
            elif ch == ']':
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0:
                top_level_parts.append(''.join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            top_level_parts.append(''.join(current).strip())

        for p in top_level_parts:
            if not p:
                continue
            if '=' in p:
                k, v = p.split('=', 1)
                k = k.strip()
                v = v.strip()
                if v.startswith('[') and v.endswith(']'):
                    params[k] = [int(x.strip()) for x in v[1:-1].split(',') if x.strip()]
                else:
                    params[k] = int(v)
            else:
                positional_values.append(int(p))

        # [严格] 禁止纯位置参数 — 语义不明确
        if positional_values:
            raise ValueError(
                f"特征 '{name}' 配置使用了位置参数 {positional_values}，"
                f"请使用关键字参数格式。例如: '{name}(period={positional_values[0]})' "
                f"或 '{name}(period={positional_values})' (展开多列)"
            )

        return name.upper(), params

    def _expand_params(self, name: str, params: Dict, calc_fn: Callable) -> List[Dict]:
        """展开多参数为多个特征配置
        [重构] 叉积展开逻辑，替代旧的 __default_params 隐式映射

        展开规则:
        - 标量参数: 在所有展开实例中保持固定
        - 列表参数: 做叉积展开，每个值产生一个独立实例
        - 无列表参数: 返回单个实例（不展开）

        示例:
          RSI(period=[5,7,10]) → [{'period':5}, {'period':7}, {'period':10}]
          VOL_RATIO(short=[5,10],long=20) → [{'short':5,'long':20}, {'short':10,'long':20}]
          MACD(fast=12,slow=26,signal=9) → [{'fast':12,'slow':26,'signal':9}]
        """
        import itertools

        # 分离列表参数和标量参数
        list_params = {}
        scalar_params = {}
        for k, v in params.items():
            if isinstance(v, list):
                list_params[k] = v
            else:
                scalar_params[k] = v

        if not list_params:
            return [params] if params else [{}]

        # 叉积展开: 生成所有列表参数值的组合
        keys = list(list_params.keys())
        value_lists = [list_params[k] for k in keys]
        combinations = list(itertools.product(*value_lists))

        result = []
        for combo in combinations:
            expanded = dict(scalar_params)
            for k, v in zip(keys, combo):
                expanded[k] = v
            result.append(expanded)

        return result

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
                # [修复] 列名拼接按函数签名参数顺序，而非字典插入顺序
                # 旧实现: pcfg.values() 顺序不可控，VOL_RATIO 可能变成 VOL_RATIO_20_5
                # 新实现: 按 calc_fn 签名中参数定义的顺序拼接，保证 VOL_RATIO_5_20
                import inspect
                sig_param_names = [pn for pn in inspect.signature(calc_fn).parameters.keys() if pn != 'data']
                for pcfg in expanded:
                    if pcfg:
                        # 按函数签名顺序排列参数值，确保列名一致性
                        ordered_values = [pcfg.get(pn, '') for pn in sig_param_names if pn in pcfg]
                        col_name = f"{name}{{{','.join(str(v) for v in ordered_values)}}}"
                    else:
                        col_name = name
                    self._feature_names.append(col_name)
                    self._feature_configs[col_name] = {
                        'name': name,
                        'calc_fn': calc_fn,
                        'params': pcfg,
                        'category': category,
                    }

        # Regime特征（统一 {} 格式列名）
        if self.config.get('regime'):
            regime_feats = [
                ('TREND_STRENGTH{20}', calc_trend_strength, {'period': 20}),
                ('VOL_REGIME{20,60}', calc_vol_regime, {'short': 20, 'long': 60}),
                ('VOL_CHG{5}', calc_vol_chg, {'period': 5}),
                ('MOM_CHG{5}', calc_mom_chg, {'period': 5}),
                ('UP_RATIO{10}', calc_up_ratio, {'period': 10}),
            ]
            for col_name, calc_fn, params in regime_feats:
                self._feature_names.append(col_name)
                self._feature_configs[col_name] = {
                    'name': col_name,
                    'calc_fn': calc_fn,
                    'params': params,
                    'category': 'regime',
                }

        # [新增] 市场广度特征接入 (market_breadth=True 时启用)
        # 数据来源：需在 DataFrame 中预置 advance/decline/new_highs/new_lows 等列
        # 默认关闭，因为需要外部数据源，不是所有场景都可用
        if self.config.get('market_breadth'):
            from .market_breadth import register_breadth_features
            register_breadth_features()
            breadth_feats = [
                ('ADV_DEC_RATIO', _FEATURE_CALC_REGISTRY.get('ADV_DEC_RATIO'), {}),
                ('NEW_HIGH_LOW', _FEATURE_CALC_REGISTRY.get('NEW_HIGH_LOW'), {}),
                ('MCCLELLAN_OSC', _FEATURE_CALC_REGISTRY.get('MCCLELLAN_OSC'), {}),
                ('ARMS_INDEX', _FEATURE_CALC_REGISTRY.get('ARMS_INDEX'), {}),
            ]
            for col_name, calc_fn, params in breadth_feats:
                if calc_fn is None:
                    continue
                self._feature_names.append(col_name)
                self._feature_configs[col_name] = {
                    'name': col_name,
                    'calc_fn': calc_fn,
                    'params': params,
                    'category': 'market_breadth',
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
            # 差异列命名: 将参数拆解重组，确保表达式解析器可正确识别为单一特征列
            # 旧: BBWIDTH{30}_sub_TSF{7}  → 解析器拆成两个特征，错误
            # 新: BBWIDTH_TSF_sub{30,7}    → 单一 FEATURE 节点，正确
            import re as _re
            m1 = _re.match(r'^(\w+)\{([^}]+)\}$', left_col)
            m2 = _re.match(r'^(\w+)\{([^}]+)\}$', right_col)
            if m1 and m2:
                derived_name = (f"{m1.group(1)}_{m2.group(1)}_{op}"
                                f"{{{m1.group(2)},{m2.group(2)}}}")
            else:
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

        close_feats: 原始值为价格量级, 需除以 close 归一化为比例
        pct_feats:   原始值为百分比, 需除以 100
        raw_feats:   原始值已归一化/无量纲/振荡类, 不做额外处理

        ⚠️ 阈值选择建议
           close 归一化后（如 EMA{20} 的值 = EMA/close），中性点为 1 而非 0。
           → thr_0(EMA{20}) ≡ 1.0 恒成立（始终 > 0）
           → 改用 thr_1(EMA{20}) = "EMA/close > 1" = 价格在均线上方  ✅
           同理 thr_0.5(VOL_RATIO{10,20}) = "量比 > 0.5" 等。
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
                     'ROC', 'MOM_RATIO',
                     'VOL_RATIO', 'VOL_CHG', 'TREND_STRENGTH',
                     'VOL_REGIME', 'MOM_CHG', 'UP_RATIO', 'MACD', 'VAR', 'LINEARREG',
                     'MFI', 'ULTOSC', 'OBV', 'AVGPRICE', 'WCLPRICE',
                     'HT_SINE', 'HT_TRENDMODE', 'CORREL',
                     # [新增] 市场广度特征：本身已是比例/差值，不需要归一化
                     'ADV_DEC_RATIO', 'NEW_HIGH_LOW', 'MCCLELLAN_OSC', 'ARMS_INDEX'}

        name_upper = feat_name.upper()
        for cf in close_feats:
            if name_upper.startswith(cf):
                return 'close'
        for pf in pct_feats:
            if name_upper.startswith(pf):
                return 'pct'
        return 'raw'

    def get_feature_names(self) -> List[str]:
        """获取所有特征列名列表

        Returns:
            [\"RSI{5}\", \"RSI{14}\", \"MACD{12,26,9}\", ...]
        """
        return self._feature_names

    def get_feature_config(self) -> Dict:
        """获取每个特征列的完整配置（计算函数、参数、所属分类等）

        Returns:
            {col_name: {'name': ..., 'calc_fn': ..., 'params': ..., 'category': ...}}
        """
        return self._feature_configs
