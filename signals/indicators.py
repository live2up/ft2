# ft2/signals/indicators.py - 指标计算函数库
"""
纯指标计算函数库

所有函数签名统一为: calc_xxx(data: pd.DataFrame, params: dict) -> pd.Series
或: calc_xxx_single(series: pd.Series, params: dict) -> pd.Series

特点:
- 只负责数学计算，不涉及信号标准化
- 返回原始指标值
- 可独立使用（无需加载 signals 模块）
"""

import pandas as pd
import numpy as np
from typing import Dict, Any


# =============================================================================
# 均线交叉类指标
# =============================================================================

def calc_ma_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """简单移动平均交叉信号"""
    short = params.get('short', 5)
    long = params.get('long', 20)
    
    ma_short = data['close'].rolling(short).mean()
    ma_long = data['close'].rolling(long).mean()
    
    return (ma_short - ma_long) / ma_long


def calc_ema_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """指数移动平均交叉信号"""
    short = params.get('short', 12)
    long = params.get('long', 26)
    
    ema_short = data['close'].ewm(span=short).mean()
    ema_long = data['close'].ewm(span=long).mean()
    
    return (ema_short - ema_long) / ema_long


def calc_wma_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """加权移动平均交叉信号"""
    short = params.get('short', 5)
    long = params.get('long', 20)
    
    wma_short = _calc_wma(data['close'], short)
    wma_long = _calc_wma(data['close'], long)
    
    return (wma_short - wma_long) / wma_long


def calc_dema_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """双指数移动平均交叉信号"""
    short = params.get('short', 12)
    long = params.get('long', 26)
    
    dema_short = _calc_dema(data['close'], short)
    dema_long = _calc_dema(data['close'], long)
    
    return (dema_short - dema_long) / dema_long


def calc_tema_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """三重指数移动平均交叉信号"""
    short = params.get('short', 12)
    long = params.get('long', 26)
    
    tema_short = _calc_tema(data['close'], short)
    tema_long = _calc_tema(data['close'], long)
    
    return (tema_short - tema_long) / tema_long


def calc_kama_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """考夫曼自适应均线交叉信号"""
    short = params.get('short', 10)
    long = params.get('long', 30)
    
    kama_short = calc_kama_single(data['close'], short)
    kama_long = calc_kama_single(data['close'], long)
    
    return (kama_short - kama_long) / kama_long


def calc_t3_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """T3 超平滑均线交叉信号"""
    short = params.get('short', 12)
    long = params.get('long', 26)
    vf = params.get('vf', 0.7)
    
    t3_short = calc_t3_single(data['close'], short, vf)
    t3_long = calc_t3_single(data['close'], long, vf)
    
    return (t3_short - t3_long) / t3_long


def calc_trima_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """三角移动平均交叉信号"""
    short = params.get('short', 12)
    long = params.get('long', 26)
    
    trima_short = calc_trima_single(data['close'], short)
    trima_long = calc_trima_single(data['close'], long)
    
    return (trima_short - trima_long) / trima_long


# =============================================================================
# 单值指标（非交叉类）
# =============================================================================

def calc_accbands(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """加速通道指标"""
    period = params.get('period', 20)
    
    upper = data['high'].rolling(period).max()
    lower = data['low'].rolling(period).min()
    
    signal = (data['close'] - lower) / (upper - lower) - 0.5
    return signal.fillna(0)


def calc_midpoint(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """中点价格指标"""
    period = params.get('period', 20)
    
    highest = data['high'].rolling(period).max()
    lowest = data['low'].rolling(period).min()
    
    return (data['close'] - (highest + lowest) / 2) / data['close']


def calc_midprice(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """中点价格指标（另一种算法）"""
    period = params.get('period', 20)
    
    highest = data['high'].rolling(period).max()
    lowest = data['low'].rolling(period).min()
    
    return (highest + lowest) / 2


# =============================================================================
# 核心算法函数（可独立使用）
# =============================================================================

def calc_kama_single(series: pd.Series, period: int = 30, eff_period: int = 10, 
                     fast_len: int = 2, slow_len: int = 30) -> pd.Series:
    """
    考夫曼自适应移动平均
    
    Args:
        series: 价格序列
        period: 计算周期
        eff_period: 效率比率周期
        fast_len: 快速平滑系数
        slow_len: 慢速平滑系数
    
    Returns:
        KAMA 值序列
    """
    change = series.diff(period)
    volatility = series.diff().abs().rolling(eff_period).sum()
    er = change.abs() / volatility
    er = er.fillna(0)
    
    fast_sc = 2.0 / (fast_len + 1)
    slow_sc = 2.0 / (slow_len + 1)
    sc = (er * (fast_sc - slow_sc) + slow_sc) ** 2
    
    kama = pd.Series(index=series.index, dtype=float)
    kama.iloc[0] = series.iloc[0]
    for i in range(1, len(series)):
        if pd.isna(series.iloc[i]):
            kama.iloc[i] = kama.iloc[i-1]
        else:
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (series.iloc[i] - kama.iloc[i-1])
    
    return kama


def calc_t3_single(series: pd.Series, period: int, vf: float = 0.7) -> pd.Series:
    """
    T3 超平滑均线
    
    Args:
        series: 价格序列
        period: 计算周期
        vf: 体积因子（默认 0.7）
    
    Returns:
        T3 值序列
    """
    c1 = -vf * vf * vf
    c2 = 3 * vf * vf + 3 * vf * vf * vf
    c3 = -6 * vf * vf - 3 * vf - 3 * vf * vf * vf
    c4 = 1 + 3 * vf + vf * vf * vf + 3 * vf * vf
    
    e1 = series.ewm(span=period).mean()
    e2 = e1.ewm(span=period).mean()
    e3 = e2.ewm(span=period).mean()
    e4 = e3.ewm(span=period).mean()
    e5 = e4.ewm(span=period).mean()
    e6 = e5.ewm(span=period).mean()
    
    return c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3


def calc_trima_single(series: pd.Series, period: int) -> pd.Series:
    """
    三角移动平均
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        TRIMA 值序列
    """
    if period % 2 == 0:
        n = period + 1
        weights = np.arange(1, n // 2 + 1)
        weights = np.concatenate([weights, weights[::-1]])
    else:
        n = period
        weights = np.arange(1, n // 2 + 1)
        weights = np.concatenate([weights, [n // 2 + 1], weights[::-1]])
    
    weights = weights / weights.sum()
    return series.rolling(period).apply(lambda x: np.dot(x, weights), raw=True)


# =============================================================================
# 辅助函数
# =============================================================================

def _calc_wma(series: pd.Series, period: int) -> pd.Series:
    """加权移动平均（内部使用）"""
    weights = np.arange(1, period + 1)
    wma = series.rolling(period).apply(
        lambda x: np.dot(x, weights) / weights.sum(), raw=True
    )
    return wma


def _calc_dema(series: pd.Series, period: int) -> pd.Series:
    """双指数移动平均（内部使用）"""
    ema = series.ewm(span=period).mean()
    ema_of_ema = ema.ewm(span=period).mean()
    return 2 * ema - ema_of_ema


def _calc_tema(series: pd.Series, period: int) -> pd.Series:
    """三重指数移动平均（内部使用）"""
    ema = series.ewm(span=period).mean()
    ema_of_ema = ema.ewm(span=period).mean()
    ema_of_ema_of_ema = ema_of_ema.ewm(span=period).mean()
    return 3 * ema - 3 * ema_of_ema + ema_of_ema_of_ema
