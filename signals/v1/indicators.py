# ft2/signals/indicators.py - 指标计算函数库
"""
指标计算函数库（基于 TA-Lib 实现）

所有函数签名统一为: calc_xxx(data: pd.DataFrame, params: dict) -> pd.Series
或: calc_xxx_single(series: pd.Series, params: dict) -> pd.Series

特点:
- 基于 TA-Lib C 库，计算速度快
- 只负责数学计算，不涉及信号标准化
- 返回原始指标值
- 可独立使用（无需加载 signals 模块）
"""

import pandas as pd
import numpy as np
import talib
from typing import Dict, Any


# =============================================================================
# 均线交叉类指标
# =============================================================================

def calc_ma_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """简单移动平均交叉信号"""
    short = params.get('short', 5)
    long = params.get('long', 20)
    
    close = data['close'].values.astype(float)
    ma_short = talib.SMA(close, timeperiod=short)
    ma_long = talib.SMA(close, timeperiod=long)
    
    return pd.Series((ma_short - ma_long) / ma_long, index=data.index)


def calc_ema_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """指数移动平均交叉信号"""
    short = params.get('short', 12)
    long = params.get('long', 26)
    
    close = data['close'].values.astype(float)
    ema_short = talib.EMA(close, timeperiod=short)
    ema_long = talib.EMA(close, timeperiod=long)
    
    return pd.Series((ema_short - ema_long) / ema_long, index=data.index)


def calc_wma_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """加权移动平均交叉信号"""
    short = params.get('short', 5)
    long = params.get('long', 20)
    
    close = data['close'].values.astype(float)
    wma_short = talib.WMA(close, timeperiod=short)
    wma_long = talib.WMA(close, timeperiod=long)
    
    return pd.Series((wma_short - wma_long) / wma_long, index=data.index)


def calc_dema_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """双指数移动平均交叉信号"""
    short = params.get('short', 12)
    long = params.get('long', 26)
    
    close = data['close'].values.astype(float)
    dema_short = talib.DEMA(close, timeperiod=short)
    dema_long = talib.DEMA(close, timeperiod=long)
    
    return pd.Series((dema_short - dema_long) / dema_long, index=data.index)


def calc_tema_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """三重指数移动平均交叉信号"""
    short = params.get('short', 12)
    long = params.get('long', 26)
    
    close = data['close'].values.astype(float)
    tema_short = talib.TEMA(close, timeperiod=short)
    tema_long = talib.TEMA(close, timeperiod=long)
    
    return pd.Series((tema_short - tema_long) / tema_long, index=data.index)


def calc_kama_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """考夫曼自适应均线交叉信号"""
    short = params.get('short', 10)
    long = params.get('long', 30)
    
    close = data['close'].values.astype(float)
    kama_short = talib.KAMA(close, timeperiod=short)
    kama_long = talib.KAMA(close, timeperiod=long)
    
    return pd.Series((kama_short - kama_long) / kama_long, index=data.index)


def calc_t3_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """T3 超平滑均线交叉信号"""
    short = params.get('short', 12)
    long = params.get('long', 26)
    vf = params.get('vf', 0.7)
    
    close = data['close'].values.astype(float)
    t3_short = talib.T3(close, timeperiod=short, vfactor=vf)
    t3_long = talib.T3(close, timeperiod=long, vfactor=vf)
    
    return pd.Series((t3_short - t3_long) / t3_long, index=data.index)


def calc_trima_cross(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """三角移动平均交叉信号"""
    short = params.get('short', 12)
    long = params.get('long', 26)
    
    close = data['close'].values.astype(float)
    trima_short = talib.TRIMA(close, timeperiod=short)
    trima_long = talib.TRIMA(close, timeperiod=long)
    
    return pd.Series((trima_short - trima_long) / trima_long, index=data.index)


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
    
    close = data['close'].values.astype(float)
    result = talib.MIDPOINT(close, timeperiod=period)
    
    return pd.Series((data['close'] - result) / data['close'], index=data.index)


def calc_midprice(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """中点价格指标（高低价中点）"""
    period = params.get('period', 20)
    
    high = data['high'].values.astype(float)
    low = data['low'].values.astype(float)
    result = talib.MIDPRICE(high, low, timeperiod=period)
    
    return pd.Series((data['close'] - result) / data['close'], index=data.index)


def calc_bbands(data: pd.DataFrame, params: Dict[str, Any]) -> pd.Series:
    """布林带指标"""
    period = params.get('period', 20)
    std_dev = params.get('std_dev', 2.0)
    
    close = data['close'].values.astype(float)
    upper, middle, lower = talib.BBANDS(close, timeperiod=period, nbdevup=std_dev, nbdevdn=std_dev)
    
    return pd.Series((close - lower) / (upper - lower) - 0.5, index=data.index)


# =============================================================================
# 核心算法函数（可独立使用，返回 numpy 数组）
# =============================================================================

def calc_kama_single(series: pd.Series, period: int = 30) -> np.ndarray:
    """
    考夫曼自适应移动平均（TA-Lib 版）
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        KAMA 值数组
    """
    return talib.KAMA(series.values.astype(float), timeperiod=period)


def calc_t3_single(series: pd.Series, period: int, vf: float = 0.7) -> np.ndarray:
    """
    T3 超平滑均线（TA-Lib 版）
    
    Args:
        series: 价格序列
        period: 计算周期
        vf: 体积因子（默认 0.7）
    
    Returns:
        T3 值数组
    """
    return talib.T3(series.values.astype(float), timeperiod=period, vfactor=vf)


def calc_trima_single(series: pd.Series, period: int) -> np.ndarray:
    """
    三角移动平均（TA-Lib 版）
    
    Args:
        series: 价格序列
        period: 计算周期
    
    Returns:
        TRIMA 值数组
    """
    return talib.TRIMA(series.values.astype(float), timeperiod=period)
