# signal/threshold.py - 阈值策略
"""
阈值策略

将连续信号转换为离散交易信号
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple
import pandas as pd
import numpy as np

from .base import SignalDirection


class ThresholdPolicy(ABC):
    """
    阈值策略基类
    
    将连续信号值转换为买卖信号
    """
    
    @abstractmethod
    def apply(self, value: float) -> SignalDirection:
        """
        应用阈值策略
        
        Args:
            value: 信号值
            
        Returns:
            SignalDirection: 信号方向
        """
        pass
    
    @abstractmethod
    def apply_series(self, series: pd.Series) -> pd.Series:
        """
        应用阈值策略到序列
        
        Args:
            series: 信号序列
            
        Returns:
            pd.Series: 方向序列 -1/0/1
        """
        pass


class SimpleThreshold(ThresholdPolicy):
    """
    简单阈值策略
    
    规则：
    - value > upper_threshold → LONG
    - value < lower_threshold → SHORT
    - otherwise → NEUTRAL
    """
    
    def __init__(
        self,
        upper_threshold: float = 0,
        lower_threshold: float = 0,
        symmetric: bool = True
    ):
        self.upper_threshold = upper_threshold
        self.lower_threshold = lower_threshold
        
        if symmetric:
            self.lower_threshold = -upper_threshold
    
    def apply(self, value: float) -> SignalDirection:
        if value > self.upper_threshold:
            return SignalDirection.LONG
        elif value < self.lower_threshold:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL
    
    def apply_series(self, series: pd.Series) -> pd.Series:
        result = pd.Series(0, index=series.index)
        result[series > self.upper_threshold] = 1
        result[series < self.lower_threshold] = -1
        return result


class PercentileThreshold(ThresholdPolicy):
    """
    分位数阈值策略
    
    根据历史分位数确定阈值
    - 高于 75% 分位 → LONG
    - 低于 25% 分位 → SHORT
    """
    
    def __init__(self, upper_percentile: float = 75, lower_percentile: float = 25):
        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile
        self._upper = None
        self._lower = None
    
    def fit(self, series: pd.Series):
        """拟合阈值"""
        self._upper = series.quantile(self.upper_percentile / 100)
        self._lower = series.quantile(self.lower_percentile / 100)
    
    def apply(self, value: float) -> SignalDirection:
        if self._upper is None or self._lower is None:
            raise ValueError("Threshold not fitted. Call fit() first.")
        
        if value > self._upper:
            return SignalDirection.LONG
        elif value < self._lower:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL
    
    def apply_series(self, series: pd.Series) -> pd.Series:
        if self._upper is None or self._lower is None:
            self.fit(series)
        
        result = pd.Series(0, index=series.index)
        result[series > self._upper] = 1
        result[series < self._lower] = -1
        return result


class ZScoreThreshold(ThresholdPolicy):
    """
    Z-Score 阈值策略
    
    基于标准分的阈值
    - z > threshold → LONG
    - z < -threshold → SHORT
    """
    
    def __init__(self, threshold: float = 1.0):
        self.threshold = threshold
        self._mean = None
        self._std = None
    
    def fit(self, series: pd.Series):
        """拟合"""
        self._mean = series.mean()
        self._std = series.std()
    
    def apply(self, value: float) -> SignalDirection:
        if self._mean is None or self._std is None:
            raise ValueError("Threshold not fitted. Call fit() first.")
        
        z = (value - self._mean) / self._std if self._std > 0 else 0
        
        if z > self.threshold:
            return SignalDirection.LONG
        elif z < -self.threshold:
            return SignalDirection.SHORT
        else:
            return SignalDirection.NEUTRAL
    
    def apply_series(self, series: pd.Series) -> pd.Series:
        if self._mean is None or self._std is None:
            self.fit(series)
        
        z_score = (series - self._mean) / self._std if self._std > 0 else 0
        
        result = pd.Series(0, index=series.index)
        result[z_score > self.threshold] = 1
        result[z_score < -self.threshold] = -1
        return result


class MovingThreshold(ThresholdPolicy):
    """
    移动阈值策略
    
    使用移动平均作为动态阈值
    """
    
    def __init__(self, period: int = 20, buffer: float = 0):
        self.period = period
        self.buffer = buffer
    
    def apply_series(self, series: pd.Series) -> pd.Series:
        ma = series.rolling(self.period).mean()
        
        upper = ma + self.buffer
        lower = ma - self.buffer
        
        result = pd.Series(0, index=series.index)
        result[series > upper] = 1
        result[series < lower] = -1
        return result


class DualThreshold(ThresholdPolicy):
    """
    双阈值策略
    
    区分入场和出场阈值（避免频繁交易）
    - 高于 entry_long → LONG
    - 低于 exit_short → SHORT
    - 其他 → 保持原状态
    """
    
    def __init__(
        self,
        entry_long: float = 0,
        exit_short: float = 0,
        entry_short: float = 0,
        exit_long: float = 0
    ):
        self.entry_long = entry_long
        self.exit_short = exit_short
        self.entry_short = entry_short
        self.exit_long = exit_long
    
    def apply(self, value: float, current_position: int = 0) -> SignalDirection:
        """
        应用双阈值
        
        Args:
            value: 信号值
            current_position: 当前持仓 (1=多头, -1=空头, 0=空仓)
        """
        if current_position == 0:  # 空仓
            if value > self.entry_long:
                return SignalDirection.LONG
            elif value < self.entry_short:
                return SignalDirection.SHORT
        elif current_position == 1:  # 多头
            if value < self.exit_short:
                return SignalDirection.SHORT
        elif current_position == -1:  # 空头
            if value > self.exit_long:
                return SignalDirection.LONG
        
        return SignalDirection.NEUTRAL
    
    def apply_series(self, series: pd.Series) -> pd.Series:
        result = pd.Series(0, index=series.index)
        position = 0
        
        for i, value in enumerate(series):
            if position == 0:
                if value > self.entry_long:
                    position = 1
                elif value < self.entry_short:
                    position = -1
            elif position == 1:
                if value < self.exit_short:
                    position = -1
            elif position == -1:
                if value > self.exit_long:
                    position = 1
            
            result.iloc[i] = position
        
        return result


# 预定义阈值策略工厂
THRESHOLD_PRESETS = {
    'conservative': SimpleThreshold(upper_threshold=0.5, lower_threshold=-0.5),
    'moderate': SimpleThreshold(upper_threshold=0.2, lower_threshold=-0.2),
    'aggressive': SimpleThreshold(upper_threshold=0.0, lower_threshold=0.0),  # 只要转正/负
    'macd': SimpleThreshold(upper_threshold=0, lower_threshold=0),
    'rsi': SimpleThreshold(upper_threshold=20, lower_threshold=-20),  # RSI-50 中心化
    'boll': SimpleThreshold(upper_threshold=0.2, lower_threshold=-0.2),
}


def get_threshold_preset(name: str) -> ThresholdPolicy:
    """获取预定义阈值策略"""
    if name not in THRESHOLD_PRESETS:
        raise ValueError(f"Unknown preset: {name}. Available: {list(THRESHOLD_PRESETS.keys())}")
    return THRESHOLD_PRESETS[name]
