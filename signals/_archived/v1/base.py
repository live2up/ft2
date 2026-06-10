# signal/base.py - 信号基类
"""
信号基类定义
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np


class SignalType(Enum):
    """信号类型"""
    CONTINUOUS = "continuous"    # 连续值信号（如 MACD 值）
    DISCRETE = "discrete"        # 离散信号（如 金叉/死叉）
    DIRECTIONAL = "directional"  # 方向信号（1/0/-1）


class SignalDirection(Enum):
    """信号方向"""
    LONG = 1      # 做多
    NEUTRAL = 0   # 中立/持有
    SHORT = -1    # 做空


@dataclass
class Signal:
    """
    信号基类
    
    包含：
    - 原始值（连续）
    - 方向（离散）
    - 元数据（名称、参数、时间）
    """
    name: str
    value: float
    direction: SignalDirection = SignalDirection.NEUTRAL
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
    
    @property
    def is_long(self) -> bool:
        return self.direction == SignalDirection.LONG
    
    @property
    def is_short(self) -> bool:
        return self.direction == SignalDirection.SHORT
    
    @property
    def is_neutral(self) -> bool:
        return self.direction == SignalDirection.NEUTRAL
    
    def to_dict(self) -> dict:
        return {
            'name': self.name,
            'value': self.value,
            'direction': self.direction.value,
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata,
        }


@dataclass
class TradingSignal:
    """
    交易信号（最终输出）
    
    用于驱动交易决策
    """
    direction: SignalDirection
    strength: float = 1.0          # 信号强度 0-1
    signals: List[Signal] = field(default_factory=list)  # 组成该交易的原始信号
    timestamp: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        # 计算平均信号强度
        if self.signals and self.strength == 1.0:
            self.strength = np.mean([abs(s.value) for s in self.signals])
    
    @classmethod
    def long(cls, strength: float = 1.0, signals: List[Signal] = None) -> 'TradingSignal':
        return cls(
            direction=SignalDirection.LONG,
            strength=strength,
            signals=signals or []
        )
    
    @classmethod
    def short(cls, strength: float = 1.0, signals: List[Signal] = None) -> 'TradingSignal':
        return cls(
            direction=SignalDirection.SHORT,
            strength=strength,
            signals=signals or []
        )
    
    @classmethod
    def neutral(cls, signals: List[Signal] = None) -> 'TradingSignal':
        return cls(
            direction=SignalDirection.NEUTRAL,
            strength=0,
            signals=signals or []
        )
    
    @property
    def is_actionable(self) -> bool:
        """信号是否可执行"""
        return self.strength >= 0.5  # 阈值可调
    
    def to_dict(self) -> dict:
        return {
            'direction': self.direction.value,
            'direction_name': self.direction.name,
            'strength': self.strength,
            'signal_count': len(self.signals),
            'signal_names': [s.name for s in self.signals],
            'timestamp': self.timestamp.isoformat() if self.timestamp else None,
            'metadata': self.metadata,
        }


@dataclass
class SignalSeries:
    """
    信号序列（DataFrame 形式）
    
    用于批量处理和回测
    """
    signals: pd.DataFrame  # index=时间, columns=信号名, values=信号值
    directions: Optional[pd.DataFrame] = None  # 方向信号
    weights: Optional[pd.DataFrame] = None  # 权重
    
    @classmethod
    def from_signals(cls, signal_dict: Dict[str, pd.Series]) -> 'SignalSeries':
        """从信号字典创建"""
        return cls(
            signals=pd.DataFrame(signal_dict)
        )
    
    def get_latest(self) -> Dict[str, float]:
        """获取最新信号值"""
        if self.signals.empty:
            return {}
        return self.signals.iloc[-1].to_dict()
    
    def get_direction_series(self, threshold_policy=None) -> pd.DataFrame:
        """获取方向序列"""
        if self.directions is not None:
            return self.directions
        
        if threshold_policy is None:
            # 默认：值>0 做多，值<0 做空
            return (self.signals > 0).astype(int) - (self.signals < 0).astype(int)
        
        return threshold_policy.apply(self.signals)
