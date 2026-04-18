# signals 模块 - 择时信号层
"""
信号层：指标 → 信号 → 融合 → 择时

架构：
  原始数据 (K线)
       ↓
  SignalGenerator (指标计算)
       ↓
  Signal (连续值)
       ↓
  SignalCombiner (多信号融合)
       ↓
  CombinedSignal (综合信号)
       ↓
  ThresholdPolicy (阈值化)
       ↓
  TradingSignal (买入/持有/卖出)

目录：
  signals/
  ├── __init__.py          # 模块入口
  ├── base.py              # Signal 基类和枚举
  ├── generator.py         # 信号生成器（8种内置 + 组合 + 函数式）
  ├── combiner.py          # 信号融合器（投票/打分/加权/自适应）
  ├── registry.py          # 信号注册器和模板
  ├── threshold.py         # 阈值策略
  ├── timeframe.py         # 多周期数据处理（对齐、转换、特征）
  └── examples.py          # 使用示例
"""

from .base import (
    Signal,           # 信号基类
    SignalType,       # 信号类型枚举
    SignalDirection,  # 信号方向枚举
    TradingSignal,    # 交易信号
    SignalSeries,     # 信号序列
)

from .generator import (
    SignalGenerator,           # 生成器基类
    MASignal,                  # 均线交叉信号
    MACDSignal,                # MACD 信号
    RSISignal,                 # RSI 信号
    KDJSignal,                 # KDJ 信号
    BOLLSignal,                # 布林带信号
    VOLSignal,                 # 量能信号
    RSRSMSignal,               # RSRS 信号
    CompositeSignal,           # 组合信号
    FunctionSignal,            # 函数式信号
    CCISignal,                 # CCI 信号
    WRSignal,                  # 威廉指标信号
    ROCSignal,                 # ROC 信号
    OBVSignal,                 # OBV 信号
    DMISignal,                 # DMI 信号
    IndicatorSignal,           # 通用指标信号（新增）
    create_signal,             # 工厂函数（新增）
)

from .combiner import (
    SignalCombiner,     # 融合器基类
    VotingCombiner,     # 投票融合
    ScoringCombiner,    # 打分融合
    WeightedCombiner,   # 加权融合
    EqualWeightCombiner,# 等权融合
    AdaptiveCombiner,   # 自适应融合
)

from .registry import (
    SignalRegistry,
    SIGNAL_TEMPLATES,
    create_signal_from_template,
    create_signal_set,
)

from .threshold import (
    ThresholdPolicy,        # 阈值基类
    SimpleThreshold,        # 简单阈值
    PercentileThreshold,   # 分位数阈值
    ZScoreThreshold,       # Z-Score 阈值
    MovingThreshold,       # 移动阈值
    DualThreshold,         # 双阈值
    THRESHOLD_PRESETS,
    get_threshold_preset,
)

from .timeframe import (
    MultiTimeframeAligner,  # 多周期数据对齐器
    FrequencyConverter,     # 频率转换器
)

__all__ = [
    # 基类
    'Signal',
    'SignalType',
    'SignalDirection',
    'SignalSeries',
    'TradingSignal',
    # 生成器
    'SignalGenerator',
    'MASignal',
    'MACDSignal',
    'RSISignal',
    'KDJSignal',
    'BOLLSignal',
    'VOLSignal',
    'RSRSMSignal',
    'CompositeSignal',
    'FunctionSignal',
    'CCISignal',
    'WRSignal',
    'ROCSignal',
    'OBVSignal',
    'DMISignal',
    'IndicatorSignal',            # 通用指标信号
    'create_signal',              # 工厂函数
    # 融合器
    'SignalCombiner',
    'VotingCombiner',
    'ScoringCombiner',
    'WeightedCombiner',
    'EqualWeightCombiner',
    'AdaptiveCombiner',
    # 注册器
    'SignalRegistry',
    'SIGNAL_TEMPLATES',
    'create_signal_from_template',
    'create_signal_set',
    # 阈值策略
    'ThresholdPolicy',
    'SimpleThreshold',
    'PercentileThreshold',
    'ZScoreThreshold',
    'MovingThreshold',
    'DualThreshold',
    'THRESHOLD_PRESETS',
    'get_threshold_preset',
    # 多周期处理
    'MultiTimeframeAligner',
    'FrequencyConverter',
]
