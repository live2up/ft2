# signal/combiner.py - 信号融合器
"""
多信号融合器 - 完整使用指南

本模块提供6种信号融合策略，用于将多个技术指标信号合并为综合交易信号。

================================================================================
一、快速选择指南
================================================================================

场景1: 你有多个指标，不知道权重 → 使用 VotingCombiner（投票）
场景2: 你有多个指标，知道权重 → 使用 WeightedCombiner（固定权重）
场景3: 你有多个指标，想自动学习权重 → 使用 AdaptiveCombiner（自适应）
场景4: 你有多个指标，想简单平均 → 使用 EqualWeightCombiner（等权）
场景5: 你有多个指标，需要归一化处理 → 使用 ScoringCombiner（打分）
场景6: 你有多个指标，有时序关系 → 使用 SequenceCombiner（时序）

================================================================================
二、核心接口说明
================================================================================

所有融合器都继承自 SignalCombiner，提供两个核心方法：

1. combine(signals: List[Signal]) -> TradingSignal
   用途：对单个时间点的多个信号进行融合
   输入：信号对象列表
   输出：综合交易信号（包含方向、强度等）
   
2. combine_series(signal_series: SignalSeries) -> pd.Series
   用途：对整个时间序列的信号进行融合
   输入：信号序列（DataFrame格式）
   输出：综合信号序列（1=做多，-1=做空，0=中立）

================================================================================
三、各融合器详细用法
================================================================================

1. VotingCombiner（投票融合器）
   - 规则：少数服从多数
   - 参数：无
   - 示例：
     combiner = VotingCombiner()
     result = combiner.combine([signal1, signal2, signal3])
     
2. ScoringCombiner（打分融合器）
   - 规则：归一化后加权求和（使用tanh归一化到[-1,1]）
   - 参数：weights - 权重字典（可选，默认等权）
   - 示例：
     combiner = ScoringCombiner(weights={'RSI': 0.6, 'MACD': 0.4})
     result = combiner.combine([rsi_signal, macd_signal])
     
3. WeightedCombiner（加权融合器）
   - 规则：固定权重，直接加权（不归一化）
   - 参数：weights - 权重字典（必填）
   - 示例：
     combiner = WeightedCombiner(weights={'RSI': 0.6, 'MACD': 0.4})
     result = combiner.combine([rsi_signal, macd_signal])
     
4. AdaptiveCombiner（自适应融合器）
   - 规则：根据历史IC（信息系数）自动调整权重
   - 参数：lookback - 回溯期（默认60）
   - 方法：fit(signal_series, returns) - 先拟合权重
   - 示例：
     combiner = AdaptiveCombiner(lookback=60)
     combiner.fit(signal_series, returns)  # 先拟合
     result = combiner.combine([rsi_signal, macd_signal])
     
5. EqualWeightCombiner（等权融合器）
   - 规则：所有信号等权重平均
   - 参数：无
   - 示例：
     combiner = EqualWeightCombiner()
     result = combiner.combine([signal1, signal2, signal3])
     
6. SequenceCombiner（时序信号组合器）
   - 规则：支持并行、时序、条件过滤三种模式
   - 参数：rules - 规则列表，window - 时序窗口
   - 示例：见下方详细说明

================================================================================
四、SequenceCombiner 详细说明
================================================================================

1. 并行模式（window=0）
   - 所有信号同时成立才触发
   - 示例：
     combiner = SequenceCombiner(rules=[
         {'name': 'MA', 'generator': MASignal(5, 20), 'condition': 'state'},
         {'name': 'KDJ', 'generator': KDJSignal(), 'condition': 'state'}
     ], window=0)
     
2. 时序模式（window=N）
   - 第一个信号触发后N日内，其他信号成立才有效
   - 示例（MA金叉后5日内KDJ金叉）：
     combiner = SequenceCombiner(rules=[
         {'name': 'MA', 'generator': MASignal(5, 20), 'condition': 'cross'},
         {'name': 'KDJ', 'generator': KDJSignal(), 'condition': 'cross'}
     ], window=5)
     
3. 条件过滤模式（window=None）
   - 第一个信号持续成立期间，其他信号成立才有效
   - 示例（ADX<20期间RSI金叉有效）：
     combiner = SequenceCombiner(rules=[
         {'name': 'ADX', 'generator': DMISignal(), 'condition': 'state', 
          'threshold': 20, 'direction': 'below'},
         {'name': 'RSI', 'generator': RSISignal(), 'condition': 'cross'}
     ], window=None)

规则参数说明：
   - name: 信号名称（字符串）
   - generator: SignalGenerator实例
   - condition: 'state'（持续状态）或 'cross'（穿越触发）
   - threshold: 阈值（仅用于state条件）
   - direction: 'above'或'below'（仅用于state条件）

================================================================================
五、数据格式说明
================================================================================

SignalSeries 格式：
   - 属性：signals（DataFrame）
   - index: 时间
   - columns: 信号名称
   - values: 信号值（>0做多，<0做空，=0中立）

K线数据格式：
   - DataFrame类型
   - 通常包含：open, high, low, close, volume等列

================================================================================
六、返回值说明
================================================================================

TradingSignal 属性：
   - direction: 信号方向（LONG/SHORT/NEUTRAL）
   - strength: 信号强度（0~1）
   - signals: 原始信号列表

pd.Series 返回值：
   - index: 时间
   - values: 1（做多），-1（做空），0（中立）

================================================================================
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np

from .base import Signal, SignalDirection, TradingSignal, SignalSeries


class SignalCombiner(ABC):
    """
    信号融合器基类
    
    输入：多个信号
    输出：综合交易信号
    """
    
    def __init__(self, name: str = None):
        self.name = name or self.__class__.__name__
    
    @abstractmethod
    def combine(self, signals: List[Signal]) -> TradingSignal:
        """
        融合多个信号
        
        Args:
            signals: 信号列表
            
        Returns:
            TradingSignal: 综合交易信号
        """
        pass
    
    def combine_series(self, signal_series: SignalSeries) -> pd.Series:
        """
        融合信号序列
        
        Args:
            signal_series: 信号序列
            
        Returns:
            pd.Series: 综合信号序列
        """
        pass


class VotingCombiner(SignalCombiner):
    """
    投票融合器
    
    规则：
    - 多头信号 > 空头信号 → 做多
    - 空头信号 > 多头信号 → 做空
    - 数量相等 → 中立
    """
    
    def __init__(self):
        super().__init__("VotingCombiner")
    
    def combine(self, signals: List[Signal]) -> TradingSignal:
        long_count = sum(1 for s in signals if s.is_long)
        short_count = sum(1 for s in signals if s.is_short)
        neutral_count = sum(1 for s in signals if s.is_neutral)
        
        if long_count > short_count:
            return TradingSignal.long(
                strength=long_count / len(signals),
                signals=signals
            )
        elif short_count > long_count:
            return TradingSignal.short(
                strength=short_count / len(signals),
                signals=signals
            )
        else:
            return TradingSignal.neutral(signals=signals)
    
    def combine_series(self, signal_series: SignalSeries) -> pd.Series:
        """
        投票融合信号序列
        
        Args:
            signal_series: 信号序列，index=时间, columns=信号名
            
        Returns:
            pd.Series: 综合信号 -1/0/1
        """
        signals_df = signal_series.signals
        
        # 方向化：>0 = 1, <0 = -1, =0 = 0
        directions = np.sign(signals_df)
        
        # 投票：求和
        votes = directions.sum(axis=1)
        
        # 最终方向：>0 = 1, <0 = -1, =0 = 0
        result = np.sign(votes)
        
        return pd.Series(result, index=signals_df.index)


class ScoringCombiner(SignalCombiner):
    """
    打分融合器
    
    规则：
    1. 归一化所有信号到 [-1, 1]
    2. 加权求和
    3. >0 做多，<0 做空
    """
    
    def __init__(self, weights: Dict[str, float] = None):
        super().__init__("ScoringCombiner")
        self.weights = weights or {}
    
    def combine(self, signals: List[Signal]) -> TradingSignal:
        if not signals:
            return TradingSignal.neutral()
        
        total_score = 0
        total_weight = 0
        
        for signal in signals:
            weight = self.weights.get(signal.name, 1.0)
            
            # 归一化到 [-1, 1]
            normalized = np.tanh(signal.value)
            
            total_score += normalized * weight
            total_weight += weight
        
        if total_weight == 0:
            return TradingSignal.neutral(signals=signals)
        
        final_score = total_score / total_weight
        
        if final_score > 0:
            return TradingSignal.long(
                strength=abs(final_score),
                signals=signals
            )
        elif final_score < 0:
            return TradingSignal.short(
                strength=abs(final_score),
                signals=signals
            )
        else:
            return TradingSignal.neutral(signals=signals)
    
    def combine_series(self, signal_series: SignalSeries) -> pd.Series:
        """
        打分融合信号序列
        
        Args:
            signal_series: 信号序列
            
        Returns:
            pd.Series: 综合信号序列 [-1, 1]
        """
        signals_df = signal_series.signals
        
        # 归一化
        normalized = signals_df.apply(lambda x: np.tanh(x))
        
        # 加权
        if self.weights:
            for col in normalized.columns:
                if col in self.weights:
                    normalized[col] *= self.weights[col]
        
        # 求和归一化
        result = normalized.sum(axis=1)
        
        # 归一化到 [-1, 1]
        max_abs = result.abs().max()
        if max_abs > 0:
            result = result / max_abs
        
        return result


class WeightedCombiner(SignalCombiner):
    """
    加权融合器（固定权重）
    
    与 ScoringCombiner 类似，但权重必须指定
    """
    
    def __init__(self, weights: Dict[str, float]):
        super().__init__("WeightedCombiner")
        
        # 归一化权重
        total = sum(weights.values())
        self.weights = {k: v / total for k, v in weights.items()}
    
    def combine(self, signals: List[Signal]) -> TradingSignal:
        if not signals:
            return TradingSignal.neutral()
        
        total_score = 0
        
        for signal in signals:
            weight = self.weights.get(signal.name, 0)
            total_score += signal.value * weight
        
        if total_score > 0:
            return TradingSignal.long(
                strength=abs(total_score),
                signals=signals
            )
        elif total_score < 0:
            return TradingSignal.short(
                strength=abs(total_score),
                signals=signals
            )
        else:
            return TradingSignal.neutral(signals=signals)
    
    def combine_series(self, signal_series: SignalSeries) -> pd.Series:
        signals_df = signal_series.signals.copy()
        
        # 加权
        for col in signals_df.columns:
            if col in self.weights:
                signals_df[col] *= self.weights[col]
        
        return signals_df.sum(axis=1)


class AdaptiveCombiner(SignalCombiner):
    """
    自适应融合器
    
    根据历史 IC 自动调整权重
    """
    
    def __init__(self, lookback: int = 60):
        super().__init__("AdaptiveCombiner")
        self.lookback = lookback
        self.weights = {}
    
    def fit(self, signal_series: SignalSeries, returns: pd.Series):
        """
        根据历史数据拟合权重
        
        Args:
            signal_series: 信号序列
            returns: 未来收益率序列
        """
        signals_df = signal_series.signals
        
        # 计算每个信号的 IC
        for col in signals_df.columns:
            signal = signals_df[col].shift(1)  # 信号发生在收益之前
            ic = signal.corr(returns)
            
            if not np.isnan(ic):
                # IC 作为权重
                self.weights[col] = abs(ic)
        
        # 归一化
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}
        else:
            self.weights = {k: 1 / len(signals_df.columns) for k in signals_df.columns}
    
    def combine(self, signals: List[Signal]) -> TradingSignal:
        if not signals:
            return TradingSignal.neutral()
        
        total_score = 0
        
        for signal in signals:
            weight = self.weights.get(signal.name, 0)
            total_score += signal.value * weight
        
        if total_score > 0:
            return TradingSignal.long(strength=abs(total_score), signals=signals)
        elif total_score < 0:
            return TradingSignal.short(strength=abs(total_score), signals=signals)
        else:
            return TradingSignal.neutral(signals=signals)
    
    def combine_series(self, signal_series: SignalSeries) -> pd.Series:
        signals_df = signal_series.signals.copy()
        
        # 加权
        for col in signals_df.columns:
            if col in self.weights:
                signals_df[col] *= self.weights[col]
        
        return signals_df.sum(axis=1)


class EqualWeightCombiner(SignalCombiner):
    """
    等权融合器
    
    所有信号权重相等
    """
    
    def __init__(self):
        super().__init__("EqualWeightCombiner")
    
    def combine(self, signals: List[Signal]) -> TradingSignal:
        if not signals:
            return TradingSignal.neutral()
        
        # 计算平均信号值
        avg_value = np.mean([s.value for s in signals])
        
        # 判断方向
        if avg_value > 0:
            return TradingSignal.long(strength=abs(avg_value), signals=signals)
        elif avg_value < 0:
            return TradingSignal.short(strength=abs(avg_value), signals=signals)
        else:
            return TradingSignal.neutral(signals=signals)
    
    def combine_series(self, signal_series: SignalSeries) -> pd.Series:
        return signal_series.signals.mean(axis=1)


# =============================================================================
# 时序信号组合器
# =============================================================================

class SequenceCombiner(SignalCombiner):
    """
    时序信号组合器
    
    统一的时序逻辑抽象，支持：
    1. 并行组合（window=0）：所有信号同时成立
    2. 时序组合（window=N）：第一个信号成立后N日内，其他信号成立
    3. 条件过滤（window=None）：第一个信号持续成立期间，其他信号成立
    
    架构设计：
        单指标择时 → 直接用 SignalGenerator（保持简单）
        多指标组合 → 统一走 SequenceCombiner
    
    Example:
        # 多指标并行
        combiner = SequenceCombiner(rules=[
            {'name': 'MA5_20', 'generator': MASignal(5, 20), 'condition': 'state'},
            {'name': 'KDJ', 'generator': KDJSignal(), 'condition': 'state'}
        ], window=0)
        
        # 时序组合（MA金叉后5日内KDJ金叉）
        combiner = SequenceCombiner(rules=[
            {'name': 'MA5_20', 'generator': MASignal(5, 20), 'condition': 'cross'},
            {'name': 'KDJ', 'generator': KDJSignal(), 'condition': 'cross'}
        ], window=5)
        
        # 条件过滤（ADX<20期间RSI金叉有效）
        combiner = SequenceCombiner(rules=[
            {'name': 'ADX', 'generator': DMISignal(), 'condition': 'state', 'threshold': 20, 'direction': 'below'},
            {'name': 'RSI', 'generator': RSISignal(), 'condition': 'cross'}
        ], window=None)
    """
    
    def __init__(
        self,
        rules: List[Dict],
        window: int = 0,
        name: str = None
    ):
        """
        Args:
            rules: 信号规则列表，每项包含：
                - name: 信号名称
                - generator: SignalGenerator 实例
                - condition: 条件类型 ('state'=持续状态, 'cross'=穿越触发)
                - threshold: 阈值（可选，用于 state 条件）
                - direction: 方向（'above'/'below'，用于 state 条件）
            window: 时序窗口
                - 0: 并行组合（所有信号同时成立）
                - N: 时序组合（第一个信号成立后N日内，其他信号成立）
                - None: 条件过滤（第一个信号持续成立期间）
            name: 组合器名称
        """
        super().__init__(name or f"Sequence_{window if window is not None else 'filter'}")
        self.rules = rules
        self.window = window
        self.signal_cache = {}
    
    def combine(self, signals: List[Signal]) -> TradingSignal:
        """
        融合多个信号（单点判断）
        
        Note: 此方法主要用于兼容接口，推荐使用 combine_series
        """
        if not signals:
            return TradingSignal.neutral()
        
        if self.window == 0:
            # 并行组合：所有信号方向一致
            directions = [s.direction for s in signals]
            if all(d == SignalDirection.LONG for d in directions):
                return TradingSignal.long(signals=signals)
            elif all(d == SignalDirection.SHORT for d in directions):
                return TradingSignal.short(signals=signals)
            else:
                return TradingSignal.neutral(signals=signals)
        else:
            # 时序逻辑需要历史状态，此方法不适用
            raise ValueError("时序逻辑需要使用 combine_series 方法")
    
    def combine_series(
        self,
        signal_series: SignalSeries,
        generators: Dict[str, 'SignalGenerator'] = None,
        data: pd.DataFrame = None
    ) -> pd.Series:
        """
        融合信号序列
        
        Args:
            signal_series: 信号序列（包含预计算的信号）
            generators: 信号生成器字典（如果 signal_series 为空，需要此参数）
            data: K线数据（用于生成信号）
            
        Returns:
            pd.Series: 综合信号序列（1=做多，-1=做空，0=中立）
        """
        # 如果没有预计算信号，则生成
        if signal_series is None or signal_series.signals.empty:
            if generators is None or data is None:
                raise ValueError("需要提供 signal_series 或 (generators + data)")
            signals_dict = {}
            for rule in self.rules:
                gen = rule['generator']
                signals_dict[rule['name']] = gen.generate(data)
            signal_series = SignalSeries.from_signals(signals_dict)
        
        signals_df = signal_series.signals
        
        if self.window == 0:
            return self._combine_parallel(signals_df)
        elif self.window is None:
            return self._combine_filter(signals_df)
        else:
            return self._combine_sequence(signals_df)
    
    def _combine_parallel(self, signals_df: pd.DataFrame) -> pd.Series:
        """
        并行组合（window=0）
        
        所有信号同时成立才触发
        """
        # 方向化：>0 = 1, <0 = -1
        directions = np.sign(signals_df)
        
        # 所有信号方向一致
        product = directions.product(axis=1)
        
        # 1=全多，-1=全空，其他=不一致
        result = product.where(product.abs() == 1, 0)
        
        return result
    
    def _combine_filter(self, signals_df: pd.DataFrame) -> pd.Series:
        """
        条件过滤（window=None）
        
        第一个信号持续成立期间，其他信号成立才有效
        """
        if len(self.rules) < 2:
            return np.sign(signals_df.iloc[:, 0])
        
        # 第一个信号作为过滤器
        filter_signal = signals_df.iloc[:, 0]
        filter_rule = self.rules[0]
        
        # 计算过滤器状态
        if 'threshold' in filter_rule:
            threshold = filter_rule['threshold']
            direction = filter_rule.get('direction', 'above')
            if direction == 'above':
                filter_state = (filter_signal > threshold).astype(float)
            else:
                filter_state = (filter_signal < threshold).astype(float)
        else:
            filter_state = (filter_signal > 0).astype(float)
        
        # 其他信号的组合（投票）
        other_signals = signals_df.iloc[:, 1:]
        other_result = np.sign(other_signals.mean(axis=1))
        
        # 过滤
        result = other_result * filter_state
        
        return result
    
    def _combine_sequence(self, signals_df: pd.DataFrame) -> pd.Series:
        """
        时序组合（window=N）
        
        第一个信号成立后N日内，其他信号成立才有效
        """
        if len(self.rules) < 2:
            return np.sign(signals_df.iloc[:, 0])
        
        # 第一个信号作为触发器
        trigger_signal = signals_df.iloc[:, 0]
        trigger_cross = self._detect_cross(trigger_signal)
        
        # 其他信号
        other_signals = signals_df.iloc[:, 1:]
        
        result = pd.Series(0, index=signals_df.index)
        
        # 滑动窗口判断
        for i in range(len(signals_df)):
            # 检查是否在触发窗口内
            window_start = max(0, i - self.window)
            has_triggered = trigger_cross.iloc[window_start:i+1].any()
            
            if has_triggered:
                # 检查其他信号是否成立
                other_state = other_signals.iloc[i]
                if (other_state > 0).all():
                    result.iloc[i] = 1
                elif (other_state < 0).all():
                    result.iloc[i] = -1
        
        return result
    
    def _detect_cross(self, signal: pd.Series) -> pd.Series:
        """
        检测穿越信号（从负到正或从正到负）
        """
        sign_change = np.sign(signal) * np.sign(signal.shift(1))
        return (sign_change < 0).astype(int)
    
    def generate_signals(
        self,
        data: pd.DataFrame
    ) -> pd.DataFrame:
        """
        生成完整的信号序列（便捷方法）
        
        Args:
            data: K线数据
            
        Returns:
            pd.DataFrame: 包含所有信号和综合结果的DataFrame
        """
        # 生成所有基础信号
        signals_dict = {}
        for rule in self.rules:
            gen = rule['generator']
            signals_dict[rule['name']] = gen.generate(data)
        
        signals_df = pd.DataFrame(signals_dict)
        signal_series = SignalSeries.from_signals(signals_dict)
        
        # 组合
        combined = self.combine_series(signal_series)
        
        # 返回完整结果
        result_df = signals_df.copy()
        result_df['combined'] = combined
        
        return result_df
