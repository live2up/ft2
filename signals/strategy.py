# signals/strategy.py - 策略适配器
"""
信号策略适配器 - 将信号层接入回测引擎

功能：
1. 将信号生成器转换为标准策略（供 core/engine 使用）
2. 支持单信号、多信号融合、阈值处理
3. 自动将信号转换为订单（不修改 core 模块）

架构：
  SignalGenerator → SignalStrategy → on_bar() → AccountManager.order_xxx()

使用示例：
    from signals import MASignal, SimpleThreshold
    from signals.strategy import SignalStrategy
    
    strategy = SignalStrategy(
        generator=MASignal(5, 20),
        threshold=SimpleThreshold(0, 0),
        position_percent=1.0
    )
    
    engine.run(strategy, start_time, end_time)
"""

from typing import Dict, List, Optional, Callable
import pandas as pd
import numpy as np

from .base import Signal, SignalDirection, SignalSeries
from .generator import SignalGenerator
from .combiner import SignalCombiner
from .threshold import ThresholdPolicy


class SignalStrategy:
    """
    信号驱动策略 - 将 signals 模块接入 core 回测引擎
    
    工作流程：
    1. on_bar() 被引擎调用
    2. 从 context 获取数据
    3. 生成信号（单信号 或 融合信号）
    4. 阈值化处理（连续值 → 方向）
    5. 执行交易（调用 account.order_xxx）
    """
    
    def __init__(
        self,
        generator: SignalGenerator = None,
        combiner: SignalCombiner = None,
        threshold: ThresholdPolicy = None,
        generators: List[SignalGenerator] = None,
        position_percent: float = 1.0,
        symbol: str = None,
        signal_to_position: Callable = None
    ):
        """
        Args:
            generator: 单信号生成器（优先使用）
            combiner: 信号融合器（多信号时必需）
            threshold: 阈值策略（None 时使用信号默认方向）
            generators: 信号生成器列表（与 combiner 配合使用）
            position_percent: 仓位比例（0-1，默认满仓）
            symbol: 交易标的（None 时从 context 自动获取）
            signal_to_position: 自定义信号转仓位函数
                               签名: func(signal_value) -> int (1/-1/0)
        
        Example:
            # 单信号策略
            strategy = SignalStrategy(
                generator=MASignal(5, 20),
                threshold=SimpleThreshold(0, 0)
            )
            
            # 多信号融合策略
            strategy = SignalStrategy(
                combiner=VotingCombiner(),
                generators=[MASignal(5,20), MACDSignal(), RSISignal()],
                position_percent=0.8
            )
        """
        self.generator = generator
        self.combiner = combiner
        self.threshold = threshold
        self.generators = generators or []
        self.position_percent = position_percent
        self.symbol = symbol
        self.signal_to_position = signal_to_position
        
        self._current_position = 0
        self._last_signal_value = None
        
    def on_bar(self, context, bars):
        """
        标准策略接口，供 core/engine.run() 调用
        
        Args:
            context: 数据上下文（core.storage.context）
            bars: 当前K线数据列表
        """
        self._bars = bars
        
        symbol = self._get_symbol(context)
        if symbol is None:
            return
        
        data = self._get_data(context, symbol)
        if data is None or len(data) < 2:
            return
        
        signal_value, direction = self._generate_signal(data)
        if signal_value is None:
            return
        
        self._last_signal_value = signal_value
        
        if self.signal_to_position:
            position = self.signal_to_position(signal_value)
        elif self.threshold:
            pos_direction = self.threshold.apply(signal_value)
            position = pos_direction.value
        else:
            position = direction.value
        
        self._execute(context, symbol, position)
    
    def _generate_signal(self, data):
        """
        生成信号（支持单信号 或 多信号融合）
        
        Returns:
            tuple: (signal_value, direction)
        """
        if self.generator:
            signal = self.generator.generate_latest(data)
            return signal.value, signal.direction
        
        elif self.combiner and self.generators:
            signals = [gen.generate_latest(data) for gen in self.generators]
            combined = self.combiner.combine(signals)
            return combined.strength * combined.direction.value, combined.direction
        
        return None, SignalDirection.NEUTRAL
    
    def _execute(self, context, symbol, target_position):
        """
        执行交易
        
        Args:
            context: 数据上下文
            symbol: 标的代码
            target_position: 目标仓位 (1=多头, -1=空头, 0=空仓)
        """
        if target_position == self._current_position:
            return
        
        account = context.account if hasattr(context, 'account') else None
        if account is None:
            return
        
        if target_position == 1:
            if self._current_position == 0:
                account.order_percent(symbol, self.position_percent, 1)
            elif self._current_position == -1:
                account.order_percent(symbol, self.position_percent * 2, 1)
            self._current_position = 1
            
        elif target_position == -1:
            if self._current_position == 0:
                account.order_percent(symbol, self.position_percent, 2)
            elif self._current_position == 1:
                account.order_percent(symbol, self.position_percent * 2, 2)
            self._current_position = -1
            
        elif target_position == 0:
            if self._current_position == 1:
                account.order_percent(symbol, 1.0, 2)
            elif self._current_position == -1:
                account.order_percent(symbol, 1.0, 1)
            self._current_position = 0
    
    def _get_symbol(self, context):
        """获取交易标的"""
        if self.symbol:
            return self.symbol
        
        symbols = context.symbols
        if isinstance(symbols, set) and len(symbols) > 0:
            return symbols.pop()
        
        return None
    
    def _get_data(self, context, symbol):
        """
        获取DataFrame格式的数据
        
        Returns:
            pd.DataFrame 或 None
        """
        for bar in self._bars:
            if bar.get('symbol') == symbol:
                df = pd.DataFrame([bar])
                df.set_index('eob', inplace=True)
                return df
        
        freq = '1d'
        if self._bars:
            freq = self._bars[0].get('frequency', '1d')
        
        try:
            raw_data = context.data(symbol=symbol, frequency=freq, count=100)
            if isinstance(raw_data, pd.DataFrame):
                df = raw_data.copy()
                if 'eob' in df.columns:
                    df.set_index('eob', inplace=True)
                return df
            elif isinstance(raw_data, list):
                df = pd.DataFrame(raw_data)
                if 'eob' in df.columns:
                    df.set_index('eob', inplace=True)
                return df
        except:
            pass
        
        return None
    
    def get_signal_value(self):
        """获取最新信号值（供外部查询）"""
        return self._last_signal_value


class MultiSymbolStrategy:
    """
    多标的信号策略
    
    为不同标的配置不同的信号生成器
    """
    
    def __init__(self, symbol_configs: Dict[str, Dict]):
        """
        Args:
            symbol_configs: 标的配置字典
                {
                    'symbol1': {
                        'generator': MASignal(5, 20),
                        'threshold': SimpleThreshold(0, 0),
                        'position_percent': 0.5
                    },
                    'symbol2': {...}
                }
        """
        self.strategies = {}
        
        for symbol, config in symbol_configs.items():
            self.strategies[symbol] = SignalStrategy(
                generator=config.get('generator'),
                combiner=config.get('combiner'),
                threshold=config.get('threshold'),
                generators=config.get('generators'),
                position_percent=config.get('position_percent', 1.0),
                symbol=symbol
            )
    
    def on_bar(self, context, bars):
        """为每个标的执行策略"""
        for symbol, strategy in self.strategies.items():
            symbol_bars = [b for b in bars if b.get('symbol') == symbol]
            if symbol_bars:
                strategy._bars = symbol_bars
                strategy.on_bar(context, bars)
