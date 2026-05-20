"""
GP 多频率适应度评估器——Phase C 核心组件

设计思路：
--------
1. GP 进化时，每个候选因子表达式需要在多个频率下评估
2. fitness = max(Sharpe_ME, Sharpe_W, Sharpe_5D)，天然输出最优频率
3. 复用 V2 Pipeline 进行回测评估（替代 V1 的纯 IC 评判）
4. 适合集成到任何 GP 框架（DEAP、gplearn、PySR）的适应度函数中

使用方式：
--------
>>> evaluator = GPMultiFreqEvaluator(returns, cost_rate=0.001)
>>> # 在 GP 适应度函数中调用：
>>> fitness, best_freq = evaluator.evaluate(factor_values)
>>> # fitness = max(Sharpe across all frequencies)
>>> # best_freq = 'W'  (Sharpe 最高的频率)

或直接用于任意因子表达式的评估：
>>> sharpe_me = evaluator.evaluate_single(factor_values, 'ME')
>>> sharpe_w  = evaluator.evaluate_single(factor_values, 'W')

依赖说明：
--------
依赖 .scheduler (FixedScheduler/IntervalScheduler)、.allocator (TopNEqualWeight)、
.pipeline (FactorPipeline)，不依赖 GP 框架本身。
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import numpy as np
import pandas as pd

from .scheduler import FixedScheduler, IntervalScheduler, RebalanceScheduler
from .allocator import TopNEqualWeight, WeightAllocator
from .pipeline import FactorPipeline, BacktestResult


@dataclass
class GPEvaluationResult:
    """GP 适应度评估结果
    
    包含各频率的绩效和最优频率信息。
    """
    fitness: float = 0.0              # 多频率最优 Sharpe
    best_frequency: str = 'ME'        # 最优频率
    best_sharpe: float = 0.0          # 最优频率的 Sharpe
    frequency_sharpes: Dict[str, float] = None  # 各频率 Sharpe

    def __post_init__(self):
        if self.frequency_sharpes is None:
            self.frequency_sharpes = {}

    def to_dict(self) -> Dict:
        return {
            'fitness': round(self.fitness, 3),
            'best_frequency': self.best_frequency,
            'best_sharpe': round(self.best_sharpe, 3),
            'frequency_sharpes': {
                k: round(v, 3) for k, v in self.frequency_sharpes.items()
            },
        }


class GPMultiFreqEvaluator:
    """GP 多频率适应度评估器
    
    为 GP 符号回归提供多频率适应度评估。
    适应度 = max(Sharpe_ME, Sharpe_W, Sharpe_5D, ...)
    天然输出"因子表达式 + 最优频率"的组合。
    
    典型用法（在 GP fitness 函数中）：
    
    ```python
    evaluator = GPMultiFreqEvaluator(returns, top_n=5)
    
    def gp_fitness(individual):
        # individual 是 GP 个体（表达式树）
        expr = str(individual)  # 转为表达式字符串
        factor_values = evaluate_expression(expr, data)
        result = evaluator.evaluate(factor_values)
        return (result.fitness,)  # DEAP 需要返回元组
    ```
    """

    def __init__(self,
                 returns: pd.DataFrame,
                 frequencies: List[str] = None,
                 top_n: int = 5,
                 cost_rate: float = 0.001,
                 rf_annual: float = 0.0):
        """初始化多频率评估器
        
        Args:
            returns: 各标的日收益率 DataFrame
            frequencies: 评估频率列表，默认 ['ME', 'W', '5D']
            top_n: 持仓数量
            cost_rate: 交易成本率
            rf_annual: 无风险利率
        """
        self.returns = returns
        self.frequencies = frequencies or ['ME', 'W', '5D']
        self.top_n = top_n
        self.cost_rate = cost_rate
        self.rf_annual = rf_annual

        # 确保 index 是 DatetimeIndex
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns = self.returns.copy()
            self.returns.index = pd.to_datetime(self.returns.index)

        # 预构建调度器和分配器
        self._schedulers = self._build_schedulers()
        self._allocator = TopNEqualWeight(top_n)

    def evaluate(self,
                 factor_values: pd.DataFrame) -> GPEvaluationResult:
        """多频率评估：计算各频率 Sharpe，返回最优值
        
        Args:
            factor_values: 因子值 DataFrame (index=日期, columns=标的)
                          GP 表达式 safe_eval 的输出
            
        Returns:
            GPEvaluationResult: 包含 fitness 和最优频率
        """
        if factor_values is None or factor_values.shape[0] < 60:
            return GPEvaluationResult(fitness=-999.0)

        frequency_sharpes = {}
        best_sharpe = -np.inf
        best_freq = self.frequencies[0]

        for freq in self.frequencies:
            scheduler = self._schedulers.get(freq)
            if scheduler is None:
                continue

            sharpe = self.evaluate_single(factor_values, freq)
            frequency_sharpes[freq] = sharpe

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_freq = freq

        fitness = best_sharpe

        return GPEvaluationResult(
            fitness=fitness,
            best_frequency=best_freq,
            best_sharpe=best_sharpe,
            frequency_sharpes=frequency_sharpes,
        )

    def evaluate_single(self,
                        factor_values: pd.DataFrame,
                        frequency: str) -> float:
        """单频率评估：返回该频率下的 Sharpe
        
        Args:
            factor_values: 因子值 DataFrame
            frequency: 频率标识 ('ME', 'W', '5D', etc.)
            
        Returns:
            float: Sharpe 比率（失败返回 -999.0）
        """
        scheduler = self._schedulers.get(frequency)
        if scheduler is None:
            return -999.0

        try:
            pipeline = FactorPipeline(
                returns=self.returns,
                scheduler=scheduler,
                allocator=self._allocator,
                cost_rate=self.cost_rate,
                rf_annual=self.rf_annual,
            )
            result = pipeline.evaluate(factor_values)
            return float(result.sharpe_ratio)

        except Exception:
            return -999.0

    def compare_all(self,
                    factor_values: pd.DataFrame) -> Dict[str, BacktestResult]:
        """全频率评估：返回所有频率的完整回测结果
        
        Args:
            factor_values: 因子值 DataFrame
            
        Returns:
            Dict[str, BacktestResult]: {频率: 回测结果}
        """
        pipeline = FactorPipeline(
            returns=self.returns,
            scheduler=FixedScheduler('ME'),  # 会被 compare_frequencies 覆盖
            allocator=self._allocator,
            cost_rate=self.cost_rate,
            rf_annual=self.rf_annual,
        )
        return pipeline.compare_frequencies(factor_values, self.frequencies)

    def _build_schedulers(self) -> Dict[str, RebalanceScheduler]:
        """构建频率→调度器映射"""
        schedulers = {}
        for freq in self.frequencies:
            freq_upper = freq.upper()
            if freq_upper in ('ME', 'W', 'M'):
                schedulers[freq] = FixedScheduler(freq_upper)
            elif freq.endswith('D') or freq.endswith('d'):
                days = int(freq.replace('D', '').replace('d', ''))
                schedulers[freq] = IntervalScheduler(days)
            else:
                try:
                    days = int(freq)
                    schedulers[freq] = IntervalScheduler(days)
                except ValueError:
                    pass
        return schedulers

    def __repr__(self) -> str:
        return (f"GPMultiFreqEvaluator(freqs={self.frequencies}, "
                f"topn={self.top_n})")
