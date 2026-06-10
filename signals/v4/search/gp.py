"""
signals/v3/search/gp.py — GP 遗传算法搜索 (v3 独立引擎版)
=============================================================================
继承 v3.gp_optimizer.GPOptimizer 的种群/交叉/变异逻辑,
替换 _evaluate_signals 为 v3.EngineV3.backtest(mode='fast')。

效果: 4000 次评估全部走 core 时间线 + ETF 费率, 约 33 分钟
=============================================================================
"""
import numpy as np
import pandas as pd
from typing import List, Dict

# ── v3 本地导入 ──
from ..gp_optimizer import (
    GPOptimizer, Individual, GenerationHistory,
    NODE_CONFIG, SEED_EXPRESSIONS,
)
from ..expression_v3 import Expression, parse_expression
from ..features import FeatureSpace
from ..engine import EngineV3


class GPSearch(GPOptimizer):
    """
    v3 GP 搜索器 — 算法同 v2, 回测用 v3 引擎。

    用法:
        gs = GPSearch(fs, train_data, test_data, population_size=80, generations=50)
        gs.run()
        gs.report()
        top10 = gs.elite_set(10)  # → 可选 v3 full 模式验证
    """

    def __init__(self, *args, start_date: str = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._start_date = start_date

    def _evaluate_signals(self, signals: np.ndarray, prices: pd.Series) -> Dict:
        """
        [重构] 用 v3.EngineV3(mode='fast') 替代 v2.run_backtest_from_signal
        """
        try:
            # 构造隐式 data (fast 模式只需 close)
            df = pd.DataFrame({'close': prices.values}, index=prices.index)
            result = EngineV3.backtest(
                signals, df, mode='fast', start_date=self._start_date)
            return {
                'sharpe': result.sharpe,
                'annual_return': result.cagr * 100,    # 转百分比 对齐 v2 接口
                'max_drawdown': abs(result.max_drawdown) * 100,
                'trade_count': result.trades,
            }
        except Exception:
            return {'sharpe': 0, 'annual_return': 0, 'max_drawdown': 100, 'trade_count': 0}

    def elite_set(self, n: int = 10) -> List[Individual]:
        """返回 Top-N elite, 用于 full 模式验证"""
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        return sorted_pop[:n]
