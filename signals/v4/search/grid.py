"""
signals/v3/search/grid.py — 参数网格搜索 (v3 独立引擎版)
=============================================================================
v3 独立版，笛卡尔积展开 + v3.EngineV3.backtest(mode='fast|full')。
=============================================================================
"""
import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Any

from ..expression_v3 import Expression
from ..features import FeatureSpace
from ..engine import EngineV3


class GridSearch:
    """
    v3 网格搜索器。

    用法:
        gs = GridSearch(template, param_grid, data, fs, start_date='2020-01-01')
        result = gs.run()
        print(result.head(5))
    """

    def __init__(self, template: str, param_grid: Dict[str, List],
                 data: pd.DataFrame, feature_space: FeatureSpace,
                 initial_capital: float = 1_000_000,
                 start_date: str = None,
                 symbol: str = '399317.SZ'):
        self.template = template
        self.param_grid = param_grid
        self.data = data
        self.fs = feature_space
        self.capital = initial_capital
        self.start_date = start_date
        self.symbol = symbol

    def run(self, mode: str = 'fast') -> pd.DataFrame:
        """
        展开参数网格并逐组合回测。

        Args:
            mode: 'fast' (搜索) / 'full' (小规模精确评估)

        Returns:
            DataFrame: 按 Sharpe 降序排列
        """
        # 笛卡尔积展开
        keys = list(self.param_grid.keys())
        values = list(self.param_grid.values())

        results = []
        total = 1
        for v in values:
            total *= len(v)

        for i, combo in enumerate(itertools.product(*values)):
            expr_str = self.template
            placeholder_count = expr_str.count('?')
            params = combo[:placeholder_count]

            for p in params:
                expr_str = expr_str.replace('?', str(p), 1)

            success = False
            for attempt in range(3):
                try:
                    expr = Expression(expr_str, feature_space=self.fs)
                    signal = expr.generate(self.data)

                    if mode == 'fast':
                        bt = EngineV3.backtest(
                            signal, self.data, symbol=self.symbol,
                            mode='fast', start_date=self.start_date)
                        results.append({
                            '表达式': expr_str,
                            '参数': str(params),
                            'Sharpe': bt.sharpe,
                            '年化': bt.cagr,
                            '最大回撤': bt.max_drawdown,
                            '交易': bt.trades,
                        })
                        success = True
                        break
                    else:
                        from core.analyzer import AccountAnalyzer
                        analyzer = EngineV3.backtest(
                            signal, self.data, symbol=self.symbol,
                            mode='full', start_date=self.start_date)
                        m = analyzer.metrics()

                        def _v(name, d=0):
                            for k, v in m.items():
                                if isinstance(v, dict) and v.get('name') == name:
                                    val = v['value']
                                    return val[0] if isinstance(val, tuple) else val
                            return d

                        results.append({
                            '表达式': expr_str,
                            '参数': str(params),
                            'Sharpe': _v('夏普比率'),
                            '年化': _v('年化收益率'),
                            '最大回撤': _v('最大回撤'),
                            '交易': len(analyzer.account.trade_records) // 2,
                        })
                        success = True
                        break
                except Exception:
                    continue

            if not success:
                results.append({
                    '表达式': expr_str, '参数': str(params),
                    'Sharpe': 0, '年化': 0, '最大回撤': 0, '交易': 0,
                })

        df = pd.DataFrame(results)
        return df.sort_values('Sharpe', ascending=False).reset_index(drop=True)
