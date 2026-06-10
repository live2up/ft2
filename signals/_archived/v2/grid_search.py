"""
signals/v2/grid_search.py — 参数网格搜索编排器
=============================================================================

对表达式模板进行参数网格搜索，自动遍历所有参数组合，
对每个组合执行回测 + Walk-Forward 验证，汇总排序输出。


============================================================================
                         架构层级（竖式）
============================================================================

 ┌─────────────────────────────────────────────────────────────────────┐
 │  输入：表达式模板 + 参数网格                                          │
 │                                                                      │
 │  template = "thr_mean(ATR(?))"                                      │
 │  param_grid = {'?': [7, 10, 14, 20]}                                │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 1  —  参数展开 (笛卡尔积)                                       │
 │                                                                      │
 │  ? ∈ [7, 10, 14, 20] → 4 个表达式                                   │
 │  thr_mean(ATR(7)), thr_mean(ATR(10)), ...                            │
 │                                                                      │
 │  多参数: template="ATR(?) & TRIMA(?)"                                 │
 │  param_grid={'?': [[7,40], [7,60], [14,60]]} → 3 个表达式           │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 2  —  逐组合回测                                                │
 │                                                                      │
 │  每个组合: run_backtest() → 记录 Sharpe/年化/回撤/胜率               │
 │  可选: walk_forward() → 记录稳定性得分                                │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 3  —  汇总排序输出                                              │
 │                                                                      │
 │  results.sort(key='Sharpe') → 最佳参数组合                           │
 │  GridResult → 排名表 / 最佳表达式 / 参数热力图                       │
 └─────────────────────────────────────────────────────────────────────┘

============================================================================
"""

import itertools
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
import warnings

from .expression import Expression
from .features import FeatureSpace
from .validator import run_backtest, walk_forward, WalkForwardResult
from .presets import PRESETS, from_preset

warnings.filterwarnings('ignore')


@dataclass
class GridResult:
    params: Dict
    expression_str: str
    total_return: float = 0.0
    annual_return: float = 0.0
    benchmark_return: float = 0.0
    excess_return: float = 0.0
    max_drawdown: float = 0.0
    sharpe: float = 0.0
    sortino: float = 0.0
    calmar: float = 0.0
    win_rate: float = 0.0
    trade_count: int = 0
    avg_hold_days: float = 0.0
    wf_sharpe: float = 0.0
    wf_stability: float = 0.0

    def __repr__(self):
        return (f"GridResult(sharpe={self.sharpe:.3f}, annual={self.annual_return:.1f}%, "
                f"wf_sharpe={self.wf_sharpe:.3f}, params={self.params})")


class GridSearch:
    """
    参数网格搜索器

    Args:
        feature_space: FeatureSpace 实例
        data: OHLCV DataFrame
        initial_capital: 回测初始资金
        sort_by: 排序字段（默认 'sharpe'）
    """

    def __init__(self, feature_space: FeatureSpace, data: pd.DataFrame,
                 initial_capital: float = 1_000_000,
                 sort_by: str = 'sharpe'):
        self.feature_space = feature_space
        self.data = data
        self.initial_capital = initial_capital
        self.sort_by = sort_by
        self._templates: List[Tuple[str, Dict, Any]] = []
        self._results: List[GridResult] = []

    def from_preset(self, preset_name: str,
                    param_grid: Dict[str, List[Any]] = None) -> 'GridSearch':
        """
        从预设模板创建网格搜索

        Args:
            preset_name: PRESETS中的模板名
            param_grid: 参数字典，key为占位符名，value为参数值列表
                如 {'?': [7, 10, 14]} 对应 "thr_mean(ATR(?))" 的单参数
                如 {'?': [[7, 40], [7, 60]]} 对应双参数模板（每项是列表）
        """
        preset = PRESETS.get(preset_name)
        if preset is None:
            available = ', '.join(PRESETS.keys())
            raise ValueError(f"未知模板: {preset_name}。可用: {available}")

        if param_grid is None:
            raise ValueError("必须提供 param_grid")

        self._templates.append((preset.template, param_grid, preset_name))
        return self

    def from_template(self, template: str,
                      param_grid: Dict[str, List[Any]],
                      name: str = None) -> 'GridSearch':
        """
        从自定义表达式模板创建网格搜索

        Args:
            template: 表达式模板，用 ? 做占位符，如 "thr_mean(ATR(?))"
            param_grid: 参数字典
            name: 可选的模板名
        """
        self._templates.append((template, param_grid, name or 'custom'))
        return self

    def _expand_combinations(self, template: str,
                              param_grid: Dict[str, List[Any]]) -> List[Tuple[str, Dict]]:
        placeholders = sorted(param_grid.keys())
        values_list = [param_grid[k] for k in placeholders]

        results = []
        for combo in itertools.product(*values_list):
            expr_str = template
            params = {}
            for ph, val in zip(placeholders, combo):
                if isinstance(val, list):
                    params[f'arg_{ph}'] = tuple(val)
                else:
                    params[f'arg_{ph}'] = val

            if isinstance(combo[0], list):
                flat = combo[0]
            else:
                flat = list(combo)

            expr_str = template
            for val in flat if isinstance(flat, list) else [flat] if len(flat) == 1 else flat:
                if isinstance(val, list):
                    for v in val:
                        expr_str = expr_str.replace('?', str(v), 1)
                else:
                    expr_str = expr_str.replace('?', str(val), 1)

            results.append((expr_str, params))

        return results

    def run(self, do_walk_forward: bool = False,
            wf_train: str = '2Y', wf_test: str = '1Y', wf_step: str = '6M',
            verbose: bool = True) -> pd.DataFrame:
        """
        执行网格搜索

        Args:
            do_walk_forward: 是否对每个组合执行 Walk-Forward
            wf_train/wf_test/wf_step: Walk-Forward 参数
            verbose: 是否打印进度

        Returns:
            DataFrame: 按排序字段降序排列的结果表
        """
        self._results = []

        for template, param_grid, template_name in self._templates:
            if verbose:
                print(f"\n网格搜索: {template_name}")
                print(f"  模板: {template}")

            combinations = self._expand_combinations(template, param_grid)
            total = len(combinations)
            if verbose:
                print(f"  参数组合数: {total}")

            for idx, (expr_str, params) in enumerate(combinations):
                try:
                    if verbose:
                        print(f"  [{idx + 1}/{total}] {expr_str} ...", end=' ')

                    expr = Expression(expr_str, feature_space=self.feature_space)
                    bt = run_backtest(expr, self.data, self.initial_capital)

                    result = GridResult(
                        params=params,
                        expression_str=expr_str,
                        total_return=bt.total_return,
                        annual_return=bt.annual_return,
                        benchmark_return=bt.benchmark_total_return,
                        excess_return=bt.excess_return,
                        max_drawdown=bt.max_drawdown,
                        sharpe=bt.sharpe,
                        sortino=bt.sortino,
                        calmar=bt.calmar,
                        win_rate=bt.win_rate,
                        trade_count=bt.trade_count,
                        avg_hold_days=bt.avg_hold_days,
                    )

                    if do_walk_forward:
                        try:
                            wf = walk_forward(expr, self.data, wf_train, wf_test, wf_step)
                            result.wf_sharpe = wf.mean_sharpe
                            result.wf_stability = wf.stability_score
                        except Exception:
                            pass

                    self._results.append(result)

                    if verbose:
                        print(f"Sharpe={bt.sharpe:.3f}")

                except Exception as e:
                    if verbose:
                        print(f"FAILED: {e}")

        self._results.sort(key=lambda x: getattr(x, self.sort_by, 0), reverse=True)

        if verbose and self._results:
            print(f"\n{'=' * 60}")
            print(f"网格搜索完成，共 {len(self._results)} 个有效结果")
            print(f"Top 5 (按 {self.sort_by}):")
            for i, r in enumerate(self._results[:5], 1):
                print(f"  {i}. {r.expression_str:40s}  Sharpe={r.sharpe:.3f}  "
                      f"年化={r.annual_return:+.1f}%  回撤={r.max_drawdown:.1f}%")

        return self.to_dataframe()

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for r in self._results:
            rows.append({
                '表达式': r.expression_str,
                '参数': str(r.params),
                '夏普': round(r.sharpe, 3),
                '年化(%)': round(r.annual_return, 2),
                '超额(%)': round(r.excess_return, 2),
                '回撤(%)': round(r.max_drawdown, 2),
                'Sortino': round(r.sortino, 3),
                'Calmar': round(r.calmar, 3),
                '胜率(%)': round(r.win_rate, 1),
                '交易次数': r.trade_count,
                '平均持仓(天)': round(r.avg_hold_days, 1),
                'WF夏普': round(r.wf_sharpe, 3) if r.wf_sharpe != 0 else None,
                'WF稳定性': round(r.wf_stability, 3) if r.wf_stability != 0 else None,
            })
        return pd.DataFrame(rows)

    def best_params(self) -> Optional[Dict]:
        if not self._results:
            return None
        return self._results[0].params

    def best_expression(self) -> Optional[str]:
        if not self._results:
            return None
        return self._results[0].expression_str

    def top_results(self, n: int = 5) -> List[GridResult]:
        return self._results[:n]
