"""
factor/v5/llm/eval_utils.py — 快速评估工具

用于人在回路的快速验证：生成表达式 → 一行求 IC/IR → 观察 → 反馈 LLM。
依赖 signals/v4 (Expression) + factor/v5 (FactorExpression/Validator/Backtest)。

"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from signals.v4 import Expression
from factor.v5.expression import FactorExpression
from factor.v5.validator import FactorValidator


@dataclass
class FactorICResult:
    """单因子 IC 评估结果"""
    expression: str
    ic_mean: float
    ic_std: float
    icir: float
    positive_ratio: float
    t_stat: float
    n_days: int
    error: str = ""


def quick_ic(expression: str,
             panel_data: Dict[str, np.ndarray],
             returns: pd.DataFrame,
             method: str = 'spearman') -> FactorICResult:
    """单条表达式的快速 IC 评估

    Args:
        expression: 因子表达式字符串
        panel_data: OHLCV 面板数据 {变量: ndarray(T,N)}
        returns: 日收益率 DataFrame (index=日期, columns=品种)
        method: IC 计算方法 'spearman' | 'pearson'

    Returns:
        FactorICResult: IC 均值/标准差/ICIR/正占比/t统计量

    Example:
        >>> ic = quick_ic("cs_rank(ts_roc(CLOSE, 20))", panel_data, returns)
        >>> print(f"IC={ic.ic_mean:.4f}, IR={ic.icir:.2f}")
    """
    try:
        expr = FactorExpression(expression)
        values = expr.evaluate(panel_data)
        fv = pd.DataFrame(values, index=returns.index, columns=returns.columns)

        validator = FactorValidator(fv, returns)
        ic = validator.information_coefficient(method=method)

        return FactorICResult(
            expression=expression,
            ic_mean=round(ic.get('mean', 0), 6),
            ic_std=round(ic.get('std', 0), 6),
            icir=round(ic.get('ir', 0), 4),
            positive_ratio=round(ic.get('positive_ratio', 0), 4),
            t_stat=round(ic.get('t_stat', 0), 4),
            n_days=len(ic.get('daily_ics', [])),
        )
    except Exception as e:
        return FactorICResult(expression=expression, error=str(e),
                              ic_mean=0, ic_std=0, icir=0,
                              positive_ratio=0, t_stat=0, n_days=0)


def quick_ic_batch(expressions: List[str],
                   panel_data: Dict[str, np.ndarray],
                   returns: pd.DataFrame,
                   method: str = 'spearman',
                   min_icir: float = 0.0,
                   verbose: bool = True) -> List[FactorICResult]:
    """批量 IC 评估，返回排序结果

    Args:
        expressions: 表达式列表
        panel_data: OHLCV 面板数据
        returns: 日收益率 DataFrame
        method: IC 计算方法
        min_icir: 最低 ICIR 阈值 (低于的不返回)

    Returns:
        按 |IC均值| 降序排列的结果列表

    Example:
        >>> results = quick_ic_batch(candidates, panel_data, returns)
        >>> for r in results[:5]:
        ...     print(f"{r.icir:.2f}  {r.expression}")
    """
    results = []
    for expr in expressions:
        ic = quick_ic(expr, panel_data, returns, method)
        if verbose and not ic.error:
            print(f"  IC={ic.ic_mean:+.4f}  IR={ic.icir:+.2f}  Pos={ic.positive_ratio:.0%}  {expr[:70]}")
        elif verbose and ic.error:
            print(f"  X FAIL: {expr[:70]} — {ic.error}")
        if abs(ic.icir) >= min_icir and not ic.error:
            results.append(ic)

    results.sort(key=lambda x: abs(x.ic_mean), reverse=True)
    return results


def quick_rank_panel(expression: str,
                     assets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """快速生成因子排名面板

    Args:
        expression: 因子表达式
        assets: {品种代码: OHLCV DataFrame}

    Returns:
        DataFrame(日期×品种), 值=每日截面排名(0~1)

    Example:
        >>> ranked = quick_rank_panel("cs_rank(ts_roc(CLOSE, 20))", assets)
        >>> print(ranked.iloc[-1].nlargest(3))  # 最近一天 Top3
    """
    expr = Expression(expression)
    return expr.rank_panel(assets)


def quick_sharpe(expression: str,
                 assets: Dict[str, pd.DataFrame],
                 top_n: int = 3,
                 rebalance: str = 'W') -> float:
    """快速因子轮动回测 (返回 Sharpe)

    Args:
        expression: 因子表达式
        assets: {品种代码: OHLCV DataFrame}
        top_n: 持仓数
        rebalance: 调仓频率 'D'/'W'/'M'

    Returns:
        年化 Sharpe
    """
    from factor.v5.engine import FacEngine

    ranked = quick_rank_panel(expression, assets)
    analyzer = FacEngine.backtest(ranked, assets, top_n=top_n, rebalance=rebalance)
    return analyzer.sharpe_ratio()
