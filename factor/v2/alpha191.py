"""
alpha191.py — 191 Alpha 因子探索框架（ExpressionFactor + 批量探索器）
=============================================================================

提供 ExpressionFactor 适配器和 Alpha191Explorer 批量探索器。
因子公式定义在独立的 alpha101.py 中，可通过 alpha191 统一导入。

核心组件：
  1) ExpressionFactor: 将 FactorExpression 包装为 Factor 子类
  2) Alpha191Explorer: 批量探索（检验 + 回测）
  3) 从 alpha101 导入 101 个因子公式

使用方式：
  >>> from factor.v2.alpha191 import expression_factor, ALPHA101
  >>> ef = expression_factor('alpha001')
  >>> fv = ef.calculate(data, symbols, dates)
  >>> explorer = Alpha191Explorer(data, returns)
  >>> results = explorer.run(['alpha001', 'alpha006', 'alpha012'])

[新增] 2026-05-30 GTJA 191 Alpha 因子库 v2 适配
=============================================================================
"""

from typing import Dict, List, Any
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .base import Factor, FactorMetadata, FactorCategory, FactorFrequency
from .expression import FactorExpression
from .gp_primitives import delay
from .validator import FactorValidator
from .pipeline import FactorPipeline
from .scheduler import FixedScheduler
from .allocator import TopNEqualWeight

# 从 alpha101 导入公式数据（向后兼容别名）
from .alpha101 import ALPHA101 as ALPHA_FORMULAS
from .alpha101 import ALPHA101_CATEGORIES as ALPHA_CATEGORIES

# 直接导出，方便外部使用
__all__ = [
    'ExpressionFactor', 'expression_factor',
    'Alpha191Explorer', 'AlphaResult',
    'ALPHA101', 'ALPHA_FORMULAS',
    'ALPHA101_CATEGORIES', 'ALPHA_CATEGORIES',
]


# ============================================================================
# Part 1: ExpressionFactor 适配器
# ============================================================================

class ExpressionFactor(Factor):
    """将 FactorExpression 字符串包装为 Factor 子类

    使表达式因子可以接入 grid_search、pipeline、validator 等所有 v2 基础设施。

    关键设计：
    - calculate() 自动注入衍生字段 'returns' 和 'vwap'，公式中可直接引用
    - 接收 pd.DataFrame 字典（与 grid_search 一致），内部转为 ndarray 求值

    Example:
        >>> ef = ExpressionFactor("ts_rank(sub(close, delay(close, 20)), 10)")
        >>> fv = ef.calculate({'close': close_df}, symbols, dates)
    """

    def __init__(self,
                 expression_str: str,
                 name: str = 'expr_factor',
                 category: FactorCategory = FactorCategory.CUSTOM,
                 description: str = ''):
        metadata = FactorMetadata(
            name=name,
            description=description or f'表达式因子: {expression_str[:60]}',
            category=category,
            frequency=FactorFrequency.DAILY,
            parameters={'expression': expression_str},
        )
        super().__init__(metadata)
        self.expression_str = expression_str
        self.expr = FactorExpression(expression_str)

    def calculate(self,
                  data: Dict[str, pd.DataFrame],
                  symbols: List[str],
                  dates: List[Any]) -> pd.DataFrame:
        """计算因子值

        将 DataFrame 数据转为 ndarray，注入衍生字段后求值表达式。

        Args:
            data: {field: DataFrame(index=日期, columns=标的)}
            symbols: 标的列表
            dates: 日期列表

        Returns:
            pd.DataFrame: 因子值，index=dates, columns=symbols
        """
        # 提取 ndarray（按列对齐）
        ndarray_data = {}
        for field in ['open', 'high', 'low', 'close', 'volume', 'amount']:
            if field in data:
                df = data[field]
                ndarray_data[field] = np.asarray(df.values, dtype=float)

        if not ndarray_data:
            raise ValueError("数据字典为空，至少需要 close")

        # 注入衍生字段
        close_arr = ndarray_data.get('close')
        vol_arr = ndarray_data.get('volume')
        if close_arr is not None:
            # returns = close / delay(close, 1) - 1
            close_delayed = delay(close_arr, 1)
            safe_delayed = np.where(np.abs(close_delayed) > 1e-10, close_delayed, np.nan)
            ndarray_data['returns'] = close_arr / safe_delayed - 1.0
            ndarray_data['returns'] = np.nan_to_num(ndarray_data['returns'], nan=0.0, posinf=0.0, neginf=0.0)

            # vwap = ts_sum(close * volume, 1) / ts_sum(volume, 1)（日频）
            if vol_arr is not None:
                amount_arr = close_arr * vol_arr
                ndarray_data['vwap'] = np.where(
                    np.abs(vol_arr) > 1e-10,
                    amount_arr / vol_arr,
                    close_arr,
                )

        # 求值
        result_ndarray = self.expr.evaluate(ndarray_data)

        # 转回 DataFrame
        result = pd.DataFrame(result_ndarray, index=dates[:result_ndarray.shape[0]],
                              columns=symbols[:result_ndarray.shape[1]])
        return result

    def __repr__(self):
        return f"ExpressionFactor({self.expression_str[:60]}...)"


def expression_factor(alpha_id: str, **kwargs) -> ExpressionFactor:
    """根据 alpha ID 创建 ExpressionFactor 实例

    Args:
        alpha_id: 因子 ID，如 'alpha001'、'alpha042'
        **kwargs: 传递给 ExpressionFactor 的额外参数

    Returns:
        ExpressionFactor 实例
    """
    formula = ALPHA_FORMULAS.get(alpha_id)
    if formula is None:
        raise KeyError(f"未知因子: {alpha_id}，可用: {list(ALPHA_FORMULAS.keys())[:10]}...")
    return ExpressionFactor(formula, name=alpha_id, **kwargs)


# ============================================================================
# Part 2: Alpha191Explorer — 批量探索器
# ============================================================================

@dataclass
class AlphaResult:
    """单个 Alpha 因子的检验 + 回测结果"""
    alpha_id: str
    ic_mean: float = 0.0
    ic_ir: float = 0.0
    sharpe: float = 0.0
    annual_return: float = 0.0
    max_drawdown: float = 0.0
    hit_rate: float = 0.0
    expression: str = ''
    error: str = ''

    def __repr__(self):
        return (f"AlphaResult({self.alpha_id}, IC={self.ic_mean:.4f}, "
                f"IR={self.ic_ir:.2f}, Sharpe={self.sharpe:.2f})")


class Alpha191Explorer:
    """191 Alpha 批量探索器

    对一个因子列表执行：表达式求值 → IC 检验 → Pipeline 回测 → 汇总排序。

    Example:
        >>> explorer = Alpha191Explorer(data, returns)
        >>> results = explorer.run(['alpha001', 'alpha006', 'alpha012'])
        >>> best = explorer.best(min_ic=0.02, min_sharpe=0.5)
    """

    def __init__(self,
                 data: Dict[str, pd.DataFrame],
                 returns: pd.DataFrame,
                 cost_rate: float = 0.001,
                 freq: str = 'ME',
                 top_n: int = 10):
        """
        Args:
            data: OHLCV 数据字典 {field: DataFrame}
            returns: 日收益率 DataFrame
            cost_rate: 交易成本费率
            freq: 调仓频率 ('ME', 'W', '5D' 等)
            top_n: 持仓数
        """
        self.data = data
        self.returns = returns
        self.cost_rate = cost_rate
        self.freq = freq
        self.top_n = top_n

        # 对齐日期
        if not isinstance(self.returns.index, pd.DatetimeIndex):
            self.returns = self.returns.copy()
            self.returns.index = pd.to_datetime(self.returns.index)

        self.symbols = list(self.returns.columns)
        self.dates = list(self.returns.index)
        self.future_returns = self.returns.shift(-1)

        self.results: List[AlphaResult] = []

    def run(self,
            alpha_ids: List[str] = None,
            verbose: bool = True) -> List[AlphaResult]:
        """批量探索因子

        Args:
            alpha_ids: 因子 ID 列表，None 表示全部 101 个（已定义）
            verbose: 是否打印进度

        Returns:
            List[AlphaResult]: 按 IC×IR 降序排列的结果
        """
        if alpha_ids is None:
            alpha_ids = list(ALPHA_FORMULAS.keys())

        self.results = []
        scheduler = FixedScheduler(self.freq)
        allocator = TopNEqualWeight(self.top_n)

        for i, aid in enumerate(alpha_ids):
            if verbose:
                print(f"[{i+1}/{len(alpha_ids)}] {aid}...", end=' ')

            try:
                # 1) 创建因子并计算
                formula = ALPHA_FORMULAS.get(aid)
                if formula is None:
                    if verbose:
                        print("跳过(未定义)")
                    continue

                ef = ExpressionFactor(formula, name=aid)
                fv = ef.calculate(self.data, self.symbols, self.dates)

                if fv.isna().all().all():
                    if verbose:
                        print("全NaN")
                    continue

                # 2) IC 检验
                validator = FactorValidator(fv, self.future_returns)
                ic_result = validator.information_coefficient()
                ic_mean = ic_result.get('mean', 0.0)
                ic_ir = ic_result.get('ir', 0.0)
                hr = validator.hit_rate()

                if ic_mean is None or np.isnan(ic_mean):
                    if verbose:
                        print("IC=NaN")
                    continue

                # 3) Pipeline 回测
                pipeline = FactorPipeline(
                    returns=self.returns,
                    scheduler=scheduler,
                    allocator=allocator,
                    cost_rate=self.cost_rate,
                )
                bt = pipeline.evaluate(fv)

                result = AlphaResult(
                    alpha_id=aid,
                    ic_mean=float(ic_mean),
                    ic_ir=float(ic_ir) if not np.isnan(ic_ir) else 0.0,
                    sharpe=bt.sharpe_ratio,
                    annual_return=bt.annual_return,
                    max_drawdown=bt.max_drawdown,
                    hit_rate=float(hr) if not np.isnan(hr) else 0.0,
                    expression=formula,
                )
                self.results.append(result)

                if verbose:
                    print(f"IC={ic_mean:.4f} IR={ic_ir:.2f} Sharpe={bt.sharpe_ratio:.2f}")

            except Exception as e:
                if verbose:
                    print(f"失败: {e}")
                self.results.append(AlphaResult(alpha_id=aid, error=str(e)))

        # 排序：IC×IR 降序
        self.results.sort(
            key=lambda r: abs(r.ic_mean) * max(r.ic_ir, 0),
            reverse=True,
        )

        return self.results

    def best(self, min_ic: float = 0.0, min_sharpe: float = 0.0,
             top_n: int = 20) -> List[AlphaResult]:
        """获取满足阈值的 Top N 因子

        Args:
            min_ic: 最低 |IC| 阈值
            min_sharpe: 最低 Sharpe 阈值
            top_n: 返回数量

        Returns:
            List[AlphaResult]
        """
        filtered = [
            r for r in self.results
            if abs(r.ic_mean) >= min_ic and r.sharpe >= min_sharpe and not r.error
        ]
        return filtered[:top_n]

    def to_dataframe(self) -> pd.DataFrame:
        """汇总为 DataFrame"""
        if not self.results:
            return pd.DataFrame()

        rows = []
        for r in self.results:
            rows.append({
                'Alpha': r.alpha_id,
                '|IC|': round(abs(r.ic_mean), 4),
                'IC': round(r.ic_mean, 4),
                'IR': round(r.ic_ir, 2),
                'Sharpe': round(r.sharpe, 2),
                '年化收益': f"{r.annual_return:.1%}",
                '最大回撤': f"{r.max_drawdown:.1%}",
                '命中率': f"{r.hit_rate:.1%}",
                '错误': r.error[:40] if r.error else '',
            })

        return pd.DataFrame(rows)

    def __repr__(self):
        n = len(self.results)
        valid = sum(1 for r in self.results if not r.error)
        return f"Alpha191Explorer(total={n}, valid={valid})"
