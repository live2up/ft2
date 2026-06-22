# [重构] 2026-06-12 对齐 signals/v4/expression.py 参数规范
"""
factor/v4/expression.py — 因子表达式引擎 (基于 signals.v4 AST DSL)

薄包装层：直接使用 signals.v4 AST DSL 解析和求值，无语法转换。

因子表达式使用 Python infix 语法：
    ts_rank((CLOSE - ts_delay(CLOSE, 20)), 10)
    cs_rank(ts_corr(CLOSE, VOLUME, 20))
    safe_max(CLOSE, ts_delay(CLOSE, 5)) if ...

用法:
    >>> from factor.v4 import FactorExpression
    >>> expr = FactorExpression("ts_rank((CLOSE - ts_delay(CLOSE, 20)), 10)")
    >>> panel = expr.evaluate(data_dict)  # → ndarray(T, N)
"""
import ast
import numpy as np
from typing import Dict

import signals.v4.ast_dsl as dsl
from utils.ast import CsResolver, _has_any_cs, _is_outer_cs_rank_call, _eval_colwise

# signals.v4 函数设计为 1D 数组，因子面板为 2D (T,N)，需逐列求值。
_BASE_VARS = {'open', 'high', 'low', 'close', 'volume', 'amount'}


class _ExpressionFromAST:
    """轻量 AST 包装器: 跳过 parse, 直接用已有 AST 树求值"""

    def __init__(self, tree: ast.Expression, name: str = ''):
        self._tree = tree
        self.name = name

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """[重构] 2026-06-22 复用 _eval_colwise 消除重复逻辑"""
        data_norm = {}
        T, N = None, None
        for k, v in data.items():
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 2:
                T, N = arr.shape
            key = k.upper() if k.lower() in _BASE_VARS else k
            data_norm[key] = arr

        if N is None:
            result = dsl.evaluate(self._tree, data_norm)
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return _eval_colwise(self._tree, data_norm, T, N)


class FactorExpression:
    """因子表达式 — 字符串 → AST → ndarray(T,N)

    对齐 signals/v4 Expression 参数规范:
      - expr_str: 表达式字符串
      - variables: 依赖变量列表
      - functions: 依赖函数列表
      - complexity: AST 节点数

    Example:
        >>> expr = FactorExpression("cs_rank(ts_roc(CLOSE, 20))")
        >>> panel = expr.evaluate(data_dict)    # ndarray(T,N)
        >>> ranked = expr.evaluate_ranked(data_dict)  # 截面排名 0~1
        >>> print(expr.variables)               # ['CLOSE']
        >>> print(expr.functions)               # ['cs_rank', 'ts_roc']
    """

    def __init__(self, expr_str: str, name: str = None):
        self.expr_str = expr_str.strip()
        self.name = name or expr_str[:60]
        self._tree = dsl.parse_expression(self.expr_str)
        self.variables = dsl.get_variables(self._tree)
        self.functions = dsl.get_functions(self._tree)
        self.complexity = sum(1 for _ in ast.walk(self._tree.body))
        # [重构] 2026-06-22 cs_rank 检测委托给 CsResolver (注册表驱动)
        self._has_cs_rank = _has_any_cs(self._tree)
        self._is_outer_cs_rank = _is_outer_cs_rank_call(self._tree)

    @property
    def features_used(self) -> list:
        """[新增] 2026-06-22 对齐 signals/v4 Expression 参数规范"""
        return self.variables

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """求值为 (T, N) ndarray。

        注意: 含截面函数的表达式请使用 evaluate_ranked()。
        evaluate() 逐列求值, 截面函数收到 1D 数组会退化。
        """
        # [新增] 2026-06-22 运行时警告: 含 cs_rank 的表达式走错入口
        if self._has_cs_rank:
            import warnings
            warnings.warn(
                f"FactorExpression.evaluate(): 表达式含截面函数, "
                f"逐列求值会导致截面函数退化。请使用 evaluate_ranked()。"
                f"\n  表达式: {self.expr_str[:80]}"
            )
        # 规范化数据
        data_norm = {}
        T, N = None, None
        for k, v in data.items():
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 2:
                T, N = arr.shape
            key = k.upper() if k.lower() in _BASE_VARS else k
            data_norm[key] = arr

        # 1D
        if N is None:
            result = dsl.evaluate(self._tree, data_norm)
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        # 2D: 逐列 (复用 cs_resolver 的逐列求值器)
        return _eval_colwise(self._tree, data_norm, T, N)

    def evaluate_ranked(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """截面排名求值 → 0~1 排名矩阵

        [重构] 2026-06-22 委托给 CsResolver 单遍 bottom-up 解析器。
        支持注册表中所有截面函数的嵌套/组合。

        Returns:
            ndarray(T,N), 每行截面排名(pct, 0~1), 值越大排名越高
        """
        # 规范化数据 (与 evaluate 共用逻辑)
        data_norm = {}
        for k, v in data.items():
            arr = np.asarray(v, dtype=float)
            key = k.upper() if k.lower() in _BASE_VARS else k
            data_norm[key] = arr

        return CsResolver().resolve(self._tree, data_norm)

    def __repr__(self):
        return f"FactorExpression({self.expr_str[:60]!r})"

    def __str__(self):
        return self.expr_str
