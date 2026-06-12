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

# signals.v4 函数设计为 1D 数组，因子面板为 2D (T,N)，需逐列求值。
_BASE_VARS = {'open', 'high', 'low', 'close', 'volume', 'amount'}


class _ExpressionFromAST:
    """轻量 AST 包装器: 跳过 parse, 直接用已有 AST 树求值"""

    def __init__(self, tree: ast.Expression, name: str = ''):
        self._tree = tree
        self.name = name

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
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

        result = np.full((T, N), 0.0)
        for j in range(N):
            col_data = {}
            for k, v in data_norm.items():
                col_data[k] = v[:, j] if isinstance(v, np.ndarray) and v.ndim == 2 else v
            col_result = dsl.evaluate(self._tree, col_data)
            result[:, j] = np.nan_to_num(
                col_result[:T].ravel(), nan=0.0, posinf=0.0, neginf=0.0)
        return result


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
        # 检测是否为 cs_rank(...) 包装的表达式
        self._has_cs_rank = (
            isinstance(self._tree.body, ast.Call)
            and isinstance(self._tree.body.func, ast.Name)
            and self._tree.body.func.id == 'cs_rank'
        )

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """求值为 (T, N) ndarray。

        signals.v4 函数设计为 1D 数组，因子面板为 2D (T,N)，
        逐列求值确保兼容性。
        """
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

        # 2D: 逐列
        result = np.full((T, N), 0.0)
        for j in range(N):
            col_data = {}
            for k, v in data_norm.items():
                col_data[k] = v[:, j] if isinstance(v, np.ndarray) and v.ndim == 2 else v
            col_result = dsl.evaluate(self._tree, col_data)
            result[:, j] = np.nan_to_num(
                col_result[:T].ravel(), nan=0.0, posinf=0.0, neginf=0.0,
            )
        return result

    def evaluate_ranked(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """截面排名求值 → 0~1 排名矩阵

        AST 级别处理: 检测 cs_rank() 外层包装，剥离内层表达式求值，
        再每日截面排名。解决逐列求值时 cs_rank 失去跨品种上下文的问题。

        Returns:
            ndarray(T,N), 每行截面排名(pct, 0~1), 值越大排名越高
        """
        # AST 级别剥离 cs_rank: 取 Call 的第一个参数作为内层表达式
        if self._has_cs_rank:
            inner_node = self._tree.body.args[0]
            inner_tree = ast.Expression(body=inner_node)
            inner_name = ast.unparse(inner_node)
        else:
            inner_tree = self._tree
            inner_name = self.expr_str

        # 对内层表达式求值 (逐列 2D)
        inner_expr = _ExpressionFromAST(inner_tree, inner_name)
        vals = inner_expr.evaluate(data)

        from scipy.stats import rankdata
        T, N = vals.shape
        ranked = np.full((T, N), 0.0)
        for t in range(T):
            row = vals[t]
            valid = ~np.isnan(row)
            if valid.sum() > 0:
                rk = rankdata(row[valid])
                ranked[t, valid] = np.nan_to_num(
                    rk / valid.sum(), nan=0.5, posinf=0.5, neginf=0.5)
        return ranked

    def __repr__(self):
        return f"FactorExpression({self.expr_str[:60]!r})"

    def __str__(self):
        return self.expr_str
