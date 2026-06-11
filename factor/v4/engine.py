"""
factor/v4/engine.py — V4 因子表达式引擎 (基于 signals.v4 AST DSL)
=============================================================================
薄包装层：直接使用 signals.v4 AST DSL 解析和求值，无语法转换。

因子表达式使用 Python infix 语法：
    ts_rank((CLOSE - ts_delay(CLOSE, 20)), 10)
    cs_rank(ts_corr(CLOSE, VOLUME, 20))
    safe_max(CLOSE, ts_delay(CLOSE, 5)) if ...

用法:
    >>> from factor.v4 import V4FactorExpression
    >>> expr = V4FactorExpression("ts_rank((CLOSE - ts_delay(CLOSE, 20)), 10)")
    >>> panel = expr.evaluate(data_dict)  # → ndarray(T, N)
=============================================================================
"""
import numpy as np
from typing import Dict

import signals.v4.ast_dsl as dsl

# signals.v4 函数设计为 1D 数组，因子面板为 2D (T,N)，需逐列求值。
_BASE_VARS = {'open', 'high', 'low', 'close', 'volume', 'amount'}


class V4FactorExpression:
    """V4 因子表达式 — 字符串 → AST → ndarray(T,N)

    Example:
        >>> expr = V4FactorExpression("cs_rank(ts_roc(CLOSE, 20))")
        >>> panel = expr.evaluate(data_dict)    # ndarray(T,N)
        >>> print(expr.terminals)               # {'CLOSE'}
        >>> print(expr.functions)               # ['cs_rank', 'ts_roc']
    """

    def __init__(self, expression: str):
        self.source = expression
        self._tree = dsl.parse_expression(expression)
        self.terminals = set(dsl.get_variables(self._tree))
        self.functions = dsl.get_functions(self._tree)

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

    def __repr__(self):
        return f"V4FactorExpression({self.source[:60]!r})"

    def __str__(self):
        return self.source
