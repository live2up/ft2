# [重构] 2026-06-22 继承 utils.ast.AstExpression 公共基类
"""
factor/v5/expression.py — 因子表达式引擎 (继承 AstExpression)
=============================================================================

FactorExpression 是 AstExpression 的因子子类，在基类的解析+自省能力之上，
增加 2D 面板 → ndarray(T,N) 的求值能力。

职责:
  - evaluate(data_dict)         → 逐列求值 ndarray(T,N)
  - evaluate_ranked(data_dict)  → 截面排名 0~1  (委托 CsResolver)

对应的内部类 _ExpressionFromAST 是 GP 引擎使用的轻量包装器，
接收已有 AST 树跳过 parse 步骤直接求值。

与 signals/v4/Expression 的关系:
  - 共同基类: utils.ast.AstExpression (parse→variables→functions→complexity)
  - 差异: FactorExpression 输入 Dict[str, ndarray], 输出 ndarray(T,N)
          Expression 输入 pd.DataFrame, 输出 pd.Series

用法:
  >>> from factor.v5 import FactorExpression
  >>> expr = FactorExpression("cs_rank(ts_roc(CLOSE, 20))")
  >>> panel = expr.evaluate(data_dict)    # ndarray(T,N)
  >>> ranked = expr.evaluate_ranked(data_dict)  # 截面排名 0~1
=============================================================================
"""
import ast
import numpy as np
from typing import Dict

from utils.ast.v2 import AstExpression, CsResolver, normalize_data_keys, evaluate, eval_colwise


class _ExpressionFromAST:
    """轻量 AST 包装器: 接收已有 AST 树直接求值 (跳过 parse)

    使用场景: GP 引擎在内部生成/变异 ast.Expression 对象，
    直接传入此包装器求值，避免反复 parse 字符串。

    与 AstExpression 的区别:
      - AstExpression: expr_str → parse → tree → evaluate
      - _ExpressionFromAST: tree (已有) → evaluate
    """

    def __init__(self, tree: ast.Expression, name: str = ''):
        self._tree = tree
        self.name = name

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """逐列求值 1D 或 2D 数据 → ndarray"""
        data_norm = normalize_data_keys(data)
        T, N = None, None
        for v in data_norm.values():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                T, N = v.shape
                break

        if N is None:
            result = evaluate(self._tree, data_norm)
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        return eval_colwise(self._tree, data_norm, T, N)


class FactorExpression(AstExpression):
    """因子表达式 — 字符串 → AST → ndarray(T,N)

    继承 AstExpression 的解析+自省能力，提供 2D 面板求值。

    Attributes (继承自 AstExpression):
        expr_str, name, _tree, variables, functions, complexity
        _has_cs, _is_outer_cs_rank

    Example:
        >>> expr = FactorExpression("cs_rank(ts_roc(CLOSE, 20))")
        >>> panel = expr.evaluate(data_dict)    # ndarray(T,N)
        >>> ranked = expr.evaluate_ranked(data_dict)  # 截面排名 0~1
        >>> print(expr.variables)               # ['CLOSE']
        >>> print(expr.functions)               # ['cs_rank', 'ts_roc']
    """

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """求值为 (T, N) ndarray（逐列求值，时序函数 1D 安全）

        注意: 含截面函数的表达式请使用 evaluate_ranked()。
        evaluate() 逐列求值, 截面函数收到 1D 数组会退化。
        """
        # 运行时警告: 含截面函数的表达式走错入口
        if self._has_cs:
            import warnings
            warnings.warn(
                f"FactorExpression.evaluate(): 表达式含截面函数, "
                f"逐列求值会导致截面函数退化。请使用 evaluate_ranked()。"
                f"\n  表达式: {self.expr_str[:80]}"
            )
        # 规范化数据
        data_norm = normalize_data_keys(data)
        T, N = None, None
        for v in data_norm.values():
            if isinstance(v, np.ndarray) and v.ndim == 2:
                T, N = v.shape
                break

        # 1D
        if N is None:
            result = evaluate(self._tree, data_norm)
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        # 2D: 逐列求值
        return eval_colwise(self._tree, data_norm, T, N)

    def evaluate_ranked(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """截面排名求值 → 0~1 排名矩阵

        委托给 CsResolver 单遍 bottom-up 解析器。
        支持注册表中所有截面函数 (cs_rank/cs_zscore/cs_scale/...) 的嵌套/组合。

        Returns:
            ndarray(T,N), 每行截面排名(pct, 0~1), 值越大排名越高
        """
        data_norm = normalize_data_keys(data)
        return CsResolver().resolve(self._tree, data_norm)
