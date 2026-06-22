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


def _any_csrank_in_ast(tree):
    """[修复] 2026-06-22 扫描整棵 AST 中是否存在 cs_rank 调用

    原 _has_cs_rank 只检测最外层，导致 cs_rank(A) + cs_rank(B)
    这种组合表达式被误判为无 cs_rank，逐列求值时 cs_rank 收到 1D
    返回全 0.5，信息完全丢失。
    """
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == 'cs_rank'):
            return True
    return False


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
        # [修复] 2026-06-22 检测 cs_rank 存在性，拆分为两个标志:
        #   _has_cs_rank:  AST 中任意位置存在 cs_rank 调用 (全树扫描)
        #   _is_outer_cs_rank: 最外层本身就是 cs_rank 调用 (原 _has_cs_rank 语义)
        # 修复前只检测最外层，导致 cs_rank(A) + cs_rank(B) 组合被误判为无 cs_rank
        self._has_cs_rank = _any_csrank_in_ast(self._tree)
        self._is_outer_cs_rank = (
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

    def _cross_sectional_rank(self, vals: np.ndarray) -> np.ndarray:
        """每日截面排名 → 0~1"""
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

    def evaluate_ranked(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """截面排名求值 → 0~1 排名矩阵

        统一流程:
          1. 剥离外层 cs_rank (如有)
          2. 在剩余 AST 中检测内层 cs_rank 调用
          3. 无内层 cs_rank → 直接逐列求值 + 截面排名
          4. 有内层 cs_rank → AST 替换为预计算截面排名变量, 再逐列求值 + 截面排名

        支持示例:
          cs_rank(A)                         → 剥离, 求值A, 排名
          0.4*A + 0.3*B                     → 直接求值, 排名
          0.4*cs_rank(A) + 0.3*cs_rank(B)   → AST替换, 各自排名, 加权, 排名
          cs_rank(ts_delta(cs_rank(A), 5))  → 剥离外层, 内层cs_rank替换, 求值, 排名

        Returns:
            ndarray(T,N), 每行截面排名(pct, 0~1), 值越大排名越高
        """
        import copy

        # Step 1: 剥离外层 cs_rank
        # [修复] 2026-06-22 区分两种情况:
        #   _is_outer_cs_rank=True  → 外层是 cs_rank, 剥离后处理内层
        #   _has_cs_rank=True 但 _is_outer_cs_rank=False → 组合 cs_rank
        #     (如 cs_rank(A)+cs_rank(B)), 不剥离, 整树进入内层处理
        if self._is_outer_cs_rank:
            tree_to_eval = ast.Expression(body=self._tree.body.args[0])
        elif self._has_cs_rank:
            tree_to_eval = self._tree
        else:
            tree_to_eval = self._tree

        # Step 2: 检测内层 cs_rank 节点
        inner_csrank_nodes = []
        for node in ast.walk(tree_to_eval):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Name)
                    and node.func.id == 'cs_rank'
                    and node.args):
                inner_csrank_nodes.append(node)

        # Step 3: 无内层 cs_rank — 直接求值
        if not inner_csrank_nodes:
            vals = _ExpressionFromAST(tree_to_eval, self.expr_str).evaluate(data)
            return self._cross_sectional_rank(vals)

        # Step 4: 有内层 cs_rank — 从内到外逐层替换
        # 多轮迭代: 每轮只替换"参数中不含 cs_rank"的叶子节点, 直到全部替换完
        import copy
        current_tree = copy.deepcopy(tree_to_eval)
        current_data = dict(data)  # 注入预计算变量的数据集

        while True:
            # 收集本轮可替换的 cs_rank 节点 (参数中不含 cs_rank)
            leaf_nodes = []
            for node in ast.walk(current_tree):
                if (isinstance(node, ast.Call)
                        and isinstance(node.func, ast.Name)
                        and node.func.id == 'cs_rank'
                        and node.args):
                    # 检查参数是否还包含 cs_rank
                    has_inner_csrank = False
                    for child in ast.walk(node.args[0]):
                        if (isinstance(child, ast.Call)
                                and isinstance(child.func, ast.Name)
                                and child.func.id == 'cs_rank'):
                            has_inner_csrank = True
                            break
                    if not has_inner_csrank:
                        leaf_nodes.append(node)

            if not leaf_nodes:
                # 检查是否还有未替换的 cs_rank (不应发生, 安全检查)
                remaining = any(
                    isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'cs_rank'
                    for n in ast.walk(current_tree)
                )
                if remaining:
                    raise ValueError(f"evaluate_ranked: 无法解析的 cs_rank 嵌套: {self.expr_str}")
                break

            # 预计算每个叶子 cs_rank 的截面排名
            n_replace = 0
            sub_arrays = []
            for node in leaf_nodes:
                n_replace += 1
                var_name = f'_CSR{n_replace}_{id(node) & 0xFFFF}'
                inner_tree = ast.Expression(body=node.args[0])
                inner_expr = _ExpressionFromAST(inner_tree, var_name)
                inner_vals = inner_expr.evaluate(current_data)
                ranked = self._cross_sectional_rank(inner_vals)
                sub_arrays.append((var_name, ranked, node))

            # AST 替换
            class _LeafCSRTransformer(ast.NodeTransformer):
                def __init__(self, subs):
                    self._subs = subs  # [(var_name, _, node), ...]
                    self._count = 0

                def visit_Call(self, node):
                    node = self.generic_visit(node)  # bottom-up
                    if (isinstance(node.func, ast.Name)
                            and node.func.id == 'cs_rank'):
                        self._count += 1
                        idx = self._count - 1
                        if idx < len(self._subs):
                            return ast.Name(id=self._subs[idx][0], ctx=ast.Load())
                    return node

            transformer = _LeafCSRTransformer(sub_arrays)
            current_tree = transformer.visit(current_tree)
            ast.fix_missing_locations(current_tree)

            # 注入变量
            for var_name, arr, _ in sub_arrays:
                current_data[var_name] = arr

        # 最终求值
        vals = _ExpressionFromAST(current_tree, self.expr_str).evaluate(current_data)
        return self._cross_sectional_rank(vals)

    def __repr__(self):
        return f"FactorExpression({self.expr_str[:60]!r})"

    def __str__(self):
        return self.expr_str
