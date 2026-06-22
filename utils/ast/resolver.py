"""
utils/ast/resolver.py — 编排层 (公共基础设施)

在四层架构中的位置: 第4层(编排) — cs_* 截面函数嵌套解算

核心设计:
  1. 自动发现: 从 FUNC_REGISTRY 中以 cs_ 前缀自动识别截面函数
     新增 cs_* 函数无需任何配置, 自动获得嵌套/组合支持
  2. 单遍 bottom-up: ast.NodeTransformer 天然保证内层先于外层处理
  3. 混合求值: 时序函数逐列求值(1D安全), 截面函数全面板求值(2D正确)
  4. 剥离外层 cs_rank: resolve() 返回截面排名(0~1), 外层 cs_rank 由最终排名替代

支持嵌套模式:
  - 简单: cs_rank(A)
  - 嵌套: cs_rank(ts_delta(cs_rank(A), 5))
  - 组合: cs_rank(A) + cs_rank(B)*0.3
  - 混合: cs_rank(A) + cs_rank(ts_delta(cs_rank(A), 15))*0.3
  - 多截面: cs_rank(A) + cs_zscore(B)        [自动支持所有 cs_* 函数]

[重构] 2026-06-22 从 factor/v4 提取到 utils/ast 公共层
"""

import ast
import copy
import numpy as np
from typing import Dict, Set, Optional

from .registry import FUNC_REGISTRY
from .dsl import normalize_data_keys, evaluate, eval_colwise, cross_sectional_rank

# ============================================================
# 截面函数检测 — 运行时动态查询 FUNC_REGISTRY (支持热注册)
# ============================================================

def _is_cs_function(name: str) -> bool:
    """[新增] 2026-06-22 运行时动态检测: 是否为截面函数 (支持 register_function 热注册)"""
    return name.startswith('cs_') and name in FUNC_REGISTRY

def _get_cs_functions() -> Set[str]:
    """[新增] 2026-06-22 获取当前所有 cs_* 函数名 (调试用)"""
    return {name for name in FUNC_REGISTRY if name.startswith('cs_')}


# ============================================================
# CsResolver — 单遍截面函数解析器
# ============================================================

class CsResolver:
    """单遍截面函数解析器

    替代原有 while-loop 多轮迭代, 使用 ast.NodeTransformer bottom-up
    遍历实现单遍处理。所有截面函数通过注册表 CS_FUNCTIONS 管理。

    Usage:
        >>> from utils.ast import CsResolver
        >>> resolver = CsResolver()
        >>> ranked = resolver.resolve(tree, data)  # → ndarray(T,N) 0~1
    """

    def __init__(self):
        self._counter = 0
        self._T = 0
        self._N = 0

    def resolve(self, tree: ast.Expression, data: Dict[str, np.ndarray]) -> np.ndarray:
        """解析 AST 中所有截面函数 → 截面排名 ndarray(T,N)

        流程:
          1. 规范化数据 key, 检测维度
          2. 剥离外层 cs_rank (最终排名替代)
          3. 无截面函数 → 逐列求值 + 截面排名
          4. 有截面函数 → 单遍 bottom-up 替换 → 逐列求值 + 截面排名
        """
        data_norm = self._normalize_data(data)
        if self._N == 0:
            # 1D 数据, 直接求值
            result = evaluate(tree, data_norm)
            return np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

        self._counter = 0

        # Step 0: 剥离外层 cs_rank (resolve 返回值总是截面排名, 外层 cs_rank 冗余)
        is_outer_cs_rank = _is_outer_cs_rank_call(tree)
        if is_outer_cs_rank:
            work_tree = ast.Expression(body=tree.body.args[0])
        else:
            work_tree = tree

        # Step 1: 检测截面函数存在性
        if not _has_any_cs(work_tree):
            # 路径A: 无截面函数 → 逐列求值 + 截面排名
            vals = eval_colwise(work_tree, data_norm, self._T, self._N)
            return cross_sectional_rank(vals)

        # 路径B: 有截面函数 → 单遍 bottom-up 替换
        work_tree_copy = copy.deepcopy(work_tree)
        current_data = dict(data_norm)  # 注入预计算变量的数据集

        transformer = _CSVisitor(self, current_data)
        resolved_tree = transformer.visit(work_tree_copy)
        ast.fix_missing_locations(resolved_tree)

        # 最终: 逐列求值替换后的 AST + 截面排名
        vals = eval_colwise(resolved_tree, current_data, self._T, self._N)
        return cross_sectional_rank(vals)

    def _normalize_data(self, data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """[重构] 2026-06-22 复用 utils/ast normalize_data_keys"""
        result = normalize_data_keys(data)
        for v in result.values():
            arr = np.asarray(v, dtype=float)
            if arr.ndim == 2:
                self._T, self._N = arr.shape
                break
        return result

    def _resolve_node(self, node: ast.Call, data: Dict[str, np.ndarray]) -> ast.Name:
        """处理单个截面函数节点 → 替换为预计算变量名

        调用时机: _CSVisitor 确认 node 的子节点中已无截面函数,
        可以安全地用完整 2D 面板求值。

        步骤:
          1. 逐列求值内部表达式 → (T,N) 因子值
          2. 应用截面函数 (2D 面板上)
          3. 注入为预计算变量
          4. 返回 Name 节点替换原 Call 节点
        """
        func_name = node.func.id
        inner_tree = ast.Expression(body=node.args[0])

        # 逐列求值内部表达式 (时序函数需要 1D)
        inner_vals = eval_colwise(inner_tree, data, self._T, self._N)

        # 应用截面变换 (在完整 2D 面板上)
        if func_name == 'cs_rank':
            result = cross_sectional_rank(inner_vals)
        else:
            # 扩展点: 其他截面函数从 registry 获取实现
            fn = FUNC_REGISTRY.get(func_name)
            if fn is not None:
                result = np.asarray(fn(inner_vals), dtype=float)
            else:
                result = inner_vals

        # 注入预计算变量
        self._counter += 1
        var_name = f'_CS{self._counter}_{id(node) & 0xFFFF:04x}'
        data[var_name] = result

        return ast.Name(id=var_name, ctx=ast.Load())


# ============================================================
# _CSVisitor — AST 访问器 (bottom-up 替换)
# ============================================================

class _CSVisitor(ast.NodeTransformer):
    """Bottom-up 遍历, 在 visit_Call 中替换截面函数节点

    设计保证:
      - generic_visit 先处理子节点, 后检查当前节点
      - 处理外层 cs_rank 时, 内层 cs_rank 已被替换为 Name 节点
      - 安全检查: args 子树中无截面函数 → 可安全求值
    """

    def __init__(self, resolver: CsResolver, data: Dict[str, np.ndarray]):
        self._resolver = resolver
        self._data = data

    def visit_Call(self, node: ast.Call):
        # 1. Bottom-up: 先处理所有子节点
        node = self.generic_visit(node)

        # 2. 检查当前节点是否为截面函数
        if not (isinstance(node.func, ast.Name)
                and _is_cs_function(node.func.id)
                and node.args):
            return node

        # 3. 安全检查: args 子树中是否还有未处理的截面函数
        if _has_any_cs_in_expr(node.args[0]):
            # 仍有嵌套截面函数未处理 → 本轮不处理, 留给下一轮
            # (在单遍 bottom-up 中不会发生, 保留为防御)
            return node

        # 4. 安全: 可以求值替换
        return self._resolver._resolve_node(node, self._data)


# ============================================================
# 辅助函数
# ============================================================

def _is_outer_cs_rank_call(tree: ast.Expression) -> bool:
    """检测最外层是否为 cs_rank 调用"""
    body = tree.body
    return (isinstance(body, ast.Call)
            and isinstance(body.func, ast.Name)
            and body.func.id == 'cs_rank')


def _has_any_cs(tree: ast.Expression) -> bool:
    """检测 AST 中是否包含任意注册的截面函数 (运行时动态查询)"""
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and _is_cs_function(node.func.id)):
            return True
    return False


def _has_any_cs_in_expr(expr_node) -> bool:
    """检测表达式节点子树中是否包含截面函数 (不检查自身, 运行时动态查询)"""
    for node in ast.walk(expr_node):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and _is_cs_function(node.func.id)):
            return True
    return False
