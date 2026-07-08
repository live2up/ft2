"""
utils/gp/v5/ast_utils.py — AST 纯函数工具
=============================================================================
[抽取] 2026-07-06 从 engine.py 拆分，纯函数无状态，可被 factor/signals 共享。
[新增] 2026-07-08 新增 _extract_subtrees 用于 Motif 库构建。
"""
import ast
import copy
import logging
from typing import List

logger = logging.getLogger(__name__)

from utils.ast.dsl import ast_depth, ast_node_count


# ============================================================
# AST 工具函数
# ============================================================

def _expr_str(tree: ast.Expression) -> str:
    try:
        return ast.unparse(tree.body)
    except Exception:
        return '<invalid>'


def _collect_replaceable(tree: ast.Expression, mode: str = 'any') -> list:
    """收集可替换的语义子树节点

    自动排除：
    - Load/Store 等元信息节点
    - Call.func 位置的 Name 节点（函数名不是子树）

    mode:
      'any'      — 所有语义节点
      'value'    — 产生数值的子树
      'bool'     — 产生布尔值的子树
    """
    func_names = set()
    for node in ast.walk(tree.body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_names.add(id(node.func))

    if mode == 'any':
        meaningful = (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
                      ast.IfExp, ast.Call, ast.Name, ast.Constant)
    elif mode == 'value':
        meaningful = (ast.BinOp, ast.UnaryOp, ast.Call, ast.Name, ast.Constant, ast.IfExp)
    elif mode == 'bool':
        meaningful = (ast.BoolOp, ast.Compare, ast.UnaryOp, ast.Call)
    else:
        meaningful = (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
                      ast.IfExp, ast.Call, ast.Name, ast.Constant)

    return [n for n in ast.walk(tree.body)
            if isinstance(n, meaningful) and id(n) not in func_names]


def _parent_map(tree: ast.Expression) -> dict:
    parents = {}
    for node in ast.walk(tree.body):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _is_func_name_position(parent: ast.AST, old_node: ast.AST) -> bool:
    """检查 old_node 是否在 Call.func 位置（函数名）"""
    return isinstance(parent, ast.Call) and parent.func is old_node


def _is_int_arg_position(parent: ast.AST, old_node: ast.AST) -> bool:
    """检查 old_node 是否在父节点的整数参数位（如 ts_rank(x, 20) 中的 20）"""
    if not isinstance(parent, ast.Call):
        return False
    for i, arg in enumerate(parent.args):
        if arg is old_node and i > 0:
            if isinstance(old_node, ast.Constant) and isinstance(old_node.value, int):
                return True
    return False


def _replace_subtree(tree: ast.Expression, old_node: ast.AST, new_node: ast.AST) -> bool:
    """将 tree 中的 old_node 替换为 new_node

    安全检查：
    - 不允许替换 Call.func 位置的节点
    - 不允许将非整数子树插入函数的整数参数位
    """
    if tree.body is old_node:
        tree.body = new_node
        return True

    parents = _parent_map(tree)
    if old_node not in parents:
        return False

    parent = parents[old_node]

    if _is_func_name_position(parent, old_node):
        return False

    if _is_int_arg_position(parent, old_node):
        if not (isinstance(new_node, ast.Constant) and isinstance(new_node.value, int)):
            return False

    for field_name, field_value in ast.iter_fields(parent):
        if isinstance(field_value, list):
            for i, item in enumerate(field_value):
                if item is old_node:
                    field_value[i] = new_node
                    return True
        elif field_value is old_node:
            setattr(parent, field_name, new_node)
            return True
    return False


# ============================================================
# AST 轻量简化
# ============================================================

def _simplify_ast(tree: ast.Expression) -> ast.Expression:
    """后处理简化 AST，消除双重否定和恒等运算。

    在随机生成/变异/交叉后调用，节省节点数并提升可读性。
    不改变语义，只做结构简化。
    """
    def _walk(node):
        if node is None:
            return None
        for child in ast.iter_child_nodes(node):
            _walk_child = _walk(child)
            for field_name, field_value in ast.iter_fields(node):
                if isinstance(field_value, list):
                    for i, item in enumerate(field_value):
                        if item is child and _walk_child is not None:
                            field_value[i] = _walk_child
                elif field_value is child and _walk_child is not None:
                    setattr(node, field_name, _walk_child)

        # neg(neg(x)) → x
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.UnaryOp) and isinstance(node.operand.op, ast.USub):
                return node.operand.operand
        # not(not(x)) → x
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            if isinstance(node.operand, ast.UnaryOp) and isinstance(node.operand.op, ast.Not):
                return node.operand.operand
        # cos(neg(x)) → cos(x)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == 'cos':
            if node.args and isinstance(node.args[0], ast.UnaryOp) and isinstance(node.args[0].op, ast.USub):
                node.args[0] = node.args[0].operand
        # x * 1 → x, 1 * x → x
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
            if isinstance(node.right, ast.Constant) and node.right.value == 1:
                return node.left
            if isinstance(node.left, ast.Constant) and node.left.value == 1:
                return node.right
        # x + 0 → x, 0 + x → x
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                return node.left
            if isinstance(node.left, ast.Constant) and node.left.value == 0:
                return node.right
        # x - 0 → x
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            if isinstance(node.right, ast.Constant) and node.right.value == 0:
                return node.left
        # x / 1 → x
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            if isinstance(node.right, ast.Constant) and node.right.value == 1:
                return node.left
        # x - x → 0
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            try:
                if ast.unparse(node.left) == ast.unparse(node.right):
                    return ast.Constant(value=0.0)
            except Exception:
                pass
        # x / x → 1
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
            try:
                if ast.unparse(node.left) == ast.unparse(node.right):
                    return ast.Constant(value=1.0)
            except Exception:
                pass
        return node

    tree = copy.deepcopy(tree)
    tree.body = _walk(tree.body)
    ast.fix_missing_locations(tree)
    return tree


# ============================================================
# AST 规范化（用于缓存 key 语义去重）
# ============================================================

def _canonicalize_key(tree: ast.Expression,
                      expr_str: str = None,
                      memo: dict = None,
                      lock=None) -> str:
    """生成规范化的缓存 key 字符串。

    安全规则:
      - Add/Mult 交换律排序（按子树字符串字典序）
      - 纯常数折叠（1+2→3 等）
      - 不处理非交换函数参数（如 ts_corr(a,b) ≠ ts_corr(b,a)）
    """
    if expr_str is None:
        expr_str = _expr_str(tree)

    if memo is not None:
        if lock is not None:
            with lock:
                if expr_str in memo:
                    return memo[expr_str]
        else:
            if expr_str in memo:
                return memo[expr_str]

    def _canonicalize(node):
        for field_name, field_value in ast.iter_fields(node):
            if isinstance(field_value, list):
                new_list = []
                for item in field_value:
                    if isinstance(item, ast.AST):
                        new_list.append(_canonicalize(item))
                    else:
                        new_list.append(item)
                setattr(node, field_name, new_list)
            elif isinstance(field_value, ast.AST):
                setattr(node, field_name, _canonicalize(field_value))

        # 纯常数折叠
        if isinstance(node, ast.BinOp):
            if isinstance(node.left, ast.Constant) and isinstance(node.right, ast.Constant):
                try:
                    l, r = node.left.value, node.right.value
                    if isinstance(node.op, ast.Add):
                        return ast.Constant(value=l + r)
                    elif isinstance(node.op, ast.Sub):
                        return ast.Constant(value=l - r)
                    elif isinstance(node.op, ast.Mult):
                        return ast.Constant(value=l * r)
                    elif isinstance(node.op, ast.Div) and r != 0:
                        return ast.Constant(value=l / r)
                except Exception:
                    pass

            # 交换律排序：Add/Mult 按子树字符串排序
            if isinstance(node.op, (ast.Add, ast.Mult)):
                left_str = ast.unparse(node.left)
                right_str = ast.unparse(node.right)
                if right_str < left_str:
                    node.left, node.right = node.right, node.left

        return node

    new_tree = copy.deepcopy(tree)
    new_tree.body = _canonicalize(new_tree.body)
    ast.fix_missing_locations(new_tree)
    key = _expr_str(new_tree)

    if memo is not None:
        if lock is not None:
            with lock:
                memo[expr_str] = key
        else:
            memo[expr_str] = key
    return key


def _extract_subtrees(tree: ast.Expression,
                       min_depth: int = 1,
                       max_depth: int = 3) -> List[ast.AST]:
    """提取所有深度在 [min_depth, max_depth] 范围内的有效子树

    [新增] 2026-07-08 用于 Motif 库构建：从高 fitness 个体中提取有潜力的子结构。
    排除：纯函数名节点、Load/Store 元节点、单变量/常数节点。
    """
    func_names = set()
    for node in ast.walk(tree.body):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            func_names.add(id(node.func))

    meaningful = (ast.BinOp, ast.UnaryOp, ast.BoolOp, ast.Compare,
                  ast.IfExp, ast.Call)

    subtrees = []
    for node in ast.walk(tree.body):
        if isinstance(node, meaningful) and id(node) not in func_names:
            d = ast_depth(node)
            if min_depth <= d <= max_depth:
                subtrees.append(node)
    return subtrees