"""
utils/gp/v5/engine.py — 遗传编程引擎核心 (AST 原生)
=============================================================================

[抽取] 2026-07-05 从 factor/v5/gp_engine.py 抽取，供 factor 和 signals 共享。
[重构] 2026-07-05 消除模块级全局变量 _TREE_GEN_CONFIG 和 _CANONICALIZE_MEMO,
       所有随机函数接收 cfg 参数, 支持多实例多线程并行。

"""
import ast
import copy
import random
import logging
import threading
import numpy as np
from typing import Dict, List, Optional, Callable
from dataclasses import fields as dataclass_fields

from utils.ast.dsl import parse_expression, ast_depth, ast_node_count
from .config import (
    GP_VARIABLES, GP_CONSTANTS,
    TS_FUNCTIONS, TS_FUNCTIONS_2ARG,
    FEATURE_FUNCTIONS_1ARG, FEATURE_FUNCTIONS_3ARG,
    RATIO_FUNCTIONS, MATH_FUNCTIONS,
    TreeGenConfig, Individual,
    DEFAULT_TREE_GEN_CONFIG, DEFAULT_GP_CONFIG,
    _FILL_VAR_KEYS, _FILL_TS_KEYS, _FILL_MATH_KEYS,
    _FILL_FEATURE_KEYS, _FILL_GROUP_KEYS, _fill_weights,
)

logger = logging.getLogger(__name__)

# [重构] 2026-07-05 删除模块级全局 _TREE_GEN_CONFIG, 所有随机函数改为接收 cfg 参数
# [重构] 2026-07-05 删除 _CANONICALIZE_MEMO, 改用 expr_str 做 key (见 _canonicalize_key)


# ============================================================
# AST 工具函数
# ============================================================

def _expr_str(tree: ast.Expression) -> str:
    try:
        return ast.unparse(tree.body)
    except Exception:
        return '<invalid>'


def _collect_replaceable(tree: ast.Expression, mode: str = 'any') -> List[ast.AST]:
    """收集可替换的语义子树节点

    自动排除：
    - Load/Store 等元信息节点
    - Call.func 位置的 Name 节点（函数名不是子树）

    mode:
      'any'      — 所有语义节点（用于变异）
      'value'    — 产生数值的子树（BinOp/Call/Name/Constant/IfExp/UnaryOp-USub）
      'bool'     — 产生布尔值的子树（Compare/BoolOp/UnaryOp-Not/Call）
    """
    # 排除函数名位：构建 Call.func 节点集合
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


def _parent_map(tree: ast.Expression) -> Dict[ast.AST, ast.AST]:
    parents = {}
    for node in ast.walk(tree.body):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _is_func_name_position(parent: ast.AST, old_node: ast.AST) -> bool:
    """检查 old_node 是否在 Call.func 位置（函数名）"""
    if isinstance(parent, ast.Call) and parent.func is old_node:
        return True
    return False


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
    - 不允许替换 Call.func 位置的节点（函数名不是子树）
    - 不允许将非整数子树插入函数的整数参数位
    """
    # 根节点替换
    if tree.body is old_node:
        tree.body = new_node
        return True

    parents = _parent_map(tree)
    if old_node not in parents:
        return False

    parent = parents[old_node]

    # 函数名位置不可替换
    if _is_func_name_position(parent, old_node):
        return False

    # 类型安全：整数参数位只接受整数常量
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
# 随机树生成
# [重构] 2026-07-05 所有函数接收 cfg: TreeGenConfig, 用 cfg.rng 替代 random
# ============================================================

def _random_variable(cfg: TreeGenConfig) -> ast.Name:
    # [重构] 2026-07-04 支持 var_allowlist (白名单) 和 var_weights (权重偏置)
    # [重构] 2026-07-05 接收 cfg 参数, 用 cfg.rng
    rng = cfg.rng
    # allowlist 优先: 只在白名单中等概率抽
    if cfg.var_allowlist:
        choices = list(cfg.var_allowlist & set(GP_VARIABLES))
        if choices:
            return ast.Name(id=rng.choice(choices), ctx=ast.Load())
        # allowlist 与 GP_VARIABLES 无交集 → 退化到全变量
    # 否则按权重抽
    vw = cfg.var_weights
    if vw:
        var_names = list(vw.keys())
        var_weights = [vw[v] for v in var_names]
        return ast.Name(id=rng.choices(var_names, weights=var_weights, k=1)[0], ctx=ast.Load())
    return ast.Name(id=rng.choice(GP_VARIABLES), ctx=ast.Load())


def _random_constant(cfg: TreeGenConfig) -> ast.Constant:
    return ast.Constant(value=cfg.rng.choice(GP_CONSTANTS))


def _random_terminal(cfg: TreeGenConfig) -> ast.AST:
    if cfg.rng.random() < 0.85:
        return _random_variable(cfg)
    return _random_constant(cfg)


def _random_ts_call(cfg: TreeGenConfig, depth: int) -> ast.Call:
    # [重构] 2026-07-04 支持 func_allowlist (白名单) 和 ts_weights (权重偏置)
    # [优化] 2026-07-04 func_allowlist 作为过滤器而非替代品: 先按权重选, 再卡白名单
    # [重构] 2026-07-05 接收 cfg 参数, 用 cfg.rng
    rng = cfg.rng
    tw = cfg.ts_weights
    if tw:
        # 权重组 → 选函数
        func_names = list(tw.keys())
        func_weights = [tw[fn] for fn in func_names]
        if cfg.func_allowlist:
            # [优化] 白名单过滤: 权重组 ∩ 白名单
            filtered = [(n, w) for n, w in zip(func_names, func_weights) if n in cfg.func_allowlist]
            if filtered:
                func_names, func_weights = zip(*filtered)
            # 交集为空 → 保留原权重组 (不降级)
        func_name = rng.choices(func_names, weights=func_weights, k=1)[0]
    elif cfg.func_allowlist:
        available = [f for f in cfg.func_allowlist if f in TS_FUNCTIONS]
        func_name = rng.choice(available) if available else rng.choice(list(TS_FUNCTIONS.keys()))
    else:
        func_name = rng.choice(list(TS_FUNCTIONS.keys()))
    windows = TS_FUNCTIONS.get(func_name, [10, 20])
    arg = _grow_tree(cfg, depth - 1, prefer_variable=True)
    window = ast.Constant(value=rng.choice(windows))
    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=[arg, window], keywords=[],
    )


# [新增] 2026-06-23 数学原语树节点 (单参数, 无窗口)
def _random_math_call(cfg: TreeGenConfig, depth: int) -> ast.Call:
    # [重构] 2026-07-04 支持 func_allowlist
    # [优化] 2026-07-04 func_allowlist 作为过滤器而非替代品
    # [重构] 2026-07-05 接收 cfg 参数, 用 cfg.rng
    rng = cfg.rng
    mw = cfg.math_weights
    if mw:
        func_names = list(mw.keys())
        func_weights = [mw[fn] for fn in func_names]
        if cfg.func_allowlist:
            filtered = [(n, w) for n, w in zip(func_names, func_weights) if n in cfg.func_allowlist]
            if filtered:
                func_names, func_weights = zip(*filtered)
        func_name = rng.choices(func_names, weights=func_weights, k=1)[0]
    elif cfg.func_allowlist:
        available = [f for f in cfg.func_allowlist if f in MATH_FUNCTIONS]
        func_name = rng.choice(available) if available else rng.choice(MATH_FUNCTIONS)
    else:
        func_name = rng.choice(MATH_FUNCTIONS)
    arg = _grow_tree(cfg, depth - 1, prefer_variable=True)
    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=[arg], keywords=[],
    )


def _random_feature_call(cfg: TreeGenConfig, depth: int) -> ast.Call:
    # 特征子组选择 (feature_1arg / feature_3arg / ratio)
    # [重构] 2026-07-05 接收 cfg 参数, 用 cfg.rng
    rng = cfg.rng
    fw = cfg.feature_weights
    if fw:
        groups = list(fw.keys())
        weights = [fw[g] for g in groups]
        chosen = rng.choices(groups, weights=weights, k=1)[0]
    else:
        r = rng.random()
        if r < 0.5:
            chosen = 'feature_1arg'
        elif r < 0.8:
            chosen = 'feature_3arg'
        else:
            chosen = 'ratio'

    if chosen == 'feature_1arg' and FEATURE_FUNCTIONS_1ARG:
        func_name = rng.choice(list(FEATURE_FUNCTIONS_1ARG.keys()))
        windows = FEATURE_FUNCTIONS_1ARG[func_name]
        arg = _grow_tree(cfg, depth - 1, prefer_variable=True)
        return ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=[arg, ast.Constant(value=rng.choice(windows))], keywords=[],
        )
    elif chosen == 'feature_3arg' and FEATURE_FUNCTIONS_3ARG:
        func_name = rng.choice(list(FEATURE_FUNCTIONS_3ARG.keys()))
        windows = FEATURE_FUNCTIONS_3ARG[func_name]
        return ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=[ast.Name(id='HIGH', ctx=ast.Load()),
                  ast.Name(id='LOW', ctx=ast.Load()),
                  ast.Name(id='CLOSE', ctx=ast.Load()),
                  ast.Constant(value=rng.choice(windows))], keywords=[],
        )
    else:
        func_name = rng.choice(list(RATIO_FUNCTIONS.keys()))
        param_pairs = RATIO_FUNCTIONS[func_name]
        short, long = rng.choice(param_pairs)
        var = 'AMOUNT' if 'amt' in func_name else 'VOLUME'
        return ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=[ast.Name(id=var, ctx=ast.Load()),
                  ast.Constant(value=short),
                  ast.Constant(value=long)], keywords=[],
        )


def _random_compare(cfg: TreeGenConfig, left: ast.AST) -> ast.Compare:
    rng = cfg.rng
    threshold = rng.choice([0, 0, 0, 1.0, 1.5, 2.0, -1.0])
    op = rng.choice([ast.Gt(), ast.Lt(), ast.GtE(), ast.LtE()])
    return ast.Compare(left=left, ops=[op], comparators=[ast.Constant(value=threshold)])


def _mode_filtered_groups(gw: dict, mode: str) -> dict:
    """按 mode 过滤函数大类

    [新增] 2026-07-04
    continuous: 禁用 comparison/logic/ternary/unary_op(Not) — 只生成连续值表达式
    predicate:  禁用 ts_function/feature_function/math_function — 只生成布尔表达式
    hybrid/None: 全开放
    """
    if not mode or mode == 'hybrid':
        return gw
    if mode == 'continuous':
        # 连续值模式: 禁用所有产出布尔值的节点类型
        invalid = {'comparison', 'logic', 'ternary'}
    elif mode == 'predicate':
        # 布尔模式: 禁用所有产出连续值的节点类型
        invalid = {'ts_function', 'feature_function', 'math_function', 'binary_op', 'unary_op'}
    else:
        return gw
    return {k: v for k, v in gw.items() if k not in invalid}


def _grow_tree(cfg: TreeGenConfig, depth: int, prefer_variable: bool = False) -> ast.AST:
    """生长随机表达式树

    与 v3 的关键区别：直接生成 Python AST 节点
    - 支持 infix 运算符: +, -, *, /
    - 支持比较: >, <, >=, <=
    - 支持逻辑: and, or
    - 支持三元: a if cond else b
    - 支持 not

    [重构] 2026-07-04 大类权重支持 mode 过滤
    [重构] 2026-07-05 接收 cfg 参数, 用 cfg.rng
    """
    rng = cfg.rng
    if depth <= 1 or (prefer_variable and rng.random() < 0.7):
        return _random_terminal(cfg)

    # 从 cfg 读函数大类权重，按 mode 过滤
    gw = _mode_filtered_groups(cfg.group_weights, cfg.mode)
    groups = list(gw.keys())
    gweights = [gw[g] for g in groups]
    chosen = rng.choices(groups, weights=gweights, k=1)[0]

    if chosen == 'ts_function':
        # 时序函数
        return _random_ts_call(cfg, depth)
    elif chosen == 'feature_function':
        # 特征函数
        return _random_feature_call(cfg, depth)
    elif chosen == 'math_function':
        # [新增] 2026-06-23 数学原语: sin, cos, exp, log, sqrt, abs, tanh
        return _random_math_call(cfg, depth)
    elif chosen == 'comparison':
        # 比较: expr > 0
        return _random_compare(cfg, _grow_tree(cfg, depth - 1, prefer_variable=True))
    elif chosen == 'logic':
        # 逻辑: ... and/or ...
        left = _grow_tree(cfg, depth - 1)
        right = _grow_tree(cfg, depth - 1)
        op = ast.And() if rng.random() < 0.6 else ast.Or()
        return ast.BoolOp(op=op, values=[left, right])
    elif chosen == 'binary_op':
        # 二元: a + b, a * b
        left = _grow_tree(cfg, depth - 1, prefer_variable=True)
        right = _grow_tree(cfg, depth - 1, prefer_variable=True)
        op = rng.choice([ast.Add(), ast.Sub(), ast.Mult(), ast.Div()])
        return ast.BinOp(left=left, op=op, right=right)
    elif chosen == 'unary_op':
        # 一元: -x (continuous 模式), not x (predicate 模式)
        # [重构] 2026-07-04 continuous 模式下禁用 Not (无意义: not(连续值))
        # [修复] 2026-07-04 消除 neg(neg(x)) → x (双重否定=恒等，省2节点)
        operand = _grow_tree(cfg, depth - 1)
        if cfg.mode == 'continuous':
            if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.USub):
                return operand.operand  # neg(neg(x)) → x
            return ast.UnaryOp(op=ast.USub(), operand=operand)
        elif cfg.mode == 'predicate':
            if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.Not):
                return operand.operand  # not(not(x)) → x
            return ast.UnaryOp(op=ast.Not(), operand=operand)
        else:
            # hybrid: 各 50%
            if rng.random() < 0.5:
                if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.USub):
                    return operand.operand
                return ast.UnaryOp(op=ast.USub(), operand=operand)
            if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.Not):
                return operand.operand
            return ast.UnaryOp(op=ast.Not(), operand=operand)
    else:
        # 三元: a if cond else b
        cond = _grow_tree(cfg, depth - 1)
        a_val = _grow_tree(cfg, depth - 1, prefer_variable=True)
        b_val = _grow_tree(cfg, depth - 1, prefer_variable=True)
        return ast.IfExp(test=cond, body=a_val, orelse=b_val)


def _random_tree(cfg: TreeGenConfig, max_depth: int = 6) -> ast.Expression:
    rng = cfg.rng
    # [修复] 2026-07-06 P1-6: max_depth<2 时 randint(2, max_depth) 崩溃, 加下限保护
    depth = rng.randint(2, max(2, max_depth))
    body = _grow_tree(cfg, depth)
    tree = ast.Expression(body=body)
    ast.fix_missing_locations(tree)
    # [新增] 2026-07-04 随机生成后简化 AST
    return _simplify_ast(tree)


def _random_tree_explore(user_cfg: TreeGenConfig, max_depth: int = 6) -> ast.Expression:
    """ε-greedy 探索通道: 无视用户权重，用默认全空间生成树

    [新增] 2026-07-05 P0: 用等概率权重但继承用户的 mode。
    保持权重全部开放，但保留类型安全（continuous不含if）。
    [重构] 2026-07-05 不再切换全局变量, 改为构建临时 explore_cfg, 共享 user_cfg.rng
    """
    user_mode = user_cfg.mode if user_cfg else None
    explore_cfg = TreeGenConfig(
        mode=user_mode,
        group_weights=DEFAULT_TREE_GEN_CONFIG.group_weights,
        ts_weights=DEFAULT_TREE_GEN_CONFIG.ts_weights,
        math_weights=DEFAULT_TREE_GEN_CONFIG.math_weights,
        feature_weights=DEFAULT_TREE_GEN_CONFIG.feature_weights,
        rng=user_cfg.rng,  # 共享 rng 保持可复现性
    )
    rng = explore_cfg.rng
    # [修复] 2026-07-06 P1-6: max_depth<2 时 randint(2, max_depth) 崩溃, 加下限保护
    depth = rng.randint(2, max(2, max_depth))
    body = _grow_tree(explore_cfg, depth)
    tree = ast.Expression(body=body)
    ast.fix_missing_locations(tree)
    return _simplify_ast(tree)


# [新增] 2026-07-04 AST 轻量简化: 消除 neg(neg(x))/x*1/x+0 等冗余
def _simplify_ast(tree: ast.Expression) -> ast.Expression:
    """后处理简化 AST，消除双重否定和恒等运算。

    在随机生成/变异/交叉后调用，节省节点数并提升可读性。
    不改变语义，只做结构简化。
    """
    def _walk(node):
        if node is None:
            return None
        # 递归处理子节点
        for child in ast.iter_child_nodes(node):
            _walk_child = _walk(child)
            # 替换子节点
            for field_name, field_value in ast.iter_fields(node):
                if isinstance(field_value, list):
                    for i, item in enumerate(field_value):
                        if item is child and _walk_child is not None:  # noqa: E711
                            field_value[i] = _walk_child
                elif field_value is child and _walk_child is not None:  # noqa: E711
                    setattr(node, field_name, _walk_child)

        # neg(neg(x)) → x
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            if isinstance(node.operand, ast.UnaryOp) and isinstance(node.operand.op, ast.USub):
                return node.operand.operand
        # not(not(x)) → x
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            if isinstance(node.operand, ast.UnaryOp) and isinstance(node.operand.op, ast.Not):
                return node.operand.operand
        # cos(neg(x)) → cos(x) (cos 是偶函数)
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
        # x - x → 0 (自减恒为0)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Sub):
            try:
                if ast.unparse(node.left) == ast.unparse(node.right):
                    return ast.Constant(value=0.0)
            except Exception:
                pass
        # x / x → 1 (自除恒为1)
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


# [新增] 2026-07-05 AST 规范化：用于缓存 key 语义去重
# 仅做不改变数值语义的安全变换：交换律排序、纯常数折叠
# 注意：规范化后的表达式仅用于缓存匹配，报告展示仍用原始 expression_str
# [重构] 2026-07-05 去掉 _CANONICALIZE_MEMO (id-based 不安全), 改用 expr_str 做 memo
# [重构] 2026-07-06 P1-1: memo 从模块级移到实例级 (GPEngine._canonicalize_memo),
#       并加 threading.Lock 保护多线程并发写, 避免跨实例污染和 check-then-set 竞态

def _canonicalize_key(tree: ast.Expression,
                      expr_str: str = None,
                      memo: Dict[str, str] = None,
                      lock=None) -> str:
    """生成规范化的缓存 key 字符串。

    [重构] 2026-07-05 直接返回字符串 key, 不再返回 AST 副本 (避免 deepcopy 开销)
    [重构] 2026-07-06 P1-1: memo + lock 由调用方传入, 实例级隔离
    安全规则:
      - Add/Mult 交换律排序（按子树字符串字典序）
      - 纯常数折叠（1+2→3 等）
      - 不处理非交换函数参数（如 ts_corr(a,b) ≠ ts_corr(b,a)）
    """
    if expr_str is None:
        expr_str = _expr_str(tree)

    # 加锁的 check-then-set, 避免多线程并发规范化
    if memo is not None:
        if lock is not None:
            with lock:
                if expr_str in memo:
                    return memo[expr_str]
        else:
            if expr_str in memo:
                return memo[expr_str]

    def _canonicalize(node):
        # 自底向上递归处理子节点
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


# ============================================================
# 变异算子
# [重构] 2026-07-05 所有函数接收 cfg: TreeGenConfig, 用 cfg.rng
# ============================================================

def _mutate_subtree(cfg: TreeGenConfig, tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """子树替换变异"""
    rng = cfg.rng
    new_tree = copy.deepcopy(tree)
    candidates = _collect_replaceable(new_tree)
    if not candidates:
        return new_tree
    target = rng.choice(candidates)
    replacement = _grow_tree(cfg, rng.randint(1, max_depth))
    _replace_subtree(new_tree, target, replacement)
    ast.fix_missing_locations(new_tree)
    # [新增] 2026-07-04 变异后简化 AST
    return _simplify_ast(new_tree)


def _mutate_constant(cfg: TreeGenConfig, tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """常数微调变异 — 小常量(<0.1)仅做离散跳变，防过拟合精度"""
    rng = cfg.rng
    new_tree = copy.deepcopy(tree)
    constants = [n for n in ast.walk(new_tree.body)
                 if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]
    if not constants:
        return new_tree
    target = rng.choice(constants)
    v = target.value
    if isinstance(v, int):
        delta = rng.choice([-5, -2, -1, 1, 2, 5])
        target.value = max(1, v + delta)
    elif abs(v) < 0.1:
        # ε常量/除法稳定项 → 离散跳变到标准值，不加噪声
        target.value = rng.choice([0.001, 0.01, 0.02, 0.05])
    else:
        noise = rng.gauss(0, abs(v) * 0.1 + 0.01)
        target.value = round(v + noise, 4)  # [修复] 2026-07-05 保留4位精度
    return new_tree


def _mutate_window(cfg: TreeGenConfig, tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """窗口参数变异"""
    rng = cfg.rng
    new_tree = copy.deepcopy(tree)
    calls = [n for n in ast.walk(new_tree.body)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    if not calls:
        return new_tree
    target = rng.choice(calls)
    for i, arg in enumerate(target.args):
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            delta = rng.choice([-5, -2, -1, 1, 2, 5])
            arg.value = max(2, arg.value + delta)
            return new_tree
    return new_tree


def _mutate_logic(cfg: TreeGenConfig, tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """逻辑变异: and↔or, 添加/移除 not"""
    rng = cfg.rng
    new_tree = copy.deepcopy(tree)
    bool_ops = [n for n in ast.walk(new_tree.body) if isinstance(n, ast.BoolOp)]
    not_ops = [n for n in ast.walk(new_tree.body)
               if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)]

    if bool_ops and rng.random() < 0.5:
        target = rng.choice(bool_ops)
        target.op = ast.Or() if isinstance(target.op, ast.And) else ast.And()
    elif not_ops and rng.random() < 0.6:
        # 移除 not: not(X) → X
        target = rng.choice(not_ops)
        _replace_subtree(new_tree, target, target.operand)
        ast.fix_missing_locations(new_tree)
    else:
        # 添加 not: X → not(X)
        candidates = [n for n in _collect_replaceable(new_tree)
                      if isinstance(n, (ast.Compare, ast.BoolOp))]
        if candidates:
            target = rng.choice(candidates)
            not_node = ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(target))
            _replace_subtree(new_tree, target, not_node)
            ast.fix_missing_locations(new_tree)

    return new_tree


def _mutate_insert_condition(cfg: TreeGenConfig, tree: ast.Expression, max_depth: int = 3) -> ast.Expression:
    """条件插入变异: 用 if-else 或 and/or 包装子树"""
    rng = cfg.rng
    new_tree = copy.deepcopy(tree)

    if rng.random() < 0.5:
        # expr → expr if condition else 0
        # 只选择产生数值的子树（排除函数名、布尔表达式）
        candidates = [n for n in _collect_replaceable(new_tree, mode='value')
                      if isinstance(n, (ast.BinOp, ast.Call))]
        if candidates:
            target = rng.choice(candidates)
            condition = _grow_tree(cfg, max_depth)
            if not isinstance(condition, (ast.Compare, ast.BoolOp)):
                condition = ast.Compare(
                    left=condition, ops=[ast.Gt()],
                    comparators=[ast.Constant(value=0)])
            ifelse = ast.IfExp(
                test=condition,
                body=copy.deepcopy(target),
                orelse=ast.Constant(value=0))
            _replace_subtree(new_tree, target, ifelse)
    else:
        # expr → expr and/or extra_condition
        candidates = [n for n in _collect_replaceable(new_tree, mode='bool')
                      if isinstance(n, (ast.Compare, ast.BoolOp))]
        if candidates:
            target = rng.choice(candidates)
            extra_cond = _grow_tree(cfg, max_depth)
            if not isinstance(extra_cond, (ast.Compare, ast.BoolOp)):
                extra_cond = ast.Compare(
                    left=extra_cond, ops=[ast.Gt()],
                    comparators=[ast.Constant(value=0)])
            op = ast.And() if rng.random() < 0.6 else ast.Or()
            combined = ast.BoolOp(op=op, values=[copy.deepcopy(target), extra_cond])
            _replace_subtree(new_tree, target, combined)

    ast.fix_missing_locations(new_tree)
    return new_tree


# 注意: _MUTATE_OPS 列表保留模块级引用, 但调用时需要传 cfg
# GPEngine._mutate 中通过 self._mutate_weights 索引并传 cfg
_MUTATE_OPS = [
    _mutate_subtree,
    _mutate_constant,
    _mutate_window,
    _mutate_logic,
    _mutate_insert_condition,
]


# ============================================================
# GPEngine
# ============================================================

class GPEngine:
    """因子组合优化 GP 引擎

    定位：配合智能体做精细因子组合搜索。
    - 种子驱动：种群主要来自智能体发现的因子表达式
    - 组合优先：搜索最优的组合方式(and/or/if-else/加权/条件门控)
    - Python AST 原生：表达能力 = 智能体能写出的任何表达式
    """

    def __init__(self, data: Dict[str, np.ndarray],
                 fitness_calculator=None,
                 config: Dict = None,
                 seed_expressions: List[str] = None,
                 random_seed: int = None,
                 # [新增] 2026-07-03 三层级加权偏置引导
                 tree_gen_config: TreeGenConfig = None,
                 # [新增] 2026-07-05 求值函数注入 (因子/信号各自注入)
                 evaluator: Optional[Callable] = None,
                 # 向后兼容的别名
                 future_returns=None, returns=None):
        self.data = data
        self.future_returns = future_returns
        self.returns = returns
        self.fitness_calc = fitness_calculator
        # [新增] evaluator: (data, tree) → values, 因子端用 _ExpressionFromAST
        self._evaluator = evaluator

        # 配置
        cfg = dict(DEFAULT_GP_CONFIG)
        if config:
            cfg.update(config)
        self.population_size = cfg['population_size']
        self.generations = cfg['generations']
        self.max_depth = cfg['max_depth']
        self.tournament_size = cfg['tournament_size']
        self.crossover_prob = cfg['crossover_prob']
        self.mutation_prob = cfg['mutation_prob']
        self.elite_count = max(1, int(self.population_size * cfg['elite_ratio']))
        self.seed_ratio = cfg['seed_ratio']
        self.random_inject_count = max(1, int(self.population_size * cfg['random_inject_ratio']))
        self._save_random_inject: int = self.random_inject_count  # [新增] 2026-07-05 用于停滞恢复
        self.parsimony_penalty = cfg['parsimony_penalty']

        # 变异算子权重
        w = cfg
        self._mutate_weights = [
            (w['mutate_subtree_weight'], _mutate_subtree),
            (w['mutate_constant_weight'], _mutate_constant),
            (w['mutate_window_weight'], _mutate_window),
            (w['mutate_logic_weight'], _mutate_logic),
            (w['mutate_insert_cond_weight'], _mutate_insert_condition),
        ]

        # 种子
        self.seed_expressions = seed_expressions or []

        # [重构] 2026-07-04 树生成概率配置构建
        # 用户设了的 key 保留，没设的填默认值；权重 fill_value=0 (禁止)
        # [重构] 2026-07-05 注入独立 rng, 支持多实例并行 + 可复现性
        rng = random.Random(random_seed) if random_seed is not None else random.Random()
        self.tree_gen_config = self._build_tree_config(tree_gen_config, rng)

        # 数据形状
        self._shape = None
        for arr in data.values():
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                self._shape = arr.shape
                break
        if self._shape is None:
            self._shape = (1, 1)

        # 状态
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []

        # [新增] 2026-07-04 表达式缓存 + 多进程
        self._fitness_cache: Dict[str, tuple] = {}  # canonical_key → (fitness, depth, nodes)
        self._parallel_workers: int = config.get('parallel_workers', 0) if config else 0
        # [新增] 2026-07-06 P1-1: AST 规范化 memo 移到实例级 + 加锁
        # 之前是模块级 _CANONICALIZE_STR_MEMO, 多实例多线程并发写会污染
        self._canonicalize_memo: Dict[str, str] = {}
        self._canonicalize_lock = threading.Lock()

        # [新增] 2026-07-05 SQLite 持久化缓存 — 跨 run 复用
        self._cache_db: str = config.get('cache_db', '') if config else ''
        self._fitness_hash: str = ''
        if self._cache_db:
            import hashlib, sqlite3
            # 配置指纹: 数据形状 + 参数
            fingerprint = f"{self._shape}_{self.parsimony_penalty:.4f}"
            self._fitness_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:12]
            self._init_sqlite_cache()

        # [新增] 2026-07-05 方向演化追踪 — 按语义签名分组，观察探索路径
        self.direction_log: Dict[str, List[float]] = {}  # signature → [fitness_per_gen]
        self._direction_best_expr: Dict[str, tuple] = {}  # signature → (best_fitness, expr_str)
        self._direction_snapshot: int = 0  # [已弃用] 全局快照偏移, 见 _direction_per_sig_snapshot
        # [修复] 2026-07-06 P1-2: 之前用全局偏移套到每个签名上, 新签名的记录永远取不到。
        # 改为按签名独立记录偏移。
        self._direction_per_sig_snapshot: Dict[str, int] = {}

        # [新增] 2026-07-05 P0 停滞检测 (AW-MEP 风格)
        self._stagnation_counter: int = 0
        self._last_best_fitness: float = -999.0
        self._save_crossover_prob: float = self.crossover_prob
        self._save_mutation_prob: float = self.mutation_prob
        self._stagnation_threshold: int = 3  # 连续N代无改善触发

        # [新增] 2026-07-05 P0 ε-greedy 探索通道
        self._explore_ratio: float = config.get('explore_ratio', 0.15) if config else 0.15

        # [新增] 2026-07-05 搜索优化: Lexicase 选择
        self._use_lexicase: bool = config.get('lexicase', False) if config else False

        # [重构] 2026-07-05 不再 seed 全局 random/np.random, 改用实例 rng
        # [修复] 2026-07-06 P1-5: 之前仍调用 np.random.seed() 设全局状态,
        # 多实例多线程下会互相覆盖, 不可复现。完全移除, 由 fitness_calc 自行管理随机数
        # (调用方应传入 np.random.Generator 而非依赖全局状态)。
        # 如 fitness_calc 用到 numpy 随机, 调用方需在 fitness_calc 内部用 default_rng(seed)。

    # [新增] 2026-07-05 SQLite 持久化缓存 — 跨 run 复用
    def _init_sqlite_cache(self):
        """创建缓存表和加载已有记录到内存"""
        import sqlite3
        conn = sqlite3.connect(self._cache_db)
        conn.execute('''CREATE TABLE IF NOT EXISTS expressions (
            expr_hash TEXT PRIMARY KEY,
            expression TEXT NOT NULL,
            fitness REAL, depth INTEGER, nodes INTEGER,
            fitness_hash TEXT, created TEXT DEFAULT (datetime('now'))
        )''')
        conn.execute('CREATE INDEX IF NOT EXISTS idx_expr_hash ON expressions(fitness_hash)')
        conn.commit()
        conn.close()

    def _load_sqlite_cache(self):
        """从 SQLite 加载当前配置指纹的缓存到内存"""
        if not self._cache_db:
            return
        import sqlite3
        conn = sqlite3.connect(self._cache_db)
        rows = conn.execute(
            'SELECT expression, fitness, depth, nodes FROM expressions WHERE fitness_hash=?',
            (self._fitness_hash,)
        ).fetchall()
        conn.close()
        for expr, fit, dep, nod in rows:
            self._fitness_cache[expr] = (fit, dep, nod)
        return len(rows)

    def _save_sqlite_cache(self):
        """将内存缓存增量写入 SQLite (INSERT OR REPLACE)"""
        if not self._cache_db:
            return
        import sqlite3
        conn = sqlite3.connect(self._cache_db)
        for expr, (fit, dep, nod) in self._fitness_cache.items():
            import hashlib
            eh = hashlib.md5(expr.encode()).hexdigest()[:16]
            conn.execute(
                'INSERT OR REPLACE INTO expressions(expr_hash, expression, fitness, depth, nodes, fitness_hash) VALUES(?,?,?,?,?,?)',
                (eh, expr, fit, dep, nod, self._fitness_hash)
            )
        conn.commit()
        conn.close()

    # [重构] 2026-07-04 树配置构建逻辑从 __init__ 抽离
    # [重构] 2026-07-05 接收 rng 参数, 注入到最终 config
    def _build_tree_config(self, user_config: TreeGenConfig = None,
                           rng: random.Random = None) -> TreeGenConfig:
        """构建最终的 TreeGenConfig: 用户配置 + 默认填充 + mode/allowlist 透传 + rng 注入"""
        if user_config is None:
            return TreeGenConfig(
                group_weights=DEFAULT_TREE_GEN_CONFIG.group_weights,
                ts_weights=DEFAULT_TREE_GEN_CONFIG.ts_weights,
                math_weights=DEFAULT_TREE_GEN_CONFIG.math_weights,
                feature_weights=DEFAULT_TREE_GEN_CONFIG.feature_weights,
                rng=rng or random.Random(),
            )
        filled = {}
        for field in dataclass_fields(TreeGenConfig):
            if field.name == 'rng':
                # [新增] rng 注入: 优先用外部传入的 (来自 GPEngine 实例)
                filled[field.name] = rng or user_config.rng or random.Random()
                continue
            user_val = getattr(user_config, field.name)
            default_val = getattr(DEFAULT_TREE_GEN_CONFIG, field.name)
            if user_val is None:
                # 用户没设 → 用默认
                # mode/var_allowlist/func_allowlist 的默认值是 None (不做额外处理)
                filled[field.name] = default_val
            elif field.name in ('mode', 'var_allowlist', 'func_allowlist',
                                 'adaptive', 'adaptive_lr', 'adaptive_every'):
                # [新增] 透传模式和自适应参数 (不做 fill)
                filled[field.name] = user_val
            else:
                # 权重字段 → 自动填充
                key_set = {
                    'group_weights': _FILL_GROUP_KEYS,
                    'ts_weights': _FILL_TS_KEYS,
                    'math_weights': _FILL_MATH_KEYS,
                    'feature_weights': _FILL_FEATURE_KEYS,
                    'var_weights': _FILL_VAR_KEYS,
                }.get(field.name, [])
                filled[field.name] = _fill_weights(user_val, key_set)
        return TreeGenConfig(**filled)

    # ── 适应度评估 ──

    def _quick_filter(self, ind: Individual) -> bool:
        """[新增] 预筛: 快速拒绝明显无效的表达式，省去昂贵回测
        
        Reject:
          - window=1 时序函数 (纯噪声)
          - 裸变量无函数 (在截面中无数学变换=无区分力)
          - 纯常数
          - 自指运算: x-x=0, x/x=1 (被_simplify_ast消除，但防变异逃逸)
        """
        expr = ind.expression_str or _expr_str(ind.tree)
        import re
        # window=1: ts_roc(x, 1) / ts_delta(x, 1) etc.
        if re.search(r'ts_\w+\([^,]+, 1\)', expr):
            return False
        # 纯常数: 0.05, 1.0/R, etc.
        vars_found = set(re.findall(r'\b[A-Z][A-Z_0-9]+\b', expr))
        if not vars_found:
            return False
        # 裸变量(无函数): CLOSE, HIGH → 无数学变换，截面区分力弱
        funcs_found = set(re.findall(r'([a-z_]+)\(', expr))
        if not funcs_found and len(vars_found) <= 2:
            return False
        # ts 函数输入常数: ts_roc(0.5, 20) → 无时序变化，纯浪费
        if re.search(r'ts_\w+\(-?\d+\.?\d*,', expr):
            return False
        return True

    def _evaluate_individual(self, ind: Individual) -> float:
        # [新增] 2026-07-04 表达式缓存: 命中直接返回 (含 metadata)
        # [重构] 2026-07-05 用 _canonicalize_key 生成规范化字符串 key, 实现语义去重
        #        保留原始 expression_str 用于报告展示
        if not ind.expression_str:
            ind.expression_str = _expr_str(ind.tree)
        key = _canonicalize_key(ind.tree, ind.expression_str,
                                self._canonicalize_memo, self._canonicalize_lock)

        if key in self._fitness_cache:
            cached_fit, cached_depth, cached_nodes = self._fitness_cache[key]
            ind.fitness = cached_fit
            ind.depth = cached_depth
            ind.node_count = cached_nodes
            return cached_fit

        # [新增] 预筛: 跳过明显无效的表达式
        if not self._quick_filter(ind):
            return -999.0

        try:
            if self._evaluator:
                factor_values = self._evaluator(self.data, ind.tree)
            else:
                return -999.0  # 未注入 evaluator
        except Exception as e:
            if ind.is_seed:
                logger.warning(f"[GP] 种子求值异常: {e}")
            return -999.0

        if not np.isfinite(factor_values).all():
            return -999.0
        if np.allclose(factor_values, factor_values.flat[0], atol=1e-10):
            return -999.0

        fitness = self.fitness_calc.compute(factor_values)
        # [修复] 2026-07-06 P1-4: nan/inf 通过 `< -998.0` 检查 (nan<x 恒为 False),
        # 会污染选择和缓存。改用 isfinite 拦截 nan/inf。
        if not np.isfinite(fitness) or fitness < -998.0:
            return -999.0

        ind.depth = ast_depth(ind.tree)
        ind.node_count = ast_node_count(ind.tree)
        penalty = self.parsimony_penalty * ind.node_count
        ind.fitness = fitness * (1.0 - penalty)

        # [新增] 存缓存 (含 metadata: depth + node_count)
        self._fitness_cache[key] = (ind.fitness, ind.depth, ind.node_count)
        return ind.fitness

    @staticmethod
    def _expr_signature(expr_str: str) -> str:
        """从表达式提取语义签名，用于方向追踪

        [新增] 2026-07-05
        [修复] 2026-07-05 变量按出现频率排序取前4, 而非字母序 (更能代表表达式特征)
        签名格式: "m:math_funcs|t:ts_funcs|v:vars"
        例: "m:gauss|t:roc|v:REL_AMOUNT" — gauss 变换成交量变化率

        同签名表达式 = 同一探索方向，不同签名 = 不同方向
        """
        if not expr_str:
            return "unknown"
        # 提取函数名
        import re
        from collections import Counter
        funcs = set(re.findall(r'([a-z_]+)\(', expr_str))
        math_sig = '+'.join(sorted(f for f in funcs if f in MATH_FUNCTIONS)) or 'none'
        ts_sig = '+'.join(sorted(f for f in funcs if f in TS_FUNCTIONS or f in TS_FUNCTIONS_2ARG)) or 'none'
        # [修复] 变量按出现频率排序取前4 (原 sorted(set(...))[:4] 是字母序, 不代表重要性)
        var_counts = Counter(re.findall(r'\b([A-Z][A-Z_0-9]+)\b', expr_str))
        vars_sig = '+'.join(v for v, _ in var_counts.most_common(4)) or 'none'
        return f"m:{math_sig}|t:{ts_sig}|v:{vars_sig}"

    def direction_report(self, min_fitness: float = 0.3) -> str:
        """方向探索报告 — 按语义签名分组，展示各方向的最佳表达式
        
        [新增] 2026-07-05
        [增强] 2026-07-05 每方向附加最佳表达式，可直接评估
        """
        if not self.direction_log:
            return "无方向记录"
        lines = ["", "=" * 50, "方向探索报告 (★=SR≥0.8  =0.5+ ·=0.3+)", "=" * 50]
        sig_best = {}
        for sig, fits in self.direction_log.items():
            if fits and max(fits) >= min_fitness:
                sig_best[sig] = {'best': max(fits), 'n': len(fits), 'latest': fits[-1]}
        
        for sig, info in sorted(sig_best.items(), key=lambda x: -x[1]['best']):
            flag = "★" if info['best'] >= 0.8 else " " if info['best'] >= 0.5 else "·"
            best_expr = ""
            if sig in self._direction_best_expr:
                _, best_expr = self._direction_best_expr[sig]
            expr_short = best_expr[:70] if best_expr else ""
            lines.append(
                f"  {flag} best={info['best']:.3f}  n={info['n']:3d}  {sig}"
            )
            if expr_short:
                lines.append(f"       └─ {expr_short}")
        lines.append("=" * 50)
        return '\n'.join(lines)

    def _update_direction_weights(self):
        """EMA 闭环方向权重更新 — 扫描 direction_log 自动提权好方向

        [新增] 2026-07-05 P0: 从方向监控数据中学习，让好方向的 var/ts/math 权重自动提升。
        EMA 公式: new_weight = (1-lr)*old + lr*(fitness/max_fitness)
        [修复] 2026-07-05 EMA 更新后做归一化, 防止权重发散或归零

        只处理自上次 snapshot 以来的新记录，避免每代重复计算。
        """
        cfg = self.tree_gen_config
        if not cfg or not cfg.adaptive:
            return

        # 收集自上次更新以来的新方向记录
        # [修复] 2026-07-06 P1-2: 改为按签名独立偏移, 新签名也能被处理
        new_records: Dict[str, List[float]] = {}
        total_entries = 0
        for sig, fits in self.direction_log.items():
            offset = self._direction_per_sig_snapshot.get(sig, 0)
            new_fits = fits[offset:]
            if new_fits:
                new_records[sig] = new_fits
                total_entries += len(new_fits)
            self._direction_per_sig_snapshot[sig] = len(fits)

        if total_entries < 10:
            return  # 样本太少，不够更新

        # 计算每个方向的分值 (用最佳 fitness，加权出现频率)
        sig_scores = {}
        for sig, fits in new_records.items():
            if not fits:
                continue
            best = max(fits)
            # 评分 = 最佳 × log(1+出现次数) — 兼顾质量和稳定性
            sig_scores[sig] = best * np.log1p(len(fits))

        if not sig_scores:
            return

        max_score = max(sig_scores.values())
        if max_score <= 0.1:
            return

        # 提取各签名中的变量/函数，按分值得分聚合权重
        var_scores: Dict[str, float] = {}
        ts_scores: Dict[str, float] = {}
        math_scores: Dict[str, float] = {}

        for sig, score in sig_scores.items():
            norm_score = score / max_score  # 归一化到 [0, 1]

            # 解析签名: "m:gauss|t:ts_roc|v:REL_AMOUNT"
            parts = {}
            for p in sig.split('|'):
                if ':' in p:
                    k, v = p.split(':', 1)
                    parts[k] = v.split('+') if v != 'none' else []

            for v in parts.get('v', []):
                var_scores[v] = var_scores.get(v, 0) + norm_score
            for f in parts.get('t', []):
                ts_scores[f] = ts_scores.get(f, 0) + norm_score
            for f in parts.get('m', []):
                math_scores[f] = math_scores.get(f, 0) + norm_score

        lr = cfg.adaptive_lr

        # EMA 更新 var_weights
        var_w = cfg.var_weights
        if var_w and var_scores:
            for v in var_w:
                old = var_w.get(v, 0)
                score = var_scores.get(v, 0)
                var_w[v] = (1 - lr) * old + lr * score * 3.0  # 放大使得 3.0 区间
            # [修复] 2026-07-05 归一化 + 最小权重保护, 防止方向被完全关闭
            self._normalize_weights(var_w, min_w=0.05)

        # EMA 更新 ts_weights
        ts_w = cfg.ts_weights
        if ts_w and ts_scores:
            for f in ts_w:
                old = ts_w.get(f, 0)
                score = ts_scores.get(f, 0)
                ts_w[f] = (1 - lr) * old + lr * score * 3.0
            self._normalize_weights(ts_w, min_w=0.05)

        # EMA 更新 math_weights
        mw = cfg.math_weights
        if mw and math_scores:
            for f in mw:
                old = mw.get(f, 0)
                score = math_scores.get(f, 0)
                mw[f] = (1 - lr) * old + lr * score * 3.0
            self._normalize_weights(mw, min_w=0.05)

    @staticmethod
    def _normalize_weights(weights: Dict[str, float], min_w: float = 0.05):
        """[新增] 2026-07-05 权重归一化: 除以总和 + 最小值保护

        防止 EMA 长期运行后某些权重发散或归零, 保持探索多样性。
        """
        if not weights:
            return
        total = sum(weights.values())
        if total <= 0:
            # 全零 → 等概率
            for k in weights:
                weights[k] = 1.0 / len(weights)
            return
        for k in weights:
            weights[k] = max(min_w, weights[k] / total)

    def _adapt_operators(self, gen_best_fitness: float):
        """AW-MEP 风格停滞检测 + 算子自适应

        [新增] 2026-07-05 P0: 连续 stagnation_threshold 代无改善 → 加大变异、降低交叉。
        [修复] 2026-07-05 改善后恢复 random_inject_count 到初始值 (原仅恢复 cross/mut)
        """
        if gen_best_fitness > self._last_best_fitness + 1e-4:
            # 有改善 → 重置计数 + 恢复默认
            self._stagnation_counter = 0
            self._last_best_fitness = gen_best_fitness
            self.crossover_prob = self._save_crossover_prob
            self.mutation_prob = self._save_mutation_prob
            self.random_inject_count = self._save_random_inject  # [修复] 恢复随机注入
        else:
            self._stagnation_counter += 1
            if self._stagnation_counter >= self._stagnation_threshold:
                # 停滞 → 加力破局
                self.mutation_prob = min(0.5, self.mutation_prob * 1.3)
                self.crossover_prob = max(0.2, self.crossover_prob * 0.85)
                if self._stagnation_counter >= 5:
                    # 严重停滞 → 随机注入翻倍
                    self.random_inject_count = min(
                        int(self.population_size * 0.3),
                        self.random_inject_count * 2
                    )

    def _evaluate_population(self):
        # [重构] 2026-07-04 多进程并行求值 + 缓存
        unevaluated = [ind for ind in self.population if ind.fitness == -999.0]
        if not unevaluated:
            return

        if self._parallel_workers > 1 and len(unevaluated) >= 10:
            self._evaluate_parallel(unevaluated)
        else:
            for ind in unevaluated:
                self._evaluate_individual(ind)

    def _evaluate_parallel(self, individuals: List[Individual]):
        """多线程并行求值 — ThreadPoolExecutor 避免 pickle 序列化问题

        [重构] 2026-07-04 从 multiprocessing 改为 ThreadPoolExecutor:
        子进程需要 pickle fitness_calc，对 __main__ 定义的类不可行。
        线程共享内存，numpy 操作释放 GIL，实测加速比接近进程方案。
        [重构] 2026-07-05 用 _canonicalize_key 生成缓存 key
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 预计算 expression_str（原始可读）和规范化缓存 key
        for ind in individuals:
            if not ind.expression_str:
                ind.expression_str = _expr_str(ind.tree)

        n_workers = min(self._parallel_workers, len(individuals))
        futures_map = {}

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for ind in individuals:
                # 先查缓存（用规范化 key，实现语义去重）
                key = _canonicalize_key(ind.tree, ind.expression_str,
                                self._canonicalize_memo, self._canonicalize_lock)
                if key in self._fitness_cache:
                    ind.fitness, ind.depth, ind.node_count = self._fitness_cache[key]
                    continue
                futures_map[executor.submit(
                    self._evaluate_individual, ind
                )] = ind

            for future in as_completed(futures_map):
                try:
                    future.result()
                except Exception as e:
                    logger.debug(f"[GP] 并行求值异常: {e}")

    # ── 选择 ──

    def _tournament_select(self) -> Individual:
        # [重构] 2026-07-05 用实例 rng 替代全局 random
        rng = self.tree_gen_config.rng
        candidates = rng.sample(self.population,
                                min(self.tournament_size, len(self.population)))
        return max(candidates, key=lambda x: x.fitness)

    def _lexicase_select(self) -> Individual:
        """方向多样性选择 — 按语义方向选父母

        [新增] 2026-07-05
        [修复] 2026-07-06 P1-3: 之前 for sig in cases 循环里 sig 未被使用, 只是反复
        用整体 fitness 做 20% 容差过滤, 等价于普通锦标赛, 没有方向多样性。
        改为: 从每个签名的 best_expr 反查对应个体, 组成"方向代表池", 随机选一个。
        这样真正实现了"按方向多样性选父母"。
        如果不开 Lexicase 或方向记录太少，退回锦标赛。
        """
        rng = self.tree_gen_config.rng
        valid = [i for i in self.population if i.fitness > -999 and i.expression_str]
        if len(valid) < self.tournament_size:
            return self._tournament_select()

        if len(self._direction_best_expr) < 2:
            return self._tournament_select()

        # 从每个方向的最佳表达式反查对应个体 (按 expression_str 匹配)
        valid_by_expr = {i.expression_str: i for i in valid if i.expression_str}
        representatives = []
        for sig, (best_fit, best_expr) in self._direction_best_expr.items():
            ind = valid_by_expr.get(best_expr)
            if ind is not None:
                representatives.append(ind)

        if not representatives:
            return self._tournament_select()

        # 加权选择: fitness 越高被选概率越大, 但保留方向多样性
        # 用 softmax-like 权重, 避免某方向垄断
        weights = []
        for ind in representatives:
            w = max(ind.fitness, 0.01)  # 避免 0/负权重
            weights.append(w)
        total_w = sum(weights)
        if total_w <= 0:
            return rng.choice(representatives)
        probs = [w / total_w for w in weights]
        return rng.choices(representatives, weights=probs, k=1)[0]

    def _select_parent(self) -> Individual:
        """统一选择器 — Lexicase 或锦标赛
        
        [新增] 2026-07-05
        """
        if self._use_lexicase:
            return self._lexicase_select()
        return self._tournament_select()

    # ── 交叉 ──

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        """子树交换交叉（带类型兼容性检查 + 深度保护）

        [修复] 2026-07-06 P0-1: 之前 test_tree=deepcopy(child_tree) 后用 n1
        (来自原 child_tree) 调 _replace_subtree, n1 不在 test_tree 中, 永远返回 False,
        深度保护完全失效。改为直接在 child_tree 上替换, 超深则回滚到 backup。
        同时每次循环重新收集 candidates, 避免回滚后 n1 失效。
        """
        rng = self.tree_gen_config.rng
        child_tree = copy.deepcopy(p1.tree)
        donor_tree = p2.tree  # 只读

        candidates2_all = _collect_replaceable(donor_tree)
        if not candidates2_all:
            return Individual(tree=child_tree, generation=p1.generation + 1)

        # 尝试最多3次找到类型兼容的交换对
        for _ in range(3):
            # 每次循环重新收集 child_tree 的可替换节点 (回滚后节点对象已变)
            candidates1_all = _collect_replaceable(child_tree)
            if not candidates1_all:
                break
            n1 = rng.choice(candidates1_all)
            n2 = rng.choice(candidates2_all)
            # 类型兼容：同为值类型或同为布尔类型
            n1_is_bool = isinstance(n1, (ast.BoolOp, ast.Compare))
            n2_is_bool = isinstance(n2, (ast.BoolOp, ast.Compare))
            if n1_is_bool != n2_is_bool:
                continue

            backup = copy.deepcopy(child_tree)
            replaced = _replace_subtree(child_tree, n1, copy.deepcopy(n2))
            if not replaced:
                child_tree = backup
                continue
            ast.fix_missing_locations(child_tree)
            if ast_depth(child_tree) > self.max_depth:
                child_tree = backup  # 越深, 回滚
                continue
            # [新增] 2026-07-04 交叉后简化 AST
            return Individual(tree=_simplify_ast(child_tree), generation=p1.generation + 1)

        # 所有尝试都失败，返回父代副本
        return Individual(tree=child_tree, generation=p1.generation + 1)

    # ── 变异 ──

    def _mutate(self, individual: Individual) -> Individual:
        """加权随机选择变异算子

        [修复] 2026-07-06 P1-7: 之前变异后不校验产物深度, 与 P0-1 叠加导致深度无界增长。
        现在统一在变异后校验, 越深则放弃变异返回原个体副本。
        """
        cfg = self.tree_gen_config
        rng = cfg.rng
        total = sum(w for w, _ in self._mutate_weights)
        r = rng.random() * total
        cumulative = 0
        for weight, mutate_fn in self._mutate_weights:
            cumulative += weight
            if r <= cumulative:
                try:
                    new_tree = mutate_fn(cfg, individual.tree, max_depth=min(self.max_depth, 4))
                except Exception:
                    new_tree = copy.deepcopy(individual.tree)
                # [修复] P1-7: 校验产物深度, 越深则放弃
                if ast_depth(new_tree) > self.max_depth:
                    new_tree = copy.deepcopy(individual.tree)
                return Individual(tree=new_tree, generation=individual.generation + 1)
        # fallback
        new_tree = _mutate_subtree(cfg, individual.tree, max_depth=min(self.max_depth, 4))
        if ast_depth(new_tree) > self.max_depth:
            new_tree = copy.deepcopy(individual.tree)
        return Individual(tree=new_tree, generation=individual.generation + 1)

    # ── 初始化 ──

    def _initialize_population(self):
        cfg = self.tree_gen_config
        rng = cfg.rng
        self.population = []

        # 1. 种子个体
        seed_count = int(self.population_size * self.seed_ratio)
        for expr_str in self.seed_expressions[:seed_count]:
            try:
                ind = Individual.from_expr(expr_str, generation=0, is_seed=True)
                self.population.append(ind)
            except Exception as e:
                logger.warning(f"种子解析失败: {expr_str[:60]} ({e})")

        # 2. 种子变异体（扩展种子邻域）
        seed_trees = [ind.tree for ind in self.population if ind.is_seed]
        while len(self.population) < seed_count and seed_trees:
            base = rng.choice(seed_trees)
            mutated = _mutate_subtree(cfg, base, max_depth=3)
            self.population.append(Individual(tree=mutated, generation=0))

        # 3. 随机个体 (ε-greedy: 部分用全空间探索)
        remaining = self.population_size - len(self.population)
        explore_n = int(remaining * self._explore_ratio)
        while len(self.population) < self.population_size:
            if len(self.population) < seed_count + explore_n:
                tree = _random_tree_explore(cfg, self.max_depth)  # 探索通道
            else:
                tree = _random_tree(cfg, self.max_depth)
            self.population.append(Individual(tree=tree, generation=0))

    # ── 进化循环 ──

    def _next_generation(self, gen: int) -> List[Individual]:
        cfg = self.tree_gen_config
        rng = cfg.rng
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_pop = []

        # 精英保留
        for i in range(min(self.elite_count, len(sorted_pop))):
            elite = sorted_pop[i]
            new_pop.append(Individual(
                tree=copy.deepcopy(elite.tree),
                fitness=elite.fitness,
                expression_str=elite.expression_str,
                generation=gen + 1,
                depth=elite.depth, node_count=elite.node_count,
            ))

        # 遗传操作
        while len(new_pop) < self.population_size - self.random_inject_count:
            r = rng.random()
            if r < self.crossover_prob:
                child = self._crossover(self._select_parent(), self._select_parent())
            elif r < self.crossover_prob + self.mutation_prob:
                child = self._mutate(self._select_parent())
            else:
                parent = self._select_parent()
                child = Individual(tree=copy.deepcopy(parent.tree), generation=gen + 1)
            new_pop.append(child)

        # 随机注入 (ε-greedy: 部分用全空间探索)
        inject_n = self.random_inject_count
        explore_n = min(inject_n, int(inject_n * self._explore_ratio * 2))  # 探索占探索比率×2
        for i in range(inject_n):
            if i < explore_n:
                tree = _random_tree_explore(cfg, self.max_depth)
            else:
                tree = _random_tree(cfg, self.max_depth)
            new_pop.append(Individual(tree=tree, generation=gen + 1))

        return new_pop[:self.population_size]

    def run(self, verbose: bool = True) -> 'GPEngine':
        """运行 GP 搜索

        [重构] 2026-07-04 用 _activate_config() 上下文管理器替替代手动 global 操作，
        确保配置正确激活/恢复，避免嵌套 run() 或多实例场景下的污染。
        [重构] 2026-07-05 删除 _activate_config, 不再需要全局变量切换
        """
        # [新增] 2026-07-05 从 SQLite 加载历史缓存
        n_loaded = self._load_sqlite_cache()
        if n_loaded and verbose:
            logger.info(f"[GP] 加载 SQLite 缓存 {n_loaded} 条 (fitness_hash={self._fitness_hash})")

        self._initialize_population()
        for gen in range(self.generations):
            self._evaluate_population()
            gen_best = max(self.population, key=lambda x: x.fitness)

            valid = [i for i in self.population if i.fitness > -999]
            gen_stats = {
                'generation': gen,
                'best_fitness': gen_best.fitness,
                'best_depth': gen_best.depth,
                'best_expression': gen_best.expression_str,
                'avg_fitness': np.mean([i.fitness for i in valid]) if valid else -999,
                'valid_count': len(valid),
            }
            self.history.append(gen_stats)

            # [新增] 2026-07-05 方向追踪: 记录当前种群中各语义方向的最优 fitness
            for ind in valid:
                if ind.fitness > -999:
                    sig = self._expr_signature(ind.expression_str or _expr_str(ind.tree))
                    if sig not in self.direction_log:
                        self.direction_log[sig] = []
                    self.direction_log[sig].append(ind.fitness)
                    # 记录该方向的最佳表达式
                    prev = self._direction_best_expr.get(sig, (-999, ''))
                    if ind.fitness > prev[0]:
                        self._direction_best_expr[sig] = (ind.fitness, ind.expression_str or _expr_str(ind.tree))

            # [新增] 2026-07-05 P0: 闭环方向权重 EMA 更新
            cfg = self.tree_gen_config
            if cfg and cfg.adaptive and gen > 0 and gen % cfg.adaptive_every == 0:
                self._update_direction_weights()

            # [新增] 2026-07-05 P0: 停滞检测 + 算子自适应 (AW-MEP 风格)
            self._adapt_operators(gen_best.fitness)

            if verbose and gen % 5 == 0:
                logger.info(f"Gen {gen:3d} | best_f={gen_best.fitness:.3f} "
                            f"depth={gen_best.depth} valid={gen_stats['valid_count']}")

            if self.best_individual is None or gen_best.fitness > self.best_individual.fitness:
                self.best_individual = gen_best

            if gen < self.generations - 1:
                self.population = self._next_generation(gen)
        # [新增] 2026-07-05 保存缓存到 SQLite
        self._save_sqlite_cache()
        return self

    def best(self) -> Optional[Individual]:
        return self.best_individual

    def top(self, n: int = 10) -> List[Individual]:
        return sorted([i for i in self.population if i.fitness > -999],
                      key=lambda x: x.fitness, reverse=True)[:n]

    def report(self) -> str:
        if not self.history:
            return "尚未运行 GP 搜索"
        lines = ["=" * 60, "因子组合优化 GP 搜索报告", "=" * 60,
                 f"种群: {self.population_size}  代数: {self.generations}  "
                 f"最大深度: {self.max_depth}  种子: {len(self.seed_expressions)}", ""]
        if self.best_individual:
            b = self.best_individual
            lines += [f"最优: fitness={b.fitness:.3f}, depth={b.depth}, nodes={b.node_count}",
                      f"表达式: {b.expression_str}", ""]
        lines += [f"{'Gen':>4}  {'Best_F':>8}  {'Depth':>6}  {'Valid':>6}",
                  "-" * 35]
        for s in self.history:
            if s['generation'] % 5 == 0 or s['generation'] == self.history[-1]['generation']:
                lines.append(f"{s['generation']:4d}  {s['best_fitness']:8.3f}  "
                             f"{s['best_depth']:6d}  {s['valid_count']:6d}")
        return '\n'.join(lines)
