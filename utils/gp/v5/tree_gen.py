"""
utils/gp/v5/tree_gen.py — 随机树生成 + 变异算子
=============================================================================
[抽取] 2026-07-06 从 engine.py 拆分。
"""
import ast
import copy
import random
import logging

from utils.ast.dsl import ast_depth
from .config import (
    GP_VARIABLES, GP_CONSTANTS,
    TreeGenConfig, get_full_default_weights,
)
from .ast_utils import (
    _collect_replaceable, _replace_subtree, _simplify_ast,
)

logger = logging.getLogger(__name__)


# ============================================================
# 随机树生成
# ============================================================

def _random_variable(cfg: TreeGenConfig) -> ast.Name:
    rng = cfg.rng
    if cfg.var_allowlist:
        choices = list(cfg.var_allowlist & set(GP_VARIABLES))
        if choices:
            return ast.Name(id=rng.choice(choices), ctx=ast.Load())
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


def _get_func_spec(func_name: str):
    """从 FUNC_REGISTRY 惰性查询 FunctionSpec。"""
    from utils.ast.functions import FUNC_REGISTRY
    return FUNC_REGISTRY.get(func_name.lower())


def _random_data_call(cfg: TreeGenConfig, depth: int) -> ast.Call:
    """统一数据函数生成器：读取 FUNC_REGISTRY 的 data_arity / data_vars / param_pool。

    涵盖 ts_function / ta_function / cs_function / feature_function 四个分类，
    从 cfg.ts_weights 选择函数。
    """
    rng = cfg.rng
    tw = cfg.ts_weights
    if tw:
        func_names = list(tw.keys())
        func_weights = [tw[fn] for fn in func_names]
        if cfg.func_allowlist:
            filtered = [(n, w) for n, w in zip(func_names, func_weights) if n in cfg.func_allowlist]
            if filtered:
                func_names, func_weights = zip(*filtered)
        func_name = rng.choices(func_names, weights=func_weights, k=1)[0]
    elif cfg.func_allowlist:
        from utils.ast.functions import FUNC_REGISTRY
        available = [f for f in cfg.func_allowlist if f in FUNC_REGISTRY]
        func_name = rng.choice(available) if available else rng.choice(list(FUNC_REGISTRY.keys()))
    else:
        from utils.ast.functions import FUNC_REGISTRY
        func_name = rng.choice(list(FUNC_REGISTRY.keys()))

    spec = _get_func_spec(func_name)
    args = []

    # 数据参数：data_vars 优先（固定变量如 H,L,C），否则生成子树
    if spec and spec.data_vars:
        for v in spec.data_vars:
            args.append(ast.Name(id=v, ctx=ast.Load()))
    elif spec:
        for _ in range(spec.data_arity):
            args.append(_grow_tree(cfg, depth - 1, prefer_variable=True))
    else:
        args.append(_grow_tree(cfg, depth - 1, prefer_variable=True))

    # 配置参数：从 param_pool 抽选
    if spec and spec.param_pool:
        chosen = rng.choice(spec.param_pool)
        if isinstance(chosen, tuple):
            for p in chosen:
                args.append(ast.Constant(value=p))
        else:
            args.append(ast.Constant(value=chosen))

    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=args, keywords=[],
    )


def _random_math_call(cfg: TreeGenConfig, depth: int) -> ast.Call:
    """统一数学函数生成器：从 cfg.math_weights 选择，按 data_arity 生成子树。"""
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
        from utils.ast.functions import FUNC_REGISTRY
        available = [f for f in cfg.func_allowlist if f in FUNC_REGISTRY]
        func_name = rng.choice(available) if available else rng.choice(list(FUNC_REGISTRY.keys()))
    else:
        from utils.ast.functions import FUNC_REGISTRY
        func_name = rng.choice(list(FUNC_REGISTRY.keys()))

    spec = _get_func_spec(func_name)
    args = []
    if spec:
        for _ in range(spec.data_arity):
            args.append(_grow_tree(cfg, depth - 1, prefer_variable=True))
    else:
        args.append(_grow_tree(cfg, depth - 1, prefer_variable=True))

    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=args, keywords=[],
    )


def _random_compare(cfg: TreeGenConfig, left: ast.AST) -> ast.Compare:
    rng = cfg.rng
    threshold = rng.choice([0, 0, 0, 1.0, 1.5, 2.0, -1.0])
    op = rng.choice([ast.Gt(), ast.Lt(), ast.GtE(), ast.LtE()])
    return ast.Compare(left=left, ops=[op], comparators=[ast.Constant(value=threshold)])


def _mode_filtered_groups(gw: dict, mode: str) -> dict:
    if not mode or mode == 'hybrid':
        return gw
    if mode == 'continuous':
        invalid = {'comparison', 'logic', 'ternary'}
    elif mode == 'predicate':
        invalid = {'ts_function', 'cs_function', 'ta_function', 'feature_function',
                   'math_function', 'binary_op', 'unary_op'}
    else:
        return gw
    return {k: v for k, v in gw.items() if k not in invalid}


def _grow_tree(cfg: TreeGenConfig, depth: int, prefer_variable: bool = False) -> ast.AST:
    rng = cfg.rng
    if depth <= 1 or (prefer_variable and rng.random() < 0.7):
        return _random_terminal(cfg)

    gw = _mode_filtered_groups(cfg.group_weights, cfg.mode)
    groups = list(gw.keys())
    gweights = [gw[g] for g in groups]
    chosen = rng.choices(groups, weights=gweights, k=1)[0]

    if chosen in ('ts_function', 'cs_function', 'ta_function', 'feature_function'):
        return _random_data_call(cfg, depth)
    elif chosen == 'math_function':
        return _random_math_call(cfg, depth)
    elif chosen == 'comparison':
        return _random_compare(cfg, _grow_tree(cfg, depth - 1, prefer_variable=True))
    elif chosen == 'logic':
        left = _grow_tree(cfg, depth - 1)
        right = _grow_tree(cfg, depth - 1)
        op = ast.And() if rng.random() < 0.6 else ast.Or()
        return ast.BoolOp(op=op, values=[left, right])
    elif chosen == 'binary_op':
        left = _grow_tree(cfg, depth - 1, prefer_variable=True)
        right = _grow_tree(cfg, depth - 1, prefer_variable=True)
        op = rng.choice([ast.Add(), ast.Sub(), ast.Mult(), ast.Div()])
        return ast.BinOp(left=left, op=op, right=right)
    elif chosen == 'unary_op':
        operand = _grow_tree(cfg, depth - 1)
        if cfg.mode == 'continuous':
            if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.USub):
                return operand.operand
            return ast.UnaryOp(op=ast.USub(), operand=operand)
        elif cfg.mode == 'predicate':
            if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.Not):
                return operand.operand
            return ast.UnaryOp(op=ast.Not(), operand=operand)
        else:
            if rng.random() < 0.5:
                if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.USub):
                    return operand.operand
                return ast.UnaryOp(op=ast.USub(), operand=operand)
            if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.Not):
                return operand.operand
            return ast.UnaryOp(op=ast.Not(), operand=operand)
    else:
        cond = _grow_tree(cfg, depth - 1)
        a_val = _grow_tree(cfg, depth - 1, prefer_variable=True)
        b_val = _grow_tree(cfg, depth - 1, prefer_variable=True)
        return ast.IfExp(test=cond, body=a_val, orelse=b_val)


def _random_tree(cfg: TreeGenConfig, max_depth: int = 6) -> ast.Expression:
    rng = cfg.rng
    depth = rng.randint(2, max(2, max_depth))
    body = _grow_tree(cfg, depth)
    tree = ast.Expression(body=body)
    ast.fix_missing_locations(tree)
    return _simplify_ast(tree)


def _random_tree_explore(user_cfg: TreeGenConfig, max_depth: int = 6) -> ast.Expression:
    """ε-greedy 探索通道: 无视用户权重，用默认全空间生成树（含自定义注册函数）"""
    user_mode = user_cfg.mode if user_cfg else None
    full_defaults = get_full_default_weights()
    explore_cfg = TreeGenConfig(
        mode=user_mode,
        group_weights=full_defaults['group_weights'],
        ts_weights=full_defaults['ts_weights'],
        math_weights=full_defaults['math_weights'],
        rng=user_cfg.rng,
    )
    rng = explore_cfg.rng
    depth = rng.randint(2, max(2, max_depth))
    body = _grow_tree(explore_cfg, depth)
    tree = ast.Expression(body=body)
    ast.fix_missing_locations(tree)
    return _simplify_ast(tree)


# ============================================================
# 变异算子
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
    return _simplify_ast(new_tree)


def _mutate_constant(cfg: TreeGenConfig, tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """常数微调变异"""
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
        target.value = rng.choice([0.001, 0.01, 0.02, 0.05])
    else:
        noise = rng.gauss(0, abs(v) * 0.1 + 0.01)
        target.value = round(v + noise, 4)
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
        target = rng.choice(not_ops)
        _replace_subtree(new_tree, target, target.operand)
        ast.fix_missing_locations(new_tree)
    else:
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


_MUTATE_OPS = [
    _mutate_subtree,
    _mutate_constant,
    _mutate_window,
    _mutate_logic,
    _mutate_insert_condition,
]