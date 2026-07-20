"""
utils/gp/v5/tree_gen.py — 随机树生成 + 变异算子
=============================================================================
[抽取] 2026-07-06 从 engine.py 拆分。
"""
import ast
import copy
import random
import logging

from utils.ast.v2.dsl import ast_depth, walk_nodes
from utils.ast.v2.spec import make_var, make_const, make_compare, make_boolop, make_ifexp, make_unaryop
from .config import (
    GP_VARIABLES, GP_CONSTANTS,
    TreeGenConfig, get_full_default_weights, _filter_funcs_by_var_scope,
)
from .ast_utils import (
    _collect_replaceable, _replace_subtree, _simplify_ast, _parent_map,
)

logger = logging.getLogger(__name__)


# ============================================================
# 随机树生成
# ============================================================

def _random_variable(cfg: TreeGenConfig) -> ast.Name:
    rng = cfg.rng
    vw = cfg.var_weights
    # [修复] 2026-07-08 var_allowlist 不再与 GP_VARIABLES 取交集.
    # 原实现 list(var_allowlist & set(GP_VARIABLES)) 会过滤掉所有自定义变量
    # (如 REL_CLOSE/MY_VAR), 导致 var_allowlist 完全失效.
    # 新逻辑: 优先与 var_weights keys 取交集 (保持权重一致性),
    #         若无 var_weights 则直接使用 var_allowlist (用户显式白名单已是最严约束).
    # 自定义变量通过 ast.register_variable() 注册合法性 + var_weights 指定生成偏置.
    if cfg.var_allowlist:
        if vw:
            choices = list(cfg.var_allowlist & set(vw.keys()))
        else:
            choices = list(cfg.var_allowlist)
        if choices:
            return make_var(rng.choice(choices))
    if vw:
        var_names = list(vw.keys())
        var_weights = [vw[v] for v in var_names]
        return make_var(rng.choices(var_names, weights=var_weights, k=1)[0])
    return make_var(rng.choice(GP_VARIABLES))


def _random_constant(cfg: TreeGenConfig) -> ast.Constant:
    return make_const(cfg.rng.choice(GP_CONSTANTS))


def _random_terminal(cfg: TreeGenConfig) -> ast.AST:
    if cfg.rng.random() < 0.85:
        return _random_variable(cfg)
    return _random_constant(cfg)


def _get_func_spec(func_name: str):
    """从 FUNC_REGISTRY 惰性查询 FunctionSpec。"""
    from utils.ast.v2.registry import FUNC_REGISTRY
    return FUNC_REGISTRY.get(func_name.lower())


def _random_call(cfg: TreeGenConfig, depth: int, weight_key: str) -> ast.Call:
    """统一函数调用生成器：从 ast FunctionSpec 读取全部参数元数据。

    [重构] 2026-07-08 合并 _random_data_call + _random_math_call。
    所有函数元数据 (data_args / data_vars / param_pool / param_ranges)
    统一从 FUNC_REGISTRY 读取，GP 不再维护参数知识。

    Args:
        weight_key: 'ts_weights' 或 'math_weights'，指定从哪个权重池选函数
    """
    rng = cfg.rng
    weights = getattr(cfg, weight_key, None)
    if weights:
        func_names = list(weights.keys())
        func_weights = [weights[fn] for fn in func_names]
        if cfg.func_allowlist:
            filtered = [(n, w) for n, w in zip(func_names, func_weights) if n in cfg.func_allowlist]
            if filtered:
                func_names, func_weights = zip(*filtered)
        func_name = rng.choices(func_names, weights=func_weights, k=1)[0]
    elif cfg.func_allowlist:
        from utils.ast.v2.registry import FUNC_REGISTRY
        available = [f for f in cfg.func_allowlist if f in FUNC_REGISTRY]
        func_name = rng.choice(available) if available else rng.choice(list(FUNC_REGISTRY.keys()))
    else:
        from utils.ast.v2.registry import FUNC_REGISTRY
        func_name = rng.choice(list(FUNC_REGISTRY.keys()))

    spec = _get_func_spec(func_name)
    args = _build_call_args(cfg, spec, depth)
    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=args, keywords=[],
    )


def _build_call_args(cfg: TreeGenConfig, spec, depth: int) -> list:
    """根据 FunctionSpec 构建函数参数列表。

    [重构] 2026-07-08 提取公共逻辑，供生成和变异复用。
    顺序: data_vars/子树 → param_pool → param_ranges
    """
    rng = cfg.rng
    args = []

    # 1. 数据参数: data_vars 优先 (固定变量), 否则按 data_args 生成子树
    if spec and spec.data_vars:
        for v in spec.data_vars:
            args.append(make_var(v))
    elif spec:
        for _ in range(spec.data_args):
            args.append(_grow_tree(cfg, depth - 1, prefer_variable=True))
    else:
        args.append(_grow_tree(cfg, depth - 1, prefer_variable=True))

    # 2. 离散配置参数: 从 param_pool 抽选
    if spec and spec.param_pool:
        chosen = rng.choice(spec.param_pool)
        if isinstance(chosen, tuple):
            for p in chosen:
                args.append(make_const(p))
        else:
            args.append(make_const(chosen))

    # 3. 连续范围参数: 从 param_ranges 采样
    if spec and spec.param_ranges:
        for pr in spec.param_ranges:
            args.append(make_const(_sample_param_range(pr, rng)))

    return args


def _sample_param_range(pr, rng) -> float:
    """从 ParamRange 采样一个值。"""
    if pr.pool is not None:
        return rng.choice(pr.pool)
    if pr.dtype == 'int':
        lo = int(pr.min_val) if pr.min_val is not None else 1
        hi = int(pr.max_val) if pr.max_val is not None else 100
        return rng.randint(lo, hi)
    lo = pr.min_val if pr.min_val is not None else 0.0
    hi = pr.max_val if pr.max_val is not None else 1.0
    return rng.uniform(lo, hi)


def _random_compare(cfg: TreeGenConfig, left: ast.AST) -> ast.Compare:
    rng = cfg.rng
    threshold = rng.choice([0, 0, 0, 1.0, 1.5, 2.0, -1.0])
    op = rng.choice([ast.Gt, ast.Lt, ast.GtE, ast.LtE])
    return make_compare(left, op, make_const(threshold))


def _mode_filtered_groups(gw: dict, mode: str) -> dict:
    if not mode or mode == 'hybrid':
        return gw
    if mode == 'continuous':
        invalid = {'comparison', 'logic', 'ternary'}
    elif mode == 'predicate':
        invalid = {'ts_function', 'cs_function', 'ta_function', 'feature_function', 'signal_function',
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

    if chosen in ('ts_function', 'cs_function', 'ta_function', 'feature_function', 'signal_function'):
        return _random_call(cfg, depth, 'ts_weights')
    elif chosen == 'math_function':
        return _random_call(cfg, depth, 'math_weights')
    elif chosen == 'comparison':
        return _random_compare(cfg, _grow_tree(cfg, depth - 1, prefer_variable=True))
    elif chosen == 'logic':
        left = _grow_tree(cfg, depth - 1)
        right = _grow_tree(cfg, depth - 1)
        op = ast.And if rng.random() < 0.6 else ast.Or
        return make_boolop(op, left, right)
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
            return make_unaryop(ast.USub, operand)
        elif cfg.mode == 'predicate':
            if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.Not):
                return operand.operand
            return make_unaryop(ast.Not, operand)
        else:
            if rng.random() < 0.5:
                if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.USub):
                    return operand.operand
                return make_unaryop(ast.USub, operand)
            if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.Not):
                return operand.operand
            return make_unaryop(ast.Not, operand)
    else:
        cond = _grow_tree(cfg, depth - 1)
        a_val = _grow_tree(cfg, depth - 1, prefer_variable=True)
        b_val = _grow_tree(cfg, depth - 1, prefer_variable=True)
        return make_ifexp(cond, a_val, b_val)


def _random_tree(cfg: TreeGenConfig, max_depth: int = 6) -> ast.Expression:
    rng = cfg.rng
    depth = rng.randint(2, max(2, max_depth))
    body = _grow_tree(cfg, depth)
    tree = ast.Expression(body=body)
    ast.fix_missing_locations(tree)
    return _simplify_ast(tree)


def _random_tree_explore(user_cfg: TreeGenConfig, max_depth: int = 6) -> ast.Expression:
    """ε-greedy 探索通道: 无视用户权重，用默认全空间生成树（含自定义注册函数）

    [修复] 2026-07-09 传递 var_allowlist / func_allowlist，避免探索通道生成白名单外变量。
    [修复] 2026-07-20 同步应用 _filter_funcs_by_var_scope，防止 data_vars 超限函数从探索通道漏入。
    """
    user_mode = user_cfg.mode if user_cfg else None
    full_defaults = get_full_default_weights()
    ts_w = full_defaults['ts_weights']
    math_w = full_defaults['math_weights']
    allowlist = user_cfg.var_allowlist if user_cfg else None
    if allowlist:
        ts_w, math_w = _filter_funcs_by_var_scope(ts_w, math_w, allowlist)
    explore_cfg = TreeGenConfig(
        mode=user_mode,
        group_weights=full_defaults['group_weights'],
        ts_weights=ts_w,
        math_weights=math_w,
        var_allowlist=allowlist,
        func_allowlist=user_cfg.func_allowlist if user_cfg else None,
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

def _make_explore_config(cfg: TreeGenConfig) -> TreeGenConfig:
    """创建无偏置探索配置：全量函数等权重，仅保留白名单约束。

    [新增] 2026-07-20 P0-2 变异用全空间：打破"变异强化主导方向"的正反馈。
    原 _mutate_subtree 用用户偏置权重(cfg)生成替换子树 → 变异只在已有方向内微调。
    现在用全量等权探索配置 → 变异能引入完全不同的函数/结构，真正起到探索作用。
    """
    from .config import get_full_default_weights
    full_defaults = get_full_default_weights()
    return TreeGenConfig(
        mode=cfg.mode,
        group_weights=full_defaults['group_weights'],
        ts_weights=full_defaults['ts_weights'],
        math_weights=full_defaults['math_weights'],
        var_allowlist=cfg.var_allowlist if cfg else None,
        func_allowlist=cfg.func_allowlist if cfg else None,
        rng=cfg.rng,
    )


def _mutate_subtree(cfg: TreeGenConfig, tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """子树替换变异

    [修改] 2026-07-20 P0-2: 替换子树用无偏置全空间生成，不再使用用户偏置权重。
    原理：如果用户权重偏 ts_rank(×3)，变异也只能生成 ts_rank 子树 →
    种群收敛后变异失去探索能力。用全空间等权生成让变异成为真正的多样性来源。
    """
    rng = cfg.rng
    new_tree = copy.deepcopy(tree)
    candidates = _collect_replaceable(new_tree)
    if not candidates:
        return new_tree
    target = rng.choice(candidates)
    explore_cfg = _make_explore_config(cfg)
    replacement = _grow_tree(explore_cfg, rng.randint(1, max_depth))
    _replace_subtree(new_tree, target, replacement)
    ast.fix_missing_locations(new_tree)
    return _simplify_ast(new_tree)


def _mutate_param(cfg: TreeGenConfig, tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """参数变异: spec 感知，从 FunctionSpec 重新采样参数。

    [重构] 2026-07-08 合并 _mutate_constant + _mutate_window。
    读取 param_pool / param_ranges，在合法范围内重新采样:
    - 窗口参数 (int): 从 param_pool 重新采样，无 pool 则 ±delta
    - 范围参数 (float): 从 param_ranges 范围内重新采样
    - 通用常数: 高斯扰动
    """
    rng = cfg.rng
    new_tree = copy.deepcopy(tree)

    # 收集所有 (常量节点, 所属 Call, 参数索引)
    parents = _parent_map(new_tree)
    candidates = []
    for node in walk_nodes(new_tree):
        # [修复] 2026-07-08 排除 bool (bool 是 int 子类, 不应参与参数变异)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
            parent = parents.get(node)
            if isinstance(parent, ast.Call) and isinstance(parent.func, ast.Name):
                arg_idx = parent.args.index(node)
                candidates.append((node, parent, arg_idx))
            else:
                candidates.append((node, None, -1))

    if not candidates:
        return new_tree

    target, call_node, arg_idx = rng.choice(candidates)

    # 尝试从 spec 读取参数约束
    if call_node is not None:
        spec = _get_func_spec(call_node.func.id)
        if spec:
            # 计算参数布局: [data_args...] [param_pool_args...] [param_ranges_args...]
            n_data = len(spec.data_vars) if spec.data_vars else spec.data_args
            n_pool = 0
            if spec.param_pool:
                sample = spec.param_pool[0]
                n_pool = len(sample) if isinstance(sample, tuple) else 1

            pool_start = n_data
            pool_end = pool_start + n_pool
            constraint_start = pool_end

            # 窗口参数: 从 param_pool 重新采样
            if pool_start <= arg_idx < pool_end:
                if spec.param_pool:
                    chosen = rng.choice(spec.param_pool)
                    if isinstance(chosen, tuple):
                        target.value = chosen[arg_idx - pool_start]
                    else:
                        target.value = chosen
                    return new_tree

            # 范围参数: 从 param_ranges 重新采样
            if arg_idx >= constraint_start and spec.param_ranges:
                ci = arg_idx - constraint_start
                if 0 <= ci < len(spec.param_ranges):
                    target.value = _sample_param_range(spec.param_ranges[ci], rng)
                    return new_tree

    # fallback: 通用常数扰动
    v = target.value
    if isinstance(v, int) and not isinstance(v, bool):
        delta = rng.choice([-5, -2, -1, 1, 2, 5])
        target.value = max(1, v + delta)
    elif isinstance(v, float) and abs(v) < 0.1:
        target.value = rng.choice([0.001, 0.01, 0.02, 0.05])
    else:
        noise = rng.gauss(0, abs(v) * 0.1 + 0.01)
        target.value = round(v + noise, 4)
    return new_tree


def _mutate_logic(cfg: TreeGenConfig, tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """逻辑变异: and↔or, 添加/移除 not"""
    rng = cfg.rng
    new_tree = copy.deepcopy(tree)
    nodes = walk_nodes(new_tree)
    bool_ops = [n for n in nodes if isinstance(n, ast.BoolOp)]
    not_ops = [n for n in nodes
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
            not_node = make_unaryop(ast.Not, copy.deepcopy(target))
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
                condition = make_compare(condition, ast.Gt, make_const(0))
            ifelse = make_ifexp(condition, copy.deepcopy(target), make_const(0))
            _replace_subtree(new_tree, target, ifelse)
    else:
        candidates = [n for n in _collect_replaceable(new_tree, mode='bool')
                      if isinstance(n, (ast.Compare, ast.BoolOp))]
        if candidates:
            target = rng.choice(candidates)
            extra_cond = _grow_tree(cfg, max_depth)
            if not isinstance(extra_cond, (ast.Compare, ast.BoolOp)):
                extra_cond = make_compare(extra_cond, ast.Gt, make_const(0))
            op = ast.And if rng.random() < 0.6 else ast.Or
            combined = make_boolop(op, copy.deepcopy(target), extra_cond)
            _replace_subtree(new_tree, target, combined)

    ast.fix_missing_locations(new_tree)
    return new_tree