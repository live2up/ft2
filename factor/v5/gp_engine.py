"""
factor/v5/gp_engine.py — 因子组合优化 GP 引擎 (Python AST 原生)
=============================================================================

定位：配合智能体做精细因子组合搜索，而非大规模符号回归。

核心设计：
  1. 种子驱动 — 种群主要来自智能体发现的因子表达式，少量随机注入
  2. Python AST 原生 — 表达能力 = 智能体能写出的任何表达式
  3. 组合优先 — 随机树/变异侧重组合结构(and/or/if-else/比较/加权)
  4. 薄适配层 — 求值用 _ExpressionFromAST(逐列2D), 序列化用 ast.unparse

与 v3 GP 的根本区别：
  - v3: 自研 ASTNode + evaluate_node, 只能生成函数调用链 add(x,y)/gt(x,y)
  - v4: Python ast 模块, 原生支持 x+y / x>y / x and y / a if c else b

用法：
  >>> from factor.v5.gp_engine import GPEngine
  >>> engine = GPEngine(
  ...     data=data_dict,
  ...     fitness_calculator=fitness_calc,
  ...     seed_expressions=[
  ...         'ts_zscore(CLOSE, 20) > 0 and amt_ratio(AMOUNT, 5, 20) > 1',
  ...         'persist(ts_roc(CLOSE, 5) > 0, 2)',
  ...     ],
  ... )
  >>> engine.run()
  >>> best = engine.best()
  >>> print(best.expression_str)

[新增] 2026-06-18 v4 GP 引擎，基于 Python ast 模块
=============================================================================
"""

import ast
import copy
import random
import logging
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, fields as dataclass_fields

from utils.ast.dsl import parse_expression, ast_depth, ast_node_count
from .expression import _ExpressionFromAST

logger = logging.getLogger(__name__)


# ============================================================
# 配置
# ============================================================

GP_VARIABLES = ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'AMOUNT']

GP_CONSTANTS = [0.0, 0.5, 1.0, -1.0, 2.0, 0.01, 0.02, 0.05, 1.5, 3.0]

# 时序函数 + 典型窗口参数
TS_FUNCTIONS = {
    'ts_rank': [5, 10, 20, 60],
    'ts_zscore': [10, 20, 60],
    'ts_mean': [5, 10, 20, 60],
    'ts_std': [10, 20, 60],
    'ts_sum': [5, 10, 20],
    'ts_delta': [1, 5, 10, 20],
    'ts_delay': [1, 5, 10, 20],
    'ts_roc': [5, 10, 20],
    'ts_decay_linear': [5, 10, 20],
    'ts_skew': [20, 60],
    'ts_kurt': [20, 60],
    'ts_resid': [10, 20],
    'ts_slope': [10, 20],
    'ts_rsq': [10, 20],
    'ts_intercept': [10, 20],
    'ts_predict': [10, 20],
}

# 双参数时序函数（需要两个输入序列）— 只能通过种子/交叉传播
TS_FUNCTIONS_2ARG = {
    'ts_corr': [10, 20, 60],
    'ts_cov': [10, 20],
    'ts_reg_slope': [5, 10],
    'ts_reg_resid': [5, 10],
    'ts_reg_rsq': [5, 10],
}

# 特征函数
FEATURE_FUNCTIONS_1ARG = {
    'linearreg': [10, 20],
    'tsf': [10, 20],
}

FEATURE_FUNCTIONS_3ARG = {
    'natr': [5, 14],
    'atr': [14],
}

# 量比/额比函数
RATIO_FUNCTIONS = {
    'amt_ratio': [(5, 20)],
    'vol_ratio': [(5, 20)],
}

# [新增] 2026-06-23 数学原语 (逐元素, 单参数, 无窗口)
#   cos:   周期震荡, 主瓣[0,π]单调, 窄范围等价钟形
#   gauss: exp(-x²), 轻尾钟形, 对极端值惩罚最重, 适合宽范围变量(std>100)
#   p4:    exp(-x⁴), 平顶陡降, 对微小变化不敏感, 适合极窄范围(std~0.01)
#   tanh:  S形, 保留符号方向, 适合有方向性的信号
#   注: cos/gauss/p4 在窄范围(std<1)上等价, 因cs_rank后排序一致
MATH_FUNCTIONS = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'tanh', 'gauss', 'p4', 'neg']

# [新增] 2026-07-03 TreeGenConfig — 树生成概率配置 (WFGP: Weight-Focused GP)
#   三层级偏置: group_weights(大类) → fn_weights(子函数) → var_weights(变量)
#   默认 = 现有等概率行为 (100% 兼容，无需改动调用方)
# [重构] 2026-07-04 增加 mode + var_allowlist/func_allowlist，fill_value 默认改为 0
@dataclass
class TreeGenConfig:
    """树生成概率配置

    控制 GP 随机生树时变量/函数的选择偏置，引导搜索往特定方向聚焦。
    默认 None = 等概率 (完全兼容现有行为)。

    用法:
        # 权重偏置：指定方向权重更高，未指定的权重=0 (不出现)
        cfg = TreeGenConfig(
            var_weights={'AMOUNT': 3, 'VOLUME': 2},
            ts_weights={'ts_corr': 3, 'ts_cov': 3},
        )
        # 白名单：只允许指定变量 (其余=0，完全禁止)
        cfg = TreeGenConfig(var_allowlist={'AMOUNT', 'VOLUME', 'SHARE'})
        # 模式过滤：continuous 禁用 comparison/logic/ternary
        cfg = TreeGenConfig(mode='continuous')

    Attributes:
        mode: 表达式模式 'continuous'/'predicate'/'hybrid' (None=hybrid兼容)
        group_weights: 函数大类权重
        ts_weights:  时序子函数权重 (未指定=0，完全不出现在随机生成中)
        var_weights: 变量权重 (未指定=0)
        var_allowlist: 变量白名单 (None=不限制, set={'AMOUNT','VOLUME'}=只允许这些)
        func_allowlist: 函数白名单 (None=不限制)
    """
    mode: Optional[str] = None           # continuous | predicate | hybrid
    group_weights: Optional[Dict[str, float]] = None
    ts_weights: Optional[Dict[str, float]] = None
    math_weights: Optional[Dict[str, float]] = None
    feature_weights: Optional[Dict[str, float]] = None
    var_weights: Optional[Dict[str, float]] = None
    var_allowlist: Optional[set] = None    # [新增] 变量白名单 (None=不限制)
    func_allowlist: Optional[set] = None   # [新增] 函数白名单

DEFAULT_TREE_GEN_CONFIG = TreeGenConfig(
    group_weights={'ts_function':25, 'feature_function':13, 'math_function':12,
                   'comparison':13, 'logic':11, 'binary_op':12,
                   'unary_op':9, 'ternary':5},
    ts_weights={fn: 1.0 for fn in TS_FUNCTIONS},
    math_weights={fn: 1.0 for fn in MATH_FUNCTIONS},
    feature_weights={'feature_1arg': 0.5, 'feature_3arg': 0.3, 'ratio': 0.2},
)
_TREE_GEN_CONFIG: TreeGenConfig = DEFAULT_TREE_GEN_CONFIG

# [重构] 2026-07-04 权重填充: 用户没设的 key 填 0 (禁止)，用户设了额外 key 保留
# 与旧版区别: fill_value 从 0.1 改为 0 — 0 权重 = 真正不出现在随机生成中
_FILL_VAR_KEYS = GP_VARIABLES[:]
_FILL_TS_KEYS = list(TS_FUNCTIONS.keys())
_FILL_MATH_KEYS = MATH_FUNCTIONS[:]
_FILL_FEATURE_KEYS = list(DEFAULT_TREE_GEN_CONFIG.feature_weights.keys())
_FILL_GROUP_KEYS = list(DEFAULT_TREE_GEN_CONFIG.group_weights.keys())


def _fill_weights(user_weights, default_keys, fill_value=0):
    """填充权重: 用户设了的用用户值，没设的填 fill_value (默认 0=禁止)
    
    [重构] 2026-07-04 fill_value 从 0.1 改为 0: 权重=0 表示完全不出现在随机生成中。
    用户添加额外 key 时保留原值，不受 fill_value 影响。
    """
    filled = {}
    for key in default_keys:
        if user_weights and key in user_weights:
            filled[key] = user_weights[key]
        else:
            filled[key] = fill_value
    # 用户可能加了额外 key (如预计算变量名) — 保留原始值
    if user_weights:
        for key in user_weights:
            if key not in filled:
                filled[key] = user_weights[key]
    # [新增] 过滤掉权重为 0 的 key (完全禁止)，但保留至少一个 key 防全空
    nonzero = {k: v for k, v in filled.items() if v > 0}
    if not nonzero and filled:
        # 全为 0 → 降级为等概率 (全部 1.0)
        nonzero = {k: 1.0 for k in filled}
    return nonzero


DEFAULT_GP_CONFIG = {
    'population_size': 200,
    'generations': 20,
    'max_depth': 10,
    'tournament_size': 5,
    'crossover_prob': 0.6,
    'mutation_prob': 0.25,
    'elite_ratio': 0.05,
    'seed_ratio': 0.4,
    'random_inject_ratio': 0.05,
    'parsimony_penalty': 0.001,
    # 变异算子权重
    'mutate_subtree_weight': 0.30,
    'mutate_constant_weight': 0.20,
    'mutate_window_weight': 0.20,
    'mutate_logic_weight': 0.15,
    'mutate_insert_cond_weight': 0.15,
}


# ============================================================
# Individual
# ============================================================

@dataclass
class Individual:
    """GP 个体"""
    tree: ast.Expression
    fitness: float = -999.0
    expression_str: str = ''
    depth: int = 0
    node_count: int = 0
    generation: int = 0
    is_seed: bool = False

    @staticmethod
    def from_expr(expr_str: str, generation: int = 0, is_seed: bool = False) -> 'Individual':
        """从表达式字符串创建个体"""
        tree = parse_expression(expr_str)
        return Individual(
            tree=tree,
            expression_str=expr_str,
            depth=ast_depth(tree),
            node_count=ast_node_count(tree),
            generation=generation,
            is_seed=is_seed,
        )


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
# ============================================================

def _random_variable() -> ast.Name:
    # [重构] 2026-07-04 支持 var_allowlist (白名单) 和 var_weights (权重偏置)
    cfg = _TREE_GEN_CONFIG
    # allowlist 优先: 只在白名单中等概率抽
    if cfg.var_allowlist:
        choices = list(cfg.var_allowlist & set(GP_VARIABLES))
        if choices:
            return ast.Name(id=random.choice(choices), ctx=ast.Load())
        # allowlist 与 GP_VARIABLES 无交集 → 退化到全变量
    # 否则按权重抽
    vw = cfg.var_weights
    if vw:
        var_names = list(vw.keys())
        var_weights = [vw[v] for v in var_names]
        return ast.Name(id=random.choices(var_names, weights=var_weights, k=1)[0], ctx=ast.Load())
    return ast.Name(id=random.choice(GP_VARIABLES), ctx=ast.Load())


def _random_constant() -> ast.Constant:
    return ast.Constant(value=random.choice(GP_CONSTANTS))


def _random_terminal() -> ast.AST:
    if random.random() < 0.85:
        return _random_variable()
    return _random_constant()


def _random_ts_call(depth: int) -> ast.Call:
    # [重构] 2026-07-04 支持 func_allowlist (白名单) 和 ts_weights (权重偏置)
    # [优化] 2026-07-04 func_allowlist 作为过滤器而非替代品: 先按权重选, 再卡白名单
    cfg = _TREE_GEN_CONFIG
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
        func_name = random.choices(func_names, weights=func_weights, k=1)[0]
    elif cfg.func_allowlist:
        available = [f for f in cfg.func_allowlist if f in TS_FUNCTIONS]
        func_name = random.choice(available) if available else random.choice(list(TS_FUNCTIONS.keys()))
    else:
        func_name = random.choice(list(TS_FUNCTIONS.keys()))
    windows = TS_FUNCTIONS.get(func_name, [10, 20])
    arg = _grow_tree(depth - 1, prefer_variable=True)
    window = ast.Constant(value=random.choice(windows))
    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=[arg, window], keywords=[],
    )


# [新增] 2026-06-23 数学原语树节点 (单参数, 无窗口)
def _random_math_call(depth: int) -> ast.Call:
    # [重构] 2026-07-04 支持 func_allowlist
    # [优化] 2026-07-04 func_allowlist 作为过滤器而非替代品
    cfg = _TREE_GEN_CONFIG
    mw = cfg.math_weights
    if mw:
        func_names = list(mw.keys())
        func_weights = [mw[fn] for fn in func_names]
        if cfg.func_allowlist:
            filtered = [(n, w) for n, w in zip(func_names, func_weights) if n in cfg.func_allowlist]
            if filtered:
                func_names, func_weights = zip(*filtered)
        func_name = random.choices(func_names, weights=func_weights, k=1)[0]
    elif cfg.func_allowlist:
        available = [f for f in cfg.func_allowlist if f in MATH_FUNCTIONS]
        func_name = random.choice(available) if available else random.choice(MATH_FUNCTIONS)
    else:
        func_name = random.choice(MATH_FUNCTIONS)
    arg = _grow_tree(depth - 1, prefer_variable=True)
    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=[arg], keywords=[],
    )


def _random_feature_call(depth: int) -> ast.Call:
    # 特征子组选择 (feature_1arg / feature_3arg / ratio)
    fw = _TREE_GEN_CONFIG.feature_weights
    if fw:
        groups = list(fw.keys())
        weights = [fw[g] for g in groups]
        chosen = random.choices(groups, weights=weights, k=1)[0]
    else:
        r = random.random()
        if r < 0.5:
            chosen = 'feature_1arg'
        elif r < 0.8:
            chosen = 'feature_3arg'
        else:
            chosen = 'ratio'

    if chosen == 'feature_1arg' and FEATURE_FUNCTIONS_1ARG:
        func_name = random.choice(list(FEATURE_FUNCTIONS_1ARG.keys()))
        windows = FEATURE_FUNCTIONS_1ARG[func_name]
        arg = _grow_tree(depth - 1, prefer_variable=True)
        return ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=[arg, ast.Constant(value=random.choice(windows))], keywords=[],
        )
    elif chosen == 'feature_3arg' and FEATURE_FUNCTIONS_3ARG:
        func_name = random.choice(list(FEATURE_FUNCTIONS_3ARG.keys()))
        windows = FEATURE_FUNCTIONS_3ARG[func_name]
        return ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=[ast.Name(id='HIGH', ctx=ast.Load()),
                  ast.Name(id='LOW', ctx=ast.Load()),
                  ast.Name(id='CLOSE', ctx=ast.Load()),
                  ast.Constant(value=random.choice(windows))], keywords=[],
        )
    else:
        func_name = random.choice(list(RATIO_FUNCTIONS.keys()))
        param_pairs = RATIO_FUNCTIONS[func_name]
        short, long = random.choice(param_pairs)
        var = 'AMOUNT' if 'amt' in func_name else 'VOLUME'
        return ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=[ast.Name(id=var, ctx=ast.Load()),
                  ast.Constant(value=short),
                  ast.Constant(value=long)], keywords=[],
        )


def _random_compare(left: ast.AST) -> ast.Compare:
    threshold = random.choice([0, 0, 0, 1.0, 1.5, 2.0, -1.0])
    op = random.choice([ast.Gt(), ast.Lt(), ast.GtE(), ast.LtE()])
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


def _grow_tree(depth: int, prefer_variable: bool = False) -> ast.AST:
    """生长随机表达式树

    与 v3 的关键区别：直接生成 Python AST 节点
    - 支持 infix 运算符: +, -, *, /
    - 支持比较: >, <, >=, <=
    - 支持逻辑: and, or
    - 支持三元: a if cond else b
    - 支持 not

    [重构] 2026-07-04 大类权重支持 mode 过滤
    """
    if depth <= 1 or (prefer_variable and random.random() < 0.7):
        return _random_terminal()

    # 从 TreeGenConfig 读函数大类权重，按 mode 过滤
    cfg = _TREE_GEN_CONFIG
    gw = _mode_filtered_groups(cfg.group_weights, cfg.mode)
    groups = list(gw.keys())
    gweights = [gw[g] for g in groups]
    chosen = random.choices(groups, weights=gweights, k=1)[0]

    if chosen == 'ts_function':
        # 时序函数
        return _random_ts_call(depth)
    elif chosen == 'feature_function':
        # 特征函数
        return _random_feature_call(depth)
    elif chosen == 'math_function':
        # [新增] 2026-06-23 数学原语: sin, cos, exp, log, sqrt, abs, tanh
        return _random_math_call(depth)
    elif chosen == 'comparison':
        # 比较: expr > 0
        return _random_compare(_grow_tree(depth - 1, prefer_variable=True))
    elif chosen == 'logic':
        # 逻辑: ... and/or ...
        left = _grow_tree(depth - 1)
        right = _grow_tree(depth - 1)
        op = ast.And() if random.random() < 0.6 else ast.Or()
        return ast.BoolOp(op=op, values=[left, right])
    elif chosen == 'binary_op':
        # 二元: a + b, a * b
        left = _grow_tree(depth - 1, prefer_variable=True)
        right = _grow_tree(depth - 1, prefer_variable=True)
        op = random.choice([ast.Add(), ast.Sub(), ast.Mult(), ast.Div()])
        return ast.BinOp(left=left, op=op, right=right)
    elif chosen == 'unary_op':
        # 一元: -x (continuous 模式), not x (predicate 模式)
        # [重构] 2026-07-04 continuous 模式下禁用 Not (无意义: not(连续值))
        # [修复] 2026-07-04 消除 neg(neg(x)) → x (双重否定=恒等，省2节点)
        operand = _grow_tree(depth - 1)
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
            if random.random() < 0.5:
                if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.USub):
                    return operand.operand
                return ast.UnaryOp(op=ast.USub(), operand=operand)
            if isinstance(operand, ast.UnaryOp) and isinstance(operand.op, ast.Not):
                return operand.operand
            return ast.UnaryOp(op=ast.Not(), operand=operand)
    else:
        # 三元: a if cond else b
        cond = _grow_tree(depth - 1)
        a_val = _grow_tree(depth - 1, prefer_variable=True)
        b_val = _grow_tree(depth - 1, prefer_variable=True)
        return ast.IfExp(test=cond, body=a_val, orelse=b_val)


def _random_tree(max_depth: int = 6) -> ast.Expression:
    depth = random.randint(2, max_depth)
    body = _grow_tree(depth)
    tree = ast.Expression(body=body)
    ast.fix_missing_locations(tree)
    # [新增] 2026-07-04 随机生成后简化 AST
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
        return node
    
    tree = copy.deepcopy(tree)
    tree.body = _walk(tree.body)
    ast.fix_missing_locations(tree)
    return tree


# ============================================================
# 变异算子
# ============================================================

def _mutate_subtree(tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """子树替换变异"""
    new_tree = copy.deepcopy(tree)
    candidates = _collect_replaceable(new_tree)
    if not candidates:
        return new_tree
    target = random.choice(candidates)
    replacement = _grow_tree(random.randint(1, max_depth))
    _replace_subtree(new_tree, target, replacement)
    ast.fix_missing_locations(new_tree)
    # [新增] 2026-07-04 变异后简化 AST
    return _simplify_ast(new_tree)


def _mutate_constant(tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """常数微调变异"""
    new_tree = copy.deepcopy(tree)
    constants = [n for n in ast.walk(new_tree.body)
                 if isinstance(n, ast.Constant) and isinstance(n.value, (int, float))]
    if not constants:
        return new_tree
    target = random.choice(constants)
    v = target.value
    if isinstance(v, int):
        delta = random.choice([-5, -2, -1, 1, 2, 5])
        target.value = max(1, v + delta)
    else:
        noise = random.gauss(0, abs(v) * 0.1 + 0.01)
        target.value = v + noise
    return new_tree


def _mutate_window(tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """窗口参数变异"""
    new_tree = copy.deepcopy(tree)
    calls = [n for n in ast.walk(new_tree.body)
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
    if not calls:
        return new_tree
    target = random.choice(calls)
    for i, arg in enumerate(target.args):
        if isinstance(arg, ast.Constant) and isinstance(arg.value, int):
            delta = random.choice([-5, -2, -1, 1, 2, 5])
            arg.value = max(2, arg.value + delta)
            return new_tree
    return new_tree


def _mutate_logic(tree: ast.Expression, max_depth: int = 4) -> ast.Expression:
    """逻辑变异: and↔or, 添加/移除 not"""
    new_tree = copy.deepcopy(tree)
    bool_ops = [n for n in ast.walk(new_tree.body) if isinstance(n, ast.BoolOp)]
    not_ops = [n for n in ast.walk(new_tree.body)
               if isinstance(n, ast.UnaryOp) and isinstance(n.op, ast.Not)]

    if bool_ops and random.random() < 0.5:
        target = random.choice(bool_ops)
        target.op = ast.Or() if isinstance(target.op, ast.And) else ast.And()
    elif not_ops and random.random() < 0.6:
        # 移除 not: not(X) → X
        target = random.choice(not_ops)
        _replace_subtree(new_tree, target, target.operand)
        ast.fix_missing_locations(new_tree)
    else:
        # 添加 not: X → not(X)
        candidates = [n for n in _collect_replaceable(new_tree)
                      if isinstance(n, (ast.Compare, ast.BoolOp))]
        if candidates:
            target = random.choice(candidates)
            not_node = ast.UnaryOp(op=ast.Not(), operand=copy.deepcopy(target))
            _replace_subtree(new_tree, target, not_node)
            ast.fix_missing_locations(new_tree)

    return new_tree


def _mutate_insert_condition(tree: ast.Expression, max_depth: int = 3) -> ast.Expression:
    """条件插入变异: 用 if-else 或 and/or 包装子树"""
    new_tree = copy.deepcopy(tree)

    if random.random() < 0.5:
        # expr → expr if condition else 0
        # 只选择产生数值的子树（排除函数名、布尔表达式）
        candidates = [n for n in _collect_replaceable(new_tree, mode='value')
                      if isinstance(n, (ast.BinOp, ast.Call))]
        if candidates:
            target = random.choice(candidates)
            condition = _grow_tree(max_depth)
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
            target = random.choice(candidates)
            extra_cond = _grow_tree(max_depth)
            if not isinstance(extra_cond, (ast.Compare, ast.BoolOp)):
                extra_cond = ast.Compare(
                    left=extra_cond, ops=[ast.Gt()],
                    comparators=[ast.Constant(value=0)])
            op = ast.And() if random.random() < 0.6 else ast.Or()
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
                 # 向后兼容的别名（discover.py 可能传这些参数）
                 future_returns=None, returns=None):
        self.data = data
        self.future_returns = future_returns
        self.returns = returns
        self.fitness_calc = fitness_calculator

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
        self.tree_gen_config = self._build_tree_config(tree_gen_config)

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
        self._fitness_cache: Dict[str, tuple] = {}  # expr_str → (fitness, depth, nodes)
        self._parallel_workers: int = config.get('parallel_workers', 0) if config else 0

        # [新增] 2026-07-05 方向演化追踪 — 按语义签名分组，观察探索路径
        self.direction_log: Dict[str, List[float]] = {}  # signature → [fitness_per_gen]

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    # [重构] 2026-07-04 树配置构建逻辑从 __init__ 抽离
    def _build_tree_config(self, user_config: TreeGenConfig = None) -> TreeGenConfig:
        """构建最终的 TreeGenConfig: 用户配置 + 默认填充 + mode/allowlist 透传"""
        if user_config is None:
            return DEFAULT_TREE_GEN_CONFIG
        filled = {}
        for field in dataclass_fields(TreeGenConfig):
            user_val = getattr(user_config, field.name)
            default_val = getattr(DEFAULT_TREE_GEN_CONFIG, field.name)
            if user_val is None:
                # 用户没设 → 用默认
                # mode/var_allowlist/func_allowlist 的默认值是 None (不做额外处理)
                filled[field.name] = default_val
            elif field.name in ('mode', 'var_allowlist', 'func_allowlist'):
                # [新增] 透传模式和白名单 (不做 fill)
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

    def _evaluate_individual(self, ind: Individual) -> float:
        # [新增] 2026-07-04 表达式缓存: 命中直接返回 (含 metadata)
        key = ind.expression_str or _expr_str(ind.tree)
        if key and key in self._fitness_cache:
            cached_fit, cached_depth, cached_nodes = self._fitness_cache[key]
            ind.fitness = cached_fit
            ind.depth = cached_depth
            ind.node_count = cached_nodes
            ind.expression_str = key
            return cached_fit

        try:
            expr = _ExpressionFromAST(ind.tree)
            factor_values = expr.evaluate(self.data)
        except Exception as e:
            if ind.is_seed:
                logger.warning(f"[GP] 种子求值异常: {e}")
            return -999.0

        if not np.isfinite(factor_values).all():
            return -999.0
        if np.allclose(factor_values, factor_values.flat[0], atol=1e-10):
            return -999.0

        fitness = self.fitness_calc.compute(factor_values)
        if fitness < -998.0:
            return -999.0

        ind.expression_str = key
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
        签名格式: "m:math_funcs|t:ts_funcs|v:vars"
        例: "m:gauss|t:roc|v:REL_AMOUNT" — gauss 变换成交量变化率
        
        同签名表达式 = 同一探索方向，不同签名 = 不同方向
        """
        if not expr_str:
            return "unknown"
        # 提取函数名
        import re
        funcs = set(re.findall(r'([a-z_]+)\(', expr_str))
        math_sig = '+'.join(sorted(f for f in funcs if f in MATH_FUNCTIONS)) or 'none'
        ts_sig = '+'.join(sorted(f for f in funcs if f in TS_FUNCTIONS or f in TS_FUNCTIONS_2ARG)) or 'none'
        # 提取变量（大写标识符）
        vars_sig = '+'.join(sorted(set(re.findall(r'\b([A-Z][A-Z_0-9]+)\b', expr_str)))[:4]) or 'none'
        return f"m:{math_sig}|t:{ts_sig}|v:{vars_sig}"

    def direction_report(self, min_fitness: float = 0.3) -> str:
        """方向探索报告 — 按语义签名分组，展示各方向的最佳和进展
        
        [新增] 2026-07-05
        """
        if not self.direction_log:
            return "无方向记录"
        lines = ["", "=" * 50, "方向探索报告 (按语义签名分组)", "=" * 50]
        # 按签名聚合: 最佳 fitness + 首次发现代
        sig_best = {}
        for sig, fits in self.direction_log.items():
            if fits and max(fits) >= min_fitness:
                sig_best[sig] = {'best': max(fits), 'n': len(fits), 'latest': fits[-1]}
        
        for sig, info in sorted(sig_best.items(), key=lambda x: -x[1]['best']):
            flag = "★" if info['best'] >= 0.8 else " " if info['best'] >= 0.5 else "·"
            lines.append(
                f"  {flag} best={info['best']:.3f}  n={info['n']:3d}  latest={info['latest']:.3f}  {sig}"
            )
        lines.append("=" * 50)
        return '\n'.join(lines)

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
        """
        from concurrent.futures import ThreadPoolExecutor, as_completed

        # 预计算 expression_str
        for ind in individuals:
            if not ind.expression_str:
                ind.expression_str = _expr_str(ind.tree)

        n_workers = min(self._parallel_workers, len(individuals))
        futures_map = {}

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for ind in individuals:
                # 先查缓存
                key = ind.expression_str
                if key and key in self._fitness_cache:
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
        candidates = random.sample(self.population,
                                   min(self.tournament_size, len(self.population)))
        return max(candidates, key=lambda x: x.fitness)

    # ── 交叉 ──

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        """子树交换交叉（带类型兼容性检查）"""
        child_tree = copy.deepcopy(p1.tree)
        donor_tree = p2.tree  # 只读

        # 值子树和布尔子树分开收集
        candidates1_all = _collect_replaceable(child_tree)
        candidates2_all = _collect_replaceable(donor_tree)
        if not candidates1_all or not candidates2_all:
            return Individual(tree=child_tree, generation=p1.generation + 1)

        # 尝试最多3次找到类型兼容的交换对
        for _ in range(3):
            n1 = random.choice(candidates1_all)
            n2 = random.choice(candidates2_all)
            # 类型兼容：同为值类型或同为布尔类型
            n1_is_bool = isinstance(n1, (ast.BoolOp, ast.Compare))
            n2_is_bool = isinstance(n2, (ast.BoolOp, ast.Compare))
            if n1_is_bool != n2_is_bool:
                continue

            # 深度保护
            test_tree = copy.deepcopy(child_tree)
            _replace_subtree(test_tree, n1, copy.deepcopy(n2))
            ast.fix_missing_locations(test_tree)
            if ast_depth(test_tree) > self.max_depth:
                continue

            _replace_subtree(child_tree, n1, copy.deepcopy(n2))
            ast.fix_missing_locations(child_tree)
            # [新增] 2026-07-04 交叉后简化 AST
            return Individual(tree=_simplify_ast(child_tree), generation=p1.generation + 1)

        # 所有尝试都失败，返回父代副本
        return Individual(tree=child_tree, generation=p1.generation + 1)

    # ── 变异 ──

    def _mutate(self, individual: Individual) -> Individual:
        """加权随机选择变异算子"""
        total = sum(w for w, _ in self._mutate_weights)
        r = random.random() * total
        cumulative = 0
        for weight, mutate_fn in self._mutate_weights:
            cumulative += weight
            if r <= cumulative:
                try:
                    new_tree = mutate_fn(individual.tree, max_depth=min(self.max_depth, 4))
                except Exception:
                    new_tree = copy.deepcopy(individual.tree)
                return Individual(tree=new_tree, generation=individual.generation + 1)
        # fallback
        new_tree = _mutate_subtree(individual.tree, max_depth=min(self.max_depth, 4))
        return Individual(tree=new_tree, generation=individual.generation + 1)

    # ── 初始化 ──

    def _initialize_population(self):
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
            base = random.choice(seed_trees)
            mutated = _mutate_subtree(base, max_depth=3)
            self.population.append(Individual(tree=mutated, generation=0))

        # 3. 随机个体
        while len(self.population) < self.population_size:
            tree = _random_tree(self.max_depth)
            self.population.append(Individual(tree=tree, generation=0))

    # ── 进化循环 ──

    def _next_generation(self, gen: int) -> List[Individual]:
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
            r = random.random()
            if r < self.crossover_prob:
                child = self._crossover(self._tournament_select(), self._tournament_select())
            elif r < self.crossover_prob + self.mutation_prob:
                child = self._mutate(self._tournament_select())
            else:
                parent = self._tournament_select()
                child = Individual(tree=copy.deepcopy(parent.tree), generation=gen + 1)
            new_pop.append(child)

        # 随机注入
        for _ in range(self.random_inject_count):
            tree = _random_tree(self.max_depth)
            new_pop.append(Individual(tree=tree, generation=gen + 1))

        return new_pop[:self.population_size]

    def run(self, verbose: bool = True) -> 'GPEngine':
        """运行 GP 搜索
        
        [重构] 2026-07-04 用 _activate_config() 上下文管理器替替代手动 global 操作，
        确保配置正确激活/恢复，避免嵌套 run() 或多实例场景下的污染。
        """
        with self._activate_config():
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

                if verbose and gen % 5 == 0:
                    logger.info(f"Gen {gen:3d} | best_f={gen_best.fitness:.3f} "
                                f"depth={gen_best.depth} valid={gen_stats['valid_count']}")

                if self.best_individual is None or gen_best.fitness > self.best_individual.fitness:
                    self.best_individual = gen_best

                if gen < self.generations - 1:
                    self.population = self._next_generation(gen)
        return self

    def _activate_config(self):
        """上下文管理器: 激活当前实例的 tree_gen_config 到模块全局
        
        [重构] 2026-07-04 封装 global 操作为 context manager
        """
        class _ConfigContext:
            def __enter__(self2):
                global _TREE_GEN_CONFIG
                self2._saved = _TREE_GEN_CONFIG
                new_config = self.tree_gen_config
                if new_config is None:
                    new_config = DEFAULT_TREE_GEN_CONFIG
                _TREE_GEN_CONFIG = new_config
                return self
            
            def __exit__(self2, *args):
                global _TREE_GEN_CONFIG
                _TREE_GEN_CONFIG = self2._saved
                return False
        
        return _ConfigContext()

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
