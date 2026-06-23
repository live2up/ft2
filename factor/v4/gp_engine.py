"""
factor/v4/gp_engine.py — 因子组合优化 GP 引擎 (Python AST 原生)
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
  >>> from factor.v4.gp_engine import GPEngine
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
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

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
    'ts_regression_residual': [10, 20],
}

# 双参数时序函数（需要两个输入序列）
TS_FUNCTIONS_2ARG = {
    'ts_corr': [10, 20, 60],
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
MATH_FUNCTIONS = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'tanh']

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
    return ast.Name(id=random.choice(GP_VARIABLES), ctx=ast.Load())


def _random_constant() -> ast.Constant:
    return ast.Constant(value=random.choice(GP_CONSTANTS))


def _random_terminal() -> ast.AST:
    if random.random() < 0.85:
        return _random_variable()
    return _random_constant()


def _random_ts_call(depth: int) -> ast.Call:
    func_name = random.choice(list(TS_FUNCTIONS.keys()))
    windows = TS_FUNCTIONS[func_name]
    arg = _grow_tree(depth - 1, prefer_variable=True)
    window = ast.Constant(value=random.choice(windows))
    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=[arg, window], keywords=[],
    )


# [新增] 2026-06-23 数学原语树节点 (单参数, 无窗口)
def _random_math_call(depth: int) -> ast.Call:
    func_name = random.choice(MATH_FUNCTIONS)
    arg = _grow_tree(depth - 1, prefer_variable=True)
    return ast.Call(
        func=ast.Name(id=func_name, ctx=ast.Load()),
        args=[arg], keywords=[],
    )


def _random_feature_call(depth: int) -> ast.Call:
    r = random.random()
    if r < 0.5 and FEATURE_FUNCTIONS_1ARG:
        func_name = random.choice(list(FEATURE_FUNCTIONS_1ARG.keys()))
        windows = FEATURE_FUNCTIONS_1ARG[func_name]
        arg = _grow_tree(depth - 1, prefer_variable=True)
        return ast.Call(
            func=ast.Name(id=func_name, ctx=ast.Load()),
            args=[arg, ast.Constant(value=random.choice(windows))], keywords=[],
        )
    elif r < 0.8 and FEATURE_FUNCTIONS_3ARG:
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


def _grow_tree(depth: int, prefer_variable: bool = False) -> ast.AST:
    """生长随机表达式树

    与 v3 的关键区别：直接生成 Python AST 节点
    - 支持 infix 运算符: +, -, *, /
    - 支持比较: >, <, >=, <=
    - 支持逻辑: and, or
    - 支持三元: a if cond else b
    - 支持 not
    """
    if depth <= 1 or (prefer_variable and random.random() < 0.7):
        return _random_terminal()

    r = random.random()

    if r < 0.25:
        # 时序函数
        return _random_ts_call(depth)
    elif r < 0.38:
        # 特征函数
        return _random_feature_call(depth)
    elif r < 0.50:
        # [新增] 2026-06-23 数学原语: sin, cos, exp, log, sqrt, abs, tanh
        return _random_math_call(depth)
    elif r < 0.63:
        # 比较: expr > 0
        return _random_compare(_grow_tree(depth - 1, prefer_variable=True))
    elif r < 0.74:
        # 逻辑: ... and/or ...
        left = _grow_tree(depth - 1)
        right = _grow_tree(depth - 1)
        op = ast.And() if random.random() < 0.6 else ast.Or()
        return ast.BoolOp(op=op, values=[left, right])
    elif r < 0.86:
        # 二元: a + b, a * b
        left = _grow_tree(depth - 1, prefer_variable=True)
        right = _grow_tree(depth - 1, prefer_variable=True)
        op = random.choice([ast.Add(), ast.Sub(), ast.Mult(), ast.Div()])
        return ast.BinOp(left=left, op=op, right=right)
    elif r < 0.95:
        # 一元: -x, not x
        operand = _grow_tree(depth - 1)
        if random.random() < 0.5:
            return ast.UnaryOp(op=ast.USub(), operand=operand)
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
    return new_tree


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

        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    # ── 适应度评估 ──

    def _evaluate_individual(self, ind: Individual) -> float:
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
            return -999.0  # 常数因子无区分度

        fitness = self.fitness_calc.compute(factor_values)
        if fitness < -998.0:
            return -999.0

        ind.expression_str = _expr_str(ind.tree)
        ind.depth = ast_depth(ind.tree)
        ind.node_count = ast_node_count(ind.tree)
        penalty = self.parsimony_penalty * ind.node_count
        ind.fitness = fitness * (1.0 - penalty)
        return ind.fitness

    def _evaluate_population(self):
        for ind in self.population:
            if ind.fitness == -999.0:
                self._evaluate_individual(ind)

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
            return Individual(tree=child_tree, generation=p1.generation + 1)

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
        """运行 GP 搜索"""
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

            if verbose and gen % 5 == 0:
                logger.info(f"Gen {gen:3d} | best_f={gen_best.fitness:.3f} "
                            f"depth={gen_best.depth} valid={gen_stats['valid_count']}")

            if self.best_individual is None or gen_best.fitness > self.best_individual.fitness:
                self.best_individual = gen_best

            if gen < self.generations - 1:
                self.population = self._next_generation(gen)

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
