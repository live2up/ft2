"""
signals/v4/search/gp.py — GP 遗传算法 (基于 Python ast)
=============================================================================
直接操作 Python ast 树，生成/交叉/变异/求值全部用标准库。
v4 独立实现，不依赖 v3 GPOptimizer。

[重构] 2026-06-22 扩展 TERMINALS/PRIMITIVES 对齐 utils/ast 注册表。
        新增: OPEN, RETURNS, VWAP 变量; ts_mean/ts_std/ts_sum 等时序函数;
        cs_scale/cs_winsorize/cs_normalize/cs_quantile 截面函数。
=============================================================================

用法:
  >>> from signals.v4.search.gp import GPSearch
  >>> from d2_api import d2_api
  >>> raw_data = d2_api.kline.query('399317.SZ', '2019-07-01', '2025-12-31')
  >>> gs = GPSearch(raw_data, population_size=50, generations=20)
  >>> gs.run()
  >>> gs.report()
=============================================================================
"""
import ast, random, numpy as np, pandas as pd
from typing import List, Dict, Optional
from dataclasses import dataclass, field

from ..expression import Expression
from ..engine import SigEngine


# ============================================================
# GP 配置
# ============================================================

# [重构] 2026-06-22 扩展 TERMINALS 和 PRIMITIVES，
# 对齐 utils/ast/functions.py (72原语) 和 variables.py (70+变量前缀)
TERMINALS = ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'AMOUNT',
             'RETURNS', 'VWAP']

PRIMITIVES = {
    # 一元函数 (带窗口参数): func(leaf, period)
    1: ['rsi', 'ts_roc', 'ts_zscore', 'ts_av_diff', 'ema', 'bb_width',
        'ts_mean', 'ts_std', 'ts_sum', 'ts_min', 'ts_max',
        'ts_delta', 'ts_scale', 'ts_decay_linear',
        'ts_resid'],
    # 二元函数 (两个数组参数+窗口): func(x, y, d)
    2: ['ts_corr', 'ts_cov', 'ts_regression'],
    # 种子表达式 (用于初始化)
    'seed': [
        "ts_mean(CLOSE, 20)", "ts_mean(CLOSE, 50)",
        "ts_std(CLOSE, 20)", "ts_rank(CLOSE, 20)",
        "ts_delta(CLOSE, 10)", "ts_delay(CLOSE, 5)",
        "ts_decay_linear(CLOSE, 20)",
        "atr(HIGH, LOW, CLOSE, 14)",
        "bb_width(CLOSE, 20)", "ema(CLOSE, 20)",
        "macd(CLOSE, 12, 26, 9)",
        "adx(HIGH, LOW, CLOSE, 14)",
        "ts_zscore(amt_ratio(AMOUNT,5,20), 60)",
    ],
    # 截面函数
    'cs': ['cs_rank', 'cs_zscore', 'cs_scale', 'cs_winsorize',
           'cs_normalize', 'cs_quantile'],
    # 算术运算
    'arith': ['+', '-', '*'],
}

# 参数变异池 (window 参数的可选值)
PARAM_POOL = {
    'window': [3, 5, 7, 10, 14, 20, 30, 50, 60, 120],
    'period': [7, 14, 20, 30],
}


# ============================================================
# GP 树操作 (基于 Python ast)
# ============================================================

def random_tree(depth: int = 3) -> ast.Module:
    """随机生成一棵表达式树"""
    if depth <= 1:
        return _random_leaf()
    r = random.random()
    if r < 0.25:
        return _random_leaf()
    elif r < 0.55:
        return _random_unary_tree(depth)
    elif r < 0.85:
        return _random_binary_tree(depth)
    else:
        return _random_cs_tree(depth)


def _random_leaf() -> ast.Module:
    r = random.random()
    if r < 0.5:
        v = random.choice(TERMINALS)
    elif r < 0.8:
        v = str(random.choice(PARAM_POOL['window']))
    else:
        v = str(round(random.uniform(0.1, 2.0), 2))
    return ast.parse(v, mode='eval')


def _random_unary_tree(depth: int) -> ast.Module:
    """随机一元函数: func(expr, period)"""
    func = random.choice(PRIMITIVES[1])
    inner = _random_leaf() if depth <= 2 else random_tree(depth - 1)
    inner_str = ast.unparse(inner)
    param = random.choice(PARAM_POOL['period'] if func == 'rsi' else PARAM_POOL['window'])
    return ast.parse(f"{func}({inner_str}, {param})", mode='eval')


def _random_binary_tree(depth: int) -> ast.Module:
    """从种子模板随机选一个表达式"""
    return ast.parse(random.choice(PRIMITIVES['seed']), mode='eval')


def _random_cs_tree(depth: int) -> ast.Module:
    """随机截面: cs_func(unary(leaf))"""
    func = random.choice(PRIMITIVES['cs'])
    inner = _random_unary_tree(depth - 1) if depth > 2 else _random_leaf()
    inner_str = ast.unparse(inner)
    return ast.parse(f"{func}({inner_str})", mode='eval')


def mutate_tree(tree: ast.Module, depth: int = 2) -> ast.Module:
    """变异: 随机替换一个子节点"""
    nodes = [n for n in ast.walk(tree.body)
             if not isinstance(n, ast.Load) and not isinstance(n, ast.Module)]
    if not nodes:
        return tree
    target = random.choice(nodes)
    replacement = _clean_tree(random_tree(depth))
    replacer = _NodeReplacer(target, replacement)
    result = replacer.visit(tree)
    return _clean_tree(result)


def mutate_param(tree: ast.Module) -> ast.Module:
    """参数变异: 随机改一个 window/period 参数"""
    tree = _deep_copy_tree(tree)
    for node in ast.walk(tree.body):
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            if random.random() < 0.3:
                if isinstance(node.value, int) and node.value in PARAM_POOL['window']:
                    node.value = random.choice(PARAM_POOL['window'])
                elif isinstance(node.value, int) and node.value in PARAM_POOL['period']:
                    node.value = random.choice(PARAM_POOL['period'])
    return tree


def crossover_trees(tree1: ast.Module, tree2: ast.Module) -> ast.Module:
    """交叉: 从 tree2 取一个子树替换 tree1 中一个节点"""
    tree1 = _deep_copy_tree(tree1)
    tree2 = _deep_copy_tree(tree2)
    nodes1 = [n for n in ast.walk(tree1.body)
              if not isinstance(n, (ast.Load, ast.Module))]
    nodes2 = [n for n in ast.walk(tree2.body)
              if not isinstance(n, (ast.Load, ast.Module))]
    if not nodes1 or not nodes2:
        return tree1
    target = random.choice(nodes1)
    replacement = random.choice(nodes2)
    replacer = _NodeReplacer(target, replacement)
    return _clean_tree(replacer.visit(tree1))


def _clean_tree(tree: ast.Module) -> ast.Module:
    """清理树的 location, 避免递归引用"""
    try:
        return ast.parse(ast.unparse(tree), mode='eval')
    except Exception:
        return tree


def _deep_copy_tree(tree: ast.Module) -> ast.Module:
    """深拷贝 ast 树"""
    try:
        return ast.parse(ast.unparse(tree), mode='eval')
    except Exception:
        return tree


def _unwrap_expr(node):
    """去掉外层 Expression 包装"""
    if isinstance(node, ast.Expression):
        return node.body
    return node


class _NodeReplacer(ast.NodeTransformer):
    def __init__(self, target, replacement):
        self.target = target
        self.replacement = replacement

    def generic_visit(self, node):
        if node is self.target:
            return self.replacement
        return super().generic_visit(node)


def tree_depth(tree: ast.Module) -> int:
    """计算树的最大深度"""
    def _d(node, level=0):
        children = list(ast.iter_child_nodes(node))
        if not children:
            return level
        return max(_d(c, level + 1) for c in children)
    return _d(tree)


def tree_complexity(tree: ast.Module) -> int:
    """计算节点数"""
    return sum(1 for _ in ast.walk(tree.body)) - 1


# ============================================================
# GP 核心
# ============================================================

@dataclass
class GPIndividual:
    """GP 个体"""
    tree: ast.Module
    fitness: float = -999.0
    sharpe: float = 0.0
    expression_str: str = ''
    generation: int = 0

    def __repr__(self):
        return f"GPIndv(fitness={self.fitness:.3f}, expr={self.expression_str[:50]})"


class GPSearch:
    """GP 信号搜索器

    用法:
        gs = GPSearch(raw_data, population_size=50, generations=20)
        gs.run()
        gs.report()
        for ind in gs.elite_set(5):
            expr = Expression(ind.expression_str)
            signal = expr.generate(raw_data)
            analyzer = SigEngine.backtest(signal, raw_data, mode='full')
    """

    def __init__(self, raw_data: pd.DataFrame,
                 population_size: int = 50,
                 generations: int = 20,
                 max_depth: int = 5,
                 crossover_prob: float = 0.7,
                 mutation_prob: float = 0.2,
                 param_mutation_prob: float = 0.3,
                 elite_count: int = 5,
                 start_date: str = None,
                 random_seed: int = None):
        self.raw_data = raw_data
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.param_mutation_prob = param_mutation_prob
        self.elite_count = elite_count
        self.start_date = start_date

        if random_seed is not None:
            random.seed(random_seed)

        self.population: List[GPIndividual] = []
        self.best: Optional[GPIndividual] = None
        self.history: List[Dict] = []

    def run(self):
        """运行 GP"""
        # 初始化种群
        self._init_population()

        for gen in range(self.generations):
            self._evaluate_population()
            self._record_gen(gen)
            self._next_generation()

        self._evaluate_population()
        self._record_gen(self.generations)

    # ── 种群操作 ──

    def _init_population(self):
        for _ in range(self.population_size):
            tree = random_tree(random.randint(2, self.max_depth))
            ind = GPIndividual(
                tree=tree,
                expression_str=ast.unparse(tree),
                generation=0,
            )
            self.population.append(ind)

    def _evaluate_population(self):
        for ind in self.population:
            if ind.fitness == -999.0:
                self._evaluate_individual(ind)

    def _evaluate_individual(self, ind: GPIndividual):
        try:
            expr = Expression(ind.expression_str)
            signal = expr.generate(self.raw_data)
            r = SigEngine.backtest(signal, self.raw_data, mode='fast',
                                     start_date=self.start_date)
            ind.fitness = r.sharpe_ratio() or 0
            ind.sharpe = r.sharpe_ratio() or 0
        except Exception:
            ind.fitness = -1.0

    def _next_generation(self):
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)

        # 精英保留
        new_pop = [GPIndividual(
            tree=sorted_pop[i].tree,
            expression_str=sorted_pop[i].expression_str,
            fitness=sorted_pop[i].fitness,
            generation=sorted_pop[i].generation + 1,
        ) for i in range(self.elite_count)]

        # 生成新个体
        while len(new_pop) < self.population_size:
            parent1 = self._tournament_select(sorted_pop)
            r = random.random()

            if r < self.crossover_prob:
                parent2 = self._tournament_select(sorted_pop)
                child_tree = crossover_trees(parent1.tree, parent2.tree)
            elif r < self.crossover_prob + self.mutation_prob:
                child_tree = mutate_tree(parent1.tree, random.randint(1, 3))
            elif r < self.crossover_prob + self.mutation_prob + self.param_mutation_prob:
                child_tree = mutate_param(parent1.tree)
            else:
                child_tree = random_tree(random.randint(2, self.max_depth))

            child_str = ast.unparse(child_tree)
            # 深度保护
            try:
                d = tree_depth(child_tree)
                if d > self.max_depth:
                    continue
            except Exception:
                continue

            new_pop.append(GPIndividual(
                tree=child_tree,
                expression_str=child_str,
                generation=parent1.generation + 1,
            ))

        self.population = new_pop[:self.population_size]

    def _tournament_select(self, sorted_pop, size=3):
        candidates = random.sample(sorted_pop[:len(sorted_pop)//2], min(size, len(sorted_pop)))
        return max(candidates, key=lambda x: x.fitness)

    def _record_gen(self, gen: int):
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        best = sorted_pop[0]
        avg_fit = np.mean([ind.fitness for ind in sorted_pop if ind.fitness > -1.0])
        valid = sum(1 for ind in sorted_pop if ind.fitness > 0)

        self.history.append({
            'generation': gen,
            'best_fitness': round(best.fitness, 3),
            'avg_fitness': round(avg_fit, 3),
            'valid_count': valid,
            'best_expr': best.expression_str[:80],
        })
        print(f"Gen {gen:2d} | best={best.fitness:.3f} avg={avg_fit:.3f} valid={valid}/{len(sorted_pop)}")
        self.best = best

    # ── 公共接口 ──

    def report(self) -> pd.DataFrame:
        return pd.DataFrame(self.history)

    def elite_set(self, n: int = 10) -> List[GPIndividual]:
        return sorted(self.population, key=lambda x: x.fitness, reverse=True)[:n]
