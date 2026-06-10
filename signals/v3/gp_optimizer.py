"""
signals/v2/gp_optimizer.py — 遗传算法择时优化器
=============================================================================

基于 AIdev GP V5 的成熟架构，从 AIdev 移植。


============================================================================
                         架构层级（竖式）
============================================================================

 ┌─────────────────────────────────────────────────────────────────────┐
 │  输入：FeatureSpace + 训练/测试数据 + NODE_CONFIG                     │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 1  —  初始化种群                                               │
 │                                                                      │
 │  随机树生成: grow 方法(随机深度) / full 方法(满树)                    │
 │  种子注入: 从 PRESETS 和 SEED_EXPRESSIONS 注入已知有效表达式          │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 2  —  适应度评估                                               │
 │                                                                      │
 │  每个个体: Expression.generate(train_data) → 信号                     │
 │           run_backtest_from_signal()  →  回测                        │
 │                                                                      │
 │  fitness = train_sharpe × overfit_penalty - complexity_penalty       │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 3  —  选择 & 遗传操作                                           │
 │                                                                      │
 │  精英保留: Top N 直接进入下一代                                       │
 │  适应度共享: 惩罚过于相似的个体，维持多样性                            │
 │  子树交叉: 两棵树的随机子树交换                                        │
 │  子树变异: 随机子树替换为新的随机子树                                  │
 │  随机注入: 每代引入少量全新随机个体，维持探索                          │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 4  —  终止 & 输出                                               │
 │                                                                      │
 │  达到最大代数 / 收敛 → 输出最佳个体                                   │
 │  report() → 代际收敛曲线 / 种群统计 / Top N 表达式                    │
 └─────────────────────────────────────────────────────────────────────┘


============================================================================
                         NODE_CONFIG（节点分布权重）
============================================================================

 ┌──────────────┬──────────────────────────────────────────────────┐
 │  'operator'   │  add:3 / sub:5 / mul:3 / div:2 / max:2 / min:4 │
 │              │  &:3 / |:2                                       │
 │  'function'   │  abs:2 / sign:1 / sqrt:1 / square:1 / neg:2    │
 │              │  relu:2 / tanh:1 / sigmoid:1                     │
 │  'threshold'  │  thr_0:3 / thr_mean:4 / thr_med:2              │
 │              │  thr_roll_mean:5 / thr_roll_med:3               │
 │  'persist'    │  persist_2:3 / persist_3:4 / persist_5:2        │
 │  'switch'     │  switch:3 (if_then / regime_switch)             │
 │  'constant'   │  0.5:2 / 1.0:1 / -1.0:1 / 0.0:1               │
 └──────────────┴──────────────────────────────────────────────────┘


============================================================================
                         技术路线（两条互补路线）
============================================================================

 ┌─────────────────┬──────────────────────────────────────────────┐
 │  路线A: PySR    │  pysr_adapter.py                              │
 │                 │  Julia 后端，多目标 Pareto 优化                │
 │                 │  目标: 拟合未来收益率 → IC/回测验证            │
 ├─────────────────┼──────────────────────────────────────────────┤
 │  路线B: 本文件   │  gp_optimizer.py                             │
 │  (自定义 SR)    │  复用现有种群/算子/选择/共享框架               │
 │                 │  目标: train_sharpe 最大化                    │
 └─────────────────┴──────────────────────────────────────────────┘

============================================================================
"""

import numpy as np
import pandas as pd
import random
import json
import warnings
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field

from .expression import (
    NodeType, TreeNode, Expression,
    parse_expression, np_persist
)
from .features import FeatureSpace, _rolling_mean
from .presets import PRESETS

warnings.filterwarnings('ignore')


# ============================================================
# 节点配置（控制随机树生成时的节点分布权重）
# ============================================================

NODE_CONFIG = {
    'operator': {'add': 3, 'sub': 5, 'mul': 3, 'div': 2, 'max': 2, 'min': 4, '&': 3, '|': 2},
    'function': {'abs': 2, 'sign': 1, 'sqrt': 1, 'square': 1, 'neg': 2, 'relu': 2, 'tanh': 1, 'sigmoid': 1},
    'threshold': {'thr_0': 3, 'thr_mean': 4, 'thr_med': 2, 'thr_roll_mean': 5, 'thr_roll_med': 3},
    'persist': {'persist_2': 3, 'persist_3': 4, 'persist_5': 2},
    'switch': {'switch': 3},  # if_then / regime_switch
    'constant': {'0.5': 2, '1.0': 1, '-1.0': 1, '0.0': 1},
}
ROLL_WINDOWS = [10, 20, 30, 60]

# 种子表达式（规范版，使用 {} 列引用语法）
# 来源: AIdev V4/V5 共识 + Phase 1B/3 验证有效的公式
SEED_EXPRESSIONS = [
    # === 波动率 - 趋势差异类（Phase 1B 衍生类最佳）===
    'thr_mean(STDDEV_TSF_sub{20,7})',
    'thr_mean(BBWIDTH_TSF_sub{30,7})',
    'thr_mean(ATR_TSF_sub{7,7})',
    'thr_roll_mean(BBWIDTH_TSF_sub{30,7}, 30)',

    # === GP V4.2 三维结构（Phase 3 验证 Sharpe 1.08）===
    'thr_mean(BBWIDTH{20}) - (thr_mean(VOL_RATIO{10,20}) min thr_mean(TSF{7}))',
    'thr_mean(BBWIDTH{30}) - (thr_mean(VOL_RATIO{10,20}) min thr_mean(TSF{7}))',

    # === 波动率 + 趋势组合（V3经典模式）===
    'thr_mean(ATR{7}) & thr_mean(TRIMA{60})',
    'thr_mean(ATR{7}) & (ADX{7} - 25) > 0',

    # === 条件分支（GP V5 发现）===
    'switch(thr_mean(ADX{14}), thr_mean(BBWIDTH{30}), thr_mean(neg(BBWIDTH{30})))',
    'switch(thr_mean(VOL_REGIME{20,60}), thr_mean(BBWIDTH{30}), thr_mean(neg(BBWIDTH{30})))',

    # === 信号确认 ===
    'persist(thr_mean(BBWIDTH{30}) sub thr_mean(TSF{7}), 3)',
    'persist(thr_roll_mean(BBWIDTH{30}, 30) sub thr_roll_med(TSF{7}, 20), 3)',

    # === 量波比（GP V4.1）===
    'thr_mean(VOL_RATIO{5,20}) div thr_med(ATR{7} sub TSF{7})',

    # === 滚动阈值变体 ===
    'thr_roll_mean(BBWIDTH{30}, 30) sub (thr_roll_med(VOL_RATIO{10,20}, 30) min thr_roll_mean(TSF{3}, 10))',
    'thr_roll_mean(UP_RATIO{10}, 20) sub thr_roll_med(BBWIDTH{30}, 30)',
]


# ============================================================
# Individual / History 数据结构
# ============================================================

@dataclass
class Individual:
    tree: TreeNode
    expression_str: str = ''
    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_annual: float = 0.0
    test_annual: float = 0.0
    max_drawdown: float = 0.0
    fitness: float = 0.0
    complexity: int = 0
    overfit_ratio: float = 0.0
    trade_count: int = 0

    def __repr__(self):
        return (f"Individual(fit={self.fitness:.4f}, train={self.train_sharpe:.4f}, "
                f"test={self.test_sharpe:.4f}, expr={self.expression_str[:60]})")


@dataclass
class GenerationHistory:
    generation: int
    best_fitness: float
    best_expr: str
    best_train_sharpe: float
    best_test_sharpe: float
    avg_fitness: float
    median_fitness: float
    min_fitness: float
    population_size: int
    elite_count: int
    stagnation: int


# ============================================================
# GPOptimizer 核心类
# ============================================================

class GPOptimizer:
    """
    遗传算法择时优化器

    Args:
        feature_space: v2 FeatureSpace 实例
        train_data: 训练集 OHLCV DataFrame
        test_data: 测试集 OHLCV DataFrame
        population_size: 种群大小（默认80）
        generations: 进化代数（默认50）
        crossover_rate: 交叉概率（默认0.50）
        mutation_rate: 变异概率（默认0.25）
        injection_rate: 随机注入概率（默认0.10）
        overfit_penalty: 过拟合惩罚权重（默认0.5，0=无惩罚 1=全惩罚）
        complexity_penalty: 复杂度惩罚系数（默认0.001）
        sharing_radius: 适应度共享半径（默认0.7）
        stagnation_threshold: 停滞代数阈值（默认5）
        max_depth: 树最大深度（默认6）
        max_size: 树最大节点数（默认25）
        seed_ratio: 种子比例（默认0.30）
        early_stop: 是否启用早停（默认True）
    """

    def __init__(self, feature_space: FeatureSpace,
                 train_data: pd.DataFrame, test_data: pd.DataFrame,
                 population_size: int = 80, generations: int = 50,
                 crossover_rate: float = 0.50, mutation_rate: float = 0.25,
                 injection_rate: float = 0.10,
                 overfit_penalty: float = 0.5, complexity_penalty: float = 0.001,
                 sharing_radius: float = 0.7, stagnation_threshold: int = 5,
                 max_depth: int = 6, max_size: int = 25,
                 seed_ratio: float = 0.30, early_stop: bool = True):
        self._fs = feature_space
        self._train_data = train_data
        self._test_data = test_data

        self.population_size = population_size
        self.generations = generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.injection_rate = injection_rate
        self.overfit_penalty = overfit_penalty
        self.complexity_penalty = complexity_penalty
        self.sharing_radius = sharing_radius
        self.stagnation_threshold = stagnation_threshold
        self.max_depth = max_depth
        self.max_size = max_size
        self.seed_ratio = seed_ratio
        self.early_stop = early_stop

        self._train_feature_df: Optional[pd.DataFrame] = None
        self._test_feature_df: Optional[pd.DataFrame] = None
        self._feature_names: List[str] = []
        self._train_feature_data: Dict[str, np.ndarray] = {}
        self._test_feature_data: Dict[str, np.ndarray] = {}

        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[GenerationHistory] = []
        self._stagnation_count = 0

    def _prepare_features(self):
        self._train_feature_df = self._fs.fit_transform(self._train_data)
        self._test_feature_df = self._fs.fit_transform(self._test_data)
        self._feature_names = self._fs.get_feature_names()
        # 对齐特征矩阵与原始数据的索引（fit_transform 可能丢弃了 NaN 行）
        self._train_data = self._train_data.loc[self._train_feature_df.index]
        self._test_data = self._test_data.loc[self._test_feature_df.index]
        self._train_feature_data = {col: self._train_feature_df[col].values
                                     for col in self._feature_names}
        self._test_feature_data = {col: self._test_feature_df[col].values
                                    for col in self._feature_names}

    # ---- 随机节点/树生成 ----

    def _random_node(self, allowed_types: Optional[Set[str]] = None) -> TreeNode:
        if allowed_types is None:
            allowed_types = {'feature', 'constant', 'operator', 'function',
                            'threshold', 'persist', 'switch'}

        type_weights = {}
        for t in allowed_types:
            if t == 'feature':
                type_weights[t] = len(self._feature_names)
            elif t in NODE_CONFIG:
                type_weights[t] = sum(NODE_CONFIG[t].values())
            else:
                type_weights[t] = 1

        total_weight = sum(type_weights.values())
        r = random.random() * total_weight
        cumulative = 0
        chosen_type = None
        for t, w in type_weights.items():
            cumulative += w
            if r < cumulative:
                chosen_type = t
                break

        if chosen_type == 'feature':
            return TreeNode(NodeType.FEATURE, random.choice(self._feature_names))
        elif chosen_type == 'constant':
            consts = list(NODE_CONFIG['constant'].keys())
            weights = list(NODE_CONFIG['constant'].values())
            return TreeNode(NodeType.CONSTANT, random.choices(consts, weights=weights, k=1)[0])
        elif chosen_type == 'operator':
            ops = list(NODE_CONFIG['operator'].keys())
            weights = list(NODE_CONFIG['operator'].values())
            return TreeNode(NodeType.OPERATOR, random.choices(ops, weights=weights, k=1)[0])
        elif chosen_type == 'function':
            funcs = list(NODE_CONFIG['function'].keys())
            weights = list(NODE_CONFIG['function'].values())
            return TreeNode(NodeType.FUNCTION, random.choices(funcs, weights=weights, k=1)[0])
        elif chosen_type == 'threshold':
            threshs = list(NODE_CONFIG['threshold'].keys())
            weights = list(NODE_CONFIG['threshold'].values())
            value = random.choices(threshs, weights=weights, k=1)[0]
            param = None
            if value in ('thr_roll_mean', 'thr_roll_med'):
                param = random.choice(ROLL_WINDOWS)
            return TreeNode(NodeType.THRESHOLD, value, param=param)
        elif chosen_type == 'persist':
            persists = list(NODE_CONFIG['persist'].keys())
            weights = list(NODE_CONFIG['persist'].values())
            value = random.choices(persists, weights=weights, k=1)[0]
            param = int(value.split('_')[1])
            return TreeNode(NodeType.PERSIST, 'persist', param=param)
        elif chosen_type == 'switch':
            return TreeNode(NodeType.SWITCH, 'switch')
        else:
            return TreeNode(NodeType.FEATURE, random.choice(self._feature_names))

    def _random_tree(self, max_depth: int = 3, method: str = 'grow') -> TreeNode:
        if max_depth <= 1 or (method == 'grow' and random.random() < 0.3):
            return self._random_node({'feature', 'constant'})

        node = self._random_node({'operator', 'function', 'threshold', 'persist', 'switch'})

        if node.node_type == NodeType.OPERATOR:
            node.children = [
                self._random_tree(max_depth - 1, method),
                self._random_tree(max_depth - 1, method)
            ]
        elif node.node_type in (NodeType.FUNCTION, NodeType.THRESHOLD, NodeType.PERSIST):
            node.children = [self._random_tree(max_depth - 1, method)]
        elif node.node_type == NodeType.SWITCH:
            node.children = [
                self._random_tree(max_depth - 1, method),
                self._random_tree(max_depth - 1, method),
                self._random_tree(max_depth - 1, method)
            ]

        return node

    def _parse_seed(self, expr: str) -> Optional[TreeNode]:
        try:
            return parse_expression(expr, set(self._feature_names))
        except Exception:
            return None

    # ---- 种群初始化 ----

    def _init_population(self):
        self.population = []

        seed_count = int(self.population_size * self.seed_ratio)
        seed_pool = SEED_EXPRESSIONS[:min(seed_count, len(SEED_EXPRESSIONS))]

        for expr_str in seed_pool:
            tree = self._parse_seed(expr_str)
            if tree and tree.depth() <= self.max_depth and tree.size() <= self.max_size:
                ind = Individual(tree=tree, expression_str=expr_str)
                self.population.append(ind)

        remaining = self.population_size - len(self.population)
        for _ in range(remaining):
            depth = random.randint(2, min(4, self.max_depth))
            method = random.choice(['grow', 'full'])
            tree = self._random_tree(depth, method)
            if tree.depth() <= self.max_depth and tree.size() <= self.max_size:
                expr_str = tree.to_string()
                ind = Individual(tree=tree, expression_str=expr_str)
                self.population.append(ind)

        while len(self.population) < self.population_size:
            tree = self._random_tree(2, 'grow')
            expr_str = tree.to_string()
            ind = Individual(tree=tree, expression_str=expr_str)
            self.population.append(ind)

    # ---- 适应度评估 ----

    def _evaluate_signals(self, signals: np.ndarray, prices: pd.Series) -> Dict:
        """[v3 独立] 基类默认实现; GPSearch 子类会覆盖为 EngineV3 版本"""
        raise NotImplementedError(
            "GPOptimizer._evaluate_signals 应由子类覆盖。"
            "v3 请使用 signals.v3.GPSearch (自动覆盖为 EngineV3 fast 模式)"
        )

    def _evaluate_individual(self, ind: Individual) -> Individual:
        try:
            train_signals = ind.tree.evaluate(self._train_feature_data)
            train_signals = np.nan_to_num(train_signals, nan=0.0, posinf=0.0, neginf=0.0)

            train_result = self._evaluate_signals(train_signals, self._train_data['close'])
            ind.train_sharpe = train_result['sharpe']
            ind.train_annual = train_result['annual_return']

            test_signals = ind.tree.evaluate(self._test_feature_data)
            test_signals = np.nan_to_num(test_signals, nan=0.0, posinf=0.0, neginf=0.0)

            test_result = self._evaluate_signals(test_signals, self._test_data['close'])
            ind.test_sharpe = test_result['sharpe']
            ind.test_annual = test_result['annual_return']
            ind.max_drawdown = test_result['max_drawdown']
            ind.trade_count = test_result['trade_count']

            ind.complexity = ind.tree.size()

            if ind.train_sharpe > 0:
                ind.overfit_ratio = min(ind.test_sharpe / ind.train_sharpe, 2.0)
            else:
                ind.overfit_ratio = 0.0

            if ind.overfit_ratio < 0 or ind.train_sharpe <= 0 or ind.test_sharpe <= -5:
                ind.fitness = -999.0
            else:
                penalty = min(ind.overfit_ratio, 1.0)
                ind.fitness = (ind.train_sharpe
                               * (self.overfit_penalty + (1 - self.overfit_penalty) * penalty)
                               - self.complexity_penalty * ind.complexity)

            if ind.train_sharpe <= -1 or ind.test_sharpe <= -5:
                ind.fitness = -999.0

        except Exception:
            ind.fitness = -1.0
            ind.train_sharpe = -10.0
            ind.test_sharpe = -10.0

        return ind

    # ---- 适应度共享 ----

    def _fitness_sharing(self, population: List[Individual]) -> List[float]:
        shared_fitness = []
        for i, ind in enumerate(population):
            if ind.fitness <= -900:
                shared_fitness.append(ind.fitness)
                continue
            niche_count = 1.0
            for j, other in enumerate(population):
                if i != j and other.fitness > -900:
                    dist = self._tree_distance(ind.tree, other.tree)
                    if dist < self.sharing_radius:
                        share = 1.0 - (dist / self.sharing_radius)
                        niche_count += share
            shared_fitness.append(ind.fitness / niche_count)
        return shared_fitness

    def _tree_distance(self, t1: TreeNode, t2: TreeNode) -> float:
        s1 = t1.to_string()
        s2 = t2.to_string()
        if s1 == s2:
            return 0.0
        f1 = set(s1.replace('(', ' ').replace(')', ' ').replace(',', ' ').split())
        f2 = set(s2.replace('(', ' ').replace(')', ' ').replace(',', ' ').split())
        intersection = len(f1 & f2)
        union = len(f1 | f2)
        if union == 0:
            return 1.0
        return 1.0 - intersection / union

    # ---- 选择 ----

    def _select_parent(self, shared_fitness: List[float]) -> Individual:
        valid = [(i, sf) for i, sf in enumerate(shared_fitness) if sf > -900]
        if not valid:
            return random.choice(self.population)
        total = sum(sf for _, sf in valid)
        if total <= 0:
            return random.choice(self.population)
        r = random.random() * total
        cumulative = 0
        for i, sf in valid:
            cumulative += sf
            if r < cumulative:
                return self.population[i]
        return self.population[valid[-1][0]]

    # ---- 遗传算子 ----

    def _crossover(self, parent1: Individual, parent2: Individual) -> Tuple[TreeNode, TreeNode]:
        t1 = parent1.tree.copy()
        t2 = parent2.tree.copy()
        n1 = t1.size()
        n2 = t2.size()
        if n1 <= 1 or n2 <= 1:
            return t1, t2

        cut_point1 = random.randint(1, n1 - 1)
        cut_point2 = random.randint(1, n2 - 1)

        subtree1 = self._get_subtree(t1, cut_point1)
        subtree2 = self._get_subtree(t2, cut_point2)

        t1_new = self._replace_subtree(t1, cut_point1, subtree2)
        t2_new = self._replace_subtree(t2, cut_point2, subtree1)

        if t1_new.depth() > self.max_depth or t1_new.size() > self.max_size:
            t1_new = t1
        if t2_new.depth() > self.max_depth or t2_new.size() > self.max_size:
            t2_new = t2

        return t1_new, t2_new

    def _mutate(self, parent: Individual) -> TreeNode:
        tree = parent.tree.copy()
        n = tree.size()
        if n <= 1:
            return tree

        cut_point = random.randint(1, n - 1)
        new_subtree = self._random_tree(random.randint(1, 3), 'grow')
        new_tree = self._replace_subtree(tree, cut_point, new_subtree)

        if new_tree.depth() > self.max_depth or new_tree.size() > self.max_size:
            return tree
        return new_tree

    def _get_subtree(self, tree: TreeNode, index: int) -> TreeNode:
        nodes = []
        self._collect_nodes(tree, nodes)
        if 0 <= index < len(nodes):
            return nodes[index].copy()
        return tree.copy()

    def _collect_nodes(self, tree: TreeNode, nodes: List[TreeNode]):
        nodes.append(tree)
        for child in tree.children:
            self._collect_nodes(child, nodes)

    def _replace_subtree(self, tree: TreeNode, index: int, new_subtree: TreeNode) -> TreeNode:
        if index == 0:
            return new_subtree.copy()

        nodes = []
        parent_map = {}
        self._collect_with_parents(tree, None, nodes, parent_map)

        if 0 <= index < len(nodes):
            target = nodes[index]
            parent = parent_map[target]
            if parent:
                for i, child in enumerate(parent.children):
                    if child is target:
                        parent.children[i] = new_subtree.copy()
                        break
        return tree

    def _collect_with_parents(self, tree: TreeNode, parent: Optional[TreeNode],
                              nodes: List[TreeNode], parent_map: Dict):
        nodes.append(tree)
        parent_map[tree] = parent
        for child in tree.children:
            self._collect_with_parents(child, tree, nodes, parent_map)

    def _inject_random(self) -> Individual:
        depth = random.randint(2, 4)
        tree = self._random_tree(depth, 'grow')
        if tree.depth() > self.max_depth or tree.size() > self.max_size:
            tree = self._random_tree(2, 'grow')
        expr_str = tree.to_string()
        return Individual(tree=tree, expression_str=expr_str)

    # ---- 主进化循环 ----

    def run(self, verbose: bool = True) -> Individual:
        self._prepare_features()

        if verbose:
            print("=" * 70)
            print("GP 遗传算法择时优化器 启动")
            print(f"  种群: {self.population_size}  |  代数: {self.generations}")
            print(f"  交叉/变异/注入: {self.crossover_rate}/{self.mutation_rate}/{self.injection_rate}")
            print(f"  过拟合惩罚: {self.overfit_penalty}  |  复杂度惩罚: {self.complexity_penalty}")
            print(f"  最大深度: {self.max_depth}  |  最大节点: {self.max_size}")
            print(f"  特征数: {len(self._feature_names)}")
            print(f"  训练: {len(self._train_data)}条  |  测试: {len(self._test_data)}条")
            print("=" * 70)

        self._init_population()
        if verbose:
            print(f"\n初始化种群: {len(self.population)} 个体")

        for ind in self.population:
            self._evaluate_individual(ind)

        self.best_individual = max(self.population, key=lambda x: x.fitness)
        if verbose:
            print(f"初始最优: fitness={self.best_individual.fitness:.4f}  "
                  f"train={self.best_individual.train_sharpe:.4f}  "
                  f"test={self.best_individual.test_sharpe:.4f}")

        stagnation_count = 0
        best_fitness_history = []

        for gen in range(self.generations):
            shared_fitness = self._fitness_sharing(self.population)

            if stagnation_count >= self.stagnation_threshold:
                cr = self.crossover_rate * 0.6
                mr = self.mutation_rate * 1.5
                ir = self.injection_rate * 2.0
            else:
                cr = self.crossover_rate
                mr = self.mutation_rate
                ir = self.injection_rate

            new_population = []

            elite_count = max(2, int(self.population_size * 0.05))
            sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
            for i in range(elite_count):
                expr_str = sorted_pop[i].tree.to_string()
                new_population.append(Individual(
                    tree=sorted_pop[i].tree.copy(),
                    expression_str=expr_str
                ))

            while len(new_population) < self.population_size:
                r = random.random()
                if r < cr:
                    p1 = self._select_parent(shared_fitness)
                    p2 = self._select_parent(shared_fitness)
                    t1, t2 = self._crossover(p1, p2)
                    new_population.append(Individual(tree=t1, expression_str=t1.to_string()))
                    if len(new_population) < self.population_size:
                        new_population.append(Individual(tree=t2, expression_str=t2.to_string()))
                elif r < cr + mr:
                    p = self._select_parent(shared_fitness)
                    t = self._mutate(p)
                    new_population.append(Individual(tree=t, expression_str=t.to_string()))
                else:
                    new_population.append(self._inject_random())

            while len(new_population) > self.population_size:
                new_population.pop()

            for ind in new_population:
                self._evaluate_individual(ind)

            self.population = new_population

            current_best = max(self.population, key=lambda x: x.fitness)
            best_fitness_history.append(current_best.fitness)

            avg_fitness = np.mean([ind.fitness for ind in self.population
                                   if ind.fitness > -900])
            median_fitness = float(np.median([ind.fitness for ind in self.population
                                              if ind.fitness > -900])) if any(ind.fitness > -900 for ind in self.population) else -999

            if current_best.fitness > self.best_individual.fitness:
                self.best_individual = Individual(
                    tree=current_best.tree.copy(),
                    expression_str=current_best.expression_str,
                    train_sharpe=current_best.train_sharpe,
                    test_sharpe=current_best.test_sharpe,
                    train_annual=current_best.train_annual,
                    test_annual=current_best.test_annual,
                    max_drawdown=current_best.max_drawdown,
                    fitness=current_best.fitness,
                    complexity=current_best.complexity,
                    overfit_ratio=current_best.overfit_ratio,
                    trade_count=current_best.trade_count,
                )
                stagnation_count = 0
            else:
                stagnation_count += 1

            self.history.append(GenerationHistory(
                generation=gen + 1,
                best_fitness=current_best.fitness,
                best_expr=current_best.expression_str[:80],
                best_train_sharpe=current_best.train_sharpe,
                best_test_sharpe=current_best.test_sharpe,
                avg_fitness=avg_fitness if not np.isnan(avg_fitness) else -999,
                median_fitness=median_fitness,
                min_fitness=min(ind.fitness for ind in self.population),
                population_size=len(self.population),
                elite_count=elite_count,
                stagnation=stagnation_count,
            ))

            if verbose:
                stag_mark = " [停滞]" if stagnation_count > 0 else ""
                print(f"\nGen {gen + 1:3d}/{self.generations}{stag_mark}:  "
                      f"best={current_best.fitness:.4f}  "
                      f"train_s={current_best.train_sharpe:.3f}  "
                      f"test_s={current_best.test_sharpe:.3f}  "
                      f"avg={avg_fitness:.4f}")

            if self.early_stop and stagnation_count >= self.stagnation_threshold * 3:
                if verbose:
                    print(f"\n  早停于第 {gen + 1} 代（连续 {stagnation_count} 代无改进）")
                break

        if verbose:
            print(f"\n{'=' * 70}")
            print(f"进化完成。最优个体:")
            print(f"  表达式: {self.best_individual.expression_str}")
            print(f"  Fitness: {self.best_individual.fitness:.4f}")
            print(f"  Train Sharpe: {self.best_individual.train_sharpe:.4f}")
            print(f"  Test Sharpe:  {self.best_individual.test_sharpe:.4f}")
            print(f"  Train 年化:   {self.best_individual.train_annual:.1f}%")
            print(f"  Test 年化:    {self.best_individual.test_annual:.1f}%")
            print(f"  Max Drawdown:  {self.best_individual.max_drawdown:.1f}%")
            print(f"  复杂度: {self.best_individual.complexity}")
            print(f"  过拟合比: {self.best_individual.overfit_ratio:.3f}")
            print(f"  交易次数: {self.best_individual.trade_count}")

        return self.best_individual

    # ---- 结果输出 ----

    def to_expression(self) -> Expression:
        if self.best_individual is None:
            raise RuntimeError("尚未运行优化器，请先调用 run()")
        return Expression(self.best_individual.tree.copy(),
                          feature_space=self._fs,
                          name=f'GP_{self.best_individual.complexity}')

    def report(self) -> pd.DataFrame:
        if not self.history:
            return pd.DataFrame()
        rows = []
        for h in self.history:
            rows.append({
                '代': h.generation,
                '最佳Fitness': round(h.best_fitness, 4),
                '最佳TrainSharpe': round(h.best_train_sharpe, 4),
                '最佳TestSharpe': round(h.best_test_sharpe, 4),
                '平均Fitness': round(h.avg_fitness, 4),
                '中位Fitness': round(h.median_fitness, 4),
                '停滞代数': h.stagnation,
                '精英数': h.elite_count,
                '最佳表达式': h.best_expr[:70],
            })
        return pd.DataFrame(rows)

    def top_individuals(self, n: int = 10) -> List[Individual]:
        valid = [ind for ind in self.population if ind.fitness > -900]
        valid.sort(key=lambda x: x.fitness, reverse=True)
        seen = set()
        unique = []
        for ind in valid:
            sig = ind.tree.to_string()
            if sig not in seen:
                seen.add(sig)
                unique.append(ind)
                if len(unique) >= n:
                    break
        return unique

    def save_best(self, path: str):
        expr = self.to_expression()
        expr.save(path)

    def to_dict(self) -> Dict:
        return {
            'best_individual': {
                'expression': self.best_individual.expression_str if self.best_individual else None,
                'train_sharpe': self.best_individual.train_sharpe if self.best_individual else 0,
                'test_sharpe': self.best_individual.test_sharpe if self.best_individual else 0,
                'fitness': self.best_individual.fitness if self.best_individual else 0,
                'complexity': self.best_individual.complexity if self.best_individual else 0,
                'overfit_ratio': self.best_individual.overfit_ratio if self.best_individual else 0,
            },
            'config': {
                'population_size': self.population_size,
                'generations': self.generations,
                'crossover_rate': self.crossover_rate,
                'mutation_rate': self.mutation_rate,
                'max_depth': self.max_depth,
                'max_size': self.max_size,
            },
            'history': [{
                'generation': h.generation,
                'best_fitness': h.best_fitness,
                'best_train_sharpe': h.best_train_sharpe,
                'best_test_sharpe': h.best_test_sharpe,
                'avg_fitness': h.avg_fitness,
            } for h in self.history],
        }

    def save_report(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
