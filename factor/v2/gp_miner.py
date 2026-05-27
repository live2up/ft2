"""
factor/v2/gp_miner.py — 因子 GP 符号回归搜索
=============================================================================

遗传编程符号回归，自动发现有效的因子表达式。

核心链路：
  初始种群 → 适应度评估(ICIR) → 选择/交叉/变异 → 新一代 → ... → 最优个体

关键设计：
  1. 适应度 = IC × (IC / IC_std) = IC × ICIR × 100
     每日一个 IC 值（~1455 个样本），比 Sharpe（~70 次调仓）统计更稳定
  2. 分类种子注入：按因子类别注入已知好因子作为初始种群
  3. 防过拟合：树深限制 + 复杂度惩罚 + population 中 Top N 多样性保持

============================================================================
                         架构层级
============================================================================

┌─────────────────────────────────────────────────────────────────────┐
│  Step 1 — 初始化种群                                                  │
│                                                                      │
│  随机生成 90% 个体（grow 方法随机深度）                               │
│  种子注入 10% 个体（从已知好因子注入）                                 │
│  → Population = Individual[]                                          │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 2 — 适应度评估                                                  │
│                                                                      │
│  对每个个体:                                                          │
│    Individual.tree → FactorExpression.evaluate(data) → factor_panel   │
│    → 计算每日 IC → ic_mean, ic_std → fitness = ic_mean/ic_std*100    │
│                                                                      │
│  无效个体(factor_panel全相同/NaN过多) → fitness = -999               │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 3 — 选择 & 遗传操作                                             │
│                                                                      │
│  精英保留: Top 5% 直接进入下一代                                       │
│  锦标赛选择: 5 个随机个体中选最优                                      │
│  子树交叉: 两棵树的随机子树交换                                        │
│  子树变异: 随机子树替换为新的随机子树                                  │
│  随机注入: 每代引入 5% 全新随机个体，维持探索                          │
└────────────────────┬────────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────────┐
│  Step 4 — 终止 & 输出                                                 │
│                                                                      │
│  达到最大代数 / 收敛 → 输出最佳个体                                   │
│  report() → 代际收敛 / 种群统计 / Top N 表达式                         │
└─────────────────────────────────────────────────────────────────────┘

============================================================================
"""

import random
import logging
import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Set, Any
from dataclasses import dataclass, field

from .expression import (
    FactorExpression, ASTNode, NodeType, evaluate_node,
    VARIABLE_MAP, UNARY_FUNCTIONS, BINARY_FUNCTIONS, PRIMITIVE_FUNCTIONS,
)

logger = logging.getLogger(__name__)

# ============================================================
# GP 配置
# ============================================================

# 终端变量集
TERMINALS = ['close', 'volume', 'open', 'high', 'low', 'amount']

# 函数集（不含 cs_rank 等需要上下文的原语在带参数函数集中）
UNARY_OPS = ['abs', 'sqrt', 'log', 'neg']
BINARY_OPS = ['add', 'sub', 'mul', 'div', 'max', 'min']

# 带参数的时序/截面原语 (函数名, 参数key, 参数候选值)
PRIMITIVE_WITH_PARAMS = [
    ('ts_rank',       'period', [5, 10, 15, 20, 30, 60]),
    ('ts_zscore',     'period', [10, 20, 30, 60]),
    ('delay',         'period', [1, 5, 10, 20, 60]),
    ('decay_linear',  'period', [5, 10, 20, 30]),
    ('cs_zscore',     'period', [20]),       # period 预留但当前未使用
    ('cs_rank',       None, []),              # 无超参数
    ('signed_power',  'exponent', [2.0]),
]

# 可随机生成的常量
CONSTANTS = [0.0, -1.0, 1.0, 0.5, 2.0]

# 默认 GP 超参
DEFAULT_GP_CONFIG = {
    'population_size': 300,
    'generations': 30,
    'max_depth': 8,
    'tournament_size': 5,
    'crossover_prob': 0.75,
    'mutation_prob': 0.15,
    'elite_ratio': 0.05,
    'random_inject_ratio': 0.05,
    'parsimony_penalty': 0.0005,  # 复杂度惩罚系数
}


# ============================================================
# Individual — GP 个体
# ============================================================

@dataclass
class Individual:
    """GP 个体：一棵表达式树 + 适应度"""
    tree: ASTNode
    fitness: float = -999.0
    expression_str: str = ''
    ic_mean: float = 0.0
    ic_std: float = 0.0
    icir: float = 0.0
    generation: int = 0
    depth: int = 0
    node_count: int = 0
    is_seed: bool = False

    def __repr__(self):
        return (f"Individual(fitness={self.fitness:.3f}, "
                f"ic_mean={self.ic_mean:.4f}, depth={self.depth}, "
                f"expr={self.expression_str[:60]}...)")


# ============================================================
# FactorGPMiner — 因子 GP 矿机
# ============================================================

class FactorGPMiner:
    """因子符号回归 GP 搜索器

    发现有效的单因子表达式。适应度基于截面 IC 的均值和稳定性（ICIR）。

    Example:
        >>> gp = FactorGPMiner(
        ...     data=panel_dict,
        ...     future_returns=returns_df,
        ...     population_size=300,
        ...     generations=30,
        ... )
        >>> gp.run()
        >>> best = gp.best()
        >>> print(best.expression_str)
    """

    def __init__(self,
                 data: Dict[str, np.ndarray],
                 future_returns: pd.DataFrame,
                 population_size: int = 300,
                 generations: int = 30,
                 max_depth: int = 8,
                 crossover_prob: float = 0.75,
                 mutation_prob: float = 0.15,
                 tournament_size: int = 5,
                 elite_ratio: float = 0.05,
                 random_inject_ratio: float = 0.05,
                 parsimony_penalty: float = 0.0005,
                 seed_expressions: List[str] = None,
                 allowed_categories: List[str] = None,
                 random_seed: int = None):
        """
        Args:
            data: 面板数据字典 {col: ndarray(T,N)}
            future_returns: 未来收益率 DataFrame (index=日期, cols=标的)
            population_size: 种群大小
            generations: 进化代数
            max_depth: 最大树深
            crossover_prob: 交叉概率
            mutation_prob: 变异概率
            tournament_size: 锦标赛选择大小
            elite_ratio: 精英保留比例
            random_inject_ratio: 每代随机注入比例
            parsimony_penalty: 复杂度惩罚系数
            seed_expressions: 种子表达式列表（已知好因子注入初始种群）
            allowed_categories: 限制搜索的函数类别，None=全部
                e.g. ['ts_rank', 'delay', 'sub'] → 简单动量类
            random_seed: 随机种子（可选）
        """
        self.data = data
        self.future_returns = future_returns
        self.population_size = population_size
        self.generations = generations
        self.max_depth = max_depth
        self.crossover_prob = crossover_prob
        self.mutation_prob = mutation_prob
        self.tournament_size = tournament_size
        self.elite_count = max(1, int(population_size * elite_ratio))
        self.random_inject_count = max(1, int(population_size * random_inject_ratio))
        self.parsimony_penalty = parsimony_penalty
        self.seed_expressions = seed_expressions or []
        self.allowed_categories = allowed_categories
        self.random_seed = random_seed

        # 运行时状态
        self.population: List[Individual] = []
        self.best_individual: Optional[Individual] = None
        self.history: List[Dict] = []

        # 推断形状
        for arr in data.values():
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                self._shape = arr.shape
                break
        else:
            self._shape = (1, 1)

        # 预计算每日 IC 所需的收益率展开
        self._returns_flat = self._prepare_returns()

        # 随机种子
        if random_seed is not None:
            random.seed(random_seed)
            np.random.seed(random_seed)

    def _prepare_returns(self) -> np.ndarray:
        """将收益率 DataFrame 展平为 1D 数组（与 factor_values.ravel() 对齐）"""
        vals = self.future_returns.values
        return np.asarray(vals, dtype=float).ravel()

    # ============================================================
    # 随机树生成
    # ============================================================

    def _random_terminal(self) -> ASTNode:
        """随机生成终端节点（变量或常量）"""
        if random.random() < 0.85:
            var = random.choice(TERMINALS)
            return ASTNode(NodeType.VARIABLE, var)
        else:
            val = random.choice(CONSTANTS)
            return ASTNode(NodeType.CONSTANT, val)

    def _random_function(self) -> Tuple[str, Dict, str]:
        """随机选择函数/原语，返回 (name, params, type)

        type: 'unary' | 'binary' | 'primitive'
        """
        r = random.random()

        # 25% 一元函数
        if r < 0.25:
            return random.choice(UNARY_OPS), {}, 'unary'

        # 30% 二元函数
        elif r < 0.55:
            return random.choice(BINARY_OPS), {}, 'binary'

        # 45% 时序/截面原语
        else:
            prim_name, param_key, param_vals = random.choice(PRIMITIVE_WITH_PARAMS)
            params = {}
            if param_key and param_vals:
                params[param_key] = random.choice(param_vals)
            return prim_name, params, 'primitive'

    def _grow_random_tree(self, depth: int) -> ASTNode:
        """Grow 方法：随机深度生成表达式树

        Args:
            depth: 当前剩余深度

        Returns:
            ASTNode: 随机树根节点
        """
        # 到达深度限制或随机终止 → 终端节点
        if depth <= 1 or random.random() < 0.25:
            return self._random_terminal()

        fn_name, params, fn_type = self._random_function()

        if fn_type == 'unary':
            child = self._grow_random_tree(depth - 1)
            return ASTNode(NodeType.UNARY, fn_name, [child])

        elif fn_type == 'binary':
            left = self._grow_random_tree(depth - 1)
            right = self._grow_random_tree(depth - 1)
            return ASTNode(NodeType.FUNCTION, fn_name, [left, right])

        else:  # primitive
            # 部分原语是二元（correlation 已从函数集移除）
            if fn_name == 'correlation':
                left = self._grow_random_tree(depth - 1)
                right = self._grow_random_tree(depth - 1)
                node = ASTNode(NodeType.FUNCTION, fn_name, [left, right], params)
            else:
                child = self._grow_random_tree(depth - 1)
                node = ASTNode(NodeType.FUNCTION, fn_name, [child], params)

            return node

    # ============================================================
    # 适应度评估
    # ============================================================

    def _evaluate_individual(self, ind: Individual) -> float:
        """评估个体适应度 = IC × ICIR × 100 × (1 - 复杂度惩罚)

        Returns:
            float: 适应度值，-999 表示无效个体
        """
        import traceback
        try:
            # 求值因子面板
            factor_values = evaluate_node(ind.tree, self.data)
        except Exception as e:
            if ind.is_seed:
                logger.warning(f"[GP Eval] 种子求值异常: {e}\n{traceback.format_exc()}")
            return -999.0

        # 无效检测：全 NaN 或全相同
        if np.all(np.isnan(factor_values)):
            if ind.is_seed:
                logger.warning(f"[GP Eval] 种子全NaN: {ind.expression_str[:60]}")
            return -999.0
        if np.nanstd(factor_values) < 1e-10:
            if ind.is_seed:
                logger.warning(f"[GP Eval] 种子零方差: std={np.nanstd(factor_values):.2e}")
            return -999.0

        # 展平因子值
        factor_flat = factor_values.ravel()

        # 计算截面 IC（每日一个 IC 值）
        T, N = self._shape
        daily_ics = []
        for t in range(T):
            fv = factor_values[t, :]
            rv = self.future_returns.iloc[t].values
            mask = ~np.isnan(fv) & ~np.isnan(rv)
            if mask.sum() < 5:
                continue
            corr = np.corrcoef(fv[mask], rv[mask])[0, 1]
            if not np.isnan(corr):
                daily_ics.append(corr)

        if len(daily_ics) < 30:
            if ind.is_seed:
                logger.warning(f"[GP Eval] 种子IC不足: n_ics={len(daily_ics)}, expr={ind.expression_str[:60]}")
            return -999.0

        ind.ic_mean = float(np.mean(daily_ics))
        ind.ic_std = float(np.std(daily_ics, ddof=1))

        if ind.ic_std < 1e-10:
            return -999.0

        ind.icir = ind.ic_mean / ind.ic_std

        # 更新元数据
        ind.depth = FactorExpression._calc_depth(ind.tree)
        ind.node_count = FactorExpression._count_nodes(ind.tree)
        ind.expression_str = ind.tree.to_str()

        # 适应度 = IC × ICIR（既看预测力，也看稳定性）
        fitness = abs(ind.ic_mean) * ind.icir * 100.0

        # 复杂度惩罚
        penalty = self.parsimony_penalty * ind.node_count
        fitness = fitness * (1.0 - penalty)

        ind.fitness = fitness
        return fitness

    def _evaluate_population(self):
        """评估所有个体的适应度"""
        for ind in self.population:
            if ind.fitness == -999.0 or ind.fitness == 0.0:
                self._evaluate_individual(ind)

    # ============================================================
    # 遗传操作
    # ============================================================

    def _tournament_select(self) -> Individual:
        """锦标赛选择：随机选 tournament_size 个个体，返回最优"""
        candidates = random.sample(
            self.population, min(self.tournament_size, len(self.population))
        )
        return max(candidates, key=lambda x: x.fitness)

    def _crossover(self, parent1: Individual, parent2: Individual) -> Individual:
        """子树交叉：随机选择子树位置交换"""
        tree1 = self._clone_tree(parent1.tree)
        tree2 = self._clone_tree(parent2.tree)

        # 随机选择交叉点
        nodes1 = self._collect_nodes(tree1)
        nodes2 = self._collect_nodes(tree2)

        if not nodes1 or not nodes2:
            return Individual(tree1)

        n1 = random.choice(nodes1)
        n2 = random.choice(nodes2)

        # 深度检查：防止交叉后超深
        new_depth = self._calc_depth_after_swap(tree1, n1, n2)
        if new_depth > self.max_depth:
            return Individual(tree1, generation=parent1.generation + 1)

        # 执行交叉
        self._replace_node(tree1, n1, self._clone_tree(n2))
        return Individual(tree1, generation=parent1.generation + 1)

    def _mutate(self, individual: Individual) -> Individual:
        """子树变异：随机替换一个子树"""
        tree = self._clone_tree(individual.tree)
        nodes = self._collect_nodes(tree)

        if not nodes:
            return Individual(tree, generation=individual.generation + 1)

        target = random.choice(nodes)
        replacement = self._grow_random_tree(
            random.randint(1, min(self.max_depth, 4))
        )

        self._replace_node(tree, target, replacement)
        return Individual(tree, generation=individual.generation + 1)

    # ============================================================
    # 主循环
    # ============================================================

    def run(self, verbose: bool = True) -> 'FactorGPMiner':
        """运行 GP 搜索

        Args:
            verbose: 是否打印每代统计信息

        Returns:
            self（可通过 .best() 获取结果）
        """
        # Step 1: 初始化种群
        self._initialize_population()

        # Step 2: 进化循环
        for gen in range(self.generations):
            self._evaluate_population()

            # 记录最佳个体
            gen_best = max(self.population, key=lambda x: x.fitness)
            gen_stats = {
                'generation': gen,
                'best_fitness': gen_best.fitness,
                'best_ic_mean': gen_best.ic_mean,
                'best_icir': gen_best.icir,
                'best_depth': gen_best.depth,
                'best_expression': gen_best.expression_str,
                'avg_fitness': np.mean([i.fitness for i in self.population
                                        if i.fitness > -999]),
                'valid_count': sum(1 for i in self.population if i.fitness > -999),
            }
            self.history.append(gen_stats)

            if verbose and gen % 5 == 0:
                logger.info(
                    f"Gen {gen:3d} | best_f={gen_best.fitness:.3f} "
                    f"IC={gen_best.ic_mean:.4f} "
                    f"depth={gen_best.depth} valid={gen_stats['valid_count']}"
                )

            # 更新全局最优
            if self.best_individual is None or gen_best.fitness > self.best_individual.fitness:
                self.best_individual = gen_best

            # 最后一代不繁殖
            if gen == self.generations - 1:
                break

            # Step 3: 产生下一代
            self.population = self._next_generation(gen)

        # 最终评估
        self._evaluate_population()
        final_best = max(self.population, key=lambda x: x.fitness)
        if final_best.fitness > (self.best_individual.fitness if self.best_individual else -999):
            self.best_individual = final_best

        return self

    def best(self) -> Optional[Individual]:
        """返回最优个体"""
        return self.best_individual

    def report(self) -> str:
        """生成搜索报告"""
        if not self.history:
            return "尚未运行 GP 搜索"
        lines = [
            "=" * 60,
            "因子 GP 符号回归搜索报告",
            "=" * 60,
            f"种群: {self.population_size}  代数: {self.generations}  最大深度: {self.max_depth}",
            "",
        ]
        if self.best_individual:
            best = self.best_individual
            lines += [
                f"最优个体: fitness={best.fitness:.3f}, "
                f"IC={best.ic_mean:.4f}, ICIR={best.icir:.3f}",
                f"深度: {best.depth}, 节点数: {best.node_count}",
                f"表达式: {best.expression_str}",
                "",
            ]
        lines += [
            "代际收敛:",
            f"{'Gen':>4}  {'Best_F':>8}  {'IC_mean':>8}  {'ICIR':>8}  {'Depth':>6}  {'Valid':>6}",
            "-" * 55,
        ]
        for s in self.history:
            if s['generation'] % 5 == 0 or s['generation'] == self.history[-1]['generation']:
                lines.append(
                    f"{s['generation']:4d}  {s['best_fitness']:8.3f}  "
                    f"{s['best_ic_mean']:8.4f}  {s['best_icir']:8.3f}  "
                    f"{s['best_depth']:6d}  {s['valid_count']:6d}"
                )
        return '\n'.join(lines)

    def top(self, n: int = 10) -> List[Individual]:
        """Top N 个最优个体"""
        sorted_pop = sorted(
            [i for i in self.population if i.fitness > -999],
            key=lambda x: x.fitness, reverse=True
        )
        return sorted_pop[:n]

    # ============================================================
    # 内部方法
    # ============================================================

    def _initialize_population(self):
        """初始化种群：随机生成 + 种子注入"""
        self.population = []

        # 90% 随机生成
        random_count = self.population_size - len(self.seed_expressions)
        for _ in range(random_count):
            tree = self._grow_random_tree(
                random.randint(2, self.max_depth)
            )
            ind = Individual(tree, generation=0)
            self.population.append(ind)

        # 10% 种子注入
        for expr_str in self.seed_expressions:
            try:
                fe = FactorExpression(expr_str)
                ind = Individual(fe.ast, generation=0, is_seed=True)
                ind.depth = fe.depth
                ind.node_count = fe.node_count
                ind.expression_str = expr_str
                self.population.append(ind)
            except Exception as e:
                logger.warning(f"种子表达式解析失败: {expr_str} ({e})")

        # 截断到 population_size
        if len(self.population) > self.population_size:
            self.population = self.population[:self.population_size]

    def _next_generation(self, gen: int) -> List[Individual]:
        """产生下一代种群"""
        sorted_pop = sorted(self.population,
                            key=lambda x: x.fitness, reverse=True)

        new_pop = []

        # 精英保留
        for i in range(self.elite_count):
            elite = sorted_pop[i]
            new_pop.append(Individual(
                self._clone_tree(elite.tree),
                fitness=elite.fitness,
                expression_str=elite.expression_str,
                ic_mean=elite.ic_mean, ic_std=elite.ic_std,
                icir=elite.icir, generation=gen + 1,
                depth=elite.depth, node_count=elite.node_count,
            ))

        # 遗传操作填充剩余
        while len(new_pop) < self.population_size - self.random_inject_count:
            r = random.random()

            if r < self.crossover_prob:
                p1 = self._tournament_select()
                p2 = self._tournament_select()
                child = self._crossover(p1, p2)
            elif r < self.crossover_prob + self.mutation_prob:
                parent = self._tournament_select()
                child = self._mutate(parent)
            else:
                parent = self._tournament_select()
                tree = self._clone_tree(parent.tree)
                child = Individual(tree, generation=gen + 1)

            new_pop.append(child)

        # 随机注入
        for _ in range(self.random_inject_count):
            tree = self._grow_random_tree(
                random.randint(2, self.max_depth)
            )
            new_pop.append(Individual(tree, generation=gen + 1))

        # 截断
        return new_pop[:self.population_size]

    # ---- 树操作辅助 ----

    @staticmethod
    def _clone_tree(node: ASTNode) -> ASTNode:
        """深拷贝 AST 树"""
        children = [FactorGPMiner._clone_tree(c) for c in node.children]
        params = dict(node.params)
        return ASTNode(node.node_type, node.value, children, params)

    @staticmethod
    def _collect_nodes(node: ASTNode) -> List[ASTNode]:
        """收集树中所有非叶节点"""
        result = []
        if node.children:
            result.append(node)
        for child in node.children:
            result += FactorGPMiner._collect_nodes(child)
        return result

    @staticmethod
    def _calc_depth_after_swap(root: ASTNode,
                                old_node: ASTNode,
                                new_node: ASTNode) -> int:
        """计算交换后的树深"""
        new_root = FactorGPMiner._clone_tree(root)
        FactorGPMiner._replace_node(new_root, old_node, FactorGPMiner._clone_tree(new_node))
        return FactorExpression._calc_depth(new_root)

    @staticmethod
    def _replace_node(root: ASTNode, target: ASTNode,
                      replacement: ASTNode) -> bool:
        """在树中替换节点（引用比较），返回是否成功"""
        if root is target:
            root.node_type = replacement.node_type
            root.value = replacement.value
            root.children = replacement.children
            root.params = dict(replacement.params)
            return True
        for child in root.children:
            if FactorGPMiner._replace_node(child, target, replacement):
                return True
        return False
