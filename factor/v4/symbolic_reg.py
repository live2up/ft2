"""
factor/v4/symbolic_reg.py — 符号回归 (Symbolic Regression)
=============================================================================
修复: 
  v2 — bloat修复: 只保留subtree变异, 禁止常数调优
  v2 — log安全: 注册 sr_log 保留符号, 消除 log(-1.4) 语义错误
  v2 — 节点数上限: max_nodes=60, 超过则丢弃
  v2 — 预测裁剪: 极端值截断到±10σ, 防止数值爆炸

定位：
  因子发现之外的另一条腿。
    - 因子发现: 技术指标组合 → 选股/择时 (Sharpe)
    - 符号回归: 数学公式拟合 → 预测 (MSE)
=============================================================================
"""
import ast
import copy
import random
import logging
from typing import Dict, List

import numpy as np

from . import gp_engine as gp
from .gp_engine import Individual, _ExpressionFromAST, ast_node_count
from .gp_engine import GP_CONSTANTS

logger = logging.getLogger(__name__)

# ============================================================
# 0. 注册 sr_log: 保留符号的 log (避免 log(-1.4) 语义错误)
# ============================================================
# [修改] 2026-07-16 factor/v4 固定引用 utils.ast.v1, 保持与原始算法一致
from utils.ast.v1 import register_function

# safe_log 取绝对值丢掉符号: log(-1.4) 实际算 log(1.4), 语义错误
# sr_log: sign(x) * log(|x| + eps), 保留符号信息
def _sr_log(x):
    return np.sign(x) * np.log(np.maximum(np.abs(x), 1e-10))

register_function('sr_log', _sr_log)


# ============================================================
# 1. 树生成器（纯数学原语，无 if-else/and/or/not/compare）
# ============================================================

SR_VARIABLES: List[str] = []

# 一元数学原语: 用 sr_log 替代 log
SR_UNARY = ['sin', 'cos', 'exp', 'sr_log', 'sqrt', 'abs', 'tanh']


def _sr_terminal() -> ast.AST:
    if random.random() < 0.8:
        return ast.Name(id=random.choice(SR_VARIABLES), ctx=ast.Load())
    return ast.Constant(value=random.choice(GP_CONSTANTS))


def _sr_grow_tree(depth: int, max_nodes: int = 60, used_nodes: int = 0) -> ast.AST:
    """生成纯数学表达式树, 带节点数限制

    Args:
        depth: 剩余深度
        max_nodes: 最大节点数 (超出则提前返回终端)
        used_nodes: 已用节点计数器 (列表包装以在递归中引用)
    """
    if depth <= 1:
        return _sr_terminal()

    r = random.random()

    if r < 0.10 or used_nodes >= max_nodes:
        return _sr_terminal()

    elif r < 0.35:
        # 二元运算
        left = _sr_grow_tree(depth - 1, max_nodes, used_nodes + 1)
        right = _sr_grow_tree(depth - 1, max_nodes, used_nodes + 2)
        return ast.BinOp(
            left=left,
            op=random.choice([ast.Add(), ast.Sub(), ast.Mult()]),  # 去掉 Div (易数值爆炸)
            right=right,
        )

    elif r < 0.70:
        # 一元函数
        fn = random.choice(SR_UNARY)
        arg = _sr_grow_tree(depth - 1, max_nodes, used_nodes + 1)
        return ast.Call(
            func=ast.Name(id=fn, ctx=ast.Load()),
            args=[arg], keywords=[],
        )

    else:
        # 嵌套: sin(x) + cos(y)
        lfn = random.choice(SR_UNARY)
        rfn = random.choice(SR_UNARY)
        left = _sr_grow_tree(depth - 2, max_nodes, used_nodes + 1)
        right = _sr_grow_tree(depth - 2, max_nodes, used_nodes + 2)
        return ast.BinOp(
            left=ast.Call(func=ast.Name(id=lfn), args=[left], keywords=[]),
            op=random.choice([ast.Add(), ast.Sub(), ast.Mult()]),
            right=ast.Call(func=ast.Name(id=rfn), args=[right], keywords=[]),
        )


def _sr_random_tree(max_depth: int = 5, max_nodes: int = 60) -> ast.Expression:
    """创建随机符号回归表达式树, 确保在节点数限制内"""
    for _ in range(10):  # 最多尝试10次
        depth = random.randint(2, max_depth)
        body = _sr_grow_tree(depth, max_nodes)
        tree = ast.Expression(body=body)
        ast.fix_missing_locations(tree)
        if ast_node_count(tree) <= max_nodes:
            return tree
    # fallback: 单个变量
    tree = ast.Expression(body=_sr_terminal())
    ast.fix_missing_locations(tree)
    return tree


# ============================================================
# 2. 树简化
# ============================================================

def _fold_constants(node: ast.AST) -> ast.AST:
    if isinstance(node, ast.BinOp):
        node.left = _fold_constants(node.left)
        node.right = _fold_constants(node.right)
        lc = isinstance(node.left, ast.Constant) and isinstance(node.left.value, (int, float))
        rc = isinstance(node.right, ast.Constant) and isinstance(node.right.value, (int, float))
        if rc:
            v = node.right.value
            if isinstance(node.op, ast.Add) and v == 0: return node.left
            if isinstance(node.op, ast.Sub) and v == 0: return node.left
            if isinstance(node.op, ast.Mult) and v == 1: return node.left
            if isinstance(node.op, ast.Mult) and v == 0: return ast.Constant(value=0.0)
        if lc:
            v = node.left.value
            if isinstance(node.op, ast.Mult) and v == 0: return ast.Constant(value=0.0)
            if isinstance(node.op, ast.Mult) and v == 1: return node.right
            if isinstance(node.op, ast.Add) and v == 0: return node.right
    elif isinstance(node, ast.UnaryOp):
        node.operand = _fold_constants(node.operand)
        if isinstance(node.op, ast.USub) and isinstance(node.operand, ast.UnaryOp) and isinstance(node.operand.op, ast.USub):
            return node.operand.operand
    elif isinstance(node, ast.Call):
        node.args = [_fold_constants(a) for a in node.args]
    return node


def simplify_tree(expr_str: str) -> str:
    try:
        tree = ast.parse(expr_str, mode='eval')
        tree = ast.Expression(body=_fold_constants(tree.body))
        s = ast.unparse(tree.body)
        # 替换 sr_log → log (输出更直观)
        s = s.replace('sr_log(', 'log(')
        return s
    except Exception:
        return expr_str


# ============================================================
# 3. Fitness: 负 MSE (带数值保护)
# ============================================================

class MSEFitness:
    """负均方误差 — 带 NaN/Inf/常数检测"""

    def __init__(self, y_true: np.ndarray, clip_std: float = 10.0):
        self.y_true = np.asarray(y_true, float).ravel()
        self.y_std = np.nanstd(self.y_true) or 1.0
        self.clip = clip_std * self.y_std  # 裁剪阈值

    def compute(self, pred: np.ndarray) -> float:
        pred = np.asarray(pred, float).ravel()
        # 对齐长度
        min_len = min(len(pred), len(self.y_true))
        pred, y = pred[:min_len], self.y_true[:min_len]

        # 1. NaN/Inf 过滤
        finite = np.isfinite(pred)
        if finite.sum() < 10:
            return -999.0

        # 2. 裁剪极端值 (防数值爆炸)
        pred = np.clip(pred, -self.clip, self.clip)

        # 3. 常数检测 (预测值无变化 → 无信息)
        if np.nanstd(pred) < 1e-8:
            return -999.0

        # 4. 方差比检测 (预测方差远小于真实方差 → 退化)
        pred_var = np.nanvar(pred)
        true_var = np.nanvar(y)
        if true_var > 1e-8 and pred_var / true_var < 0.001:
            return -999.0

        # 5. MSE
        valid = ~np.isnan(pred) & ~np.isnan(y)
        if valid.sum() < 10:
            return -999.0
        mse = np.nanmean((y[valid] - pred[valid]) ** 2)
        return -mse


# ============================================================
# 4. 入口函数
# ============================================================

def symbolic_regression(
    X: Dict[str, np.ndarray],
    y: np.ndarray,
    generations: int = 20,
    population_size: int = 200,
    max_depth: int = 5,
    max_nodes: int = 40,
    parsimony_penalty: float = 0.01,
    random_inject_ratio: float = 0.15,
    stall_generations: int = 5,
    random_seed: int = None,
    verbose: bool = True,
) -> gp.GPEngine:
    """运行符号回归

    v2 改进:
      - 只保留 subtree 变异 (去掉 constant/window/logic 变异 → bloat减少)
      - sr_log 保留符号信息 (vs safe_log 取绝对值丢符号)
      - max_nodes 节点数上限 (配合 depth 双重限制)
      - MSE 裁剪 + NaN/常数检测 (防数值爆炸)
      - 停滞检测: 连续 stall_generations 代无改善则注入随机个体 (默认5代)
      - 默认 random_inject_ratio 从 5% → 15% 保持种群多样性

    Args:
        X: 特征 {'x1': ndarray, 'x2': ndarray, ...}
        y: 目标 ndarray
        generations: 进化代数
        population_size: 种群大小
        max_depth: 树最大深度
        max_nodes: 树最大节点数
        parsimony_penalty: 稀疏惩罚 (越大公式越简短)
        random_inject_ratio: 每代随机注入比例 (保持多样性)
        stall_generations: 停滞几代后触发额外注入
        random_seed: 随机种子

    Returns:
        GPEngine (已运行)
    """
    global SR_VARIABLES
    SR_VARIABLES = list(X.keys())

    # 备份
    _orig_random_tree = gp._random_tree
    _orig_mutate_weights = gp._MUTATE_OPS[:]

    try:
        # 替换树生成器
        gp._random_tree = lambda d=max_depth: _sr_random_tree(d, max_nodes)
        # 替换变异算子: 只保留子树变异
        gp._MUTATE_OPS = [gp._mutate_subtree]

        engine = gp.GPEngine(
            data=X,
            fitness_calculator=MSEFitness(y),
            config={
                'population_size': population_size,
                'generations': generations,
                'max_depth': max_depth,
                'seed_ratio': 0.0,
                'parsimony_penalty': parsimony_penalty,
                'random_inject_ratio': random_inject_ratio,
                'mutate_subtree_weight': 1.0,
                'mutate_constant_weight': 0.0,
                'mutate_window_weight': 0.0,
                'mutate_logic_weight': 0.0,
                'mutate_insert_cond_weight': 0.0,
            },
            random_seed=random_seed,
        )

        # 替换实例的变异方法: 只保留子树变异 + 节点数保护
        def _sr_mutate(individual, max_depth=4):
            for _ in range(5):
                new_tree = gp._mutate_subtree(individual.tree, max_depth=min(max_depth, 3))
                if ast_node_count(new_tree) <= max_nodes:
                    return Individual(tree=new_tree, generation=individual.generation + 1)
            return Individual(tree=copy.deepcopy(individual.tree), generation=individual.generation + 1)
        engine._mutate = _sr_mutate

        # 替换 run 方法: 加入停滞检测 + 多样性注入
        _orig_run = engine.run
        _orig_next_gen = engine._next_generation

        def _sr_run(verbose=True):
            engine._initialize_population()
            stall_count = 0
            best_ever = -999.0

            for gen in range(engine.generations):
                engine._evaluate_population()
                gen_best = max(engine.population, key=lambda x: x.fitness)

                # 停滞检测
                if gen_best.fitness > best_ever + 1e-6:
                    best_ever = gen_best.fitness
                    stall_count = 0
                else:
                    stall_count += 1

                # 停滞 > 阈值 → 注入随机个体 (替换底部30%)
                if stall_count >= stall_generations:
                    n_inject = max(5, int(engine.population_size * 0.3))
                    sorted_pop = sorted(engine.population, key=lambda x: x.fitness)
                    for i in range(min(n_inject, len(sorted_pop) // 2)):
                        new_ind = Individual(
                            tree=_sr_random_tree(max_depth, max_nodes),
                            generation=gen + 1,
                        )
                        engine.population[engine.population.index(sorted_pop[i])] = new_ind
                    stall_count = 0
                    if verbose:
                        logger.info(f"  Gen {gen:3d} | 停滞触发注入 {n_inject} 新个体 | best={best_ever:.4f}")

                # 记录历史
                valid = [i for i in engine.population if i.fitness > -999]
                gen_stats = {
                    'generation': gen,
                    'best_fitness': gen_best.fitness,
                    'best_depth': gen_best.depth,
                    'best_expression': gen_best.expression_str,
                    'avg_fitness': np.mean([i.fitness for i in valid]) if valid else -999,
                    'valid_count': len(valid),
                }
                engine.history.append(gen_stats)

                if engine.best_individual is None or gen_best.fitness > engine.best_individual.fitness:
                    engine.best_individual = Individual(
                        tree=copy.deepcopy(gen_best.tree),
                        fitness=gen_best.fitness,
                        expression_str=gen_best.expression_str,
                        depth=gen_best.depth, node_count=gen_best.node_count,
                        generation=gen,
                    )

                if verbose and (gen % 5 == 0 or gen == engine.generations - 1):
                    logger.info(f"Gen {gen:3d} | best_f={gen_best.fitness:.3f} "
                                f"depth={gen_best.depth} valid={gen_stats['valid_count']}")

                if gen < engine.generations - 1:
                    engine.population = _orig_next_gen(gen)

            return engine

        engine.run = _sr_run

        if verbose:
            logging.basicConfig(level=logging.INFO, format='[GP] %(message)s')

        engine.run(verbose=verbose)
        return engine

    finally:
        gp._random_tree = _orig_random_tree
        gp._MUTATE_OPS = _orig_mutate_weights


# ============================================================
# 5. 快捷工具
# ============================================================

def evaluate_formula(expr_str: str, X: Dict[str, np.ndarray]) -> np.ndarray:
    try:
        tree = ast.parse(expr_str, mode='eval')
    except SyntaxError:
        return np.full(len(next(iter(X.values()))), np.nan)
    tree = ast.Expression(body=tree.body)
    return _ExpressionFromAST(tree).evaluate(X)


def formula_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    yt, yp = np.asarray(y_true, float).ravel(), np.asarray(y_pred, float).ravel()
    v = ~(np.isnan(yt) | np.isnan(yp))
    if v.sum() < 10:
        return float('inf')
    return float(np.nanmean((yt[v] - yp[v]) ** 2))
