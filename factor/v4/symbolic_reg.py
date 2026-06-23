"""
factor/v4/symbolic_reg.py — 符号回归 (Symbolic Regression)

定位：因子发现之外的另一条腿。
  - 因子发现: 技术指标组合 → 选股/择时 (Sharpe)
  - 符号回归: 数学公式拟合 → 预测 (MSE)

核心思路：
  1. 替换 GP 的树生成器：从技术指标函数 → 数学原语
  2. 替换 fitness：从 Sharpe → 负 MSE
  3. 复用 GPEngine 的进化循环、交叉、变异

用法:
  >>> from factor.v4.symbolic_reg import symbolic_regression
  >>> engine = symbolic_regression(
  ...     X={'x1': arr1, 'x2': arr2},
  ...     y=target, generations=20,
  ... )
  >>> print(engine.best().expression_str)

  # 集成到 FactorExplorer:
  >>> ex = FactorExplorer()
  >>> engine = ex.symbolic_reg(X, y)
"""

import ast
import random
from typing import Dict

import numpy as np

from .gp_engine import GPEngine, Individual, _ExpressionFromAST, _replace_subtree
from .gp_engine import GP_CONSTANTS


# ============================================================
# 1. 符号回归树生成器（替换 _grow_tree 的技术指标版）
# ============================================================

# 终端: 用户传入的特征变量名，构造时设置
SR_VARIABLES = ['x1', 'x2', 'x3', 'x4', 'x5']

# 数学原语（一元）
SR_UNARY = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'tanh']


def _sr_terminal() -> ast.AST:
    """终端: 随机变量或常量"""
    if random.random() < 0.8:
        return ast.Name(id=random.choice(SR_VARIABLES), ctx=ast.Load())
    return ast.Constant(value=random.choice(GP_CONSTANTS))


def _sr_grow_tree(depth: int) -> ast.AST:
    """生成符号回归表达式树（数学原语，无技术指标）"""
    if depth <= 1:
        return _sr_terminal()

    r = random.random()

    if r < 0.25:
        # 终端（提前终止生长）
        return _sr_terminal()

    elif r < 0.50:
        # 二元运算: x + y, x * y, x - y, x / y
        left = _sr_grow_tree(depth - 1)
        right = _sr_grow_tree(depth - 1)
        op = random.choice([ast.Add(), ast.Sub(), ast.Mult(), ast.Div()])
        return ast.BinOp(left=left, op=op, right=right)

    elif r < 0.75:
        # 一元函数: sin(x), exp(x), log(x), sqrt(x), abs(x)
        fn = random.choice(SR_UNARY)
        arg = _sr_grow_tree(depth - 1)
        return ast.Call(
            func=ast.Name(id=fn, ctx=ast.Load()),
            args=[arg], keywords=[],
        )

    elif r < 0.90:
        # 二元函数嵌套: sin(x) + cos(y)
        left_fn = random.choice(SR_UNARY)
        right_fn = random.choice(SR_UNARY)
        op = random.choice([ast.Add(), ast.Sub(), ast.Mult()])
        return ast.BinOp(
            left=ast.Call(func=ast.Name(id=left_fn), args=[_sr_grow_tree(depth - 2)], keywords=[]),
            op=op,
            right=ast.Call(func=ast.Name(id=right_fn), args=[_sr_grow_tree(depth - 2)], keywords=[]),
        )

    else:
        # Compare: expr > 0 (隐式门控)
        left = _sr_grow_tree(depth - 1)
        return ast.Compare(
            left=left, ops=[random.choice([ast.Gt(), ast.Lt()])],
            comparators=[ast.Constant(value=0)],
        )


def _sr_random_tree(max_depth: int = 6) -> ast.Expression:
    """创建随机符号回归表达式树"""
    depth = random.randint(2, max_depth)
    body = _sr_grow_tree(depth)
    tree = ast.Expression(body=body)
    ast.fix_missing_locations(tree)
    return tree


# ============================================================
# 2. Fitness: 负 MSE（符号回归标准）
# ============================================================

class MSEFitness:
    """负均方误差 (GP 最大化)"""

    def __init__(self, y_true: np.ndarray):
        self.y_true = np.asarray(y_true, float)

    def compute(self, pred: np.ndarray) -> float:
        pred = np.asarray(pred, float)
        # 扩展 1D 预测到 2D
        if pred.ndim == 2 and self.y_true.ndim == 1:
            pred = pred[:, 0]
        if len(pred.shape) > 1:
            pred = pred.ravel()
        if len(self.y_true.shape) > 1:
            y = self.y_true.ravel()
        else:
            y = self.y_true
        # 对齐长度
        min_len = min(len(pred), len(y))
        pred, y = pred[:min_len], y[:min_len]
        # 过滤 NaN
        valid = ~(np.isnan(pred) | np.isnan(y))
        if valid.sum() < 10:
            return -999.0
        mse = np.nanmean((y[valid] - pred[valid]) ** 2)
        return -mse  # GP 最大化


# ============================================================
# 3. 入口函数
# ============================================================

def symbolic_regression(
    X: Dict[str, np.ndarray],
    y: np.ndarray,
    generations: int = 20,
    population_size: int = 200,
    max_depth: int = 8,
    random_seed: int = None,
    verbose: bool = True,
    **gp_kwargs,
) -> GPEngine:
    """运行符号回归

    Args:
        X: 特征字典 {'x1': ndarray(T,), 'x2': ndarray(T,), ...}
           特征名会作为 GP 终端变量出现在表达式中。
        y: 目标 ndarray(T,) — GP 拟合目标
        generations: 进化代数
        population_size: 种群大小
        max_depth: 树最大深度
        random_seed: 随机种子
        verbose: 是否打印进度
        **gp_kwargs: 传递给 GPEngine 的其他参数

    Returns:
        GPEngine (已运行), 通过 .best().expression_str 取最优公式
    """
    # 设置变量名
    global SR_VARIABLES
    SR_VARIABLES = list(X.keys())

    # 配置
    cfg = dict(gp_kwargs.get('config', {}))
    cfg.setdefault('population_size', population_size)
    cfg.setdefault('generations', generations)
    cfg.setdefault('max_depth', max_depth)
    cfg.setdefault('seed_ratio', 0.0)   # 符号回归无种子
    gp_kwargs['config'] = cfg
    gp_kwargs['random_seed'] = random_seed

    # 创建 GP 引擎
    engine = GPEngine(
        data=X,
        fitness_calculator=MSEFitness(y),
        seed_expressions=gp_kwargs.pop('seed_expressions', None),
        **gp_kwargs,
    )

    # 替换树生成器：符号回归版 vs 技术指标版
    engine._create_random_individual = lambda depth: Individual(
        tree=_sr_random_tree(depth), generation=0,
    )

    # 修改 _initialize_population 中的随机个体生成
    _orig_init = engine._initialize_population

    def _sr_init():
        _orig_init()
        # 将随机个体替换为符号回归树
        for i, ind in enumerate(engine.population):
            if not ind.is_seed and not (hasattr(ind, 'from_seed') and ind.is_seed):
                engine.population[i] = Individual(
                    tree=_sr_random_tree(max_depth), generation=0,
                )

    engine._initialize_population = _sr_init

    if verbose:
        import logging
        logging.basicConfig(level=logging.INFO, format='%(message)s')

    engine.run(verbose=verbose)
    return engine


# ============================================================
# 4. 快捷工具: 拟合评估
# ============================================================

def evaluate_formula(expr_str: str, X: Dict[str, np.ndarray]) -> np.ndarray:
    """对给定表达式求值（在 X 数据上计算）"""
    tree = ast.parse(expr_str, mode='eval')
    tree = ast.Expression(body=tree.body)
    ae = _ExpressionFromAST(tree)
    return ae.evaluate(X)


def formula_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """公式拟合 MSE"""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    valid = ~(np.isnan(y_true) | np.isnan(y_pred))
    if valid.sum() < 10:
        return float('inf')
    return float(np.nanmean((y_true[valid] - y_pred[valid]) ** 2))
