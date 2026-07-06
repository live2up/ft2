"""
utils/gp/v5/config.py — GP 配置、权重管理、个体定义
=============================================================================
[抽取] 2026-07-05 从 factor/v5/gp_engine.py 抽取，供 factor 和 signals 共享
"""

import ast
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, Optional

from utils.ast.dsl import parse_expression, ast_depth, ast_node_count

logger = logging.getLogger(__name__)

# ============================================================
# 配置
# ============================================================

GP_VARIABLES = ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'AMOUNT']
GP_CONSTANTS = [0.0, 0.5, 1.0, -1.0, 2.0, 0.01, 0.02, 0.05, 1.5, 3.0]

TS_FUNCTIONS = {
    'ts_rank': [5, 10, 20, 60], 'ts_zscore': [10, 20, 60],
    'ts_mean': [5, 10, 20, 60], 'ts_std': [10, 20, 60],
    'ts_sum': [5, 10, 20], 'ts_delta': [1, 5, 10, 20],
    'ts_delay': [1, 5, 10, 20], 'ts_roc': [5, 10, 20],
    'ts_decay_linear': [5, 10, 20], 'ts_skew': [20, 60],
    'ts_kurt': [20, 60], 'ts_resid': [10, 20],
    'ts_slope': [10, 20], 'ts_rsq': [10, 20],
    'ts_intercept': [10, 20], 'ts_predict': [10, 20],
}

TS_FUNCTIONS_2ARG = {
    'ts_corr': [10, 20, 60], 'ts_cov': [10, 20],
    'ts_reg_slope': [5, 10], 'ts_reg_resid': [5, 10],
    'ts_reg_rsq': [5, 10],
}

FEATURE_FUNCTIONS_1ARG = {'linearreg': [10, 20], 'tsf': [10, 20]}
FEATURE_FUNCTIONS_3ARG = {'natr': [5, 14], 'atr': [14]}
RATIO_FUNCTIONS = {'amt_ratio': [(5, 20)], 'vol_ratio': [(5, 20)]}

MATH_FUNCTIONS = ['sin', 'cos', 'exp', 'log', 'sqrt', 'abs', 'tanh', 'gauss', 'p4', 'neg']

# ============================================================
# TreeGenConfig — 树生成概率配置
# ============================================================

@dataclass
class TreeGenConfig:
    """树生成概率配置

    控制 GP 随机生树时变量/函数的选择偏置，引导搜索往特定方向聚焦。
    默认 None = 等概率 (完全兼容现有行为)。

    [重构] 2026-07-05 新增 rng 字段: 每个实例持有独立随机数生成器,
    消除模块级全局 random 状态, 支持多实例多线程并行 + 可复现性。

    用法:
        cfg = TreeGenConfig(var_weights={'AMOUNT': 3, 'VOLUME': 2})
        cfg = TreeGenConfig(var_allowlist={'AMOUNT', 'VOLUME'})
        cfg = TreeGenConfig(mode='continuous')
        cfg = TreeGenConfig(rng=random.Random(42))  # 可复现
    """
    mode: Optional[str] = None
    group_weights: Optional[Dict[str, float]] = None
    ts_weights: Optional[Dict[str, float]] = None
    math_weights: Optional[Dict[str, float]] = None
    feature_weights: Optional[Dict[str, float]] = None
    var_weights: Optional[Dict[str, float]] = None
    var_allowlist: Optional[set] = None
    func_allowlist: Optional[set] = None
    adaptive: bool = False
    adaptive_lr: float = 0.3
    adaptive_every: int = 3
    rng: random.Random = field(default_factory=random.Random)

DEFAULT_TREE_GEN_CONFIG = TreeGenConfig(
    group_weights={'ts_function':25, 'feature_function':13, 'math_function':12,
                   'comparison':13, 'logic':11, 'binary_op':12,
                   'unary_op':9, 'ternary':5},
    ts_weights={fn: 1.0 for fn in TS_FUNCTIONS},
    math_weights={fn: 1.0 for fn in MATH_FUNCTIONS},
    feature_weights={'feature_1arg': 0.5, 'feature_3arg': 0.3, 'ratio': 0.2},
)

_FILL_VAR_KEYS = GP_VARIABLES[:]
_FILL_TS_KEYS = list(TS_FUNCTIONS.keys())
_FILL_MATH_KEYS = MATH_FUNCTIONS[:]
_FILL_FEATURE_KEYS = list(DEFAULT_TREE_GEN_CONFIG.feature_weights.keys())
_FILL_GROUP_KEYS = list(DEFAULT_TREE_GEN_CONFIG.group_weights.keys())

def _fill_weights(user_weights, default_keys, fill_value=0):
    """填充权重: 用户设了的用用户值，没设的填 fill_value (默认 0=禁止)"""
    filled = {}
    for key in default_keys:
        if user_weights and key in user_weights:
            filled[key] = user_weights[key]
        else:
            filled[key] = fill_value
    if user_weights:
        for key in user_weights:
            if key not in filled:
                filled[key] = user_weights[key]
    nonzero = {k: v for k, v in filled.items() if v > 0}
    if not nonzero and filled:
        nonzero = {k: 1.0 for k in filled}
    return nonzero

DEFAULT_GP_CONFIG = {
    'population_size': 200, 'generations': 20, 'max_depth': 10,
    'tournament_size': 5, 'crossover_prob': 0.6, 'mutation_prob': 0.25,
    'elite_ratio': 0.05, 'seed_ratio': 0.4, 'random_inject_ratio': 0.05,
    'parsimony_penalty': 0.001,
    'mutate_subtree_weight': 0.30, 'mutate_constant_weight': 0.20,
    'mutate_window_weight': 0.20, 'mutate_logic_weight': 0.15,
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
            tree=tree, expression_str=expr_str,
            depth=ast_depth(tree), node_count=ast_node_count(tree),
            generation=generation, is_seed=is_seed,
        )
