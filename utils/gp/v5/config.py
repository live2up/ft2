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

# [重构] 2026-07-08 GP 默认变量/常数池
# GP_VARIABLES: fallback 变量池 — 仅当用户未设 var_weights 且未设 var_allowlist 时使用.
#   自定义变量不经此池: 通过 ast.register_variable() 注册合法性 + var_weights 指定生成偏置.
#   ast 是单一事实来源 (变量合法性 + 函数元数据), GP 仅消费 ast 注册表.
# GP_CONSTANTS: GP 随机生成常数节点的数值池 (与 ast SAFE_CONSTANTS 不同:
#   SAFE_CONSTANTS 是解析时的命名常量 True/pi/e, GP_CONSTANTS 是生成时的数值候选)
GP_VARIABLES = ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME', 'AMOUNT']
GP_CONSTANTS = [0.0, 0.5, 1.0, -1.0, 2.0, 0.01, 0.02, 0.05, 1.5, 3.0]

# [重构] 2026-07-07 移除 TS_FUNCTIONS / MATH_FUNCTIONS 等硬编码映射，
# 改从 utils.ast.functions.FUNC_REGISTRY 动态读取。
# GP 所需的所有函数元数据（分类 / 参数个数 / 参数池）统一由注册表管理。

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
    var_weights: Optional[Dict[str, float]] = None
    var_allowlist: Optional[set] = None
    func_allowlist: Optional[set] = None
    adaptive: bool = False
    adaptive_lr: float = 0.3
    adaptive_every: int = 3
    rng: random.Random = field(default_factory=random.Random)

DEFAULT_TREE_GEN_CONFIG = TreeGenConfig(
    group_weights={'ts_function':30, 'math_function':15,
                   'comparison':15, 'logic':13, 'binary_op':13,
                   'unary_op':9, 'ternary':5},
    # [重构] 2026-07-07 移除 feature_weights，所有函数统一由 ts_weights/math_weights 控制。
    # ts_weights/math_weights 默认为空，由 get_full_default_weights() 从 FUNC_REGISTRY 动态填充。
    ts_weights={},
    math_weights={},
)

_FILL_VAR_KEYS = GP_VARIABLES[:]
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


# [重构] 2026-07-07 从 FUNC_CATEGORIES 动态读取函数分类，自动适配 group_weights
# feature_function 合并到 ts_weights（ta_function 同类处理）
_FUNCTION_GROUP_MAP = {
    'ts_function': ('ts_weights', [10, 20]),
    'cs_function': ('ts_weights', [10, 20]),
    'ta_function': ('ts_weights', [10, 20]),
    'feature_function': ('ts_weights', [10, 20]),
    'math_function': ('math_weights', None),
    # comparison / logic / binary_op / unary_op / ternary 无子权重池
}


def _get_funcs_by_group(group_name: str) -> list:
    """从 FUNC_CATEGORIES 获取指定 group_weights 大类下的函数名列表。"""
    from utils.ast.functions import FUNC_CATEGORIES
    return sorted(set(FUNC_CATEGORIES.get(group_name, [])))


def get_full_default_weights() -> dict:
    """获取包含自定义注册函数的完整默认权重配置。

    在 DEFAULT_TREE_GEN_CONFIG 基础上，合并通过 register_function 注册的自定义函数。
    自定义函数默认权重 1.0。ts_weights 涵盖 ts/cs/ta/feature_function，
    math_weights 涵盖 math_function。
    """
    ts_w = dict(DEFAULT_TREE_GEN_CONFIG.ts_weights)
    math_w = dict(DEFAULT_TREE_GEN_CONFIG.math_weights)

    for name in _get_funcs_by_group('ts_function'):
        ts_w.setdefault(name, 1.0)
    for name in _get_funcs_by_group('cs_function'):
        ts_w.setdefault(name, 1.0)
    for name in _get_funcs_by_group('ta_function'):
        ts_w.setdefault(name, 1.0)
    for name in _get_funcs_by_group('feature_function'):
        ts_w.setdefault(name, 1.0)
    for name in _get_funcs_by_group('math_function'):
        math_w.setdefault(name, 1.0)

    return {
        'group_weights': DEFAULT_TREE_GEN_CONFIG.group_weights,
        'ts_weights': ts_w,
        'math_weights': math_w,
    }


def _get_fill_keys(field_name: str) -> list:
    """按权重字段名获取默认 key 集合（含自定义注册函数），替代废弃的 _FILL_TS_KEYS 等。

    由 engine.py _build_tree_config 在合并用户权重时调用，
    从 FUNC_REGISTRY 动态读取，确保自定义注册函数也被纳入。
    """
    if field_name == 'ts_weights':
        return (_get_funcs_by_group('ts_function') +
                _get_funcs_by_group('cs_function') +
                _get_funcs_by_group('ta_function') +
                _get_funcs_by_group('feature_function'))
    elif field_name == 'math_weights':
        return _get_funcs_by_group('math_function')
    elif field_name == 'var_weights':
        return list(GP_VARIABLES)
    elif field_name == 'group_weights':
        return list(_FILL_GROUP_KEYS)
    return []


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
