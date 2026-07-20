"""
utils/gp/v5/config.py — GP 配置、权重管理、个体定义
=============================================================================
[抽取] 2026-07-05 从 factor/v5/gp_engine.py 抽取，供 factor 和 signals 共享
"""

import ast
import random
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from utils.ast.v2.dsl import parse_expression, ast_depth, ast_node_count

logger = logging.getLogger(__name__)

# ============================================================
# 基础池
# ============================================================

# [重构] 2026-07-08 GP 默认变量/常数池
# GP_VARIABLES: fallback 变量池 — 仅当用户未设 var_weights 且未设 var_allowlist 时使用。
#   自定义变量不经此池: 通过 ast.register_variable() 注册合法性 + var_weights 指定生成偏置。
#   ast 是单一事实来源 (变量合法性 + 函数元数据), GP 仅消费 ast 注册表。
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
    # [新增] 2026-07-05 模式过滤: continuous=纯数值, predicate=条件信号, hybrid=全空间
    mode: Optional[str] = None
    # [权重] 结构组权重: 控制 ts/math/comparison/logic/binary/unary/ternary 的出现概率
    group_weights: Optional[Dict[str, float]] = None
    # [权重] 时序函数池: key=函数名, value=生成偏置。未声明的函数默认屏蔽 (fill_value=0)
    ts_weights: Optional[Dict[str, float]] = None
    # [权重] 数学函数池: key=函数名, value=生成偏置。未声明的函数默认屏蔽
    math_weights: Optional[Dict[str, float]] = None
    # [权重] 变量池: key=变量名, value=生成偏置。未声明的变量默认屏蔽
    var_weights: Optional[Dict[str, float]] = None
    # [白名单] 变量白名单: 仅从该集合采样，与 var_weights 取交集
    var_allowlist: Optional[set] = None
    # [白名单] 函数白名单: 仅从该集合采样，与 ts_weights/math_weights 取交集
    func_allowlist: Optional[set] = None
    # [自适应] 是否启用方向追踪 + EMA 权重更新
    adaptive: bool = False
    # [自适应] EMA 学习率: 越大权重更新越快 (默认 0.3)
    adaptive_lr: float = 0.3
    # [自适应] 每 N 代触发一次 EMA 更新 (默认 3)
    adaptive_every: int = 3
    # [复现] 每个实例独立 RNG，支持多线程并行和结果复现
    rng: random.Random = field(default_factory=random.Random)

DEFAULT_TREE_GEN_CONFIG = TreeGenConfig(
    # [权重] 结构组默认权重: 控制 7 类 AST 结构的生成概率
    # ts_function 最高(30) → 优先生成函数调用; ternary 最低(5) → 最少出现
    group_weights={'ts_function':30, 'math_function':15,
                   'comparison':15, 'logic':13, 'binary_op':13,
                   'unary_op':9, 'ternary':5},
    # [重构] 2026-07-07 移除 feature_weights，所有函数统一由 ts_weights/math_weights 控制。
    # ts_weights/math_weights 默认为空，由 get_full_default_weights() 从 FUNC_REGISTRY 动态填充。
    ts_weights={},
    math_weights={},
)

_FILL_GROUP_KEYS = list(DEFAULT_TREE_GEN_CONFIG.group_weights.keys())


def _filter_funcs_by_var_scope(ts_weights: dict, math_weights: dict,
                                var_allowlist: set) -> tuple:
    """自动排除 data_vars 超限的函数。

    [新增] 2026-07-20 配合 var_allowlist 使用：
    当用户限定变量范围时，自动禁用内部注入超限变量的函数。
    例：var_allowlist={'REL_HIGH'}, atr_sma 的 data_vars=['HIGH','LOW','CLOSE']
    → LOW/CLOSE 不在 allowlist → atr_sma 权重置 0。

    这是严格模式——var_allowlist 是硬白名单，一切变量出入都必须遵守。
    如果要逐步开放多类变量，应使用 var_weights 权重聚焦（非黑名单式的偏置），
    而非 var_allowlist：
      - var_allowlist → 严格单变量/少变量探索
      - var_weights   → 偏置但不禁止，自然过渡到多变量

    原理：_build_call_args 用 data_vars 直接注入变量，绕过 _random_variable
    的 allowlist 检查。与其在生成层拦截，不如直接从函数池排除。
    """
    if not var_allowlist:
        return ts_weights, math_weights

    from utils.ast.v2.registry import FUNC_REGISTRY

    def _in_scope(func_name: str) -> bool:
        spec = FUNC_REGISTRY.get(func_name)
        if spec and spec.data_vars:
            return all(v in var_allowlist for v in spec.data_vars)
        return True  # 无 data_vars 的函数不受限

    filtered_ts = {}
    excluded_ts = []
    for f, w in (ts_weights or {}).items():
        if w > 0 and not _in_scope(f):
            excluded_ts.append(f)
        else:
            filtered_ts[f] = w

    filtered_math = {}
    excluded_math = []
    for f, w in (math_weights or {}).items():
        if w > 0 and not _in_scope(f):
            excluded_math.append(f)
        else:
            filtered_math[f] = w

    if excluded_ts or excluded_math:
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"[GP] var_allowlist={var_allowlist} → "
            f"自动排除函数: {sorted(excluded_ts + excluded_math)}"
        )

    return filtered_ts, filtered_math


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


def _get_funcs_by_group(group_name: str) -> list:
    """从 FUNC_CATEGORIES 获取指定 group_weights 大类下的函数名列表。"""
    from utils.ast.v2.registry import FUNC_CATEGORIES
    return sorted(set(FUNC_CATEGORIES.get(group_name, [])))


def get_full_default_weights() -> dict:
    """获取包含自定义注册函数的完整默认权重配置。

    在 DEFAULT_TREE_GEN_CONFIG 基础上，合并通过 register_function 注册的自定义函数。
    自定义函数默认权重 1.0。ts_weights 涵盖 ts/cs/ta/feature_function，
    math_weights 涵盖 math_function，var_weights 涵盖 GP_VARIABLES。
    """
    ts_w = dict(DEFAULT_TREE_GEN_CONFIG.ts_weights)
    math_w = dict(DEFAULT_TREE_GEN_CONFIG.math_weights)
    var_w = dict(DEFAULT_TREE_GEN_CONFIG.var_weights) if DEFAULT_TREE_GEN_CONFIG.var_weights else {}

    for name in _get_funcs_by_group('ts_function'):
        ts_w.setdefault(name, 1.0)
    for name in _get_funcs_by_group('cs_function'):
        ts_w.setdefault(name, 1.0)
    for name in _get_funcs_by_group('ta_function'):
        ts_w.setdefault(name, 1.0)
    for name in _get_funcs_by_group('feature_function'):
        ts_w.setdefault(name, 1.0)
    for name in _get_funcs_by_group('signal_function'):
        ts_w.setdefault(name, 1.0)
    for name in _get_funcs_by_group('math_function'):
        math_w.setdefault(name, 1.0)
    for name in GP_VARIABLES:
        var_w.setdefault(name, 1.0)

    return {
        'group_weights': DEFAULT_TREE_GEN_CONFIG.group_weights,
        'ts_weights': ts_w,
        'math_weights': math_w,
        'var_weights': var_w,
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
                _get_funcs_by_group('feature_function') +
                _get_funcs_by_group('signal_function'))
    elif field_name == 'math_weights':
        return _get_funcs_by_group('math_function')
    elif field_name == 'var_weights':
        return list(GP_VARIABLES)
    elif field_name == 'group_weights':
        return list(_FILL_GROUP_KEYS)
    return []


# ============================================================
# GP 默认配置
# ============================================================

DEFAULT_GP_CONFIG = {
    # [种群] 每代个体数 (默认 200，kb03 使用 500 提升探索)
    'population_size': 500,
    # [迭代] 进化代数 (默认 20，kb03 使用 40 确保收敛)
    'generations': 40,
    # [深度] 树最大深度 (默认 10，kb03 使用 5 防止过拟合、保持可解释)
    'max_depth': 5,
    # [选择] 锦标赛大小 (每次随机抽取 6 个，选 fitness 最高)
    'tournament_size': 6,
    # [交叉] 交叉概率 (60% 后代来自交叉)
    'crossover_prob': 0.6,
    # [变异] 变异概率 (25% 后代来自变异)
    'mutation_prob': 0.25,
    # [精英] 精英比例 (前 5% 直接进入下一代，不变化)
    'elite_ratio': 0.05,
    # [种子] 种子个体比例 (15% 来自用户提供的 seed_expressions)
    'seed_ratio': 0.15,
    # [注入] 随机注入比例 (15% 完全随机生成，维持多样性)
    'random_inject_ratio': 0.15,
    # [并行] 默认 4 线程评估个体 (n_workers=min(workers, unevaluated))
    'parallel_workers': 4,
    # [惩罚] 复杂度惩罚系数 (fitness *= (1 - penalty * node_count))
    'parsimony_penalty': 0.001,
    # [选择器] True=使用 ε-Lexicase 选择 (保留方向多样性)，False=锦标赛选择
    'lexicase': True,
    # [ε-lexicase] ε 阈值 (0.05 = 允许 fitness 差距 5% 内的个体同为代表)
    'epsilon': 0.05,
    # [年龄] True=启用年龄机制，fitness 随年龄衰减，防止种群老龄化
    'age_enabled': True,
    # [年龄] 年龄惩罚系数 (fitness *= 1/(1 + lr*age)，默认 0.05)
    'age_penalty_lr': 0.05,
    # [Motif] True=启用 Motif 库，提取高频子树作为种子
    'motif_enabled': True,
    # [Motif] 每 N 代更新一次 Motif 库
    'motif_update_every': 3,
    # [Motif] 仅提取 fitness >= 此阈值的个体的子树
    'motif_min_fitness': 0.0,
    # [Motif] Motif 子树最大深度（默认 max_depth//2）
    'motif_max_depth_ratio': 0.5,
    # [Motif] 每次初始化时最多注入多少 motif 种子
    'motif_inject_count': 5,
    # [变异] 子树替换权重 (30% 变异操作中子树替换)
    'mutate_subtree_weight': 0.30,
    # [变异] 参数变异权重 (40% 变异操作中重新采样参数: 窗口/常数)
    'mutate_param_weight': 0.40,
    # [变异] 逻辑变异权重 (15% 变异操作中 and↔or / 加/删 not)
    'mutate_logic_weight': 0.15,
    # [变异] 条件插入权重 (15% 变异操作中用 if-else/and/or 包装子树)
    'mutate_insert_cond_weight': 0.15,

    # [岛屿] 岛屿模式参数
    'num_islands': 1,          # 岛屿数量 (1=禁用岛屿模式)
    'migrate_every': 5,        # 每 N 代迁移一次
    'migrate_count': 2,        # 每岛每次迁出 Top-k
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
    age: int = 1  # [新增] 2026-07-08 年龄: 个体存活代数，用于年龄惩罚
    signature: str = ''  # [新增] 2026-07-08 缓存 _expr_signature 结果，避免重复 regex

    @staticmethod
    def from_expr(expr_str: str, generation: int = 0, is_seed: bool = False) -> 'Individual':
        """从表达式字符串创建个体"""
        tree = parse_expression(expr_str)
        return Individual(
            tree=tree, expression_str=expr_str,
            depth=ast_depth(tree), node_count=ast_node_count(tree),
            generation=generation, is_seed=is_seed,
        )


# ============================================================
# GenerationSnapshot — 每代预计算快照
# ============================================================

@dataclass
class GenerationSnapshot:
    """每代评价完成后预计算的快照，后代选择只读。

    [新增] 2026-07-08 替代零散的 _sig_index / lexicase 暴力计算。
    run() 中一代只构造一次，后代 _lexicase_select / motif / 方向更新
    直接读快照，避免每代 ~425 次重复遍历。
    """
    generation: int
    valid: List[Individual]                             # 有效个体
    sorted_valid: List[Individual]                      # 按 fitness 降序
    by_signature: Dict[str, List[Individual]]           # 方向→个体索引
    sig_best: Dict[str, Tuple[float, str]]              # 方向→(best_fitness, best_expr)
    lexicase_pool: List[Individual]                     # ε-lexicase 代表池（预去重）


# ============================================================
# AURORA 扩展 (v6)
# ============================================================

@dataclass
class EncodedIndividual:
    """AURORA 个体: Individual + 潜在向量 φ + 因子统计摘要

    [新增] 2026-07-20 v6 AURORA 引擎。在标准 Individual 基础上
    附加编码器产出的低维行为描述子和因子统计特征。
    因子值是 ndarray，不存入 dataclass（内存过大），仅在评估时暂存。
    """
    tree: 'ast.Expression'
    fitness: float = -999.0
    expression_str: str = ''
    depth: int = 0
    node_count: int = 0
    generation: int = 0
    is_seed: bool = False
    age: int = 1
    signature: str = ''
    phi: Optional['np.ndarray'] = None      # 潜在向量 (latent_dim,)
    stats: Optional['np.ndarray'] = None    # 因子统计摘要 (input_dim,)

    @staticmethod
    def from_individual(ind: Individual) -> 'EncodedIndividual':
        """从标准 Individual 升级为 EncodedIndividual"""
        import numpy as np
        return EncodedIndividual(
            tree=ind.tree, fitness=ind.fitness,
            expression_str=ind.expression_str,
            depth=ind.depth, node_count=ind.node_count,
            generation=ind.generation, is_seed=ind.is_seed,
            age=ind.age, signature=ind.signature,
        )

    @staticmethod
    def from_expr(expr_str: str, generation: int = 0, is_seed: bool = False) -> 'EncodedIndividual':
        """从表达式字符串创建个体 (对齐 Individual.from_expr)"""
        import numpy as np
        tree = parse_expression(expr_str)
        return EncodedIndividual(
            tree=tree, expression_str=expr_str,
            depth=ast_depth(tree), node_count=ast_node_count(tree),
            generation=generation, is_seed=is_seed,
        )


# AURORA 默认配置
AURA_DEFAULT_CONFIG = {
    # [Archive] 非结构化 archive 容量
    'archive_size': 300,
    # [Archive] DNS 局部竞争的邻居数 k
    'dns_k_neighbors': 10,
    # [Archive] 每个签名最多占用槽位数 (0=不限制) — sig_quota 签名多样性保护
    'sig_quota': 5,
    # [编码器] 输入维度 (因子统计特征数)
    'encoder_input_dim': 8,
    # [编码器] 潜在空间维度
    'encoder_latent_dim': 6,
    # [编码器] 隐藏层维度
    'encoder_hidden_dim': 16,
    # [编码器] 学习率
    'encoder_lr': 0.01,
    # [编码器] 每代训练步数
    'encoder_steps_per_update': 50,
    # [编码器] 重训间隔 (代数)
    'encoder_update_every': 5,
    # [编码器] 损失类型: 'mse' (自编码器重建) 或 'triplet' (三元组)
    'encoder_loss_type': 'mse',
    # [灭绝] 触发周期 (代数)
    'extinction_period': 30,
    # [灭绝] 保留比例 (含精英)
    'extinction_retention': 0.10,
    # [生成] 每代产生的子代数 (替换旧版的 population_size)
    'children_per_gen': 200,
}
