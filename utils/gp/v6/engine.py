"""
utils/gp/v5/engine.py — 遗传编程引擎核心 (AST 原生)
=============================================================================
[抽取] 2026-07-05 从 factor/v5/gp_engine.py 抽取，供 factor 和 signals 共享。
[重构] 2026-07-06 拆分为 ast_utils / tree_gen / cache / engine 四文件。
"""
import ast
import copy
import random
import logging
import threading
import itertools
import numpy as np
import os
from typing import Dict, List, Optional, Callable
from dataclasses import fields as dataclass_fields

from utils.ast.v2.dsl import parse_expression, ast_depth, ast_node_count
from .config import (
    GP_VARIABLES,
    TreeGenConfig, Individual, GenerationSnapshot,
    DEFAULT_TREE_GEN_CONFIG, DEFAULT_GP_CONFIG,
    _fill_weights, _filter_funcs_by_var_scope,
    get_full_default_weights, _get_funcs_by_group, _get_fill_keys,
)
from .ast_utils import (
    _expr_str, _collect_replaceable, _replace_subtree, _simplify_ast, _canonicalize_key,
    _extract_subtrees,
)
from .tree_gen import (
    _grow_tree, _random_tree, _random_tree_explore,
    _mutate_subtree, _mutate_param,
    _mutate_logic, _mutate_insert_condition,
)
from .cache import FitnessCache

logger = logging.getLogger(__name__)


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
                 tree_gen_config: TreeGenConfig = None,
                 evaluator: Optional[Callable] = None,
                 future_returns=None, returns=None,
                 cache_db: str = '',
                 source: str = ''):
        self.data = data
        self.future_returns = future_returns
        self.returns = returns
        self.fitness_calc = fitness_calculator
        self._evaluator = evaluator
        self._source = source

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
        # [删除] 2026-07-21 _save_random_inject — 仅被已删除的 _adapt_operators 使用
        self.parsimony_penalty = cfg['parsimony_penalty']

        # [新增] 2026-07-08 年龄机制: 防止种群老龄化，保持多样性
        self._age_enabled: bool = cfg.get('age_enabled', False)
        self._age_penalty_lr: float = cfg.get('age_penalty_lr', 0.05)

        # 变异算子权重
        # [重构] 2026-07-08 _mutate_constant + _mutate_window → _mutate_param
        # [修复] 2026-07-08 mode='continuous' 时自动屏蔽 logic/insert_condition 算子
        w = cfg
        mode = tree_gen_config.mode if tree_gen_config else 'hybrid'
        _logic_w = 0.0 if mode == 'continuous' else w['mutate_logic_weight']
        _cond_w = 0.0 if mode == 'continuous' else w['mutate_insert_cond_weight']
        self._mutate_weights = [
            (w['mutate_subtree_weight'], _mutate_subtree),
            (w['mutate_param_weight'], _mutate_param),
            (_logic_w, _mutate_logic),
            (_cond_w, _mutate_insert_condition),
        ]

        # 种子
        self.seed_expressions = seed_expressions or []

        # TreeGenConfig 构建
        rng = random.Random(random_seed) if random_seed is not None else random.Random()
        self.tree_gen_config = self._build_tree_config(tree_gen_config, rng)

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

        # 缓存 (SQLite + 内存)
        self._parallel_workers = config.get('parallel_workers', 4) if config else 4
        self._canonicalize_memo: Dict[str, str] = {}
        self._canonicalize_lock = threading.Lock()
        # [重构] 2026-07-08 缓存路径独立参数，不再从 config 读取，避免与算法参数混淆。
        # 优先级: cache_db 显式参数 > config['cache_db'] > 默认工作目录/output/.gp_cache.db
        cache_db = cache_db or (config.get('cache_db', '') if config else '')
        if not cache_db:
            cache_db = os.path.join(os.getcwd(), 'output', '.gp_cache.db')
        cache_dir = os.path.dirname(cache_db)
        if cache_dir:
            os.makedirs(cache_dir, exist_ok=True)
        fitness_hash = ''
        import hashlib
        fingerprint = f"{self._shape}_{self.parsimony_penalty:.4f}"
        fitness_hash = hashlib.md5(fingerprint.encode()).hexdigest()[:12]
        # [新增] 2026-07-10 溯源信息: source + session_id
        session_id = ''
        if self._source:
            seed_val = random_seed if random_seed is not None else 0
            session_id = f"{self._source}_seed{seed_val}"
        self.fitness_cache = FitnessCache(cache_db, fitness_hash,
                                          source=self._source,
                                          session_id=session_id)

        # [删除] 2026-07-21 清理 v5 死代码:
        #   方向演化追踪/stagnation检测/ε-Lexicase/Motif/快照/岛屿模型
        #   v6 用 AURORA Archive + DNS + 方向感知选择替代

        # 但仍保留 _use_lexicase 标志:
        #   AURORAEngine._select_parent 在 archive 空时 fallback 到 GPEngine._select_parent,
        #   需要此标志决定走 tournament 还是 lexicase
        self._use_lexicase: bool = config.get('lexicase', False) if config else False
        self._num_islands: int = config.get('num_islands', 1) if config else 1

        # ε-greedy 探索 (仍被 _fill_one_population → _initialize_archive 使用)
        self._explore_ratio: float = config.get('explore_ratio', 0.15) if config else 0.15

    # ── 配置构建 ──

    def _build_tree_config(self, user_config: TreeGenConfig = None,
                           rng: random.Random = None) -> TreeGenConfig:
        # [新增] 2026-07-07 自定义注册函数也纳入默认权重池
        full_defaults = get_full_default_weights()
        if user_config is None:
            # [修复] 2026-07-08 移除已删除的 feature_weights 引用
            return TreeGenConfig(
                group_weights=full_defaults['group_weights'],
                ts_weights=full_defaults['ts_weights'],
                math_weights=full_defaults['math_weights'],
                var_weights=full_defaults.get('var_weights', {}),
                rng=rng or random.Random(),
            )
        filled = {}
        for field in dataclass_fields(TreeGenConfig):
            if field.name == 'rng':
                filled[field.name] = rng or user_config.rng or random.Random()
                continue
            user_val = getattr(user_config, field.name)
            if user_val is None:
                # [简化] 2026-07-08 统一回退到 full_defaults
                default_val = full_defaults.get(field.name)
                if default_val is not None:
                    filled[field.name] = default_val
                else:
                    filled[field.name] = getattr(DEFAULT_TREE_GEN_CONFIG, field.name)
            elif field.name in ('mode', 'var_allowlist', 'func_allowlist',
                                'adaptive', 'adaptive_lr', 'adaptive_every'):
                filled[field.name] = user_val
            else:
                # [重构] 2026-07-07 用 _get_fill_keys 动态获取 key 集合，不再依赖 _FILL_TS_KEYS 等硬编码常量
                key_set = _get_fill_keys(field.name)
                filled[field.name] = _fill_weights(user_val, key_set)
        # [新增] 2026-07-20 变量范围自动过滤：var_allowlist 有值时，
        # 排除 data_vars 超限的函数（如 atr_sma 需要 HIGH/LOW/CLOSE）
        if filled.get('var_allowlist'):
            filled['ts_weights'], filled['math_weights'] = _filter_funcs_by_var_scope(
                filled['ts_weights'], filled['math_weights'], filled['var_allowlist']
            )
        return TreeGenConfig(**filled)

    # ── 适应度评估 ──

    def _quick_filter(self, ind: Individual) -> bool:
        """预筛: 拒绝明显无效的表达式"""
        expr = ind.expression_str or _expr_str(ind.tree)
        import re
        if re.search(r'ts_\w+\([^,]+, 1\)', expr):
            return False
        vars_found = set(re.findall(r'\b[A-Z][A-Z_0-9]+\b', expr))
        if not vars_found:
            return False
        funcs_found = set(re.findall(r'([a-z_]+)\(', expr))
        if not funcs_found and len(vars_found) <= 2:
            return False
        if re.search(r'ts_\w+\(-?\d+\.?\d*,', expr):
            return False
        return True

    def _evaluate_individual(self, ind: Individual) -> float:
        if not ind.expression_str:
            ind.expression_str = _expr_str(ind.tree)
        key = _canonicalize_key(ind.tree, ind.expression_str,
                                self._canonicalize_memo, self._canonicalize_lock)

        cached = self.fitness_cache.get(key)
        if cached is not None:
            ind.fitness, ind.depth, ind.node_count = cached
            raw_fitness = cached[0]
            # [修复] 2026-07-21 缓存命中时仍需求值，供 AURORA 编码阶段复用
            # 不调 evaluator → _factor_values 缺失 → 种子被 archive 误拒
            try:
                if self._evaluator:
                    ind._factor_values = self._evaluator(self.data, ind.tree)
            except Exception:
                ind._factor_values = None
        else:
            if not self._quick_filter(ind):
                return -999.0

            try:
                if self._evaluator:
                    factor_values = self._evaluator(self.data, ind.tree)
                    # [新增] 2026-07-21 缓存因子值，供 AURORA 编码阶段复用，省掉重复求值
                    ind._factor_values = factor_values
                else:
                    return -999.0
            except Exception as e:
                if ind.is_seed:
                    logger.warning(f"[GP] 种子求值异常: {e}")
                return -999.0

            # [修复] 2026-07-20 预热期 NaN 问题：
            # ts_kurt(HIGH,21) → 前20天窗口不足 → NaN。
            # 原 isfinite().all() 一刀切 → ts_kurt/skew/rsq 整族被误杀。
            # 正确逻辑：只要求有效值占比 ≥ 50%（排除全预热/全停牌表达式），
            # 具体 NaN 语义（预热/停牌）由下游 compute() + FacEngine start_date 分层处理。
            fv = np.asarray(factor_values, dtype=float)
            valid_mask = np.isfinite(fv)
            valid_ratio = valid_mask.sum() / valid_mask.size
            if valid_ratio < 0.5:
                return -999.0  # 一半以上 NaN → 大概率是窗口过大或全停牌

            # 常数值检查：只看有效值区域（排除预热NaN干扰）
            valid_vals = fv[valid_mask]
            if np.allclose(valid_vals, valid_vals[0], atol=1e-10):
                return -999.0  # 全相同 → 无截面区分力

            fitness = self.fitness_calc.compute(factor_values)
            if not np.isfinite(fitness) or fitness < -998.0:
                return -999.0

            ind.depth = ast_depth(ind.tree)
            ind.node_count = ast_node_count(ind.tree)
            penalty = self.parsimony_penalty * ind.node_count
            raw_fitness = fitness * (1.0 - penalty)
            self.fitness_cache.put(key, (raw_fitness, ind.depth, ind.node_count))

        # [新增] 2026-07-08 年龄惩罚: 随代数衰减，防止种群老龄化
        if self._age_enabled and raw_fitness > -999.0:
            age_penalty = 1.0 / (1.0 + self._age_penalty_lr * ind.age)
            raw_fitness = raw_fitness * age_penalty

        ind.fitness = raw_fitness
        return raw_fitness

    def _evaluate_population(self):
        unevaluated = [ind for ind in self.population if ind.fitness == -999.0]
        if not unevaluated:
            return

        if self._parallel_workers > 1 and len(unevaluated) >= 10:
            self._evaluate_parallel(unevaluated)
        else:
            for ind in unevaluated:
                self._evaluate_individual(ind)

    def _evaluate_parallel(self, individuals: List[Individual]):
        from concurrent.futures import ThreadPoolExecutor, as_completed
        for ind in individuals:
            if not ind.expression_str:
                ind.expression_str = _expr_str(ind.tree)

        n_workers = min(self._parallel_workers, len(individuals))
        futures_map = {}

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            for ind in individuals:
                key = _canonicalize_key(ind.tree, ind.expression_str,
                                self._canonicalize_memo, self._canonicalize_lock)
                if self.fitness_cache.get(key) is not None:
                    cached = self.fitness_cache.get(key)
                    ind.fitness, ind.depth, ind.node_count = cached
                    continue
                futures_map[executor.submit(
                    self._evaluate_individual, ind
                )] = ind

            for future in as_completed(futures_map):
                try:
                    future.result()
                except Exception as e:
                    logger.debug(f"[GP] 并行求值异常: {e}")

    # ── 方向追踪 ──

    @staticmethod
    def _structural_skeleton(expr_str: str) -> str:
        """结构骨架：变量→X，常数→C，保留嵌套关系。"""
        import re
        skel = re.sub(r'\b[A-Z][A-Z_0-9]+\b', 'X', expr_str)
        return re.sub(r'\b\d+(?:\.\d+)?\b', 'C', skel)

    @staticmethod
    def _expr_signature(expr_str: str) -> str:
        if not expr_str:
            return "unknown"
        import re
        from collections import Counter
        from utils.ast.v2.registry import FUNC_REGISTRY
        funcs = set(re.findall(r'([a-z_]+)\(', expr_str))
        math_sig = '+'.join(sorted(
            f for f in funcs
            if f in FUNC_REGISTRY and FUNC_REGISTRY[f].category == 'math_function'
        )) or 'none'
        ts_sig = '+'.join(sorted(
            f for f in funcs
            if f in FUNC_REGISTRY and FUNC_REGISTRY[f].category in (
                'ts_function', 'cs_function', 'ta_function', 'feature_function',
                'signal_function')
        )) or 'none'
        var_counts = Counter(re.findall(r'\b([A-Z][A-Z_0-9]+)\b', expr_str))
        vars_sig = '+'.join(f'{v}:{c}' for v, c in var_counts.most_common(4)) or 'none'
        # [新增] 2026-07-20 结构骨架: 变量→X, 常数→C, 保留嵌套关系
        # 让 Lexicase 按树形分组而非仅按函数名，解决单变量收敛到同一结构的问题
        skel = GPEngine._structural_skeleton(expr_str)
        if len(skel) > 80:
            import hashlib
            skel = skel[:40] + '..' + hashlib.md5(skel.encode()).hexdigest()[:8]
        return f"m:{math_sig}|t:{ts_sig}|v:{vars_sig}|s:{skel}"

    def direction_report(self, min_fitness: float = 0.3) -> str:
        if not self.direction_log:
            return "无方向记录"
        lines = ["", "=" * 50, "方向探索报告 (★=fit≥0.8   =0.5+ ·=0.3+)", "=" * 50]
        sig_best = {}
        for sig, fits in self.direction_log.items():
            if fits and max(fits) >= min_fitness:
                sig_best[sig] = {'best': max(fits), 'n': len(fits), 'latest': fits[-1]}
        for sig, info in sorted(sig_best.items(), key=lambda x: -x[1]['best']):
            flag = "★" if info['best'] >= 0.8 else " " if info['best'] >= 0.5 else "·"
            best_expr = ""
            if sig in self._direction_best_expr:
                _, best_expr = self._direction_best_expr[sig]
            expr_short = best_expr[:70] if best_expr else ""
            lines.append(f"  {flag} best={info['best']:.3f}  n={info['n']:3d}  {sig}")
            if expr_short:
                lines.append(f"       └─ {expr_short}")
        lines.append("=" * 50)
        return '\n'.join(lines)

    def summary(self, title: str = '') -> str:
        """输出 GP 启动配置摘要，供调用方在 run() 前打印。

        [新增] 2026-07-08 标准化启动参数输出，替代各脚本手写 print。
        """
        cfg = self.tree_gen_config
        mode = cfg.mode if cfg and cfg.mode else 'hybrid'

        # 变量
        var_names = list(self.data.keys())
        # 启用的函数
        ts_funcs = [k for k, v in (cfg.ts_weights or {}).items() if v > 0]
        math_funcs = [k for k, v in (cfg.math_weights or {}).items() if v > 0]

        lines = []
        if title:
            lines.append(f"{'='*60}")
            lines.append(f"{title}")
        lines.append(f"{'='*60}")
        lines.append(f"[GP v5 启动配置]")
        lines.append(f"  变量({len(var_names)}): {', '.join(var_names)}")
        lines.append(f"  数据形状: {self._shape}")
        lines.append(f"  种子表达式: {len(self.seed_expressions)} 个")
        if self.seed_expressions:
            show_n = min(5, len(self.seed_expressions))
            for s in self.seed_expressions[:show_n]:
                lines.append(f"    {s}")
            if len(self.seed_expressions) > show_n:
                lines.append(f"    ... 共 {len(self.seed_expressions)} 个")
        lines.append(f"  原语 — TS({len(ts_funcs)}): {', '.join(ts_funcs[:20])}")
        if len(ts_funcs) > 20:
            lines.append(f"         ... 共 {len(ts_funcs)} 个")
        lines.append(f"  原语 — MATH({len(math_funcs)}): {', '.join(math_funcs[:20])}")
        if len(math_funcs) > 20:
            lines.append(f"           ... 共 {len(math_funcs)} 个")
        lines.append(f"{'─'*60}")
        lines.append(f"[GP 参数]")
        lines.append(f"  population_size={self.population_size}, generations={self.generations}, max_depth={self.max_depth}")
        lines.append(f"  tournament_size={self.tournament_size}, elite_count={self.elite_count}")
        lines.append(f"  crossover_prob={self.crossover_prob}, mutation_prob={self.mutation_prob}")
        lines.append(f"  seed_ratio={self.seed_ratio}, random_inject_ratio={self.random_inject_count/self.population_size:.2f}")
        lines.append(f"  explore_ratio={self._explore_ratio:.2f}, mode='{mode}'")
        lines.append(f"  parsimony_penalty={self.parsimony_penalty}")
        lines.append(f"  变异权重: subtree={self._mutate_weights[0][0]:.2f}, param={self._mutate_weights[1][0]:.2f}")
        if len(self._mutate_weights) > 2:
            lines.append(f"            logic={self._mutate_weights[2][0]:.2f}, insert_cond={self._mutate_weights[3][0]:.2f}")
        lines.append(f"{'─'*60}")
        lines.append(f"[高级特性]")
        lines.append(f"  lexicase={'ON' if self._use_lexicase else 'OFF'}, epsilon={self._epsilon:.3f}"
                     f", elite_max_per_sig={self._elite_max_per_sig}")
        lines.append(f"  年龄机制: {'ON' if self._age_enabled else 'OFF'}, penalty_lr={self._age_penalty_lr}")
        lines.append(f"  Motif库: {'ON' if self._motif_enabled else 'OFF'}, update_every={self._motif_update_every}, "
                     f"inject={self._motif_inject_count}, max_depth={self._motif_max_depth}")
        lines.append(f"  停滞检测: threshold={self._stagnation_threshold}")
        lines.append(f"  自适应权重: {'ON' if (cfg and cfg.adaptive) else 'OFF'}, "
                     f"lr={cfg.adaptive_lr if cfg else 0.3}, every={cfg.adaptive_every if cfg else 3}")
        lines.append(f"{'='*60}")
        return '\n'.join(lines)

    # [删除] 2026-07-21 _update_direction_weights — v5 自适应权重 EMA 闭环
    #   v6 用 AURORA softmax(fitness/T) 方向感知选择替代, 不修改生成层权重

    # [删除] 2026-07-21 _normalize_weights — v5 自适应权重辅助, v6 不需要

    # [删除] 2026-07-21 _adapt_operators — v5 停滞检测 + 算子自适应, v6 不需要

    # ── Motif 库管理 ──

    def _update_motif_library(self):
        """从当前种群 Top 30% 个体中提取高频子树，更新 Motif 库"""
        if not self._motif_enabled or not self._snap:
            return

        # [优化] 2026-07-08 从快照 sorted_valid 取，避免重新排序
        valid = [i for i in self._snap.sorted_valid if i.fitness > self._motif_min_fitness]
        if not valid:
            return

        top_n = max(1, int(len(valid) * 0.3))
        top_individuals = valid[:top_n]

        for ind in top_individuals:
            expr_str = ind.expression_str or _expr_str(ind.tree)
            subtrees = _extract_subtrees(ind.tree, min_depth=1, max_depth=self._motif_max_depth)
            for subtree in subtrees:
                try:
                    subtree_str = _expr_str(subtree)
                    if not subtree_str or subtree_str == expr_str:
                        continue
                    # 用 canonical key 去重
                    canonical = _canonicalize_key(
                        subtree, subtree_str,
                        self._canonicalize_memo, self._canonicalize_lock
                    )
                    if canonical in self._motif_library:
                        entry = self._motif_library[canonical]
                        entry['count'] += 1
                        entry['fitness_sum'] += ind.fitness
                    else:
                        self._motif_library[canonical] = {
                            'count': 1,
                            'fitness_sum': ind.fitness,
                            'expr': subtree_str,
                            'depth': ast_depth(subtree),
                        }
                except Exception:
                    continue

    def _get_motif_seeds(self, n: int = 5) -> List[str]:
        """返回 Top-N Motif 表达式，按 出现次数 × 平均 fitness 排序"""
        if not self._motif_library:
            return []

        scored = []
        for key, entry in self._motif_library.items():
            avg_fitness = entry['fitness_sum'] / entry['count'] if entry['count'] > 0 else 0
            score = entry['count'] * max(avg_fitness, 0.01)
            scored.append((score, entry['expr']))

        scored.sort(key=lambda x: -x[0])
        return [expr for _, expr in scored[:n]]

    def get_motif_seeds(self, n: int = 10) -> List[str]:
        """公开 API：获取当前 Motif 库 Top-N 种子表达式（供跨轮次复用）"""
        return self._get_motif_seeds(n)

    # ── 选择 ──

    def _tournament_select(self) -> Individual:
        rng = self.tree_gen_config.rng
        candidates = rng.sample(self.population,
                                min(self.tournament_size, len(self.population)))
        return max(candidates, key=lambda x: x.fitness)

    def _lexicase_select(self) -> Individual:
        rng = self.tree_gen_config.rng

        # [重构] 2026-07-08 从快照 lexicase_pool 直接 O(1) 抽样
        # [修复] 2026-07-09 改为每方向抽 fitness 最高代表 + 均匀随机选，
        #            避免比例抽样使弱方向虽有代表却几乎不会被选到。
        if not self._snap or len(self._snap.lexicase_pool) < 2:
            return self._tournament_select()

        pool = self._snap.lexicase_pool

        # 按 signature 分组，每方向选最佳代表
        sig_reps: Dict[str, Individual] = {}
        for ind in pool:
            sig = ind.signature
            if not sig:
                continue
            if sig not in sig_reps or ind.fitness > sig_reps[sig].fitness:
                sig_reps[sig] = ind

        reps = list(sig_reps.values())
        if len(reps) < 2:
            return self._tournament_select()
        return rng.choice(reps)

    def _select_parent(self) -> Individual:
        if self._use_lexicase:
            return self._lexicase_select()
        return self._tournament_select()

    # ── 交叉 ──

    def _crossover(self, p1: Individual, p2: Individual) -> Individual:
        rng = self.tree_gen_config.rng
        child_tree = copy.deepcopy(p1.tree)
        donor_tree = p2.tree

        candidates2_all = _collect_replaceable(donor_tree)
        if not candidates2_all:
            return Individual(tree=child_tree, generation=p1.generation + 1)

        for _ in range(3):
            candidates1_all = _collect_replaceable(child_tree)
            if not candidates1_all:
                break
            n1 = rng.choice(candidates1_all)
            n2 = rng.choice(candidates2_all)
            n1_is_bool = isinstance(n1, (ast.BoolOp, ast.Compare))
            n2_is_bool = isinstance(n2, (ast.BoolOp, ast.Compare))
            if n1_is_bool != n2_is_bool:
                continue
            backup = copy.deepcopy(child_tree)
            replaced = _replace_subtree(child_tree, n1, copy.deepcopy(n2))
            if not replaced:
                child_tree = backup
                continue
            ast.fix_missing_locations(child_tree)
            if ast_depth(child_tree) > self.max_depth:
                child_tree = backup
                continue
            return Individual(tree=_simplify_ast(child_tree), generation=p1.generation + 1)

        return Individual(tree=child_tree, generation=p1.generation + 1)

    # ── 变异 ──

    def _mutate(self, individual: Individual) -> Individual:
        cfg = self.tree_gen_config
        rng = cfg.rng
        total = sum(w for w, _ in self._mutate_weights)
        r = rng.random() * total
        cumulative = 0
        for weight, mutate_fn in self._mutate_weights:
            cumulative += weight
            if r <= cumulative:
                try:
                    new_tree = mutate_fn(cfg, individual.tree, max_depth=min(self.max_depth, 4))
                except Exception:
                    new_tree = copy.deepcopy(individual.tree)
                if ast_depth(new_tree) > self.max_depth:
                    new_tree = copy.deepcopy(individual.tree)
                return Individual(tree=new_tree, generation=individual.generation + 1)
        new_tree = _mutate_subtree(cfg, individual.tree, max_depth=min(self.max_depth, 4))
        if ast_depth(new_tree) > self.max_depth:
            new_tree = copy.deepcopy(individual.tree)
        return Individual(tree=new_tree, generation=individual.generation + 1)

    # ── 初始化 ──

    def _fill_one_population(self, seeds: List[str], pop_size: int, cfg: TreeGenConfig, rng) -> List[Individual]:
        """填充一个独立的子种群（被单模式和岛屿模式共用）"""
        pop: List[Individual] = []
        seed_count = int(pop_size * self.seed_ratio)
        for expr_str in seeds[:seed_count]:
            try:
                ind = Individual.from_expr(expr_str, generation=0, is_seed=True)
                ind.age = 1
                pop.append(ind)
            except Exception as e:
                logger.warning(f"种子解析失败: {expr_str[:60]} ({e})")

        seed_trees = [ind.tree for ind in pop if ind.is_seed]
        while len(pop) < seed_count and seed_trees:
            base = rng.choice(seed_trees)
            mutated = _mutate_subtree(cfg, base, max_depth=3)
            pop.append(Individual(tree=mutated, generation=0, age=1))

        remaining = pop_size - len(pop)
        explore_n = int(remaining * self._explore_ratio)
        while len(pop) < pop_size:
            if len(pop) < seed_count + explore_n:
                tree = _random_tree_explore(cfg, self.max_depth)
            else:
                tree = _random_tree(cfg, self.max_depth)
            pop.append(Individual(tree=tree, generation=0, age=1))
        return pop

    def _initialize_population(self):
        cfg = self.tree_gen_config
        rng = cfg.rng
        self.population = []

        # [新增] 2026-07-08 Motif 种子: 从 Motif 库抽取高质量子树
        motif_seeds = self._get_motif_seeds(self._motif_inject_count) if self._motif_enabled else []
        all_seeds = self.seed_expressions + motif_seeds

        # [新增] 2026-07-09 岛屿模式: 分 N 个独立子种群，种子 round-robin 分配
        if self._num_islands > 1:
            self._islands = []
            for i in range(self._num_islands):
                island_seeds = all_seeds[i::self._num_islands]
                self._islands.append(
                    self._fill_one_population(island_seeds, self._island_size, cfg, rng)
                )
            self.population = self._islands[0]
            return

        # 单模式
        self.population = self._fill_one_population(all_seeds, self.population_size, cfg, rng)

    # ── 进化循环 ──

    def _next_generation(self, gen: int, *,
                         pop_size: int = None,
                         elite_count: int = None,
                         inject_count: int = None,
                         explore_ratio: float = None) -> List[Individual]:
        """生成下一代种群。

        Args:
            gen: 当前代数
            pop_size: 种群大小（None=使用 self.population_size）
            elite_count: 精英数量（None=使用 self.elite_count）
            inject_count: 随机注入数量（None=使用 self.random_inject_count）
            explore_ratio: 探索比例（None=使用 self._explore_ratio）
        """
        _pop_size = pop_size if pop_size is not None else self.population_size
        _elite = elite_count if elite_count is not None else self.elite_count
        _inject = inject_count if inject_count is not None else self.random_inject_count
        _explore = explore_ratio if explore_ratio is not None else self._explore_ratio

        cfg = self.tree_gen_config
        rng = cfg.rng
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_pop = []

        # [新增] 2026-07-09 岛屿模式: 精英走方向配额, 每方向最多1个代表
        # 原问题: 纯 fitness 排序让冠军及其常数微调变体全数占据精英席, 岛内多样性被压制
        # 修复: 按 signature 分组取各方向最佳代表, 再按 fitness 降序填满 elite_count
        # 单模式保持原行为 (纯 fitness 排序), 仅岛屿/lexicase 模式启用方向配额
        if (self._num_islands > 1 or self._use_lexicase) and _elite > 0:
            # [重构] 2026-07-09 方向配额精英: 每方向最多 elite_max_per_sig 个代表
            # None = 不限制 → 等同于原始 lexicase 的纯 fitness 排序
            if self._elite_max_per_sig is None:
                elites = sorted_pop[:min(_elite, len(sorted_pop))]
            else:
                sig_top: Dict[str, List[Individual]] = {}
                no_sig: List[Individual] = []
                for ind in sorted_pop:
                    sig = ind.signature
                    if not sig and ind.expression_str:
                        sig = self._expr_signature(ind.expression_str)
                    if sig:
                        lst = sig_top.setdefault(sig, [])
                        if len(lst) < self._elite_max_per_sig:
                            lst.append(ind)
                    else:
                        no_sig.append(ind)
                elite_candidates = []
                for lst in sig_top.values():
                    elite_candidates.extend(lst)
                elite_candidates.sort(key=lambda x: x.fitness, reverse=True)
                elite_candidates.extend(no_sig)
                elites = elite_candidates[:min(_elite, len(elite_candidates))]
        else:
            elites = sorted_pop[:min(_elite, len(sorted_pop))]

        for elite in elites:
            new_pop.append(Individual(
                tree=copy.deepcopy(elite.tree),
                fitness=elite.fitness,
                expression_str=elite.expression_str,
                generation=gen + 1,
                depth=elite.depth, node_count=elite.node_count,
                age=elite.age + 1,  # [新增] 2026-07-08 精英个体年龄+1
            ))

        while len(new_pop) < _pop_size - _inject:
            r = rng.random()
            if r < self.crossover_prob:
                child = self._crossover(self._select_parent(), self._select_parent())
            elif r < self.crossover_prob + self.mutation_prob:
                child = self._mutate(self._select_parent())
            else:
                parent = self._select_parent()
                child = Individual(tree=copy.deepcopy(parent.tree), generation=gen + 1)
            child.age = 1  # [新增] 2026-07-08 新个体初始年龄=1
            new_pop.append(child)

        inject_n = _inject
        # [修复] 2026-07-09 按总人口算探索数，保持每代 15% 探索强度不衰减
        explore_n = min(inject_n, int(_pop_size * _explore))

        # [新增] 2026-07-09 注入 motif 种子（取代部分纯随机注入，保持多样性）
        motif_injected = 0
        if self._motif_enabled:
            motif_seeds = self._get_motif_seeds(self._motif_inject_count)
            for expr in motif_seeds:
                if len(new_pop) >= _pop_size:
                    break
                try:
                    ind = Individual.from_expr(expr, generation=gen + 1)
                    ind.age = 1
                    new_pop.append(ind)
                    motif_injected += 1
                except Exception:
                    continue

        remaining_inject = max(0, inject_n - motif_injected)
        for i in range(remaining_inject):
            if i < explore_n:
                tree = _random_tree_explore(cfg, self.max_depth)
            else:
                tree = _random_tree(cfg, self.max_depth)
            new_pop.append(Individual(tree=tree, generation=gen + 1, age=1))

        return new_pop[:_pop_size]

    # [重构] 2026-07-09 岛屿迁移: 从"精英迁移"改为"多样性迁移"
    def _migrate(self):
        """环形拓扑多样性迁移: Island i → Island (i+1) % N

        [原问题] 纯 Top-k 迁移让冠军骨架扩散到所有岛, 各岛同质化塌陷.
        [新策略]
          - 迁出: 源岛各方向最佳代表中, 接收岛缺失的方向优先 (Top-k 兜底)
          - 替换: 接收岛"方向冗余"个体优先 (同方向保留最佳, 其余按 fitness 升序替换),
                 避免替换掉接收岛稀少方向的唯一代表
        """
        n = self._num_islands
        k = min(self._migrate_count, max(1, self._island_size // 5))

        for i in range(n):
            src = self._islands[i]
            dst = self._islands[(i + 1) % n]

            # ── 选迁出个体: 接收岛缺失的方向优先, Top-k 兜底 ──
            src_sig_best: Dict[str, Individual] = {}
            for ind in src:
                sig = ind.signature
                if not sig and ind.expression_str:
                    sig = self._expr_signature(ind.expression_str)
                if sig:
                    if sig not in src_sig_best or ind.fitness > src_sig_best[sig].fitness:
                        src_sig_best[sig] = ind

            dst_sigs: set = set()
            for ind in dst:
                sig = ind.signature
                if not sig and ind.expression_str:
                    sig = self._expr_signature(ind.expression_str)
                if sig:
                    dst_sigs.add(sig)

            # 优先迁出缺失方向代表 (按 fitness 降序), 不足时用源岛 Top-k 补齐
            missing_reps = [ind for sig, ind in src_sig_best.items() if sig not in dst_sigs]
            missing_reps.sort(key=lambda x: x.fitness, reverse=True)
            migrants = missing_reps[:k]

            if len(migrants) < k:
                src_sorted = sorted(src, key=lambda x: x.fitness, reverse=True)
                existing_ids = {id(m) for m in migrants}
                for ind in src_sorted:
                    if len(migrants) >= k:
                        break
                    if id(ind) not in existing_ids:
                        migrants.append(ind)
                        existing_ids.add(id(ind))

            if not migrants:
                continue

            # ── 选替换位置: 接收岛方向冗余个体优先 (同方向保留最佳) ──
            dst_by_sig: Dict[str, List[int]] = {}
            no_sig_idx: List[int] = []
            for idx, ind in enumerate(dst):
                sig = ind.signature
                if not sig and ind.expression_str:
                    sig = self._expr_signature(ind.expression_str)
                if sig:
                    dst_by_sig.setdefault(sig, []).append(idx)
                else:
                    no_sig_idx.append(idx)

            # 冗余索引: 每方向除 fitness 最高外其余加入冗余池, 无签名个体也加入
            redundant_idx: List[int] = []
            for sig, idxs in dst_by_sig.items():
                if len(idxs) > 1:
                    idxs_sorted = sorted(idxs, key=lambda j: dst[j].fitness)  # 升序, 最差在前
                    redundant_idx.extend(idxs_sorted[:-1])  # 排除该方向最佳
            redundant_idx.extend(no_sig_idx)
            redundant_idx.sort(key=lambda j: dst[j].fitness)  # 冗余池按 fitness 升序

            # 冗余不足时用全局 Bottom-k 补齐 (兜底)
            if len(redundant_idx) < k:
                all_idx_sorted = sorted(range(len(dst)), key=lambda j: dst[j].fitness)
                existing = set(redundant_idx)
                for idx in all_idx_sorted:
                    if len(redundant_idx) >= k:
                        break
                    if idx not in existing:
                        redundant_idx.append(idx)
                        existing.add(idx)

            # 执行替换
            replace_indices = redundant_idx[:k]
            for j, migrant in enumerate(migrants[:k]):
                if j >= len(replace_indices):
                    break
                replace_idx = replace_indices[j]
                migrant_copy = copy.deepcopy(migrant)
                migrant_copy.generation = max(migrant_copy.generation, dst[replace_idx].generation)
                dst[replace_idx] = migrant_copy

    # ── 快照 ──

    def _build_snapshot(self, gen: int) -> GenerationSnapshot:
        """构建每代快照：方向追踪 + signature 缓存 + lexicase_pool 预计算。

        合并原来分散在 run() 中的三个遍历为一趟，减少重复计算。
        """
        valid = [i for i in self.population if i.fitness > -999]
        by_sig: Dict[str, List[Individual]] = {}

        for ind in valid:
            sig = self._expr_signature(ind.expression_str or _expr_str(ind.tree))
            ind.signature = sig  # 缓存避免 lexicase 重复计算

            # 方向追踪
            if sig not in self.direction_log:
                self.direction_log[sig] = []
            self.direction_log[sig].append(ind.fitness)
            prev = self._direction_best_expr.get(sig, (-999, ''))
            if ind.fitness > prev[0]:
                self._direction_best_expr[sig] = (ind.fitness, ind.expression_str or _expr_str(ind.tree))

            # 按 sig 分组
            if ind.expression_str:
                by_sig.setdefault(sig, []).append(ind)

        # lexicase_pool 预计算
        lex_pool: List[Individual] = []
        seen_expr: set = set()
        for sig, inds in by_sig.items():
            best_this_gen = max(ind.fitness for ind in inds)
            threshold = best_this_gen - abs(best_this_gen) * self._epsilon
            for ind in inds:
                if ind.fitness >= threshold and ind.expression_str not in seen_expr:
                    seen_expr.add(ind.expression_str)
                    lex_pool.append(ind)

        return GenerationSnapshot(
            generation=gen,
            valid=[i for i in valid if i.expression_str],
            sorted_valid=sorted(
                [i for i in valid if i.expression_str],
                key=lambda x: x.fitness, reverse=True,
            ),
            by_signature=by_sig,
            sig_best=dict(self._direction_best_expr),
            lexicase_pool=lex_pool,
        )

    # ── 主循环 ──

    def run(self, verbose: bool = True, callback: Callable[[int, Individual, Dict], None] = None) -> 'GPEngine':
        """运行 GP 搜索

        Args:
            verbose: 是否打印日志
            callback: 可选，每代结束后调用 callback(gen, best_individual, stats)
        """
        n_loaded = self.fitness_cache.load()
        if n_loaded and verbose:
            logger.info(f"[GP] 加载 SQLite 缓存 {n_loaded} 条")

        self._initialize_population()
        for gen in range(self.generations):
            self._evaluate_and_merge()
            self._post_generation(gen, verbose, callback)
            if gen < self.generations - 1:
                self._advance_to_next_generation(gen)

        self.fitness_cache.save()
        return self

    def _evaluate_and_merge(self):
        """评估当前种群（岛屿模式逐岛评估后合并）"""
        if self._num_islands > 1:
            for i in range(self._num_islands):
                self.population = self._islands[i]
                self._evaluate_population()
                self._islands[i] = self.population
            self.population = list(itertools.chain.from_iterable(self._islands))
        else:
            self._evaluate_population()

    def _post_generation(self, gen: int, verbose: bool, callback):
        """每代评估后的统一后处理：统计 → 快照 → 自适应 → 回调 → 日志"""
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

        # [删除] 2026-07-21 移除 v5 自适应权重/停滞检测/Motif/方向追踪
        #   v6 用 AURORA Archive + 方向感知选择替代, 这些代码从未被调用

        if callback:
            callback(gen, gen_best, gen_stats)

        if verbose and gen % 5 == 0:
            logger.info(f"Gen {gen:3d} | best_f={gen_best.fitness:.3f} "
                        f"depth={gen_best.depth} valid={gen_stats['valid_count']}")

        if self.best_individual is None or gen_best.fitness > self.best_individual.fitness:
            self.best_individual = gen_best

    def _advance_to_next_generation(self, gen: int):
        """进化到下一代：迁移 + 生成新种群"""
        if self._num_islands > 1:
            if gen > 0 and gen % self._migrate_every == 0:
                self._migrate()
            for i in range(self._num_islands):
                self.population = self._islands[i]
                self._islands[i] = self._next_generation(gen,
                    pop_size=self._island_size,
                    elite_count=max(1, int(self._island_size * self.elite_count / self.population_size)),
                    inject_count=max(0, int(self._island_size * 0.03)),
                    explore_ratio=max(0.03, self._explore_ratio * 0.25))
            self.population = self._islands[0]
        else:
            self.population = self._next_generation(gen)

    # ── 结果 ──

    def best(self) -> Optional[Individual]:
        return self.best_individual

    def top(self, n: int = 10) -> List[Individual]:
        # [新增] 2026-07-09 岛屿模式: 从所有岛合并收集
        pool = self.population
        if self._num_islands > 1 and self._islands:
            pool = list(itertools.chain.from_iterable(self._islands))
        return sorted([i for i in pool if i.fitness > -999],
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