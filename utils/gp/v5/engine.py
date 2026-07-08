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
import numpy as np
import os
from typing import Dict, List, Optional, Callable
from dataclasses import fields as dataclass_fields

from utils.ast.dsl import parse_expression, ast_depth, ast_node_count
from .config import (
    GP_VARIABLES,
    TreeGenConfig, Individual,
    DEFAULT_TREE_GEN_CONFIG, DEFAULT_GP_CONFIG,
    _fill_weights,
    get_full_default_weights, _get_funcs_by_group, _get_fill_keys,
)
from .ast_utils import (
    _expr_str, _collect_replaceable, _replace_subtree, _simplify_ast, _canonicalize_key,
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
                 cache_db: str = ''):
        self.data = data
        self.future_returns = future_returns
        self.returns = returns
        self.fitness_calc = fitness_calculator
        self._evaluator = evaluator

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
        self._save_random_inject: int = self.random_inject_count
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
        self._parallel_workers = config.get('parallel_workers', 0) if config else 0
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
        self.fitness_cache = FitnessCache(cache_db, fitness_hash)

        # 方向演化追踪
        self.direction_log: Dict[str, List[float]] = {}
        self._direction_best_expr: Dict[str, tuple] = {}
        self._direction_per_sig_snapshot: Dict[str, int] = {}

        # 停滞检测
        self._stagnation_counter: int = 0
        self._last_best_fitness: float = -999.0
        self._save_crossover_prob: float = self.crossover_prob
        self._save_mutation_prob: float = self.mutation_prob
        self._stagnation_threshold: int = 3

        # ε-greedy 探索
        self._explore_ratio: float = config.get('explore_ratio', 0.15) if config else 0.15

        # [新增] 2026-07-08 ε-Lexicase: 允许 fitness 差距 ε 内的个体同为代表，保留多样性
        self._use_lexicase: bool = config.get('lexicase', False) if config else False
        self._epsilon: float = config.get('epsilon', 0.05) if config else 0.05

        # [新增] 2026-07-08 Motif 库: 提取高频子树作为高质量种子
        self._motif_enabled: bool = cfg.get('motif_enabled', True)
        self._motif_update_every: int = cfg.get('motif_update_every', 3)
        self._motif_min_fitness: float = cfg.get('motif_min_fitness', 0.0)
        self._motif_max_depth: int = max(1, int(self.max_depth * cfg.get('motif_max_depth_ratio', 0.5)))
        self._motif_inject_count: int = cfg.get('motif_inject_count', 5)
        # Motif 库结构: {canonical_key: {"count": int, "fitness_sum": float, "expr": str, "depth": int}}
        self._motif_library: Dict[str, Dict] = {}

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
        else:
            if not self._quick_filter(ind):
                return -999.0

            try:
                if self._evaluator:
                    factor_values = self._evaluator(self.data, ind.tree)
                else:
                    return -999.0
            except Exception as e:
                if ind.is_seed:
                    logger.warning(f"[GP] 种子求值异常: {e}")
                return -999.0

            if not np.isfinite(factor_values).all():
                return -999.0
            if np.allclose(factor_values, factor_values.flat[0], atol=1e-10):
                return -999.0

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
    def _expr_signature(expr_str: str) -> str:
        if not expr_str:
            return "unknown"
        import re
        from collections import Counter
        from utils.ast.functions import FUNC_REGISTRY
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
        # [优化] 2026-07-08 保留变量频次编码 v:count，提升方向追踪精度。
        # 旧: "CLOSE+OPEN" 丢失频次; 新: "CLOSE:3+OPEN:1" 用于加权。
        vars_sig = '+'.join(f'{v}:{c}' for v, c in var_counts.most_common(4)) or 'none'
        return f"m:{math_sig}|t:{ts_sig}|v:{vars_sig}"

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
        lines.append(f"  lexicase={'ON' if self._use_lexicase else 'OFF'}, epsilon={self._epsilon:.3f}")
        lines.append(f"  年龄机制: {'ON' if self._age_enabled else 'OFF'}, penalty_lr={self._age_penalty_lr}")
        lines.append(f"  Motif库: {'ON' if self._motif_enabled else 'OFF'}, update_every={self._motif_update_every}, "
                     f"inject={self._motif_inject_count}, max_depth={self._motif_max_depth}")
        lines.append(f"  停滞检测: threshold={self._stagnation_threshold}")
        lines.append(f"  自适应权重: {'ON' if (cfg and cfg.adaptive) else 'OFF'}, "
                     f"lr={cfg.adaptive_lr if cfg else 0.3}, every={cfg.adaptive_every if cfg else 3}")
        lines.append(f"{'='*60}")
        return '\n'.join(lines)

    def _update_direction_weights(self):
        """EMA 闭环方向权重更新"""
        cfg = self.tree_gen_config
        if not cfg or not cfg.adaptive:
            return

        new_records: Dict[str, List[float]] = {}
        total_entries = 0
        for sig, fits in self.direction_log.items():
            offset = self._direction_per_sig_snapshot.get(sig, 0)
            new_fits = fits[offset:]
            if new_fits:
                new_records[sig] = new_fits
                total_entries += len(new_fits)
            self._direction_per_sig_snapshot[sig] = len(fits)

        if total_entries < 10:
            return

        sig_scores = {}
        for sig, fits in new_records.items():
            if not fits:
                continue
            best = max(fits)
            sig_scores[sig] = best * np.log1p(len(fits))

        if not sig_scores:
            return

        max_score = max(sig_scores.values())
        if max_score <= 0.1:
            return

        var_scores: Dict[str, float] = {}
        ts_scores: Dict[str, float] = {}
        math_scores: Dict[str, float] = {}

        for sig, score in sig_scores.items():
            norm_score = score / max_score
            parts = {}
            for p in sig.split('|'):
                if ':' in p:
                    k, v = p.split(':', 1)
                    parts[k] = v.split('+') if v != 'none' else []
            # [优化] 2026-07-08 解析 v:count 编码，按频次加权。
            for item in parts.get('v', []):
                if ':' in item:
                    v, cnt = item.split(':')
                    var_scores[v] = var_scores.get(v, 0) + norm_score * float(cnt)
                else:
                    var_scores[item] = var_scores.get(item, 0) + norm_score
            # ts_scores / math_scores 仍按原始逻辑（无频次编码）
            for f in parts.get('t', []):
                ts_scores[f] = ts_scores.get(f, 0) + norm_score
            for f in parts.get('m', []):
                math_scores[f] = math_scores.get(f, 0) + norm_score

        lr = cfg.adaptive_lr
        var_w = cfg.var_weights
        if var_w and var_scores:
            for v in var_w:
                old = var_w.get(v, 0)
                score = var_scores.get(v, 0)
                var_w[v] = (1 - lr) * old + lr * score * 3.0
            self._normalize_weights(var_w, min_w=0.05)
        ts_w = cfg.ts_weights
        if ts_w and ts_scores:
            for f in ts_w:
                old = ts_w.get(f, 0)
                score = ts_scores.get(f, 0)
                ts_w[f] = (1 - lr) * old + lr * score * 3.0
            self._normalize_weights(ts_w, min_w=0.05)
        mw = cfg.math_weights
        if mw and math_scores:
            for f in mw:
                old = mw.get(f, 0)
                score = math_scores.get(f, 0)
                mw[f] = (1 - lr) * old + lr * score * 3.0
            self._normalize_weights(mw, min_w=0.05)

    @staticmethod
    def _normalize_weights(weights: Dict[str, float], min_w: float = 0.05):
        if not weights:
            return
        total = sum(weights.values())
        if total <= 0:
            for k in weights:
                weights[k] = 1.0 / len(weights)
            return
        for k in weights:
            weights[k] = max(min_w, weights[k] / total)

    def _adapt_operators(self, gen_best_fitness: float):
        """AW-MEP 风格停滞检测 + 算子自适应"""
        if gen_best_fitness > self._last_best_fitness + 1e-4:
            self._stagnation_counter = 0
            self._last_best_fitness = gen_best_fitness
            self.crossover_prob = self._save_crossover_prob
            self.mutation_prob = self._save_mutation_prob
            self.random_inject_count = self._save_random_inject
        else:
            self._stagnation_counter += 1
            if self._stagnation_counter >= self._stagnation_threshold:
                self.mutation_prob = min(0.5, self.mutation_prob * 1.3)
                self.crossover_prob = max(0.2, self.crossover_prob * 0.85)
                if self._stagnation_counter >= 5:
                    self.random_inject_count = min(
                        int(self.population_size * 0.3),
                        self.random_inject_count * 2,
                    )

    # ── Motif 库管理 ──

    def _update_motif_library(self):
        """从当前种群 Top 30% 个体中提取高频子树，更新 Motif 库"""
        if not self._motif_enabled:
            return

        valid = [i for i in self.population if i.fitness > self._motif_min_fitness]
        if not valid:
            return

        # 取 Top 30%
        valid.sort(key=lambda x: x.fitness, reverse=True)
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
        valid = [i for i in self.population if i.fitness > -999 and i.expression_str]
        if len(valid) < self.tournament_size:
            return self._tournament_select()
        if len(self._direction_best_expr) < 2:
            return self._tournament_select()

        # [新增] 2026-07-08 ε-lexicase: 对每个方向，取 fitness 在 [best*(1-ε), best] 区间内的所有个体
        # 而非仅取 best 表达式。保留更多方向代表，提升多样性。
        representatives = []
        for sig, (best_fit, best_expr) in self._direction_best_expr.items():
            # 计算该方向的 fitness 阈值（允许 ε 差距）
            threshold = best_fit * (1.0 - self._epsilon)
            # 收集该方向下所有 fitness >= threshold 的个体
            for ind in valid:
                ind_sig = self._expr_signature(ind.expression_str)
                if ind_sig == sig and ind.fitness >= threshold:
                    representatives.append(ind)

        # 去重（同一表达式可能出现多次）
        seen = set()
        unique_reps = []
        for ind in representatives:
            if ind.expression_str not in seen:
                seen.add(ind.expression_str)
                unique_reps.append(ind)
        representatives = unique_reps

        if not representatives:
            return self._tournament_select()

        weights = [max(ind.fitness, 0.01) for ind in representatives]
        total_w = sum(weights)
        if total_w <= 0:
            return rng.choice(representatives)
        probs = [w / total_w for w in weights]
        return rng.choices(representatives, weights=probs, k=1)[0]

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

    def _initialize_population(self):
        cfg = self.tree_gen_config
        rng = cfg.rng
        self.population = []

        # [新增] 2026-07-08 Motif 种子: 从 Motif 库抽取高质量子树
        motif_seeds = self._get_motif_seeds(self._motif_inject_count) if self._motif_enabled else []
        all_seeds = self.seed_expressions + motif_seeds

        seed_count = int(self.population_size * self.seed_ratio)
        for expr_str in all_seeds[:seed_count]:
            try:
                ind = Individual.from_expr(expr_str, generation=0, is_seed=True)
                ind.age = 1  # [新增] 2026-07-08 种子初始年龄=1
                self.population.append(ind)
            except Exception as e:
                logger.warning(f"种子解析失败: {expr_str[:60]} ({e})")

        seed_trees = [ind.tree for ind in self.population if ind.is_seed]
        while len(self.population) < seed_count and seed_trees:
            base = rng.choice(seed_trees)
            mutated = _mutate_subtree(cfg, base, max_depth=3)
            self.population.append(Individual(tree=mutated, generation=0, age=1))

        remaining = self.population_size - len(self.population)
        explore_n = int(remaining * self._explore_ratio)
        while len(self.population) < self.population_size:
            if len(self.population) < seed_count + explore_n:
                tree = _random_tree_explore(cfg, self.max_depth)
            else:
                tree = _random_tree(cfg, self.max_depth)
            self.population.append(Individual(tree=tree, generation=0, age=1))

    # ── 进化循环 ──

    def _next_generation(self, gen: int) -> List[Individual]:
        cfg = self.tree_gen_config
        rng = cfg.rng
        sorted_pop = sorted(self.population, key=lambda x: x.fitness, reverse=True)
        new_pop = []

        for i in range(min(self.elite_count, len(sorted_pop))):
            elite = sorted_pop[i]
            new_pop.append(Individual(
                tree=copy.deepcopy(elite.tree),
                fitness=elite.fitness,
                expression_str=elite.expression_str,
                generation=gen + 1,
                depth=elite.depth, node_count=elite.node_count,
                age=elite.age + 1,  # [新增] 2026-07-08 精英个体年龄+1
            ))

        while len(new_pop) < self.population_size - self.random_inject_count:
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

        inject_n = self.random_inject_count
        explore_n = min(inject_n, int(inject_n * self._explore_ratio * 2))
        for i in range(inject_n):
            if i < explore_n:
                tree = _random_tree_explore(cfg, self.max_depth)
            else:
                tree = _random_tree(cfg, self.max_depth)
            new_pop.append(Individual(tree=tree, generation=gen + 1, age=1))

        return new_pop[:self.population_size]

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

            # 方向追踪
            for ind in valid:
                if ind.fitness > -999:
                    sig = self._expr_signature(ind.expression_str or _expr_str(ind.tree))
                    if sig not in self.direction_log:
                        self.direction_log[sig] = []
                    self.direction_log[sig].append(ind.fitness)
                    prev = self._direction_best_expr.get(sig, (-999, ''))
                    if ind.fitness > prev[0]:
                        self._direction_best_expr[sig] = (ind.fitness, ind.expression_str or _expr_str(ind.tree))

            # 方向权重 EMA 更新
            cfg = self.tree_gen_config
            if cfg and cfg.adaptive and gen > 0 and gen % cfg.adaptive_every == 0:
                self._update_direction_weights()

            # [新增] 2026-07-08 Motif 库更新: 每 N 代从 Top 个体提取子树
            if self._motif_enabled and gen > 0 and gen % self._motif_update_every == 0:
                self._update_motif_library()

            # 算子自适应
            self._adapt_operators(gen_best.fitness)

            # callback
            if callback:
                callback(gen, gen_best, gen_stats)

            if verbose and gen % 5 == 0:
                logger.info(f"Gen {gen:3d} | best_f={gen_best.fitness:.3f} "
                            f"depth={gen_best.depth} valid={gen_stats['valid_count']}")

            if self.best_individual is None or gen_best.fitness > self.best_individual.fitness:
                self.best_individual = gen_best

            if gen < self.generations - 1:
                self.population = self._next_generation(gen)

        self.fitness_cache.save()
        return self

    # ── 结果 ──

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