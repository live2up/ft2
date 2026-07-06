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
from typing import Dict, List, Optional, Callable
from dataclasses import fields as dataclass_fields

from utils.ast.dsl import parse_expression, ast_depth, ast_node_count
from .config import (
    GP_VARIABLES,
    TS_FUNCTIONS, TS_FUNCTIONS_2ARG, MATH_FUNCTIONS,
    TreeGenConfig, Individual,
    DEFAULT_TREE_GEN_CONFIG, DEFAULT_GP_CONFIG,
    _FILL_VAR_KEYS, _FILL_TS_KEYS, _FILL_MATH_KEYS,
    _FILL_FEATURE_KEYS, _FILL_GROUP_KEYS, _fill_weights,
)
from .ast_utils import (
    _expr_str, _collect_replaceable, _replace_subtree, _simplify_ast, _canonicalize_key,
)
from .tree_gen import (
    _grow_tree, _random_tree, _random_tree_explore,
    _mutate_subtree, _mutate_constant, _mutate_window,
    _mutate_logic, _mutate_insert_condition, _MUTATE_OPS,
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
                 future_returns=None, returns=None):
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

        # 变异算子权重
        w = cfg
        self._mutate_weights = [
            (w['mutate_subtree_weight'], _mutate_subtree),
            (w['mutate_constant_weight'], _mutate_constant),
            (w['mutate_window_weight'], _mutate_window),
            (w['mutate_logic_weight'], _mutate_logic),
            (w['mutate_insert_cond_weight'], _mutate_insert_condition),
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
        # [改] 2026-07-06 默认缓存到 output/.gp_cache.db
        cache_db = config.get('cache_db', '') if config else ''
        if not cache_db:
            cache_db = 'output/.gp_cache.db'
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

        # Lexicase 选择
        self._use_lexicase: bool = config.get('lexicase', False) if config else False

    # ── 配置构建 ──

    def _build_tree_config(self, user_config: TreeGenConfig = None,
                           rng: random.Random = None) -> TreeGenConfig:
        if user_config is None:
            return TreeGenConfig(
                group_weights=DEFAULT_TREE_GEN_CONFIG.group_weights,
                ts_weights=DEFAULT_TREE_GEN_CONFIG.ts_weights,
                math_weights=DEFAULT_TREE_GEN_CONFIG.math_weights,
                feature_weights=DEFAULT_TREE_GEN_CONFIG.feature_weights,
                rng=rng or random.Random(),
            )
        filled = {}
        for field in dataclass_fields(TreeGenConfig):
            if field.name == 'rng':
                filled[field.name] = rng or user_config.rng or random.Random()
                continue
            user_val = getattr(user_config, field.name)
            default_val = getattr(DEFAULT_TREE_GEN_CONFIG, field.name)
            if user_val is None:
                filled[field.name] = default_val
            elif field.name in ('mode', 'var_allowlist', 'func_allowlist',
                                'adaptive', 'adaptive_lr', 'adaptive_every'):
                filled[field.name] = user_val
            else:
                key_set = {
                    'group_weights': _FILL_GROUP_KEYS,
                    'ts_weights': _FILL_TS_KEYS,
                    'math_weights': _FILL_MATH_KEYS,
                    'feature_weights': _FILL_FEATURE_KEYS,
                    'var_weights': _FILL_VAR_KEYS,
                }.get(field.name, [])
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
            return cached[0]

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
        ind.fitness = fitness * (1.0 - penalty)

        self.fitness_cache.put(key, (ind.fitness, ind.depth, ind.node_count))
        return ind.fitness

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
        funcs = set(re.findall(r'([a-z_]+)\(', expr_str))
        math_sig = '+'.join(sorted(f for f in funcs if f in MATH_FUNCTIONS)) or 'none'
        ts_sig = '+'.join(sorted(f for f in funcs if f in TS_FUNCTIONS or f in TS_FUNCTIONS_2ARG)) or 'none'
        var_counts = Counter(re.findall(r'\b([A-Z][A-Z_0-9]+)\b', expr_str))
        vars_sig = '+'.join(v for v, _ in var_counts.most_common(4)) or 'none'
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
            for v in parts.get('v', []):
                var_scores[v] = var_scores.get(v, 0) + norm_score
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

        valid_by_expr = {i.expression_str: i for i in valid if i.expression_str}
        representatives = []
        for sig, (best_fit, best_expr) in self._direction_best_expr.items():
            ind = valid_by_expr.get(best_expr)
            if ind is not None:
                representatives.append(ind)

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

        seed_count = int(self.population_size * self.seed_ratio)
        for expr_str in self.seed_expressions[:seed_count]:
            try:
                ind = Individual.from_expr(expr_str, generation=0, is_seed=True)
                self.population.append(ind)
            except Exception as e:
                logger.warning(f"种子解析失败: {expr_str[:60]} ({e})")

        seed_trees = [ind.tree for ind in self.population if ind.is_seed]
        while len(self.population) < seed_count and seed_trees:
            base = rng.choice(seed_trees)
            mutated = _mutate_subtree(cfg, base, max_depth=3)
            self.population.append(Individual(tree=mutated, generation=0))

        remaining = self.population_size - len(self.population)
        explore_n = int(remaining * self._explore_ratio)
        while len(self.population) < self.population_size:
            if len(self.population) < seed_count + explore_n:
                tree = _random_tree_explore(cfg, self.max_depth)
            else:
                tree = _random_tree(cfg, self.max_depth)
            self.population.append(Individual(tree=tree, generation=0))

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
            new_pop.append(child)

        inject_n = self.random_inject_count
        explore_n = min(inject_n, int(inject_n * self._explore_ratio * 2))
        for i in range(inject_n):
            if i < explore_n:
                tree = _random_tree_explore(cfg, self.max_depth)
            else:
                tree = _random_tree(cfg, self.max_depth)
            new_pop.append(Individual(tree=tree, generation=gen + 1))

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