"""
utils/gp/v6/aurora_engine.py — AURORA 启发式 GP 引擎
=============================================================================
[新增] 2026-07-20 基于 GECCO 2025 AURORA-XCon 论文的核心思想。
在 v5 GPEngine 基础上替换选择/种群管理机制:

  论文映射:
    机器人状态轨迹 → 因子输出统计摘要 (8维)
    行为描述子 φ     → 编码器潜在向量 (6维)
    QD Archive        → 非结构化 archive, DNS 局部竞争
    高斯变异          → 子树/参数/逻辑变异 (复用 v5)
    周期性灭绝        → 每 N 代清空 archive, 保留 k%

  核心改动 (相对 GPEngine):
    1. 选择: tournament/lexicase → archive 均匀随机采样
    2. 种群: fitness 排序淘汰 → DNS φ-空间局部竞争
    3. 新增: 编码器在线训练 (MSE 自编码器 或 Triplet Loss)
    4. 新增: 周期性灭绝事件

  复用 GPEngine 的:
    变异算子 (_mutate, _crossover)
    适应度评估 (_evaluate_individual, _evaluate_population)
    AST 缓存 (FitnessCache)
    树生成配置 (TreeGenConfig)
=============================================================================
"""

import ast
import copy
import random
import logging
from typing import Dict, List, Optional, Callable

import numpy as np

from .config import (
    EncodedIndividual, Individual, TreeGenConfig,
    AURA_DEFAULT_CONFIG,
)
from .engine import GPEngine
from .tree_gen import (
    _random_tree, _random_tree_explore, _mutate_subtree,
)

logger = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════
# FactorEncoder — 轻量 MLP 编码器
# ══════════════════════════════════════════════════════════════

class FactorEncoder:
    """2层 MLP: 因子统计摘要 → 潜在向量 φ

    支持两种训练模式:
      - mse:    自编码器 (encoder + decoder), 重建损失
      - triplet: 三元组损失 (φ 空间聚类), 只需 encoder

    设计取舍:
      - 纯 numpy 实现, 无 PyTorch/tensorflow 依赖
      - 参数 ~300 (8→16→6), CPU 上训练 <0.1s/次
      - 手动反向传播, 代码量换零依赖
    """

    def __init__(self, input_dim: int = 8, latent_dim: int = 6,
                 hidden_dim: int = 16, loss_type: str = 'mse',
                 seed: int = None):
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.loss_type = loss_type

        rng = np.random.RandomState(seed) if seed else np.random.RandomState()

        # Encoder: input → hidden → latent
        self.W1 = rng.randn(input_dim, hidden_dim) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden_dim)
        self.W2 = rng.randn(hidden_dim, latent_dim) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros(latent_dim)

        # Decoder (仅 mse 模式): latent → hidden → input
        self.W3 = rng.randn(latent_dim, hidden_dim) * np.sqrt(2.0 / latent_dim)
        self.b3 = np.zeros(hidden_dim)
        self.W4 = rng.randn(hidden_dim, input_dim) * np.sqrt(2.0 / hidden_dim)
        self.b4 = np.zeros(input_dim)

    def encode(self, stats: np.ndarray) -> np.ndarray:
        """stats (N, input_dim) → phi (N, latent_dim)"""
        h = np.maximum(0, stats @ self.W1 + self.b1)
        return h @ self.W2 + self.b2

    def decode(self, phi: np.ndarray) -> np.ndarray:
        """phi (N, latent_dim) → reconstructed stats (N, input_dim)"""
        h = np.maximum(0, phi @ self.W3 + self.b3)
        return h @ self.W4 + self.b4

    def _train_mse(self, stats: np.ndarray, lr: float) -> float:
        """单步 MSE 自编码器训练, 返回 loss"""
        N = stats.shape[0]

        # Forward
        h1 = np.maximum(0, stats @ self.W1 + self.b1)
        phi = h1 @ self.W2 + self.b2
        h2 = np.maximum(0, phi @ self.W3 + self.b3)
        recon = h2 @ self.W4 + self.b4

        loss = np.mean((recon - stats) ** 2)

        # Backward through decoder
        d_recon = 2 * (recon - stats) / N
        d_h2 = d_recon @ self.W4.T
        d_h2[h2 <= 0] = 0

        grad_W4 = h2.T @ d_recon
        grad_b4 = d_recon.sum(0)
        grad_W3 = phi.T @ d_h2
        grad_b3 = d_h2.sum(0)

        # Backward through latent
        d_phi = d_h2 @ self.W3.T

        # Backward through encoder
        grad_W2 = h1.T @ d_phi
        grad_b2 = d_phi.sum(0)
        d_h1 = d_phi @ self.W2.T
        d_h1[h1 <= 0] = 0
        grad_W1 = stats.T @ d_h1
        grad_b1 = d_h1.sum(0)

        # Update
        self.W1 -= lr * grad_W1; self.b1 -= lr * grad_b1
        self.W2 -= lr * grad_W2; self.b2 -= lr * grad_b2
        self.W3 -= lr * grad_W3; self.b3 -= lr * grad_b3
        self.W4 -= lr * grad_W4; self.b4 -= lr * grad_b4

        return float(loss)

    def _train_triplet(self, anchors: np.ndarray, positives: np.ndarray,
                       negatives: np.ndarray, margin: float, lr: float) -> float:
        """单步 Triplet Loss 训练, 返回 loss"""
        N = anchors.shape[0]

        # Forward
        def _fwd(x):
            h = np.maximum(0, x @ self.W1 + self.b1)
            return h, h @ self.W2 + self.b2

        h_a, phi_a = _fwd(anchors)
        h_p, phi_p = _fwd(positives)
        h_n, phi_n = _fwd(negatives)

        d_pos = np.sum((phi_a - phi_p) ** 2, axis=1)
        d_neg = np.sum((phi_a - phi_n) ** 2, axis=1)
        margin_loss = d_pos - d_neg + margin
        active = margin_loss > 0

        if not active.any():
            return 0.0

        active_mask = active[:, None].astype(float)
        # φ gradients
        g_phi_a = 2 * (phi_n - phi_p) * active_mask / N
        g_phi_p = -2 * (phi_a - phi_p) * active_mask / N
        g_phi_n = 2 * (phi_a - phi_n) * active_mask / N

        # W2, b2
        g_W2 = (h_a.T @ g_phi_a + h_p.T @ g_phi_p + h_n.T @ g_phi_n)
        g_b2 = g_phi_a.sum(0) + g_phi_p.sum(0) + g_phi_n.sum(0)

        # ReLU backward
        def _back_h(g_phi, h):
            g_h = g_phi @ self.W2.T
            g_h[h <= 0] = 0
            return g_h

        g_h_a = _back_h(g_phi_a, h_a)
        g_h_p = _back_h(g_phi_p, h_p)
        g_h_n = _back_h(g_phi_n, h_n)

        # W1, b1
        g_W1 = (anchors.T @ g_h_a + positives.T @ g_h_p + negatives.T @ g_h_n)
        g_b1 = g_h_a.sum(0) + g_h_p.sum(0) + g_h_n.sum(0)

        # Update
        self.W1 -= lr * g_W1; self.b1 -= lr * g_b1
        self.W2 -= lr * g_W2; self.b2 -= lr * g_b2

        return float(margin_loss[active].mean())

    def fit(self, archive_individuals: List[EncodedIndividual],
            steps: int = 50, lr: float = 0.01) -> List[float]:
        """用 archive 数据训练编码器, 返回 loss 历史"""
        valid_stats = [ind.stats for ind in archive_individuals
                       if ind.stats is not None]
        if len(valid_stats) < 10:
            return []
        stats_arr = np.stack(valid_stats)

        losses = []
        for _ in range(steps):
            if self.loss_type == 'triplet' and len(stats_arr) >= 3:
                # 构造三元组
                idx = np.random.choice(len(stats_arr), size=min(len(stats_arr), 64),
                                       replace=False)
                batch = stats_arr[idx]
                fitnesses = np.array([archive_individuals[i].fitness
                                      for i in idx])
                # 对每个锚点, 选 fitness 最近的正例和最远的负例
                anchors, positives, negatives = [], [], []
                for i in range(len(batch)):
                    diff = np.abs(fitnesses - fitnesses[i])
                    diff[i] = np.inf
                    p_idx = diff.argmin()
                    n_idx = diff.argmax()
                    anchors.append(batch[i])
                    positives.append(batch[p_idx])
                    negatives.append(batch[n_idx])
                margin = self._compute_dynamic_margin(stats_arr)
                loss = self._train_triplet(
                    np.array(anchors), np.array(positives),
                    np.array(negatives), margin, lr)
            else:
                # MSE 自编码器
                idx = np.random.choice(len(stats_arr),
                                       size=min(len(stats_arr), 64),
                                       replace=False)
                batch = stats_arr[idx]
                loss = self._train_mse(batch, lr)
            losses.append(loss)
        return losses

    def _compute_dynamic_margin(self, stats_arr: np.ndarray) -> float:
        """动态边界: archive 中任意两解间的最小 φ 距离"""
        if len(stats_arr) < 2:
            return 1.0
        sample_idx = np.random.choice(len(stats_arr),
                                       size=min(len(stats_arr), 50),
                                       replace=False)
        phi_sample = self.encode(stats_arr[sample_idx])
        dists = []
        for i in range(len(phi_sample)):
            d = np.sum((phi_sample[i] - phi_sample) ** 2, axis=1)
            d[i] = np.inf
            dists.append(d.min())
        return float(np.mean(dists)) if dists else 1.0


# ══════════════════════════════════════════════════════════════
# 因子统计提取
# ══════════════════════════════════════════════════════════════

def extract_factor_stats(factor_values: np.ndarray) -> np.ndarray:
    """从因子输出矩阵提取 8 维统计摘要。

    factor_values: (T_dates, N_assets) 的因子值矩阵
    返回: (8,) 的统计向量
      0: mean          均值
      1: std           标准差
      2: skew          偏度
      3: kurt          峰度
      4: hit_rate      >0 的比例
      5: autocorr_lag1 一阶自相关
      6: top_ratio     头部(>80%分位) vs 整体均值
      7: turnover      相邻两天排名的平均变化率
    """
    fv = np.nan_to_num(np.asarray(factor_values, dtype=float),
                       nan=0.0, posinf=0.0, neginf=0.0)
    flat = fv.ravel()
    flat = flat[np.isfinite(flat)]
    if len(flat) < 2:
        return np.zeros(8)

    stats = np.zeros(8)
    # Clip: 防止异常值导致编码器 matmul 溢出
    flat = np.clip(flat, -100.0, 100.0)
    stats[0] = float(np.mean(flat))
    stats[1] = float(np.std(flat)) if len(flat) > 1 else 0.0

    # Skew & Kurt
    if stats[1] > 1e-10:
        centered = flat - stats[0]
        stats[2] = float(np.mean(centered ** 3) / (stats[1] ** 3))
        stats[3] = float(np.mean(centered ** 4) / (stats[1] ** 4))
        stats[2] = max(-5.0, min(5.0, stats[2]))
        stats[3] = max(1.0, min(15.0, stats[3]))

    stats[4] = float(np.mean(flat > 0))

    # Autocorr lag-1 (行均值序列)
    if fv.ndim == 2 and fv.shape[0] > 1:
        row_mean = np.mean(fv, axis=1)
        row_mean = row_mean[np.isfinite(row_mean)]
        if len(row_mean) > 1:
            rm = row_mean - np.mean(row_mean)
            denom = np.sum(rm ** 2)
            if denom > 1e-10:
                stats[5] = float(np.sum(rm[:-1] * rm[1:]) / denom)
                stats[5] = max(-1.0, min(1.0, stats[5]))

    # Top ratio
    thresh = np.percentile(flat, 80) if len(flat) >= 5 else stats[0]
    top_vals = flat[flat >= thresh]
    stats[6] = float(np.mean(top_vals) / (abs(stats[0]) + 1e-10)) if len(top_vals) > 0 else 1.0

    # Turnover (日间排名变化)
    if fv.ndim == 2 and fv.shape[0] >= 2:
        rank_changes = []
        for t in range(1, min(fv.shape[0], 252)):
            try:
                r0 = np.argsort(np.argsort(fv[t - 1]))
                r1 = np.argsort(np.argsort(fv[t]))
                valid = np.isfinite(r0) & np.isfinite(r1)
                if valid.sum() > 1:
                    rank_changes.append(
                        np.mean(np.abs(r0[valid] - r1[valid])) / len(valid))
            except Exception:
                pass
        if rank_changes:
            stats[7] = float(np.mean(rank_changes[-20:]))

    return np.nan_to_num(stats, nan=0.0)


# ══════════════════════════════════════════════════════════════
# AuroraArchive — 非结构化 archive + DNS 局部竞争
# ══════════════════════════════════════════════════════════════

class AuroraArchive:
    """非结构化解档案, DNS (Dominated Novelty Search) 局部竞争替换。

    与 MAP-Elites 网格不同:
      - 无预定义网格, 个体在连续 φ 空间中按距离竞争
      - 容量上限 max_size, 满了触发局部替换
      - 替换规则: φ 空间内相互不被 fitness-dominates 的个体都保留
    """

    def __init__(self, max_size: int = 300, k_neighbors: int = 10):
        self.max_size = max_size
        self.k_neighbors = k_neighbors
        self._individuals: List[EncodedIndividual] = []
        self._phi_cache: Optional[np.ndarray] = None  # (N, latent_dim)
        self._stats_cache: Optional[np.ndarray] = None  # (N, input_dim)

    def __len__(self) -> int:
        return len(self._individuals)

    def __iter__(self):
        return iter(self._individuals)

    def _rebuild_cache(self):
        """重建 φ 缓存 (用于距离计算)"""
        phis = []
        stats_list = []
        for ind in self._individuals:
            if ind.phi is not None:
                phis.append(ind.phi)
                stats_list.append(ind.stats if ind.stats is not None
                                  else np.zeros_like(ind.phi))
        if phis:
            self._phi_cache = np.stack(phis)
            self._stats_cache = np.stack(stats_list)
        else:
            self._phi_cache = None
            self._stats_cache = None

    def uniform_sample(self, rng: random.Random) -> Optional[EncodedIndividual]:
        """均匀随机采样一个个体 (用于选择父代)"""
        if not self._individuals:
            return None
        return rng.choice(self._individuals)

    def try_add(self, individual: EncodedIndividual) -> bool:
        """尝试将个体加入 archive, DNS 局部竞争决定去留。

        1. archive 未满 → 直接加入
        2. archive 已满 → 在 φ 空间找 k 近邻, 检查支配关系
           - 如果个体被某个近邻支配 (距离近且 fitness 低) → 拒绝
           - 如果个体支配某个近邻 → 替换它
           - 否则 → 随机替换一个近邻
        """
        if individual.phi is None:
            return False

        # 容量未满
        if len(self._individuals) < self.max_size:
            self._individuals.append(individual)
            self._phi_cache = None  # 延迟重建
            return True

        # 容量已满 → DNS 局部竞争
        return self._dns_replace(individual)

    def _dns_replace(self, candidate: EncodedIndividual) -> bool:
        """DNS 局部竞争: 在 φ 空间找 k 近邻, 按支配关系决定替换"""
        if self._phi_cache is None:
            self._rebuild_cache()
        if self._phi_cache is None or len(self._phi_cache) < 1:
            self._individuals.append(candidate)
            return True

        # 计算 candidate 到所有 archive 个体的 φ 距离
        phi_c = candidate.phi.reshape(1, -1)
        dists = np.sum((self._phi_cache - phi_c) ** 2, axis=1)
        k = min(self.k_neighbors, len(dists))
        neighbor_idx = np.argpartition(dists, k)[:k]

        # 检查支配关系
        dominated_by = False
        for ni in neighbor_idx:
            neighbor = self._individuals[ni]
            if neighbor.fitness > candidate.fitness and dists[ni] < np.median(dists):
                # neighbor 更近且 fitness 更高 → candidate 被支配
                dominated_by = True
                break

        if dominated_by:
            return False  # 被支配, 拒绝

        # candidate 不被支配 → 找可以替换的目标
        # 优先替换: φ 接近且 fitness 更低的个体
        targets = []
        for ni in neighbor_idx:
            neighbor = self._individuals[ni]
            if candidate.fitness > neighbor.fitness:
                targets.append((ni, dists[ni]))

        if targets:
            # 替换 fitness 最低的邻近个体
            targets.sort(key=lambda x: self._individuals[x[0]].fitness)
            replace_idx = targets[0][0]
        else:
            # 所有近邻 fitness 都更高 → 随机替换最远的近邻
            farthest = neighbor_idx[np.argmax(dists[neighbor_idx])]
            replace_idx = farthest

        self._individuals[replace_idx] = candidate
        self._phi_cache = None
        return True

    def get_all(self) -> List[EncodedIndividual]:
        return list(self._individuals)

    def get_best(self) -> Optional[EncodedIndividual]:
        if not self._individuals:
            return None
        return max(self._individuals, key=lambda x: x.fitness)

    def extinction(self, retention: float = 0.10,
                   rng: random.Random = None) -> None:
        """周期性灭绝: 保留 best + 随机 retention% 的个体"""
        if not self._individuals or len(self._individuals) <= 1:
            return

        rng = rng or random.Random()
        best = self.get_best()
        if best is None:
            return

        retain_n = max(2, int(len(self._individuals) * retention))
        survivors = [best]
        others = [ind for ind in self._individuals if ind is not best]
        if others and retain_n > 1:
            survivors.extend(rng.sample(others,
                                        min(retain_n - 1, len(others))))

        self._individuals = survivors
        self._phi_cache = None
        self._stats_cache = None
        logger.info(
            f"[AURORA] 灭绝: {len(survivors)}/{retain_n + len(others) + 1} "
            f"个体存活 (保留 {retention:.0%})"
        )


# ══════════════════════════════════════════════════════════════
# AURORAEngine — 继承 GPEngine, 覆写选择和种群管理
# ══════════════════════════════════════════════════════════════

class AURORAEngine(GPEngine):
    """AURORA 启发式 GP 引擎

    继承 GPEngine, 复用: 变异/交叉/求值/缓存/TreeGenConfig
    覆写: 选择方式 (archive 均匀采样) + 种群管理 (DNS archive + 编码器 + 灭绝)
    """

    def __init__(self, data, fitness_calculator=None,
                 config: Dict = None,
                 seed_expressions: List[str] = None,
                 random_seed: int = None,
                 tree_gen_config: TreeGenConfig = None,
                 evaluator: Optional[Callable] = None,
                 future_returns=None, returns=None,
                 cache_db: str = '', source: str = '',
                 aura_config: Dict = None):
        """初始化 AURORA 引擎。

        aura_config: AURORA 专用配置, 覆盖 AURA_DEFAULT_CONFIG。
        """
        # 父类初始化 (GPEngine)
        super().__init__(
            data=data, fitness_calculator=fitness_calculator,
            config=config, seed_expressions=seed_expressions,
            random_seed=random_seed, tree_gen_config=tree_gen_config,
            evaluator=evaluator, future_returns=future_returns,
            returns=returns, cache_db=cache_db, source=source,
        )

        # AURORA 配置
        ac = dict(AURA_DEFAULT_CONFIG)
        if aura_config:
            ac.update(aura_config)
        self._archive_size = ac['archive_size']
        self._dns_k = ac['dns_k_neighbors']
        self._enc_input_dim = ac['encoder_input_dim']
        self._enc_latent_dim = ac['encoder_latent_dim']
        self._enc_hidden_dim = ac['encoder_hidden_dim']
        self._enc_lr = ac['encoder_lr']
        self._enc_steps = ac['encoder_steps_per_update']
        self._enc_update_every = ac['encoder_update_every']
        self._enc_loss_type = ac['encoder_loss_type']
        self._extinction_period = ac['extinction_period']
        self._extinction_retention = ac['extinction_retention']
        self._children_per_gen = ac['children_per_gen']

        # 编码器
        self.encoder = FactorEncoder(
            input_dim=self._enc_input_dim,
            latent_dim=self._enc_latent_dim,
            hidden_dim=self._enc_hidden_dim,
            loss_type=self._enc_loss_type,
            seed=random_seed,
        )

        # Archive
        self.archive = AuroraArchive(
            max_size=self._archive_size,
            k_neighbors=self._dns_k,
        )

        # 状态
        self._gen_best_individual: Optional[EncodedIndividual] = None
        self._encoder_losses: List[float] = []

    # ── 选择覆写 ──

    def _select_parent(self) -> Individual:
        """AURORA: 从 archive 均匀随机采样父代"""
        parent = self.archive.uniform_sample(self.tree_gen_config.rng)
        if parent is not None:
            return parent
        # fallback: archive 为空时用锦标赛
        return super()._select_parent()

    # ── 编码 ──

    def _encode_individual(self, ind: EncodedIndividual, factor_values: np.ndarray):
        """对单个个体提取统计 + 编码为 φ"""
        if factor_values is None:
            return
        ind.stats = extract_factor_stats(factor_values)
        ind.phi = self.encoder.encode(ind.stats.reshape(1, -1)).ravel()

    def _encode_population(self, population: List[EncodedIndividual],
                           factor_values_map: Dict[str, np.ndarray]):
        """批量编码种群。factor_values_map: {expr_str → factor_values}"""
        for ind in population:
            fv = factor_values_map.get(ind.expression_str)
            self._encode_individual(ind, fv)

    # ── 子代生成 ──

    def _generate_children(self, gen: int, count: int) -> List[Individual]:
        """生成子代 (复用 GPEngine 的交叉/变异/随机生成)"""
        cfg = self.tree_gen_config
        rng = cfg.rng
        children = []

        while len(children) < count:
            r = rng.random()
            if r < self.crossover_prob and len(self.archive) >= 2:
                p1 = self._select_parent()
                p2 = self._select_parent()
                if p1 is not None and p2 is not None:
                    child = self._crossover(p1, p2)
                else:
                    child = Individual(tree=_random_tree(cfg, self.max_depth),
                                       generation=gen + 1, age=1)
            elif r < self.crossover_prob + self.mutation_prob:
                parent = self._select_parent()
                if parent is not None:
                    child = self._mutate(parent)
                else:
                    child = Individual(tree=_random_tree(cfg, self.max_depth),
                                       generation=gen + 1, age=1)
            else:
                parent = self._select_parent()
                if parent is not None:
                    child = Individual(tree=copy.deepcopy(parent.tree),
                                       generation=gen + 1, age=1)
                else:
                    child = Individual(tree=_random_tree(cfg, self.max_depth),
                                       generation=gen + 1, age=1)

            child.age = 1
            children.append(child)

        return children[:count]

    # ── 编码器训练 ──

    def _train_encoder_step(self):
        """用当前 archive 数据训练编码器"""
        individuals = self.archive.get_all()
        valid = [ind for ind in individuals
                 if ind.stats is not None and ind.fitness > -999]
        if len(valid) < 10:
            return

        losses = self.encoder.fit(valid, steps=self._enc_steps,
                                  lr=self._enc_lr)
        self._encoder_losses.extend(losses)

        # 重新编码
        for ind in valid:
            if ind.stats is not None:
                ind.phi = self.encoder.encode(
                    ind.stats.reshape(1, -1)).ravel()
        # 重置 archive 的 φ 缓存
        self.archive._rebuild_cache()

    # ── 主循环覆写 ──

    def run(self, verbose: bool = True,
            callback: Callable[[int, Individual, Dict], None] = None) -> 'AURORAEngine':
        """AURORA 主循环。

        每代:
          1. 生成子代 (交叉/变异/随机)
          2. 评估子代
          3. 提取统计 + 编码为 φ
          4. DNS 局部竞争加入 archive
          5. 周期性: 训练编码器 / 灭绝事件
        """
        n_loaded = self.fitness_cache.load()
        if n_loaded and verbose:
            logger.info(f"[AURORA] 加载 SQLite 缓存 {n_loaded} 条")

        # ── 初始化: 种子 + 随机个体 → archive ──
        self._initialize_archive()

        for gen in range(self.generations):
            # 1. 生成子代
            children = self._generate_children(gen, self._children_per_gen)

            # 2. 评估子代 (复用 GPEngine._evaluate_population)
            self.population = children
            self._evaluate_population()
            evaluated = [ind for ind in self.population if ind.fitness > -999]

            # 3+4. 编码 + DNS 归档
            factor_values_map = {}  # 这里需要 evaluator 求值
            for ind in evaluated:
                # 重新求值拿到 factor_values (用于统计提取)
                if self._evaluator and ind.expression_str:
                    try:
                        fv = self._evaluator(self.data, ind.tree)
                        factor_values_map[ind.expression_str] = fv
                    except Exception:
                        pass

            for ind in evaluated:
                enc_ind = EncodedIndividual.from_individual(ind)
                fv = factor_values_map.get(ind.expression_str)
                self._encode_individual(enc_ind, fv)
                self.archive.try_add(enc_ind)

            # 5. 统计
            gen_best = self.archive.get_best()
            valid_count = len(self.archive)
            gen_stats = {
                'generation': gen,
                'best_fitness': gen_best.fitness if gen_best else -999,
                'best_expression': gen_best.expression_str if gen_best else '',
                'archive_size': valid_count,
                'encoder_loss': self._encoder_losses[-1] if self._encoder_losses else 0.0,
            }
            self.history.append(gen_stats)

            if gen_best and (self._gen_best_individual is None or
                             gen_best.fitness > self._gen_best_individual.fitness):
                self._gen_best_individual = gen_best

            # 6. 编码器重训
            if gen > 0 and gen % self._enc_update_every == 0:
                self._train_encoder_step()

            # 7. 灭绝事件
            if gen > 0 and gen % self._extinction_period == 0:
                self.archive.extinction(
                    retention=self._extinction_retention,
                    rng=self.tree_gen_config.rng,
                )

            # 8. 回调
            if callback:
                callback(gen, gen_best or Individual(
                    tree=ast.Expression(body=None)), gen_stats)

            if verbose and gen % 5 == 0:
                best_f = gen_best.fitness if gen_best else -999
                logger.info(
                    f"Gen {gen:3d} | best_f={best_f:.3f} "
                    f"archive={valid_count} | children={len(evaluated)}"
                )

        self.fitness_cache.save()
        return self

    # ── 初始化 ──

    def _initialize_archive(self):
        """初始化 archive: 种子表达式 + 随机生成"""
        cfg = self.tree_gen_config
        rng = cfg.rng

        seeds = list(self.seed_expressions)
        initial = []

        # 种子个体
        seed_n = int(self._archive_size * self.seed_ratio)
        for expr_str in seeds[:seed_n]:
            try:
                ind = EncodedIndividual.from_expr(expr_str, generation=0, is_seed=True)
                initial.append(ind)
            except Exception as e:
                logger.warning(f"[AURORA] 种子解析失败: {expr_str[:60]} ({e})")

        # 种子变异补齐
        while len(initial) < seed_n and seeds:
            base = Individual.from_expr(rng.choice(seeds))
            mutated = _mutate_subtree(cfg, base.tree, max_depth=3)
            initial.append(EncodedIndividual(
                tree=mutated, generation=0, age=1, is_seed=True))

        # 随机个体填充
        remaining = self._archive_size - len(initial)
        explore_n = int(remaining * self._explore_ratio)
        for i in range(remaining):
            if i < explore_n and hasattr(self, '_explore_ratio'):
                tree = _random_tree_explore(cfg, self.max_depth)
            else:
                tree = _random_tree(cfg, self.max_depth)
            initial.append(EncodedIndividual(tree=tree, generation=0, age=1))

        # 评估初始种群
        self.population = initial[:self._archive_size]
        self._evaluate_population()

        # 提取统计 + 编码 + 归档
        for ind in self.population:
            if ind.fitness <= -999:
                continue
            enc_ind = ind if isinstance(ind, EncodedIndividual) \
                      else EncodedIndividual.from_individual(ind)
            if self._evaluator and enc_ind.expression_str:
                try:
                    fv = self._evaluator(self.data, enc_ind.tree)
                    self._encode_individual(enc_ind, fv)
                except Exception:
                    pass
            self.archive.try_add(enc_ind)

        # 初始编码器训练
        if len(self.archive) >= 10:
            self._train_encoder_step()

        logger.info(
            f"[AURORA] 初始化: archive={len(self.archive)}/{self._archive_size}"
        )

    # ── 结果 ──

    def best(self) -> Optional[Individual]:
        return self._gen_best_individual or self.archive.get_best()

    def top(self, n: int = 10) -> List[Individual]:
        individuals = self.archive.get_all()
        return sorted([i for i in individuals if i.fitness > -999],
                      key=lambda x: x.fitness, reverse=True)[:n]

    def report(self) -> str:
        if not self.history:
            return "尚未运行 AURORA GP 搜索"
        lines = [
            "=" * 60,
            "AURORA 启发式 GP 搜索报告",
            "=" * 60,
            f"Archive 容量: {self._archive_size}  代数: {self.generations}  "
            f"最大深度: {self.max_depth}  种子: {len(self.seed_expressions)}",
            f"编码器: loss={self._enc_loss_type}  latent_dim={self._enc_latent_dim}",
            f"灭绝周期: {self._extinction_period}  "
            f"保留比例: {self._extinction_retention:.0%}",
            "",
        ]
        best_ind = self.best()
        if best_ind:
            lines += [
                f"最优: fitness={best_ind.fitness:.3f}, "
                f"depth={best_ind.depth}",
                f"表达式: {best_ind.expression_str}",
                "",
            ]
        lines += [
            f"{'Gen':>4}  {'Best_F':>8}  {'Archive':>8}  {'EncLoss':>8}",
            "-" * 40,
        ]
        for s in self.history:
            if s['generation'] % 5 == 0 or s['generation'] == self.history[-1]['generation']:
                lines.append(
                    f"{s['generation']:4d}  {s['best_fitness']:8.3f}  "
                    f"{s['archive_size']:8d}  "
                    f"{s.get('encoder_loss', 0):8.4f}"
                )
        return '\n'.join(lines)
