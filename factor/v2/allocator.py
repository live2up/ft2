"""
权重分配器抽象层

设计思路：
--------
1. WeightAllocator(ABC)：权重分配抽象基类，将因子得分数组转为持仓权重数组
2. TopNEqualWeight：前 N 名等权持有，其余 0
3. ScoreProportional：正得分按比例分配（归一化到总权重=1），负得分剔除
4. RiskParity：基于历史协方差矩阵的风险平价权重

使用方式：
--------
>>> allocator = TopNEqualWeight(top_n=5)
>>> weights = allocator.allocate(scores)  # scores: shape (N,) 的因子得分
>>> # weights: shape (N,) 的持仓权重，和为 1（全为正时）或 ≤1

与 V1 的关系：
------------
V1 中权重分配逻辑内嵌在回测脚本中，每次手工实现。
V2 抽象为独立模块，Pipeline 通过依赖注入使用，便于切换和扩展。

依赖说明：
--------
numpy + scipy（RiskParity 需要 scipy.optimize），不依赖 ft2 其他模块。
"""

from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
import warnings


class WeightAllocator(ABC):
    """权重分配器抽象基类
    
    将因子得分数组转换为持仓权重数组。
    权重数组满足：
    - 非负（纯多头）或允许做空（由子类决定）
    - 总和 ≤ 1（允许空仓/部分持仓）
    """

    @abstractmethod
    def allocate(self, scores: np.ndarray) -> np.ndarray:
        """将因子得分转为持仓权重
        
        Args:
            scores: 因子得分数组，shape (N,)，值越大表示越看多
            
        Returns:
            np.ndarray: 持仓权重数组，shape (N,)，非负且和 ≤ 1
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class TopNEqualWeight(WeightAllocator):
    """前 N 名等权持有
    
    选取得分最高的 top_n 个标的，等权分配。
    其余标的权重为 0。
    
    典型场景：横截面选股，每月调仓持有 Top 5/10 行业。
    """

    def __init__(self, top_n: int = 5):
        """初始化
        
        Args:
            top_n: 持有的标的数量
            
        Raises:
            ValueError: 如果 top_n < 1
        """
        if top_n < 1:
            raise ValueError(f"top_n 必须 >= 1，当前值: {top_n}")
        self.top_n = top_n

    def allocate(self, scores: np.ndarray) -> np.ndarray:
        """Top N 等权分配
        
        处理边界：
        - NaN 得分排在最末尾（不会被选中）
        - 如果有效得分数量 < top_n，则只持有有效得分的标的
        
        Args:
            scores: 因子得分数组，shape (N,)
            
        Returns:
            np.ndarray: 权重数组，shape (N,)
        """
        n = len(scores)
        if n == 0:
            return np.array([])

        # NaN 处理：将 NaN 替换为 -inf，确保排在末尾
        valid_mask = ~np.isnan(scores)
        valid_scores = np.where(valid_mask, scores, -np.inf)
        n_valid = valid_mask.sum()

        # 实际持有的数量 = min(top_n, 有效标的数)
        actual_n = min(self.top_n, n_valid)
        if actual_n == 0:
            return np.zeros(n)

        # 选出 top N 的索引
        # [修复] 2026-05-20 argpartition 顺序不确定 → 加 argsort 固定排序
        # 旧实现: np.argpartition(-valid_scores, actual_n-1)[:actual_n]
        # 问题: argpartition 不保证 top N 内部顺序确定性，导致相同因子每次运行
        #       产生不同持仓 → 换手率计算不稳定
        top_indices = np.argpartition(-valid_scores, actual_n - 1)[:actual_n]
        top_indices = top_indices[np.argsort(-valid_scores[top_indices])]

        weights = np.zeros(n)
        weights[top_indices] = 1.0 / actual_n

        return weights

    def __repr__(self) -> str:
        return f"TopNEqualWeight(top_n={self.top_n})"


class ScoreProportional(WeightAllocator):
    """正得分按比例分配
    
    只有得分为正的标的才分配权重，权重与其得分成正比。
    负得分或 NaN 得分的标的权重为 0。
    
    典型场景：信号强弱决定仓位大小，信号越强仓位越重。
    """

    def allocate(self, scores: np.ndarray) -> np.ndarray:
        """正得分按比例分配
        
        Args:
            scores: 因子得分数组，shape (N,)
            
        Returns:
            np.ndarray: 权重数组，shape (N,)，和 = 1（有正得分时）
        """
        n = len(scores)
        if n == 0:
            return np.array([])

        # 只保留正得分，NaN 视为 0
        positive_scores = np.nan_to_num(scores, nan=0.0)
        positive_scores = np.maximum(positive_scores, 0)

        total = positive_scores.sum()
        if total <= 0:
            return np.zeros(n)

        return positive_scores / total

    def __repr__(self) -> str:
        return "ScoreProportional()"


class RiskParity(WeightAllocator):
    """风险平价权重
    
    基于历史协方差矩阵，使每个持仓对组合风险的贡献相等。
    
    使用 scipy.optimize.minimize 求解：
        minimize Σ_i (w_i * (Σw)_i - target_risk_contrib)²
        subject to w_i >= 0, Σ w_i = 1
    
    注意：需要至少 top_n × 2 个历史数据点才能可靠估计协方差。
    """

    def __init__(self, lookback: int = 60):
        """初始化
        
        Args:
            lookback: 用于估计协方差的历史窗口（交易日数）
        """
        if lookback < 10:
            raise ValueError(f"lookback 必须 >= 10（最小 10 个观测），当前值: {lookback}")
        self.lookback = lookback

    def allocate(self, scores: np.ndarray,
                 returns_history: Optional[np.ndarray] = None) -> np.ndarray:
        """风险平价权重分配
        
        Args:
            scores: 因子得分数组，shape (N,)，用于筛选标的（取 top_n 个）
            returns_history: 历史收益率矩阵，shape (lookback, M)，可选。
                           如果为 None 则退化为等权分配。
                           
        Returns:
            np.ndarray: 权重数组，shape (N,)
        """
        n = len(scores)
        if n == 0:
            return np.array([])

        # 无历史收益数据时退化为等权
        if returns_history is None or returns_history.shape[1] == 0:
            warnings.warn("RiskParity: 无历史收益数据，退化为等权分配")
            # 取所有非 NaN 得分标的等权
            valid_mask = ~np.isnan(scores)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                return np.zeros(n)
            weights = np.zeros(n)
            weights[valid_mask] = 1.0 / n_valid
            return weights

        # 基于历史收益计算风险平价
        try:
            from scipy.optimize import minimize

            # 估计协方差矩阵
            cov = np.cov(returns_history, rowvar=False)
            # 添加小量正则化防止奇异矩阵
            cov = cov + np.eye(cov.shape[0]) * 1e-6

            m = cov.shape[0]  # 实际参与计算的标的数

            def risk_parity_objective(w):
                portfolio_risk = w @ cov @ w
                if portfolio_risk <= 0:
                    return 1e10
                sigma = np.sqrt(portfolio_risk)
                # 边际风险贡献
                marginal_contrib = cov @ w
                # 风险贡献 = w_i * (Σw)_i / σ
                risk_contrib = w * marginal_contrib / sigma
                # 目标：每个标的风险贡献相等 = sigma / m
                target = sigma / m
                return np.sum((risk_contrib - target) ** 2)

            # 初始权重：等权
            w0 = np.ones(m) / m
            constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]
            bounds = [(0, 1) for _ in range(m)]

            result = minimize(
                risk_parity_objective, w0,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints,
                options={'maxiter': 1000, 'ftol': 1e-10}
            )

            if result.success:
                opt_weights = result.x
                # 归一化
                opt_weights = np.maximum(opt_weights, 0)
                total = opt_weights.sum()
                if total > 0:
                    opt_weights = opt_weights / total
            else:
                warnings.warn(f"RiskParity: 优化未收敛，退化为等权。{result.message}")
                opt_weights = np.ones(m) / m

            # 扩展回原始 N 维度（可能有 NaN 得分被排除的标的）
            full_weights = np.zeros(n)
            full_weights[:m] = opt_weights
            return full_weights

        except ImportError:
            warnings.warn("RiskParity: scipy 不可用，退化为等权分配")
            valid_mask = ~np.isnan(scores)
            n_valid = valid_mask.sum()
            if n_valid == 0:
                return np.zeros(n)
            weights = np.zeros(n)
            weights[valid_mask] = 1.0 / n_valid
            return weights

    def __repr__(self) -> str:
        return f"RiskParity(lookback={self.lookback})"
