"""
因子组合器模块

设计思路：
---------
1. 多因子组合：将多个因子组合成综合因子
2. 权重优化：支持多种权重分配方法
3. 因子正交化：去除因子间的多重共线性
4. 组合优化：最大化IC/IR，最小化波动率等

使用方式：
---------
1. 创建FactorCombiner实例
2. 添加需要组合的因子
3. 选择组合方法和参数
4. 执行组合计算
5. 获取组合因子结果
"""

import warnings
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any, Tuple
from enum import Enum
from dataclasses import dataclass
import pandas as pd
import numpy as np
from scipy import stats, optimize
# 可选依赖：scikit-learn
try:
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False
    PCA = None
    StandardScaler = None
import logging

from .base import Factor, FactorMetadata, FactorCategory, FactorFrequency
from .validator import FactorValidator

logger = logging.getLogger(__name__)


class CombinationMethod(Enum):
    """组合方法枚举"""
    EQUAL_WEIGHT = "equal_weight"          # 等权组合
    IC_WEIGHT = "ic_weight"                # IC加权
    IR_WEIGHT = "ir_weight"                # IR加权
    MAX_DIVERSIFICATION = "max_diversification"  # 最大分散化
    MIN_VARIANCE = "min_variance"          # 最小方差
    MAX_SHARPE = "max_sharpe"              # 最大夏普比率
    RISK_PARITY = "risk_parity"            # 风险平价
    PCA = "pca"                            # 主成分分析


class OrthogonalizationMethod(Enum):
    """正交化方法枚举"""
    NONE = "none"                          # 不进行正交化
    RESIDUAL = "residual"                  # 残差正交化
    PCA = "pca"                            # PCA正交化
    GRAM_SCHMIDT = "gram_schmidt"          # 格拉姆-施密特正交化


class StandardizationMethod(Enum):
    """因子标准化方法枚举"""
    NONE = "none"                          # 不标准化（直接用原始值）
    ZSCORE = "zscore"                      # 横截面Z-score标准化
    RANK = "rank"                          # 横截面排名标准化（百分位）
    MINMAX = "minmax"                      # 横截面Min-Max标准化（归一化到[0,1]）


@dataclass
class CombinationResult:
    """组合结果数据类"""
    combined_factor: pd.DataFrame          # 组合后的因子值
    weights: Dict[str, float]              # 各因子权重
    method: CombinationMethod              # 组合方法
    orthogonalization: OrthogonalizationMethod  # 正交化方法
    standardization: StandardizationMethod      # 标准化方法
    metrics: Dict[str, Any]                # 组合指标
    timestamp: datetime = datetime.now()   # 时间戳


class FactorCombiner:
    """因子组合器"""
    
    def __init__(self, 
                 factor_values: Optional[Dict[str, pd.DataFrame]] = None,
                 future_returns: Optional[pd.DataFrame] = None):
        """
        初始化因子组合器
        
        Args:
            factor_values: 因子值字典，key为因子名称，value为因子值DataFrame
            future_returns: 未来收益率DataFrame（用于计算IC/IR等指标）
        """
        self.factor_values = factor_values or {}
        self.future_returns = future_returns
        
        # 缓存
        self._orthogonalized_factors = {}
        self._combination_results = {}
        
    def add_factor(self, name: str, factor_values: pd.DataFrame):
        """
        添加因子
        
        Args:
            name: 因子名称
            factor_values: 因子值DataFrame
        """
        self.factor_values[name] = factor_values
        # 清空相关缓存
        self._orthogonalized_factors.pop(name, None)
        
    def remove_factor(self, name: str):
        """
        移除因子
        
        Args:
            name: 因子名称
        """
        if name in self.factor_values:
            del self.factor_values[name]
            # 清空相关缓存
            self._orthogonalized_factors.pop(name, None)
            
    def set_future_returns(self, future_returns: pd.DataFrame):
        """
        设置未来收益率
        
        Args:
            future_returns: 未来收益率DataFrame
        """
        self.future_returns = future_returns
        
    def _validate_data(self) -> bool:
        """
        验证数据有效性
        
        Returns:
            bool: 数据是否有效
        """
        if not self.factor_values:
            logger.error("没有因子数据")
            return False
            
        # 检查所有因子数据形状是否一致
        shapes = set()
        for name, values in self.factor_values.items():
            if values is None or values.empty:
                logger.error(f"因子 {name} 数据为空")
                return False
            shapes.add(values.shape)
            
        if len(shapes) > 1:
            logger.error("因子数据形状不一致")
            return False
            
        return True
    
    def _align_data(self, factor_names: Optional[List[str]] = None) -> Tuple[pd.DataFrame, ...]:
        """
        对齐数据
        
        Args:
            factor_names: 因子名称列表，如果为None则使用所有因子
            
        Returns:
            Tuple[pd.DataFrame, ...]: 对齐后的因子数据元组
        """
        if not self._validate_data():
            raise ValueError("数据无效")
            
        if factor_names is None:
            factor_names = list(self.factor_values.keys())
            
        # 获取第一个因子的索引作为基准
        first_factor = self.factor_values[factor_names[0]]
        common_index = first_factor.index
        common_columns = first_factor.columns
        
        # 对齐所有因子
        aligned_factors = []
        for name in factor_names:
            factor_data = self.factor_values[name]
            
            # 重新索引以对齐
            aligned = factor_data.reindex(index=common_index, columns=common_columns)
            aligned_factors.append(aligned)
            
        # 对齐未来收益率（如果存在）
        aligned_returns = None
        if self.future_returns is not None:
            aligned_returns = self.future_returns.reindex(
                index=common_index, columns=common_columns
            )
            
        return tuple(aligned_factors), aligned_returns
    
    def orthogonalize(self, 
                     factor_names: List[str],
                     method: OrthogonalizationMethod = OrthogonalizationMethod.RESIDUAL,
                     reference_factor: Optional[str] = None) -> Dict[str, pd.DataFrame]:
        """
        因子正交化
        
        Args:
            factor_names: 需要正交化的因子名称列表
            method: 正交化方法
            reference_factor: 参考因子（用于残差正交化）
            
        Returns:
            Dict[str, pd.DataFrame]: 正交化后的因子值
        """
        if not factor_names:
            return {}
            
        # 检查缓存
        cache_key = f"{'_'.join(sorted(factor_names))}_{method.value}"
        if cache_key in self._orthogonalized_factors:
            return self._orthogonalized_factors[cache_key]
            
        # 对齐数据
        aligned_factors, _ = self._align_data(factor_names)
        
        if method == OrthogonalizationMethod.NONE:
            # 不进行正交化，直接返回原始数据
            result = {name: factor for name, factor in zip(factor_names, aligned_factors)}
            self._orthogonalized_factors[cache_key] = result
            return result
            
        elif method == OrthogonalizationMethod.RESIDUAL:
            # 残差正交化
            if reference_factor is None:
                # 使用第一个因子作为参考
                reference_idx = 0
            else:
                reference_idx = factor_names.index(reference_factor)
                
            reference = aligned_factors[reference_idx]
            result = {}
            
            for i, (name, factor) in enumerate(zip(factor_names, aligned_factors)):
                if i == reference_idx:
                    # 参考因子保持不变
                    result[name] = factor
                else:
                    # 对其他因子进行残差正交化
                    orthogonalized = self._residual_orthogonalization(factor, reference)
                    result[name] = orthogonalized
                    
        elif method == OrthogonalizationMethod.PCA:
            # PCA正交化
            result = self._pca_orthogonalization(factor_names, aligned_factors)
            
        elif method == OrthogonalizationMethod.GRAM_SCHMIDT:
            # 格拉姆-施密特正交化
            result = self._gram_schmidt_orthogonalization(factor_names, aligned_factors)
            
        else:
            raise ValueError(f"不支持的正交化方法: {method}")
            
        self._orthogonalized_factors[cache_key] = result
        return result
    
    def _residual_orthogonalization(self, 
                                   factor: pd.DataFrame, 
                                   reference: pd.DataFrame) -> pd.DataFrame:
        """
        残差正交化
        
        Args:
            factor: 需要正交化的因子
            reference: 参考因子
            
        Returns:
            pd.DataFrame: 正交化后的因子
        """
        # 将DataFrame转换为二维数组以便计算
        factor_flat = factor.values.flatten()
        reference_flat = reference.values.flatten()
        
        # 去除NaN值
        valid_mask = ~np.isnan(factor_flat) & ~np.isnan(reference_flat)
        if valid_mask.sum() < 10:
            logger.warning("有效数据点不足，跳过正交化")
            return factor
            
        factor_valid = factor_flat[valid_mask]
        reference_valid = reference_flat[valid_mask]
        
        # 线性回归：factor = alpha + beta * reference + epsilon
        # 使用最小二乘法估计beta
        X = np.column_stack([np.ones_like(reference_valid), reference_valid])
        beta = np.linalg.lstsq(X, factor_valid, rcond=None)[0]
        
        # 计算残差：epsilon = factor - (alpha + beta * reference)
        epsilon = factor_valid - (beta[0] + beta[1] * reference_valid)
        
        # 将残差填充回原始形状
        result_flat = np.full_like(factor_flat, np.nan)
        result_flat[valid_mask] = epsilon
        
        # 重塑为原始形状
        result = pd.DataFrame(
            result_flat.reshape(factor.shape),
            index=factor.index,
            columns=factor.columns
        )
        
        return result
    
    def _pca_orthogonalization(self, 
                              factor_names: List[str],
                              aligned_factors: List[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        PCA正交化
        
        Args:
            factor_names: 因子名称列表
            aligned_factors: 对齐后的因子数据
            
        Returns:
            Dict[str, pd.DataFrame]: PCA正交化后的因子
        """
        if not HAS_SKLEARN:
            logger.warning("scikit-learn未安装，跳过PCA正交化，使用原始因子")
            return {name: factor for name, factor in zip(factor_names, aligned_factors)}
        
        # 将因子数据堆叠为三维数组 (n_factors, n_dates, n_symbols)
        factor_array = np.stack([f.values for f in aligned_factors])
        
        # 重塑为二维数组 (n_factors, n_dates * n_symbols)
        n_factors, n_dates, n_symbols = factor_array.shape
        factor_2d = factor_array.reshape(n_factors, -1)
        
        # 去除包含NaN的列
        valid_cols = ~np.any(np.isnan(factor_2d), axis=0)
        if valid_cols.sum() < n_factors * 10:
            logger.warning("有效数据点不足，跳过PCA正交化")
            return {name: factor for name, factor in zip(factor_names, aligned_factors)}
            
        factor_valid = factor_2d[:, valid_cols]
        
        # 标准化
        scaler = StandardScaler()
        factor_scaled = scaler.fit_transform(factor_valid.T).T
        
        # PCA分解
        pca = PCA(n_components=n_factors)
        pca_components = pca.fit_transform(factor_scaled.T).T
        
        # 将主成分转换回原始形状
        result = {}
        for i, name in enumerate(factor_names):
            # 重建因子值（使用第i个主成分）
            factor_reconstructed = np.full(factor_2d.shape[1], np.nan)
            factor_reconstructed[valid_cols] = pca_components[i]
            
            # 重塑为原始形状
            factor_reshaped = factor_reconstructed.reshape(n_dates, n_symbols)
            result[name] = pd.DataFrame(
                factor_reshaped,
                index=aligned_factors[0].index,
                columns=aligned_factors[0].columns
            )
            
        return result
    
    def _gram_schmidt_orthogonalization(self,
                                       factor_names: List[str],
                                       aligned_factors: List[pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        格拉姆-施密特正交化
        
        Args:
            factor_names: 因子名称列表
            aligned_factors: 对齐后的因子数据
            
        Returns:
            Dict[str, pd.DataFrame]: 正交化后的因子
        """
        # 将因子数据堆叠为三维数组 (n_factors, n_dates, n_symbols)
        factor_array = np.stack([f.values for f in aligned_factors])
        
        # 重塑为二维数组 (n_factors, n_dates * n_symbols)
        n_factors, n_dates, n_symbols = factor_array.shape
        factor_2d = factor_array.reshape(n_factors, -1)
        
        # 去除包含NaN的列
        valid_cols = ~np.any(np.isnan(factor_2d), axis=0)
        if valid_cols.sum() < n_factors * 10:
            logger.warning("有效数据点不足，跳过格拉姆-施密特正交化")
            return {name: factor for name, factor in zip(factor_names, aligned_factors)}
            
        factor_valid = factor_2d[:, valid_cols]
        
        # 格拉姆-施密特正交化
        orthogonal_basis = []
        
        for i in range(n_factors):
            # 当前因子
            v = factor_valid[i].copy()
            
            # 减去在前面的正交基上的投影
            for u in orthogonal_basis:
                projection = np.dot(v, u) / np.dot(u, u)
                v = v - projection * u
                
            # 归一化
            norm = np.linalg.norm(v)
            if norm > 1e-10:
                v = v / norm
                
            orthogonal_basis.append(v)
            
        # 将正交基转换回原始形状
        result = {}
        for i, name in enumerate(factor_names):
            factor_reconstructed = np.full(factor_2d.shape[1], np.nan)
            factor_reconstructed[valid_cols] = orthogonal_basis[i]
            
            # 重塑为原始形状
            factor_reshaped = factor_reconstructed.reshape(n_dates, n_symbols)
            result[name] = pd.DataFrame(
                factor_reshaped,
                index=aligned_factors[0].index,
                columns=aligned_factors[0].columns
            )
            
        return result
    
    def calculate_weights(self,
                         factor_names: List[str],
                         method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT,
                         lookforward: int = 1,
                         **kwargs) -> Dict[str, float]:
        """
        计算因子权重
        
        Args:
            factor_names: 因子名称列表
            method: 组合方法
            lookforward: 未来期数（用于计算IC/IR）
            **kwargs: 方法特定参数
            
        Returns:
            Dict[str, float]: 因子权重字典
        """
        if not factor_names:
            return {}
            
        # 对齐数据
        aligned_factors, aligned_returns = self._align_data(factor_names)
        
        if method == CombinationMethod.EQUAL_WEIGHT:
            # 等权组合
            weight = 1.0 / len(factor_names)
            return {name: weight for name in factor_names}
            
        elif method == CombinationMethod.IC_WEIGHT:
            # IC加权
            return self._ic_weighted_weights(factor_names, aligned_factors, 
                                           aligned_returns, lookforward)
            
        elif method == CombinationMethod.IR_WEIGHT:
            # IR加权
            return self._ir_weighted_weights(factor_names, aligned_factors,
                                           aligned_returns, lookforward)
            
        elif method == CombinationMethod.MAX_DIVERSIFICATION:
            # 最大分散化权重
            return self._max_diversification_weights(factor_names, aligned_factors)
            
        elif method == CombinationMethod.MIN_VARIANCE:
            # 最小方差权重
            return self._min_variance_weights(factor_names, aligned_factors)
            
        elif method == CombinationMethod.MAX_SHARPE:
            # 最大夏普比率权重
            return self._max_sharpe_weights(factor_names, aligned_factors,
                                          aligned_returns, lookforward)
            
        elif method == CombinationMethod.RISK_PARITY:
            # 风险平价权重
            return self._risk_parity_weights(factor_names, aligned_factors)
            
        elif method == CombinationMethod.PCA:
            # PCA权重
            return self._pca_weights(factor_names, aligned_factors)
            
        else:
            raise ValueError(f"不支持的组合方法: {method}")
    
    def _ic_weighted_weights(self,
                            factor_names: List[str],
                            aligned_factors: List[pd.DataFrame],
                            aligned_returns: Optional[pd.DataFrame],
                            lookforward: int) -> Dict[str, float]:
        """
        IC加权权重
        
        Args:
            factor_names: 因子名称列表
            aligned_factors: 对齐后的因子数据
            aligned_returns: 对齐后的收益率数据
            lookforward: 未来期数
            
        Returns:
            Dict[str, float]: IC加权权重
        """
        if aligned_returns is None:
            logger.warning("没有收益率数据，使用等权组合")
            weight = 1.0 / len(factor_names)
            return {name: weight for name in factor_names}
            
        # 计算每个因子的IC
        ics = []
        for i, factor in enumerate(aligned_factors):
            validator = FactorValidator(factor_values=factor, 
                                       future_returns=aligned_returns)
            ic_result = validator.information_coefficient(lookforward=lookforward)
            ic_mean = ic_result.get('mean', 0)
            ics.append(abs(ic_mean))  # 使用IC的绝对值
            
        # 归一化权重
        total_ic = sum(ics)
        if total_ic == 0:
            weight = 1.0 / len(factor_names)
            return {name: weight for name in factor_names}
            
        weights = {name: ic / total_ic for name, ic in zip(factor_names, ics)}
        return weights
    
    def _ir_weighted_weights(self,
                            factor_names: List[str],
                            aligned_factors: List[pd.DataFrame],
                            aligned_returns: Optional[pd.DataFrame],
                            lookforward: int) -> Dict[str, float]:
        """
        IR加权权重
        
        Args:
            factor_names: 因子名称列表
            aligned_factors: 对齐后的因子数据
            aligned_returns: 对齐后的收益率数据
            lookforward: 未来期数
            
        Returns:
            Dict[str, float]: IR加权权重
        """
        if aligned_returns is None:
            logger.warning("没有收益率数据，使用等权组合")
            weight = 1.0 / len(factor_names)
            return {name: weight for name in factor_names}
            
        # 计算每个因子的IR
        irs = []
        for i, factor in enumerate(aligned_factors):
            validator = FactorValidator(factor_values=factor,
                                       future_returns=aligned_returns)
            ir_value = validator.information_ratio(lookforward=lookforward)
            irs.append(abs(ir_value))  # 使用IR的绝对值
            
        # 归一化权重
        total_ir = sum(irs)
        if total_ir == 0:
            weight = 1.0 / len(factor_names)
            return {name: weight for name in factor_names}
            
        weights = {name: ir / total_ir for name, ir in zip(factor_names, irs)}
        return weights
    
    def _max_diversification_weights(self,
                                   factor_names: List[str],
                                   aligned_factors: List[pd.DataFrame]) -> Dict[str, float]:
        """
        最大分散化权重
        
        Args:
            factor_names: 因子名称列表
            aligned_factors: 对齐后的因子数据
            
        Returns:
            Dict[str, float]: 最大分散化权重
        """
        # 计算因子相关性矩阵
        n_factors = len(factor_names)
        correlations = np.eye(n_factors)
        
        for i in range(n_factors):
            for j in range(i + 1, n_factors):
                # 计算因子i和j的相关性
                factor_i = aligned_factors[i].values.flatten()
                factor_j = aligned_factors[j].values.flatten()
                
                # 去除NaN值
                valid_mask = ~np.isnan(factor_i) & ~np.isnan(factor_j)
                if valid_mask.sum() < 10:
                    corr = 0
                else:
                    corr = np.corrcoef(factor_i[valid_mask], factor_j[valid_mask])[0, 1]
                    if np.isnan(corr):
                        corr = 0
                        
                correlations[i, j] = corr
                correlations[j, i] = corr
                
        # 最大分散化组合：权重与1/波动率成比例
        # 这里简化处理，使用等权
        weight = 1.0 / n_factors
        return {name: weight for name in factor_names}
    
    def _min_variance_weights(self,
                            factor_names: List[str],
                            aligned_factors: List[pd.DataFrame]) -> Dict[str, float]:
        """
        最小方差权重
        
        Args:
            factor_names: 因子名称列表
            aligned_factors: 对齐后的因子数据
            
        Returns:
            Dict[str, float]: 最小方差权重
        """
        n_factors = len(factor_names)
        
        # 计算协方差矩阵
        cov_matrix = np.eye(n_factors)
        
        for i in range(n_factors):
            for j in range(i, n_factors):
                factor_i = aligned_factors[i].values.flatten()
                factor_j = aligned_factors[j].values.flatten()
                
                # 去除NaN值
                valid_mask = ~np.isnan(factor_i) & ~np.isnan(factor_j)
                if valid_mask.sum() < 10:
                    cov = 0 if i != j else 1
                else:
                    cov = np.cov(factor_i[valid_mask], factor_j[valid_mask])[0, 1]
                    if np.isnan(cov):
                        cov = 0 if i != j else 1
                        
                cov_matrix[i, j] = cov
                cov_matrix[j, i] = cov
                
        # 最小方差组合：w = Σ⁻¹ * 1 / (1ᵀ * Σ⁻¹ * 1)
        try:
            cov_inv = np.linalg.inv(cov_matrix)
            ones = np.ones(n_factors)
            weights_raw = cov_inv @ ones / (ones.T @ cov_inv @ ones)
            
            # 确保权重和为1且非负
            weights_raw = np.maximum(weights_raw, 0)
            weights_raw = weights_raw / weights_raw.sum()
            
        except np.linalg.LinAlgError:
            # 矩阵不可逆，使用等权
            logger.warning("协方差矩阵不可逆，使用等权组合")
            weights_raw = np.ones(n_factors) / n_factors
            
        weights = {name: float(w) for name, w in zip(factor_names, weights_raw)}
        return weights
    
    def _max_sharpe_weights(self,
                           factor_names: List[str],
                           aligned_factors: List[pd.DataFrame],
                           aligned_returns: Optional[pd.DataFrame],
                           lookforward: int) -> Dict[str, float]:
        """
        最大夏普比率权重
        
        Args:
            factor_names: 因子名称列表
            aligned_factors: 对齐后的因子数据
            aligned_returns: 对齐后的收益率数据
            lookforward: 未来期数
            
        Returns:
            Dict[str, float]: 最大夏普比率权重
        """
        if aligned_returns is None:
            logger.warning("没有收益率数据，使用最小方差组合")
            return self._min_variance_weights(factor_names, aligned_factors)
            
        n_factors = len(factor_names)
        
        # 计算预期收益（使用IC作为代理）
        expected_returns = np.zeros(n_factors)
        for i, factor in enumerate(aligned_factors):
            validator = FactorValidator(factor_values=factor,
                                       future_returns=aligned_returns)
            ic_result = validator.information_coefficient(lookforward=lookforward)
            expected_returns[i] = abs(ic_result.get('mean', 0))
            
        # 计算协方差矩阵
        cov_matrix = np.eye(n_factors)
        for i in range(n_factors):
            for j in range(i, n_factors):
                factor_i = aligned_factors[i].values.flatten()
                factor_j = aligned_factors[j].values.flatten()
                
                valid_mask = ~np.isnan(factor_i) & ~np.isnan(factor_j)
                if valid_mask.sum() < 10:
                    cov = 0 if i != j else 1
                else:
                    cov = np.cov(factor_i[valid_mask], factor_j[valid_mask])[0, 1]
                    if np.isnan(cov):
                        cov = 0 if i != j else 1
                        
                cov_matrix[i, j] = cov
                cov_matrix[j, i] = cov
                
        # 最大夏普比率组合：w = Σ⁻¹ * μ / (1ᵀ * Σ⁻¹ * μ)
        try:
            cov_inv = np.linalg.inv(cov_matrix)
            weights_raw = cov_inv @ expected_returns / (np.ones(n_factors).T @ cov_inv @ expected_returns)
            
            # 确保权重和为1且非负
            weights_raw = np.maximum(weights_raw, 0)
            weights_raw = weights_raw / weights_raw.sum()
            
        except np.linalg.LinAlgError:
            # 矩阵不可逆，使用IC加权
            logger.warning("协方差矩阵不可逆，使用IC加权组合")
            return self._ic_weighted_weights(factor_names, aligned_factors,
                                           aligned_returns, lookforward)
            
        weights = {name: float(w) for name, w in zip(factor_names, weights_raw)}
        return weights
    
    def _risk_parity_weights(self,
                           factor_names: List[str],
                           aligned_factors: List[pd.DataFrame]) -> Dict[str, float]:
        """
        风险平价权重
        
        Args:
            factor_names: 因子名称列表
            aligned_factors: 对齐后的因子数据
            
        Returns:
            Dict[str, float]: 风险平价权重
        """
        n_factors = len(factor_names)
        
        # 计算波动率
        volatilities = np.zeros(n_factors)
        for i, factor in enumerate(aligned_factors):
            factor_flat = factor.values.flatten()
            valid_mask = ~np.isnan(factor_flat)
            if valid_mask.sum() > 10:
                volatilities[i] = np.std(factor_flat[valid_mask])
            else:
                volatilities[i] = 1.0
                
        # 风险平价：权重与1/波动率成比例
        inv_vol = 1.0 / volatilities
        weights_raw = inv_vol / inv_vol.sum()
        
        weights = {name: float(w) for name, w in zip(factor_names, weights_raw)}
        return weights
    
    def _pca_weights(self,
                    factor_names: List[str],
                    aligned_factors: List[pd.DataFrame]) -> Dict[str, float]:
        """
        PCA权重（使用第一主成分）
        
        Args:
            factor_names: 因子名称列表
            aligned_factors: 对齐后的因子数据
            
        Returns:
            Dict[str, float]: PCA权重
        """
        if not HAS_SKLEARN:
            logger.warning("scikit-learn未安装，使用等权组合")
            weight = 1.0 / len(factor_names)
            return {name: weight for name in factor_names}
        
        # 将因子数据堆叠为二维数组
        factor_array = np.stack([f.values for f in aligned_factors])
        n_factors, n_dates, n_symbols = factor_array.shape
        factor_2d = factor_array.reshape(n_factors, -1)
        
        # 去除包含NaN的列
        valid_cols = ~np.any(np.isnan(factor_2d), axis=0)
        if valid_cols.sum() < n_factors * 10:
            logger.warning("有效数据点不足，使用等权组合")
            weight = 1.0 / len(factor_names)
            return {name: weight for name in factor_names}
            
        factor_valid = factor_2d[:, valid_cols]
        
        # 标准化
        scaler = StandardScaler()
        factor_scaled = scaler.fit_transform(factor_valid.T).T
        
        # PCA分解
        pca = PCA(n_components=1)
        pca.fit(factor_scaled.T)
        
        # 第一主成分的载荷作为权重
        weights_raw = np.abs(pca.components_[0])  # 使用绝对值
        weights_raw = weights_raw / weights_raw.sum()
        
        weights = {name: float(w) for name, w in zip(factor_names, weights_raw)}
        return weights
    
    def standardize(self,
                    factor: pd.DataFrame,
                    method: StandardizationMethod = StandardizationMethod.ZSCORE,
                    clip: float = 3.0) -> pd.DataFrame:
        """
        因子横截面标准化
        
        对每个时间截面（每行）独立进行标准化，去除因子量纲差异。
        这是多因子组合前的必要步骤，否则量纲不同的因子直接加权
        会被量纲大的因子主导。
        
        Args:
            factor: 因子值DataFrame，index为日期，columns为标的代码
            method: 标准化方法
            clip: Z-score截尾阈值（仅ZSCORE方法有效），默认3.0即±3σ
            
        Returns:
            pd.DataFrame: 标准化后的因子值
        """
        if method == StandardizationMethod.NONE:
            return factor.copy()
            
        result = factor.copy()
        
        if method == StandardizationMethod.ZSCORE:
            # 横截面Z-score: (x - mean) / std，每行独立计算
            row_mean = result.mean(axis=1, skipna=True)
            row_std = result.std(axis=1, ddof=1, skipna=True)
            
            # 避免除以0
            row_std = row_std.replace(0, np.nan)
            
            result = result.sub(row_mean, axis=0).div(row_std, axis=0)
            
            # 截尾去极值
            if clip is not None and clip > 0:
                result = result.clip(-clip, clip)
                
            # NaN填充为0（中庸得分）
            result = result.fillna(0)
            
        elif method == StandardizationMethod.RANK:
            # 横截面排名百分位: rank / N，映射到[0, 1]
            result = result.rank(axis=1, pct=True, na_option='keep')
            # 映射到[-1, 1]区间，均值接近0
            result = result * 2 - 1
            result = result.fillna(0)
            
        elif method == StandardizationMethod.MINMAX:
            # 横截面Min-Max: (x - min) / (max - min)，映射到[0, 1]
            row_min = result.min(axis=1, skipna=True)
            row_max = result.max(axis=1, skipna=True)
            row_range = row_max - row_min
            
            # 避免除以0
            row_range = row_range.replace(0, np.nan)
            
            result = result.sub(row_min, axis=0).div(row_range, axis=0)
            result = result.fillna(0.5)  # NaN填充为中值
            
        else:
            raise ValueError(f"不支持的标准化方法: {method}")
            
        return result
    
    def combine(self,
               factor_names: List[str],
               method: CombinationMethod = CombinationMethod.EQUAL_WEIGHT,
               orthogonalization: OrthogonalizationMethod = OrthogonalizationMethod.NONE,
               standardization: StandardizationMethod = StandardizationMethod.ZSCORE,
               lookforward: int = 1,
               clip: float = 3.0,
               **kwargs) -> CombinationResult:
        """
        组合因子
        
        Args:
            factor_names: 因子名称列表
            method: 组合方法
            orthogonalization: 正交化方法
            standardization: 标准化方法（默认Z-score，组合前去除量纲差异）
            lookforward: 未来期数
            clip: Z-score截尾阈值（仅ZSCORE方法有效），默认3.0
            **kwargs: 额外参数
            
        Returns:
            CombinationResult: 组合结果
        """
        if not factor_names:
            raise ValueError("需要至少一个因子进行组合")
            
        # 检查缓存
        cache_key = f"{'_'.join(sorted(factor_names))}_{method.value}_{orthogonalization.value}_{standardization.value}"
        if cache_key in self._combination_results:
            logger.info(f"使用缓存的组合结果: {cache_key}")
            return self._combination_results[cache_key]
            
        # 正交化处理
        if orthogonalization != OrthogonalizationMethod.NONE:
            orthogonalized = self.orthogonalize(factor_names, method=orthogonalization)
            # 使用正交化后的因子进行组合
            aligned_factors, aligned_returns = self._align_data(factor_names)
            # 替换为正交化后的因子
            aligned_factors = [orthogonalized[name] for name in factor_names]
        else:
            aligned_factors, aligned_returns = self._align_data(factor_names)
            
        # 横截面标准化（组合前去除量纲差异）
        aligned_factors = [self.standardize(f, method=standardization, clip=clip)
                          for f in aligned_factors]
            
        # 计算权重
        weights = self.calculate_weights(factor_names, method, lookforward, **kwargs)
        
        # 组合因子：加权平均
        combined = None
        for i, (name, factor) in enumerate(zip(factor_names, aligned_factors)):
            weight = weights.get(name, 0)
            if weight == 0:
                continue
                
            weighted_factor = factor * weight
            
            if combined is None:
                combined = weighted_factor
            else:
                combined = combined + weighted_factor
                
        if combined is None:
            raise ValueError("组合因子计算失败")
            
        # 计算组合指标
        metrics = self._calculate_combination_metrics(combined, aligned_returns, 
                                                    lookforward, weights)
        
        # 创建结果对象
        result = CombinationResult(
            combined_factor=combined,
            weights=weights,
            method=method,
            orthogonalization=orthogonalization,
            standardization=standardization,
            metrics=metrics
        )
        
        # 缓存结果
        self._combination_results[cache_key] = result
        
        return result
    
    def _calculate_combination_metrics(self,
                                     combined_factor: pd.DataFrame,
                                     future_returns: Optional[pd.DataFrame],
                                     lookforward: int,
                                     weights: Dict[str, float]) -> Dict[str, Any]:
        """
        计算组合指标
        
        Args:
            combined_factor: 组合因子
            future_returns: 未来收益率
            lookforward: 未来期数
            weights: 因子权重
            
        Returns:
            Dict[str, Any]: 组合指标
        """
        metrics = {
            'weights': weights,
            'effective_number': self._calculate_effective_number(weights),
            'weight_concentration': self._calculate_weight_concentration(weights)
        }
        
        if future_returns is not None:
            # 计算组合因子的IC/IR
            validator = FactorValidator(factor_values=combined_factor,
                                       future_returns=future_returns)
            
            ic_result = validator.information_coefficient(lookforward=lookforward)
            metrics['ic'] = ic_result.get('mean', np.nan)
            metrics['ir'] = ic_result.get('ir', np.nan)
            
            # 计算分组收益
            group_result = validator.group_returns(lookforward=lookforward)
            metrics['group_returns'] = group_result.get('returns', {})
            metrics['long_short_spread'] = group_result.get('spread', np.nan)
            metrics['monotonicity'] = group_result.get('monotonicity', np.nan)
            
        return metrics
    
    def _calculate_effective_number(self, weights: Dict[str, float]) -> float:
        """
        计算有效因子数量
        
        Args:
            weights: 因子权重
            
        Returns:
            float: 有效因子数量
        """
        weight_array = np.array(list(weights.values()))
        weight_array = weight_array[weight_array > 0]  # 只考虑正权重
        
        if len(weight_array) == 0:
            return 0
            
        # 有效数量 = 1 / Σ(w_i²)
        return 1.0 / np.sum(weight_array ** 2)
    
    def _calculate_weight_concentration(self, weights: Dict[str, float]) -> float:
        """
        计算权重集中度（赫芬达尔指数）
        
        Args:
            weights: 因子权重
            
        Returns:
            float: 赫芬达尔指数（0-1，越大表示越集中）
        """
        weight_array = np.array(list(weights.values()))
        return float(np.sum(weight_array ** 2))
    
    def clear_cache(self):
        """清空缓存"""
        self._orthogonalized_factors.clear()
        self._combination_results.clear()
        logger.info("组合器缓存已清空")