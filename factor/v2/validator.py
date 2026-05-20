"""
因子检验器模块（V2 自包含版本）

[重构] 2026-05-20 从 factor/validator.py 移植，使 V2 模块完全独立不依赖 V1。
保留核心验证方法：IC/IR/换手率/衰减率/分组收益/命中率/稳定性。

设计思路：
---------
1. 统计检验：IC、IR、换手率、衰减率等
2. 分组检验：十分组收益、多空组合收益
3. 稳定性检验：时间序列稳定性、截面稳定性

使用方式：
---------
>>> from factor.v2.validator import FactorValidator
>>> validator = FactorValidator(factor_values, future_returns)
>>> ic = validator.information_coefficient()
>>> ir = validator.information_ratio()
>>> decay = validator.decay_rate(max_lookforward=20)
"""

import warnings
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import pandas as pd
import numpy as np
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ValidationMetric(Enum):
    """验证指标枚举"""
    IC = "ic"
    IR = "ir"
    TURNOVER = "turnover"
    DECAY = "decay"
    HIT_RATE = "hit_rate"
    GROUP_RETURN = "group_return"
    LONG_SHORT = "long_short"
    MONOTONICITY = "monotonicity"
    STABILITY = "stability"


@dataclass
class ValidationResult:
    """验证结果数据类"""
    metric: ValidationMetric
    value: Any
    confidence: float = 0.95
    p_value: Optional[float] = None
    description: str = ""
    timestamp: datetime = field(default_factory=datetime.now)


def validation_metric(name: str = None,
                     desc: str = '',
                     metric_type: str = 'float',
                     order: int = 99):
    """
    验证指标装饰器，用于标记验证方法

    Args:
        name: 指标中文名称，默认使用函数名
        desc: 指标描述
        metric_type: 数据类型，默认 'float'
        order: 排序号，用于报告中的顺序

    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)

        wrapper._is_validation_metric = True
        wrapper._metric_name = name or func.__name__
        wrapper._metric_desc = desc
        wrapper._metric_type = metric_type
        wrapper._metric_order = order

        return wrapper
    return decorator


class FactorValidator:
    """因子检验器（V2 自包含版本）"""

    def __init__(self,
                 factor_values: Optional[pd.DataFrame] = None,
                 future_returns: Optional[pd.DataFrame] = None,
                 group_count: int = 10):
        """
        初始化因子检验器

        Args:
            factor_values: 因子值DataFrame（index为日期，columns为标的）
            future_returns: 未来收益率DataFrame（与factor_values相同结构）
            group_count: 分组数量，默认10组
        """
        self.factor_values = factor_values
        self.future_returns = future_returns
        self.group_count = group_count

        self._validation_results = {}
        self._group_assignments = None

        self.ic_lookforward = 1
        self.min_valid_ratio = 0.7

    def set_data(self,
                factor_values: pd.DataFrame,
                future_returns: pd.DataFrame):
        """
        设置数据

        Args:
            factor_values: 因子值DataFrame
            future_returns: 未来收益率DataFrame
        """
        self.factor_values = factor_values
        self.future_returns = future_returns
        self.clear_cache()

    def clear_cache(self):
        """清空缓存"""
        self._validation_results.clear()
        self._group_assignments = None

    def _validate_data(self) -> bool:
        """
        验证数据有效性

        Returns:
            bool: 数据是否有效
        """
        if self.factor_values is None or self.future_returns is None:
            logger.error("因子值或未来收益率数据未设置")
            return False

        if self.factor_values.shape != self.future_returns.shape:
            logger.error("因子值和未来收益率数据形状不匹配")
            return False

        if self.factor_values.empty or self.future_returns.empty:
            logger.error("数据为空")
            return False

        valid_ratio = (self.factor_values.notna().sum().sum() /
                      (self.factor_values.shape[0] * self.factor_values.shape[1]))

        if valid_ratio < self.min_valid_ratio:
            logger.warning(f"数据完整度较低: {valid_ratio:.2%}")

        return True

    def _get_group_assignments(self) -> pd.DataFrame:
        """
        获取分组分配

        Returns:
            pd.DataFrame: 分组分配（与因子值相同结构，值为分组编号1-group_count）
        """
        if self._group_assignments is not None:
            return self._group_assignments

        if not self._validate_data():
            raise ValueError("数据无效")

        group_assignments = pd.DataFrame(
            np.nan,
            index=self.factor_values.index,
            columns=self.factor_values.columns
        )

        for date_idx in self.factor_values.index:
            factor_slice = self.factor_values.loc[date_idx]
            valid_mask = factor_slice.notna()
            valid_values = factor_slice[valid_mask]

            if len(valid_values) < self.group_count:
                continue

            ranks = valid_values.rank(method='first')
            groups = pd.qcut(ranks, q=self.group_count, labels=False) + 1
            group_assignments.loc[date_idx, valid_mask] = groups

        self._group_assignments = group_assignments
        return group_assignments

    @validation_metric(name='信息系数(IC)', desc='因子值与未来收益率的相关系数', order=10)
    def information_coefficient(self,
                               lookforward: Optional[int] = None,
                               method: str = 'spearman') -> Dict[str, Any]:
        """
        计算信息系数（IC）

        Args:
            lookforward: 未来期数，默认为self.ic_lookforward
            method: 相关系数计算方法，'spearman'（默认）或'pearson'

        Returns:
            Dict[str, Any]: IC统计结果
        """
        if not self._validate_data():
            return {'mean': np.nan, 'std': np.nan, 'ir': np.nan}

        lookforward = lookforward or self.ic_lookforward

        factor_aligned = self.factor_values.iloc[:-lookforward] if lookforward > 0 else self.factor_values
        returns_aligned = self.future_returns.iloc[lookforward:] if lookforward > 0 else self.future_returns

        common_dates = factor_aligned.index.intersection(returns_aligned.index)
        if len(common_dates) == 0:
            logger.error("因子值和未来收益率日期无法对齐")
            return {'mean': np.nan, 'std': np.nan, 'ir': np.nan}

        factor_aligned = factor_aligned.loc[common_dates]
        returns_aligned = returns_aligned.loc[common_dates]

        daily_ics = []
        valid_dates = []

        for date_idx in common_dates:
            factor_slice = factor_aligned.loc[date_idx]
            returns_slice = returns_aligned.loc[date_idx]

            valid_mask = factor_slice.notna() & returns_slice.notna()
            if valid_mask.sum() < 10:
                continue

            factor_valid = factor_slice[valid_mask]
            returns_valid = returns_slice[valid_mask]

            if method == 'spearman':
                ic, p_value = stats.spearmanr(factor_valid, returns_valid)
            elif method == 'pearson':
                ic, p_value = stats.pearsonr(factor_valid, returns_valid)
            else:
                raise ValueError(f"不支持的相关系数计算方法: {method}")

            if not np.isnan(ic):
                daily_ics.append(ic)
                valid_dates.append(date_idx)

        if not daily_ics:
            return {'mean': np.nan, 'std': np.nan, 'ir': np.nan}

        daily_ics = np.array(daily_ics)

        result = {
            'mean': float(np.mean(daily_ics)),
            'std': float(np.std(daily_ics)),
            'ir': float(np.mean(daily_ics) / np.std(daily_ics) if np.std(daily_ics) != 0 else np.nan),
            'positive_ratio': float(np.mean(daily_ics > 0)),
            't_stat': float(stats.ttest_1samp(daily_ics, 0).statistic if len(daily_ics) > 1 else np.nan),
            'p_value': float(stats.ttest_1samp(daily_ics, 0).pvalue if len(daily_ics) > 1 else np.nan),
            'daily_ics': daily_ics.tolist(),
            'dates': [d.strftime('%Y-%m-%d') for d in valid_dates]
        }

        return result

    @validation_metric(name='信息比率(IR)', desc='IC均值与标准差的比值', order=11)
    def information_ratio(self, lookforward: Optional[int] = None) -> float:
        """
        计算信息比率（IR）

        Args:
            lookforward: 未来期数

        Returns:
            float: 信息比率
        """
        ic_result = self.information_coefficient(lookforward)
        return ic_result.get('ir', np.nan)

    @validation_metric(name='换手率', desc='因子排名变化率', order=20)
    def turnover_rate(self, lookforward: int = 1) -> Dict[str, float]:
        """
        计算换手率

        Args:
            lookforward: 未来期数

        Returns:
            Dict[str, float]: 各分位换手率
        """
        if not self._validate_data():
            return {'mean': np.nan, 'top': np.nan, 'bottom': np.nan}

        group_assignments = self._get_group_assignments()

        turnover_rates = []
        top_turnovers = []
        bottom_turnovers = []

        dates = sorted(group_assignments.index)
        for i in range(lookforward, len(dates)):
            current_date = dates[i - lookforward]
            next_date = dates[i]

            current_groups = group_assignments.loc[current_date]
            next_groups = group_assignments.loc[next_date]

            valid_mask = current_groups.notna() & next_groups.notna()
            if valid_mask.sum() == 0:
                continue

            current_valid = current_groups[valid_mask]
            next_valid = next_groups[valid_mask]

            rank_change = (current_valid != next_valid).mean()
            turnover_rates.append(rank_change)

            top_mask = current_valid == self.group_count
            if top_mask.any():
                top_turnover = (current_valid[top_mask] != next_valid[top_mask]).mean()
                top_turnovers.append(top_turnover)

            bottom_mask = current_valid == 1
            if bottom_mask.any():
                bottom_turnover = (current_valid[bottom_mask] != next_valid[bottom_mask]).mean()
                bottom_turnovers.append(bottom_turnover)

        if not turnover_rates:
            return {'mean': np.nan, 'top': np.nan, 'bottom': np.nan}

        result = {
            'mean': float(np.mean(turnover_rates)),
            'top': float(np.mean(top_turnovers)) if top_turnovers else np.nan,
            'bottom': float(np.mean(bottom_turnovers)) if bottom_turnovers else np.nan,
            'std': float(np.std(turnover_rates))
        }

        return result

    @validation_metric(name='衰减率', desc='因子预测能力随时间衰减的速度', order=21)
    def decay_rate(self, max_lookforward: int = 20) -> Dict[str, Any]:
        """
        计算衰减率

        Args:
            max_lookforward: 最大未来期数

        Returns:
            Dict[str, Any]: 衰减率结果
        """
        if not self._validate_data():
            return {'half_life': np.nan, 'decay_rates': []}

        ic_means = []
        lookforwards = list(range(1, min(max_lookforward + 1, len(self.factor_values) // 2)))

        for lookforward in lookforwards:
            ic_result = self.information_coefficient(lookforward)
            ic_mean = ic_result.get('mean', np.nan)
            ic_means.append(ic_mean if not np.isnan(ic_mean) else np.nan)

        valid_ics = [(i+1, ic) for i, ic in enumerate(ic_means) if not np.isnan(ic)]
        if len(valid_ics) < 3:
            return {'half_life': np.nan, 'decay_rates': ic_means}

        lookforwards_valid, ics_valid = zip(*valid_ics)

        try:
            log_ics = np.log(np.abs(ics_valid))
            slope, intercept = np.polyfit(lookforwards_valid, log_ics, 1)
            decay_rate_val = -slope
            half_life = np.log(2) / decay_rate_val if decay_rate_val > 0 else np.inf
        except:
            decay_rate_val = np.nan
            half_life = np.nan

        result = {
            'half_life': float(half_life),
            'decay_rate': float(decay_rate_val),
            'ic_means': ic_means,
            'lookforwards': lookforwards
        }

        return result

    @validation_metric(name='命中率', desc='因子方向预测正确的比例', order=40)
    def hit_rate(self, lookforward: int = 1) -> float:
        """
        计算命中率

        Args:
            lookforward: 未来期数

        Returns:
            float: 命中率
        """
        if not self._validate_data():
            return np.nan

        factor_aligned = self.factor_values.iloc[:-lookforward] if lookforward > 0 else self.factor_values
        returns_aligned = self.future_returns.iloc[lookforward:] if lookforward > 0 else self.future_returns

        common_dates = factor_aligned.index.intersection(returns_aligned.index)
        if len(common_dates) == 0:
            return np.nan

        hit_counts = 0
        total_counts = 0

        for date_idx in common_dates:
            factor_slice = factor_aligned.loc[date_idx]
            returns_slice = returns_aligned.loc[date_idx]

            valid_mask = factor_slice.notna() & returns_slice.notna()
            if valid_mask.sum() < 10:
                continue

            factor_valid = factor_slice[valid_mask]
            returns_valid = returns_slice[valid_mask]

            factor_median = factor_valid.median()
            returns_median = returns_valid.median()

            factor_above = factor_valid > factor_median
            returns_above = returns_valid > returns_median

            hits = (factor_above == returns_above).sum()
            hit_counts += hits
            total_counts += len(factor_valid)

        if total_counts == 0:
            return np.nan

        return hit_counts / total_counts
