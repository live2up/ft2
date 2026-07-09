"""
因子检验器模块（V5 自包含版本）

[重构] 2026-05-20 从 factor/validator.py 移植，使 V5 模块完全独立不依赖 V1。
保留核心验证方法：IC/IR/换手率/衰减率/分组收益/命中率/稳定性。

设计思路：
---------
1. 统计检验：IC、IR、换手率、衰减率等
2. 分组检验：十分组收益、多空组合收益
3. 稳定性检验：时间序列稳定性、截面稳定性

使用方式：
---------
>>> from factor.v5.validator import FactorValidator
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

        # [修复] 2026-05-20 多期IC对齐：lookforward>1 时累积N期前瞻收益
        # 旧实现 factor[:-N] vs future_returns[N:]，只偏移日期范围，未真正使用多期收益
        # 导致 decay_rate() 中所有 lookforward 实际都算的是1日IC，衰减曲线无意义
        # 新实现：用1期前瞻收益反向滚动计算N期累积收益，再直接对齐
        if lookforward == 1:
            factor_aligned = self.factor_values.iloc[:-1]
            returns_aligned = self.future_returns.iloc[1:]
        else:
            # fw_cum[t] = (1+r[t→t+1])*(1+r[t+1→t+2])*...*(1+r[t+N-1→t+N]) - 1
            fw_cum = (1.0 + self.future_returns).iloc[::-1].rolling(
                window=lookforward, min_periods=lookforward
            ).apply(np.prod, raw=True).iloc[::-1] - 1.0
            tail = lookforward - 1
            factor_aligned = self.factor_values.iloc[:-tail] if tail > 0 else self.factor_values.iloc[:-1]
            returns_aligned = fw_cum.iloc[:-tail] if tail > 0 else fw_cum.iloc[1:]

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

    # [新增] 2026-05-27 Bootstrap Sharpe 置信区间
    @staticmethod
    def bootstrap_sharpe(nav_series: pd.Series,
                         n_bootstrap: int = 1000,
                         ci: float = 0.90,
                         freq: str = 'monthly') -> Dict[str, float]:
        """Bootstrap 重采样计算 Sharpe 置信区间

        对净值序列的收益率做有放回重采样，计算每次的 Sharpe，
        输出指定置信水平的区间。

        原理：不假设收益率服从正态分布，通过重采样经验分布
        直接估计 Sharpe 的不确定性。

        Args:
            nav_series: 日频净值序列
            n_bootstrap: 重采样次数，默认 1000
            ci: 置信水平，默认 0.90（输出 5%~95% 分位）
            freq: 重采样频率，'monthly'（月频，推荐）或 'daily'（日频）
                 月频更稳健（减少序列自相关影响），日频更精细

        Returns:
            Dict[str, float]: {
                'sharpe': 原始 Sharpe,
                'mean': Bootstrap 均值,
                'std': Bootstrap 标准差,
                'ci_lower': 置信下限,
                'ci_upper': 置信上限,
                'n_bootstrap': 实际重采样次数,
            }
        """
        daily_ret = nav_series.pct_change().dropna()
        if len(daily_ret) < 20:
            return {'sharpe': 0, 'mean': 0, 'std': 0,
                    'ci_lower': 0, 'ci_upper': 0, 'n_bootstrap': 0}

        # 原始 Sharpe
        orig_sharpe = float(
            daily_ret.mean() / daily_ret.std() * np.sqrt(252)
            if daily_ret.std() > 1e-10 else 0
        )

        # 按频率重采样
        if freq == 'monthly':
            # 月频：按月份分组，每组内累加收益
            monthly_ret = daily_ret.resample('ME').apply(
                lambda x: np.prod(1 + x.values) - 1
            ).dropna()
            if len(monthly_ret) < 12:
                return {'sharpe': orig_sharpe, 'mean': orig_sharpe, 'std': 0,
                        'ci_lower': orig_sharpe, 'ci_upper': orig_sharpe,
                        'n_bootstrap': 0}
            sample_data = monthly_ret.values
            annual_factor = 12.0
        else:
            sample_data = daily_ret.values
            annual_factor = 252.0

        # Bootstrap 重采样
        n = len(sample_data)
        boot_sharpes = []
        rng = np.random.default_rng()

        for _ in range(n_bootstrap):
            indices = rng.integers(0, n, size=n)
            boot_ret = sample_data[indices]
            boot_mean = np.mean(boot_ret)
            boot_std = np.std(boot_ret, ddof=1)
            if boot_std > 1e-10:
                boot_sharpe = boot_mean / boot_std * np.sqrt(annual_factor)
                boot_sharpes.append(boot_sharpe)

        if len(boot_sharpes) < 100:
            return {'sharpe': orig_sharpe, 'mean': orig_sharpe, 'std': 0,
                    'ci_lower': orig_sharpe, 'ci_upper': orig_sharpe,
                    'n_bootstrap': len(boot_sharpes)}

        boot_sharpes = np.array(boot_sharpes)
        alpha = (1.0 - ci) / 2.0

        return {
            'sharpe': orig_sharpe,
            'mean': float(np.mean(boot_sharpes)),
            'std': float(np.std(boot_sharpes, ddof=1)),
            'ci_lower': float(np.percentile(boot_sharpes, alpha * 100)),
            'ci_upper': float(np.percentile(boot_sharpes, (1 - alpha) * 100)),
            'n_bootstrap': len(boot_sharpes),
        }

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

        # [修复] 2026-05-20 同 IC 对齐逻辑：lookforward>1 时累积N期前瞻收益
        if lookforward == 1:
            factor_aligned = self.factor_values.iloc[:-1]
            returns_aligned = self.future_returns.iloc[1:]
        else:
            fw_cum = (1.0 + self.future_returns).iloc[::-1].rolling(
                window=lookforward, min_periods=lookforward
            ).apply(np.prod, raw=True).iloc[::-1] - 1.0
            tail = lookforward - 1
            factor_aligned = self.factor_values.iloc[:-tail] if tail > 0 else self.factor_values.iloc[:-1]
            returns_aligned = fw_cum.iloc[:-tail] if tail > 0 else fw_cum.iloc[1:]

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

    # [新增] 2026-05-27 分年度 IC 统计
    @validation_metric(name='分年度IC', desc='每年 IC 均值、ICIR、正占比', order=12)
    def yearly_ic(self, lookforward: Optional[int] = None,
                  method: str = 'spearman') -> Dict[int, Dict[str, float]]:
        """计算每年 IC 统计

        对每年内的日频 IC 序列独立计算均值、ICIR、正占比。

        Args:
            lookforward: 未来期数
            method: 相关系数计算方法

        Returns:
            Dict[int, Dict[str, float]]: {年份: {mean, ir, positive_ratio, n_days}}
        """
        if not self._validate_data():
            return {}

        lookforward = lookforward or self.ic_lookforward

        # 对齐因子和收益
        if lookforward == 1:
            factor_aligned = self.factor_values.iloc[:-1]
            returns_aligned = self.future_returns.iloc[1:]
        else:
            fw_cum = (1.0 + self.future_returns).iloc[::-1].rolling(
                window=lookforward, min_periods=lookforward
            ).apply(np.prod, raw=True).iloc[::-1] - 1.0
            tail = lookforward - 1
            factor_aligned = self.factor_values.iloc[:-tail] if tail > 0 else self.factor_values.iloc[:-1]
            returns_aligned = fw_cum.iloc[:-tail] if tail > 0 else fw_cum.iloc[1:]

        common_dates = factor_aligned.index.intersection(returns_aligned.index)
        if len(common_dates) == 0:
            return {}

        factor_aligned = factor_aligned.loc[common_dates]
        returns_aligned = returns_aligned.loc[common_dates]

        # 按年份分组计算 IC
        yearly_ics: Dict[int, list] = {}
        for date_idx in common_dates:
            factor_slice = factor_aligned.loc[date_idx]
            returns_slice = returns_aligned.loc[date_idx]
            valid_mask = factor_slice.notna() & returns_slice.notna()
            if valid_mask.sum() < 10:
                continue
            factor_valid = factor_slice[valid_mask]
            returns_valid = returns_slice[valid_mask]
            if method == 'spearman':
                ic, _ = stats.spearmanr(factor_valid, returns_valid)
            elif method == 'pearson':
                ic, _ = stats.pearsonr(factor_valid, returns_valid)
            else:
                raise ValueError(f"不支持的相关系数计算方法: {method}")
            if not np.isnan(ic):
                year = date_idx.year
                if year not in yearly_ics:
                    yearly_ics[year] = []
                yearly_ics[year].append(ic)

        result = {}
        for year in sorted(yearly_ics.keys()):
            ics = np.array(yearly_ics[year])
            if len(ics) < 5:
                continue
            mean_ic = float(np.mean(ics))
            std_ic = float(np.std(ics))
            result[int(year)] = {
                'mean': mean_ic,
                'ir': float(mean_ic / std_ic) if std_ic > 1e-10 else 0.0,
                'positive_ratio': float(np.mean(ics > 0)),
                'n_days': len(ics),
            }

        return result
