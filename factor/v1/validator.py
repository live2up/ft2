"""
因子检验器模块

设计思路：
---------
1. 统计检验：IC、IR、换手率、衰减率等
2. 分组检验：十分组收益、多空组合收益
3. 稳定性检验：时间序列稳定性、截面稳定性
4. 装饰器模式：类似analyzer.py的@metric装饰器
5. 报告生成：HTML/JSON格式的检验报告

使用方式：
---------
1. 创建FactorValidator实例
2. 提供因子值和未来收益率数据
3. 执行各种检验
4. 获取检验结果和报告
"""

import warnings
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from pathlib import Path
import json
import logging

logger = logging.getLogger(__name__)


class ValidationMetric(Enum):
    """验证指标枚举"""
    IC = "ic"                     # 信息系数
    IR = "ir"                     # 信息比率
    TURNOVER = "turnover"         # 换手率
    DECAY = "decay"               # 衰减率
    HIT_RATE = "hit_rate"         # 命中率
    GROUP_RETURN = "group_return" # 分组收益
    LONG_SHORT = "long_short"     # 多空收益
    MONOTONICITY = "monotonicity" # 单调性
    STABILITY = "stability"       # 稳定性


@dataclass
class ValidationResult:
    """验证结果数据类"""
    metric: ValidationMetric      # 指标类型
    value: Any                    # 指标值
    confidence: float = 0.95      # 置信水平
    p_value: Optional[float] = None  # p值
    description: str = ""         # 描述
    timestamp: datetime = field(default_factory=datetime.now)  # 时间戳


def validation_metric(name: str = None, 
                     desc: str = '', 
                     metric_type: str = 'float', 
                     order: int = 99):
    """
    验证指标装饰器，用于标记验证方法
    
    Args:
        name: 指标中文名称，默认使用函数名
        desc: 指标描述
        metric_type: 数据类型，默认 'float'，可选 'int', 'dict', 'list' 等
        order: 排序号，用于报告中的顺序
        
    Returns:
        装饰器函数
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(self, *args, **kwargs):
            return func(self, *args, **kwargs)
            
        # 添加验证指标元数据
        wrapper._is_validation_metric = True
        wrapper._metric_name = name or func.__name__
        wrapper._metric_desc = desc
        wrapper._metric_type = metric_type
        wrapper._metric_order = order
        
        return wrapper
    return decorator


class FactorValidator:
    """因子检验器"""
    
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
        
        self._validation_results = {}  # 缓存验证结果
        self._group_assignments = None  # 缓存分组结果
        
        # 配置
        self.ic_lookforward = 1  # IC计算的未来期数
        self.min_valid_ratio = 0.7  # 最小有效数据比例
        
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
            
        # 检查数据完整性
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
            
        # 按日期分组，对每个截面的因子值进行分组
        group_assignments = pd.DataFrame(
            np.nan, 
            index=self.factor_values.index, 
            columns=self.factor_values.columns
        )
        
        for date_idx in self.factor_values.index:
            factor_slice = self.factor_values.loc[date_idx]
            
            # 去除NaN值
            valid_mask = factor_slice.notna()
            valid_values = factor_slice[valid_mask]
            
            if len(valid_values) < self.group_count:
                # 有效数据不足，跳过该日期
                continue
                
            # 按因子值排序并分组
            ranks = valid_values.rank(method='first')
            groups = pd.qcut(ranks, q=self.group_count, labels=False) + 1  # 1-based分组
            
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
        
        # 对齐因子值和未来收益率
        factor_aligned = self.factor_values.iloc[:-lookforward] if lookforward > 0 else self.factor_values
        returns_aligned = self.future_returns.iloc[lookforward:] if lookforward > 0 else self.future_returns
        
        # 确保索引对齐
        common_dates = factor_aligned.index.intersection(returns_aligned.index)
        if len(common_dates) == 0:
            logger.error("因子值和未来收益率日期无法对齐")
            return {'mean': np.nan, 'std': np.nan, 'ir': np.nan}
            
        factor_aligned = factor_aligned.loc[common_dates]
        returns_aligned = returns_aligned.loc[common_dates]
        
        # 计算每日IC
        daily_ics = []
        valid_dates = []
        
        for date_idx in common_dates:
            factor_slice = factor_aligned.loc[date_idx]
            returns_slice = returns_aligned.loc[date_idx]
            
            # 去除NaN值
            valid_mask = factor_slice.notna() & returns_slice.notna()
            if valid_mask.sum() < 10:  # 至少需要10个有效数据点
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
        
        # 计算每日换手率
        turnover_rates = []
        top_turnovers = []  # 前10%组换手率
        bottom_turnovers = []  # 后10%组换手率
        
        dates = sorted(group_assignments.index)
        for i in range(lookforward, len(dates)):
            current_date = dates[i - lookforward]
            next_date = dates[i]
            
            current_groups = group_assignments.loc[current_date]
            next_groups = group_assignments.loc[next_date]
            
            # 去除NaN值
            valid_mask = current_groups.notna() & next_groups.notna()
            if valid_mask.sum() == 0:
                continue
                
            current_valid = current_groups[valid_mask]
            next_valid = next_groups[valid_mask]
            
            # 总体换手率（排名变化的比例）
            rank_change = (current_valid != next_valid).mean()
            turnover_rates.append(rank_change)
            
            # 前10%组换手率
            top_mask = current_valid == self.group_count  # 第10组（因子值最高）
            if top_mask.any():
                top_turnover = (current_valid[top_mask] != next_valid[top_mask]).mean()
                top_turnovers.append(top_turnover)
                
            # 后10%组换手率
            bottom_mask = current_valid == 1  # 第1组（因子值最低）
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
            
        # 计算不同未来期数的IC
        ic_means = []
        lookforwards = list(range(1, min(max_lookforward + 1, len(self.factor_values) // 2)))
        
        for lookforward in lookforwards:
            ic_result = self.information_coefficient(lookforward)
            ic_mean = ic_result.get('mean', np.nan)
            if not np.isnan(ic_mean):
                ic_means.append(ic_mean)
            else:
                ic_means.append(np.nan)
                
        # 拟合衰减曲线（指数衰减）
        valid_ics = [(i+1, ic) for i, ic in enumerate(ic_means) if not np.isnan(ic)]
        if len(valid_ics) < 3:
            return {'half_life': np.nan, 'decay_rates': ic_means}
            
        lookforwards_valid, ics_valid = zip(*valid_ics)
        
        try:
            # 指数衰减拟合：IC(t) = IC0 * exp(-λ*t)
            log_ics = np.log(np.abs(ics_valid))
            slope, intercept = np.polyfit(lookforwards_valid, log_ics, 1)
            decay_rate = -slope  # λ
            half_life = np.log(2) / decay_rate if decay_rate > 0 else np.inf
        except:
            decay_rate = np.nan
            half_life = np.nan
            
        result = {
            'half_life': float(half_life),
            'decay_rate': float(decay_rate),
            'ic_means': ic_means,
            'lookforwards': lookforwards
        }
        
        return result
    
    @validation_metric(name='分组收益', desc='按因子分组的未来收益率', order=30)
    def group_returns(self, 
                     lookforward: int = 1,
                     value_weighted: bool = False,
                     weights: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
        """
        计算分组收益
        
        Args:
            lookforward: 未来期数
            value_weighted: 是否市值加权
            weights: 权重DataFrame（如果value_weighted为True且weights不为None）
            
        Returns:
            Dict[str, Any]: 分组收益结果
        """
        if not self._validate_data():
            return {'returns': {}, 'spread': np.nan}
            
        group_assignments = self._get_group_assignments()
        
        # 对齐分组和未来收益率
        groups_aligned = group_assignments.iloc[:-lookforward] if lookforward > 0 else group_assignments
        returns_aligned = self.future_returns.iloc[lookforward:] if lookforward > 0 else self.future_returns
        
        # 确保索引对齐
        common_dates = groups_aligned.index.intersection(returns_aligned.index)
        if len(common_dates) == 0:
            return {'returns': {}, 'spread': np.nan}
            
        groups_aligned = groups_aligned.loc[common_dates]
        returns_aligned = returns_aligned.loc[common_dates]
        
        # 计算每日分组收益
        group_daily_returns = {i: [] for i in range(1, self.group_count + 1)}
        
        for date_idx in common_dates:
            groups_slice = groups_aligned.loc[date_idx]
            returns_slice = returns_aligned.loc[date_idx]
            
            for group_num in range(1, self.group_count + 1):
                # 获取该组的标的
                group_mask = groups_slice == group_num
                if not group_mask.any():
                    group_daily_returns[group_num].append(np.nan)
                    continue
                    
                group_returns = returns_slice[group_mask]
                
                if value_weighted:
                    if weights is not None:
                        # 市值加权收益
                        weight_slice = weights.loc[date_idx]
                        group_weights = weight_slice[group_mask]
                        group_weights_normalized = group_weights / group_weights.sum()
                        group_return = (group_returns * group_weights_normalized).sum()
                    else:
                        # 等权收益
                        group_return = group_returns.mean()
                else:
                    # 等权收益
                    group_return = group_returns.mean()
                    
                group_daily_returns[group_num].append(group_return)
                
        # 计算平均分组收益
        group_avg_returns = {}
        for group_num in range(1, self.group_count + 1):
            returns = group_daily_returns[group_num]
            valid_returns = [r for r in returns if not np.isnan(r)]
            if valid_returns:
                group_avg_returns[group_num] = float(np.mean(valid_returns))
            else:
                group_avg_returns[group_num] = np.nan
                
        # 计算多空收益（第10组 - 第1组）
        if 10 in group_avg_returns and 1 in group_avg_returns:
            long_short_spread = group_avg_returns[10] - group_avg_returns[1]
        else:
            long_short_spread = np.nan
            
        result = {
            'returns': group_avg_returns,
            'spread': float(long_short_spread),
            'monotonicity': self._calculate_monotonicity(group_avg_returns),
            'daily_returns': group_daily_returns
        }
        
        return result
    
    def _calculate_monotonicity(self, group_returns: Dict[int, float]) -> float:
        """
        计算单调性
        
        Args:
            group_returns: 分组收益字典
            
        Returns:
            float: 单调性得分（0-1）
        """
        # 提取有效收益
        valid_returns = [(group, ret) for group, ret in group_returns.items() 
                        if not np.isnan(ret)]
        if len(valid_returns) < 3:
            return np.nan
            
        groups, returns = zip(*sorted(valid_returns))
        returns = np.array(returns)
        
        # 计算Spearman秩相关系数
        spearman_corr, _ = stats.spearmanr(groups, returns)
        
        # 单调性得分为相关系数的绝对值
        return float(abs(spearman_corr))
    
    @validation_metric(name='多空收益', desc='最高组与最低组的收益差', order=31)
    def long_short_return(self, lookforward: int = 1) -> float:
        """
        计算多空收益
        
        Args:
            lookforward: 未来期数
            
        Returns:
            float: 多空收益
        """
        group_result = self.group_returns(lookforward)
        return group_result.get('spread', np.nan)
    
    @validation_metric(name='单调性', desc='分组收益的单调递增程度', order=32)
    def monotonicity(self, lookforward: int = 1) -> float:
        """
        计算单调性
        
        Args:
            lookforward: 未来期数
            
        Returns:
            float: 单调性得分
        """
        group_result = self.group_returns(lookforward)
        return group_result.get('monotonicity', np.nan)
    
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
            
        # 对齐因子值和未来收益率
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
            
            # 去除NaN值
            valid_mask = factor_slice.notna() & returns_slice.notna()
            if valid_mask.sum() < 10:
                continue
                
            factor_valid = factor_slice[valid_mask]
            returns_valid = returns_slice[valid_mask]
            
            # 计算中位数
            factor_median = factor_valid.median()
            returns_median = returns_valid.median()
            
            # 判断方向
            factor_above = factor_valid > factor_median
            returns_above = returns_valid > returns_median
            
            # 计算命中率
            hits = (factor_above == returns_above).sum()
            hit_counts += hits
            total_counts += len(factor_valid)
            
        if total_counts == 0:
            return np.nan
            
        return hit_counts / total_counts
    
    @validation_metric(name='稳定性', desc='因子IC的时间序列稳定性', order=50)
    def stability(self, 
                 window: int = 60,
                 lookforward: int = 1) -> Dict[str, float]:
        """
        计算稳定性
        
        Args:
            window: 滚动窗口大小
            lookforward: 未来期数
            
        Returns:
            Dict[str, float]: 稳定性指标
        """
        ic_result = self.information_coefficient(lookforward)
        daily_ics = ic_result.get('daily_ics', [])
        
        if not daily_ics or len(daily_ics) < window:
            return {'rolling_std': np.nan, 'max_drawdown': np.nan}
            
        daily_ics_array = np.array(daily_ics)
        
        # 滚动标准差
        rolling_std = pd.Series(daily_ics_array).rolling(window=window).std().dropna()
        avg_rolling_std = rolling_std.mean() if not rolling_std.empty else np.nan
        
        # IC的最大回撤
        cumulative = np.cumprod(1 + daily_ics_array)
        running_max = np.maximum.accumulate(cumulative)
        drawdown = (running_max - cumulative) / running_max
        max_drawdown = drawdown.max() if len(drawdown) > 0 else np.nan
        
        result = {
            'rolling_std': float(avg_rolling_std),
            'max_drawdown': float(max_drawdown),
            'autocorrelation': self._calculate_autocorrelation(daily_ics_array)
        }
        
        return result
    
    def _calculate_autocorrelation(self, series: np.ndarray, max_lag: int = 5) -> List[float]:
        """
        计算自相关系数
        
        Args:
            series: 时间序列
            max_lag: 最大滞后阶数
            
        Returns:
            List[float]: 各阶自相关系数
        """
        if len(series) < max_lag * 2:
            return [np.nan] * max_lag
            
        autocorrs = []
        for lag in range(1, max_lag + 1):
            if lag >= len(series):
                autocorrs.append(np.nan)
                continue
                
            corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]
            autocorrs.append(float(corr))
            
        return autocorrs
    
    def run_all_validations(self, 
                           lookforward: int = 1,
                           save_report: bool = False,
                           output_dir: str = ".") -> Dict[str, Any]:
        """
        运行所有验证
        
        Args:
            lookforward: 未来期数
            save_report: 是否保存报告
            output_dir: 输出目录
            
        Returns:
            Dict[str, Any]: 所有验证结果
        """
        if not self._validate_data():
            logger.error("数据无效，无法运行验证")
            return {}
            
        # 收集所有@validation_metric装饰的方法
        validation_methods = []
        for name in dir(self):
            if name.startswith('_'):
                continue
            attr = getattr(self, name, None)
            if callable(attr) and hasattr(attr, '_is_validation_metric'):
                validation_methods.append((name, attr))
                
        # 按order排序
        validation_methods.sort(key=lambda x: getattr(x[1], '_metric_order', 99))
        
        # 运行验证
        results = {}
        for method_name, method in validation_methods:
            try:
                # 根据方法参数决定是否传递lookforward
                import inspect
                sig = inspect.signature(method)
                params = list(sig.parameters.keys())
                
                if 'lookforward' in params:
                    result = method(lookforward=lookforward)
                else:
                    result = method()
                    
                results[method_name] = {
                    'name': getattr(method, '_metric_name', method_name),
                    'desc': getattr(method, '_metric_desc', ''),
                    'value': result,
                    'order': getattr(method, '_metric_order', 99)
                }
                
                logger.info(f"验证 {method_name} 完成")
                
            except Exception as e:
                logger.error(f"验证 {method_name} 失败: {e}")
                results[method_name] = {
                    'name': getattr(method, '_metric_name', method_name),
                    'desc': getattr(method, '_metric_desc', ''),
                    'value': None,
                    'error': str(e),
                    'order': getattr(method, '_metric_order', 99)
                }
                
        # 缓存结果
        self._validation_results = results
        
        # 生成报告
        if save_report:
            self.generate_report(results, output_dir)
            
        return results
    
    def generate_report(self, 
                       results: Optional[Dict[str, Any]] = None,
                       output_dir: str = ".") -> str:
        """
        生成验证报告
        
        Args:
            results: 验证结果，如果为None则使用缓存结果
            output_dir: 输出目录
            
        Returns:
            str: 报告文件路径
        """
        if results is None:
            results = self._validation_results
            
        if not results:
            logger.warning("没有验证结果可生成报告")
            return ""
            
        # 准备报告数据
        report_data = {
            'title': '因子验证报告',
            'created_at': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'factor_shape': self.factor_values.shape if self.factor_values is not None else None,
            'results': []
        }
        
        # 整理结果
        for method_name, result_info in results.items():
            if 'error' in result_info:
                continue
                
            report_data['results'].append({
                'name': result_info['name'],
                'description': result_info['desc'],
                'value': result_info['value'],
                'order': result_info['order']
            })
            
        # 按order排序
        report_data['results'].sort(key=lambda x: x['order'])
        
        # 生成JSON报告
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = output_path / f"factor_validation_{timestamp}.json"
        
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(report_data, f, ensure_ascii=False, indent=2, default=str)
            
        logger.info(f"验证报告已保存至: {json_path}")
        
        return str(json_path)
    
    def plot_ic_series(self, 
                      lookforward: int = 1,
                      save_path: Optional[str] = None):
        """
        绘制IC时间序列图
        
        Args:
            lookforward: 未来期数
            save_path: 保存路径，如果为None则显示图表
        """
        ic_result = self.information_coefficient(lookforward)
        daily_ics = ic_result.get('daily_ics', [])
        dates = ic_result.get('dates', [])
        
        if not daily_ics or not dates:
            logger.warning("没有IC数据可绘制")
            return
            
        plt.figure(figsize=(12, 6))
        
        # 转换日期
        date_objs = [datetime.strptime(d, '%Y-%m-%d') for d in dates]
        
        # 绘制IC序列
        plt.plot(date_objs, daily_ics, label='Daily IC', linewidth=1)
        plt.axhline(y=0, color='r', linestyle='--', alpha=0.5)
        plt.axhline(y=np.mean(daily_ics), color='g', linestyle='--', 
                   label=f'Mean IC: {np.mean(daily_ics):.4f}')
        
        # 填充正负区域
        plt.fill_between(date_objs, 0, daily_ics, where=np.array(daily_ics)>=0, 
                        alpha=0.3, color='green')
        plt.fill_between(date_objs, 0, daily_ics, where=np.array(daily_ics)<0, 
                        alpha=0.3, color='red')
        
        plt.title(f'Information Coefficient Time Series (Lookforward={lookforward})')
        plt.xlabel('Date')
        plt.ylabel('IC')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"IC图表已保存至: {save_path}")
        else:
            plt.show()
            
    def plot_group_returns(self, 
                          lookforward: int = 1,
                          save_path: Optional[str] = None):
        """
        绘制分组收益图
        
        Args:
            lookforward: 未来期数
            save_path: 保存路径，如果为None则显示图表
        """
        group_result = self.group_returns(lookforward)
        returns = group_result.get('returns', {})
        
        if not returns:
            logger.warning("没有分组收益数据可绘制")
            return
            
        groups = list(returns.keys())
        values = list(returns.values())
        
        plt.figure(figsize=(10, 6))
        
        # 绘制柱状图
        bars = plt.bar(groups, values, color='steelblue', alpha=0.7)
        
        # 标记最高和最低组
        max_idx = np.nanargmax(values)
        min_idx = np.nanargmin(values)
        bars[max_idx].set_color('green')
        bars[min_idx].set_color('red')
        
        # 添加数值标签
        for i, (group, value) in enumerate(zip(groups, values)):
            if not np.isnan(value):
                plt.text(group, value, f'{value:.4f}', 
                        ha='center', va='bottom' if value >= 0 else 'top')
                
        plt.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        plt.title(f'Group Returns (Lookforward={lookforward})')
        plt.xlabel('Group (1=Lowest Factor, 10=Highest Factor)')
        plt.ylabel('Average Return')
        plt.grid(True, alpha=0.3, axis='y')
        
        # 添加多空收益标注
        spread = group_result.get('spread', np.nan)
        if not np.isnan(spread):
            plt.text(0.5, 0.95, f'Long-Short Spread: {spread:.4f}', 
                    transform=plt.gca().transAxes, fontsize=12,
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"分组收益图表已保存至: {save_path}")
        else:
            plt.show()