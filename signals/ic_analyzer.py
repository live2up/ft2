# signals/ic_analyzer.py - 通用IC分析模块
"""
IC（信息系数）分析器 - 完整的信号预测能力评估体系

功能：
1. 基础IC（Pearson + Rank IC）
2. 多窗口滚动IC（30/60/120日）
3. IC衰减曲线（多持有期）
4. 统计显著性检验（t值/p值）
5. 年度/分阶段IC分析
6. 累积IC曲线
7. 信号换手率分析
8. IC分布形态分析

架构：
  ICAnalyzer.analyze(signals, prices, **config) → 返回完整IC计算结果字典

使用示例：
    from signals import ICAnalyzer, MASignal
    
    signals = MASignal(5, 20).generate(df)
    
    ic_result = ICAnalyzer.analyze(signals, df['close'])
    
    ic_result = ICAnalyzer.analyze(
        signals, df['close'],
        rolling_windows=[30, 60, 120],
        holding_periods=[1, 3, 5, 10, 20],
        annualize=True
    )
    
    ic_result = {
        'basic': {...},           # 基础统计
        'rolling': {...},         # 多窗口滚动IC
        'decay': {...},           # IC衰减曲线
        'significance': {...},    # 统计检验
        'annual': {...},          # 年度分析
        'cumulative': {...},      # 累积IC
        'turnover': {...},        # 换手率
        'distribution': {...},    # 分布形态
    }
"""

from typing import Dict, List, Optional, Union
import pandas as pd
import numpy as np
from scipy import stats


class ICAnalyzer:
    """
    通用IC分析器
    
    计算信号与未来收益率的相关性，提供完整的统计检验
    """
    
    @staticmethod
    def analyze(
        signals: pd.Series,
        prices: pd.Series,
        rolling_windows: List[int] = None,
        holding_periods: List[int] = None,
        alpha: float = 0.05,
        annualize: bool = True,
    ) -> Dict:
        """
        执行完整IC分析
        
        Args:
            signals: 信号序列（与prices对齐）
            prices: 价格序列
            rolling_windows: 滚动窗口列表，默认 [30, 60, 120]
            holding_periods: 持有期列表，默认 [1, 3, 5, 10, 20]
            alpha: 显著性水平，默认 0.05
            annualize: 是否进行年度分析，默认 True
            
        Returns:
            dict: 完整IC分析结果
        """
        rolling_windows = rolling_windows or [30, 60, 120]
        holding_periods = holding_periods or [1, 3, 5, 10, 20]
        
        signal_shifted = signals.shift(1)
        
        forward_returns = {}
        for period in holding_periods:
            forward_returns[period] = prices.shift(-period) / prices - 1
        
        result = {}
        
        result['basic'] = ICAnalyzer._basic_ic(signal_shifted, forward_returns, holding_periods)
        
        result['rolling'] = ICAnalyzer._rolling_ic(signal_shifted, forward_returns[5], rolling_windows)
        
        result['decay'] = ICAnalyzer._ic_decay(signal_shifted, forward_returns, holding_periods)
        
        result['significance'] = ICAnalyzer._significance_test(signal_shifted, forward_returns[5], alpha)
        
        if annualize and len(signal_shifted) > 365:
            result['annual'] = ICAnalyzer._annual_ic(signal_shifted, forward_returns[5])
        
        result['cumulative'] = ICAnalyzer._cumulative_ic(signal_shifted, forward_returns[5])
        
        result['turnover'] = ICAnalyzer._turnover_analysis(signals)
        
        result['distribution'] = ICAnalyzer._distribution_analysis(signal_shifted, forward_returns[5])
        
        return result
    
    @staticmethod
    def _calculate_ic(signal: pd.Series, returns: pd.Series, method: str = 'pearson') -> float:
        """
        计算IC值
        
        Args:
            signal: 信号序列
            returns: 收益率序列
            method: 'pearson' 或 'spearman'
            
        Returns:
            float: IC值
        """
        valid_mask = signal.notna() & returns.notna()
        signal_valid = signal[valid_mask]
        returns_valid = returns[valid_mask]
        
        if len(signal_valid) < 10:
            return np.nan
        
        if method == 'spearman':
            ic, _ = stats.spearmanr(signal_valid, returns_valid)
        else:
            ic = signal_valid.corr(returns_valid)
        
        return ic
    
    @staticmethod
    def _basic_ic(signal: pd.Series, forward_returns: Dict, holding_periods: List[int]) -> Dict:
        """基础IC统计"""
        result = {}
        
        for period in holding_periods:
            returns = forward_returns[period]
            
            pearson_ic = ICAnalyzer._calculate_ic(signal, returns, 'pearson')
            spearman_ic = ICAnalyzer._calculate_ic(signal, returns, 'spearman')
            
            result[f'IC_{period}d'] = {
                'pearson_ic': pearson_ic,
                'rank_ic': spearman_ic,
            }
        
        base_returns = forward_returns.get(5, forward_returns.get(holding_periods[0]))
        if base_returns is not None:
            ic_60 = signal.rolling(60).corr(base_returns)
            ic_valid = ic_60.dropna()
            
            result['summary'] = {
                'ic_mean': ic_valid.mean(),
                'ic_std': ic_valid.std(),
                'ic_ir': ic_valid.mean() / ic_valid.std() if ic_valid.std() > 0 else 0,
                'ic_positive_ratio': (ic_valid > 0).sum() / len(ic_valid) * 100,
                'ic_median': ic_valid.median(),
                'ic_skewness': ic_valid.skew(),
                'ic_kurtosis': ic_valid.kurtosis(),
            }
        else:
            result['summary'] = {}
        
        return result
    
    @staticmethod
    def _rolling_ic(signal: pd.Series, returns: pd.Series, windows: List[int]) -> Dict:
        """多窗口滚动IC"""
        result = {}
        
        for window in windows:
            ic_rolling = signal.rolling(window).corr(returns)
            ic_valid = ic_rolling.dropna()
            
            result[f'window_{window}d'] = {
                'series': ic_rolling,
                'mean': ic_valid.mean(),
                'std': ic_valid.std(),
                'ir': ic_valid.mean() / ic_valid.std() if ic_valid.std() > 0 else 0,
                'positive_ratio': (ic_valid > 0).sum() / len(ic_valid) * 100,
                'min': ic_valid.min(),
                'max': ic_valid.max(),
            }
        
        return result
    
    @staticmethod
    def _ic_decay(signal: pd.Series, forward_returns: Dict, holding_periods: List[int]) -> Dict:
        """IC衰减曲线"""
        decay_data = []
        
        for period in holding_periods:
            returns = forward_returns[period]
            pearson_ic = ICAnalyzer._calculate_ic(signal, returns, 'pearson')
            rank_ic = ICAnalyzer._calculate_ic(signal, returns, 'spearman')
            
            decay_data.append({
                'holding_period': period,
                'pearson_ic': pearson_ic,
                'rank_ic': rank_ic,
            })
        
        if len(decay_data) >= 2:
            first_ic = abs(decay_data[0]['pearson_ic'])
            last_ic = abs(decay_data[-1]['pearson_ic'])
            decay_rate = (first_ic - last_ic) / (holding_periods[-1] - holding_periods[0]) if first_ic > 0 else 0
        else:
            decay_rate = 0
        
        return {
            'data': decay_data,
            'decay_rate': decay_rate,
        }
    
    @staticmethod
    def _significance_test(signal: pd.Series, returns: pd.Series, alpha: float = 0.05) -> Dict:
        """统计显著性检验"""
        valid_mask = signal.notna() & returns.notna()
        signal_valid = signal[valid_mask]
        returns_valid = returns[valid_mask]
        
        n = len(signal_valid)
        
        if n < 10:
            return {'significant': False, 'message': '样本量不足'}
        
        pearson_ic = signal_valid.corr(returns_valid)
        t_stat = pearson_ic * np.sqrt((n - 2) / (1 - pearson_ic**2)) if abs(pearson_ic) < 1 else float('inf')
        p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
        
        rank_ic, rank_p_value = stats.spearmanr(signal_valid, returns_valid)
        
        return {
            'pearson_ic': pearson_ic,
            't_statistic': t_stat,
            'p_value': p_value,
            'significant': p_value < alpha,
            'confidence_level': f'{(1-alpha)*100:.0f}%',
            'rank_ic': rank_ic,
            'rank_p_value': rank_p_value,
            'rank_significant': rank_p_value < alpha,
            'sample_size': n,
        }
    
    @staticmethod
    def _annual_ic(signal: pd.Series, returns: pd.Series) -> Dict:
        """年度IC分析"""
        signal_df = pd.DataFrame({'signal': signal, 'returns': returns})
        signal_df['year'] = signal_df.index.year
        
        annual_results = {}
        
        for year, group in signal_df.groupby('year'):
            year_signal = group['signal']
            year_returns = group['returns']
            
            ic = ICAnalyzer._calculate_ic(year_signal, year_returns)
            rank_ic = ICAnalyzer._calculate_ic(year_signal, year_returns, 'spearman')
            
            valid_mask = year_signal.notna() & year_returns.notna()
            n = valid_mask.sum()
            
            if n >= 30:
                t_stat = ic * np.sqrt((n - 2) / (1 - ic**2)) if abs(ic) < 1 else float('inf')
                p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=n-2))
            else:
                t_stat = np.nan
                p_value = np.nan
            
            annual_results[int(year)] = {
                'pearson_ic': ic,
                'rank_ic': rank_ic,
                'sample_size': n,
                't_statistic': t_stat,
                'p_value': p_value,
            }
        
        return annual_results
    
    @staticmethod
    def _cumulative_ic(signal: pd.Series, returns: pd.Series) -> Dict:
        """累积IC分析"""
        valid_mask = signal.notna() & returns.notna()
        signal_valid = signal[valid_mask]
        returns_valid = returns[valid_mask]
        
        ic_series = signal_valid.rolling(60).corr(returns_valid)
        cumulative_ic = ic_series.cumsum()
        
        return {
            'series': cumulative_ic,
            'final_value': cumulative_ic.iloc[-1] if len(cumulative_ic) > 0 else 0,
        }
    
    @staticmethod
    def _turnover_analysis(signals: pd.Series) -> Dict:
        """信号换手率分析"""
        signal_changes = signals.diff().abs()
        turnover = signal_changes.mean()
        
        return {
            'mean_turnover': turnover,
            'max_turnover': signal_changes.max(),
            'min_turnover': signal_changes.min(),
            'turnover_std': signal_changes.std(),
            'high_turnover_days': (signal_changes > signal_changes.mean() + 2*signal_changes.std()).sum(),
        }
    
    @staticmethod
    def _distribution_analysis(signal: pd.Series, returns: pd.Series) -> Dict:
        """IC分布形态分析"""
        ic_60 = signal.rolling(60).corr(returns)
        ic_valid = ic_60.dropna()
        
        if len(ic_valid) < 10:
            return {'message': '样本量不足'}
        
        jb_stat, jb_p_value = stats.jarque_bera(ic_valid)
        
        return {
            'mean': ic_valid.mean(),
            'median': ic_valid.median(),
            'std': ic_valid.std(),
            'skewness': ic_valid.skew(),
            'kurtosis': ic_valid.kurtosis(),
            'min': ic_valid.min(),
            'max': ic_valid.max(),
            'percentile_5': ic_valid.quantile(0.05),
            'percentile_25': ic_valid.quantile(0.25),
            'percentile_75': ic_valid.quantile(0.75),
            'percentile_95': ic_valid.quantile(0.95),
            'is_normal': jb_p_value > 0.05,
            'normality_p_value': jb_p_value,
        }
