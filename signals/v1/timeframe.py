# signal/timeframe.py - 多周期数据处理
"""
多周期数据对齐器

核心功能：
1. 不同频率信号的对齐和聚合
2. 多周期特征创建
3. 时间窗口分析
"""

from typing import Dict, List, Optional, Union, Callable, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, time


class MultiTimeframeAligner:
    """
    多周期数据对齐器
    
    解决多周期策略中的核心问题：将不同频率的数据对齐到统一的时间索引
    
    主要功能：
    1. 信号对齐：将高频信号聚合到低频
    2. 特征对齐：对齐不同频率的特征数据
    3. 数据合并：合并多个数据源
    """
    
    # 常用频率映射
    FREQUENCY_MAP = {
        'tick': '1s',
        '1s': '1s',
        '1min': '1min',
        '1m': '1min',
        '5min': '5min',
        '5m': '5min',
        '15min': '15min',
        '15m': '15min',
        '30min': '30min',
        '30m': '30min',
        '60min': '60min',
        '1h': '1h',
        '2h': '2h',
        '4h': '4h',
        'daily': 'D',
        'd': 'D',
        'weekly': 'W',
        'w': 'W',
        'monthly': 'M',
        'mth': 'M',
    }
    
    # 默认聚合方法
    AGG_METHODS = {
        'price': 'last',      # 价格取最后一个
        'volume': 'sum',      # 成交量求和
        'signal': 'last',     # 信号取最后一个
        'return': 'sum',      # 收益率求和
        'volatility': 'mean', # 波动率取平均
    }
    
    @staticmethod
    def infer_frequency(series: pd.Series) -> str:
        """
        推断时间序列的频率
        
        Args:
            series: 时间序列，必须有DatetimeIndex
            
        Returns:
            频率字符串，如 '1min', '5min', '1h', 'D' 等
            
        Raises:
            ValueError: 如果索引不是DatetimeIndex
        """
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have DatetimeIndex")
        
        if len(series) < 2:
            return 'unknown'
        
        # 尝试推断频率
        try:
            freq = pd.infer_freq(series.index)
            if freq:
                return freq
        except:
            pass
        
        # 计算时间间隔
        if len(series) >= 2:
            intervals = series.index[1:] - series.index[:-1]
            median_interval = intervals.median()
            
            # 转换为标准频率
            seconds = median_interval.total_seconds()
            if seconds <= 1:
                return '1s'
            elif seconds <= 60:
                return f'{int(seconds)}s'
            elif seconds <= 3600:
                minutes = int(seconds / 60)
                return f'{minutes}min'
            elif seconds <= 86400:
                hours = int(seconds / 3600)
                return f'{hours}h'
            else:
                days = int(seconds / 86400)
                return f'{days}D'
        
        return 'unknown'
    
    @staticmethod
    def align_signals(
        signals_dict: Dict[str, pd.Series],
        target_freq: str = 'D',
        method: str = 'last',
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        将不同频率的信号对齐到目标频率
        
        Args:
            signals_dict: 信号字典 {信号名: 信号序列}
            target_freq: 目标频率，支持pandas频率字符串
            method: 聚合方法 ('last', 'mean', 'sum', 'max', 'min', 'first')
            fill_method: 填充方法 ('ffill', 'bfill', 'nearest', None)
            
        Returns:
            对齐后的DataFrame，index为目标频率
            
        Example:
            >>> daily_signal = pd.Series(...)  # 日线信号
            >>> hourly_signal = pd.Series(...)  # 小时线信号
            >>> aligned = MultiTimeframeAligner.align_signals(
            ...     {'daily': daily_signal, 'hourly': hourly_signal},
            ...     target_freq='D'
            ... )
        """
        if not signals_dict:
            return pd.DataFrame()
        
        aligned_signals = {}
        
        for name, series in signals_dict.items():
            if series.empty:
                continue
            
            # 确保是DatetimeIndex
            if not isinstance(series.index, pd.DatetimeIndex):
                raise ValueError(f"Signal '{name}' must have DatetimeIndex")
            
            # 推断原始频率
            source_freq = MultiTimeframeAligner.infer_frequency(series)
            
            # 如果已经是目标频率或无法推断，直接使用
            if source_freq == target_freq or source_freq == 'unknown':
                aligned_signals[name] = series
                continue
            
            # 高频转低频：需要聚合
            try:
                # 使用resample进行聚合
                if method == 'last':
                    resampled = series.resample(target_freq).last()
                elif method == 'mean':
                    resampled = series.resample(target_freq).mean()
                elif method == 'sum':
                    resampled = series.resample(target_freq).sum()
                elif method == 'max':
                    resampled = series.resample(target_freq).max()
                elif method == 'min':
                    resampled = series.resample(target_freq).min()
                elif method == 'first':
                    resampled = series.resample(target_freq).first()
                else:
                    raise ValueError(f"Unknown aggregation method: {method}")
                
                # 填充缺失值
                if fill_method and resampled.isna().any():
                    if fill_method == 'ffill':
                        resampled = resampled.ffill()
                    elif fill_method == 'bfill':
                        resampled = resampled.bfill()
                    elif fill_method == 'nearest':
                        # 使用最近的非空值填充
                        resampled = resampled.fillna(method='ffill').fillna(method='bfill')
                
                aligned_signals[name] = resampled
                
            except Exception as e:
                raise ValueError(f"Failed to align signal '{name}': {e}")
        
        # 合并所有信号
        if not aligned_signals:
            return pd.DataFrame()
        
        # 创建DataFrame并对齐索引
        aligned_df = pd.DataFrame(aligned_signals)
        
        # 确保所有信号有相同的时间索引（取交集）
        common_index = None
        for name, series in aligned_signals.items():
            if common_index is None:
                common_index = series.dropna().index
            else:
                common_index = common_index.intersection(series.dropna().index)
        
        if len(common_index) == 0:
            # 如果没有共同索引，使用第一个信号的索引
            first_series = list(aligned_signals.values())[0]
            aligned_df = aligned_df.reindex(first_series.index)
        else:
            aligned_df = aligned_df.reindex(common_index)
        
        return aligned_df.dropna(how='all')
    
    @staticmethod
    def align_to_reference(
        signals_dict: Dict[str, pd.Series],
        reference_index: pd.DatetimeIndex,
        fill_method: str = 'ffill'
    ) -> pd.DataFrame:
        """
        将所有信号对齐到参考时间索引
        
        Args:
            signals_dict: 信号字典
            reference_index: 参考时间索引
            fill_method: 填充方法
            
        Returns:
            对齐到参考索引的DataFrame
        """
        aligned_signals = {}
        
        for name, series in signals_dict.items():
            if series.empty:
                continue
            
            # 重新索引到参考索引
            aligned = series.reindex(reference_index, method=fill_method)
            aligned_signals[name] = aligned
        
        return pd.DataFrame(aligned_signals)
    
    @staticmethod
    def create_multi_timeframe_features(
        data_dict: Dict[str, pd.DataFrame],
        feature_funcs: Dict[str, Callable],
        target_freq: str = 'D'
    ) -> pd.DataFrame:
        """
        从多周期数据创建特征
        
        Args:
            data_dict: 多周期数据字典 {频率名: DataFrame}
            feature_funcs: 特征函数字典 {特征名: 函数}
            target_freq: 目标频率
            
        Returns:
            特征DataFrame
            
        Example:
            >>> daily_data = pd.DataFrame(...)  # 日线数据
            >>> hourly_data = pd.DataFrame(...)  # 小时线数据
            >>> features = MultiTimeframeAligner.create_multi_timeframe_features(
            ...     {'daily': daily_data, 'hourly': hourly_data},
            ...     feature_funcs={
            ...         'daily_return': lambda df: df['close'].pct_change(),
            ...         'hourly_vol': lambda df: df['close'].pct_change().rolling(24).std()
            ...     }
            ... )
        """
        all_features = {}
        
        for freq_name, data in data_dict.items():
            if data.empty:
                continue
            
            # 为每个数据源计算特征
            for feature_name, func in feature_funcs.items():
                try:
                    # 计算特征
                    feature_series = func(data)
                    
                    if feature_series is not None:
                        # 对齐到目标频率
                        aligned = MultiTimeframeAligner.align_signals(
                            {f'{freq_name}_{feature_name}': feature_series},
                            target_freq=target_freq,
                            method='last'
                        )
                        
                        if not aligned.empty:
                            col_name = aligned.columns[0]
                            all_features[col_name] = aligned[col_name]
                            
                except Exception as e:
                    print(f"Warning: Failed to compute feature '{feature_name}' for '{freq_name}': {e}")
                    continue
        
        # 合并所有特征
        if not all_features:
            return pd.DataFrame()
        
        # 对齐所有特征序列
        return MultiTimeframeAligner.align_signals(all_features, target_freq=target_freq)
    
    @staticmethod
    def merge_timeframes(
        high_freq_data: pd.DataFrame,
        low_freq_data: pd.DataFrame,
        on: str = 'date'
    ) -> pd.DataFrame:
        """
        合并高低频数据
        
        Args:
            high_freq_data: 高频数据
            low_freq_data: 低频数据
            on: 合并的列名（通常是日期列）
            
        Returns:
            合并后的DataFrame
            
        Note:
            低频数据会向前填充到高频数据
        """
        # 确保有日期列
        if on not in high_freq_data.columns:
            high_freq_data = high_freq_data.reset_index()
            if on not in high_freq_data.columns:
                high_freq_data = high_freq_data.rename(columns={'index': on})
        
        if on not in low_freq_data.columns:
            low_freq_data = low_freq_data.reset_index()
            if on not in low_freq_data.columns:
                low_freq_data = low_freq_data.rename(columns={'index': on})
        
        # 合并
        merged = pd.merge_asof(
            high_freq_data.sort_values(on),
            low_freq_data.sort_values(on),
            on=on,
            suffixes=('_high', '_low')
        )
        
        return merged.set_index(on) if on in merged.columns else merged
    
    @staticmethod
    def extract_time_window_features(
        intraday_data: pd.DataFrame,
        window_defs: Dict[str, Tuple[str, str]],
        feature_types: List[str] = None
    ) -> pd.DataFrame:
        """
        提取时间窗口特征（如早盘、午盘、尾盘）
        
        Args:
            intraday_data: 日内数据，必须有DatetimeIndex
            window_defs: 窗口定义 {窗口名: (开始时间, 结束时间)}
            feature_types: 特征类型列表 ['return', 'volume', 'volatility', 'range']
            
        Returns:
            窗口特征DataFrame（日频率）
        """
        if not isinstance(intraday_data.index, pd.DatetimeIndex):
            raise ValueError("intraday_data must have DatetimeIndex")
        
        if feature_types is None:
            feature_types = ['return', 'volume', 'volatility', 'range']
        
        features_dict = {}
        
        # 按日期分组
        for date, day_data in intraday_data.groupby(intraday_data.index.date):
            date_str = date.strftime('%Y-%m-%d')
            
            for window_name, (start_str, end_str) in window_defs.items():
                # 解析时间
                start_time = pd.to_datetime(start_str).time()
                end_time = pd.to_datetime(end_str).time()
                
                # 提取窗口数据
                window_mask = (day_data.index.time >= start_time) & (day_data.index.time <= end_time)
                window_data = day_data[window_mask]
                
                if len(window_data) < 2:
                    # 数据不足，填充NaN
                    for ft in feature_types:
                        features_dict.setdefault(f'{window_name}_{ft}', {})[date_str] = np.nan
                    continue
                
                # 计算特征
                for ft in feature_types:
                    if ft == 'return':
                        # 窗口收益率
                        if 'close' in window_data.columns and 'open' in window_data.columns:
                            value = window_data['close'].iloc[-1] / window_data['open'].iloc[0] - 1
                        else:
                            value = np.nan
                    
                    elif ft == 'volume':
                        # 窗口成交量
                        if 'volume' in window_data.columns:
                            value = window_data['volume'].sum()
                        else:
                            value = np.nan
                    
                    elif ft == 'volatility':
                        # 窗口波动率
                        if 'close' in window_data.columns:
                            returns = window_data['close'].pct_change()
                            value = returns.std()
                        else:
                            value = np.nan
                    
                    elif ft == 'range':
                        # 价格范围（最高-最低）/开盘
                        if all(col in window_data.columns for col in ['high', 'low', 'open']):
                            price_range = window_data['high'].max() - window_data['low'].min()
                            value = price_range / window_data['open'].iloc[0]
                        else:
                            value = np.nan
                    
                    else:
                        value = np.nan
                    
                    features_dict.setdefault(f'{window_name}_{ft}', {})[date_str] = value
        
        # 转换为DataFrame
        features_df = pd.DataFrame(features_dict)
        features_df.index = pd.to_datetime(features_df.index)
        
        return features_df.sort_index()


class FrequencyConverter:
    """
    频率转换器
    
    专门处理频率转换和重采样
    """
    
    @staticmethod
    def convert_frequency(
        series: pd.Series,
        target_freq: str,
        agg_method: str = 'last'
    ) -> pd.Series:
        """
        转换时间序列频率
        
        Args:
            series: 原始时间序列
            target_freq: 目标频率
            agg_method: 聚合方法
            
        Returns:
            转换频率后的序列
        """
        if series.empty:
            return series
        
        if not isinstance(series.index, pd.DatetimeIndex):
            raise ValueError("Series must have DatetimeIndex")
        
        # 检查是否需要转换
        current_freq = MultiTimeframeAligner.infer_frequency(series)
        if current_freq == target_freq:
            return series
        
        # 执行转换
        if agg_method == 'last':
            return series.resample(target_freq).last()
        elif agg_method == 'mean':
            return series.resample(target_freq).mean()
        elif agg_method == 'sum':
            return series.resample(target_freq).sum()
        elif agg_method == 'max':
            return series.resample(target_freq).max()
        elif agg_method == 'min':
            return series.resample(target_freq).min()
        elif agg_method == 'first':
            return series.resample(target_freq).first()
        elif agg_method == 'ohlc':
            # 对于价格数据，返回OHLC
            if isinstance(series, pd.Series):
                resampled = series.resample(target_freq).ohlc()
                return resampled['close']  # 默认返回收盘价
            else:
                raise ValueError("ohlc method requires Series input")
        else:
            raise ValueError(f"Unknown aggregation method: {agg_method}")
    
    @staticmethod
    def infer_aggregation_method(
        series_name: str,
        data_type: str = None
    ) -> str:
        """
        根据序列名称推断聚合方法
        
        Args:
            series_name: 序列名称
            data_type: 数据类型
            
        Returns:
            推荐的聚合方法
        """
        name_lower = series_name.lower()
        
        # 基于名称的推断
        if any(word in name_lower for word in ['price', 'close', 'open', 'high', 'low', 'last']):
            return 'last'
        elif any(word in name_lower for word in ['volume', 'amount', 'sum', 'total']):
            return 'sum'
        elif any(word in name_lower for word in ['return', 'pnl', 'profit']):
            return 'sum'
        elif any(word in name_lower for word in ['volatility', 'std', 'risk', 'var']):
            return 'mean'
        elif any(word in name_lower for word in ['signal', 'indicator', 'score', 'alpha']):
            return 'last'
        elif any(word in name_lower for word in ['count', 'num', 'quantity']):
            return 'sum'
        else:
            # 基于数据类型的推断
            if data_type == 'price':
                return 'last'
            elif data_type == 'volume':
                return 'sum'
            elif data_type == 'return':
                return 'sum'
            else:
                return 'mean'  # 默认取平均