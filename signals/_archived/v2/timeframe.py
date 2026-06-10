"""
signals/v2/timeframe.py — 多周期特征计算
=============================================================================

提供工具函数，将 FeatureSpace 扩展到多周期（周线/月线）。
核心思路：下采样 → 在各周期上计算特征 → 前向填充对齐到日线 → 前缀区分。

与 v2 框架的关系：
  - 不修改 FeatureSpace 内部逻辑
  - 通过外部重采样 + 列名前缀实现多周期特征共存
  - 表达式里可直接引用 "W_ATR{7}" / "M_RSI{14}"

使用示例：
    from signals.v2 import FeatureSpace
    from signals.v2.timeframe import compute_multitimeframe_features

    fs = FeatureSpace()
    multi_features = compute_multitimeframe_features(
        fs, df,
        timeframes={'W': 'W-FRI', 'M': 'M'},
    )
    # multi_features 包含: ATR{7}, W_ATR{7}, M_ATR{7}, ...

============================================================================
"""

import copy
import logging
from typing import Dict, Optional
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


# ============================================================
# 核心工具
# ============================================================

def resample_ohlcv(data: pd.DataFrame, freq: str) -> pd.DataFrame:
    """
    将日线 OHLCV 下采样到周线/月线

    Args:
        data: 日线 DataFrame，需包含 open/high/low/close/volume
        freq: pandas 频率字符串，如 'W-FRI'（周五为周结束日）、'M'（月末）

    Returns:
        下采样后的 OHLCV DataFrame
    """
    required = ['open', 'high', 'low', 'close', 'volume']
    ohlc_cols = [c for c in required if c in data.columns]

    agg_map = {}
    for col in ohlc_cols:
        if col == 'volume':
            agg_map[col] = 'sum'
        elif col == 'open':
            agg_map[col] = 'first'
        elif col == 'high':
            agg_map[col] = 'max'
        elif col == 'low':
            agg_map[col] = 'min'
        elif col == 'close':
            agg_map[col] = 'last'

    resampled = data[ohlc_cols].resample(freq).agg(agg_map)
    resampled = resampled.dropna()

    # 保留原始数据中的其他列（如 advance/decline 等广度特征）
    extra_cols = [c for c in data.columns if c not in ohlc_cols]
    if extra_cols:
        for col in extra_cols:
            if pd.api.types.is_numeric_dtype(data[col]):
                resampled[col] = data[col].resample(freq).last()

    return resampled


def align_to_daily(low_freq_df: pd.DataFrame, daily_index: pd.DatetimeIndex,
                    prefix: str = '') -> pd.DataFrame:
    """
    将低频特征 DataFrame 前向填充对齐到日线索引

    Args:
        low_freq_df: 低频特征 DataFrame（周线/月线）
        daily_index: 日线 DatetimeIndex
        prefix: 列名前缀，如 'W_' 或 'M_'

    Returns:
        对齐到日线的特征 DataFrame（列名带前缀）
    """
    aligned = low_freq_df.reindex(daily_index, method='ffill')
    if prefix:
        aligned = aligned.add_prefix(prefix)
    return aligned


# ============================================================
# 多周期特征计算（核心入口）
# ============================================================

def compute_multitimeframe_features(
    feature_space,
    data: pd.DataFrame,
    timeframes: Optional[Dict[str, str]] = None,
    prefix_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """
    在多时间周期上计算 FeatureSpace 特征，对齐到日线，列名加前缀

    Args:
        feature_space: FeatureSpace 实例（仅用其配置和计算逻辑，不修改状态）
        data: 日线 OHLCV DataFrame
        timeframes: {前缀: pandas频率} 映射
                    默认: {'W': 'W-FRI', 'M': 'M'}
        prefix_map: {频率: 列名前缀} 映射
                    默认: {'W-FRI': 'W_', 'M': 'M_'}

    Returns:
        合并后的特征 DataFrame:
        - 第一组: 日线特征（无前缀），如 ATR{7}, RSI{14}
        - 后续组: 各周期特征（带前缀），如 W_ATR{7}, M_RSI{14}

    Example:
        fs = FeatureSpace()
        features = compute_multitimeframe_features(
            fs, df,
            timeframes={'W': 'W-FRI', 'M': 'M'},
        )
        # features.columns:
        # ['ATR{7}', 'RSI{14}', ..., 'W_ATR{7}', 'W_RSI{14}', ..., 'M_ATR{7}', ...]
    """
    timeframes = timeframes or {'W': 'W-FRI', 'M': 'M'}
    prefix_map = prefix_map or {'W-FRI': 'W_', 'M': 'M_'}

    all_features = []

    # Step 1: 日线特征（无前缀）
    daily_features = feature_space.fit_transform(data)
    all_features.append(daily_features)

    # Step 2: 各周期特征
    daily_index = data.index
    for label, freq in timeframes.items():
        try:
            resampled = resample_ohlcv(data, freq)
            if len(resampled) < 10:
                continue  # 数据太少，跳过

            # 在新实例上计算特征（避免污染原 FeatureSpace 状态）
            tf_fs = copy.deepcopy(feature_space)
            tf_features = tf_fs.fit_transform(resampled)

            prefix = prefix_map.get(freq, f'{label}_')
            aligned = align_to_daily(tf_features, daily_index, prefix=prefix)
            all_features.append(aligned)
        except Exception as e:
            logger.warning(f"多周期特征计算失败 (tf={label}, freq={freq}): {e}")

    # Step 3: 合并所有周期特征
    result = pd.concat(all_features, axis=1)
    result = result.replace([np.inf, -np.inf], np.nan)

    return result


# ============================================================
# 便捷函数
# ============================================================

def resample_and_compute(
    feature_space,
    data: pd.DataFrame,
    freq: str,
    prefix: str = '',
) -> pd.DataFrame:
    """
    在单个指定周期上计算特征并对齐到日线

    Args:
        feature_space: FeatureSpace 实例
        data: 日线数据
        freq: pandas 频率，如 'W-FRI'
        prefix: 列名前缀

    Returns:
        带前缀的对齐特征 DataFrame
    """
    resampled = resample_ohlcv(data, freq)
    tf_fs = copy.deepcopy(feature_space)
    features = tf_fs.fit_transform(resampled)
    return align_to_daily(features, data.index, prefix=prefix)
