"""
signals/v4/market_breadth.py — 市场广度特征计算
=============================================================================

突破 AIdev GP 的根本局限：特征池缺少市场广度指标。
市场广度指标是"温度计"（驱动因素），优于技术指标（滞后反映）。


============================================================================
                         特征类别（竖式）
============================================================================

 ┌─────────────────────────────────────────────────────────────────────┐
 │  输入：全市场股票池的日线数据                                         │
 │  (需包含每只股票的涨跌标记、新高新低标记)                              │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌──────────────────────────────────────────────────────────────────────┐
 │  特征计算（4 大类别）                                                  │
 │                                                                       │
 │  1. ADV_DEC_RATIO     涨跌家数比率                                     │
 │     = 上涨家数 / 下跌家数    → 市场宽度                                │
 │                                                                       │
 │  2. SECTOR_DIFFUSION  行业扩散度                                       │
 │     = 上涨行业数 / 总行业数 → 板块轮动广度                             │
 │                                                                       │
 │  3. NEW_HIGH_LOW      新高新低差                                       │
 │     = 创N日新高家数 - 创N日新低家数 → 极端情绪                         │
 │                                                                       │
 │  4. MCCLELLAN_OSCILLATOR  麦克莱伦振荡器                               │
 │     = EMA(ADV-DEC, 19) - EMA(ADV-DEC, 39) → 中期广度动量              │
 └──────────────────────────────────────────────────────────────────────┘


============================================================================
                         与现有框架的关系
============================================================================

 ┌─────────────────────┬───────────────────────────────────────────┐
 │  features.py         │ OHLCV 技术指标 → 原子价格特征             │
 │  market_breadth.py   │ 全市场统计 → 市场氛围特征 (新增维度)       │
 │  结合使用            │ 技术指标 + 广度指标 → 更完整特征空间       │
 └─────────────────────┴───────────────────────────────────────────┘

============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional, Callable, Tuple


def _rolling(arr: np.ndarray, window: int, func=np.nanmean) -> np.ndarray:
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        seg = arr[max(0, i - window + 1):i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) > 0:
            result[i] = func(valid)
    return result


# ============================================================
# 市场广度特征：需要全市场数据源
# ============================================================

def calc_advance_decline_ratio(advance: np.ndarray, decline: np.ndarray,
                                smooth: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """
    涨跌家数比率

    Args:
        advance: 每日上涨家数
        decline: 每日下跌家数
        smooth: 平滑窗口

    Returns:
        (raw, smoothed_raw)
    """
    total = advance + decline
    ratio = np.where(total > 0, (advance - decline) / total, 0)
    smoothed = _rolling(ratio, smooth)
    return ratio, smoothed


def calc_sector_diffusion(sector_returns: pd.DataFrame, lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    行业扩散度 — 申万行业指数上涨比例

    Args:
        sector_returns: DataFrame，列=行业，值=日收益率
        lookback: 回看窗口

    Returns:
        (diffusion, smoothed)
    """
    is_positive = (sector_returns > 0).astype(float)
    diffusion = is_positive.mean(axis=1).values
    smoothed = _rolling(diffusion, lookback)
    return diffusion, smoothed


def calc_new_high_low(new_highs: np.ndarray, new_lows: np.ndarray,
                       total: int, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """
    新高新低差

    Args:
        new_highs: 每日创N日新高家数
        new_lows: 每日创N日新低家数
        total: 全市场股票总数
        window: 平滑窗口

    Returns:
        (raw, smoothed)
    """
    ratio = np.where(total > 0, (new_highs - new_lows) / total, 0)
    smoothed = _rolling(ratio, window)
    return ratio, smoothed


def calc_mcclellan_oscillator(advance: np.ndarray, decline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    麦克莱伦振荡器

    McClellan Oscillator = 差值(EMA19) - EMA(EMA39)
    经典的美国市场广度指标

    Args:
        advance: 上涨家数
        decline: 下跌家数

    Returns:
        (oscillator, smoothed)
    """
    diff = advance - decline

    # EMA(19)
    ema19 = np.full(len(diff), np.nan, dtype=float)
    alpha19 = 2 / (19 + 1)
    ema19[0] = diff[0]
    for i in range(1, len(diff)):
        ema19[i] = alpha19 * diff[i] + (1 - alpha19) * ema19[i - 1]

    # EMA(39)
    ema39 = np.full(len(diff), np.nan, dtype=float)
    alpha39 = 2 / (39 + 1)
    ema39[0] = diff[0]
    for i in range(1, len(diff)):
        ema39[i] = alpha39 * diff[i] + (1 - alpha39) * ema39[i - 1]

    oscillator = ema19 - ema39
    smoothed = _rolling(oscillator, 5)
    return oscillator, smoothed


def calc_arms_index(advance: np.ndarray, decline: np.ndarray,
                     adv_volume: np.ndarray, decl_volume: np.ndarray) -> np.ndarray:
    """
    Arms Index (TRIN) — 衡量市场买卖压力

    TRIN = (上涨家数/下跌家数) / (上涨量/下跌量)

    Args:
        advance/decline: 涨跌家数
        adv_volume/decl_volume: 涨跌成交量
    """
    adv_ratio = np.where(decline > 0, advance / decline, 0)
    vol_ratio = np.where(decl_volume > 0, adv_volume / decl_volume, 0)
    trin = np.where(vol_ratio > 0, adv_ratio / vol_ratio, 0)
    return trin


# ============================================================
# 将市场广度特征注册到FeatureSpace
# ============================================================

def _breadth_ad_ratio(data: pd.DataFrame, **params) -> Tuple[np.ndarray, np.ndarray]:
    """市场广度-涨跌比: 从DataFrame提取advance/decline列计算"""
    smooth = params.get('smooth', 5)
    if 'advance' in data.columns and 'decline' in data.columns:
        return calc_advance_decline_ratio(data['advance'].values, data['decline'].values, smooth)
    return np.full(len(data), np.nan), np.full(len(data), np.nan)


def _breadth_new_high_low(data: pd.DataFrame, **params) -> Tuple[np.ndarray, np.ndarray]:
    """市场广度-新高新低差: 从DataFrame提取new_highs/new_lows列计算"""
    window = params.get('window', 20)
    # [修复] total 应为全市场股票总数，用 len(data) 会取到交易日数（错误）
    total = params.get('total', 5000)  # 默认 5000，按实际全市场股票数传入
    if 'new_highs' in data.columns and 'new_lows' in data.columns:
        return calc_new_high_low(data['new_highs'].values, data['new_lows'].values,
                                  total, window)
    return np.full(len(data), np.nan), np.full(len(data), np.nan)


def _breadth_mcclellan(data: pd.DataFrame, **params) -> Tuple[np.ndarray, np.ndarray]:
    """市场广度-麦克莱伦振荡器: 从DataFrame提取advance/decline列计算"""
    if 'advance' in data.columns and 'decline' in data.columns:
        return calc_mcclellan_oscillator(data['advance'].values, data['decline'].values)
    return np.full(len(data), np.nan), np.full(len(data), np.nan)


def _breadth_arms(data: pd.DataFrame, **params) -> Tuple[np.ndarray, np.ndarray]:
    """市场广度-Arms指数(TRIN): 从DataFrame提取advance/decline/adv_vol/decl_vol列计算"""
    if all(c in data.columns for c in ['advance', 'decline', 'adv_volume', 'decl_volume']):
        trin = calc_arms_index(data['advance'].values, data['decline'].values,
                                data['adv_volume'].values, data['decl_volume'].values)
        return trin, _rolling(trin, 5)
    return np.full(len(data), np.nan), np.full(len(data), np.nan)


# [移除] 2026-06-22 register_breadth_features() —
# 引用已删除的 v3 FeatureSpace._FEATURE_CALC_REGISTRY,
# v4 中市场广度通过 extra_features 字典直接注入 Expression.generate()。
