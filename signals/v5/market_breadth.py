"""
signals/v5/market_breadth.py — 市场广度特征计算
=============================================================================
[重构] 2026-07-07 从 v4 升级到 v5。
v5 版: 仅保留核心 calc_* 函数，通过 extra_features 注入 Expression.generate()。
"""

import numpy as np
import pandas as pd
from typing import Tuple


def _rolling(arr: np.ndarray, window: int, func=np.nanmean) -> np.ndarray:
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        seg = arr[max(0, i - window + 1):i + 1]
        valid = seg[~np.isnan(seg)]
        if len(valid) > 0:
            result[i] = func(valid)
    return result


def calc_advance_decline_ratio(advance: np.ndarray, decline: np.ndarray,
                                smooth: int = 5) -> Tuple[np.ndarray, np.ndarray]:
    """涨跌家数比率 = (上涨-下跌)/总数, 返回 (raw, smoothed)"""
    total = advance + decline
    ratio = np.where(total > 0, (advance - decline) / total, 0)
    smoothed = _rolling(ratio, smooth)
    return ratio, smoothed


def calc_sector_diffusion(sector_returns: pd.DataFrame, lookback: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """行业扩散度 — 上涨行业比例, 返回 (diffusion, smoothed)"""
    is_positive = (sector_returns > 0).astype(float)
    diffusion = is_positive.mean(axis=1).values
    smoothed = _rolling(diffusion, lookback)
    return diffusion, smoothed


def calc_new_high_low(new_highs: np.ndarray, new_lows: np.ndarray,
                       total: int, window: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """新高新低差 = (新高-新低)/总数, 返回 (raw, smoothed)"""
    ratio = np.where(total > 0, (new_highs - new_lows) / total, 0)
    smoothed = _rolling(ratio, window)
    return ratio, smoothed


def calc_mcclellan_oscillator(advance: np.ndarray, decline: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """麦克莱伦振荡器 = EMA19(adv-dec) - EMA39(adv-dec), 返回 (osc, smoothed)"""
    diff = advance - decline
    ema19 = np.full(len(diff), np.nan, dtype=float)
    alpha19 = 2 / (19 + 1)
    ema19[0] = diff[0]
    for i in range(1, len(diff)):
        ema19[i] = alpha19 * diff[i] + (1 - alpha19) * ema19[i - 1]
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
    """Arms Index (TRIN) = (上涨/下跌) / (上涨量/下跌量)"""
    adv_ratio = np.where(decline > 0, advance / decline, 0)
    vol_ratio = np.where(decl_volume > 0, adv_volume / decl_volume, 0)
    return np.where(vol_ratio > 0, adv_ratio / vol_ratio, 0)
