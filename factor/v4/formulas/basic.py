"""
factor/v3/formulas/basic.py - 因子原子基元

[重构] 从 v3/formulas.py 拆分
"""

from typing import Dict
from ..base import FactorCategory

# ═══════════════════════════════════════════════════════════════════════════
#
# 经典因子构建块，覆盖 6 个核心方向。
# 每个基元都是最小粒度的因子原子，GP 以此为基础组合出复杂因子。
# 作为 GP 种子公式的质量保障。
#
# [新增] 2026-06-01 v3 基础因子库
# ═══════════════════════════════════════════════════════════════════════════

BASIC_FACTORS: Dict[str, str] = {
    # ── 动量类 ──
    'mom_5d':    'sub(close, delay(close, 5))',               # 5日动量
    'mom_10d':   'sub(close, delay(close, 10))',              # 10日动量
    'mom_20d':   'sub(close, delay(close, 20))',              # 20日动量
    'mom_60d':   'sub(close, delay(close, 60))',              # 60日动量

    # ── 反转类 ──
    'rev_5d':    'neg(sub(close, delay(close, 5)))',           # 5日反转
    'rev_20d':   'neg(sub(close, delay(close, 20)))',          # 20日反转

    # ── 波动率类 ──
    'vol_10d':   'ts_std(close, 10)',                          # 10日波动率
    'vol_20d':   'ts_std(close, 20)',                          # 20日波动率
    'vol_60d':   'ts_std(close, 60)',                          # 60日波动率
    'range_pct': 'div(sub(high, low), close)',                 # 日内波幅率 (high-low)/close

    # ── 成交量类 ──
    'vol_ratio_10d': 'div(volume, ts_mean(volume, 10))',       # 10日量比
    'vol_ratio_20d': 'div(volume, ts_mean(volume, 20))',       # 20日量比
    'dollar_volume': 'mul(sub(close, open), volume)',          # 买入金额 (涨跌方向×量)

    # ── 截面标准化类 ──
    'cs_rank_mom_20d':    'cs_rank(sub(close, delay(close, 20)))',    # 截面排名动量
    'cs_zscore_mom_10d':  'ts_zscore(sub(close, delay(close, 10)), 20)',  # 时序标准化动量
    'cs_rank_volume':     'cs_rank(volume)',                            # 截面排名量

    # ── 均线交叉类 ──
    'ma_cross_5_20':    'sub(sma(close, 5, 0), sma(close, 20, 0))',   # MA5-MA20
    'ma_cross_10_60':   'sub(sma(close, 10, 0), sma(close, 60, 0))',  # MA10-MA60
    'sma_10d':          'sma(close, 10, 0)',                           # 10日均线（用于组合）

    # ── 量价相关性 ──
    'corr_close_vol_20d': 'correlation(close, volume, 20)',             # 价量相关性
}

BASIC_FACTORS_CATEGORIES: Dict[str, FactorCategory] = {
    'mom_5d': FactorCategory.MOMENTUM, 'mom_10d': FactorCategory.MOMENTUM,
    'mom_20d': FactorCategory.MOMENTUM, 'mom_60d': FactorCategory.MOMENTUM,
    'rev_5d': FactorCategory.REVERSAL, 'rev_20d': FactorCategory.REVERSAL,
    'vol_10d': FactorCategory.VOLATILITY, 'vol_20d': FactorCategory.VOLATILITY,
    'vol_60d': FactorCategory.VOLATILITY, 'range_pct': FactorCategory.VOLATILITY,
    'vol_ratio_10d': FactorCategory.VOLUME, 'vol_ratio_20d': FactorCategory.VOLUME,
    'dollar_volume': FactorCategory.VOLUME,
    'cs_rank_mom_20d': FactorCategory.TECHNICAL, 'cs_zscore_mom_10d': FactorCategory.TECHNICAL,
    'cs_rank_volume': FactorCategory.TECHNICAL,
    'ma_cross_5_20': FactorCategory.PRICE, 'ma_cross_10_60': FactorCategory.PRICE,
    'sma_10d': FactorCategory.PRICE,
    'corr_close_vol_20d': FactorCategory.TECHNICAL,
}
