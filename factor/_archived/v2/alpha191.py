"""
alpha191.py — 国泰 191 Alpha 因子公式（v2 表达式语法）
=============================================================================

基于国泰安（GTJA/CSMAR）191 个 Alpha 因子，翻译为 factor v2
表达式引擎的 S-表达式语法。

与 alpha101.py（WQ 因子）的区别：
  - 因子来源：国泰安 191 因子 vs WorldQuant 101 因子
  - 设计思路：GT191 更侧重 A 股市场微观结构，涵盖更多资金流/情绪/技术指标
  - 表达式语法：完全相同，使用同一套 S-表达式引擎

翻译规则：
  RANK(x)        → cs_rank(x)
  TSRANK(x, d)   → ts_rank(x, d)
  CORR(x, y, d)  → correlation(x, y, d)
  DELTA(x, d)    → sub(x, delay(x, d))
  SUM(x, d)      → ts_sum(x, d)
  MEAN(x, d)     → ts_mean(x, d)
  STD(x, d)      → ts_std(x, d)
  MAX(x, d)      → ts_max(x, d)
  MIN(x, d)      → ts_min(x, d)
  SMA(x, n, m)   → sma(x, n, m)
  SIGNEDPOWER    → signed_power(x, a)
  LOWDAY(x, d)   → ts_argmin(x, d)
  HIGHDAY(x, d)  → ts_argmax(x, d)
  COVIANCE(x,y,d)→ covariance(x, y, d)
  REGBETA(x,y,d) → regbeta(x, y, d)
  IFELSE(c,a,b)  → ifelse(c, a, b)
  RET            → returns
  VWAP           → vwap

[新增] 2026-06-01  填充全部 191 因子表达式
=============================================================================
"""

from typing import Dict
from .base import FactorCategory


# ============================================================================
# GT191 因子公式定义
# ============================================================================

ALPHA191: Dict[str, str] = {

    # ═══════════════════════════════════════════════════════════════════
    #  动量类 (gtja_001 ~ gtja_030)
    # ═══════════════════════════════════════════════════════════════════

    # 001: N 日收益率排名
    'gtja_001': 'cs_rank(sub(close, delay(close, 5)))',

    # 002: 20 日收益率 × 5 日成交量变化
    'gtja_002': 'mul(sub(close, delay(close, 20)), sub(volume, delay(volume, 5)))',

    # 003: 成交量加权收益率排名
    'gtja_003': 'cs_rank(mul(sub(close, delay(close, 1)), volume))',

    # 004: 5 日动量 × 成交量比
    'gtja_004': 'mul(sub(close, delay(close, 5)), div(volume, ts_mean(volume, 20)))',

    # 005: 日收益 × 成交量比排名
    'gtja_005': 'cs_rank(mul(sub(close, delay(close, 1)), div(volume, ts_mean(volume, 20))))',

    # 006: 10 日动量排名 × 波动率倒数
    'gtja_006': 'mul(ts_rank(sub(close, delay(close, 1)), 10), neg(ts_std(sub(close, delay(close, 1)), 10)))',

    # 007: 20 日收益率滚动排名
    'gtja_007': 'ts_rank(sub(close, delay(close, 1)), 20)',

    # 008: 收益率加成交量双重排名
    'gtja_008': 'add(ts_rank(sub(close, delay(close, 1)), 5), ts_rank(volume, 5))',

    # 009: 4 日指数加权动量（decay_linear 近似）
    'gtja_009': 'decay_linear(sub(close, delay(close, 1)), 4)',

    # 010: 5 日收益率 × 20 日波动率反向
    'gtja_010': 'mul(sub(close, delay(close, 5)), neg(ts_std(sub(close, delay(close, 1)), 20)))',

    # 011: 成交量加速（volume delta × 价格 delta）
    'gtja_011': 'mul(sub(volume, delay(volume, 5)), sub(close, delay(close, 5)))',

    # 012: 20 日动量衰减加权
    'gtja_012': 'decay_linear(sub(close, delay(close, 1)), 20)',

    # 013: (high+low)/2 动量 vs close 动量差
    'gtja_013': 'sub(sub(div(add(high, low), 2), delay(div(add(high, low), 2), 20)), sub(close, delay(close, 20)))',

    # 014: ts_argmax(close, 30) — 距最高点天数排名
    'gtja_014': 'neg(cs_rank(ts_argmax(close, 30)))',

    # 015: ts_argmin(close, 30) — 距最低点天数
    'gtja_015': 'cs_rank(ts_argmin(close, 30))',

    # 016: close 的 ts_max - ts_min 振幅
    'gtja_016': 'sub(ts_max(close, 20), ts_min(close, 20))',

    # 017: (close - ts_min) / (ts_max - ts_min)  随机指标
    'gtja_017': 'div(sub(close, ts_min(close, 20)), sub(ts_max(close, 20), ts_min(close, 20)))',

    # 018: 10 日收益率 × 日收益率加速度
    'gtja_018': 'mul(sub(close, delay(close, 10)), sub(sub(close, delay(close, 1)), delay(sub(close, delay(close, 1)), 1)))',

    # 019: 收益率的 ts_mean - ts_std 差值
    'gtja_019': 'sub(ts_mean(sub(close, delay(close, 1)), 20), ts_std(sub(close, delay(close, 1)), 20))',

    # 020: close 与 vwap 偏离排名
    'gtja_020': 'cs_rank(sub(close, vwap))',

    # 021: SMA 趋势偏离
    'gtja_021': 'cs_rank(sub(close, sma(close, 20, 1)))',

    # 022: (close - sma) / sma 标准化偏离
    'gtja_022': 'div(sub(close, sma(close, 20, 1)), sma(close, 20, 1))',

    # 023: close / sma(close, 20) 比率
    'gtja_023': 'div(close, sma(close, 20, 1))',

    # 024: 20 日动量相对波动率
    'gtja_024': 'div(sub(close, delay(close, 20)), ts_std(sub(close, delay(close, 1)), 20))',

    # 025: 5 日动量 × correlation(close, volume, 20)
    'gtja_025': 'mul(sub(close, delay(close, 5)), correlation(close, volume, 20))',

    # 026: 20 日 ts_mean(close) 偏离
    'gtja_026': 'sub(close, ts_mean(close, 20))',

    # 027: 5 日 ts_max(close) 突破
    'gtja_027': 'sub(close, ts_max(close, 5))',

    # 028: (ts_max - ts_min) / ts_mean 标准化振幅
    'gtja_028': 'div(sub(ts_max(close, 20), ts_min(close, 20)), ts_mean(close, 20))',

    # 029: 收益率 × ts_rank(volume, 20)
    'gtja_029': 'mul(sub(close, delay(close, 1)), ts_rank(volume, 20))',

    # 030: 10 日 ts_rank(close) 与 ts_rank(volume) 差值
    'gtja_030': 'sub(ts_rank(close, 10), ts_rank(volume, 10))',

    # ═══════════════════════════════════════════════════════════════════
    #  反转类 (gtja_031 ~ gtja_060)
    # ═══════════════════════════════════════════════════════════════════

    # 031: 负的 5 日 ts_rank  — 经典短期反转
    'gtja_031': 'neg(ts_rank(sub(close, delay(close, 1)), 5))',

    # 032: 负的 20 日 momentum × 5 日 volume
    'gtja_032': 'neg(mul(sub(close, delay(close, 20)), ts_rank(volume, 5)))',

    # 033: 负的 (close - vwap) × ts_rank(volume)
    'gtja_033': 'neg(mul(sub(close, vwap), ts_rank(volume, 5)))',

    # 034: 负的 decay_linear 收益率
    'gtja_034': 'neg(decay_linear(sub(close, delay(close, 1)), 10))',

    # 035: ts_zscore 反转 — 偏离均值后回归
    'gtja_035': 'neg(ts_zscore(sub(close, delay(close, 1)), 20))',

    # 036: ts_rank 反转 — 多种窗口
    'gtja_036': 'neg(ts_rank(sub(close, delay(close, 1)), 30))',

    # 037: 负的 ts_rank(close, 45) — 中期反转
    'gtja_037': 'neg(ts_rank(close, 45))',

    # 038: 负的 ts_rank(close, 60) — 长期反转
    'gtja_038': 'neg(ts_rank(close, 60))',

    # 039: 反转 × 成交量放大
    'gtja_039': 'neg(mul(sub(close, delay(close, 5)), div(volume, ts_mean(volume, 20))))',

    # 040: cs_rank(-returns) × 负相关 — 双重反转
    'gtja_040': 'neg(cs_rank(mul(sub(close, delay(close, 1)), div(volume, delay(volume, 1)))))',

    # 041: (close-low) 相对 (high-low) 比例 — 下影线
    'gtja_041': 'div(sub(close, low), add(sub(high, low), 0.001))',

    # 042: RSI 风格 — 14日上涨力度
    'gtja_042': 'div(ts_mean(ifelse(sub(close, delay(close, 1)), sub(close, delay(close, 1)), 0.0), 14), add(ts_mean(abs(sub(close, delay(close, 1))), 14), 0.001))',

    # 043: 负的 (high - delay(high, 10)) — 高位回落
    'gtja_043': 'neg(sub(high, delay(high, 10)))',

    # 044: (close - open) × volume 的 cs_rank — 日间反转信号
    'gtja_044': 'neg(cs_rank(mul(sub(close, open), volume)))',

    # 045: 负的 correlation(ts_rank(close, 5), ts_rank(volume, 5), 10)
    'gtja_045': 'neg(correlation(ts_rank(close, 5), ts_rank(volume, 5), 10))',

    # 046: 负的 covariance(close, volume, 10)
    'gtja_046': 'neg(covariance(close, volume, 10))',

    # 047: (close-open)/open 与 volume 相关性反转
    'gtja_047': 'neg(ts_rank(mul(div(sub(close, open), add(open, 0.001)), volume), 5))',

    # 048: ts_argmax 突破反转 — 从高点到现在的天数
    'gtja_048': 'cs_rank(ts_argmax(close, 60))',

    # 049: ts_argmin 触底反弹 — 从低点到现在的天数
    'gtja_049': 'neg(cs_rank(ts_argmin(close, 60)))',

    # 050: (close - ts_min) / (ts_max - ts_min) × -1  — 反向随机
    'gtja_050': 'neg(div(sub(close, ts_min(close, 20)), add(sub(ts_max(close, 20), ts_min(close, 20)), 0.001)))',

    # 051: 负的 ts_rank(returns, 20) — 收益率排名反转
    'gtja_051': 'neg(ts_rank(sub(close, delay(close, 1)), 20))',

    # 052: cs_rank(neg(close-delay(close,1))) × ts_rank(volume, 20)
    'gtja_052': 'mul(cs_rank(neg(sub(close, delay(close, 1)))), ts_rank(volume, 20))',

    # 053: (open - delay(close, 1)) / delay(close, 1) — 隔夜跳空
    'gtja_053': 'div(sub(open, delay(close, 1)), add(delay(close, 1), 0.001))',

    # 054: correlation(neg(sub(close, delay(close, 1))), ts_rank(volume, 5), 20)
    'gtja_054': 'correlation(neg(sub(close, delay(close, 1))), ts_rank(volume, 5), 20)',

    # 055: 60日动量反转
    'gtja_055': 'neg(ts_rank(sub(close, delay(close, 60)), 10))',

    # 056: 负的 (close - ts_max(close, 20)) — 超买回调
    'gtja_056': 'neg(sub(close, ts_max(close, 20)))',

    # 057: 负的 ts_rank(div(close, sma(close, 20, 1)), 10) — 均值偏离反转
    'gtja_057': 'neg(ts_rank(div(close, sma(close, 20, 1)), 10))',

    # 058: ifelse 反转 — 连跌后买入
    'gtja_058': 'neg(ifelse(sub(delay(close, 1), close), sub(close, delay(close, 1)), 0.0))',

    # 059: 负的 (5日return/vol) × ts_rank(volume)
    'gtja_059': 'neg(mul(div(sub(close, delay(close, 5)), add(ts_std(sub(close, delay(close, 1)), 20), 0.001)), ts_rank(volume, 20)))',

    # 060: cs_rank(neg(trend)) × correlation(close, volume, 10)
    'gtja_060': 'mul(cs_rank(neg(sub(close, delay(close, 10)))), correlation(close, volume, 10))',

    # ═══════════════════════════════════════════════════════════════════
    #  波动率类 (gtja_061 ~ gtja_080)
    # ═══════════════════════════════════════════════════════════════════

    # 061: 负的 ts_std(returns, 20) — 低波动率偏好
    'gtja_061': 'neg(ts_std(sub(close, delay(close, 1)), 20))',

    # 062: 负的 ts_std(returns, 10) — 短期低波
    'gtja_062': 'neg(ts_std(sub(close, delay(close, 1)), 10))',

    # 063: 负的 ts_std(returns, 60) — 长期低波
    'gtja_063': 'neg(ts_std(sub(close, delay(close, 1)), 60))',

    # 064: 负的 ts_zscore(returns, 20) — Z-score 低波
    'gtja_064': 'neg(ts_zscore(sub(close, delay(close, 1)), 20))',

    # 065: 负的 ts_zscore(returns, 10)
    'gtja_065': 'neg(ts_zscore(sub(close, delay(close, 1)), 10))',

    # 066: 负的 ts_zscore(returns, 60)
    'gtja_066': 'neg(ts_zscore(sub(close, delay(close, 1)), 60))',

    # 067: 下行波动率 — 只统计负收益的 std
    'gtja_067': 'neg(ts_std(ifelse(sub(close, delay(close, 1)), 0.0, sub(close, delay(close, 1))), 20))',

    # 068: (high-low)/close 振幅 — 低振幅偏好
    'gtja_068': 'neg(div(sub(high, low), add(close, 0.001)))',

    # 069: ts_mean(high-low) / close — 平均振幅
    'gtja_069': 'neg(div(ts_mean(sub(high, low), 20), close))',

    # 070: 波动率变化 — 标准差的一阶差分
    'gtja_070': 'sub(ts_std(sub(close, delay(close, 1)), 10), delay(ts_std(sub(close, delay(close, 1)), 10), 1))',

    # 071: 负的 covariance(returns, 前一日returns, 20) — 波动率持续性
    'gtja_071': 'neg(covariance(sub(close, delay(close, 1)), delay(sub(close, delay(close, 1)), 1), 20))',

    # 072: 负的 (high-low) / ts_mean(close, 20) — 相对振幅
    'gtja_072': 'neg(div(sub(high, low), add(ts_mean(close, 20), 0.001)))',

    # 073: 负的 ts_std(volume, 20) / ts_mean(volume, 20) — 量波动率
    'gtja_073': 'neg(div(ts_std(volume, 20), add(ts_mean(volume, 20), 0.001)))',

    # 074: 波动率 × 成交量背离
    'gtja_074': 'sub(neg(ts_std(sub(close, delay(close, 1)), 20)), ts_rank(volume, 20))',

    # 075: 负的 ts_rank(ts_std(returns, 20), 20) — 波动的排名
    'gtja_075': 'neg(ts_rank(ts_std(sub(close, delay(close, 1)), 20), 20))',

    # 076: regression beta (returns, market) — 简化用 ts_mean 近似Beta
    'gtja_076': 'neg(regbeta(sub(close, delay(close, 1)), sub(close, delay(close, 1)), 60))',

    # 077: (close - ts_min) / (ts_max - ts_min) 波动率位置
    'gtja_077': 'neg(ts_std(div(sub(close, ts_min(close, 20)), add(sub(ts_max(close, 20), ts_min(close, 20)), 0.001)), 10))',

    # 078: 负的 SMA 波动
    'gtja_078': 'neg(ts_std(sub(close, sma(close, 10, 1)), 20))',

    # 079: ts_sum(abs(returns), 20) 的负排名 — 低总波动
    'gtja_079': 'neg(ts_rank(ts_sum(abs(sub(close, delay(close, 1))), 20), 20))',

    # 080: neg(ts_std(close)/ts_mean(close)) — 变异系数
    'gtja_080': 'neg(div(ts_std(close, 20), add(ts_mean(close, 20), 0.001)))',

    # ═══════════════════════════════════════════════════════════════════
    #  成交量类 (gtja_081 ~ gtja_100)
    # ═══════════════════════════════════════════════════════════════════

    # 081: 成交量比 — volume/20日均量
    'gtja_081': 'div(volume, add(ts_mean(volume, 20), 0.001))',

    # 082: volume / ts_mean(volume, 5) — 短期量比
    'gtja_082': 'div(volume, add(ts_mean(volume, 5), 0.001))',

    # 083: volume / ts_mean(volume, 60) — 长期量比
    'gtja_083': 'div(volume, add(ts_mean(volume, 60), 0.001))',

    # 084: ts_rank(volume, 5) — 短期量能排名
    'gtja_084': 'ts_rank(volume, 5)',

    # 085: ts_rank(volume, 20) — 中期量能排名
    'gtja_085': 'ts_rank(volume, 20)',

    # 086: sub(volume, delay(volume, 1)) — 成交量一日变化
    'gtja_086': 'sub(volume, delay(volume, 1))',

    # 087: sub(volume, delay(volume, 5)) — 成交量5日变化
    'gtja_087': 'sub(volume, delay(volume, 5))',

    # 088: ts_sum(volume, 5) / ts_sum(volume, 20) — 近期量占比
    'gtja_088': 'div(ts_sum(volume, 5), add(ts_sum(volume, 20), 0.001))',

    # 089: ts_sum(volume, 10) / ts_sum(volume, 60) — 中期量占比
    'gtja_089': 'div(ts_sum(volume, 10), add(ts_sum(volume, 60), 0.001))',

    # 090: ts_mean(volume, 5) - ts_mean(volume, 20) — 量能均线交叉
    'gtja_090': 'sub(ts_mean(volume, 5), ts_mean(volume, 20))',

    # 091: ts_max(volume, 20) / volume — 量能峰值比
    'gtja_091': 'div(ts_max(volume, 20), add(volume, 0.001))',

    # 092: volume / ts_mean(volume, 20) × returns — 量价共振
    'gtja_092': 'mul(div(volume, add(ts_mean(volume, 20), 0.001)), sub(close, delay(close, 1)))',

    # 093: correlation(volume, close, 10) — 量价相关性
    'gtja_093': 'correlation(volume, close, 10)',

    # 094: correlation(volume, sub(close, delay(close, 1)), 10) — 量收益相关性
    'gtja_094': 'correlation(volume, sub(close, delay(close, 1)), 10)',

    # 095: ts_rank(sub(volume, delay(volume, 5)), 10) — 量变化排名
    'gtja_095': 'ts_rank(sub(volume, delay(volume, 5)), 10)',

    # 096: 成交量衰减加权
    'gtja_096': 'decay_linear(sub(volume, ts_mean(volume, 20)), 10)',

    # 097: ts_zscore(volume, 20) — 成交量 Z-score
    'gtja_097': 'ts_zscore(volume, 20)',

    # 098: volume × abs(returns) — 成交金额（量×价变）
    'gtja_098': 'mul(volume, abs(sub(close, delay(close, 1))))',

    # 099: 缩量 — 负的成交量排名
    'gtja_099': 'neg(ts_rank(volume, 10))',

    # 100: ts_std(volume, 20) — 成交量波动
    'gtja_100': 'ts_std(volume, 20)',

    # ═══════════════════════════════════════════════════════════════════
    #  资金流类 (gtja_101 ~ gtja_130)
    # ═══════════════════════════════════════════════════════════════════

    # 101: amount / close ≈ volume — 资金流代理
    'gtja_101': 'cs_rank(volume)',

    # 102: (close-open) × volume / abs(close-open) — 资金流向
    #      用 close-open 符号判断买卖方向，volume 做权重
    'gtja_102': 'mul(sign(sub(close, open)), volume)',

    # 103: amount = volume × vwap（vwap ≈ close 当 volume ⊥ price）
    'gtja_103': 'cs_rank(mul(volume, vwap))',

    # 104: (close - open) × volume / open — 资金流比率
    'gtja_104': 'div(mul(sub(close, open), volume), add(open, 0.001))',

    # 105: ts_sum(mul(sub(close, open), volume), 5) — 5日累计资金流
    'gtja_105': 'ts_sum(mul(sign(sub(close, open)), volume), 5)',

    # 106: ts_sum(mul(sub(close, open), volume), 20) — 20日累计资金流
    'gtja_106': 'ts_sum(mul(sign(sub(close, open)), volume), 20)',

    # 107: 资金流加速度
    'gtja_107': 'sub(ts_sum(mul(sign(sub(close, open)), volume), 5), delay(ts_sum(mul(sign(sub(close, open)), volume), 5), 5))',

    # 108: correlation(amount_flow, returns, 20)
    'gtja_108': 'correlation(mul(sign(sub(close, open)), volume), sub(close, delay(close, 1)), 20)',

    # 109: ts_rank(mul(sub(close, open), volume), 10)
    'gtja_109': 'ts_rank(mul(sub(close, open), volume), 10)',

    # 110: 资金流 / 总流通量 — 标准化资金流
    'gtja_110': 'div(mul(sub(close, open), volume), add(ts_mean(mul(close, volume), 20), 0.001))',

    # 111: ts_sum(volume, 5) / ts_sum(volume, 20) × returns — 放量上涨
    'gtja_111': 'mul(div(ts_sum(volume, 5), add(ts_sum(volume, 20), 0.001)), sub(close, delay(close, 1)))',

    # 112: ts_mean(mul(volume, returns), 5) — 平均资金流强度
    'gtja_112': 'ts_mean(mul(volume, sub(close, delay(close, 1))), 5)',

    # 113: 资金流 × 波动率
    'gtja_113': 'div(ts_mean(mul(volume, sub(close, delay(close, 1))), 5), add(ts_std(sub(close, delay(close, 1)), 20), 0.001))',

    # 114: ts_rank(mul(volume, vwap), 20) — 成交额排名
    'gtja_114': 'ts_rank(mul(volume, vwap), 20)',

    # 115: (high*low)^0.5 / close — 价格位置 vs 中期价
    'gtja_115': 'div(sqrt(mul(high, low)), close)',

    # 116: (high - close) / close — 上影线比例
    'gtja_116': 'neg(div(sub(high, close), add(close, 0.001)))',

    # 117: (close - low) / close — 下影线比例
    'gtja_117': 'div(sub(close, low), add(close, 0.001))',

    # 118: (close - open) / (high - low) — K线实体比例
    'gtja_118': 'div(sub(close, open), add(sub(high, low), 0.001))',

    # 119: CCI 风格 — (close - ts_mean) / (0.015 × ts_mean_deviation)
    'gtja_119': 'div(sub(close, ts_mean(div(add(high, low), 2), 20)), add(mul(0.015, ts_mean(abs(sub(close, ts_mean(close, 20))), 20)), 0.001))',

    # 120: RSI 简化 — 上涨均值/总变动均值
    'gtja_120': 'sub(div(ts_mean(ifelse(sub(close, delay(close, 1)), sub(close, delay(close, 1)), 0.0), 14), add(ts_mean(abs(sub(close, delay(close, 1))), 14), 0.001)), 0.5)',

    # 121: 威廉指标 — (ts_max(high, 14) - close) / (ts_max - ts_min)
    'gtja_121': 'neg(div(sub(ts_max(high, 14), close), add(sub(ts_max(high, 14), ts_min(low, 14)), 0.001)))',

    # 122: 资金流 10 日衰减加权
    'gtja_122': 'decay_linear(mul(sign(sub(close, open)), volume), 10)',

    # 123: IFELSE — 上涨放量则为正，缩量则为负
    'gtja_123': 'ifelse(sub(close, delay(close, 1)), mul(sub(volume, ts_mean(volume, 20)), sub(close, delay(close, 1))), neg(mul(sub(volume, ts_mean(volume, 20)), abs(sub(close, delay(close, 1))))))',

    # 124: 量价背离 — 价涨量缩
    'gtja_124': 'sub(neg(ts_rank(volume, 10)), ts_rank(sub(close, delay(close, 1)), 10))',

    # 125: ts_rank(volume, 10) × ts_rank(returns, 5)
    'gtja_125': 'mul(ts_rank(volume, 10), ts_rank(sub(close, delay(close, 1)), 5))',

    # 126: SMA 成交量趋势
    'gtja_126': 'sub(sma(volume, 5, 1), sma(volume, 20, 1))',

    # 127: 资金流 × 振幅比
    'gtja_127': 'mul(mul(sign(sub(close, open)), volume), div(sub(high, low), add(close, 0.001)))',

    # 128: ts_argmax(volume, 20) — 距放量日天数
    'gtja_128': 'cs_rank(ts_argmax(volume, 20))',

    # 129: ts_std(mul(volume, returns), 20) — 资金流波动
    'gtja_129': 'ts_std(mul(volume, sub(close, delay(close, 1))), 20)',

    # 130: decay_linear(close, 5) / decay_linear(close, 20) — 短期/长期趋势比
    'gtja_130': 'div(decay_linear(close, 5), add(decay_linear(close, 20), 0.001))',

    # ═══════════════════════════════════════════════════════════════════
    #  基本面类 (gtja_131 ~ gtja_160)
    #  注：基本面因子依赖 EP/BP/ROE 等外源数据，表达式引擎当前不直接支持。
    #      以下提供可以用价格数据代理的因子，其余留空待数据扩展。
    # ═══════════════════════════════════════════════════════════════════

    # 131: 市值 proxy — ts_mean(close × volume, 20)  日均成交额
    'gtja_131': 'neg(ts_mean(mul(close, volume), 20))',

    # 132: 小市值偏好 — 负的 ts_rank(close × volume)
    'gtja_132': 'neg(cs_rank(mul(close, volume)))',

    # 133: 价格×成交量 排名 × 收益率
    'gtja_133': 'mul(neg(cs_rank(mul(close, volume))), sub(close, delay(close, 1)))',

    # 134: 价格中位数偏离
    'gtja_134': 'sub(close, ts_mean(close, 60))',

    # 135: 相对强度 — close / ts_max(close, 252)
    'gtja_135': 'div(close, add(ts_max(close, 252), 0.001))',

    # 136: 过去 60 日收益率 — 长期表现
    'gtja_136': 'sub(close, delay(close, 60))',

    # 137: 过去 120 日收益率 — 超长期表现
    'gtja_137': 'sub(close, delay(close, 120))',

    # 138: ts_rank(close/ts_max(close,252), 20) — 距高点位置排名
    'gtja_138': 'ts_rank(div(close, add(ts_max(close, 252), 0.001)), 20)',

    # 139: ts_argmax(close, 252) — 距年内最高点天数
    'gtja_139': 'cs_rank(ts_argmax(close, 252))',

    # 140: 长期反转 — neg(returns_252)
    'gtja_140': 'neg(sub(close, delay(close, 252)))',

    # 141-160: 外源数据因子（EP/BP/ROE/CFP/SP/毛利率等）
    # 这些因子依赖基本面数据（PE, PB, ROE, 营收增长率等），
    # 表达式引擎当前仅支持 OHLCV 数据，待 v3 扩展数据层后补充。
    # 预留 ID 命名空间，确保编号连续性。

    # ═══════════════════════════════════════════════════════════════════
    #  技术指标类 (gtja_161 ~ gtja_191)
    # ═══════════════════════════════════════════════════════════════════

    # 161: MACD 快慢线差值 — EMA12 - EMA26（用 SMA 近似）
    'gtja_161': 'sub(sma(close, 12, 1), sma(close, 26, 1))',

    # 162: MACD 信号线 — SMA(MACD, 9)
    'gtja_162': 'sma(sub(sma(close, 12, 1), sma(close, 26, 1)), 9, 1)',

    # 163: MACD 柱 — MACD 线 - 信号线
    'gtja_163': 'sub(sub(sma(close, 12, 1), sma(close, 26, 1)), sma(sub(sma(close, 12, 1), sma(close, 26, 1)), 9, 1))',

    # 164: 布林带宽度 — 2 × ts_std(close, 20) / sma(close, 20)
    'gtja_164': 'div(mul(2, ts_std(close, 20)), add(sma(close, 20, 1), 0.001))',

    # 165: 布林带 %B — (close - lower) / (upper - lower)
    'gtja_165': 'div(sub(close, sub(sma(close, 20, 1), mul(2, ts_std(close, 20)))), add(mul(4, ts_std(close, 20)), 0.001))',

    # 166: 布林带上轨突破 — close - upper_band
    'gtja_166': 'sub(close, add(sma(close, 20, 1), mul(2, ts_std(close, 20))))',

    # 167: 布林带下轨 — close - lower_band
    'gtja_167': 'sub(close, sub(sma(close, 20, 1), mul(2, ts_std(close, 20))))',

    # 168: (close - sma(20)) / ts_std(20) — Z-score 偏离
    'gtja_168': 'div(sub(close, sma(close, 20, 1)), add(ts_std(close, 20), 0.001))',

    # 169: EMA5 - EMA10 金叉死叉
    'gtja_169': 'sub(sma(close, 5, 1), sma(close, 10, 1))',

    # 170: EMA5 - EMA20
    'gtja_170': 'sub(sma(close, 5, 1), sma(close, 20, 1))',

    # 171: EMA10 - EMA30
    'gtja_171': 'sub(sma(close, 10, 1), sma(close, 30, 1))',

    # 172: EMA5 - EMA60 长短期趋势
    'gtja_172': 'sub(sma(close, 5, 1), sma(close, 60, 1))',

    # 173: 三重均线 — SMA5>SMA10>SMA20 趋势确认
    'gtja_173': 'add(sub(sma(close, 5, 1), sma(close, 10, 1)), sub(sma(close, 10, 1), sma(close, 20, 1)))',

    # 174: (close - open) / open — 日内收益率
    'gtja_174': 'div(sub(close, open), add(open, 0.001))',

    # 175: (high - open) / open — 日内最高涨幅
    'gtja_175': 'div(sub(high, open), add(open, 0.001))',

    # 176: (open - low) / open — 日内最大跌幅（取负）
    'gtja_176': 'neg(div(sub(open, low), add(open, 0.001)))',

    # 177: ts_mean(high-low, 5) / close — 5日平均波幅
    'gtja_177': 'div(ts_mean(sub(high, low), 5), close)',

    # 178: ts_mean(high-low, 20) / close — 20日平均波幅
    'gtja_178': 'div(ts_mean(sub(high, low), 20), close)',

    # 179: (close - vwap) / vwap — 收盘价偏离均价
    'gtja_179': 'div(sub(close, vwap), add(vwap, 0.001))',

    # 180: 成交量加权收盘价 — vwap 的 cs_rank
    'gtja_180': 'cs_rank(vwap)',

    # 181: ifelse(returns>0, volume, neg(volume)) — 涨量正/跌量负
    'gtja_181': 'ifelse(sub(close, delay(close, 1)), volume, neg(volume))',

    # 182: ts_rank(close, 10) - ts_rank(close, 60) — 短期vs长期排名差
    'gtja_182': 'sub(ts_rank(close, 10), ts_rank(close, 60))',

    # 183: ts_rank(close, 10) - ts_rank(close, 20)
    'gtja_183': 'sub(ts_rank(close, 10), ts_rank(close, 20))',

    # 184: SMA 交叉 × 成交量 — 金叉放量
    'gtja_184': 'mul(sub(sma(close, 5, 1), sma(close, 20, 1)), div(volume, add(ts_mean(volume, 20), 0.001)))',

    # 185: OBV 风格 — 成交量 × 价格变化方向
    'gtja_185': 'ts_sum(mul(sign(sub(close, delay(close, 1))), volume), 20)',

    # 186: ts_corr(sma(close,5), sma(close,20), 30) — 均线趋势一致性
    'gtja_186': 'correlation(sma(close, 5, 1), sma(close, 20, 1), 30)',

    # 187: 连涨下跌信号 — ifelse(returns>0, returns, neg(returns)) × volume
    'gtja_187': 'mul(ifelse(sub(close, delay(close, 1)), sub(close, delay(close, 1)), neg(sub(close, delay(close, 1)))), div(volume, add(ts_mean(volume, 10), 0.001)))',

    # 188: ts_rank(sma(close, 5), 20) — 短期趋势排名
    'gtja_188': 'ts_rank(sma(close, 5, 1), 20)',

    # 189: 多空信号 — cs_rank(close) - cs_rank(returns)
    'gtja_189': 'sub(cs_rank(close), cs_rank(sub(close, delay(close, 1))))',

    # 190: ts_rank(ts_std(close, 10), 20) — 波动率的趋势
    'gtja_190': 'ts_rank(ts_std(close, 10), 20)',

    # 191: correlation(cs_rank(close), cs_rank(volume), 20) — 量价排名相关
    'gtja_191': 'correlation(cs_rank(close), cs_rank(volume), 20)',
}


# ============================================================================
# 因子分类映射
# ============================================================================

ALPHA191_CATEGORIES: Dict[str, FactorCategory] = {}

# 动量类
for i in range(1, 31):
    ALPHA191_CATEGORIES[f'gtja_{i:03d}'] = FactorCategory.MOMENTUM

# 反转类
for i in range(31, 61):
    ALPHA191_CATEGORIES[f'gtja_{i:03d}'] = FactorCategory.REVERSAL

# 波动率类
for i in range(61, 81):
    ALPHA191_CATEGORIES[f'gtja_{i:03d}'] = FactorCategory.VOLATILITY

# 成交量/量价类
for i in range(81, 101):
    ALPHA191_CATEGORIES[f'gtja_{i:03d}'] = FactorCategory.VOLUME

# 资金流类
for i in range(101, 131):
    ALPHA191_CATEGORIES[f'gtja_{i:03d}'] = FactorCategory.VOLUME

# 基本面类（VALUE ≈ 估值/规模相关，最接近基本面）
for i in range(131, 161):
    ALPHA191_CATEGORIES[f'gtja_{i:03d}'] = FactorCategory.VALUE

# 技术指标类
for i in range(161, 192):
    ALPHA191_CATEGORIES[f'gtja_{i:03d}'] = FactorCategory.TECHNICAL


# ============================================================================
# 向后兼容别名（alpha191 原有接口）
# ============================================================================

from .expression_factor import (
    ExpressionFactor,
    expression_factor,
    AlphaExplorer,
    AlphaResult,
)
from .alpha101 import (
    ALPHA101,
    ALPHA101_CATEGORIES,
)
ALPHA_FORMULAS = ALPHA101
ALPHA_CATEGORIES = ALPHA101_CATEGORIES


__all__ = [
    'ALPHA101', 'ALPHA101_CATEGORIES',
    'ALPHA_FORMULAS', 'ALPHA_CATEGORIES',
    'ALPHA191', 'ALPHA191_CATEGORIES',
    'ExpressionFactor', 'expression_factor',
    'AlphaExplorer', 'AlphaResult',
]
