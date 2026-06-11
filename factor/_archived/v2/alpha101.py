"""
alpha101.py — WorldQuant 101 Formulaic Alphas（v2 表达式语法）

基于 Zura Kakushadze 论文 "101 Formulaic Alphas" (2015)，
全部翻译为 factor v2 表达式引擎的小写 S-表达式语法。

翻译规则（GTJA/WQ → v2）：
  RANK(x)        → cs_rank(x)
  TSRANK(x, d)   → ts_rank(x, d)
  CORR(x, y, d)  → correlation(x, y, d)
  COVIANCE(x,y,d)→ covariance(x, y, d)
  DELTA(x, d)    → sub(x, delay(x, d))
  SUM(x, d)      → ts_sum(x, d)
  MEAN(x, d)     → ts_mean(x, d)
  STD(x, d)      → ts_std(x, d)
  MAX(x, d)      → ts_max(x, d)
  MIN(x, d)      → ts_min(x, d)
  SMA(x, n, m)   → sma(x, n, m)
  DECAYLINEAR    → decay_linear(x, d)
  SIGNEDPOWER    → signed_power(x, a)
  TSARGMAX(x,d)  → ts_argmax(x, d)
  TSARGMIN(x,d)  → ts_argmin(x, d)
  REGBETA(x,y,d) → regbeta(x, y, d)
  IFELSE(c,a,b)  → ifelse(c, a, b)
  RETURNS        → 衍生字段 'returns'（ExpressionFactor 自动注入）
  VWAP           → 衍生字段 'vwap'
  ADV(d)         → ts_mean(volume, d)

已知简化（不影响排序/相关性）：
  - scale() 省略：传递标量缩放，cs_rank 中不影响排名
  - indneutralize() 省略：行业中性化暂未实现
  - 浮点参数取整：原始 GP 优化出的非整周期取最近整数

使用方式：
  >>> from factor.v2.alpha101 import ALPHA101, ALPHA101_CATEGORIES
  >>> len(ALPHA101)  # 101
  >>> ALPHA101['alpha042']  # 获取公式字符串

[新增] 2026-05-30  从 alpha191.py 拆出，独立为纯数据模块
=============================================================================
"""

from typing import Dict
from .base import FactorCategory


# ============================================================================
# 101 Alpha 公式定义
# ============================================================================

ALPHA101: Dict[str, str] = {
    # ── Alpha #001 ──
    # 原始: (rank(ts_argmax(signedpower(((returns<0)?stddev(returns,20):close), 2), 5)) - 0.5)
    'alpha001': 'sub(cs_rank(ts_argmax(signed_power(ifelse(neg(returns), ts_std(returns, 20), close), 2), 5)), 0.5)',

    # ── Alpha #002 ──
    # 原始: -correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)
    'alpha002': 'neg(correlation(cs_rank(sub(log(volume), delay(log(volume), 2))), cs_rank(div(sub(close, open), open)), 6))',

    # ── Alpha #003 ──
    # 原始: -correlation(rank(open), rank(volume), 10)
    'alpha003': 'neg(correlation(cs_rank(open), cs_rank(volume), 10))',

    # ── Alpha #004 ──
    # 原始: -ts_rank(rank(low), 9)
    'alpha004': 'neg(ts_rank(cs_rank(low), 9))',

    # ── Alpha #005 ──
    # 原始: rank((open - ts_mean(open, 10))) * (-abs(delta(close, 1)))
    'alpha005': 'mul(cs_rank(sub(open, ts_mean(open, 10))), neg(abs(sub(close, delay(close, 1)))))',

    # ── Alpha #006 ──
    # 原始: -correlation(open, volume, 10)
    'alpha006': 'neg(correlation(open, volume, 10))',

    # ── Alpha #007 ──
    # 原始: ((adv20 < volume) ? ((-ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : -1)
    'alpha007': 'ifelse(sub(volume, ts_mean(volume, 20)), mul(neg(ts_rank(abs(sub(close, delay(close, 7))), 60)), sign(sub(close, delay(close, 7)))), -1)',

    # ── Alpha #008 ──
    # 原始: -rank(((ts_sum(open, 5) * ts_sum(returns, 5)) - delay((ts_sum(open, 5) * ts_sum(returns, 5)), 10)))
    'alpha008': 'neg(cs_rank(sub(mul(ts_sum(open, 5), ts_sum(returns, 5)), delay(mul(ts_sum(open, 5), ts_sum(returns, 5)), 10))))',

    # ── Alpha #009 ──
    # 原始: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1) : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : -delta(close, 1)))
    'alpha009': 'ifelse(ts_min(sub(close, delay(close, 1)), 5), sub(close, delay(close, 1)), ifelse(neg(ts_max(sub(close, delay(close, 1)), 5)), sub(close, delay(close, 1)), neg(sub(close, delay(close, 1)))))',

    # ── Alpha #010 ──
    # 原始: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1): ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : -delta(close, 1))))
    'alpha010': 'cs_rank(ifelse(ts_min(sub(close, delay(close, 1)), 4), sub(close, delay(close, 1)), ifelse(neg(ts_max(sub(close, delay(close, 1)), 4)), sub(close, delay(close, 1)), neg(sub(close, delay(close, 1))))))',

    # ── Alpha #011 ──
    # 原始: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    'alpha011': 'mul(add(cs_rank(ts_max(sub(vwap, close), 3)), cs_rank(ts_min(sub(vwap, close), 3))), cs_rank(sub(volume, delay(volume, 3))))',

    # ── Alpha #012 ──
    # 原始: sign(delta(volume, 1)) * (-delta(close, 1))
    'alpha012': 'mul(sign(sub(volume, delay(volume, 1))), neg(sub(close, delay(close, 1))))',

    # ── Alpha #013 ──
    # 原始: -rank(covariance(rank(close), rank(volume), 5))
    'alpha013': 'neg(cs_rank(covariance(cs_rank(close), cs_rank(volume), 5)))',

    # ── Alpha #014 ──
    # 原始: ((-rank(delta(returns, 3))) * correlation(open, volume, 10))
    'alpha014': 'mul(neg(cs_rank(sub(returns, delay(returns, 3)))), correlation(open, volume, 10))',

    # ── Alpha #015 ──
    # 原始: -ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3)
    'alpha015': 'neg(ts_sum(cs_rank(correlation(cs_rank(high), cs_rank(volume), 3)), 3))',

    # ── Alpha #016 ──
    # 原始: -rank(covariance(rank(high), rank(volume), 5))
    'alpha016': 'neg(cs_rank(covariance(cs_rank(high), cs_rank(volume), 5)))',

    # ── Alpha #017 ──
    # 原始: (((-rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
    'alpha017': 'mul(mul(neg(cs_rank(ts_rank(close, 10))), cs_rank(sub(sub(close, delay(close, 1)), delay(sub(close, delay(close, 1)), 1)))), cs_rank(ts_rank(div(volume, ts_mean(volume, 20)), 5)))',

    # ── Alpha #018 ──
    # 原始: -rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))
    'alpha018': 'neg(cs_rank(add(add(ts_std(abs(sub(close, open)), 5), sub(close, open)), correlation(close, open, 10))))',

    # ── Alpha #019 ──
    # 原始: ((-sign((close - delay(close, 7)) + delta(close, 7))) * (1 + rank(1 + ts_sum(returns, 250))))
    'alpha019': 'mul(neg(sign(add(sub(close, delay(close, 7)), sub(close, delay(close, 7))))), add(1, cs_rank(add(1, ts_sum(returns, 250)))))',

    # ── Alpha #020 ──
    # 原始: (((-rank(open - delay(high, 1))) * rank(open - delay(close, 1))) * rank(open - delay(low, 1)))
    'alpha020': 'mul(mul(neg(cs_rank(sub(open, delay(high, 1)))), cs_rank(sub(open, delay(close, 1)))), cs_rank(sub(open, delay(low, 1))))',

    # ── Alpha #021 ──
    # 原始: (((ts_sum(close, 8) / 8) + ts_std(close, 8)) < (ts_sum(close, 2) / 2)) ? -1 : 1
    'alpha021': 'ifelse(sub(add(div(ts_sum(close, 8), 8), ts_std(close, 8)), div(ts_sum(close, 2), 2)), -1, 1)',

    # ── Alpha #022 ──
    # 原始: -delta(correlation(high, volume, 5), 5) * rank(ts_std(close, 20))
    'alpha022': 'mul(neg(sub(correlation(high, volume, 5), delay(correlation(high, volume, 5), 5))), cs_rank(ts_std(close, 20)))',

    # ── Alpha #023 ──
    # 原始: (((ts_sum(high, 20) / 20) < high) ? (-delta(high, 2)) : 0)
    'alpha023': 'ifelse(sub(div(ts_sum(high, 20), 20), high), neg(sub(high, delay(high, 2))), 0)',

    # ── Alpha #024 ──
    # 原始: (((delta(ts_sum(close, 100) / 100, 100) / delay(close, 100)) <= 0.05) ? (-close / (ts_sum(close, 100)/100 - close)) : (-log(close)))
    'alpha024': 'ifelse(sub(0.05, div(sub(div(ts_sum(close, 100), 100), delay(div(ts_sum(close, 100), 100), 100)), delay(close, 100))), div(neg(close), sub(div(ts_sum(close, 100), 100), close)), neg(log(close)))',

    # ── Alpha #025 ──
    # 原始: rank((-returns * adv20 * vwap * (high - close)))
    'alpha025': 'cs_rank(mul(mul(mul(neg(returns), ts_mean(volume, 20)), vwap), sub(high, close)))',

    # ── Alpha #026 ──
    # 原始: -ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    'alpha026': 'neg(ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3))',

    # ── Alpha #027 ──
    # 原始: ((0.5 < rank(correlation(rank(volume), rank(vwap), 6))) ? -1 : 1)
    'alpha027': 'ifelse(sub(cs_rank(correlation(cs_rank(volume), cs_rank(vwap), 6)), 0.5), 1, -1)',

    # ── Alpha #028 ──
    # 原始: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    'alpha028': 'sub(add(correlation(ts_mean(volume, 20), low, 5), div(add(high, low), 2)), close)',

    # ── Alpha #029 ──
    # 原始: (min(product(rank(rank(log(ts_sum(rank(rank((-rank(rank(delta(close, 1)))))), 15)), 1))), 1), 5) + ts_rank(delay(-returns, 6), 5))
    'alpha029': 'add(ts_min(cs_rank(log(ts_sum(cs_rank(neg(cs_rank(sub(close, delay(close, 1))))), 15))), 5), ts_rank(delay(neg(returns), 6), 5))',

    # ── Alpha #030 ──
    # 原始: (((1.0 - rank(((sign((close - delay(close, 1))) + sign(delay(close, 1) - delay(close, 2))) + sign(delay(close, 2) - delay(close, 3)))))) * ts_sum(volume, 5)) / ts_sum(volume, 20))
    'alpha030': 'div(mul(sub(1, cs_rank(add(add(sign(sub(close, delay(close, 1))), sign(sub(delay(close, 1), delay(close, 2)))), sign(sub(delay(close, 2), delay(close, 3)))))), ts_sum(volume, 5)), ts_sum(volume, 20))',

    # ── Alpha #031 ──
    # 原始: ((rank(rank(rank(decay_linear((-rank(rank(delta(close, 10)))), 10)))) + rank((-delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    'alpha031': 'add(add(cs_rank(decay_linear(neg(cs_rank(sub(close, delay(close, 10)))), 10)), cs_rank(neg(sub(close, delay(close, 3))))), sign(correlation(ts_mean(volume, 20), low, 12)))',

    # ── Alpha #032 ──
    # 原始: (scale(((ts_sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
    'alpha032': 'add(sub(div(ts_sum(close, 7), 7), close), mul(20, correlation(vwap, delay(close, 5), 230)))',

    # ── Alpha #033 ──
    # 原始: rank((-((1 - (open / close))^1)))
    'alpha033': 'cs_rank(neg(sub(1, div(open, close))))',

    # ── Alpha #034 ──
    # 原始: rank(((1 - rank((ts_std(returns, 2) / ts_std(returns, 5)))) + (1 - rank(delta(close, 1)))))
    'alpha034': 'cs_rank(add(sub(1, cs_rank(div(ts_std(returns, 2), ts_std(returns, 5)))), sub(1, cs_rank(sub(close, delay(close, 1))))))',

    # ── Alpha #035 ──
    # 原始: ((ts_rank(volume, 32) * (1 - ts_rank(((close + high) - low), 16))) * (1 - ts_rank(returns, 32)))
    'alpha035': 'mul(mul(ts_rank(volume, 32), sub(1, ts_rank(sub(add(close, high), low), 16))), sub(1, ts_rank(returns, 32)))',

    # ── Alpha #036 ──
    # 原始: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(ts_rank(delay((-returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank(((ts_sum(close, 200) / 200) - open) * (close - open))))
    'alpha036': 'add(add(add(add(mul(2.21, cs_rank(correlation(sub(close, open), delay(volume, 1), 15))), mul(0.7, cs_rank(sub(open, close)))), mul(0.73, cs_rank(ts_rank(delay(neg(returns), 6), 5)))), cs_rank(abs(correlation(vwap, ts_mean(volume, 20), 6)))), mul(0.6, cs_rank(mul(sub(div(ts_sum(close, 200), 200), open), sub(close, open)))))',

    # ── Alpha #037 ──
    # 原始: rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close))
    'alpha037': 'add(cs_rank(correlation(delay(sub(open, close), 1), close, 200)), cs_rank(sub(open, close)))',

    # ── Alpha #038 ──
    # 原始: ((-rank(ts_rank(close, 10))) * rank((close / open)))
    'alpha038': 'mul(neg(cs_rank(ts_rank(close, 10))), cs_rank(div(close, open)))',

    # ── Alpha #039 ──
    # 原始: ((-rank(delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9))))) * (1 + rank(ts_sum(returns, 250))))
    'alpha039': 'mul(neg(cs_rank(mul(sub(close, delay(close, 7)), sub(1, cs_rank(decay_linear(div(volume, ts_mean(volume, 20)), 9)))))), add(1, cs_rank(ts_sum(returns, 250))))',

    # ── Alpha #040 ──
    # 原始: ((-rank(ts_std(high, 10))) * correlation(high, volume, 10))
    'alpha040': 'mul(neg(cs_rank(ts_std(high, 10))), correlation(high, volume, 10))',

    # ── Alpha #041 ──
    # 原始: ((high * low)^0.5 - vwap)
    'alpha041': 'sub(sqrt(mul(high, low)), vwap)',

    # ── Alpha #042 ──
    # 原始: rank((vwap - close)) / rank((vwap + close))
    'alpha042': 'div(cs_rank(sub(vwap, close)), cs_rank(add(vwap, close)))',

    # ── Alpha #043 ──
    # 原始: (ts_rank(adv20, 20) * ts_rank(((-delta(close, 7))), 8))
    'alpha043': 'mul(ts_rank(ts_mean(volume, 20), 20), ts_rank(neg(sub(close, delay(close, 7))), 8))',

    # ── Alpha #044 ──
    # 原始: (-correlation(high, rank(volume), 5))
    'alpha044': 'neg(correlation(high, cs_rank(volume), 5))',

    # ── Alpha #045 ──
    # 原始: ((-rank(delta((((close * 0.6) + (open * 0.4))), 1))) * rank(correlation(vwap, adv20, 15)))
    'alpha045': 'mul(neg(cs_rank(sub(add(mul(close, 0.6), mul(open, 0.4)), delay(add(mul(close, 0.6), mul(open, 0.4)), 1)))), cs_rank(correlation(vwap, ts_mean(volume, 20), 15)))',

    # ── Alpha #046 ──
    # 原始: ((ts_mean(close, 3) + ts_mean(close, 6) + ts_mean(close, 12) + ts_mean(close, 24)) / (4 * close))
    'alpha046': 'div(add(add(add(ts_mean(close, 3), ts_mean(close, 6)), ts_mean(close, 12)), ts_mean(close, 24)), mul(4, close))',

    # ── Alpha #047 ──
    # 原始: sma((ts_max(high, 6) - close) / ((ts_max(high, 6) - ts_min(low, 6))) * 100, 9, 1)
    'alpha047': 'sma(mul(div(sub(ts_max(high, 6), close), sub(ts_max(high, 6), ts_min(low, 6))), 100), 9, 1)',

    # ── Alpha #048 ──
    # 原始: (-correlation(rank(high), rank(volume), 5) * ts_rank(volume, 5))
    'alpha048': 'mul(neg(correlation(cs_rank(high), cs_rank(volume), 5)), ts_rank(volume, 5))',

    # ── Alpha #049 ──
    # 原始: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < -0.05) ? 1 : (-close / delay(close, 20)))
    'alpha049': 'ifelse(add(sub(div(sub(delay(close, 20), delay(close, 10)), 10), div(sub(delay(close, 10), close), 10)), 0.05), div(neg(close), delay(close, 20)), 1)',

    # ── Alpha #050 ──
    # 原始: (-ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5))
    'alpha050': 'neg(ts_max(cs_rank(correlation(cs_rank(volume), cs_rank(vwap), 5)), 5))',

    # ── Alpha #051 ──
    # 原始: (((((delay(close, 20) - delay(close, 10)) / 10) - ((delay(close, 10) - close) / 10)) < -0.05) ? 1 : -1)
    'alpha051': 'ifelse(add(sub(div(sub(delay(close, 20), delay(close, 10)), 10), div(sub(delay(close, 10), close), 10)), 0.05), -1, 1)',

    # ── Alpha #052 ──
    # 原始: (((-ts_min(low, 5) + delay(ts_min(low, 5), 5)) * rank(((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    'alpha052': 'mul(mul(add(neg(ts_min(low, 5)), delay(ts_min(low, 5), 5)), cs_rank(div(sub(ts_sum(returns, 240), ts_sum(returns, 20)), 220))), ts_rank(volume, 5))',

    # ── Alpha #053 ──
    # 原始: -delta((((close - low) - (high - close)) / (close - low)), 9)
    'alpha053': 'neg(sub(div(sub(sub(close, low), sub(high, close)), sub(close, low)), delay(div(sub(sub(close, low), sub(high, close)), sub(close, low)), 9)))',

    # ── Alpha #054 ──
    # 原始: ((-((low - close) * (open^5))) / ((low - high) * (close^5)))
    'alpha054': 'div(neg(mul(sub(low, close), signed_power(open, 5))), mul(sub(low, high), signed_power(close, 5)))',

    # ── Alpha #055 ──
    # 原始: (-correlation(rank(((close - ts_min(low, 12)) / (ts_max(high, 12) - ts_min(low, 12)))), rank(volume), 6))
    'alpha055': 'neg(correlation(cs_rank(div(sub(close, ts_min(low, 12)), sub(ts_max(high, 12), ts_min(low, 12)))), cs_rank(volume), 6))',

    # ── Alpha #056 ──
    # 原始: (0 - (1 * (rank((ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3))) * rank((returns * cap)))))
    'alpha056': 'neg(mul(cs_rank(div(ts_sum(returns, 10), ts_sum(ts_sum(returns, 2), 3))), cs_rank(returns)))',

    # ── Alpha #057 ──
    # 原始: (0 - (1 * ((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))))
    'alpha057': 'neg(div(sub(close, vwap), decay_linear(cs_rank(ts_argmax(close, 30)), 2)))',

    # ── Alpha #058 ──
    # 原始: -ts_rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), volume, 3.92795), 7.89291), 5.50322)
    'alpha058': 'neg(ts_rank(decay_linear(correlation(vwap, volume, 4), 8), 6))',

    # ── Alpha #059 ──
    # 原始: -ts_rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), volume, 4.78795), 5.65416), 4.08144)
    'alpha059': 'neg(ts_rank(decay_linear(correlation(vwap, volume, 5), 6), 4))',

    # ── Alpha #060 ──
    # 原始: (0 - (1 * ((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume))) - scale(rank(ts_argmax(close, 10))))))
    'alpha060': 'neg(sub(mul(2, cs_rank(mul(div(sub(sub(close, low), sub(high, close)), sub(high, low)), volume))), cs_rank(ts_argmax(close, 10))))',

    # ── Alpha #061 ──
    # 原始: (rank((vwap - ts_min(vwap, 16.1219))) < rank(correlation(vwap, adv180, 17.9282)))
    'alpha061': 'sub(cs_rank(sub(vwap, ts_min(vwap, 16))), cs_rank(correlation(vwap, ts_mean(volume, 180), 18)))',

    # ── Alpha #062 ──
    # 原始: ((rank(correlation(vwap, ts_sum(adv20, 22.4101), 9.91009))) < rank(((rank(open) + rank(open)) < (rank(((high + low) / 2)) + rank(high)))))
    'alpha062': 'sub(cs_rank(correlation(vwap, ts_sum(ts_mean(volume, 20), 22), 10)), cs_rank(sub(add(cs_rank(open), cs_rank(open)), add(cs_rank(div(add(high, low), 2)), cs_rank(high)))))',

    # ── Alpha #063 ──
    # 原始: -((rank(decay_linear(delta(IndNeutralize(close, IndClass.industry), 2.25164), 8.22237))) - rank(decay_linear(ts_sum(volume, 10) / adv10, 0.35445)))
    'alpha063': 'neg(sub(cs_rank(decay_linear(sub(close, delay(close, 2)), 8)), cs_rank(decay_linear(div(ts_sum(volume, 10), ts_mean(volume, 10)), 1))))',

    # ── Alpha #064 ──
    # 原始: ((rank(correlation(ts_sum(((open * 0.178404) + (low * (1 - 0.178404))), 12.7054), ts_sum(adv120, 12.7054), 16.6208))) < rank(delta(((((high + low) / 2) * 0.178404) + (vwap * (1 - 0.178404))), 3.69741)))
    'alpha064': 'sub(cs_rank(correlation(ts_sum(add(mul(open, 0.178), mul(low, 0.822)), 13), ts_sum(ts_mean(volume, 120), 13), 17)), cs_rank(sub(add(mul(div(add(high, low), 2), 0.178), mul(vwap, 0.822)), delay(add(mul(div(add(high, low), 2), 0.178), mul(vwap, 0.822)), 4))))',

    # ── Alpha #065 ──
    # 原始: ((rank(correlation(((open * 0.00817205) + (vwap * (1 - 0.00817205))), ts_sum(adv60, 8.6911), 6.40374))) < rank((open - ts_min(open, 13.635))))
    'alpha065': 'sub(cs_rank(correlation(add(mul(open, 0.008), mul(vwap, 0.992)), ts_sum(ts_mean(volume, 60), 9), 6)), cs_rank(sub(open, ts_min(open, 14))))',

    # ── Alpha #066 ──
    # 原始: ((rank(decay_linear(delta(vwap, 3.51013), 7.23052)) + ts_rank(decay_linear(((((low* 0.96633) + (low * (1 - 0.96633))) - vwap) / (open - ((high + low) / 2))), 11.4157), 6.72611)) * -1)
    'alpha066': 'neg(add(cs_rank(decay_linear(sub(vwap, delay(vwap, 4)), 7)), ts_rank(decay_linear(div(sub(low, vwap), sub(open, div(add(high, low), 2))), 11), 7)))',

    # ── Alpha #067 ──
    # 原始: sma((close - sma(close, 10, 1)) / sma(close, 10, 1) * 100, 2, 1)
    'alpha067': 'sma(mul(div(sub(close, sma(close, 10, 1)), sma(close, 10, 1)), 100), 2, 1)',

    # ── Alpha #068 ──
    # 原始: sma(((high + low) / 2 - sma((high + low) / 2, 15, 2)) / (0.5 * sma(abs((high + low) / 2 - sma((high + low) / 2, 15, 2)), 15, 2)) * 100, 2, 1)
    'alpha068': 'sma(mul(div(sub(div(add(high, low), 2), sma(div(add(high, low), 2), 15, 2)), mul(0.5, sma(abs(sub(div(add(high, low), 2), sma(div(add(high, low), 2), 15, 2))), 15, 2))), 100), 2, 1)',

    # ── Alpha #069 ──
    # 原始: ((rank(ts_max(delta(IndNeutralize(vwap, IndClass.industry), 2.72412), 4.79344))^ts_rank(correlation(((close*0.490655) + (vwap*(1-0.490655))), adv20, 4.92416), 9.0615)) * -1)
    'alpha069': 'neg(signed_power(cs_rank(ts_max(sub(vwap, delay(vwap, 3)), 5)), ts_rank(correlation(add(mul(close, 0.491), mul(vwap, 0.509)), ts_mean(volume, 20), 5), 9)))',

    # ── Alpha #070 ──
    # 原始: sma((close - sma(close, 10, 2)) / sma(close, 10, 2) * 100, 2, 1)
    'alpha070': 'sma(mul(div(sub(close, sma(close, 10, 2)), sma(close, 10, 2)), 100), 2, 1)',

    # ── Alpha #071 ──
    # 原始: ((rank(decay_linear(correlation(ts_rank(close, 3.43976), cs_rank(volume), 5.078), 7.2288))) * -1)
    'alpha071': 'neg(cs_rank(decay_linear(correlation(ts_rank(close, 3), cs_rank(volume), 5), 7)))',

    # ── Alpha #072 ──
    # 原始: (ts_rank(decay_linear(correlation(((high + low) / 2), adv40, 8.93355), 3.64587), 1.54837))
    'alpha072': 'ts_rank(decay_linear(correlation(div(add(high, low), 2), ts_mean(volume, 40), 9), 4), 2)',

    # ── Alpha #073 ──
    # 原始: (ts_rank(decay_linear(correlation(((high + low) / 2), adv40, 4.798), 3.354), 1.548))
    'alpha073': 'ts_rank(decay_linear(correlation(div(add(high, low), 2), ts_mean(volume, 40), 5), 3), 2)',

    # ── Alpha #074 ──
    # 原始: (rank(correlation(close, ts_sum(adv30, 37.4843), 15.1365)) < rank(correlation(rank(((high * 0.0261661) + (vwap * (1 - 0.0261661)))), rank(volume), 11.4791)))
    'alpha074': 'sub(cs_rank(correlation(close, ts_sum(ts_mean(volume, 30), 37), 15)), cs_rank(correlation(cs_rank(add(mul(high, 0.026), mul(vwap, 0.974))), cs_rank(volume), 11)))',

    # ── Alpha #075 ──
    # 原始: (rank(correlation(vwap, volume, 4.24304)) < rank(correlation(rank(low), rank(adv50), 12.4413)))
    'alpha075': 'sub(cs_rank(correlation(vwap, volume, 4)), cs_rank(correlation(cs_rank(low), cs_rank(ts_mean(volume, 50)), 12)))',

    # ── Alpha #076 ──
    # 原始: (ts_rank(decay_linear(correlation(vwap, volume, 17.2927), 20.4309), 2.52082))
    'alpha076': 'ts_rank(decay_linear(correlation(vwap, volume, 17), 20), 3)',

    # ── Alpha #077 ──
    # 原始: (ts_rank(decay_linear(((high + low) / 2 + high - (vwap + high)), 9.40452), 4.26891))
    'alpha077': 'ts_rank(decay_linear(sub(add(div(add(high, low), 2), high), add(vwap, high)), 9), 4)',

    # ── Alpha #078 ──
    # 原始: (rank(correlation(ts_sum(((low * 0.352233) + (vwap * (1 - 0.352233))), 19.7428), ts_sum(adv40, 19.7428), 6.83313))^rank(correlation(rank(vwap), rank(volume), 5.77478)))
    'alpha078': 'signed_power(cs_rank(correlation(ts_sum(add(mul(low, 0.352), mul(vwap, 0.648)), 20), ts_sum(ts_mean(volume, 40), 20), 7)), cs_rank(correlation(cs_rank(vwap), cs_rank(volume), 6)))',

    # ── Alpha #079 ──
    # 原始: (rank(delta(IndNeutralize(((close * 0.60733) + (open * (1 - 0.60733))), IndClass.industry), 3.77542))^ts_rank(correlation(ts_rank(vwap, 3.30162), ts_rank(adv81, 8.98462), 11.671), 2.61946))
    'alpha079': 'signed_power(cs_rank(sub(add(mul(close, 0.607), mul(open, 0.393)), delay(add(mul(close, 0.607), mul(open, 0.393)), 4))), ts_rank(correlation(ts_rank(vwap, 3), ts_rank(ts_mean(volume, 81), 9), 12), 3))',

    # ── Alpha #080 ──
    # 原始: ((rank(sign(delta(IndNeutralize(((open * 0.868128) + (high * (1 - 0.868128))), IndClass.industry), 4.04545)))^ts_rank(correlation(high, adv10, 5.11456), 5.53756)) * -1)
    'alpha080': 'neg(mul(cs_rank(sign(sub(add(mul(open, 0.868), mul(high, 0.132)), delay(add(mul(open, 0.868), mul(high, 0.132)), 4)))), ts_rank(correlation(high, ts_mean(volume, 10), 5), 6)))',

    # ── Alpha #081 ──
    # 原始: ((rank(log(product(rank((rank(correlation(vwap, ts_sum(adv10, 49.6054), 8.47743))^4)), 14.9655))) < rank(correlation(rank(vwap), rank(volume), 5.07914))) * -1)
    'alpha081': 'neg(sub(cs_rank(log(cs_rank(signed_power(cs_rank(correlation(vwap, ts_sum(ts_mean(volume, 10), 50), 8)), 4)))), cs_rank(correlation(cs_rank(vwap), cs_rank(volume), 5))))',

    # ── Alpha #082 ──
    # 原始: (ts_min(rank(decay_linear(delta(open, 1.46063), 14.8717)), 4.94768) * -1)
    'alpha082': 'neg(ts_min(cs_rank(decay_linear(sub(open, delay(open, 1)), 15)), 5))',

    # ── Alpha #083 ──
    # 原始: ((rank(delay(((high - low) / (ts_sum(close, 5) / 5)), 2)) * rank(rank(volume))) / (((high - low) / (ts_sum(close, 5) / 5)) / (vwap - close)))
    'alpha083': 'div(mul(cs_rank(delay(div(sub(high, low), div(ts_sum(close, 5), 5)), 2)), cs_rank(cs_rank(volume))), div(div(sub(high, low), div(ts_sum(close, 5), 5)), sub(vwap, close)))',

    # ── Alpha #084 ──
    # 原始: signedpower(ts_rank((vwap - ts_max(vwap, 15.3217)), 20.7127), delta(close, 4.96796))
    'alpha084': 'signed_power(ts_rank(sub(vwap, ts_max(vwap, 15)), 21), sub(close, delay(close, 5)))',

    # ── Alpha #085 ──
    # 原始: (rank(correlation(((high * 0.876703) + (close * (1 - 0.876703))), adv30, 9.61331))^rank(correlation(ts_rank(((high + low) / 2), 3.70596), ts_rank(volume, 10.1595), 14.9085)))
    'alpha085': 'signed_power(cs_rank(correlation(add(mul(high, 0.877), mul(close, 0.123)), ts_mean(volume, 30), 10)), cs_rank(correlation(ts_rank(div(add(high, low), 2), 4), ts_rank(volume, 10), 15)))',

    # ── Alpha #086 ──
    # 原始: ((ts_rank(correlation(close, ts_sum(adv20, 14.7444), 6.00046), 20.4195) < rank(((open + close) - (vwap + open)))) * -1)
    'alpha086': 'neg(sub(ts_rank(correlation(close, ts_sum(ts_mean(volume, 20), 15), 6), 20), cs_rank(sub(add(open, close), add(vwap, open)))))',

    # ── Alpha #087 ──
    # 原始: (sma(((close - sma(close, 10, 2)) / sma(close, 10, 2)) * 100, 1.68304, 2.11422))
    'alpha087': 'sma(mul(div(sub(close, sma(close, 10, 2)), sma(close, 10, 2)), 100), 2, 2)',

    # ── Alpha #088 ──
    # 原始: (rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8.06882)))
    'alpha088': 'cs_rank(decay_linear(sub(add(cs_rank(open), cs_rank(low)), add(cs_rank(high), cs_rank(close))), 8))',

    # ── Alpha #089 ──
    # 原始: (ts_rank(decay_linear(correlation(((low * 0.967285) + (low * (1 - 0.967285))), adv10, 6.94279), 5.51607), 3.79744) - ts_rank(decay_linear(delta(IndNeutralize(vwap, IndClass.industry), 3.48158), 10.1463), 15.3012))
    'alpha089': 'sub(ts_rank(decay_linear(correlation(low, ts_mean(volume, 10), 7), 6), 4), ts_rank(decay_linear(sub(vwap, delay(vwap, 3)), 10), 15))',

    # ── Alpha #090 ──
    # 原始: ((rank((close - ts_max(close, 4.66719)))^ts_rank(correlation(IndNeutralize(vwap, IndClass.industry), adv40, 8.66405), 7.00965)) * -1)
    'alpha090': 'neg(mul(cs_rank(sub(close, ts_max(close, 5))), ts_rank(correlation(vwap, ts_mean(volume, 40), 9), 7)))',

    # ── Alpha #091 ──
    # 原始: (sma(sma((close - sma(close, 8, 1)) / sma(close, 8, 1) * 100, 3, 1), 3, 1))
    'alpha091': 'sma(sma(mul(div(sub(close, sma(close, 8, 1)), sma(close, 8, 1)), 100), 3, 1), 3, 1)',

    # ── Alpha #092 ──
    # 原始: (ts_rank(decay_linear(correlation(((high + low) / 2 + close), adv10, 12.3721), 7.35093), 16.446))
    'alpha092': 'ts_rank(decay_linear(correlation(add(div(add(high, low), 2), close), ts_mean(volume, 10), 12), 7), 16)',

    # ── Alpha #093 ──
    # 原始: (ts_rank(decay_linear(correlation(IndNeutralize(vwap, IndClass.industry), adv81, 17.4193), 5.70597), 5.5743))
    'alpha093': 'ts_rank(decay_linear(correlation(vwap, ts_mean(volume, 81), 17), 6), 6)',

    # ── Alpha #094 ──
    # 原始: ((rank((vwap - ts_min(vwap, 11.5783)))^ts_rank(correlation(ts_rank(vwap, 19.6462), ts_rank(adv60, 4.02992), 18.0926), 2.70756)) * -1)
    'alpha094': 'neg(mul(cs_rank(sub(vwap, ts_min(vwap, 12))), ts_rank(correlation(ts_rank(vwap, 20), ts_rank(ts_mean(volume, 60), 4), 18), 3)))',

    # ── Alpha #095 ──
    # 原始: (rank((open - ts_min(open, 12.4105))) < ts_rank((rank(correlation(ts_sum(((high + low) / 2), 19.1351), ts_sum(adv40, 19.1351), 12.8742))^5), 11.7584))
    'alpha095': 'sub(cs_rank(sub(open, ts_min(open, 12))), ts_rank(signed_power(cs_rank(correlation(ts_sum(div(add(high, low), 2), 19), ts_sum(ts_mean(volume, 40), 19), 13)), 5), 12))',

    # ── Alpha #096 ──
    # 原始: (ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 5.69878), 7.17348), 4.73834) * -1)
    'alpha096': 'neg(ts_rank(decay_linear(correlation(cs_rank(vwap), cs_rank(volume), 6), 7), 5))',

    # ── Alpha #097 ──
    # 原始: ((rank(decay_linear(delta(IndNeutralize(((low * 0.721001) + (vwap * (1 - 0.721001))), IndClass.industry), 3.3705), 20.4523)) - ts_rank(decay_linear(ts_rank(correlation(ts_rank(low, 7.87871), ts_rank(adv60, 17.255), 4.97547), 18.5925), 15.1801), 6.8346)) * -1)
    'alpha097': 'neg(sub(cs_rank(decay_linear(sub(add(mul(low, 0.721), mul(vwap, 0.279)), delay(add(mul(low, 0.721), mul(vwap, 0.279)), 3)), 20)), ts_rank(decay_linear(ts_rank(correlation(ts_rank(low, 8), ts_rank(ts_mean(volume, 60), 17), 5), 19), 15), 7)))',

    # ── Alpha #098 ──
    # 原始: (rank(decay_linear(correlation(vwap, ts_sum(adv5, 26.4719), 4.58418), 7.18088)) - 5.23607)
    'alpha098': 'sub(cs_rank(decay_linear(correlation(vwap, ts_sum(ts_mean(volume, 5), 26), 5), 7)), 5.236)',

    # ── Alpha #099 ──
    # 原始: ((rank(correlation(ts_sum(((high + low) / 2), 19.8975), ts_sum(adv60, 19.8975), 8.8136)) < rank(correlation(low, volume, 6.28259))) * -1)
    'alpha099': 'neg(sub(cs_rank(correlation(ts_sum(div(add(high, low), 2), 20), ts_sum(ts_mean(volume, 60), 20), 9)), cs_rank(correlation(low, volume, 6))))',

    # ── Alpha #100 ──
    # 原始: (0 - (1 * ((1.5 * scale(indneutralize(indneutralize(rank(((((close - low) - (high - close)) / (high - low)) * volume), IndClass.subindustry), IndClass.subindustry))) - scale(indneutralize((correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))), IndClass.subindustry)))))
    'alpha100': 'neg(sub(mul(1.5, cs_rank(mul(div(sub(sub(close, low), sub(high, close)), sub(high, low)), volume))), sub(correlation(close, cs_rank(ts_mean(volume, 20)), 5), cs_rank(ts_argmin(close, 30)))))',

    # ── Alpha #101 ──
    # 原始: ((close - open) / ((high - low) + 0.001))
    'alpha101': 'div(sub(close, open), add(sub(high, low), 0.001))',
}


# ============================================================================
# 因子分类映射
# ============================================================================

ALPHA101_CATEGORIES: Dict[str, FactorCategory] = {
    'alpha001': FactorCategory.MOMENTUM,
    'alpha002': FactorCategory.MOMENTUM,
    'alpha003': FactorCategory.VOLUME,
    'alpha004': FactorCategory.REVERSAL,
    'alpha005': FactorCategory.MOMENTUM,
    'alpha006': FactorCategory.VOLUME,
    'alpha007': FactorCategory.MOMENTUM,
    'alpha008': FactorCategory.MOMENTUM,
    'alpha009': FactorCategory.MOMENTUM,
    'alpha010': FactorCategory.MOMENTUM,
    'alpha011': FactorCategory.VOLUME,
    'alpha012': FactorCategory.MOMENTUM,
    'alpha013': FactorCategory.VOLUME,
    'alpha014': FactorCategory.MOMENTUM,
    'alpha015': FactorCategory.VOLUME,
    'alpha016': FactorCategory.VOLUME,
    'alpha017': FactorCategory.MOMENTUM,
    'alpha018': FactorCategory.VOLATILITY,
    'alpha019': FactorCategory.MOMENTUM,
    'alpha020': FactorCategory.PRICE,
    'alpha021': FactorCategory.PRICE,
    'alpha022': FactorCategory.VOLUME,
    'alpha023': FactorCategory.PRICE,
    'alpha024': FactorCategory.MOMENTUM,
    'alpha025': FactorCategory.MOMENTUM,
    'alpha026': FactorCategory.VOLUME,
    'alpha027': FactorCategory.VOLUME,
    'alpha028': FactorCategory.PRICE,
    'alpha029': FactorCategory.MOMENTUM,
    'alpha030': FactorCategory.MOMENTUM,
    'alpha031': FactorCategory.MOMENTUM,
    'alpha032': FactorCategory.PRICE,
    'alpha033': FactorCategory.PRICE,
    'alpha034': FactorCategory.MOMENTUM,
    'alpha035': FactorCategory.VOLUME,
    'alpha041': FactorCategory.PRICE,
    'alpha042': FactorCategory.PRICE,
    'alpha043': FactorCategory.MOMENTUM,
    'alpha044': FactorCategory.VOLUME,
    'alpha046': FactorCategory.PRICE,
    'alpha047': FactorCategory.TECHNICAL,
    'alpha048': FactorCategory.VOLUME,
    'alpha049': FactorCategory.MOMENTUM,
    'alpha050': FactorCategory.VOLUME,
    'alpha051': FactorCategory.MOMENTUM,
    'alpha052': FactorCategory.MOMENTUM,
    'alpha053': FactorCategory.PRICE,
    'alpha054': FactorCategory.PRICE,
    'alpha055': FactorCategory.VOLUME,
    'alpha057': FactorCategory.PRICE,
    'alpha060': FactorCategory.PRICE,
    'alpha067': FactorCategory.TECHNICAL,
    'alpha068': FactorCategory.TECHNICAL,
    'alpha070': FactorCategory.TECHNICAL,
    'alpha082': FactorCategory.PRICE,
    'alpha083': FactorCategory.VOLUME,
    'alpha084': FactorCategory.PRICE,
    'alpha087': FactorCategory.TECHNICAL,
    'alpha088': FactorCategory.PRICE,
    'alpha091': FactorCategory.TECHNICAL,
    'alpha098': FactorCategory.VOLUME,
    'alpha100': FactorCategory.PRICE,
    'alpha101': FactorCategory.PRICE,
}
