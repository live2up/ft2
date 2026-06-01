"""
factor/v3/formulas.py — 统一公式库（WQ101 + GT191）
=============================================================================

合并 v2 的 alpha101.py + alpha191.py。
包含 ~272 个因子公式，全部为 v3 引擎的 S-表达式语法。

模块结构：
  Part 1: ALPHA101 + 分类映射 （WorldQuant 101 Alpha）
  Part 2: ALPHA191 + 分类映射 （国泰安 191 Alpha，含 20 个预留基本面缺口）

使用方式：
  >>> from factor.v3.formulas import ALPHA101, ALPHA191
  >>> len(ALPHA101)  # 101
  >>> len(ALPHA191)  # 171（20 个预留）

[重构] 2026-06-01 从 v2 alpha101.py + alpha191.py 合并为 v3/formulas.py
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
