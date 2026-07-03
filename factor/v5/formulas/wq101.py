"""
kb03/formulas/wq101.py — WorldQuant 101 Alpha 因子 (V4 AST 语法)

结构: ALPHA101[name] = (v4_expr, category)
每个因子一个条目，公式和分类在一起。

翻译规则 (原始 WQ101 表 → V4 AST):
  rank(x)           → cs_rank(x)
  correlation(x,y,w)→ ts_corr(x,y,w)
  delay(x,w)        → ts_delay(x,w)
  delta(x,w)        → (x - ts_delay(x,w))
  signedpower(x,a)  → signed_power(x,a)
  scale(x)          → cs_scale(x) (WQ101 的 scale 是截面缩放)
  ts_rank(x,w)      → ts_rank(x,w)
  decay_linear(x,w) → ts_decay_linear(x,w)
  ts_sum(x,w)       → ts_sum(x,w)
  ts_max(x,w)       → ts_max(x,w)
  ts_min(x,w)       → ts_min(x,w)
  ts_argmax(x,w)    → ts_argmax(x,w)
  ts_argmin(x,w)    → ts_argmin(x,w)
  condition ? a : b → (a if condition else b)
  returns           → RETURNS
  adv20             → ts_mean(VOLUME,20)

注意: WQ101 的 ifelse 条件是显式比较运算符 (> < >= <= !=)
      绝非简单的非零判断。每条条件公式都对照原始定义手写。
"""
from typing import Dict, Tuple

ALPHA101: Dict[str, Tuple[str, str]] = {
    # ═══════════════════════════════════════════════════════════════
    # Alpha #001
    # 原始: (rank(ts_argmax(signedpower(((returns<0)?stddev(returns,20):close), 2), 5)) - 0.5)
    # ═══════════════════════════════════════════════════════════════
    "alpha001": (
        "(cs_rank(ts_argmax(signed_power("
        "(ts_std(RETURNS, 20) if RETURNS < 0 else CLOSE), 2), 5)) - 0.5)", "动量"),

    # ═══════════════════════════════════════════════════════════════
    # Alpha #002
    # 原始: -correlation(rank(delta(log(volume), 2)), rank((close-open)/open), 6)
    # ═══════════════════════════════════════════════════════════════
    "alpha002": (
        "(-ts_corr(cs_rank((log(VOLUME) - ts_delay(log(VOLUME), 2))), "
        "cs_rank(((CLOSE - OPEN) / OPEN)), 6))", "动量"),

    # Alpha #003: -correlation(rank(open), rank(volume), 10)
    "alpha003": ("(-ts_corr(cs_rank(OPEN), cs_rank(VOLUME), 10))", "量价"),

    # Alpha #004: -ts_rank(rank(low), 9)
    "alpha004": ("(-ts_rank(cs_rank(LOW), 9))", "反转"),

    # Alpha #005: rank((open - ts_mean(open, 10))) * (-abs(delta(close, 1)))
    "alpha005": ("(cs_rank((OPEN - ts_mean(OPEN, 10))) * (-abs((CLOSE - ts_delay(CLOSE, 1)))))", "动量"),

    # Alpha #006: -correlation(open, volume, 10)
    "alpha006": ("(-ts_corr(OPEN, VOLUME, 10))", "量价"),

    # ═══════════════════════════════════════════════════════════════
    # Alpha #007
    # 原始: ((adv20 < volume) ? ((-ts_rank(abs(delta(close, 7)), 60)) * sign(delta(close, 7))) : -1)
    # ═══════════════════════════════════════════════════════════════
    "alpha007": (
        "(((-ts_rank(abs((CLOSE - ts_delay(CLOSE, 7))), 60)) * sign((CLOSE - ts_delay(CLOSE, 7)))) "
        "if VOLUME > ts_mean(VOLUME, 20) else -1)", "动量"),

    # Alpha #008: -rank(((ts_sum(open, 5) * ts_sum(returns, 5)) - delay((ts_sum(open, 5) * ts_sum(returns, 5)), 10)))
    "alpha008": (
        "(-cs_rank(((ts_sum(OPEN, 5) * ts_sum(RETURNS, 5)) - "
        "ts_delay((ts_sum(OPEN, 5) * ts_sum(RETURNS, 5)), 10))))", "动量"),

    # ═══════════════════════════════════════════════════════════════
    # Alpha #009
    # 原始: ((0 < ts_min(delta(close, 1), 5)) ? delta(close, 1)
    #       : ((ts_max(delta(close, 1), 5) < 0) ? delta(close, 1) : -delta(close, 1)))
    # ═══════════════════════════════════════════════════════════════
    "alpha009": (
        "((CLOSE - ts_delay(CLOSE, 1)) "
        "if 0 < ts_min((CLOSE - ts_delay(CLOSE, 1)), 5) "
        "else ((CLOSE - ts_delay(CLOSE, 1)) "
        "if ts_max((CLOSE - ts_delay(CLOSE, 1)), 5) < 0 "
        "else (-(CLOSE - ts_delay(CLOSE, 1)))))", "动量"),

    # ═══════════════════════════════════════════════════════════════
    # Alpha #010: 同 alpha009 但窗口=4, 外层 rank
    # 原始: rank(((0 < ts_min(delta(close, 1), 4)) ? delta(close, 1)
    #       : ((ts_max(delta(close, 1), 4) < 0) ? delta(close, 1) : -delta(close, 1))))
    # ═══════════════════════════════════════════════════════════════
    "alpha010": (
        "cs_rank(((CLOSE - ts_delay(CLOSE, 1)) "
        "if 0 < ts_min((CLOSE - ts_delay(CLOSE, 1)), 4) "
        "else ((CLOSE - ts_delay(CLOSE, 1)) "
        "if ts_max((CLOSE - ts_delay(CLOSE, 1)), 4) < 0 "
        "else (-(CLOSE - ts_delay(CLOSE, 1))))))", "动量"),

    # Alpha #011: ((rank(ts_max((vwap - close), 3)) + rank(ts_min((vwap - close), 3))) * rank(delta(volume, 3)))
    "alpha011": (
        "((cs_rank(ts_max((VWAP - CLOSE), 3)) + cs_rank(ts_min((VWAP - CLOSE), 3))) "
        "* cs_rank((VOLUME - ts_delay(VOLUME, 3))))", "量价"),

    # Alpha #012: sign(delta(volume, 1)) * (-delta(close, 1))
    "alpha012": ("(sign((VOLUME - ts_delay(VOLUME, 1))) * (-(CLOSE - ts_delay(CLOSE, 1))))", "动量"),

    # Alpha #013: -rank(covariance(rank(close), rank(volume), 5))
    "alpha013": ("(-cs_rank(ts_cov(cs_rank(CLOSE), cs_rank(VOLUME), 5)))", "量价"),

    # Alpha #014: ((-rank(delta(returns, 3))) * correlation(open, volume, 10))
    "alpha014": (
        "((-cs_rank((RETURNS - ts_delay(RETURNS, 3)))) * ts_corr(OPEN, VOLUME, 10))", "动量"),

    # Alpha #015: -ts_sum(rank(correlation(rank(high), rank(volume), 3)), 3)
    "alpha015": ("(-ts_sum(cs_rank(ts_corr(cs_rank(HIGH), cs_rank(VOLUME), 3)), 3))", "量价"),

    # Alpha #016: -rank(covariance(rank(high), rank(volume), 5))
    "alpha016": ("(-cs_rank(ts_cov(cs_rank(HIGH), cs_rank(VOLUME), 5)))", "量价"),

    # Alpha #017: (((-rank(ts_rank(close, 10))) * rank(delta(delta(close, 1), 1))) * rank(ts_rank((volume / adv20), 5)))
    "alpha017": (
        "(((-cs_rank(ts_rank(CLOSE, 10))) "
        "* cs_rank(((CLOSE - ts_delay(CLOSE, 1)) - ts_delay((CLOSE - ts_delay(CLOSE, 1)), 1)))) "
        "* cs_rank(ts_rank((VOLUME / ts_mean(VOLUME, 20)), 5)))", "动量"),

    # Alpha #018: -rank(((stddev(abs((close - open)), 5) + (close - open)) + correlation(close, open, 10)))
    "alpha018": (
        "(-cs_rank(((ts_std(abs((CLOSE - OPEN)), 5) + (CLOSE - OPEN)) + ts_corr(CLOSE, OPEN, 10))))", "波动率"),

    # Alpha #019: ((-sign((close - delay(close, 7)) + delta(close, 7))) * (1 + rank(1 + ts_sum(returns, 250))))
    "alpha019": (
        "((-sign(((CLOSE - ts_delay(CLOSE, 7)) + (CLOSE - ts_delay(CLOSE, 7))))) "
        "* (1 + cs_rank((1 + ts_sum(RETURNS, 250)))))", "动量"),

    # Alpha #020: (((-rank(open - delay(high, 1))) * rank(open - delay(close, 1))) * rank(open - delay(low, 1)))
    "alpha020": (
        "(((-cs_rank((OPEN - ts_delay(HIGH, 1)))) "
        "* cs_rank((OPEN - ts_delay(CLOSE, 1)))) "
        "* cs_rank((OPEN - ts_delay(LOW, 1))))", "价格"),

    # ═══════════════════════════════════════════════════════════════
    # Alpha #021
    # 原始: (((ts_sum(close, 8) / 8) + ts_std(close, 8)) < (ts_sum(close, 2) / 2)) ? -1 : 1
    # ═══════════════════════════════════════════════════════════════
    "alpha021": (
        "(-1 if (ts_mean(CLOSE, 8) + ts_std(CLOSE, 8)) < ts_mean(CLOSE, 2) else 1)", "价格"),

    # Alpha #022: -delta(correlation(high, volume, 5), 5) * rank(ts_std(close, 20))
    "alpha022": (
        "((-(ts_corr(HIGH, VOLUME, 5) - ts_delay(ts_corr(HIGH, VOLUME, 5), 5))) "
        "* cs_rank(ts_std(CLOSE, 20)))", "波动率"),

    # ═══════════════════════════════════════════════════════════════
    # Alpha #023
    # 原始: ((ts_sum(high, 20) / 20) < high) ? (-delta(high, 2)) : 0
    # ═══════════════════════════════════════════════════════════════
    "alpha023": (
        "(-(HIGH - ts_delay(HIGH, 2)) if ts_mean(HIGH, 20) < HIGH else 0)", "价格"),

    # ═══════════════════════════════════════════════════════════════
    # Alpha #024
    # 原始: ((delta(MA100(close), 100) / delay(close, 100)) <= 0.05) ?
    #       (-close / (MA100(close) - close)) : (-log(close))
    # ═══════════════════════════════════════════════════════════════
    "alpha024": (
        "(((-CLOSE) / ((ts_mean(CLOSE, 100) - CLOSE) + 0.001)) "
        "if ((ts_mean(CLOSE, 100) - ts_delay(ts_mean(CLOSE, 100), 100)) "
        "/ (ts_delay(CLOSE, 100) + 0.001)) <= 0.05 "
        "else (-log(CLOSE)))", "动量"),

    # Alpha #025: rank((-returns * adv20 * vwap * (high - close)))
    "alpha025": (
        "cs_rank(((((-RETURNS) * ts_mean(VOLUME, 20)) * VWAP) * (HIGH - CLOSE)))", "动量"),

    # Alpha #026: -ts_max(correlation(ts_rank(volume, 5), ts_rank(high, 5), 5), 3)
    "alpha026": (
        "(-ts_max(ts_corr(ts_rank(VOLUME, 5), ts_rank(HIGH, 5), 5), 3))", "量价"),

    # ═══════════════════════════════════════════════════════════════
    # Alpha #027
    # 原始: (0.5 < rank(correlation(rank(volume), rank(vwap), 6))) ? -1 : 1
    # ═══════════════════════════════════════════════════════════════
    "alpha027": (
        "(-1 if cs_rank(ts_corr(cs_rank(VOLUME), cs_rank(VWAP), 6)) > 0.5 else 1)", "量价"),

    # Alpha #028: scale(((correlation(adv20, low, 5) + ((high + low) / 2)) - close))
    "alpha028": (
        "((ts_corr(ts_mean(VOLUME, 20), LOW, 5) + ((HIGH + LOW) / 2)) - CLOSE)", "价格"),

    # Alpha #029: (min(product(rank(rank(log(ts_sum(rank(rank((-rank(rank(delta(close, 1)))))), 15)))), 1)), 5) + ts_rank(delay(-returns, 6), 5)
    "alpha029": (
        "(ts_min(cs_rank(log(ts_sum(cs_rank((-cs_rank((CLOSE - ts_delay(CLOSE, 1))))), 15))), 5) "
        "+ ts_rank(ts_delay((-RETURNS), 6), 5))", "动量"),

    # Alpha #030: (((1.0 - rank(((sign(delta(close,1)) + sign(delay(close,1)-delay(close,2))) + sign(delay(close,2)-delay(close,3)))))) * ts_sum(volume, 5)) / ts_sum(volume, 20)
    "alpha030": (
        "(((1 - cs_rank(((sign((CLOSE - ts_delay(CLOSE, 1))) "
        "+ sign((ts_delay(CLOSE, 1) - ts_delay(CLOSE, 2)))) "
        "+ sign((ts_delay(CLOSE, 2) - ts_delay(CLOSE, 3)))))) "
        "* ts_sum(VOLUME, 5)) / ts_sum(VOLUME, 20))", "动量"),

    # Alpha #031: ((rank(rank(rank(decay_linear((-rank(rank(delta(close, 10)))), 10)))) + rank((-delta(close, 3)))) + sign(scale(correlation(adv20, low, 12))))
    "alpha031": (
        "((cs_rank(ts_decay_linear((-cs_rank((CLOSE - ts_delay(CLOSE, 10)))), 10)) "
        "+ cs_rank((-(CLOSE - ts_delay(CLOSE, 3))))) "
        "+ sign(ts_corr(ts_mean(VOLUME, 20), LOW, 12)))", "动量"),

    # Alpha #032: (scale(((ts_sum(close, 7) / 7) - close)) + (20 * scale(correlation(vwap, delay(close, 5), 230))))
    "alpha032": (
        "(((ts_sum(CLOSE, 7) / 7) - CLOSE) + (20 * ts_corr(VWAP, ts_delay(CLOSE, 5), 230)))", "价格"),

    # Alpha #033: rank((-((1 - (open / close)))))
    "alpha033": ("cs_rank((-(1 - (OPEN / CLOSE))))", "价格"),

    # Alpha #034: rank(((1 - rank((ts_std(returns, 2) / ts_std(returns, 5)))) + (1 - rank(delta(close, 1)))))
    "alpha034": (
        "cs_rank(((1 - cs_rank((ts_std(RETURNS, 2) / ts_std(RETURNS, 5)))) "
        "+ (1 - cs_rank((CLOSE - ts_delay(CLOSE, 1))))))", "动量"),

    # Alpha #035: ((ts_rank(volume, 32) * (1 - ts_rank(((close + high) - low), 16))) * (1 - ts_rank(returns, 32)))
    "alpha035": (
        "((ts_rank(VOLUME, 32) * (1 - ts_rank(((CLOSE + HIGH) - LOW), 16))) "
        "* (1 - ts_rank(RETURNS, 32)))", "量价"),

    # Alpha #036: (((((2.21 * rank(correlation((close - open), delay(volume, 1), 15))) + (0.7 * rank((open - close)))) + (0.73 * rank(ts_rank(delay((-returns), 6), 5)))) + rank(abs(correlation(vwap, adv20, 6)))) + (0.6 * rank(((ts_sum(close, 200) / 200) - open) * (close - open))))
    "alpha036": (
        "(((((2.21 * cs_rank(ts_corr((CLOSE - OPEN), ts_delay(VOLUME, 1), 15))) "
        "+ (0.7 * cs_rank((OPEN - CLOSE)))) "
        "+ (0.73 * cs_rank(ts_rank(ts_delay((-RETURNS), 6), 5)))) "
        "+ cs_rank(abs(ts_corr(VWAP, ts_mean(VOLUME, 20), 6)))) "
        "+ (0.6 * cs_rank((((ts_mean(CLOSE, 200) - OPEN) * (CLOSE - OPEN))))))", "动量"),

    # Alpha #037: (rank(correlation(delay((open - close), 1), close, 200)) + rank((open - close)))
    "alpha037": (
        "(cs_rank(ts_corr(ts_delay((OPEN - CLOSE), 1), CLOSE, 200)) + cs_rank((OPEN - CLOSE)))", "价格"),

    # Alpha #038: ((-rank(ts_rank(close, 10))) * rank((close / open)))
    "alpha038": ("((-cs_rank(ts_rank(CLOSE, 10))) * cs_rank((CLOSE / OPEN)))", "价格"),

    # Alpha #039: ((-rank((delta(close, 7) * (1 - rank(decay_linear((volume / adv20), 9)))))) * (1 + rank(ts_sum(returns, 250))))
    "alpha039": (
        "((-cs_rank(((CLOSE - ts_delay(CLOSE, 7)) "
        "* (1 - cs_rank(ts_decay_linear((VOLUME / ts_mean(VOLUME, 20)), 9)))))) "
        "* (1 + cs_rank(ts_sum(RETURNS, 250))))", "动量"),

    # Alpha #040: ((-rank(ts_std(high, 10))) * correlation(high, volume, 10))
    "alpha040": (
        "((-cs_rank(ts_std(HIGH, 10))) * ts_corr(HIGH, VOLUME, 10))", "波动率"),

    # Alpha #041: (sqrt((high * low)) - vwap)
    "alpha041": ("(sqrt((HIGH * LOW)) - VWAP)", "价格"),

    # Alpha #042: (rank((vwap - close)) / rank((vwap + close)))
    "alpha042": ("(cs_rank((VWAP - CLOSE)) / cs_rank((VWAP + CLOSE)))", "价格"),

    # Alpha #043: (ts_rank(adv20, 20) * ts_rank((-(close - delay(close, 7))), 8))
    "alpha043": (
        "(ts_rank(ts_mean(VOLUME, 20), 20) * ts_rank((-(CLOSE - ts_delay(CLOSE, 7))), 8))", "动量"),

    # Alpha #044: -correlation(high, rank(volume), 5)
    "alpha044": ("(-ts_corr(HIGH, cs_rank(VOLUME), 5))", "量价"),

    # Alpha #045: ((-rank(((close * 0.6 + open * 0.4) - delay((close * 0.6 + open * 0.4), 1)))) * rank(correlation(vwap, adv20, 15)))
    "alpha045": (
        "((-cs_rank((((CLOSE * 0.6) + (OPEN * 0.4)) "
        "- ts_delay(((CLOSE * 0.6) + (OPEN * 0.4)), 1)))) "
        "* cs_rank(ts_corr(VWAP, ts_mean(VOLUME, 20), 15)))", "量价"),

    # Alpha #046: ((MA3(close) + MA6(close) + MA12(close) + MA24(close)) / (4 * close))
    "alpha046": (
        "(((ts_mean(CLOSE, 3) + ts_mean(CLOSE, 6)) "
        "+ ts_mean(CLOSE, 12) + ts_mean(CLOSE, 24)) / (4 * CLOSE))", "价格"),

    # Alpha #047: delay(mean((((max6(high) - close) / (max6(high) - min6(low))) * 100), 9), 1)
    "alpha047": (
        "ts_delay(ts_mean((((ts_max(HIGH, 6) - CLOSE) "
        "/ (ts_max(HIGH, 6) - ts_min(LOW, 6))) * 100), 9), 1)", "技术指标"),

    # Alpha #048: ((-correlation(rank(high), rank(volume), 5)) * ts_rank(volume, 5))
    "alpha048": (
        "((-ts_corr(cs_rank(HIGH), cs_rank(VOLUME), 5)) * ts_rank(VOLUME, 5))", "量价"),

    # ═══════════════════════════════════════════════════════════════
    # Alpha #049
    # 原始: (( (delta(close,20)/20 - delta(close,10)/10) + 0.05) < 0) ?
    #       (-close / delay(close,20)) : 1
    # ═══════════════════════════════════════════════════════════════
    "alpha049": (
        "(((-CLOSE) / (ts_delay(CLOSE, 20) + 0.001)) "
        "if ((((ts_delay(CLOSE, 20) - ts_delay(CLOSE, 10)) / 10) "
        "- ((ts_delay(CLOSE, 10) - CLOSE) / 10)) + 0.05) < 0 "
        "else 1)", "动量"),

    # Alpha #050: -ts_max(rank(correlation(rank(volume), rank(vwap), 5)), 5)
    "alpha050": ("(-ts_max(cs_rank(ts_corr(cs_rank(VOLUME), cs_rank(VWAP), 5)), 5))", "量价"),

    # ═══════════════════════════════════════════════════════════════
    # Alpha #051: 条件同 alpha049，但返回 -1 / 1
    # ═══════════════════════════════════════════════════════════════
    "alpha051": (
        "(-1 if ((((ts_delay(CLOSE, 20) - ts_delay(CLOSE, 10)) / 10) "
        "- ((ts_delay(CLOSE, 10) - CLOSE) / 10)) + 0.05) < 0 else 1)", "动量"),

    # Alpha #052: (((-min(low, 5) + delay(min(low, 5), 5)) * rank(((ts_sum(returns, 240) - ts_sum(returns, 20)) / 220))) * ts_rank(volume, 5))
    "alpha052": (
        "((((-ts_min(LOW, 5)) + ts_delay(ts_min(LOW, 5), 5)) "
        "* cs_rank(((ts_sum(RETURNS, 240) - ts_sum(RETURNS, 20)) / 220))) "
        "* ts_rank(VOLUME, 5))", "动量"),

    # Alpha #053: -((((close - low) - (high - close)) / (close - low)) - delta((((close - low) - (high - close)) / (close - low)), 9))
    "alpha053": (
        "(-((((CLOSE - LOW) - (HIGH - CLOSE)) / (CLOSE - LOW)) "
        "- ts_delay((((CLOSE - LOW) - (HIGH - CLOSE)) / (CLOSE - LOW)), 9)))", "价格"),

    # Alpha #054: (-((low - close) * signed_power(open, 5))) / ((low - high) * signed_power(close, 5))
    "alpha054": (
        "((-((LOW - CLOSE) * signed_power(OPEN, 5))) / ((LOW - HIGH) * signed_power(CLOSE, 5)))", "价格"),

    # Alpha #055: -correlation(rank(((close - min(low, 12)) / (max(high, 12) - min(low, 12)))), rank(volume), 6)
    "alpha055": (
        "(-ts_corr(cs_rank(((CLOSE - ts_min(LOW, 12)) "
        "/ (ts_max(HIGH, 12) - ts_min(LOW, 12)))), cs_rank(VOLUME), 6))", "量价"),

    # Alpha #056: -(rank((ts_sum(returns, 10) / ts_sum(ts_sum(returns, 2), 3))) * rank(returns))
    "alpha056": (
        "(-(cs_rank((ts_sum(RETURNS, 10) / ts_sum(ts_sum(RETURNS, 2), 3))) * cs_rank(RETURNS)))", "动量"),

    # Alpha #057: -((close - vwap) / decay_linear(rank(ts_argmax(close, 30)), 2))
    "alpha057": (
        "(-((CLOSE - VWAP) / ts_decay_linear(cs_rank(ts_argmax(CLOSE, 30)), 2)))", "价格"),

    # Alpha #058: -ts_rank(decay_linear(correlation(vwap, volume, 4), 8), 6)
    "alpha058": ("(-ts_rank(ts_decay_linear(ts_corr(VWAP, VOLUME, 4), 8), 6))", "量价"),

    # Alpha #059: -ts_rank(decay_linear(correlation(vwap, volume, 5), 6), 4)
    "alpha059": ("(-ts_rank(ts_decay_linear(ts_corr(VWAP, VOLUME, 5), 6), 4))", "量价"),

    # Alpha #060: -((2 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - scale(rank(ts_argmax(close, 10))))
    "alpha060": (
        "(-((2 * cs_rank(((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)) * VOLUME))) "
        "- cs_rank(ts_argmax(CLOSE, 10))))", "价格"),

    # Alpha #061: (rank((vwap - min(vwap, 16))) - rank(correlation(vwap, mean(adv180), 18)))
    "alpha061": (
        "(cs_rank((VWAP - ts_min(VWAP, 16))) "
        "- cs_rank(ts_corr(VWAP, ts_mean(VOLUME, 180), 18)))", "波动率"),

    # Alpha #062: (rank(correlation(vwap, sum(adv20, 22), 10)) - rank(((rank(open) + rank(open)) - (rank(((high + low) / 2)) + rank(high)))))
    "alpha062": (
        "(cs_rank(ts_corr(VWAP, ts_sum(ts_mean(VOLUME, 20), 22), 10)) "
        "- cs_rank(((cs_rank(OPEN) + cs_rank(OPEN)) - (cs_rank(((HIGH + LOW) / 2)) + cs_rank(HIGH)))))", "波动率"),

    # Alpha #063: -(rank(decay_linear(delta(close, 2), 8)) - rank(decay_linear((ts_sum(volume, 10) / adv20), 1)))
    "alpha063": (
        "(-(cs_rank(ts_decay_linear((CLOSE - ts_delay(CLOSE, 2)), 8)) "
        "- cs_rank(ts_decay_linear((ts_sum(VOLUME, 10) / ts_mean(VOLUME, 10)), 1))))", "动量"),

    # Alpha #064: (rank(correlation(sum(((open*0.178)+(low*0.822)), 13), sum(mean(volume, 120), 13), 17)) - rank(((((high+low)/2)*0.178)+(vwap*0.822)) - delay(((((high+low)/2)*0.178)+(vwap*0.822)), 4)))
    "alpha064": (
        "(cs_rank(ts_corr(ts_sum(((OPEN * 0.178) + (LOW * 0.822)), 13), "
        "ts_sum(ts_mean(VOLUME, 120), 13), 17)) "
        "- cs_rank((((((HIGH + LOW) / 2) * 0.178) + (VWAP * 0.822)) "
        "- ts_delay(((((HIGH + LOW) / 2) * 0.178) + (VWAP * 0.822)), 4))))", "波动率"),

    # Alpha #065: (rank(correlation(((open*0.008)+(vwap*0.992)), sum(mean(volume, 60), 9), 6)) - rank((open - min(open, 14))))
    "alpha065": (
        "(cs_rank(ts_corr(((OPEN * 0.008) + (VWAP * 0.992)), "
        "ts_sum(ts_mean(VOLUME, 60), 9), 6)) "
        "- cs_rank((OPEN - ts_min(OPEN, 14))))", "波动率"),

    # Alpha #066: -(rank(decay_linear(delta(vwap, 4), 7)) + ts_rank(decay_linear(((low - vwap) / (open - ((high + low) / 2))), 11), 7))
    "alpha066": (
        "(-(cs_rank(ts_decay_linear((VWAP - ts_delay(VWAP, 4)), 7)) "
        "+ ts_rank(ts_decay_linear(((LOW - VWAP) / (OPEN - ((HIGH + LOW) / 2))), 11), 7)))", "波动率"),

    # Alpha #067: delay(mean((((close - delay(ma10(close), 1)) / delay(ma10(close), 1)) * 100), 2), 1)
    "alpha067": (
        "ts_delay(ts_mean((((CLOSE - ts_delay(ts_mean(CLOSE, 10), 1)) "
        "/ ts_delay(ts_mean(CLOSE, 10), 1)) * 100), 2), 1)", "技术指标"),

    # Alpha #068: delay(mean(((((high+low)/2) - delay(ma15((high+low)/2), 2)) / (0.5 * delay(ma15(abs(((high+low)/2) - delay(ma15((high+low)/2), 2))), 2))) * 100), 2), 1)
    "alpha068": (
        "ts_delay(ts_mean((((((HIGH + LOW) / 2) "
        "- ts_delay(ts_mean(((HIGH + LOW) / 2), 15), 2)) "
        "/ (0.5 * ts_delay(ts_mean(abs((((HIGH + LOW) / 2) "
        "- ts_delay(ts_mean(((HIGH + LOW) / 2), 15), 2))), 15), 2))) * 100), 2), 1)", "技术指标"),

    # Alpha #069: -signed_power(rank(max(delta(vwap, 3), 5)), ts_rank(correlation(((close*0.491)+(vwap*0.509)), adv20, 5), 9))
    "alpha069": (
        "(-signed_power(cs_rank(ts_max((VWAP - ts_delay(VWAP, 3)), 5)), "
        "ts_rank(ts_corr(((CLOSE * 0.491) + (VWAP * 0.509)), ts_mean(VOLUME, 20), 5), 9)))", "动量"),

    # Alpha #070: delay(mean((((close - delay(ma10(close), 2)) / delay(ma10(close), 2)) * 100), 2), 1)
    "alpha070": (
        "ts_delay(ts_mean((((CLOSE - ts_delay(ts_mean(CLOSE, 10), 2)) "
        "/ ts_delay(ts_mean(CLOSE, 10), 2)) * 100), 2), 1)", "技术指标"),

    # Alpha #071: -rank(decay_linear(correlation(ts_rank(close, 3), cs_rank(volume), 5), 7))
    "alpha071": (
        "(-cs_rank(ts_decay_linear(ts_corr(ts_rank(CLOSE, 3), cs_rank(VOLUME), 5), 7)))", "动量"),

    # Alpha #072: ts_rank(decay_linear(correlation(((high + low) / 2), mean(volume, 40), 9), 4), 2)
    "alpha072": (
        "ts_rank(ts_decay_linear(ts_corr(((HIGH + LOW) / 2), ts_mean(VOLUME, 40), 9), 4), 2)", "波动率"),

    # Alpha #073: ts_rank(decay_linear(correlation(((high + low) / 2), mean(volume, 40), 5), 3), 2)
    "alpha073": (
        "ts_rank(ts_decay_linear(ts_corr(((HIGH + LOW) / 2), ts_mean(VOLUME, 40), 5), 3), 2)", "波动率"),

    # Alpha #074: (rank(correlation(close, sum(mean(volume, 30), 37), 15)) - rank(correlation(rank(((high*0.026)+(vwap*0.974))), rank(volume), 11)))
    "alpha074": (
        "(cs_rank(ts_corr(CLOSE, ts_sum(ts_mean(VOLUME, 30), 37), 15)) "
        "- cs_rank(ts_corr(cs_rank(((HIGH * 0.026) + (VWAP * 0.974))), cs_rank(VOLUME), 11)))", "动量"),

    # Alpha #075: (rank(correlation(vwap, volume, 4)) - rank(correlation(rank(low), rank(mean(volume, 50)), 12)))
    "alpha075": (
        "(cs_rank(ts_corr(VWAP, VOLUME, 4)) "
        "- cs_rank(ts_corr(cs_rank(LOW), cs_rank(ts_mean(VOLUME, 50)), 12)))", "波动率"),

    # Alpha #076: ts_rank(decay_linear(correlation(vwap, volume, 17), 20), 3)
    "alpha076": ("ts_rank(ts_decay_linear(ts_corr(VWAP, VOLUME, 17), 20), 3)", "波动率"),

    # Alpha #077: ts_rank(decay_linear(((((high + low) / 2) + high) - (vwap + high)), 9), 4)
    "alpha077": (
        "ts_rank(ts_decay_linear(((((HIGH + LOW) / 2) + HIGH) - (VWAP + HIGH)), 9), 4)", "技术指标"),

    # Alpha #078: signed_power(rank(correlation(sum(((low*0.352)+(vwap*0.648)), 20), sum(mean(volume, 40), 20), 7)), rank(correlation(rank(vwap), rank(volume), 6)))
    "alpha078": (
        "signed_power(cs_rank(ts_corr(ts_sum(((LOW * 0.352) + (VWAP * 0.648)), 20), "
        "ts_sum(ts_mean(VOLUME, 40), 20), 7)), "
        "cs_rank(ts_corr(cs_rank(VWAP), cs_rank(VOLUME), 6)))", "波动率"),

    # Alpha #079: signed_power(rank((((close*0.607)+(open*0.393)) - delay(((close*0.607)+(open*0.393)), 4))), ts_rank(correlation(ts_rank(vwap, 3), ts_rank(mean(volume, 81), 9), 12), 3))
    "alpha079": (
        "signed_power(cs_rank((((CLOSE * 0.607) + (OPEN * 0.393)) "
        "- ts_delay(((CLOSE * 0.607) + (OPEN * 0.393)), 4))), "
        "ts_rank(ts_corr(ts_rank(VWAP, 3), ts_rank(ts_mean(VOLUME, 81), 9), 12), 3))", "动量"),

    # Alpha #080: -(rank(sign((((open*0.868)+(high*0.132)) - delay(((open*0.868)+(high*0.132)), 4)))) * ts_rank(correlation(high, mean(volume, 10), 5), 6))
    "alpha080": (
        "(-(cs_rank(sign((((OPEN * 0.868) + (HIGH * 0.132)) "
        "- ts_delay(((OPEN * 0.868) + (HIGH * 0.132)), 4)))) "
        "* ts_rank(ts_corr(HIGH, ts_mean(VOLUME, 10), 5), 6)))", "动量"),

    # Alpha #081: -(rank(log(rank(signed_power(rank(correlation(vwap, sum(mean(volume, 10), 50), 8)), 4)))) - rank(correlation(rank(vwap), rank(volume), 5)))
    "alpha081": (
        "(-(cs_rank(log(cs_rank(signed_power(cs_rank(ts_corr(VWAP, "
        "ts_sum(ts_mean(VOLUME, 10), 50), 8)), 4)))) "
        "- cs_rank(ts_corr(cs_rank(VWAP), cs_rank(VOLUME), 5))))", "动量"),

    # Alpha #082: -min(rank(decay_linear(delta(open, 1), 15)), 5)
    "alpha082": ("(-ts_min(cs_rank(ts_decay_linear((OPEN - ts_delay(OPEN, 1)), 15)), 5))", "价格"),

    # Alpha #083: ((rank(delay(((high - low) / (mean(close, 5))), 2)) * rank(rank(volume))) / (((high - low) / (mean(close, 5))) / (vwap - close)))
    "alpha083": (
        "((cs_rank(ts_delay(((HIGH - LOW) / (ts_mean(CLOSE, 5))), 2)) * cs_rank(cs_rank(VOLUME))) "
        "/ (((HIGH - LOW) / (ts_mean(CLOSE, 5))) / (VWAP - CLOSE)))", "量价"),

    # Alpha #084: signed_power(ts_rank((vwap - max(vwap, 15)), 21), delta(close, 5))
    "alpha084": (
        "signed_power(ts_rank((VWAP - ts_max(VWAP, 15)), 21), (CLOSE - ts_delay(CLOSE, 5)))", "价格"),

    # Alpha #085: signed_power(rank(correlation(((high*0.877)+(close*0.123)), mean(volume, 30), 10)), rank(correlation(ts_rank(((high+low)/2), 4), ts_rank(volume, 10), 15)))
    "alpha085": (
        "signed_power(cs_rank(ts_corr(((HIGH * 0.877) + (CLOSE * 0.123)), "
        "ts_mean(VOLUME, 30), 10)), "
        "cs_rank(ts_corr(ts_rank(((HIGH + LOW) / 2), 4), ts_rank(VOLUME, 10), 15)))", "波动率"),

    # Alpha #086: -(ts_rank(correlation(close, sum(adv20, 15), 6), 20) - rank(((open + close) - (vwap + open))))
    "alpha086": (
        "(-(ts_rank(ts_corr(CLOSE, ts_sum(ts_mean(VOLUME, 20), 15), 6), 20) "
        "- cs_rank(((OPEN + CLOSE) - (VWAP + OPEN)))))", "动量"),

    # Alpha #087: delay(mean(delay(mean((((close - delay(ma10(close), 1)) / delay(ma10(close), 1)) * 100), 3), 1), 3), 1)
    "alpha087": (
        "ts_delay(ts_mean(ts_delay(ts_mean((((CLOSE - ts_delay(ts_mean(CLOSE, 8), 1)) "
        "/ ts_delay(ts_mean(CLOSE, 8), 1)) * 100), 3), 1), 3), 1)", "技术指标"),

    # Alpha #088: rank(decay_linear(((rank(open) + rank(low)) - (rank(high) + rank(close))), 8))
    "alpha088": (
        "cs_rank(ts_decay_linear(((cs_rank(OPEN) + cs_rank(LOW)) - (cs_rank(HIGH) + cs_rank(CLOSE))), 8))", "价格"),

    # Alpha #089: (ts_rank(decay_linear(correlation(low, mean(volume, 10), 7), 6), 4) - ts_rank(decay_linear(delta(vwap, 3), 10), 15))
    "alpha089": (
        "(ts_rank(ts_decay_linear(ts_corr(LOW, ts_mean(VOLUME, 10), 7), 6), 4) "
        "- ts_rank(ts_decay_linear((VWAP - ts_delay(VWAP, 3)), 10), 15))", "波动率"),

    # Alpha #090: -(rank((close - max(close, 5))) * ts_rank(correlation(vwap, mean(volume, 40), 9), 7))
    "alpha090": (
        "(-(cs_rank((CLOSE - ts_max(CLOSE, 5))) "
        "* ts_rank(ts_corr(VWAP, ts_mean(VOLUME, 40), 9), 7)))", "动量"),

    # Alpha #091: delay(mean(delay(mean((((close - delay(ma8(close), 1)) / delay(ma8(close), 1)) * 100), 3), 1), 3), 1)
    "alpha091": (
        "ts_delay(ts_mean(ts_delay(ts_mean((((CLOSE - ts_delay(ts_mean(CLOSE, 8), 1)) "
        "/ ts_delay(ts_mean(CLOSE, 8), 1)) * 100), 3), 1), 3), 1)", "技术指标"),

    # Alpha #092: ts_rank(decay_linear(correlation((((high + low) / 2) + close), mean(volume, 10), 12), 7), 16)
    "alpha092": (
        "ts_rank(ts_decay_linear(ts_corr((((HIGH + LOW) / 2) + CLOSE), "
        "ts_mean(VOLUME, 10), 12), 7), 16)", "波动率"),

    # Alpha #093: ts_rank(decay_linear(correlation(vwap, mean(volume, 81), 17), 6), 6)
    "alpha093": ("ts_rank(ts_decay_linear(ts_corr(VWAP, ts_mean(VOLUME, 81), 17), 6), 6)", "波动率"),

    # Alpha #094: -(rank((vwap - min(vwap, 12))) * ts_rank(correlation(ts_rank(vwap, 20), ts_rank(mean(volume, 60), 4), 18), 3))
    "alpha094": (
        "(-(cs_rank((VWAP - ts_min(VWAP, 12))) "
        "* ts_rank(ts_corr(ts_rank(VWAP, 20), ts_rank(ts_mean(VOLUME, 60), 4), 18), 3)))", "波动率"),

    # Alpha #095: (rank((open - min(open, 12))) - ts_rank(signed_power(rank(correlation(sum(((high + low) / 2), 19), sum(mean(volume, 40), 19), 13)), 5), 12))
    "alpha095": (
        "(cs_rank((OPEN - ts_min(OPEN, 12))) "
        "- ts_rank(signed_power(cs_rank(ts_corr(ts_sum(((HIGH + LOW) / 2), 19), "
        "ts_sum(ts_mean(VOLUME, 40), 19), 13)), 5), 12))", "动量"),

    # Alpha #096: -ts_rank(decay_linear(correlation(rank(vwap), rank(volume), 6), 7), 5)
    "alpha096": ("(-ts_rank(ts_decay_linear(ts_corr(cs_rank(VWAP), cs_rank(VOLUME), 6), 7), 5))", "波动率"),

    # Alpha #097: -(rank(decay_linear((((low*0.721)+(vwap*0.279)) - delay(((low*0.721)+(vwap*0.279)), 3)), 20)) - ts_rank(decay_linear(ts_rank(correlation(ts_rank(low, 8), ts_rank(mean(volume, 60), 17), 5), 19), 15), 7))
    "alpha097": (
        "(-(cs_rank(ts_decay_linear((((LOW * 0.721) + (VWAP * 0.279)) "
        "- ts_delay(((LOW * 0.721) + (VWAP * 0.279)), 3)), 20)) "
        "- ts_rank(ts_decay_linear(ts_rank(ts_corr(ts_rank(LOW, 8), "
        "ts_rank(ts_mean(VOLUME, 60), 17), 5), 19), 15), 7)))", "动量"),

    # Alpha #098: (rank(decay_linear(correlation(vwap, sum(mean(volume, 5), 26), 5), 7)) - 5.236)
    "alpha098": (
        "(cs_rank(ts_decay_linear(ts_corr(VWAP, ts_sum(ts_mean(VOLUME, 5), 26), 5), 7)) - 5.236)", "量价"),

    # Alpha #099: -(rank(correlation(sum(((high + low) / 2), 20), sum(mean(volume, 60), 20), 9)) - rank(correlation(low, volume, 6)))
    "alpha099": (
        "(-(cs_rank(ts_corr(ts_sum(((HIGH + LOW) / 2), 20), "
        "ts_sum(ts_mean(VOLUME, 60), 20), 9)) "
        "- cs_rank(ts_corr(LOW, VOLUME, 6))))", "波动率"),

    # Alpha #100: -((1.5 * scale(rank(((((close - low) - (high - close)) / (high - low)) * volume)))) - (correlation(close, rank(adv20), 5) - rank(ts_argmin(close, 30))))
    "alpha100": (
        "(-((1.5 * cs_rank(((((CLOSE - LOW) - (HIGH - CLOSE)) / (HIGH - LOW)) * VOLUME))) "
        "- (ts_corr(CLOSE, cs_rank(ts_mean(VOLUME, 20)), 5) - cs_rank(ts_argmin(CLOSE, 30)))))", "价格"),

    # Alpha #101: ((close - open) / ((high - low) + 0.001))
    "alpha101": ("((CLOSE - OPEN) / ((HIGH - LOW) + 0.001))", "价格"),
}