"""
kb03/formulas/gt191.py — 国泰君安 191 因子 (V4 AST 语法)

结构: ALPHA191[name] = (v4_expr, category)
每个因子一个条目，公式和分类在一起。

翻译规则同 wq101.py。GT191 的 ifelse 条件全部是 `close > delay(close,1)` 型（即正收益判断），
非零校验→显式 > 比较。
"""
from typing import Dict, Tuple

ALPHA191: Dict[str, Tuple[str, str]] = {
    # ── 动量类 (gtja_001 ~ gtja_030) ──
    "gtja_001": ("cs_rank((CLOSE - ts_delay(CLOSE, 5)))", "动量"),
    "gtja_002": ("((CLOSE - ts_delay(CLOSE, 20)) * (VOLUME - ts_delay(VOLUME, 5)))", "动量"),
    "gtja_003": ("cs_rank(((CLOSE - ts_delay(CLOSE, 1)) * VOLUME))", "动量"),
    "gtja_004": ("((CLOSE - ts_delay(CLOSE, 5)) * (VOLUME / ts_mean(VOLUME, 20)))", "动量"),
    "gtja_005": ("cs_rank(((CLOSE - ts_delay(CLOSE, 1)) * (VOLUME / ts_mean(VOLUME, 20))))", "动量"),
    "gtja_006": ("(ts_rank((CLOSE - ts_delay(CLOSE, 1)), 10) * (-ts_std((CLOSE - ts_delay(CLOSE, 1)), 10)))", "动量"),
    "gtja_007": ("ts_rank((CLOSE - ts_delay(CLOSE, 1)), 20)", "动量"),
    "gtja_008": ("(ts_rank((CLOSE - ts_delay(CLOSE, 1)), 5) + ts_rank(VOLUME, 5))", "动量"),
    "gtja_009": ("ts_decay_linear((CLOSE - ts_delay(CLOSE, 1)), 4)", "动量"),
    "gtja_010": ("((CLOSE - ts_delay(CLOSE, 5)) * (-ts_std((CLOSE - ts_delay(CLOSE, 1)), 20)))", "动量"),
    "gtja_011": ("((VOLUME - ts_delay(VOLUME, 5)) * (CLOSE - ts_delay(CLOSE, 5)))", "动量"),
    "gtja_012": ("ts_decay_linear((CLOSE - ts_delay(CLOSE, 1)), 20)", "动量"),
    "gtja_013": ("((((HIGH + LOW) / 2) - ts_delay(((HIGH + LOW) / 2), 20)) - (CLOSE - ts_delay(CLOSE, 20)))", "动量"),
    "gtja_014": ("(-cs_rank(ts_argmax(CLOSE, 30)))", "动量"),
    "gtja_015": ("cs_rank(ts_argmin(CLOSE, 30))", "动量"),
    "gtja_016": ("(ts_max(CLOSE, 20) - ts_min(CLOSE, 20))", "动量"),
    "gtja_017": ("((CLOSE - ts_min(CLOSE, 20)) / (ts_max(CLOSE, 20) - ts_min(CLOSE, 20)))", "动量"),
    "gtja_018": ("((CLOSE - ts_delay(CLOSE, 10)) * ((CLOSE - ts_delay(CLOSE, 1)) - ts_delay((CLOSE - ts_delay(CLOSE, 1)), 1)))", "动量"),
    "gtja_019": ("(ts_mean((CLOSE - ts_delay(CLOSE, 1)), 20) - ts_std((CLOSE - ts_delay(CLOSE, 1)), 20))", "动量"),
    "gtja_020": ("cs_rank((CLOSE - VWAP))", "动量"),
    "gtja_021": ("cs_rank((CLOSE - ts_delay(ts_mean(CLOSE, 20), 1)))", "动量"),
    "gtja_022": ("((CLOSE - ts_delay(ts_mean(CLOSE, 20), 1)) / ts_delay(ts_mean(CLOSE, 20), 1))", "动量"),
    "gtja_023": ("(CLOSE / ts_delay(ts_mean(CLOSE, 20), 1))", "动量"),
    "gtja_024": ("((CLOSE - ts_delay(CLOSE, 20)) / ts_std((CLOSE - ts_delay(CLOSE, 1)), 20))", "动量"),
    "gtja_025": ("((CLOSE - ts_delay(CLOSE, 5)) * ts_corr(CLOSE, VOLUME, 20))", "动量"),
    "gtja_026": ("(CLOSE - ts_mean(CLOSE, 20))", "动量"),
    "gtja_027": ("(CLOSE - ts_max(CLOSE, 5))", "动量"),
    "gtja_028": ("((ts_max(CLOSE, 20) - ts_min(CLOSE, 20)) / ts_mean(CLOSE, 20))", "动量"),
    "gtja_029": ("((CLOSE - ts_delay(CLOSE, 1)) * ts_rank(VOLUME, 20))", "动量"),
    "gtja_030": ("(ts_rank(CLOSE, 10) - ts_rank(VOLUME, 10))", "动量"),

    # ── 反转类 (gtja_031 ~ gtja_060) ──
    "gtja_031": ("(-ts_rank((CLOSE - ts_delay(CLOSE, 1)), 5))", "反转"),
    "gtja_032": ("(-((CLOSE - ts_delay(CLOSE, 20)) * ts_rank(VOLUME, 5)))", "反转"),
    "gtja_033": ("(-((CLOSE - VWAP) * ts_rank(VOLUME, 5)))", "反转"),
    "gtja_034": ("(-ts_decay_linear((CLOSE - ts_delay(CLOSE, 1)), 10))", "反转"),
    "gtja_035": ("(-ts_zscore((CLOSE - ts_delay(CLOSE, 1)), 20))", "反转"),
    "gtja_036": ("(-ts_rank((CLOSE - ts_delay(CLOSE, 1)), 30))", "反转"),
    "gtja_037": ("(-ts_rank(CLOSE, 45))", "反转"),
    "gtja_038": ("(-ts_rank(CLOSE, 60))", "反转"),
    "gtja_039": ("(-((CLOSE - ts_delay(CLOSE, 5)) * (VOLUME / ts_mean(VOLUME, 20))))", "反转"),
    "gtja_040": ("(-cs_rank(((CLOSE - ts_delay(CLOSE, 1)) * (VOLUME / ts_delay(VOLUME, 1)))))", "反转"),
    "gtja_041": ("((CLOSE - LOW) / ((HIGH - LOW) + 0.001))", "反转"),
    "gtja_042": (
        "(ts_mean((CLOSE - ts_delay(CLOSE, 1) "
        "if CLOSE > ts_delay(CLOSE, 1) else 0.0), 14) "
        "/ (ts_mean(abs(CLOSE - ts_delay(CLOSE, 1)), 14) + 0.001))", "反转"),
    "gtja_043": ("(-(HIGH - ts_delay(HIGH, 10)))", "反转"),
    "gtja_044": ("(-cs_rank(((CLOSE - OPEN) * VOLUME)))", "反转"),
    "gtja_045": ("(-ts_corr(ts_rank(CLOSE, 5), ts_rank(VOLUME, 5), 10))", "反转"),
    "gtja_046": ("(-ts_cov(CLOSE, VOLUME, 10))", "反转"),
    "gtja_047": ("(-ts_rank((((CLOSE - OPEN) / (OPEN + 0.001)) * VOLUME), 5))", "反转"),
    "gtja_048": ("cs_rank(ts_argmax(CLOSE, 60))", "反转"),
    "gtja_049": ("(-cs_rank(ts_argmin(CLOSE, 60)))", "反转"),
    "gtja_050": ("(-((CLOSE - ts_min(CLOSE, 20)) / ((ts_max(CLOSE, 20) - ts_min(CLOSE, 20)) + 0.001)))", "反转"),
    "gtja_051": ("(-ts_rank((CLOSE - ts_delay(CLOSE, 1)), 20))", "反转"),
    "gtja_052": ("(cs_rank((-(CLOSE - ts_delay(CLOSE, 1)))) * ts_rank(VOLUME, 20))", "反转"),
    "gtja_053": ("((OPEN - ts_delay(CLOSE, 1)) / (ts_delay(CLOSE, 1) + 0.001))", "反转"),
    "gtja_054": ("ts_corr((-(CLOSE - ts_delay(CLOSE, 1))), ts_rank(VOLUME, 5), 20)", "反转"),
    "gtja_055": ("(-ts_rank((CLOSE - ts_delay(CLOSE, 60)), 10))", "反转"),
    "gtja_056": ("(-(CLOSE - ts_max(CLOSE, 20)))", "反转"),
    "gtja_057": ("(-ts_rank((CLOSE / ts_delay(ts_mean(CLOSE, 20), 1)), 10))", "反转"),
    "gtja_058": (
        "(-((CLOSE - ts_delay(CLOSE, 1)) "
        "if ts_delay(CLOSE, 1) > CLOSE else 0.0))", "反转"),
    "gtja_059": (
        "(-(((CLOSE - ts_delay(CLOSE, 5)) / (ts_std((CLOSE - ts_delay(CLOSE, 1)), 20) + 0.001)) "
        "* ts_rank(VOLUME, 20)))", "反转"),
    "gtja_060": ("(cs_rank((-(CLOSE - ts_delay(CLOSE, 10)))) * ts_corr(CLOSE, VOLUME, 10))", "反转"),

    # ── 波动率类 (gtja_061 ~ gtja_080) ──
    "gtja_061": ("(-ts_std((CLOSE - ts_delay(CLOSE, 1)), 20))", "波动率"),
    "gtja_062": ("(-ts_std((CLOSE - ts_delay(CLOSE, 1)), 10))", "波动率"),
    "gtja_063": ("(-ts_std((CLOSE - ts_delay(CLOSE, 1)), 60))", "波动率"),
    "gtja_064": ("(-ts_zscore((CLOSE - ts_delay(CLOSE, 1)), 20))", "波动率"),
    "gtja_065": ("(-ts_zscore((CLOSE - ts_delay(CLOSE, 1)), 10))", "波动率"),
    "gtja_066": ("(-ts_zscore((CLOSE - ts_delay(CLOSE, 1)), 60))", "波动率"),
    "gtja_067": (
        "(-ts_std((0.0 if CLOSE > ts_delay(CLOSE, 1) "
        "else (CLOSE - ts_delay(CLOSE, 1))), 20))", "波动率"),
    "gtja_068": ("(-((HIGH - LOW) / (CLOSE + 0.001)))", "波动率"),
    "gtja_069": ("(-(ts_mean((HIGH - LOW), 20) / CLOSE))", "波动率"),
    "gtja_070": ("(ts_std((CLOSE - ts_delay(CLOSE, 1)), 10) - ts_delay(ts_std((CLOSE - ts_delay(CLOSE, 1)), 10), 1))", "波动率"),
    "gtja_071": ("(-ts_cov((CLOSE - ts_delay(CLOSE, 1)), ts_delay((CLOSE - ts_delay(CLOSE, 1)), 1), 20))", "波动率"),
    "gtja_072": ("(-((HIGH - LOW) / (ts_mean(CLOSE, 20) + 0.001)))", "波动率"),
    "gtja_073": ("(-(ts_std(VOLUME, 20) / (ts_mean(VOLUME, 20) + 0.001)))", "波动率"),
    "gtja_074": ("((-ts_std((CLOSE - ts_delay(CLOSE, 1)), 20)) - ts_rank(VOLUME, 20))", "波动率"),
    "gtja_075": ("(-ts_rank(ts_std((CLOSE - ts_delay(CLOSE, 1)), 20), 20))", "波动率"),
    # 对日收益自身回归: y=x=日收益, rettype=0(斜率)→ 等价于 ts_slope(ΔlogC,60)
    "gtja_076": ("(-ts_reg_slope((CLOSE - ts_delay(CLOSE, 1)), (CLOSE - ts_delay(CLOSE, 1)), 60))", "波动率"),
    "gtja_077": (
        "(-ts_std(((CLOSE - ts_min(CLOSE, 20)) "
        "/ ((ts_max(CLOSE, 20) - ts_min(CLOSE, 20)) + 0.001)), 10))", "波动率"),
    "gtja_078": ("(-ts_std((CLOSE - ts_delay(ts_mean(CLOSE, 10), 1)), 20))", "波动率"),
    "gtja_079": ("(-ts_rank(ts_sum(abs((CLOSE - ts_delay(CLOSE, 1))), 20), 20))", "波动率"),
    "gtja_080": ("(-(ts_std(CLOSE, 20) / (ts_mean(CLOSE, 20) + 0.001)))", "波动率"),

    # ── 量价类 (gtja_081 ~ gtja_130) ──
    "gtja_081": ("(VOLUME / (ts_mean(VOLUME, 20) + 0.001))", "量价"),
    "gtja_082": ("(VOLUME / (ts_mean(VOLUME, 5) + 0.001))", "量价"),
    "gtja_083": ("(VOLUME / (ts_mean(VOLUME, 60) + 0.001))", "量价"),
    "gtja_084": ("ts_rank(VOLUME, 5)", "量价"),
    "gtja_085": ("ts_rank(VOLUME, 20)", "量价"),
    "gtja_086": ("(VOLUME - ts_delay(VOLUME, 1))", "量价"),
    "gtja_087": ("(VOLUME - ts_delay(VOLUME, 5))", "量价"),
    "gtja_088": ("(ts_sum(VOLUME, 5) / (ts_sum(VOLUME, 20) + 0.001))", "量价"),
    "gtja_089": ("(ts_sum(VOLUME, 10) / (ts_sum(VOLUME, 60) + 0.001))", "量价"),
    "gtja_090": ("(ts_mean(VOLUME, 5) - ts_mean(VOLUME, 20))", "量价"),
    "gtja_091": ("(ts_max(VOLUME, 20) / (VOLUME + 0.001))", "量价"),
    "gtja_092": ("((VOLUME / (ts_mean(VOLUME, 20) + 0.001)) * (CLOSE - ts_delay(CLOSE, 1)))", "量价"),
    "gtja_093": ("ts_corr(VOLUME, CLOSE, 10)", "量价"),
    "gtja_094": ("ts_corr(VOLUME, (CLOSE - ts_delay(CLOSE, 1)), 10)", "量价"),
    "gtja_095": ("ts_rank((VOLUME - ts_delay(VOLUME, 5)), 10)", "量价"),
    "gtja_096": ("ts_decay_linear((VOLUME - ts_mean(VOLUME, 20)), 10)", "量价"),
    "gtja_097": ("ts_zscore(VOLUME, 20)", "量价"),
    "gtja_098": ("(VOLUME * abs((CLOSE - ts_delay(CLOSE, 1))))", "量价"),
    "gtja_099": ("(-ts_rank(VOLUME, 10))", "量价"),
    "gtja_100": ("ts_std(VOLUME, 20)", "量价"),
    "gtja_101": ("cs_rank(VOLUME)", "量价"),
    "gtja_102": ("(sign((CLOSE - OPEN)) * VOLUME)", "量价"),
    "gtja_103": ("cs_rank((VOLUME * VWAP))", "量价"),
    "gtja_104": ("(((CLOSE - OPEN) * VOLUME) / (OPEN + 0.001))", "量价"),
    "gtja_105": ("ts_sum((sign((CLOSE - OPEN)) * VOLUME), 5)", "量价"),
    "gtja_106": ("ts_sum((sign((CLOSE - OPEN)) * VOLUME), 20)", "量价"),
    "gtja_107": ("(ts_sum((sign((CLOSE - OPEN)) * VOLUME), 5) - ts_delay(ts_sum((sign((CLOSE - OPEN)) * VOLUME), 5), 5))", "量价"),
    "gtja_108": ("ts_corr((sign((CLOSE - OPEN)) * VOLUME), (CLOSE - ts_delay(CLOSE, 1)), 20)", "量价"),
    "gtja_109": ("ts_rank(((CLOSE - OPEN) * VOLUME), 10)", "量价"),
    "gtja_110": ("(((CLOSE - OPEN) * VOLUME) / (ts_mean((CLOSE * VOLUME), 20) + 0.001))", "量价"),
    "gtja_111": ("((ts_sum(VOLUME, 5) / (ts_sum(VOLUME, 20) + 0.001)) * (CLOSE - ts_delay(CLOSE, 1)))", "量价"),
    "gtja_112": ("ts_mean((VOLUME * (CLOSE - ts_delay(CLOSE, 1))), 5)", "量价"),
    "gtja_113": ("(ts_mean((VOLUME * (CLOSE - ts_delay(CLOSE, 1))), 5) / (ts_std((CLOSE - ts_delay(CLOSE, 1)), 20) + 0.001))", "量价"),
    "gtja_114": ("ts_rank((VOLUME * VWAP), 20)", "量价"),
    "gtja_115": ("(sqrt((HIGH * LOW)) / CLOSE)", "量价"),
    "gtja_116": ("(-((HIGH - CLOSE) / (CLOSE + 0.001)))", "量价"),
    "gtja_117": ("((CLOSE - LOW) / (CLOSE + 0.001))", "量价"),
    "gtja_118": ("((CLOSE - OPEN) / ((HIGH - LOW) + 0.001))", "量价"),
    "gtja_119": (
        "((CLOSE - ts_mean(((HIGH + LOW) / 2), 20)) "
        "/ ((0.015 * ts_mean(abs((CLOSE - ts_mean(CLOSE, 20))), 20)) + 0.001))", "量价"),
    "gtja_120": (
        "((ts_mean((CLOSE - ts_delay(CLOSE, 1) "
        "if CLOSE > ts_delay(CLOSE, 1) else 0.0), 14) "
        "/ (ts_mean(abs(CLOSE - ts_delay(CLOSE, 1)), 14) + 0.001)) - 0.5)", "量价"),
    "gtja_121": ("(-((ts_max(HIGH, 14) - CLOSE) / ((ts_max(HIGH, 14) - ts_min(LOW, 14)) + 0.001)))", "量价"),
    "gtja_122": ("ts_decay_linear((sign((CLOSE - OPEN)) * VOLUME), 10)", "量价"),
    "gtja_123": (
        "(((VOLUME - ts_mean(VOLUME, 20)) * (CLOSE - ts_delay(CLOSE, 1))) "
        "if CLOSE > ts_delay(CLOSE, 1) "
        "else (-(VOLUME - ts_mean(VOLUME, 20)) * abs(CLOSE - ts_delay(CLOSE, 1))))", "量价"),
    "gtja_124": ("((-ts_rank(VOLUME, 10)) - ts_rank((CLOSE - ts_delay(CLOSE, 1)), 10))", "量价"),
    "gtja_125": ("(ts_rank(VOLUME, 10) * ts_rank((CLOSE - ts_delay(CLOSE, 1)), 5))", "量价"),
    "gtja_126": ("(ts_delay(ts_mean(VOLUME, 5), 1) - ts_delay(ts_mean(VOLUME, 20), 1))", "量价"),
    "gtja_127": ("((sign((CLOSE - OPEN)) * VOLUME) * ((HIGH - LOW) / (CLOSE + 0.001)))", "量价"),
    "gtja_128": ("cs_rank(ts_argmax(VOLUME, 20))", "量价"),
    "gtja_129": ("ts_std((VOLUME * (CLOSE - ts_delay(CLOSE, 1))), 20)", "量价"),
    "gtja_130": ("(ts_decay_linear(CLOSE, 5) / (ts_decay_linear(CLOSE, 20) + 0.001))", "量价"),

    # ── 估值类 (gtja_131 ~ gtja_140) ──
    "gtja_131": ("(-ts_mean((CLOSE * VOLUME), 20))", "估值"),
    "gtja_132": ("(-cs_rank((CLOSE * VOLUME)))", "估值"),
    "gtja_133": ("((-cs_rank((CLOSE * VOLUME))) * (CLOSE - ts_delay(CLOSE, 1)))", "估值"),
    "gtja_134": ("(CLOSE - ts_mean(CLOSE, 60))", "估值"),
    "gtja_135": ("(CLOSE / (ts_max(CLOSE, 252) + 0.001))", "估值"),
    "gtja_136": ("(CLOSE - ts_delay(CLOSE, 60))", "估值"),
    "gtja_137": ("(CLOSE - ts_delay(CLOSE, 120))", "估值"),
    "gtja_138": ("ts_rank((CLOSE / (ts_max(CLOSE, 252) + 0.001)), 20)", "估值"),
    "gtja_139": ("cs_rank(ts_argmax(CLOSE, 252))", "估值"),
    "gtja_140": ("(-(CLOSE - ts_delay(CLOSE, 252)))", "估值"),

    # ── 量价类 (gtja_141 ~ gtja_160) — 补充完整 191 因子 ──
    # 来源: 国泰君安《基于短周期价量特征的多因子选股体系》(2017)
    # Alpha141: (-1 * TSRANK(MIN(VWAP-LOW,0),5))
    "gtja_141": ("(-ts_rank(ts_min(VWAP - LOW, 5)))", "量价"),
    # Alpha142: RANK(REGBETA(Δlog(CLOSE), Δlog(VOLUME),5))*-1 → 收益率~换手率的5日回归斜率(负号=放量下跌)
    "gtja_142": ("(-cs_rank(ts_reg_slope(ts_delta(log(CLOSE), 1), ts_delta(log(VOLUME), 1), 5)))", "量价"),
    # Alpha143: RANK(REGRESI(Δlog(CLOSE), Δlog(VOLUME),5))*-1 → 收益率~换手率的5日回归残差
    "gtja_143": ("(-cs_rank(ts_reg_resid(ts_delta(log(CLOSE), 1), ts_delta(log(VOLUME), 1), 5)))", "量价"),
    # Alpha144: RANK(REGBETA(Δlog(HIGH), Δlog(VOLUME),5))*-1
    "gtja_144": ("(-cs_rank(ts_reg_slope(ts_delta(log(HIGH), 1), ts_delta(log(VOLUME), 1), 5)))", "量价"),
    # Alpha145: RANK(REGRESI(Δlog(HIGH), Δlog(VOLUME),5))*-1
    "gtja_145": ("(-cs_rank(ts_reg_resid(ts_delta(log(HIGH), 1), ts_delta(log(VOLUME), 1), 5)))", "量价"),
    # Alpha146: RANK(REGBETA(Δlog(LOW), Δlog(VOLUME),5))*-1
    "gtja_146": ("(-cs_rank(ts_reg_slope(ts_delta(log(LOW), 1), ts_delta(log(VOLUME), 1), 5)))", "量价"),
    # Alpha147: RANK(REGRESI(Δlog(LOW), Δlog(VOLUME),5))*-1
    "gtja_147": ("(-cs_rank(ts_reg_resid(ts_delta(log(LOW), 1), ts_delta(log(VOLUME), 1), 5)))", "量价"),
    # Alpha148: RANK(REGBETA(Δlog(VWAP), Δlog(VOLUME),5))*-1
    "gtja_148": ("(-cs_rank(ts_reg_slope(ts_delta(log(VWAP), 1), ts_delta(log(VOLUME), 1), 5)))", "量价"),
    # Alpha149: RANK(REGRESI(Δlog(VWAP), Δlog(VOLUME),5))*-1 → VWAP~VOLUME 回归残差
    "gtja_149": ("(-cs_rank(ts_reg_resid(ts_delta(log(VWAP), 1), ts_delta(log(VOLUME), 1), 5)))", "量价"),
    # Alpha150: RANK(COUNT(CLOSE>DELAY(CLOSE,1),5))*-1
    "gtja_150": ("(-cs_rank(ts_sum((CLOSE > ts_delay(CLOSE, 1)), 5)))", "量价"),
    # Alpha151: RANK(COUNT(CLOSE>DELAY(OPEN,1),5))*-1
    "gtja_151": ("(-cs_rank(ts_sum((CLOSE > ts_delay(OPEN, 1)), 5)))", "量价"),
    # Alpha152: RANK(COUNT(CLOSE>DELAY(HIGH,1),5))*-1
    "gtja_152": ("(-cs_rank(ts_sum((CLOSE > ts_delay(HIGH, 1)), 5)))", "量价"),
    # Alpha153: RANK(COUNT(HIGH>DELAY(LOW,1),5))*-1
    "gtja_153": ("(-cs_rank(ts_sum((HIGH > ts_delay(LOW, 1)), 5)))", "量价"),
    # Alpha154: RANK(COUNT(LOW>DELAY(VWAP,1),5))*-1
    "gtja_154": ("(-cs_rank(ts_sum((LOW > ts_delay(VWAP, 1)), 5)))", "量价"),
    # Alpha155: RANK(COUNT(VWAP>DELAY(VWAP,1),5))*-1
    "gtja_155": ("(-cs_rank(ts_sum((VWAP > ts_delay(VWAP, 1)), 5)))", "量价"),
    # Alpha156: RANK(COUNT(VOLUME>DELAY(CLOSE,1),5))*-1
    "gtja_156": ("(-cs_rank(ts_sum((VOLUME > ts_delay(CLOSE, 1)), 5)))", "量价"),
    # Alpha157: RANK(COUNT(VOLUME>DELAY(OPEN,1),5))*-1
    "gtja_157": ("(-cs_rank(ts_sum((VOLUME > ts_delay(OPEN, 1)), 5)))", "量价"),
    # Alpha158: RANK(COUNT(VOLUME>DELAY(HIGH,1),5))*-1
    "gtja_158": ("(-cs_rank(ts_sum((VOLUME > ts_delay(HIGH, 1)), 5)))", "量价"),
    # Alpha159: RANK(COUNT(VOLUME>DELAY(LOW,1),5))*-1
    "gtja_159": ("(-cs_rank(ts_sum((VOLUME > ts_delay(LOW, 1)), 5)))", "量价"),
    # Alpha160: RANK(COUNT(VOLUME>DELAY(VWAP,1),5))*-1
    "gtja_160": ("(-cs_rank(ts_sum((VOLUME > ts_delay(VWAP, 1)), 5)))", "量价"),

    # ── 技术指标类 (gtja_161 ~ gtja_191) ──
    "gtja_161": ("(ts_delay(ts_mean(CLOSE, 12), 1) - ts_delay(ts_mean(CLOSE, 26), 1))", "技术指标"),
    "gtja_162": ("ts_delay(ts_mean((ts_delay(ts_mean(CLOSE, 12), 1) - ts_delay(ts_mean(CLOSE, 26), 1)), 9), 1)", "技术指标"),
    "gtja_163": (
        "((ts_delay(ts_mean(CLOSE, 12), 1) - ts_delay(ts_mean(CLOSE, 26), 1)) "
        "- ts_delay(ts_mean((ts_delay(ts_mean(CLOSE, 12), 1) - ts_delay(ts_mean(CLOSE, 26), 1)), 9), 1))", "技术指标"),
    "gtja_164": ("((2 * ts_std(CLOSE, 20)) / (ts_delay(ts_mean(CLOSE, 20), 1) + 0.001))", "技术指标"),
    "gtja_165": (
        "((CLOSE - (ts_delay(ts_mean(CLOSE, 20), 1) - (2 * ts_std(CLOSE, 20)))) "
        "/ ((4 * ts_std(CLOSE, 20)) + 0.001))", "技术指标"),
    "gtja_166": ("(CLOSE - (ts_delay(ts_mean(CLOSE, 20), 1) + (2 * ts_std(CLOSE, 20))))", "技术指标"),
    "gtja_167": ("(CLOSE - (ts_delay(ts_mean(CLOSE, 20), 1) - (2 * ts_std(CLOSE, 20))))", "技术指标"),
    "gtja_168": ("((CLOSE - ts_delay(ts_mean(CLOSE, 20), 1)) / (ts_std(CLOSE, 20) + 0.001))", "技术指标"),
    "gtja_169": ("(ts_delay(ts_mean(CLOSE, 5), 1) - ts_delay(ts_mean(CLOSE, 10), 1))", "技术指标"),
    "gtja_170": ("(ts_delay(ts_mean(CLOSE, 5), 1) - ts_delay(ts_mean(CLOSE, 20), 1))", "技术指标"),
    "gtja_171": ("(ts_delay(ts_mean(CLOSE, 10), 1) - ts_delay(ts_mean(CLOSE, 30), 1))", "技术指标"),
    "gtja_172": ("(ts_delay(ts_mean(CLOSE, 5), 1) - ts_delay(ts_mean(CLOSE, 60), 1))", "技术指标"),
    "gtja_173": (
        "((ts_delay(ts_mean(CLOSE, 5), 1) - ts_delay(ts_mean(CLOSE, 10), 1)) "
        "+ (ts_delay(ts_mean(CLOSE, 10), 1) - ts_delay(ts_mean(CLOSE, 20), 1)))", "技术指标"),
    "gtja_174": ("((CLOSE - OPEN) / (OPEN + 0.001))", "技术指标"),
    "gtja_175": ("((HIGH - OPEN) / (OPEN + 0.001))", "技术指标"),
    "gtja_176": ("(-((OPEN - LOW) / (OPEN + 0.001)))", "技术指标"),
    "gtja_177": ("(ts_mean((HIGH - LOW), 5) / CLOSE)", "技术指标"),
    "gtja_178": ("(ts_mean((HIGH - LOW), 20) / CLOSE)", "技术指标"),
    "gtja_179": ("((CLOSE - VWAP) / (VWAP + 0.001))", "技术指标"),
    "gtja_180": ("cs_rank(VWAP)", "技术指标"),
    "gtja_181": ("(VOLUME if CLOSE > ts_delay(CLOSE, 1) else (-VOLUME))", "技术指标"),
    "gtja_182": ("(ts_rank(CLOSE, 10) - ts_rank(CLOSE, 60))", "技术指标"),
    "gtja_183": ("(ts_rank(CLOSE, 10) - ts_rank(CLOSE, 20))", "技术指标"),
    "gtja_184": (
        "((ts_delay(ts_mean(CLOSE, 5), 1) - ts_delay(ts_mean(CLOSE, 20), 1)) "
        "* (VOLUME / (ts_mean(VOLUME, 20) + 0.001)))", "技术指标"),
    "gtja_185": ("ts_sum((sign((CLOSE - ts_delay(CLOSE, 1))) * VOLUME), 20)", "技术指标"),
    "gtja_186": ("ts_corr(ts_delay(ts_mean(CLOSE, 5), 1), ts_delay(ts_mean(CLOSE, 20), 1), 30)", "技术指标"),
    "gtja_187": (
        "(((CLOSE - ts_delay(CLOSE, 1)) "
        "if CLOSE > ts_delay(CLOSE, 1) "
        "else (-(CLOSE - ts_delay(CLOSE, 1)))) "
        "* (VOLUME / (ts_mean(VOLUME, 10) + 0.001)))", "技术指标"),
    "gtja_188": ("ts_rank(ts_delay(ts_mean(CLOSE, 5), 1), 20)", "技术指标"),
    "gtja_189": ("(cs_rank(CLOSE) - cs_rank((CLOSE - ts_delay(CLOSE, 1))))", "技术指标"),
    "gtja_190": ("ts_rank(ts_std(CLOSE, 10), 20)", "技术指标"),
    "gtja_191": ("ts_corr(cs_rank(CLOSE), cs_rank(VOLUME), 20)", "技术指标"),
}