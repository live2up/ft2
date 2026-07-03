"""
kb03/formulas/alpha158.py — Qlib Alpha158 因子 (V4 AST 语法)

来源: 微软 Qlib 开源项目 (https://github.com/microsoft/qlib)
结构: ALPHA158[name] = (v4_expr, category)

Alpha158 共 158 个因子，分为三大类:
  1. K线基础特征 (9个) — 单日 OHLC 形态
  2. 价格特征 (4个) — 当日价格相对收盘价比值
  3. 滚动窗口技术指标 (145个) — 29类指标 × 5个窗口(5/10/20/30/60)

翻译规则 (Qlib 原始 → V4 AST):
  Ref(x, d)        → ts_delay(x, d)
  Mean(x, d)       → ts_mean(x, d)
  Std(x, d)        → ts_std(x, d)
  Max(x, d)        → ts_max(x, d)
  Min(x, d)        → ts_min(x, d)
  Slope(x, d)      → ts_slope(x, d)
  Rsquare(x, d)    → ts_rsq(x, d)
  Resi(x, d)       → ts_resid(x, d)
  Quantile(x,d,p)  → ts_quantile(x, d, p)
  Rank(x, d)       → ts_rank(x, d)
  Corr(x, y, d)    → ts_corr(x, y, d)
  IdxMax(x, d)     → ts_argmax(x, d)
  IdxMin(x, d)     → ts_argmin(x, d)
  Greater(a, b)    → (a if a > b else b)
  Less(a, b)       → (a if a < b else b)
  Sum(x, d)        → ts_sum(x, d)
  Abs(x)           → abs(x)
  Log(x)           → log(x)
  Sqrt(x)          → sqrt(x)
  Sign(x)          → sign(x)
  VWAP             → (HIGH + LOW + CLOSE) / 3  (行业指数近似)

注意: Alpha158 的 VWAP 在 Qlib 中是真实成交均价，
      行业指数无真实 VWAP，用 (H+L+C)/3 近似。
"""
from typing import Dict, Tuple

ALPHA158: Dict[str, Tuple[str, str]] = {
    # ═══════════════════════════════════════════════════════════════
    # 1. K线基础特征 (9个)
    # ═══════════════════════════════════════════════════════════════
    "KMID":  ("((CLOSE - OPEN) / OPEN)", "K线"),
    "KLEN":  ("((HIGH - LOW) / OPEN)", "K线"),
    "KMID2": ("((CLOSE - OPEN) / ((HIGH - LOW) + 1e-12))", "K线"),
    "KUP":   ("((HIGH - (OPEN if OPEN > CLOSE else CLOSE)) / OPEN)", "K线"),
    "KUP2":  ("((HIGH - (OPEN if OPEN > CLOSE else CLOSE)) / ((HIGH - LOW) + 1e-12))", "K线"),
    "KLOW":  ("(((OPEN if OPEN < CLOSE else CLOSE) - LOW) / OPEN)", "K线"),
    "KLOW2": ("(((OPEN if OPEN < CLOSE else CLOSE) - LOW) / ((HIGH - LOW) + 1e-12))", "K线"),
    "KSFT":  ("((2 * CLOSE - HIGH - LOW) / OPEN)", "K线"),
    "KSFT2": ("((2 * CLOSE - HIGH - LOW) / ((HIGH - LOW) + 1e-12))", "K线"),

    # ═══════════════════════════════════════════════════════════════
    # 2. 价格特征 (4个)
    # ═══════════════════════════════════════════════════════════════
    "OPEN0":  ("(OPEN / CLOSE)", "价格"),
    "HIGH0":  ("(HIGH / CLOSE)", "价格"),
    "LOW0":   ("(LOW / CLOSE)", "价格"),
    "VWAP0":  ("(((HIGH + LOW + CLOSE) / 3) / CLOSE)", "价格"),

    # ═══════════════════════════════════════════════════════════════
    # 3. 滚动窗口技术指标 (145个) = 29类 × 5窗口
    # 窗口: 5, 10, 20, 30, 60
    # ═══════════════════════════════════════════════════════════════

    # ── 3.1 价格趋势与动量类 ──
    # ROC: Ref($close, d)/$close
    "ROC5":  ("(ts_delay(CLOSE, 5) / CLOSE)", "动量"),
    "ROC10": ("(ts_delay(CLOSE, 10) / CLOSE)", "动量"),
    "ROC20": ("(ts_delay(CLOSE, 20) / CLOSE)", "动量"),
    "ROC30": ("(ts_delay(CLOSE, 30) / CLOSE)", "动量"),
    "ROC60": ("(ts_delay(CLOSE, 60) / CLOSE)", "动量"),

    # MA: Mean($close, d)/$close
    "MA5":  ("(ts_mean(CLOSE, 5) / CLOSE)", "动量"),
    "MA10": ("(ts_mean(CLOSE, 10) / CLOSE)", "动量"),
    "MA20": ("(ts_mean(CLOSE, 20) / CLOSE)", "动量"),
    "MA30": ("(ts_mean(CLOSE, 30) / CLOSE)", "动量"),
    "MA60": ("(ts_mean(CLOSE, 60) / CLOSE)", "动量"),

    # STD: Std($close, d)/$close
    "STD5":  ("(ts_std(CLOSE, 5) / CLOSE)", "波动率"),
    "STD10": ("(ts_std(CLOSE, 10) / CLOSE)", "波动率"),
    "STD20": ("(ts_std(CLOSE, 20) / CLOSE)", "波动率"),
    "STD30": ("(ts_std(CLOSE, 30) / CLOSE)", "波动率"),
    "STD60": ("(ts_std(CLOSE, 60) / CLOSE)", "波动率"),

    # BETA: Slope($close, d)/$close  (对时间回归斜率)
    "BETA5":  ("(ts_slope(CLOSE, 5) / CLOSE)", "动量"),
    "BETA10": ("(ts_slope(CLOSE, 10) / CLOSE)", "动量"),
    "BETA20": ("(ts_slope(CLOSE, 20) / CLOSE)", "动量"),
    "BETA30": ("(ts_slope(CLOSE, 30) / CLOSE)", "动量"),
    "BETA60": ("(ts_slope(CLOSE, 60) / CLOSE)", "动量"),

    # RSQR: Rsquare($close, d)  (对时间回归 R²)
    "RSQR5":  ("ts_rsq(CLOSE, 5)", "动量"),
    "RSQR10": ("ts_rsq(CLOSE, 10)", "动量"),
    "RSQR20": ("ts_rsq(CLOSE, 20)", "动量"),
    "RSQR30": ("ts_rsq(CLOSE, 30)", "动量"),
    "RSQR60": ("ts_rsq(CLOSE, 60)", "动量"),

    # RESI: Resi($close, d)/$close  (对时间回归残差)
    "RESI5":  ("(ts_resid(CLOSE, 5) / CLOSE)", "动量"),
    "RESI10": ("(ts_resid(CLOSE, 10) / CLOSE)", "动量"),
    "RESI20": ("(ts_resid(CLOSE, 20) / CLOSE)", "动量"),
    "RESI30": ("(ts_resid(CLOSE, 30) / CLOSE)", "动量"),
    "RESI60": ("(ts_resid(CLOSE, 60) / CLOSE)", "动量"),

    # ── 3.2 价格位置类 ──
    # MAX: Max($high, d)/$close
    "MAX5":  ("(ts_max(HIGH, 5) / CLOSE)", "价格"),
    "MAX10": ("(ts_max(HIGH, 10) / CLOSE)", "价格"),
    "MAX20": ("(ts_max(HIGH, 20) / CLOSE)", "价格"),
    "MAX30": ("(ts_max(HIGH, 30) / CLOSE)", "价格"),
    "MAX60": ("(ts_max(HIGH, 60) / CLOSE)", "价格"),

    # MIN: Min($low, d)/$close
    "MIN5":  ("(ts_min(LOW, 5) / CLOSE)", "价格"),
    "MIN10": ("(ts_min(LOW, 10) / CLOSE)", "价格"),
    "MIN20": ("(ts_min(LOW, 20) / CLOSE)", "价格"),
    "MIN30": ("(ts_min(LOW, 30) / CLOSE)", "价格"),
    "MIN60": ("(ts_min(LOW, 60) / CLOSE)", "价格"),

    # QTLU: Quantile($close, d, 0.8)/$close  (80%分位数值)
    "QTLU5":  ("(ts_quantile(CLOSE, 5, 0.8) / CLOSE)", "价格"),
    "QTLU10": ("(ts_quantile(CLOSE, 10, 0.8) / CLOSE)", "价格"),
    "QTLU20": ("(ts_quantile(CLOSE, 20, 0.8) / CLOSE)", "价格"),
    "QTLU30": ("(ts_quantile(CLOSE, 30, 0.8) / CLOSE)", "价格"),
    "QTLU60": ("(ts_quantile(CLOSE, 60, 0.8) / CLOSE)", "价格"),

    # QTLD: Quantile($close, d, 0.2)/$close  (20%分位数值)
    "QTLD5":  ("(ts_quantile(CLOSE, 5, 0.2) / CLOSE)", "价格"),
    "QTLD10": ("(ts_quantile(CLOSE, 10, 0.2) / CLOSE)", "价格"),
    "QTLD20": ("(ts_quantile(CLOSE, 20, 0.2) / CLOSE)", "价格"),
    "QTLD30": ("(ts_quantile(CLOSE, 30, 0.2) / CLOSE)", "价格"),
    "QTLD60": ("(ts_quantile(CLOSE, 60, 0.2) / CLOSE)", "价格"),

    # RANK: Rank($close, d)
    "RANK5":  ("ts_rank(CLOSE, 5)", "价格"),
    "RANK10": ("ts_rank(CLOSE, 10)", "价格"),
    "RANK20": ("ts_rank(CLOSE, 20)", "价格"),
    "RANK30": ("ts_rank(CLOSE, 30)", "价格"),
    "RANK60": ("ts_rank(CLOSE, 60)", "价格"),

    # RSV: ($close-Min($low,d))/(Max($high,d)-Min($low,d)+1e-12)
    "RSV5":  ("((CLOSE - ts_min(LOW, 5)) / (ts_max(HIGH, 5) - ts_min(LOW, 5) + 1e-12))", "价格"),
    "RSV10": ("((CLOSE - ts_min(LOW, 10)) / (ts_max(HIGH, 10) - ts_min(LOW, 10) + 1e-12))", "价格"),
    "RSV20": ("((CLOSE - ts_min(LOW, 20)) / (ts_max(HIGH, 20) - ts_min(LOW, 20) + 1e-12))", "价格"),
    "RSV30": ("((CLOSE - ts_min(LOW, 30)) / (ts_max(HIGH, 30) - ts_min(LOW, 30) + 1e-12))", "价格"),
    "RSV60": ("((CLOSE - ts_min(LOW, 60)) / (ts_max(HIGH, 60) - ts_min(LOW, 60) + 1e-12))", "价格"),

    # ── 3.3 时间序列位置类 ──
    # IMAX: IdxMax($high, d)/d
    "IMAX5":  ("(ts_argmax(HIGH, 5) / 5)", "价格"),
    "IMAX10": ("(ts_argmax(HIGH, 10) / 10)", "价格"),
    "IMAX20": ("(ts_argmax(HIGH, 20) / 20)", "价格"),
    "IMAX30": ("(ts_argmax(HIGH, 30) / 30)", "价格"),
    "IMAX60": ("(ts_argmax(HIGH, 60) / 60)", "价格"),

    # IMIN: IdxMin($low, d)/d
    "IMIN5":  ("(ts_argmin(LOW, 5) / 5)", "价格"),
    "IMIN10": ("(ts_argmin(LOW, 10) / 10)", "价格"),
    "IMIN20": ("(ts_argmin(LOW, 20) / 20)", "价格"),
    "IMIN30": ("(ts_argmin(LOW, 30) / 30)", "价格"),
    "IMIN60": ("(ts_argmin(LOW, 60) / 60)", "价格"),

    # IMXD: (IdxMax($high,d)-IdxMin($low,d))/d
    "IMXD5":  ("((ts_argmax(HIGH, 5) - ts_argmin(LOW, 5)) / 5)", "价格"),
    "IMXD10": ("((ts_argmax(HIGH, 10) - ts_argmin(LOW, 10)) / 10)", "价格"),
    "IMXD20": ("((ts_argmax(HIGH, 20) - ts_argmin(LOW, 20)) / 20)", "价格"),
    "IMXD30": ("((ts_argmax(HIGH, 30) - ts_argmin(LOW, 30)) / 30)", "价格"),
    "IMXD60": ("((ts_argmax(HIGH, 60) - ts_argmin(LOW, 60)) / 60)", "价格"),

    # ── 3.4 价格-成交量关联类 ──
    # CORR: Corr($close, Log($volume+1), d)
    "CORR5":  ("ts_corr(CLOSE, log(VOLUME + 1), 5)", "量价"),
    "CORR10": ("ts_corr(CLOSE, log(VOLUME + 1), 10)", "量价"),
    "CORR20": ("ts_corr(CLOSE, log(VOLUME + 1), 20)", "量价"),
    "CORR30": ("ts_corr(CLOSE, log(VOLUME + 1), 30)", "量价"),
    "CORR60": ("ts_corr(CLOSE, log(VOLUME + 1), 60)", "量价"),

    # CORD: Corr($close/Ref($close,1), Log($volume/Ref($volume,1)+1), d)
    "CORD5":  ("ts_corr((CLOSE / ts_delay(CLOSE, 1)), log((VOLUME / ts_delay(VOLUME, 1)) + 1), 5)", "量价"),
    "CORD10": ("ts_corr((CLOSE / ts_delay(CLOSE, 1)), log((VOLUME / ts_delay(VOLUME, 1)) + 1), 10)", "量价"),
    "CORD20": ("ts_corr((CLOSE / ts_delay(CLOSE, 1)), log((VOLUME / ts_delay(VOLUME, 1)) + 1), 20)", "量价"),
    "CORD30": ("ts_corr((CLOSE / ts_delay(CLOSE, 1)), log((VOLUME / ts_delay(VOLUME, 1)) + 1), 30)", "量价"),
    "CORD60": ("ts_corr((CLOSE / ts_delay(CLOSE, 1)), log((VOLUME / ts_delay(VOLUME, 1)) + 1), 60)", "量价"),

    # ── 3.5 涨跌统计类 ──
    # CNTP: Mean($close>Ref($close,1), d)
    "CNTP5":  ("ts_mean((CLOSE > ts_delay(CLOSE, 1)), 5)", "动量"),
    "CNTP10": ("ts_mean((CLOSE > ts_delay(CLOSE, 1)), 10)", "动量"),
    "CNTP20": ("ts_mean((CLOSE > ts_delay(CLOSE, 1)), 20)", "动量"),
    "CNTP30": ("ts_mean((CLOSE > ts_delay(CLOSE, 1)), 30)", "动量"),
    "CNTP60": ("ts_mean((CLOSE > ts_delay(CLOSE, 1)), 60)", "动量"),

    # CNTN: Mean($close<Ref($close,1), d)
    "CNTN5":  ("ts_mean((CLOSE < ts_delay(CLOSE, 1)), 5)", "动量"),
    "CNTN10": ("ts_mean((CLOSE < ts_delay(CLOSE, 1)), 10)", "动量"),
    "CNTN20": ("ts_mean((CLOSE < ts_delay(CLOSE, 1)), 20)", "动量"),
    "CNTN30": ("ts_mean((CLOSE < ts_delay(CLOSE, 1)), 30)", "动量"),
    "CNTN60": ("ts_mean((CLOSE < ts_delay(CLOSE, 1)), 60)", "动量"),

    # CNTD: CNTP - CNTN
    "CNTD5":  ("(ts_mean((CLOSE > ts_delay(CLOSE, 1)), 5) - ts_mean((CLOSE < ts_delay(CLOSE, 1)), 5))", "动量"),
    "CNTD10": ("(ts_mean((CLOSE > ts_delay(CLOSE, 1)), 10) - ts_mean((CLOSE < ts_delay(CLOSE, 1)), 10))", "动量"),
    "CNTD20": ("(ts_mean((CLOSE > ts_delay(CLOSE, 1)), 20) - ts_mean((CLOSE < ts_delay(CLOSE, 1)), 20))", "动量"),
    "CNTD30": ("(ts_mean((CLOSE > ts_delay(CLOSE, 1)), 30) - ts_mean((CLOSE < ts_delay(CLOSE, 1)), 30))", "动量"),
    "CNTD60": ("(ts_mean((CLOSE > ts_delay(CLOSE, 1)), 60) - ts_mean((CLOSE < ts_delay(CLOSE, 1)), 60))", "动量"),

    # ── 3.6 RSI类指标 ──
    # SUMP: Sum(Greater($close-Ref($close,1),0),d)/(Sum(Abs($close-Ref($close,1)),d)+1e-12)
    "SUMP5":  ("(ts_sum((CLOSE - ts_delay(CLOSE, 1)) if (CLOSE - ts_delay(CLOSE, 1)) > 0 else 0, 5) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 5) + 1e-12))", "动量"),
    "SUMP10": ("(ts_sum((CLOSE - ts_delay(CLOSE, 1)) if (CLOSE - ts_delay(CLOSE, 1)) > 0 else 0, 10) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 10) + 1e-12))", "动量"),
    "SUMP20": ("(ts_sum((CLOSE - ts_delay(CLOSE, 1)) if (CLOSE - ts_delay(CLOSE, 1)) > 0 else 0, 20) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 20) + 1e-12))", "动量"),
    "SUMP30": ("(ts_sum((CLOSE - ts_delay(CLOSE, 1)) if (CLOSE - ts_delay(CLOSE, 1)) > 0 else 0, 30) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 30) + 1e-12))", "动量"),
    "SUMP60": ("(ts_sum((CLOSE - ts_delay(CLOSE, 1)) if (CLOSE - ts_delay(CLOSE, 1)) > 0 else 0, 60) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 60) + 1e-12))", "动量"),

    # SUMN: Sum(Greater(Ref($close,1)-$close,0),d)/(Sum(Abs($close-Ref($close,1)),d)+1e-12)
    "SUMN5":  ("(ts_sum((ts_delay(CLOSE, 1) - CLOSE) if (ts_delay(CLOSE, 1) - CLOSE) > 0 else 0, 5) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 5) + 1e-12))", "动量"),
    "SUMN10": ("(ts_sum((ts_delay(CLOSE, 1) - CLOSE) if (ts_delay(CLOSE, 1) - CLOSE) > 0 else 0, 10) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 10) + 1e-12))", "动量"),
    "SUMN20": ("(ts_sum((ts_delay(CLOSE, 1) - CLOSE) if (ts_delay(CLOSE, 1) - CLOSE) > 0 else 0, 20) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 20) + 1e-12))", "动量"),
    "SUMN30": ("(ts_sum((ts_delay(CLOSE, 1) - CLOSE) if (ts_delay(CLOSE, 1) - CLOSE) > 0 else 0, 30) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 30) + 1e-12))", "动量"),
    "SUMN60": ("(ts_sum((ts_delay(CLOSE, 1) - CLOSE) if (ts_delay(CLOSE, 1) - CLOSE) > 0 else 0, 60) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 60) + 1e-12))", "动量"),

    # SUMD: SUMP - SUMN
    "SUMD5":  ("((ts_sum((CLOSE - ts_delay(CLOSE, 1)) if (CLOSE - ts_delay(CLOSE, 1)) > 0 else 0, 5) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 5) + 1e-12)) - (ts_sum((ts_delay(CLOSE, 1) - CLOSE) if (ts_delay(CLOSE, 1) - CLOSE) > 0 else 0, 5) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 5) + 1e-12)))", "动量"),
    "SUMD10": ("((ts_sum((CLOSE - ts_delay(CLOSE, 1)) if (CLOSE - ts_delay(CLOSE, 1)) > 0 else 0, 10) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 10) + 1e-12)) - (ts_sum((ts_delay(CLOSE, 1) - CLOSE) if (ts_delay(CLOSE, 1) - CLOSE) > 0 else 0, 10) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 10) + 1e-12)))", "动量"),
    "SUMD20": ("((ts_sum((CLOSE - ts_delay(CLOSE, 1)) if (CLOSE - ts_delay(CLOSE, 1)) > 0 else 0, 20) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 20) + 1e-12)) - (ts_sum((ts_delay(CLOSE, 1) - CLOSE) if (ts_delay(CLOSE, 1) - CLOSE) > 0 else 0, 20) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 20) + 1e-12)))", "动量"),
    "SUMD30": ("((ts_sum((CLOSE - ts_delay(CLOSE, 1)) if (CLOSE - ts_delay(CLOSE, 1)) > 0 else 0, 30) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 30) + 1e-12)) - (ts_sum((ts_delay(CLOSE, 1) - CLOSE) if (ts_delay(CLOSE, 1) - CLOSE) > 0 else 0, 30) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 30) + 1e-12)))", "动量"),
    "SUMD60": ("((ts_sum((CLOSE - ts_delay(CLOSE, 1)) if (CLOSE - ts_delay(CLOSE, 1)) > 0 else 0, 60) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 60) + 1e-12)) - (ts_sum((ts_delay(CLOSE, 1) - CLOSE) if (ts_delay(CLOSE, 1) - CLOSE) > 0 else 0, 60) / (ts_sum(abs(CLOSE - ts_delay(CLOSE, 1)), 60) + 1e-12)))", "动量"),

    # ── 3.7 成交量技术指标 ──
    # VMA: Mean($volume, d)/($volume+1e-12)
    "VMA5":  ("(ts_mean(VOLUME, 5) / (VOLUME + 1e-12))", "量价"),
    "VMA10": ("(ts_mean(VOLUME, 10) / (VOLUME + 1e-12))", "量价"),
    "VMA20": ("(ts_mean(VOLUME, 20) / (VOLUME + 1e-12))", "量价"),
    "VMA30": ("(ts_mean(VOLUME, 30) / (VOLUME + 1e-12))", "量价"),
    "VMA60": ("(ts_mean(VOLUME, 60) / (VOLUME + 1e-12))", "量价"),

    # VSTD: Std($volume, d)/($volume+1e-12)
    "VSTD5":  ("(ts_std(VOLUME, 5) / (VOLUME + 1e-12))", "量价"),
    "VSTD10": ("(ts_std(VOLUME, 10) / (VOLUME + 1e-12))", "量价"),
    "VSTD20": ("(ts_std(VOLUME, 20) / (VOLUME + 1e-12))", "量价"),
    "VSTD30": ("(ts_std(VOLUME, 30) / (VOLUME + 1e-12))", "量价"),
    "VSTD60": ("(ts_std(VOLUME, 60) / (VOLUME + 1e-12))", "量价"),

    # WVMA: Std(Abs($close/Ref($close,1)-1)*$volume,d)/(Mean(Abs($close/Ref($close,1)-1)*$volume,d)+1e-12)
    "WVMA5":  ("(ts_std(abs((CLOSE / ts_delay(CLOSE, 1) - 1)) * VOLUME, 5) / (ts_mean(abs((CLOSE / ts_delay(CLOSE, 1) - 1)) * VOLUME, 5) + 1e-12))", "量价"),
    "WVMA10": ("(ts_std(abs((CLOSE / ts_delay(CLOSE, 1) - 1)) * VOLUME, 10) / (ts_mean(abs((CLOSE / ts_delay(CLOSE, 1) - 1)) * VOLUME, 10) + 1e-12))", "量价"),
    "WVMA20": ("(ts_std(abs((CLOSE / ts_delay(CLOSE, 1) - 1)) * VOLUME, 20) / (ts_mean(abs((CLOSE / ts_delay(CLOSE, 1) - 1)) * VOLUME, 20) + 1e-12))", "量价"),
    "WVMA30": ("(ts_std(abs((CLOSE / ts_delay(CLOSE, 1) - 1)) * VOLUME, 30) / (ts_mean(abs((CLOSE / ts_delay(CLOSE, 1) - 1)) * VOLUME, 30) + 1e-12))", "量价"),
    "WVMA60": ("(ts_std(abs((CLOSE / ts_delay(CLOSE, 1) - 1)) * VOLUME, 60) / (ts_mean(abs((CLOSE / ts_delay(CLOSE, 1) - 1)) * VOLUME, 60) + 1e-12))", "量价"),

    # VSUMP: Sum(Greater($volume-Ref($volume,1),0),d)/(Sum(Abs($volume-Ref($volume,1)),d)+1e-12)
    "VSUMP5":  ("(ts_sum((VOLUME - ts_delay(VOLUME, 1)) if (VOLUME - ts_delay(VOLUME, 1)) > 0 else 0, 5) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 5) + 1e-12))", "量价"),
    "VSUMP10": ("(ts_sum((VOLUME - ts_delay(VOLUME, 1)) if (VOLUME - ts_delay(VOLUME, 1)) > 0 else 0, 10) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 10) + 1e-12))", "量价"),
    "VSUMP20": ("(ts_sum((VOLUME - ts_delay(VOLUME, 1)) if (VOLUME - ts_delay(VOLUME, 1)) > 0 else 0, 20) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 20) + 1e-12))", "量价"),
    "VSUMP30": ("(ts_sum((VOLUME - ts_delay(VOLUME, 1)) if (VOLUME - ts_delay(VOLUME, 1)) > 0 else 0, 30) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 30) + 1e-12))", "量价"),
    "VSUMP60": ("(ts_sum((VOLUME - ts_delay(VOLUME, 1)) if (VOLUME - ts_delay(VOLUME, 1)) > 0 else 0, 60) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 60) + 1e-12))", "量价"),

    # VSUMN: Sum(Greater(Ref($volume,1)-$volume,0),d)/(Sum(Abs($volume-Ref($volume,1)),d)+1e-12)
    "VSUMN5":  ("(ts_sum((ts_delay(VOLUME, 1) - VOLUME) if (ts_delay(VOLUME, 1) - VOLUME) > 0 else 0, 5) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 5) + 1e-12))", "量价"),
    "VSUMN10": ("(ts_sum((ts_delay(VOLUME, 1) - VOLUME) if (ts_delay(VOLUME, 1) - VOLUME) > 0 else 0, 10) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 10) + 1e-12))", "量价"),
    "VSUMN20": ("(ts_sum((ts_delay(VOLUME, 1) - VOLUME) if (ts_delay(VOLUME, 1) - VOLUME) > 0 else 0, 20) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 20) + 1e-12))", "量价"),
    "VSUMN30": ("(ts_sum((ts_delay(VOLUME, 1) - VOLUME) if (ts_delay(VOLUME, 1) - VOLUME) > 0 else 0, 30) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 30) + 1e-12))", "量价"),
    "VSUMN60": ("(ts_sum((ts_delay(VOLUME, 1) - VOLUME) if (ts_delay(VOLUME, 1) - VOLUME) > 0 else 0, 60) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 60) + 1e-12))", "量价"),

    # VSUMD: VSUMP - VSUMN
    "VSUMD5":  ("((ts_sum((VOLUME - ts_delay(VOLUME, 1)) if (VOLUME - ts_delay(VOLUME, 1)) > 0 else 0, 5) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 5) + 1e-12)) - (ts_sum((ts_delay(VOLUME, 1) - VOLUME) if (ts_delay(VOLUME, 1) - VOLUME) > 0 else 0, 5) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 5) + 1e-12)))", "量价"),
    "VSUMD10": ("((ts_sum((VOLUME - ts_delay(VOLUME, 1)) if (VOLUME - ts_delay(VOLUME, 1)) > 0 else 0, 10) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 10) + 1e-12)) - (ts_sum((ts_delay(VOLUME, 1) - VOLUME) if (ts_delay(VOLUME, 1) - VOLUME) > 0 else 0, 10) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 10) + 1e-12)))", "量价"),
    "VSUMD20": ("((ts_sum((VOLUME - ts_delay(VOLUME, 1)) if (VOLUME - ts_delay(VOLUME, 1)) > 0 else 0, 20) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 20) + 1e-12)) - (ts_sum((ts_delay(VOLUME, 1) - VOLUME) if (ts_delay(VOLUME, 1) - VOLUME) > 0 else 0, 20) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 20) + 1e-12)))", "量价"),
    "VSUMD30": ("((ts_sum((VOLUME - ts_delay(VOLUME, 1)) if (VOLUME - ts_delay(VOLUME, 1)) > 0 else 0, 30) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 30) + 1e-12)) - (ts_sum((ts_delay(VOLUME, 1) - VOLUME) if (ts_delay(VOLUME, 1) - VOLUME) > 0 else 0, 30) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 30) + 1e-12)))", "量价"),
    "VSUMD60": ("((ts_sum((VOLUME - ts_delay(VOLUME, 1)) if (VOLUME - ts_delay(VOLUME, 1)) > 0 else 0, 60) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 60) + 1e-12)) - (ts_sum((ts_delay(VOLUME, 1) - VOLUME) if (ts_delay(VOLUME, 1) - VOLUME) > 0 else 0, 60) / (ts_sum(abs(VOLUME - ts_delay(VOLUME, 1)), 60) + 1e-12)))", "量价"),
}
