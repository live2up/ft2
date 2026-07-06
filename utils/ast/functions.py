# -*- coding: utf-8 -*-
"""
utils/ast/functions.py — 原语层 (公共基础设施)
=============================================================================

在五层架构中的位置: 第2层(原语) — 定义"能算什么"

  函数定义 = 85 个 | FUNC_REGISTRY 条目 = 88 个 (含 3 别名)

═══════════════════════════════════════════════════════════════
命名规范 (对齐 WQ101 行业标准)

  ◆ 函数命名 (snake_case, 小写 + 前缀)
    前缀约定:
      ts_       → Time Series (窗口滚动, 只用历史数据)
      ts_reg_   → 双变量滚动线性回归 y~x (ts_reg_slope/y~x残差等)
      cs_       → Cross Sectional (截面统计, 需完整 2D 面板)
      expanding_→ 扩展窗口 (起始→当前, 无固定窗口)
      无前缀    → 逐元素数学 / talib 特征 / 信号

    回归函数命名区分:
      ts_reg_*(y, x, d)  → 双变量 y~x 回归 (如 CLOSE~VOLUME)
      ts_*(x, w)         → 单变量 x~t 趋势回归 (如 CLOSE~time)

    回归后缀统一 (统计学标准缩写):
      slope      = 斜率
      intercept  = 截距
      resid      = residual (残差), 保留 ts_resi 别名兼容 Alpha158
      predict    = 预测值
      rsq        = R-squared (R²), 保留 ts_rsquare 别名兼容

    参数约定:
      窗口: d         (对齐 WQ101, 非 w=window)
      输入: x, y      (单变量 x, 双变量 x,y)
      可选: 有默认值   (如 cs_scale(x, scale=1.0))

    WQ101 兼容别名 (注册为短名, 等价于 ts_* 版本):
      corr  → ts_corr,  roc  → ts_roc,  kurt → ts_kurt

  ◆ 变量 vs 函数同名 (设计如此, 互补而非冲突):
      函数: rsi(CLOSE, 14)     → 实时算, 参数灵活
      变量: ts_rank(RSI_14, 60) → 预计算, 性能优先

  ◆ 统计量约定 (对齐 WQ101 / GT191)
    ddof=1 (样本):   ts_std, ts_skew, ts_kurt, ts_cov, cs_zscore, cs_winsorize
    NaN 处理:        NaN = 缺失数据应忽略, 使用 nan* 版本函数
                     (nanmean/nanmax/nanargmax 等)
    cs_rank 并列:    competition ranking (method='min'), 见 SciPy rankdata 文档

原则:
  - 所有滚动窗口只用历史数据（无前向偏差）
  - 冷启动保护：前 window-1 天返回 NaN
  - 1D 数组输入, 1D 数组输出 (由上层逐列调用实现 2D 面板)

═══════════════════════════════════════════════════════════════
现有函数清单 (88 个 FUNC_REGISTRY 条目, 按分类)

  ┌─ 内部工具 (3)
  │  _rolling / _expanding / _persist

  ├─ 时序 ts_ (28 + 2 lambda + 5 别名 = 35 条目)
  │  基础:      mean / std / sum / max / min / median
  │  周期:      delta / delay / roc
  │  统计:      rank / zscore / scale / quantile / av_diff
  │             var(λ) / logret(λ) / decay_linear / product
  │  相关:      corr / cov
  │  分布形状:  skew / kurt / argmax / argmin
  │  x~t趋势:   slope / resid / rsq / intercept / predict
  │  [y~x回归]  ts_reg_slope / ts_reg_intercept / ts_reg_resid
  │             ts_reg_predict / ts_reg_rsq (均调用 ts_regression)

  ├─ 扩张 expanding_ (4)
  │  expanding_mean / expanding_median / expanding_std / expanding_percentile

  ├─ 截面 cs_ (6)
  │  cs_rank / cs_zscore / cs_scale / cs_winsorize
  │  cs_normalize / cs_quantile(=cs_rank)

  ├─ 数学 (17)
  │  基础:      abs / log / sqrt / sign / exp / safe_max / safe_min
  │  三角函数:  sin / cos
  │  激活:      tanh / sigmoid / relu / softsign
  │  钟形族:    gauss / p4
  │  自定义:    signed_power / square_sigmoid

  ├─ 信号 (1)
  │  persist

  ├─ 特征 ta-lib (21)
  │  趋势:      ema / wma / dema / kama / trima / tsf / linearreg
  │  波动:      atr / atr_sma / natr / stddev / var / hv / bb_width
  │  动量:      rsi / macd / adx / cci
  │  量比:      vol_ratio / amt_ratio
  │  平滑:      wilder_smooth

  ├─ WQ别名 (3) — 逐步取缔, 统一改用 ts_ 版
  │  corr          → 改用 ts_corr
  │  roc           → 改用 ts_roc
  │  kurt          → 改用 ts_kurt

  ├─ 旧名别名 (3) — [2026-07-03] 已停止生产使用, 后续删除
  │  ts_resi                    → 改用 ts_resid
  │  ts_regression_residual     → 改用 ts_resid
  │  ts_rsquare                 → 改用 ts_rsq

  └─ 注册 (2)
     register_function / unregister_function

═══════════════════════════════════════════════════════════════
潜在扩展函数 (按优先级, 从 WQ101/GT191/Alpha158 需求整理)

  ★ P0 — 逻辑/条件 (无此算子会导致表达式级功能缺失)
    is_nan(x)                    → 判断 NaN (当前用 `x != x` 替代, 不直观)
    if_else(cond, a, b)         → 条件选择 (当前用 cond*a+(1-cond)*b 替代, 欠明确)
    ts_last(x, d)               → 取窗口内最后一个有效值 (等价延迟, 语义清晰)
    ts_count_nans(x, d)         → 窗口内 NaN 计数 (诊断缺失率)

  ★ P1 — 组别运算 (31行业/个股级需要)
    group_rank(x, group)        → 组内排名 (替代 cs_rank 后加行业层)
    group_zscore(x, group)      → 组内标准化
    group_mean(x, group)        → 组内均值 (去行业均值 = 行业中性化)
    group_neutralize(x, group)   → 行业中性化

  ★ P2 — 时序辅助 (提升表达力)
    ts_step(x, d)               → 窗口内是否从未为 0 (持久性检测)
    ts_backfill(x, d)           → 前向填充 NaN (数据修复)
    ts_hump(x, d)               → 窗口内是否中间高两端低 (驼峰检测)
    ts_kth_element(x, d, k)     → 窗口内第 k 大/小值 (稳健极值)

  ★ P3 — 高级统计 (特定策略需求)
    ts_hurst(x, d)              → Hurst 指数 (均值回归/趋势检测)
    ts_entropy(x, d)            → 样本熵/近似熵 (复杂度测量)
    ts_dcorr(x, y, d)           → 距离相关性 (非线性相关性)
    ts_variance_ratio(x, d, k)  → 方差比检验 (随机游走检测)
    cs_winsorize_quantile(x, l, u) → 分位数截尾 (现行 std 版外的新变体)

  ★ P4 — 变换/向量 (低优先级)
    bucket(x, n)                → 连续值分箱 (截面离散化)
    vec_sum(x) / vec_avg(x)     → 向量归约 (替代 ts_sum 的语义更明确)
    trade_when(cond, expr)      → 条件触发执行 (状态机逻辑)

═══════════════════════════════════════════════════════════════
公共模式 / 可复用抽象

  # 防除零模式 (6处: ts_roc/ts_zscore/ts_scale/cs_zscore/cs_scale/ts_regression)
  result = value / divisor if divisor > 1e-10 else 0.0

  # NaN过滤模式 (7处: ts_product/ts_regression/ts_resi/
  #                  ts_decay_linear/ts_scale/cs_zscore/cs_scale/cs_winsorize)
  valid = ~np.isnan(arr)
  # 当前分散在各函数中, 未来可抽为 _nan_filter(seg, min_valid=1) 辅助函数

  # 防除零+NaN (可抽公共函数, 暂时分散):
  def _safe_div(num, den, fallback=0.0, eps=1e-10):
      return num / den if abs(den) > eps else fallback

═══════════════════════════════════════════════════════════════
[重构] 2026-06-22 从 registry.py 拆分, 独立为 functions.py
[修正] 2026-06-25 cs_rank→min排名, ts_*→nan*版, 对齐WQ标准
[重构] 2026-07-06 ts_*/cs_* 函数用 numba @njit 加速, _rolling 作为 fallback 保留
═══════════════════════════════════════════════════════════════
"""
import numpy as np
import talib
from numba import njit
from typing import Dict, Callable


# ============================================================
# 内部工具函数 (1D 安全防护)
# ============================================================

def _rolling(x: np.ndarray, window: int, func, *a, **kw):
    """[修复] 2026-06-22 加 2D 防护: 时序函数只接受 1D, 误传 2D 会静默错误

    设计选择: 不过滤 NaN, 由各 func 自行处理 (各函数语义不同: nanmean/argmax 等)
    调用方约定: func 应使用 nan 安全版本 (np.nanmean, np.nanmax 等)
    [对齐] WQ标准: NaN=缺失数据应忽略, 不应污染窗口"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"_rolling 只接受 1D 数组, 收到 {x.ndim}D shape={x.shape}。"
            f"2D 面板需逐列调用。"
        )
    r = np.full_like(x, np.nan)
    for i in range(window - 1, len(x)):
        r[i] = func(x[i - window + 1 : i + 1], *a, **kw)
    return r


def _expanding(x: np.ndarray, func, min_p: int = 20, *a, **kw):
    """扩展窗口计算，自动跳过 NaN（与 v3 _expanding_mean 语义一致）

    [修复] 2026-06-22 加 2D 防护"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"_expanding 只接受 1D 数组, 收到 {x.ndim}D shape={x.shape}。"
            f"2D 面板需逐列调用。"
        )
    r = np.full_like(x, np.nan)
    for i in range(min_p - 1, len(x)):
        valid = x[:i + 1]
        valid = valid[~np.isnan(valid)]
        if len(valid) > 0:
            r[i] = func(valid, *a, **kw)
    return r


def _persist(x: np.ndarray, n: int) -> np.ndarray:
    """[修复] 2026-06-22 加 2D 防护"""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(
            f"_persist 只接受 1D 数组, 收到 {x.ndim}D shape={x.shape}。"
            f"2D 面板需逐列调用。"
        )
    r = np.full_like(x, 0.0)
    s = x > 0
    for i in range(n - 1, len(x)):
        if np.all(s[i - n + 1 : i + 1]):
            r[i] = 1.0
    return r


# ============================================================
# [重构] 2026-07-06 numba @njit 核心函数
# 所有 ts_*/cs_* 的计算逻辑在此实现, 上层 ts_* 为薄包装
# NaN 处理: 跳过窗口内 NaN, 全 NaN 返回 NaN (ts_resid 返回 0.0)
# ============================================================

@njit(cache=True)
def _ts_mean_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动均值, 跳过 NaN"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        s = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                s += x[j]
                cnt += 1
        if cnt > 0:
            r[i] = s / cnt
    return r


@njit(cache=True)
def _ts_std_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动标准差 (ddof=1), 跳过 NaN, count<2 返回 NaN"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        s = 0.0
        sq = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                s += x[j]
                sq += x[j] * x[j]
                cnt += 1
        if cnt >= 2:
            mean = s / cnt
            var = (sq - cnt * mean * mean) / (cnt - 1)
            if var < 0.0:
                var = 0.0
            r[i] = np.sqrt(var)
    return r


@njit(cache=True)
def _ts_sum_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动求和, 跳过 NaN, 全 NaN 返回 NaN"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        s = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                s += x[j]
                cnt += 1
        if cnt > 0:
            r[i] = s
    return r


@njit(cache=True)
def _ts_max_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动最大值, 跳过 NaN"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        m = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                if cnt == 0 or x[j] > m:
                    m = x[j]
                cnt += 1
        if cnt > 0:
            r[i] = m
    return r


@njit(cache=True)
def _ts_min_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动最小值, 跳过 NaN"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        m = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                if cnt == 0 or x[j] < m:
                    m = x[j]
                cnt += 1
        if cnt > 0:
            r[i] = m
    return r


@njit(cache=True)
def _ts_median_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动中位数, 收集有效值后排序取中位"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                cnt += 1
        if cnt > 0:
            valid = np.empty(cnt)
            k = 0
            for j in range(i - d + 1, i + 1):
                if not np.isnan(x[j]):
                    valid[k] = x[j]
                    k += 1
            sorted_v = np.sort(valid)
            mid = cnt // 2
            if cnt % 2 == 1:
                r[i] = sorted_v[mid]
            else:
                r[i] = (sorted_v[mid - 1] + sorted_v[mid]) / 2.0
    return r


@njit(cache=True)
def _ts_rank_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动排名
    [修复] 2026-07-06 当前值 x[i] 为 NaN 时返回 NaN (原用 valid[cnt-1] 返回过期排名, 污染截面)"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        # [修复] 当前值为 NaN 时无法计算排名, 保持 NaN
        if np.isnan(x[i]):
            continue
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                cnt += 1
        if cnt > 0:
            valid = np.empty(cnt)
            k = 0
            for j in range(i - d + 1, i + 1):
                if not np.isnan(x[j]):
                    valid[k] = x[j]
                    k += 1
            sorted_v = np.sort(valid)
            # [修复] 用 x[i] (当前值) 而非 valid[cnt-1] (最后有效值)
            pos = np.searchsorted(sorted_v, x[i])
            r[i] = (pos + 1) / cnt
    return r


@njit(cache=True)
def _ts_argmax_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动 argmax, 返回距今天数 (0=当前, d-1=最远)"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        best_val = 0.0
        best_idx = -1
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                if cnt == 0 or x[j] > best_val:
                    best_val = x[j]
                    best_idx = j
                cnt += 1
        if cnt > 0:
            r[i] = i - best_idx
    return r


@njit(cache=True)
def _ts_argmin_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动 argmin, 返回距今天数 (0=当前, d-1=最远)"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        best_val = 0.0
        best_idx = -1
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                if cnt == 0 or x[j] < best_val:
                    best_val = x[j]
                    best_idx = j
                cnt += 1
        if cnt > 0:
            r[i] = i - best_idx
    return r


@njit(cache=True)
def _ts_corr_core(x, y, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动相关系数, 同时跳过 x/y 的 NaN, 有效值<3 返回 NaN
    [修复] 2026-07-06 std≈0 时返回 NaN (原 0.0 与 ts_zscore 不一致, 掩盖无法计算的事实)"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        sx = 0.0
        sy = 0.0
        sxy = 0.0
        sxx = 0.0
        syy = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not (np.isnan(x[j]) or np.isnan(y[j])):
                sx += x[j]
                sy += y[j]
                sxy += x[j] * y[j]
                sxx += x[j] * x[j]
                syy += y[j] * y[j]
                cnt += 1
        if cnt >= 3:
            mx = sx / cnt
            my = sy / cnt
            cov_xy = (sxy - cnt * mx * my) / (cnt - 1)
            var_x = (sxx - cnt * mx * mx) / (cnt - 1)
            var_y = (syy - cnt * my * my) / (cnt - 1)
            if var_x < 0.0:
                var_x = 0.0
            if var_y < 0.0:
                var_y = 0.0
            std_x = np.sqrt(var_x)
            std_y = np.sqrt(var_y)
            # [修复] std≈0 时保持 NaN (原 r[i]=0.0 与 ts_zscore 不一致)
            if std_x > 1e-10 and std_y > 1e-10:
                r[i] = cov_xy / (std_x * std_y)
    return r


@njit(cache=True)
def _ts_cov_core(x, y, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动协方差 (ddof=1), 同时跳过 x/y 的 NaN"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        sx = 0.0
        sy = 0.0
        sxy = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not (np.isnan(x[j]) or np.isnan(y[j])):
                sx += x[j]
                sy += y[j]
                sxy += x[j] * y[j]
                cnt += 1
        if cnt >= 3:
            mx = sx / cnt
            my = sy / cnt
            cov_xy = (sxy - cnt * mx * my) / (cnt - 1)
            r[i] = cov_xy
    return r


@njit(cache=True)
def _ts_skew_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动偏度, 有效值<3 返回 NaN
    [修复] 2026-07-06 std≈0 时返回 NaN (原 0.0 与 ts_zscore 不一致)"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        s = 0.0
        sq = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                s += x[j]
                sq += x[j] * x[j]
                cnt += 1
        if cnt >= 3:
            mean = s / cnt
            var = (sq - cnt * mean * mean) / (cnt - 1)
            if var < 0.0:
                var = 0.0
            std = np.sqrt(var)
            # [修复] std≈0 时保持 NaN (原 r[i]=0.0)
            if std > 1e-10:
                m3 = 0.0
                for j in range(i - d + 1, i + 1):
                    if not np.isnan(x[j]):
                        z = (x[j] - mean) / std
                        m3 += z * z * z
                r[i] = m3 / cnt
    return r


@njit(cache=True)
def _ts_kurt_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动超额峰度, 有效值<4 返回 NaN
    [修复] 2026-07-06 std≈0 时返回 NaN (原 0.0 与 ts_zscore 不一致)"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        s = 0.0
        sq = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                s += x[j]
                sq += x[j] * x[j]
                cnt += 1
        if cnt >= 4:
            mean = s / cnt
            var = (sq - cnt * mean * mean) / (cnt - 1)
            if var < 0.0:
                var = 0.0
            std = np.sqrt(var)
            # [修复] std≈0 时保持 NaN (原 r[i]=0.0)
            if std > 1e-10:
                m4 = 0.0
                for j in range(i - d + 1, i + 1):
                    if not np.isnan(x[j]):
                        z = (x[j] - mean) / std
                        m4 += z * z * z * z
                r[i] = m4 / cnt - 3.0
    return r


@njit(cache=True)
def _ts_scale_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动缩放, x[i]/sum(|x|), sum_abs<=1e-10 返回 0"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        s = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                s += abs(x[j])
                cnt += 1
        if cnt > 0:
            if s > 1e-10:
                r[i] = x[i] / s
            else:
                r[i] = 0.0
    return r


@njit(cache=True)
def _ts_decay_linear_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 线性衰减加权均值, 权重=[1,2,...,n_valid]"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                cnt += 1
        if cnt > 0:
            wsum = 0.0
            wsum_x = 0.0
            k = 1
            for j in range(i - d + 1, i + 1):
                if not np.isnan(x[j]):
                    w = float(k)
                    wsum += w
                    wsum_x += x[j] * w
                    k += 1
            r[i] = wsum_x / wsum
    return r


@njit(cache=True)
def _ts_product_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动乘积, 跳过 NaN, 全 NaN 返回 NaN"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        p = 1.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                p *= x[j]
                cnt += 1
        if cnt > 0:
            r[i] = p
    return r


@njit(cache=True)
def _ts_quantile_core(x, d, p):
    """[修复] 2026-07-06 WQ nearest-rank: 收集有效值后排序取 ceil(p*cnt)-1 位 (0-indexed)
    替代 np.nanpercentile (linear interpolation), 对齐 WQ/Alpha158 标准
    [修复] 2026-07-06 off-by-one: 原 floor(p*cnt) 在 p*cnt 为整数时多取一位"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                cnt += 1
        if cnt > 0:
            # 收集有效值
            valid = np.empty(cnt)
            k = 0
            for j in range(i - d + 1, i + 1):
                if not np.isnan(x[j]):
                    valid[k] = x[j]
                    k += 1
            # 排序取 nearest-rank (1-indexed: ceil(p*cnt), 转 0-indexed: -1)
            sorted_v = np.sort(valid)
            # [修复] ceil(p*cnt)-1 替代 floor(p*cnt), 并限制范围
            idx = int(np.ceil(p * cnt)) - 1
            if idx < 0:
                idx = 0
            elif idx > cnt - 1:
                idx = cnt - 1
            r[i] = sorted_v[idx]
    return r


@njit(cache=True)
def _ts_zscore_core(x, d):
    """[重构] 2026-07-06 numba @njit 加速: 滚动 Z-score, std≈0 或冷启动返回 NaN"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        s = 0.0
        sq = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                s += x[j]
                sq += x[j] * x[j]
                cnt += 1
        if cnt >= 2:
            mean = s / cnt
            var = (sq - cnt * mean * mean) / (cnt - 1)
            if var < 0.0:
                var = 0.0
            std = np.sqrt(var)
            if std > 1e-10:
                r[i] = (x[i] - mean) / std
    return r


@njit(cache=True)
def _ts_regression_core(y, x, d, rettype):
    """[重构] 2026-07-06 numba @njit 加速: 双变量 y~x 滚动线性回归
    rettype: 0=slope, 1=intercept, 2=resid(y[i]-(a+b*x[i])), 3=predict(a+b*x[i]), 4=rsq"""
    n = len(y)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        sx = 0.0
        sy = 0.0
        sxy = 0.0
        sxx = 0.0
        cnt = 0
        for j in range(i - d + 1, i + 1):
            if not (np.isnan(y[j]) or np.isnan(x[j])):
                sx += x[j]
                sy += y[j]
                sxy += x[j] * y[j]
                sxx += x[j] * x[j]
                cnt += 1
        if cnt >= 3:
            mx = sx / cnt
            my = sy / cnt
            var_x = (sxx - cnt * mx * mx) / (cnt - 1)
            cov_xy = (sxy - cnt * mx * my) / (cnt - 1)
            if var_x < 0.0:
                var_x = 0.0
            if var_x > 1e-15:
                slope = cov_xy / var_x
                intercept = my - slope * mx
                if rettype == 0:
                    r[i] = slope
                elif rettype == 1:
                    r[i] = intercept
                elif rettype == 2:
                    r[i] = y[i] - (intercept + slope * x[i])
                elif rettype == 3:
                    r[i] = intercept + slope * x[i]
                elif rettype == 4:
                    ss_res = 0.0
                    ss_tot = 0.0
                    for j in range(i - d + 1, i + 1):
                        if not (np.isnan(y[j]) or np.isnan(x[j])):
                            pred = intercept + slope * x[j]
                            ss_res += (y[j] - pred) ** 2
                            ss_tot += (y[j] - my) ** 2
                    if ss_tot > 1e-10:
                        r[i] = 1.0 - ss_res / ss_tot
                    else:
                        r[i] = 0.0
    return r


@njit(cache=True)
def _ts_linear_reg_core(x, window, rettype, fill_value):
    """[重构] 2026-07-06 numba @njit 加速: 单变量 x~t 趋势线性回归
    rettype: 0=slope, 1=intercept, 2=resid, 3=predict, 4=rsq
    fill_value: 冷启动/数据不足时的填充值 (ts_resid 传 0.0, 其他传 NaN)
    [修复] 2026-07-06 window<3 时尊重 fill_value (原硬编码 0.0 导致 ts_slope 等返回虚假 0)
    [修复] 2026-07-06 rettype=2/3 用 x[i] 和 window-1 (原用 valid[cnt-1] 和 cnt-1 返回过期值)"""
    n = len(x)
    if window < 3:
        # [修复] 尊重 fill_value, 不硬编码 0.0
        return np.full(n, fill_value)
    r = np.full(n, fill_value)
    for i in range(window - 1, n):
        # [修复] 当前值 x[i] 为 NaN 时无法计算残差/预测, 保持 fill_value
        if np.isnan(x[i]):
            continue
        cnt = 0
        for j in range(i - window + 1, i + 1):
            if not np.isnan(x[j]):
                cnt += 1
        if cnt >= 3:
            valid = np.empty(cnt)
            k = 0
            for j in range(i - window + 1, i + 1):
                if not np.isnan(x[j]):
                    valid[k] = x[j]
                    k += 1
            # t = [0, 1, ..., cnt-1]
            mean_t = (cnt - 1) / 2.0
            mean_x = 0.0
            for k in range(cnt):
                mean_x += valid[k]
            mean_x /= cnt
            sum_t_dev_sq = 0.0
            sum_tx_dev = 0.0
            for k in range(cnt):
                t_dev = k - mean_t
                sum_t_dev_sq += t_dev * t_dev
                sum_tx_dev += t_dev * (valid[k] - mean_x)
            var_t = sum_t_dev_sq / (cnt - 1)
            cov_tx = sum_tx_dev / (cnt - 1)
            if var_t > 1e-15:
                b = cov_tx / var_t
                a = mean_x - b * mean_t
                if rettype == 0:
                    r[i] = b
                elif rettype == 1:
                    r[i] = a
                elif rettype == 2:
                    # [修复] 用 x[i] 和当前时刻 window-1 (原用 valid[cnt-1] 和 cnt-1)
                    predicted = a + b * (window - 1)
                    r[i] = x[i] - predicted
                elif rettype == 3:
                    # [修复] 预测当前时刻 window-1 (原用 cnt-1)
                    r[i] = a + b * (window - 1)
                elif rettype == 4:
                    ss_res = 0.0
                    ss_tot = 0.0
                    for k in range(cnt):
                        pred = a + b * k
                        ss_res += (valid[k] - pred) ** 2
                        ss_tot += (valid[k] - mean_x) ** 2
                    if ss_tot > 1e-10:
                        r[i] = 1.0 - ss_res / ss_tot
                    else:
                        r[i] = 0.0
    return r


@njit(cache=True)
def _cs_zscore_core(x):
    """[重构] 2026-07-06 numba @njit 加速: 2D 截面 Z-score, 逐行计算, 跳过 NaN"""
    n_rows, n_cols = x.shape
    r = np.full((n_rows, n_cols), np.nan)
    for i in range(n_rows):
        s = 0.0
        sq = 0.0
        cnt = 0
        for j in range(n_cols):
            if not np.isnan(x[i, j]):
                s += x[i, j]
                sq += x[i, j] * x[i, j]
                cnt += 1
        if cnt > 1:
            mean = s / cnt
            var = (sq - cnt * mean * mean) / (cnt - 1)
            if var < 0.0:
                var = 0.0
            std = np.sqrt(var)
            for j in range(n_cols):
                if not np.isnan(x[i, j]):
                    if std > 1e-10:
                        r[i, j] = (x[i, j] - mean) / std
                    else:
                        r[i, j] = 0.0
    return r


@njit(cache=True)
def _cs_scale_core(x, scale):
    """[重构] 2026-07-06 numba @njit 加速: 2D 截面缩放, x/sum(|x|)*scale"""
    n_rows, n_cols = x.shape
    r = np.full((n_rows, n_cols), np.nan)
    for i in range(n_rows):
        s = 0.0
        cnt = 0
        for j in range(n_cols):
            if not np.isnan(x[i, j]):
                s += abs(x[i, j])
                cnt += 1
        if cnt > 0:
            if s > 1e-10:
                for j in range(n_cols):
                    if not np.isnan(x[i, j]):
                        r[i, j] = x[i, j] / s * scale
            else:
                for j in range(n_cols):
                    if not np.isnan(x[i, j]):
                        r[i, j] = 0.0
    return r


@njit(cache=True)
def _cs_winsorize_core(x, std_n):
    """[重构] 2026-07-06 numba @njit 加速: 2D 截面缩尾, 逐行 +/-std_n*std"""
    n_rows, n_cols = x.shape
    r = x.copy()
    for i in range(n_rows):
        s = 0.0
        sq = 0.0
        cnt = 0
        for j in range(n_cols):
            if not np.isnan(x[i, j]):
                s += x[i, j]
                sq += x[i, j] * x[i, j]
                cnt += 1
        if cnt > 1:
            mean = s / cnt
            var = (sq - cnt * mean * mean) / (cnt - 1)
            if var < 0.0:
                var = 0.0
            std = np.sqrt(var)
            lo = mean - std_n * std
            hi = mean + std_n * std
            for j in range(n_cols):
                if not np.isnan(x[i, j]):
                    if x[i, j] < lo:
                        r[i, j] = lo
                    elif x[i, j] > hi:
                        r[i, j] = hi
    return r


# ============================================================
# 时序函数 (ts_) — 窗口滚动, 只用历史数据
# ============================================================

# [修正] 2026-06-25 改用 nan* 版本, 对齐 WQ NaN=缺失数据应忽略的规范
# [重构] 2026-07-06 改用 numba @njit 加速, _rolling 作为 fallback 保留
def ts_mean(x, d):       return _ts_mean_core(np.asarray(x, float), int(d))
def ts_std(x, d):        return _ts_std_core(np.asarray(x, float), int(d))
def ts_sum(x, d):
    """滚动窗口求和, 忽略 NaN"""
    return _ts_sum_core(np.asarray(x, float), int(d))
def ts_max(x, d):        return _ts_max_core(np.asarray(x, float), int(d))
def ts_min(x, d):        return _ts_min_core(np.asarray(x, float), int(d))
def ts_median(x, d):     return _ts_median_core(np.asarray(x, float), int(d))

def ts_delta(x, d):
    x = np.asarray(x, float); r = np.full_like(x, np.nan)
    r[d:] = x[d:] - x[:-d]; return r

def ts_delay(x, d):
    x = np.asarray(x, float); r = np.full_like(x, np.nan)
    r[d:] = x[:-d]; return r

def ts_rank(x, d):
    """[修复] 2026-07-06 过滤 NaN 后计算排名, 避免 NaN 干扰 searchsorted
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_rank_core(np.asarray(x, float), int(d))

def ts_corr(x, y, d):
    """[修复] 2026-07-06 过滤 NaN 后计算, 对齐 WQ NaN=缺失数据应忽略规范
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_corr_core(np.asarray(x, float), np.asarray(y, float), int(d))

def ts_cov(x, y, d):
    """Rolling covariance (样本协方差, ddof=1)
    [修复] 2026-07-06 过滤 NaN 后计算, 对齐 WQ 规范
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_cov_core(np.asarray(x, float), np.asarray(y, float), int(d))

def ts_skew(x, d):
    """[修复] 2026-07-06 过滤 NaN 后计算, 对齐 WQ 规范
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_skew_core(np.asarray(x, float), int(d))

def ts_kurt(x, d):
    """[修复] 2026-07-06 过滤 NaN 后计算, 对齐 WQ 规范
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_kurt_core(np.asarray(x, float), int(d))

def ts_argmax(x, d):
    """[重构] 2026-07-06 numba @njit 加速"""
    return _ts_argmax_core(np.asarray(x, float), int(d))

def ts_argmin(x, d):
    """[重构] 2026-07-06 numba @njit 加速"""
    return _ts_argmin_core(np.asarray(x, float), int(d))

def ts_roc(x, d):
    """Rate of change: (x[t]-x[t-d])/x[t-d]"""
    x = np.asarray(x, float); r = np.full_like(x, np.nan)
    r[d:] = (x[d:] - x[:-d]) / np.where(np.abs(x[:-d]) > 1e-10, x[:-d], np.nan)
    return r

def ts_zscore(x, d):
    """滚动 Z-score (样本标准差, ddof=1)
    [修复] 2026-07-06 std≈0 或冷启动期返回 NaN (原返回0违反冷启动保护, 产生虚假信号)
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_zscore_core(np.asarray(x, float), int(d))

def ts_scale(x, d):
    """[修正] 2026-06-25 过滤 NaN, 对齐 WQ 标准
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_scale_core(np.asarray(x, float), int(d))

def ts_quantile(x, d, p=0.5):
    """滚动分位数值: 返回窗口内 p 分位数的值 (非排名)
    [重构] 2026-07-06 numba @njit 加速

    p=0.5 → 中位数, p=0.8 → 80%分位数, p=0.2 → 20%分位数
    """
    return _ts_quantile_core(np.asarray(x, float), int(d), float(p))

def ts_av_diff(x, d):
    """Current minus rolling mean (deviation)
    [重构] 2026-07-06 numba @njit 加速"""
    x = np.asarray(x, float)
    return x - _ts_mean_core(x, int(d))

def ts_decay_linear(x, d):
    """[修正] 2026-06-25 过滤 NaN 后加权, 对齐 WQ 标准
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_decay_linear_core(np.asarray(x, float), int(d))

def ts_product(x, d):
    """Rolling product over d periods
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_product_core(np.asarray(x, float), int(d))

def ts_regression(y, x, d, rettype=2):
    """Rolling linear regression: y = alpha + beta*x + eps

    Args:
        y: 因变量
        x: 自变量
        d: 窗口
        rettype: 0=斜率, 1=截距, 2=残差, 3=预测值, 4=R²

    [重构] 2026-07-06 numba @njit 加速
    """
    return _ts_regression_core(np.asarray(y, float), np.asarray(x, float), int(d), int(rettype))


def ts_resid(x, window):
    """时序回归残差: 对时间做线性回归 x=a+b*t，返回当前值-预测值
    [重构] 2026-07-06 numba @njit 加速
    [修复] 2026-07-06 冷启动期返回 NaN (原 0.0 违反冷启动保护, 掩盖数据不足)"""
    return _ts_linear_reg_core(np.asarray(x, float), int(window), 2, np.nan)


def ts_slope(x, window):
    """时序线性回归斜率: 对时间做线性回归 x=a+b*t，返回斜率 b
    [重构] 2026-07-06 numba @njit 加速, 默认填充值 NaN"""
    return _ts_linear_reg_core(np.asarray(x, float), int(window), 0, np.nan)


def ts_rsq(x, window):
    """时序线性回归 R²: 对时间做线性回归 x=a+b*t，返回 R²
    [重构] 2026-07-06 numba @njit 加速, 默认填充值 NaN"""
    return _ts_linear_reg_core(np.asarray(x, float), int(window), 4, np.nan)


def ts_intercept(x, window):
    """时序线性回归截距: 对时间做线性回归 x=a+b*t，返回截距 a
    [重构] 2026-07-06 numba @njit 加速, 默认填充值 NaN"""
    return _ts_linear_reg_core(np.asarray(x, float), int(window), 1, np.nan)


def ts_predict(x, window):
    """时序线性回归预测值: 对时间做线性回归 x=a+b*t，返回当前时刻预测值 a+b*(w-1)
    [重构] 2026-07-06 numba @njit 加速, 默认填充值 NaN"""
    return _ts_linear_reg_core(np.asarray(x, float), int(window), 3, np.nan)


# ============================================================
# 扩张统计 (expanding_) — 起始→当前, 无固定窗口
# ============================================================

def expanding_mean(x, min_p=20):    return _expanding(x, np.mean, min_p)
def expanding_median(x, min_p=20):  return _expanding(x, np.median, min_p)
def expanding_std(x, min_p=20):     return _expanding(x, lambda a: np.std(a, ddof=1), min_p)
def expanding_percentile(x, p, min_p=20):
    return _expanding(x, lambda a: np.percentile(a, p * 100), min_p)


# ============================================================
# 截面函数 (cs_) — 每日跨品种统计, 需完整 2D 面板
# ============================================================

def cs_rank(x):
    """[修正] 2026-06-25 改为 method='min' (最小排名), 对齐 WQ/DolphinDB 行业标准.
    WQ rank 规范: 并列值取最小排名, range [1/N, 1]. 见 DolphinDB WQ101 实现.
    旧版 average 排名对离散信号(ts_argmax等)引入虚假区分度, 连续信号无影响."""
    x = np.asarray(x, float)
    if x.ndim == 1: return np.full_like(x, 0.5)
    from scipy.stats import rankdata
    r = np.full_like(x, np.nan)
    for i in range(x.shape[0]):
        v = ~np.isnan(x[i])
        if v.sum() > 0:
            rk = rankdata(x[i][v], method='min') / v.sum()
            r[i][v] = rk
    return r

def cs_zscore(x):
    """[重构] 2026-07-06 numba @njit 加速 (2D); 1D 保持原逻辑返回 zeros"""
    x = np.asarray(x, float)
    if x.ndim == 1: return np.zeros_like(x)
    return _cs_zscore_core(x)

def cs_scale(x, scale=1.0):
    """Cross-sectional scale: sum(abs(x)) = scale
    [重构] 2026-07-06 numba @njit 加速 (2D); 1D 保持原逻辑"""
    x = np.asarray(x, float)
    if x.ndim == 1: return x / (np.sum(np.abs(x)) + 1e-10) * scale
    return _cs_scale_core(x, float(scale))

def cs_winsorize(x, std=4.0):
    """Cross-sectional winsorize at +/-std (样本标准差, ddof=1)
    [重构] 2026-07-06 numba @njit 加速 (2D); 1D 保持原逻辑整体截尾"""
    x = np.asarray(x, float)
    if x.ndim == 1:
        m, s = np.mean(x), np.std(x, ddof=1)
        return np.clip(x, m - std * s, m + std * s)
    return _cs_winsorize_core(x, float(std))

def cs_normalize(x, use_std=False, limit=0.0):
    """Cross-sectional normalize"""
    if use_std:
        return cs_scale(cs_zscore(x))
    return cs_scale(x, 1.0)

def cs_quantile(x, driver='gaussian', sigma=1.0):
    """Cross-sectional quantile (等价 cs_rank)"""
    return cs_rank(x)


# ============================================================
# 数学函数 — 逐元素安全运算
# ============================================================

def safe_abs(x):          return np.abs(x)
def safe_log(x):
    """[修复] 2026-07-06 非正数返回 NaN (原 abs(x) 导致 log(-1)==log(1) 掩盖负数错误)
    若需处理负数输入的量级, 请用 sign(x)*log(abs(x)+1) 显式表达"""
    x = np.asarray(x, float)
    result = np.full_like(x, np.nan)
    mask = x > 1e-10
    result[mask] = np.log(x[mask])
    return result
def safe_sqrt(x):         return np.sqrt(np.maximum(x, 0.0))
def safe_sign(x):         return np.sign(x)
def safe_exp(x):          return np.exp(np.clip(x, -50, 50))
def safe_tanh(x):         return np.tanh(x)
def safe_sigmoid(x):      return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))
def safe_relu(x):         return np.maximum(x, 0.0)

# [新增] 2026-06-23 钟形函数族: 中心最高, 两边递减, 不同尾重
#   cos(x)      — 周期震荡, 主瓣单调到π后振荡, 窄范围等价于钟形
#   gauss(x)    — exp(-x²), 轻尾钟形, 对极端值惩罚最重, 适合宽范围变量
#   p4(x)       — exp(-x⁴), 平顶陡降, 对微小变化不敏感, 适合极窄范围
#   exp_neg(x)  — exp(-|x|), 尖峰中尾, 介于cos和gauss之间
def safe_neg(x):
    """负号: -x (命名沿用历史, 无 NaN 特殊处理)"""
    return -np.asarray(x, float)

def safe_gauss(x):        return np.exp(-np.asarray(x, float)**2)
def safe_p4(x):           return np.exp(-np.asarray(x, float)**4)

def safe_softsign(x):
    """Soft Sign: x / (1 + |x|), 值域 (-1, 1)
    奇函数，奇性: 奇（f(-x)=-f(x)），原点斜率=1
    与 tanh 差异: 多项式尾 vs 指数尾，极端值仍有区分度
    tanh(2)=0.96, softsign(2)=0.67
    tanh(5)=0.9999, softsign(5)=0.833
    排序假设: "方向重要，且方向和幅度都有区分度"
    """
    x = np.asarray(x, float)
    return x / (1.0 + np.abs(x))

def safe_square_sigmoid(x):
    """平方 sigmoid: x²/(1+x²), 值域 [0, 1)
    倒钟形: 谷在0=0, ±1=0.5, ±3=0.9, ±5=0.96, ±10=0.99
    与钟形族(cos/gauss/p4)相反: 它们峰在0，这个是谷在0
    排序假设: "偏离0才有信息，越极端置信度越高"
    组合用法: sign(x)*square_sigmoid(x) 同时获取方向+极端度
    """
    x = np.asarray(x, float)
    return x**2 / (1.0 + x**2)

def signed_power(x, exponent=2.0):
    """带符号幂变换: sign(x) * |x|^exponent
    保留方向，非线性放大/压缩幅度。
    exponent>1 放大极端值，exponent<1 压缩振幅。
    """
    return np.sign(x) * np.power(np.abs(x), float(exponent))

def safe_max(x, y):    return np.maximum(x, y)
def safe_min(x, y):    return np.minimum(x, y)





# ============================================================
# 信号函数
# ============================================================
#
# ── 非对称买卖模式（单表达式即可实现，无需额外状态机） ──
#
# 方向非对称:  上涨追动量，下跌等反转
#   "ts_roc(CLOSE, 20) if SECTOR_UP > 0.5 else -ts_roc(CLOSE, 10)"
#   → 广度好时做多追涨，广度差时做多抄底
#
# 时长非对称:  买入需持续性确认，卖出单日即可
#   "persist(ts_roc(CLOSE, 5) > 0, 2) if BREADTH_L > 0.6 else -1 if CLOSE < ts_delay(CLOSE, 1) * 0.98 else 0"
#   → 买入需要连续2天确认，破位当日立即卖出
#
# 状态门控:    趋势/震荡两套逻辑
#   "(ts_roc(CLOSE, 20) if adx(HIGH, LOW, CLOSE, 14) > 30 else -ts_roc(CLOSE, 5) if adx(HIGH, LOW, CLOSE, 14) < 15 else 0)"
#   → 强趋势追涨，弱趋势反转，中间观望
#
# 量价背离:    价格高位但量不配合 → 卖出
#   "-1 if (CLOSE / ts_max(CLOSE, 20) > 0.98 and VOLUME < ts_mean(VOLUME, 5)) else 0"
#
# 所有模式输出为连续信号线，引擎层按 >0 做多 / <0 做空 解释。

def persist(x, n=3):
    return _persist(x, n)


# ============================================================
# 特征计算函数（从原始 OHLCV 数组实时算，无需 FeatureSpace）
# ============================================================

def _feature_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """RSI centered: (RSI-50)/50, range [-1,1]. talib-aligned.
    [修复] 2026-07-06 保留冷启动期 NaN (原 nan_to_num 填充 50.0 掩盖缺失数据, 与冷启动保护原则冲突)"""
    c = np.asarray(close, float)
    result = talib.RSI(c, timeperiod=period)
    return (result - 50) / 50


def _feature_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR - Wilder smoothed. talib-aligned."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.ATR(h, l, c, timeperiod=period)


def _feature_atr_sma(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR-SMA: simple rolling mean of TR (original V4, kept for compat)
    [修复] 2026-07-06 用 shift 替代 roll, 避免首位引入末尾数据 (NaN 安全)"""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    prev_c = np.empty_like(c)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    # [重构] 2026-07-06 改用 ts_mean (numba @njit 加速)
    return ts_mean(tr, period)


def _wilder_smooth(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder smoothing: S[t]=(S[t-1]*(p-1)+x[t])/p, seed=SMA. talib-aligned."""
    x = np.asarray(x, float)
    n = len(x)
    r = np.full(n, np.nan)
    if n < period:
        return r
    r[period - 1] = np.mean(x[:period])
    for i in range(period, n):
        if np.isnan(r[i - 1]):
            r[i] = np.nan
        else:
            r[i] = (r[i - 1] * (period - 1) + x[i]) / period
    return r


def _feature_ema(x: np.ndarray, period: int = 20) -> np.ndarray:
    """EMA: recursive, talib-aligned. k=2/(p+1), seed=SMA."""
    x = np.asarray(x, float)
    n = len(x)
    k = 2.0 / (period + 1)
    r = np.full(n, np.nan)
    if n < period:
        return r
    r[period - 1] = np.mean(x[:period])
    for i in range(period, n):
        if np.isnan(r[i - 1]):
            r[i] = np.nan
        else:
            r[i] = (x[i] - r[i - 1]) * k + r[i - 1]
    return r


def _feature_bbwidth(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Bollinger Band Width = (upper-lower)/middle. talib-aligned."""
    c = np.asarray(close, float)
    upper, middle, lower = talib.BBANDS(c, timeperiod=period, nbdevup=2, nbdevdn=2, matype=0)
    return np.where(middle > 0, (upper - lower) / middle, 0)


def _feature_stddev(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Standard Deviation. talib-aligned."""
    c = np.asarray(close, float)
    return talib.STDDEV(c, timeperiod=period, nbdev=1)


def _feature_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX - Average Directional Index. talib-aligned."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.ADX(h, l, c, timeperiod=period)


def _feature_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """CCI - Commodity Channel Index. talib-aligned."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.CCI(h, l, c, timeperiod=period)


def _feature_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD histogram = 2*(DIF-DEA). talib-aligned."""
    c = np.asarray(close, float)
    dif, dea, hist = talib.MACD(c, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    return 2 * (dif - dea)


def _feature_trima(close: np.ndarray, period: int = 40) -> np.ndarray:
    """TRIMA - Triangular Moving Average. talib-aligned."""
    c = np.asarray(close, float)
    return talib.TRIMA(c, timeperiod=period)


def _feature_tsf(close: np.ndarray, period: int = 7) -> np.ndarray:
    """TSF - Time Series Forecast. talib-aligned."""
    c = np.asarray(close, float)
    return talib.TSF(c, timeperiod=period)


def _feature_kama(close: np.ndarray, period: int = 30) -> np.ndarray:
    """KAMA - Kaufman Adaptive Moving Average. talib-aligned."""
    c = np.asarray(close, float)
    return talib.KAMA(c, timeperiod=period)


def _feature_wma(close: np.ndarray, period: int = 20) -> np.ndarray:
    """WMA - Weighted Moving Average. talib-aligned."""
    c = np.asarray(close, float)
    return talib.WMA(c, timeperiod=period)


def _feature_dema(close: np.ndarray, period: int = 20) -> np.ndarray:
    """DEMA - Double Exponential Moving Average. talib-aligned."""
    c = np.asarray(close, float)
    return talib.DEMA(c, timeperiod=period)


def _feature_hv(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Historical Volatility: annualized std of daily returns x100."""
    c = np.asarray(close, float)
    rets = np.diff(c) / np.where(c[:-1] > 0, c[:-1], 1)
    rets = np.insert(rets, 0, 0)
    return _rolling(rets, period, lambda a: np.std(a, ddof=1) * np.sqrt(252) * 100)


def _feature_natr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """NATR - Normalized ATR. talib-aligned."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.NATR(h, l, c, timeperiod=period)


def _feature_var(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Variance. talib-aligned."""
    c = np.asarray(close, float)
    return talib.VAR(c, timeperiod=period)


def _feature_linearreg(close: np.ndarray, period: int = 20) -> np.ndarray:
    """Linear Regression value. talib-aligned."""
    c = np.asarray(close, float)
    return talib.LINEARREG(c, timeperiod=period)


def _feature_vol_ratio(close: np.ndarray, volume: np.ndarray, short: int = 5, long: int = 20) -> np.ndarray:
    """Volume ratio = SMA(vol,short) / SMA(vol,long). No talib equivalent.
    [重构] 2026-07-06 改用 ts_mean (numba @njit 加速)"""
    v = np.asarray(volume, float)
    # [重构] 2026-07-06 改用 ts_mean (numba @njit 加速)
    vs = ts_mean(v, short)
    vl = ts_mean(v, long)
    return np.where(vl > 0, vs / vl, 0)


def _feature_amt_ratio(amount: np.ndarray, short: int = 5, long: int = 20) -> np.ndarray:
    """Amount ratio = short_mean / long_mean
    [重构] 2026-07-06 改用 ts_mean (numba @njit 加速)"""
    a = np.asarray(amount, float)
    # [重构] 2026-07-06 改用 ts_mean (numba @njit 加速)
    return np.where(ts_mean(a, long) > 0,
                    ts_mean(a, short) / ts_mean(a, long), 0)


# ============================================================
# 函数注册表
# ============================================================

FUNC_REGISTRY: Dict[str, Callable] = {
    # ── 时序 (ts_) ──
    'ts_mean':   ts_mean,   'ts_std':    ts_std,
    'ts_sum':    ts_sum,    'ts_max':    ts_max,
    'ts_min':    ts_min,    'ts_median': ts_median,
    'ts_delta':  ts_delta,  'ts_delay':  ts_delay,
    'ts_rank':   ts_rank,   'ts_corr':   ts_corr,
    'ts_skew':   ts_skew,   'ts_kurt':   ts_kurt,
    'ts_argmax': ts_argmax, 'ts_argmin': ts_argmin,
    'ts_roc':    ts_roc,
    'ts_cov':      ts_cov,
    'ts_var':      lambda x, d: ts_std(x, d) ** 2,
    'ts_logret':   lambda x: safe_log(x / ts_delay(x, 1)),
    'ts_zscore':   ts_zscore,
    'ts_scale':    ts_scale,
    'ts_quantile': ts_quantile,
    'ts_av_diff':  ts_av_diff,
    'ts_decay_linear': ts_decay_linear,
    'ts_product':  ts_product,
    'ts_regression': ts_regression,
    'ts_reg_slope':      lambda y, x, d: ts_regression(y, x, d, 0),
    'ts_reg_intercept':  lambda y, x, d: ts_regression(y, x, d, 1),
    'ts_reg_resid':      lambda y, x, d: ts_regression(y, x, d, 2),
    'ts_reg_predict':    lambda y, x, d: ts_regression(y, x, d, 3),
    'ts_reg_rsq':        lambda y, x, d: ts_regression(y, x, d, 4),
    'ts_resid': ts_resid,                    # 统计学标准缩写
    'ts_slope': ts_slope,
    'ts_rsq': ts_rsq,                        # 统计学标准缩写
    'ts_intercept': ts_intercept,
    'ts_predict': ts_predict,

    # ── 扩张统计 (expanding_) ──
    'expanding_mean': expanding_mean, 'expanding_median': expanding_median,
    'expanding_std':  expanding_std,  'expanding_percentile': expanding_percentile,

    # ── 截面 (cs_) ──
    'cs_rank':   cs_rank,   'cs_zscore':   cs_zscore,
    'cs_scale':  cs_scale,  'cs_winsorize': cs_winsorize,
    'cs_quantile': cs_quantile,

    # ── 数学 ──
    'abs': safe_abs, 'log': safe_log, 'sqrt': safe_sqrt,
    'sign': safe_sign, 'exp': safe_exp, 'tanh': safe_tanh,
    'sigmoid': safe_sigmoid, 'relu': safe_relu, 'softsign': safe_softsign,
    'sin': lambda x: np.sin(x), 'cos': lambda x: np.cos(x),
    'gauss': safe_gauss, 'p4': safe_p4, 'neg': safe_neg, 'square_sigmoid': safe_square_sigmoid,
    'signed_power': signed_power,
    'safe_max': safe_max, 'safe_min': safe_min,

    # ── 信号 ──
    'persist': persist,

    # ── 特征计算 (从 OHLCV 实时算, 无需 FeatureSpace) ──
    'rsi':         _feature_rsi,
    'atr':         _feature_atr,
    'atr_sma':     _feature_atr_sma,
    'bb_width':    _feature_bbwidth,
    'stddev':      _feature_stddev,
    'adx':         _feature_adx,
    'cci':         _feature_cci,
    'macd':        _feature_macd,
    'trima':       _feature_trima,
    'ema':         _feature_ema,
    'wilder_smooth': _wilder_smooth,
    'tsf':         _feature_tsf,
    'kama':        _feature_kama,
    'wma':         _feature_wma,
    'dema':        _feature_dema,
    'hv':          _feature_hv,
    'natr':        _feature_natr,
    'var':         _feature_var,
    'linearreg':   _feature_linearreg,
    'vol_ratio':   _feature_vol_ratio,
    'amt_ratio':   _feature_amt_ratio,

    # ── 旧名别名 — [2026-07-03] 已停止生产使用, 后续删除 ──
    'ts_resi': ts_resid,                     # → 改用 ts_resid
    'ts_regression_residual': ts_resid,      # → 改用 ts_resid
    'ts_rsquare': ts_rsq,                    # → 改用 ts_rsq
}

# 安全常量 (表达式中的 True/False/None/pi/e)
SAFE_CONSTANTS = {'True': 1.0, 'False': 0.0, 'None': 0.0, 'pi': np.pi, 'e': np.e}


# ============================================================
# 函数分类索引 (供 LLM 理解可用原语)
# ============================================================

FUNC_CATEGORIES = {
    '时序统计': ['ts_mean', 'ts_std', 'ts_sum', 'ts_max', 'ts_min', 'ts_median',
                'ts_delta', 'ts_delay', 'ts_rank', 'ts_corr', 'ts_cov',
                'ts_skew', 'ts_kurt', 'ts_argmax', 'ts_argmin',
                'ts_roc', 'ts_zscore', 'ts_scale', 'ts_quantile',
                'ts_av_diff', 'ts_decay_linear', 'ts_product', 'ts_var', 'ts_logret',
                'ts_regression', 'ts_resi', 'ts_reg_slope', 'ts_reg_intercept',
                'ts_reg_resid', 'ts_reg_predict', 'ts_reg_rsq',
                'ts_intercept', 'ts_predict', 'ts_resid', 'ts_rsq'],
    '扩张统计': ['expanding_mean', 'expanding_median', 'expanding_std', 'expanding_percentile'],
    '截面算子': ['cs_rank', 'cs_zscore', 'cs_scale', 'cs_winsorize', 'cs_quantile'],
    '特征计算': ['rsi', 'atr', 'atr_sma', 'macd', 'adx', 'cci', 'bb_width', 'stddev',
                'ema', 'tsf', 'kama', 'trima', 'wma', 'dema', 'hv', 'natr',
                'var', 'linearreg', 'vol_ratio', 'amt_ratio', 'wilder_smooth'],
    '数学运算': ['abs', 'log', 'sqrt', 'sign', 'exp', 'tanh', 'sigmoid', 'relu',
                'sin', 'cos', 'gauss', 'p4', 'softsign', 'square_sigmoid',
                'signed_power', 'safe_max', 'safe_min'],
    '信号确认': ['persist'],
}


def get_func_category(name: str) -> str:
    """查询函数所属分类"""
    for cat, names in FUNC_CATEGORIES.items():
        if name in names:
            return cat
    return '自定义'


# ============================================================
# 临时自定义注册 (LLM 探索时热添加，无需修改 functions.py)
# ============================================================
#
# 使用场景:
#   每个探索脚本是独立进程, 跑完即销毁。脚本顶部注册 → 全文使用 →
#   进程退出自动清零, 无需手动清理。不存在跨脚本污染问题。
#
# 用法:
#   from utils.ast import register_function
#   register_function('my_indicator', lambda x, w: np.convolve(x, np.ones(w)/w, 'same'))
#   expr = Expression("MY_VAR > 0 and my_indicator(CLOSE, 10) > 0")
#
# 兼容旧路径:
#   from signals.v4 import register_function  # 仍可用，重导出链

def register_function(name: str, func: Callable) -> None:
    """临时注册自定义函数到表达式引擎。

    进程级全局注册, 当前脚本内所有 Expression 生效。
    脚本退出自动销毁, 无泄漏风险。

    Args:
        name: 函数名 (表达式中的调用名)
        func: 函数实现，签名为 (*np.ndarray) -> np.ndarray
    """
    name_lower = name.lower()
    if name_lower in FUNC_REGISTRY:
        import warnings
        warnings.warn(
            f"register_function: '{name}' 已存在，将被覆盖。"
            f"原函数: {FUNC_REGISTRY[name_lower].__name__}"
        )
    FUNC_REGISTRY[name_lower] = func


def unregister_function(name: str) -> bool:
    """注销自定义函数，返回是否成功。内置函数不可注销。"""
    return FUNC_REGISTRY.pop(name.lower(), None) is not None
