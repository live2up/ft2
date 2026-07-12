# -*- coding: utf-8 -*-
"""
utils/ast/functions.py — 原语层 (公共基础设施)
=============================================================================

在五层架构中的位置: 第2层(原语) — 定义"能算什么"

  函数定义 = 86 个 | FUNC_REGISTRY 条目 = 91 个 (含 4 别名)

═══════════════════════════════════════════════════════════════
命名规范 (对齐 WQ101 行业标准)

  ◆ 函数命名 (snake_case, 小写 + 前缀)
    前缀约定:
      ts_       → Time Series (窗口滚动, 只用历史数据)
      reg_      → 双变量滚动线性回归 y~x (reg_slope/reg_resid 等)
      cs_       → Cross Sectional (截面统计, 需完整 2D 面板)
      expanding_→ 扩展窗口 (起始→当前, 无固定窗口)
      无前缀    → 逐元素数学 / talib 特征 / 信号

    回归函数命名区分:
      reg_*(y, x, d)  → 双变量 y~x 回归 (如 CLOSE~VOLUME)
      ts_*(x, w)      → 单变量 x~t 趋势回归 (如 CLOSE~time)

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

  ◆ 统计量约定 (对齐 WQ101 / GT191, 主流实现 pandas/scipy)
    ddof=1 (样本):   ts_std, ts_cov, cs_zscore, cs_winsorize
    ddof=0 (总体):   ts_skew, ts_kurt (偏度/峰度的 z-score 用总体方差, 然后施加 Fisher-Pearson 样本修正, 对齐 pandas rolling().skew()/.kurt())
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

  ├─ 时序 ts_ (31 + 1 lambda = 32 条目)
  │  基础:      mean / std / sum / max / min / median
  │  周期:      delta / delay / roc
  │  统计:      rank / zscore / scale / quantile / av_diff
  │             var(λ) / decay_linear / product
  │  相关:      corr / cov / autocorr
  │  持续性:    step
  │  形态:      hump
  │  分布形状:  skew / kurt / argmax / argmin
  │  x~t趋势:   slope / resid / rsq / intercept / predict
  │  自回归:    ar_resid
  │  [y~x回归]  ts_reg_slope / ts_reg_intercept / ts_reg_resid
  │             ts_reg_predict / ts_reg_rsq (均调用 ts_regression)

  ├─ 扩张 expanding_ (4)
  │  expanding_mean / expanding_median / expanding_std / expanding_percentile

  ├─ 截面 cs_ (6)
  │  cs_rank / cs_zscore / cs_scale / cs_winsorize
  │  cs_normalize / cs_quantile

  ├─ 数学 (18)
  │  基础:      abs / log / sqrt / sign / exp / max / min / logret
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

  ├─ 旧名别名 (4) — [2026-07-03] 已停止生产使用, 后续删除
  │  ts_resi                    → 改用 ts_resid
  │  ts_regression_residual     → 改用 reg_resid
  │  ts_rsquare                 → 改用 ts_rsq
  │  ts_logret                  → 改用 logret

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

  # 防除零模式 (6处: ts_roc/ts_zscore/ts_scale/cs_zscore/cs_scale/regression)
  result = value / divisor if divisor > 1e-10 else 0.0

  # NaN过滤模式 (7处: ts_product/regression/ts_resid/
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
from dataclasses import dataclass, field
from numba import njit
from typing import Any, Callable, Dict, List, Optional, Union


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


# [修复] 2026-07-06 窗口参数校验: d<1 时 range(d-1, n) 会从 -1 开始, 静默污染末元素
def _validate_window(d, name):
    """校验窗口参数 d >= 1, 返回 int(d)"""
    d_int = int(d)
    if d_int < 1:
        raise ValueError(
            f"窗口参数 d={d} 无效, 必须 >= 1 (函数: {name})"
        )
    return d_int


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
    """滚动排名: 当前值 x[i] 在窗口内的归一化排名 [1/cnt, 1]。

    算法: 收集窗口内有效值 → 排序 → searchsorted 定位 x[i] → (pos+1)/cnt
    返回: 值域 [1/cnt, 1], 冷启动/NaN 返回 NaN
    [对齐] WQ101 ts_rank(x,d): 当前值在窗口内的排名位置, 归一化到 (0,1]
    [分歧] ft2 输出范围 (0, 1], 最低得 1/cnt 而非 0。这是为避免 0 值被下游算子
           静默忽略, 保留最小区分度。对相对排名无影响。
    [重构] 2026-07-06 numba @njit 加速
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
    """滚动窗口最大值位置: 返回最大值距今天数。

    算法: 遍历窗口 [i-d+1, i], 记录最大值及其索引 → i - best_idx
    返回: 值域 [0, d-1], 0=当前日出现最大值, d-1=最远日出现最大值
    冷启动/全 NaN 窗口返回 NaN
    [对齐] WQ101 ts_argmax(x,d): 返回窗口内最大值位置。ft2 采用距今天数 (0=当前)，
            比 WQ101/DolphinDB 的 0=最旧约定更直观，使用时注意方向相反。
    [重构] 2026-07-06 numba @njit 加速"""
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
    """滚动窗口最小值位置: 返回最小值距今天数。

    算法: 遍历窗口 [i-d+1, i], 记录最小值及其索引 → i - best_idx
    返回: 值域 [0, d-1], 0=当前日出现最小值, d-1=最远日出现最小值
    冷启动/全 NaN 窗口返回 NaN
    [对齐] WQ101 ts_argmin(x,d): 返回窗口内最小值位置。ft2 采用距今天数 (0=当前)，
            比 WQ101/DolphinDB 的 0=最旧约定更直观，使用时注意方向相反。
    [重构] 2026-07-06 numba @njit 加速"""
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
    """滚动 Pearson 相关系数: cov(x,y) / (std_x * std_y)。

    算法: 同时跳过 x/y 的 NaN → 计算样本协方差和方差 (ddof=1) → 相关系数
    返回: 值域 [-1, 1], 有效值<3 或 std_x/std_y≈0 时返回 NaN
    [对齐] WQ101 correlation(x,y,d): Pearson 相关系数, ddof=1, 跳过 NaN
    [重构] 2026-07-06 numba @njit 加速
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
    """滚动样本协方差: cov(x,y) = Σ((x-mean_x)*(y-mean_y)) / (n-1)。

    算法: 同时跳过 x/y 的 NaN → 计算样本协方差 (ddof=1)
    返回: 值域无界, 有效值<3 返回 NaN
    [对齐] WQ101 covariance(x,y,d): 样本协方差, ddof=1, 跳过 NaN
    [重构] 2026-07-06 numba @njit 加速"""
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
    """样本偏度 (Fisher-Pearson): 衡量分布不对称程度。

    算法: 总体偏度 g1 = m3/m2^1.5 → Fisher-Pearson 样本修正 G1 = g1 * sqrt(n*(n-1))/(n-2)
    返回: 值域无界, 有效值<3 返回 NaN
    [对齐] pandas/scipy skew(bias=False): Fisher-Pearson 样本偏度
    [分歧] 使用总体方差 ddof=0 计算 m2, 与 ddof=1 的 ts_std 不一致。
           标准做法应统一 ddof, 但当前公式与 m3/m4 分母保持一致 (均为 cnt)。
            对排名类因子无影响, 对阈值类用法有 ~7.4% 偏差 (n=20 时)。 (审核报告 P2-3)
    [修复] 2026-07-06 统一为 Fisher-Pearson 样本修正"""
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
            # 总体方差 ddof=0 (m2 需与 m3/m4 分母一致)
            m2 = (sq - cnt * mean * mean) / cnt
            if m2 < 0.0:
                m2 = 0.0
            std = np.sqrt(m2)
            if std > 1e-10:
                m3 = 0.0
                for j in range(i - d + 1, i + 1):
                    if not np.isnan(x[j]):
                        z = (x[j] - mean) / std
                        m3 += z * z * z
                # 总体偏度 → Fisher-Pearson 样本偏度修正
                g1 = m3 / cnt
                r[i] = g1 * np.sqrt(cnt * (cnt - 1)) / (cnt - 2)
    return r


@njit(cache=True)
def _ts_kurt_core(x, d):
    """样本超额峰度 (Fisher): 衡量分布尾重程度。

    算法: 总体超额峰度 g2 = m4/m2^2 - 3 → Fisher 样本修正 G2 = (n-1)/((n-2)(n-3)) * ((n+1)*g2+6)
    返回: 值域无界, 有效值<4 返回 NaN
    [对齐] pandas/scipy kurt(bias=False): Fisher 样本超额峰度 (正态分布=0)
    [分歧] 使用总体方差 ddof=0 计算 m2, 与 ddof=1 的 ts_std 不一致。
           标准做法应统一 ddof, 但当前公式与 m3/m4 分母保持一致 (均为 cnt)。
            对排名类因子无影响, 对阈值类用法有偏差 (审核报告 P2-3)。
    [修复] 2026-07-06 统一为 Fisher 样本超额峰度修正"""
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
            # 总体方差 ddof=0 (m2 需与 m4 分母一致)
            m2 = (sq - cnt * mean * mean) / cnt
            if m2 < 0.0:
                m2 = 0.0
            std = np.sqrt(m2)
            if std > 1e-10:
                m4 = 0.0
                for j in range(i - d + 1, i + 1):
                    if not np.isnan(x[j]):
                        z = (x[j] - mean) / std
                        m4 += z * z * z * z
                # 总体超额峰度 → Fisher 样本超额峰度修正
                g2 = m4 / cnt - 3.0
                r[i] = ((cnt + 1) * g2 + 6) * (cnt - 1) / ((cnt - 2) * (cnt - 3))
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
    """线性衰减加权均值: 近期数据权重更高。

    算法: 收集窗口内有效值 → 按出现顺序分配权重 [1,2,...,cnt] → 加权平均
    返回: 加权均值, 全 NaN 窗口返回 NaN
    [对齐] WQ101 decay_linear(x,d): 线性时间加权, 近期权重高
    [分歧] ft2 权重按有效值出现顺序分配; WQ101/DolphinDB 按绝对时间位置分配。
            窗口含 NaN 时两者结果不同 (例如 [NaN, a, b, NaN, c] 的权重分配不同)。
            当前实现更简洁, 且对干净数据无差异, 故保留。
    [重构] 2026-07-06 numba @njit 加速"""
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
    """滚动分位数值: 返回窗口内 p 分位数的值 (非排名)。

    算法: 收集有效值 → 排序 → nearest-rank 取 ceil(p*cnt)-1 位 (0-indexed)
    返回: 值域 [min, max], 有效值<1 返回 NaN
    [注意] WQ101 原始公式集无 ts_quantile 算子 (分位数由 rank/percentile 排名实现)
           本函数采用 nearest-rank 方法, 与 np.percentile linear 插值不同
    [修复] 2026-07-06 用 ceil(p*cnt)-1 替代 floor(p*cnt), 并限制范围"""
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
    """滚动 Z-score: (x[i] - mean) / std, 样本标准差 ddof=1。

    算法: 计算窗口均值和样本标准差 → (x[i] - mean) / std
    返回: 值域无界, std≈0 或冷启动返回 NaN
    [对齐] WQ101 无直接对应, ft2 扩展用于标准化
    [重构] 2026-07-06 numba @njit 加速"""
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
def _ts_autocorr_core(x, lag, d):
    """滚动自相关系数: 序列 x 与其 lag 期延迟值在窗口 d 内的 Pearson 相关系数。

    算法: 收集窗口内 x[t] 和 x[t-lag] 都有效的配对 → 计算样本协方差和方差 (ddof=1) → 相关系数
    返回: 值域 [-1, 1], 有效配对<3 或 std≈0 返回 NaN
    [对齐] WQ101 无直接对应, ft2 扩展用于动量持续性检测
    [新增] 2026-07-10"""
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
            k = j - lag
            if k < 0:
                continue
            if not (np.isnan(x[j]) or np.isnan(x[k])):
                sx += x[j]
                sy += x[k]
                sxy += x[j] * x[k]
                sxx += x[j] * x[j]
                syy += x[k] * x[k]
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
            if std_x > 1e-10 and std_y > 1e-10:
                r[i] = cov_xy / (std_x * std_y)
    return r


@njit(cache=True)
def _ts_step_core(x, d):
    """窗口内符号持续性: 窗口内所有值是否都 >= 0 (从未翻转为负)。

    算法: 检查窗口 [i-d+1, i] 内所有非 NaN 值是否都 >= 0
    返回: 1.0 (全部 >= 0) 或 0.0 (出现 < 0), 冷启动/全 NaN 返回 NaN
    [对齐] WQ101 无直接对应, ft2 扩展用于趋势持续性检测
    [注意] 与 persist 不同: persist 是"连续满足条件的天数"(累计), ts_step 是"整个窗口内是否从未翻转"(二值)
    [新增] 2026-07-10"""
    n = len(x)
    r = np.full(n, np.nan)
    for i in range(d - 1, n):
        all_nonneg = True
        has_valid = False
        for j in range(i - d + 1, i + 1):
            if not np.isnan(x[j]):
                has_valid = True
                if x[j] < 0:
                    all_nonneg = False
                    break
        if has_valid:
            r[i] = 1.0 if all_nonneg else 0.0
    return r


@njit(cache=True)
def _ts_hump_core(x, d):
    """窗口内驼峰检测: 是否中间高两端低 (先涨后跌形态)。

    算法: 检查窗口 [i-d+1, i] 内最大值位置是否在中间区域 [d*0.3, d*0.7],
          且最大值 > 窗口起始值 且 最大值 > 窗口结束值
    返回: 1.0 (驼峰形态) 或 0.0 (非驼峰), 冷启动/全 NaN 返回 NaN
    [对齐] WQ101 无直接对应, ft2 扩展用于形态检测
    [注意] 与 persist 不同: persist 检测同向持续性, ts_hump 检测先涨后跌的倒 U 形态
    [新增] 2026-07-10"""
    n = len(x)
    r = np.full(n, np.nan)
    lo = int(d * 0.3)
    hi = int(d * 0.7)
    for i in range(d - 1, n):
        window = x[i - d + 1: i + 1]
        # 找最大值及其位置
        max_val = 0.0
        max_idx = 0
        valid_cnt = 0
        for j in range(d):
            if not np.isnan(window[j]):
                valid_cnt += 1
                if valid_cnt == 1 or window[j] > max_val:
                    max_val = window[j]
                    max_idx = j
        if valid_cnt < 3:
            continue
        # 最大值位置在中间区域
        if max_idx < lo or max_idx > hi:
            r[i] = 0.0
            continue
        # 两端值
        left_val = window[0]
        right_val = window[d - 1]
        has_left = not np.isnan(left_val)
        has_right = not np.isnan(right_val)
        if not has_left:
            left_val = max_val
        if not has_right:
            right_val = max_val
        if max_val > left_val and max_val > right_val:
            r[i] = 1.0
        else:
            r[i] = 0.0
    return r


@njit(cache=True)
def _ts_regression_core(y, x, d, rettype):
    """双变量 y~x 滚动线性回归: y = a + b*x + eps。

    算法: 对有效 (x,y) 对做最小二乘 → 斜率 b, 截距 a
    返回: rettype 指定类型
        - 0: 斜率 b (x 对 y 的影响强度)
        - 1: 截距 a
        - 2: 残差 y[i] - (a+b*x[i])
        - 3: 预测值 a + b*x[i]
        - 4: R²
    返回: 有效值<3 或 var_x≈0 返回 NaN
    [对齐] WQ101 无直接对应算子, ft2 扩展用于量价回归等场景
    [重构] 2026-07-06 numba @njit 加速"""
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
    """单变量 x~t 趋势线性回归: x = a + b*t + eps。

    算法: 对时间 t=[0,1,...,cnt-1] 和有效值 x 做最小二乘 → 截距 a, 斜率 b
    返回: rettype 指定类型
        - 0: 斜率 b (趋势强度)
        - 1: 截距 a
        - 2: 残差 x[i] - (a+b*(window-1))
        - 3: 预测值 a + b*(window-1)
        - 4: R²
    冷启动/NaN: window<3 或当前值 NaN 返回 fill_value
    [分歧] 窗口含 NaN 时, 有效值被重新编号为 t=[0,1,...,cnt-1], 时间轴被压缩。
            WQ101 标准实现通常保留原始时间位置, 因此 NaN 会导致斜率失真。
            当前实现更简洁, 且对干净数据无差异, 故保留。 (审核报告 P2-2)
    [重构] 2026-07-06 numba @njit 加速
    [修复] 2026-07-06 window<3 尊重 fill_value, rettype=2/3 用 x[i] 和 window-1"""
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
                    # [修复] 2026-07-12 预测时间用 cnt-1 而非 window-1 (有NaN时 cnt<window, window-1 外推偏差)
                    predicted = a + b * (cnt - 1)
                    r[i] = x[i] - predicted
                elif rettype == 3:
                    # [修复] 2026-07-12 预测时间用 cnt-1 而非 window-1
                    r[i] = a + b * (cnt - 1)
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
    """2D 截面 Z-score: (x - mean) / std, 样本标准差 ddof=1。

    算法: 逐行计算均值和样本标准差 → 标准化
    返回: 值域无界, std≈0 返回 0.0, 冷启动返回 NaN
    [对齐] WQ101 无直接对应, ft2 扩展用于截面标准化
    [重构] 2026-07-06 numba @njit 加速 (2D)"""
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
    """2D 截面缩放: sum(|x|) = scale。

    算法: 逐行计算 sum(|x|) → x / sum(|x|) * scale
    返回: 值域无界, sum_abs≈0 返回 0.0
    [对齐] WQ101 scale(x, a=1): 截面缩放, sum(|x|) = a
    [重构] 2026-07-06 numba @njit 加速 (2D)"""
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
    """2D 截面缩尾 (Winsorize): 按 +/-std_n*std 截尾。

    算法: 逐行计算 mean 和 std → 超出 [mean-std_n*std, mean+std_n*std] 的值截尾
    返回: 值域 [mean-std_n*std, mean+std_n*std]
    [对齐] WQ101 无直接对应, ft2 扩展用于异常值处理
    [重构] 2026-07-06 numba @njit 加速 (2D)"""
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


@njit(cache=True)
def _cs_rank_core(panel):
    """2D 截面 min-rank (竞争排名): 每行独立排名, 并列值取最小排名, 归一化到 (0, 1]。

    算法: 每行收集有效值 → argsort 升序 → 同值组取最小排名 → 除以 cnt
    返回: 值域 (0, 1], 全 NaN 行保持 NaN
    [对齐] WQ101 rank(x): 截面排名, 最低得 0, 最高得 1
    [分歧] ft2 输出范围 (0, 1], 最低得 1/N 而非 0。这是为避免 0 值被下游算子
            (如乘法、除法) 静默忽略, 保留最小区分度。对相对排名无影响。
    [对齐] DolphinDB rank(): method='min' 竞争排名
    [重构] 2026-07-06 numba @njit 加速 (~28x), 替代 scipy rankdata"""
    n_days, n_stocks = panel.shape
    out = np.full((n_days, n_stocks), np.nan)
    for i in range(n_days):
        vals = np.empty(n_stocks)
        idxs = np.empty(n_stocks, dtype=np.int64)
        cnt = 0
        for j in range(n_stocks):
            if not np.isnan(panel[i, j]):
                vals[cnt] = panel[i, j]
                idxs[cnt] = j
                cnt += 1
        if cnt == 0:
            continue
        order = np.argsort(vals[:cnt])
        sorted_vals = vals[:cnt][order]
        min_rank = np.empty(cnt, dtype=np.float64)
        pos = 0
        while pos < cnt:
            g_start = pos
            g_val = sorted_vals[pos]
            while pos < cnt and sorted_vals[pos] == g_val:
                pos += 1
            rank = (g_start + 1) / cnt
            for t in range(g_start, pos):
                min_rank[order[t]] = rank
        for j in range(cnt):
            out[i, idxs[j]] = min_rank[j]
    return out


# ============================================================
# 时序函数 (ts_) — 窗口滚动, 只用历史数据
# ============================================================

# [修正] 2026-06-25 改用 nan* 版本, 对齐 WQ NaN=缺失数据应忽略的规范
# [重构] 2026-07-06 改用 numba @njit 加速, _rolling 作为 fallback 保留
def ts_mean(x, d):       return _ts_mean_core(np.asarray(x, float), _validate_window(d, 'ts_mean'))
def ts_std(x, d):        return _ts_std_core(np.asarray(x, float), _validate_window(d, 'ts_std'))
def ts_sum(x, d):
    """滚动窗口求和, 忽略 NaN"""
    return _ts_sum_core(np.asarray(x, float), _validate_window(d, 'ts_sum'))
def ts_max(x, d):        return _ts_max_core(np.asarray(x, float), _validate_window(d, 'ts_max'))
def ts_min(x, d):        return _ts_min_core(np.asarray(x, float), _validate_window(d, 'ts_min'))
def ts_median(x, d):     return _ts_median_core(np.asarray(x, float), _validate_window(d, 'ts_median'))

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
    """滚动协方差 (样本协方差, ddof=1)
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
    """滚动窗口最大值位置: 返回最大值距今天数。

    返回: 值域 [0, d-1], 0=当前日出现最大值, d-1=最远日出现最大值
    [对齐] WQ101 ts_argmax(x,d): 返回窗口内最大值位置。ft2 采用距今天数 (0=当前)，
            比 WQ101/DolphinDB 的 0=最旧约定更直观，使用时注意方向相反。
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_argmax_core(np.asarray(x, float), int(d))


def ts_argmin(x, d):
    """滚动窗口最小值位置: 返回最小值距今天数。

    返回: 值域 [0, d-1], 0=当前日出现最小值, d-1=最远日出现最小值
    [对齐] WQ101 ts_argmin(x,d): 返回窗口内最小值位置。ft2 采用距今天数 (0=当前)，
            比 WQ101/DolphinDB 的 0=最旧约定更直观，使用时注意方向相反。
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_argmin_core(np.asarray(x, float), int(d))

def ts_roc(x, d):
    """变化率 (Rate of Change): (x[t] - x[t-d]) / x[t-d]。

    算法: 当前值减 d 天前值, 除以前值 → 百分比变化
    返回: 值域无界, x[t-d]≈0 返回 NaN
    [对齐] WQ101 delta(x,d) 不含除法, ft2 ts_roc 是相对变化率
           若需绝对变化用 ts_delta(x, d)
    [修复] 2026-07-06 x[t-d] 接近 0 时返回 NaN (原硬编码除零行为)"""
    x = np.asarray(x, float); r = np.full_like(x, np.nan)
    r[d:] = (x[d:] - x[:-d]) / np.where(np.abs(x[:-d]) > 1e-10, x[:-d], np.nan)
    return r

def ts_zscore(x, d):
    """滚动 Z-score (样本标准差, ddof=1)
    [修复] 2026-07-06 std≈0 或冷启动期返回 NaN (原返回0违反冷启动保护, 产生虚假信号)
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_zscore_core(np.asarray(x, float), int(d))

def ts_autocorr(x, lag, d):
    """[新增] 2026-07-10 滚动自相关系数: x 与其 lag 期延迟值的 Pearson 相关系数"""
    return _ts_autocorr_core(np.asarray(x, float), int(lag), _validate_window(d, 'ts_autocorr'))

def ts_step(x, d):
    """[新增] 2026-07-10 窗口内符号持续性: 是否所有值都 >= 0 (从未翻转为负)"""
    return _ts_step_core(np.asarray(x, float), _validate_window(d, 'ts_step'))

def ts_hump(x, d):
    """[新增] 2026-07-10 窗口内驼峰检测: 是否中间高两端低 (先涨后跌形态)"""
    return _ts_hump_core(np.asarray(x, float), _validate_window(d, 'ts_hump'))

def ts_scale(x, d):
    """[修正] 2026-06-25 过滤 NaN, 对齐 WQ 标准
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_scale_core(np.asarray(x, float), int(d))

def ts_quantile(x, d, p=0.5):
    """滚动分位数值: 返回窗口内 p 分位数的值 (非排名)
    [重构] 2026-07-06 numba @njit 加速

    p=0.5 → 中位数, p=0.8 → 80%分位数, p=0.2 → 20%分位数
    """
    return _ts_quantile_core(np.asarray(x, float), _validate_window(d, 'ts_quantile'), float(p))

def ts_av_diff(x, d):
    """当前值减滚动均值 (偏离度)
    [重构] 2026-07-06 numba @njit 加速"""
    x = np.asarray(x, float)
    return x - _ts_mean_core(x, _validate_window(d, 'ts_av_diff'))

def ts_decay_linear(x, d):
    """[修正] 2026-06-25 过滤 NaN 后加权, 对齐 WQ 标准
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_decay_linear_core(np.asarray(x, float), _validate_window(d, 'ts_decay_linear'))

def ts_product(x, d):
    """d 期滚动乘积
    [重构] 2026-07-06 numba @njit 加速"""
    return _ts_product_core(np.asarray(x, float), _validate_window(d, 'ts_product'))

def regression(y, x, d, rettype=2):
    """滚动线性回归: y = alpha + beta*x + eps

    Args:
        y: 因变量
        x: 自变量
        d: 窗口
        rettype: 0=斜率, 1=截距, 2=残差, 3=预测值, 4=R²

    [重构] 2026-07-06 numba @njit 加速
    """
    return _ts_regression_core(np.asarray(y, float), np.asarray(x, float), int(d), int(rettype))


# [兼容] 2026-07-03 旧名别名（已停止生产使用，后续删除）
ts_regression_residual = lambda y, x, d: regression(y, x, d, 2)  # 默认 rettype=2 → 残差


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


def ts_ar_resid(x, order=5):
    """自回归残差: X(t) ~ c + φ₁X(t-1) + ... + φpX(t-p), 返回残差 ε(t)。

    与 ts_resid 的区别:
      ts_resid(x, w)    = X(t) - (a + b×t)          → 偏离"匀速直线运动"
      ts_ar_resid(x, p) = X(t) - AR(p) prediction   → 偏离"自身惯用模式"

    算法: 滚动窗口 OLS 拟合 AR(p), 窗口 d = 3×order+5 (保证足够自由度)。
    冷启动: 前 d+order-1 天返回 NaN。
    单参数设计: 仅 order 可调, 窗口自动推导, 降低 GP 搜索难度。

    [新增] 2026-07-12 自回归残差, 适用于成交额/波动率等有周期性的序列
    """
    x = np.asarray(x, float)
    p = int(order)
    d = 3 * p + 5  # 窗口自动推导, 保证 OLS 自由度充足
    n = len(x)
    r = np.full(n, np.nan)
    min_start = max(d + p - 1, p - 1)
    for i in range(min_start, n):
        # 构建 d 个观测: y[t] = X[t], 特征 = [1, X[t-1], ..., X[t-p]]
        X_mat = np.ones((d, p + 1))
        y_vec = np.empty(d)
        for j in range(d):
            t = i - d + 1 + j
            y_vec[j] = x[t]
            for lag in range(1, p + 1):
                X_mat[j, lag] = x[t - lag]
        try:
            beta = np.linalg.lstsq(X_mat, y_vec, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue
        pred = beta[0]
        for lag in range(1, p + 1):
            pred += beta[lag] * x[i - lag]
        r[i] = x[i] - pred
    return r


# ============================================================
# 扩张统计 (expanding_) — 起始→当前, 无固定窗口
# ============================================================

def expanding_mean(x, min_p=20):    return _expanding(x, np.mean, max(1, int(min_p)))
def expanding_median(x, min_p=20):  return _expanding(x, np.median, max(1, int(min_p)))
def expanding_std(x, min_p=20):     return _expanding(x, lambda a: np.std(a, ddof=1), max(1, int(min_p)))
def expanding_percentile(x, p, min_p=20):
    return _expanding(x, lambda a: np.percentile(a, p * 100), max(1, int(min_p)))


# ============================================================
# 截面函数 (cs_) — 每日跨品种统计, 需完整 2D 面板
# ============================================================

def cs_rank(x):
    """截面排名: 并列值取最小排名, 归一化到 (0, 1]。

    算法: 2D 面板调用 _cs_rank_core; 1D 输入返回全 0.5 (截面排名需至少 2 个值)
    返回: 值域 (0, 1], 全 NaN 行保持 NaN
    [对齐] WQ101 rank(x): 截面排名, 最低得 0, 最高得 1
    [分歧] ft2 输出范围 (0, 1], 最低得 1/N 而非 0。这是为避免 0 值被下游算子
           静默忽略, 保留最小区分度。对相对排名无影响。
    [对齐] DolphinDB rank(): method='min' 竞争排名
    [修复] 2026-07-06 numba @njit 加速 (~28x), 替代 scipy rankdata"""
    x = np.asarray(x, float)
    if x.ndim == 1: return np.full_like(x, 0.5)
    return _cs_rank_core(x)

def cs_zscore(x):
    """[重构] 2026-07-06 numba @njit 加速 (2D); 1D 保持原逻辑返回 zeros"""
    x = np.asarray(x, float)
    if x.ndim == 1: return np.zeros_like(x)
    return _cs_zscore_core(x)

def cs_scale(x, scale=1.0):
    """截面缩放: sum(abs(x)) = scale
    [重构] 2026-07-06 numba @njit 加速 (2D); 1D 保持原逻辑"""
    x = np.asarray(x, float)
    if x.ndim == 1: return x / (np.sum(np.abs(x)) + 1e-10) * scale
    return _cs_scale_core(x, float(scale))

def cs_winsorize(x, std=4.0):
    """截面缩尾: 按 +/-std 截尾 (样本标准差, ddof=1)
    [重构] 2026-07-06 numba @njit 加速 (2D); 1D 保持原逻辑整体截尾"""
    x = np.asarray(x, float)
    if x.ndim == 1:
        m, s = np.mean(x), np.std(x, ddof=1)
        return np.clip(x, m - std * s, m + std * s)
    return _cs_winsorize_core(x, float(std))

def cs_normalize(x, use_std=False):
    """截面归一化"""
    if use_std:
        return cs_scale(cs_zscore(x))
    return cs_scale(x, 1.0)

def cs_quantile(x, q=0.5, interpolation='linear'):
    """截面分位数: 返回每行第 q 分位数的值。

    对齐: DolphinDB quantile(X, q, interpolation), Skelf quantile(x, q)
    算法: 每行排序后使用 numpy.nanquantile 计算分位数, 支持 NaN 跳过。
    返回: 与 x 同形状的 Panel, 每行所有元素 = 该行的 q 分位数值。
    用途: 与 ts_quantile 对应, ts_quantile 是时序滚动分位数, cs_quantile 是截面分位数。

    Example:
        median = cs_quantile(returns, 0.5)    # 截面中位数
        q75 = cs_quantile(returns, 0.75)       # 截面 75% 分位数
        demeaned = returns - cs_quantile(returns, 0.5)  # 减去截面中位数

    [修复] 2026-07-12 对齐行业标准, 替代原 RINT 实现
    """
    x = np.asarray(x, float)
    if x.ndim == 1:
        return np.full_like(x, np.nan)
    n_rows, n_cols = x.shape
    out = np.full((n_rows, n_cols), np.nan)
    for i in range(n_rows):
        q_val = np.nanquantile(x[i], q, method=interpolation)
        out[i, :] = q_val
    return out


# ============================================================
# 数学函数 — 逐元素安全运算
# ============================================================

def safe_abs(x):          return np.abs(x)
def safe_log(x):
    """自然对数 ln(x), 仅对正数定义。

    算法: x > 1e-10 时返回 ln(x), 否则返回 NaN
    返回: 值域 (-inf, +inf), 非正数返回 NaN
    [注意] 阈值 1e-10 排除极小正数, 金融价格无影响, 但对标准化因子值可能丢失信息
           若需处理负数/零, 用 sign(x)*log(abs(x)+1) 显式表达
    [修复] 2026-07-06 非正数返回 NaN (原 abs(x) 导致 log(-1)==log(1) 掩盖负数错误)"""
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
    """RSI 中心化: (RSI-50)/50, 值域 [-1,1]. talib 对齐.
    [修复] 2026-07-06 保留冷启动期 NaN (原 nan_to_num 填充 50.0 掩盖缺失数据, 与冷启动保护原则冲突)"""
    c = np.asarray(close, float)
    result = talib.RSI(c, timeperiod=period)
    return (result - 50) / 50


def _feature_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR - Wilder 平滑. talib 对齐."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.ATR(h, l, c, timeperiod=period)


def _feature_atr_sma(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ATR-SMA: TR 的简单滚动均值 (V4 原始版本，保留兼容)
    [修复] 2026-07-06 用 shift 替代 roll, 避免首位引入末尾数据 (NaN 安全)"""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    prev_c = np.empty_like(c)
    prev_c[0] = c[0]
    prev_c[1:] = c[:-1]
    tr = np.maximum(h - l, np.maximum(np.abs(h - prev_c), np.abs(l - prev_c)))
    # [重构] 2026-07-06 改用 ts_mean (numba @njit 加速)
    return ts_mean(tr, period)


def _wilder_smooth(x: np.ndarray, period: int) -> np.ndarray:
    """Wilder 平滑: S[t]=(S[t-1]*(p-1)+x[t])/p, 初始值=SMA. talib 对齐."""
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
    """EMA: 递归计算, talib 对齐. k=2/(p+1), 初始值=SMA."""
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
    """布林带宽度 = (上轨-下轨)/中轨. talib 对齐.
    [修复] 2026-07-09 NaN 透传, 对齐 WQ 规范: talib 在停牌/冷启动返回 NaN, 不应被填 0
           伪装成"最窄布林带"; middle 为 NaN 或 <=0 时返回 NaN (NaN=缺失, 不参与截面排名)."""
    c = np.asarray(close, float)
    upper, middle, lower = talib.BBANDS(c, timeperiod=period, nbdevup=2, nbdevdn=2, matype=0)
    # [修复] 2026-07-09 middle NaN/<=0 → NaN (WQ: 缺失不排名); errstate 抑制 0 除警告
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(middle > 0, (upper - lower) / middle, np.nan)


def _feature_stddev(close: np.ndarray, period: int = 20) -> np.ndarray:
    """标准差. talib 对齐."""
    c = np.asarray(close, float)
    return talib.STDDEV(c, timeperiod=period, nbdev=1)


def _feature_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """ADX - 平均趋向指数. talib 对齐."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.ADX(h, l, c, timeperiod=period)


def _feature_cci(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """CCI - 商品通道指数. talib 对齐."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.CCI(h, l, c, timeperiod=period)


def _feature_macd(close: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> np.ndarray:
    """MACD 柱状图 = 2*(DIF-DEA). talib 对齐."""
    c = np.asarray(close, float)
    dif, dea, hist = talib.MACD(c, fastperiod=fast, slowperiod=slow, signalperiod=signal)
    return 2 * (dif - dea)


def _feature_trima(close: np.ndarray, period: int = 40) -> np.ndarray:
    """TRIMA - 三角移动平均. talib 对齐."""
    c = np.asarray(close, float)
    return talib.TRIMA(c, timeperiod=period)


def _feature_tsf(close: np.ndarray, period: int = 7) -> np.ndarray:
    """TSF - 时间序列预测. talib 对齐."""
    c = np.asarray(close, float)
    return talib.TSF(c, timeperiod=period)


def _feature_kama(close: np.ndarray, period: int = 30) -> np.ndarray:
    """KAMA - 卡夫曼自适应移动平均. talib 对齐."""
    c = np.asarray(close, float)
    return talib.KAMA(c, timeperiod=period)


def _feature_wma(close: np.ndarray, period: int = 20) -> np.ndarray:
    """WMA - 加权移动平均. talib 对齐."""
    c = np.asarray(close, float)
    return talib.WMA(c, timeperiod=period)


def _feature_dema(close: np.ndarray, period: int = 20) -> np.ndarray:
    """DEMA - 双指数移动平均. talib 对齐."""
    c = np.asarray(close, float)
    return talib.DEMA(c, timeperiod=period)


def _feature_hv(close: np.ndarray, period: int = 20) -> np.ndarray:
    """历史波动率: 日收益率年化标准差 x100.
    [修复] 2026-07-09 np.std→nanstd, 对齐 WQ 规范: 窗口内 NaN(停牌)应跳过, 用剩余有效值计算;
           与 ts_std(numba nan-aware core) 行为一致, 不再因单日停牌丢弃整段波动率."""
    c = np.asarray(close, float)
    rets = np.diff(c) / np.where(c[:-1] > 0, c[:-1], 1)
    rets = np.insert(rets, 0, 0)

    def _hv_std(a: np.ndarray) -> float:
        # [修复] 2026-07-09 仅取有效值, 全 NaN/有效值<=1 时返回 NaN (WQ: 窗口全缺失→缺失)
        valid = a[~np.isnan(a)]
        if valid.size <= 1:
            return np.nan
        return np.nanstd(valid, ddof=1) * np.sqrt(252) * 100

    return _rolling(rets, period, _hv_std)


def _feature_natr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int = 14) -> np.ndarray:
    """NATR - 归一化 ATR. talib 对齐."""
    h, l, c = np.asarray(high, float), np.asarray(low, float), np.asarray(close, float)
    return talib.NATR(h, l, c, timeperiod=period)


def _feature_var(close: np.ndarray, period: int = 20) -> np.ndarray:
    """方差. talib 对齐."""
    c = np.asarray(close, float)
    return talib.VAR(c, timeperiod=period)


def _feature_linearreg(close: np.ndarray, period: int = 20) -> np.ndarray:
    """线性回归值. talib 对齐."""
    c = np.asarray(close, float)
    return talib.LINEARREG(c, timeperiod=period)


def _feature_vol_ratio(close: np.ndarray, volume: np.ndarray, short: int = 5, long: int = 20) -> np.ndarray:
    """量比 = SMA(成交量,短期) / SMA(成交量,长期). 无 talib 等价.
    [重构] 2026-07-06 改用 ts_mean (numba @njit 加速)"""
    v = np.asarray(volume, float)
    # [重构] 2026-07-06 改用 ts_mean (numba @njit 加速)
    vs = ts_mean(v, short)
    vl = ts_mean(v, long)
    # [修复] 2026-07-09 NaN 透传, 对齐 WQ 规范: 长期均量 NaN(停牌)或 <=0 时返回 NaN,
    #        不填 0 伪装"地量"; ts_mean 已 nan-aware, 此处仅放行 NaN
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(vl > 0, vs / vl, np.nan)


def _feature_amt_ratio(amount: np.ndarray, short: int = 5, long: int = 20) -> np.ndarray:
    """额比 = 短期均值 / 长期均值
    [重构] 2026-07-06 改用 ts_mean (numba @njit 加速)"""
    a = np.asarray(amount, float)
    # [重构] 2026-07-06 改用 ts_mean (numba @njit 加速)
    ms = ts_mean(a, short)
    ml = ts_mean(a, long)
    # [修复] 2026-07-09 NaN 透传, 对齐 WQ 规范: 长期均值 NaN(停牌)或 <=0 时返回 NaN,
    #        不填 0 伪装"低额比"; 顺带消除 ts_mean(a,long) 重复调用
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(ml > 0, ms / ml, np.nan)


# ============================================================
# 函数注册表
# ============================================================

# [重构] 2026-07-07 引入 FunctionSpec: 把函数实现、GP 分类、数据参数个数、
# 参数候选池封装到一个 dataclass 中。GP 生成器可直接读取 data_args/param_pool
# 生成合法表达式，不再依赖 TS_FUNCTIONS/TS_FUNCTIONS_2ARG 等硬编码分池。
# [新增] 2026-07-08 ParamRange: 描述带值域约束的配置参数 (如浮点比例、
# 正整数窗口等)。param_pool 描述离散候选, param_ranges 描述连续范围约束,
# 两者互补。GP 生成器优先用 param_pool, 其次用 param_ranges 采样。
# [调整] 2026-07-08 ParamConstraint → ParamRange, data_arity → data_args,
# param_constraints → param_ranges

@dataclass
class VarSpec:
    """变量规格 — 描述变量名、所属类别和匹配模式。

    Args:
        name: 变量名 (ALL_CAPS, 如 'CLOSE', 'REL')
        category: 所属类别 (如 '原始OHLCV', '相对基准')
        is_prefix: True=前缀通配 (REL_CLOSE 通过), False=精确匹配 (只认 CLOSE)
        description: 描述信息 (仅文档/调试)
    """
    name: str
    category: str = '自定义'
    is_prefix: bool = False
    description: str = ''


@dataclass
class ParamRange:
    """参数值域约束 — 描述 param_pool 无法覆盖的带范围参数。

    用于函数签名中 param_pool 之外的额外参数 (如 ts_quantile 的 p, cs_scale 的 scale)。

    Args:
        name: 参数名 (仅用于文档/调试, 不参与生成逻辑)
        dtype: 'int' 或 'float', 默认 'float'
        min_val: 最小值 (含), None 表示无下界
        max_val: 最大值 (含), None 表示无上界
        pool: 离散候选列表。非 None 时优先于 min/max 采样
    """
    name: str
    dtype: str = 'float'
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    pool: Optional[List[Any]] = None


@dataclass
class FunctionSpec:
    """函数注册项：实现 + 表达式/GP 元数据。

    Args:
        func: 函数实现，签名为 (*np.ndarray) -> np.ndarray。
        category: GP 大类，对齐 group_weights 的 key。
        data_args: 需要由 GP 生成子树填充的数据序列参数个数。
            例如 ts_mean(x, d) 为 1，ts_corr(x, y, d) 为 2。
            data_vars 非空时自动取 len(data_vars)，无需手动指定。
        param_pool: 第一组配置参数候选列表。元素为标量时生成一个常数参数；
            为 tuple 时生成多个常数参数；为 None 时不附加配置参数。
        param_ranges: 额外配置参数值域约束列表 (带类型/范围)。
            用于 param_pool 无法覆盖的参数 (如浮点比例、连续范围)。
            GP 生成器在 param_pool 之后按顺序追加这些参数。
        data_vars: 固定变量名列表。非 None 时 GP 直接填入这些变量名，
            而非生成随机子树。用于指标计算类函数 (如 natr 固定需要 H/L/C)。
            设置时 data_args 自动取 len(data_vars)。
    """
    func: Callable
    category: str
    data_args: int = 1
    param_pool: Optional[List[Any]] = None
    param_ranges: Optional[List[ParamRange]] = None
    data_vars: Optional[List[str]] = None

    def __call__(self, *args, **kwargs):
        # [兼容] 保持 FUNC_REGISTRY[name](...) 可直接调用
        return self.func(*args, **kwargs)

    def __getattr__(self, name):
        # [兼容] 旧代码可能访问 FUNC_REGISTRY[name].__name__ / __doc__ 等属性
        return getattr(self.func, name)


# ============================================================
# 函数分类索引 (供 LLM 理解可用原语)
# ============================================================

# [重构] 2026-07-07 FUNC_CATEGORIES 不再硬编码，由 _register 自动维护
# 人只填一次 _register(..., category=...)，两个 dict 自动同步
FUNC_CATEGORIES: Dict[str, List[str]] = {}

# 有效函数分类 — GP group_weights 的 key，超出此范围的 category 会被拒绝
VALID_FUNC_CATEGORIES = frozenset([
    'ts_function', 'cs_function', 'math_function',
    'ta_function', 'feature_function',
])


def get_func_category(name: str) -> str:
    """查询函数所属 GP 大类（直接对齐 group_weights key）"""
    spec = FUNC_REGISTRY.get(name.lower())
    # [重构] 2026-07-07 FUNC_CATEGORIES 由 _register 自动维护，与 FUNC_REGISTRY 完全同步
    # 无需兜底遍历，直接返回 spec.category
    return spec.category if spec is not None else 'math_function'


def _register(name: str, func: Callable, category: str,
              data_args: Optional[int] = None,
              param_pool: Optional[List[Any]] = None,
              param_ranges: Optional[List[ParamRange]] = None,
              data_vars: Optional[List[str]] = None) -> None:
    """统一注册内置/自定义函数，同时维护 FUNC_REGISTRY 与 FUNC_CATEGORIES。

    [规范] 2026-07-08 data_vars 非空时, data_args 自动取 len(data_vars),
    无需手动指定。消除 data_args 与 data_vars 的冗余。
    [调整] 2026-07-08 data_arity → data_args, param_constraints → param_ranges
    """
    name_lower = name.lower()
    if category not in VALID_FUNC_CATEGORIES:
        raise ValueError(
            f"无效函数分类 '{category}' (函数 '{name_lower}')。"
            f"有效分类: {sorted(VALID_FUNC_CATEGORIES)}"
        )
    # data_vars 优先: 有固定变量时, data_args 自动推导
    if data_vars is not None:
        data_args = len(data_vars)
    elif data_args is None:
        data_args = 1
    FUNC_REGISTRY[name_lower] = FunctionSpec(
        func=func, category=category,
        data_args=data_args, param_pool=param_pool,
        param_ranges=param_ranges,
        data_vars=data_vars,
    )
    if category not in FUNC_CATEGORIES:
        FUNC_CATEGORIES[category] = []
    if name_lower not in FUNC_CATEGORIES[category]:
        FUNC_CATEGORIES[category].append(name_lower)


FUNC_REGISTRY: Dict[str, FunctionSpec] = {}

# [重构] 2026-07-07 内置函数统一通过 _register 注册，param_pool 直接内联

# ── 时序 (ts_) ──
_register('ts_mean', ts_mean, 'ts_function', param_pool=[5, 10, 20, 60])
_register('ts_std', ts_std, 'ts_function', param_pool=[10, 20, 60])
_register('ts_sum', ts_sum, 'ts_function', param_pool=[5, 10, 20])
_register('ts_max', ts_max, 'ts_function', param_pool=[10, 20])
_register('ts_min', ts_min, 'ts_function', param_pool=[10, 20])
_register('ts_median', ts_median, 'ts_function', param_pool=[10, 20])
_register('ts_delta', ts_delta, 'ts_function', param_pool=[1, 5, 10, 20])
_register('ts_delay', ts_delay, 'ts_function', param_pool=[1, 5, 10, 20])
_register('ts_rank', ts_rank, 'ts_function', param_pool=[5, 10, 20, 60])
_register('ts_corr', ts_corr, 'ts_function', data_args=2, param_pool=[10, 20, 60])
_register('ts_skew', ts_skew, 'ts_function', param_pool=[20, 60])
_register('ts_kurt', ts_kurt, 'ts_function', param_pool=[20, 60])
_register('ts_argmax', ts_argmax, 'ts_function', param_pool=[10, 20])
_register('ts_argmin', ts_argmin, 'ts_function', param_pool=[10, 20])
_register('ts_roc', ts_roc, 'ts_function', param_pool=[5, 10, 20])
_register('ts_cov', ts_cov, 'ts_function', data_args=2, param_pool=[5, 10, 20])
_register('ts_var', lambda x, d: ts_std(x, d) ** 2, 'ts_function', param_pool=[10, 20])
_register('logret', lambda x: safe_log(x / ts_delay(x, 1)), 'math_function')
_register('ts_zscore', ts_zscore, 'ts_function', param_pool=[10, 20, 60])
_register('ts_autocorr', ts_autocorr, 'ts_function', param_pool=[(1, 10), (5, 20), (10, 60)])
_register('ts_step', ts_step, 'ts_function', param_pool=[5, 10, 20])
_register('ts_hump', ts_hump, 'ts_function', param_pool=[10, 20])
_register('ts_scale', ts_scale, 'ts_function', param_pool=[10, 20])
_register('ts_quantile', ts_quantile, 'ts_function', param_pool=[5, 10, 20],
          param_ranges=[ParamRange('p', 'float', 0.0, 1.0)])
_register('ts_av_diff', ts_av_diff, 'ts_function', param_pool=[10, 20])
_register('ts_decay_linear', ts_decay_linear, 'ts_function', param_pool=[5, 10, 20])
_register('ts_product', ts_product, 'ts_function', param_pool=[10, 20])
# ── 双变量回归 reg_ (y~x, 3 参数: y, x, d) ──
_register('reg_slope', lambda y, x, d: regression(y, x, d, 0), 'ts_function', data_args=2, param_pool=[5, 10])
_register('reg_intercept', lambda y, x, d: regression(y, x, d, 1), 'ts_function', data_args=2, param_pool=[5, 10])
_register('reg_resid', lambda y, x, d: regression(y, x, d, 2), 'ts_function', data_args=2, param_pool=[5, 10])
_register('reg_predict', lambda y, x, d: regression(y, x, d, 3), 'ts_function', data_args=2, param_pool=[5, 10])
_register('reg_rsq', lambda y, x, d: regression(y, x, d, 4), 'ts_function', data_args=2, param_pool=[5, 10])
_register('ts_slope', ts_slope, 'ts_function', param_pool=[10, 20])
_register('ts_intercept', ts_intercept, 'ts_function', param_pool=[10, 20])
_register('ts_resid', ts_resid, 'ts_function', param_pool=[10, 20])
_register('ts_predict', ts_predict, 'ts_function', param_pool=[10, 20])
_register('ts_rsq', ts_rsq, 'ts_function', param_pool=[10, 20])
_register('ts_ar_resid', ts_ar_resid, 'ts_function', param_pool=[3, 5, 10])

# ── 扩张统计 (expanding_) ──
_register('expanding_mean', expanding_mean, 'ts_function', param_pool=[20, 60])
_register('expanding_median', expanding_median, 'ts_function', param_pool=[20, 60])
_register('expanding_std', expanding_std, 'ts_function', param_pool=[20, 60])
_register('expanding_percentile', expanding_percentile, 'ts_function', param_pool=[(0.1, 20), (0.5, 20), (0.9, 20)])

# ── 截面 (cs_) ──
_register('cs_rank', cs_rank, 'cs_function')
_register('cs_zscore', cs_zscore, 'cs_function')
_register('cs_scale', cs_scale, 'cs_function',
          param_ranges=[ParamRange('scale', 'float', 0.5, 2.0)])
_register('cs_winsorize', cs_winsorize, 'cs_function',
          param_ranges=[ParamRange('std', 'float', 2.0, 5.0)])
_register('cs_quantile', cs_quantile, 'cs_function')
# [修复] 2026-07-07 补上 cs_normalize 注册（旧硬编码列表中有但从未注册）
_register('cs_normalize', cs_normalize, 'cs_function')

# ── 数学 ──
_register('abs', safe_abs, 'math_function')
_register('log', safe_log, 'math_function')
_register('sqrt', safe_sqrt, 'math_function')
_register('sign', safe_sign, 'math_function')
_register('exp', safe_exp, 'math_function')
_register('tanh', safe_tanh, 'math_function')
_register('sigmoid', safe_sigmoid, 'math_function')
_register('relu', safe_relu, 'math_function')
_register('softsign', safe_softsign, 'math_function')
_register('sin', lambda x: np.sin(x), 'math_function')
_register('cos', lambda x: np.cos(x), 'math_function')
_register('gauss', safe_gauss, 'math_function')
_register('p4', safe_p4, 'math_function')
_register('neg', safe_neg, 'math_function')
_register('square_sigmoid', safe_square_sigmoid, 'math_function')
_register('signed_power', signed_power, 'math_function',
          param_ranges=[ParamRange('exponent', 'float', 0.5, 4.0)])
_register('max', safe_max, 'math_function', data_args=2)
_register('min', safe_min, 'math_function', data_args=2)

# ── 信号 ──
_register('persist', persist, 'ts_function', param_pool=[3, 5, 10])

# ── 特征计算 (从 OHLCV 实时算, 无需 FeatureSpace) ──
_register('rsi', _feature_rsi, 'ta_function', param_pool=[14, 20])
_register('atr', _feature_atr, 'ta_function', data_args=3, param_pool=[14],
           data_vars=['HIGH', 'LOW', 'CLOSE'])
_register('atr_sma', _feature_atr_sma, 'ta_function', data_args=3, param_pool=[14],
           data_vars=['HIGH', 'LOW', 'CLOSE'])
_register('bb_width', _feature_bbwidth, 'ta_function', param_pool=[20])
_register('stddev', _feature_stddev, 'ta_function', param_pool=[10, 20])
_register('adx', _feature_adx, 'ta_function', data_args=3, param_pool=[14, 20],
           data_vars=['HIGH', 'LOW', 'CLOSE'])
_register('cci', _feature_cci, 'ta_function', data_args=3, param_pool=[14, 20],
           data_vars=['HIGH', 'LOW', 'CLOSE'])
_register('macd', _feature_macd, 'ta_function', param_pool=[(12, 26, 9)])
_register('trima', _feature_trima, 'ta_function', param_pool=[40])
_register('ema', _feature_ema, 'ta_function', param_pool=[5, 10, 20, 60])
_register('wilder_smooth', _wilder_smooth, 'feature_function', param_pool=[10, 20])
_register('tsf', _feature_tsf, 'ta_function', param_pool=[10, 20])
_register('kama', _feature_kama, 'ta_function', param_pool=[30])
_register('wma', _feature_wma, 'ta_function', param_pool=[5, 10, 20, 60])
_register('dema', _feature_dema, 'ta_function', param_pool=[10, 20])
_register('hv', _feature_hv, 'ta_function', param_pool=[20, 60])
_register('natr', _feature_natr, 'ta_function', data_args=3, param_pool=[5, 14],
           data_vars=['HIGH', 'LOW', 'CLOSE'])
_register('var', _feature_var, 'ta_function', param_pool=[10, 20])
_register('linearreg', _feature_linearreg, 'ta_function', param_pool=[10, 20])
# [修复] 2026-07-08 vol_ratio 签名 (close, volume, short, long), data_vars 需含 CLOSE
_register('vol_ratio', _feature_vol_ratio, 'ta_function', param_pool=[(5, 20)],
           data_vars=['CLOSE', 'VOLUME'])
_register('amt_ratio', _feature_amt_ratio, 'ta_function', param_pool=[(5, 20)],
           data_vars=['AMOUNT'])

# ── 旧名别名 — [2026-07-03] 已停止生产使用, 后续删除 ──
_register('ts_resi', ts_resid, 'ts_function', param_pool=[10, 20])
_register('ts_regression_residual', lambda y, x, d: regression(y, x, d, 2), 'ts_function', data_args=2, param_pool=[5, 10])
_register('ts_rsquare', ts_rsq, 'ts_function', param_pool=[10, 20])
_register('ts_logret', lambda x: safe_log(x / ts_delay(x, 1)), 'math_function')  # [废弃] 改用 logret

# 安全常量 (表达式中的 True/False/None/pi/e)
SAFE_CONSTANTS = {'True': 1.0, 'False': 0.0, 'None': 0.0, 'pi': np.pi, 'e': np.e}


# ============================================================
# 变量层 — 合并自 variables.py
# [重构] 2026-07-08 将 variables.py 合并到 functions.py,
# 统一管理"能引用什么"(变量)和"能算什么"(函数)两类元数据。
# [新增] 2026-07-08 VarSpec + _VAR_REGISTRY, 替代双表分离模式。
# ============================================================

# ── 单一事实来源: _VAR_REGISTRY ──
# is_prefix=True  → 前缀通配 (REL_CLOSE 通过, 后缀任意 ASCII 字母数字)
# is_prefix=False → 精确匹配 (只认变量名本身, 拒绝后缀)
_VAR_REGISTRY: Dict[str, VarSpec] = {
    # ── 第1组: 原始 OHLCV ──
    'OPEN':    VarSpec('OPEN', '原始OHLCV', description='开盘价'),
    'HIGH':    VarSpec('HIGH', '原始OHLCV', description='最高价'),
    'LOW':     VarSpec('LOW', '原始OHLCV', description='最低价'),
    'CLOSE':   VarSpec('CLOSE', '原始OHLCV', description='收盘价'),
    'VOLUME':  VarSpec('VOLUME', '原始OHLCV', description='成交量'),
    'AMOUNT':  VarSpec('AMOUNT', '原始OHLCV', description='成交额'),
    'VWAP':    VarSpec('VWAP', '原始OHLCV', description='均价 (WQ065)'),
    'RETURNS': VarSpec('RETURNS', '原始OHLCV', description='收益率'),

    # ── 第2组: 派生变量 (前缀通配) ──
    'REL':   VarSpec('REL', '相对基准', is_prefix=True, description='REL_CLOSE/REL_AMOUNT/REL_VOLUME'),
    'BENCH': VarSpec('BENCH', '相对基准', is_prefix=True, description='BENCH_CLOSE/BENCH_RETURNS'),
    'SHARE': VarSpec('SHARE', '市场份额', description='跨品种成交额占比'),
    'DOWNSIDE_VOL': VarSpec('DOWNSIDE_VOL', '下行风险', description='下行标准差'),

    # ── 第3组: 基本面 ──
    'PE_TTM_INDEX':  VarSpec('PE_TTM_INDEX', '基本面', description='滚动市盈率'),
    'PB_MRQ':        VarSpec('PB_MRQ', '基本面', description='市净率'),
    'TURNOVERRATIO': VarSpec('TURNOVERRATIO', '基本面', description='换手率'),
    'TOTALCAPITAL':  VarSpec('TOTALCAPITAL', '基本面', description='总市值'),

    # ── 第4组: talib 指标 (变量通道, 为性能预留) ──
    'ATR':    VarSpec('ATR', '波动率'),
    'STDDEV': VarSpec('STDDEV', '波动率'),
    'HV':     VarSpec('HV', '波动率'),
    'NATR':   VarSpec('NATR', '波动率'),
    'BBWIDTH': VarSpec('BBWIDTH', '通道指标'),
    'TRIMA': VarSpec('TRIMA', '趋势指标'), 'SMA': VarSpec('SMA', '趋势指标'),
    'MA':    VarSpec('MA', '趋势指标'), 'EMA': VarSpec('EMA', '趋势指标'),
    'TSF':   VarSpec('TSF', '趋势指标'), 'WMA': VarSpec('WMA', '趋势指标'),
    'DEMA':  VarSpec('DEMA', '趋势指标'), 'KAMA': VarSpec('KAMA', '趋势指标'),
    'ADX':   VarSpec('ADX', '趋势指标'),
    'RSI':   VarSpec('RSI', '动量指标'), 'CCI': VarSpec('CCI', '动量指标'),
    'MACD':  VarSpec('MACD', '动量指标'), 'MFI': VarSpec('MFI', '动量指标'),
    'ULTOSC': VarSpec('ULTOSC', '动量指标'), 'ROC': VarSpec('ROC', '动量指标'),
    'LINEARREG': VarSpec('LINEARREG', '统计指标'),
    'VAR':   VarSpec('VAR', '统计指标'), 'CORREL': VarSpec('CORREL', '统计指标'),
    'VOL_RATIO': VarSpec('VOL_RATIO', '量价指标'), 'VOL_CHG': VarSpec('VOL_CHG', '量价指标'),
    'OBV':   VarSpec('OBV', '量价指标'), 'UP_RATIO': VarSpec('UP_RATIO', '量价指标'),
    'AVGPRICE': VarSpec('AVGPRICE', '价格水平'), 'WCLPRICE': VarSpec('WCLPRICE', '价格水平'),

    # ── 第5组: 行业广度 / 市场级变量 ──
    'SECTOR_UP': VarSpec('SECTOR_UP', '市场宽度'), 'SECTOR_MOM': VarSpec('SECTOR_MOM', '市场宽度'),
    'SECTOR_AD': VarSpec('SECTOR_AD', '市场宽度'),
    'BREADTH_S': VarSpec('BREADTH_S', '市场宽度'), 'BREADTH_M': VarSpec('BREADTH_M', '市场宽度'),
    'BREADTH_L': VarSpec('BREADTH_L', '市场宽度'), 'BREADTH_AMT': VarSpec('BREADTH_AMT', '市场宽度'),
    'DISP':   VarSpec('DISP', '市场结构'), 'ROTSPD': VarSpec('ROTSPD', '市场结构'),
    'NHL':    VarSpec('NHL', '市场结构'), 'SKEW': VarSpec('SKEW', '市场结构'),
    'IND_CORR': VarSpec('IND_CORR', '市场结构'),
    'VMED':   VarSpec('VMED', '资金结构'), 'VDISP': VarSpec('VDISP', '资金结构'),
    'VSKEW':  VarSpec('VSKEW', '资金结构'),
    'TAILUP': VarSpec('TAILUP', '尾部风险'), 'TAILDOWN': VarSpec('TAILDOWN', '尾部风险'),
    'TAILNET': VarSpec('TAILNET', '尾部风险'),
}

# ── 向后兼容导出 (自动从 _VAR_REGISTRY 推导) ──
VALID_VAR_PREFIXES: List[str] = list(_VAR_REGISTRY.keys())
VAR_CATEGORIES: Dict[str, list] = {}
for spec in _VAR_REGISTRY.values():
    VAR_CATEGORIES.setdefault(spec.category, []).append(spec.name)


def get_var_category(name: str) -> str:
    """查询变量所属分类"""
    upper = name.upper()
    spec = _VAR_REGISTRY.get(upper)
    if spec is not None:
        return spec.category
    # 前缀匹配: 查找 is_prefix=True 的注册项
    for spec in _VAR_REGISTRY.values():
        if spec.is_prefix and upper.startswith(spec.name + '_'):
            return spec.category
    return '自定义'


def is_valid_variable(name: str) -> bool:
    """检查变量名是否合法（匹配已注册变量）

    规则:
      1. 精确匹配 _VAR_REGISTRY 中的精确变量 (如 CLOSE, SECTOR_UP)
      2. 前缀通配: is_prefix=True 的条目 + '_' + 后缀 (如 REL_CLOSE)
         后缀只能含 ASCII 字母数字和下划线
    """
    upper = name.upper()
    spec = _VAR_REGISTRY.get(upper)
    if spec is not None:
        return True
    # 前缀通配匹配: 只查 is_prefix=True 的条目
    for spec in _VAR_REGISTRY.values():
        if spec.is_prefix and upper.startswith(spec.name + '_'):
            rest = upper[len(spec.name) + 1:]
            if rest and all(c.isascii() and (c.isalnum() or c == '_') for c in rest):
                return True
    return False


def register_variable(name: str, category: str = '自定义',
                      is_prefix: bool = False,
                      description: str = '') -> None:
    """临时注册自定义变量到表达式引擎。

    添加 VarSpec 到 _VAR_REGISTRY，同时同步到 VALID_VAR_PREFIXES。

    Args:
        name: 变量名 (ALL_CAPS)
        category: 所属类别, 默认 '自定义'
        is_prefix: True=前缀通配, False=精确匹配
        description: 描述信息

    进程级全局注册, 当前脚本内所有 Expression 生效。
    脚本退出自动销毁, 无泄漏风险。
    """
    upper = name.upper()
    if upper not in _VAR_REGISTRY:
        spec = VarSpec(upper, category=category, is_prefix=is_prefix, description=description)
        _VAR_REGISTRY[upper] = spec
        _sync_var_backward_compat()


def unregister_variable(name: str) -> bool:
    """注销自定义变量，返回是否成功。"""
    upper = name.upper()
    if upper in _VAR_REGISTRY:
        del _VAR_REGISTRY[upper]
        _sync_var_backward_compat()
        return True
    return False


def _sync_var_backward_compat():
    """同步 VALID_VAR_PREFIXES 和 VAR_CATEGORIES 与 _VAR_REGISTRY 一致"""
    global VALID_VAR_PREFIXES, VAR_CATEGORIES
    VALID_VAR_PREFIXES = list(_VAR_REGISTRY.keys())
    cats = {}
    for spec in _VAR_REGISTRY.values():
        cats.setdefault(spec.category, []).append(spec.name)
    VAR_CATEGORIES = cats


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

def register_function(
    name: str,
    func: Callable,
    category: str = 'math_function',
    data_args: Optional[int] = None,
    param_pool: Optional[List[Any]] = None,
    param_ranges: Optional[List[ParamRange]] = None,
    data_vars: Optional[List[str]] = None,
) -> None:
    """临时注册自定义函数到表达式引擎，并加入 GP 权重池。

    进程级全局注册，当前脚本内所有 Expression 生效。
    脚本退出自动销毁，无泄漏风险。

    Args:
        name: 函数名（表达式中的调用名）
        func: 函数实现，签名为 (*np.ndarray) -> np.ndarray
        category: GP 大类，直接对齐 group_weights 的函数类 key。
                  默认 'math_function'。有效分类见 VALID_FUNC_CATEGORIES。
        data_args: 数据序列参数个数。GP 生成器据此生成正确数量的子树。
                  例如 ts_mean 为 1，ts_corr 为 2，natr 为 3。
                  默认 1。
        param_pool: 配置参数候选列表。元素为标量时生成一个常数参数；
                  为 tuple 时生成多个常数参数；为 None 时不附加配置参数。
        param_ranges: 额外配置参数值域约束列表 (带类型/范围)。
                  用于 param_pool 无法覆盖的参数 (如浮点比例、连续范围)。
        data_vars: 固定变量名列表。非 None 时 GP 直接填入这些变量名，
                  而非生成随机子树。例如 natr 的 data_vars=['HIGH','LOW','CLOSE']。
    """
    name_lower = name.lower()
    if name_lower in FUNC_REGISTRY:
        import warnings
        warnings.warn(
            f"register_function: '{name}' 已存在，将被覆盖。"
            f"原函数: {FUNC_REGISTRY[name_lower].__name__}"
        )
    # [重构] 2026-07-07 统一走 _register，保证 FUNC_REGISTRY 与 FUNC_CATEGORIES 同步
    _register(name_lower, func, category, data_args=data_args,
              param_pool=param_pool, param_ranges=param_ranges,
              data_vars=data_vars)


def unregister_function(name: str) -> bool:
    """注销自定义函数，返回是否成功。内置函数不可注销。"""
    name_lower = name.lower()
    removed = FUNC_REGISTRY.pop(name_lower, None) is not None
    if removed:
        for cat_names in FUNC_CATEGORIES.values():
            if name_lower in cat_names:
                cat_names.remove(name_lower)
                break
    return removed
