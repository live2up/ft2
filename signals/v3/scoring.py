"""
signals/v3/scoring.py — 连续值打分择时框架
=============================================================================
v3 独立版本。将多个连续信号加权合成为复合分数，通过三区状态机生成持仓决策。

[修复] 2026-06-10 zscore/rank 改为 expanding 统计量，消除全序列前向偏差:
  - zscore: expanding mean/std (t时刻只用[0:t]的数据)
  - rank: expanding percentile rank (t时刻只统计历史)
  旧实现用 np.nanmean/np.nanstd 全序列 → score[i] 含未来信息 → 虚高 Sharpe

[修复] 2026-06-20 three_zone_backtest 改走 EngineV3.backtest(mode='fast'):
  - 三区状态机只生成 position 信号 (1.0=持仓, 0.0=空仓)
  - 撮合/费率/指标计算全部由 ft2.core Engine 驱动
  - 与 EngineV3.backtest 结果完全一致，无自研回测偏差

=============================================================================
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Callable


# ============================================================
# Expanding 统计工具 (防前向偏差)
# ============================================================

def _expanding_zscore(values: np.ndarray, warmup: int = 20) -> np.ndarray:
    """expanding z-score: 每时刻只用历史数据计算 mean/std

    zscore[i] = (values[i] - mean[0:i+1]) / std[0:i+1], 截断到 [-3, 3]

    Args:
        warmup: 冷启动 bar 数，前 warmup 条统计量不稳定，返回 0（中性）
    """
    n = len(values)
    if warmup >= n:
        return np.zeros(n)           # 序列太短，全中性

    result = np.full(n, np.nan)
    # [兼容] np.nancumsum 仅 NumPy≥2.0 可用，手动实现
    v_clean = np.where(np.isnan(values), 0.0, values)
    v2_clean = np.where(np.isnan(values), 0.0, values ** 2)
    cumsum = np.cumsum(v_clean)
    cumsum2 = np.cumsum(v2_clean)
    count = np.arange(1, n + 1, dtype=float)

    for i in range(warmup, n):
        mean = cumsum[i] / count[i]
        var = cumsum2[i] / count[i] - mean ** 2
        std = np.sqrt(max(var, 1e-10))
        result[i] = np.clip((values[i] - mean) / std, -3.0, 3.0)

    result[:warmup] = 0.0
    return np.nan_to_num(result, nan=0.0)


def _expanding_rank(values: np.ndarray, warmup: int = 20) -> np.ndarray:
    """expanding percentile rank: t时刻在历史中的百分位

    rank[i] = (#values[0:i+1] <= values[i]) / (i+1), 值域 [0, 1]

    Args:
        warmup: 冷启动 bar 数，前 warmup 条排名不稳定，返回 0.5（中位）
    """
    n = len(values)
    if warmup >= n:
        return np.full(n, 0.5)       # 序列太短，全中位

    result = np.full(n, np.nan)

    for i in range(warmup, n):
        window = values[:i + 1]
        result[i] = np.sum(window <= values[i]) / (i + 1)

    result[:warmup] = 0.5
    return np.nan_to_num(result, nan=0.5)


# ============================================================
# 信号定义
# ============================================================
class ScoredSignal:
    """单个评分信号的定义

    Parameters
    ----------
    name : str
        特征列名（需存在于 feature_df 中）
    weight : float
        权重（可正可负，负权重=反向信号）
    transform : str or callable
        变换方式:
        - 'raw'            : 原始值（需已归一化到 -1~1）
        - 'rank'           : expanding 百分位排名 0~1（无前向偏差）
        - 'zscore'         : expanding Z-score [-3, 3]（无前向偏差）
        - 'rank_full'      : 全序列排名 0~1（⚠ 含未来信息，仅用于对比实验）
        - 'zscore_full'    : 全序列 Z-score（⚠ 含未来信息，仅用于对比实验）
        - 'binary'         : thr_0 二值化 {0, 1}
        - 'neg_log'        : 负对数变换（用于波动率等正偏态分布）
        - callable         : 自定义变换函数 fn(values) → array
    """

    def __init__(self, name: str, weight: float = 1.0,
                 transform: str = 'raw', warmup: int = 20):
        """
        Args:
            name: 特征列名
            weight: 权重（正=正向，负=反向）
            transform: 变换方式 (raw/rank/zscore/binary/neg_log/rank_full/zscore_full)
            warmup: expanding 变换的冷启动 bar 数（仅 rank/zscore 有效）
                    前 warmup 条 → zscore=0(中性), rank=0.5(中位)
                    默认 20 ≈ 一个月交易日。可网格搜索调优
        """
        self.name = name
        self.weight = weight
        self.transform = transform
        self.warmup = warmup

    def apply(self, series: pd.Series) -> np.ndarray:
        values = series.values.copy()
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        if self.transform == 'raw':
            return values

        # [修复] expanding rank — 无前向偏差（默认）
        if self.transform == 'rank':
            return _expanding_rank(values, self.warmup)

        # [保留] 全序列 rank — 含未来信息，仅用于对比实验
        if self.transform == 'rank_full':
            return pd.Series(values).rank(pct=True).values

        # [修复] expanding zscore — 无前向偏差（默认）
        if self.transform == 'zscore':
            return _expanding_zscore(values, self.warmup)

        # [保留] 全序列 zscore — 含未来信息，仅用于对比实验
        if self.transform == 'zscore_full':
            mean, std = np.nanmean(values), np.nanstd(values)
            if std < 1e-10:
                return np.zeros_like(values)
            return np.clip((values - mean) / std, -3.0, 3.0)

        if self.transform == 'binary':
            return np.where(values > 0, 1.0, 0.0)

        if self.transform == 'neg_log':
            v = np.clip(values, 0.0, None)
            # [修复] 2026-06-20 用 expanding max 替代全序列 max，消除前向偏差
            expanding_max = np.maximum.accumulate(v)
            expanding_max = np.where(expanding_max > 1e-10, expanding_max, 1e-10)
            return np.where(v > 1e-10, -np.log(v / expanding_max), 0.0)

        if callable(self.transform):
            return self.transform(values)

        return values


# ============================================================
# 复合评分器
# ============================================================
class CompositeScorer:
    """将多个 ScoredSignal 加权求和为复合分数"""

    def __init__(self, signals: List[ScoredSignal],
                 feature_df: Optional[pd.DataFrame] = None):
        self.signals = signals
        self.feature_df = feature_df

    def set_feature_df(self, feature_df: pd.DataFrame):
        self.feature_df = feature_df

    def add_signal(self, signal: ScoredSignal):
        self.signals.append(signal)

    def compute(self) -> pd.Series:
        """加权求和所有信号，返回复合分数 Series"""
        if self.feature_df is None:
            raise ValueError("请先通过 set_feature_df() 设置特征数据")

        score = np.zeros(len(self.feature_df))
        total_weight = 0.0

        for sig in self.signals:
            if sig.name not in self.feature_df.columns:
                continue
            weighted = sig.apply(self.feature_df[sig.name]) * sig.weight
            score += weighted
            total_weight += abs(sig.weight)

        if total_weight > 0:
            score = score / total_weight

        return pd.Series(score, index=self.feature_df.index,
                         name='composite_score')


# ============================================================
# 三区状态机
# ============================================================
def three_zone_backtest(score: pd.Series,
                        data: pd.DataFrame,
                        entry_threshold: float = 0.3,
                        exit_threshold: float = -0.3,
                        symbol: str = '399317.SZ',
                        initial_capital: float = 1_000_000,
                        start_date: str = None,
                        fee_config: dict = None) -> 'BacktestResult':
    """
    三区状态机回测 — ft2.core Engine 驱动

    复合分数通过三区逻辑转为持仓信号:
      score >  entry_thr  →  开多仓 (signal=1)
      score <  exit_thr   →  清仓 (signal=0)
      其余情况            →  保持原持仓

    生成 position 信号后交给 EngineV3.backtest(mode='fast') 做撮合，
    与 ft2.core 结果完全一致。

    Parameters
    ----------
    score : pd.Series
        复合分数序列（index=日期）
    data : pd.DataFrame
        OHLCV DataFrame (index=DatetimeIndex)
    entry_threshold : float
        开多阈值
    exit_threshold : float
        清仓阈值
    symbol : str
        交易标的
    initial_capital : float
        初始资金
    start_date : str
        回测起始日
    fee_config : dict
        费率配置, None=零费率

    Returns
    -------
    BacktestResult
        含 sharpe / cagr / max_drawdown / trades / nav / position / score
    """
    from .engine import EngineV3

    # [修复] 2026-06-20 检查 score 与 data index 对齐
    if not score.index.equals(data.index):
        common = score.index.intersection(data.index)
        if len(common) < len(score) * 0.9:
            import warnings
            warnings.warn(f"score.index 与 data.index 仅 {len(common)}/{len(score)} 重叠，回测可能不准确")

    # 三区状态机 → position 信号
    score_vals = score.fillna(0).values
    n = len(score_vals)
    position = np.zeros(n)
    prev_pos = 0.0

    for i in range(n):
        s = score_vals[i]
        if s > entry_threshold:
            position[i] = 1.0
        elif s < exit_threshold:
            position[i] = 0.0
        else:
            position[i] = prev_pos
        prev_pos = position[i]

    # position 信号 → pd.Series (与 data index 对齐)
    signal = pd.Series(position, index=score.index, name='signal')

    # 走 EngineV3 fast 模式
    analyzer = EngineV3.backtest(
        signal, data, symbol=symbol, mode='fast',
        initial_capital=initial_capital, start_date=start_date,
        fee_config=fee_config)

    return BacktestResult(
        analyzer=analyzer,
        position=pd.Series(position, index=score.index, name='position'),
        score=score_vals,
    )


# ============================================================
# 回测结果容器
# ============================================================
class BacktestResult:
    """三区回测结果 — 包装 AccountAnalyzer

    所有指标返回小数 (与 AccountAnalyzer 一致):
      sharpe=1.23, cagr=0.085, max_drawdown=-0.105, win_rate=0.55
    展示时用 .1% 格式化自动转百分比。
    """

    def __init__(self, analyzer, position, score):
        self._analyzer = analyzer
        self.position = position
        self.score = score

    @property
    def sharpe(self):
        return self._analyzer.sharpe_ratio() or 0

    @property
    def cagr(self):
        return self._analyzer.annualized_return() or 0

    @property
    def total_return(self):
        return self._analyzer.return_rate() or 0

    @property
    def max_drawdown(self):
        dd = self._analyzer.max_drawdown()
        return dd[0] if dd else 0  # 小数, 如 -0.105

    @property
    def annual_vol(self):
        return self._analyzer.volatility() or 0

    @property
    def trades(self):
        return len(self._analyzer.trade_profits)

    @property
    def win_rate(self):
        return self._analyzer.win_rate() or 0

    @property
    def calmar(self):
        return self._analyzer.calmar_ratio() or 0

    @property
    def nav(self):
        assets = self._analyzer.daily_assets
        dates_sorted = sorted(assets.keys())
        return pd.Series(
            [assets[d] for d in dates_sorted],
            index=pd.DatetimeIndex(dates_sorted),
            name='nav'
        )

    # 兼容旧属性名 (量纲与 cagr 一致, 均为小数)
    @property
    def annual_return(self):
        return self._analyzer.annualized_return() or 0

    @property
    def trade_count(self):
        return len(self._analyzer.trade_profits)  # fast模式=0, full模式=交易回合数

    def __repr__(self):
        return (f"BacktestResult(SR={self.sharpe:.3f}, CAGR={self.cagr:.1%}, "
                f"MDD={self.max_drawdown:.1%}, trades={self.trades})")
