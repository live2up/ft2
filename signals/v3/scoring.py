"""
signals/v3/scoring.py — 连续值打分择时框架
=============================================================================
v3 独立版本。将多个连续信号加权合成为复合分数，通过三区状态机生成持仓决策。

[修复] 2026-06-10 zscore/rank 改为 expanding 统计量，消除全序列前向偏差:
  - zscore: expanding mean/std (t时刻只用[0:t]的数据)
  - rank: expanding percentile rank (t时刻只统计历史)
  旧实现用 np.nanmean/np.nanstd 全序列 → score[i] 含未来信息 → 虚高 Sharpe

=============================================================================
"""
import numpy as np
import pandas as pd
from typing import List, Optional, Callable


# ============================================================
# Expanding 统计工具 (防前向偏差)
# ============================================================

def _expanding_zscore(values: np.ndarray, min_warmup: int = 20) -> np.ndarray:
    """expanding z-score: 每时刻只用历史数据计算 mean/std

    zscore[i] = (values[i] - mean[0:i+1]) / std[0:i+1], 截断到 [-3, 3]
    [修复] 替代原 np.nanmean/np.nanstd 全序列版本，消除前向偏差

    Args:
        min_warmup: 冷启动天数，前 min_warmup 天统计量不稳定，返回 0（中性）
    """
    n = len(values)
    result = np.full(n, np.nan)
    cumsum = np.nancumsum(values)
    cumsum2 = np.nancumsum(values ** 2)
    count = np.arange(1, n + 1, dtype=float)

    for i in range(min_warmup, n):
        mean = cumsum[i] / count[i]
        var = cumsum2[i] / count[i] - mean ** 2
        std = np.sqrt(max(var, 1e-10))
        result[i] = np.clip((values[i] - mean) / std, -3.0, 3.0)

    # 冷启动期间 → 中性
    result[:min_warmup] = 0.0
    return np.nan_to_num(result, nan=0.0)


def _expanding_rank(values: np.ndarray, min_warmup: int = 20) -> np.ndarray:
    """expanding percentile rank: t时刻在历史中的百分位

    rank[i] = (#values[0:i+1] <= values[i]) / (i+1), 值域 [0, 1]
    [修复] 替代原 pd.Series.rank(pct=True) 全序列版本，消除前向偏差

    Args:
        min_warmup: 冷启动天数，前 min_warmup 天排名不稳定，返回 0.5（中位）
    """
    n = len(values)
    result = np.full(n, np.nan)

    for i in range(min_warmup, n):
        window = values[:i + 1]
        result[i] = np.sum(window <= values[i]) / (i + 1)

    result[:min_warmup] = 0.5  # 冷启动 → 中位
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
            warmup: expanding 变换的冷启动天数（仅 zscore/rank 有效）,
                    默认 20。前 warmup 天 zscore→0(中性), rank→0.5(中位)
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
            mx = v.max()
            return np.where(v > 1e-10, -np.log(v / (mx + 1e-10)), 0.0) if mx > 1e-10 else np.zeros_like(v)

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
                        prices: pd.Series,
                        entry_threshold: float = 0.3,
                        exit_threshold: float = -0.3) -> 'BacktestResult':
    """
    三区状态机回测 — 无前向偏差（状态机只依赖历史）

    复合分数通过三区逻辑转为持仓:
      score >  entry_thr  →  开多仓
      score <  exit_thr   →  清仓
      其余情况            →  保持原持仓

    Parameters
    ----------
    score : pd.Series
        复合分数序列（index=日期）
    prices : pd.Series
        价格序列（index=日期，用于计算收益）
    entry_threshold : float
        开多阈值
    exit_threshold : float
        清仓阈值

    Returns
    -------
    BacktestResult
        含 nav / sharpe / annual_return / max_drawdown / trade_count 等属性
    """
    score = score.fillna(0).values
    prices = prices.reindex(score.index).fillna(method='ffill').values

    n = len(score)
    position = np.zeros(n)
    prev_pos = 0.0

    for i in range(n):
        s = score[i]
        if s > entry_threshold:
            position[i] = 1.0
        elif s < exit_threshold:
            position[i] = 0.0
        else:
            position[i] = prev_pos
        prev_pos = position[i]

    # 绩效计算
    ret = pd.Series(prices).pct_change().shift(-1).fillna(0).values
    strategy_ret = position * ret
    nav = (1 + strategy_ret).cumprod()
    nav[0] = 1.0

    trades = int((np.diff(position, prepend=0) != 0).sum())
    sharpe = float(np.mean(strategy_ret) / (np.std(strategy_ret) + 1e-10) * np.sqrt(252))
    dd = np.minimum.accumulate(nav / np.maximum.accumulate(nav)) - 1
    max_dd = float(np.min(dd))
    ann = float(nav[-1] ** (252 / n) - 1) if n > 0 else 0.0

    pos_series = pd.Series(position, index=score.index, name='position')

    return BacktestResult(
        nav=pd.Series(nav, index=score.index, name='nav'),
        position=pos_series,
        sharpe=sharpe,
        annual_return=ann * 100,
        max_drawdown=max_dd * 100,
        trade_count=trades,
        score=score,
    )


# ============================================================
# 回测结果容器
# ============================================================
class BacktestResult:
    """三区回测结果"""

    def __init__(self, nav, position, sharpe, annual_return,
                 max_drawdown, trade_count, score):
        self.nav = nav
        self.position = position
        self.sharpe = sharpe
        self.annual_return = annual_return
        self.max_drawdown = max_drawdown
        self.trade_count = trade_count
        self.score = score

    def __repr__(self):
        return (f"BacktestResult(sharpe={self.sharpe:.3f}, "
                f"annual={self.annual_return:.1f}%, "
                f"max_dd={self.max_drawdown:.1f}%, "
                f"trades={self.trade_count})")
