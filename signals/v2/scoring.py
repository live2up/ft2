"""
scoring.py — 多条件打分择时框架
=============================================================================
将多个连续信号加权合成为复合分数，通过三区状态机生成持仓决策。
补充 validator.py 的二进制信号回测，支持连续信号的"开多/保持/清仓"逻辑。

核心用法:
  from signals.v2 import CompositeScorer, ScoredSignal, three_zone_backtest

  # 1. 定义信号组合
  signals = [
      ScoredSignal('CLOSE_MA_RATIO{10}', weight=0.4),
      ScoredSignal('MOM_CHAIN{10}',      weight=0.3),
      ScoredSignal('RETURN_ABS{5}',      weight=0.3, transform='neg_log'),
  ]

  # 2. 计算复合分数
  scorer = CompositeScorer(signals, feature_df)
  score = scorer.compute()

  # 3. 三区回测
  bt = three_zone_backtest(score, prices, entry_thr=0.3, exit_thr=-0.3)
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import List, Optional, Callable, Dict


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
        - 'raw'    : 原始值（需已归一化到 -1~1）
        - 'rank'   : 百分位排名 0~1
        - 'zscore' : Z-score 标准化，截断到 [-3, 3]
        - 'binary' : thr_0 二值化 {0, 1}
        - 'neg_log': 负对数变换（用于波动率等正偏态分布）
        - callable : 自定义变换函数 fn(values) → array
    """

    def __init__(self, name: str, weight: float = 1.0,
                 transform: str = 'raw'):
        self.name = name
        self.weight = weight
        self.transform = transform

    def apply(self, series: pd.Series) -> np.ndarray:
        values = series.values.copy()
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        if self.transform == 'raw':
            return values

        if self.transform == 'rank':
            return pd.Series(values).rank(pct=True).values

        if self.transform == 'zscore':
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
    三区状态机回测

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
    """三区回测结果，保持与 validator.BTResult 风格一致"""

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
