"""
signals/v3/scoring.py — 连续值打分 (继承 v2)
=============================================================================
"""
from signals.v2.scoring import (
    ScoredSignal,
    CompositeScorer,
    three_zone_backtest,
    BacktestResult,
)

__all__ = ['ScoredSignal', 'CompositeScorer', 'three_zone_backtest', 'BacktestResult']
