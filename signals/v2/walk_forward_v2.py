"""
signals/v2/walk_forward_v2.py — 滚动窗口验证 (core 引擎版)
=============================================================================

[新增] 2026-06-03 signals/v2.1
与 v2 原版 walk_forward() 的区别:
  原版: 每窗口仅收集 train_sharpe / test_sharpe (简化回测)
  v2.1: 每窗口跑 core 引擎 → 收集 AccountAnalyzer 全套指标
        → 窗口间稳定性分析（夏普/回撤/胜率/盈亏比分布）


============================================================================
                         Walk-Forward 流程
============================================================================

 train_data (expand)          test_data
 ┌──────────────────┐ ┌──────────────────┐
 │ 计算信号 → core   │ │ 计算信号 → core   │ 
 │ 引擎回测(训练)    │ │ 引擎回测(测试)    │
 └──────────────────┘ └──────────────────┘
                               │
                   ┌───────────┼───────────┐
                   ▼           ▼           ▼
              夏普          最大回撤     胜率
              ...          ...          ...
                    
                   汇总 → WalkForwardCoreResult


============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field

from .validator import run_backtest_with_core

# Window metrics to collect from AccountAnalyzer
WINDOW_METRICS = [
    ('sharpe_ratio', 'sharpe'),
    ('return_rate', 'total_return'),
    ('annualized_return', 'annual_return'),
    ('volatility', 'annual_vol'),
    ('max_drawdown', 'max_drawdown'),
    ('sortino_ratio', 'sortino'),
    ('win_rate', 'win_rate'),
    ('avg_profit_loss_ratio', 'profit_loss_ratio'),
    ('avg_holding_period', 'avg_hold_days'),
    ('var', 'var_95'),
    ('cvar', 'cvar_95'),
    ('kelly_criterion', 'kelly'),
]


@dataclass
class WalkForwardCoreResult:
    """core 引擎版 Walk-Forward 验证结果"""

    windows: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    label: str = ""

    @property
    def train_sharpes(self) -> List[float]:
        return [w.get('train', {}).get('sharpe', 0) or 0 for w in self.windows]

    @property
    def test_sharpes(self) -> List[float]:
        return [w.get('test', {}).get('sharpe', 0) or 0 for w in self.windows]

    @property
    def stability_score(self) -> float:
        """夏普稳定性 = mean / sigma"""
        sharpes = self.test_sharpes
        if len(sharpes) < 2:
            return 0.0
        mu = np.mean(sharpes)
        sigma = np.std(sharpes, ddof=1)
        return float(mu / sigma) if sigma > 1e-10 else 0.0

    @property
    def negative_count(self) -> int:
        return sum(1 for s in self.test_sharpes if s < 0)

    @property
    def overfit_ratio(self) -> float:
        """过拟合比 = mean(train_sharpe) / mean(test_sharpe)"""
        t_mean = np.mean(self.test_sharpes) if self.test_sharpes else 0
        tr_mean = np.mean(self.train_sharpes) if self.train_sharpes else 0
        return float(tr_mean / t_mean) if abs(t_mean) > 1e-10 else float('inf')

    def to_dataframe(self) -> pd.DataFrame:
        """窗口级别指标表"""
        rows = []
        for w in self.windows:
            row = {
                'train_start': w.get('train_start', ''),
                'train_end': w.get('train_end', ''),
                'test_start': w.get('test_start', ''),
                'test_end': w.get('test_end', ''),
            }
            for scope in ('train', 'test'):
                metrics = w.get(scope, {})
                for metric_key, short_key in WINDOW_METRICS:
                    val = metrics.get(short_key)
                    if val is not None:
                        row[f'{scope}_{short_key}'] = round(val, 4)
            rows.append(row)
        return pd.DataFrame(rows)

    def stability_report(self) -> Dict:
        """窗口间指标稳定性报告"""
        report = {}
        for metric_key, short_key in WINDOW_METRICS:
            train_vals = [w.get('train', {}).get(short_key)
                         for w in self.windows]
            test_vals = [w.get('test', {}).get(short_key)
                        for w in self.windows]
            train_vals = [v for v in train_vals if v is not None]
            test_vals = [v for v in test_vals if v is not None]

            if len(test_vals) >= 2:
                report[short_key] = {
                    'train_mean': round(np.mean(train_vals), 4) if train_vals else None,
                    'test_mean': round(np.mean(test_vals), 4),
                    'test_std': round(np.std(test_vals, ddof=1), 4),
                    'test_cv': round(np.std(test_vals, ddof=1) / abs(np.mean(test_vals)), 4)
                               if abs(np.mean(test_vals)) > 1e-10 else None,
                }

        report['sharpe_stability'] = self.stability_score
        report['overfit_ratio'] = self.overfit_ratio
        report['negative_sharpe_windows'] = f"{self.negative_count}/{len(self.windows)}"
        return report

    def __repr__(self):
        n = len(self.windows)
        if n == 0:
            return "WalkForwardCoreResult(无窗口)"
        return (
            f"WalkForwardCoreResult(窗口={n}, "
            f"测试夏普均值={np.mean(self.test_sharpes):.2f}, "
            f"稳定性={self.stability_score:.2f}, "
            f"负夏普窗={self.negative_count}/{n})"
        )


def _parse_timedelta(s: str) -> pd.Timedelta:
    """解析时间窗口字符串 '2Y' / '6M' / '1Y' → Timedelta"""
    s = s.strip()
    unit_map = {'Y': 'years', 'y': 'years', 'M': 'months', 'm': 'months',
                'D': 'days', 'd': 'days'}
    unit = 'days'
    for u in unit_map:
        if s.endswith(u):
            unit = unit_map[u]
            s = s[:-1]
            break
    val = int(s)
    if unit == 'years':
        return pd.Timedelta(days=int(val * 365.25))
    elif unit == 'months':
        return pd.Timedelta(days=int(val * 30.5))
    return pd.Timedelta(days=val)


def _collect_window_metrics(analyzer) -> Dict:
    """从 AccountAnalyzer 收集一个窗口的指标"""
    metrics = {}
    for method_name, short_key in WINDOW_METRICS:
        try:
            val = getattr(analyzer, method_name)()
            if isinstance(val, tuple):
                val = val[0]  # max_drawdown 返回 (rate, peak, trough)
            metrics[short_key] = val
        except Exception:
            metrics[short_key] = None
    return metrics


def walk_forward_with_core(
    expr_or_gen,
    data: pd.DataFrame,
    symbol: str = '399317.SZ',
    freq: str = '1d',
    train_size: str = '2Y',
    test_size: str = '1Y',
    step: str = '6M',
    initial_capital: float = 1_000_000,
    long_only: bool = True,
    anchor_start: bool = False,
    note_fields: List[str] = None,
) -> WalkForwardCoreResult:
    """
    用 core 引擎跑 Walk-Forward 滚动验证。

    每窗口流程:
      1. 切分 train/test 数据
      2. 重置特征空间（防前瞻偏差）
      3. 计算信号 → run_backtest_with_core → 收集全套指标

    Args:
        expr_or_gen: Expression 实例（需有 generate 方法）
        data: OHLCV DataFrame
        symbol: 交易标的
        freq: 频率
        train_size: 训练窗口 ('2Y'=2年)
        test_size: 测试窗口 ('1Y'=1年)
        step: 步进长度 ('6M'=6个月)
        initial_capital: 初始资金
        long_only: 仅做多
        anchor_start: True=锚定起点(扩展窗口), False=滚动窗口
        note_fields: 附加 bar 字段写入 TradeRecord.note

    Returns:
        WalkForwardCoreResult: 含窗口指标 + 稳定性报告

    Example:
        >>> from signals.v2 import walk_forward_with_core, Expression, FeatureSpace
        >>> fs = FeatureSpace()
        >>> expr = Expression("thr_mean(ATR{7})", fs)
        >>> wf = walk_forward_with_core(expr, df, symbol='399317.SZ')
        >>> print(wf.stability_report())
        >>> df_metrics = wf.to_dataframe()
    """
    train_td = _parse_timedelta(train_size)
    test_td = _parse_timedelta(test_size)
    step_td = _parse_timedelta(step)

    result = WalkForwardCoreResult()

    if isinstance(data.index, pd.DatetimeIndex):
        dates = data.index
    else:
        dates = pd.to_datetime(data.index)

    # 确保日期排列
    data_sorted = data.sort_index()

    train_start = dates[0]
    windows = []

    while True:
        train_end = train_start + train_td
        test_end = train_end + test_td

        if test_end > dates[-1]:
            break

        train_mask = (dates >= train_start) & (dates < train_end)
        test_mask = (dates >= train_end) & (dates <= test_end)
        train_data = data_sorted.loc[train_mask]
        all_data = data_sorted.loc[train_mask | test_mask]

        if len(train_data) < 30 or len(test_data) < 10:
            break

        # ── 训练集回测（仅用 train_data，防前瞻偏差）──
        try:
            if hasattr(expr_or_gen, '_feature_df'):
                expr_or_gen._feature_df = None
            if hasattr(expr_or_gen, '_feature_space'):
                expr_or_gen._feature_space = None

            train_analyzer = run_backtest_with_core(
                expr_or_gen, train_data, symbol=symbol, freq=freq,
                initial_capital=initial_capital, long_only=long_only,
                note_fields=note_fields,
            )
            train_metrics = _collect_window_metrics(train_analyzer)
        except Exception as e:
            train_metrics = {'error': str(e)}

        # ── 测试集回测（all_data=train+test，train作lookback，test执行） ──
        try:
            if hasattr(expr_or_gen, '_feature_df'):
                expr_or_gen._feature_df = None
            if hasattr(expr_or_gen, '_feature_space'):
                expr_or_gen._feature_space = None

            test_analyzer = run_backtest_with_core(
                expr_or_gen, all_data, symbol=symbol, freq=freq,
                initial_capital=initial_capital, long_only=long_only,
                note_fields=note_fields,
            )
            test_metrics = _collect_window_metrics(test_analyzer)
        except Exception as e:
            test_metrics = {'error': str(e)}

        windows.append({
            'train_start': str(train_start.date()),
            'train_end': str(train_end.date()),
            'test_start': str(train_end.date()),
            'test_end': str(test_end.date()),
            'train': train_metrics,
            'test': test_metrics,
        })

        if anchor_start:
            # 扩展窗口模式
            train_start = dates[0]
            train_td += step_td
            if train_td > test_td * 6:
                break
        else:
            # 滚动窗口模式
            train_start += step_td

        # 安全检查
        if train_start + train_td + test_td > dates[-1]:
            break

    result.windows = windows
    result.label = f"{train_size}/{test_size}/{step}"

    # 汇总统计
    test_s = [w.get('test', {}).get('sharpe') for w in windows
              if 'error' not in w.get('test', {})]
    test_s = [s for s in test_s if s is not None]
    train_s = [w.get('train', {}).get('sharpe') for w in windows
               if 'error' not in w.get('train', {})]
    train_s = [s for s in train_s if s is not None]

    result.summary = {
        'total_windows': len(windows),
        'mean_test_sharpe': round(np.mean(test_s), 4) if test_s else None,
        'std_test_sharpe': round(np.std(test_s, ddof=1), 4) if len(test_s) > 1 else None,
        'min_test_sharpe': round(min(test_s), 4) if test_s else None,
        'max_test_sharpe': round(max(test_s), 4) if test_s else None,
        'negative_sharpe_count': sum(1 for s in test_s if s < 0),
        'stability_score': round(
            np.mean(test_s) / np.std(test_s, ddof=1), 4
        ) if len(test_s) > 1 and np.std(test_s, ddof=1) > 1e-10 else 0.0,
        'decay_ratio': round(
            np.mean(train_s) / np.mean(test_s), 4
        ) if test_s and train_s and abs(np.mean(test_s)) > 1e-10 else None,
    }

    return result
