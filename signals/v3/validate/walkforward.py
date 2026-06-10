"""
signals/v3/validate/walkforward.py — Walk-Forward 验证 (v3 独立)
=============================================================================

两种模式:
  1. Series 模式: walkforward_validate(signal, data, ...)
     预计算信号直接切片 → EngineV3 full (向后兼容)
     适用于: 特征已保证无前向偏差的信号

  2. Expression 模式: walkforward_validate_expr(expr_str, data, fs, ...)
     每窗口独立 fit FeatureSpace + generate Expression → EngineV3 full
     适用于: 检测特征/变换层的前向泄露 (zscore/rank 等)

[新增] 2026-06-10 Expression 模式 — 每窗口重算特征，真实 OOS 验证
=============================================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

from ..wf_result import WalkForwardCoreResult
from ..engine import EngineV3


# ── 时间窗口解析 ──
def _parse_td(s: str) -> pd.Timedelta:
    """解析时间窗口字符串 → Timedelta"""
    s = s.strip().upper()
    if s.endswith('Y'):
        return pd.Timedelta(days=int(s[:-1]) * 365)
    elif s.endswith('M'):
        return pd.Timedelta(days=int(s[:-1]) * 30)
    return pd.Timedelta(days=int(s.replace('D', '')))


# ── 窗口级回测 ──
def _run_window(seg_sig, seg_data, symbol, initial_capital):
    """单窗口 full 模式回测，返回指标 dict"""
    try:
        analyzer = EngineV3.backtest(
            seg_sig, seg_data, symbol=symbol, mode='full',
            initial_capital=initial_capital)
        m = analyzer.metrics()

        def _v(name, d=0):
            for k, v in m.items():
                if isinstance(v, dict) and v.get('name') == name:
                    val = v['value']
                    return val[0] if isinstance(val, tuple) else val
            return d

        return {
            'sharpe': _v('夏普比率'),
            'annual_return': _v('年化收益率'),
            'max_drawdown': _v('最大回撤'),
            'win_rate': _v('胜率'),
            'trades': len(analyzer.account.trade_records) // 2,
        }
    except Exception as e:
        return {
            'sharpe': 0, 'annual_return': 0,
            'max_drawdown': 0, 'win_rate': 0, 'trades': 0,
            'error': str(e),
        }


# ============================================================
# 模式 1: Series 入口 — 预计算信号切片 (向后兼容)
# ============================================================

def walkforward_validate(
    signal: pd.Series,
    data: pd.DataFrame,
    symbol: str = '399317.SZ',
    initial_capital: float = 1_000_000,
    start_date: str = None,
    train_size: str = '2Y',
    test_size: str = '1Y',
    step: str = '6M',
    anchor_start: bool = False,
) -> WalkForwardCoreResult:
    """
    Walk-Forward 验证 (Series 模式)。

    信号预计算，每窗口直接切片回测。
    要求: signal 必须无前向偏差（如用 expanding_zscore 而非全序列 zscore）。

    Args:
        signal: 预计算信号序列 (全量)
        data: OHLCV DataFrame
        train_size / test_size / step: '2Y' / '1Y' / '6M'
        anchor_start: True=扩展窗口, False=滚动窗口

    Returns:
        WalkForwardCoreResult: windows/summary/stability_score
    """
    train_td = _parse_td(train_size)
    test_td = _parse_td(test_size)
    step_td = _parse_td(step)

    windows = []
    train_start = data.index[0]

    for wi in range(20):
        train_end = train_start + train_td
        test_end = train_end + test_td

        if test_end > data.index[-1]:
            break

        train_mask = (data.index >= train_start) & (data.index < train_end)
        test_mask = (data.index >= train_end) & (data.index < test_end)
        train_data = data.loc[train_mask]
        test_data = data.loc[test_mask]

        if len(train_data) < 60 or len(test_data) < 20:
            if anchor_start:
                train_td += step_td
                if train_td > test_td * 6:
                    break
                continue
            train_start += step_td
            continue

        window = {
            'train_start': str(train_start.date()),
            'train_end': str(train_end.date()),
            'test_start': str(train_end.date()),
            'test_end': str(test_end.date()),
        }

        train_sig = signal.reindex(train_data.index).fillna(0)
        test_sig = signal.reindex(test_data.index).fillna(0)

        window['train'] = _run_window(train_sig, train_data, symbol, initial_capital)
        window['test'] = _run_window(test_sig, test_data, symbol, initial_capital)
        windows.append(window)

        if anchor_start:
            train_start = data.index[0]
            train_td += step_td
            if train_td > test_td * 6:
                break
        else:
            train_start += step_td

    return _summarize(windows)


# ============================================================
# 模式 2: Expression 入口 — 每窗口重算特征 (真实 OOS)
# ============================================================

def walkforward_validate_expr(
    expr_str: str,
    data: pd.DataFrame,
    feature_space: 'FeatureSpace' = None,
    symbol: str = '399317.SZ',
    initial_capital: float = 1_000_000,
    train_size: str = '2Y',
    test_size: str = '1Y',
    step: str = '6M',
    anchor_start: bool = False,
    verbose: bool = True,
) -> WalkForwardCoreResult:
    """
    Walk-Forward 验证 (Expression 模式) — 真实 OOS。

    每窗口独立:
      1. FeatureSpace.fit(train_data)  → train_features
      2. Expression.generate(train_data) → train_signal
      3. EngineV3.backtest(train_signal, train_data, mode='full')
      4. 测试窗口同步骤 1~3

    这才是严格的无前向偏差验证 —— 信号计算层和回测层双重隔离。

    Args:
        expr_str: 表达式字符串 (如 "thr_mean(ATR{7}) & thr_mean(EMA{20})")
        data: OHLCV DataFrame
        feature_space: FeatureSpace 实例 (None→创建默认)
        train_size / test_size / step: '2Y' / '1Y' / '6M'
        anchor_start: True=扩展窗口, False=滚动窗口
        verbose: 是否打印每窗口进度

    Returns:
        WalkForwardCoreResult: windows/summary/stability_score

    Example:
        >>> from signals.v3 import FeatureSpace, walkforward_validate_expr
        >>> fs = FeatureSpace()
        >>> wf = walkforward_validate_expr("thr_mean(ATR{7})", df, fs)
        >>> print(f"OOS Sharpe: {wf.summary['mean_test_sharpe']:.3f}")
        >>> print(f"稳定性: {wf.stability_score:.2f}")
        >>> print(f"负Sharpe窗口: {wf.negative_count}/{wf.summary['total_windows']}")
    """
    from ..expression import Expression
    from ..features import FeatureSpace as FS

    if feature_space is None:
        feature_space = FS()

    train_td = _parse_td(train_size)
    test_td = _parse_td(test_size)
    step_td = _parse_td(step)

    windows = []
    train_start = data.index[0]

    for wi in range(20):
        train_end = train_start + train_td
        test_end = train_end + test_td

        if test_end > data.index[-1]:
            break

        train_mask = (data.index >= train_start) & (data.index < train_end)
        test_mask = (data.index >= train_end) & (data.index < test_end)
        train_data = data.loc[train_mask]
        test_data = data.loc[test_mask]

        if len(train_data) < 60 or len(test_data) < 20:
            if anchor_start:
                train_td += step_td
                if train_td > test_td * 6:
                    break
                continue
            train_start += step_td
            continue

        window = {
            'train_start': str(train_start.date()),
            'train_end': str(train_end.date()),
            'test_start': str(train_end.date()),
            'test_end': str(test_end.date()),
        }

        if verbose:
            print(f"  WF[{wi+1}] {window['train_start']}→{window['test_end']} "
                  f"train={len(train_data)} test={len(test_data)} ...", end=' ')

        # ── 关键: 每窗口独立 fit + generate ──
        for phase, seg_data in [('train', train_data), ('test', test_data)]:
            try:
                # 1) fit 特征空间（仅窗口内数据）
                fs = feature_space.fit(seg_data)          # ← 重新 fit
                # 2) 生成信号
                expr = Expression(expr_str, feature_space=fs)
                seg_sig = expr.generate(seg_data)          # ← 重新 generate
                # 3) 回测
                window[phase] = _run_window(seg_sig, seg_data, symbol, initial_capital)
            except Exception as e:
                window[phase] = {
                    'sharpe': 0, 'annual_return': 0,
                    'max_drawdown': 0, 'win_rate': 0, 'trades': 0,
                    'error': str(e),
                }

        windows.append(window)

        if verbose:
            ts = window.get('test', {}).get('sharpe', 0)
            trs = window.get('train', {}).get('sharpe', 0)
            print(f"train_SR={trs:.3f} test_SR={ts:.3f}")

        if anchor_start:
            train_start = data.index[0]
            train_td += step_td
            if train_td > test_td * 6:
                break
        else:
            train_start += step_td

    return _summarize(windows)


# ============================================================
# 汇总统计
# ============================================================

def _summarize(windows: List[Dict]) -> WalkForwardCoreResult:
    test_sharpes = [w.get('test', {}).get('sharpe', 0) or 0 for w in windows]
    train_sharpes = [w.get('train', {}).get('sharpe', 0) or 0 for w in windows]

    summary = {
        'total_windows': len(windows),
        'mean_test_sharpe': float(np.mean(test_sharpes)) if test_sharpes else 0,
        'mean_train_sharpe': float(np.mean(train_sharpes)) if train_sharpes else 0,
        'min_test_sharpe': float(np.min(test_sharpes)) if test_sharpes else 0,
        'max_test_sharpe': float(np.max(test_sharpes)) if test_sharpes else 0,
        'negative_count': sum(1 for s in test_sharpes if s < 0),
        'positive_count': sum(1 for s in test_sharpes if s >= 0),
        'stability_score': float(np.mean(test_sharpes) / np.std(test_sharpes, ddof=1))
                           if len(test_sharpes) > 1 and np.std(test_sharpes, ddof=1) > 0 else 0,
        'overfit_ratio': float(np.mean(train_sharpes) / np.mean(test_sharpes))
                         if np.mean(test_sharpes) != 0 else float('inf'),
    }

    return WalkForwardCoreResult(windows=windows, summary=summary)
