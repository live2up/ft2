"""
signals/v3/validate/walkforward.py — Walk-Forward 验证 (v3 独立引擎版)
=============================================================================
每窗口用 v3.EngineV3(mode='full')。
=============================================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, List

from ..wf_result import WalkForwardCoreResult
from ..engine import EngineV3


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
    v3 Walk-Forward 验证 — 每窗口用 EngineV3(mode='full')。

    Args:
        signal: 预计算信号 (全量)
        data: OHLCV DataFrame
        train_size / test_size / step: '2Y' / '1Y' / '6M'
        anchor_start: True=扩展窗口, False=滚动窗口

    Returns:
        WalkForwardCoreResult: windows/summary/stability_score
    """
    def _parse_td(s: str) -> pd.Timedelta:
        s = s.strip().upper()
        if s.endswith('Y'):
            return pd.Timedelta(days=int(s[:-1]) * 365)
        elif s.endswith('M'):
            return pd.Timedelta(days=int(s[:-1]) * 30)
        return pd.Timedelta(days=int(s.replace('D', '')))

    train_td = _parse_td(train_size)
    test_td = _parse_td(test_size)
    step_td = _parse_td(step)

    windows = []
    train_start = data.index[0]

    for wi in range(20):  # 安全上限
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

        # 信号截取
        train_sig = signal.reindex(train_data.index).fillna(0)
        test_sig = signal.reindex(test_data.index).fillna(0)

        for phase, seg_data, seg_sig in [
            ('train', train_data, train_sig),
            ('test', test_data, test_sig)
        ]:
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

                window[phase] = {
                    'sharpe': _v('夏普比率'),
                    'annual_return': _v('年化收益率'),
                    'max_drawdown': _v('最大回撤'),
                    'win_rate': _v('胜率'),
                    'trades': len(analyzer.account.trade_records) // 2,
                }
            except Exception as e:
                window[phase] = {
                    'sharpe': 0, 'annual_return': 0,
                    'max_drawdown': 0, 'win_rate': 0, 'trades': 0,
                    'error': str(e),
                }

        windows.append(window)

        if anchor_start:
            train_start = data.index[0]
            train_td += step_td
            if train_td > test_td * 6:
                break
        else:
            train_start += step_td

    # 汇总
    test_sharpes = [w.get('test', {}).get('sharpe', 0) or 0 for w in windows]
    summary = {
        'total_windows': len(windows),
        'mean_test_sharpe': float(np.mean(test_sharpes)) if test_sharpes else 0,
        'negative_count': sum(1 for s in test_sharpes if s < 0),
        'stability_score': float(np.mean(test_sharpes) / np.std(test_sharpes, ddof=1))
                           if len(test_sharpes) > 1 and np.std(test_sharpes) > 0 else 0,
    }

    return WalkForwardCoreResult(windows=windows, summary=summary)
