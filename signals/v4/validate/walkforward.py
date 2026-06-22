"""
signals/v4/validate/walkforward.py — Walk-Forward 滚动窗口验证

核心指标:
  - mean_sharpe:     各窗口均值 (越高越好)
  - stability:       1 - std(sharpe) (越接近1越好)
  - negative_count:  负 Sharpe 窗口数 (越少越好)
  - win_rate:        正向窗口比例 (越接近100%越好)

用法:
  >>> from signals.v4.validate import validate_walkforward
  >>> wf = validate_walkforward("atr(HIGH,LOW,CLOSE,7) > ts_mean(atr(HIGH,LOW,CLOSE,7),30)", data)
  >>> print(f"均值SR={wf['mean_sharpe']:.3f} 稳定性={wf['stability']:.3f} 正窗口={wf['win_rate']:.0%}")

[规范化] 2026-06-22 函数名改为 validate_walkforward, 对齐 validate_single/compare_signals 的 verb_noun 模式。
"""
import pandas as pd, numpy as np
from typing import Dict, List
from ..expression import Expression
from ..engine import SigEngine


def validate_walkforward(expr_str: str, data: pd.DataFrame,
                         train: str = '2Y', test: str = '1Y',
                         step: str = '6M',
                         symbol: str = '399317.SZ') -> Dict:
    """
    Walk-Forward 滚动窗口验证

    Args:
        expr_str: V4 表达式
        data: OHLCV DataFrame (index=DatetimeIndex)
        train: 训练窗口长度, '2Y' / '3Y' / '1Y'
        test: 测试窗口长度, '1Y' / '6M'
        step: 滑动步长, '6M' / '1Y'
        symbol: 交易标的

    Returns:
        Dict: {'windows':[...], 'mean_sharpe':..., 'stability':..., ...}
    """
    expr = Expression(expr_str)

    # 生成窗口
    windows = _generate_windows(data.index, train, test, step)
    results = []

    for win in windows:
        train_start, train_end, test_start, test_end = win
        try:
            seg = data[test_start:test_end]
            if len(seg) < 30:
                continue
            signal = expr.generate(data[train_start:test_end])
            # 只取测试期部分
            test_signal = signal[test_start:test_end]
            r = SigEngine.backtest(test_signal, data[test_start:test_end],
                                    mode='fast', symbol=symbol)
            dd_result = r.max_drawdown()
            results.append({
                'window': f"{test_start.strftime('%Y-%m')} ~ {test_end.strftime('%Y-%m')}",
                'sharpe': r.sharpe_ratio() or 0,
                'cagr': r.annualized_return() or 0,
                'mdd': dd_result[0] if dd_result else 0,
                'trades': len(r.trade_profits),
            })
        except Exception as e:
            results.append({
                'window': f"{test_start.strftime('%Y-%m')} ~ {test_end.strftime('%Y-%m')}",
                'sharpe': 0, 'cagr': 0, 'mdd': 0, 'trades': 0,
                'error': str(e),
            })

    if not results:
        return {'windows': [], 'mean_sharpe': 0, 'stability': 0,
                'negative_count': 0, 'win_rate': 0}

    srs = [r['sharpe'] for r in results]
    positive = sum(1 for s in srs if s > 0)

    return {
        'windows': results,
        'mean_sharpe': np.mean(srs),
        'sharpe_std': np.std(srs),
        'stability': max(0, 1 - np.std(srs)),
        'negative_count': sum(1 for s in srs if s < 0),
        'win_rate': positive / len(srs) if len(srs) > 0 else 0,
        'min_sharpe': min(srs),
        'max_sharpe': max(srs),
    }


def _generate_windows(index: pd.DatetimeIndex, train: str,
                      test: str, step: str):
    """生成滚动窗口起止日期"""
    windows = []
    start = index.min()
    end = index.max()

    # 解析时间偏移
    train_offset = _parse_offset(train)
    test_offset = _parse_offset(test)
    step_offset = _parse_offset(step)

    current = start + train_offset
    while current + test_offset <= end:
        windows.append((
            current - train_offset,   # train_start
            current - pd.Timedelta(days=1),  # train_end
            current,                  # test_start
            current + test_offset,    # test_end
        ))
        current += step_offset
    return windows


def _parse_offset(offset: str) -> pd.offsets.DateOffset:
    """解析 '2Y' / '6M' / '1Y' 为 DateOffset

    [修复] 2026-06-22 使用 pd.DateOffset 替代 30天/365天近似，
    避免月度窗口漂移（30天 vs 实际自然月差异超过 10%）。
    回退逻辑保留，以防传入未识别后缀。
    """
    if offset.endswith('Y'):
        return pd.offsets.DateOffset(years=int(offset[:-1]))
    elif offset.endswith('M'):
        return pd.offsets.DateOffset(months=int(offset[:-1]))
    elif offset.endswith('D'):
        return pd.offsets.DateOffset(days=int(offset[:-1]))
    return pd.offsets.DateOffset(years=1)
