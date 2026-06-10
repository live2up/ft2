"""
signals/v3/validate/compare.py — 多信号对比 (v3 独立)
=============================================================================
"""
import pandas as pd
from typing import List, Dict, Any

from ..engine import EngineCore


def compare_signals(
    signals: List[Dict[str, Any]],
    data: pd.DataFrame,
    symbol: str = '399317.SZ',
    initial_capital: float = 1_000_000,
    start_date: str = None,
) -> pd.DataFrame:
    """
    多信号批量 full 模式对比。

    Args:
        signals: List[Dict], 每个 dict 含:
            - 'name': 信号名称
            - 'expr': 表达式字符串 (可选, Expression 计算)
            - 'signal': pd.Series (可选, 直接传信号)
        data: OHLCV DataFrame
        symbol: 标的
        initial_capital: 初始资金
        start_date: 回测起始日

    Returns:
        DataFrame: 按 Sharpe 降序排列的对比表
    """
    results = []
    for i, sig in enumerate(signals):
        name = sig.get('name', f'信号{i+1}')
        try:
            if 'signal' in sig and sig['signal'] is not None:
                signal_series = sig['signal']
            elif 'expr' in sig:
                from ..expression_v3 import Expression
                from ..features import FeatureSpace
                fs = sig.get('feature_space') or FeatureSpace().fit(data)
                signal_series = Expression(sig['expr'], feature_space=fs).generate(data)
            else:
                print(f"  [{name}] 跳过: 无 signal 或 expr")
                continue

            analyzer = EngineCore.backtest(
                signal_series, data, symbol=symbol, mode='full',
                initial_capital=initial_capital, start_date=start_date)

            m = analyzer.metrics()

            def _v(key, d=0):
                for k, v in m.items():
                    if isinstance(v, dict) and v.get('name') == key:
                        val = v['value']
                        return val[0] if isinstance(val, tuple) else val
                return d

            results.append({
                '排名': 0, '名称': name,
                'Sharpe': round(float(_v('夏普比率')), 3),
                '年化': float(_v('年化收益率')),
                '最大回撤': float(_v('最大回撤')),
                '交易': len(analyzer.account.trade_records) // 2,
                '胜率': float(_v('胜率')),
                '_analyzer': analyzer,
            })
            print(f"  [{name}] SR={results[-1]['Sharpe']:.3f}")
        except Exception as e:
            print(f"  [{name}] FAIL: {e}")

    results.sort(key=lambda r: r['Sharpe'], reverse=True)
    for i, r in enumerate(results):
        r['排名'] = i + 1
    return pd.DataFrame(results)
