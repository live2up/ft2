"""
signals/v4/validate/compare.py — 多信号对比验证

用法:
  >>> from signals.v4.validate import compare_signals
  >>> df = compare_signals([
  ...     {'name': 'ATR突破', 'expr': 'atr(HIGH,LOW,CLOSE,7) > ts_mean(atr(HIGH,LOW,CLOSE,7),30)'},
  ...     {'name': 'RSI超卖', 'expr': 'rsi(CLOSE,14) < -0.3'},
  ... ], data)
"""
import pandas as pd, numpy as np
from ..expression import Expression
from ..engine import EngineCore


def compare_signals(signals: list, data: pd.DataFrame,
                    symbol: str = '399317.SZ',
                    start_date: str = None,
                    mode: str = 'full') -> pd.DataFrame:
    """
    多信号对比回测

    Args:
        signals: [{'name': '名称', 'expr': 'V4表达式'}, ...]
        data: OHLCV DataFrame
        symbol: 交易标的
        start_date: 回测起始日
        mode: 'fast' (快速) / 'full' (标准, 含费率)

    Returns:
        DataFrame, columns=[排名, 名称, 表达式, Sharpe, 年化, 最大回撤, 胜率, 交易]
    """
    results = []
    for i, sig in enumerate(signals):
        name = sig.get('name', f'信号{i+1}')
        expr_str = sig['expr']
        try:
            expr = Expression(expr_str)
            signal = expr.generate(data)
            result = EngineCore.backtest(signal, data, mode=mode,
                                         symbol=symbol, start_date=start_date)
            # full/fast 统一返回 AccountAnalyzer，直接调方法
            dd = result.max_drawdown()
            results.append({
                '名称': name, '表达式': expr_str,
                'Sharpe': round(result.sharpe_ratio() or 0, 3),
                '年化': result.annualized_return() or 0,
                '最大回撤': dd[0] if dd else 0,
                '胜率': result.win_rate() or 0,
                '交易': len(result.trade_profits),
            })
        except Exception as e:
            results.append({
                '名称': name, '表达式': expr_str,
                'Sharpe': 0, '年化': 0, '最大回撤': 0, '胜率': 0, '交易': 0,
                'error': str(e),
            })

    df = pd.DataFrame(results).sort_values('Sharpe', ascending=False)
    df.insert(0, '排名', range(1, len(df) + 1))
    return df


def signal_correlation(signals: list, data: pd.DataFrame) -> pd.DataFrame:
    """
    信号相关性分析 — 检测冗余信号, 辅助融合决策

    Args:
        signals: [{'name': '名称', 'expr': 'V4表达式'}, ...]
        data: OHLCV DataFrame

    Returns:
        DataFrame: 相关性矩阵 (n×n), 值越高越冗余

    解读:
        >0.8  高度冗余, 融合无增益
        0.5~0.8 中度相关, 谨慎融合
        <0.3  低相关, 适合融合
    """
    signal_series = {}
    for sig in signals:
        name = sig.get('name', sig['expr'][:30])
        try:
            expr = Expression(sig['expr'])
            signal_series[name] = expr.generate(data).fillna(0)
        except Exception as e:
            print(f"  X {name}: {e}")

    if len(signal_series) < 2:
        return pd.DataFrame()

    df_signals = pd.DataFrame(signal_series)
    return df_signals.corr().round(3)

