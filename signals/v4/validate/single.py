"""
signals/v4/validate/single.py — 单信号标准回测验证

用法:
  >>> from signals.v4.validate import validate_single
  >>> analyzer = validate_single("rsi(CLOSE,14) > 0.3", data, start_date='2020-01-01')
  >>> analyzer.to_notebook("RSI 策略")
"""
import pandas as pd
from ..expression import Expression
from ..engine import SigEngine


def validate_single(expr_str: str, data: pd.DataFrame,
                    symbol: str = '399317.SZ',
                    start_date: str = None,
                    bench_label: str = '399317.SZ'):
    """
    单信号标准回测 (ft2.core Engine, full 模式, 含费率)

    Args:
        expr_str: V4 表达式, 如 "rsi(CLOSE, 14) > 0.3"
        data: OHLCV DataFrame
        symbol: 交易标的
        start_date: 回测起始日
        bench_label: 基准标签

    Returns:
        AccountAnalyzer (含 metrics/to_notebook 方法)
    """
    expr = Expression(expr_str)
    signal = expr.generate(data)
    return SigEngine.backtest(signal, data, mode='full',
                               symbol=symbol, start_date=start_date,
                               bench_label=bench_label)
