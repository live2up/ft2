"""
signals/v3/validate/single.py — 单信号验证
=============================================================================
"""
import pandas as pd
from typing import List, Optional

from ..engine import EngineCore


def validate_single(
    signal: pd.Series,
    data: pd.DataFrame,
    symbol: str = '399317.SZ',
    initial_capital: float = 1_000_000,
    start_date: str = None,
    title: str = "单信号回测",
    bench_label: str = None,
    note_fields: List[str] = None,
):
    """
    单信号 full 模式验证 → AccountAnalyzer。

    Returns:
        AccountAnalyzer: 可链式调用 .to_notebook(title, header=, footer=)
    """
    return EngineCore.backtest(
        signal, data, symbol=symbol, mode='full',
        initial_capital=initial_capital, start_date=start_date,
        note_fields=note_fields, bench_label=bench_label)
