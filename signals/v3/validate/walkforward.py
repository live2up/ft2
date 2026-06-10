"""
signals/v3/validate/walkforward.py — Walk-Forward 验证 (v3 独立)
=============================================================================

两种模式:
  1. Series 模式: walkforward_validate(signal, data, ...)
     预计算信号直接切片 → EngineV3 full (向后兼容)

  2. Expression 模式: walkforward_validate_expr(expr_str, data, fs, ...)
     每窗口独立 fit FeatureSpace + generate Expression → EngineV3 full
     严格 OOS: 特征计算 + 回测双层隔离

[新增] 2026-06-10 Expression 模式 — 每窗口重算特征，真实 OOS 验证

参数命名规范:
  train_size/test_size/step  窗口周期 ('2Y'/'1Y'/'6M')
  min_train_bars/min_test_bars 窗口最小 bar 数 (默认 60/20)
  initial_capital             初始资金 (所有 EngineV3 入口统一)
  start_date                  回测起始日 (所有 EngineV3 入口统一)
  warmup                      ScoredSignal 冷启动 bar 数 (scoring 层)
=============================================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union

from ..wf_result import WalkForwardCoreResult
from ..engine import EngineV3


# ── 时间窗口解析 ──
def _parse_td(s: str) -> pd.Timedelta:
    """解析时间窗口字符串 → Timedelta ('2Y' → 730 days)"""
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
    train_size: str = '2Y',
    test_size: str = '1Y',
    step: str = '6M',
    min_train_bars: int = 60,
    min_test_bars: int = 20,
    anchor_start: bool = False,
) -> WalkForwardCoreResult:
    """
    Walk-Forward 验证 (Series 模式) — 预计算信号切片回测。

    要求: signal 必须无前向偏差（如用 expanding_zscore 而非全序列 zscore）。

    Args:
        signal: 预计算信号序列 (全量, pd.Series)
        data: OHLCV DataFrame (index=DatetimeIndex)
        symbol: 交易标的
        initial_capital: 初始资金
        train_size: 训练窗口长度 ('2Y'=2年, '18M'=18月, '500D'=500天)
        test_size: 测试窗口长度
        step: 窗口滑动步长
        min_train_bars: 训练窗口最少 bar 数 (默认 60 ≈ 一个季度)
        min_test_bars: 测试窗口最少 bar 数 (默认 20 ≈ 一个月)
        anchor_start: True=扩展窗口(训练集不断扩大), False=滚动窗口

    Returns:
        WalkForwardCoreResult

    Example:
        >>> wf = walkforward_validate(signal, df, train_size='2Y', test_size='1Y')
        >>> print(f"OOS SR={wf.summary['mean_test_sharpe']:.3f}, "
        ...       f"stab={wf.stability_score:.2f}")
    """
    train_td = _parse_td(train_size)
    test_td = _parse_td(test_size)
    step_td = _parse_td(step)

    windows = []
    train_start = data.index[0]

    while True:
        train_end = train_start + train_td
        test_end = train_end + test_td
        if test_end > data.index[-1]:
            break

        train_mask = (data.index >= train_start) & (data.index < train_end)
        test_mask = (data.index >= train_end) & (data.index < test_end)
        train_data = data.loc[train_mask]
        test_data = data.loc[test_mask]

        if len(train_data) < min_train_bars or len(test_data) < min_test_bars:
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
    expression: str,
    data: pd.DataFrame,
    feature_space: 'FeatureSpace' = None,
    symbol: str = '399317.SZ',
    initial_capital: float = 1_000_000,
    train_size: str = '2Y',
    test_size: str = '1Y',
    step: str = '6M',
    min_train_bars: int = 60,
    min_test_bars: int = 20,
    anchor_start: bool = False,
    verbose: bool = True,
) -> WalkForwardCoreResult:
    """
    Walk-Forward 验证 (Expression 模式) — 真实 OOS。

    每窗口独立:
      1. fs.fit(train_data) → train_features
      2. expr.generate(train_data) → train_signal
      3. EngineV3.backtest(train_signal, train_data, mode='full')
      4. 测试窗口同步骤 1~3

    Args:
        expression: 表达式字符串 ("thr_mean(ATR{7}) & thr_mean(EMA{20})")
        data: OHLCV DataFrame
        feature_space: FeatureSpace 实例 (None→创建默认)
        symbol: 交易标的
        initial_capital: 初始资金
        train_size: 训练窗口 ('2Y')
        test_size: 测试窗口 ('1Y')
        step: 滑动步长 ('6M')
        min_train_bars: 训练窗口最少 bar 数 (默认 60)
        min_test_bars: 测试窗口最少 bar 数 (默认 20)
        anchor_start: True=扩展窗口, False=滚动窗口
        verbose: 是否打印每窗口进度

    Returns:
        WalkForwardCoreResult

    Example:
        >>> fs = FeatureSpace()
        >>> wf = walkforward_validate_expr("thr_mean(ATR{7})", df, fs)
        >>> print(f"OOS SR={wf.summary['mean_test_sharpe']:.3f}")
    """
    from ..expression import Expression
    from ..features import FeatureSpace as FS, DEFAULT_CONFIG

    # [修复] 使用局部 FeatureSpace 副本，不修改外部传入的实例
    _base_config = feature_space.config if feature_space else DEFAULT_CONFIG
    feature_space = None  # 释放外部引用

    train_td = _parse_td(train_size)
    test_td = _parse_td(test_size)
    step_td = _parse_td(step)

    windows = []
    train_start = data.index[0]

    wi = 0
    while True:
        train_end = train_start + train_td
        test_end = train_end + test_td
        if test_end > data.index[-1]:
            break

        train_mask = (data.index >= train_start) & (data.index < train_end)
        test_mask = (data.index >= train_end) & (data.index < test_end)
        train_data = data.loc[train_mask]
        test_data = data.loc[test_mask]

        if len(train_data) < min_train_bars or len(test_data) < min_test_bars:
            if anchor_start:
                train_td += step_td
                if train_td > test_td * 6:
                    break
                continue
            train_start += step_td
            continue

        wi += 1
        window = {
            'train_start': str(train_start.date()),
            'train_end': str(train_end.date()),
            'test_start': str(train_end.date()),
            'test_end': str(test_end.date()),
        }

        if verbose:
            print(f"  WF[{wi}] {window['train_start']}→{window['test_end']} "
                  f"train={len(train_data)} test={len(test_data)} ...", end=' ')

        for phase, seg_data in [('train', train_data), ('test', test_data)]:
            try:
                fs = FS(config=_base_config).fit(seg_data)  # 每窗口新实例
                expr = Expression(expression, feature_space=fs)
                seg_sig = expr.generate(seg_data)
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
    """汇总窗口统计 → WalkForwardCoreResult
    [修复] 排除含 error 的失败窗口，不计入均值
    [修复] overfit_ratio 用差值而非比值（避免负 test_SR 时符号混乱）
    """
    valid_windows = [w for w in windows
                     if 'error' not in w.get('train', {})
                     and 'error' not in w.get('test', {})]
    failed_count = len(windows) - len(valid_windows)

    test_sharpes = [w.get('test', {}).get('sharpe', 0) or 0 for w in valid_windows]
    train_sharpes = [w.get('train', {}).get('sharpe', 0) or 0 for w in valid_windows]

    mean_test = float(np.mean(test_sharpes)) if test_sharpes else 0
    mean_train = float(np.mean(train_sharpes)) if train_sharpes else 0

    summary = {
        'total_windows': len(windows),
        'valid_windows': len(valid_windows),
        'failed_windows': failed_count,
        'mean_test_sharpe': mean_test,
        'mean_train_sharpe': mean_train,
        'min_test_sharpe': float(np.min(test_sharpes)) if test_sharpes else 0,
        'max_test_sharpe': float(np.max(test_sharpes)) if test_sharpes else 0,
        'negative_count': sum(1 for s in test_sharpes if s < 0),
        'positive_count': sum(1 for s in test_sharpes if s > 0),
        'stability_score': float(mean_test / np.std(test_sharpes, ddof=1))
                           if len(test_sharpes) > 1 and np.std(test_sharpes, ddof=1) > 0 else 0,
        'overfit_ratio': float(mean_train - mean_test),
    }

    return WalkForwardCoreResult(windows=windows, summary=summary)
