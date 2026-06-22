# [重构] 2026-06-22 继承 utils.ast.AstExpression 公共基类
"""
signals/v4/expression.py — 择时信号表达式 (继承 AstExpression)
=============================================================================

Expression 是 AstExpression 的择时子类，在基类的解析+自省能力之上，
增加单品种 DataFrame → 1D 信号序列的求值能力。

职责:
  - generate(ohlcv_df)         → 单品种择时信号 pd.Series
  - evaluate_panel({code: df}) → 多品种逐列求值 pd.DataFrame
  - rank_panel({code: df})     → 多品种截面排名 pd.DataFrame

与 factor/v4/FactorExpression 的关系:
  - 共同基类: utils.ast.AstExpression (parse→variables→functions→complexity)
  - 差异: Expression 输入 pd.DataFrame(含OHLCV列), 输出 pd.Series
          FactorExpression 输入 Dict[str, ndarray], 输出 ndarray(T,N)

用法:
  >>> from signals.v4 import Expression
  >>> expr = Expression("(CLOSE / ts_mean(CLOSE, 50) - 1) * 100")
  >>> signal = expr.generate(ohlcv_df)  # -> pd.Series
  >>> from signals.v4 import EngineCore
  >>> result = EngineCore.backtest(signal, ohlcv_df, mode='fast')
=============================================================================
"""
import pandas as pd
import numpy as np
from typing import Dict, List

from utils.ast.expr_base import AstExpression
from utils.ast.dsl import evaluate
from utils.ast import CsResolver, normalize_data_keys


class Expression(AstExpression):
    """择时信号表达式 — 继承 AstExpression + 单品种求值

    generate() 将 DataFrame 列映射为 ALL_CAPS 变量名，
    逐日求值后返回连续信号序列 (值 >0 做多)。

    Attributes (继承自 AstExpression):
        expr_str, name, _tree, variables, functions, complexity

    Example:
        >>> expr = Expression("rsi(CLOSE, 14) < -0.3 and ts_roc(CLOSE, 5) > 0")
        >>> signal = expr.generate(ohlcv_df)
        >>> print(signal.describe())
    """

    def generate(self, data: pd.DataFrame,
                 extra_features: Dict[str, np.ndarray] = None) -> pd.Series:
        """
        从 OHLCV DataFrame 生成信号序列（单品种择时）

        Args:
            data: OHLCV DataFrame (index=DatetimeIndex, 含 open/high/low/close/volume)
            extra_features: 额外预计算特征 {特征名: np.ndarray}

        Returns:
            pd.Series, index 对齐 data，值 >0 做多
        """
        data_dict = _build_data_dict(data, self.variables, extra_features)
        result = evaluate(self._tree, data_dict)
        if result.size == 1:
            result = np.full(len(data), result.item())
        result = result.flatten()
        # 对齐索引：若 result 短于 data，对齐到 data 尾部
        if len(result) < len(data):
            offset = len(data) - len(result)
            return pd.Series(result, index=data.index[offset:], name=self.name)
        return pd.Series(result[:len(data)],
                        index=data.index[:len(result)], name=self.name)

    def evaluate_panel(self, assets: Dict[str, pd.DataFrame],
                       align: str = 'inner') -> pd.DataFrame:
        """
        多品种批量求值（因子轮动用）

        Args:
            assets: {品种代码: OHLCV DataFrame}
            align: 'inner'=取公共日期, 'outer'=取并集（NaN填充）

        Returns:
            DataFrame(index=日期, columns=品种代码)，值为连续因子值
            每列是该品种在同一表达式下的求值结果
        """
        results = {}
        for code, df in assets.items():
            try:
                results[code] = self.generate(df)
            except Exception as e:
                import warnings
                warnings.warn(f"品种 {code} 求值失败: {e}")

        if not results:
            raise ValueError("所有品种求值均失败")

        panel = pd.DataFrame(results)

        # 日期对齐
        if align == 'inner':
            common_index = panel.index
            for col in panel.columns:
                common_index = common_index.intersection(panel[col].dropna().index)
            panel = panel.loc[common_index]

        return panel

    def rank_panel(self, assets: Dict[str, pd.DataFrame],
                   align: str = 'inner') -> pd.DataFrame:
        """
        多品种批量求值 + 截面排名（一步到位）

        含 cs_* 截面函数时 → CsResolver (需完整2D面板)
        无截面函数时 → 逐品种求值 + 每日 scipy 排名

        Returns:
            DataFrame(index=日期, columns=品种代码)，值为 0~1 截面百分位排名
        """
        # 含截面函数 → 委托 CsResolver (需要完整2D面板)
        if self._has_cs:
            return self._rank_panel_via_csresolver(assets, align)

        # 无截面函数 → 原有逻辑 (逐品种求值安全)
        panel = self.evaluate_panel(assets, align=align)
        # 每日截面排名
        from scipy.stats import rankdata
        ranked = panel.copy()
        for i in range(len(panel)):
            row = panel.iloc[i].values
            valid = ~np.isnan(row)
            if valid.sum() > 0:
                rk = rankdata(row[valid]) / valid.sum()
                ranked.iloc[i, valid] = rk
        return ranked

    def _rank_panel_via_csresolver(self, assets: Dict[str, pd.DataFrame],
                                    align: str = 'inner') -> pd.DataFrame:
        """委托 CsResolver 正确处理截面函数 (需要完整2D面板)

        [重构] 2026-06-22 直接使用 utils.ast.CsResolver,
        不再经过 factor.v4.FactorExpression 中转 (消除跨模块循环依赖)。
        """
        symbols = list(assets.keys())

        # 收集所有品种的日期交集 → 确保面板对齐
        common_dates = assets[symbols[0]].index
        for sym in symbols[1:]:
            common_dates = common_dates.intersection(assets[sym].index)
        dates = sorted(common_dates)

        # 为每个需要的变量构建 (T,N) ndarray
        panel_data = {}
        for var in self.variables:
            var_upper = var.upper()
            mat = np.full((len(dates), len(symbols)), np.nan)
            for j, sym in enumerate(symbols):
                df = assets[sym]
                col_map = {c.upper().strip(): c for c in df.columns}
                col_name = col_map.get(var_upper, col_map.get(var_upper.lower()))
                if col_name:
                    mat[:, j] = df[col_name].reindex(dates).values.astype(float)
            panel_data[var_upper] = mat

        # 直接使用 CsResolver (单遍 bottom-up 处理嵌套/组合截面函数)
        ranked = CsResolver().resolve(self._tree, panel_data)

        return pd.DataFrame(ranked, index=dates, columns=symbols)


def stateful_signal(data: pd.DataFrame, buy_expr: str, sell_expr: str,
                    max_hold: int = None,
                    extra_features: Dict[str, np.ndarray] = None) -> pd.Series:
    """状态机信号: 空仓等BUY→持仓→SELL/超时→空仓

    将非对称 BUY/SELL 表达式转换为单一持仓信号序列，
    适用于择时策略的 BUY/SELL 分离设计。

    [新增] 2026-06-22 从 AI_zeshi 模板提取到 v4 公共 API。

    Args:
        data: OHLCV DataFrame
        buy_expr: 买入触发表达式（值 >0 触发建仓）
        sell_expr: 卖出触发表达式（值 >0 触发平仓），可为 None
        max_hold: 最长持仓天数，None = 不超时
        extra_features: 注入 Expression.generate() 的额外特征

    Returns:
        pd.Series, index 对齐 data, 值 {0, 1}
    """
    buy_sig = Expression(buy_expr).generate(data, extra_features=extra_features)
    sell_sig = None
    if sell_expr is not None:
        sell_sig = Expression(sell_expr).generate(data, extra_features=extra_features)

    pos, hold = 0, 0
    sig = np.zeros(len(data), dtype=int)
    for i in range(len(data)):
        if pos == 0 and buy_sig.iloc[i] > 0:
            sig[i] = 1; pos = 1; hold = 1
        elif pos == 1:
            hold += 1
            sell_trigger = (sell_sig.iloc[i] > 0) if sell_sig is not None else False
            timeout = (max_hold is not None and hold >= max_hold)
            if sell_trigger or timeout:
                sig[i] = 0; pos = 0; hold = 0
            else:
                sig[i] = 1
        # pos == 0: sig[i] stays 0

    return pd.Series(sig, index=data.index, name=f"STF(B={buy_expr[:30]}, S={sell_expr[:20] if sell_expr else 'None'})")


def _build_data_dict(data: pd.DataFrame,
                     required_vars: List[str],
                     extra_features: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
    """从 OHLCV DataFrame 构建求值数据字典

    列名大小写不敏感，自动映射 open/high/low/close/volume/amount。
    支持衍生字段 RETURNS/VWAP 的自动计算。
    """
    data_dict = {}
    n = len(data)

    # 列名映射（大小写不敏感）
    col_map = {c.upper().strip(): c for c in data.columns}

    # 标准 OHLCV
    for std, alt in [('CLOSE','close'), ('OPEN','open'), ('HIGH','high'),
                      ('LOW','low'), ('VOLUME','volume'), ('AMOUNT','amount')]:
        key = std if std in col_map else (alt if alt in col_map else None)
        if key:
            data_dict[std] = data[col_map[key]].values.astype(float)

    # 衍生字段
    for var in required_vars:
        u = var.upper()
        if u in ('RETURNS', 'RET') and 'CLOSE' in data_dict:
            c = data_dict['CLOSE']
            r = np.full(n, np.nan); r[1:] = c[1:] / c[:-1] - 1
            data_dict['RETURNS'] = r; data_dict['RET'] = r
        elif u == 'VWAP' and 'AMOUNT' in data_dict and 'VOLUME' in data_dict:
            v = data_dict['VOLUME']
            data_dict['VWAP'] = np.where(v > 0, data_dict['AMOUNT'] / v, data_dict['CLOSE'])

    # 注入额外特征
    if extra_features:
        for name, arr in extra_features.items():
            data_dict[name.upper()] = np.asarray(arr, dtype=float)

    return data_dict
