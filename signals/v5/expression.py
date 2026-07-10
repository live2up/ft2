"""
signals/v5/expression.py — 择时信号表达式 (直连 utils.ast.v2)

[v5] 2026-07-10 从 v4 迁移，移除兼容重导出，直接导入 ast.v2。
"""
import ast
import pandas as pd
import numpy as np
from typing import Dict, List

from utils.ast.v2.spec import AstExpression
from utils.ast.v2.dsl import evaluate
from utils.ast.v2 import CsResolver, normalize_data_keys


class Expression(AstExpression):
    """择时信号表达式 — 继承 AstExpression + 单品种求值

    generate() 将 DataFrame 列映射为 ALL_CAPS 变量名，
    逐日求值后返回连续信号序列 (值 >0 做多)。
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

        Returns:
            DataFrame(index=日期, columns=品种代码)，值为 0~1 截面百分位排名
        """
        if self._has_cs:
            return self._rank_panel_via_csresolver(assets, align)

        panel = self.evaluate_panel(assets, align=align)
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
        """委托 CsResolver 正确处理截面函数 (需要完整2D面板)"""
        symbols = list(assets.keys())

        common_dates = assets[symbols[0]].index
        for sym in symbols[1:]:
            common_dates = common_dates.intersection(assets[sym].index)
        dates = sorted(common_dates)

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

        ranked = CsResolver().resolve(self._tree, panel_data)

        return pd.DataFrame(ranked, index=dates, columns=symbols)


def stateful_signal(data: pd.DataFrame, buy_expr: str, sell_expr: str,
                    max_hold: int = None,
                    extra_features: Dict[str, np.ndarray] = None) -> pd.Series:
    """状态机信号: 空仓等BUY→持仓→SELL/超时→空仓

    [v5] 2026-07-10 从 v4 原样迁移，逻辑不变。
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

    return pd.Series(sig, index=data.index, name=f"STF(B={buy_expr[:30]}, S={sell_expr[:20] if sell_expr else 'None'})")


def _build_data_dict(data: pd.DataFrame,
                     required_vars: List[str] = None,
                     extra_features: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
    """从 OHLCV DataFrame 构建求值数据字典

    [v5] 2026-07-10 从 v4 原样迁移。
    
    Args:
        data: OHLCV DataFrame
        required_vars: 需要的变量列表。None 表示包含所有标准变量和派生变量
        extra_features: 额外预计算特征
    """
    data_dict = {}
    n = len(data)

    col_map = {c.upper().strip(): c for c in data.columns}

    for std, alt in [('CLOSE','close'), ('OPEN','open'), ('HIGH','high'),
                      ('LOW','low'), ('VOLUME','volume'), ('AMOUNT','amount')]:
        key = std if std in col_map else (alt if alt in col_map else None)
        if key:
            data_dict[std] = data[col_map[key]].values.astype(float)

    # 派生变量：required_vars=None 时生成所有，否则按需生成
    if required_vars is None:
        if 'CLOSE' in data_dict:
            c = data_dict['CLOSE']
            r = np.full(n, np.nan); r[1:] = c[1:] / c[:-1] - 1
            data_dict['RETURNS'] = r; data_dict['RET'] = r
        if 'AMOUNT' in data_dict and 'VOLUME' in data_dict:
            v = data_dict['VOLUME']
            data_dict['VWAP'] = np.where(v > 0, data_dict['AMOUNT'] / v, data_dict['CLOSE'])
    else:
        for var in required_vars:
            u = var.upper()
            if u in ('RETURNS', 'RET') and 'CLOSE' in data_dict:
                c = data_dict['CLOSE']
                r = np.full(n, np.nan); r[1:] = c[1:] / c[:-1] - 1
                data_dict['RETURNS'] = r; data_dict['RET'] = r
            elif u == 'VWAP' and 'AMOUNT' in data_dict and 'VOLUME' in data_dict:
                v = data_dict['VOLUME']
                data_dict['VWAP'] = np.where(v > 0, data_dict['AMOUNT'] / v, data_dict['CLOSE'])

    if extra_features:
        for name, arr in extra_features.items():
            data_dict[name.upper()] = np.asarray(arr, dtype=float)

    return data_dict


# ============================================================
# GP 桥接层 (utils/gp/v5 专用)
# ============================================================

class _SignalFromAST:
    """轻量 AST 包装器: 接收已有 AST 树直接求值为 1D 信号序列

    使用场景: GP 引擎在内部生成/变异 ast.Expression 对象，
    直接传入此包装器求值，避免反复 parse 字符串。

    与 _ExpressionFromAST (factor/v5) 的区别:
      - factor 端: 输出 2D 面板，保留 NaN 用于截面排名
      - signal 端: 输出 1D 序列，NaN → 0 表示无信号
    """

    def __init__(self, tree: ast.Expression, name: str = ''):
        self._tree = tree
        self.name = name

    def evaluate(self, data: Dict[str, np.ndarray]) -> np.ndarray:
        """直接求值 AST 树 → 1D 信号序列

        Args:
            data: {变量名: np.ndarray}，由 _build_gp_data_dict 构建

        Returns:
            np.ndarray, 1D 信号序列，NaN → 0
        """
        data_norm = normalize_data_keys(data)
        result = evaluate(self._tree, data_norm)
        result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)
        return result.flatten()


def _build_gp_data_dict(data: pd.DataFrame,
                        extra_features: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
    """从 OHLCV DataFrame 构建 GP 求值数据字典

    提取所有标准变量 + 常用派生变量 (RETURNS, RET, VWAP)。

    Args:
        data: OHLCV DataFrame (index=DatetimeIndex)
        extra_features: 额外预计算特征 {特征名: np.ndarray}

    Returns:
        Dict[str, np.ndarray]，可直接传入 gp/v5 evaluator
    """
    return _build_data_dict(data, required_vars=None, extra_features=extra_features)
