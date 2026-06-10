"""
signals/v4/expression.py — V4 Expression 类（基于 Python AST）
=============================================================================
用法：
  >>> from signals.v4 import Expression
  >>> expr = Expression("(CLOSE / ts_mean(CLOSE, 50) - 1) * 100")
  >>> signal = expr.generate(ohlcv_df)  # -> pd.Series
  >>> from signals.v4 import EngineCore
  >>> result = EngineCore.backtest(signal, ohlcv_df, mode='fast')
=============================================================================
"""
import ast
import pandas as pd
import numpy as np
from typing import Dict, Optional, List

from .ast_dsl import parse_expression, evaluate, get_variables, get_functions


class Expression:
    """
    V4 信号表达式（基于 Python AST）
    
    支持：
      - 原始 OHLCV: CLOSE, OPEN, HIGH, LOW, VOLUME, AMOUNT
      - 预计算特征: RSI_14, ATR_7, EMA_20, MACD_12_26_9, ...（需传入 extra_features）
      - 时序函数: ts_mean, ts_std, ts_max, ts_min, ts_delta, ts_delay, ...
      - 扩张统计: expanding_mean, expanding_median, expanding_std
      - 数学函数: abs, log, sqrt, sign, tanh, sigmoid, relu, exp
      - 信号确认: persist(expr, n)
      - 任意算术/比较/逻辑: + - * / > < >= <= and or, a if cond else b
    """
    
    def __init__(self, expr_str: str, name: str = None):
        self.expr_str = expr_str.strip()
        self.name = name or expr_str[:60]
        self._tree = parse_expression(self.expr_str)
        self.variables = get_variables(self._tree)
        self.functions = get_functions(self._tree)
        self.complexity = sum(1 for _ in ast.walk(self._tree.body))
    
    @property
    def features_used(self) -> List[str]:
        return self.variables
    
    def __repr__(self):
        return f"Expression({self.expr_str!r})"
    
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
        return pd.Series(result.flatten()[:len(data)],
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
        
        用法:
            expr = Expression("(CLOSE / ts_mean(CLOSE, 50) - 1) * 100")
            panel = expr.evaluate_panel({'399967': df1, '399970': df2, ...})
            # → 可用于截面排名选 Top N
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
        
        Returns:
            DataFrame(index=日期, columns=品种代码)，值为 0~1 截面百分位排名
            每日每品种的排名值（1=最强，0=最弱）
        """
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


def _build_data_dict(data: pd.DataFrame,
                     required_vars: List[str],
                     extra_features: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
    """从 OHLCV DataFrame 构建求值数据字典"""
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
