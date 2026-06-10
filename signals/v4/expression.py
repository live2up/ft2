"""
signals/v4/expression.py — Expression 类
=============================================================================
包装 Python AST 表达式，提供与 v3 类似的接口。

核心改进（对比 v3）：
  - 基于 Python ast，无需自研 Parser
  - 直接读写 CLOSE/OPEN/HIGH/LOW/VOLUME 等原始 OHLCV
  - 任意特征间算术运算
  - 输出连续值信号（非 0/1）

用法：
  >>> from signals.v4 import Expression
  >>> expr = Expression("(CLOSE / ts_mean(CLOSE, 50) - 1) * 100")
  >>> signal = expr.generate(ohlcv_df)  # -> pd.Series
  >>> result = EngineV3.backtest(signal, ohlcv_df, mode='fast')
=============================================================================
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Set

from .ast_dsl import parse_expression, evaluate, get_variables, get_functions
from .registry import is_valid_variable


class Expression:
    """
    V4 信号表达式（基于 Python AST）
    
    支持：
      - 原始 OHLCV: CLOSE, OPEN, HIGH, LOW, VOLUME, AMOUNT
      - 预计算特征: RSI_14, ATR_7, EMA_20, MACD_12_26_9, ...
      - 时序函数: ts_mean, ts_std, ts_max, ts_min, ts_delta, ts_delay, ts_corr, ...
      - 扩张统计: expanding_mean, expanding_median, expanding_std
      - 数学函数: abs, log, sqrt, sign, tanh, sigmoid, relu, exp
      - 信号确认: persist(expr, n)
      - 任意算术/比较/逻辑: + - * / > < >= <= and or, a if cond else b
    """
    
    def __init__(self, expr_str: str, name: str = None):
        self.expr_str = expr_str.strip()
        self.name = name or expr_str[:60]
        
        # 解析 + 校验
        self._tree = parse_expression(self.expr_str)
        
        # 自省
        self.variables: List[str] = get_variables(self._tree)
        self.functions: List[str] = get_functions(self._tree)
        self.complexity: int = sum(1 for _ in self._tree.body.body  # type: ignore
                                   ) if hasattr(self._tree, 'body') else 1
    
    @property
    def features_used(self) -> List[str]:
        """使用的特征/变量列表"""
        return self.variables
    
    def __repr__(self):
        return f"Expression({self.expr_str!r})"
    
    def _to_str(self) -> str:
        return self.expr_str
    
    def generate(self, data: pd.DataFrame,
                 extra_features: Dict[str, np.ndarray] = None) -> pd.Series:
        """
        从 OHLCV DataFrame 生成信号序列
        
        Args:
            data: OHLCV DataFrame (index=DatetimeIndex, columns 含 open/high/low/close/volume)
            extra_features: 额外预计算特征 {特征名: np.ndarray}
        
        Returns:
            pd.Series, index 与 data 对齐，值为信号强度（>0 做多，<=0 空仓）
        """
        # 构建数据字典
        data_dict = _build_data_dict(data, self.variables, extra_features)
        
        # 求值
        result = evaluate(self._tree, data_dict)
        
        # 展平标量 → broadcast
        if result.size == 1:
            result = np.full(len(data), result.item())
        
        # 返回 Series
        return pd.Series(result.flatten()[:len(data)], 
                        index=data.index[:len(result)], 
                        name=self.name)
    
    def generate_cached(self, cached_data: Dict[str, np.ndarray]) -> np.ndarray:
        """从缓存数据字典直接求值（用于批量评估）"""
        return evaluate(self._tree, cached_data)


def _build_data_dict(data: pd.DataFrame,
                     required_vars: List[str],
                     extra_features: Dict[str, np.ndarray] = None) -> Dict[str, np.ndarray]:
    """
    从 OHLCV DataFrame 构建求值所需的数据字典
    
    包含：
      1. 原始 OHLCV 列（自动映射大小写）
      2. 动态计算所需特征（按需计算，不预计算全部）
      3. extra_features 中已有的特征
    """
    data_dict = {}
    n = len(data)
    
    # 检测列名（大小写不敏感）
    col_map = {}
    for col in data.columns:
        col_map[col.upper()] = col
    
    # 标准 OHLCV 映射
    ohlcv_std = {
        'CLOSE': 'close', 'OPEN': 'open', 'HIGH': 'high',
        'LOW': 'low', 'VOLUME': 'volume', 'AMOUNT': 'amount',
    }
    for std_name, default_col in ohlcv_std.items():
        if std_name in col_map:
            data_dict[std_name] = data[col_map[std_name]].values.astype(float)
        elif default_col in col_map:
            data_dict[std_name] = data[col_map[default_col]].values.astype(float)
    
    # 衍生字段
    if 'RETURNS' in required_vars or 'RET' in required_vars:
        if 'CLOSE' in data_dict:
            ret = np.full(n, np.nan)
            ret[1:] = data_dict['CLOSE'][1:] / data_dict['CLOSE'][:-1] - 1
            data_dict['RETURNS'] = ret
            data_dict['RET'] = ret
    
    if 'VWAP' in required_vars:
        if 'AMOUNT' in data_dict and 'VOLUME' in data_dict:
            vol = data_dict['VOLUME']
            data_dict['VWAP'] = np.where(vol > 0, data_dict['AMOUNT'] / vol, data_dict['CLOSE'])
    
    # 需要计算的特征
    need_compute = set()
    for var in required_vars:
        upper = var.upper()
        if upper not in data_dict and upper not in ('RETURNS', 'RET'):
            need_compute.add(upper)
    
    # 按需计算特征
    if need_compute:
        _compute_features(data_dict, need_compute, col_map, data, n)
    
    # 注入额外特征
    if extra_features:
        for name, arr in extra_features.items():
            data_dict[name.upper()] = np.asarray(arr, dtype=float)
    
    return data_dict


def _compute_features(data_dict: Dict[str, np.ndarray],
                      needed: Set[str],
                      col_map: Dict[str, str],
                      data: pd.DataFrame,
                      n: int):
    """按需计算特征（惰性，不预计算全部 37 列）"""
    from .registry import FUNC_REGISTRY
    from pandas import Series
    
    for feat_name in sorted(needed):
        if feat_name in data_dict:
            continue
        
        # 解析特征名: RSI_14 → (RSI, [14]), VOL_RATIO_5_20 → (VOL_RATIO, [5, 20])
        parts = feat_name.split('_')
        
        # 尝试从 FeatureSpace 计算函数注册表导入
        try:
            # 尝试多段参数: MACD_12_26_9, VOL_RATIO_5_20
            for i in range(len(parts), 1, -1):
                func_key = '_'.join(parts[:i])
                param_str = '_'.join(parts[i:])
                if func_key in FUNC_REGISTRY:
                    params = []
                    for p in param_str.split('_'):
                        if p.isdigit():
                            params.append(int(p))
                        elif p.replace('.', '').isdigit():
                            params.append(float(p))
                    if params:
                        result = FUNC_REGISTRY[func_key](
                            *[Series(data_dict.get(c, np.zeros(n))) for c in ['CLOSE', 'OPEN', 'HIGH', 'LOW', 'VOLUME']]
                        )
                        continue
            
            # 尝试单段特征名: RSI_14 → func_key=RSI, params=[14]
            func_key = parts[0]
            if func_key in FUNC_REGISTRY and len(parts) > 1:
                params = []
                for p in parts[1:]:
                    if p.isdigit():
                        params.append(int(p))
                    elif p.replace('.', '').isdigit():
                        params.append(float(p))
                if params:
                    result = FUNC_REGISTRY[func_key](*params)
                    continue
        
        except Exception:
            pass
        
        # 无法计算 → 保持 NaN
        data_dict[feat_name] = np.full(n, np.nan)
