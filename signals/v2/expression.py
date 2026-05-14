"""
signals/v2/expression.py — 统一表达式引擎
=============================================================================

核心抽象：Expression 是信号的一等公民。
传统指标和 GP 表达式用同一套语言描述，编译为可执行的信号生成器。


============================================================================
                         架构层级（竖式）
============================================================================

 ┌─────────────────────────────────────────────────────────────────────┐
 │  输入：表达式字符串                                                   │
 │  "thr_mean(ATR(7)) & thr_mean(TRIMA(60))"                           │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Tokenizer（词法分析）                                                │
 │                                                                      │
 │  "thr_mean(ATR(7))" → [NAME("thr_mean"), LPAREN,                     │
 │                        NAME("ATR"), LPAREN, NUMBER("7"), RPAREN, RPAREN] │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Parser（递归下降语法分析）→ AST 树                                   │
 │                                                                      │
 │  THR_MEAN                                                           │
 │    └── FEATURE("ATR", [7])  ─→ 查找 DataFrame 列: ATR{7}            │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  AST 节点类型（7 种）                                                 │
 │                                                                      │
 │  ┌──────────────┬──────────────────────────────────────────────┐    │
 │  │   节点类型    │    说明 / 语法                                │    │
 │  ├──────────────┼──────────────────────────────────────────────┤    │
 │  │ FEATURE      │ 特征引用    ATR{7} → 查找列 ATR{7}             │    │
 │  │ CONSTANT     │ 常量        -1.0, 0.5, 2.0                     │    │
 │  │ OPERATOR     │ 二元运算    +, -, *, /, &, |, >, <            │    │
 │  │ FUNCTION     │ 一元函数    abs, sqrt, sigmoid, tanh, relu    │    │
 │  │ THRESHOLD    │ 阈值化      thr_0 / thr_mean / thr_med        │    │
 │  │ PERSIST      │ 信号确认    persist(expr, n) 连续N日          │    │
 │  │ SWITCH       │ 条件分支    if_then(cond, a, b)               │    │
 │  └──────────────┴──────────────────────────────────────────────┘    │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  TreeNode.evaluate(feature_data) → 信号序列 (np.ndarray)             │
 │                                                                      │
 │  递归求值：叶子节点取特征值 → 逐层向上计算 → 输出信号数组              │
└────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Expression.generate(data) → 最终信号 (pd.Series)                    │
 │                                                                      │
 │  1. 调用 FeatureSpace.fit_transform(data) → 获取特征矩阵             │
 │  2. 调用 tree.evaluate(feature_data) → 生成信号数组                  │
 │  3. 返回 pd.Series(signals, index=data.index)                        │
 └─────────────────────────────────────────────────────────────────────┘


============================================================================
                         参数规范速查
============================================================================

 ┌──────────────────────────┬───────────────────────────────────────┐
 │       语法元素            │       说明 / 示例                      │
 ├──────────────────────────┼───────────────────────────────────────┤
 │ 特征引用 ({} 列引用)      │ ATR{7}  → 列名 ATR{7}                 │
 │                          │ MACD{12,26,9} → 列名 MACD{12,26,9}    │
 │                          │                                       │
 │                          │ 花括号 {} 专门用于"引用特征数据列"，    │
 │                          │ 圆括号 () 专门用于"函数/变换调用"。     │
 │                          │ 两者分工明确，表达式所见即列名。        │
 │                          │                                       │
 │                          │ 旧 () 语法仍向后兼容：                 │
 │                          │ RSI(14) 与 RSI{14} 产生相同 AST。       │
 │                          │ 推荐使用 {} 写法保持语义一致性。        │
 │                          │                                       │
 │                          │ TreeNode._feature_key() 转换:           │
 │                          │ FEATURE("ATR",[7]) → "ATR{7}"          │
 │                          │                                       │
 │                          │ 与 Config 层对照:                      │
 │                          │ Config.features: "ATR(period=[7,14])"  │
 │                          │ Expression:  "thr_mean(ATR{7})"        │
 │                          │                                        │
 ├──────────────────────────┼───────────────────────────────────────┤
 │ 阈值函数                  │ thr_0(x)    : x > 0 → 1.0 / 0.0      │
│                          │ thr_50(x)   : x > 50 → 1.0 / 0.0     │
│                          │ thr_1(x)    : x > 1 → 1.0 / 0.0       │
│                          │ thr_0.5(x)  : x > 0.5 → 1.0 / 0.0     │
│                          │ 任意 thr_<数字>(x) 均可用              │
│                          │ thr_mean(x) : x > mean(x) → 1.0 / 0.0 │
│                          │ thr_med(x)  : x > median(x) → 1.0/0.0 │
│                          │ thr_roll_mean(x, w) : x > rolling_mean │
│                          │ thr_roll_med(x, w)  : x > rolling_med  │
│                          │ thr_zscore(x, w, k) : 布林带式突破     │
│                          │ thr_pct(x, p) : x > 历史p分位数        │
│                          │ thr_range(x, lo, hi) : 区间过滤        │
 ├──────────────────────────┼───────────────────────────────────────┤
 │ 二元运算                  │ expr1 & expr2 : AND (np.minimum)      │
 │                          │ expr1 | expr2 : OR  (np.maximum)       │
 │                          │ expr1 sub expr2 : 减法                  │
 ├──────────────────────────┼───────────────────────────────────────┤
 │ 信号确认                  │ persist(expr, 3) : 连续3日同向才触发  │
 ├──────────────────────────┼───────────────────────────────────────┤
 │ 运算符重载 (Python 语法)  │ expr1 & expr2   : AND 组合             │
 │                          │ expr1 | expr2   : OR 组合              │
 │                          │ -expr           : REVERSE 反转          │
 │                          │ expr1 + expr2   : 加权求和              │
 └──────────────────────────┴───────────────────────────────────────┘


============================================================================
                         架构原则（来自 AIdev 探索结论）
============================================================================

 1. AST 是信号的正确表达形式 — GP V4/V5 已验证
 2. 安全计算内置 — div 防零、sqrt 防负、sigmoid 防溢出
 3. 自省能力 — complexity / features_used / depth
 4. 序列化 — Expression 可 save/load JSON，可持久化复用
 5. 运算符重载 — &(AND) / |(OR) / -(REVERSE) / +(加权)

============================================================================

语法示例（推荐 {} 列引用语法）：
    简单:    "RSI{14}"
    变换:    "thr_mean(RSI{14})"
    阈值:    "thr_roll_mean(BBWIDTH{30}, 60)"
    二元:    "thr_mean(BBWIDTH{30}) sub thr_mean(TSF{7})"
    条件:    "if_then(ADX{14} > 25, trend_expr, range_expr)"
    确认:    "persist(thr_mean(BBWIDTH{30}), 3)"
    组合:    "thr_mean(ATR{7}) & thr_mean(TRIMA{60})"

============================================================================
"""

import sys
import os
import json
import re
import numpy as np
import pandas as pd
from enum import Enum
from typing import Dict, List, Optional, Tuple, Set, Callable, Any, Union

from .features import FeatureSpace, _rolling_mean, NORMALIZE_MAP


# ============================================================
# AST 节点定义
# ============================================================

class NodeType(Enum):
    FEATURE = 'feature'
    CONSTANT = 'constant'
    OPERATOR = 'operator'
    FUNCTION = 'function'
    THRESHOLD = 'threshold'
    PERSIST = 'persist'
    SWITCH = 'switch'


class TreeNode:
    """表达式树节点 — 移植自AIdev GP V5"""

    BINARY_OPS = {
        'add': np.add,
        'sub': np.subtract,
        'mul': np.multiply,
        'div': lambda x, y: np.where(np.abs(y) > 1e-10, x / y, 0.0),
        'max': np.maximum,
        'min': np.minimum,
        '>': lambda x, y: np.where(x > y, 1.0, 0.0),
        '<': lambda x, y: np.where(x < y, 1.0, 0.0),
        '&': lambda x, y: np.minimum(x, y),
        '|': lambda x, y: np.maximum(x, y),
    }

    UNARY_FUNCS = {
        'abs': np.abs,
        'sign': np.sign,
        'sqrt': lambda x: np.sqrt(np.abs(x)),
        'square': np.square,
        'neg': lambda x: -x,
        'relu': lambda x: np.maximum(x, 0),
        'tanh': np.tanh,
        'sigmoid': lambda x: 1 / (1 + np.exp(-np.clip(x, -500, 500))),
        'log': lambda x: np.log(np.maximum(np.abs(x), 1e-10)),
        'exp': lambda x: np.exp(np.clip(x, -50, 50)),
    }

    # [修复] thr_mean/thr_med 不再在此处用全序列均值实现，改为 expanding 方式(无前瞻偏差)
    # 旧实现 'thr_mean': lambda x: np.where(x > np.nanmean(x), 1.0, 0.0) 用了全序列均值含未来数据
    # 新实现移到 evaluate() 中，调用 _expanding_mean / _expanding_median
    # thr_0 保持不变：指标>0做多，对中心化后的RSI/MFI等含义是"高于中性线"
    THRESHOLD_FUNCS = {
        'thr_0': lambda x: np.where(x > 0, 1.0, 0.0),
        'thr_mean': None,
        'thr_med': None,
    }

    def __init__(self, node_type: NodeType, value: str,
                 children: Optional[List['TreeNode']] = None,
                 param: Optional[Any] = None):
        """AST 树节点

        Args:
            node_type: 节点类型 (FEATURE / CONSTANT / OPERATOR / ...)
            value: 节点值（特征名/运算符/函数名/常量字符串）
            children: 子树列表（OPERATOR 有左右子树，THRESHOLD 有入参子树）
            param: 额外参数（FEATURE 的参数列表、THRESHOLD 的窗口大小等）
        """
        self.node_type = node_type
        self.value = value
        self.children = children if children else []
        self.param = param

    def _feature_key(self) -> str:
        """FEATURE节点: 从 value + params 构造 DataFrame 列名

        FEATURE("RSI", [14]) → "RSI{14}"
        FEATURE("MACD", [12, 26, 9]) → "MACD{12,26,9}"
        FEATURE("ULTOSC") → "ULTOSC"
        """
        if self.param is not None:
            if isinstance(self.param, list):
                return f"{self.value}{{{','.join(str(p) for p in self.param)}}}"
            return f"{self.value}{{{self.param}}}"
        return self.value

    def evaluate(self, feature_data: Dict[str, np.ndarray]) -> np.ndarray:
        if self.node_type == NodeType.FEATURE:
            col_name = self._feature_key()
            # 1) 精确匹配: RSI{14} → 直接命中
            if col_name in feature_data:
                return feature_data[col_name].copy()
            # 2) {} 前缀匹配: VOL_RATIO{10} → 匹配 VOL_RATIO{10,20}
            #    花括号格式去掉末尾 } 加 , 定位额外参数前缀
            #    "RSI{14}" 这种单参数特征精确匹配已命中，不会进此分支
            prefix_matches = []
            if '}' in col_name:
                brace_prefix = col_name.rstrip('}') + ','
                prefix_matches = [k for k in feature_data
                                  if k.startswith(brace_prefix)]
            # 3) 旧 _ 格式向后兼容: 将 {} 转 _ 再试
            if not prefix_matches:
                alt_name = col_name.replace('{', '_').replace('}', '')
                if alt_name in feature_data:
                    return feature_data[alt_name].copy()
                prefix_matches = [k for k in feature_data
                                  if k.startswith(alt_name + '_')
                                  and '_MA' not in k.replace(alt_name + '_', '', 1).split('_sub_')[0]
                                  and '_sub_' not in k]
            if len(prefix_matches) == 1:
                return feature_data[prefix_matches[0]].copy()
            if len(prefix_matches) > 1:
                raise KeyError(
                    f"特征 '{col_name}' 前缀匹配到多列: {prefix_matches}，"
                    f"请使用完整参数指定，如 VOL_RATIO{10,20}"
                )
            # 无匹配 → 报错
            available = [k for k in feature_data if col_name[:3] in k]
            raise KeyError(f"特征列 '{col_name}' 不存在于特征矩阵中 (相近列: {available[:10]})")
        elif self.node_type == NodeType.CONSTANT:
            # 返回单元素数组而非标量，确保 numpy 广播兼容
            return np.array([float(self.value)])
        elif self.node_type == NodeType.OPERATOR:
            if len(self.children) < 2:
                return self.children[0].evaluate(feature_data) if self.children else np.zeros(1)
            left = self.children[0].evaluate(feature_data)
            right = self.children[1].evaluate(feature_data)
            return self.BINARY_OPS[self.value](left, right)
        elif self.node_type == NodeType.FUNCTION:
            arg = self.children[0].evaluate(feature_data) if self.children else np.zeros(1)
            return self.UNARY_FUNCS[self.value](arg)
        elif self.node_type == NodeType.THRESHOLD:
            arg = self.children[0].evaluate(feature_data) if self.children else np.zeros(1)
            if self.value == 'thr_roll_mean':
                w = self.param if self.param is not None else 30
                rolling_val = _rolling_mean(arg, w)
                return np.where(arg > rolling_val, 1.0, 0.0)
            if self.value == 'thr_roll_med':
                w = self.param if self.param is not None else 30
                rolling_val = _rolling_median(arg, w)
                return np.where(arg > rolling_val, 1.0, 0.0)
            if self.value == 'thr_mean':
                expanding_val = _expanding_mean(arg)
                return np.where(arg > expanding_val, 1.0, 0.0)
            if self.value == 'thr_med':
                expanding_val = _expanding_median(arg)
                return np.where(arg > expanding_val, 1.0, 0.0)
            # [新增] thr_zscore: x > rolling_mean(x, w) + k * rolling_std(x, w)，布林带式突破
            # 用法: thr_zscore(RSI(14), 30, 1.0) — RSI突破30日均值+1倍标准差
            if self.value == 'thr_zscore':
                w = self.param.get('window', 30) if isinstance(self.param, dict) else (self.param or 30)
                k = self.param.get('k', 1.0) if isinstance(self.param, dict) else 1.0
                rolling_m = _rolling_mean(arg, w)
                rolling_s = _rolling_std(arg, w)
                return np.where(rolling_s > 1e-10,
                                np.where(arg > rolling_m + k * rolling_s, 1.0, 0.0), 0.0)
            # [新增] thr_pct: x > expanding_percentile(x, p)，历史分位数突破
            # 用法: thr_pct(RSI(14), 0.8) — RSI处于历史前20%
            if self.value == 'thr_pct':
                p = self.param if self.param is not None else 0.5
                pct_val = _expanding_percentile(arg, p)
                return np.where(arg > pct_val, 1.0, 0.0)
            # [新增] thr_range: lo < x < hi，区间过滤
            # 用法: thr_range(RSI(14), -0.3, 0.3) — RSI在[-0.3, 0.3]区间内做多
            if self.value == 'thr_range':
                lo = self.param.get('lo', -1e10) if isinstance(self.param, dict) else -1e10
                hi = self.param.get('hi', 1e10) if isinstance(self.param, dict) else 1e10
                return np.where((arg > lo) & (arg < hi), 1.0, 0.0)
            if self.value in self.THRESHOLD_FUNCS:
                return self.THRESHOLD_FUNCS[self.value](arg)
            # 通用 thr_<数字>: x > 阈值 (如 thr_50(RSI{14}))
            if self.param is not None and isinstance(self.param, (int, float)):
                return np.where(arg > self.param, 1.0, 0.0)
            return np.where(arg > 0, 1.0, 0.0)
        elif self.node_type == NodeType.PERSIST:
            arg = self.children[0].evaluate(feature_data) if self.children else np.zeros(1)
            n = self.param if self.param is not None else 3
            return np_persist(arg, n)
        elif self.node_type == NodeType.SWITCH:
            if len(self.children) < 3:
                return self.children[0].evaluate(feature_data) if self.children else np.zeros(1)
            cond_val = self.children[0].evaluate(feature_data)
            branch_a = self.children[1].evaluate(feature_data)
            branch_b = self.children[2].evaluate(feature_data)
            return np.where(cond_val > 0, branch_a, branch_b)
        raise ValueError(f"未知节点类型: {self.node_type}")

    def get_features(self) -> Set[str]:
        """递归收集 AST 中所有引用到的特征列名

        Returns:
            { "RSI{14}", "ROC{5}", "VOL_RATIO{10,20}", ... }
        """
        feats = set()
        if self.node_type == NodeType.FEATURE:
            feats.add(self._feature_key())
        for child in self.children:
            feats.update(child.get_features())
        return feats

    def depth(self) -> int:
        """AST 树的深度 = 从根到叶子节点的最长路径"""
        if not self.children:
            return 1
        return 1 + max(child.depth() for child in self.children)

    def size(self) -> int:
        """AST 树的节点总数 = 表达式复杂度的衡量"""
        return 1 + sum(child.size() for child in self.children)

    def to_string(self) -> str:
        if self.node_type == NodeType.FEATURE:
            if self.param is not None:
                if isinstance(self.param, list):
                    return f"{self.value}{{{', '.join(str(p) for p in self.param)}}}"
                return f"{self.value}{{{self.param}}}"
            return self.value
        elif self.node_type == NodeType.CONSTANT:
            return self.value
        elif self.node_type == NodeType.OPERATOR:
            left = self.children[0].to_string()
            right = self.children[1].to_string()
            return f"({left} {self.value} {right})"
        elif self.node_type == NodeType.FUNCTION:
            arg = self.children[0].to_string()
            return f"{self.value}({arg})"
        elif self.node_type == NodeType.THRESHOLD:
            arg = self.children[0].to_string()
            if self.value in ('thr_roll_mean', 'thr_roll_med'):
                return f"{self.value}({arg}, {self.param})"
            # [新增] 新阈值函数的序列化
            if self.value == 'thr_zscore':
                w = self.param.get('window', 30) if isinstance(self.param, dict) else 30
                k = self.param.get('k', 1.0) if isinstance(self.param, dict) else 1.0
                return f"thr_zscore({arg}, {w}, {k})"
            if self.value == 'thr_pct':
                return f"thr_pct({arg}, {self.param})"
            if self.value == 'thr_range':
                lo = self.param.get('lo', -1e10) if isinstance(self.param, dict) else -1e10
                hi = self.param.get('hi', 1e10) if isinstance(self.param, dict) else 1e10
                return f"thr_range({arg}, {lo}, {hi})"
            return f"{self.value}({arg})"
        elif self.node_type == NodeType.PERSIST:
            arg = self.children[0].to_string()
            return f"persist({arg}, {self.param})"
        elif self.node_type == NodeType.SWITCH:
            cond = self.children[0].to_string()
            a = self.children[1].to_string()
            b = self.children[2].to_string()
            return f"switch({cond}, {a}, {b})"
        return str(self.value)

    def copy(self) -> 'TreeNode':
        """深度克隆整棵 AST 树（递归拷贝所有子节点）"""
        return TreeNode(
            self.node_type, self.value,
            [child.copy() for child in self.children],
            self.param
        )

    def to_dict(self) -> Dict:
        """序列化为字典（JSON 兼容），用于保存/传输"""
        result = {
            'type': self.node_type.value,
            'value': self.value,
        }
        if self.param is not None:
            result['param'] = self.param
        if self.children:
            result['children'] = [child.to_dict() for child in self.children]
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'TreeNode':
        """从字典反序列化为 AST 树（与 to_dict() 对称）"""
        node_type = NodeType(data['type'])
        children = [cls.from_dict(c) for c in data.get('children', [])]
        param = data.get('param')
        return cls(node_type, data['value'], children, param)


def np_persist(arr: np.ndarray, n: int) -> np.ndarray:
    """信号确认滤波器：连续 N 个信号同向才触发，否则保持中性的 0

    用途：
      消除单日噪声信号的虚假触发。例如 persist(signal, 3) 要求连续3日做多信号才做多，
      连续3日做空才做空，其余情况持仓不变（信号 0）。

    Args:
        arr: 原始信号序列，取值 { -1, 0, 1 }
        n: 最低持续天数

    Returns:
        滤波后信号序列，仅当连续 n 期同向时输出 +1/-1，否则 0
    """
    result = np.zeros_like(arr)
    for i in range(n - 1, len(arr)):
        segment = arr[i - n + 1:i + 1]
        if np.all(segment > 0):
            result[i] = 1.0
        elif np.all(segment < 0):
            result[i] = -1.0
    return result


# [修复] expanding 函数增加 min_warmup 参数，冷启动期间返回 NaN
# 旧实现: 从第0天就开始计算，expanding_mean 样本极少导致信号不稳定
# 新实现: 前 min_warmup 天返回 NaN，由 nan_to_num 转为 0(不持仓)，避免冷启动噪音
_EXPANDING_WARMUP = 20


def _expanding_mean(arr: np.ndarray, min_warmup: int = _EXPANDING_WARMUP) -> np.ndarray:
    """扩展窗口均值：第t天的均值只用到[0:t+1]的数据，无前瞻偏差
    [修复说明] 替代原有的 np.nanmean(x) 全序列均值实现。
    原实现在第t天判断时用了t+1到末尾的未来数据(前瞻偏差)，
    回测虚增收益且实盘不可复现。expanding mean 严格只用历史数据。
    [优化] 向量化实现，去掉 for 循环，cumsum/count 直接相除即可。"""
    cumsum = np.nancumsum(arr)
    count = np.cumsum(~np.isnan(arr))
    result = np.where(count > 0, cumsum / count, np.nan)
    # [修复] 冷启动保护: 前 min_warmup 天设为 NaN，避免样本不足导致信号不稳定
    result[:min(min_warmup, len(result))] = np.nan
    return result


def _expanding_median(arr: np.ndarray, min_warmup: int = _EXPANDING_WARMUP) -> np.ndarray:
    """扩展窗口中位数：第 t 天的中位数只用到 [0:t+1] 的数据，无前瞻偏差

    用途：
      替代 np.nanmedian(x) 全序列中位数（含未来数据），消除前瞻偏差。

    冷启动保护：
      前 min_warmup 天返回 NaN，避免少量样本下中位数不稳定。
    """
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(min_warmup, len(arr)):
        segment = arr[:i + 1]
        valid = segment[~np.isnan(segment)]
        if len(valid) > 0:
            result[i] = np.median(valid)
    return result


def _rolling_median(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动窗口中位数：固定窗口 [t-window+1:t+1]

    用途：
      thr_roll_med 的底层计算。原实现错误地调用了 _rolling_mean（计算均值而非中位数），
      此函数提供了正确的中位数计算。

    与 _expanding_median 的区别：
      - expanding: 窗口从 0 增长到当前（start→current）
      - rolling: 窗口大小固定为 window（current-window+1→current）
    """
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        segment = arr[i - window + 1:i + 1]
        valid = segment[~np.isnan(segment)]
        if len(valid) > 0:
            result[i] = np.median(valid)
    return result


def _rolling_std(arr: np.ndarray, window: int) -> np.ndarray:
    """滚动窗口标准差（样本标准差 ddof=1）

    用途：
      thr_zscore 的底层计算，用于布林带式突破信号：
        signal = x[t] > rolling_mean(x, w) + k * rolling_std(x, w)
    """
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(window - 1, len(arr)):
        segment = arr[i - window + 1:i + 1]
        valid = segment[~np.isnan(segment)]
        if len(valid) > 1:
            result[i] = np.std(valid, ddof=1)
    return result


def _expanding_percentile(arr: np.ndarray, percentile: float,
                           min_warmup: int = _EXPANDING_WARMUP) -> np.ndarray:
    """扩展窗口分位数：第t天的p分位数只用到[0:t+1]的数据，无前瞻偏差
    用于 thr_pct: 判断当前值是否突破历史分位数"""
    result = np.full_like(arr, np.nan, dtype=float)
    for i in range(min_warmup, len(arr)):
        segment = arr[:i + 1]
        valid = segment[~np.isnan(segment)]
        if len(valid) > 0:
            result[i] = np.percentile(valid, percentile * 100)
    return result


# ============================================================
# 表达式解析器
# ============================================================

class Token:
    """词法分析 token 类型枚举（字符串常数，用于类型判断）

    LPAREN/RPAREN: 分组括号 ()
    LBRACE/RBRACE: 列引用花括号 {}      ← {} 语法新增
    COMMA:         参数分隔符
    NAME:          标识符（函数名/特征名/运算符别名）
    NUMBER:        数字字面量（含负数和浮点数）
    OP:            运算符（> < & | + -）
    EOF:           流结束标记
    """
    LPAREN, RPAREN, LBRACE, RBRACE, COMMA, NAME, NUMBER, OP, EOF = (
        'LPAREN', 'RPAREN', 'LBRACE', 'RBRACE', 'COMMA', 'NAME', 'NUMBER', 'OP', 'EOF'
    )

    def __init__(self, type_: str, value: str):
        self.type = type_
        self.value = value

    def __repr__(self):
        return f"Token({self.type}, '{self.value}')"


OPS = {'add', 'sub', 'mul', 'div', 'max', 'min', '>', '<', '&', '|', '+'}


class Tokenizer:
    """词法分析器：表达式字符串 → Token 序列

    支持的语法元素:
      括号:      ( )
      花括号:    { }       ← 列引用语法
      逗号:      ,
      数字:      3.14, -5, 20
      标识符:    RSI, thr_mean, ROC
      运算符:    > < & | + -
    """

    def __init__(self, source: str):
        self.source = source
        self.pos = 0
        self.tokens: List[Token] = []

    def tokenize(self) -> List[Token]:
        while self.pos < len(self.source):
            ch = self.source[self.pos]
            if ch.isspace():
                self.pos += 1
                continue
            if ch == '(':
                self.tokens.append(Token(Token.LPAREN, '('))
                self.pos += 1
            elif ch == ')':
                self.tokens.append(Token(Token.RPAREN, ')'))
                self.pos += 1
            elif ch == ',':
                self.tokens.append(Token(Token.COMMA, ','))
                self.pos += 1
            elif ch == '{':
                self.tokens.append(Token(Token.LBRACE, '{'))
                self.pos += 1
            elif ch == '}':
                self.tokens.append(Token(Token.RBRACE, '}'))
                self.pos += 1
            elif ch.isdigit() or ch == '.' or (ch == '-' and self.pos + 1 < len(self.source) and self.source[self.pos + 1].isdigit()):
                self._number()
            elif ch.isalpha() or ch == '_':
                self._name()
            elif ch in '><+-&|':
                self._op()
            else:
                self.pos += 1
        self.tokens.append(Token(Token.EOF, ''))
        return self.tokens

    def _number(self):
        """扫描数字字面量（含负号和浮点数），生成 NUMBER token"""
        start = self.pos
        if self.source[self.pos] == '-':
            self.pos += 1
        while self.pos < len(self.source) and (self.source[self.pos].isdigit() or self.source[self.pos] == '.'):
            self.pos += 1
        val = self.source[start:self.pos]
        self.tokens.append(Token(Token.NUMBER, val))

    def _name(self):
        """扫描标识符（字母、数字、下划线），生成 NAME token

        额外支持标识符后缀：
          · 小数后缀:  thr_0.5  → 完整标识符 thr_0.5
          · 负数后缀:  thr_-5   → 完整标识符 thr_-5  (仅当名以 _ 结尾时触发，
            避免与 thr_ - 5 这样的减法表达式混淆)

        命名运算符: sub/add/mul/div/min/max 识别为 OP token 而非 NAME
        """
        start = self.pos
        while self.pos < len(self.source) and (self.source[self.pos].isalnum() or self.source[self.pos] == '_'):
            self.pos += 1
        # 后缀扩展: .数字 (thr_0.5)
        if (self.pos + 1 < len(self.source) and self.source[self.pos] == '.'
                and self.source[self.pos + 1].isdigit()):
            self.pos += 1
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self.pos += 1
        # 后缀扩展: -数字 (thr_-5)，仅当名以 _ 结尾（防止 thr - 5 误识别）
        elif (self.pos + 1 < len(self.source) and self.source[self.pos] == '-'
              and self.source[self.pos + 1].isdigit()
              and self.source[start:self.pos].endswith('_')):
            self.pos += 1  # 消耗 '-'
            while self.pos < len(self.source) and self.source[self.pos].isdigit():
                self.pos += 1
            # 再尝试消耗小数部分: thr_-0.5
            if (self.pos + 1 < len(self.source) and self.source[self.pos] == '.'
                    and self.source[self.pos + 1].isdigit()):
                self.pos += 1
                while self.pos < len(self.source) and self.source[self.pos].isdigit():
                    self.pos += 1
        val = self.source[start:self.pos]
        # 命名运算符识别: sub/add/mul/div/min/max 生成 OP token
        if val in OPS:
            self.tokens.append(Token(Token.OP, val))
        else:
            self.tokens.append(Token(Token.NAME, val))

    def _op(self):
        """扫描运算符（支持单字符和双字符），生成 OP token"""
        val = self.source[self.pos]
        if self.pos + 1 < len(self.source):
            two = val + self.source[self.pos + 1]
            if two in OPS:
                self.tokens.append(Token(Token.OP, two))
                self.pos += 2
                return
        self.tokens.append(Token(Token.OP, val))
        self.pos += 1


class Parser:
    """递归下降解析器"""

    def __init__(self, tokens: List[Token], feature_names: Optional[Set[str]] = None):
        self.tokens = tokens
        self.pos = 0
        self.feature_names = feature_names or set()

    def peek(self) -> Token:
        """查看下一个 token 但不消耗它（lookahead）"""
        return self.tokens[self.pos] if self.pos < len(self.tokens) else Token(Token.EOF, '')

    def advance(self) -> Token:
        """消耗并返回当前 token，pos 前进 1"""
        t = self.peek()
        self.pos += 1
        return t

    def match(self, type_: str) -> bool:
        """检查下一个 token 是否匹配指定类型（不消耗）"""
        return self.peek().type == type_

    def expect(self, type_: str, value: Optional[str] = None):
        """期望下一个 token 为指定类型（可选匹配值），否则抛出 SyntaxError"""
        t = self.advance()
        if t.type != type_:
            raise SyntaxError(f"期望 {type_}，得到 {t.type}('{t.value}') at pos {self.pos}")
        if value is not None and t.value != value:
            raise SyntaxError(f"期望 '{value}'，得到 '{t.value}'")

    # [修复] 运算符优先级分层，从平坦左结合改为三级优先级
    # 旧实现: 所有运算符同级左结合, "A > 0 & B > 0" 解析为 ((A > 0) & B) > 0
    # 新实现: >/< (比较) > &/| (逻辑) > +/-/*/÷ (算术), 正确解析为 (A > 0) & (B > 0)
    # 优先级(数值越大越先绑定): comparison(6) > logic(4) > additive(2) > multiplicative(3)
    # 实际解析从低优先级开始，高优先级在更深的递归层先归约
    OP_PRECEDENCE = {
        '|': 1, 'add': 2, 'sub': 2,
        'mul': 3, 'div': 3, 'min': 3, 'max': 3,
        '&': 4,
        '>': 6, '<': 6,
    }

    def parse(self) -> TreeNode:
        tree = self._parse_expr()
        if self.peek().type != Token.EOF:
            pass
        return tree

    def _parse_expr(self) -> TreeNode:
        # [修复] 从最低优先级开始递归下降解析，保证高优先级运算符先绑定
        return self._parse_logic_or()

    def _parse_logic_or(self) -> TreeNode:
        """优先级1: | (逻辑或)"""
        left = self._parse_logic_and()
        while self.peek().type == Token.OP and self._normalize_op(self.peek().value) == '|':
            op_token = self.advance()
            op_str = self._normalize_op(op_token.value)
            right = self._parse_logic_and()
            left = TreeNode(NodeType.OPERATOR, op_str, [left, right])
        return left

    def _parse_logic_and(self) -> TreeNode:
        """优先级4: & (逻辑与)"""
        left = self._parse_comparison()
        while self.peek().type == Token.OP and self._normalize_op(self.peek().value) == '&':
            op_token = self.advance()
            op_str = self._normalize_op(op_token.value)
            right = self._parse_comparison()
            left = TreeNode(NodeType.OPERATOR, op_str, [left, right])
        return left

    def _parse_comparison(self) -> TreeNode:
        """优先级6: > < (比较)"""
        left = self._parse_additive()
        while self.peek().type == Token.OP and self._normalize_op(self.peek().value) in ('>', '<'):
            op_token = self.advance()
            op_str = self._normalize_op(op_token.value)
            right = self._parse_additive()
            left = TreeNode(NodeType.OPERATOR, op_str, [left, right])
        return left

    def _parse_additive(self) -> TreeNode:
        """优先级2: + - (加减)"""
        left = self._parse_multiplicative()
        while self.peek().type == Token.OP and self._normalize_op(self.peek().value) in ('add', 'sub', '+'):
            op_token = self.advance()
            op_str = self._normalize_op(op_token.value)
            right = self._parse_multiplicative()
            left = TreeNode(NodeType.OPERATOR, op_str, [left, right])
        return left

    def _parse_multiplicative(self) -> TreeNode:
        """优先级3: * / min max (乘除)"""
        left = self._parse_primary()
        while self.peek().type == Token.OP and self._normalize_op(self.peek().value) in ('mul', 'div', 'min', 'max'):
            op_token = self.advance()
            op_str = self._normalize_op(op_token.value)
            right = self._parse_primary()
            left = TreeNode(NodeType.OPERATOR, op_str, [left, right])
        return left

    def _normalize_op(self, op_val: str) -> str:
        """统一运算符表示：+ → add，- → sub，&& → &，|| → |"""
        op_map = {'+': 'add', '-': 'sub', '&&': '&', '||': '|'}
        return op_map.get(op_val, op_val)

    def _parse_primary(self) -> TreeNode:
        """解析基本元素：数字 → CONSTANT、标识符 → NAME/CALL/BRAVE_REF、括号 → 子表达式"""
        token = self.peek()
        if token.type == Token.NUMBER:
            return self._parse_number()
        elif token.type == Token.NAME:
            return self._parse_name_or_call()
        elif token.type == Token.LPAREN:
            self.expect(Token.LPAREN)
            inner = self._parse_expr()
            self.expect(Token.RPAREN)
            return inner
        else:
            raise SyntaxError(f"意外的 token: {token}")

    def _parse_number(self) -> TreeNode:
        """解析数字常量 → CONSTANT 节点"""
        t = self.advance()
        return TreeNode(NodeType.CONSTANT, t.value)

    def _parse_name_or_call(self) -> TreeNode:
        name = self.advance().value
        if self.peek().type == Token.LPAREN:
            # 函数调用: name(args...)
            return self._parse_call(name)
        elif self.peek().type == Token.LBRACE:
            # 列引用: NAME{params} → FEATURE 节点
            return self._parse_brace_ref(name)
        # 普通名称: 可能是特征名
        if name in self.feature_names or not self.feature_names:
            return TreeNode(NodeType.FEATURE, name)
        return TreeNode(NodeType.FEATURE, name)

    def _parse_brace_ref(self, name: str) -> TreeNode:
        """解析 NAME{param1,param2,...} → FEATURE 节点

        RSI{14}       → FEATURE("RSI", param=[14])
        MACD{12,26,9} → FEATURE("MACD", param=[12, 26, 9])
        """
        self.expect(Token.LBRACE)
        params = []
        while self.peek().type != Token.RBRACE:
            if self.peek().type == Token.NUMBER:
                params.append(int(float(self.advance().value)))
            elif self.peek().type == Token.COMMA:
                self.advance()
            else:
                break
        self.expect(Token.RBRACE)
        if params:
            return TreeNode(NodeType.FEATURE, name, param=params)
        return TreeNode(NodeType.FEATURE, name)

    def _parse_call(self, func_name: str) -> TreeNode:
        """解析 NAME(args...) 函数调用语法

        支持所有内置函数（阈值、一元运算、信号确认、条件分支）
        以及默认的 FEATURE 构建（参数进入 param）
        """
        self.expect(Token.LPAREN)
        args = []
        while self.peek().type != Token.RPAREN:
            arg_node = self._parse_expr()
            args.append(arg_node)
            if self.peek().type == Token.COMMA:
                self.advance()
            elif self.peek().type == Token.RPAREN:
                break
        self.expect(Token.RPAREN)
        return self._build_call(func_name, args)

    def _build_call(self, func_name: str, args: List[TreeNode]) -> TreeNode:
        """根据函数名构建对应的AST节点"""
        name = func_name.lower()

        # persist(expr, n)
        if name == 'persist':
            n = 3
            if len(args) >= 2 and args[1].node_type == NodeType.CONSTANT:
                n = int(float(args[1].value))
            return TreeNode(NodeType.PERSIST, 'persist', [args[0]], param=n)

        # switch(cond, a, b) — 条件分支
        if name == 'switch' or name == 'if_then':
            if len(args) >= 3:
                return TreeNode(NodeType.SWITCH, 'switch', [args[0], args[1], args[2]])
            return args[0] if args else TreeNode(NodeType.CONSTANT, '0')

        # 通用阈值函数: thr_<数字>(x) — 显式阈值比较
        #   thr_1(EMA{20})   → "EMA/close > 1" = 价格在均线上方
        #   thr_50(RSI{14})  → "中心化RSI > 50" (若RSI未中心化则为原始RSI > 50)
        #   thr_0.5(VOL_RATIO{10,20}) → "量比 > 0.5"
        thr_numeric = re.match(r'^thr_(-?\d+(?:\.\d+)?)$', name)
        if thr_numeric:
            threshold = float(thr_numeric.group(1))
            return TreeNode(NodeType.THRESHOLD, name, [args[0]], param=threshold)

        # 阈值函数: thr_0, thr_mean, thr_med, thr_roll_mean, thr_roll_med
        # [新增] thr_zscore, thr_pct, thr_range
        if name in ('thr_0', 'thr_mean', 'thr_med', 'thr_roll_mean', 'thr_roll_med',
                     'thr_zscore', 'thr_pct', 'thr_range'):
            param = None
            if name in ('thr_roll_mean', 'thr_roll_med') and len(args) >= 2:
                if args[1].node_type == NodeType.CONSTANT:
                    param = int(float(args[1].value))
            # [新增] thr_zscore(x, window, k) — 布林带式突破
            elif name == 'thr_zscore' and len(args) >= 2:
                param = {'window': 30, 'k': 1.0}
                if args[1].node_type == NodeType.CONSTANT:
                    param['window'] = int(float(args[1].value))
                if len(args) >= 3 and args[2].node_type == NodeType.CONSTANT:
                    param['k'] = float(args[2].value)
            # [新增] thr_pct(x, p) — 历史分位数突破
            elif name == 'thr_pct' and len(args) >= 2:
                if args[1].node_type == NodeType.CONSTANT:
                    param = float(args[1].value)
            # [新增] thr_range(x, lo, hi) — 区间过滤
            elif name == 'thr_range' and len(args) >= 3:
                param = {'lo': -1e10, 'hi': 1e10}
                if args[1].node_type == NodeType.CONSTANT:
                    param['lo'] = float(args[1].value)
                if args[2].node_type == NodeType.CONSTANT:
                    param['hi'] = float(args[2].value)
            return TreeNode(NodeType.THRESHOLD, name, [args[0]], param=param)

        # 一元函数
        if name in TreeNode.UNARY_FUNCS:
            return TreeNode(NodeType.FUNCTION, name, [args[0]])

        # 默认为特征函数: RSI(14) → FEATURE("RSI", param=[14])
        # KDJ(9, 3, 3) → FEATURE("KDJ", param=[9, 3, 3])
        feature_params = []
        for a in args:
            if a.node_type == NodeType.CONSTANT:
                feature_params.append(int(float(a.value)))
        if feature_params:
            return TreeNode(NodeType.FEATURE, func_name, param=feature_params)
        return TreeNode(NodeType.FEATURE, func_name)


def parse_expression(expr_str: str, feature_names: Optional[Set[str]] = None) -> TreeNode:
    """解析表达式字符串 → AST树"""
    tokenizer = Tokenizer(expr_str)
    tokens = tokenizer.tokenize()
    parser = Parser(tokens, feature_names)
    return parser.parse()


# ============================================================
# Expression 类 — 信号的一等公民
# ============================================================

class Expression:
    """
    统一表达式信号

    用法:
        # 简单表达式
        expr = Expression("RSI(14)", feature_space=fs)
        signals = expr.generate(data)

        # 复杂表达式（GP结果）
        expr = Expression("thr_mean(BBWIDTH(30)) sub thr_mean(TSF(7))", feature_space=fs)

        # 运算符组合
        expr = Expression("ATR(7)", fs) & Expression("TRIMA(60)", fs)

        # 自省
        print(expr.complexity)    # 节点数
        print(expr.features_used) # ['BBWIDTH_30', 'TSF_7']

        # 序列化
        expr.save("best.json")
        loaded = Expression.load("best.json")
    """

    def __init__(self, source: Union[str, TreeNode],
                 feature_space: Optional[FeatureSpace] = None,
                 feature_df: Optional[pd.DataFrame] = None,
                 name: Optional[str] = None):
        """
        Args:
            source: 表达式字符串 或 TreeNode
            feature_space: FeatureSpace实例（用于自动计算特征）
            feature_df: 预计算的特征矩阵（替代feature_space）
            name: 可选的信号名称
        """
        if isinstance(source, TreeNode):
            self._tree = source
            self.source = source.to_string()
        else:
            self.source = source
            feature_names = set()
            if feature_space:
                feature_names = set(feature_space.get_feature_names())
            elif feature_df is not None:
                feature_names = set(feature_df.columns)
            self._tree = parse_expression(self.source, feature_names)

        self._feature_space = feature_space
        self._feature_df = feature_df
        self._name = name or f"Expr_{self.source[:30]}"
        self._compiled = None

        # 自省属性
        self.complexity = self._tree.size()
        self.depth = self._tree.depth()
        self.features_used = sorted(list(self._tree.get_features()))
        self.expression = self._tree.to_string()

    def generate(self, data: pd.DataFrame) -> pd.Series:
        """生成信号序列"""
        if self._feature_df is not None:
            feature_data = {col: self._feature_df[col].values for col in self._feature_df.columns}
        elif self._feature_space is not None:
            features = self._feature_space.fit_transform(data)
            self._feature_df = features
            feature_data = {col: features[col].values for col in features.columns}
        else:
            raise RuntimeError("需要提供 feature_space 或 feature_df")

        signals = self._tree.evaluate(feature_data)
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)

        index = data.index if self._feature_df is None else self._feature_df.index
        return pd.Series(signals, index=index, name=self._name)

    def generate_from_features(self, feature_df: pd.DataFrame) -> pd.Series:
        """直接从特征DataFrame生成信号（GPU风格求值）"""
        feature_data = {col: feature_df[col].values for col in feature_df.columns}
        signals = self._tree.evaluate(feature_data)
        signals = np.nan_to_num(signals, nan=0.0, posinf=0.0, neginf=0.0)
        return pd.Series(signals, index=feature_df.index, name=self._name)

    # ---- 运算符重载（Python 语法 → AST 组合） ----

    def __and__(self, other: 'Expression') -> 'Expression':
        """& 运算符：AND 信号组合 (np.minimum)"""
        return self._binary_op(other, '&', 'AND')

    def __or__(self, other: 'Expression') -> 'Expression':
        """| 运算符：OR 信号组合 (np.maximum)"""
        return self._binary_op(other, '|', 'OR')

    def __neg__(self) -> 'Expression':
        """- 运算符：信号反转（做多↔做空反转）"""
        tree = TreeNode(NodeType.FUNCTION, 'neg', [self._tree.copy()])
        expr = Expression(tree, self._feature_space, self._feature_df,
                          name=f'REV_{self._name}')
        expr.source = f'neg({self.source})'
        return expr

    def __add__(self, other: 'Expression') -> 'Expression':
        """+ 运算符：两个信号值相加（等权合并）"""
        return self._binary_op(other, 'add', 'ADD')

    def __sub__(self, other: 'Expression') -> 'Expression':
        """- 运算符：信号差值"""
        return self._binary_op(other, 'sub', 'SUB')

    def __mul__(self, other: 'Expression') -> 'Expression':
        """* 运算符：信号相乘（条件与）"""
        return self._binary_op(other, 'mul', 'MUL')

    def _binary_op(self, other: 'Expression', op_name: str, label: str) -> 'Expression':
        """通用二元运算构造器：组合两个 Expression 的 AST 树为新节点

        此方法创建一棵新的 OPERATOR 节点，左右子树分别是 self 和 other 的 AST 拷贝。
        合并时优先使用 self 的 feature_space / feature_df。

        Args:
            other: 参与运算的另一个表达式
            op_name: 内部运算符名称（'&', '|', 'add', 'sub', 'mul'）
            label: 拼接名称用的显示标签（'AND', 'OR', 'ADD', 'SUB', 'MUL'）
        """
        tree = TreeNode(NodeType.OPERATOR, op_name, [self._tree.copy(), other._tree.copy()])
        expr = Expression(tree, self._feature_space or other._feature_space,
                          self._feature_df or other._feature_df,
                          name=f'{self._name}_{label}_{other._name}')
        expr.source = f'({self.source}) {op_name} ({other.source})'
        return expr

    # ---- 序列化 ----

    def to_dict(self) -> Dict:
        """转字典（含表达式字符串、AST 树、自省信息），用于保存/传输"""
        return {
            'source': self.source,
            'expression': self.expression,
            'name': self._name,
            'tree': self._tree.to_dict(),
            'complexity': self.complexity,
            'depth': self.depth,
            'features_used': self.features_used,
        }

    def to_json(self) -> str:
        """转 JSON 字符串"""
        return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: Dict, feature_space: Optional[FeatureSpace] = None,
                  feature_df: Optional[pd.DataFrame] = None) -> 'Expression':
        """从字典还原 Expression

        Args:
            data: to_dict() 输出的字典
            feature_space: 可选的 FeatureSpace 实例
            feature_df: 可选的预计算特征矩阵
        """
        tree = TreeNode.from_dict(data.get('tree', data.get('ast', {})))
        expr = cls(tree, feature_space, feature_df, name=data.get('name'))
        expr.source = data.get('source', expr.source)
        return expr

    @classmethod
    def from_json(cls, json_str: str, feature_space: Optional[FeatureSpace] = None,
                  feature_df: Optional[pd.DataFrame] = None) -> 'Expression':
        """从 JSON 字符串还原 Expression"""
        data = json.loads(json_str)
        return cls.from_dict(data, feature_space, feature_df)

    def save(self, path: str):
        """保存表达式到 JSON 文件"""
        with open(path, 'w', encoding='utf-8') as f:
            f.write(self.to_json())

    @classmethod
    def load(cls, path: str, feature_space: Optional[FeatureSpace] = None,
             feature_df: Optional[pd.DataFrame] = None) -> 'Expression':
        """从 JSON 文件加载 Expression"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data, feature_space, feature_df)

    def __repr__(self):
        return f"Expression('{self.source[:60]}', complexity={self.complexity}, features={self.features_used})"

    def __str__(self):
        return self.source


# ============================================================
# Persist / RegimeSwitch 变换器
# ============================================================

def persist(expr: Expression, n: int = 3) -> Expression:
    """信号确认包装器：连续 n 个信号同向才触发

    用法:
        filtered = persist(expr, 3)   # 连续 3 日同向才触发

    相当于表达式语法:
        persist(thr_mean(RSI{14}), 3)
    """
    tree = TreeNode(NodeType.PERSIST, 'persist', [expr._tree.copy()], param=n)
    return Expression(tree, expr._feature_space, expr._feature_df,
                      name=f'Persist({n})_{expr._name}')


def regime_switch(condition: Expression, expr_a: Expression, expr_b: Expression) -> Expression:
    """条件分支信号：condition 成立时走 expr_a，否则走 expr_b

    用法:
        trend_signal = regime_switch(adx_signal, trend_expr, range_expr)

    相当于表达式语法:
        if_then(ADX{14} > 25, ATR_breakout, BBWIDTH_reversal)
    """
    tree = TreeNode(NodeType.SWITCH, 'switch',
                     [condition._tree.copy(), expr_a._tree.copy(), expr_b._tree.copy()])
    fs = condition._feature_space or expr_a._feature_space or expr_b._feature_space
    fd = condition._feature_df or expr_a._feature_df or expr_b._feature_df
    return Expression(tree, fs, fd, name=f'Switch_{condition._name}')


def parse_and_build(expr_str: str,
                    feature_space: Optional[FeatureSpace] = None,
                    feature_df: Optional[pd.DataFrame] = None) -> Expression:
    """便捷函数：从字符串创建 Expression，一行替代两步操作

    相当于:
        tree = parse_expression(expr_str)
        expr = Expression(tree, feature_space, feature_df)
    """
    return Expression(expr_str, feature_space, feature_df)
