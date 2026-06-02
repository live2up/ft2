# symbol_classifier.py
# 品种分类器（代码 → 品种类型）—— 数据系统 L1 原子核心层
# Version: 2026-05-29
#
# 定位：
#   - 整个数据系统中唯一负责"代码 → 品种类型"映射的模块
#   - 纯规则驱动，零外部依赖，可嵌入任何项目
#   - 被数据维护层（kline_query/kline_repair）和数据接口层（d2_controller）共同引用
#
# 使用方式：
#   from symbol_classifier import classify_symbol, classify_symbols, SYMBOL_TYPE_MAP
#
# 识别优先级：
#   1. 后缀规则（.SI/.SL/.SWI/.TI/.CSI → index，.BJ → stock）
#   2. SYMBOL_TYPE_MAP 精确映射（内置边界指数 + 用户自定义）
#   3. 数字规则（.SH/.SZ：5/1开头 → etf，000开头 → index，399开头 → index）
#   4. 无法识别 → None

from typing import Dict, List, Optional


# =============================================================================
# 品种类型常量（消除魔法字符串）
# =============================================================================

SEC_TYPE_INDEX = 'index'
SEC_TYPE_STOCK = 'stock'
SEC_TYPE_ETF = 'etf'
SEC_TYPE_UNKNOWN = None


# =============================================================================
# 代码后缀 → 品种类型映射规则
# =============================================================================

SYMBOL_TYPE_RULES: List[Dict] = [
    {'suffix': '.SI', 'type': 'index'},
    {'suffix': '.SL', 'type': 'index'},
    {'suffix': '.SWI', 'type': 'index'},
    {'suffix': '.TI', 'type': 'index'},
    {'suffix': '.CSI', 'type': 'index'},
    {'suffix': '.BJ', 'type': 'stock'},
]


# =============================================================================
# 强制代码类型映射（用于 classify_symbol() 规则无法识别的边界代码）
# =============================================================================

SYMBOL_TYPE_MAP: Dict[str, str] = {
    # 规则已自动覆盖所有 000.SH（上证指数）、399.SH/SZ（深证指数）
    # 以下仅保留少数核心指数作为参考标记，识别走规则不依赖此映射：
    '000300.SH': 'index',   # 沪深300
    '000688.SH': 'index',   # 科创50
    '000905.SH': 'index',   # 中证500
    '000852.SH': 'index',   # 中证1000
    '399001.SZ': 'index',   # 深证成指
    '399006.SZ': 'index',   # 创业板指
}


# =============================================================================
# 缓存
# =============================================================================

_classify_cache: Dict[str, Optional[str]] = {}


def classify_symbol(symbol: str) -> Optional[str]:
    """
    识别单个代码的品种类型

    规则（按优先级）：
      1. .SI/.SL/.SWI/.TI/.CSI → index
      2. .BJ → stock
      3. SYMBOL_TYPE_MAP 强制映射查表（内置边界指数 + 用户自定义）
      4. .SH/.SZ 数字规则：
         - 以 5/1 开头 → etf（ETF 编码规则）
         - 以 000 开头 → index（上证指数编码规则）
         - 以 399 开头 → index（深交所指数编码规则）
         - .SH 以 6 开头 → stock（上交所股票编码规则）
         - .SZ 以 0/2/3 开头（非399） → stock（深交所股票编码规则）
      5. 仍无法识别 → None

    Args:
        symbol: 代码，如 '000001.SH', '399317.SZ', '511880.SH', '000300.CSI'

    Returns:
        品种类型：'index' / 'stock' / 'etf' / None（无法识别）
    """
    if not symbol or not isinstance(symbol, str):
        return None

    symbol_std = symbol.upper().strip()

    if symbol_std in _classify_cache:
        return _classify_cache[symbol_std]

    for rule in SYMBOL_TYPE_RULES:
        if symbol_std.endswith(rule['suffix']):
            _classify_cache[symbol_std] = rule['type']
            return rule['type']

    if symbol_std in SYMBOL_TYPE_MAP:
        _classify_cache[symbol_std] = SYMBOL_TYPE_MAP[symbol_std]
        return SYMBOL_TYPE_MAP[symbol_std]

    if symbol_std.endswith('.SH') or symbol_std.endswith('.SZ'):
        import re
        match = re.match(r'^(\d+)\.(SH|SZ)$', symbol_std)
        if match:
            code_num = match.group(1)
            suffix = match.group(2)
            first = code_num[0] if code_num else ''
            if first in ('5', '1'):
                _classify_cache[symbol_std] = 'etf'
                return 'etf'
            if suffix == 'SH' and code_num.startswith('000'):
                _classify_cache[symbol_std] = 'index'
                return 'index'
            if code_num.startswith('399'):
                _classify_cache[symbol_std] = 'index'
                return 'index'
            if suffix == 'SH' and first == '6':
                _classify_cache[symbol_std] = 'stock'
                return 'stock'
            if suffix == 'SZ' and first in ('0', '2', '3'):
                _classify_cache[symbol_std] = 'stock'
                return 'stock'

    _classify_cache[symbol_std] = None
    return None


def classify_symbols(symbols: List[str]) -> Dict[str, List[str]]:
    """
    批量识别代码并分组

    Args:
        symbols: 代码列表

    Returns:
        {'index': [...], 'stock': [...], 'etf': [...], 'unknown': [...]}
    """
    result: Dict[str, List[str]] = {
        'index': [],
        'stock': [],
        'etf': [],
        'unknown': [],
    }

    for symbol in symbols:
        dtype = classify_symbol(symbol)
        if dtype:
            result[dtype].append(symbol)
        else:
            result['unknown'].append(symbol)

    return result


def clear_classify_cache(symbol: Optional[str] = None) -> None:
    """清除品种识别缓存"""
    global _classify_cache
    if symbol:
        _classify_cache.pop(symbol.upper().strip(), None)
    else:
        _classify_cache = {}


# =============================================================================
# 导出
# =============================================================================

__all__ = [
    'SEC_TYPE_INDEX',
    'SEC_TYPE_STOCK',
    'SEC_TYPE_ETF',
    'SEC_TYPE_UNKNOWN',
    'SYMBOL_TYPE_RULES',
    'SYMBOL_TYPE_MAP',
    'classify_symbol',
    'classify_symbols',
    'clear_classify_cache',
]
