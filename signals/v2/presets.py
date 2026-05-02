"""
signals/v2/presets.py - 表达式模板库

内置预配置的表达式模板，替代旧的 SIGNAL_TEMPLATES。
每个模板是一个参数化表达式，通过 from_preset() 填充参数。

模板语法：
    'ATR(?)'  # ? 表示可参数化的占位符，类似SQL prepared statement
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .expression import Expression
from .features import FeatureSpace


@dataclass
class ExpressionPreset:
    """表达式模板"""
    name: str
    template: str
    description: str
    category: str
    param_count: int
    example: str


PRESETS: Dict[str, ExpressionPreset] = {
    # ============================
    # Overlap 趋势类
    # ============================
    'ma_cross': ExpressionPreset(
        name='ma_cross',
        template='thr_mean(SMA(?))',
        description='均线偏离均值信号（short周期）',
        category='overlap',
        param_count=1,
        example='MA交叉(short=5)'
    ),
    'ema': ExpressionPreset(
        name='ema',
        template='thr_mean(EMA(?))',
        description='EMA偏离均值信号',
        category='overlap',
        param_count=1,
        example='EMA(20)'
    ),
    'trima': ExpressionPreset(
        name='trima',
        template='thr_mean(TRIMA(?))',
        description='三角加权移动平均偏离均值',
        category='overlap',
        param_count=1,
        example='TRIMA(60)'
    ),
    'boll': ExpressionPreset(
        name='boll',
        template='thr_mean(BBWIDTH(?))',
        description='布林带宽度偏离均值信号',
        category='overlap',
        param_count=1,
        example='BBWIDTH(20)'
    ),

    # ============================
    # Momentum 动量类
    # ============================
    'rsi': ExpressionPreset(
        name='rsi',
        template='RSI(?)',
        description='RSI信号（已中心化：RSI-50）',
        category='momentum',
        param_count=1,
        example='RSI(14)'
    ),
    'macd': ExpressionPreset(
        name='macd',
        template='MACD(?, ?, ?)',
        description='MACD柱线信号',
        category='momentum',
        param_count=3,
        example='MACD(12, 26, 9)'
    ),
    'adx': ExpressionPreset(
        name='adx',
        template='thr_mean(ADX(?))',
        description='ADX趋势强度偏离均值信号',
        category='momentum',
        param_count=1,
        example='ADX(14)'
    ),
    'cci': ExpressionPreset(
        name='cci',
        template='thr_mean(CCI(?))',
        description='CCI偏离均值信号',
        category='momentum',
        param_count=1,
        example='CCI(14)'
    ),

    # ============================
    # Volatility 波动率类
    # ============================
    'atr': ExpressionPreset(
        name='atr',
        template='thr_mean(ATR(?))',
        description='ATR偏离均值信号（AIdev V2最优）',
        category='volatility',
        param_count=1,
        example='ATR(10)'
    ),
    'bbw': ExpressionPreset(
        name='bbw',
        template='thr_mean(BBWIDTH(?))',
        description='布林带宽度偏离均值信号',
        category='volatility',
        param_count=1,
        example='BBWIDTH(30)'
    ),
    'hv': ExpressionPreset(
        name='hv',
        template='thr_mean(HV(?))',
        description='历史波动率偏离均值信号',
        category='volatility',
        param_count=1,
        example='HV(20)'
    ),
    'natr': ExpressionPreset(
        name='natr',
        template='thr_mean(NATR(?))',
        description='归一化ATR偏离均值信号',
        category='volatility',
        param_count=1,
        example='NATR(10)'
    ),

    # ============================
    # Combo 组合类
    # ============================
    'atr_trima': ExpressionPreset(
        name='atr_trima',
        template='thr_mean(ATR(?)) & thr_mean(TRIMA(?))',
        description='ATR×TRIMA组合（AIdev V3最优）',
        category='combo',
        param_count=2,
        example='ATR(7) & TRIMA(60)'
    ),
    'atr_adx': ExpressionPreset(
        name='atr_adx',
        template='thr_mean(ATR(?)) & thr_mean(ADX(?))',
        description='ATR×ADX组合',
        category='combo',
        param_count=2,
        example='ATR(7) & ADX(14)'
    ),
    'bbw_tsf': ExpressionPreset(
        name='bbw_tsf',
        template='thr_mean(BBWIDTH(?)) sub thr_mean(TSF(?))',
        description='BBW×TSF组合（AIdev V4共识特征）',
        category='combo',
        param_count=2,
        example='BBWIDTH(30) sub TSF(7)'
    ),
    'bbw_tsf_vol': ExpressionPreset(
        name='bbw_tsf_vol',
        template='thr_mean(BBWIDTH(?)) sub (VOL_RATIO(?, ?, ma=True) & thr_mean(TSF(?)))',
        description='BBW×VOL_RATIO×TSF三维结构（AIdev V4.3最优）',
        category='combo',
        param_count=4,
        example='BBWIDTH(30) sub (VOL_RATIO(5) & TSF(7))'
    ),

    # ============================
    # Regime 市场状态类
    # ============================
    'trend_strength': ExpressionPreset(
        name='trend_strength',
        template='thr_mean(TREND_STRENGTH(?))',
        description='趋势强度偏离均值信号',
        category='regime',
        param_count=1,
        example='TREND_STRENGTH(20)'
    ),
    'vol_regime': ExpressionPreset(
        name='vol_regime',
        template='thr_mean(VOL_REGIME(?))',
        description='波动率regime信号',
        category='regime',
        param_count=1,
        example='VOL_REGIME(20)'
    ),
    'vol_chg': ExpressionPreset(
        name='vol_chg',
        template='thr_mean(VOL_CHG(?))',
        description='量能变化率信号',
        category='regime',
        param_count=1,
        example='VOL_CHG(5)'
    ),
}


def from_preset(preset_name: str, params: List[Any],
                feature_space: FeatureSpace) -> Expression:
    """
    从模板创建Expression

    Args:
        preset_name: 模板名称
        params: 参数值列表（按?占位符顺序）
        feature_space: FeatureSpace实例

    Returns:
        Expression 实例
    """
    preset = PRESETS.get(preset_name)
    if preset is None:
        available = ', '.join(PRESETS.keys())
        raise ValueError(f"未知模板: {preset_name}。可用模板: {available}")

    if len(params) != preset.param_count:
        raise ValueError(f"模板 '{preset_name}' 需要 {preset.param_count} 个参数，提供了 {len(params)}")

    expr_str = preset.template
    for param in params:
        expr_str = expr_str.replace('?', str(param), 1)

    return Expression(expr_str, feature_space=feature_space, name=f'{preset_name}_{params}')


# 扩展 Expression 类
Expression.from_preset = staticmethod(from_preset)
Expression.list_presets = staticmethod(lambda: list(PRESETS.keys()))
Expression.get_preset_info = staticmethod(lambda name: PRESETS.get(name))
