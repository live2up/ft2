"""
ft2.signals.v1 — 信号管理器 v1（已归档）

自 2026-05-25 起，v1 模块归档至本子目录。
新开发请使用 signals.v2，v1 仅保留用于向后兼容。

包含：
- base: 信号基类（Signal / SignalType / ...）
- generator: 信号生成器（100+ TA-Lib 指标信号）
- combiner: 信号融合器（投票/打分/加权/自适应/时序）
- threshold: 阈值策略
- registry: 信号注册器
- backtest: 轻量回测
- ic_analyzer: IC 分析器
- indicators: 指标计算函数库
- timeframe: 多周期数据处理
"""

from .base import (
    Signal,
    SignalType,
    SignalDirection,
    TradingSignal,
    SignalSeries,
)
from .generator import (
    SignalGenerator,
    MASignal, BOLLSignal, SARSignal, MIDPOINTSignal, DMISignal, AROONSignal,
    SMASignal, EMASignal, WMASignal, DEMASignal, TEMASignal, KAMASignal,
    T3Signal, TRIMASignal, MAMASignal, MAVPSignal,
    AVGPRICESignal, MEDPRICESignal, TYPPRICESignal, WCLPRICESignal,
    HTTRENDLINESignal, MIDPRICESignal, SAREXTSignal, ACCBANDSSignal,
    MACDSignal, RSISignal, KDJSignal, CCISignal, WRSignal, ROCSignal,
    STOCHFSignal, CMOSignal, MFISignal, MOMSignal, PPOSignal, STOCHRSISignal,
    TRIXSignal, ULTOSCSignal, WILLRSignal,
    ADXSignal, ADXRSignal, APOSignal, DXSignal, IMISignal,
    MACDEXTSignal, MACDFIXSignal,
    PLUSDISignal, MINUSDISignal, PLUSDMSignal, MINUSDMSignal,
    ROCPSignal, ROCRSignal, ROCR100Signal, STOCHSignal, BOPSignal,
    OBVSignal, VOLSignal, ADOSCSignal, ADSignal,
    ATRSignal, STDDEVSignal, NATRSignal, TRANGESignal, AVGDEVSignal,
    BBWidthSignal, RealizedVolatilitySignal,
    HTDCCPERIODSignal, HTDCPHASESignal, HTPHASORSignal, HTSINESignal, HTTRENDMODESignal,
    BETASignal, RSRSMSignal, CORRELSignal,
    LINEARREGSignal, LINEARREGANGLESignal, LINEARREGINTERCEPTSignal, LINEARREGSLOPESignal,
    TSFSignal, VARSignal,
    ADDSignal, SUBSignal, MULTSignal, DIVSignal, SUMSignal,
    MAXSignal, MINSignal, MAXINDEXSignal, MININDEXSignal,
    MINMAXSignal, MINMAXINDEXSignal,
    ACOSSignal, ASINSignal, ATANSignal, CEILSignal, COSSignal, COSHSignal,
    EXPSignal, FLOORSignal, LNSignal, LOG10Signal,
    SINSignal, SINHSignal, SQRTSignal, TANSignal, TANHSignal,
    AROONOSCSignal,
    CompositeSignal, FunctionSignal, IndicatorSignal,
    create_signal,
)
from .combiner import (
    SignalCombiner,
    VotingCombiner, ScoringCombiner, WeightedCombiner,
    EqualWeightCombiner, AdaptiveCombiner, SequenceCombiner,
)
from .registry import (
    SignalRegistry, SIGNAL_TEMPLATES,
    create_signal_from_template, create_signal_set,
)
from .threshold import (
    ThresholdPolicy, SimpleThreshold, PercentileThreshold,
    ZScoreThreshold, MovingThreshold, DualThreshold,
    THRESHOLD_PRESETS, get_threshold_preset,
)
from .timeframe import MultiTimeframeAligner, FrequencyConverter
from .backtest import run_backtest, BacktestResult
from .ic_analyzer import ICAnalyzer

__all__ = [
    'Signal', 'SignalType', 'SignalDirection', 'SignalSeries', 'TradingSignal',
    'SignalGenerator',
    # 趋势
    'MASignal', 'BOLLSignal', 'SARSignal', 'MIDPOINTSignal', 'DMISignal', 'AROONSignal',
    'SMASignal', 'EMASignal', 'WMASignal', 'DEMASignal', 'TEMASignal', 'KAMASignal',
    'T3Signal', 'TRIMASignal', 'MAMASignal', 'MAVPSignal',
    'AVGPRICESignal', 'MEDPRICESignal', 'TYPPRICESignal', 'WCLPRICESignal',
    'HTTRENDLINESignal', 'MIDPRICESignal', 'SAREXTSignal', 'ACCBANDSSignal',
    # 动量
    'MACDSignal', 'RSISignal', 'KDJSignal', 'CCISignal', 'WRSignal', 'ROCSignal',
    'STOCHFSignal', 'CMOSignal', 'MFISignal', 'MOMSignal', 'PPOSignal', 'STOCHRSISignal',
    'TRIXSignal', 'ULTOSCSignal', 'WILLRSignal',
    'ADXSignal', 'ADXRSignal', 'APOSignal', 'DXSignal', 'IMISignal',
    'MACDEXTSignal', 'MACDFIXSignal',
    'PLUSDISignal', 'MINUSDISignal', 'PLUSDMSignal', 'MINUSDMSignal',
    'ROCPSignal', 'ROCRSignal', 'ROCR100Signal', 'STOCHSignal', 'BOPSignal',
    # 成交量
    'OBVSignal', 'VOLSignal', 'ADOSCSignal', 'ADSignal',
    # 波动率
    'ATRSignal', 'STDDEVSignal', 'NATRSignal', 'TRANGESignal', 'AVGDEVSignal',
    'BBWidthSignal', 'RealizedVolatilitySignal',
    # 周期
    'HTDCCPERIODSignal', 'HTDCPHASESignal', 'HTPHASORSignal', 'HTSINESignal', 'HTTRENDMODESignal',
    # 统计
    'BETASignal', 'RSRSMSignal', 'CORRELSignal',
    'LINEARREGSignal', 'LINEARREGANGLESignal', 'LINEARREGINTERCEPTSignal', 'LINEARREGSLOPESignal',
    'TSFSignal', 'VARSignal',
    # 数学
    'ADDSignal', 'SUBSignal', 'MULTSignal', 'DIVSignal', 'SUMSignal',
    'MAXSignal', 'MINSignal', 'MAXINDEXSignal', 'MININDEXSignal',
    'MINMAXSignal', 'MINMAXINDEXSignal',
    'ACOSSignal', 'ASINSignal', 'ATANSignal', 'CEILSignal', 'COSSignal', 'COSHSignal',
    'EXPSignal', 'FLOORSignal', 'LNSignal', 'LOG10Signal',
    'SINSignal', 'SINHSignal', 'SQRTSignal', 'TANSignal', 'TANHSignal',
    'AROONOSCSignal',
    'CompositeSignal', 'FunctionSignal', 'IndicatorSignal', 'create_signal',
    'SignalCombiner', 'VotingCombiner', 'ScoringCombiner', 'WeightedCombiner',
    'EqualWeightCombiner', 'AdaptiveCombiner', 'SequenceCombiner',
    'SignalRegistry', 'SIGNAL_TEMPLATES', 'create_signal_from_template', 'create_signal_set',
    'ThresholdPolicy', 'SimpleThreshold', 'PercentileThreshold',
    'ZScoreThreshold', 'MovingThreshold', 'DualThreshold',
    'THRESHOLD_PRESETS', 'get_threshold_preset',
    'MultiTimeframeAligner', 'FrequencyConverter',
    'run_backtest', 'BacktestResult',
    'ICAnalyzer',
]