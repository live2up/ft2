# signals 模块 - 择时信号层
"""
信号层：指标 → 信号 → 融合 → 择时 → 策略执行

架构：
  原始数据 (K线)
       ↓
  SignalGenerator (指标计算)
       ↓
  Signal (连续值)
       ↓
  SignalCombiner (多信号融合)
       ↓
  CombinedSignal (综合信号)
       ↓
  ThresholdPolicy (阈值化)
       ↓
  TradingSignal (买入/持有/卖出)
       ↓
  SignalStrategy (策略适配器 → 调用 core 引擎)

目录：
  signals/
  ├── __init__.py          # 模块入口
  ├── base.py              # Signal 基类和枚举
  ├── generator.py         # 信号生成器（14种内置 + 组合 + 函数式）
  ├── combiner.py          # 信号融合器（投票/打分/加权/自适应/时序）
  ├── threshold.py         # 阈值策略（简单/分位数/Z-Score/双阈值）
  ├── registry.py          # 信号注册器和模板
  ├── timeframe.py         # 多周期数据处理（对齐、转换、特征）
  ├── strategy.py          # 策略适配器（连接 signals 和 core 引擎）
  ├── examples.py          # 使用示例
  └── test_strategy.py     # 策略适配器测试
"""

from .base import (
    Signal,           # 信号基类
    SignalType,       # 信号类型枚举
    SignalDirection,  # 信号方向枚举
    TradingSignal,    # 交易信号
    SignalSeries,     # 信号序列
)

from .generator import (
    SignalGenerator,           # 生成器基类
    # Overlap Studies
    MASignal,                  # 均线交叉信号
    BOLLSignal,                # 布林带信号
    SARSignal,                 # SAR 抛物线转向
    MIDPOINTSignal,            # MIDPOINT 中点
    DMISignal,                 # DMI 趋向指标
    AROONSignal,               # AROON 阿隆指标
    SMASignal,                 # SMA 简单移动平均
    EMASignal,                 # EMA 指数移动平均
    WMASignal,                 # WMA 加权移动平均
    DEMASignal,                # DEMA 双指数移动平均
    TEMASignal,                # TEMA 三重指数移动平均
    KAMASignal,                # KAMA 考夫曼自适应
    T3Signal,                  # T3 三重指数平滑
    TRIMASignal,               # TRIMA 三角移动平均
    MAMASignal,                # MAMA 梅斯自适应
    MAVPSignal,                # MAVP 可变周期
    AVGPRICESignal,            # AVGPRICE 平均价格
    MEDPRICESignal,            # MEDPRICE 中间价格
    TYPPRICESignal,            # TYPPRICE 典型价格
    WCLPRICESignal,            # WCLPRICE 加权收盘价
    HTTRENDLINESignal,         # HT_TRENDLINE 希尔伯特趋势线
    MIDPRICESignal,            # MIDPRICE 中间价
    SAREXTSignal,              # SAREXT SAR 扩展版
    ACCBANDSSignal,            # ACCBANDS 加速度带
    # Momentum Indicators
    MACDSignal,                # MACD 信号
    RSISignal,                 # RSI 信号
    KDJSignal,                 # KDJ 信号
    CCISignal,                 # CCI 信号
    WRSignal,                  # 威廉指标信号
    ROCSignal,                 # ROC 信号
    STOCHFSignal,              # 随机指标信号
    CMOSignal,                 # CMO 钱德动量
    MFISignal,                 # MFI 资金流量
    MOMSignal,                 # MOM 动量
    PPOSignal,                 # PPO 比例 MACD
    STOCHRSISignal,            # STOCHRSI 随机 RSI
    TRIXSignal,                # TRIX 三重指数平滑
    ULTOSCSignal,              # ULTOSC 终极振荡器
    WILLRSignal,               # WILLR 威廉指标
    ADXSignal,                 # ADX 平均趋向指数
    ADXRSignal,                # ADXR 平均趋向指数评估
    APOSignal,                 # APO 绝对价格振荡器
    DXSignal,                  # DX 趋向指数
    IMISignal,                 # IMI 日内动量指数
    MACDEXTSignal,             # MACDEXT MACD 扩展
    MACDFIXSignal,             # MACDFIX MACD 固定参数
    PLUSDISignal,              # PLUS_DI 正向趋向
    MINUSDISignal,             # MINUS_DI 负向趋向
    PLUSDMSignal,              # PLUS_DM 正向趋向动量
    MINUSDMSignal,             # MINUS_DM 负向趋向动量
    ROCPSignal,                # ROCP 价格变化率
    ROCRSignal,                # ROCR 价格变化比率
    ROCR100Signal,             # ROCR100 价格变化比率(100)
    STOCHSignal,               # STOCH 慢速随机
    BOPSignal,                 # BOP 均势
    # Volume Indicators
    OBVSignal,                 # OBV 信号
    VOLSignal,                 # 量能信号
    ADOSCSignal,               # ADOSC 蔡金振荡器
    ADSignal,                  # AD 蔡金线
    # Volatility Indicators
    ATRSignal,                 # ATR 平均真实波幅
    STDDEVSignal,              # STDDEV 标准差
    NATRSignal,                # NATR 归一化 ATR
    TRANGESignal,              # TRANGE 真实波幅
    AVGDEVSignal,              # AVGDEV 平均偏差
    # Cycle Indicators
    HTDCCPERIODSignal,         # HT_DCPERIOD 希尔伯特周期
    HTDCPHASESignal,           # HT_DCPHASE 希尔伯特相位
    HTPHASORSignal,            # HT_PHASOR 希尔伯特相量
    HTSINESignal,              # HT_SINE 希尔伯特正弦
    HTTRENDMODESignal,         # HT_TRENDMODE 希尔伯特模式
    # Statistic Functions
    BETASignal,                # BETA 贝塔系数
    RSRSMSignal,               # RSRS 阻力支撑相对强度
    CORRELSignal,              # CORREL 相关系数
    LINEARREGSignal,           # LINEARREG 线性回归
    LINEARREGANGLESignal,      # LINEARREG_ANGLE 线性回归角度
    LINEARREGINTERCEPTSignal,  # LINEARREG_INTERCEPT 线性回归截距
    LINEARREGSLOPESignal,      # LINEARREG_SLOPE 线性回归斜率
    TSFSignal,                 # TSF 时间序列预测
    VARSignal,                 # VAR 方差
    # Math Operators
    ADDSignal,                 # ADD 加法
    SUBSignal,                 # SUB 减法
    MULTSignal,                # MULT 乘法
    DIVSignal,                 # DIV 除法
    SUMSignal,                 # SUM 求和
    MAXSignal,                 # MAX 最大值
    MINSignal,                 # MIN 最小值
    MAXINDEXSignal,            # MAXINDEX 最大值索引
    MININDEXSignal,            # MININDEX 最小值索引
    MINMAXSignal,              # MINMAX 最大值最小值
    MINMAXINDEXSignal,         # MINMAXINDEX 最大值最小值索引
    # Math Transform (数学变换)
    ACOSSignal, ASINSignal, ATANSignal, CEILSignal, COSSignal, COSHSignal,
    EXPSignal, FLOORSignal, LNSignal, LOG10Signal,
    SINSignal, SINHSignal, SQRTSignal, TANSignal, TANHSignal,
    # Momentum 补充
    AROONOSCSignal,            # AROONOSC 阿隆振荡器
    # 组合/通用
    CompositeSignal,           # 组合信号
    FunctionSignal,            # 函数式信号
    IndicatorSignal,           # 通用指标信号
    create_signal,             # 工厂函数
)

from .combiner import (
    SignalCombiner,     # 融合器基类
    VotingCombiner,     # 投票融合
    ScoringCombiner,    # 打分融合
    WeightedCombiner,   # 加权融合
    EqualWeightCombiner,# 等权融合
    AdaptiveCombiner,   # 自适应融合
    SequenceCombiner,   # 时序融合（新增）
)

from .registry import (
    SignalRegistry,
    SIGNAL_TEMPLATES,
    create_signal_from_template,
    create_signal_set,
)

from .threshold import (
    ThresholdPolicy,        # 阈值基类
    SimpleThreshold,        # 简单阈值
    PercentileThreshold,   # 分位数阈值
    ZScoreThreshold,       # Z-Score 阈值
    MovingThreshold,       # 移动阈值
    DualThreshold,         # 双阈值
    THRESHOLD_PRESETS,
    get_threshold_preset,
)

from .timeframe import (
    MultiTimeframeAligner,  # 多周期数据对齐器
    FrequencyConverter,     # 频率转换器
)

from .backtest import (
    run_backtest,           # 轻量回测函数
    BacktestResult,         # 回测结果对象
)

from .ic_analyzer import (
    ICAnalyzer,             # 通用IC分析器
)

__all__ = [
    # 基类
    'Signal', 'SignalType', 'SignalDirection', 'SignalSeries', 'TradingSignal',
    # 生成器基类
    'SignalGenerator',
    # Overlap Studies (趋势指标)
    'MASignal', 'BOLLSignal', 'SARSignal', 'MIDPOINTSignal', 'DMISignal', 'AROONSignal',
    'SMASignal', 'EMASignal', 'WMASignal', 'DEMASignal', 'TEMASignal', 'KAMASignal',
    'T3Signal', 'TRIMASignal', 'MAMASignal', 'MAVPSignal',
    'AVGPRICESignal', 'MEDPRICESignal', 'TYPPRICESignal', 'WCLPRICESignal',
    'HTTRENDLINESignal', 'MIDPRICESignal', 'SAREXTSignal', 'ACCBANDSSignal',
    # Momentum Indicators (动量指标)
    'MACDSignal', 'RSISignal', 'KDJSignal', 'CCISignal', 'WRSignal', 'ROCSignal',
    'STOCHFSignal', 'CMOSignal', 'MFISignal', 'MOMSignal', 'PPOSignal', 'STOCHRSISignal',
    'TRIXSignal', 'ULTOSCSignal', 'WILLRSignal',
    'ADXSignal', 'ADXRSignal', 'APOSignal', 'DXSignal', 'IMISignal',
    'MACDEXTSignal', 'MACDFIXSignal',
    'PLUSDISignal', 'MINUSDISignal', 'PLUSDMSignal', 'MINUSDMSignal',
    'ROCPSignal', 'ROCRSignal', 'ROCR100Signal', 'STOCHSignal', 'BOPSignal',
    # Volume Indicators (成交量指标)
    'OBVSignal', 'VOLSignal', 'ADOSCSignal', 'ADSignal',
    # Volatility Indicators (波动率指标)
    'ATRSignal', 'STDDEVSignal', 'NATRSignal', 'TRANGESignal', 'AVGDEVSignal',
    # Cycle Indicators (周期指标)
    'HTDCCPERIODSignal', 'HTDCPHASESignal', 'HTPHASORSignal', 'HTSINESignal', 'HTTRENDMODESignal',
    # Statistic Functions (统计函数)
    'BETASignal', 'RSRSMSignal', 'CORRELSignal',
    'LINEARREGSignal', 'LINEARREGANGLESignal', 'LINEARREGINTERCEPTSignal', 'LINEARREGSLOPESignal',
    'TSFSignal', 'VARSignal',
    # Math Operators (数学运算)
    'ADDSignal', 'SUBSignal', 'MULTSignal', 'DIVSignal', 'SUMSignal',
    'MAXSignal', 'MINSignal', 'MAXINDEXSignal', 'MININDEXSignal',
    'MINMAXSignal', 'MINMAXINDEXSignal',
    # Math Transform (数学变换)
    'ACOSSignal', 'ASINSignal', 'ATANSignal', 'CEILSignal', 'COSSignal', 'COSHSignal',
    'EXPSignal', 'FLOORSignal', 'LNSignal', 'LOG10Signal',
    'SINSignal', 'SINHSignal', 'SQRTSignal', 'TANSignal', 'TANHSignal',
    # Momentum 补充
    'AROONOSCSignal',
    # 组合/通用
    'CompositeSignal', 'FunctionSignal', 'IndicatorSignal', 'create_signal',
    # 融合器
    'SignalCombiner',
    'VotingCombiner',
    'ScoringCombiner',
    'WeightedCombiner',
    'EqualWeightCombiner',
    'AdaptiveCombiner',
    'SequenceCombiner',    # 时序融合
    # 注册器
    'SignalRegistry',
    'SIGNAL_TEMPLATES',
    'create_signal_from_template',
    'create_signal_set',
    # 阈值策略
    'ThresholdPolicy',
    'SimpleThreshold',
    'PercentileThreshold',
    'ZScoreThreshold',
    'MovingThreshold',
    'DualThreshold',
    'THRESHOLD_PRESETS',
    'get_threshold_preset',
    # 多周期处理
    'MultiTimeframeAligner',
    'FrequencyConverter',
    # 轻量回测
    'run_backtest',
    'BacktestResult',
    # IC分析器
    'ICAnalyzer',
]
