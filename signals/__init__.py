"""
ft2.signals — 择时信号层

版本：
  v2/ — 当前主力（表达式引擎 + Pipeline + GP 优化 + 网格搜索 + 参数量化）
  v1/ — 已归档（2026-05-25），只读存档（TA-Lib 信号生成器 + 融合器 + 回测）

使用方式：
  from signals.v2 import Expression, SignalPipeline, FeatureSpace
  from signals.v1 import MASignal, run_backtest   # 仅向后兼容

禁止跨版本依赖，v1 与 v2 各自独立。
"""