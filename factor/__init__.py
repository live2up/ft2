"""
ft2.factor — 因子挖掘与回测框架

版本：
  v4/ — 当前主力（基于 signals.v4 AST DSL + ft2.core Engine + 67原语，探索灵活）
  v3/ — 保留，仅供历史测试对照
  v2/ — 已归档
  v1/ — 已归档（2026-05-25），只读存档

使用方式：
  from factor.v4 import FactorExpression, FactorLibrary, EngineCore     # 推荐
  from factor.v3 import FactorPipeline, FactorExpression, FactorDiscoveryEngine  # 仅历史对照

禁止跨版本依赖，v1/v2/v3/v4 各自独立。
"""