"""
ft2.factor — 因子挖掘与回测框架

版本：
  v2/ — 当前主力（Pipeline + Scheduler + Allocator + Validator + Grid/BO搜索 + GP）
  v1/ — 已归档（2026-05-25），只读存档，不再发展

使用方式：
  from factor.v2 import FactorPipeline, Factor, FactorGridSearch
  from factor.v1 import FactorCalculator, FactorValidator   # 仅向后兼容

禁止跨版本依赖，v1 与 v2 各自独立。
"""