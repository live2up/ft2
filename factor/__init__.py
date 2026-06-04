"""
ft2.factor — 因子挖掘与回测框架

版本：
  v3/ — 推荐版本（GP发现引擎 + 可插拔适应度 + 迭代探索 + 自增长因子库）
  v2/ — 稳定版本，不再新增功能
  v1/ — 已归档（2026-05-25），只读存档

使用方式：
  from factor.v3 import FactorPipeline, FactorExpression, FactorDiscoveryEngine
  from factor.v2 import FactorPipeline, FactorGridSearch        # 向后兼容
  from factor.v1 import FactorCalculator, FactorValidator        # 仅向后兼容

禁止跨版本依赖，v1/v2/v3 各自独立。
"""