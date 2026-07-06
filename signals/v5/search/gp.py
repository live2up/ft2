"""
signals/v5/search/gp.py — 信号 GP 引擎 (wrapper)
=============================================================================
[重构] 2026-07-07 基于 utils/gp/v5/engine.py 核心重写，
继承 GPEngine 并注入信号端 evaluator (_SignalFromAST)。

与 factor/v5/gp_engine.py 的关系:
  - 共同基类: utils.gp.v5.engine.GPEngine (进化循环/选择/交叉/变异/方向追踪)
  - 差异: factor 端注入 _ExpressionFromAST (2D 面板求值)
          signals 端注入 _SignalFromAST (1D 信号求值 + SigEngine 回测)

用法:
  >>> from signals.v5.search.gp import SigGPEngine, TreeGenConfig
  >>> from signals.v5 import SigEngine
  >>> from signals.v5.expression import _build_gp_data_dict

  >>> # 适应度: 用 SigEngine 算 SR
  >>> class SigFitness:
  ...     def __init__(self, data, start_date=None):
  ...         self.data = data
  ...         self.start_date = start_date
  ...     def compute(self, signal_values):
  ...         signal = pd.Series(signal_values.ravel(), index=self.data.index[:len(signal_values)])
  ...         r = SigEngine.backtest(signal, self.data, mode='fast',
  ...                                 start_date=self.start_date)
  ...         return r.sharpe_ratio() or -999.0

  >>> data_dict = _build_gp_data_dict(ohlcv_df)
  >>> engine = SigGPEngine(
  ...     data=data_dict,
  ...     fitness_calculator=SigFitness(ohlcv_df),
  ...     seed_expressions=["ts_mean(CLOSE, 20) > ts_mean(CLOSE, 50)"],
  ...     random_seed=42,
  ... ).run(verbose=True)
  >>> best = engine.best()
  >>> print(best.expression_str, best.fitness)
=============================================================================
"""
from utils.gp.v5.engine import GPEngine as _BaseGPEngine
from utils.gp.v5.config import TreeGenConfig, Individual
from ..expression import _SignalFromAST


class SigGPEngine(_BaseGPEngine):
    """信号端 GP 引擎 — 注入 _SignalFromAST evaluator

    继承 utils.gp.v5.GPEngine 核心 (进化循环/选择/交叉/变异/方向追踪/缓存)，
    自动注入信号端 evaluator，将 AST 树 → 1D 信号序列。

    构造参数同基类 GPEngine:
      data: Dict[str, np.ndarray] — 由 _build_gp_data_dict(ohlcv_df) 构建
      fitness_calculator: 必须有 .compute(signal_values) → float 方法
      seed_expressions: 种子表达式列表
      config: 引擎级参数字典 (覆盖 DEFAULT_GP_CONFIG)
      tree_gen_config: TreeGenConfig 权重聚焦配置
      random_seed: 随机种子 (可复现)
    """

    def __init__(self, *args, _no_auto_eval=False, **kwargs):
        # auto-inject signal evaluator (can be overridden)
        if not _no_auto_eval and 'evaluator' not in kwargs:
            kwargs['evaluator'] = lambda data, tree: _SignalFromAST(tree).evaluate(data)
        super().__init__(*args, **kwargs)


# 重新导出 (保持 API 兼容)
__all__ = ['SigGPEngine', 'TreeGenConfig', 'Individual', '_SignalFromAST']
