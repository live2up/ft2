"""
factor/v4/industry_fitness.py — 行业轮动自适应适应度

核心设计:
  N < 50:   Pipeline Sharpe做fitness (IC不可信, 跳过IC门控)
  N 50~100: 宽松IC门控 + Pipeline Sharpe (IC半可信)
  N ≥ 100:  WQB风格IC门控 + Pipeline Sharpe (IC可信)

Sharpe计算: 使用 ft2.core EngineCore (fast 模式) 替代 FactorPipeline
  - 日期+标的交集对齐、NaN→0、逐日累乘净值、√252年化
  - 调仓周期由scheduler/rebalance参数控制

IC门控: 根据截面宽度N自适应
  N < 50:   跳过(IC统计量噪声大, 会误杀好因子)
  N 50~100: HR + 分年度稳定性(宽松)
  N ≥ 100:  ICIR + HR + 分年度稳定性(严格, WQB风格)

[新增] 2026-06-05
[重构] 2026-06-18 FactorPipeline → EngineCore.backtest(mode='fast')
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

# [迁移] 2026-06-18 从 discover.py 迁入 FitnessCalculator + _compute_daily_ics (discover.py 已删除)


class FitnessCalculator:
    """适应度计算器基类

    [迁移] 2026-06-18 从 discover.py 迁入，仅保留 industry_fitness 所需的基类。
    """

    def __init__(self, data: Dict[str, np.ndarray], future_returns: pd.DataFrame,
                 returns: pd.DataFrame = None, cost_rate: float = 0.0,
                 scheduler=None, allocator=None):
        self.data = data
        self.future_returns = future_returns
        self.returns = returns
        self.cost_rate = cost_rate
        self.scheduler = scheduler
        self.allocator = allocator
        self._shape = self._infer_shape(data)

    @staticmethod
    def _infer_shape(data) -> Tuple[int, int]:
        for arr in data.values():
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                return arr.shape
        return (1, 1)

    def compute(self, factor_values: np.ndarray) -> float:
        raise NotImplementedError

    def _validate(self, factor_values: np.ndarray) -> bool:
        if np.all(np.isnan(factor_values)):
            return False
        if np.nanstd(factor_values) < 1e-10:
            return False
        return True


def _compute_daily_ics(factor_values: np.ndarray, future_returns: pd.DataFrame
                       ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """计算日频 IC 序列 + 有效日索引

    [迁移] 2026-06-18 从 discover.py 迁入
    """
    T = factor_values.shape[0]
    daily_ics = []
    valid_indices = []
    for t in range(T):
        fv = factor_values[t, :]
        rv = future_returns.iloc[t].values
        mask = ~np.isnan(fv) & ~np.isnan(rv)
        if mask.sum() < 5:
            continue
        corr = np.corrcoef(fv[mask], rv[mask])[0, 1]
        if not np.isnan(corr):
            daily_ics.append(corr)
            valid_indices.append(t)
    if len(daily_ics) < 30:
        return None, None
    return np.array(daily_ics), np.array(valid_indices, dtype=int)

logger = logging.getLogger(__name__)


class IndustryFitness(FitnessCalculator):
    """行业轮动自适应适应度

    根据截面宽度N自适应选择评估策略:
      N < 50:   IC统计量噪声大 → 跳过IC门控, 直接Pipeline Sharpe
      N 50~100: IC半可信       → 宽松IC门控 + Pipeline Sharpe
      N ≥ 100:  IC可信         → WQB风格门控 + Pipeline Sharpe

    Sharpe计算:
      直接复用FactorPipeline.evaluate(), 与手动回测结果一致。
      调仓周期由scheduler参数控制, 默认月末(FixedScheduler('ME'))。

    参数:
      top_n:           选Top N个行业持有, 默认3
      scheduler:       调度器, 默认None(自动创建FixedScheduler('ME'))
                       支持: FixedScheduler('ME'/'W'), IntervalScheduler(N天)
      allocator:       分配器, 默认None(自动创建TopNEqualWeight(top_n))
      min_hr:          IC门控HR阈值(N≥50时生效), 默认0.5
      min_yearly:      IC门控分年度稳定性(N≥50时生效), 默认0.5
      icir_threshold:  IC门控绝对阈值(N≥100时生效), 默认0.15
      cost_rate:       交易费率, 默认0.0(纯信号测试)

    用法:
      # 默认: 月末调仓 + Top3等权
      IndustryFitness(panel, returns, returns)

      # 周末调仓
      IndustryFitness(panel, returns, returns, scheduler=FixedScheduler('W'))

      # 5天调仓
      IndustryFitness(panel, returns, returns, scheduler=IntervalScheduler(5))

      # Top5行业
      IndustryFitness(panel, returns, returns, top_n=5)

      # GPEngine中通过FitnessMode使用
      gp = GPEngine(..., fitness_mode=FitnessMode.INDUSTRY,
                    scheduler=FixedScheduler('W'), ...)
    """

    def __init__(self, data: Dict[str, np.ndarray],
                 future_returns: pd.DataFrame,
                 returns: pd.DataFrame = None,
                 cost_rate: float = 0.0,
                 scheduler=None, allocator=None,
                 # Industry特有参数
                 top_n: int = 3,
                 min_hr: float = 0.5,
                 min_yearly: float = 0.5,
                 icir_threshold: float = 0.15):
        super().__init__(data, future_returns, returns, cost_rate,
                         scheduler, allocator)
        self.top_n = top_n
        self.min_hr = min_hr
        self.min_yearly = min_yearly
        self.icir_threshold = icir_threshold

        self._N = self._shape[1]  # 截面宽度

    def _pipeline_sharpe(self, factor_values: np.ndarray) -> float:
        """EngineCore fast Sharpe — 替代 FactorPipeline

        [重构] 2026-06-18 FactorPipeline → EngineCore.backtest(mode='fast')
        """
        if self.returns is None:
            return -999.0

        T, N = factor_values.shape
        try:
            from .engine import EngineCore
            fv_df = pd.DataFrame(factor_values,
                                 index=self.returns.index[:T],
                                 columns=self.returns.columns[:N])
            rebalance = self.scheduler if self.scheduler is not None else 'ME'
            _top_n = getattr(self.allocator, 'top_n', self.top_n) if self.allocator is not None else self.top_n
            analyzer = EngineCore.backtest(fv_df, returns=self.returns,
                                           top_n=_top_n, rebalance=rebalance, mode='fast')
            return float(analyzer.sharpe_ratio())
        except Exception as e:
            logger.debug(f"IndustryFitness EngineCore Sharpe失败: {e}")
            return -999.0

    def _ic_gate(self, factor_values: np.ndarray) -> Tuple[bool, Dict[str, float]]:
        """IC门控(N ≥ 50时生效)

        返回: (pass, stats_dict)
        """
        daily_ics, valid_indices = _compute_daily_ics(factor_values, self.future_returns)

        if daily_ics is None:
            return False, {'icir': 0.0, 'hr': 0.0, 'yearly_stability': 0.0}

        ic_mean = float(np.mean(daily_ics))
        ic_std = float(np.std(daily_ics, ddof=1))
        icir = ic_mean / ic_std if ic_std > 1e-10 else 0.0
        hr = float(np.mean(daily_ics > 0))

        # 分年度稳定性
        yearly_stability = 0.5
        if (self.future_returns is not None and
                hasattr(self.future_returns, 'index') and
                hasattr(self.future_returns.index, 'year')):
            valid_years = np.asarray(self.future_returns.index[valid_indices].year)
            yearly_means = []
            for yr in sorted(set(valid_years)):
                yr_ics = daily_ics[valid_years == yr]
                if len(yr_ics) > 0:
                    yearly_means.append(float(np.mean(yr_ics)))
            if yearly_means:
                main_dir = np.sign(ic_mean)
                yearly_stability = float(np.mean(
                    [1.0 for ym in yearly_means if np.sign(ym) == main_dir]))

        stats = {'icir': icir, 'hr': hr, 'yearly_stability': yearly_stability}

        # 门控条件(根据N自适应)
        if self._N >= 100:
            # IC可信: WQB风格严格门控
            if abs(icir) < self.icir_threshold:
                return False, stats
            if hr < self.min_hr:
                return False, stats
            if yearly_stability < self.min_yearly:
                return False, stats
        elif self._N >= 50:
            # IC半可信: 只卡HR(方向稳定性), 放宽ICIR
            if hr < self.min_hr:
                return False, stats
            if yearly_stability < 0.4:  # 更宽松
                return False, stats
        else:
            # N < 50: IC不可信, 跳过门控
            pass

        return True, stats

    def compute(self, factor_values: np.ndarray) -> float:
        """主入口: 自适应评估

        N < 50:   废因子过滤 → 快速Sharpe
        N 50~100: IC门控(宽松) → 快速Sharpe
        N ≥ 100:  IC门控(严格) → 快速Sharpe

        注意: factor_values必须与self.returns行列对齐。
        GPEngine内部evaluate时, factor_values的shape=(T, N)与data一致,
        所以不需要额外对齐。外部调用batch_evaluate时也需要保证对齐。
        """
        if not self._validate(factor_values):
            return -999.0

        # IC门控(N ≥ 50时)
        if self._N >= 50:
            passed, stats = self._ic_gate(factor_values)
            if not passed:
                return -999.0

        # Pipeline Sharpe
        try:
            return self._pipeline_sharpe(factor_values)
        except Exception as e:
            logger.debug(f"IndustryFitness 快速Sharpe失败: {e}")
            return -999.0

    @property
    def mode_label(self) -> str:
        """当前评估模式标签"""
        if self._N >= 100:
            return f"INDUSTRY[strict](N={self._N})"
        elif self._N >= 50:
            return f"INDUSTRY[loose](N={self._N})"
        else:
            return f"INDUSTRY[sharpe_only](N={self._N})"

    def batch_evaluate(self, factor_dict: Dict[str, np.ndarray],
                       dates=None, returns=None) -> pd.DataFrame:
        """批量评估多个因子, 返回四维评估表

        Args:
            factor_dict: {因子名: (T, N) ndarray}
            dates: 日期索引(可选, 默认用self.returns.index)
            returns: 收益率(可选, 默认用self.returns)

        Returns:
            DataFrame: 每行一个因子, 列=评估指标
        """
        import pandas as pd
        from scipy import stats as sp_stats

        if returns is None:
            returns = self.returns
        if dates is None:
            dates = returns.index

        rows = []
        for name, fv in factor_dict.items():
            if fv.shape != self._shape:
                continue

            row = {'name': name}

            # IC统计
            daily_ics, valid_indices = _compute_daily_ics(fv, self.future_returns)
            if daily_ics is not None:
                ic_mean = float(np.mean(daily_ics))
                ic_std = float(np.std(daily_ics, ddof=1))
                row['ic_mean'] = ic_mean
                row['icir'] = ic_mean / ic_std if ic_std > 1e-10 else 0.0
                row['hr'] = float(np.mean(daily_ics > 0))

                # 分年度
                if (self.future_returns is not None and
                        hasattr(self.future_returns, 'index') and
                        hasattr(self.future_returns.index, 'year')):
                    valid_years = np.asarray(self.future_returns.index[valid_indices].year)
                    yearly_means = []
                    for yr in sorted(set(valid_years)):
                        yr_ics = daily_ics[valid_years == yr]
                        if len(yr_ics) > 0:
                            yearly_means.append(float(np.mean(yr_ics)))
                    if yearly_means:
                        main_dir = np.sign(ic_mean)
                        row['yearly_stability'] = float(np.mean(
                            [1.0 for ym in yearly_means if np.sign(ym) == main_dir]))

            # [修复] 2026-06-18 原引用不存在的 _quick_sharpe，改用 _pipeline_sharpe
            row['quick_sharpe'] = self._pipeline_sharpe(fv)

            rows.append(row)

        return pd.DataFrame(rows).sort_values('quick_sharpe', ascending=False)
