"""
因子组合器抽象层

设计思路：
--------
1. FactorCombiner(ABC)：因子组合抽象基类，将多个因子值 DataFrame 合并为一个
2. EqualWeightCombiner：等权组合，所有因子权重相同
3. FixedWeightCombiner：固定权重组合，GA/Bayesian 优化后的静态权重
4. ExpandingICCombiner：Expanding IC 动态加权，逐日计算权重（无前瞻偏差）

关键约束：
--------
组合过程必须无前瞻偏差——t 时刻的组合权重只能用 t 及之前的信息。

与 pipeline 的关系：
----------------
Combiner 在 pipeline 外部完成因子组合，输出单个组合因子 DataFrame。
pipeline.evaluate() 只接收单因子输入，职责保持单一。

[新增] 2026-05-22 从各业务脚本中抽取因子组合逻辑，统一抽象。
旧问题：组合逻辑散落在 4 个脚本中，IC 加权取 iloc[-1] 导致前瞻偏差，
等权/IC 加权/固定权重的接口不统一，无法独立单测。
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


def cross_section_zscore(fv: pd.DataFrame) -> pd.DataFrame:
    """截面 z-score 标准化

    每个交易日做一次截面 z-score：(x - mean) / std。
    std=0 时保留 0（全相同因子的截面无区分度，不应产生极端值）。

    [新增] 2026-05-22 从各业务脚本中提取，统一实现。
    旧问题：4 个文件各有一份相同实现，维护困难。
    """
    mean = fv.mean(axis=1)
    std = fv.std(axis=1)
    std_safe = std.replace(0, 1.0)
    return fv.sub(mean, axis=0).div(std_safe, axis=0)


class FactorCombiner(ABC):
    """因子组合器抽象基类

    将多个因子值 DataFrame 合并为一个组合因子 DataFrame。
    所有无状态组合逻辑统一继承此类，确保接口一致、可独立单测。

    约定：
    - 输入的 factor_values_list 中每个 fv 应已做截面标准化
    - 输出的组合因子值 DataFrame 的 index/columns 与输入一致
    - 子类 combine() 必须保证无前瞻偏差
    """

    @abstractmethod
    def combine(self, factor_values_list: List[Tuple[str, pd.DataFrame]],
                returns: pd.DataFrame = None) -> pd.DataFrame:
        """组合多个因子

        Args:
            factor_values_list: [(name, fv)] 各因子值 DataFrame（已做截面标准化）
            returns: 日收益率 DataFrame，IC 类组合器需要，等权/固定权重可不传

        Returns:
            pd.DataFrame: 组合后的因子值
        """
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class EqualWeightCombiner(FactorCombiner):
    """等权组合

    所有因子权重相同，简单平均。
    适用于初步探索因子组合效果，或无先验信息时的基准方案。
    """

    def combine(self, factor_values_list: List[Tuple[str, pd.DataFrame]],
                returns: pd.DataFrame = None) -> pd.DataFrame:
        n = len(factor_values_list)
        if n == 0:
            raise ValueError("factor_values_list 不能为空")
        if n == 1:
            return factor_values_list[0][1].copy()

        return sum(fv for _, fv in factor_values_list) / n

    def __repr__(self) -> str:
        return "EqualWeightCombiner()"


class FixedWeightCombiner(FactorCombiner):
    """固定权重组合

    每个因子使用预设的固定权重，适用于 GA/Bayesian 优化后已知最优权重的场景。
    权重在构造时确定，不随时间变化，无前瞻偏差风险（权重来自优化结果，
    但优化结果本身如果用全样本则有偏差，那是优化阶段的问题而非组合器的问题）。

    典型用法：
    >>> combiner = FixedWeightCombiner(weights={'Reversal': 0.6, 'MomRate': 0.4})
    >>> combined = combiner.combine([('Reversal', fv1), ('MomRate', fv2)])
    """

    def __init__(self, weights: Dict[str, float]):
        if not weights:
            raise ValueError("weights 不能为空")
        total = sum(weights.values())
        if total <= 0:
            raise ValueError("权重之和必须 > 0")
        self.weights = {k: v / total for k, v in weights.items()}

    def combine(self, factor_values_list: List[Tuple[str, pd.DataFrame]],
                returns: pd.DataFrame = None) -> pd.DataFrame:
        if not factor_values_list:
            raise ValueError("factor_values_list 不能为空")

        combined = None
        for name, fv in factor_values_list:
            w = self.weights.get(name, 0.0)
            if w == 0.0:
                continue
            if combined is None:
                combined = fv * w
            else:
                combined = combined + fv * w

        if combined is None:
            raise ValueError("所有因子权重为 0，无法组合")

        return combined

    def __repr__(self) -> str:
        w_str = ', '.join(f'{k}={v:.3f}' for k, v in self.weights.items())
        return f"FixedWeightCombiner({w_str})"


class ExpandingICCombiner(FactorCombiner):
    """Expanding IC 动态加权（无前瞻偏差）

    核心逻辑：
    - 对每个交易日 t，计算各因子截至 t 日的 expanding mean Rank IC
    - 用 t 日的 IC 值作为当日组合权重（IC<0 的因子权重为 0）
    - expanding IC 在 min_periods 之前为 NaN → 退化为等权

    与旧版 compute_expanding_ic_weights 的区别：
    - 旧版：取 iloc[-1] 末尾值作为全期间固定权重 → 前瞻偏差（P0 bug）
    - 新版：逐日取 expanding IC[t] 作为当日权重 → 无前瞻偏差

    [新增] 2026-05-22 替代旧版 compute_expanding_ic_weights，修复 P0 前瞻偏差。
    """

    def __init__(self, min_periods: int = 60):
        if min_periods < 1:
            raise ValueError(f"min_periods 必须 >= 1，当前值: {min_periods}")
        self.min_periods = min_periods

    def combine(self, factor_values_list: List[Tuple[str, pd.DataFrame]],
                returns: pd.DataFrame = None) -> pd.DataFrame:
        if not factor_values_list:
            raise ValueError("factor_values_list 不能为空")

        n_factors = len(factor_values_list)

        if n_factors == 1:
            return factor_values_list[0][1].copy()

        if returns is None:
            raise ValueError("ExpandingICCombiner 需要 returns 参数来计算 IC")

        # [修复] 2026-05-22 逐日计算 expanding IC 权重，替代旧版 iloc[-1] 固定权重
        # 旧版问题：取 expanding IC 末尾值作为全期间权重，包含未来信息（前瞻偏差）
        # 新版做法：每个交易日取当日的 expanding IC 值作为权重，仅用历史数据
        ic_weight_df = self._compute_time_varying_ic_weights(factor_values_list, returns)

        common_dates = ic_weight_df.index
        common_symbols = factor_values_list[0][1].columns

        combined = pd.DataFrame(0.0, index=common_dates, columns=common_symbols)
        for j, (name, fv) in enumerate(factor_values_list):
            w_col = ic_weight_df.columns[j]
            # 逐行加权：combined[t] += weight[t] * fv[t]
            for t in common_dates:
                if t in fv.index:
                    combined.loc[t] += ic_weight_df.loc[t, w_col] * fv.loc[t]

        return combined

    def _compute_time_varying_ic_weights(self,
                                          factor_values_list: List[Tuple[str, pd.DataFrame]],
                                          returns: pd.DataFrame) -> pd.DataFrame:
        """计算逐日 expanding IC 权重

        对每个交易日 t：
        1. 计算各因子在 t 日的 Rank IC（因子值 vs 次日收益率的截面排名相关系数）
        2. 对 IC 序列做 expanding mean（min_periods 之前为 NaN）
        3. 取 expanding_mean[t] 作为该因子在 t 日的 IC 估计值
        4. IC < 0 的因子权重设为 0，正 IC 归一化到和为 1
        5. 所有因子 IC <= 0 时退化为等权

        Args:
            factor_values_list: [(name, fv)] 各因子值
            returns: 日收益率

        Returns:
            pd.DataFrame: 逐日权重，columns = 各因子名，index = 共同日期
        """
        future_returns = returns.shift(-1)
        n_factors = len(factor_values_list)

        # 计算每个因子的逐日 IC 和 expanding mean IC
        expanding_ic_list = []
        all_dates = None

        for name, fv in factor_values_list:
            common = fv.index.intersection(future_returns.index)
            cols = fv.columns.intersection(future_returns.columns)
            if len(common) < self.min_periods or len(cols) < 3:
                expanding_ic_list.append(None)
                continue

            fv_sub = fv.loc[common, cols]
            fr_sub = future_returns.loc[common, cols]

            daily_ics = []
            dates = []
            for date in common:
                row_fv = fv_sub.loc[date].dropna()
                row_fr = fr_sub.loc[date].dropna()
                common_idx = row_fv.index.intersection(row_fr.index)
                if len(common_idx) < 3:
                    daily_ics.append(np.nan)
                    dates.append(date)
                    continue
                ic = row_fv.loc[common_idx].rank().corr(
                    row_fr.loc[common_idx].rank(), method='pearson'
                )
                daily_ics.append(ic)
                dates.append(date)

            ic_s = pd.Series(daily_ics, index=dates)
            expanding_mean_ic = ic_s.expanding(min_periods=self.min_periods).mean()
            expanding_ic_list.append(expanding_mean_ic)

            if all_dates is None:
                all_dates = common
            else:
                all_dates = all_dates.intersection(common)

        # 构建逐日权重矩阵
        weight_records = []
        for t in all_dates:
            weights_at_t = []
            for j, ic_s in enumerate(expanding_ic_list):
                if ic_s is not None and t in ic_s.index and not np.isnan(ic_s.loc[t]):
                    val = ic_s.loc[t]
                    weights_at_t.append(max(val, 0.0))
                else:
                    weights_at_t.append(0.0)

            total = sum(weights_at_t)
            if total <= 0:
                # 所有因子 IC <= 0 或 expanding 尚未满 min_periods → 退化为等权
                weights_at_t = [1.0 / n_factors] * n_factors
            else:
                weights_at_t = [w / total for w in weights_at_t]

            weight_records.append(weights_at_t)

        factor_names = [name for name, _ in factor_values_list]
        weight_df = pd.DataFrame(weight_records, index=all_dates, columns=factor_names)
        return weight_df

    def __repr__(self) -> str:
        return f"ExpandingICCombiner(min_periods={self.min_periods})"
