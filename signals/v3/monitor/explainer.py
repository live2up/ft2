"""
signals/v2/explainer.py — 白盒子解释引擎
=============================================================================

对任意信号的择时表现进行结构化解释，生成人类可读的分析报告。


============================================================================
                         架构层级（竖式）
============================================================================

 ┌─────────────────────────────────────────────────────────────────────┐
 │  输入：Expression + FeatureSpace + OHLCV DataFrame                   │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 1  —  市场状态分割                                              │
 │                                                                      │
 │  三重维度交叉分割：                                                    │
 │    · 波动率:  高 / 中 / 低                                            │
 │    · 趋势:    强 / 弱                                                 │
 │    · 成交量:  放量 / 缩量                                             │
 │                                                                      │
 │  组合: 3 × 2 × 2 = 12 种市场状态                                      │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 2  —  逐状态绩效分析 (RegimePerformance)                        │
 │                                                                      │
 │  每个状态: n_days / pct_of_total / win_rate / avg_return            │
 │  判断信号在该状态下是否「有效 / 中性 / 无效」                          │
 └────────────────────┬────────────────────────────────────────────────┘
                      │
                      ▼
 ┌─────────────────────────────────────────────────────────────────────┐
 │  Step 3  —  生成解释报告 (ExplanationReport)                          │
 │                                                                      │
 │  · 信号整体有效性评分                                                 │
 │  · 最佳 / 最差市场状态                                                │
 │  · 可操作的改进建议 (调整阈值 / 增加过滤条件)                          │
 │  · 人类可读的解释文本                                                 │
 └─────────────────────────────────────────────────────────────────────┘

============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field

from ..expression import Expression, TreeNode


@dataclass
class RegimePerformance:
    """单一市场状态下的信号表现"""
    name: str
    n_days: int
    pct_of_total: float
    ic: float
    mean_return_5d: float
    signal_correct_rate: float

    @property
    def is_effective(self) -> bool:
        return abs(self.ic) > 0.05

    @property
    def effectiveness_label(self) -> str:
        if abs(self.ic) > 0.15:
            return '显著有效'
        elif abs(self.ic) > 0.08:
            return '有效'
        elif abs(self.ic) > 0.04:
            return '弱有效'
        else:
            return '无效'


@dataclass
class ExplanationReport:
    """完整解释报告"""
    expression_name: str
    expression_str: str
    features_used: List[str]
    overall_ic: float
    overall_annual_return: float
    overall_sharpe: float
    overall_max_dd: float
    overall_win_rate: float

    regime_performances: List[RegimePerformance]
    best_regime: Optional[RegimePerformance] = None
    worst_regime: Optional[RegimePerformance] = None

    failure_conditions: List[str] = field(default_factory=list)
    success_conditions: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

    @property
    def text(self) -> str:
        return self._build_text()

    def _build_text(self) -> str:
        lines = [
            f"=== {self.expression_name} 信号解释 ===",
            f"表达式: {self.expression_str}",
            f"使用特征: {', '.join(self.features_used)}",
            "",
            f"总体表现: IC={self.overall_ic:.4f} 年化={self.overall_annual_return:.1f}% 夏普={self.overall_sharpe:.2f} 回撤={self.overall_max_dd:.1f}%",
            "",
            "--- 市场状态分解 ---",
        ]
        for rp in self.regime_performances:
            lines.append(
                f"  [{rp.effectiveness_label}] {rp.name}: "
                f"IC={rp.ic:.4f} 日均={rp.mean_return_5d:.3f}% "
                f"天数={rp.n_days}({rp.pct_of_total:.0f}%)"
            )
        if self.best_regime:
            lines.append(f"\n  最佳环境: {self.best_regime.name}")
        if self.worst_regime:
            lines.append(f"  最差环境: {self.worst_regime.name}")

        if self.success_conditions:
            lines.append("\n--- 成功条件 ---")
            for c in self.success_conditions:
                lines.append(f"  ✓ {c}")
        if self.failure_conditions:
            lines.append("\n--- 失效条件 ---")
            for c in self.failure_conditions:
                lines.append(f"  ✗ {c}")
        if self.improvement_suggestions:
            lines.append("\n--- 改进建议 ---")
            for s in self.improvement_suggestions:
                lines.append(f"  → {s}")

        return '\n'.join(lines)

    def __repr__(self):
        return self.text


class Explainer:
    """
    白盒子解释器

    将信号在不同市场状态下的表现进行分解，
    解释信号何时有效、何时失效，并给出改进建议。
    """

    def __init__(self, feature_space=None):
        self._fs = feature_space

    def explain(self, expression: str, data: pd.DataFrame,
                forward_period: int = 5,
                target_col: Optional[str] = None,
                feature_space=None) -> ExplanationReport:
        """
        解释一个表达式的择时表现

        Args:
            expression: 表达式字符串，如 "RSI(14)"
            data: OHLCV DataFrame, 需含 forward_return_Nd 列
            forward_period: 预测天数
            target_col: 目标列名，默认 f'forward_return_{forward_period}d'
            feature_space: FeatureSpace，若不传则用构造时的

        Returns:
            ExplanationReport 含结构化解译
        """
        fs = feature_space or self._fs
        if fs is None:
            raise ValueError("需要提供 FeatureSpace")

        if target_col is None:
            target_col = f'forward_return_{forward_period}d'

        if target_col not in data.columns:
            data = data.copy()
            data[target_col] = data['close'].shift(-forward_period) / data['close'] - 1

        expr = Expression(expression, feature_space=fs)
        signal = expr.generate(data).reindex(data.index)
        signal_clean = signal.fillna(signal.median() if signal.notna().any() else 0)

        target = data[target_col].reindex(signal.index)

        overall_ic = signal_clean.corr(target) if len(signal_clean) > 10 else 0.0

        daily_ret = data['close'].pct_change().reindex(signal.index)
        positions = (signal_clean > signal_clean.median()).astype(float)
        strategy_ret = daily_ret * positions.shift(1).fillna(0)
        nav = (1 + strategy_ret.fillna(0)).cumprod()
        total = nav.iloc[-1] - 1
        years = max((nav.index[-1] - nav.index[0]).days / 365.25, 0.1)
        ann = (1 + total) ** (1 / years) - 1 if years > 0 else 0
        vol = strategy_ret.std() * np.sqrt(252)
        sharpe = ann / vol if vol > 0 else 0
        dd = (nav / nav.cummax() - 1).min()
        win_rate = (strategy_ret > 0).mean()

        regimes = self._segment_regimes(data, signal)
        regime_perfs = self._evaluate_regimes(regimes, signal_clean, target)

        best = max(regime_perfs, key=lambda r: abs(r.ic)) if regime_perfs else None
        worst = min(regime_perfs, key=lambda r: abs(r.ic)) if regime_perfs else None

        failure_conds, success_conds = self._derive_conditions(regime_perfs)
        suggestions = self._derive_suggestions(regime_perfs, expression)

        return ExplanationReport(
            expression_name=expression,
            expression_str=expression,
            features_used=expr.features_used,
            overall_ic=overall_ic,
            overall_annual_return=ann * 100,
            overall_sharpe=sharpe,
            overall_max_dd=dd * 100,
            overall_win_rate=win_rate * 100,
            regime_performances=regime_perfs,
            best_regime=best,
            worst_regime=worst,
            failure_conditions=failure_conds,
            success_conditions=success_conds,
            improvement_suggestions=suggestions,
        )

    def compare(self, expressions: List[str], data: pd.DataFrame,
                feature_space=None) -> List[ExplanationReport]:
        """批量解释多个表达式"""
        reports = []
        for expr in expressions:
            report = self.explain(expr, data, feature_space=feature_space)
            reports.append(report)
        return reports

    def _segment_regimes(self, data: pd.DataFrame,
                         signal: pd.Series) -> Dict[str, pd.Series]:
        """将数据按市场状态分割"""
        regimes = {}
        close = data['close'].reindex(signal.index)
        volume = data.get('volume', pd.Series(1, index=signal.index)).reindex(signal.index)

        # 波动率分段: 基于20日振幅
        ret_20d = close.pct_change(20).abs()
        if ret_20d.notna().sum() > 30:
            q33 = ret_20d.quantile(0.33)
            q67 = ret_20d.quantile(0.67)
            regimes['高波动(振幅>67%分位)'] = ret_20d >= q67
            regimes['中波动'] = (ret_20d >= q33) & (ret_20d < q67)
            regimes['低波动(振幅<33%分位)'] = ret_20d < q33

        # 趋势分段: 基于40日均线偏离
        ma40 = close.rolling(40).mean()
        if ma40.notna().sum() > 50:
            trend_dev = (close - ma40) / ma40 * 100
            regimes['上升趋势(价格>>MA40)'] = trend_dev > 3
            regimes['震荡(价格≈MA40)'] = (trend_dev >= -3) & (trend_dev <= 3)
            regimes['下跌趋势(价格<<MA40)'] = trend_dev < -3

        # 成交量分段
        vol_ma20 = volume.rolling(20).mean()
        if vol_ma20.notna().sum() > 30:
            vol_ratio = volume / vol_ma20
            regimes['放量(成交>1.3倍均量)'] = vol_ratio > 1.3
            regimes['正常量'] = (vol_ratio >= 0.7) & (vol_ratio <= 1.3)
            regimes['缩量(成交<0.7倍均量)'] = vol_ratio < 0.7

        return regimes

    def _evaluate_regimes(self, regimes: Dict[str, pd.Series],
                          signal: pd.Series, target: pd.Series
                          ) -> List[RegimePerformance]:
        """计算各状态下信号表现"""
        results = []
        n_total = signal.notna().sum()
        for name, mask in regimes.items():
            idx = mask[mask].index.intersection(signal.index).intersection(target.index)
            if len(idx) < 20:
                continue
            sig_sub = signal.loc[idx]
            tgt_sub = target.loc[idx]
            ic = sig_sub.corr(tgt_sub) if len(sig_sub) > 10 else 0.0
            pred_dir = (sig_sub > sig_sub.median()).astype(int)
            actual_dir = (tgt_sub > 0).astype(int)
            correct = (pred_dir == actual_dir).mean()
            results.append(RegimePerformance(
                name=name,
                n_days=len(idx),
                pct_of_total=len(idx) / max(n_total, 1) * 100,
                ic=ic,
                mean_return_5d=tgt_sub.mean() * 100,
                signal_correct_rate=correct * 100,
            ))
        return results

    def _derive_conditions(self, regime_perfs: List[RegimePerformance]
                           ) -> Tuple[List[str], List[str]]:
        """提取失效和成功条件"""
        failures = []
        successes = []
        for rp in regime_perfs:
            if rp.is_effective and abs(rp.ic) > 0.08:
                direction = '正相关' if rp.ic > 0 else '负相关'
                successes.append(f"{rp.name}时信号有效 (IC={rp.ic:.3f}, {direction})")
            elif not rp.is_effective and rp.n_days > 50:
                failures.append(f"{rp.name}时信号接近失效 (IC={rp.ic:.3f})")
        return failures, successes

    def _derive_suggestions(self, regime_perfs: List[RegimePerformance],
                            expression: str) -> List[str]:
        """生成改进建议"""
        suggestions = []
        n_effective = sum(1 for r in regime_perfs if r.is_effective)
        if n_effective == 0:
            suggestions.append(f"该信号在所有市场状态下均不显著，建议替换或重新设计")
        elif n_effective < len(regime_perfs) * 0.5:
            suggestions.append(
                f"信号仅在{len([r for r in regime_perfs if r.is_effective])}/{len(regime_perfs)}种市场有效，"
                f"建议加入市场状态过滤"
            )

        weak_vol = [r for r in regime_perfs if '高波动' in r.name and not r.is_effective]
        if weak_vol:
            suggestions.append("在高波动时期信号失效，建议加入波动率门控(提高入场阈值)")

        weak_vol_low = [r for r in regime_perfs if '低波动' in r.name and not r.is_effective]
        if weak_vol_low:
            suggestions.append("在低波动时期信号失效，建议在低波时降低仓位")

        weak_trend = [r for r in regime_perfs if '下跌' in r.name and not r.is_effective]
        if weak_trend:
            suggestions.append("在下跌趋势中信号失效，建议加入趋势过滤器(仅在上行/震荡时使用)")

        vol_related = [r for r in regime_perfs if '放量' in r.name and r.is_effective]
        if vol_related:
            suggestions.append("放量环境信号有效，建议加入成交量确认(仅在放量时执行)")

        if not suggestions:
            suggestions.append("当前信号在各市场状态表现均衡，可维持现策略")

        return suggestions


def explain_signal(expression: str, data: pd.DataFrame,
                   feature_space=None,
                   forward_period: int = 5) -> ExplanationReport:
    """
    快捷函数: 解释一个表达式信号

    用法:
        report = explain_signal("RSI(14)", df, fs)
        print(report.text)
    """
    explainer = Explainer(feature_space)
    return explainer.explain(expression, data, forward_period=forward_period)
