"""
signals/v5/wf_result.py — Walk-Forward 结果容器 (v5 独立版)
=============================================================================
[重构] 2026-07-07 从 v4 升级到 v5, 无 v3 依赖。
=============================================================================
"""
import numpy as np
import pandas as pd
from typing import Dict, List
from dataclasses import dataclass, field

# [优化] 2026-06-22 直接使用 AccountAnalyzer 的 @metric 方法名，不重复包装
METRICS = [
    'sharpe_ratio', 'return_rate', 'annualized_return', 'volatility',
    'max_drawdown', 'sortino_ratio', 'win_rate', 'avg_holding_period',
    'var', 'cvar', 'kelly_criterion',
]


@dataclass
class WalkForwardResult:
    """ft2.core 引擎版 Walk-Forward 验证结果 (v5)"""

    windows: List[Dict] = field(default_factory=list)
    summary: Dict = field(default_factory=dict)
    label: str = ""

    @property
    def train_sharpes(self) -> List[float]:
        return [w.get('train', {}).get('sharpe_ratio', 0) or 0 for w in self.windows]

    @property
    def test_sharpes(self) -> List[float]:
        return [w.get('test', {}).get('sharpe_ratio', 0) or 0 for w in self.windows]

    @property
    def stability_score(self) -> float:
        """夏普稳定性 = mean / sigma"""
        sharpes = self.test_sharpes
        if len(sharpes) < 2:
            return 0.0
        mu = np.mean(sharpes)
        sigma = np.std(sharpes, ddof=1)
        return float(mu / sigma) if sigma > 1e-10 else 0.0

    @property
    def negative_count(self) -> int:
        return sum(1 for s in self.test_sharpes if s < 0)

    @property
    def overfit_ratio(self) -> float:
        """过拟合比 = mean(train_sharpe) / mean(test_sharpe)"""
        t_mean = np.mean(self.test_sharpes) if self.test_sharpes else 0
        tr_mean = np.mean(self.train_sharpes) if self.train_sharpes else 0
        return float(tr_mean / t_mean) if abs(t_mean) > 1e-10 else float('inf')

    def to_dataframe(self) -> pd.DataFrame:
        """窗口级别指标表"""
        rows = []
        for w in self.windows:
            row = {
                'train_start': w.get('train_start', ''),
                'train_end': w.get('train_end', ''),
                'test_start': w.get('test_start', ''),
                'test_end': w.get('test_end', ''),
            }
            for scope in ('train', 'test'):
                metrics = w.get(scope, {})
                for m in METRICS:
                    val = metrics.get(m)
                    if val is not None:
                        row[f'{scope}_{m}'] = round(val, 4)
            rows.append(row)
        return pd.DataFrame(rows)

    def stability_report(self) -> Dict:
        """窗口间指标稳定性报告"""
        report = {'n_windows': len(self.windows)}
        for scope in ('train', 'test'):
            for m in METRICS:
                vals = [w.get(scope, {}).get(m) for w in self.windows]
                vals = [v for v in vals if v is not None]
                if len(vals) >= 2:
                    report[f'{scope}_{m}_mean'] = float(np.mean(vals))
                    report[f'{scope}_{m}_std'] = float(np.std(vals, ddof=1))
        return report
