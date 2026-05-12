"""
signals/v2/pysr_adapter.py - PySR 符号回归适配器

将 v2 的 FeatureSpace 连接到 PySR (Julia 后端) 符号回归引擎。
PySR 搜索数学公式来预测未来收益率，然后对发现的公式进行
IC/ICIR 验证 + v2 回测验证。

用法:
    from signals.v2 import FeatureSpace, PySRAdapter

    fs = FeatureSpace().fit(data)

    adapter = PySRAdapter(fs, data, holding_period=5)
    adapter.prepare_data(test_start='2024-01-01')

    model = adapter.run_search(niterations=40)

    results = adapter.validate_formulas(model, top_n=15)
    print(adapter.report())

    best_expr = adapter.best_formula()

注意:
- 需要安装 PySR: pip install pysr
- PySR 需要 Julia 运行时环境
- 符号回归的目标变量是未来N日收益率（连续值）
- 验证时自动转换：预测值 → 阈值化 → 择时信号 → 回测

自定义SR路线说明：
  本模块采用 PySR (Julia 后端) 路线，与 AIdev 002zeshi_pysr 一致。
  另一条路线是自定义 SR 引擎（基于 v2 TreeNode），不依赖 Julia，
  可直接复用 v2 的 Expression.parse() 解析能力，与 GP 优化器共享
  AST 基础。两种路线对比：
  ┌──────────┬────────────────────┬──────────────────────┐
  │          │  PySR 路线         │  自定义SR 路线         │
  ├──────────┼────────────────────┼──────────────────────┤
  │ 依赖     │  Julia + PySR     │  纯 Python            │
  │ 算法     │  多目标Pareto      │  单目标GA             │
  │ 算子集   │  可配置            │  与GP共享NODE_CONFIG   │
  │ 过拟合控制│  Pareto前沿       │  train/test划分       │
  │ 与v2集成 │  适配层(本文件)    │  原生 TreeNode         │
  │ 实现难度 │  低(已有轮子)      │  中(需约800行)         │
  └──────────┴────────────────────┴──────────────────────┘
  当前选择 PySR 路线。后续可根据需要实现自定义SR版本。
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import warnings

from .expression import Expression
from .features import FeatureSpace
from .validator import run_backtest, run_backtest_from_signal, walk_forward

warnings.filterwarnings('ignore')


# ============================================================
# 默认 PySR 配置（保守版，参考 AIdev PySR V2）
# ============================================================

DEFAULT_PYSR_CONFIG = {
    'populations': 8,
    'population_size': 50,
    'niterations': 40,
    'maxsize': 20,
    'maxdepth': 6,
    'binary_operators': ['+', '-', '*', '/', 'max', 'min'],
    'unary_operators': ['sqrt', 'abs', 'tanh'],
    'constraints': {
        '/': (-1, 9),
    },
    'nested_constraints': {},
    'complexity_of_operators': {
        '+': 1, '-': 1, '*': 1,
        '/': 2, 'max': 2, 'min': 2,
        'sqrt': 2, 'abs': 1, 'tanh': 2,
    },
    'complexity_of_constants': 2,
    'select_k_features': None,
    'warm_start': False,
    'progress': True,
    'timeout_in_seconds': 600,
    'random_state': 42,
}

# 公式验证阈值
DEFAULT_VALIDATION_THRESHOLDS = {
    'train_ic_min': 0.01,
    'test_ic_min': 0.005,
    'train_sharpe_min': -1.0,
    'trade_count_min': 5,
}

# 信号转换方法
SIGNAL_METHODS = ['zero', 'mean', 'rolling_mean']


# ============================================================
# 数据结构
# ============================================================

@dataclass
class SRFormulaResult:
    equation: str
    sympy_expr: str
    complexity: int
    loss: float
    score: float

    train_ic: float = 0.0
    train_icir: float = 0.0
    test_ic: float = 0.0
    test_icir: float = 0.0
    rank_train_ic: float = 0.0
    rank_test_ic: float = 0.0

    train_sharpe: float = 0.0
    test_sharpe: float = 0.0
    train_annual: float = 0.0
    test_annual: float = 0.0
    train_drawdown: float = 0.0
    test_drawdown: float = 0.0
    trade_count: int = 0
    overfit_ratio: float = 0.0
    best_signal_method: str = ''

    wf_mean_sharpe: float = 0.0
    wf_stability: float = 0.0

    is_valid: bool = False

    def __repr__(self):
        return (f"SRFormula(IC={self.test_ic:.4f}, SR={self.test_sharpe:.3f}, "
                f"eq={self.equation[:50]})")


# ============================================================
# PySRAdapter 核心类
# ============================================================

class PySRAdapter:
    """
    PySR 符号回归适配器

    Args:
        feature_space: v2 FeatureSpace 实例
        data: OHLCV DataFrame
        holding_period: 目标变量——未来N日收益率
        initial_capital: 回测初始资金
        config: PySR 配置（None=使用默认配置）
    """

    def __init__(self, feature_space: FeatureSpace, data: pd.DataFrame,
                 holding_period: int = 5,
                 initial_capital: float = 1_000_000,
                 config: Optional[Dict] = None):
        self._fs = feature_space
        self._data = data
        self.holding_period = holding_period
        self.initial_capital = initial_capital
        self._config = {**DEFAULT_PYSR_CONFIG}
        if config:
            self._deep_merge(self._config, config)

        self._feature_df: Optional[pd.DataFrame] = None
        self._feature_names: List[str] = []
        self._y: Optional[pd.Series] = None

        self._X_train: Optional[np.ndarray] = None
        self._X_test: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None
        self._y_test: Optional[np.ndarray] = None
        self._train_prices: Optional[pd.Series] = None
        self._test_prices: Optional[pd.Series] = None

        self._model = None
        self._results: List[SRFormulaResult] = []

    def _deep_merge(self, base: Dict, overrides: Dict):
        for key, val in overrides.items():
            if isinstance(val, dict) and key in base and isinstance(base[key], dict):
                self._deep_merge(base[key], val)
            else:
                base[key] = val

    # ---- 数据准备 ----

    def prepare_data(self, test_start: str = '2024-01-01',
                     use_selected_features: bool = True,
                     selected_features: List[str] = None) -> 'PySRAdapter':
        """
        准备 PySR 数据

        Args:
            test_start: 测试集开始日期
            use_selected_features: 是否使用精选特征子集
            selected_features: 自定义精选特征列表（None则使用默认）
        """
        self._feature_df = self._fs.fit_transform(self._data)
        self._feature_names = self._fs.get_feature_names()

        close = self._data['close']
        future_return = close.shift(-self.holding_period) / close - 1.0
        self._y = future_return.loc[self._feature_df.index]

        feature_cols = self._feature_names
        if use_selected_features and selected_features is None:
            selected_features = self._get_default_selected_features()
        if selected_features:
            available = [f for f in selected_features if f in self._feature_df.columns]
            feature_cols = available
            print(f"使用精选特征: {len(available)}/{len(self._feature_names)} 个")

        train_mask = self._feature_df.index < test_start
        test_mask = self._feature_df.index >= test_start

        self._X_train = self._feature_df.loc[train_mask, feature_cols].values
        self._X_test = self._feature_df.loc[test_mask, feature_cols].values
        self._y_train = self._y.loc[train_mask].values
        self._y_test = self._y.loc[test_mask].values
        self._train_prices = close.loc[self._feature_df.index[train_mask]]
        self._test_prices = close.loc[self._feature_df.index[test_mask]]
        self._feature_names = feature_cols

        print(f"数据准备完成:")
        print(f"  训练: {len(self._X_train)} 条, 测试: {len(self._X_test)} 条")
        print(f"  特征数: {len(feature_cols)}")
        print(f"  目标: 未来{self.holding_period}日收益率")

        return self

    def _get_default_selected_features(self) -> List[str]:
        all_features = set(self._feature_names)
        preferred = [
            'ATR_7', 'ATR_14', 'STDDEV_20', 'BBWIDTH_20', 'BBWIDTH_30',
            'HV_20', 'NATR_14', 'TRIMA_40', 'TRIMA_60',
            'TSF_3', 'TSF_5', 'TSF_7', 'TSF_14',
            'ADX_14', 'RSI_14', 'CCI_14', 'MACD_12_26_9',
            'VOL_RATIO_5', 'VOL_RATIO_10', 'VOL_RATIO_10_MA',
            'TREND_STRENGTH_20', 'VOL_REGIME_20', 'VOL_CHG_5', 'MOM_CHG_5', 'UP_RATIO_10',
            'BBWIDTH_30_sub_TSF_7', 'ATR_7_sub_TSF_7', 'STDDEV_20_sub_TSF_7',
        ]
        return [f for f in preferred if f in all_features]

    # ---- PySR 搜索 ----

    def run_search(self, custom_config: Optional[Dict] = None) -> Any:
        """
        执行 PySR 符号回归搜索

        Args:
            custom_config: 覆盖默认配置的参数

        Returns:
            PySR model 对象
        """
        try:
            from pysr import PySRRegressor
        except ImportError:
            raise ImportError(
                "需要安装 PySR: pip install pysr\n"
                "PySR 还需要 Julia 运行时，请参考 https://github.com/MilesCranmer/PySR"
            )

        if self._X_train is None:
            raise RuntimeError("请先调用 prepare_data()")

        cfg = {**self._config}
        if custom_config:
            self._deep_merge(cfg, custom_config)

        pysr_cfg = cfg.get('pysr', cfg)

        model = PySRRegressor(
            populations=pysr_cfg['populations'],
            population_size=pysr_cfg['population_size'],
            niterations=pysr_cfg['niterations'],
            maxsize=pysr_cfg['maxsize'],
            maxdepth=pysr_cfg['maxdepth'],
            binary_operators=pysr_cfg['binary_operators'],
            unary_operators=pysr_cfg['unary_operators'],
            constraints=pysr_cfg.get('constraints', {}),
            nested_constraints=pysr_cfg.get('nested_constraints', {}),
            complexity_of_operators=pysr_cfg.get('complexity_of_operators', None),
            complexity_of_constants=pysr_cfg.get('complexity_of_constants', 1),
            warm_start=pysr_cfg.get('warm_start', False),
            progress=pysr_cfg.get('progress', True),
            timeout_in_seconds=pysr_cfg.get('timeout_in_seconds', 600),
            model_selection='best',
            verbosity=1,
            random_state=pysr_cfg.get('random_state', 42),
        )

        print(f"\nPySR 搜索参数:")
        print(f"  种群: {pysr_cfg['populations']} × {pysr_cfg['population_size']}")
        print(f"  迭代: {pysr_cfg['niterations']}")
        print(f"  最大复杂度: {pysr_cfg['maxsize']}, 最大深度: {pysr_cfg['maxdepth']}")
        print(f"  二元算子: {pysr_cfg['binary_operators']}")
        print(f"  一元算子: {pysr_cfg['unary_operators']}")
        print(f"  特征数: {len(self._feature_names)}")
        print(f"  训练样本: {len(self._X_train)}")
        print(f"  超时: {pysr_cfg.get('timeout_in_seconds', 600)}秒")

        model.fit(self._X_train, self._y_train, variable_names=self._feature_names)
        self._model = model

        return model

    # ---- 公式验证 ----

    def _compute_ic(self, y_pred: np.ndarray, y_true: np.ndarray,
                    window: int = 60) -> Tuple[float, float]:
        pred_s = pd.Series(y_pred).reset_index(drop=True)
        true_s = pd.Series(y_true).reset_index(drop=True)
        rolling_ic = pred_s.rolling(window, min_periods=window // 2).corr(true_s)
        ic_mean = rolling_ic.dropna().mean()
        ic_std = rolling_ic.dropna().std()
        icir = ic_mean / ic_std if ic_std and ic_std > 1e-10 else 0.0
        return float(ic_mean), float(icir)

    def _compute_rank_ic(self, y_pred: np.ndarray, y_true: np.ndarray,
                          window: int = 60) -> Tuple[float, float]:
        pred_s = pd.Series(y_pred).rank().reset_index(drop=True)
        true_s = pd.Series(y_true).rank().reset_index(drop=True)
        rolling_ic = pred_s.rolling(window, min_periods=window // 2).corr(true_s)
        ic_mean = rolling_ic.dropna().mean()
        ic_std = rolling_ic.dropna().std()
        icir = ic_mean / ic_std if ic_std and ic_std > 1e-10 else 0.0
        return float(ic_mean), float(icir)

    def _formula_to_signal(self, y_pred: np.ndarray, method: str = 'zero') -> np.ndarray:
        if method == 'zero':
            return np.where(y_pred > 0, 1.0, 0.0)
        elif method == 'mean':
            threshold = np.nanmean(y_pred)
            return np.where(y_pred > threshold, 1.0, 0.0)
        elif method == 'median':
            threshold = np.nanmedian(y_pred)
            return np.where(y_pred > threshold, 1.0, 0.0)
        elif method == 'rolling_mean':
            s = pd.Series(y_pred)
            roll = s.rolling(60, min_periods=10).mean()
            return np.where(y_pred > roll.values, 1.0, 0.0)
        else:
            return np.where(y_pred > 0, 1.0, 0.0)

    def validate_formulas(self, model=None, top_n: int = 15,
                          signal_methods: List[str] = None,
                          do_walk_forward: bool = False) -> List[SRFormulaResult]:
        """
        验证 PySR 发现的 Pareto 前沿公式

        Args:
            model: PySR model（默认使用上次搜索的模型）
            top_n: 验证前 N 个公式
            signal_methods: 信号转换方法列表
            do_walk_forward: 是否执行 Walk-Forward 验证

        Returns:
            按 test_icir 降序排列的公式列表
        """
        if model is not None:
            self._model = model
        if self._model is None:
            raise RuntimeError("没有 PySR 模型，请先调用 run_search()")
        if signal_methods is None:
            signal_methods = SIGNAL_METHODS

        equations = self._model.equations_
        if isinstance(equations, list):
            equations = equations[-1]

        print(f"\n验证 Pareto 前沿公式 (共 {len(equations)} 个)...")

        X_test = self._X_test
        y_test = self._y_test
        prices_train = self._train_prices
        prices_test = self._test_prices

        for idx in range(min(len(equations), top_n)):
            row = equations.iloc[idx]
            equation = str(row.get('equation', ''))
            sympy_expr = str(row.get('sympy_format', equation))
            complexity = int(row.get('complexity', 0))
            loss = float(row.get('loss', 0))
            score = float(row.get('score', 0))

            print(f"\n  [{idx + 1}] complexity={complexity}, loss={loss:.6f}")
            print(f"      {equation[:80]}")

            try:
                y_pred_train = self._model.predict(self._X_train, idx)
                y_pred_test = self._model.predict(X_test, idx)
            except Exception as e:
                print(f"      预测失败: {e}")
                continue

            y_pred_train = np.nan_to_num(y_pred_train, nan=0.0, posinf=0.0, neginf=0.0)
            y_pred_test = np.nan_to_num(y_pred_test, nan=0.0, posinf=0.0, neginf=0.0)

            best_result = None
            best_fitness = -np.inf

            for method in signal_methods:
                result = SRFormulaResult(
                    equation=equation, sympy_expr=sympy_expr,
                    complexity=complexity, loss=loss, score=score,
                    best_signal_method=method,
                )

                result.train_ic, result.train_icir = self._compute_ic(
                    y_pred_train, self._y_train)
                result.test_ic, result.test_icir = self._compute_ic(
                    y_pred_test, y_test)
                result.rank_train_ic, _ = self._compute_rank_ic(
                    y_pred_train, self._y_train)
                result.rank_test_ic, _ = self._compute_rank_ic(
                    y_pred_test, y_test)

                train_signals = self._formula_to_signal(y_pred_train, method)
                test_signals = self._formula_to_signal(y_pred_test, method)

                try:
                    train_bt = run_backtest_from_signal(
                        train_signals, prices_train, long_only=True)
                    result.train_sharpe = train_bt.sharpe
                    result.train_annual = train_bt.annual_return
                    result.train_drawdown = train_bt.max_drawdown

                    test_bt = run_backtest_from_signal(
                        test_signals, prices_test, long_only=True)
                    result.test_sharpe = test_bt.sharpe
                    result.test_annual = test_bt.annual_return
                    result.test_drawdown = test_bt.max_drawdown
                    result.trade_count = max(train_bt.trade_count, test_bt.trade_count)
                except Exception:
                    pass

                if result.train_sharpe > 0:
                    result.overfit_ratio = min(result.test_sharpe / result.train_sharpe, 2.0)
                else:
                    result.overfit_ratio = 0.0

                result.is_valid = (
                    result.train_ic > DEFAULT_VALIDATION_THRESHOLDS['train_ic_min'] and
                    result.test_ic > DEFAULT_VALIDATION_THRESHOLDS['test_ic_min'] and
                    result.train_sharpe > DEFAULT_VALIDATION_THRESHOLDS['train_sharpe_min'] and
                    result.trade_count >= DEFAULT_VALIDATION_THRESHOLDS['trade_count_min']
                )

                fitness = result.test_icir * 10 + max(result.test_sharpe, 0)
                if fitness > best_fitness:
                    best_fitness = fitness
                    best_result = result

            if best_result:
                if do_walk_forward and best_result.is_valid:
                    try:
                        expr = Expression(f"thr_0({best_result.equation.split(' ')[0]})",
                                         feature_space=self._fs)
                        wf = walk_forward(expr, self._data, '2Y', '1Y', '6M')
                        best_result.wf_mean_sharpe = wf.mean_sharpe
                        best_result.wf_stability = wf.stability_score
                    except Exception:
                        pass

                self._results.append(best_result)
                tag = "✓" if best_result.is_valid else "✗"
                print(f"      {tag} IC: train={best_result.train_ic:.4f}, "
                      f"test={best_result.test_ic:.4f} | "
                      f"ICIR: {best_result.test_icir:.4f} | "
                      f"Sharpe: train={best_result.train_sharpe:.3f}, "
                      f"test={best_result.test_sharpe:.3f}")

        self._results.sort(key=lambda x: x.test_icir, reverse=True)
        return self._results

    def best_formula(self) -> Optional[SRFormulaResult]:
        if not self._results:
            return None
        return self._results[0]

    def best_formula_expression(self) -> Optional[str]:
        best = self.best_formula()
        if best:
            return best.equation
        return None

    # ---- 批量搜索（多持有期） ----

    def batch_search(self, holding_periods: List[int] = [3, 5, 10],
                     test_start: str = '2024-01-01',
                     niterations: int = 20) -> Dict[int, List[SRFormulaResult]]:
        """
        批量搜索多个持有周期

        Returns:
            {holding_period: [SRFormulaResult, ...]}
        """
        all_results = {}
        for hp in holding_periods:
            print(f"\n{'=' * 60}")
            print(f"持有期: {hp} 日")
            self.holding_period = hp
            self.prepare_data(test_start=test_start)
            self.run_search(custom_config={'niterations': niterations})
            results = self.validate_formulas(top_n=10)
            all_results[hp] = results
        return all_results

    # ---- 报告输出 ----

    def report(self) -> pd.DataFrame:
        if not self._results:
            return pd.DataFrame()
        rows = []
        for i, r in enumerate(self._results, 1):
            rows.append({
                '排名': i,
                '复杂度': r.complexity,
                '训练IC': round(r.train_ic, 4),
                '测试IC': round(r.test_ic, 4),
                '训练ICIR': round(r.train_icir, 4),
                '测试ICIR': round(r.test_icir, 4),
                'Rank训练IC': round(r.rank_train_ic, 4),
                'Rank测试IC': round(r.rank_test_ic, 4),
                '训练Sharpe': round(r.train_sharpe, 4),
                '测试Sharpe': round(r.test_sharpe, 4),
                '训练年化(%)': round(r.train_annual, 2),
                '测试年化(%)': round(r.test_annual, 2),
                '训练回撤(%)': round(r.train_drawdown, 2),
                '测试回撤(%)': round(r.test_drawdown, 2),
                '过拟合比': round(r.overfit_ratio, 3),
                '交易次数': r.trade_count,
                '信号方法': r.best_signal_method,
                '有效': '✓' if r.is_valid else '✗',
                '公式': r.equation[:70],
            })
        return pd.DataFrame(rows)

    def to_dict(self) -> Dict:
        return {
            'config': {
                'holding_period': self.holding_period,
                'pysr_config': {k: v for k, v in self._config.items()
                                if not callable(v)},
            },
            'summary': {
                'total_formulas': len(self._results),
                'valid_formulas': sum(1 for r in self._results if r.is_valid),
                'best_test_ic': max((r.test_ic for r in self._results), default=0),
                'best_test_icir': max((r.test_icir for r in self._results), default=0),
                'best_test_sharpe': max((r.test_sharpe for r in self._results if r.is_valid), default=0),
            },
            'formulas': [{
                'equation': r.equation,
                'sympy_expr': r.sympy_expr,
                'complexity': r.complexity,
                'test_ic': r.test_ic,
                'test_icir': r.test_icir,
                'test_sharpe': r.test_sharpe,
                'test_annual': r.test_annual,
                'is_valid': r.is_valid,
            } for r in self._results],
        }

    def save_report(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False, default=str)
