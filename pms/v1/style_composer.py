"""
pms/v1/style_composer.py — 多风格组合管理器
============================================================

定位：因子信号器 → 回测引擎 之间的组合管理层。
每个风格独立选品、独立管理资金，合并后统一执行。

核心类:
  StyleConfig     — 单个风格的配置 (面板 + 资金 + top_n)
  StyleManager    — 多风格组合管理 (独立选品 + 合并 + 来源追踪)
  StyleBacktest   — 回测入口 (委托 core.Engine)

用法:
  >>> from pms.v1 import StyleManager, StyleConfig, StyleBacktest
  >>> 
  >>> mgr = StyleManager([
  >>>     StyleConfig(name='强势', panel=panel_strength, capital_weight=0.5, top_n=2),
  >>>     StyleConfig(name='反转', panel=panel_reversal, capital_weight=0.5, top_n=2),
  >>> ])
  >>> 
  >>> analyzer = StyleBacktest.backtest(mgr, assets, rebalance='D', mode='vector')
  >>> print(analyzer.sharpe_ratio())
  >>> print(analyzer.style_breakdown())
============================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field

from core.engine import Engine
from core.account import OrderSide
from core.storage import context
from core.analyzer import AccountAnalyzer


# ============================================================
# StyleConfig — 单个风格的配置
# ============================================================

@dataclass
class StyleConfig:
    """单个风格的配置
    
    Attributes:
        name: 风格名称（'强势', '反转'等, 用于来源追踪）
        panel: 因子排名面板 DataFrame(index=日期, columns=品种)
        capital_weight: 该风格占用的资金比例 (0~1)
        top_n: 该风格保留的品种数
    """
    name: str
    panel: pd.DataFrame
    capital_weight: float = 0.5
    top_n: int = 2
    
    def __post_init__(self):
        if not isinstance(self.panel.index, pd.DatetimeIndex):
            self.panel.index = pd.to_datetime(self.panel.index)


# ============================================================
# StyleManager — 多风格组合管理
# ============================================================

class StyleManager:
    """多风格组合管理器 — 每日独立选品 + 合并 + 来源追踪
    
    每个风格用各自的面板独立选 top_n 品种, 按 capital_weight 分配资金。
    同一品种被多个风格选中 → 累加权重 (共识自然翻倍)。
    """

    def __init__(self, styles: List[StyleConfig]):
        assert sum(s.capital_weight for s in styles) == 1.0, \
            f"资金权重合计应为 1.0, 当前 {sum(s.capital_weight for s in styles)}"
        self.styles = styles
        
        # 求所有风格的公共日期交集
        self.dates = styles[0].panel.index.sort_values()
        for s in styles[1:]:
            self.dates = self.dates.intersection(s.panel.index)
        self.dates = self.dates.sort_values()
        
        # 所有涉及品种
        self.symbols = set()
        for s in styles:
            self.symbols.update(s.panel.columns.tolist())
        self.symbols = sorted(self.symbols)

    # ---- 核心：每日选品+合并 ----

    def compose_day(self, date) -> Dict[str, Tuple[float, List[str]]]:
        """给定日期, 独立选品 + 合并权重 + 标注来源
        
        Args:
            date: 交易日 (pd.Timestamp 或日期字符串)

        Returns:
            dict: {品种代码: (权重, [来源风格列表])}
                  例: {'801010.SI': (0.50, ['强势', '反转']),
                        '801020.SI': (0.25, ['强势']),
                        '801030.SI': (0.25, ['反转'])}
        """
        positions: Dict[str, list] = {}  # {code: [累计权重, 来源列表]}

        for style in self.styles:
            if date not in style.panel.index:
                continue
            
            row = style.panel.loc[date]
            # 选出该风格 Top-N 排名品种
            top_codes = row.nlargest(style.top_n).index.tolist()
            if not top_codes:
                continue
            
            # 每个品种分配权重: 风格资金占比 / 风格选品数
            weight_per_stock = style.capital_weight / len(top_codes)
            
            for code in top_codes:
                if code not in positions:
                    positions[code] = [0.0, []]
                positions[code][0] += weight_per_stock       # 累加权重
                positions[code][1].append(style.name)         # 累加来源

        return {code: (info[0], info[1]) for code, info in positions.items()}
    
    def compose_all(self) -> Dict[pd.Timestamp, Dict[str, Tuple[float, List[str]]]]:
        """逐日完整组合"""
        return {d: self.compose_day(d) for d in self.dates}

    # ---- 分析 ----

    def overlap_matrix(self) -> pd.DataFrame:
        """两两风格选股重合率 (日频平均)"""
        n = len(self.styles)
        mat = np.zeros((n, n))
        counts = np.zeros((n, n))
        
        for date in self.dates:
            daily_picks = {}
            for i, style in enumerate(self.styles):
                if date not in style.panel.index:
                    continue
                daily_picks[i] = set(
                    style.panel.loc[date].nlargest(style.top_n).index.tolist()
                )
            
            for i in range(n):
                for j in range(n):
                    if i in daily_picks and j in daily_picks:
                        a, b = daily_picks[i], daily_picks[j]
                        if len(a) + len(b) > 0:
                            mat[i, j] += len(a & b) / max(len(a), len(b))
                            counts[i, j] += 1
        
        mat = np.where(counts > 0, mat / counts, 0)
        return pd.DataFrame(
            mat, 
            index=[s.name for s in self.styles],
            columns=[s.name for s in self.styles],
        )
    
    def style_daily_picks(self, style_name: str) -> pd.DataFrame:
        """某风格每日选品矩阵 (T × N, bool)"""
        style = next((s for s in self.styles if s.name == style_name), None)
        if style is None:
            raise ValueError(f"风格 '{style_name}' 不存在")
        
        picks = pd.DataFrame(
            False,
            index=self.dates,
            columns=self.symbols,
        )
        for date in self.dates:
            if date in style.panel.index:
                top = style.panel.loc[date].nlargest(style.top_n).index
                picks.loc[date, top] = True
        return picks


# ============================================================
# StyleBacktest — 回测入口
# ============================================================

class StyleBacktest:
    """多风格组合回测 — 委托 core.Engine
    
    三种模式:
      vector — 纯矩阵向量化 (~最快, 无 TradeRecord)
      fast   — 事件驱动 + FastAccount
      full   — 事件驱动 + AccountManager (完整记录)
    """

    @staticmethod
    def backtest(
        manager: StyleManager,
        assets: Dict[str, pd.DataFrame],
        rebalance: str = 'D',
        mode: str = 'vector',
        initial_capital: float = 1_000_000,
        start_date: str = None,
        bench_label: str = None,
        fee_config: dict = None,
    ) -> AccountAnalyzer:
        """多风格组合回测统一入口

        Args:
            manager: StyleManager (多风格配置)
            assets: {品种代码: OHLCV DataFrame}
            rebalance: 'D'/'W'/'M'/'ME'/'5D'
            mode: 'vector'/'fast'/'full'
            initial_capital: 初始资金
            start_date: 回测起始日
            bench_label: 基准品种代码 (full 模式有效)
            fee_config: 费率配置

        Returns:
            AccountAnalyzer (base_dir=None, to_notebook(base_dir=...) 时由调用方指定输出目录)
        """
        if mode == 'vector':
            return StyleBacktest._run_vectorized(
                manager, assets, rebalance, initial_capital, start_date, fee_config,
            )
        else:
            return StyleBacktest._run_engine(
                manager, assets, rebalance, mode, initial_capital,
                start_date, bench_label, fee_config,
            )

    # ---- vector 模式 ----

    @staticmethod
    def _run_vectorized(manager, assets, rebalance, capital, start_date, fee_config):
        """向量化: 预计算每日权重 → 矩阵运算"""
        dates = manager.dates
        if start_date:
            dates = dates[dates >= pd.Timestamp(start_date)]
        symbols = manager.symbols
        sym_to_idx = {c: i for i, c in enumerate(symbols)}
        n = len(dates)
        
        # 构建价格 + 收益率矩阵
        price_arr = np.full((n, len(symbols)), np.nan)
        for code in symbols:
            if code in assets:
                price_arr[:, sym_to_idx[code]] = (
                    assets[code].loc[:, 'close'].reindex(dates).values
                )
        ret_arr = np.diff(price_arr, axis=0) / (price_arr[:-1] + 1e-12)
        ret_arr = np.nan_to_num(ret_arr, nan=0.0)
        
        # 预计算每日权重
        daily_weights = StyleBacktest._precompute_weights(
            manager, dates, symbols, sym_to_idx,
        )
        
        # 调仓日
        rb_dates = _make_rebalance_set(dates, rebalance)
        
        # 费率
        fee = fee_config or {'commission_rate': 0.0, 'stamp_tax_rate': 0.0, 'min_commission': 0.0}
        
        nav = float(capital)
        daily_nav = {}
        current_weights = np.zeros(len(symbols))
        
        for i in range(n):
            date = dates[i]
            is_rb = date in rb_dates
            
            # 当日收益
            if i > 0 and current_weights.sum() > 0:
                day_ret = np.dot(current_weights, ret_arr[i - 1, :])
                nav *= (1.0 + day_ret)
            
            # 调仓
            if is_rb:
                new_weights = daily_weights.get(date, np.zeros(len(symbols)))
                
                # 手续费
                dw = np.abs(new_weights - current_weights)
                turnover = dw.sum() / 2
                if turnover > 0:
                    nav -= max(turnover * nav * fee['commission_rate'], fee['min_commission'])
                    nav -= turnover * nav * fee['stamp_tax_rate']
                
                current_weights = new_weights
            
            daily_nav[date.date()] = round(nav, 2)
        
        return AccountAnalyzer(daily_assets=daily_nav)

    @staticmethod
    def _precompute_weights(manager, dates, symbols, sym_to_idx):
        """预计算每日权重向量"""
        daily = {}
        for d in dates:
            pos = manager.compose_day(d)
            w = np.zeros(len(symbols))
            for code, (weight, _sources) in pos.items():
                if code in sym_to_idx:
                    w[sym_to_idx[code]] = weight
            daily[d] = w
        return daily

    # ---- Engine 模式 (fast/full) ----

    @staticmethod
    def _run_engine(manager, assets, rebalance, mode, capital,
                    start_date, bench_label, fee_config):
        """事件驱动模式: core.Engine 执行"""
        dates = manager.dates
        if start_date:
            dates = dates[dates >= pd.Timestamp(start_date)]
        symbols = manager.symbols
        rb_dates = _make_rebalance_set(dates, rebalance)
        
        engine = Engine(init_cash=capital, fee_config=fee_config)
        context.mode = 'backtest'
        
        # 注入数据
        for code in symbols:
            if code in assets:
                df = assets[code].copy()
                if not isinstance(df.index, pd.DatetimeIndex):
                    df.index = pd.to_datetime(df.index)
                if 'eob' not in df.columns:
                    df['eob'] = df.index
                context.unsubscribe(code, '1d')
                context.subscribe(code, '1d', count=300)
                engine.add_data(code, '1d', df)
        
        class _StyleStrategy:
            def __init__(self):
                self._rb_dates = rb_dates
                self._targets = {}     # 最近一次调仓的目标仓位
            
            def on_bar(self, ctx, bars):
                if not bars:
                    return
                current_date = pd.Timestamp(bars[0].get('eob')).normalize()
                
                if current_date not in self._rb_dates:
                    return
                
                # 计算目标仓位
                pos = manager.compose_day(current_date)
                self._targets = pos
                
                # 平仓不在目标内的
                for code, hold in list(ctx.account.get_position().items()):
                    if hold.get('volume', 0) <= 0:
                        continue
                    if code not in pos:
                        try:
                            ctx.account.order_percent(code, 1.0, OrderSide.Sell)
                        except (ValueError, RuntimeError):
                            pass
                
                # 开仓/调整
                for code, (weight, _sources) in pos.items():
                    try:
                        ctx.account.order_percent(code, weight, OrderSide.Buy)
                    except (ValueError, RuntimeError):
                        pass
        
        start_t = dates[0].to_pydatetime()
        end_t = dates[-1].to_pydatetime()
        
        if mode == 'fast':
            analyzer = engine.run_fast(_StyleStrategy(), start_t, end_t)
        else:
            engine.run(_StyleStrategy, start_t, end_t)
            analyzer = AccountAnalyzer(engine.account)
            
            if bench_label and bench_label in assets:
                bench_df = assets[bench_label].copy()
                if 'eob' not in bench_df.columns:
                    bench_df['eob'] = bench_df.index
                bench_eng = Engine(init_cash=capital, fee_config=fee_config)
                context.unsubscribe(bench_label, '1d')
                context.subscribe(bench_label, '1d', count=3000)
                bench_eng.add_data(bench_label, '1d', bench_df)
                from core.account import BenchHolder
                bench_eng.run(BenchHolder, start_t, end_t)
                bench_an = AccountAnalyzer(bench_eng.account)
                analyzer.set_benchmark(bench_an.daily_assets, bench_label)
        
        return analyzer


# ============================================================
# 工具函数
# ============================================================

def _make_rebalance_set(dates: pd.DatetimeIndex, rb: str) -> Set[pd.Timestamp]:
    """生成调仓日集合 (对齐 FacEngine 逻辑)"""
    rb = str(rb)
    if rb.endswith('D') and rb[:-1].isdigit():
        interval = int(rb[:-1])
        return {pd.Timestamp(dates[i].date())
                for i in range(interval - 1, len(dates), interval)}
    
    if rb == 'W':
        # 每ISO周最后一个交易日 (避免 resample('W') 兼容问题)
        iso = dates.isocalendar()
        seen = set()
        result = set()
        for i in range(len(dates)):
            key = (iso['year'].iloc[i], iso['week'].iloc[i])
            seen.add(key)
        # 找每周最后出现的位置
        last_positions = {}
        for i in range(len(dates)):
            key = (iso['year'].iloc[i], iso['week'].iloc[i])
            last_positions[key] = i
        return {pd.Timestamp(dates[i].date()) for i in last_positions.values()}
    
    freq_map = {'M': 'MS', 'ME': 'ME'}
    freq = freq_map.get(rb, rb)
    # [修复] 用非 NaN 值避免 dropna 清空全部
    s = pd.Series(1, index=dates)
    return {pd.Timestamp(d.date()) for d in s.resample(freq).last().index}