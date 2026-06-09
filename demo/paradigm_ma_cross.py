# paradigm_ma_cross.py
# 回测范式：均线金叉择时 + 基准买入持有对比
# 数据源: d2api → 回测引擎: ft2.core → 报告: Notebook
#
# 运行方式：
#   cd d:\01-Doc\Quant\ft2\tmp
#   python paradigm_ma_cross.py

import sys
sys.path.insert(0, r'd:\01-Doc\Quant\d2api')
sys.path.insert(0, r'd:\01-Doc\Quant\ft2')

import pandas as pd

from d2_api import d2_api
from core import Engine, AccountAnalyzer, context, account, OrderSide, BenchHolder


# ============================================================================
# 1. 数据获取
# ============================================================================

SYMBOL = '399317.SZ'          # 国证A指
DATA_START = '2022-01-01'     # 数据起点（含预热期，引擎只入缓存不调 on_bar）
BACKTEST_START = '2023-01-01' # 回测起点（开始交易）
END_DATE = '2026-06-09'
INTERVAL = '1d'

print(f'获取 {SYMBOL} {DATA_START} ~ {END_DATE} 日线数据...')
df_raw = d2_api.kline.query(SYMBOL, DATA_START, END_DATE, interval=INTERVAL)
print(f'共 {len(df_raw)} 条记录，时间范围 {df_raw.index[0]} ~ {df_raw.index[-1]}')

# d2api 返回的 DataFrame 以 time 为 datetime 索引，转换为引擎需要的 records 格式
# 引擎要求每条 bar 包含 eob (end-of-bar) 字段
records = []
for idx, row in df_raw.iterrows():
    rec = row.to_dict()
    # 将 time 索引转为 eob（引擎的 bar 结束时间）
    rec['eob'] = idx.to_pydatetime()
    # symbol 字段已在 d2api 返回的列中，无需额外设置
    records.append(rec)

# 回测区间 — engine.run() 已自动适配 str/date/datetime/Timestamp
# 引擎会遍历所有数据构建 timeline，但只在 [start, end] 区间内调用 on_bar
# 预热期数据仅入缓存，策略不可见，保证均线等技术指标初始值正确


# ============================================================================
# 2. 配置回测上下文
# ============================================================================

context.mode = 'backtest'
# [注意] fields 必须包含 'close,eob'，account._get_price() 依赖 eob 做时间比对
#       策略中用 context.data() 查询字段可不同于订阅字段（子集即可）
context.subscribe(SYMBOL, '1d', count=300, fields='close,eob')


# ============================================================================
# 3. 策略定义：双均线交叉择时
# ============================================================================

class MACrossover:
    """
    双均线交叉择时策略
    - 金叉（短均上穿长均）→ 全仓买入
    - 死叉（短均下穿长均）→ 全仓卖出
    """

    def __init__(self, short_period=5, long_period=20):
        self.short_period = short_period
        self.long_period = long_period

    def on_bar(self, context, bars):
        # 取足够的收盘价数据
        need_bars = self.long_period + 2  # +2 保证能计算前一根 bar 的均线值
        data = context.data(SYMBOL, '1d', count=need_bars, fields='close')

        if isinstance(data, pd.DataFrame):
            closes = data['close'].tolist()
        else:
            closes = [d['close'] for d in data]

        if len(closes) < self.long_period + 1:
            return

        # 当前 bar 的均线值
        short_now = sum(closes[-self.short_period:]) / self.short_period
        long_now = sum(closes[-self.long_period:]) / self.long_period
        # 前一根 bar 的均线值
        short_prev = sum(closes[-self.short_period-1:-1]) / self.short_period
        long_prev = sum(closes[-self.long_period-1:-1]) / self.long_period

        has_position = account.get_position(SYMBOL)['volume'] > 0

        if short_prev <= long_prev and short_now > long_now and not has_position:
            account.order_percent(SYMBOL, 1.0, OrderSide.Buy, note='金叉买入')
        elif short_prev >= long_prev and short_now < long_now and has_position:
            account.order_percent(SYMBOL, 1.0, OrderSide.Sell, note='死叉卖出')


# ============================================================================
# 4. 运行基准策略（买入持有）
# ============================================================================

print('\n运行基准策略（买入持有）...')
bench_engine = Engine()
bench_engine.add_data(SYMBOL, '1d', records)
bench_engine.run(BenchHolder, BACKTEST_START, END_DATE)
bench_analyzer = AccountAnalyzer(account)

bench_trades = len(account.trade_records)
bench_nav = account.snapshots[-1].nav if account.snapshots else 0
print(f'  基准成交 {bench_trades} 笔，最终净值 {bench_nav:,.0f}')

# [重要] 基准跑完后重置账户（恢复初始资金+清空持仓/记录）
#   Engine 缓存已实例化隔离，无需 context.reset()
account.reset(init_cash=1e6)


# ============================================================================
# 5. 运行择时策略
# ============================================================================

print('\n运行择时策略（MA5/20 交叉）...')
engine = Engine()
engine.add_data(SYMBOL, '1d', records)
engine.run(MACrossover, BACKTEST_START, END_DATE)
strategy_analyzer = AccountAnalyzer(account)

strat_trades = len(account.trade_records)
strat_nav = account.snapshots[-1].nav if account.snapshots else 0
print(f'  策略成交 {strat_trades} 笔，最终净值 {strat_nav:,.0f}')


# ============================================================================
# 6. 输出 Notebook 报告（含基准对比）
# ============================================================================

print('\n生成 Notebook 报告...')
strategy_analyzer.set_benchmark(bench_analyzer.daily_assets, '买入持有')
html_path = strategy_analyzer.to_notebook(f'{SYMBOL} MA{5}/{20}交叉择时')
print(f'报告已生成: {html_path}')
