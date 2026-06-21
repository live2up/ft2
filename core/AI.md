# core 模块 - AI 快速上手

> 回测引擎核心 + HTML 报告（已融合 notebook 模块）
>
> **版本：v2.5 | 更新日期：2026-06-21**

---

## 架构概览

```
Engine → AccountManager / FastAccount → AccountAnalyzer → Notebook → HTML
  │           │     │                           │               │
  │ 时间线驱动  │     │ 自动扣费+记录净值          │ 指标计算       │ 渲染层
  ▼           ▼     ▼                           ▼               ▼
run() →  AccountManager      run_fast() → FastAccount
全快照  ctx.account.         零快照  ctx.account.  (替换 self.account)
       order_percent()              order_percent() → 接口完全一致
                                      mark() 记录净值
  → AccountAnalyzer(account)        → AccountAnalyzer(daily_assets)
```

**v2.5 重构要点（2026-06-21）：**
- 新增 `FastAccount` 轻量账户：走 `context.data()` 缓存取价，无快照/TradeRecord
- `run_fast()` 替换 `self.account` 为 `FastAccount`，策略统一用 `ctx.account` 下单
- fast/full 下单接口完全一致：`ctx.account.order_percent(symbol, 1.0, OrderSide.Buy)`
- `Engine(fee_config={...})` 构造时传入费率，full/fast 共享同一费源
- 默认费率改为零 (commission_rate=0, stamp_tax_rate=0, min_commission=0)

---

## 1. Engine — 回测引擎

```python
from core.engine import Engine
from core.storage import context
from core.account import OrderSide

engine = Engine(init_cash=1e6)  # 默认零费率
# 或自定义费率:
engine = Engine(init_cash=1e6, fee_config={'commission_rate': 0.0003, 'stamp_tax_rate': 0.001, 'min_commission': 5.0})
context.mode = 'backtest'
context.subscribe('399317.SZ', '1d', count=300)

# 加载数据（DataFrame 需有 eob, symbol 列）
engine.add_data('399317.SZ', '1d', df)

# ── full 模式: AccountManager → TradeRecord + Snapshot → AccountAnalyzer(account) ──
engine.run(MyStrategy, start_time, end_time)
from core.analyzer import AccountAnalyzer
analyzer = AccountAnalyzer(engine.account)       # Path 1: 快照模式 (全部指标)

# ── fast 模式: FastAccount → 零快照 → AccountAnalyzer(daily_assets) ──
# run_fast() 内部将 self.account 替换为 FastAccount，策略 ctx.account 透明切换
analyzer = engine.run_fast(FastStrategy(), start_time, end_time)
                                                  # Path 2: 净值模式 (仅资产指标)
```

- `Engine.timeline` 是 `OrderedDict[eob → List[bar]]`，按时间排序驱动；多频率数据共线
- 每个 bar 先入缓存（`_add_bar`），再调 `on_bar`
- `run()` / `run_fast()` 共用 `_drive_timeline()` 时间线循环，差异只在 `snapshot=True/False`
- `run_fast()` 不生成 TradeRecord/snapshots，FastAccount 自动管理仓位+费率+净值，约 6x 快于 full
- `ctx.account` 在 fast 模式下指向 FastAccount，接口兼容 AccountManager（order_percent/order_volume/get_position）
- 引擎在 on_bar 后自动调用 `ctx.account.mark()` 记录日末净值，策略无需手动调用
- `start/end` 自动 clamp 到时间线边界，`init_snapshot` 锚定时间线上 start 之前的真实 bar
- `run()`/`run_fast()` 入口注册 `_active_engine`，退出时恢复，支持嵌套多引擎

---

## 2. AccountManager / FastAccount — 账户管理

```python
# 策略内部通过 context.account 访问（委托到活跃 Engine 的账户）
# full 模式 → AccountManager (快照+TradeRecord)
# fast 模式 → FastAccount (零快照, 缓存取价, 自动记录净值)
def on_bar(self, context, bars):
    account = context.account

    # 下单（接口完全一致）
    account.order_percent('399317.SZ', 1.0, OrderSide.Buy, note="金叉买入")
    account.order_volume('399317.SZ', 100, OrderSide.Sell)

    # 查询
    account.get_position(symbol)  # {'volume': ..., 'cost_price': ...}
    account.get_account()         # {'cash': ..., 'nav': ...}
```

- **AccountManager**: 完整账户，快照聚合 + TradeRecord + FIFO 平仓匹配
- **FastAccount**: 轻量账户，通过 `context.data()` 缓存获取价格，自动扣费+记录净值
- `run_fast()` 时将 `self.account` 替换为 `FastAccount`，`ctx.account` 透明切换
- 费用配置：通过 `Engine(fee_config={...})` 构造时传入，full/fast 共享同一费源
- 默认费率：零费（commission_rate=0, stamp_tax_rate=0, min_commission=0）
- 交易单位：full 模式自动识别（stock/etf→100，index→1）；fast 模式简化处理
- `TradeRecord.note`：仅 full 模式，追溯每笔交易触发原因

---

## 3. AccountAnalyzer — 分析器

```python
from core.analyzer import AccountAnalyzer

# ── 两种输入路径 ──

# Path 1: 快照模式 (full 回测) — 全部指标可用
analyzer = AccountAnalyzer(account=engine.account)
# _daily_assets ← _aggregate_daily_assets(snapshots)
# _trade_profits ← _calculate_profit(trade_records)

# Path 2: 净值模式 (fast 回测 / 外部数据) — 仅资产指标可用
analyzer = AccountAnalyzer(daily_assets={date(2024,1,2): 1e6, ...})
# _daily_assets ← 直接赋值, _trade_profits = []

# 两种路径均归一到 Dict[date, float]，返回同一套 @metric 指标
# fast 模式下交易指标 (胜率/盈亏比等) 返回 None
```

### 3.1 指标定义：@metric 装饰器

**所有指标通过 `@metric` 声明元数据，一处定义全局生效：**

```python
@metric(name='夏普比率', group='风险', fmt='.2f', order=30)
def sharpe_ratio(self): return 0.57  # 纯数字

@metric(name='年化波动率', group='风险', fmt='.1%', order=20)
def volatility(self): return 0.125  # 输出显示 12.5%
```

| 参数 | 说明 | 默认 |
|------|------|------|
| `name` | 展示名 | 函数名 |
| `group` | 分组：`收益`/`风险`/`交易` | `''` |
| `fmt` | 输出格式：`.1%` 百分比 / `.2f` 数值 / `.1f` | `.2f` |
| `desc` | 描述（显示在 notebook metric-desc） | `''` |
| `order` | 组内排序 | `99` |

**新增指标只需加 `@metric`，`to_notebook/to_excel` 自动拾取。**

### 3.2 统一收集

```python
analyzer.metrics()  # → Dict[name → {name, value, group, desc, fmt, order}]
                    #    一次调用收集所有 @metric 方法结果
```

### 3.3 输出

```python
# Notebook 交互式报告（推荐，自动保存 HTML）
analyzer.to_notebook("策略回测")
# 内部: Notebook(title) → 构建 cells(指标表格/净值图/交易记录) → Jinja2 渲染 → 输出 .html

# 带自定义内容的报告（header/footer 回调）
analyzer.to_notebook("策略回测",
    header=lambda nb: nb.markdown("## 策略参数\n- MA5/20 交叉择时"),
    footer=lambda nb: nb.markdown("## 结论\n年化超额 +5.2%"))

# header/footer 支持完整 Notebook API（chart/table/section 等）
def add_header(nb):
    nb.metrics({'夏普': '1.85', '胜率': '52%'}, title="核心 KPI", columns=2)

def add_footer(nb):
    with nb.section("归因分析"):
        nb.table(factor_data, columns=['日期', '因子暴露', '收益贡献'])

analyzer.to_notebook("策略回测", header=add_header, footer=add_footer)

# Excel 报告
analyzer.to_excel("策略回测")
# → Sheet1: 回测指标(分组)  Sheet2: 每日资产  Sheet3: 交易记录
```

**to_notebook() 内部渲染链路：**

```
AccountAnalyzer.to_notebook()
  ├── 1. from notebook import Notebook → nb = Notebook(title)
  ├── 2. 构建 cells:
  │     ├── nb.section("指标分析") → nb.table(指标对比/基础指标/交易指标)
  │     ├── nb.section("收益走势图") → nb.chart('perf', {xAxis+series+datazoom})
  │     └── nb.section("交易记录") → nb.table(trades, page={size:10})
  ├── 3. nb.export_html()
  │     ├── 数据序列化为 JSON-LD → 注入 <head>
  │     ├── Jinja2 渲染 template/notebook.html
  │     └── 输出到 base_dir（调用者脚本所在目录）
  └── 前端: Vue3 初始化 → ECharts 绑定 → ft-table 渲染
```

**模板文件位置：**
| 文件 | 用途 |
|------|------|
| `template/notebook.html` | Jinja2 主模板，JSON-LD 注入 + Vue3 挂载点 |
| `template/js/notebook3D.js` | Vue3 渲染逻辑（图表/表格/section 交互） |
| `template/js/notebook3D.css` | 样式 |
| `template/js/ft-table.js` | 表格 Vue 组件（冻结/分页/热力图） |
| `template/js/echarts.min.js` | ECharts 库 |

**Notebook 报告层次（to_notebook 自动生成）：**

```
策略回测报告
├── 指标分析 (section)
│   ├── 基础/交易指标 (table)
│   └── [有基准时] 策略 vs 基准对比表 (table)
├── 收益走势图 (section)
│   └── 净值曲线 / 叠加对比图 (chart:perf, height=500px, datazoom slider)
└── 交易记录 (section)
    └── 全部成交明细 (table, page=10)
```

### 3.4 完整指标列表

| group | 指标名 | 方法 | fmt | order |
|-------|--------|------|-----|-------|
| 收益 | 累计收益率 | `return_rate()` | `.1%` | 10 |
| 收益 | 年化收益率 | `annualized_return()` | `.1%` | 11 |
| 风险 | 最大回撤 | `max_drawdown()` | `.1%` | 20 |
| 风险 | 夏普比率 | `sharpe_ratio()` | `.2f` | 21 |
| 风险 | 年化波动率 | `volatility()` | `.1%` | 22 |
| 风险 | 索提诺比率 | `sortino_ratio()` | `.2f` | 23 |
| 风险 | VaR(95%) | `var()` | `.1%` | 24 |
| 风险 | CVaR(95%) | `cvar()` | `.1%` | 25 |
| 风险 | UPI / 溃疡绩效指数 | `upi()` | `.2f` | 26 |
| 风险 | Ulcer Index | `ulcer_index()` | `.2f` | 27 |
| 风险 | Calmar / 卡尔玛比率 | `calmar_ratio()` | `.2f` | 28 |
| 风险 | 最大回撤持续天数 | `max_dd_duration()` | `.0f` | 29 |
| 风险 | 最大回撤恢复天数 | `max_dd_recovery()` | `.0f` | 30 |
| 交易 | 胜率 | `win_rate()` | `.1%` | 40 |
| 交易 | 平均盈亏比 | `avg_profit_loss_ratio()` | `.2f` | 41 |
| 交易 | 平均持仓天数 | `avg_holding_period()` | `.1f` | 42 |
| 交易 | 持仓天数比例 | `position_holding_ratio()` | `.1%` | 43 |
| 交易 | 凯利公式最优仓位 | `kelly_criterion()` | `.1%` | 50 |
| 交易 | 半凯利仓位 | `kelly_fraction()` | `.1%` | 51 |

> 每个指标返回纯数字，格式化由 `@metric(fmt=...)` 控制。新增指标只需加 `@metric`，`to_notebook/to_excel` 自动拾取。

### 3.5 时间区间切片

```python
analyzer.getTimeRange('3m')     # 近3月
analyzer.getTimeRange('1y')     # 近1年
analyzer.getTimeRange('1m', '6m', '1y', '2y', '3y', '5y', 'all')
```

### 3.6 公共查询

```python
analyzer.daily_assets          # Dict[date, nav]
analyzer.trade_profits         # List[Dict] 逐笔盈浮
analyzer.returns('1m,3m')      # 多区间收益率
analyzer.get_daily_total_assets()
analyzer.get_largest_profit_trades(5)
analyzer.get_largest_loss_trades(5)
```

### 3.7 基准对比（2026-06-02 新增）

```python
from core import BenchHolder

# 推荐：同一份 bars 跑两个策略，共享 init_snapshot(start_time-1天)，日期天然对齐
bench_engine = Engine()
# ... 加载相同数据到 bench_engine ...
bench_engine.run(BenchHolder, start_time, end_time)  # 买入持有（基准＝标的本身）
bench_analyzer = AccountAnalyzer(account)

engine.run(MyStrategy, start_time, end_time)          # 择时策略
strategy_analyzer = AccountAnalyzer(account)

# 注入基准 → to_notebook() 自动生成对比 section
strategy_analyzer.set_benchmark(bench_analyzer.daily_assets, '国证A指')
strategy_analyzer.to_notebook("策略 vs 基准")
```

**set_benchmark 接受的格式：**

| 格式 | 示例 |
|------|------|
| `Dict[date, float]` | `{date(2024,1,3): 100000.0, ...}` — 每日净值 |
| `List[Dict]` | `[{'date': date(2024,1,3), 'assets': 100000.0}, ...]` |

**对比输出内容：**

| Section | 内容 |
|---------|------|
| 对比 Table | 策略 vs 基准同构指标（收益/风险类，交易类跳过） |
| 净值叠加图 | 归一化至 1.0 的双线对比 |
| 超额累计曲线 | 每日超额累计乘积，起点=1.0 |
| 超额指标 | 超额收益、年化超额、信息比率、跟踪误差、日超额胜率 |

**对齐机制：**
- 双方共用引擎 `init_snapshot(start_time - 1 day)` → `dates[0]` 恒为初始资金
- 策略盘中交易 vs 基准前收盘价入场 → 超额 metric 在交集日期计算，严格对齐
- 外部基准数据缺少 init 日期时，自动向前扩展首日值补齐

### 3.8 Notebook 直接使用（高级）

当 `to_notebook()` 的预置结构不够用时，可直接操作 Notebook 构建自定义报告：

```python
from notebook import Notebook

nb = Notebook("自定义分析报告")

# 顶层 KPI 卡片
nb.metrics({'总收益率': '45.2%', '夏普比率': '1.85', '最大回撤': '-12.5%'},
           title="核心指标", columns=4)

# Section 层次组织
with nb.section("收益分析"):
    nb.chart('line', {'xAxis': dates, 'series': [...]},
             title='净值曲线', height='400px',
             series_opts={'is_smooth': True})

with nb.section("风险分析"):
    nb.chart('line', {'xAxis': dates, 'series': [{'name': '回撤', 'data': dd}]},
             title='回撤序列')

with nb.section("交易明细", collapsed=True):
    nb.table(trades, columns=['日期', '标的', '方向', '价格', '数量'],
             freeze={'left': 1}, page={'size': 20})

nb.export_html()  # → 输出到当前脚本所在目录
```

**Notebook 核心 API 速查：**

| 方法 | 用途 | 示例 |
|------|------|------|
| `nb.metrics(data, title, columns)` | KPI 指标卡片 | `List[Dict]` 或 `Dict`，支持 `color` |
| `nb.table(data, columns, title, **opts)` | 数据表格 | `freeze`/`heatmap`/`page` 配置 |
| `nb.chart(type, data, title, height, **opts)` | 图表 | `line`/`bar`/`area`/`scatter`/`kline`/`pie`/`heatmap` |
| `nb.chartg(type, data, height, **opts)` | Grid 累加图表 | 多图共享时间轴纵向堆叠 |
| `nb.section(title, collapsed)` | 章节容器 | `with nb.section(...)` 上下文管理器 |
| `nb.text/markdown/code/divider/html` | 文本/布局 | 补充说明、分隔线、原始 HTML |
| `nb.pyecharts(chart, title, height)` | pyecharts 原生 | 当 chart() 封装不够用时 |

**数据格式规范（关键）：**
- 图表标准格式：`{'xAxis': [...], 'series': [{'name': '...', 'data': [...]}]}`
- 图表也支持 DataFrame（line/bar/area: 第一列→xAxis, 余列→series；kline: 自动识 OHLC 字段）
- scatter **不支持** DataFrame，必须传 dict（`series[0].data` 为 `[[x,y], ...]`）
- xAxis 数据原样透传，前端不做日期格式转换

**表格增强选项：**
- `freeze={'left': 2}` — 冻结前 2 列（标识列固定）
- `heatmap={'start': 2, 'axis': 'column'}` — 数值列着色（红涨绿跌）
- `page={'size': 20}` / `page=False` — 分页控制（≤20 行建议禁用）

---

## 关键设计原则

1. **计算层返回纯数字，呈现层负责格式化** — 方法返回 `0.159`，`fmt='.1%'` 控制输出 `15.9%`
2. **`@metric` 声明式驱动** — 指标元数据集中管理，输出层零硬编码
3. **引擎天然防未来** — `eob` 时间线 + `context.now` 保证每时刻只能看到 ≤当前的数据
4. **频率无限制** — `freq` 是纯字符串 key，支持 `'1d'`/`'m10'`/`'my_signal'` 等任意自定义频率；多频数据共线推进
5. **fast/full 双模式** — `run()` 全快照 + AccountManager，`run_fast()` 替换为 FastAccount，接口一致，共用 `_drive_timeline()` 时间线循环
6. **初始快照独立** — `init_snapshot(start_time前一根真实bar)` 作为 `snapshots[0]`，分析层零推断、零补偿
7. **缓存实例化隔离** — `_cache`/`bar_data_set`/`account` 归 Engine 实例所有，多引擎天然隔离，无需 `account.reset()`
8. **基准＝真实策略** — `BenchHolder` 走与主策略相同的引擎+数据+账户通道，对比结果无偏差
9. **Notebook 渲染层内聚** — `analyzer.to_notebook()` 内部完成全链路；支持 `header/footer` 回调扩展

---

## 文件清单

```
core/
├── engine.py          # 回测引擎 (Engine + timeline 驱动，run/run_fast)
├── account.py         # 账户管理 (AccountManager + FastAccount + OrderSide + BenchHolder)
├── analyzer.py        # 分析器 (AccountAnalyzer + @metric + to_notebook/to_excel)
├── storage.py         # 数据存储 (context 上下文)
├── symbol_classifier.py # 品种分类
├── __init__.py        # 统一导出
└── AI.md              # 本文件

notebook/              # Notebook 渲染模块（core 内部依赖）
├── notebook.py        # Notebook 主类 (cell 构建 + Jinja2 渲染)
├── cell.py            # Cell/Section 数据类 + CellBuilder + 图表构建器
├── min_pyecharts.py   # 精简 ECharts option 构建器 (668行)
└── __init__.py

template/              # HTML 模板（Jinja2 + Vue3 前端）
├── notebook.html      # 主模板 (JSON-LD 注入 + Vue3 挂载)
└── js/
    ├── notebook3D.js  # Vue3 渲染逻辑
    ├── notebook3D.css # 样式
    ├── ft-table.js    # 表格组件
    └── echarts.min.js # ECharts 库
```
