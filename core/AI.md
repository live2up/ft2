# core 模块 - AI 快速上手

> 回测引擎核心
>
> **版本：v2.0 | 更新日期：2026-05-30**

---

## 架构概览

```
Engine → AccountManager → AccountAnalyzer
  │           │                 │
  │ 时间线驱动  │ 订单/持仓/费用   │ 指标计算+报告输出
  ▼           ▼                 ▼
context.now  account.order_   analyzer.to_notebook()
逐bar推进    percent()         analyzer.to_excel()
```

---

## 1. Engine — 回测引擎

```python
from core.engine import Engine
from core.storage import context
from core.account import account, OrderSide

engine = Engine()
context.mode = 'backtest'
context.subscribe('399317.SZ', '1d', count=300)

# 加载数据（DataFrame 需有 eob, symbol 列）
engine.add_data('399317.SZ', '1d', df)

# 运行（传策略类，引擎自动实例化）
engine.run(MyStrategy, start_time, end_time)
```

- `Engine.timeline` 是 `OrderedDict[eob → List[bar]]`，按时间排序驱动
- 每个 bar 先入缓存（`_add_bar`），再调 `on_bar`
- 策略实例化在 `run()` 内完成，`context.data()` 只能看到 ≤当前时间的数据

---

## 2. AccountManager — 账户管理

```python
from core.account import account, OrderSide

# 下单（可选 note 记录信号备注）
account.order_percent('399317.SZ', 1.0, OrderSide.Buy, note="温度计75度")
account.order_volume('399317.SZ', 100, OrderSide.Sell)

# 查询
account.get_account()        # {'cash': ..., 'nav': ...}
account.get_position(symbol) # {'volume': ..., 'cost_price': ...}
account.trade_records        # List[TradeRecord] 含 note 字段
account.snapshots            # List[AccountSnapshot]
```

- 费用计算：佣金 0.03% + 印花税 0.1%（卖）+ 最低 5 元
- 策略层可覆盖：`account.fee_config = {...}`
- `TradeRecord.note`：追溯每笔交易触发原因

---

## 3. AccountAnalyzer — 分析器

```python
from core.analyzer import AccountAnalyzer
analyzer = AccountAnalyzer(account)
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
# Notebook 交互式报告（推荐）
nb = analyzer.to_notebook("策略回测")
nb.export_html()

# Excel 报告
analyzer.to_excel("策略回测")
# → Sheet1: 回测指标(分组)  Sheet2: 每日资产  Sheet3: 交易记录
```

### 3.4 完整指标列表

| group | 指标 | fmt |
|-------|------|-----|
| 收益 | 累计收益率、年化收益率 | `.1%` |
| 风险 | 年化波动率、最大回撤、夏普比率、索提诺比率 | 混合 |
| 风险 | VaR(95%)、CVaR(95%)、Ulcer Index、UPI | 混合 |
| 交易 | 胜率、平均盈亏比、平均持仓时间、凯利/半凯利 | 混合 |

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

---

## 关键设计原则

1. **计算层返回纯数字，呈现层负责格式化** — 方法返回 `0.159`，`fmt='.1%'` 控制输出 `15.9%`
2. **`@metric` 声明式驱动** — 指标元数据集中管理，输出层零硬编码
3. **引擎天然防未来** — `eob` 时间线 + `context.now` 保证每时刻只能看到 ≤当前的数据
4. **频率无限制** — `freq` 是纯字符串 key，支持 `'1d'`/`'m10'`/`'my_signal'` 等任意自定义频率
