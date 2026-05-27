# ft2 — 量化回测框架

> AI 助手自动加载文件。本文件包含：项目简介 → 模块索引 → 使用速查 → 开发规范

---

## 1. 项目简介

ft2 是量化回测框架，提供因子挖掘、回测引擎、择时信号、HTML 报告生成能力。主要用于维护阶段，供外部项目导入使用。

**导入方式：**

```python
import sys; sys.path.insert(0, r'd:\01-Doc\Quant\ft2')
from notebook import Notebook
from core.analyzer import AccountAnalyzer
from factor import Factor, FactorRegistry, FactorValidator
from signals import MASignal, run_backtest, ICAnalyzer
```

---

## 2. 模块索引

| 模块 | 用途 | 何时使用 | 深度文档 |
|------|------|---------|---------|
| `notebook` | HTML 报告生成 | 需要输出可视化报告 | `notebook/AI.md` |
| `core` | 回测引擎核心 | 需要运行回测、分析结果 | `core/AI.md` |
| `factor` | 因子挖掘体系 | 需要计算因子、验证 IC | `factor/AI.md` |
| `signals` | 择时信号研究 | 需要生成信号、轻量回测 | `signals/AI.md` |
| `template` | HTML/JS 模板资源 | 报告模板、ECharts、ft-table | — |

**深度文档规则：** 上面速查不够用时，必须读取对应模块的 `AI.md`。例如操作 notebook 图表时读 `notebook/AI.md`，操作 factor 验证时读 `factor/AI.md`。

---

## 3. 使用速查

### 3.1 notebook — HTML 报告生成器

基于 Jinja2 + Vue3 + ECharts + ft-table，Python 端只负责数据组装。

```python
from notebook import Notebook
nb = Notebook("报告标题")

nb.metrics({'收益率': '45.2%', '夏普': '1.85'}, title="核心指标")  # 指标卡片
nb.chart('line', {'xAxis': dates, 'series': [...]}, title='净值') # 图表
nb.table(data, columns=['code', 'name'], heatmap={'start': 2})     # 表格(支持热力图)
with nb.section("详细", collapsed=True): ...                       # 折叠章节
nb.chartg('line', nav_data, height=300)  # Grid累加(同时间轴多图纵向堆叠)
nb.pyecharts(chart_obj)                  # pyecharts 原生对象
nb.export_html()                         # 输出到调用者脚本所在目录
```

**数据约定：** 图表用 `{'xAxis': [...], 'series': [...]}` 标准格式；表格用 `List[Dict]`；DataFrame 自动转换。

### 3.2 core — 回测引擎

```python
from core.analyzer import AccountAnalyzer
analyzer = AccountAnalyzer(account)

analyzer.sharpe_ratio()                          # 夏普比率
analyzer.max_drawdown()                          # 最大回撤
analyzer.returns()                               # 收益率序列
analyzer.nav()                                   # 净值序列
analyzer.getTimeRange('3m')                      # 时间区间切片: '1m'/'3m'/'1y'/'ytd'

from core.engine import Engine
from core.account import AccountManager
```

### 3.3 factor — 因子挖掘

```python
from factor import Factor, FactorRegistry, FactorValidator

class MyFactor(Factor):
    def calc(self, data):
        return data['close'].pct_change(20)

registry = FactorRegistry()
registry.register(MyFactor())

validator = FactorValidator(factor_values, returns)
ic = validator.calc_ic()          # ic_mean, ic_std, ir
validator.group_return(n_groups=5) # 分组收益
validator.monotonic_test()         # 单调性检验
```

**因子值格式：** DataFrame, index=日期, columns=股票代码

### 3.4 signals — 择时信号

```python
from signals import MASignal, MACDSignal, RSISignal, run_backtest, ICAnalyzer

signals = MASignal(5, 20).generate(df)

result = run_backtest(MASignal(5, 20), df, initial_capital=1e6)
# result.total_return / .sharpe / .max_drawdown / .win_rate / .trade_count

ic = ICAnalyzer.analyze(signals, df['close'])
# ic['basic']['summary']['ic_mean'], ic['significance']['p_value']
```

**架构原则：** signals/ 只负责计算，不依赖 Notebook；输出由调用方用 Notebook 组织。

---

## 4. 通用数据约定

- DataFrame: `index=日期`, `columns=股票代码`
- 表格数据: `List[Dict]` 或 `DataFrame`
- 图表数据: `{'xAxis': [...], 'series': [...]}`
- 时间格式: 日线 `'YYYY-MM-DD'`，分钟线 `'YYYY-MM-DD HH:MM:SS'`

---

## 5. 开发规范

### 5.1 代码修改必须注释

```python
# [修改类型] YYYY-MM-DD 简短描述
# 详细说明修改原因
```

类型：`[修复]` / `[优化]` / `[新增]` / `[重构]` / `[调整]`

### 5.2 工作流程

- 讨论阶段先不修改代码，确认方案后实施
- 修改前先阅读相关模块的 AI.md 和源码

### 5.3 版本目录自包含

每个 `vN/` 目录必须完全自包含：只能 `from .xxx import`，禁止跨版本 `from v(N-1)` 或 `from ..` 依赖。必备基类（如 `base.py`）必须在当前 `vN/` 内自有一份。

### 5.4 其他

- 不删除现有注释，除非函数逻辑彻底改变
- 不提交敏感信息
- 新增函数/类必须添加 docstring

---

> 最后更新：2026-05-27
