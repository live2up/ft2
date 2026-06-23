# ft2 — 量化回测框架

> AI 助手自动加载。**根据任务领域，按需读取对应模块的 AI.md。**

## 按需读取规则

| 任务涉及 | 应先读取 |
|----------|----------|
| 回测引擎 / 账户管理 / 指标计算 / HTML 报告 | `core/AI.md` |
| Notebook 可视化报告 / 图表 / 表格 | `notebook/AI.md` |
| 因子挖掘 / GP符号回归 / 因子检验 / 因子组合 | `factor/AI.md` |
| 择时信号 / 表达式引擎 / Walk-Forward / IC分析 | `signals/AI.md` |
| 多模块交叉（如 信号回测→报告输出） | 按依赖顺序读取多个 AI.md |
| 项目历史 / 已有结论 / 避免重复踩坑 | `docs/项目记忆库.md` |

> 以上 `AI.md` 各自包含完整的 API 速查、架构图、参数说明和注意事项。
> `docs/项目记忆库.md` 从 CodeBuddy Memory 导出，记录所有关键版本结论和死胡同。
> 不依赖 IDE 特定的 Skill 机制，跨平台、跨用户可用。

## 快速导入

```python
import sys; sys.path.insert(0, r'd:\01-Doc\Quant\ft2')
from core import Engine, AccountAnalyzer, context, OrderSide, BenchHolder
from notebook import Notebook
from factor.v4 import FacEngine, FactorExpression, FactorLibrary, FactorValidator  # v4 因子
from signals.v4 import Expression, SigEngine  # v4 择时信号
```

## 数据约定

- DataFrame: `index=日期`, `columns=股票代码`
- 图表: `{'xAxis': [...], 'series': [...]}`

## 开发规范

- 代码修改添加注释: `# [修复] 2026-05-29 描述`
- 修改前先读取源码确认
- 临时脚本/测试文件放 `tmp/` 目录，根目录保持整洁

## 外部模块调用准则

AI Agent 或外部项目调用 ft2 时，遵循以下原则保障上下文沟通简洁：

### 1. 命名对齐 ft2 规范

| 场景 | ft2 规范 | 外部模块应使用 |
|------|----------|---------------|
| OHLCV 数据 | `data`（不是 `raw_data` / `df` / `klines`） | `data` |
| 因子排名面板 | `panel`（不是 `ranked` / `factor_df` / `fv`） | `panel = expr.evaluate_ranked(data)` |
| 回测结果 | `analyzer`（不是 `result` / `backtest` / `r` / `bt`） | `analyzer = FacEngine.backtest(...)` 或 `analyzer = SigEngine.backtest(...)` |
| 信号序列 | `signal`（不是 `sig` / `signals` / `pred`） | `signal` |
| 额外特征 | `extra_features`（不是 `features` / `feats` / `ctx`） | `extra_features` |
| 表达式字符串 | `expr` / `buy_expr` / `sell_expr` | `expr` / `buy_expr` / `sell_expr` |
| 品种名称映射 | `symbol_names`（如 `{'801010.SI': '农林牧渔(申万)'}`） | 嵌入 DataFrame `name` 列即可自动传递 |

### 2. 零封装原则

直接调用 ft2 API，不要创建中间封装层：

```python
# 好 — 直接调 signals.v4，无中间函数
from signals.v4 import stateful_signal, SigEngine
signal = stateful_signal(data, buy_expr, sell_expr, max_hold=10)
r = SigEngine.backtest(signal, data, mode='fast')

# 好 — 直接调 factor.v4，无中间函数
from factor.v4 import FactorExpression, FacEngine
panel = FactorExpression("cs_rank(ts_roc(CLOSE, 20))").evaluate_ranked(data)
r = FacEngine.backtest(panel, assets, top_n=3, mode='full')

# 坏 — 又包一层，Agent 需要学两套 API
def my_backtest(data, b, s):
    sig = stateful_signal(data, b, s)
    return SigEngine.backtest(sig, data)
r = my_backtest(data, buy_expr, sell_expr)
```

### 3. 显式传参，不依赖闭包

```python
# 好 — 参数显式传递
signal = stateful_signal(data, buy_expr, sell_expr, extra_features=ef)
r = SigEngine.backtest(signal, data, mode='fast', start_date=start_date)

# 好 — factor.v4 显式传参
r = FacEngine.backtest(panel, assets, top_n=3, mode='full',
    bench_label='399317.SZ', fee_config={'commission_rate': 0.0003})

# 坏 — 依赖闭包变量，模板复制后跑不通
def stateful_engine(data, buy, sell):
    return stateful_signal(data, buy, sell, extra_features=extra_features)  # extra_features 从哪来？
```

### 4. 能用 v4 就用 v4

ft2 已有功能的不要手写：

```python
# 好 — 用 v4 现成函数
from signals.v4.validate import signal_correlation
corr = signal_correlation(signals, data)

from factor.v4 import FactorExpression
panel = FactorExpression("cs_rank(ts_roc(CLOSE, 20))").evaluate_ranked(data)

# 坏 — 自实现同样逻辑
def my_correlation(signals, data):
    ...  # 45 行手写计算

def my_momentum(data):
    ...  # 手写截面排名+时序ROC
```

### 5. 保持 import 链简单

```python
# 好 — 直接从 v4 顶层导入
from signals.v4 import stateful_signal, SigEngine
from factor.v4 import FacEngine, FactorExpression

# 坏 — 多层 import 跳转
from signals.v4.expression import stateful_signal
from signals.v4.engine import SigEngine
from factor.v4.engine import FacEngine
from factor.v4.expression import FactorExpression
```

### 6. 调用链不超过 2 层

```
  模板 -> signals.v4          ✓ 好
  模板 -> factor.v4           ✓ 好
  模板 -> 自封装层 -> v4      ✗ 中间层无新增价值
```
