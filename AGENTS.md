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

> 以上 `AI.md` 各自包含完整的 API 速查、架构图、参数说明和注意事项。
> 不依赖 IDE 特定的 Skill 机制，跨平台可用。

## 快速导入

```python
import sys; sys.path.insert(0, r'd:\01-Doc\Quant\ft2')
from core import Engine, AccountAnalyzer, context, account, OrderSide, BenchHolder
from notebook import Notebook
from factor.v3 import FactorExpression, GPEngine, FactorDiscoveryEngine  # v3 推荐
from signals.v2 import Expression, FeatureSpace, run_backtest_with_core  # v2.1 推荐
```

## 数据约定

- DataFrame: `index=日期`, `columns=股票代码`
- 图表: `{'xAxis': [...], 'series': [...]}`

## 开发规范

- 代码修改添加注释: `# [修复] 2026-05-29 描述`
- 修改前先读取源码确认
- 临时脚本/测试文件放 `tmp/` 目录，根目录保持整洁