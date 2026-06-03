# ft2 — 量化回测框架

> AI 助手自动加载。详细用法请使用对应的 Skill。

## 导入

```python
import sys; sys.path.insert(0, r'd:\01-Doc\Quant\ft2')
from notebook import Notebook
from core import Engine, AccountAnalyzer
from factor.v2 import Factor, FactorPipeline, FactorValidator  # v2 仍在探索
from factor.v3 import FactorExpression, GPEngine, FactorDiscoveryEngine  # v3 新架构
from signals.v2 import Expression, FeatureSpace, run_backtest
```

## 模块

| 模块 | Skill | 用途 |
|------|-------|------|
| core | ft2-core | 回测引擎 + HTML 报告（已融合 notebook） |
| factor | ft2-factor | 因子发现 (v3: GP符号回归+可插拔适应度) |
| signals | ft2-signals | 择时信号 |

## 数据约定

- DataFrame: `index=日期`, `columns=股票代码`
- 图表: `{'xAxis': [...], 'series': [...]}`

## 开发规范

- 代码修改添加注释: `# [修复] 2026-05-29 描述`
- 修改前先读取源码确认
- 临时脚本/测试文件放 `tmp/` 目录，根目录保持整洁