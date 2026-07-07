# ft2 — 量化回测框架

> 单标的择时 + 多资产因子轮动 + AST DSL 符号回归

---

## 架构总览

**ft2.core** — 回测引擎底座
- Engine（事件驱动/向量化双模式）、AccountMgr、FastAccount、AccountAnalyzer、BenchHolder

**衍生模块**（基于 core 构建）：
- **signals** — 择时策略。AST DSL 表达式引擎，buy/sell 状态机，GP/Grid 搜索，walkforward 验证
- **factor** — 因子轮动。截面排名，IC/IR 检验，GP 符号回归，LLM 因子生成，WQ101/GT191 公式库
- **pms** — 组合管理。多风格选品，多信号合并执行

**辅助模块**（被 signals/factor 共用）：
- **utils/ast** — DSL 扩展。72+原语注册、变量白名单、AST 解析/求值/安全校验

**展示模块**（独立，各模块均可调用）：
- **ft2.notebook** — HTML 报告生成器。Notebook → ECharts 图表 + ft-table 表格 + Jinja2 模板 → 交互式报告

---

## 模块职责

| 模块 | 定位 | 核心功能 |
|------|------|----------|
| **ft2.core** | 基础底座 | 回测引擎（事件驱动/向量化双模式），账户管理，指标计算（Sharpe / CAGR / MDD / WF），基准持有 |
| **signals** | 衍生模块 | 单标的择时策略。AST DSL 表达式引擎，buy/sell 状态机，GP/Grid 搜索，walkforward 验证，市场宽度 |
| **factor** | 衍生模块 | 多资产因子轮动。截面排名，IC/IR/Bootstrap 检验，GP 符号回归，LLM 因子生成，WQ101/GT191 公式库 |
| **pms** | 衍生模块 | 组合管理系统。多风格选品，多信号合并执行 |
| **utils/ast** | 辅助模块 | 为 signals/factor 提供 DSL 能力。原语注册、变量白名单、AST 解析/求值/安全校验 |
| **ft2.notebook** | 展示模块 | 独立 HTML 报告生成器。ECharts + ft-table + Jinja2 → 交互式报告。不依赖 core，任意模块均可调用 |

---

## 调用关系

衍生模块（signals / factor / pms）→ 调用 ft2.core 回测 → 结果传给 ft2.notebook 出报告

utils/ast ← 被 signals / factor 调用，提供 DSL 解析能力

ft2.notebook ← 任意模块独立调用，无依赖

## 数据流

原始 OHLCV 数据 → signals / factor / pms 计算信号/排名/配置 → ft2.core 回测执行 → 指标计算 → ft2.notebook → HTML 报告

---

## 示例报告

> 以下 HTML 报告由 ft2.notebook 生成，托管于 GitHub Pages（CDN: `livepu/ft2`）。

| 报告 | 说明 |
|------|------|
| [Notebook 综合展示测试](https://cdn.jsdelivr.net/gh/livepu/ft2@master/demo/Notebook%20%E7%BB%BC%E5%90%88%E5%B1%95%E7%A4%BA%E6%B5%8B%E8%AF%95.html) | Notebook 渲染能力全量测试：图表类(折线/柱状/饼图/散点/K线/热力/Grid) · 表格类 · 文本类 |
| [双风格并行报告](https://cdn.jsdelivr.net/gh/livepu/ft2@master/demo/23-a-%E5%8F%8C%E9%A3%8E%E6%A0%BC%E5%B9%B6%E8%A1%8C_%E5%BC%BARD+%E7%A8%B3AE_D%E9%A2%91.html) | 因子轮动回测报告：双风格(强RD+稳AE)并行策略，日频调仓 |