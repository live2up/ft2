# ft2 — 量化回测框架

> 快速测试 · 研究输出

---

## 架构总览

| 定位 | 模块 | 核心能力 |
|------|------|----------|
| **基础底座** | **ft2.core** | 回测引擎（事件驱动/向量化双模式），账户管理（AccountMgr + FastAccount），指标计算（Sharpe / CAGR / MDD / WF），基准持有 |
| **衍生模块** | **signals** | 单标的择时策略。AST DSL 表达式引擎，buy/sell 状态机，GP/Grid 搜索，walkforward 验证，市场宽度 |
| | **factor** | 多资产因子轮动。截面排名，IC/IR/Bootstrap 检验，GP 符号回归，LLM 因子生成，WQ101/GT191 公式库 |
| | **pms** | 组合管理系统。多风格选品，多信号合并执行 |
| **辅助模块** | **utils/ast** | 为 signals/factor 提供 DSL 能力。85 原语注册、变量白名单、AST 解析/求值/安全校验 |
| **展示模块** | **ft2.notebook** | 独立 HTML 报告生成器。ECharts + ft-table + Jinja2 → 交互式报告。不依赖 core，任意模块可调用 |

---

## 设计亮点

一个 DIY 量化研究框架，核心思路是**去黑箱**——控制每一层、理解每一行。

| 模块 | 亮点 |
|------|------|
| **ft2.core** | **严谨的回测框架**：fast/full 双模式，向量化 38ms/次 快速扫描，事件驱动完整回测，两种模式 SR 偏差 < 0.000001。AccountAnalyzer 覆盖全部常用指标（SR / CAGR / MDD / WF / 换手 / 胜率 / 盈亏比），一个 Engine 两套 Account 满足从快速扫描到精细回放的全场景需求 |
| **signals** | **Python AST DSL**：直接使用 Python `ast.parse()` 做表达式引擎，不是造轮子写 parser。72 个原语全对齐 WQ101 命名规范，写表达式就是写 Python。`stateful_signal` 支持 buy/sell 分离的状态机策略 |
| **factor** | **GP 符号回归**：AST 树直接当基因组，`ast.NodeTransformer` 做遗传算子，无需编译、无需序列化、无需第三方 ML 库。种子驱动 + SQLite 缓存 + 方向演化追踪，30 代进化即可收敛。vs gplearn（固定原语集 + IC 代理适应度）直接用真实回测 Sharpe 做适应度；vs DEAP（百行配置搭框架）5 行 `GPEngine` 直出；vs 商业黑箱（不可调试）AST 表达式可读可改可直接写通达信 |
| **pms** | **多风格选品**：每个风格独立面板、独立资金、独立选品后合并权重，多头共识自然翻倍 |
| **utils/ast** | **一套 DSL 喂饱两个模块**：85 个原语（ts_ / cs_ / expanding_ / reg_）在 `utils/ast` 一次实现，signals 和 factor 共用。`cs_rank` 自动解算 2D 面板对单品种表达式透明。AST 安全校验拦截 unsafe eval |
| **ft2.notebook** | **替代 Jupyter 输出**：自研 668 行 min_pyecharts 做 ECharts option 组装 + 4 个 Vue3 前端组件（ft-table / ECharts / Grid / Markdown），一个 `Notebook` 对象聚合图表/表格/文本/代码块 → 单页 HTML。离线 `file://` 可直接打开。比 Jupyter 更灵活：一回测一报告，版本可追踪、可分享、可存档 |

---

## 调用关系

衍生模块（signals / factor / pms）→ 调用 ft2.core 回测 → 结果传给 ft2.notebook 出报告

utils/ast ← 被 signals / factor 调用，提供 DSL 解析能力

ft2.notebook ← 任意模块独立调用，无依赖

## 数据流

原始 OHLCV 数据 → signals / factor / pms 计算信号/排名/配置 → ft2.core 回测执行 → 指标计算 → ft2.notebook → HTML 报告

---

## 示例报告

预览文件位于 `demo/` 目录，用浏览器直接打开即可查看：

- **Notebook 综合展示测试** — `demo/Notebook 综合展示测试.html`（Notebook 渲染能力全量测试）
- **双风格并行报告** — `demo/23-a-双风格并行_强RD+稳AE_D频.html`（因子轮动回测报告）