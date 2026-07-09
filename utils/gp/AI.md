# utils/gp — 遗传编程（GP）引擎

> 本目录是**遗传编程引擎的共享基础设施**，供 `factor` 和 `signals` 模块复用。
> 当前实现为 **v5**，核心代码在 `v5/` 子目录（`engine.py` / `config.py` / `tree_gen.py` / `ast_utils.py` / `cache.py`）。
> 本文只讲「怎么用」，行为一律以 `v5/` 当前代码为准。

> **版本 v5 | 更新 2026-07-09** — 同步引擎最新默认配置与新增能力：ε-Lexicase 选择（默认开）、年龄机制（默认开）、Motif 种子库（默认开）、岛屿模型、4 算子变异（subtree/param/logic/insert_cond）；`DEFAULT_GP_CONFIG` 默认种群/代数/深度已调为 500/40/5。

---

## 1. 定位与能力

`GPEngine` 是一个 **因子组合优化 GP 引擎**：

- **Python AST 原生**：搜索空间 = 智能体能写出的任何合法表达式（变量 + 时序函数 + 数学函数 + 比较/逻辑/三元运算）。
- **种子驱动**：种群主要来自智能体（或人类）发现的因子表达式种子，在此基础上组合/变异。
- **组合优先**：重点搜索最优组合方式（and / or / if-else / 加权 / 条件门控），而非从零随机搜索。

> 注意：v5 引擎**只做搜索与评估调度**，不绑定任何具体的因子语义。
> 它接收一个 `evaluator`（把 AST 树 → 因子值面板）和一个 `fitness_calculator`（把因子值面板 → 标量适应度）。
> 因子语义由调用方（如 `factor.v5.gp_engine` 包装层）通过 `evaluator` 注入。

### v5 的核心特征：权重聚焦探索

v5 最关键的升级，是**搜索空间可被参数「聚焦」**，而非全程等概率随机：

- 通过 `TreeGenConfig` 的 `var_weights` / `ts_weights` / `math_weights` 给变量和函数分配选择权重，把搜索流量导向你认定的有效方向（详见 §5）。
- 配合 `var_allowlist` / `func_allowlist` 可**硬白名单锁定**搜索范围。

**这种「权重聚焦」是 v5 的招牌能力**：它让 GP 从「盲搜」变成「在你指定的方向上精搜」，
大幅减少无效表达式、提高找到有效因子的效率。使用 v5 时应**有意识地设计权重与白名单**，而非放任默认等概率。

---

## 2. 模块结构

```
utils/gp/
├── __init__.py            # 仅含一句 docstring，无导出
├── AI.md                  # 本文
└── v5/                    # GP 引擎 v5
    ├── __init__.py        # 统一导出 GPEngine / TreeGenConfig / Individual / FitnessCache / 工具函数
    ├── engine.py          # GPEngine 主类：进化循环、选择、交叉、变异、方向追踪
    ├── config.py          # TreeGenConfig + Individual + DEFAULT_GP_CONFIG / DEFAULT_TREE_GEN_CONFIG + 原语表
    ├── tree_gen.py        # 随机树生成 + 4 种变异算子（subtree/param/logic/insert_cond）
    ├── ast_utils.py       # AST 纯函数：_simplify_ast / _canonicalize_key / _collect_replaceable / _replace_subtree
    └── cache.py           # FitnessCache：内存 + SQLite 双级缓存
```

统一导入：

```python
from utils.gp.v5 import (
    GPEngine, Individual, TreeGenConfig,
    DEFAULT_GP_CONFIG, DEFAULT_TREE_GEN_CONFIG,
)
from utils.gp.v5.config import GP_VARIABLES, TS_FUNCTIONS, MATH_FUNCTIONS  # 原语表
```

---

## 3. 快速上手

### 3.1 经 factor 包装层（推荐，已自动注入 evaluator）

`factor.v5.gp_engine.GPEngine` 是 `utils.gp.v5.GPEngine` 的子类，自动注入因子端 evaluator
（`_ExpressionFromAST(tree).evaluate(data)`）。**用因子搜索时一律走这个，不用自己传 evaluator。**

```python
import numpy as np
from factor.v5.gp_engine import GPEngine, TreeGenConfig
from factor.v5 import FacEngine                       # 用于适应度计算

# data: {列名(str): ndarray(T, N)}，裸 OHLCV + REL_* 终端变量面板
data = {...}

# 适应度：用 FacEngine 算 SR
class SRFitness:
    def __init__(self, returns, top_n=2, rebalance='D', start_date=None):
        self.kw = dict(returns=returns, top_n=top_n, rebalance=rebalance,
                       mode='vector', start_date=start_date)
    def compute(self, fv):
        fv = np.nan_to_num(fv, 0.0)
        r = FacEngine.backtest(fv, **self.kw)
        return getattr(r, 'sharpe', -999.0)

engine = GPEngine(
    data=data,
    fitness_calculator=SRFitness(returns, top_n=2),
    seed_expressions=[
        "cs_rank(ts_roc(CLOSE, 20))",           # 种子：智能体发现的好因子
        "gauss(ts_roc(AMOUNT, 40))",
    ],
    random_seed=42,                             # 可复现
).run(verbose=True)

best = engine.best()
print(best.expression_str, best.fitness)
```

### 3.2 直接用核心（必须传 evaluator）

直接用 `utils.gp.v5.GPEngine` 时**必须传 `evaluator`**，否则所有个体 fitness 会被判为 `-999`：

```python
from utils.gp.v5 import GPEngine, TreeGenConfig

def my_evaluator(data: dict, tree) -> np.ndarray:
    """把 AST 树 → 因子值面板 ndarray(T, N)。返回非有限值会被判 -999。"""
    return _ExpressionFromAST(tree).evaluate(data)   # 用你自己的 AST 求值器

engine = GPEngine(
    data=data,
    fitness_calculator=my_fitness,
    evaluator=my_evaluator,        # ← 必须
    random_seed=42,
)
```

---

## 4. GPEngine 构造参数

| 参数 | 类型 | 默认 | 说明 |
|------|------|------|------|
| `data` | `Dict[str, np.ndarray]` | — | 终端变量面板，值形状需为 2D `(T, N)`（引擎据此推断数据形状） |
| `fitness_calculator` | 对象 | `None` | 必须有 `.compute(factor_values) -> float` 方法 |
| `evaluator` | `Callable` | `None` | `(data, tree) -> ndarray`；直用核心时**必传** |
| `config` | `Dict` | `None` | 覆盖 `DEFAULT_GP_CONFIG` 的引擎级参数（见 §6） |
| `seed_expressions` | `List[str]` | `None` | 种子表达式字符串列表，按比例注入初始种群 |
| `random_seed` | `int` | `None` | 传入后种群可复现（内置独立 `random.Random`） |
| `tree_gen_config` | `TreeGenConfig` | `None` | 树生成偏置（见 §5） |
| `cache_db` | `str` | `''` | 缓存 DB 路径；优先级：`cache_db` 参数 > `config['cache_db']` > `./output/.gp_cache.db` |
| `future_returns` / `returns` | — | `None` | 透传字段，引擎本身不消费，供 `fitness_calculator` 使用 |

---

## 5. TreeGenConfig — 权重聚焦的核心配置

`TreeGenConfig` 是 v5「权重聚焦探索」的**唯一控制入口**。它决定随机生树时
**变量 / 函数 / 运算**各自被选择的概率，从而把搜索流量导向你指定的方向。
默认值 `None` = 等概率（但会浪费大量搜索在无效组合上）。

```python
cfg = TreeGenConfig(
    var_weights={'AMOUNT': 3, 'VOLUME': 2},   # 变量权重聚焦：AMOUNT 出现概率是 VOLUME 的 1.5 倍
    var_allowlist={'CLOSE', 'AMOUNT'},         # 硬白名单：只允许这两个变量出现
    func_allowlist={'ts_rank', 'cos'},         # 硬白名单：只允许这两个函数
    ts_weights={'ts_rank': 2, 'ts_cov': 1},    # 函数级聚焦（未列即禁用）
    mode='continuous',                          # None / 'continuous' / 'predicate' / 'hybrid'
    adaptive=True,                              # 开启 EMA 方向权重闭环（默认 False）
    adaptive_lr=0.3, adaptive_every=3,
    rng=random.Random(42),                      # 可复现
)
```

### 5.1 权重参数速查（权重聚焦的关键旋钮）

| 字段 | 类型 | 聚焦对象 | 行为要点 |
|------|------|----------|----------|
| `var_weights` | `Dict[str, float]` | 变量 | **部分传入 = 未列出的变量被填 0 即禁用**；权重越高该变量被选中概率越大 |
| `ts_weights` | `Dict[str, float]` | 时序/截面/特征函数 | 控制 `ts_*` / `cs_*` / `linearreg` / `natr` / `amt_ratio` 等**全部非数学函数**；**未列即禁**（键来自 `FUNC_REGISTRY` 动态填充，不再有独立 `feature_weights`） |
| `math_weights` | `Dict[str, float]` | 数学函数 | 控制 `sin/cos/exp/...` 等，键来自 `FUNC_REGISTRY` 的 `math_function` 分类 |
| `group_weights` | `Dict[str, float]` | 运算大类 | 控制 ts/math/comparison/logic/binary_op/unary_op/ternary **七大类**的相对概率（一般不改） |

> **权重聚焦的本质**：权重不是「附加偏好」，而是**选择概率分布**。引擎用
> `random.choices(keys, weights=...)` 按权重抽样，所以 `var_weights={'AMOUNT':3}`
> 等价于「整个种群只会出现 AMOUNT」。想「偏置但不禁止」，必须把全部 6 个变量都写上、按相对比例赋权。

### 5.2 白名单与模式

| 字段 | 作用 |
|------|------|
| `var_allowlist` / `func_allowlist` | **硬白名单**，优先级高于权重；与权重联合使用时取交集 |
| `mode='continuous'` | 生成连续数值表达式，禁用比较/逻辑/三元（适合纯因子值搜索） |
| `mode='predicate'` | 只生成布尔谓词（适合择时信号的条件分支） |
| `mode=None` / `'hybrid'` | 全开，混合数值与布尔 |

### 5.3 自适应权重闭环（adaptive）

开启 `adaptive=True` 后，引擎每 `adaptive_every` 代统计各「方向签名」（`m:数学|t:时序|v:变量`）
的历史最佳 fitness，用 EMA 把表现好的变量/函数权重调高、差的压低（`lr=adaptive_lr`，默认 0.3），
并 `max(weight, 0.05)` 归一化。这是**权重聚焦的自动版**——你给初值，引擎帮你迭代收敛。
需累计 ≥10 条方向记录才首次更新（**默认关**，需显式开启）。

### 5.4 内置原语（权重键的来源）

权重字段的合法键**不再是手写常量表**，而是运行时从 `utils.ast.functions.FUNC_REGISTRY` 动态读取（见 `config.get_full_default_weights()`）：
- 变量 `var_weights` 的键：`GP_VARIABLES = ['CLOSE','OPEN','HIGH','LOW','VOLUME','AMOUNT']`（仅 fallback 池；自定义变量经 `ast.register_variable()` 注册后由 `var_weights` 指定偏置，不在此列）
- `ts_weights` 的键：**所有非数学函数** —— `ts_function` / `cs_function` / `ta_function` / `feature_function` / `signal_function` 分类下的全部函数（如 `ts_rank/ts_roc/ts_cov/linearreg/natr/amt_ratio/...`）
- `math_weights` 的键：`math_function` 分类（如 `sin/cos/exp/log/sqrt/abs/tanh/gauss/p4/neg`）

> **坑**：权重键拼错（如 `'Ts_Rank'`）会被忽略、该键退化为禁用（变量黑名单语义见 §5.1 表）。

---

## 6. 引擎级参数（config）

`DEFAULT_GP_CONFIG` 默认值如下，可通过构造 `config=` 字典覆盖（**当前默认已为强探索配置**，kb03 等实战脚本即直接复用此默认）：

| 参数 | 默认 | 说明 |
|------|------|------|
| `population_size` | 500 | 种群规模（kb03 用 500 提升探索广度） |
| `generations` | 40 | 进化代数（kb03 用 40 确保收敛） |
| `max_depth` | 5 | 树最大深度（浅树防过拟合、保可解释；变异子树深度上限取 `min(max_depth, 4)`） |
| `tournament_size` | 6 | 锦标赛选择规模 |
| `crossover_prob` | 0.6 | 交叉概率（与 `mutation_prob` 共用一个 r 区间） |
| `mutation_prob` | 0.25 | 变异概率 |
| `elite_ratio` | 0.05 | 精英保留比例（至少 1 个） |
| `seed_ratio` | 0.15 | 初始种群中种子表达式占比 |
| `random_inject_ratio` | 0.15 | 每代随机注入个体占比（探索多样性） |
| `parsimony_penalty` | 0.001 | 简洁度惩罚：`fitness *= (1 - penalty * node_count)` |
| `lexicase` | True | **ε-Lexicase 选择默认开**（见 §7） |
| `epsilon` | 0.05 | ε-Lexicase 容差：fitness 差距 ≤5% 的个体同为代表，保留方向多样性 |
| `age_enabled` | True | **年龄机制默认开**：fitness 随个体年龄衰减，防种群老龄化（见 §7） |
| `age_penalty_lr` | 0.05 | 年龄惩罚系数：`fitness *= 1/(1 + lr*age)` |
| `motif_enabled` | True | **Motif 种子库默认开**（见 §7） |
| `motif_update_every` | 3 | 每 N 代从 Top 30% 个体提取高频子树更新 Motif 库 |
| `motif_min_fitness` | 0.0 | 仅提取 fitness ≥ 此阈值的个体子树 |
| `motif_max_depth_ratio` | 0.5 | Motif 子树最大深度（= `max_depth * ratio`） |
| `motif_inject_count` | 5 | 每次初始化/注入最多掺入的 Motif 种子数 |
| `num_islands` | 1 | 岛屿数量（>1 启用岛屿模型，见 §7） |
| `migrate_every` | 5 | 岛屿模式：每 N 代迁移一次 |
| `migrate_count` | 2 | 岛屿模式：每岛每次迁出/替换个体数 |
| `elite_max_per_sig` | 1 | 方向配额精英：每「方向签名」最多保留 N 个代表（`None`=不限制）；岛屿/lexicase 模式下生效 |
| `mutate_*_weight` | 见下 | 4 种变异算子权重 |

变异算子权重默认：`subtree=0.30, param=0.40, logic=0.15, insert_cond=0.15`。
> `param` 算子统一负责常数与窗口的重新采样：spec 从 `param_pool`/`param_ranges` 感知式选取（排除 bool 节点）。

---

## 7. 内置算法与默认开关

| 算法 | 默认 | 行为 / 开启方式 |
|------|------|------|
| **AW-MEP 停滞检测** | **开** | 连续 3 代无提升 → 变异率 ×1.3（上限 0.5）、交叉率 ×0.85（下限 0.2）；连续 5 代 → 随机注入数翻倍（上限 30%） |
| **ε-greedy 探索** | **开（explore_ratio=0.15）** | 约 15% 个体无视用户权重、用全默认空间生成（`_random_tree_explore`，仍遵守 `var_allowlist`/`func_allowlist`），保持探索广度。由 `config={'explore_ratio': 0.15}` 控制 |
| **ε-Lexicase 选择** | **开（epsilon=0.05）** | 按「方向签名」分组，每组取 fitness 最高代表 + 均匀随机选，使小众方向也能被选中。fitness 差距 ≤`epsilon`(5%) 的个体同为代表。由 `config={'lexicase': True, 'epsilon': 0.05}` 控制；**岛屿模式会强制开启** |
| **年龄机制（age）** | **开（age_enabled=True）** | 个体 fitness 随存活代数衰减 `fitness *= 1/(1 + lr*age)`，淘汰长期占席的老个体，维持多样性。由 `config={'age_enabled': True, 'age_penalty_lr': 0.05}` 控制 |
| **Motif 种子库** | **开（motif_enabled=True）** | 每 `motif_update_every` 代从当前种群 Top 30% 提取高频子树，作为高质量种子掺入初始化与每代注入，加速收敛。由 `config={'motif_enabled': ..., 'motif_inject_count': 5}` 控制 |
| **岛屿模型（islands）** | **关（num_islands=1）** | 分 N 个独立子种群隔离演化 + 环形拓扑多样性迁移（迁出接收岛缺失方向、替换冗余个体）。开：`config={'num_islands': 4, 'migrate_every': 5, 'migrate_count': 2}`，会自动开启 Lexicase |
| **EMA 方向权重** | **关** | 闭环自适应权重。开：`TreeGenConfig(adaptive=True)`。需累计 ≥10 条方向记录才更新 |
| **AST 简化 `_simplify_ast`** | **开** | 每次生成/变异/交叉后调用，消除 `neg(neg(x))`、`cos(neg(x))→cos(x)`、`x*1→x`、`x+0→x`、`x-x→0` 等恒等/冗余 |
| **预筛 `_quick_filter`** | **开** | 拒绝无效表达式：窗口=1 的时序函数、`ts_func(负数, ...)`、无变量、仅常量+≤2 变量 |
| **FitnessCache（SQLite）** | **开** | 默认缓存到 `output/.gp_cache.db`，按数据形状+惩罚生成指纹，跨运行复用（懒加载） |

> **默认开关**：v5 当前默认 **Lexicase / 年龄机制 / Motif 库 均为开启**，EMA 自适应权重仍为关闭（需 `adaptive=True` 显式开启）。放任全部默认即可获得较稳健的多样性保护，但仍建议按任务主动设计 `TreeGenConfig` 权重。

**v5 的设计主线是「权重聚焦 + 多样性保障」**：`TreeGenConfig` 的权重/白名单把先验有效方向「点亮」，
上述多样性机制（ε-greedy / AW-MEP / 年龄机制 / Motif 库 / ε-Lexicase）协同防止过早收敛与过拟合，
`adaptive=True` 进一步把「点亮」变成自动迭代收敛。**用 v5 时务必主动设计这些参数**，
放任默认等概率会退化成低效率的盲搜。

---

## 8. 运行与结果读取

```python
engine.run(verbose=True)            # verbose: 每 5 代打印日志
# 可选回调：每代结束 callback(gen, best_individual, stats)
engine.run(callback=lambda gen, best, stats: print(gen, best.fitness))
```

结果 API：

| 方法 / 属性 | 返回 | 说明 |
|-------------|------|------|
| `engine.best()` | `Individual` | 全局最优个体 |
| `engine.top(n=10)` | `List[Individual]` | 当前种群 fitness 前 n（岛屿模式自动合并所有岛） |
| `engine.report()` | `str` | 文本版搜索报告（每 5 代 best_fitness / depth / valid） |
| `engine.direction_report()` | `str` | 方向探索报告（按语义签名聚合各方向最佳 fitness，★=≥0.8） |
| `engine.summary(title='')` | `str` | 启动配置摘要（变量/原语/参数/高级特性开关一览），建议 `run()` 前打印 |
| `engine.get_motif_seeds(n=10)` | `List[str]` | 当前 Motif 库 Top-N 种子表达式（跨轮次复用高质量子树） |
| `engine.history` | `List[Dict]` | 每代统计：`generation/best_fitness/best_depth/best_expression/avg_fitness/valid_count` |

`Individual` 字段：`tree`(ast.Expression) / `fitness` / `expression_str` / `depth` / `node_count` / `generation` / `is_seed` / `age`（存活代数，年龄惩罚用） / `signature`（方向签名缓存）。

---

## 9. 缓存机制

`FitnessCache` 为内存 + SQLite 双级（**懒加载**）：

- 构造时按 `数据形状 + parsimony_penalty` 生成 12 位 `fitness_hash`，隔离不同实验。
- `get()` 先查内存，未命中再按 `expr_hash` 从 SQLite **单条懒加载**（避免一次性把 10 万+ 历史缓存载入内存）；`save()` 结尾批量写回。
- 同一表达式（经 `_canonicalize_key` 规范化，含加乘交换律排序、常数折叠）只评估一次。
- 换缓存路径即可隔离不同项目：`cache_db='my_run/.gp_cache.db'`（构造参数，优先级高于 `config['cache_db']`）或 `config={'cache_db': ...}`。

---

## 10. 与 factor / signals 的关系

- `utils.gp.v5` 是**纯算法核心**，不依赖因子语义。
- `factor.v5.gp_engine.GPEngine` 是其子类，仅注入 `_ExpressionFromAST` evaluator（自动把树转成因子面板）。
- 同理 `signals` 若需 GP，复用同一核心、注入自己的 evaluator 即可。
- **新探索统一用 `factor.v5.gp_engine` / `factor.v5`**；直用 `utils.gp.v5` 核心时务必传 `evaluator`。

---

## 11. 常见坑

1. **直用核心忘了传 `evaluator`** → 所有个体 fitness = -999。经 `factor.v5.gp_engine` 则自动注入，无此问题。
2. **`var_weights` 是黑名单式** → 只写想要的变量，其余自动禁用；想要"附加偏置"得把全部 6 个变量都写上。
3. **适应度函数必须返回有限标量** → 返回 NaN / 常数面板 / `< -998` 会被判 -999 并计入缓存（浪费搜索）。
4. **变异子树深度上限 = min(max_depth, 4)** → 深树主要靠交叉和初始随机树，变异是小步调整（max_depth 默认 5）。
5. **adaptive 默认关闭** → 需权重闭环时显式 `TreeGenConfig(adaptive=True)`，且需累计 ≥10 条方向记录才首次更新。**Lexicase 默认已开（ε-Lexicase）**，岛屿模式会自动强制开启；若想退化为锦标赛选择用 `config={'lexicase': False}`。
6. **缓存跨运行复用** → 改了 `fitness_calculator` 逻辑但未换 `cache_db`，可能读到旧适应度。换 DB 或清文件可强制重算。
