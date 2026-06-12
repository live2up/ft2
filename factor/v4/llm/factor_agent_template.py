"""
因子发现 Agent 模板 2026-06-12 — LLM 生成因子 + IC/回测验证

用法: Agent 复制本模板到工作目录，改 EXPLORE_IDEA + 填充候选区，运行。
      观察 IC/Sharpe 结果 → 反馈 Agent → 迭代优化。

流程:
  1. LLM 批量生成因子表达式 (按 EXPLORE_IDEA)
  2. IC 快速筛选 → ICIR 排名表
  3. ICIR>阈值的因子做轮动回测 → Sharpe 排名
  4. Top N 入库 + HTML 报告
  5. 历史去重: 跳过已入库因子

依赖:
  - factor/v4/llm/ (LLMGenerator + eval_utils)
  - signals/v4 (Expression + AST DSL)
  - factor/v4 (V4FactorExpression + Validator + Backtest)
"""
import sys, os, json
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, r'D:\01-Doc\Quant\ft2')
sys.path.insert(0, r'D:\01-Doc\Quant\d2api')

from datetime import datetime
import numpy as np, pandas as pd
from factor.v4.llm import (
    LLMGenerator, quick_ic_batch, quick_rank_panel, quick_sharpe,
)
from factor.v4.base import FactorLibrary, LibraryEntry
from notebook import Notebook

# ══════════════════════════════════════════════════════════════
# Agent 配置区 (Agent 在此填充)
# ══════════════════════════════════════════════════════════════

# LLM provider: "deepseek" | "openai" | "custom"
LLM_PROVIDER = "deepseek"
LLM_MODEL = "deepseek-chat"
LLM_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")

# 探索配置
EXPLORE_IDEA = "量价背离方向，价涨量缩的反转信号"  # 因子想法 (自然语言)
N_GENERATE = 20          # 每轮生成数量
N_TOP_IC = 10            # IC 筛选 Top N 做回测
ICIR_THRESHOLD = 1.0     # 最低 ICIR 阈值
REBALANCE = 'W'          # 调仓频率 'W'|'M'|'5D'
TOP_N_HOLD = 3           # 持仓品种数

# ══════════════════════════════════════════════════════════════
# 数据加载 (根据实际数据源修改)
# ══════════════════════════════════════════════════════════════

# TODO: 替换为实际数据加载
# from d2_api import d2_api
# 加载行业指数 OHLCV 面板数据
# assets = {code: d2_api.kline.query(code, ...) for code in CODES}
# panel_data = build_panel_dict(assets)
# returns = build_returns_df(assets)

# 示例: 从 pickle 加载 (替换为实际路径)
import pickle
DATA_PATH = "tmp/panel_data.pkl"  # TODO: 修改为实际路径
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"数据文件不存在: {DATA_PATH}。请先准备面板数据。")

with open(DATA_PATH, 'rb') as f:
    saved = pickle.load(f)
    panel_data = saved['panel']        # {变量: ndarray(T,N)}
    assets = saved['assets']           # {品种代码: OHLCV DataFrame}
    returns = saved['returns']         # DataFrame(日期×品种)
    dates = saved['dates']
    symbols = saved['symbols']

print(f"数据: {len(symbols)} 品种, {len(dates)} 天 ({dates[0]} ~ {dates[-1]})")

# ══════════════════════════════════════════════════════════════
# 去重: 加载已入库因子
# ══════════════════════════════════════════════════════════════

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

LIB_PATH = os.path.join(OUTPUT_DIR, 'factor_library.json')
if os.path.exists(LIB_PATH):
    library = FactorLibrary.load(LIB_PATH)
    seen_exprs = set(e.expression for e in library._entries.values())
    print(f"[历史] 因子库已收录 {len(seen_exprs)} 个因子\n")
else:
    library = FactorLibrary()
    seen_exprs = set()

# ══════════════════════════════════════════════════════════════
# Step 1: LLM 生成因子表达式
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Step 1: LLM 生成 — {EXPLORE_IDEA[:50]}")
print(f"{'='*60}")

gen = LLMGenerator(
    provider=LLM_PROVIDER,
    model=LLM_MODEL,
    api_key=LLM_API_KEY,
)

# 用 Top 因子上下文引导生成
top_context = ""
if library.size() > 0:
    top_factors = library.top(5, sort_by='icir')
    top_context = "\n".join(
        f"  {e.alpha_id}: {e.expression} (ICIR={e.icir:.2f})"
        for e in top_factors
    )

candidates = gen.generate(
    idea=EXPLORE_IDEA,
    n=N_GENERATE,
    context=top_context,
    avoid=list(seen_exprs),
)

# 去重 + 过滤空
candidates = [c for c in candidates if c and c not in seen_exprs]
print(f"有效候选: {len(candidates)} 个\n")

if not candidates:
    print("无新候选因子，退出。")
    exit(0)

# ══════════════════════════════════════════════════════════════
# Step 2: IC 快速筛选
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Step 2: IC 筛选 (ICIR>{ICIR_THRESHOLD})")
print(f"{'='*60}")

ic_results = quick_ic_batch(candidates, panel_data, returns,
                            min_icir=ICIR_THRESHOLD)
ic_results = ic_results[:N_TOP_IC]

if not ic_results:
    print("无因子通过 IC 筛选。")
    exit(0)

print(f"\n通过 IC 筛选: {len(ic_results)} 个")
for i, r in enumerate(ic_results):
    print(f"  #{i+1} IC={r.ic_mean:+.4f}  IR={r.icir:+.2f}  Pos={r.positive_ratio:.0%}  t={r.t_stat:.1f}  {r.expression[:80]}")

# ══════════════════════════════════════════════════════════════
# Step 3: 轮动回测
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"Step 3: 轮动回测 (Top{TOP_N_HOLD}, {REBALANCE}调仓)")
print(f"{'='*60}")

backtest_results = []
for ic_r in ic_results:
    try:
        sharpe = quick_sharpe(ic_r.expression, assets,
                              top_n=TOP_N_HOLD, rebalance=REBALANCE)
        backtest_results.append({
            'expression': ic_r.expression,
            'ic_mean': ic_r.ic_mean,
            'icir': ic_r.icir,
            'sharpe': sharpe,
            'positive_ratio': ic_r.positive_ratio,
            't_stat': ic_r.t_stat,
        })
        print(f"  SR={sharpe:.3f}  IC={ic_r.ic_mean:+.4f}  {ic_r.expression[:70]}")
    except Exception as e:
        print(f"  X FAIL: {ic_r.expression[:70]} — {e}")
        backtest_results.append({
            'expression': ic_r.expression, 'ic_mean': ic_r.ic_mean,
            'icir': ic_r.icir, 'sharpe': -99, 'positive_ratio': 0, 't_stat': 0,
        })

bt_df = pd.DataFrame(backtest_results).sort_values('sharpe', ascending=False)

# ══════════════════════════════════════════════════════════════
# Step 4: 入库 + 报告
# ══════════════════════════════════════════════════════════════

print(f"\n{'='*60}")
print(f"结果汇总 (Top {min(5, len(bt_df))})")
print(f"{'='*60}")

new_entries = []
for _, r in bt_df.head(10).iterrows():
    if r['sharpe'] > 0.3:
        rank = len(new_entries) + 1
        tag = "++" if r['sharpe'] > 1.0 else ("+" if r['sharpe'] > 0.5 else "~")
        entry_id = f"llm_{datetime.now().strftime('%m%d%H%M')}_{rank:03d}"
        entry = LibraryEntry(
            alpha_id=entry_id,
            expression=r['expression'],
            icir=r['icir'],
            sharpe=r['sharpe'],
            source='llm',
        )
        new_entries.append(entry)
        print(f"  {tag} SR={r['sharpe']:.3f}  IC={r['ic_mean']:+.4f}  IR={r['icir']:.2f}  {r['expression'][:70]}")

if new_entries:
    library.register_batch(new_entries)
    library.save(LIB_PATH)
    print(f"\n入库: {len(new_entries)} 个 → {LIB_PATH}")

# ══════════════════════════════════════════════════════════════
# HTML 报告
# ══════════════════════════════════════════════════════════════

title = f"factor_{EXPLORE_IDEA[:15].replace(' ', '_')}_{datetime.now().strftime('%m%d_%H%M%S')}"

nb = Notebook(f"因子发现: {EXPLORE_IDEA[:30]}")

# 概览
nb.metrics([
    {'name': '探索想法', 'value': EXPLORE_IDEA[:40]},
    {'name': 'LLM生成', 'value': str(len(candidates))},
    {'name': 'IC筛选通过', 'value': str(len(ic_results)), 'color': 'green'},
    {'name': '入库', 'value': str(len(new_entries)), 'color': 'green'},
    {'name': f'Top1 Sharpe', 'value': f"{bt_df.iloc[0]['sharpe']:.3f}" if len(bt_df) > 0 else 'N/A'},
    {'name': f'Top1 IC', 'value': f"{bt_df.iloc[0]['ic_mean']:+.4f}" if len(bt_df) > 0 else 'N/A'},
], columns=3)

# IC 排名表
ic_rows = [{
    '排名': i+1,
    '表达式': r.expression,
    'IC均值': r.ic_mean,
    'IR': r.icir,
    '正占比': f"{r.positive_ratio:.0%}",
    't值': r.t_stat,
} for i, r in enumerate(ic_results)]
nb.table(ic_rows, columns=['排名','表达式','IC均值','IR','正占比','t值'],
         freeze={'left': 1}, heatmap={'start': 2, 'axis': 'column'},
         title=f'IC 排名 ({EXPLORE_IDEA[:30]})')

# 回测排名表
bt_rows = [{
    '排名': i+1,
    '表达式': r['expression'],
    'Sharpe': round(r['sharpe'], 3),
    'IC均值': round(r['ic_mean'], 4),
    'IR': round(r['icir'], 2),
} for i, (_, r) in enumerate(bt_df.head(10).iterrows())]
nb.table(bt_rows, columns=['排名','表达式','Sharpe','IC均值','IR'],
         freeze={'left': 1}, heatmap={'start': 2, 'axis': 'column'},
         title=f'轮动回测排名 (Top{TOP_N_HOLD}, {REBALANCE})')

report_path = nb.export_html(title)
print(f"\n报告: {report_path}")
