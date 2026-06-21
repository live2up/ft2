"""
demo/signal_backtest_demo.py — signals.v4 择时回测范式 (EngineCore)
====================================================================
[更新] 2026-06-21 从 signals.v3 迁移到 signals.v4 (EngineCore)
[重命名] 2026-06-10 core_backtest_notebook → signal_backtest_demo（避免误解为裸 core 引擎）

信号探测/择时回测的标准范式：
  Expression → EngineCore.backtest() → AccountAnalyzer.to_notebook()

展示 header/footer 回调能力：
  - header: 对比表 / KPI卡片 / 自定义图表
  - footer: 结论 / 方法论说明

v3 → v4 关键差异（必须注意）:
  ┌──────────────────────┬──────────────────────────────────┐
  │ v3 语义                │ v4 等价                          │
  ├──────────────────────┼──────────────────────────────────┤
  │ thr_mean(X)           │ X > expanding_mean(X)            │
  │ thr_med(X)            │ X > expanding_median(X)          │
  │ thr_roll_mean(X, w)   │ X > ts_mean(X, w)               │
  │ thr_roll_med(X, w)    │ X > ts_median(X, w)             │
  │ div (binary)          │ and (均先用 np.minimum 取 min)   │
  │ sub                   │ -                                │
  │ &                     │ and                              │
  │ |                     │ or                               │
  ├──────────────────────┼──────────────────────────────────┤
  │ ATR{7}                │ atr(H,L,C,7) / CLOSE  ← /CLOSE! │
  │ TSF{7}                │ tsf(C,7) / CLOSE       ← /CLOSE! │
  │ STDDEV{20}            │ stddev(C,20) / CLOSE    ← /CLOSE! │
  │ BBWIDTH{20}           │ bb_width(C,20)        ← raw      │
  │ VOL_RATIO{5,20}       │ vol_ratio(C,V,5,20)   ← raw      │
  │ RANGE_PCT             │ (HIGH-LOW)/CLOSE                  │
  └──────────────────────┴──────────────────────────────────┘

  ⚠️ v3 FeatureSpace 对 ATR/TSF/STDDEV 做了 /close 归一化,
     v4 函数返回原始值, 因此需显式加 /CLOSE 才能行为一致.

  ⚠️ v3 的 thr_mean(A) + thr_mean(B) 是先各自二值化再相加:
      (A>expanding_mean(A)) + (B>expanding_mean(B))
     不是 (A+B) > expanding_mean(A+B) !

用法:
  D:/Programs/mamba/envs/py313/python.exe demo/signal_backtest_demo.py
输出:
  tmp/EngineV4_demo_最优信号.html
"""
import sys, os, numpy as np, pandas as pd

sys.path.insert(0, r'D:\01-Doc\Quant\ft2')
sys.path.insert(0, r'D:\01-Doc\Quant\d2api')

from d2_api import d2_api
from signals.v4 import Expression, EngineCore


# ============================================================
# 0. 配置
# ============================================================
SYMBOL = '399317.SZ'          # 国证A指
DATA_START = '2019-07-01'     # 数据起点（含预热期）
START_DATE = '2020-01-01'     # 回测起点
END_DATE = '2025-12-31'
INITIAL_CAPITAL = 1_000_000


# ============================================================
# 1. 数据获取
# ============================================================
data = d2_api.kline.query(SYMBOL, DATA_START, END_DATE, interval='1d')
print(f"数据: {len(data)} 行, {data.columns.tolist()}")


# ============================================================
# 2. 最优公式定义 + 信号计算
# ============================================================
# [更新] 2026-06-21 从 v3 适配 v4，严格保持语义一致
#
# v3 表达式模式 → v4 翻译:
#   thr_mean(TERM)            → TERM > expanding_mean(TERM)
#   thr_med(TERM)             → TERM > expanding_median(TERM)
#   thr_roll_med(TERM, w)     → TERM > ts_median(TERM, w)
#   thr_mean(A) op thr_mean(B)→ (A>expanding_mean(A)) op (B>expanding_mean(B))
#                               注意: 先各自阈值化，再把二进制结果做 op
#   div 在二进制上等于 and      → 用 and (np.minimum)

TOP_FORMULAS = [
    # ── P4-量波比: thr_mean(VOL_RATIO{5,20}) div thr_roll_med((ATR{3} sub TSF{7}), 30) ──
    ("P4-量波比", "VOL_RATIO/(ATR-TSF) 滚动",
     "(vol_ratio(CLOSE,VOLUME,5,20) > expanding_mean(vol_ratio(CLOSE,VOLUME,5,20)))"
     " and (((atr(HIGH,LOW,CLOSE,3) - tsf(CLOSE,7)) / CLOSE) > ts_median((atr(HIGH,LOW,CLOSE,3) - tsf(CLOSE,7)) / CLOSE, 30))"),

    # ── T28-加法: thr_mean(ATR{7} - TSF{7}) + thr_mean(BBWIDTH{20} - TSF{7}) ──
    ("T28-加法", "ATR-TSF+BW-TSF",
     "(((atr(HIGH,LOW,CLOSE,7) - tsf(CLOSE,7)) / CLOSE) > expanding_mean(((atr(HIGH,LOW,CLOSE,7) - tsf(CLOSE,7)) / CLOSE)))"
     " + "
     "((bb_width(CLOSE,20) - (tsf(CLOSE,7) / CLOSE)) > expanding_mean(bb_width(CLOSE,20) - (tsf(CLOSE,7) / CLOSE)))"),

    # ── P3-量波比: thr_mean(VOL_RATIO{5,20}) div thr_med((ATR{7} sub TSF{7})) ──
    ("P3-量波比", "VOL_RATIO/(ATR-TSF)",
     "(vol_ratio(CLOSE,VOLUME,5,20) > expanding_mean(vol_ratio(CLOSE,VOLUME,5,20)))"
     " and (((atr(HIGH,LOW,CLOSE,7) - tsf(CLOSE,7)) / CLOSE) > expanding_median((atr(HIGH,LOW,CLOSE,7) - tsf(CLOSE,7)) / CLOSE))"),

    # ── P4-偏离: thr_mean(ATR{7}) - thr_mean(TSF{7}) ──
    ("P4-偏离", "ATR - TSF",
     "((atr(HIGH,LOW,CLOSE,7) / CLOSE) > expanding_mean(atr(HIGH,LOW,CLOSE,7) / CLOSE))"
     " - "
     "((tsf(CLOSE,7) / CLOSE) > expanding_mean(tsf(CLOSE,7) / CLOSE))"),

    # ── T28-三重: thr_mean(BBWIDTH{20}) + thr_mean(ATR{7}) + thr_mean(STDDEV{20}) ──
    ("T28-三重", "BBWIDTH+ATR+STDDEV",
     "(bb_width(CLOSE,20) > expanding_mean(bb_width(CLOSE,20)))"
     " + "
     "((atr(HIGH,LOW,CLOSE,7) / CLOSE) > expanding_mean(atr(HIGH,LOW,CLOSE,7) / CLOSE))"
     " + "
     "((stddev(CLOSE,20) / CLOSE) > expanding_mean(stddev(CLOSE,20) / CLOSE))"),

    # ── T28-终极: (thr_mean(ATR{10} - TSF{10}) + thr_mean(BBWIDTH{20} - TSF{7})) * thr_mean(RANGE_PCT) ──
    ("T28-终极", "(ATR-TSF+BW-TSF)×RANGE_PCT",
     "((((atr(HIGH,LOW,CLOSE,10) - tsf(CLOSE,10)) / CLOSE) > expanding_mean(((atr(HIGH,LOW,CLOSE,10) - tsf(CLOSE,10)) / CLOSE)))"
     " + "
     "((bb_width(CLOSE,20) - (tsf(CLOSE,7) / CLOSE)) > expanding_mean(bb_width(CLOSE,20) - (tsf(CLOSE,7) / CLOSE))))"
     " * "
     "(((HIGH - LOW) / CLOSE) > expanding_mean((HIGH - LOW) / CLOSE))"),
]

# 对每个公式: Expression → signal → EngineCore.backtest
results = []
for short_id, short_name, expr_str in TOP_FORMULAS:
    print(f"\n[{short_id}] {expr_str[:60]} ...", end=" ")
    try:
        signal = Expression(expr_str).generate(data)
        # [更新] 2026-06-21 v4: EngineCore.backtest 替代 EngineV3.backtest
        analyzer = EngineCore.backtest(
            signal, data, symbol=SYMBOL, mode='full',
            initial_capital=INITIAL_CAPITAL, start_date=START_DATE,
        )

        # [更新] 2026-06-21 使用 AccountAnalyzer 方法调用，替代 metrics() dict 遍历
        sharpe = analyzer.sharpe_ratio()
        cagr = analyzer.annualized_return()
        dd = analyzer.max_drawdown()
        dd_max = dd[0] if dd else 0.0  # (value, start, end)
        trades = len(analyzer.trade_profits)

        # 处理 None 返回值（无交易时某些指标为 None）
        if sharpe is None: sharpe = 0.0
        if cagr is None: cagr = 0.0
        sharpe = round(sharpe, 3)

        tag = "OK" if sharpe > 0.5 else ("~" if sharpe > 0 else "X")
        print(f"{tag} SR={sharpe:.3f} CAGR={cagr:.1%} DD={dd_max:.1%} 交易={trades}")

        # daily_assets 用于 header 图表
        daily = analyzer.daily_assets
        dates = sorted(daily.keys())
        nav_arr = np.array([daily[d] for d in dates])
        cummax = np.maximum.accumulate(nav_arr)
        dd_arr = (nav_arr / cummax - 1)

        results.append({
            'id': short_id, 'name': short_name, 'expr': expr_str,
            'analyzer': analyzer, 'sharpe': sharpe, 'cagr': cagr,
            'mdd': dd_max, 'trades': trades,
            'daily': daily, 'dates': dates, 'nav': nav_arr, 'dd': dd_arr,
        })
    except Exception as e:
        print(f"FAIL: {e}")

# 基准: 手动计算 (与策略 daily_assets 日期对齐)
closes = data.loc[data.index >= pd.Timestamp(START_DATE), 'close'].values
bench_ret = np.diff(closes) / closes[:-1]
bench_cagr = (closes[-1]/closes[0])**(252/len(closes)) - 1
bench_mdd = float(np.min(closes / np.maximum.accumulate(closes) - 1))
bench_sharpe = (bench_cagr-0.02)/(np.std(bench_ret,ddof=1)*np.sqrt(252))
print(f"\n基准: CAGR={bench_cagr:.1%} SR={bench_sharpe:.2f} MDD={bench_mdd:.1%}")


# ============================================================
# 3. 构建 Notebook（选 Top1 analyzer 做主报告，header/footer 扩展）
# ============================================================
results.sort(key=lambda r: r['sharpe'], reverse=True)
top = results[0]

# ── 3a. footer: 结论 ──
def add_footer(nb):
    with nb.section("结论与方法论"):
        nb.markdown(
            "### EngineCore (signals.v4) 特点\n"
            "- **统一入口**: `EngineCore.backtest()` 替代 v3 的 EngineV3\n"
            "- **双模式**: `full` / `fast` 统一返回 AccountAnalyzer\n"
            "- **Python AST**: 表达式使用原生 Python 语法\n"
            "\n"
            "### v3 → v4 适配要点\n"
            "- **阈值时机**: v3 先对各项二值化再组合 (`thr_mean(A)+thr_mean(B)`),\n"
            "  v4 等价: `(A>expanding_mean(A)) + (B>expanding_mean(B))`\n"
            "- **归一化差异**: v3 FeatureSpace 对 ATR/TSF/STDDEV 做了 `/close`,\n"
            "  v4 需显式加 `/CLOSE`\n"
            "- **div→and**: v3 二进制 `div` 等效 v4 `and` (均用 np.minimum)\n"
            "\n"
            "### 信号形态比引擎更重要\n"
            "- 从 v15（简化引擎 SR=1.0~1.95）到 v18（core引擎 含费率），下降 20~40%\n"
            "- **波动率类信号最稳健**: ATR/BBWIDTH/STDDEV 的 TSF 差分组合衰减仅 15~25%\n"
            "\n"
            "### 最优公式\n"
            f"- 第1名 **{top['id']}** Sharpe={top['sharpe']:.3f}\n"
        )

# ── 3b. header: 全量对比表 + KPI ──
def add_header(nb):
    # KPI 卡片
    valid_n = sum(1 for r in results if r['sharpe'] > 0.5)
    nb.metrics([
        {'name': '测试公式', 'value': str(len(results))},
        {'name': f'有效(SR>0.5)', 'value': str(valid_n), 'color': 'green'},
        {'name': 'Top1 SR', 'value': f"{results[0]['sharpe']:.3f}", 'color': 'green'},
        {'name': '基准 SR', 'value': f'{bench_sharpe:.2f}'},
        {'name': '基准 CAGR', 'value': f'{bench_cagr:.1%}'},
        {'name': f'标的 {SYMBOL}', 'value': f'{START_DATE[:4]}~{END_DATE[:4]}'},
    ], title="概览", columns=3)

    # 对比表格
    comparison = []
    for i, r in enumerate(results):
        comparison.append({
            '排名': i+1, 'ID': r['id'], '公式': r['name'],
            '表达式': r['expr'][:45],
            'Sharpe': r['sharpe'], '年化': f"{r['cagr']:.1%}",
            '最大回撤': f"{r['mdd']:.1%}", '交易': r['trades'],
        })
    comparison.append({
        '排名': '—', 'ID': SYMBOL, '公式': '基准',
        '表达式': '指数持有',
        'Sharpe': round(bench_sharpe, 3), '年化': f"{bench_cagr:.1%}",
        '最大回撤': f"{bench_mdd:.1%}", '交易': 1,
    })

    nb.table(comparison,
        columns=['排名','ID','公式','表达式','Sharpe','年化','最大回撤','交易'],
        freeze={'left': 2},
        heatmap={'start': 4, 'end': -2, 'axis': 'column'},
        title='信号对比')

    # header 内追加净值叠加图（选 Top3）
    dates_all = results[0]['dates']
    series_list = []
    for i in range(min(3, len(results))):
        r = results[i]
        nav_norm = r['nav'] / INITIAL_CAPITAL
        series_list.append({'name': f"#{i+1} {r['id']}", 'data': [round(v,4) for v in nav_norm]})

    # 基准净值（START_DATE 截断，与策略 daily_assets 日期对齐）
    bench_data = data.loc[data.index >= pd.Timestamp(START_DATE)]
    bench_norm = closes / closes[0]

    if len(bench_norm) > len(dates_all):
        bench_norm = bench_norm[:len(dates_all)]
    series_list.append({'name': '基准', 'data': [round(v,4) for v in bench_norm]})

    nb.chart('line', {
        'xAxis': [d.strftime('%Y-%m-%d') if hasattr(d,'strftime') else str(d) for d in dates_all],
        'series': series_list,
    }, title='Top3 净值对比（归一化至1.0）', height='350px', series_opts={'is_smooth': True})

    # 回撤对比
    dd_series = []
    for i in range(min(3, len(results))):
        r = results[i]
        dd_pct = [round(d*100, 2) for d in r['dd']]
        dd_series.append({'name': f"#{i+1} {r['id']}", 'data': dd_pct})
    nb.chart('line', {
        'xAxis': [d.strftime('%Y-%m-%d') if hasattr(d,'strftime') else str(d) for d in dates_all],
        'series': dd_series,
    }, title='Top3 回撤对比 (%)', height='200px')


# ── 主报告: analyzer.to_notebook(title, header=, footer=) ──
output_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'tmp')
os.makedirs(output_dir, exist_ok=True)

top['analyzer'].base_dir = output_dir
# set_benchmark 用 START_DATE 截断的净值 dict
bench_data = data.loc[data.index >= pd.Timestamp(START_DATE)]
bench_nav_dict = {
    d.date() if hasattr(d, 'date') else d: float(closes[i] / closes[0] * INITIAL_CAPITAL)
    for i, d in enumerate(bench_data.index)
}
top['analyzer'].set_benchmark(bench_nav_dict, SYMBOL)
top['analyzer'].to_notebook(
    f"EngineV4 demo — v18 最优择时",
    header=add_header,
    footer=add_footer)

print(f"\n>> HTML: {output_dir}/EngineV4 demo — v18 最优择时.html")

# 控制台
print(f"\n{'排名':<4} {'ID':<12} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'交易':>5}")
print("-"*48)
for i, r in enumerate(results):
    print(f"  {i+1:<3} {r['id']:<12} {r['sharpe']:>7.3f} {r['cagr']:>7.1%} {r['mdd']:>7.1%} {r['trades']:>5}")
