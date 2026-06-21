"""
demo/signal_backtest_demo.py — signals.v3 择时回测范式 (EngineV3)
====================================================================
[更新] 2026-06-10 从 signals.v2 迁移到 signals.v3 (EngineV3)
[重命名] 2026-06-10 core_backtest_notebook → signal_backtest_demo（避免误解为裸 core 引擎）

信号探测/择时回测的标准范式：
  Expression → EngineV3.backtest() → AccountAnalyzer.to_notebook()

展示 header/footer 回调能力：
  - header: 对比表 / KPI卡片 / 自定义图表
  - footer: 结论 / 方法论说明

用法:
  D:/Programs/mamba/envs/py313/python.exe demo/signal_backtest_demo.py
输出:
  tmp/EngineV3_demo_最优信号.html
"""
import sys, os, numpy as np, pandas as pd

ROOT = r'D:\01-Doc\AI_zeshi'
sys.path.insert(0, r'D:\01-Doc\Quant\ft2')
sys.path.insert(0, r'D:\01-Doc\Quant\d2api')
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'v2'))

from d2_api import d2_api
from signals.v3 import EngineV3, FeatureSpace, Expression
from shared.config import SYMBOL, DATA_START, START_DATE, END_DATE, INITIAL_CAPITAL


# ============================================================
# 1. 数据 + 特征
# ============================================================
data = d2_api.kline.query(SYMBOL, DATA_START, END_DATE, interval='1d')
print(f"数据: {len(data)} 行, {data.columns.tolist()}")

# 构建 FeatureSpace + 手动追加自定义特征
fs = FeatureSpace().fit(data)
feat_df = fs.transform(data)

close = data['close'].values
high = data['high'].values
low = data['low'].values

def _close_ma_ratio(arr, period):
    r = np.full(len(arr), np.nan)
    for i in range(period, len(arr)):
        ma = np.mean(arr[i-period:i])
        if ma > 0: r[i] = arr[i]/ma - 1
    return r

def _range_pct(h, l, c): return (h-l)/c

idx = feat_df.index
c = pd.Series(close, index=data.index).reindex(idx).values
h = pd.Series(high, index=data.index).reindex(idx).values
l = pd.Series(low, index=data.index).reindex(idx).values
feat_df['RANGE_PCT'] = _range_pct(h, l, c)
feat_df['CLOSE_MA_RATIO{10}'] = _close_ma_ratio(c, 10)
fs._feature_names = list(feat_df.columns)


# ============================================================
# 2. 最优公式定义 + 信号计算
# ============================================================
TOP_FORMULAS = [
    ("P4-量波比", "VOL_RATIO/(ATR-TSF) 滚动", "thr_mean(VOL_RATIO{5,20}) div thr_roll_med((ATR{3} sub TSF{7}), 30)"),
    ("T28-加法", "ATR-TSF+BW-TSF", "thr_mean(ATR{7} - TSF{7}) + thr_mean(BBWIDTH{20} - TSF{7})"),
    ("P3-量波比", "VOL_RATIO/(ATR-TSF)", "thr_mean(VOL_RATIO{5,20}) div thr_med((ATR{7} sub TSF{7}))"),
    ("P4-偏离", "ATR - TSF", "thr_mean(ATR{7}) - thr_mean(TSF{7})"),
    ("T28-三重", "BBWIDTH+ATR+STDDEV", "thr_mean(BBWIDTH{20}) + thr_mean(ATR{7}) + thr_mean(STDDEV{20})"),
    ("T28-终极", "(ATR-TSF+BW-TSF)×RANGE_PCT", "(thr_mean(ATR{10} - TSF{10}) + thr_mean(BBWIDTH{20} - TSF{7})) * thr_mean(RANGE_PCT)"),
]

# 对每个公式: Expression → signal → EngineV3.backtest
results = []
for short_id, short_name, expr_str in TOP_FORMULAS:
    print(f"\n[{short_id}] {expr_str[:60]} ...", end=" ")
    try:
        signal = Expression(expr_str, feature_df=feat_df).generate(data)
        # [更新] 2026-06-10 v3: EngineV3.backtest 替代 core_backtest
        # mode='full' → AccountAnalyzer, bench_label 自动跑基准
        analyzer = EngineV3.backtest(
            signal, data, symbol=SYMBOL, mode='full',
            initial_capital=INITIAL_CAPITAL, start_date=START_DATE,
        )
        m = analyzer.metrics()

        def _v(name, default=0):
            for k, v in m.items():
                if isinstance(v, dict) and v.get('name') == name:
                    val = v['value']
                    return val[0] if isinstance(val, tuple) else val
            return default

        # 从 daily_assets 计算回撤序列
        daily = analyzer.daily_assets
        dates = sorted(daily.keys())
        nav_arr = np.array([daily[d] for d in dates])
        cummax = np.maximum.accumulate(nav_arr)
        dd_arr = (nav_arr / cummax - 1)
        dd_max = float(np.min(dd_arr))

        sharpe = round(float(_v('夏普比率')), 3)
        cagr = float(_v('年化收益率'))
        trades = len(analyzer.account.trade_records) // 2

        tag = "OK" if sharpe > 0.5 else ("~" if sharpe > 0 else "X")
        print(f"{tag} SR={sharpe:.3f} CAGR={cagr:.1%} DD={dd_max:.1%} 交易={trades}")

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
            "### EngineV3 (signals.v3) 特点\n"
            "- **统一入口**: `EngineV3.backtest()` 替代 v2 的多个回测函数\n"
            "- **双模式**: `full` → AccountAnalyzer / `fast` → FastResult (GP搜索用)\n"
            "- **自动基准**: `bench_label='399317.SZ'` 参数自动跑买入持有基准\n"
            "- **费率控制**: `fee_config` 参数配置，None=零费率\n"
            "\n"
            "### 信号形态比引擎更重要\n"
            "- 从 v15（简化引擎 SR=1.0~1.95）到 v18（core引擎 含费率），下降 20~40%\n"
            "- 但 **close_ma_ratio** 从 1.95 → 0.16 的主因是 thr_mean 二值化，不是引擎差异\n"
            "- **波动率类信号最稳健**: ATR/BBWIDTH/STDDEV 的 TSF 差分组合衰减仅 15~25%\n"
            "\n"
            "### 最优公式\n"
            f"- 第1名 **{top['id']}** `{top['expr'][:50]}...` Sharpe={top['sharpe']:.3f}\n"
            "- 推荐: 滚动中位数版 VOL_RATIO/(ATR-TSF)，回撤最可控（~10%）\n"
            "- 搭配: ATR-TSF + BBWIDTH-TSF 加法组合，CAGR 可达 18%\n"
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
# [更新] 2026-06-10 set_benchmark 用 START_DATE 截断的净值 dict
bench_data = data.loc[data.index >= pd.Timestamp(START_DATE)]
bench_nav_dict = {
    d.date() if hasattr(d, 'date') else d: float(closes[i] / closes[0] * INITIAL_CAPITAL)
    for i, d in enumerate(bench_data.index)
}
top['analyzer'].set_benchmark(bench_nav_dict, SYMBOL)
top['analyzer'].to_notebook(
    f"EngineV3 demo — v18 最优择时",
    header=add_header,
    footer=add_footer)

print(f"\n>> HTML: {output_dir}/EngineV3 demo — v18 最优择时.html")

# 控制台
print(f"\n{'排名':<4} {'ID':<12} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'交易':>5}")
print("-"*48)
for i, r in enumerate(results):
    print(f"  {i+1:<3} {r['id']:<12} {r['sharpe']:>7.3f} {r['cagr']:>7.1%} {r['mdd']:>7.1%} {r['trades']:>5}")
