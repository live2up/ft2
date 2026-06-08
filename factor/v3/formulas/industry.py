"""
factor/v3/formulas/industry.py — 行业轮动专有因子

从 yinzi-3 项目探索中发现的行业轮动有效因子。
行业轮动(31行业)与个股选股(3000+股票)的信号结构完全不同:
  - 个股: 高IC因子主导, 量价因子有效
  - 行业: IC极低(SE≈0.189), 依赖非线性变形和条件切换

[新增] 2026-06-07 基于yinzi-3 Phase1~3全部探索成果
"""
from typing import Dict
from ..base import FactorCategory

# ═══════════════════════════════════════════════════════════════════════════
#
# 行业轮动专有因子
#
# 核心发现(2026-06-07):
#   1. 超跌反弹(A1)是唯一穿越牛熊的因子(熊市Sharpe=0.548)
#   2. 趋势加速(M6)牛市极强(1.741)但熊市为负(-0.096)
#   3. 条件切换(SW)全样本最优(1.230)但熊市弱(0.205)
#   4. 非线性变形 >> 线性变形: argmax/argmin >> ts_rank >> ret
#   5. 条件切换(ifelse) >> 条件增强(add+mul) >> 单因子
#   6. 月度和周度是完全不同的世界, 需独立探索
#
# ═══════════════════════════════════════════════════════════════════════════

INDUSTRY_ALPHA: Dict[str, str] = {
    # ── 穿越牛熊型 (全环境正收益) ──────────────────────────────
    #
    # A1: 超跌反弹 — yinzi-3单因子天花板, 穿越牛熊之王
    #   全样本=0.984  牛市=1.562  熊市=0.548  震荡=0.670  std=0.452
    #   逻辑: 30日内跌幅最大的行业反弹最多(截面排名)
    #   变形: ts_argmin(超跌位置) >> ts_rank(排名) >> delta(线性)
    #
    'ind_revert_30d': 'cs_rank(ts_argmin(delta(close,1),30))',

    # ── 趋势型 (牛市极强, 熊市失效) ────────────────────────────
    #
    # M6: 趋势加速 — 牛市之王
    #   全样本=0.917  牛市=1.741  熊市=-0.096  震荡=0.613  std=0.757
    #   逻辑: 60日内收盘价最高的行业趋势最强
    #
    'ind_trend_60d': 'cs_rank(ts_argmax(close,60))',

    # ── 资金面型 (牛市爆发, 熊市最差) ──────────────────────────
    #
    # RA: 资金面排名
    #   全样本=0.920  牛市=2.244  熊市=-0.901  震荡=1.203  std=1.308
    #   逻辑: 相对成交额时序排名, 资金涌入的行业更强
    #   注意: 环境标准差最大, 极端两极分化
    #
    'ind_relamt_rank_10d': 'cs_rank(ts_rank(rel_amount,10))',

    # ── 稳定型 (牛市最强但熊市负) ──────────────────────────────
    #
    # SK: 排名稳定性 — 排名波动小的行业有趋势延续性
    #   全样本=0.894  牛市=2.077  熊市=-0.587  震荡=0.443  std=1.097
    #
    'ind_rank_stable_5d': 'neg(ts_std(cs_rank(close),5))',

    # ── 跳空型 (高波动环境有效) ────────────────────────────────
    #
    # GA: 标准化跳空 — 隔夜跳空反映市场对行业的定价调整
    #   全样本=0.696  牛市=1.828  熊市=-0.687  震荡=0.826  std=1.034
    #
    'ind_gap_range': 'cs_rank(div(sub(open,delay(close,1)),add(sub(high,low),0.01)))',

    # ── 条件切换型 (框架级因子, 全样本最优) ──────────────────────
    #
    # SW: 大盘状态切换 — 牛市用超跌反弹, 非牛市用趋势加速
    #   全样本=1.230  牛市=1.951  熊市=0.205  震荡=1.136  std=0.713
    #   逻辑: 基准指数60日排名>0.7 → A1(超跌), 否则 → M6(趋势)
    #   网格搜索最优: W=60, T=0.7 (864组合102秒)
    #   注意: 熊市弱, 需与A1配合使用
    #
    'ind_switch_bull60': 'ifelse(gt(ts_rank(bench_close,60),0.7),cs_rank(ts_argmin(delta(close,1),30)),cs_rank(ts_argmax(close,60)))',

    # ── 条件增强型 (因子+状态倍增) ──────────────────────────────
    #
    # EN: 价量齐升时超跌反弹加倍
    #   全样本=0.927  牛市=1.760  熊市=0.209  震荡=0.348  std=0.700
    #
    'ind_revert_enhanced': 'add(cs_rank(ts_argmin(delta(close,1),30)),mul(and(gt(delta(close,5),0),gt(delta(amount,5),0)),cs_rank(ts_argmin(delta(close,1),30))))',

    # ── 大盘交互型 (与基准的相关性变化) ────────────────────────
    #
    # CB: 行业与大盘相关性变化 — 独立性增强的行业有alpha
    #   全样本=0.449  牛市=1.246  熊市=-0.216  震荡=-0.056  std=0.655
    #
    'ind_corr_bench_delta': 'cs_rank(delta(correlation(close,bench_close,20),20))',

    # ── 超跌量确认型 (价格跌+量增=超跌确认) ────────────────────
    #
    # RC: 排名超跌 + 量确认
    #   全样本=0.848  逻辑: 价跌量增=恐慌抛售, 反弹概率大
    #
    'ind_revert_vol_confirm': 'cs_rank(mul(neg(ts_zscore(ts_rank(close,10),10)),ts_zscore(amount,20)))',

    # ── 多层卷积型 (深层次超跌信号) ─────────────────────────────
    #
    # argmin_delta1_30d: 原始探索最优单因子, 与A1等价但展示GP可发现性
    #   全样本=0.984
    #
    'ind_argmin_delta1_30d': 'cs_rank(ts_argmin(delta(close,1),30))',

    # ── 跨通道卷积型 (价量双确认) ──────────────────────────────
    #
    # argmax_cross_60d: 价量双确认的趋势信号
    #   全样本=0.889  双频稳健
    #
    'ind_argmax_cross_60d': 'cs_rank(add(ts_argmax(close,60),ts_argmax(amount,60)))',

    # ── 份额变化回归残差型 (独立于趋势的资金行为) ──────────────
    #
    # share_residual: 份额变化去除趋势后的残差
    #   全样本=0.826  双频稳健
    #
    'ind_share_residual_20d': 'cs_rank(ts_regression_residual(share,20))',

    # ── 成交额加速度型 (资金加速流入) ──────────────────────────
    #
    # amt_accel: 成交额10日均线的10日变化
    #   全样本=0.874
    #
    'ind_amt_accel_20d': 'cs_rank(delta(ts_mean(amount,10),10))',

    # ── 偏度型 (收益分布不对称性) ──────────────────────────────
    #
    # skew_relclose: 相对收盘价偏度 — 正偏=有上涨尾部
    #   全样本=0.734
    #
    'ind_skew_relclose_60d': 'cs_rank(ts_skew(rel_close,60))',

    # ── 双频组合型 (周度主导+月度辅助) ────────────────────────
    #
    # 2026-06-07突破: 周度条件切换(SW)权重90% + 月度超跌反弹(A1)权重10%
    #   逻辑: SW是快响应因子(每周重新评估), A1是穿越保护(熊市正收益)
    #   全样本=1.281(Top3) / 1.451(Top2) / 1.479(Top2+0.1*M6)
    #   熊市=0.380(Top3) vs 纯SW熊市=0.205
    #
    'ind_dual_sw_a1': 'add(mul(0.9,ifelse(gt(ts_rank(bench_close,60),0.7),cs_rank(ts_argmin(delta(close,1),30)),cs_rank(ts_argmax(close,60)))),mul(0.1,cs_rank(ts_argmin(delta(close,1),30))))',

    # ── 波动率条件切换型 (volhi20条件) ──────────────────────
    #
    # volhi20>0.8时用RA(资金面), 否则用M6(趋势加速)
    #   全样本=1.251(Top3) — 比bull60条件(1.230)更优
    #   逻辑: 高波动环境用资金面因子更稳, 低波动用趋势因子
    #
    'ind_switch_volhi_ra_m6': 'ifelse(gt(ts_rank(ts_std(close,20),60),0.8),cs_rank(ts_rank(rel_amount,10)),cs_rank(ts_argmax(close,60)))',
}

INDUSTRY_ALPHA_CATEGORIES: Dict[str, FactorCategory] = {
    'ind_revert_30d':          FactorCategory.REVERSAL,
    'ind_trend_60d':           FactorCategory.MOMENTUM,
    'ind_relamt_rank_10d':     FactorCategory.VOLUME,
    'ind_rank_stable_5d':      FactorCategory.VOLATILITY,
    'ind_gap_range':           FactorCategory.PRICE,
    'ind_switch_bull60':       FactorCategory.TECHNICAL,    # 条件切换 = 技术框架
    'ind_revert_enhanced':     FactorCategory.TECHNICAL,    # 条件增强 = 技术框架
    'ind_corr_bench_delta':    FactorCategory.TECHNICAL,
    'ind_revert_vol_confirm':  FactorCategory.REVERSAL,
    'ind_argmin_delta1_30d':   FactorCategory.REVERSAL,     # 等价A1
    'ind_argmax_cross_60d':    FactorCategory.MOMENTUM,
    'ind_share_residual_20d':  FactorCategory.VOLUME,
    'ind_amt_accel_20d':       FactorCategory.VOLUME,
    'ind_skew_relclose_60d':   FactorCategory.VOLATILITY,
    'ind_dual_sw_a1':          FactorCategory.TECHNICAL,    # 双频组合
    'ind_switch_volhi_ra_m6':  FactorCategory.TECHNICAL,    # 波动率条件切换
}
