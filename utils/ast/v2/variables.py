# -*- coding: utf-8 -*-
"""
utils/ast/variables.py — 变量层 (公共基础设施)
=============================================================================

在五层架构中的位置: 第3层(变量) — 定义"能引用什么"

  变量 = 数据名词 (VALID_VAR_PREFIXES): 70+ 前缀

═══════════════════════════════════════════════════════════════
命名规范 (对齐 WQ101 行业标准)

  ◆ 变量命名 (ALL_CAPS)
    分类:
      原始 OHLCV:   OPEN, HIGH, LOW, CLOSE, VOLUME, AMOUNT, VWAP, RETURNS, RET
      talib 指标:   ATR, RSI, MACD, CCI, ADX, EMA, HV, NATR ... (函数+变量双通道)
      行业广度:     SECTOR_UP, BREADTH_L, DISP, ROTSPD, IND_CORR ... (仅变量)
      因子模块:     BENCH, REL, SHARE, DOWNSIDE_VOL
      LLM/用户自定义: register_variable() 运行时热注册

    变量 vs 函数同名 (设计如此, 互补而非冲突):
      函数: rsi(CLOSE, 14)     → 实时算, 参数灵活
      变量: ts_rank(RSI_14, 60) → 预计算, 性能优先

═══════════════════════════════════════════════════════════════
变量用途 (为什么需要变量层):
  1. 性能缓存 — talib 指标预计算注入, 避免每次表达式重复算
  2. 跨品种聚合 — 行业广度/市场级数据无法从单个 OHLCV 算出
  3. 变量不会替代函数 — 变量用于数据, 函数用于操作, 两者互补

[重构] 2026-06-22 从 registry.py 拆分, 独立为 variables.py
=============================================================================
"""

# ============================================================
# 合法变量前缀 — 5组, 按实际使用频率分类
# ============================================================

VALID_VAR_PREFIXES = [
    # ═══════════════════════════════════════════════════════════
    # 第1组: 活跃 — 因子表达式高频使用 (由 data_loader 注入)
    # ═══════════════════════════════════════════════════════════
    'OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT',     # 原始 OHLCV
    'VWAP',                                                  # 均价 (WQ065)
    'RETURNS',                                               # close[t]/close[t-1]-1

    # ═══════════════════════════════════════════════════════════
    # 第2组: 活跃 — 派生变量前缀 (匹配 REL_CLOSE, BENCH_RETURNS 等后缀)
    #
    # 注入方式: data_loader.ensure_data() 计算后放入 panel_dict
    # 大小写:   小写键 (rel_close) → normalize_data_keys() → 大写 (REL_CLOSE)
    # 前缀匹配: is_valid_variable('REL_CLOSE') → startswith('REL') → True
    #
    # 计算来源 (data_loader.py):
    #   REL_CLOSE   = close / bench_close      (相对基准价)
    #   REL_AMOUNT  = amount / bench_amount    (相对基准额)
    #   REL_VOLUME  = volume / bench_volume    (相对基准量)
    #   BENCH_CLOSE = 基准指数 close (铺平到 N 列)
    #   BENCH_RETURNS = bench_close 日收益率
    #   SHARE       = amount / amount.sum(axis=1) (市场份额)
    #   DOWNSIDE_VOL = 下行波动率 (仅负收益的 std)
    # ═══════════════════════════════════════════════════════════
    'REL',          # 前缀: REL_CLOSE / REL_AMOUNT / REL_VOLUME
    'BENCH',        # 前缀: BENCH_CLOSE / BENCH_RETURNS
    'SHARE',        # 完整: SHARE = amount / Σamount (跨品种总成交额占比)
    'DOWNSIDE_VOL', # 完整: DOWNSIDE_VOL = std(returns[returns<0]) (下行风险)

    # ═══════════════════════════════════════════════════════════
    # 第3组: 活跃 — 基本面变量 (从 parquet 注入)
    # ═══════════════════════════════════════════════════════════
    'PE_TTM_INDEX',     # 滚动市盈率
    'PB_MRQ',           # 市净率
    'TURNOVERRATIO',    # 换手率
    'TOTALCAPITAL',     # 总市值

    # ═══════════════════════════════════════════════════════════
    # 第4组: 预留 — talib 指标 (函数通道已覆盖, 变量通道为性能预留)
    # 用法: rsi(CLOSE,14) 等价于 RSI_14 (预计算注入后)
    # 原则: 只保留有明确经济学意义的标准指标, 可由 ts_* 推导的二级量不列
    # 现状: 以函数通道为主, 变量通道未启用
    # ═══════════════════════════════════════════════════════════
    # 波动率: ATR(真实波幅), STDDEV(滚动标准差), HV(年化波动率), NATR(归一化ATR)
    'ATR', 'STDDEV', 'HV', 'NATR',
    # 通道: BBWIDTH(布林带宽度, 波动率/价格水平)
    'BBWIDTH',
    # 趋势: 9种移动平均 + ADX(趋向指数)
    'TRIMA', 'SMA', 'MA', 'EMA', 'TSF', 'WMA', 'DEMA', 'KAMA', 'ADX',
    # 动量: RSI(超买超卖), CCI(偏离均值), MACD(趋势动量), MFI(资金流), ULTOSC(多周期), ROC(变化率)
    'RSI', 'CCI', 'MACD', 'MFI', 'ULTOSC', 'ROC',
    # 统计: LINEARREG(线性回归), VAR(方差), CORREL(相关系数)
    'LINEARREG', 'VAR', 'CORREL',
    # 量价: VOL_RATIO(量比), VOL_CHG(量变), OBV(能量潮), UP_RATIO(上涨比例)
    'VOL_RATIO', 'VOL_CHG', 'OBV', 'UP_RATIO',
    # 价格水平: AVGPRICE(均价), WCLPRICE(加权收盘)
    'AVGPRICE', 'WCLPRICE',

    # ═══════════════════════════════════════════════════════════
    # 第5组: 预留 — 行业广度 / 市场级变量 (择时信号用)
    # 来源: 需跨品种聚合 (无法从单列 OHLCV 算出)
    # 原则: 只保留一级原始量, Z-score/动量等二级推导由表达式 ts_* 完成
    # 现状: 语法白名单已预留, 数据计算管线待实现
    # ═══════════════════════════════════════════════════════════
    # ── 价格宽度: 多少品种在涨? (市场参与度) ──
    'SECTOR_UP',           # 当日上涨行业比例 [0,1]
    'SECTOR_MOM',          # 动量>0的行业比例
    'SECTOR_AD',           # 行业涨跌比 = (涨-跌)/(涨+跌)
    # ── 均线宽度: 均线上方行业密度 (趋势广度) ──
    'BREADTH_S',           # 短期均线宽度 (短)
    'BREADTH_M',           # 中期均线宽度 (中)
    'BREADTH_L',           # 长期均线宽度 (长, 经典背离信号源)
    'BREADTH_AMT',         # 成交额均线宽度
    # 离散度/轮动 — 行业有多一致? (市场结构)
    'DISP',                # 行业日收益截面标准差 (高=分化, 低=同涨同跌)
    'ROTSPD',              # 行业轮动速度 (领涨行业切换频率)
    'NHL',                 # 新高-新低行业净差
    'SKEW',                # 行业收益截面偏度 (>0=少数领涨, <0=少数暴跌)
    # 成交量结构 — 资金在往哪流? (资金集中度)
    'VMED',                # 行业中位数成交量
    'VDISP',               # 行业成交量截面离散度 (高=资金集中, 低=均匀)
    'VSKEW',               # 行业成交量截面偏度 (>0=少数行业吸金)
    # 相关性 — 市场分散还是同步? (系统性风险)
    'IND_CORR',            # 行业平均两两相关系数 (>0.7=系统性, <0.3=分散)
    # 尾部风险 — 极端行情密度 (黑天鹅预警)
    'TAILUP',              # 右尾强度 (行业极端上涨密度)
    'TAILDOWN',            # 左尾强度 (行业极端下跌密度)
    'TAILNET',             # 尾部净强度 = TAILUP - TAILDOWN
]


# ============================================================
# 变量分类索引 (供 LLM 理解可用数据)
# ============================================================

VAR_CATEGORIES = {
    '原始OHLCV': ['OPEN', 'HIGH', 'LOW', 'CLOSE', 'VOLUME', 'AMOUNT', 'VWAP', 'RETURNS'],
    '相对基准': ['REL', 'BENCH'],                     # 前缀: REL_CLOSE / BENCH_RETURNS 等
    '市场份额': ['SHARE'],
    '下行风险': ['DOWNSIDE_VOL'],
    '基本面': ['PE_TTM_INDEX', 'PB_MRQ', 'TURNOVERRATIO', 'TOTALCAPITAL'],
    '波动率': ['ATR', 'STDDEV', 'HV', 'NATR'],
    '通道指标': ['BBWIDTH'],
    '趋势指标': ['TRIMA', 'SMA', 'MA', 'EMA', 'TSF', 'WMA', 'DEMA', 'KAMA', 'ADX'],
    '动量指标': ['RSI', 'CCI', 'MACD', 'MFI', 'ULTOSC', 'ROC'],
    '统计指标': ['LINEARREG', 'VAR', 'CORREL'],
    '量价指标': ['VOL_RATIO', 'VOL_CHG', 'OBV', 'UP_RATIO'],
    '价格水平': ['AVGPRICE', 'WCLPRICE'],
    '市场宽度': ['SECTOR_UP', 'SECTOR_MOM', 'SECTOR_AD',
                'BREADTH_S', 'BREADTH_M', 'BREADTH_L', 'BREADTH_AMT'],
    '市场结构': ['DISP', 'ROTSPD', 'NHL', 'SKEW', 'IND_CORR'],
    '资金结构': ['VMED', 'VDISP', 'VSKEW'],
    '尾部风险': ['TAILUP', 'TAILDOWN', 'TAILNET'],
}


def get_var_category(prefix: str) -> str:
    """查询变量前缀所属分类"""
    upper = prefix.upper()
    for cat, prefixes in VAR_CATEGORIES.items():
        for pfx in prefixes:
            if upper == pfx or upper.startswith(pfx + '_'):
                return cat
    return '自定义'


# ============================================================
# 变量名校验
# ============================================================

def is_valid_variable(name: str) -> bool:
    """检查变量名是否合法（匹配已注册前缀）

    规则:
      1. 精确匹配任一前缀 (如 CLOSE, SECTOR_UP)
      2. 前缀 + '_' + 后缀 (如 RSI_14, BREADTH_S)
         后缀只能含 ASCII 字母数字和下划线
    """
    upper = name.upper()
    if upper in VALID_VAR_PREFIXES:
        return True
    for pfx in VALID_VAR_PREFIXES:
        if upper.startswith(pfx + '_'):
            rest = upper[len(pfx) + 1:]
            if rest and all(c.isascii() and (c.isalnum() or c == '_') for c in rest):
                return True
    return False


# ============================================================
# 临时自定义注册 (外部模板热添加，无需修改 variables.py)
# ============================================================
#
# 使用场景:
#   每个探索脚本是独立进程, 跑完即销毁。脚本顶部注册 → 全文使用 →
#   进程退出自动清零, 无需手动清理。不存在跨脚本污染问题。
#
# 用法:
#   from utils.ast import register_variable
#   register_variable('MY_VAR')
#   expr = Expression("MY_VAR > 0")
#
# 兼容旧路径:
#   from signals.v4 import register_variable  # 仍可用，重导出链

def register_variable(prefix: str) -> None:
    """临时注册自定义变量前缀到表达式引擎。

    进程级全局注册, 当前脚本内所有 Expression 生效。
    脚本退出自动销毁, 无泄漏风险。

    Args:
        prefix: 变量名前缀 (大小写不敏感)，支持 'prefix' 或 'prefix_suffix' 匹配
    """
    upper = prefix.upper()
    if upper not in VALID_VAR_PREFIXES:
        VALID_VAR_PREFIXES.append(upper)


def unregister_variable(prefix: str) -> bool:
    """注销自定义变量前缀，返回是否成功。"""
    upper = prefix.upper()
    if upper in VALID_VAR_PREFIXES:
        VALID_VAR_PREFIXES.remove(upper)
        return True
    return False
