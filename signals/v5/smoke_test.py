"""signals/v5 冒烟测试 — 验证导入、表达式解析、状态机、回测"""
import sys
import os

# 确保能导入 signals.v5
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, r'D:\01-Doc\Quant\ft2')
sys.path.insert(0, r'D:\01-Doc\Quant\d2api')

import numpy as np
import pandas as pd

print("=" * 60)
print("[v5] 冒烟测试")
print("=" * 60)

# 1. 导入测试
print("\n[1] 导入测试")
from signals.v5 import Expression, SigEngine, stateful_signal
print("    ✅ from signals.v5 import Expression, SigEngine, stateful_signal")

from utils.ast.v2 import register_function, FUNC_REGISTRY
print("    ✅ from utils.ast.v2 import register_function, FUNC_REGISTRY")

# 2. 表达式解析测试
print("\n[2] 表达式解析测试")
expr = Expression("ts_roc(CLOSE, 5) > 0 and ts_zscore(amt_ratio(AMOUNT,5,20), 60) > 0")
print(f"    expr.variables = {expr.variables}")
print(f"    expr.functions = {expr.functions}")
print(f"    expr.complexity = {expr.complexity}")
assert 'CLOSE' in expr.variables
assert 'ts_roc' in expr.functions
print("    ✅ 解析+自省正常")

# 3. 自定义原语注册测试
print("\n[3] 自定义原语注册测试")
register_function('v5_test', lambda x: x * 2, category='math_function')
expr2 = Expression("v5_test(CLOSE) > 0")
assert 'v5_test' in expr2.functions
print("    ✅ register_function 正常")

# 4. 单品种求值测试
print("\n[4] 单品种求值测试")
np.random.seed(42)
n = 100
dates = pd.date_range('2024-01-01', periods=n, freq='B')
data = pd.DataFrame({
    'open': np.random.rand(n) * 100 + 100,
    'high': np.random.rand(n) * 100 + 105,
    'low': np.random.rand(n) * 100 + 95,
    'close': np.random.rand(n) * 100 + 100,
    'volume': np.random.randint(1000, 10000, n).astype(float),
    'amount': np.random.rand(n) * 1e6 + 5e5,
}, index=dates)

sig = Expression("ts_roc(CLOSE, 5) > 0").generate(data)
assert isinstance(sig, pd.Series)
assert len(sig) == n
print(f"    signal[:5] = {sig.head().tolist()}")
print("    ✅ generate() 正常")

# 5. 状态机测试
print("\n[5] 状态机测试")
stf = stateful_signal(data, "ts_roc(CLOSE, 5) > 0", "ts_roc(CLOSE, 5) < 0", max_hold=10)
assert isinstance(stf, pd.Series)
assert set(stf.unique()).issubset({0, 1})
print(f"    pos% = {(stf > 0).mean() * 100:.1f}%")
print("    ✅ stateful_signal() 正常")

# 6. 回测测试
print("\n[6] 回测测试")
r = SigEngine.backtest(stf, data, symbol='399317.SZ', mode='fast', start_date='2024-01-01')
print(f"    SR={r.sharpe_ratio():.3f}  CAGR={r.annualized_return()*100:.1f}%  MDD={r.max_drawdown()[0]*100:.1f}%")
print("    ✅ SigEngine.backtest() 正常")

# 7. 状态机回测测试
print("\n[7] 状态机回测测试 (backtest_stateful)")
r2 = SigEngine.backtest_stateful(
    data,
    "ts_roc(CLOSE, 5) > 0",
    "ts_roc(CLOSE, 5) < 0",
    max_hold=10,
    mode='fast',
    start_date='2024-01-01'
)
print(f"    SR={r2.sharpe_ratio():.3f}  CAGR={r2.annualized_return()*100:.1f}%  MDD={r2.max_drawdown()[0]*100:.1f}%")
print("    ✅ SigEngine.backtest_stateful() 正常")

# 8. 跨模块一致性检查
print("\n[8] v4 vs v5 跨模块一致性")
from signals.v4 import Expression as ExprV4, SigEngine as EngineV4, stateful_signal as stfV4
expr_v4 = ExprV4("ts_roc(CLOSE, 5) > 0")
expr_v5 = Expression("ts_roc(CLOSE, 5) > 0")
assert expr_v4.variables == expr_v5.variables
assert expr_v4.functions == expr_v5.functions
assert expr_v4.complexity == expr_v5.complexity
print("    ✅ Expression 解析结果一致")

print("\n" + "=" * 60)
print("signals/v5 冒烟测试全部通过 ✅")
print("=" * 60)
