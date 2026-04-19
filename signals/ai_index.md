# signals 模块 AI 助手指南

## 项目定位
择时信号研究模块，专注计算，不负责输出。

## 架构原则
- **signals/ 只负责计算**：信号生成、回测、IC分析
- **test_xx.py 负责输出**：用 Notebook 组织输出 HTML 报告
- **不要混在一起**：signals/ 不依赖 Notebook

## 目录结构
```
signals/
├── signal.py           # Signal 基础类
├── generator.py        # 13种信号生成器
├── combiner.py         # 5种信号融合器
├── threshold.py        # 5种阈值策略
├── backtest.py         # 轻量回测（纯计算）
├── ic_analyzer.py      # IC分析器（纯计算）
└── examples.py         # 使用示例
```

## 核心 API

### 1. 信号生成
```python
from signals import MASignal, MACDSignal, RSISignal

signals = MASignal(5, 20).generate(df)
```

### 2. 轻量回测
```python
from signals import run_backtest

result = run_backtest(MASignal(5, 20), df, initial_capital=1e6)
# result.total_return, result.sharpe, result.max_drawdown ...
```

### 3. IC分析
```python
from signals import ICAnalyzer

ic = ICAnalyzer.analyze(signals, df['close'])
# ic['basic']['summary']['ic_mean']
# ic['significance']['p_value']
# ic['annual'], ic['turnover'], ic['distribution']
```

## BacktestResult 字段
| 类型 | 字段 |
|------|------|
| 收益 | total_return, annual_return, excess_return |
| 风险 | max_drawdown, annual_vol, downside_vol |
| 比率 | sharpe, sortino, calmar, information_ratio |
| 交易 | trade_count, win_rate, profit_loss_ratio |

## IC分析结果结构
```python
ic_result = {
    'basic': {'summary': {...}, 'IC_1d': {...}, ...},
    'rolling': {'window_30d': {...}, ...},
    'decay': {'data': [...], 'decay_rate': ...},
    'significance': {'pearson_ic': ..., 't_statistic': ..., 'p_value': ...},
    'annual': {2020: {...}, 2021: {...}, ...},
    'cumulative': {'series': ..., 'final_value': ...},
    'turnover': {'mean_turnover': ..., ...},
    'distribution': {'mean': ..., 'skewness': ..., ...},
}
```

## 测试文件路径
测试文件存放在：`d:\01-Doc\AIdev\001zeshi_simp\01_重叠研究_Overlap\test_xx.py`

## 注意事项
- 已删除 evaluator.py（旧架构混合计算和输出）
- BacktestResult 不再有 export_report() 方法
- 新测试必须用 Notebook 自己组织输出
