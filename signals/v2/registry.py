"""
signals/v2/registry.py - 表达式注册与发现

管理已发现的Expression实例：
- 注册：表达式 + 元数据
- 查询：按名称或类别检索
- 去重：基于complexity + features自动去重
- 列表：分类浏览已注册表达式
"""

import json
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

from .expression import Expression


@dataclass
class ExpressionEntry:
    """表达式注册条目"""
    name: str
    expression: Expression
    category: str = 'general'
    metadata: Dict = field(default_factory=dict)
    version: int = 1

    @property
    def source(self) -> str:
        return self.expression.source

    @property
    def complexity(self) -> int:
        return self.expression.complexity

    @property
    def features_used(self) -> List[str]:
        return self.expression.features_used

    def fingerprint(self) -> str:
        """唯一的表达式指纹"""
        key = f"{self.expression.expression}|{','.join(sorted(self.features_used))}"
        return hashlib.md5(key.encode()).hexdigest()[:12]


class ExpressionRegistry:
    """
    表达式注册表

    用法:
        registry = ExpressionRegistry()
        registry.register("gp_v43_best", expr, category="gp",
            metadata={'train_sharpe': 1.32, 'test_sharpe': 1.92})
        expr = registry.get("gp_v43_best")
        gp_list = registry.list(category="gp")
    """

    def __init__(self):
        self._entries: Dict[str, ExpressionEntry] = {}
        self._fingerprints: Dict[str, str] = {}

    def register(self, name: str, expression: Expression,
                 category: str = 'general',
                 metadata: Optional[Dict] = None,
                 overwrite: bool = False) -> ExpressionEntry:
        """注册表达式"""
        if not overwrite and name in self._entries:
            raise ValueError(f"表达式 '{name}' 已存在，设置 overwrite=True 覆盖")

        entry = ExpressionEntry(
            name=name,
            expression=expression,
            category=category,
            metadata=metadata or {},
        )
        self._entries[name] = entry
        fp = entry.fingerprint()
        self._fingerprints[fp] = name
        return entry

    def get(self, name: str) -> Optional[Expression]:
        """获取表达式"""
        entry = self._entries.get(name)
        return entry.expression if entry else None

    def get_entry(self, name: str) -> Optional[ExpressionEntry]:
        """获取完整条目（含元数据）"""
        return self._entries.get(name)

    def list(self, category: Optional[str] = None) -> List[ExpressionEntry]:
        """列出注册的表达式"""
        entries = list(self._entries.values())
        if category:
            entries = [e for e in entries if e.category == category]
        return sorted(entries, key=lambda e: e.name)

    def list_by_sharpe(self, min_sharpe: float = 0) -> List[ExpressionEntry]:
        """按夏普过滤"""
        result = []
        for entry in self._entries.values():
            ts = entry.metadata.get('test_sharpe', 0)
            if ts >= min_sharpe:
                result.append(entry)
        return sorted(result, key=lambda e: e.metadata.get('test_sharpe', 0), reverse=True)

    def is_duplicate(self, expression: Expression) -> bool:
        """检查是否与已注册表达式重复"""
        key = f"{expression.expression}|{','.join(sorted(expression.features_used))}"
        fp = hashlib.md5(key.encode()).hexdigest()[:12]
        return fp in self._fingerprints

    def remove(self, name: str):
        """移除注册的表达式"""
        if name in self._entries:
            entry = self._entries.pop(name)
            fp = entry.fingerprint()
            self._fingerprints.pop(fp, None)

    def to_dict(self) -> Dict:
        """序列化所有注册条目"""
        result = {}
        for name, entry in self._entries.items():
            result[name] = {
                'name': entry.name,
                'category': entry.category,
                'metadata': entry.metadata,
                'version': entry.version,
                'expression': entry.expression.to_dict(),
            }
        return result

    def to_json(self, path: Optional[str] = None) -> str:
        """导出为JSON"""
        data = self.to_dict()
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        if path:
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)
        return json_str

    @classmethod
    def from_json(cls, json_str_or_path: str,
                  feature_space=None, feature_df=None) -> 'ExpressionRegistry':
        """从JSON恢复注册表"""
        try:
            with open(json_str_or_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except (FileNotFoundError, OSError):
            data = json.loads(json_str_or_path)

        registry = cls()
        for name, entry_data in data.items():
            expr = Expression.from_dict(
                entry_data['expression'],
                feature_space, feature_df
            )
            registry.register(
                name=name,
                expression=expr,
                category=entry_data.get('category', 'general'),
                metadata=entry_data.get('metadata', {}),
            )
        return registry

    def __len__(self) -> int:
        return len(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __repr__(self):
        categories = {}
        for entry in self._entries.values():
            c = entry.category
            categories[c] = categories.get(c, 0) + 1
        cat_str = ', '.join(f'{c}:{n}' for c, n in categories.items())
        return f"ExpressionRegistry({len(self._entries)}条: {cat_str})"

    def select_best(self, regime: Optional[str] = None,
                    metric: str = 'test_sharpe') -> Optional[ExpressionEntry]:
        """
        根据市场状态选择最佳表达式

        Args:
            regime: 市场状态（如 '震荡市', '趋势市', '反转市'），None则不限
            metric: 排序指标

        Returns:
            最佳 ExpressionEntry
        """
        candidates = list(self._entries.values())
        if regime:
            candidates = [
                e for e in candidates
                if e.metadata.get('suitable_for', '') == regime
                or regime in e.metadata.get('suitable_regimes', [])
            ]
        if not candidates:
            return None
        return max(candidates, key=lambda e: e.metadata.get(metric, 0))

    def select_top_k(self, k: int = 3, regime: Optional[str] = None,
                     metric: str = 'test_sharpe') -> List[ExpressionEntry]:
        """选择前K个最佳表达式"""
        candidates = list(self._entries.values())
        if regime:
            candidates = [
                e for e in candidates
                if e.metadata.get('suitable_for', '') == regime
                or regime in e.metadata.get('suitable_regimes', [])
            ]
        sorted_entries = sorted(
            candidates,
            key=lambda e: e.metadata.get(metric, 0),
            reverse=True
        )
        return sorted_entries[:k]

    def ensemble(self, names: List[str], data: pd.DataFrame,
                 mode: str = 'vote', feature_space=None) -> 'pd.Series':
        """
        Ensemble投票组合多个表达式

        Args:
            names: 已注册表达式名称列表
            data: OHLCV DataFrame
            mode: 'vote' 多数投票, 'weighted' 按夏普加权, 'unanimous' 全票通过
            feature_space: FeatureSpace

        Returns:
            pd.Series 组合后的持仓信号 (0/1)
        """
        if len(names) == 0:
            raise ValueError("至少需要1个表达式")
        if len(names) == 1:
            entry = self._entries[names[0]]
            signal = entry.expression.generate(data)
            return (signal > signal.median()).astype(int)

        signals = []
        weights = []
        for name in names:
            entry = self._entries.get(name)
            if entry is None:
                continue
            sig = entry.expression.generate(data).reindex(data.index)
            signals.append(sig)
            s = entry.metadata.get('test_sharpe', 1.0)
            weights.append(max(s, 0.01))

        if not signals:
            raise ValueError("没有找到有效的表达式")

        positions = [((s > s.median()).astype(int) * 2 - 1) for s in signals]

        if mode == 'vote':
            combined = sum(positions)
            result = (combined > 0).astype(int)
        elif mode == 'weighted':
            combined = sum(p * w for p, w in zip(positions, weights))
            result = (combined > 0).astype(int)
        elif mode == 'unanimous':
            min_pos = np.minimum.reduce(positions)
            result = ((min_pos > 0) | (sum(positions) == len(positions))).astype(int)
        else:
            raise ValueError(f"不支持的ensemble模式: {mode}")

        return pd.Series(result, index=data.index)

    def update_performance(self, name: str, **metrics):
        """
        更新表达式的绩效指标

        用法:
            registry.update_performance('rsi', test_sharpe=0.57, test_ic=-0.225)
        """
        entry = self._entries.get(name)
        if entry:
            entry.metadata.update(metrics)

    def tag_regime(self, name: str, regime: str):
        """标记表达式适合的市场状态"""
        entry = self._entries.get(name)
        if entry:
            regs = entry.metadata.setdefault('suitable_regimes', [])
            if regime not in regs:
                regs.append(regime)
            entry.metadata['suitable_for'] = regime

    def regime_summary(self) -> Dict[str, List[str]]:
        """按市场状态汇总表达式"""
        summary = {}
        for entry in self._entries.values():
            for regime in entry.metadata.get('suitable_regimes', []):
                if regime not in summary:
                    summary[regime] = []
                summary[regime].append(entry.name)
            regime_single = entry.metadata.get('suitable_for')
            if regime_single and regime_single not in summary:
                summary[regime_single] = []
            if regime_single:
                summary[regime_single].append(entry.name)
        return summary

    def best_per_regime(self, metric: str = 'test_sharpe'
                        ) -> Dict[str, ExpressionEntry]:
        """每个市场状态下的最佳表达式"""
        result = {}
        summary = self.regime_summary()
        for regime, names in summary.items():
            entries = [self._entries[n] for n in names if n in self._entries]
            if entries:
                result[regime] = max(
                    entries,
                    key=lambda e: e.metadata.get(metric, 0)
                )
        return result


# 全局默认注册表
DEFAULT_REGISTRY = ExpressionRegistry()
