"""
因子计算引擎模块

设计思路：
---------
1. 批量计算：支持同时计算多个因子
2. 依赖处理：自动处理因子间的依赖关系
3. 缓存优化：缓存中间计算结果
4. 数据适配：兼容现有回测框架的数据格式
5. 并行计算：支持多进程/多线程加速

使用方式：
---------
1. 创建FactorCalculator实例
2. 注册需要计算的因子
3. 提供数据源（可以是函数或已有数据）
4. 执行批量计算
5. 获取计算结果
"""

import warnings
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any, Callable, Tuple
from collections import defaultdict, deque
import pandas as pd
import numpy as np
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging

from .base import Factor, FactorMetadata, FactorCategory, FactorFrequency

logger = logging.getLogger(__name__)


class DataSource:
    """数据源抽象类
    
    负责提供因子计算所需的数据，可以是：
    1. 内存中的数据
    2. 数据库查询
    3. API调用
    4. 文件读取
    """
    
    def __init__(self, data: Optional[Dict[str, pd.DataFrame]] = None):
        """
        初始化数据源
        
        Args:
            data: 内存数据字典，key为字段名，value为DataFrame
        """
        self._data = data or {}
        
    def get_data(self, symbols: List[str], 
                 dates: List[date], 
                 fields: List[str]) -> Dict[str, pd.DataFrame]:
        """
        获取数据
        
        Args:
            symbols: 标的列表
            dates: 日期列表
            fields: 字段列表
            
        Returns:
            Dict[str, pd.DataFrame]: 数据字典
        """
        result = {}
        
        for field in fields:
            if field in self._data:
                df = self._data[field]
                # 筛选指定的标的和日期
                if set(symbols).issubset(set(df.columns)) and set(dates).issubset(set(df.index)):
                    result[field] = df.loc[dates, symbols]
                else:
                    # 如果数据不完整，返回NaN填充的DataFrame
                    result[field] = pd.DataFrame(
                        np.nan, 
                        index=dates, 
                        columns=symbols
                    )
                    logger.warning(f"字段 {field} 数据不完整，使用NaN填充")
            else:
                # 字段不存在，返回NaN填充的DataFrame
                result[field] = pd.DataFrame(
                    np.nan, 
                    index=dates, 
                    columns=symbols
                )
                logger.warning(f"字段 {field} 不存在，使用NaN填充")
                
        return result
    
    def add_data(self, field: str, data: pd.DataFrame):
        """
        添加数据
        
        Args:
            field: 字段名
            data: 数据DataFrame
        """
        self._data[field] = data
        
    def clear(self):
        """清空数据"""
        self._data.clear()


class FactorDependencyGraph:
    """因子依赖关系图
    
    分析因子间的依赖关系，支持拓扑排序
    """
    
    def __init__(self):
        self.graph = defaultdict(set)  # 邻接表
        self.reverse_graph = defaultdict(set)  # 反向邻接表
        self.factors = {}  # 因子名称到因子的映射
        
    def add_factor(self, factor: Factor):
        """
        添加因子
        
        Args:
            factor: 因子实例
        """
        factor_name = factor.metadata.name
        self.factors[factor_name] = factor
        
        # 添加依赖关系
        dependencies = factor.metadata.dependencies
        for dep in dependencies:
            self.graph[dep].add(factor_name)
            self.reverse_graph[factor_name].add(dep)
            
    def get_dependencies(self, factor_name: str) -> List[str]:
        """
        获取因子的直接依赖
        
        Args:
            factor_name: 因子名称
            
        Returns:
            List[str]: 依赖的因子名称列表
        """
        return list(self.reverse_graph.get(factor_name, set()))
    
    def get_dependents(self, factor_name: str) -> List[str]:
        """
        获取依赖该因子的因子
        
        Args:
            factor_name: 因子名称
            
        Returns:
            List[str]: 依赖该因子的因子名称列表
        """
        return list(self.graph.get(factor_name, set()))
    
    def topological_sort(self) -> List[str]:
        """
        拓扑排序
        
        Returns:
            List[str]: 排序后的因子名称列表
        """
        # Kahn算法
        in_degree = defaultdict(int)
        for node in self.factors:
            in_degree[node] = len(self.reverse_graph.get(node, set()))
            
        # 入度为0的节点
        queue = deque([node for node in self.factors if in_degree[node] == 0])
        result = []
        
        while queue:
            node = queue.popleft()
            result.append(node)
            
            for neighbor in self.graph.get(node, set()):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
                    
        if len(result) != len(self.factors):
            # 存在环
            raise ValueError("因子依赖图中存在环")
            
        return result
    
    def get_calculation_order(self) -> List[List[str]]:
        """
        获取计算批次（同一批次的因子可以并行计算）
        
        Returns:
            List[List[str]]: 批次列表，每个批次包含可以并行计算的因子
        """
        topological_order = self.topological_sort()
        levels = defaultdict(list)
        level_map = {}
        
        # 计算每个节点的层级
        for node in topological_order:
            deps = self.reverse_graph.get(node, set())
            if not deps:
                level = 0
            else:
                level = max(level_map[dep] for dep in deps) + 1
            level_map[node] = level
            levels[level].append(node)
            
        # 按层级分组
        batches = []
        for level in sorted(levels.keys()):
            batches.append(levels[level])
            
        return batches


class FactorCalculator:
    """因子计算引擎"""
    
    def __init__(self, data_source: Optional[DataSource] = None, 
                 max_workers: int = 4,
                 use_cache: bool = True):
        """
        初始化因子计算引擎
        
        Args:
            data_source: 数据源，如果为None则创建空数据源
            max_workers: 最大工作线程数
            use_cache: 是否使用缓存
        """
        self.data_source = data_source or DataSource()
        self.max_workers = max_workers
        self.use_cache = use_cache
        
        self.dependency_graph = FactorDependencyGraph()
        self.factors = {}  # 因子名称 -> 因子实例
        self.cache = {}  # 缓存计算结果
        
        # 统计信息
        self.stats = {
            'total_calculations': 0,
            'cache_hits': 0,
            'calculation_time': 0.0
        }
        
    def register_factor(self, factor: Factor):
        """
        注册因子
        
        Args:
            factor: 因子实例
        """
        factor_name = factor.metadata.name
        
        if factor_name in self.factors:
            logger.warning(f"因子 {factor_name} 已注册，将被覆盖")
            
        self.factors[factor_name] = factor
        self.dependency_graph.add_factor(factor)
        
        logger.info(f"注册因子: {factor_name}")
        
    def register_factors(self, factors: List[Factor]):
        """
        批量注册因子
        
        Args:
            factors: 因子实例列表
        """
        for factor in factors:
            self.register_factor(factor)
            
    def get_factor(self, name: str) -> Optional[Factor]:
        """
        获取因子实例
        
        Args:
            name: 因子名称
            
        Returns:
            Optional[Factor]: 因子实例，如果不存在返回None
        """
        return self.factors.get(name)
    
    def list_factors(self) -> List[str]:
        """
        列出所有注册的因子
        
        Returns:
            List[str]: 因子名称列表
        """
        return list(self.factors.keys())
    
    def calculate_single(self, factor_name: str, 
                        symbols: List[str], 
                        dates: List[date],
                        required_fields: Optional[List[str]] = None) -> pd.DataFrame:
        """
        计算单个因子
        
        Args:
            factor_name: 因子名称
            symbols: 标的列表
            dates: 日期列表
            required_fields: 需要的字段列表，如果为None则从因子依赖中推断
            
        Returns:
            pd.DataFrame: 因子值
        """
        factor = self.get_factor(factor_name)
        if factor is None:
            raise ValueError(f"因子 {factor_name} 未注册")
            
        # 获取因子依赖的字段
        if required_fields is None:
            # 从因子依赖中推断需要的字段
            # 这里简化处理，实际应用中需要更复杂的推断逻辑
            required_fields = ['close', 'volume', 'amount']  # 默认字段
            
        # 获取数据
        data = self.data_source.get_data(symbols, dates, required_fields)
        
        # 检查缓存
        cache_key = factor.get_cache_key(symbols, dates)
        if self.use_cache and cache_key in self.cache:
            self.stats['cache_hits'] += 1
            return self.cache[cache_key]
            
        # 计算因子
        start_time = datetime.now()
        result = factor.calculate_with_cache(data, symbols, dates)
        end_time = datetime.now()
        
        # 更新统计
        self.stats['total_calculations'] += 1
        self.stats['calculation_time'] += (end_time - start_time).total_seconds()
        
        # 缓存结果
        if self.use_cache:
            self.cache[cache_key] = result
            
        return result
    
    def calculate_batch(self, factor_names: List[str],
                       symbols: List[str],
                       dates: List[date],
                       parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        批量计算因子
        
        Args:
            factor_names: 因子名称列表
            symbols: 标的列表
            dates: 日期列表
            parallel: 是否并行计算
            
        Returns:
            Dict[str, pd.DataFrame]: 因子计算结果字典
        """
        # 检查所有因子是否已注册
        for name in factor_names:
            if name not in self.factors:
                raise ValueError(f"因子 {name} 未注册")
                
        # 获取计算顺序
        calculation_order = self.dependency_graph.get_calculation_order()
        
        # 过滤出需要计算的因子
        needed_factors = set(factor_names)
        all_results = {}
        
        # 按批次计算
        for batch in calculation_order:
            # 过滤出本批次需要计算的因子
            batch_factors = [f for f in batch if f in needed_factors]
            if not batch_factors:
                continue
                
            if parallel and len(batch_factors) > 1:
                # 并行计算
                batch_results = self._calculate_batch_parallel(
                    batch_factors, symbols, dates
                )
            else:
                # 串行计算
                batch_results = {}
                for factor_name in batch_factors:
                    try:
                        result = self.calculate_single(factor_name, symbols, dates)
                        batch_results[factor_name] = result
                    except Exception as e:
                        logger.error(f"计算因子 {factor_name} 失败: {e}")
                        # 使用NaN填充失败的结果
                        batch_results[factor_name] = pd.DataFrame(
                            np.nan, index=dates, columns=symbols
                        )
                        
            all_results.update(batch_results)
            
            # 更新数据源，将计算结果作为新字段
            for factor_name, result in batch_results.items():
                self.data_source.add_data(factor_name, result)
                
        return all_results
    
    def _calculate_batch_parallel(self, factor_names: List[str],
                                symbols: List[str],
                                dates: List[date]) -> Dict[str, pd.DataFrame]:
        """
        并行计算一批因子
        
        Args:
            factor_names: 因子名称列表
            symbols: 标的列表
            dates: 日期列表
            
        Returns:
            Dict[str, pd.DataFrame]: 因子计算结果字典
        """
        results = {}
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交任务
            future_to_name = {}
            for factor_name in factor_names:
                future = executor.submit(
                    self.calculate_single, factor_name, symbols, dates
                )
                future_to_name[future] = factor_name
                
            # 收集结果
            for future in as_completed(future_to_name):
                factor_name = future_to_name[future]
                try:
                    result = future.result()
                    results[factor_name] = result
                except Exception as e:
                    logger.error(f"并行计算因子 {factor_name} 失败: {e}")
                    # 使用NaN填充失败的结果
                    results[factor_name] = pd.DataFrame(
                        np.nan, index=dates, columns=symbols
                    )
                    
        return results
    
    def calculate_all(self, symbols: List[str],
                     dates: List[date],
                     parallel: bool = True) -> Dict[str, pd.DataFrame]:
        """
        计算所有注册的因子
        
        Args:
            symbols: 标的列表
            dates: 日期列表
            parallel: 是否并行计算
            
        Returns:
            Dict[str, pd.DataFrame]: 所有因子计算结果
        """
        return self.calculate_batch(
            list(self.factors.keys()), symbols, dates, parallel
        )
    
    def get_required_fields(self, factor_names: List[str]) -> List[str]:
        """
        获取计算指定因子所需的数据字段
        
        Args:
            factor_names: 因子名称列表
            
        Returns:
            List[str]: 所需字段列表
        """
        # 这里简化处理，实际应用中需要根据因子定义推断所需字段
        # 可以从因子元数据的dependencies中推断
        required = set()
        
        for factor_name in factor_names:
            factor = self.get_factor(factor_name)
            if factor:
                # 添加基础字段
                required.update(['close', 'volume', 'amount'])
                # 添加因子依赖的其他因子
                required.update(factor.metadata.dependencies)
                
        return list(required)
    
    def clear_cache(self):
        """清空缓存"""
        self.cache.clear()
        for factor in self.factors.values():
            factor.clear_cache()
        logger.info("缓存已清空")
        
    def get_stats(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息字典
        """
        stats = self.stats.copy()
        stats['registered_factors'] = len(self.factors)
        stats['cache_size'] = len(self.cache)
        return stats
    
    def export_results(self, results: Dict[str, pd.DataFrame], 
                      output_dir: str = ".",
                      format: str = "parquet") -> Dict[str, str]:
        """
        导出计算结果
        
        Args:
            results: 计算结果字典
            output_dir: 输出目录
            format: 输出格式，支持 'parquet', 'csv', 'feather'
            
        Returns:
            Dict[str, str]: 文件路径字典
        """
        import os
        from pathlib import Path
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        file_paths = {}
        
        for factor_name, df in results.items():
            # 生成文件名
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{factor_name}_{timestamp}"
            
            if format == "parquet":
                filepath = output_path / f"{filename}.parquet"
                df.to_parquet(filepath)
            elif format == "csv":
                filepath = output_path / f"{filename}.csv"
                df.to_csv(filepath)
            elif format == "feather":
                filepath = output_path / f"{filename}.feather"
                df.to_feather(filepath)
            else:
                raise ValueError(f"不支持的格式: {format}")
                
            file_paths[factor_name] = str(filepath)
            logger.info(f"导出因子 {factor_name} 到 {filepath}")
            
        return file_paths


# 工具函数
def create_sample_data(symbols: List[str], 
                      dates: List[date],
                      seed: int = 42) -> Dict[str, pd.DataFrame]:
    """
    创建示例数据
    
    Args:
        symbols: 标的列表
        dates: 日期列表
        seed: 随机种子
        
    Returns:
        Dict[str, pd.DataFrame]: 示例数据
    """
    np.random.seed(seed)
    
    data = {}
    n_dates = len(dates)
    n_symbols = len(symbols)
    
    # 生成价格数据
    base_prices = np.random.uniform(10, 100, n_symbols)
    price_returns = np.random.normal(0.0005, 0.02, (n_dates, n_symbols))
    
    # 累积收益率得到价格序列
    price_cumulative = np.cumprod(1 + price_returns, axis=0)
    prices = base_prices * price_cumulative
    
    data['close'] = pd.DataFrame(prices, index=dates, columns=symbols)
    
    # 生成成交量数据（与价格波动相关）
    volume_base = np.random.uniform(1e6, 1e7, n_symbols)
    volume_scale = np.abs(price_returns) * 10 + 1
    volumes = volume_base * volume_scale
    
    data['volume'] = pd.DataFrame(volumes, index=dates, columns=symbols)
    
    # 生成成交额数据
    data['amount'] = data['close'] * data['volume']
    
    # 生成开盘价（略低于收盘价）
    data['open'] = data['close'] * (1 - np.random.uniform(0, 0.02, (n_dates, n_symbols)))
    
    # 生成最高价（略高于收盘价）
    data['high'] = data['close'] * (1 + np.random.uniform(0, 0.03, (n_dates, n_symbols)))
    
    # 生成最低价（略低于开盘价）
    data['low'] = data['open'] * (1 - np.random.uniform(0, 0.02, (n_dates, n_symbols)))
    
    return data