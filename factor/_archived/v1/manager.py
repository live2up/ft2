"""
因子管理器模块

设计思路：
---------
1. 因子库管理：存储和管理所有因子定义
2. 因子版本控制：支持因子版本管理和回滚
3. 因子元数据管理：维护因子的完整元数据
4. 因子依赖管理：管理因子间的依赖关系
5. 因子性能跟踪：跟踪因子的历史表现

使用方式：
---------
1. 创建FactorManager实例
2. 注册因子（从文件、数据库或代码）
3. 查询和检索因子
4. 管理因子版本
5. 导出/导入因子库
"""

import warnings
from datetime import datetime, date
from typing import Dict, List, Optional, Union, Any, Set, Tuple
from dataclasses import dataclass, field, asdict
from enum import Enum
import pandas as pd
import numpy as np
import json
# 可选依赖：PyYAML
try:
    import yaml
    HAS_YAML = True
except ImportError:
    HAS_YAML = False
    yaml = None
import pickle
from pathlib import Path
import hashlib
import logging
from copy import deepcopy

from .base import Factor, FactorMetadata, FactorCategory, FactorFrequency, FactorRegistry
from .calculator import FactorCalculator
from .validator import FactorValidator
from .combiner import FactorCombiner

logger = logging.getLogger(__name__)


class StorageFormat(Enum):
    """存储格式枚举"""
    PICKLE = "pickle"      # Python pickle格式
    JSON = "json"          # JSON格式
    YAML = "yaml"          # YAML格式
    PARQUET = "parquet"    # Parquet格式（仅因子值）


class FactorStatus(Enum):
    """因子状态枚举"""
    DRAFT = "draft"        # 草稿
    ACTIVE = "active"      # 活跃
    DEPRECATED = "deprecated"  # 已弃用
    ARCHIVED = "archived"  # 已归档


@dataclass
class FactorVersion:
    """因子版本信息"""
    version: str                    # 版本号（语义化版本，如1.0.0）
    factor_name: str               # 因子名称
    metadata: FactorMetadata       # 因子元数据
    code_hash: str                 # 代码哈希（用于验证）
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    created_by: str = ""           # 创建者
    description: str = ""          # 版本描述
    status: FactorStatus = FactorStatus.DRAFT  # 状态
    performance_metrics: Dict[str, Any] = field(default_factory=dict)  # 性能指标
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        result = asdict(self)
        result['metadata'] = asdict(self.metadata)
        result['status'] = self.status.value
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactorVersion':
        """从字典创建"""
        # 处理metadata
        metadata_dict = data.get('metadata', {})
        metadata = FactorMetadata(
            name=metadata_dict.get('name', ''),
            description=metadata_dict.get('description', ''),
            category=FactorCategory(metadata_dict.get('category', 'custom')),
            frequency=FactorFrequency(metadata_dict.get('frequency', '1d')),
            author=metadata_dict.get('author', ''),
            version=metadata_dict.get('version', '1.0.0'),
            created_at=datetime.fromisoformat(metadata_dict.get('created_at')) 
                        if metadata_dict.get('created_at') else datetime.now(),
            updated_at=datetime.fromisoformat(metadata_dict.get('updated_at')) 
                        if metadata_dict.get('updated_at') else datetime.now(),
            parameters=metadata_dict.get('parameters', {}),
            dependencies=metadata_dict.get('dependencies', [])
        )
        
        # 处理datetime字段
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
            
        return cls(
            version=data.get('version', '1.0.0'),
            factor_name=data.get('factor_name', ''),
            metadata=metadata,
            code_hash=data.get('code_hash', ''),
            created_at=created_at,
            created_by=data.get('created_by', ''),
            description=data.get('description', ''),
            status=FactorStatus(data.get('status', 'draft')),
            performance_metrics=data.get('performance_metrics', {})
        )


@dataclass
class FactorLibraryEntry:
    """因子库条目"""
    factor_name: str                       # 因子名称
    latest_version: str                    # 最新版本
    versions: Dict[str, FactorVersion]     # 所有版本
    tags: Set[str] = field(default_factory=set)  # 标签
    created_at: datetime = field(default_factory=datetime.now)  # 创建时间
    updated_at: datetime = field(default_factory=datetime.now)  # 更新时间
    
    def add_version(self, version: FactorVersion):
        """添加版本"""
        self.versions[version.version] = version
        self.latest_version = version.version
        self.updated_at = datetime.now()
        
    def get_version(self, version: str = None) -> Optional[FactorVersion]:
        """获取版本"""
        if version is None:
            version = self.latest_version
        return self.versions.get(version)
    
    def list_versions(self) -> List[str]:
        """列出所有版本"""
        return sorted(self.versions.keys())
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'factor_name': self.factor_name,
            'latest_version': self.latest_version,
            'versions': {v: ver.to_dict() for v, ver in self.versions.items()},
            'tags': list(self.tags),
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FactorLibraryEntry':
        """从字典创建"""
        # 处理versions
        versions_dict = data.get('versions', {})
        versions = {}
        for ver_str, ver_data in versions_dict.items():
            versions[ver_str] = FactorVersion.from_dict(ver_data)
            
        # 处理datetime字段
        created_at = data.get('created_at')
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at)
        elif created_at is None:
            created_at = datetime.now()
            
        updated_at = data.get('updated_at')
        if isinstance(updated_at, str):
            updated_at = datetime.fromisoformat(updated_at)
        elif updated_at is None:
            updated_at = datetime.now()
            
        return cls(
            factor_name=data.get('factor_name', ''),
            latest_version=data.get('latest_version', ''),
            versions=versions,
            tags=set(data.get('tags', [])),
            created_at=created_at,
            updated_at=updated_at
        )


class FactorManager:
    """因子管理器"""
    
    def __init__(self, 
                 storage_path: Optional[Union[str, Path]] = None,
                 auto_load: bool = True):
        """
        初始化因子管理器
        
        Args:
            storage_path: 存储路径，如果为None则使用内存存储
            auto_load: 是否自动从存储路径加载
        """
        self.storage_path = Path(storage_path) if storage_path else None
        self.library: Dict[str, FactorLibraryEntry] = {}  # 因子库
        self.factor_instances: Dict[str, Factor] = {}     # 因子实例缓存
        
        # 创建存储目录（如果存在存储路径）
        if self.storage_path and not self.storage_path.exists():
            self.storage_path.mkdir(parents=True, exist_ok=True)
            
        # 自动加载
        if auto_load and self.storage_path:
            self.load_library()
            
    def register_factor(self, 
                       factor: Factor,
                       version: str = None,
                       created_by: str = "",
                       description: str = "",
                       tags: List[str] = None,
                       status: FactorStatus = FactorStatus.DRAFT) -> str:
        """
        注册因子
        
        Args:
            factor: 因子实例
            version: 版本号，如果为None则自动生成
            created_by: 创建者
            description: 版本描述
            tags: 标签列表
            status: 因子状态
            
        Returns:
            str: 注册的版本号
        """
        factor_name = factor.metadata.name
        
        # 生成代码哈希
        code_hash = self._calculate_factor_hash(factor)
        
        # 确定版本号
        if version is None:
            version = self._generate_next_version(factor_name)
            
        # 创建版本信息
        factor_version = FactorVersion(
            version=version,
            factor_name=factor_name,
            metadata=factor.metadata,
            code_hash=code_hash,
            created_by=created_by,
            description=description,
            status=status
        )
        
        # 添加到因子库
        if factor_name not in self.library:
            self.library[factor_name] = FactorLibraryEntry(
                factor_name=factor_name,
                latest_version=version,
                versions={version: factor_version},
                tags=set(tags or [])
            )
        else:
            entry = self.library[factor_name]
            entry.add_version(factor_version)
            if tags:
                entry.tags.update(tags)
                
        # 缓存因子实例
        self.factor_instances[f"{factor_name}_{version}"] = factor
        
        logger.info(f"注册因子: {factor_name} v{version}")
        
        # 自动保存
        if self.storage_path:
            self.save_library()
            
        return version
    
    def _calculate_factor_hash(self, factor: Factor) -> str:
        """
        计算因子哈希值
        
        Args:
            factor: 因子实例
            
        Returns:
            str: 哈希值
        """
        # 这里简化处理，实际应用中可能需要更复杂的哈希计算
        # 可以考虑序列化因子类或计算源代码哈希
        content = f"{factor.metadata.name}{factor.metadata.description}{factor.metadata.category.value}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _generate_next_version(self, factor_name: str) -> str:
        """
        生成下一个版本号
        
        Args:
            factor_name: 因子名称
            
        Returns:
            str: 下一个版本号
        """
        if factor_name not in self.library:
            return "1.0.0"
            
        entry = self.library[factor_name]
        versions = entry.list_versions()
        
        if not versions:
            return "1.0.0"
            
        # 获取最新版本
        latest = versions[-1]
        
        # 解析版本号
        try:
            major, minor, patch = map(int, latest.split('.'))
            # 增加修订版本号
            return f"{major}.{minor}.{patch + 1}"
        except:
            # 版本号格式不符合语义化版本，使用时间戳
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            return f"1.0.0.{timestamp}"
    
    def get_factor(self, 
                  factor_name: str, 
                  version: str = None,
                  use_cache: bool = True) -> Optional[Factor]:
        """
        获取因子实例
        
        Args:
            factor_name: 因子名称
            version: 版本号，如果为None则使用最新版本
            use_cache: 是否使用缓存
            
        Returns:
            Optional[Factor]: 因子实例，如果不存在返回None
        """
        # 检查缓存
        cache_key = f"{factor_name}_{version if version else 'latest'}"
        if use_cache and cache_key in self.factor_instances:
            return self.factor_instances[cache_key]
            
        # 从因子库获取
        if factor_name not in self.library:
            logger.warning(f"因子 {factor_name} 不存在于因子库中")
            return None
            
        entry = self.library[factor_name]
        
        if version is None:
            version = entry.latest_version
            
        factor_version = entry.get_version(version)
        if factor_version is None:
            logger.warning(f"因子 {factor_name} 版本 {version} 不存在")
            return None
            
        # 这里简化处理，实际应用中可能需要从存储中加载因子类
        # 目前只返回缓存的实例或None
        return None
    
    def list_factors(self, 
                    tags: List[str] = None,
                    category: FactorCategory = None,
                    status: FactorStatus = None) -> List[Dict[str, Any]]:
        """
        列出因子
        
        Args:
            tags: 标签过滤
            category: 分类过滤
            status: 状态过滤
            
        Returns:
            List[Dict[str, Any]]: 因子信息列表
        """
        result = []
        
        for factor_name, entry in self.library.items():
            latest_version = entry.get_version()
            if latest_version is None:
                continue
                
            metadata = latest_version.metadata
            
            # 应用过滤条件
            if tags and not any(tag in entry.tags for tag in tags):
                continue
                
            if category and metadata.category != category:
                continue
                
            if status and latest_version.status != status:
                continue
                
            result.append({
                'name': factor_name,
                'description': metadata.description,
                'category': metadata.category.value,
                'frequency': metadata.frequency.value,
                'latest_version': entry.latest_version,
                'tags': list(entry.tags),
                'status': latest_version.status.value,
                'created_at': entry.created_at.isoformat(),
                'updated_at': entry.updated_at.isoformat()
            })
            
        return result
    
    def search_factors(self, 
                      query: str,
                      search_fields: List[str] = None) -> List[Dict[str, Any]]:
        """
        搜索因子
        
        Args:
            query: 搜索查询
            search_fields: 搜索字段，如果为None则搜索所有字段
            
        Returns:
            List[Dict[str, Any]]: 搜索结果
        """
        if search_fields is None:
            search_fields = ['name', 'description', 'tags']
            
        query_lower = query.lower()
        results = []
        
        for factor_name, entry in self.library.items():
            latest_version = entry.get_version()
            if latest_version is None:
                continue
                
            metadata = latest_version.metadata
            
            # 检查是否匹配
            matched = False
            
            if 'name' in search_fields and query_lower in factor_name.lower():
                matched = True
            elif 'description' in search_fields and query_lower in metadata.description.lower():
                matched = True
            elif 'tags' in search_fields and any(query_lower in tag.lower() for tag in entry.tags):
                matched = True
            elif 'category' in search_fields and query_lower in metadata.category.value.lower():
                matched = True
            elif 'author' in search_fields and query_lower in metadata.author.lower():
                matched = True
                
            if matched:
                results.append({
                    'name': factor_name,
                    'description': metadata.description,
                    'category': metadata.category.value,
                    'latest_version': entry.latest_version,
                    'tags': list(entry.tags),
                    'status': latest_version.status.value
                })
                
        return results
    
    def update_factor_status(self, 
                           factor_name: str, 
                           version: str,
                           status: FactorStatus,
                           update_latest: bool = False) -> bool:
        """
        更新因子状态
        
        Args:
            factor_name: 因子名称
            version: 版本号，如果为None则更新所有版本
            status: 新状态
            update_latest: 是否同时更新最新版本的状态
            
        Returns:
            bool: 是否成功
        """
        if factor_name not in self.library:
            logger.error(f"因子 {factor_name} 不存在")
            return False
            
        entry = self.library[factor_name]
        
        if version is None:
            # 更新所有版本
            for ver in entry.versions.values():
                ver.status = status
        else:
            # 更新指定版本
            factor_version = entry.get_version(version)
            if factor_version is None:
                logger.error(f"因子 {factor_name} 版本 {version} 不存在")
                return False
                
            factor_version.status = status
            
            # 如果需要，同时更新最新版本
            if update_latest and version == entry.latest_version:
                for ver in entry.versions.values():
                    if ver.version == entry.latest_version:
                        ver.status = status
                        
        entry.updated_at = datetime.now()
        
        # 自动保存
        if self.storage_path:
            self.save_library()
            
        logger.info(f"更新因子 {factor_name} 版本 {version} 状态为 {status.value}")
        return True
    
    def add_factor_tags(self, 
                       factor_name: str,
                       tags: List[str]) -> bool:
        """
        添加因子标签
        
        Args:
            factor_name: 因子名称
            tags: 标签列表
            
        Returns:
            bool: 是否成功
        """
        if factor_name not in self.library:
            logger.error(f"因子 {factor_name} 不存在")
            return False
            
        entry = self.library[factor_name]
        entry.tags.update(tags)
        entry.updated_at = datetime.now()
        
        # 自动保存
        if self.storage_path:
            self.save_library()
            
        logger.info(f"为因子 {factor_name} 添加标签: {tags}")
        return True
    
    def remove_factor_tags(self, 
                          factor_name: str,
                          tags: List[str]) -> bool:
        """
        移除因子标签
        
        Args:
            factor_name: 因子名称
            tags: 标签列表
            
        Returns:
            bool: 是否成功
        """
        if factor_name not in self.library:
            logger.error(f"因子 {factor_name} 不存在")
            return False
            
        entry = self.library[factor_name]
        for tag in tags:
            entry.tags.discard(tag)
            
        entry.updated_at = datetime.now()
        
        # 自动保存
        if self.storage_path:
            self.save_library()
            
        logger.info(f"从因子 {factor_name} 移除标签: {tags}")
        return True
    
    def delete_factor(self, 
                     factor_name: str,
                     version: str = None) -> bool:
        """
        删除因子
        
        Args:
            factor_name: 因子名称
            version: 版本号，如果为None则删除整个因子
            
        Returns:
            bool: 是否成功
        """
        if factor_name not in self.library:
            logger.error(f"因子 {factor_name} 不存在")
            return False
            
        if version is None:
            # 删除整个因子
            del self.library[factor_name]
            # 清理缓存
            cache_keys = [k for k in self.factor_instances.keys() if k.startswith(f"{factor_name}_")]
            for key in cache_keys:
                del self.factor_instances[key]
                
            logger.info(f"删除因子: {factor_name}")
        else:
            # 删除指定版本
            entry = self.library[factor_name]
            
            if version not in entry.versions:
                logger.error(f"因子 {factor_name} 版本 {version} 不存在")
                return False
                
            # 不能删除唯一版本
            if len(entry.versions) == 1:
                logger.error(f"不能删除因子 {factor_name} 的唯一版本")
                return False
                
            # 删除版本
            del entry.versions[version]
            
            # 更新最新版本
            if version == entry.latest_version:
                entry.latest_version = max(entry.versions.keys())
                
            entry.updated_at = datetime.now()
            
            # 清理缓存
            cache_key = f"{factor_name}_{version}"
            if cache_key in self.factor_instances:
                del self.factor_instances[cache_key]
                
            logger.info(f"删除因子 {factor_name} 版本 {version}")
            
        # 自动保存
        if self.storage_path:
            self.save_library()
            
        return True
    
    def save_library(self, 
                    format: StorageFormat = StorageFormat.JSON,
                    path: Optional[Union[str, Path]] = None):
        """
        保存因子库
        
        Args:
            format: 存储格式
            path: 存储路径，如果为None则使用self.storage_path
        """
        if path is None:
            if self.storage_path is None:
                raise ValueError("没有指定存储路径")
            path = self.storage_path
        else:
            path = Path(path)
            
        # 准备数据
        library_data = {
            'metadata': {
                'version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'factor_count': len(self.library)
            },
            'library': {name: entry.to_dict() for name, entry in self.library.items()}
        }
        
        # 根据格式保存
        if format == StorageFormat.JSON:
            file_path = path / "factor_library.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(library_data, f, ensure_ascii=False, indent=2, default=str)
                
        elif format == StorageFormat.YAML:
            if not HAS_YAML:
                raise ImportError("PyYAML未安装，无法保存YAML格式。请安装: pip install PyYAML")
            file_path = path / "factor_library.yaml"
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(library_data, f, default_flow_style=False, allow_unicode=True)
                
        elif format == StorageFormat.PICKLE:
            file_path = path / "factor_library.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(library_data, f)
                
        else:
            raise ValueError(f"不支持的存储格式: {format}")
            
        logger.info(f"因子库已保存至: {file_path}")
    
    def load_library(self, 
                    format: StorageFormat = StorageFormat.JSON,
                    path: Optional[Union[str, Path]] = None):
        """
        加载因子库
        
        Args:
            format: 存储格式
            path: 存储路径，如果为None则使用self.storage_path
        """
        if path is None:
            if self.storage_path is None:
                raise ValueError("没有指定存储路径")
            path = self.storage_path
        else:
            path = Path(path)
            
        # 根据格式加载
        library_data = None
        
        if format == StorageFormat.JSON:
            file_path = path / "factor_library.json"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    library_data = json.load(f)
                    
        elif format == StorageFormat.YAML:
            if not HAS_YAML:
                raise ImportError("PyYAML未安装，无法加载YAML格式。请安装: pip install PyYAML")
            file_path = path / "factor_library.yaml"
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    library_data = yaml.safe_load(f)
                    
        elif format == StorageFormat.PICKLE:
            file_path = path / "factor_library.pkl"
            if file_path.exists():
                with open(file_path, 'rb') as f:
                    library_data = pickle.load(f)
                    
        else:
            raise ValueError(f"不支持的存储格式: {format}")
            
        if library_data is None:
            logger.warning(f"因子库文件不存在: {file_path}")
            return
            
        # 解析数据
        self.library.clear()
        
        library_dict = library_data.get('library', {})
        for factor_name, entry_data in library_dict.items():
            try:
                entry = FactorLibraryEntry.from_dict(entry_data)
                self.library[factor_name] = entry
            except Exception as e:
                logger.error(f"加载因子 {factor_name} 失败: {e}")
                
        logger.info(f"因子库已加载，共 {len(self.library)} 个因子")
    
    def export_factor(self, 
                     factor_name: str,
                     version: str = None,
                     output_path: Union[str, Path] = ".",
                     format: StorageFormat = StorageFormat.JSON) -> str:
        """
        导出因子
        
        Args:
            factor_name: 因子名称
            version: 版本号，如果为None则使用最新版本
            output_path: 输出路径
            format: 导出格式
            
        Returns:
            str: 导出文件路径
        """
        if factor_name not in self.library:
            raise ValueError(f"因子 {factor_name} 不存在")
            
        entry = self.library[factor_name]
        
        if version is None:
            version = entry.latest_version
            
        factor_version = entry.get_version(version)
        if factor_version is None:
            raise ValueError(f"因子 {factor_name} 版本 {version} 不存在")
            
        # 准备导出数据
        export_data = {
            'factor': factor_version.to_dict(),
            'export_info': {
                'exported_at': datetime.now().isoformat(),
                'format': format.value
            }
        }
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 根据格式导出
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{factor_name}_v{version}_{timestamp}"
        
        if format == StorageFormat.JSON:
            file_path = output_path / f"{filename}.json"
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(export_data, f, ensure_ascii=False, indent=2, default=str)
                
        elif format == StorageFormat.YAML:
            if not HAS_YAML:
                raise ImportError("PyYAML未安装，无法导出YAML格式。请安装: pip install PyYAML")
            file_path = output_path / f"{filename}.yaml"
            with open(file_path, 'w', encoding='utf-8') as f:
                yaml.dump(export_data, f, default_flow_style=False, allow_unicode=True)
                
        elif format == StorageFormat.PICKLE:
            file_path = output_path / f"{filename}.pkl"
            with open(file_path, 'wb') as f:
                pickle.dump(export_data, f)
                
        else:
            raise ValueError(f"不支持的导出格式: {format}")
            
        logger.info(f"因子 {factor_name} v{version} 已导出至: {file_path}")
        return str(file_path)
    
    def import_factor(self, 
                     import_path: Union[str, Path],
                     format: StorageFormat = StorageFormat.JSON,
                     overwrite: bool = False) -> bool:
        """
        导入因子
        
        Args:
            import_path: 导入文件路径
            format: 导入格式
            overwrite: 是否覆盖已存在的因子
            
        Returns:
            bool: 是否成功
        """
        import_path = Path(import_path)
        if not import_path.exists():
            raise ValueError(f"导入文件不存在: {import_path}")
            
        # 根据格式加载
        import_data = None
        
        if format == StorageFormat.JSON:
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = json.load(f)
                
        elif format == StorageFormat.YAML:
            if not HAS_YAML:
                raise ImportError("PyYAML未安装，无法导入YAML格式。请安装: pip install PyYAML")
            with open(import_path, 'r', encoding='utf-8') as f:
                import_data = yaml.safe_load(f)
                
        elif format == StorageFormat.PICKLE:
            with open(import_path, 'rb') as f:
                import_data = pickle.load(f)
                
        else:
            raise ValueError(f"不支持的导入格式: {format}")
            
        if import_data is None:
            raise ValueError("导入数据为空")
            
        # 解析因子数据
        factor_data = import_data.get('factor')
        if factor_data is None:
            raise ValueError("导入数据中缺少因子信息")
            
        try:
            factor_version = FactorVersion.from_dict(factor_data)
        except Exception as e:
            raise ValueError(f"解析因子数据失败: {e}")
            
        factor_name = factor_version.factor_name
        version = factor_version.version
        
        # 检查是否已存在
        if factor_name in self.library and version in self.library[factor_name].versions:
            if not overwrite:
                logger.warning(f"因子 {factor_name} 版本 {version} 已存在，跳过导入")
                return False
            else:
                logger.info(f"覆盖已存在的因子 {factor_name} 版本 {version}")
                
        # 添加到因子库
        if factor_name not in self.library:
            self.library[factor_name] = FactorLibraryEntry(
                factor_name=factor_name,
                latest_version=version,
                versions={version: factor_version},
                tags=set()
            )
        else:
            entry = self.library[factor_name]
            entry.add_version(factor_version)
            
        logger.info(f"导入因子: {factor_name} v{version}")
        
        # 自动保存
        if self.storage_path:
            self.save_library()
            
        return True
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        获取统计信息
        
        Returns:
            Dict[str, Any]: 统计信息
        """
        total_factors = len(self.library)
        total_versions = sum(len(entry.versions) for entry in self.library.values())
        
        # 按分类统计
        category_counts = {}
        for entry in self.library.values():
            latest_version = entry.get_version()
            if latest_version:
                category = latest_version.metadata.category.value
                category_counts[category] = category_counts.get(category, 0) + 1
                
        # 按状态统计
        status_counts = {}
        for entry in self.library.values():
            for version in entry.versions.values():
                status = version.status.value
                status_counts[status] = status_counts.get(status, 0) + 1
                
        # 按作者统计
        author_counts = {}
        for entry in self.library.values():
            for version in entry.versions.values():
                author = version.metadata.author
                if author:
                    author_counts[author] = author_counts.get(author, 0) + 1
                    
        return {
            'total_factors': total_factors,
            'total_versions': total_versions,
            'category_counts': category_counts,
            'status_counts': status_counts,
            'author_counts': author_counts,
            'average_versions_per_factor': total_versions / total_factors if total_factors > 0 else 0
        }
    
    def clear_cache(self):
        """清空缓存"""
        self.factor_instances.clear()
        logger.info("因子实例缓存已清空")