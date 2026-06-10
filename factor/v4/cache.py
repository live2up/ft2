"""
因子值持久化缓存

设计思路：
--------
1. FactorCacheStore：因子值持久化缓存管理器
2. 键结构: (factor_name, lookback, date_range_start, date_range_end)
3. 双格式支持：parquet（推荐，列存压缩）和 pickle（备用）
4. 典型使用场景：BO/网格搜索时避免重复计算相同的因子值

使用方式：
--------
>>> store = FactorCacheStore('./cache')
>>> store.save('Momentum', 20, '2020-01', '2025-12', factor_df)
>>> cached = store.load('Momentum', 20, '2020-01', '2025-12')
>>> if cached is not None:  # 命中缓存
...     pass

依赖说明：
--------
仅依赖 pandas + pickle，不依赖 ft2 其他模块。
parquet 格式需要 pyarrow（可选）。
"""

import os
import pickle
import hashlib
from pathlib import Path
from typing import Optional, Dict, List
import pandas as pd
import logging

logger = logging.getLogger(__name__)


class FactorCacheStore:
    """因子值持久化缓存管理器
    
    按 (factor_name, lookback_days, start_date, end_date) 构建缓存键，
    支持 parquet 和 pickle 两种存储格式。
    
    目录结构：
    {cache_dir}/
      └── {factor_name}/
          └── {lookback}/
              └── {date_range_hash}.parquet  (或 .pkl)
    """

    def __init__(self, cache_dir: str = '.factor_cache',
                 format: str = 'parquet'):
        """初始化缓存管理器
        
        Args:
            cache_dir: 缓存根目录路径
            format: 存储格式，'parquet' 或 'pickle'
        """
        self.cache_dir = Path(cache_dir)
        self.format = format.lower()

        if self.format == 'parquet':
            try:
                import pyarrow  # noqa: F401 验证可用
            except ImportError:
                logger.warning(
                    "parquet 格式需要 pyarrow，未安装将回退到 pickle 格式"
                )
                self.format = 'pickle'

        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def _make_cache_path(self, factor_name: str, lookback: int,
                         start_date: str, end_date: str) -> Path:
        """构建缓存文件路径
        
        Args:
            factor_name: 因子名称
            lookback: 回看天数
            start_date: 起始日期字符串
            end_date: 结束日期字符串
            
        Returns:
            Path: 缓存文件路径
        """
        # 用 hash 缩短文件名，避免路径过长
        key_str = f"{start_date}_{end_date}"
        key_hash = hashlib.md5(key_str.encode()).hexdigest()[:8]
        ext = 'parquet' if self.format == 'parquet' else 'pkl'
        filename = f"{key_hash}.{ext}"

        # 安全的目录名（替换非法字符）
        safe_name = factor_name.replace('/', '_').replace('\\', '_')
        sub_dir = self.cache_dir / safe_name / str(lookback)
        sub_dir.mkdir(parents=True, exist_ok=True)

        return sub_dir / filename

    def save(self, factor_name: str, lookback: int,
             start_date: str, end_date: str,
             data: pd.DataFrame) -> bool:
        """保存因子值到缓存
        
        Args:
            factor_name: 因子名称
            lookback: 回看天数
            start_date: 起始日期（字符串，如 '2020-01-01'）
            end_date: 结束日期
            data: 因子值 DataFrame（index=日期, columns=标的）
            
        Returns:
            bool: 是否保存成功
        """
        try:
            path = self._make_cache_path(factor_name, lookback, start_date, end_date)
            if self.format == 'parquet':
                data.to_parquet(path)
            else:
                with open(path, 'wb') as f:
                    pickle.dump(data, f)
            logger.debug(f"缓存已保存: {path}")
            return True
        except Exception as e:
            logger.error(f"缓存保存失败 [{factor_name}/{lookback}]: {e}")
            return False

    def load(self, factor_name: str, lookback: int,
             start_date: str, end_date: str) -> Optional[pd.DataFrame]:
        """从缓存加载因子值
        
        Args:
            factor_name: 因子名称
            lookback: 回看天数
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            Optional[pd.DataFrame]: 缓存的因子值，未命中时返回 None
        """
        try:
            path = self._make_cache_path(factor_name, lookback, start_date, end_date)
            if not path.exists():
                return None

            if self.format == 'parquet':
                data = pd.read_parquet(path)
            else:
                with open(path, 'rb') as f:
                    data = pickle.load(f)

            logger.debug(f"缓存命中: {path}")
            return data
        except Exception as e:
            logger.warning(f"缓存读取失败 [{factor_name}/{lookback}]: {e}")
            return None

    def exists(self, factor_name: str, lookback: int,
               start_date: str, end_date: str) -> bool:
        """检查缓存是否存在
        
        Args:
            factor_name: 因子名称
            lookback: 回看天数
            start_date: 起始日期
            end_date: 结束日期
            
        Returns:
            bool: 缓存是否存在
        """
        path = self._make_cache_path(factor_name, lookback, start_date, end_date)
        return path.exists()

    def list_cached(self, factor_name: Optional[str] = None) -> List[Dict]:
        """列出已缓存的因子
        
        Args:
            factor_name: 因子名称过滤，None 表示列出所有
            
        Returns:
            List[Dict]: 缓存条目列表，每项包含 name/lookback/start/end/path
        """
        result = []
        base = self.cache_dir

        if factor_name:
            factor_dirs = [base / factor_name]
        else:
            factor_dirs = [d for d in base.iterdir() if d.is_dir()]

        for fdir in factor_dirs:
            if not fdir.is_dir():
                continue
            name = fdir.name
            for ldir in fdir.iterdir():
                if not ldir.is_dir():
                    continue
                try:
                    lookback = int(ldir.name)
                except ValueError:
                    continue
                for fpath in ldir.iterdir():
                    if fpath.suffix in ('.parquet', '.pkl'):
                        # 从文件名解析日期范围（hash 不可逆，记录 path 即可）
                        result.append({
                            'name': name,
                            'lookback': lookback,
                            'path': str(fpath),
                        })
        return result

    def clear(self, factor_name: Optional[str] = None):
        """清除缓存
        
        Args:
            factor_name: 因子名称过滤，None 表示清除所有
        """
        if factor_name:
            target = self.cache_dir / factor_name
            if target.exists():
                import shutil
                shutil.rmtree(target)
                logger.info(f"已清除因子缓存: {factor_name}")
        else:
            import shutil
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            logger.info("已清除所有缓存")

    def __repr__(self) -> str:
        return f"FactorCacheStore(dir='{self.cache_dir}', format='{self.format}')"
