# signal/registry.py - 信号注册器
"""
信号注册器

管理所有信号生成器
"""

from typing import Dict, Type, List, Optional
from .generator import SignalGenerator


class SignalRegistry:
    """
    信号注册器
    
    用于：
    1. 注册信号生成器
    2. 批量创建信号
    3. 管理信号模板
    """
    
    _generators: Dict[str, Type[SignalGenerator]] = {}
    
    @classmethod
    def register(cls, name: str = None):
        """
        注册信号生成器装饰器
        
        Usage:
            @SignalRegistry.register('my_signal')
            class MySignal(SignalGenerator):
                ...
        """
        def decorator(cls_: Type[SignalGenerator]) -> Type[SignalGenerator]:
            signal_name = name or cls_.__name__
            cls._generators[signal_name] = cls_
            return cls_
        return decorator
    
    @classmethod
    def get(cls, name: str) -> Optional[Type[SignalGenerator]]:
        """获取信号生成器类"""
        return cls._generators.get(name)
    
    @classmethod
    def create(cls, name: str, **kwargs) -> Optional[SignalGenerator]:
        """
        创建信号生成器实例
        
        Args:
            name: 信号名称
            **kwargs: 传给生成器的参数
            
        Returns:
            SignalGenerator 实例
        """
        cls_ = cls.get(name)
        if cls_ is None:
            raise ValueError(f"Unknown signal: {name}. Available: {list(cls._generators.keys())}")
        return cls_(**kwargs)
    
    @classmethod
    def list_signals(cls) -> List[str]:
        """列出所有注册的信号"""
        return list(cls._generators.keys())
    
    @classmethod
    def create_batch(cls, configs: List[Dict]) -> List[SignalGenerator]:
        """
        批量创建信号生成器
        
        Args:
            configs: 配置列表 [{'name': 'MA5_20', ...}, ...]
            
        Returns:
            SignalGenerator 列表
        """
        generators = []
        
        for config in configs:
            if isinstance(config, dict):
                name = config.pop('name')
                gen = cls.create(name, **config)
                generators.append(gen)
            else:
                generators.append(config)  # 已经是实例
        
        return generators
    
    @classmethod
    def clear(cls):
        """清空注册表"""
        cls._generators.clear()


# 预定义信号模板
SIGNAL_TEMPLATES = {
    # 均线系统
    'ma_short': {'class': 'MASignal', 'params': {'short_period': 5, 'long_period': 20}},
    'ma_medium': {'class': 'MASignal', 'params': {'short_period': 10, 'long_period': 60}},
    'ma_long': {'class': 'MASignal', 'params': {'short_period': 20, 'long_period': 120}},
    
    # MACD 系统
    'macd_standard': {'class': 'MACDSignal', 'params': {'fast': 12, 'slow': 26, 'signal': 9}},
    'macd_fast': {'class': 'MACDSignal', 'params': {'fast': 6, 'slow': 13, 'signal': 5}},
    'macd_slow': {'class': 'MACDSignal', 'params': {'fast': 19, 'slow': 39, 'signal': 9}},
    
    # RSI 系统
    'rsi_fast': {'class': 'RSISignal', 'params': {'period': 6}},
    'rsi_standard': {'class': 'RSISignal', 'params': {'period': 14}},
    'rsi_slow': {'class': 'RSISignal', 'params': {'period': 28}},
    
    # KDJ 系统
    'kdj_standard': {'class': 'KDJSignal', 'params': {'n': 9, 'm1': 3, 'm2': 3}},
    'kdj_fast': {'class': 'KDJSignal', 'params': {'n': 5, 'm1': 2, 'm2': 2}},
    
    # BOLL 系统
    'boll_standard': {'class': 'BOLLSignal', 'params': {'period': 20, 'std_dev': 2}},
    'boll_narrow': {'class': 'BOLLSignal', 'params': {'period': 20, 'std_dev': 1}},
    'boll_wide': {'class': 'BOLLSignal', 'params': {'period': 20, 'std_dev': 3}},
    
    # 量能系统
    'vol_short': {'class': 'VOLSignal', 'params': {'period': 5}},
    'vol_medium': {'class': 'VOLSignal', 'params': {'period': 10}},
    
    # RSRS 系统
    'rsrs_standard': {'class': 'RSRSMSignal', 'params': {'n': 18, 'm': 600}},
}


def create_signal_from_template(template_name: str, registry: SignalRegistry = None) -> SignalGenerator:
    """
    从模板创建信号
    
    Args:
        template_name: 模板名称
        registry: 信号注册器
        
    Returns:
        SignalGenerator 实例
    """
    if registry is None:
        registry = SignalRegistry
    
    if template_name not in SIGNAL_TEMPLATES:
        raise ValueError(f"Unknown template: {template_name}. Available: {list(SIGNAL_TEMPLATES.keys())}")
    
    template = SIGNAL_TEMPLATES[template_name]
    return registry.create(template['class'], **template['params'])


def create_signal_set(names: List[str], registry: SignalRegistry = None) -> List[SignalGenerator]:
    """
    创建一组信号
    
    Args:
        names: 信号名列表
        
    Returns:
        SignalGenerator 列表
    """
    if registry is None:
        registry = SignalRegistry
    
    signals = []
    
    for name in names:
        if name in SIGNAL_TEMPLATES:
            signals.append(create_signal_from_template(name, registry))
        else:
            signals.append(registry.create(name))
    
    return signals
