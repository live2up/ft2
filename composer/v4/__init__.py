"""composer/v4 — 多风格组合管理器

用法:
  >>> from composer.v4 import StyleConfig, StyleManager, StyleBacktest
  >>> mgr = StyleManager([StyleConfig('强势', panel_A, 0.5, 2), StyleConfig('反转', panel_B, 0.5, 2)])
  >>> analyzer = StyleBacktest.backtest(mgr, assets)
"""

from .style_composer import StyleConfig, StyleManager, StyleBacktest

__all__ = ['StyleConfig', 'StyleManager', 'StyleBacktest']
