"""
Lightweight Hydra package init for CPL/HieraMamba video backbone usage.

只导出 AMP 需要的 `Hydra` mixer，避免导入 MatrixMixer/BERT 相关路径时碰到
transformers 版本兼容问题。
"""

from .modules.hydra import Hydra
