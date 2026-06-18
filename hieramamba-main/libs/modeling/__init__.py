"""
Lightweight modeling package init.

CPL 只需要导入 `libs.modeling.video_net.HieraMambaBackbone`。原版这里会顺手
导入完整训练栈、loss、head 等模块；在 CPL 环境里这些额外导入可能触发
transformers/Hydra-BERT 版本冲突。需要运行 HieraMamba 原仓库训练脚本时，
请在对应脚本中显式导入它需要的 `.model`、`.losses`、`.optim` 等模块。
"""

from .video_net import HieraMambaBackbone, make_video_net
