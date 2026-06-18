"""
AMP backbone adapter for CPL.

这里不再重新实现 AMP block，而是直接复用 HieraMamba 的
`HieraMambaBackbone -> AnchorMambaPoolingBlockGated -> Hydra` 链路。

整体流程:
    CPL video features (B, T, C_in)
        -> 转成 HieraMamba 需要的 (B, C_in, T)
        -> HieraMambaBackbone
            1. MaskedConv1D 输入投影到 embd_dim
            2. embedding convs / absolute position embedding
            3. 多层 AnchorMambaPoolingBlockGated
                - 每 2 个 token 生成 1 个 anchor
                - 将 anchor 和原 token 交错成 packed sequence
                - 用 Hydra 做双向全局时序建模
                - gate1 融合原 packed 特征和 Hydra 输出
                - 可选 local Transformer 编码，再 gate2 融合
                - FFN 后拆回 sequence stream 和 anchor stream
        -> 把每层 sequence_fpn / anchor_fpn 从 embd_dim 投影到 CPL hidden_size
        -> sequence_fpn 跨尺度融合后作为 CPL 主视频 token

输出:
    frames:       (B, T0, hidden)，送入 CPL 的 DualTransformer
    sequence_fpn: 每层 sequence stream，时间长度依层级变化
    anchor_fpn:   每层 anchor stream，可用于替代 CPL 原先的均匀下采样 proposal 源
"""

from pathlib import Path
import importlib
import sys
import types

import torch
import torch.nn as nn
import torch.nn.functional as F


_HIERA_IMPORT_ERROR = None
_HIERA_BACKBONE_CLASS = None


def _ensure_hieramamba_paths():
    """让 CPL 可以从兄弟目录 `hieramamba-main` 直接导入原版 HieraMamba 代码。"""
    repo_root = Path(__file__).resolve().parents[3]
    hiera_root = repo_root / "hieramamba-main"
    hydra_root = hiera_root / "hydra"
    libs_root = hiera_root / "libs"
    for path in (hiera_root, hydra_root, libs_root):
        path_str = str(path)
        if path_str in sys.path:
            sys.path.remove(path_str)
        sys.path.insert(0, path_str)

    local_roots = (str(hiera_root), str(hydra_root), str(libs_root))
    for name in list(sys.modules):
        if name in ("hydra", "libs") or name.startswith(("hydra.", "libs.")):
            module_file = getattr(sys.modules[name], "__file__", "") or ""
            if not module_file or not module_file.startswith(local_roots):
                del sys.modules[name]
    return hiera_root, hydra_root, libs_root


def _register_minimal_package(name, path):
    """注册一个不执行 __init__.py 的 package，占位给相对导入使用。"""
    module = types.ModuleType(name)
    module.__path__ = [str(path)]
    module.__package__ = name
    sys.modules[name] = module
    return module


def _register_hieramamba_modules(hiera_root, hydra_root):
    """
    注册 CPL 需要的最小 HieraMamba/Hydra 包结构。

    这会绕开 `libs/__init__.py`、`libs/modeling/__init__.py` 和 `hydra/__init__.py`，
    避免导入 HieraMamba 原仓库完整训练/BERT 栈时触发 transformers 版本冲突。
    """
    libs_pkg = _register_minimal_package("libs", hiera_root / "libs")
    modeling_pkg = _register_minimal_package("libs.modeling", hiera_root / "libs" / "modeling")
    setattr(libs_pkg, "modeling", modeling_pkg)

    hydra_pkg = _register_minimal_package("hydra", hydra_root / "hydra")
    hydra_modules_pkg = _register_minimal_package("hydra.modules", hydra_root / "hydra" / "modules")
    setattr(hydra_pkg, "modules", hydra_modules_pkg)


def _patch_transformers_generation_outputs():
    """
    兼容旧 mamba_ssm/HieraMamba 对 transformers generation 输出类名的导入。

    新版 transformers 删除/移动了 GreedySearchDecoderOnlyOutput 等旧名字；
    这些名字只在 mamba_ssm 的 HuggingFace 集成导入阶段用到，CPL 的 AMP
    前向不会调用文本生成逻辑，因此补空占位即可避免无关依赖把训练启动卡死。
    """
    try:
        import transformers.generation as generation
        try:
            import transformers.generation.utils as generation_utils
        except Exception:
            generation_utils = None
    except Exception:
        return
    for name in (
        "GreedySearchDecoderOnlyOutput",
        "GreedySearchEncoderDecoderOutput",
        "SampleDecoderOnlyOutput",
        "SampleEncoderDecoderOutput",
        "BeamSearchDecoderOnlyOutput",
        "BeamSearchEncoderDecoderOutput",
        "BeamSampleDecoderOnlyOutput",
        "BeamSampleEncoderDecoderOutput",
    ):
        if not hasattr(generation, name):
            setattr(generation, name, type(name, (object,), {}))
        if generation_utils is not None and not hasattr(generation_utils, name):
            setattr(generation_utils, name, getattr(generation, name))


def _load_hieramamba_backbone():
    """
    延迟导入原版 HieraMamba backbone。

    这样即使服务器还没装 mamba_ssm / causal-conv1d，普通 CPL import 也不会失败；
    只有真正启用 AMP 时才会给出明确的依赖错误。
    """
    global _HIERA_IMPORT_ERROR, _HIERA_BACKBONE_CLASS
    if _HIERA_BACKBONE_CLASS is not None:
        return _HIERA_BACKBONE_CLASS
    hiera_root, hydra_root, _ = _ensure_hieramamba_paths()
    _register_hieramamba_modules(hiera_root, hydra_root)
    _patch_transformers_generation_outputs()
    try:
        video_net = importlib.import_module("libs.modeling.video_net")
        OriginalHieraMambaBackbone = video_net.HieraMambaBackbone
    except Exception as exc:
        _HIERA_IMPORT_ERROR = exc
        return None
    _HIERA_BACKBONE_CLASS = OriginalHieraMambaBackbone
    return _HIERA_BACKBONE_CLASS


class HieraAMPBackbone(nn.Module):
    """
    CPL 到 HieraMamba AMP 的薄适配层。

    这个类只负责输入/输出格式和维度投影，不改变 HieraMamba 原版 AMP block
    的内部处理逻辑。若 `bidirectional=True`，原版 block 内部会使用 Hydra；
    若 `bidirectional=False`，原版 block 会退回单向 Mamba2。
    """

    def __init__(
        self,
        in_dim,
        out_dim,
        embd_dim=384,
        max_seq_len=2304,
        n_heads=4,
        mha_win_size=0,
        stride=1,
        arch=(2, 0, 8),
        dropout=0.1,
        attn_pdrop=0.0,
        proj_pdrop=None,
        path_pdrop=None,
        use_abs_pe=True,
        local_window_size=5,
        pool_method="mean",
        block_type="AnchorMambaPoolingBlockGated",
        local_encoder_type="transformer",
        ffn_ratio=2,
        local_encode=True,
        local_encode_num_layers=0,
        mamba_headdim=64,
        mamba_dstate=64,
        mamba_expand=2,
        mamba_dconv=7,
        bidirectional=True,
        use_mamba=True,
        pyramid_fusion="sum",
    ):
        super().__init__()
        if not use_mamba:
            raise ValueError(
                "Direct HieraMamba AMP requires Hydra/Mamba. "
                "Set AMP.enabled=false for baseline CPL instead of use_mamba=false."
            )

        OriginalHieraMambaBackbone = _load_hieramamba_backbone()
        if OriginalHieraMambaBackbone is None:
            raise ImportError(
                "Failed to import HieraMamba backbone. Please install causal-conv1d, "
                "mamba-ssm, Hydra dependencies, and ensure hieramamba-main is present. "
                "Original error: {}".format(repr(_HIERA_IMPORT_ERROR))
            )

        assert len(arch) == 3, "arch must be (embed_convs, stem_layers, branch_layers)"
        assert arch[2] > 0, "AMP backbone needs at least one branch layer"
        self.pyramid_fusion = pyramid_fusion
        self.branch_layers = int(arch[2])

        if proj_pdrop is None:
            proj_pdrop = dropout
        if path_pdrop is None:
            path_pdrop = dropout

        self.backbone = OriginalHieraMambaBackbone(
            in_dim=in_dim,
            embd_dim=embd_dim,
            max_seq_len=max_seq_len,
            n_heads=n_heads,
            mha_win_size=mha_win_size,
            stride=stride,
            arch=tuple(arch),
            attn_pdrop=attn_pdrop,
            proj_pdrop=proj_pdrop,
            path_pdrop=path_pdrop,
            use_abs_pe=use_abs_pe,
            local_window_size=local_window_size,
            pool_method=pool_method,
            return_anchor=True,
            block_type=block_type,
            local_encoder_type=local_encoder_type,
            ffn_ratio=ffn_ratio,
            local_encode=local_encode,
            local_encode_num_layers=local_encode_num_layers,
            mamba_headdim=mamba_headdim,
            mamba_dstate=mamba_dstate,
            mamba_expand=mamba_expand,
            mamba_dconv=mamba_dconv,
            bidirectional=bidirectional,
        )

        self.out_proj = nn.Conv1d(embd_dim, out_dim, kernel_size=1)
        if pyramid_fusion == "sum":
            self.pyramid_weights = nn.Parameter(torch.zeros(self.branch_layers))
        else:
            self.pyramid_weights = None

    @property
    def uses_mamba(self):
        return True

    def _project_pyramid(self, tensors):
        """将 HieraMamba 每层 FPN 从 embd_dim 投影到 CPL hidden_size。"""
        return tuple(self.out_proj(t) for t in tensors)

    def _fuse_pyramid(self, sequence_fpn):
        """把多尺度 sequence FPN 对齐到 finest length 后加权融合。"""
        if self.pyramid_fusion != "sum":
            return sequence_fpn[0]
        target_len = sequence_fpn[0].size(-1)
        weights = torch.softmax(self.pyramid_weights[: len(sequence_fpn)], dim=0)
        fused = 0
        for weight, feat in zip(weights, sequence_fpn):
            if feat.size(-1) != target_len:
                feat = F.interpolate(feat, size=target_len, mode="linear", align_corners=True)
            fused = fused + weight * feat
        return fused

    @staticmethod
    def _mask_to_bt(mask):
        if mask.ndim == 3:
            mask = mask.squeeze(1)
        return mask.byte()

    def forward(self, x, mask):
        if x.ndim != 3:
            raise ValueError("Expected video features with shape (B, T, C)")
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)
        mask = mask.bool()

        # HieraMamba 原版 backbone 使用 (B, C, T)，CPL 数据加载器给的是 (B, T, C)。
        x = x.transpose(1, 2).contiguous()
        sequence_fpn, sequence_masks, anchor_fpn, anchor_masks = self.backbone(x, mask)

        sequence_fpn = self._project_pyramid(sequence_fpn)
        anchor_fpn = self._project_pyramid(anchor_fpn)

        # frames 是 CPL 主干使用的视频 token；anchor_fpn 保留给 proposal 源和 Gaussian prior。
        fused = self._fuse_pyramid(sequence_fpn).transpose(1, 2).contiguous()
        sequence_fpn = tuple(feat.transpose(1, 2).contiguous() for feat in sequence_fpn)
        anchor_fpn = tuple(feat.transpose(1, 2).contiguous() for feat in anchor_fpn)
        sequence_masks = tuple(self._mask_to_bt(scale_mask) for scale_mask in sequence_masks)
        anchor_masks = tuple(self._mask_to_bt(scale_mask) for scale_mask in anchor_masks)
        return {
            "frames": fused,
            "sequence_fpn": sequence_fpn,
            "sequence_masks": sequence_masks,
            "anchor_fpn": anchor_fpn,
            "anchor_masks": anchor_masks,
        }
