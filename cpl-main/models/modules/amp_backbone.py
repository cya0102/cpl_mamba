import math

import torch
import torch.nn as nn
import torch.nn.functional as F


try:
    from mamba_ssm import Mamba2
except Exception:
    Mamba2 = None


def _drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    return x.div(keep_prob) * mask.floor_()


def _sinusoid_encoding(seq_len, n_freqs):
    tics = torch.arange(seq_len, dtype=torch.float)
    freqs = 10000 ** torch.linspace(0, 1, n_freqs + 1)[:n_freqs]
    x = tics[None, :] / freqs[:, None]
    return torch.cat((torch.sin(x), torch.cos(x)))


def _downsample_mask(mask, stride):
    mask_float = mask.float()
    mask_float = F.max_pool1d(mask_float, kernel_size=stride, stride=stride, ceil_mode=True)
    return mask_float.bool()


class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        x = x.float()
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)


class LayerScale(nn.Module):
    def __init__(self, n_channels, pdrop=0.0, init_scale=1e-4):
        super().__init__()
        self.scale = nn.Parameter(init_scale * torch.ones((1, 1, n_channels)))
        self.pdrop = pdrop

    def forward(self, x):
        return _drop_path(self.scale.to(x.dtype) * x, self.pdrop, self.training)


class SwiGLUFFN(nn.Module):
    def __init__(self, dim, hidden_dim=None, dropout=0.0):
        super().__init__()
        if hidden_dim is None:
            hidden_dim = int(2 * (4 * dim) / 3)
        self.w12 = nn.Linear(dim, 2 * hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.dropout = dropout

    def forward(self, x):
        gate, up = self.w12(x).chunk(2, dim=-1)
        x = self.w2(F.silu(gate) * up)
        return F.dropout(x, p=self.dropout, training=self.training)


class ChannelLayerNorm(nn.Module):
    def __init__(self, channels, eps=1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels, 1))
        self.bias = nn.Parameter(torch.zeros(channels, 1))
        self.eps = eps

    def forward(self, x):
        x = x - torch.mean(x, dim=1, keepdim=True)
        sigma = torch.mean(x ** 2, dim=1, keepdim=True)
        return x / torch.sqrt(sigma + self.eps) * self.weight + self.bias


class MaskedConv1D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, bias=True):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=bias
        )
        if bias:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x, mask):
        if mask is None:
            mask = torch.ones_like(x[:, :1], dtype=torch.bool)
        x = self.conv(x * mask.to(x.dtype))
        if self.stride > 1:
            mask = F.interpolate(mask.float(), size=x.size(-1), mode="nearest").bool()
        return x, mask


class GatedPooling(nn.Module):
    def __init__(self, stride, d_model):
        super().__init__()
        self.stride = stride
        self.gate_proj = nn.Conv1d(2 * d_model, d_model, kernel_size=1)

    def forward(self, x):
        mean = F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride)
        max_value = F.max_pool1d(x, kernel_size=self.stride, stride=self.stride)
        gate = torch.sigmoid(self.gate_proj(torch.cat([mean, max_value], dim=1)))
        return gate * max_value + (1.0 - gate) * mean


class AttnPooling(nn.Module):
    def __init__(self, stride, d_model, nhead, dropout):
        super().__init__()
        self.stride = stride
        self.pool_q = nn.Parameter(torch.randn(1, 1, d_model) / (d_model ** 0.5))
        self.pool_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=False)

    def forward(self, x):
        bsz, dim, length = x.shape
        num_blocks = length // self.stride
        x = x.reshape(bsz, dim, num_blocks, self.stride)
        kv = x.permute(3, 0, 2, 1).reshape(self.stride, bsz * num_blocks, dim)
        query = self.pool_q.expand(-1, bsz * num_blocks, -1)
        out, _ = self.pool_attn(query, kv, kv)
        return out.squeeze(0).view(bsz, num_blocks, dim).permute(0, 2, 1)


class AnchorPooling(nn.Module):
    def __init__(self, stride, method="mean", d_model=0, nhead=1, dropout=0.0):
        super().__init__()
        self.method = method
        self.stride = stride
        if method == "mean":
            self.pooler = None
        elif method == "max":
            self.pooler = None
        elif method == "gated":
            self.pooler = GatedPooling(stride, d_model)
        elif method == "attn":
            self.pooler = AttnPooling(stride, d_model, nhead, dropout)
        else:
            raise ValueError("Unsupported anchor pool method: {}".format(method))

    def forward(self, x):
        if self.method == "mean":
            return F.avg_pool1d(x, kernel_size=self.stride, stride=self.stride)
        if self.method == "max":
            return F.max_pool1d(x, kernel_size=self.stride, stride=self.stride)
        return self.pooler(x)


class ConvSequenceMixer(nn.Module):
    def __init__(self, d_model, d_conv=7, dropout=0.0):
        super().__init__()
        padding = d_conv // 2
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, d_conv, padding=padding, groups=d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(d_model, d_model, 1),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class MambaSequenceMixer(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=64,
        d_conv=7,
        expand=2,
        headdim=64,
        dropout=0.0,
        bidirectional=True,
        use_mamba=True,
    ):
        super().__init__()
        self.use_mamba = use_mamba and Mamba2 is not None
        self.bidirectional = bidirectional
        if self.use_mamba:
            self.forward_mixer = Mamba2(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=expand,
                headdim=headdim,
            )
            if bidirectional:
                self.backward_mixer = Mamba2(
                    d_model=d_model,
                    d_state=d_state,
                    d_conv=d_conv,
                    expand=expand,
                    headdim=headdim,
                )
        else:
            self.forward_mixer = ConvSequenceMixer(d_model, d_conv=d_conv, dropout=dropout)

    @property
    def is_mamba(self):
        return self.use_mamba

    def forward(self, x):
        if not self.use_mamba:
            return self.forward_mixer(x)
        y = self.forward_mixer(x)
        if self.bidirectional:
            y_back = self.backward_mixer(torch.flip(x, dims=[1]))
            y = 0.5 * (y + torch.flip(y_back, dims=[1]))
        return y


class LocalTransformerEncoder(nn.Module):
    def __init__(self, d_model, nhead=4, dropout=0.0, ffn_ratio=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=ffn_ratio * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=1)

    def forward(self, x, mask):
        key_padding_mask = None if mask is None else ~mask
        return self.encoder(x, src_key_padding_mask=key_padding_mask)


class AnchorMambaPoolingBlock(nn.Module):
    def __init__(
        self,
        stride=2,
        d_model=384,
        nhead=4,
        dropout=0.1,
        ffn_ratio=2,
        local_encode=False,
        pool_method="mean",
        mamba_headdim=64,
        mamba_dstate=64,
        mamba_expand=2,
        mamba_dconv=7,
        bidirectional=True,
        use_mamba=True,
    ):
        super().__init__()
        self.stride = stride
        self.local_encode = local_encode
        self.anchor_pooling = AnchorPooling(
            stride=stride, method=pool_method, d_model=d_model, nhead=1, dropout=dropout
        )

        self.global_encoder = MambaSequenceMixer(
            d_model=d_model,
            d_state=mamba_dstate,
            d_conv=mamba_dconv,
            expand=mamba_expand,
            headdim=mamba_headdim,
            dropout=dropout,
            bidirectional=bidirectional,
            use_mamba=use_mamba,
        )
        self.norm_global = RMSNorm(d_model)
        self.drop_path_global = LayerScale(d_model, dropout)
        self.gate1 = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())

        if local_encode:
            self.local_encoder = LocalTransformerEncoder(
                d_model=d_model, nhead=nhead, dropout=dropout, ffn_ratio=ffn_ratio
            )
            self.norm_local = RMSNorm(d_model)
            self.drop_path_local = LayerScale(d_model, dropout)
            self.gate2 = nn.Sequential(nn.Linear(2 * d_model, d_model), nn.Sigmoid())

        self.ffn = SwiGLUFFN(d_model, hidden_dim=ffn_ratio * d_model, dropout=dropout)
        self.norm_ffn = RMSNorm(d_model)
        self.drop_path_ffn = LayerScale(d_model, dropout)

    def _interleave_anchors(self, x, mask):
        bsz, dim, length = x.shape
        stride = self.stride
        num_blocks = (length + stride - 1) // stride
        padded_len = num_blocks * stride
        pad_len = padded_len - length
        if pad_len > 0:
            x = F.pad(x, (0, pad_len))
            mask = F.pad(mask, (0, pad_len), value=False)

        anchors = self.anchor_pooling(x)
        x_blocks = x.reshape(bsz, dim, num_blocks, stride)
        anchor_mask = _downsample_mask(mask, stride)
        token_mask = mask.reshape(bsz, 1, num_blocks, stride)

        mixed = torch.cat([anchors.unsqueeze(-1), x_blocks], dim=-1)
        mixed_mask = torch.cat([anchor_mask.unsqueeze(-1), token_mask], dim=-1)
        combined = mixed.permute(0, 2, 3, 1).reshape(bsz, num_blocks * (stride + 1), dim)
        combined_mask = mixed_mask.permute(0, 2, 3, 1).reshape(bsz, num_blocks * (stride + 1))
        return combined, combined_mask, anchor_mask, num_blocks

    def _extract_outputs(self, x, original_len, num_blocks):
        bsz, _, dim = x.shape
        stride = self.stride
        x = x.reshape(bsz, num_blocks, stride + 1, dim)
        anchor_out = x[:, :, 0].permute(0, 2, 1).contiguous()
        seq_out = x[:, :, 1:].reshape(bsz, num_blocks * stride, dim)
        seq_out = seq_out[:, :original_len].permute(0, 2, 1).contiguous()
        return anchor_out, seq_out

    def forward(self, x, mask=None):
        bsz, _, length = x.shape
        if mask is None:
            mask = torch.ones((bsz, 1, length), dtype=torch.bool, device=x.device)
        elif mask.ndim == 2:
            mask = mask.unsqueeze(1)
        mask = mask.bool()

        x = x.masked_fill(~mask, 0.0)
        combined, combined_mask, anchor_mask, num_blocks = self._interleave_anchors(x, mask)

        global_out = self.global_encoder(self.norm_global(combined))
        global_out = combined + self.drop_path_global(global_out)
        global_out = global_out.masked_fill(~combined_mask.unsqueeze(-1), 0.0)

        gate1 = self.gate1(torch.cat([combined, global_out], dim=-1))
        fused = gate1 * global_out + (1.0 - gate1) * combined

        if self.local_encode:
            local_out = self.local_encoder(self.norm_local(fused), combined_mask)
            local_out = fused + self.drop_path_local(local_out)
            local_out = local_out.masked_fill(~combined_mask.unsqueeze(-1), 0.0)
            gate2 = self.gate2(torch.cat([fused, local_out], dim=-1))
            fused = gate2 * local_out + (1.0 - gate2) * fused

        ffn_out = self.ffn(self.norm_ffn(fused))
        final_out = fused + self.drop_path_ffn(ffn_out)
        final_out = final_out.masked_fill(~combined_mask.unsqueeze(-1), 0.0)

        anchor_out, seq_out = self._extract_outputs(final_out, length, num_blocks)
        return anchor_out, seq_out, anchor_mask, mask


class HieraAMPBackbone(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        embd_dim=384,
        max_seq_len=2304,
        n_heads=4,
        stride=1,
        arch=(1, 0, 3),
        dropout=0.1,
        use_abs_pe=True,
        pool_method="mean",
        ffn_ratio=2,
        local_encode=False,
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
        assert len(arch) == 3, "arch must be (embed_convs, stem_layers, branch_layers)"
        assert arch[2] > 0, "AMP backbone needs at least one branch layer"
        self.max_seq_len = max_seq_len
        self.pyramid_fusion = pyramid_fusion

        self.embd_fc = MaskedConv1D(in_dim, embd_dim, 1)
        self.embd_convs = nn.ModuleList()
        self.embd_norms = nn.ModuleList()

        remaining_stride = stride
        for _ in range(arch[0]):
            conv_stride = 2 if remaining_stride > 1 else 1
            kernel_size = 5 if conv_stride > 1 else 3
            padding = 2 if conv_stride > 1 else 1
            self.embd_convs.append(
                MaskedConv1D(
                    embd_dim,
                    embd_dim,
                    kernel_size=kernel_size,
                    stride=conv_stride,
                    padding=padding,
                    bias=False,
                )
            )
            self.embd_norms.append(ChannelLayerNorm(embd_dim))
            remaining_stride = max(remaining_stride // 2, 1)

        self.stem = nn.ModuleList()
        for _ in range(arch[1]):
            self.stem.append(
                LocalTransformerEncoder(
                    d_model=embd_dim, nhead=n_heads, dropout=dropout, ffn_ratio=ffn_ratio
                )
            )

        branch_layers = arch[2]
        local_encode_num_layers = branch_layers if local_encode_num_layers == 0 else local_encode_num_layers
        self.branch = nn.ModuleList()
        for idx in range(branch_layers):
            self.branch.append(
                AnchorMambaPoolingBlock(
                    d_model=embd_dim,
                    stride=2,
                    nhead=n_heads,
                    dropout=dropout,
                    ffn_ratio=ffn_ratio,
                    local_encode=local_encode and idx < local_encode_num_layers,
                    pool_method=pool_method,
                    mamba_headdim=mamba_headdim,
                    mamba_dstate=mamba_dstate,
                    mamba_expand=mamba_expand,
                    mamba_dconv=mamba_dconv,
                    bidirectional=bidirectional,
                    use_mamba=use_mamba,
                )
            )

        if use_abs_pe:
            pe = _sinusoid_encoding(max_seq_len, embd_dim // 2) / (embd_dim ** 0.5)
            self.register_buffer("pe", pe, persistent=False)
        else:
            self.pe = None

        self.out_proj = nn.Conv1d(embd_dim, out_dim, kernel_size=1)
        if pyramid_fusion == "sum":
            self.pyramid_weights = nn.Parameter(torch.zeros(branch_layers))
        else:
            self.pyramid_weights = None

        self.apply(self._init_weights)

    @property
    def uses_mamba(self):
        return any(block.global_encoder.is_mamba for block in self.branch)

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Conv1d)):
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def _add_positional_encoding(self, x, mask):
        if self.pe is None:
            return x * mask.to(x.dtype)
        _, _, length = x.size()
        pe = self.pe.to(dtype=x.dtype, device=x.device)
        if length > pe.size(-1):
            pe = F.interpolate(pe.unsqueeze(0), size=length, mode="linear", align_corners=True)[0]
        return (x + pe[..., :length]) * mask.to(x.dtype)

    def _project_pyramid(self, tensors):
        return tuple(self.out_proj(t) for t in tensors)

    def _fuse_pyramid(self, sequence_fpn):
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

    def forward(self, x, mask):
        if x.ndim != 3:
            raise ValueError("Expected video features with shape (B, T, C)")
        if mask.ndim == 2:
            mask = mask.unsqueeze(1)
        mask = mask.bool()

        x = x.transpose(1, 2).contiguous()
        x, mask = self.embd_fc(x, mask)
        for conv, norm in zip(self.embd_convs, self.embd_norms):
            x, mask = conv(x, mask)
            x = F.relu(norm(x), inplace=True)

        x = self._add_positional_encoding(x, mask)
        for block in self.stem:
            stem_out = block(x.transpose(1, 2), mask.squeeze(1)).transpose(1, 2)
            x = stem_out * mask.to(stem_out.dtype)

        sequence_fpn = []
        sequence_masks = []
        anchor_fpn = []
        anchor_masks = []
        for idx, block in enumerate(self.branch):
            if idx == 0:
                anchor_out, seq_out, anchor_mask, seq_mask = block(x, mask)
            else:
                anchor_out, seq_out, anchor_mask, seq_mask = block(anchor_out, anchor_mask)
            sequence_fpn.append(seq_out)
            sequence_masks.append(seq_mask)
            anchor_fpn.append(anchor_out)
            anchor_masks.append(anchor_mask)

        sequence_fpn = self._project_pyramid(sequence_fpn)
        anchor_fpn = self._project_pyramid(anchor_fpn)
        fused = self._fuse_pyramid(sequence_fpn).transpose(1, 2).contiguous()
        sequence_fpn = tuple(feat.transpose(1, 2).contiguous() for feat in sequence_fpn)
        anchor_fpn = tuple(feat.transpose(1, 2).contiguous() for feat in anchor_fpn)
        sequence_masks = tuple(mask.squeeze(1).byte() for mask in sequence_masks)
        anchor_masks = tuple(mask.squeeze(1).byte() for mask in anchor_masks)
        return {
            "frames": fused,
            "sequence_fpn": sequence_fpn,
            "sequence_masks": sequence_masks,
            "anchor_fpn": anchor_fpn,
            "anchor_masks": anchor_masks,
        }
