# HieraMamba 前向传播流程

基于论文 CVPR 2026 Figure 2 (Overview of the HieraMamba Architecture) 梳理，将模型的完整前向传播分解为六个阶段，每个阶段标注对应代码位置。

---

## 总览：Early Fusion 路径

论文 Figure 2(a) 的数据流：

```
Video Features ──→ [Video Projection] ──→ [Fusion (Early)] ──→ [Multi-Scale Video Encoder] ──→ [Fusion (Late)] ──→ Moment Decoder ──→ (ts, te)
Text Features  ──→ [Text Encoder]     ──↗                                                  ──↗
```

对应代码入口：`libs/modeling/model.py:155-178`，`HieraMamba._forward_earlyfusion()`

---

## Stage 1：特征提取 (Feature Extraction)

**论文 Section 4.2 – Feature Extraction**

输入：预提取的视频 clip 特征 $V \in \mathbb{R}^{L_v \times D_v}$，文本特征 $Q \in \mathbb{R}^{L_q \times D_q}$。
这些特征来自 frozen 的视频/文本骨干网络（如 EgoVLP、CLIP），不在 HieraMamba 中端到端训练。

**代码**：`libs/data/dataset.py` — `VideoCentricDataset.__getitem__()` 负责从磁盘加载 `.npy` / `.pt` 特征文件。

---

## Stage 2：文本编码 (Text Encoder)

**论文 Figure 2(a) "Text Encoder" → Section 4.2 – Video and Text Encoders**

将输入文本特征通过 `text_net` 编码为上下文丰富的 query 嵌入 $E \in \mathbb{R}^{L_q \times D_q}$。

| 数据集 | 骨干 | 代码 |
|--------|------|------|
| Ego4D / MAD | `TextIdentity`（简单投影 + 可选注意力池化） | `libs/modeling/text_net.py:22-89` |
| TACoS | `TextTransformer`（Transformer 编码器 + GloVe tokenization） | `libs/modeling/text_net.py:92-188` |

**代码**：
```python
# libs/modeling/model.py:108-110
def encode_text(self, tokens, token_masks):
    text, text_masks = self.text_net(tokens, token_masks)
    return text, text_masks
```

---

## Stage 3：视频投影与早期融合 (Video Projection + Early Fusion)

**论文 Figure 2(a) "Optional" 虚线箭头 → Section 4.2 – Feature Extraction**

Early Fusion 模式下（Ego4D、MAD），在视频编码之前，先将文本信息注入视频特征：

1. **视频投影**：`vid_proj` 将原始视频特征投影到 embedding 维度
2. **跨注意力融合**：`fusion` 模块将文本信息注入视频序列

```python
# libs/modeling/model.py:167-171
vid_masks = vid_masks.unsqueeze(1)
vid, vid_masks = self.project_video(vid, vid_masks)
vid_fused, vid_masks_fused = self.fusion(vid, vid_masks, text, text_masks, text_size)
```

**融合模块**：`libs/modeling/fusion.py`
- `XAttNFusion`（`'xattn'`）：多层 `TransformerDecoder` 跨注意力 + AdaLN（第 16-78 行）
- `XAttNFusion2`（`'xattn2'`）：额外拼接全局文本池化表征（第 81-128 行）

---

## Stage 4：多尺度视频编码器 (Multi-Scale Video Encoder)

**论文 Figure 2(a) "Multi-Scale Video Encoder" → Figure 2(b) "AMP Block" → Section 4.2–4.3**

这是 HieraMamba 的核心。由 `HieraMambaBackbone` 构建特征金字塔 $\mathcal{V}_{pyr} = \{\tilde{V}^{(0)}, \tilde{V}^{(1)}, ..., \tilde{V}^{(L-1)}\}$。

### 4.1 Embedding Convolutions

```python
# libs/modeling/video_net.py:150-155
x, _ = self.embd_fc(x, mask)
for conv, norm in zip(self.embd_convs, self.embd_norms):
    x, mask = conv(x, mask)
    x = F.relu(norm(x), inplace=True)
```

通过 1D 卷积逐步下采样输入序列，为 AMP 层做好准备。

### 4.2 AMP Block 堆叠（论文核心贡献）

**代码**：`libs/modeling/video_net.py:178-191`（backbone 堆叠）+ `libs/modeling/anchor_mamba.py:330-476`（单个 AMP 块）

每个 AMP 块接收上一层的 anchor stream，输出精炼后的 anchor stream 和 sequence stream，形成层级金字塔。

```python
# libs/modeling/video_net.py:178-191 — backbone 循环
for i, block in enumerate(self.branch):
    if i == 0:
        anchor_out, x_out, anchor_mask, x_mask = block(x, mask)
    else:
        anchor_out, x_out, anchor_mask, x_mask = block(anchor_out, anchor_mask)
    fpn += (x_out,)
    anchor_fpn += (anchor_out,)
```

### 4.3 单个 AMP Block 内部流程（论文 Figure 2(b)）

以 `AnchorMambaPoolingBlockGated`（`libs/modeling/anchor_mamba.py:330-476`）为例：

```
输入 x: (B, D, L)
    │
    ▼
① Generate & Interleave Anchors（论文 Figure 2(b) 底部）
    │  每 s 帧生成一个 anchor token，交错排列
    │  [a₀, t₀, t₁, a₁, t₂, t₃, ...]
    │
    ▼
② Global Encoding（论文 Figure 2(b) "Global"）
    │  RMSNorm → Hydra/Mamba2 双向扫描 → 残差连接
    │
    ▼
③ Gate 1 Fusion（论文 Figure 2(b) σ 符号）
    │  σ(W₁·[x_combined; global_out]) → 自适应融合原始输入和全局编码
    │
    ▼
④ Local Encoding（论文 Figure 2(b) "Local"，可选）
    │  RMSNorm → 窗口 Transformer（local_window_size=5）→ 残差
    │
    ▼
⑤ Gate 2 Fusion（仅在启用 Local 时）
    │
    ▼
⑥ FFN（论文 Figure 2(b) "FFN"）
    │  RMSNorm → SwiGLU FFN → 残差
    │
    ▼
⑦ Extract anchor/sequence outputs（论文 Figure 2(b) 顶部）
    │  从交错序列中分离 anchor stream 和 sequence stream
    │
    ▼
输出: anchor_out (B, D, L/s), seq_out (B, D, L), anchor_mask, mask
```

**对应代码**（`anchor_mamba.py:435-476`，`AnchorMambaPoolingBlockGated.forward`）：

```python
# ① 生成并交错 anchor
x_combined, anchor_positions, expanded_mask, anchor_mask = \
    self._generate_and_interleave_anchors(x, mask)  # 第85-145行

# ② 全局编码（Hydra 双向 SSM）
global_out = self.global_encoder(self.norm_global(x_combined))
global_out = self.drop_path_global(global_out) + x_combined  # 残差

# ③ 第一个门控融合
gate1_weights = self.gate1(torch.cat([x_combined, global_out], dim=-1))
fusion1_out = gate1_weights * global_out + (1 - gate1_weights) * x_combined

# ④ ⑤ 可选局部编码 + 第二个门控融合
if self.local_encode:
    local_out = self.local_encoder(...)
    gate2_weights = self.gate2(torch.cat([fusion1_out, local_out], dim=-1))
    fused = gate2_weights * local_out + (1 - gate2_weights) * fusion1_out

# ⑥ FFN
final_out = self.ffn(self.norm_ffn(fused)) + fused

# ⑦ 分离输出
anchor_out, seq_out = self._extract_anchor_and_sequence_outputs(final_out, ...)
```

### 4.4 Anchor 生成与交错（论文 Section 4.3.1）

**代码**：`anchor_mamba.py:85-145`（`_generate_and_interleave_anchors`）

论文核心公式：$\tilde{V} = [a_0, v_0, v_1, ..., v_{s-1}, a_1, v_s, ...]$，将每个 anchor token 放在其对应的 $s$ 个帧之前。

锚点通过 `AnchorPooling` 模块生成（`anchor_mamba.py:689-747`），支持四种池化方式：
- `MeanPooling`（默认）
- `MaxPooling`
- `AttnPooling`（CLIP 风格可学习查询）
- `GatedPooling`（自适应混合 mean/max）

---

## Stage 5：晚期融合 (Fusion + Moment Decoder)

**论文 Figure 2(a) "Fusion" → "Moment Decoder" → Section 4.2 – Fusion and Decoding**

多尺度特征金字塔 $\mathcal{V}_{pyr}$ 与文本嵌入 $E$ 一起送入跨注意力融合模块，然后经过分类和回归头预测时间戳。

```python
# libs/modeling/model.py:127-131
def fuse_and_predict(self, fpn, fpn_masks, text, text_masks, text_size=None):
    fpn, fpn_masks_fusion = self.fusion(fpn, fpn_masks, text, text_masks, text_size)
    fpn_logits, _ = self.cls_head(fpn, fpn_masks_fusion)
    fpn_offsets, fpn_masks = self.reg_head(fpn, fpn_masks_fusion)
    return fpn_logits, fpn_logits, fpn_offsets, fpn_masks
```

### 5.1 跨注意力融合

`fusion` 对每个 FPN 层的视频特征做跨注意力，以文本为 key/value：
- **代码**：`libs/modeling/fusion.py:56-66`（`XAttNFusion._forward`）
- 内部使用 `TransformerDecoder`（`libs/modeling/blocks.py:644`），支持 AdaLN 模式

### 5.2 Moment Decoder（分类头 + 回归头）

论文将这个部分称为 "lightweight decoder [75]"（参考 ActionFormer），即：

- **ClsHead**：多个 1D Conv → 输出每个候选点的置信度分数 $\in \mathbb{R}^{(bs, p)}$
  - 代码：`libs/modeling/head.py:18-69`
- **RegHead**：多个 1D Conv → 输出每个正样本点的边界偏移 $(\Delta_{start}, \Delta_{end}) \in \mathbb{R}^{(bs, p, 2)}$
  - 代码：`libs/modeling/head.py:72-116`

```python
# head.py:58-68 (ClsHead.forward)
for conv, norm in zip(self.convs, self.norms):
    x, _ = conv(x, mask)
    x = F.relu(norm(x), inplace=True)
logits, _ = self.cls_head(x, mask)  # (bs, 1, p) → squeeze → (bs, p)
```

---

## Stage 6：辅助损失（训练时）

**论文 Figure 2(c) "ACC and SPC Loss" → Section 4.4**

训练时由 `TrainerAuxiliary` 额外计算两个对比损失，引导层级特征学习紧凑且有判别力的表征：

### 6.1 Anchor-Conditioned Contrastive (ACC) Loss

**论文 Section 4.4 – Anchor-Conditioned Contrastive (ACC) Loss**

让每个 anchor 拉近其覆盖的帧 token（紧凑性），推远远距离 anchor 的帧（区分性）。

- **代码**：`libs/modeling/losses.py:16-68`（`MultiScaleMaskedContrastive`）
- 核心计算：`libs/modeling/contrastive_losses.py`（`contrastive_subsample_negative_mp`，多正样本 InfoNCE）

```python
# worker.py:800-807 — TrainerAuxiliary 中的调用
ds_contrastive_loss = self.ds_contrastive_loss(
    fpn, sequence_fpn_masks, anchor_fpn, anchor_fpn_masks
)
```

### 6.2 Segment-Pooled Contrastive (SPC) Loss

**论文 Section 4.4 – Segment-Pooled Contrastive (SPC) Loss**

将每个 GT 区间内的帧 token 池化为一个段原型，拉近同区间 token，推远区间外 token。

- **代码**：`libs/modeling/losses.py:71-171`（`MultiScaleMaskedGTPointContrastive`）

```python
# worker.py:834-836 — TrainerAuxiliary 中的调用
gt_contrastive_loss = self.gt_contrastive_loss(
    fpn_expanded, fpn_masks_split, gt_labels_split, gt_labels_span
)
```

### 6.3 总损失

$$\mathcal{L} = \mathcal{L}_{cls} + \lambda_{reg} \mathcal{L}_{reg} + \lambda_{ACC} \mathcal{L}_{ACC} + \lambda_{SPC} \mathcal{L}_{SPC}$$

**代码**：`libs/worker.py:866-868`
```python
total_loss = cls_loss + self.loss_weight * reg_loss + \
    self.ds_contrastive_weight * ds_contrastive_loss + \
    self.gt_contrastive_weight * gt_contrastive_loss
```

---

## 推理时流程（Eval）

**代码**：`libs/worker.py:900-1542`（`EvaluatorOriginal` / `EvaluatorAuxiliary`）

推理时额外步骤：
1. **滑动窗口**：将长视频切分为重叠窗口逐个推理（`worker.py:1093-1116`）
2. **候选点生成**：`PtGenerator` 在每个 FPN 层生成候选点及其回归范围（`libs/modeling/model.py:196-271`）
3. **NMS**：非极大值抑制去除冗余预测（`libs/nms/` C++ 扩展）
4. **时间戳转换**：将模型输出的归一化坐标转换为实际秒数（`worker.py:1183-1192`）
5. **Recall@k@IoU 评估**：与 GT 比较计算指标

---

## 关键文件索引

| 模块 | 文件路径 |
|------|----------|
| 顶层模型定义 | `libs/modeling/model.py` |
| HieraMamba Backbone | `libs/modeling/video_net.py` |
| AMP Block（论文核心） | `libs/modeling/anchor_mamba.py` |
| 文本编码器 | `libs/modeling/text_net.py` |
| 跨注意力融合 | `libs/modeling/fusion.py` |
| 分类/回归头 | `libs/modeling/head.py` |
| 基础模块（Conv/Attn/FFN） | `libs/modeling/blocks.py` |
| 对比损失 | `libs/modeling/losses.py`, `libs/modeling/contrastive_losses.py` |
| 训练/评估循环 | `libs/worker.py` |
| 数据加载 | `libs/data/dataset.py` |
| 配置加载 | `libs/core/opt.py` |
| Hydra 双向 SSM（子模块） | `hydra/modules/hydra.py` |
