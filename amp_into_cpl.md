# 方案 B：Anchor-aware Gaussian Proposal — AMP 融入 CPL 架构设计

## 1. 动机与核心洞察

### 1.1 两个模块的本质同构性

CPL 的 Gaussian Proposal 和 HieraMamba 的 AMP block 本质上在解决同一个问题：**从连续视频时序中提取紧凑的、语义集中的表示**。但它们的实现机制截然不同：

| 维度 | CPL Gaussian Proposal | AMP Anchor |
|------|----------------------|------------|
| 压缩方式 | 软加权（高斯分布对帧特征做 weighted sum） | 硬池化（avg pooling 将 stride 帧压缩为 1 个 anchor） |
| 位置选择 | 语义驱动：中心和宽度由 query 条件解码，可任意位置 | 结构驱动：等间隔 stride 划分，固定位置 |
| 时序建模 | 无：依赖上游 Transformer 编码的帧特征 | 有：Mamba 双向扫描 + 局部 Transformer 窗口编码 |
| 复杂度 | O(T²)（Transformer self-attention） | O(T)（Mamba 线性扫描） |
| 层级性 | 无：所有 proposal 在同一分辨率上竞争 | 有：多层 AMP 块逐级压缩，形成特征金字塔 |

两者的核心张力在于：**CPL 的 proposal 语义灵活但计算昂贵，AMP 的 anchor 高效但位置刚性**。方案 B 的目标就是让两者协同——用 anchor 提供结构性先验，用 Gaussian 提供语义灵活性。

### 1.2 融合后的系统定位

融合后的模型仍是一个**弱监督时序定位**模型（训练时无 GT 时序边界标注），但同时获得：

- AMP 的**线性复杂度**层级视频编码（解决 CPL 在长视频上 Transformer 退化的问题）
- AMP 的**多尺度特征金字塔**（让 proposal 在不同粒度上操作）
- CPL 的**可学习高斯 proposal**（保持 query-conditioned 的精确定位能力）
- CPL 的**Easy-to-Hard 负样本挖掘**（保持训练稳定性）

---

## 2. 架构总览

### 2.1 CPL 原始架构

```
frames ──→ frame_fc (Linear) ──→ DualTransformer ──→ Gaussian params (center, width)
words  ──→ word_fc  (Linear) ──↗                           │
                                                            ▼
                                              Gaussian masking (下采样4x后加权)
                                                            │
                                                            ▼
                                              Masked Transformer ──→ 重建 words
```

### 2.2 融合后的架构

```
frames ──→ AMP Backbone ──→ 特征金字塔 V_pyr ──┬──→ Ṽ⁽⁰⁾ (finest)  ──→ Gaussian masking ──→ 重建 words
            (L 层 AMP 块)       │              ├──→ Ṽ⁽¹⁾ (medium)  ──↗
                                │              ├──→ Ṽ⁽²⁾ (coarse)  ──↗ (cross-attn 补充)
                                │              └──→ Anchor 先验 ──→ 门控融合 ──→ Gaussian params
words  ──→ word_fc (Linear) ──↗
```

---

## 3. 模块级设计

### 3.1 AMP Backbone 替换 frame_fc

**现状**：CPL 的 `frame_fc` 是一个无状态的线性层，不改变时序长度，不引入跨帧交互。

**替换为**：HieraMamba 的 `HieraMambaBackbone`，由嵌入投影 + 嵌入卷积 + L 层 AMP 块组成。

**输入输出变化**：

```
原始:  frames_feat ∈ R^(B, T, input_size)  →  frame_fc  →  R^(B, T, hidden_size)
融合后: clip_feats ∈ R^(B, D_clip, T)      →  Backbone  →  V_pyr = {Ṽ⁽⁰⁾, Ṽ⁽¹⁾, ..., Ṽ⁽⁽⁻¹⁾}
                                                               其中 Ṽ⁽ˡ⁾ ∈ R^(B, embd_dim, T/2^l)
```

**关键决策**：

1. **输入特征来源**：CPL 原始直接接收预提取的 3D CNN 帧特征（如 C3D），AMP backbone 也接收类似的 clip 特征。因此输入格式兼容，只需对齐维度。

2. **backbone 预训练**：使用 HieraMamba 在 Ego4D/MAD 上预训练的权重初始化 AMP backbone。这些权重已经学会了有效的层级视频编码。

3. **backbone 是否冻结**：分阶段处理（详见第 5 节）。初期冻结以稳定训练，后期解冻浅层以适应目标域。

4. **金字塔输出维度对齐**：AMP backbone 的 `embd_dim`（默认 384）需要与 CPL 的 `hidden_size`（默认 256）对齐。通过在 backbone 输出端添加投影层实现。

### 3.2 Anchor 先验生成

这是方案 B 的核心新增模块。目标是从 AMP 金字塔的 anchor tokens 中提取 proposal 参数的先验信息。

#### 3.2.1 为什么需要 Anchor 先验

CPL 原始的 Gaussian 参数完全由 Transformer 解码器预测。在长视频上，Transformer 的 self-attention 难以捕捉远距离结构信息，导致：

- 预测的高斯中心偏移（定位不准）
- 预测的高斯宽度不合理（过宽覆盖无关内容，过窄遗漏目标）

AMP 的 anchor tokens 经过 Mamba 的全局双向扫描，已经编码了全序列的结构信息。这些 anchor 本身隐含了"哪些时间区域包含不同语义事件"的线索，可以作为 proposal 参数的结构性补充。

#### 3.2.2 先验提取过程

对金字塔的每一层 anchor 做全局池化，得到该层对视频整体结构的理解：

```
每层 anchor → 全局池化 → 该层的"视频结构摘要向量"
```

将所有层的摘要向量融合（如平均池化或注意力加权），得到一个综合的多尺度先验表示。然后通过线性投影映射为 proposal 参数（center, width）的候选值。

#### 3.2.3 先验的含义

生成的 anchor 先验可以理解为：**不看 query，仅凭视频本身的结构，哪些区域最可能包含有意义的事件**。这是一种无条件的 proposal 候选，与 Transformer 解码器生成的条件 proposal（看了 query 后的预测）形成互补。

### 3.3 门控融合机制

Transformer 解码器预测的 Gaussian 参数与 anchor 先验不能直接相加，需要一个输入依赖的门控来动态平衡两者的贡献。

#### 3.3.1 门控的设计原则

门控的 sigmoid 值应该根据以下因素动态调整：

- **视频长度**：长视频上 Mamba 的全局扫描更可靠，门控应偏向 anchor 先验；短视频上 Transformer self-attention 足够，门控应偏向 Transformer 预测
- **Query 复杂度**：简单 query（如"跑步"）的 proposal 位置比较确定，anchor 先验即可提供好的候选；复杂 query（如"一个人先拿起杯子然后走向窗边"）需要 Transformer 的语义理解
- **当前训练阶段**：训练初期 Transformer 预测不准，应更多依赖 anchor 先验；训练后期 Transformer 已经学会 query-conditioned 定位，应更多信任自己的预测

#### 3.3.2 融合后的 Gaussian 参数

最终的 Gaussian center 和 width 由门控在两个来源之间插值得到，再经过 sigmoid 归一化到 [0, 1] 范围。这保证了无论门控如何偏向，输出始终是合法的高斯参数。

### 3.4 Gaussian Masking 在 AMP 特征上的操作

#### 3.4.1 下采样的重新考虑

CPL 原始在 Gaussian masking 前先做 4 倍均匀下采样（`n_frames // 4`），纯粹为了减少计算量。引入 AMP backbone 后：

- AMP 金字塔的 `Ṽ⁽¹⁾` 已经是 T/2 长度，`Ṽ⁽²⁾` 是 T/4 长度
- 但这些层的特征经过了 Mamba 编码和 FFN 精炼，与原始帧特征的分布不同
- Gaussian masking 的核心操作是"用高斯权重对特征做加权求和"，这要求特征保留精确的时序对齐关系

**决策**：在 `Ṽ⁽⁰⁾`（最精细层）上做 Gaussian masking。理由是 `Ṽ⁽⁰⁾` 保留了原始时序分辨率和精确的边界信息，虽然经过了 AMP 块的编码，但序列流的输出与输入保持相同长度。同时利用更高层的特征作为 cross-attention 的补充上下文。

#### 3.4.2 双粒度加权

Gaussian weighting 可以在两种粒度上同时进行：

- **Token 级加权**：对 `Ṽ⁽⁰⁾` 的每个 token 按高斯权重加权，得到精细的 proposal 表征
- **Anchor 级加权**：对最接近高斯区域的 anchor tokens 按高斯权重加权，得到窗口级的 proposal 表征

两种粒度的加权结果拼接后通过投影层融合。这样 proposal 表征既有时序精度（来自 token 级），又有语义概括（来自 anchor 级）。

#### 3.4.3 对下游语义补全的影响

Gaussian 加权后的 proposal 表征被送入 Masked Transformer 重建被 masked 的 words。AMP 特征的引入意味着：

- 每个 token 已经携带全局上下文（Mamba 双向扫描）和局部精炼（Transformer 窗口编码）
- proposal 表征的信息密度更高，重建准确率应该提升
- 而重建准确率是 CPL 选择最终 proposal 的依据（loss-based strategy），因此形成正反馈循环

---

## 4. 负样本挖掘的变化

### 4.1 原始机制回顾

CPL 在同一视频内生成两个负 Gaussian proposal（左侧和右侧），训练初期让它们远离正 proposal（容易区分），训练后期逐渐逼近（越来越难）。这模拟了课程学习。

### 4.2 AMP 引入后的增强

#### 4.2.1 Anchor 边界作为离散化约束

AMP 的等间隔 anchor 划分天然定义了视频的"结构网格"。负 proposal 逼近正 proposal 时，可以利用 anchor 边界将连续的逼近过程离散化为沿 anchor 网格的步进。

具体来说：负 proposal 的高斯中心不应该连续滑动，而应该每次跳到相邻的 anchor 对应的时间窗口。这将搜索空间从连续的 [0, 1] 降低为离散的 anchor 索引，降低了训练的方差。

#### 4.2.2 负 proposal 的特征复用

原始 CPL 中，正 proposal 和负 proposal 在相同的帧特征上分别做高斯加权，计算是冗余的。引入 AMP 后，由于 anchor 已经是窗口的紧凑摘要，位于正 proposal 附近的负 proposal 可以直接引用附近的 anchor tokens，减少重复计算。

#### 4.2.3 跨尺度负样本

原始 CPL 只在同一分辨率上生成负 proposal。AMP 金字塔提供了新的可能性：在更粗粒度的层（如 `Ṽ⁽¹⁾` 或 `Ṽ⁽²⁾`）上生成负 proposal，让模型学会区分不同尺度的语义区域。这可以作为额外的辅助训练信号。

---

## 5. 训练策略

### 5.1 分阶段训练

AMP backbone 的参数量显著大于 CPL 原始的 `frame_fc`。从随机初始化端到端训练会导致不稳定。

#### 阶段一：Backbone 冻结，训练 Proposal 模块

- 冻结 AMP backbone 的所有参数（使用 HieraMamba 预训练权重）
- 只训练：Gaussian 参数预测头（Transformer decoder + 新增的 anchor 先验投影 + 门控）
- 训练 CPL 原有的三个损失：`L_rec`、`L_IVC`、`L_div`
- 目标：让 Gaussian proposal 机制适应 AMP backbone 的特征分布

#### 阶段二：浅层解冻，联合微调

- 解冻 AMP backbone 的前 1-2 层 AMP 块
- 保持深层 AMP 块和 Hydra/Mamba 参数冻结
- 所有模块联合训练
- 目标：让 backbone 的浅层特征适应目标数据集（Charades-STA / ActivityNet）的分布

#### 阶段三（可选）：全模型微调

- 解冻所有参数，使用较小的学习率
- 仅在目标数据集足够大时进行，否则有过拟合风险

### 5.2 学习率差异化

不同模块应使用不同的学习率：

| 模块 | 学习率策略 | 理由 |
|------|-----------|------|
| Mamba/SSM 参数 | 最小（如 1e-5） | SSM 状态更新对学习率敏感，过大导致训练不稳定 |
| Hydra 双向扫描 | 中等（如 5e-5） | 需要适应新的双向混合模式 |
| Transformer 局部编码 | 中等（如 5e-5） | 标准 Transformer 参数 |
| Gaussian 参数预测头 | 较大（如 1e-4） | 从零训练，需要较快收敛 |
| Anchor 先验投影 | 较大（如 1e-4） | 新增模块，从零训练 |
| 门控融合 | 较大（如 1e-4） | 新增模块，从零训练 |
| Word/Frame FC | 较大（如 1e-4） | CPL 原有模块 |

### 5.3 Curriculum Learning 的层级扩展

AMP 的层级结构天然支持训练课程：

1. **初期**：仅使用 `Ṽ⁽⁰⁾`（最精细层）做 Gaussian masking 和 proposal 生成。模型先学会在原始分辨率上定位。
2. **中期**：引入 `Ṽ⁽¹⁾` 作为 cross-attention 的补充上下文。模型开始利用粗粒度全局信息。
3. **后期**：引入全部金字塔层和 anchor 先验。模型完全利用多尺度信息。

这种层级课程与 CPL 已有的 Easy-to-Hard 负样本课程正交，可以叠加使用。

---

## 6. 损失函数设计

### 6.1 保留 CPL 原有损失

以下三个损失保持不变，作用于 proposal 层面：

- **`L_rec`（重建损失）**：衡量 Gaussian-weighted proposal 重建 masked words 的能力。融合后由于 proposal 表征更丰富，`L_rec` 应该能提供更强的梯度信号。
- **`L_IVC`（视频内对比损失）**：确保最佳 proposal 的重建损失低于 reference（全视频）和负 proposal。与 AMP 无关，保持不变。
- **`L_div`（多样性损失）**：确保多个 Gaussian proposal 之间有足够的差异。可以引入 anchor 信息来增强——如果两个 Gaussian proposal 落在同一个 anchor 窗口内，额外施加多样性惩罚。

### 6.2 可选引入 AMP 自监督损失

#### 6.2.1 L_ACC（Anchor-Conditioned Contrastive）

`L_ACC` 要求每个 anchor 与其时间窗口内的帧 token 相似（紧凑性），与远处 anchor 不同（区分性）。

**适用条件**：仅在 AMP backbone 可训练（非冻结）时可用。如果 backbone 冻结，anchor 的表示固定，`L_ACC` 的梯度无法回传。

**集成方式**：作为辅助损失加入总损失，权重较小（如 0.1），避免干扰 CPL 的主损失。

#### 6.2.2 L_SPC 的近似

原始 `L_SPC` 需要 GT 时序区间，在弱监督设定下不可用。但可以用 CPL 的最佳 proposal 近似 GT：

- 选择重建损失最小的 proposal，其 Gaussian 区间作为"伪 GT"
- 在这个伪 GT 区间内聚合帧 token 为 segment prototype
- 让 prototype 与区间内 token 做对比，与区间外 token 排斥

这种近似在训练后期（proposal 质量较高时）比较可靠，可以作为一种自训练信号。训练初期不建议使用，因为伪 GT 不准确会引入噪声。

### 6.3 总损失

```
L_total = L_rec + L_IVC + λ_div * L_div + λ_acc * L_ACC (可选) + λ_spc * L_SPC_approx (可选)
```

其中 `λ_acc` 和 `λ_spc_approx` 在训练初期设为 0，后期逐渐增大。

---

## 7. 推理流程

### 7.1 推理时的数据流

```
1. 输入: frames_feat, words_id
2. AMP Backbone → 特征金字塔 V_pyr
3. Anchor 先验提取 → anchor_prior (num_props × 2)
4. Transformer Decoder → transformer_prior (num_props × 2)
5. 门控融合 → final_gauss_param (num_props × 2) → sigmoid → (center, width)
6. Gaussian masking on Ṽ⁽⁰⁾ → K 个 proposal 表征
7. 重建 words → K 组 words_logit
8. 选择策略 (loss-based 或 vote-based) → 最终预测的 (center, width)
9. 反归一化 → (start_time, end_time)
```

### 7.2 推理效率对比

设视频长度为 T：

| 步骤 | CPL 原始 | 融合后 |
|------|---------|--------|
| 视频编码 | O(T)（Linear） | O(T)（Mamba 线性扫描） |
| 时序建模 | O(T²)（Transformer self-attn） | O(T)（Mamba）+ O(T·W)（局部窗口） |
| Proposal 生成 | O(T²)（Decoder cross-attn） | O(T²)（不变，但 T 因 backbone 下采样可更小） |
| Gaussian masking | O(T) | O(T)（在 Ṽ⁽⁰⁾ 上） |
| 语义补全 | O(T × Q) | O(T × Q)（不变） |

主要收益来自编码和时序建模阶段的复杂度降低。对于长视频（T 很大时），提升显著。

---

## 8. 关键技术风险与缓解

| 风险 | 描述 | 缓解措施 |
|------|------|----------|
| 特征分布偏移 | AMP backbone 的输出分布与 CPL 原始 `frame_fc` 差异大，导致 Gaussian 参数预测初始值远离最优 | 阶段一冻结 backbone 训练，让 proposal 模块先适应 |
| Mamba CUDA 依赖 | Mamba 的 selective scan kernel 仅支持 compute capability ≥ 8.0（Ampere+） | 在文档中明确 GPU 要求；对不支持的 GPU 回退到单向 Mamba |
| 训练不稳定 | Mamba 状态更新对学习率和初始化敏感 | 使用极小学习率（1e-5）微调 Mamba 参数；使用 gradient clipping |
| 过拟合 | AMP backbone 参数量大，在小数据集（如 Charades-STA）上可能过拟合 | 阶段一只解冻 proposal 模块；使用 dropout 和 weight decay；考虑冻结更多层 |
| Gaussian 与 Anchor 竞争 | 门控融合不当可能导致两个来源互相干扰 | 初始将门控偏向 Transformer 预测（anchor 先验仅作为正则化），随训练推进逐步放开 |

---

## 9. 评估计划

### 9.1 消融实验

| 实验 | 变量 | 目的 |
|------|------|------|
| Baseline | 原始 CPL | 确认复现 |
| +AMP backbone | 用 AMP 替换 frame_fc，冻结 backbone | 验证 AMP 特征本身的价值 |
| +Anchor 先验 | 加入 anchor 先验生成和门控融合 | 验证先验对 Gaussian 参数预测的改善 |
| +解冻浅层 | 解冻前 2 层 AMP 块 | 验证微调 backbone 的增量收益 |
| +层级课程 | 分阶段引入金字塔层 | 验证训练课程的效果 |
| +L_ACC | 加入 anchor 对比损失 | 验证自监督辅助损失的价值 |
| +跨尺度负样本 | 在粗粒度层生成负 proposal | 验证多尺度负样本的效果 |

### 9.2 评估指标

在 Charades-STA 和 ActivityNet Captions 上评估：

- **R@1, IoU={0.3, 0.5, 0.7}**：top-1 预测在不同 IoU 阈值下的召回率
- **R@5, IoU={0.3, 0.5, 0.7}**：top-5 预测的召回率
- **mIoU**：平均 IoU
- **FLOPs**：单次前向传播的计算量（验证效率提升）
- **推理速度**：实际 FPS（验证工程效率）

---

## 10. 总结

方案 B 的核心是 **Anchor-aware Gaussian Proposal**，通过三个关键设计将 AMP 融入 CPL：

1. **AMP Backbone 替换帧编码**：获得线性复杂度的层级视频特征，解决长视频 Transformer 退化问题
2. **Anchor 先验 + 门控融合**：让 AMP 的结构性先验调制 CPL 的 Gaussian 参数预测，提升 proposal 质量
3. **双粒度 Gaussian masking**：在 token 级和 anchor 级同时做高斯加权，兼顾时序精度和语义概括

整个设计保持了 CPL 的弱监督设定和端到端可训练性，同时引入了 AMP 的效率优势和层级表示能力。预期在长视频基准（ActivityNet）上获得最显著的提升。
