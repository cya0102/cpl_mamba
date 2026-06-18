import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.transformer import DualTransformer
from models.modules.amp_backbone import HieraAMPBackbone

class CPL(nn.Module):
    """
    CPL 主模型。

    原始 CPL 流程:
        video frames -> frame_fc -> DualTransformer -> Gaussian proposals
        words        -> word_fc  -> masked semantic completion

    AMP 打开后:
        video frames -> HieraAMPBackbone -> 多尺度 sequence/anchor 特征
        sequence 特征替代原 frame_fc 输出参与 CPL 的 Gaussian proposal 与语义补全；
        anchor 特征额外生成无 query 条件的视频结构先验，再通过 gate 与 CPL 原本
        query-conditioned Gaussian logits 融合。
    """
    def __init__(self, config):
        super().__init__()
        self.dropout = config['dropout']
        self.vocab_size = config['vocab_size']
        self.sigma = config["sigma"]
        self.use_negative = config['use_negative']
        self.num_props = config['num_props']
        self.max_epoch = config['max_epoch']
        self.gamma = config['gamma']
        self.hidden_size = config['hidden_size']

        self.amp_config = config.get('AMP', {})
        self.use_amp = self.amp_config.get('enabled', False)
        self.use_anchor_prior = self.use_amp and self.amp_config.get('use_anchor_prior', True)
        self.amp_proposal_source = self.amp_config.get('proposal_source', 'anchor')
        self.amp_proposal_anchor_scale = int(self.amp_config.get('proposal_anchor_scale', 1))

        # Baseline CPL 使用 frame_fc 投影原始帧特征；AMP 分支仍复用它投影 pred token。
        self.frame_fc = nn.Linear(config['frames_input_size'], self.hidden_size)
        if self.use_amp:
            backbone_config = dict(self.amp_config.get('backbone', {}))
            backbone_dropout = backbone_config.pop('dropout', self.dropout)
            # AMP backbone 输出 hidden_size 维的视频 token，保证下游 DualTransformer 不改接口。
            self.amp_backbone = HieraAMPBackbone(
                in_dim=config['frames_input_size'],
                out_dim=self.hidden_size,
                dropout=backbone_dropout,
                **backbone_config
            )
            if self.amp_config.get('freeze_backbone', False):
                for param in self.amp_backbone.parameters():
                    param.requires_grad = False

        self.word_fc = nn.Linear(config['words_input_size'], self.hidden_size)
        self.mask_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.start_vec = nn.Parameter(torch.zeros(config['words_input_size']).float(), requires_grad=True)
        self.pred_vec = nn.Parameter(torch.zeros(config['frames_input_size']).float(), requires_grad=True)

        self.trans = DualTransformer(**config['DualTransformer'])
        self.fc_comp = nn.Linear(self.hidden_size, self.vocab_size)
        self.fc_gauss = nn.Linear(self.hidden_size, self.num_props*2)
        if self.use_anchor_prior:
            # anchor_prior: 仅从视频 anchor 摘要预测 proposal 参数，提供不依赖 query 的结构先验。
            self.fc_anchor_prior = nn.Sequential(
                nn.LayerNorm(self.hidden_size),
                nn.Linear(self.hidden_size, self.hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(self.dropout),
                nn.Linear(self.hidden_size, self.num_props*2),
            )
            # anchor_gate: 根据 query 表征、anchor 摘要和视频长度，动态决定信任 CPL 预测还是 anchor 先验。
            self.fc_anchor_gate = nn.Linear(self.hidden_size * 2 + 1, self.num_props*2)
            nn.init.constant_(self.fc_anchor_gate.bias, self.amp_config.get('gate_bias', 2.0))
 
        self.word_pos_encoder = SinusoidalPositionalEmbedding(self.hidden_size, 0, 20)

        pretrained_path = self.amp_config.get('pretrained_path') if self.use_amp else None
        if pretrained_path:
            self._load_amp_pretrained(pretrained_path)

    def forward(self, frames_feat, frames_len, words_id, words_feat, words_len, weights, **kwargs):
        bsz, n_frames, _ = frames_feat.shape
        amp_out = None
        if self.use_amp:
            # 1) AMP 视频编码:
            #    输入 (B, T, C_in)，输出 finest-level sequence token (B, T', hidden)
            #    以及多尺度 anchor_fpn/anchor_masks，供 proposal 先验使用。
            frames_feat = F.dropout(frames_feat, self.dropout, self.training)
            frames_mask = _generate_mask(frames_feat, frames_len).bool()
            amp_out = self.amp_backbone(frames_feat, frames_mask)
            frames_feat = amp_out['frames']
            props_source_len = frames_feat.size(1)

            # 2) 保留 CPL 原设计中的 pred token。它不属于真实视频帧，mask 置 0；
            #    DualTransformer 仍可在最后位置产生 proposal 查询状态 h[:, -1]。
            pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
            pred_vec = F.dropout(pred_vec, self.dropout, self.training)
            pred_vec = self.frame_fc(pred_vec)
            frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
            frames_mask = amp_out['sequence_masks'][0].to(frames_feat.device)
            if frames_mask.size(1) != props_source_len:
                frame_mask_len = self._scale_lengths(frames_len, n_frames, props_source_len)
                frames_mask = _generate_mask(frames_feat[:, :props_source_len], frame_mask_len)
            frames_mask = torch.cat([frames_mask, frames_mask.new_zeros(bsz, 1)], dim=1)
        else:
            # Baseline CPL 路径保持原行为: 拼接 pred token 后做 dropout + frame_fc。
            pred_vec = self.pred_vec.view(1, 1, -1).expand(bsz, 1, -1)
            frames_feat = torch.cat([frames_feat, pred_vec], dim=1)
            frames_feat = F.dropout(frames_feat, self.dropout, self.training)
            frames_feat = self.frame_fc(frames_feat)
            props_source_len = n_frames
            frames_mask = _generate_mask(frames_feat, frames_len)

        words_feat[:, 0] = self.start_vec.to(words_feat.device)
        words_pos = self.word_pos_encoder(words_feat)
        words_feat = F.dropout(words_feat, self.dropout, self.training)
        words_feat = self.word_fc(words_feat)
        words_mask = _generate_mask(words_feat, words_len + 1)

        # 3) Query-conditioned Gaussian 预测:
        #    decoding=1 用文本上下文引导视频 token，h[:, -1] 是 pred token 的 proposal 查询状态。
        enc_out, h = self.trans(frames_feat, frames_mask, words_feat + words_pos, words_mask, decoding=1)
        trans_gauss_logits = self.fc_gauss(h[:, -1]).view(bsz, self.num_props, 2)
        anchor_gate = None
        anchor_gauss_param = None
        if self.use_anchor_prior:
            # 4) Anchor-aware Gaussian prior:
            #    多尺度 anchor 做 masked mean pooling，得到视频结构摘要；
            #    摘要预测无条件 proposal，再用 gate 与 query-conditioned proposal 融合。
            anchor_summary = self._anchor_summary(amp_out['anchor_fpn'], amp_out['anchor_masks'])
            anchor_gauss_logits = self.fc_anchor_prior(anchor_summary).view(bsz, self.num_props, 2)
            length_ratio = frames_len.float().to(frames_feat.device) / max(float(n_frames), 1.0)
            gate_input = torch.cat([h[:, -1], anchor_summary, length_ratio.unsqueeze(-1)], dim=-1)
            anchor_gate = torch.sigmoid(self.fc_anchor_gate(gate_input).view(bsz, self.num_props, 2))
            gauss_logits = anchor_gate * trans_gauss_logits + (1.0 - anchor_gate) * anchor_gauss_logits
            anchor_gauss_param = torch.sigmoid(anchor_gauss_logits).view(bsz*self.num_props, 2)
        else:
            gauss_logits = trans_gauss_logits
        gauss_param = torch.sigmoid(gauss_logits).view(bsz*self.num_props, 2)
        gauss_center = gauss_param[:, 0]
        gauss_width = gauss_param[:, 1]

        # 5) Semantic completion 的视频源:
        #    baseline 仍用原 CPL 的 4x 均匀采样；
        #    AMP 默认改用 HieraMamba anchor_fpn 的 T/4 层，避免再对增强帧做简单池化/采样。
        frames_feat, frames_mask = self._build_proposal_source(
            frames_feat[:, :props_source_len],
            frames_mask[:, :props_source_len],
            props_source_len,
            amp_out,
        )
        props_len = frames_feat.size(1)
        props_feat = frames_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, props_len, -1)
        props_mask = frames_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)

        gauss_weight = self.generate_gauss_weight(props_len, gauss_center, gauss_width)
        
        # 6) Semantic completion:
        #    每个 Gaussian proposal 作为视频注意力权重，重建被 mask 的文本 token。
        #    训练/eval 仍沿用 CPL 的 loss-based proposal 选择逻辑。
        words_feat, masked_words = self._mask_words(words_feat, words_len, weights=weights)
        words_feat = words_feat + words_pos
        words_feat = words_feat[:, :-1]
        words_mask = words_mask[:, :-1]

        words_mask1 = words_mask.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_id1 = words_id.unsqueeze(1) \
            .expand(bsz, self.num_props, -1).contiguous().view(bsz*self.num_props, -1)
        words_feat1 = words_feat.unsqueeze(1) \
            .expand(bsz, self.num_props, -1, -1).contiguous().view(bsz*self.num_props, words_mask1.size(1), -1)

        pos_weight = gauss_weight/gauss_weight.max(dim=-1, keepdim=True)[0]
        _, h, attn_weight = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=pos_weight, need_weight=True)
        words_logit = self.fc_comp(h)

        if self.use_negative:
            # 7) Easy-to-hard negative proposal mining 保持 CPL 原机制不变。
            neg_1_weight, neg_2_weight = self.negative_proposal_mining(props_len, gauss_center, gauss_width, kwargs['epoch'])
            
            _, neg_h_1 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_1_weight)
            neg_words_logit_1 = self.fc_comp(neg_h_1)
  
            _, neg_h_2 = self.trans(props_feat, props_mask, words_feat1, words_mask1, decoding=2, gauss_weight=neg_2_weight)
            neg_words_logit_2 = self.fc_comp(neg_h_2)

            _, ref_h = self.trans(frames_feat, frames_mask, words_feat, words_mask, decoding=2)
            ref_words_logit = self.fc_comp(ref_h)
        else:
            neg_words_logit_1 = None
            neg_words_logit_2 = None
            ref_words_logit = None

        output = {
            'neg_words_logit_1': neg_words_logit_1,
            'neg_words_logit_2': neg_words_logit_2,
            'ref_words_logit': ref_words_logit,
            'words_logit': words_logit,
            'words_id': words_id,
            'words_mask': words_mask,
            'width': gauss_width,
            'center': gauss_center,
            'gauss_weight': gauss_weight,
        }
        if anchor_gate is not None:
            output.update({
                'anchor_gate': anchor_gate.view(bsz*self.num_props, 2),
                'anchor_prior': anchor_gauss_param,
            })
        return output

    def _scale_lengths(self, lengths, old_len, new_len):
        """当 AMP backbone 改变时间长度时，将原始帧长度映射到新 token 长度。"""
        if old_len == new_len:
            return lengths
        scaled = torch.ceil(lengths.float() * float(new_len) / max(float(old_len), 1.0)).long()
        return scaled.clamp(min=1, max=new_len)

    def _anchor_summary(self, anchor_fpn, anchor_masks):
        """对每一层 anchor 做 masked mean pooling，再跨尺度平均成一个视频结构摘要。"""
        summaries = []
        for feat, mask in zip(anchor_fpn, anchor_masks):
            mask = mask.to(feat.device).float().unsqueeze(-1)
            denom = mask.sum(dim=1).clamp(min=1.0)
            summaries.append((feat * mask).sum(dim=1) / denom)
        return torch.stack(summaries, dim=1).mean(dim=1)

    def _build_proposal_source(self, frames_feat, frames_mask, props_source_len, amp_out):
        """
        构造 semantic completion 使用的视频 token。

        - AMP proposal_source='anchor': 使用 HieraMamba anchor_fpn 的某一层。
          默认 proposal_anchor_scale=1，对应原始 T -> T/2 -> T/4 的第二层 anchor，
          与 CPL 原先 props_source_len//4 的计算量接近，但特征来自 AMP anchor。
        - AMP proposal_source='sequence': 使用指定层 sequence_fpn。
        - 其他情况: 保留原 CPL 的均匀采样。
        """
        if self.use_amp and amp_out is not None:
            if self.amp_proposal_source == 'anchor':
                scale = min(self.amp_proposal_anchor_scale, len(amp_out['anchor_fpn']) - 1)
                return (
                    amp_out['anchor_fpn'][scale],
                    amp_out['anchor_masks'][scale].to(frames_feat.device),
                )
            if self.amp_proposal_source == 'sequence':
                scale = min(self.amp_proposal_anchor_scale, len(amp_out['sequence_fpn']) - 1)
                return (
                    amp_out['sequence_fpn'][scale],
                    amp_out['sequence_masks'][scale].to(frames_feat.device),
                )

        props_len = max(props_source_len // 4, 1)
        keep_idx = torch.linspace(0, props_source_len - 1, steps=props_len, device=frames_feat.device).long()
        return frames_feat[:, keep_idx], frames_mask[:, keep_idx]

    def _load_amp_pretrained(self, pretrained_path):
        """按 key/shape 尽量加载 HieraMamba 或 AMP checkpoint，维度不匹配的层会自动跳过。"""
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        state_dict = checkpoint
        if isinstance(checkpoint, dict):
            for key in ('model_ema', 'model', 'state_dict', 'model_parameters'):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    state_dict = checkpoint[key]
                    break
        target_state = self.amp_backbone.state_dict()
        prefixes = (
            'module.vid_net.',
            'model.vid_net.',
            'model_ema.vid_net.',
            'vid_net.',
            'module.amp_backbone.',
            'amp_backbone.',
        )
        filtered = {}
        for key, value in state_dict.items():
            if not torch.is_tensor(value):
                continue
            candidates = [key]
            candidates.extend(key[len(prefix):] for prefix in prefixes if key.startswith(prefix))
            wrapped_candidates = []
            for candidate in candidates:
                wrapped_candidates.append(candidate)
                if not candidate.startswith('backbone.'):
                    wrapped_candidates.append('backbone.' + candidate)
            for candidate in wrapped_candidates:
                if candidate in target_state and target_state[candidate].shape == value.shape:
                    filtered[candidate] = value
                    break
        missing, unexpected = self.amp_backbone.load_state_dict(filtered, strict=False)
        print(
            'Loaded AMP pretrained weights from {}: matched {}, missing {}, unexpected {}.'.format(
                pretrained_path, len(filtered), len(missing), len(unexpected)
            )
        )
    
    
    def generate_gauss_weight(self, props_len, center, width):
        # pdb.set_trace()
        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)
        center = center.unsqueeze(-1)
        width = width.unsqueeze(-1).clamp(1e-2) / self.sigma

        w = 0.3989422804014327
        weight = w/width*torch.exp(-(weight-center)**2/(2*width**2))

        return weight/weight.max(dim=-1, keepdim=True)[0]


    def negative_proposal_mining(self, props_len, center, width, epoch):
        def Gauss(pos, w1, c):
            w1 = w1.unsqueeze(-1).clamp(1e-2) / (self.sigma/2)
            c = c.unsqueeze(-1)
            w = 0.3989422804014327
            y1 = w/w1*torch.exp(-(pos-c)**2/(2*w1**2))
            return y1/y1.max(dim=-1, keepdim=True)[0]

        weight = torch.linspace(0, 1, props_len)
        weight = weight.view(1, -1).expand(center.size(0), -1).to(center.device)

        left_width = torch.clamp(center-width/2, min=0)
        left_center = left_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5
        right_width = torch.clamp(1-center-width/2, min=0)
        right_center = 1 - right_width * min(epoch/self.max_epoch, 1)**self.gamma * 0.5

        left_neg_weight = Gauss(weight, left_center, left_center)
        right_neg_weight = Gauss(weight, 1-right_center, right_center)

        return left_neg_weight, right_neg_weight

    def _mask_words(self, words_feat, words_len, weights=None):
        token = self.mask_vec.to(words_feat.device).unsqueeze(0).unsqueeze(0)
        token = self.word_fc(token)

        masked_words = []
        for i, l in enumerate(words_len):
            l = int(l)
            num_masked_words = max(l // 3, 1) 
            masked_words.append(torch.zeros([words_feat.size(1)], dtype=torch.uint8, device=words_feat.device))
            if l < 1:
                continue
            p = weights[i, :l].cpu().numpy() if weights is not None else None
            choices = np.random.choice(np.arange(1, l + 1), num_masked_words, replace=False, p=p)
            masked_words[-1][choices] = 1
        
        masked_words = torch.stack(masked_words, 0).unsqueeze(-1)
        masked_words_vec = words_feat.new_zeros(*words_feat.size()) + token
        masked_words_vec = masked_words_vec.masked_fill_(masked_words == 0, 0)
        words_feat1 = words_feat.masked_fill(masked_words == 1, 0) + masked_words_vec
        return words_feat1, masked_words


def _generate_mask(x, x_len):
    if False and int(x_len.min()) == x.size(1):
        mask = None
    else:
        mask = []
        for l in x_len:
            mask.append(torch.zeros([x.size(1)], dtype=torch.uint8, device=x.device))
            mask[-1][:int(l)] = 1
        mask = torch.stack(mask, 0)
    return mask


class SinusoidalPositionalEmbedding(nn.Module):
    """This module produces sinusoidal positional embeddings of any length.

    Padding symbols are ignored.
    """

    def __init__(self, embedding_dim, padding_idx, init_size=1024):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.weights = SinusoidalPositionalEmbedding.get_embedding(
            init_size,
            embedding_dim,
            padding_idx,
        )

    @staticmethod
    def get_embedding(num_embeddings, embedding_dim, padding_idx=None):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        import math
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(num_embeddings, -1)
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    def forward(self, input, **kwargs):
        bsz, seq_len, _ = input.size()
        max_pos = seq_len
        if self.weights is None or max_pos > self.weights.size(0):
            # recompute/expand embeddings if needed
            self.weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos,
                self.embedding_dim,
                self.padding_idx,
            )
        self.weights = self.weights.to(input.device)[:max_pos]
        return self.weights.unsqueeze(0)

    def max_positions(self):
        """Maximum number of supported positions."""
        return int(1e5)  # an arbitrary large number
