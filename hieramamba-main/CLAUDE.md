# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

HieraMamba is a research project for **video temporal grounding** — given an untrimmed video and a natural language query, the model predicts start/end timestamps of the relevant segment. It uses Mamba selective state-space models to build a hierarchical, linear-time architecture via **Anchor-MambaPooling (AMP) blocks** that compress video embeddings into multi-scale anchor tokens. CVPR 2026 paper by Joungbin An and Kristen Grauman (UT Austin).

This is a standalone research training/evaluation pipeline, not a library or package.

## Setup

```bash
# Install dependencies, build NMS C++ extension, initialize hydra submodule
./install.sh

# Or manually:
git submodule update --init --recursive
pip install -r requirements.txt
cd libs/nms && python setup_nms.py build_ext --inplace
```

Requires CUDA-capable GPU. Key deps: `torch>=2.1.0`, `mamba-ssm>=2.2.3`, `causal-conv1d>=1.2.0`.

## Training and Evaluation

```bash
# Train (config filename from opts/, experiment name)
python train.py --opt ego4d_hieramamba.yaml --name ego4d_hieramamba

# Evaluate (uses saved config from experiments/<name>/)
python eval.py --name ego4d_hieramamba --ckpt last

# Both together
./run.sh ego4d_hieramamba.yaml ego4d_hieramamba 0   # arg3 = GPU id
```

Available benchmark configs in `opts/`: `ego4d_hieramamba.yaml`, `mad_hieramamba.yaml`, `madv2_hieramamba.yaml`, `tacos_hieramamba.yaml`.

Experiments are saved under `experiments/<name>/`. Logs go to TensorBoard (`events.out.tfevents.*`).

## Architecture

```
train.py / eval.py          Entry points
libs/
  core/opt.py               Config loading — merges YAML + DEFAULTS dict
  modeling/
    model.py                Top-level models: HieraMamba, PtTransformer
    video_net.py            HieraMambaBackbone — builds feature pyramid via stacked AMP blocks
    anchor_mamba.py         AMP blocks (AnchorMambaPoolingBlockGated, AnchorMambaPoolingBlock)
    text_net.py             Text backbones: TextIdentity (pre-encoded), TextTransformer (tokenized)
    fusion.py               Cross-attention fusion: XAttNFusion, XAttNFusion2
    head.py                 ClsHead / RegHead (1D conv per FPN level)
    blocks.py               MaskedConv1D, TransformerEncoder/Decoder, MaskedMHA, FFN, SwiGLUFFN
    loss.py                 JIT-compiled focal, giou, diou losses
    losses.py               Contrastive losses: ACC (anchor-to-sequence), SPC (GT point)
    contrastive_losses.py   Multi-positive InfoNCE implementation
    optim.py                Per-component LR scheduling, warmup + cosine/multistep
  data/
    dataset.py              VideoCentricDataset / TextCentricDataset
    tokenizer.py            GloVeTokenizer (used for TACoS)
  worker.py                 TrainerOriginal/TrainerAuxiliary, EvaluatorOriginal/EvaluatorAuxiliary
  nms/                      C++ 1D NMS extension
hydra/                      Git submodule — Hydra bidirectional SSM mixing layer
```

### Model Forward Paths

**Early Fusion** (Ego4D, MAD — `fusion_before_vid=True`):
text encode → video project → cross-attention fuse → HieraMamba backbone → fuse again → cls/reg heads

**Late Fusion** (TACoS — `fusion_before_vid=False`):
text encode → HieraMamba backbone → cross-attention fuse → cls/reg heads

### Model Registry

Models and components are registered via string keys in config YAML. Key registrations:
- `'hieramamba'` → `HieraMamba`, `'pt_transformer'` → `PtTransformer`
- `'hieramamba_backbone'` → `HieraMambaBackbone`
- `'identity'` → `TextIdentity`, `'transformer'` → `TextTransformer`
- `'xattn'` → `XAttNFusion`, `'xattn2'` → `XAttNFusion2`
- Trainer/evaluator: `'TrainerAuxiliary'` / `'EvaluatorAuxiliary'` (used by all HieraMamba configs)

### Config System

YAML files in `opts/` are merged with the `DEFAULTS` dict in `libs/core/opt.py`. Config values like `model_type`, `trainer_type`, `evaluator_type`, `text_type`, `fusion_type` drive module instantiation. All hyperparameters (learning rates, loss weights, architectural dimensions, AMP block counts, pooling types) are in the YAML.

## Key Design Patterns

- **All models use masked operations** — `MaskedConv1D`, `MaskedMHA`, etc. handle variable-length sequences via length tensors. Always pass and propagate masks.
- **AMP blocks are hierarchical** — each block in the backbone takes the anchor stream from the previous level, progressively compressing sequence length. `num_stages` controls pyramid depth.
- **Contrastive losses** (ACC/SPC) are optional per-config and only used in `TrainerAuxiliary`. Ego4D and TACoS enable them; MAD disables them.
- **Microbatching** — `microbatch_size` in config controls gradient accumulation when `batch_size > microbatch_size`.
- **EMA** — trainer maintains an exponential moving average of model weights, used for checkpointing.
- **No automated tests** — validation is done via benchmark evaluation (Rank@k at IoU thresholds).
