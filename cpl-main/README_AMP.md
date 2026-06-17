# CPL + AMP

This branch adds an optional Anchor-Mamba Pooling video encoder to CPL. The
original `config/*/main.json` files still run the baseline CPL. Use the new
AMP configs for the fused model:

```bash
python train.py --config-path config/charades/amp.json --log_dir LOG_DIR --tag amp
python train.py --config-path config/activitynet/amp.json --log_dir LOG_DIR --tag amp
```

## Runtime

The AMP backbone will use `mamba_ssm.Mamba2` when it is installed. If
`mamba_ssm` is unavailable, it falls back to a lightweight convolutional
sequence mixer so the model can still be imported and shape-checked. For real
AMP training on the server, install the HieraMamba runtime dependencies:

```bash
pip install mamba-ssm causal-conv1d
```

## Config Knobs

The new options live under `model.config.AMP`:

- `enabled`: switch AMP integration on or off.
- `use_anchor_prior`: fuse anchor-derived Gaussian priors with CPL's
  query-conditioned Gaussian logits.
- `gate_bias`: initial gate bias. The default `2.0` makes the model start closer
  to original CPL and gradually learn to use anchor priors.
- `freeze_backbone`: freeze AMP parameters when using a pretrained backbone.
- `pretrained_path`: optional HieraMamba checkpoint path. The loader accepts
  checkpoints containing `model`, `model_ema`, `state_dict`, or
  `model_parameters`, and loads keys that match the AMP backbone by name and
  shape.
- `backbone`: HieraMamba-style backbone settings, including `embd_dim`, `arch`,
  `pool_method`, Mamba hyperparameters, and `pyramid_fusion`.

Recommended first server run:

1. Train with `pretrained_path: null` and `freeze_backbone: false` to verify the
   pipeline.
2. If you have a HieraMamba checkpoint with compatible feature dimensions, set
   `pretrained_path` and try `freeze_backbone: true` for a stable warm-up run.
3. After the loss is stable, unfreeze the backbone or lower the global learning
   rate before full fine-tuning.
