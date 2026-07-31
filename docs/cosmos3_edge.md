# Cosmos3-Edge in FluxVLA

FluxVLA reuses `Cosmos3FlowMatching` and `Cosmos3MoTBackbone` for all Cosmos3
variants. The backbone selects Qwen3-VL for Nano/Super and Nemotron-3 Dense VL
for Edge from `vlm_config.model_type`.

## Supported scope

- Native FluxVLA training and LIBERO evaluation.
- Official `nvidia/Cosmos3-Edge` transformer checkpoint.
- Native LIBERO 7D actions, padded internally to the checkpoint's 64D action
  projector width with embodiment domain 5.
- H16 joint video/action flow-matching training.

The current Edge configuration enables the Nemotron generator/policy tower. It
does not enable the SigLIP2 reasoner visual tower or autoregressive reasoner
generation.

## Checkpoints

Place the official files at:

```text
checkpoints/
├── Cosmos3-Edge/
│   ├── config.json
│   ├── tokenizer files
│   └── transformer/
│       ├── config.json
│       ├── diffusion_pytorch_model-00001-of-00002.safetensors
│       ├── diffusion_pytorch_model-00002-of-00002.safetensors
│       └── diffusion_pytorch_model.safetensors.index.json
└── Wan2.2-TI2V-5B/
    └── Wan2.2_VAE.pth
```

The Edge config uses the same explicit checkpoint `name_mapping` mechanism as
the PR #51 Nano/Super configs. It expects the published transformer layout:
28 layers, hidden size 2048, FFN size 9216, 16 attention heads, 8 KV heads,
64 action dimensions and 32 embodiment domains.

## LIBERO data contract

| Field | Value |
|---|---|
| images | two 128×128 views, stacked into a 256×128 video frame |
| action | `[B,16,7]`, mean/std normalized and padded internally to 64D |
| embodiment | domain 5 |
| control rate | 20 Hz |
| objective | joint action/video flow matching, weights 10/1 |
| inference | 30 UniPC steps, shift 10 |

There is no LIBERO 7D to DROID 8D conversion.

## Training configs

- Mixed four-suite training:
  `configs/cosmos3/cosmos3edge_libero_full_finetune.py`
- Single-suite training:
  `cosmos3edge_libero_{spatial,object,goal,10}_full_finetune.py`

The current recipes follow JiKun's Nano schedule while replacing the Qwen
backbone and tokenizer with Edge/Nemotron. `full_finetune` is the historical
recipe name; the model freezes the understanding pathway and trains the
generation/action pathway (`freeze_non_moe_vlm_backbone=True`).

Example:

```bash
bash scripts/train.sh \
  configs/cosmos3/cosmos3edge_libero_full_finetune.py \
  work_dirs/cosmos3edge_libero_mixed \
  --eval-after-train
```

Evaluate a trained checkpoint with the matching suite config and
`scripts/eval.sh`. When loading a FluxVLA fine-tuned safetensors file, override
`model.name_mapping=None`; the explicit Edge mapping is only for the official
Diffusers-format base transformer.

## Licensing

The adapted Nemotron/RMSNorm/ReLU²/mRoPE code retains NVIDIA's OpenMDW-1.1
identifier and is documented in
`fluxvla/models/third_party_models/cosmos3/ATTRIBUTION.txt`.
