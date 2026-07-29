# Cosmos3-Edge in FluxVLA

This document is the implementation and release gate for the native
Cosmos3-Edge integration. The Edge path reuses `Cosmos3FlowMatching`; it does
not introduce a second VLA facade. Nano/Super use the Qwen3-VL backbone, while
Edge is selected explicitly and uses the Nemotron-3 Dense VL generation
backbone.

## Current implementation boundary

- Native, in-process FluxVLA is the supported training and LIBERO evaluation
  path.
- NVIDIA's standalone Cosmos environment is a numerical oracle. It must not be
  installed into the FluxVLA environment: the checked source snapshot uses
  Torch 2.10 and `transformers>=4.57.1,<5`, while FluxVLA pins Transformers
  5.3.0.
- The vendored Cosmos subset and its OpenMDW-1.1 license, notice and provenance
  are under `fluxvla/models/third_party_models/cosmos3/`.
- GPU parity and post-training are release gates, not facts
  established by CPU unit tests. A model must not be labelled Edge-compatible
  until the parity gate below passes.

The source snapshot available during integration reports package version
`cosmos-framework==1.2.2`, but contains no Git metadata. Before publishing a
checkpoint, record a resolvable NVIDIA Git commit/tag, Hugging Face revision,
and every checkpoint shard hash.

## Environment hard gate

Driver 470 is unsupported. Run the checks from the exact job environments.
The two-A800 FluxVLA training gate is:

Download the official base snapshot before the gate. NVIDIA's repository is
`nvidia/Cosmos3-Edge` (there is no hyphen between `Cosmos` and `3`). The helper
defaults to shared storage and links the snapshots into this repository:

```bash
hf auth login

# LIBERO needs Edge base + Wan2.2 VAE.
bash scripts/download_cosmos3_edge_checkpoints.sh
```

Set `COSMOS3_CHECKPOINT_STORE` and `COSMOS3_EDGE_REVISION` to change the shared
destination or pin an immutable HF revision. Accept the NVIDIA model terms
before downloading. The complete snapshot root is required because
tokenizer/config files live outside the `transformer/` subdirectory.

The integration has been checked against Base revision
`6f58f6b4c91288838e60b6bcb2cc45d997e961de`. It publishes `(Dmax=64,
domains=32)` and a tokenizer with IDs `(pad,bos,eos,vision_start,vision_end) =
(11,1,11,20,21)`. Edge loading selects the nested
`transformer/diffusion_pytorch_model.safetensors.index.json` because the Base
root manifest omits generation/action tensors. Strict loading of all required
tensors must report 100% required and overall coverage.

The example below assumes the repository's cu124 pair (Torch 2.6/CUDA 12.4)
with its matching driver floor. If the A800 job instead uses cu128, use Torch
2.8/CUDA 12.8 and the 570.26 driver floor used by the oracle/5090 examples.

```bash
torchrun --standalone --nproc-per-node=2 \
  scripts/check_cosmos3_edge_environment.py \
  --profile flux --require-gpus 2 --require-distributed \
  --require-flash-attn --min-driver-version 550.54.14 \
  --report-dir work_dirs/cosmos3_gate_a800 \
  --hash-path checkpoints/Cosmos3-Edge \
  --hash-path checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

Run the official oracle in a separate environment. The cu128 profile should
use the driver floor selected for that image (the command below uses the CUDA
12.8 GA floor):

```bash
torchrun --standalone --nproc-per-node=2 \
  scripts/check_cosmos3_edge_environment.py \
  --profile oracle --require-gpus 2 --require-distributed \
  --require-flash-attn --min-driver-version 570.26.00 \
  --report-dir work_dirs/cosmos3_gate_oracle \
  --hash-path checkpoints/Cosmos3-Edge \
  --hash-path checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth
```

Archive each JSON output beside the run. Both training devices must pass BF16
matmul and NCCL. Do not proceed when CUDA initialization, shard hashes, or
source revisions are unresolved.

## Model and batch contract

The checkpoint config is authoritative for internal width and domain count.
External robot actions are never converted between embodiments.

| Field | Contract |
|---|---|
| `images` | `[B,3,T,H,W]`, float32 input in `[-1,1]`, cast to BF16 by the model |
| `text_token_ids` | `[B,L]`, int64, Edge tokenizer/chat template and pad ID |
| `actions` | `[B,Ha,Dmax]`, float32/BF16; zero-padded after the raw dimensions |
| `states` | `[B,Dmax]`, float32/BF16, optional state condition |
| `raw_action_dim` | `[B]`, int64; LIBERO uses 7 |
| `embodiment_ids` | `[B]`, int64; LIBERO uses domain 5 |
| `action_masks` | `[B,Ha]`, bool; false rows never contribute to action loss |
| `frame_masks` | `[B,T]`, bool; false/repeated frames never contribute to video loss |
| `conditioning_fps` | `[B]`, float32; LIBERO uses 20 by default |
| `sequence_plan` | Python list of length `B` |

LIBERO remains `[B,H,7]`. Padding to `Dmax` is only an internal projector
contract, and loss/output slicing uses `raw_action_dim`; there is no forced
conversion to another action representation.

## Checkpoint and parity gate

Checkpoint loading must identify HF single-file/indexed safetensors, nested
shards, and distributed checkpoints before mapping. Required tensor and numel
coverage must be 100%; only explicitly allowlisted, deliberately reinitialized
action/domain parameters may be absent. Unknown formats, architecture
mismatches, and unmapped required tensors are fatal.

For the same input, timestep and noise, compare FluxVLA with the standalone
official implementation before training:

1. Assert 28 generation layers, hidden size 2048, FFN size 9216, 16 attention
   heads and 8 KV heads.
2. Compare transformer velocity in BF16 with `rtol=atol=1e-2`.
3. Check H16 output shape, condition placement and distribution after UniPC
   sampling.

Run one policy sample with `--dump-first-velocity` to save FluxVLA's exact
first conditional input/velocity NPZ. Capture the same named arrays from the
official oracle process, then make the gate machine-checkable:

```bash
python scripts/compare_cosmos3_parity.py \
  --reference work_dirs/parity/official_velocity.npz \
  --candidate work_dirs/parity/fluxvla_velocity.npz \
  --rtol 1e-2 --atol 1e-2 \
  --output-json work_dirs/parity/report.json
```

The comparator requires exact equality for every shared input tensor and
shape/finite/allclose agreement for all `velocity_*` arrays; it exits nonzero
on failure. The two incompatible Python environments never need to be mixed.
The inspected NVIDIA snapshot does not ship an Edge-policy parity exporter,
and no real Edge checkpoint was present on this node. Therefore the official
NPZ producer and a passing real-weight comparison remain a release blocker;
the FluxVLA dump plus comparator alone do not establish parity.

If intermediate velocity parity fails, use the official process as an oracle
and stop native post-training. Do not compensate by loosening checkpoint
loading.

LIBERO defaults to `action_init=checkpoint`. If the inspected Edge checkpoint
contains compatible action heads but domain 5 should be task-specific, use
`cosmos3edge_libero_task_smoke_fresh_domain.py`; it sets
`action_init=fresh_domain` and row 5 while all other rows load strictly. If the
checkpoint has no action tensors at all, use
`cosmos3edge_libero_task_smoke_fresh_all.py`, whose explicit `(Dmax, domains)`
layout must first be checked against the released config. These are explicit
experiments, never an automatic fallback; the loader reports every
reinitialized key and still requires 100% coverage outside that allowlist.
`fresh_domain` cannot repair a missing or shape-incompatible action head;
`fresh_all` is required in that case.

First prove strict checkpoint loading without sampling:

```bash
python scripts/cosmos3_fluxvla_infer.py \
  --model-size edge --checkpoint checkpoints/Cosmos3-Edge \
  --vae-path checkpoints/Wan2.2-TI2V-5B/Wan2.2_VAE.pth \
  --input /path/to/one_edge_sample.json \
  --output-dir work_dirs/cosmos3_edge_load --load-only
```

The native loader accepts inspected indexed/single safetensors and has strict
mappings for native VFM and Nemotron HF layouts.
Diffusers-style aliases are implemented but remain experimental: the checked
official diffusers class is Qwen-specific and no real Edge diffusers manifest
was available. The loader rejects DCP and legacy Edge `.pt/.pth`. A released
artifact's actual key manifest remains the authority: do not waive the 100%
required-numel gate.

If a checkpoint contains complete `vision_vae.*` tensors, those embedded
weights take precedence and an unavailable external path no longer prevents
model construction. Otherwise the loader requires the original
`Wan2.2_VAE.pth` and fails before encode/decode; an uninitialized VAE is never
used as a fallback. Omit the external VAE from an environment hash manifest
only after strict checkpoint inspection proves embedded VAE coverage.

## LIBERO post-training sequence

Use the Edge base/intermediate checkpoint, domain 5, raw width 7, H16, 20 Hz,
policy mode, full windows and no prepended state. Start with one task:

The checked local Spatial `meta/info.json` advertises 30 FPS, while the PR #51
Cosmos recipe and this policy contract use 20 Hz. Before spending GPU time,
confirm whether 30 is only the encoded-video/export clock or the real control
and action sampling rate. If actions are actually 30 Hz, change training,
prompt metadata and evaluation together; FPS modulation makes a silent
20/30 mismatch a correctness bug.

1. Ten-step forward/backward check.
2. Overfit 32 windows for 100 steps; require at least 30% loss reduction.
3. Freeze the backbone and run action-projection/embedding training for 500
   steps.
4. Unfreeze the generation tower and continue to 2,000 steps with BF16, FSDP
   full-shard, batch 1/GPU and accumulation 8. Use backbone LR `4e-5`, action LR
   `2e-4`, and action/video weights `10/1`.
5. Save and resume at 500/1,000/2,000, then run evaluation in a fresh process.

The current FSDP runner writes a rank-0 full model/optimizer `.pt` for resume
and a weights-only safetensors file for deployment. This is suitable for the
smoke and partial runs only after host-RAM/disk profiling. A 4B full-finetune
run is not released until sharded `torch.distributed.checkpoint` save/resume
has been implemented and exercised; do not treat the full-state path as a
production 4B checkpoint strategy.

Action-only is a loss ablation: the first implementation still processes
future visual tokens. Removing those tokens is a separate optimization and
requires a new parity test. Do not start the four-suite benchmark unless 20
single-task trials reach at least 30% success without NaN or bounds errors.

Executable entry points, from the repository root:

```bash
# Ten-step forward/backward and checkpoint-write gate
torchrun --standalone --nproc-per-node=2 scripts/train.py \
  --config configs/cosmos3/cosmos3edge_libero_task_overfit.py \
  --work-dir work_dirs/cosmos3edge_libero_task0_10step \
  --cfg-options runner.max_steps=10 runner.save_iter_interval=10

# Exact 32-window, 100-step overfit gate (prompt dropout disabled)
torchrun --standalone --nproc-per-node=2 scripts/train.py \
  --config configs/cosmos3/cosmos3edge_libero_task_overfit.py \
  --work-dir work_dirs/cosmos3edge_libero_task0_overfit \
  --cfg-options runner.save_iter_interval=100

# 500-step action-interface smoke, task 0, followed by fresh-process eval
torchrun --standalone --nproc-per-node=2 scripts/train.py \
  --config configs/cosmos3/cosmos3edge_libero_task_smoke.py \
  --work-dir work_dirs/cosmos3edge_libero_task0_action \
  --eval-after-train

# New 2,000-step partial run, initialized from the action-smoke weights only.
# This intentionally resets optimizer/scheduler/global-step because the
# trainable parameter topology changes between the two stages.
torchrun --standalone --nproc-per-node=2 scripts/train.py \
  --config configs/cosmos3/cosmos3edge_libero_task_partial.py \
  --work-dir work_dirs/cosmos3edge_libero_task0_partial \
  --cfg-options model.pretrained_name_or_path=work_dirs/cosmos3edge_libero_task0_action/checkpoints/latest-checkpoint.safetensors \
    inference_model.pretrained_name_or_path=work_dirs/cosmos3edge_libero_task0_action/checkpoints/latest-checkpoint.safetensors \
  --eval-after-train
```

Use `--resume-from ...latest-checkpoint.pt` only within the same config and
trainable-parameter topology. Checkpoints record the exact ordered optimizer
groups; a cross-topology resume, legacy checkpoint without that manifest, or
scheduler group mismatch is rejected instead of silently changing learning
rates.

The four full-suite configs are
`cosmos3edge_libero_{spatial,object,goal,10}_posttrain.py`. They are production
post-training schedules, not smoke runs: each uses 30,000 optimizer steps and
saves every 3,000 steps. With two GPUs, one sample/GPU and accumulation 8, the
effective batch is 16. The checked Spatial dataset contains 46,186 valid H16
windows, so one epoch is `ceil(46186/16)=2,887` optimizer steps and 30,000
steps is about 10.4 epochs. The 2,000-step task-0 run is about 8.9 epochs of
that single task, but only 0.69 epoch of the complete Spatial suite; it is a
gate, not a converged full-suite recipe.

Run each suite independently from the Edge base checkpoint. Do not initialize
Object/Goal/LIBERO-10 from a Spatial-only checkpoint. A complete Spatial run
and an exact-topology restart are:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export TOKENIZERS_PARALLELISM=false

torchrun --standalone --nproc-per-node=2 scripts/train.py \
  --config configs/cosmos3/cosmos3edge_libero_spatial_posttrain.py \
  --work-dir work_dirs/cosmos3edge_libero_spatial_partial_30k \
  --eval-after-train

# Use the same config, work directory and originally planned 30,000-step
# scheduler. The .pt contains optimizer/scheduler state; safetensors does not.
torchrun --standalone --nproc-per-node=2 scripts/train.py \
  --config configs/cosmos3/cosmos3edge_libero_spatial_posttrain.py \
  --work-dir work_dirs/cosmos3edge_libero_spatial_partial_30k \
  --resume-from work_dirs/cosmos3edge_libero_spatial_partial_30k/checkpoints/latest-checkpoint.pt \
  --eval-after-train
```

Repeat the first command with these config/work-directory pairs:

```text
cosmos3edge_libero_object_posttrain.py  -> cosmos3edge_libero_object_partial_30k
cosmos3edge_libero_goal_posttrain.py    -> cosmos3edge_libero_goal_partial_30k
cosmos3edge_libero_10_posttrain.py      -> cosmos3edge_libero_10_partial_30k
```

Before launching a missing suite, verify its effective epoch size from
`meta/episodes.jsonl`: sum `max(length - 16, 0)`, divide by the effective
batch 16 and round up. Thirty thousand steps is the released starting budget;
if it is fewer than eight epochs for a replacement dataset, increase the
budget on the initial launch. Do not extend `runner.max_steps` only when
resuming: that changes the cosine schedule definition while restoring an
already-running scheduler.

Checkpoint headers set `Dmax` and the embodiment-domain count; config loading
fails instead of guessing when a real checkpoint has no unambiguous action
head. No fixed step count guarantees closed-loop convergence: accept the run
only when action/video losses are finite and plateauing and 50-trial-per-task
closed-loop success no longer improves between retained checkpoints.

## Result provenance

Every published result must include resolved config, FluxVLA commit, NVIDIA
source revision, HF model revision, checkpoint shard hashes, dataset revision,
seeds, resolution, horizon, denoising/CFG settings, latency and memory. LIBERO
scores enter the README only when the artifact and exact reproduction command
are available.

After the single-task gate, run Spatial, Object, Goal and LIBERO-10 with 50
trials per task for seeds 7, 17 and 27. Publish per-task and aggregate success
together with latency and peak memory.
Every Edge config has an explicit top-level training seed. Run each benchmark
checkpoint three times, for example:

```bash
for SEED in 7 17 27; do
  torchrun --standalone --nproc-per-node=2 scripts/eval.py \
    --config configs/cosmos3/cosmos3edge_libero_spatial_posttrain.py \
    --ckpt-path /path/to/latest-checkpoint.safetensors \
    --cfg-options seed=$SEED eval.seed=$SEED eval.inference_seed=$SEED
done
```

Repeat with the Object, Goal and LIBERO-10 configs; archive each resolved
config and summary independently.
