#!/bin/bash
# Fine-tune a mixed-LIBERO Cosmos3-Edge checkpoint on one LIBERO suite.

set -euo pipefail

SUITE="${1:-}"
MIXED_CKPT="${2:-}"
if [[ -z "${SUITE}" || -z "${MIXED_CKPT}" ]]; then
  echo "Usage: $0 <spatial|object|goal|10> <mixed_checkpoint.safetensors> [work_dir]" >&2
  exit 2
fi

case "${SUITE}" in
  spatial|object|goal)
    CONFIG="configs/cosmos3/cosmos3edge_libero_${SUITE}_full_finetune.py"
    ;;
  10|long)
    SUITE="10"
    CONFIG="configs/cosmos3/cosmos3edge_libero_10_full_finetune.py"
    ;;
  *)
    echo "Unsupported suite '${SUITE}'; use spatial, object, goal, or 10." >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ ! -f "${MIXED_CKPT}" ]]; then
  echo "Mixed checkpoint not found: ${MIXED_CKPT}" >&2
  exit 2
fi
MIXED_CKPT="$(readlink -f "${MIXED_CKPT}")"
WORK_DIR="${3:-/mnt/data/cpfs/users/leo/wk_dir/cosmos3/edge_libero_${SUITE}_mixed_finetune}"

if [[ -z "${NPROC_PER_NODE:-}" && -z "${MLP_WORKER_GPU:-}" ]]; then
  export NPROC_PER_NODE=2
fi
export WORLD_SIZE="${WORLD_SIZE:-${MLP_WORKER_NUM:-1}}"
export RANK="${RANK:-${MLP_ROLE_INDEX:-0}}"
export MASTER_ADDR="${MASTER_ADDR:-${MLP_WORKER_0_HOST:-localhost}}"
export MASTER_PORT="${MASTER_PORT:-${MLP_WORKER_0_PORT:-29500}}"

export WANDB_PROJECT="${WANDB_PROJECT:-fluxvla_cosmos3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"

bash scripts/train.sh \
  "${CONFIG}" \
  "${WORK_DIR}" \
  --eval-after-train \
  --cfg-options \
  "model.pretrained_name_or_path=${MIXED_CKPT}" \
  model.name_mapping=None
