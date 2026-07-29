#!/bin/bash
# Train Cosmos3-Edge from the official base checkpoint on one LIBERO suite.

set -euo pipefail

SUITE="${1:-}"
if [[ -z "${SUITE}" ]]; then
  echo "Usage: $0 <spatial|object|goal|10> [work_dir]" >&2
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

WORK_DIR="${2:-/mnt/data/cpfs/users/leo/wk_dir/cosmos3/edge_libero_${SUITE}_single}"

# Preserve DLC/MLP distributed settings. Default to the current two-GPU node
# only when the platform and caller did not provide a process count.
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
  --eval-after-train
