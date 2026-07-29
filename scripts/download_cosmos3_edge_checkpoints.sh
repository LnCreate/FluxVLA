#!/usr/bin/env bash
# Download the official Cosmos3-Edge snapshots without filling the repo disk.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STORE="${COSMOS3_CHECKPOINT_STORE:-/mnt/workspace/FluxVLA_checkpoints}"
EDGE_REVISION="${COSMOS3_EDGE_REVISION:-6f58f6b4c91288838e60b6bcb2cc45d997e961de}"
WAN22_REVISION="${WAN22_REVISION:-921dbaf3f1674a56f47e83fb80a34bac8a8f203e}"

HF_BIN="${HF_BIN:-hf}"
if ! command -v "${HF_BIN}" >/dev/null 2>&1; then
  if command -v huggingface-cli >/dev/null 2>&1; then
    HF_BIN=huggingface-cli
  else
    echo "ERROR: neither 'hf' nor 'huggingface-cli' is available." >&2
    exit 1
  fi
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
  if [[ "$(basename "${HF_BIN}")" == "hf" ]]; then
    if ! "${HF_BIN}" auth whoami >/dev/null 2>&1; then
      echo "ERROR: Hugging Face is not authenticated." >&2
      echo "Run: ${HF_BIN} auth login" >&2
      echo "Also accept the NVIDIA model terms on the model pages." >&2
      exit 2
    fi
  elif ! "${HF_BIN}" whoami >/dev/null 2>&1; then
    echo "ERROR: Hugging Face is not authenticated." >&2
    echo "Run: ${HF_BIN} login" >&2
    echo "Also accept the NVIDIA model terms on the model pages." >&2
    exit 2
  fi
fi

mkdir -p "${STORE}" "${ROOT}/checkpoints"

download_snapshot() {
  local repo="$1"
  local revision="$2"
  local target="$3"

  echo "Downloading ${repo}@${revision} -> ${target}"
  "${HF_BIN}" download "${repo}" \
    --revision "${revision}" \
    --local-dir "${target}"
}

link_checkpoint() {
  local name="$1"
  local target="$2"
  local link="${ROOT}/checkpoints/${name}"

  target="$(cd "${target}" && pwd)"
  if [[ -L "${link}" ]]; then
    ln -sfn "${target}" "${link}"
  elif [[ -e "${link}" ]]; then
    if [[ "$(cd "${link}" && pwd)" != "${target}" ]]; then
      echo "ERROR: refusing to replace existing ${link}." >&2
      exit 3
    fi
  else
    ln -s "${target}" "${link}"
  fi
}

EDGE_DIR="${STORE}/Cosmos3-Edge"
download_snapshot "nvidia/Cosmos3-Edge" "${EDGE_REVISION}" "${EDGE_DIR}"
link_checkpoint "Cosmos3-Edge" "${EDGE_DIR}"

VAE_LINK="${ROOT}/checkpoints/Wan2.2-TI2V-5B"
if [[ ! -f "${VAE_LINK}/Wan2.2_VAE.pth" ]]; then
  VAE_DIR="${STORE}/Wan2.2-TI2V-5B"
  echo "Downloading Wan2.2 VAE -> ${VAE_DIR}"
  "${HF_BIN}" download "Wan-AI/Wan2.2-TI2V-5B" \
    --revision "${WAN22_REVISION}" \
    --include "Wan2.2_VAE.pth" \
    --local-dir "${VAE_DIR}"
  link_checkpoint "Wan2.2-TI2V-5B" "${VAE_DIR}"
else
  echo "Using existing ${VAE_LINK}/Wan2.2_VAE.pth"
fi

echo "Checkpoint links:"
ls -ld \
  "${ROOT}/checkpoints/Cosmos3-Edge" \
  "${ROOT}/checkpoints/Wan2.2-TI2V-5B"
