#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

bash scripts/preflight_4xa100.sh
source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/hf-cache}"

accelerate launch --num_processes 4 \
  -m axolotl.cli.train \
  configs/qwen3.8-flash-next-qlora-4xa100-smoke.yaml

echo "Smoke run finished. Inspect outputs/zoria-v0-4xa100-smoke."
