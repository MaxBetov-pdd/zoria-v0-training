#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/hf-cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HF_HOME}/datasets}"

python scripts/prepare_mvp_data.py

echo
echo "Prepared files:"
du -h data/processed/zoria_v0_train.jsonl \
       data/processed/zoria_v0_audit_holdout.jsonl \
       data/processed/zoria_v0_manifest.json
