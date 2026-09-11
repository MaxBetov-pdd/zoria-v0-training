#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MIN_FREE_GB="${ZORIA_MIN_FREE_GB:-600}"

echo "== GPU =="
nvidia-smi --query-gpu=index,name,memory.total,compute_cap --format=csv,noheader
GPU_COUNT="$(nvidia-smi --query-gpu=index --format=csv,noheader | wc -l | tr -d ' ')"
if [[ "${GPU_COUNT}" -ne 4 ]]; then
  echo "ERROR: the selected large path expects exactly four GPUs." >&2
  exit 1
fi
GPU_MEMORY="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk 'BEGIN {ok=1} {gsub(/[[:space:]]/,""); if ($1 < 75000) ok=0} END {print ok}')"
if [[ "${GPU_MEMORY}" != "1" ]]; then
  echo "ERROR: every GPU must have at least 75,000 MiB VRAM." >&2
  exit 1
fi

echo
echo "== RAM =="
RAM_GB="$(awk '/MemTotal/ {printf "%.0f", $2/1024/1024}' /proc/meminfo)"
echo "${RAM_GB} GB"
if [[ "${RAM_GB}" -lt 340 ]]; then
  echo "ERROR: at least 340 GB system RAM is required for this configuration." >&2
  exit 1
fi

echo
echo "== Disk =="
FREE_GB="$(df -BG "${ROOT_DIR}" | awk 'NR==2 {gsub("G","",$4); print $4}')"
echo "${FREE_GB} GB free at ${ROOT_DIR}"
if [[ "${FREE_GB}" -lt "${MIN_FREE_GB}" ]]; then
  echo "ERROR: need at least ${MIN_FREE_GB} GB free before downloading Flash-Next." >&2
  exit 1
fi

echo
python3 --version
python3 - <<'PY'
try:
    import torch
except ImportError:
    print("torch: not installed yet")
else:
    print("torch:", torch.__version__)
    print("cuda:", torch.version.cuda)
    print("cuda_available:", torch.cuda.is_available())
    print("gpu_count:", torch.cuda.device_count())
PY

echo
echo "Preflight passed for the selected 4xA100 experimental path."
