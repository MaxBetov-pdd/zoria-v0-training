#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

if [[ -f /etc/os-release ]]; then
  . /etc/os-release
  if [[ "${ID:-}" != "ubuntu" || "${VERSION_ID:-}" != "24.04" ]]; then
    echo "WARNING: this kit is prepared for Ubuntu 24.04; detected ${PRETTY_NAME:-unknown}." >&2
  fi
fi

if command -v sudo >/dev/null 2>&1; then
  sudo apt-get update
  sudo apt-get install -y git git-lfs curl build-essential python3.12-venv
fi

if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="${HOME}/.local/bin:${PATH}"

git lfs install --skip-repo

# A100 is Ampere. CUDA 12.8 is the stable Axolotl wheel path for this host.
export UV_TORCH_BACKEND="${UV_TORCH_BACKEND:-cu128}"
uv venv --python 3.12
# shellcheck disable=SC1091
source .venv/bin/activate

uv pip install --no-build-isolation "axolotl[deepspeed]==0.18.0" torchvision
uv pip install "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@4dfa522"
uv pip install --upgrade \
  "huggingface_hub[hf_transfer]" \
  "datasets<4" \
  tqdm \
  pyyaml \
  lm-eval \
  langdetect \
  immutabledict

mkdir -p benchmarks
if [[ ! -d benchmarks/ukrainian-llm-leaderboard/.git ]]; then
  git clone https://github.com/lang-uk/ukrainian-llm-leaderboard.git \
    benchmarks/ukrainian-llm-leaderboard
  git -C benchmarks/ukrainian-llm-leaderboard checkout 74d8069
fi

echo
echo "Environment created in ${ROOT_DIR}/.venv"
echo "Next:"
echo "  source .venv/bin/activate"
echo "  bash scripts/preflight_4xa100.sh"
echo "  bash scripts/download_and_prepare.sh"
