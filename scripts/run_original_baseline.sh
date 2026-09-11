#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
source .venv/bin/activate

export TOKENIZERS_PARALLELISM=false
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
export HF_HOME="${HF_HOME:-${ROOT_DIR}/hf-cache}"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

MODEL="${ZORIA_BASE_MODEL:-Qwen/Qwen3.8-Flash-Next}"
TASKS="${ZORIA_BASELINE_TASKS:-flores_en-uk,long_flores_en-uk,xlsum_uk,belebele_ukr_Cyrl,ifeval_uk}"

# This uses the original checkpoint with 4-bit loading for a comparable
# memory-conscious baseline. Start with the small task list above. Run the
# complete ukrainian_bench group only after the smoke evaluation succeeds.
lm_eval --model hf \
  --model_args "pretrained=${MODEL},load_in_4bit=True,torch_dtype=bfloat16,device_map=auto,trust_remote_code=True" \
  --tasks "${TASKS}" \
  --batch_size auto \
  --output_path ./eval-results/original-qwen \
  --log_samples \
  --include_path ./benchmarks/ukrainian-llm-leaderboard/tasks \
  --apply_chat_template
