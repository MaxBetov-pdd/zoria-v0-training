# Zoria v0

Zoria v0 is a text-only Ukrainian adaptation of
`Qwen/Qwen3.8-Flash-Next` using 4-bit QLoRA. The first run is intentionally
small and conservative: it targets Ukrainian translation, QA, summarization,
grammar and instruction behaviour while keeping a small English/code/math
replay portion.

The VPS scripts target **Ubuntu 24.04**, Python 3.12, PyTorch 2.12.1 and
CUDA 12.8 on the selected 4×A100 host. Axolotl is installed through `uv`.
The pinned Axolotl release is `0.18.0`; the provider must expose NVIDIA
driver 570.26 or newer.

## Files

- `training_spec.md` — exact model, data caps and run order.
- `data/sources.yaml` — dataset roles and later-stage sources.
- `scripts/prepare_mvp_data.py` — streaming download, normalization and
  deduplication.
- `scripts/run_original_baseline.sh` — baseline evaluation for original Qwen.
- `configs/qwen3.8-flash-next-qlora-4xa100.yaml` — primary 300-step run.
- `configs/qwen3.8-flash-next-qlora-4xa100.yaml` — selected primary run.
- `configs/qwen3.8-flash-next-qlora-4xa100-smoke.yaml` — selected smoke test.
- `configs/qwen3.8-flash-next-qlora-4xa100-smoke.yaml` — 20-step FSDP smoke
  test.
- `colab/README.md` — Google Colab pilot instructions for Qwen3.5-9B.
- `colab/qwen35_9b_colab.py` — prepare, baseline, smoke, train and benchmark
  runner.
- `configs/qwen3.5-9b-colab-a100.yaml` — one-A100 pilot config.
- `VPS_CHECKLIST.md` — provider checks and stop conditions.
- `benchmarks/ukrainian-llm-leaderboard` — pinned evaluation repository at
  commit `74d8069`.

## VPS order

```bash
git clone https://github.com/MaxBetov-pdd/zoria-v0-training.git zoria
cd zoria
bash scripts/bootstrap_vps.sh
source .venv/bin/activate
bash scripts/preflight_4xa100.sh
bash scripts/download_and_prepare.sh
bash scripts/run_4xa100_baseline.sh
bash scripts/run_4xa100_smoke.sh
```

Do not start the 300-step run until the baseline and smoke test complete.
Do not download the full Kobza or FinePDFs corpus for v0.

The same kit is available as a [ZIP release](https://github.com/MaxBetov-pdd/zoria-v0-training/releases/latest).
