# Zoria Colab pilot

This pilot uses the newest practical Qwen checkpoint that fits a 16GB Colab
GPU:

```text
Qwen/Qwen3.5-9B
BitsAndBytes NF4
QLoRA
text-only Ukrainian SFT
```

It is an intermediate experiment. The Lambda target remains
`Qwen/Qwen3.8-Flash-Next`.

## Colab setup

Create a new Colab notebook and select a GPU runtime. Then run:

```python
!git clone https://github.com/MaxBetov-pdd/zoria-v0-training.git /content/zoria
%cd /content/zoria
!pip install -U "axolotl[deepspeed]==0.18.0" torchvision "datasets<4" tqdm pyyaml langdetect immutabledict
!pip install -U "cut-cross-entropy[transformers] @ git+https://github.com/axolotl-ai-cloud/ml-cross-entropy.git@4dfa522"
!pip install -U "lm-eval[hf]"
```

Check the GPU:

```python
!nvidia-smi
```

Run the pilot helper:

```python
!python colab/qwen35_9b_colab.py --mode prepare
!python colab/qwen35_9b_colab.py --mode baseline
!python colab/qwen35_9b_colab.py --mode smoke
!python colab/qwen35_9b_colab.py --mode train
!python colab/qwen35_9b_colab.py --mode benchmark-adapter
```

The helper uses a small cap of 3,000 examples per source. It does not add the
official leaderboard test questions to training.

`baseline` and `benchmark-adapter` are quick smoke evaluations with 50
examples per task. For the complete active leaderboard group, run:

```python
!python colab/qwen35_9b_colab.py --mode baseline-full
!python colab/qwen35_9b_colab.py --mode benchmark-adapter-full
```

The full modes use `--tasks ukrainian_bench` without `--limit`. They include
all active tasks from the pinned leaderboard repository. Some repository
tasks remain intentionally disabled upstream, such as SQuAD, Hellaswag and
the TODO MMLU task; they are not silently replaced by the smoke subset.

## One-A100 runtime

If Colab gives you an A100, use the larger pilot:

```python
!nvidia-smi --query-gpu=name,memory.total --format=csv
!python colab/qwen35_9b_colab.py --mode baseline
!python colab/qwen35_9b_colab.py --mode a100-smoke
!python colab/qwen35_9b_colab.py --mode a100-train
```

The A100 config uses 2,048-token sequences, BF16 and 300 steps. Start with
the smoke test before spending the remaining runtime.

For evaluation, the helper loads the 9B base in FP16 because an A100 has
enough VRAM and the current `lm-eval` Transformers backend does not accept
`load_in_4bit` directly for Qwen3.5. Training still uses NF4 QLoRA.

## If the 16GB runtime runs out of memory

Run the smoke config first. If it still fails:

1. change `sequence_len` from 512 to 384;
2. keep `lora_r: 16`;
3. keep `micro_batch_size: 1`;
4. keep sample packing disabled;
5. use Qwen3.5-4B as the fallback.

The Qwen3.5 family has official Axolotl QLoRA support and uses hybrid
Gated DeltaNet plus standard attention. The Colab run is text-only; vision
training is reserved for a later version.

The leaderboard repository still contains a legacy `xlsum.py` dataset script,
so the Colab environment intentionally pins `datasets<4`.
