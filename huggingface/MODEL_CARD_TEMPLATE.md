---
base_model:
- Qwen/Qwen3.5-9B
library_name: peft
language:
- uk
- en
license: apache-2.0
tags:
- ukrainian
- qlora
- peft
- zoria
---

# Zoria Qwen3.5-9B pilot

Zoria is an experimental Ukrainian language adaptation of
`Qwen/Qwen3.5-9B` trained with 4-bit BitsAndBytes QLoRA.

## Training

- Base model: `Qwen/Qwen3.5-9B`
- Method: QLoRA
- Quantization: NF4 4-bit
- LoRA rank: 32
- Context length: 1024
- Dataset manifest: `data/processed-colab/zoria_v0_manifest.json`
- Evaluation commit: `74d8069`

## Scope

This is a Colab-scale pilot. It is not the final Zoria model and is not a
replacement for the planned Qwen3.8-Flash-Next Lambda run.

## Results

Results will be added only after the baseline and adapter evaluation are
completed. Do not add leaderboard claims without raw logs and exact settings.

## License

The base model is distributed under Apache-2.0. Dataset licenses and
attributions are listed in the Zoria repository and must be checked before
redistribution.
