# Квантизация Qwen3.8-Flash-Next

## Что означает наш текущий 4-bit MVP

В конфиге Zoria сейчас указано:

```yaml
load_in_4bit: true
adapter: qlora
```

Это BitsAndBytes NF4 QLoRA: базовые веса загружаются в 4-bit, а LoRA
матрицы обучаются в BF16. Такой путь описан в официальном Axolotl guide для
Flash-Next вместе с `quantize_moe_experts: true` и
`ple_cpu_offload: true`.

При этом исходный Hugging Face checkpoint всё равно состоит из больших
safetensors-файлов. Квантизация выполняется при загрузке; она не превращает
уже скачанный полный checkpoint в маленький файл автоматически.

## Можно ли скачать готовый 4-bit checkpoint

Да, но нужно проверить три вещи:

1. **Формат:** NF4, GPTQ, AWQ, NVFP4 и AutoRound — разные форматы.
2. **Обучение:** checkpoint может поддерживать inference, но не LoRA training.
3. **Железо/backend:** часть NVFP4 kernels рассчитана на Hopper/Blackwell,
   а не на A100; часть community checkpoints поддерживает только vLLM,
   SGLang или конкретную версию Transformers.

Найденный официальный NVIDIA `nvidia/Qwen3.8-Flash-Next-NVFP4` — это
готовый quantized checkpoint для deployment. Его нельзя автоматически
подставить в наш NF4 QLoRA config. Для него нужен отдельный NVFP4 MoE-LoRA
эксперимент и проверка поддержки `qwen4_exp` на A100.

## Решение для Zoria v0

Первый воспроизводимый путь остаётся:

```text
исходный Qwen checkpoint
→ Axolotl BitsAndBytes 4-bit QLoRA
→ LoRA adapter
```

Для этого лучше иметь 1TB NVMe.

Вариант с 400GB можно рассматривать только после отдельного smoke test на
конкретном готовом NF4/GPTQ/AWQ/NVFP4 checkpoint. В этом случае он может
поместиться, но это будет уже другой training path, а не автоматически тот
же самый QLoRA эксперимент.
