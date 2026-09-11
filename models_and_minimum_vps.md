# Zoria: модели и минимальный VPS

**Дата:** 11 сентября 2026  
**Назначение:** практическая проверка выбора foundation model и инфраструктуры для Zoria v0/v1.

## Вывод

Для первой версии Zoria разумно оставить **Qwen3.8-Flash-Next** как основной foundation model и обучать только адаптеры через QLoRA. Обычный CPU VPS для этого не подходит. Минимальная рабочая конфигурация — **одна NVIDIA H200 141 GB, 256 GB RAM и 1 TB NVMe**, но для комфортной работы лучше брать **384 GB RAM, 2 TB NVMe и 28+ vCPU**.

Это позволяет делать baseline, 4-bit QLoRA, небольшое continued pretraining, SFT и eval. Полное обучение всех 125B параметров, полное обучение Engram-таблицы и крупные distributed эксперименты требуют кластера из нескольких GPU.

## Рекомендуемые модели

| Роль | Модель | Решение |
|---|---|---|
| Основной foundation | Qwen3.8-Flash-Next | Брать для Zoria v0/v1 |
| Малый pipeline/debug model | Qwen3.8-27B или аналогичный 20–32B класс | Использовать для проверки данных, eval и SFT-пайплайна |
| Большая исследовательская модель | Qwen3.8-Flash-Next full-weight | Только при доступе к multi-GPU кластеру |
| Baseline для сравнения | Vanilla Qwen3.8-Flash-Next | Обязателен для измерения украинского прироста и forgetting |

### Почему Qwen3.8-Flash-Next

По официальной документации NVIDIA модель имеет 125B параметров backbone, около 6B активных параметров на токен, 51.2B Engram n-gram table, 512 routed experts, 48 decoder layers и native context 262,144 токена. В NeMo сейчас подтверждён language-only SFT path; проверенная конфигурация в документации валидировалась на 4,096 токенах.

Большой Engram table важен для инфраструктуры: модель нельзя оценивать только по числу активных MoE-параметров. Нужны GPU VRAM, host RAM и быстрый локальный диск.

## Минимальная инфраструктура

### Вариант A: обязательный минимум для Zoria v0

- 1 × NVIDIA H200 141 GB HBM3e;
- 256 GB system RAM;
- 1 TB NVMe SSD;
- 16 vCPU;
- Linux x86_64, CUDA-совместимый драйвер;
- локальный NVMe для checkpoint/cache;
- сетевой диск только для backup, не для активного dataset/cache.

Это нижняя граница. Перед арендой нужно выполнить smoke test на 2K–4K context и проверить фактический peak VRAM.

### Вариант B: рекомендуемый рабочий VPS

- 1 × NVIDIA H200 141 GB;
- 384–512 GB RAM;
- 2 TB NVMe SSD;
- 24–32 vCPU;
- минимум 10 Gbit/s network;
- persistent volume для dataset и checkpoints;
- возможность остановить и возобновить инстанс без потери диска.

Такая конфигурация соответствует плану из исходного исследования: около 387 GB RAM, 2.06 TB SSD и 28 vCPU.

### Вариант C: дешёвая машина только для подготовки

- 32–64 vCPU;
- 128–256 GB RAM;
- 1–2 TB NVMe;
- без GPU или с небольшой GPU.

Подходит для сбора/очистки данных, dedup, токенизации, benchmark harness и подготовки shards. Не подходит для обучения Flash-Next.

## Что реально помещается на одной H200

### Реалистично

- inference и baseline evaluation;
- 4-bit QLoRA;
- adapters rank 32/64/128, затем pilot rank 256;
- Ukrainian CPT на 5–300M токенов при контроле eval;
- SFT примерно 20–50M качественных токенов;
- ограниченные preference/verifiable эксперименты;
- контексты 8K–16K как основной режим и отдельные long-context прогоны.

### Не считать минимальным VPS

- full-parameter CPT всех 125B backbone weights;
- полное обучение 51.2B Engram table;
- большой Branch-and-Merge с несколькими full-weight branches;
- meaningful full-weight training на 262K context;
- крупный RL/GRPO rollout cluster.

NVIDIA описывает full training для этой архитектуры на существенно более крупной distributed-конфигурации; одна H200 предназначена для адаптации и экспериментов, а не для полного переобучения модели.

## Практический training plan

1. Сначала скачать/проверить vanilla checkpoint и сделать baseline.
2. Запустить 2K–4K smoke test на 1–2 часа.
3. Сравнить LoRA rank 32, 64 и 128 на 5–10M токенов.
4. Проверять украинские метрики вместе с English, code, math и instruction retention.
5. Только при положительных кривых продолжать до 50–100M, затем максимум до 200–300M токенов.
6. После CPT сделать небольшой качественный SFT и verifiable post-training.

Для Flash-Next нельзя автоматически использовать blanket LoRA targeting из конфигураций для обычных Llama/Qwen-моделей. Axolotl указывает явные цели для Gated DeltaNet и предупреждает о несовместимости blanket `lora_target_linear: true` из-за QSA indexer projection.

## Оценка стоимости GPU

Публичные агрегаторы на 10 сентября 2026 показывали single-H200 предложения примерно от **$1.98/GPU-hour** на marketplace до около **$4–5/GPU-hour** для более типичных on-demand предложений. Цену нужно проверять непосредственно перед арендой: spot-инстансы могут исчезать, а диски и egress часто оплачиваются отдельно.

Грубая оценка:

- smoke test 10 часов: около $20–50 за GPU-время;
- 100 часов: около $200–500;
- 500 часов: около $1,000–2,500;

Это не включает storage, CPU/RAM, network egress и возможные простои.

## Решение для Zoria

**Сейчас:** арендовать не обычный VPS, а GPU instance с 1 × H200, 384 GB RAM, 2 TB NVMe и persistent storage.  
**До аренды:** подготовить eval harness, contamination firewall и первые 20–50M чистых украинских токенов.  
**Не делать на MVP:** tokenizer surgery, full-weight training и Engram ownership experiments.

## Источники

- [NVIDIA NeMo AutoModel: Qwen3.8-Flash-Next](https://docs.nvidia.com/nemo/automodel/model-coverage/large-language-models/qwen/qwen3-8-flash-next)
- [NVIDIA configuration reference](https://docs.nvidia.com/nemo/automodel/v0.4/nemo-automodel/nemo_automodel/components/models/qwen3_8_flash_next/config)
- [Qwen3.8-Flash-Next model card](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)
- [Axolotl Qwen3.8-Flash-Next fine-tuning documentation](https://docs.axolotl.ai/docs/models/qwen3.8-flash-next.html)
- [Current H200 marketplace snapshot](https://gpufinder.dev/gpu/h200)
- [H200 on-demand price snapshot](https://www.priceofcompute.com/gpus/h200-sxm)

## Ограничения

- Фактическая VRAM и throughput зависят от версии Transformers/NeMo/Axolotl, quantization, sequence length, batch size и CPU offload.
- Цены GPU быстро меняются и являются ориентиром, а не коммерческим предложением.
- Формулировка «минимальный VPS» относится к Zoria v0/v1 через QLoRA; для full-weight обучения она неприменима.

