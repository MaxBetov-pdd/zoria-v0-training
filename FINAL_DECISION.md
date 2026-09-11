# Zoria v0: финальное решение перед арендой

## Выбранный метод

```text
Base: Qwen/Qwen3.8-Flash-Next
Quantization: BitsAndBytes NF4 4-bit
Training: QLoRA
Stage: text-only SFT
Adapter: LoRA rank 64
```

Базовые веса заморожены. Обучаются только LoRA-матрицы. Tokenizer,
Engram/PLE table, vision encoder и архитектура не меняются.

## Точная конфигурация

- `load_in_4bit: true`
- `quantize_moe_experts: true`
- `ple_cpu_offload: true`
- `attn_implementation: sdpa`
- Cut Cross Entropy
- sequence length: 4,096
- micro batch: 1
- gradient accumulation: 4
- learning rate: `1e-4`
- LoRA rank: 64
- LoRA alpha: 128
- dropout: 0.05
- maximum: 300 optimizer steps
- seed: 42

Целевые modules и parameters уже записаны в
`configs/qwen3.8-flash-next-qlora-4xa100.yaml` и соответствуют официальному
Axolotl recipe для этой архитектуры.

## Точная модель данных

Подготовщик использует фиксированные caps:

- `lapa-llm/hermes3-uk`: 15,000;
- `lapa-llm/fiftyfive-best`: 15,000;
- `lapa-llm/wiki-facts-conversations`: 8,000;
- `lapa-llm/impossible-questions`: 4,000;
- `lapa-llm/antipropaganda-safe`: 3,000;
- `grammarly/spivavtor`: 8,000;
- `FIdo-AI/ua-squad`: 5,000;
- `lapa-llm/hermes3-en-fixed`: 7,000.

После дедупликации получается примерно 60–65 тысяч messages и ориентировочно
5–15M токенов. 2% сохраняются как audit holdout и не участвуют в обучении.

`openthoughts_no_think`, Kobza, FinePDFs и official leaderboard questions в
первый run не входят.

## Точный порядок

### 1. Подготовить VPS

Из доступных вариантов выбираем:

```text
4 × NVIDIA A100 80GB
384GB RAM
800GB SSD
48 vCPU
```

Это минимальный вариант из предложенных, где есть разумный запас по RAM и
диску для распределённого 4-bit QLoRA. Сам multi-GPU Flash-Next остаётся
экспериментальным, поэтому перед длинным запуском обязателен smoke test.

### 2. Установить окружение

```bash
bash scripts/bootstrap_vps.sh
source .venv/bin/activate
```

### 3. Проверить машину

```bash
bash scripts/preflight_4xa100.sh
```

### 4. Подготовить данные

```bash
bash scripts/download_and_prepare.sh
```

### 5. Запустить baseline оригинального Qwen

```bash
bash scripts/run_original_baseline.sh
```

Результаты baseline сохраняются отдельно от результатов Zoria. Сначала
запускается небольшой subset, затем полный `ukrainian_bench`.

### 6. Сделать 20-step smoke test

```bash
bash scripts/run_4xa100_baseline.sh
bash scripts/run_4xa100_smoke.sh
```

Проверить loss, отсутствие NaN/OOM и повторную загрузку checkpoint.

### 7. Запустить основной pilot

```bash
bash scripts/run_4xa100_primary.sh
```

### 8. После pilot

Запустить rank 32 и rank 128 на тех же данных. Выбрать checkpoint по
украинскому приросту при минимальной деградации English/code/math/reasoning.

## Решение по предложенному серверу

```text
4 × A100 80GB
48 vCPU
384GB RAM
800GB SSD
```

### Для выбранного надёжного пути

**Подходит условно**, потому что:

- Axolotl не тестировал Flash-Next multi-GPU;
- 4xA100 дают достаточный суммарный запас VRAM для FSDP-шардинга;
- 800GB дают место для исходных safetensors, кеша, данных и adapter outputs;
- стандартный NF4 QLoRA всё равно сначала скачивает исходный checkpoint.

### Medium

Не выбираем: 400GB слишком тесны, а margin по памяти меньше.

### Xlarge

Для MVP не нужен: 768GB RAM и 1.6TB SSD не дадут пропорционального прироста
по сравнению с large.

### 1x H100

Не подходит: 80GB VRAM недостаточно для выбранного Flash-Next QLoRA recipe.

### 8x H100

Технически избыточен для MVP и стоит намного дороже. Он понадобится только
для будущего full-weight/distributed research, а не для текущей QLoRA.
