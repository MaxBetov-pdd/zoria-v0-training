# Zoria v0: точная спецификация обучения

## Что обучаем

```text
Qwen/Qwen3.8-Flash-Next
→ 4-bit QLoRA
→ text-only Ukrainian SFT
→ небольшой English replay
→ один adapter Zoria v0
```

Мы не меняем tokenizer, Engram table, vision encoder или архитектуру Qwen. Base model остаётся замороженной; обучаются только LoRA-параметры.

## Что именно входит в датасет

Подготовщик `scripts/prepare_mvp_data.py` делает фиксированный набор с seed 42. Лимиты задаются по источникам:

| Source | Cap | Роль |
|---|---:|---|
| `lapa-llm/hermes3-uk` | 15,000 | основной украинский instruction/QA/reasoning mix |
| `lapa-llm/fiftyfive-best` | 15,000 | лучший слой translation data Lapa |
| `lapa-llm/wiki-facts-conversations` | 8,000 | знания, культура, factual QA и summary |
| `lapa-llm/impossible-questions` | 4,000 | uncertainty и unanswerable questions |
| `lapa-llm/antipropaganda-safe` | 3,000 | factuality/safe response; малая доля |
| `grammarly/spivavtor` | 8,000 | grammar, paraphrase, simplification, coherence |
| `FIdo-AI/ua-squad` | 5,000 | context-grounded Ukrainian QA |
| `lapa-llm/hermes3-en-fixed` | 7,000 | English/code/math/general replay |

Ориентир — около 60–65 тысяч уникальных examples после дедупликации и около 5–15M реальных токенов. Точное число токенов фиксируется после токенизации и записывается в manifest/logs; заранее подменять его оценкой нельзя.

`lapa-llm/openthoughts_no_think` сознательно не входит в primary run: в нём есть специальный system prompt про подробный Thought/Solution формат. Сначала сохраняем исходный reasoning Qwen через English/code/math replay из `hermes3-en-fixed`, а затем отдельно проверяем reasoning retention.

## Как строится train file

- публичные datasets читаются streaming-режимом;
- разные схемы приводятся к OpenAI Messages;
- удаляются exact duplicates по SHA-256 нормализованных messages;
- 2% отправляются в `zoria_v0_audit_holdout.jsonl` и не обучаются;
- остальное попадает в `zoria_v0_train.jsonl`;
- train перемешивается детерминированно;
- official leaderboard data не загружается в train.

## Порядок запусков

### 1. Baseline

До обучения запускается vanilla Qwen на выбранном подмножестве Ukrainian leaderboard и retention set. Сохраняются версии checkpoint, transformers/vLLM/lm-eval, chat template, precision и raw samples.

### 2. 4xA100 smoke

`configs/qwen3.8-flash-next-qlora-4xa100-smoke.yaml`:

- 20 optimizer steps;
- sequence length 2,048;
- LoRA rank 64;
- QLoRA 4-bit;
- `ple_cpu_offload: true`.

Успех smoke — модель загружается, loss считается, checkpoint записывается и повторно загружается, все четыре GPU используются, нет NaN/OOM/collective hang.

### 3. Primary pilot

`configs/qwen3.8-flash-next-qlora-4xa100-smoke.yaml`:

- 300 optimizer steps;
- sequence length 4,096;
- LoRA rank 64;
- learning rate `1e-4`;
- one epoch cap through `max_steps`;
- 2% validation split from train;
- checkpoints every 100 steps.

### 4. Ablation

На тех же данных и с теми же шагами запускаются rank 32 и rank 128. Нельзя одновременно менять rank, data mix, learning rate и sequence length: иначе результат нельзя интерпретировать.

### 5. Selection

Выбирается не checkpoint с минимальным train loss, а вариант с лучшим соотношением:

```text
Ukrainian leaderboard gain
− English/code/math regression
− Ukrainian hallucination/format regression
```

### 6. Финальный v0

Только лучший rank/mix запускается на полном training budget. После этого публикуются adapter, merged checkpoint если merge технически проходит, config, data manifest, evaluation logs и limitations.

## Что проверять

Украинский subset:

- FLORES en→uk;
- LongFLORES en→uk;
- WMT en↔uk;
- XLSUM uk;
- Belebele uk;
- ZNO language/history/math;
- GSM8K uk;
- IFEval uk;
- Winogrande uk.

Retention:

- English instruction;
- code;
- math;
- reasoning;
- basic refusal/format behaviour.

Отдельно держим audit holdout и не используем его для выбора во время промежуточных экспериментов.

## Решение по серверу

### `4×A100 80GB / 384GB RAM / 800GB SSD`

| Компонент | Оценка |
|---|---|
| 4×A100 | потенциально достаточно для распределённого 4-bit QLoRA, но путь experimental |
| 384GB RAM | достаточный запас для PLE/Engram CPU offload и FSDP |
| 800GB SSD | достаточный рабочий запас для исходных весов, кешей и outputs |

В репозитории Axolotl текущая Flash-Next интеграция проверена на 1×B300; multi-GPU отдельно не тестировался. Поэтому этот VPS нельзя оплачивать сразу на месяц без короткого smoke test.

По manifest Hugging Face 131 safetensors-файл модели занимают примерно 360GB decimal / 335.3GiB. Axolotl также указывает около 329.6GiB BF16 weights. На номинальном диске 400GB останется слишком мало места для временных файлов, окружения, HF cache, processed data и checkpoints при обычном NF4 QLoRA пути.

### Почему выбираем large

Минимальная безопасная конфигурация для выбранного large:

```text
4×A100 80GB
384GB RAM
800GB NVMe
```

Medium не выбираем: 400GB слишком тесны для полного исходного checkpoint и
кешей. Xlarge и 8×H100 оставляем для будущих full-weight/distributed
экспериментов.
