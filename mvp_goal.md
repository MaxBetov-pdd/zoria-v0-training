# Zoria MVP: точная цель

## Цель

Создать украинскую адаптацию Qwen3.8-Flash-Next по аналогии с тем, как MamayLM является украинской адаптацией Gemma 3:

```text
Qwen3.8-Flash-Next
+ небольшой украинский QLoRA/SFT
= Zoria v0
```

Zoria v0 должна немного улучшить результаты на Ukrainian LLM Leaderboard и сохранить исходные способности Qwen в reasoning, code, math, English и instruction following.

Главная цель — не максимальное переобучение и не новая архитектура, а хороший trade-off:

```text
Ukrainian leaderboard gain ↑
Global capability regression ↓
Training cost ↓
```

## Что означает «украинская модель»

В публичном описании корректно говорить:

> Zoria — украинская language adaptation модель Qwen3.8-Flash-Next, обученная на качественных украинских instruction и task-oriented данных.

Нужно открыто указывать базовую модель Qwen, размер LoRA и состав адаптационного датасета. Не заявлять, что Zoria обучена с нуля или имеет новую архитектуру.

## Метод MVP

1. Прогнать vanilla Qwen на leaderboard и retention tests.
2. Загрузить базовую модель в 4-bit.
3. Обучить QLoRA на украинских task-oriented данных.
4. Добавить небольшой English/code/math replay.
5. Не обучать на официальных evaluation questions.
6. Сравнить LoRA rank 32, 64 и 128 на коротких прогонах.
7. Выбрать checkpoint по Ukrainian gain и regression, а не по train loss.
8. Выпустить adapter и merged model.

## Данные MVP

Приоритет:

- украинский translation-style data;
- summarization;
- document-grounded QA;
- reading comprehension;
- Ukrainian grammar and rewriting;
- ZNO-like knowledge questions;
- GSM8K-like math in Ukrainian;
- IFEval-like constrained instructions;
- JSON and tool-call examples.

Ориентировочный mix:

- 70–80% Ukrainian task-oriented data;
- 10–15% English/general instruction replay;
- 5–10% code/math/reasoning replay;
- остальные данные — multilingual/translation retention.

Не переносить в обучение открытые leaderboard test questions. Использовать только разрешённые train splits, похожие synthetic tasks и собственный private holdout.

## Что сознательно не входит в MVP

- full continued pretraining;
- обучение всех Qwen weights;
- tokenizer surgery;
- изменение Engram table;
- Branch-and-Merge full branches;
- RL/GRPO;
- vision/OCR training;
- context training на 128K–262K;
- expert-parallel architecture work.

## Критерий успеха

Zoria v0 принимается, если по сравнению с vanilla Qwen:

- есть измеримый прирост на выбранных Ukrainian leaderboard tasks;
- нет существенного падения на reasoning, code, math и English retention;
- качество украинского заметно лучше в ручной проверке;
- результат воспроизводится из опубликованного конфига;
- adapter и evaluation logs можно предоставить СМИ, исследователям и грантовым комиссиям.

## Порядок экспериментов

### Run 0: baseline

Vanilla Qwen, полный фиксированный eval.

### Run 1: small pilot

QLoRA rank 64, небольшой датасет, короткое обучение. Проверка target modules, VRAM и качества.

### Run 2: rank ablation

Rank 32/64/128 на одинаковых данных и одинаковом количестве шагов.

### Run 3: data mix

Сравнить Ukrainian-only против Ukrainian + replay.

### Run 4: final MVP

Лучший rank и mix, затем leaderboard, retention, manual review и публикация.

### Optional Run 5

Несколько task-specific adapters и аккуратный adapter merge. Добавлять только если Run 4 уже стабильно улучшает украинские метрики.
