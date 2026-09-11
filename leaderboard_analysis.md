# Украинский leaderboard и eval-план Zoria

**Дата:** 11 сентября 2026  
**Источник:** официальный Ukrainian LLM Leaderboard Минцифры / WINWIN AI / УКУ / lang-uk.

## Что найдено

Минцифры Украины объявило Ukrainian LLM Leaderboard как национальный рейтинг качества работы моделей с украинским языком. Платформа создана WINWIN AI при Минцифры вместе с Украинским католическим университетом и инициативой lang-uk.

- Публичная страница: https://lang.org.ua/en/produkty/servisi/ukrainian-llm-leaderboard/
- Официальная публикация Минцифры: https://thedigital.gov.ua/news/shtuchnyy-intelekt/zapuskayemo-pershyy-natsionalnyy-reytynh-shi-modeley-za-rivnem-volodinnia-ukrayinskoiu
- Код evaluation и leaderboard: https://github.com/lang-uk/ukrainian-llm-leaderboard
- Результаты: https://huggingface.co/datasets/lang-uk/ukrainian-llm-leaderboard-results

Репозиторий уже склонирован в:

```text
zoria/benchmarks/ukrainian-llm-leaderboard
```

Зафиксированный commit:

```text
74d8069
```

## Какие способности измеряются

### 1. Перевод

Задачи:

- FLORES-200: English ↔ Ukrainian и дополнительные языковые пары;
- LongFLORES: перевод более длинных фрагментов;
- WMT-22: English ↔ Ukrainian.

Что это показывает для Zoria:

- знание украинской грамматики и лексики;
- устойчивость на предложениях и абзацах;
- сохранение смысла;
- качество двунаправленного перевода.

Это один из главных критериев для Ukrainian CPT. Для оценки нужны BLEU/chrF или метрики, заданные конкретной задачей. Нельзя сводить результат к одному общему score.

### 2. Суммаризация

Задача:

- XLSUM Ukrainian.

Что проверяет:

- сжатие длинного текста;
- сохранение фактов;
- естественность украинского;
- способность работать с новостным стилем.

Leaderboard прямо рассматривает summarization вместе с QA как proxy для RAG. Для Zoria это полезнее, чем обычный разговорный тест, потому что проверяет работу с исходным документом.

### 3. Question answering и comprehension

Задачи:

- Belebele Ukrainian;
- SQuAD Ukrainian;
- TriviaQA Ukrainian.

В текущем YAML репозитория `squad_uk` закомментирован, а Belebele и TriviaQA активны.

Проверяем:

- извлечение ответа из контекста;
- reading comprehension;
- закрытые и открытые вопросы;
- отсутствие выдуманных деталей;
- полезность для RAG.

### 4. Знания и reasoning

Задачи:

- ZNO-Eval: geography, history, Ukrainian language and literature, math;
- ARC Easy и ARC Challenge;
- Winogrande;
- Hellaswag;
- MMLU Ukrainian;
- TriviaQA.

В текущей конфигурации активны ZNO, ARC, Winogrande и Ukrainian MMLU aggregate через `global_mmlu_full_uk`. Hellaswag пока закомментирован.

Это измеряет знания, логический выбор ответа и reasoning. Но эти тесты нельзя считать чистой проверкой украинского языка: часть результата зависит от общих знаний и способности решать multiple-choice задачи.

### 5. Математика

Задачи:

- GSM8K Ukrainian;
- ZNO math.

Проверяем:

- понимание украинской формулировки;
- многошаговое решение;
- правильный финальный ответ;
- сохранение reasoning после украинского CPT.

Нужно отдельно хранить exact answer accuracy и полный ответ модели. Если проверять только текстовым совпадением, можно потерять правильные ответы из-за форматирования.

### 6. Instruction following

Задача:

- IFEval Ukrainian.

Проверяет соблюдение формальных ограничений: формат, количество пунктов, наличие требуемых слов, структуру ответа и другие проверяемые инструкции.

Для Zoria это важная защита от регрессии после CPT. Украинский текст может стать лучше, но instruction following ухудшиться.

### 7. Fairness

В README репозитория указаны:

- StereoSet-UK Eval;
- WinoBias-UK Natural;
- BBQ-UK;
- WinoPron-UK.

Эти тесты находятся в отдельной fairness-части результатов. Они измеряют проявляемые моделью социальные bias patterns, coreference и ответы на bias-вопросы. Это не полный alignment-тест и не доказательство безопасности модели.

## Текущий состав запуска

В `tasks/ukrainian_bench/ukrainian_bench.yaml` сейчас включены:

- Belebele Ukrainian;
- Ukrainian MMLU aggregate;
- TriviaQA Ukrainian;
- ARC Easy/Challenge Ukrainian;
- XLSUM Ukrainian;
- FLORES Ukrainian;
- LongFLORES Ukrainian;
- WMT English–Ukrainian;
- ZNO geography/history/language and literature/math;
- GSM8K Ukrainian;
- Winogrande Ukrainian;
- IFEval Ukrainian.

Закомментированы или отложены:

- SQuAD Ukrainian;
- Hellaswag Ukrainian;
- отдельная MMLU task group;
- часть дополнительных fairness и visual/alignment направлений.

## Как использовать это для Zoria

### До обучения

Снять baseline vanilla Qwen на полном наборе активных задач. Зафиксировать:

- commit репозитория;
- версии `lm-eval`, `vllm`, `datasets`;
- checkpoint revision;
- quantization;
- prompt/chat template;
- temperature, top-p и max tokens;
- GPU и batch size;
- raw logs и samples.

### После каждого CPT checkpoint

Минимальный быстрый gate:

1. FLORES en→uk;
2. XLSUM uk;
3. Belebele uk;
4. ZNO language/history/math;
5. GSM8K uk;
6. IFEval uk;
7. Winogrande uk;
8. короткий English/code/math retention set.

Полный leaderboard запуск делать для baseline, лучшего CPT checkpoint и финального SFT checkpoint.

### Что должно считаться успехом

Нельзя оптимизировать только общий средний score. Для каждого checkpoint хранить в таблице:

```text
UA translation
UA summarization
UA QA
UA knowledge/reasoning
UA math
UA instruction following
fairness
English/code/math retention
```

Checkpoint принимается, если украинский gain заметен, а global regression остаётся в заранее заданном допуске.

## Важное ограничение для 4-bit

MVP действительно можно обучать через 4-bit QLoRA. Однако текущий официальный leaderboard README указывает, что оценка quantized models находится в roadmap и ещё не является полностью отдельной стандартизированной веткой.

Поэтому для честного сравнения Zoria нужно сделать два режима:

1. **Official-compatible:** повторить настройки leaderboard максимально точно.
2. **Deployment mode:** отдельно оценить фактический 4-bit inference checkpoint.

В отчёте явно указывать:

```text
base model precision
adapter precision
inference quantization
backend
GPU memory utilization
```

Нельзя сравнивать FP16/BF16 результат одной модели с 4-bit результатом другой и делать вывод только по разнице score.

## Команда для одной H200

Пример README использует `data_parallel_size=2,tensor_parallel_size=2`, то есть рассчитан на четыре GPU. Для одной H200 стартовый вариант должен быть примерно таким:

```bash
VLLM_WORKER_MULTIPROC_METHOD=spawn lm_eval \
  --model vllm \
  --model_args pretrained=MODEL,data_parallel_size=1,tensor_parallel_size=1,gpu_memory_utilization=0.90,add_bos_token=True \
  --tasks ukrainian_bench \
  --batch_size auto \
  --output_path ./eval-results \
  --log_samples \
  --include_path ./tasks \
  --apply_chat_template
```

Для Flash-Next конкретные параметры модели, thinking tokens и chat template сначала нужно проверить на smoke test. Команду из README нельзя запускать без изменений на одной GPU.

## Приоритеты для Zoria v0

1. Baseline vanilla Qwen.
2. FLORES + LongFLORES.
3. XLSUM.
4. Belebele + TriviaQA.
5. ZNO language/history/math.
6. GSM8K Ukrainian.
7. IFEval Ukrainian.
8. Winogrande Ukrainian.
9. Полный leaderboard.
10. Fairness suite и собственный private holdout.

Главные метрики для выбора checkpoint: перевод, суммаризация, QA и IFEval. ZNO/GSM8K/Winogrande нужны как контроль знаний и reasoning, а English/code/math — как защита от catastrophic forgetting.
