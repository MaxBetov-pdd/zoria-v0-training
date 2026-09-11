# Данные MamayLM и Lapa: что взять для Zoria

## 1. MamayLM

### MamayLM v0.1

- Gemma 2 9B.
- Украинская адаптация через continual training и instruction tuning.
- Использовались украинские и английские данные, смешивание, synthetic data и model merging.
- Полный список и полный pipeline в первоначальном анонсе раскрыты ограниченно.

### MamayLM v1.0

Предобучение:

- Kobza;
- FineWeb2;
- Malyuk/UberText-связанные украинские данные;
- CulturaX и другие корпуса, вошедшие в Kobza;
- Ukrainian Wikipedia для rehydration;
- English Wikipedia, SmolTalk и Mixture of Thoughts для replay.

Instruction/SFT:

- Nemotron SFT и Post-Training;
- OpenCoder/OPC;
- Aya Collection;
- Spivavtor;
- UAlpaca;
- UA-SQuAD;
- Ukrainian StackExchange;
- Crimean Tatar parallel corpora;
- UA-Lawyer QA;
- synthetic QA по украинской истории и культуре;
- переведённые англоязычные instruction datasets.

Процесс:

- большой multilingual/украинский corpus;
- best-fit packing;
- две pretraining data части;
- model soup/merging;
- отдельные Ukrainian-focused и English-focused instruction branches;
- финальное merging.

### MamayLM v2.0

Предобучение:

- Kobza;
- FinePDFs, ориентированный на документы;
- exact deduplication;
- fuzzy deduplication;
- JQL quality filtering поверх Snowflake embeddings;
- 58M документов и 85B токенов после curation;
- rehydration: повтор высококачественных документов;
- English Wikipedia, SmolTalk и Mixture of Thoughts как replay;
- всего около 105B training tokens.

Построение sequences:

- embeddings документов;
- 320 semantic centroids;
- документы одного кластера пакуются вместе;
- best-fit packing;
- это уменьшает смешивание несвязанных документов и помогает long-context.

Instruction/SFT:

- прежний набор Nemotron, OpenCoder, Aya и украинских источников;
- Hermes 3 в украинском переводе;
- FiftyFive-Best от Lapa;
- усиленные translation, chat и function calling данные.

## 2. Lapa

### Lapa pretraining corpus

До фильтрации около 60B токенов. Основной состав:

| Источник | Документы | Токены |
|---|---:|---:|
| CulturaX Ukrainian | 24,942,577 | 15.00B |
| FineWeb2 Ukrainian | 32,124,035 | 19.11B |
| HPLT 2.0 Ukrainian | 26,244,485 | 20.71B |
| UberText 2.0 | 6,431,848 | 2.90B |
| Ukrainian News | 7,175,971 | 1.85B |
| Всего | 96,918,916 | 59.58B |

Дополнительно использовались:

- Institutional Books от Harvard Law School Library/Open Data Initiative;
- Ukrainian News для актуального контекста;
- украинская часть YODAS2, расшифрованная Whisper, для speech text;
- другие public/licensed datasets для культурного и исторического контекста.

### Умная выборка Lapa

Lapa не размечала вручную 60B токенов. Они сделали перенос quality classifiers:

1. случайно выбрали 500K украинских документов;
2. перевели их на английский сильной моделью;
3. оценили английскими quality models;
4. обучили украинские классификаторы на этих pseudo-labels;
5. проверили перенос на отложенных 10%;
6. применили ensemble scores к большому украинскому корпусу.

Оцениваемые свойства:

- educational value: FineWeb-Nemotron-Edu, F1 0.96;
- educational value: FineWeb-Mixtral-Edu, transferred F1 0.94;
- informational value: FastText-OH-ELI5, transferred F1 0.67;
- propaganda/factuality: украинский VoxCheck-based classifier, F1 0.96;
- manipulative content: classifier на UNLP 2025 data, F1 0.74;
- grammatical correctness: classifier на Ukrainian GEC data, F1 0.71.

До этого применялись language identification, Unicode normalization, exact/fuzzy deduplication и эвристики по доле символов, ссылок, чисел и пробелов. После комбинации score документы получили quality buckets 0–20. Низкокачественные документы удалили. Корпус сократился примерно с 60B до 30B токенов.

Во время CPT:

- первые 70% — regular-quality data;
- последние 30% — high-quality data.

Это и есть самая полезная для Zoria идея Lapa: **лучше небольшой отобранный корпус, чем большой сырой корпус**.

### Lapa instruction data

Опубликованный training config использует:

- `lapa-llm/hermes3-uk` — украинские conversations;
- `lapa-llm/hermes3-en-fixed` — английский replay;
- `lapa-llm/antipropaganda-safe` — безопасные ответы и factuality;
- `lapa-llm/wiki-instruction-dialogs` — Wikipedia-based instructions;
- `lapa-llm/lapa-persona-qa` — persona/identity QA;
- `lapa-llm/ua-lawyer` — украинский legal QA;
- `lapa-llm/lang-uk-fiction-gec-dialogs` — fiction/GEC/dialogues;
- `lapa-llm/fiftyfive-best` — отобранный parallel translation data;
- `lapa-llm/openthoughts_no_think` — reasoning/coding without forcing visible chain-of-thought;
- `lapa-llm/summaries` — summaries from Wiki facts;
- `lapa-llm/wiki-facts-conversations` — factual conversations;
- `lapa-llm/impossible-questions` — handling unanswerable questions.

В paper также указаны исходные task families:

- UA-GEC;
- NER-UK 2.0;
- UberText-NER-Silver;
- UA-Lawyer;
- FiftyFiveShades;
- LeetCode;
- Hermes-3;
- translated high-quality English instruction datasets;
- synthetic document-grounded QA;
- synthetic summarization, knowledge extraction и structured lists;
- rule-based grammatical error correction.

### Самые полезные маленькие выборки

1. **FiftyFive-Best:** deduplicated English–Ukrainian parallel corpus; каждая пара оценивалась шестью quality estimation models; Lapa использовала только top 10%. Это хороший готовый источник для translation.
2. **Document-grounded synthetic QA:** более 1.3M вопросов по украинским документам; типы — open QA, yes/no, multiple-choice, extraction, lists, summary, comparison и unanswerable questions.
3. **Antipropaganda-safe:** полезен для factuality и аккуратных ответов, но его долю нельзя делать слишком большой.
4. **Spivavtor/UA-GEC:** помогает естественности, грамматике и редактированию.
5. **Wiki facts/conversations:** дешёвый способ добавить украинские знания и культуру.

## 3. Что берём в Zoria MVP

### Берём сразу

- `hermes3-uk`;
- `fiftyfive-best`;
- `wiki-facts-conversations`;
- `summaries`;
- `impossible-questions`;
- `antipropaganda-safe` в небольшой доле;
- Spivavtor/UA-GEC;
- небольшой английский replay;
- небольшой code/math replay.

### Не скачиваем сразу

- весь Kobza;
- весь FinePDFs;
- 30B-token filtered corpus;
- полный 1.3M synthetic QA;
- все 25 datasets без проверки схемы и лицензии.

Сначала скачиваем/стримим небольшую выборку, нормализуем, удаляем дубликаты и вручную проверяем качество.

## 4. Практический рецепт smart sampling для Zoria

Для первых экспериментов не нужно обучать собственную Lapa-quality модель. Достаточно:

1. взять 50–200K украинских документов из Kobza или другой разрешённой коллекции;
2. удалить точные дубликаты и слишком короткие/мусорные документы;
3. посчитать язык и простые quality signals;
4. использовать готовую quality estimation модель Lapa, если её лицензия подходит;
5. оставить верхние 20–40% по качеству;
6. добавить 10–20% task-oriented instruction data;
7. добавить 10–20% English/code/math replay;
8. обучить QLoRA и сравнить с необработанной выборкой.

Нужно сохранить оба варианта:

```text
random/mixed sample
quality-filtered sample
```

Иначе мы не докажем, что умная выборка реально помогла.

## Источники

- [MamayLM v1](https://blog.mamaylm.insait.ai/index.html)
- [MamayLM v2](https://models.mamay.ai/blog/mamaylm-v2-release-en/)
- [Lapa GitHub](https://github.com/lapa-llm/lapa-llm)
- [Lapa paper](https://aclanthology.org/2026.unlp-1.14/)
- [Lapa training config](https://github.com/lapa-llm/lapa-llm/blob/main/training/lapa-12b-instructions.yml)
- [Lapa quality classifiers](https://github.com/lapa-llm/lapa-llm/tree/main/pretraining/quality-classifiers)
- [Lapa Hugging Face collection](https://huggingface.co/collections/lapa-llm/lapa-v012-release)
