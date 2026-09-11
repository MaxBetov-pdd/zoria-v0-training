# MamayLM: версии, архитектуры и обучение

**Дата исследования:** 11 сентября 2026

## Короткий вывод

MamayLM — не одна архитектура и не один обычный LoRA. Во всех публичных версиях команда брала готовую Gemma и меняла в основном данные и training pipeline:

```text
Gemma base
→ continual pretraining / adaptation
→ instruction tuning
→ отдельные language-focused branches
→ model merging
→ release
```

Архитектурно MamayLM остаётся производной базовой Gemma. Главный вклад — украинские данные, продолженное обучение, перевод/синтетика, replay, merging и evaluation.

## Версии

### MamayLM v0.1 — 2025

**Base:** Gemma 2 9B.

**Архитектура:** обычный decoder-only Transformer Gemma 2, text-only.

**Что делали:**

- адаптировали Gemma 2 под украинский;
- использовали continual training и instruction tuning;
- применяли улучшения в data mixing и model merging;
- добавляли synthetic data;
- сохраняли английские способности;
- выпустили обычную и quantized версии.

На официальном анонсе подробный полный training recipe не раскрыт так подробно, как для v1/v2. Версию нужно считать первой proof-of-concept украинской адаптацией, а не новой архитектурой.

### MamayLM v1.0 — 2025

**Base:** Gemma 3 12B; были опубликованы модели на Gemma 3 4B и 12B.

**Архитектура:** Gemma 3 multimodal model. В релизе сохранилась возможность принимать изображения, хотя основной adaptation corpus был текстовым и image-text training не использовался.

**Continual pretraining:**

- большой отфильтрованный украинско-английский корпус;
- украинский web data, Kobza, Wikipedia и специализированные источники;
- разные data mixtures;
- dataset splitting;
- model soup/merging после обучения на разных частях данных;
- сохранение visual и long-context способностей Gemma 3.

**Instruction/SFT:**

- NVIDIA Nemotron SFT и Post-Training;
- OpenCoder/OPC;
- Aya Collection;
- Spivavtor;
- UAlpaca;
- UA-SQuAD;
- Ukrainian StackExchange;
- Crimean Tatar parallel corpora;
- UA-Lawyer QA;
- synthetic/translated instruction data.

**Language branches:**

Команда обучала English-focused и Ukrainian-focused instruction models отдельно, затем объединяла их model merging. Это важная часть рецепта: одна ветка оптимизируется под украинский, другая помогает не потерять английские способности.

**Evaluation:**

ZNO, Winogrande, Hellaswag, ARC, TriviaQA, GSM8K, MMLU, IFEval, украинские переводы стандартных тестов и визуальные тесты. Визуальные способности улучшились без image-text training, по объяснению команды, за счёт усиления language module при сохранении vision pathway.

### MamayLM v2.0 — 2026

**Base:** Gemma 3, размеры 12B и 27B.

**Архитектура:** Gemma 3 multimodal; отдельной новой Mamay attention/MoE архитектуры не заявлено. Основные изменения — масштабы, data pipeline и post-training.

**Continual pretraining corpus:**

- Kobza;
- FinePDFs;
- exact deduplication;
- fuzzy deduplication;
- JQL quality filtering на Snowflake embeddings;
- 58M документов;
- 85B training tokens после curation;
- rehydration/repetition лучших документов;
- English replay: English Wikipedia, SmolTalk, Mixture of Thoughts;
- итоговый CPT — 105B training tokens.

**Sequence construction:**

- clustering документов по 320 semantic centroids;
- sequences строятся из документов одного кластера;
- best-fit packing;
- улучшение long-context и уменьшение cross-document interference.

**Post-training:**

- прежний mixed open-source SFT pipeline;
- synthetic QA;
- translation pipeline;
- Nemotron;
- OpenCoder;
- Aya;
- Spivavtor;
- UAlpaca;
- UA-SQuAD;
- Ukrainian StackExchange;
- Crimean Tatar datasets;
- UA-Lawyer QA;
- translated Hermes 3;
- FiftyFive-Best от Lapa;
- усиление machine translation и function calling.

**Merging:**

Как и в v1, English- и Ukrainian-focused instruction models обучаются отдельно и объединяются в финальную модель через более продвинутый model merging.

**Vision:**

Image-text paired training не добавлялся, но visual scores улучшились за счёт языковой части. Поэтому vision у Mamay v2 — сохранённая и косвенно улучшенная способность Gemma 3, а не отдельный украинский vision pretraining stage.

## Что именно у Mamay является архитектурой

### Реальная модельная архитектура

- v0.1: Gemma 2 9B decoder-only Transformer;
- v1: Gemma 3 4B/12B multimodal architecture;
- v2: Gemma 3 12B/27B multimodal architecture.

### Training architecture/pipeline

Вот где находится основная инженерная работа Mamay:

1. continual pretraining;
2. data mixing;
3. high-quality Ukrainian data;
4. English replay;
5. two language-focused branches;
6. model soup/model merging;
7. synthetic QA;
8. translated datasets;
9. semantic clustering;
10. best-fit packing;
11. quality filtering;
12. contamination-aware evaluation.

## Что имеет смысл перенять Zoria v0

### Обязательно

- готовый сильный foundation Qwen;
- украинский task-oriented dataset;
- небольшой English/code/math replay;
- baseline vanilla Qwen;
- leaderboard evaluation;
- сохранение исходных reasoning abilities.

### Можно добавить после первого LoRA

- два adapters: Ukrainian capability и Ukrainian instruction;
- adapter merge;
- controlled comparison Ukrainian-only против Ukrainian+replay;
- semantic filtering;
- отдельные translation/summarization/QA subsets.

### Оставить для Zoria v1/v2

- крупный continual pretraining;
- полноценные language branches;
- Branch-and-Merge для нескольких полных runs;
- JQL-scale filtering;
- 100B-token corpus;
- tokenizer surgery;
- Engram experiments;
- full-weight training.

## Главный вывод для Zoria

MamayLM v1/v2 показывает, что украинская модель может быть создана без новой neural architecture. Украинизация достигается изменением параметров готовой Gemma через data/training pipeline.

Поэтому Zoria v0 может быть честной украинской адаптацией Qwen даже при LoRA/QLoRA. Разница в масштабе: Mamay v2 — глубокий full-model adaptation project, а Zoria v0 — быстрый parameter-efficient MVP. Это нужно открыто указать в model card.

## Источники

- MamayLM v0.1: https://huggingface.co/blog/INSAIT-Institute/mamaylm
- MamayLM v1.0: https://blog.mamaylm.insait.ai/index.html
- MamayLM v2.0: https://models.mamay.ai/blog/mamaylm-v2-release-en/
- MamayLM v1 model card: https://huggingface.co/INSAIT-Institute/MamayLM-Gemma-3-12B-IT-v1.0
- MamayLM research index: https://models.mamay.ai/research/
