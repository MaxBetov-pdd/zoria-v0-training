# Zoria: разделение идей по ресурсам

## Главный принцип

Zoria v0 — это QLoRA/LoRA-украинизация готового Qwen. Архитектурные исследования не должны блокировать первый релиз. Сначала получаем измеримый прирост на украинском leaderboard, затем расширяем обучение по одному направлению и сохраняем сравнение с предыдущей версией.

## Можно сделать сразу

| Идея | Ресурс | Результат |
|---|---|---|
| Baseline vanilla Qwen | 1 GPU; часы | Понимаем стартовый уровень |
| 4-bit QLoRA, rank 32/64/128 | 1 H200; часы/дни | Zoria v0 adapter |
| Ukrainian instruction SFT | 1 H200; дни | Более естественные ответы |
| Небольшой Ukrainian CPT pilot | 1 H200; дни | Проверка, нужен ли raw-text CPT |
| English/code/math replay | Почти бесплатен относительно основного run | Снижение forgetting |
| RAG-style document QA | 1 H200; дни | Практическая полезность на документах |
| JSON/tool calling/verifiable tasks | 1 H200; дни | Проверяемый post-training |
| Rank ablations | 1 H200; дополнительные часы/дни | Выбор лучшего adapter |
| Leaderboard eval и private holdout | CPU/GPU; часы/дни | Доказательства для грантов |

## Дёшево по GPU, но требует подготовки данных

| Идея | Что требуется | Приоритет |
|---|---|---|
| Exact/fuzzy dedup | CPU, RAM, storage | Сразу |
| Contamination firewall | CPU, benchmark copies, hashes/MinHash | Сразу |
| Language/quality filtering | CPU и небольшой judge budget | Сразу |
| Semantic clustering | Embeddings и CPU/storage | После baseline |
| Best-fit/semantic packing | CPU preprocessing | После появления CPT |
| JQL-like quality scoring | Judge model/API и подготовленные labels | После первого рабочего набора |
| Quality curriculum | Разные quality buckets | После первого CPT pilot |

Эти вещи могут заметно улучшить результат без увеличения размера модели. Они требуют инженерного времени, но не большого GPU-кластера.

## Средняя сложность: одна сильная GPU

| Идея | Ограничение |
|---|---|
| 20–100M token CPT | Долго на одной GPU; нужны частые eval и checkpoints |
| 100–300M token CPT | Возможно, только если метрики продолжают расти |
| SFT 20–50M tokens | Реалистично на одной H200 |
| Partial unfreeze отдельных projections | Больше VRAM/нестабильнее QLoRA; делать после v0 |
| Большие LoRA ranks 256+ | Больше VRAM и adapter memory; нужен pilot |
| Context 32K–64K | Существенно медленнее и дороже по memory; отдельный curriculum |
| Несколько специализированных adapters | Обучаются последовательно; больше storage и eval |
| Adapter merge | Само объединение дешёвое; трудно доказать, что merge полезен |
| Малый preference/verifiable stage | Реалистично при автоматической проверке |

## Требует больше времени, но не обязательно больше GPU

### Branch-and-Merge

Можно обучать branches последовательно на одной H200 и потом объединять. Поэтому минимальный технический запуск возможен без кластера, но стоимость растёт примерно с количеством branches:

- Ukrainian general;
- legal/official;
- technical/science;
- instruction/tool use.

Для первого MVP это лишнее. После появления сильного single-adapter можно сделать controlled experiment.

### JQL и сложная фильтрация

JQL-style pipeline можно начать с маленького judge-labelled набора и дешёвого classifier. Полный масштаб Mamay повторять не нужно. Главный расход — подготовка данных, а не VRAM.

### Semantic packing

Это preprocessing: документы кластеризуются, затем пакуются в sequences. Он не меняет архитектуру Qwen и не требует размораживания модели. Его можно добавить уже на первом CPT, если pipeline готов.

## Требует существенно больше VRAM или distributed training

| Идея | Почему отложить |
|---|---|
| Full-weight CPT 125B backbone | Optimizer states, gradients и параметры не помещаются на одной H200 |
| Full training 51.2B Engram | Требует distributed ownership и намного больше памяти |
| Trainable sparse Engram | Может быть следующим research step, но сначала нужно реализовать hit-map и sparse updates |
| Полный Branch-and-Merge full-weight | Каждая branch — дорогой полный training run |
| Expert-parallel experiments | Нужны несколько GPU и routing/distributed infrastructure |
| Meaningful 128K–262K full training | Огромные activation memory и compute time |
| Большой GRPO/RL rollout | Нужны отдельные inference/training ресурсы и много samples |
| Полноценная vision adaptation | Нужны image-text data, preprocessing и дополнительные training runs |

## Tokenizer surgery

Tokenizer surgery не является хорошим следующим этапом после LoRA. В обычной Gemma-модели это уже сложное изменение, а в Flash-Next есть Engram hashing по raw token IDs. Изменение token IDs может нарушить соответствие с pretrained Engram table.

Порядок должен быть таким:

1. измерить tokenizer fertility Qwen на украинском;
2. доказать, что fertility реально ограничивает score;
3. отдельно спроектировать совместимый tokenizer/Engram experiment;
4. проводить только после гранта или сильного исследовательского обоснования.

## Рекомендуемая лестница

### Zoria v0

- vanilla baseline;
- 4-bit QLoRA;
- 1–10M quality Ukrainian instruction tokens;
- rank 32/64/128;
- leaderboard subset;
- private holdout.

### Zoria v0.1

- лучший rank;
- 5–20M mixed instruction data;
- document QA, translation, summarization, JSON, tool use;
- English/code/math retention;
- full leaderboard run.

### Zoria v1

- semantic filtering и packing;
- 20–100M CPT;
- replay mix;
- SFT;
- verifiable post-training;
- optional adapter merge.

### Zoria v1.5

- 100–300M CPT при положительных eval curves;
- partial unfreeze;
- несколько специализированных branches;
- controlled Branch-and-Merge;
- context 32K–64K.

### Zoria v2 после гранта

- multi-GPU CPT;
- full-weight или крупное partial training;
- sparse Engram adaptation;
- tokenizer research;
- long-context training;
- expert parallelism;
- vision и крупный preference/RL stage.

## Что не смешивать

- LoRA/QLoRA — способ адаптировать существующую модель;
- SFT — обучение поведения на instruction data;
- CPT — обучение на raw text и изменение языковых способностей;
- Branch-and-Merge — способ объединять независимые training branches;
- tokenizer surgery — изменение входного представления модели;
- Engram adaptation — изменение специальной памяти Flash-Next.

Это разные эксперименты. Нельзя добавлять их одновременно в первый run, иначе будет невозможно понять, что именно дало результат.
