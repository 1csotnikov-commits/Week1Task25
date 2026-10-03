# Week1Task25 — Мини-чат с RAG + памятью (production-like)

Расширение проекта Week1Task21–24: локальный пайплайн индексации + RAG + reranking/rewrite + цитаты/источники + мини-чат с памятью задачи.

Папка проекта — `C:\II\Week 1 Task 21` (продолжение, публикация в отдельный репозиторий Week1Task25).

Возможности:

1. Индексация документов (Task 21): чанкинг, эмбеддинги `nomic-ai/nomic-embed-text-v2-moe`, FAISS + SQLite, статистика;
2. RAG-запрос (Task 22): вопрос → retrieval → промт → DeepSeek → ответ; контрольные вопросы;
3. Реранкинг + query rewrite (Task 23): threshold/similarity/cross_encoder, none/heuristic/llm;
4. Цитаты и источники (Task 24): структурированный ответ, верификация, режим «не знаю»;
5. Мини-чат с памятью (Task 25): история диалога, task state (goal/facts/constraints/terms/open_questions), сессии и длинные сценарии.

Интерфейс — только CLI (интерактивный REPL), без web. PDF не поддерживаются.

## Стек

- ОС: Windows Server 2022 (совместимо с Linux/macOS)
- Python 3.10+
- `sentence-transformers` + `einops` (эмбеддинги)
- `faiss-cpu` (векторный поиск, с fallback на `numpy`)
- SQLite (метаданные и текст чанков)

## Архитектура

```
C:\II\Week 1 Task 21\
├── config/
│   └── config.yaml          # конфигурация (embedding, chunking, index, llm, rag, evaluation)
├── data/
│   ├── documents/           # пользователь кладёт сюда документы
│   ├── index/               # сгенерированные индексы (faiss.index + metadata.db)
│   ├── eval/                # контрольные вопросы (questions.yaml) и результаты
│   └── logs/                # логи RAG-запросов (опционально)
├── src/
│   ├── ingestion/           # DocumentLoader: text_loader (.md/.txt/.py/.js/.json)
│   ├── chunking/            # BaseChunker, fixed_size, structural
│   ├── embeddings/          # EmbeddingProvider, nomic_provider
│   ├── indexing/            # IndexStore, faiss_sqlite_store, pipeline
│   ├── stats/               # chunk_stats (метрики + форматы console/json/csv/markdown)
│   ├── llm/                 # LLMProvider, deepseek_provider
│   ├── retrieval/           # Retriever, VectorRetriever, RetrievedChunk
│   ├── reranking/           # Reranker, ThresholdFilter, SimilarityReranker, CrossEncoderReranker, RerankerPipeline
│   ├── rag/                 # RAGPipeline, NoRAGPipeline, prompt_builder, query_rewrite, citations (парсинг/верификация)
│   ├── evaluation/          # вопросы, генерация, прогон, эвристики, отчёты
│   ├── chat/                # ChatPipeline, ChatSession, HistoryStore, task_state, state_updater, scenarios
│   └── cli/                 # main.py (REPL), commands.py (единый источник описаний)
├── tests/                   # unittest-тесты
├── requirements.txt
└── README.md
```

## Установка

```bash
pip install -r requirements.txt
```

или вручную (основные зависимости):

```bash
pip install "sentence-transformers>=3.0,<4.0" "transformers>=4.40,<5.0" einops
```

> **Важно:** при первом запуске модель `nomic-ai/nomic-embed-text-v2-moe`
> скачается автоматически (**~1–2 ГБ**). Убедитесь, что есть доступ в интернет.

> **Совместимость версий:** не ставьте `sentence-transformers 6.x` / `transformers 5.x` —
> кастомный код модели nomic (`modeling_hf_nomic_bert.py`) несовместим с transformers 5.x
> (отсутствует метод `get_extended_attention_mask`). Используйте версии из `requirements.txt`.

### Windows Server 2022: возможные проблемы

- **`faiss-cpu` не устанавливается** — это не критично: при ошибке импорта
  автоматически включается fallback на `numpy` + перебор (для 20–30 страниц
  скорость приемлема). В консоли появится предупреждение.
- **`OSError: [WinError 1114] ... c10.dll`** при импорте `torch` — обычно
  означает отсутствие Microsoft Visual C++ Redistributable. Установите
  актуальный **VC++ Redistributable (x64)**, например:
  `https://aka.ms/vs/17/release/vc_redist.x64.exe` (тихая установка:
  `vc_redist.x64.exe /install /quiet /norestart`).

## Модель эмбеддингов

Используется только локальная модель:

```
nomic-ai/nomic-embed-text-v2-moe
```

- 768 измерений;
- поддержка русского языка;
- лицензия Apache 2.0;
- нормализованные векторы (L2-норма = 1), поэтому `IndexFlatIP`
  (inner product) эквивалентен косинусной близости.

## Запуск CLI

```bash
python -m src.cli.main
```

Откроется интерактивная консоль. Список команд — `/help`.

### Команды

| Команда | Описание |
| --- | --- |
| `/help [command]` | Справка по командам (описания — из единого источника `commands.py`) |
| `/index build <path> [--strategy fixed\|structural] [--name <name>] [--config <path>]` | Построить индекс |
| `/index list` | Список индексов |
| `/index info <name>` | Метаданные индекса |
| `/index stats <name> [--format console\|json\|csv\|markdown]` | Статистика по индексу |
| `/index compare <a> <b> [--format ...]` | Сравнить два индекса |
| `/index show-chunk <name> <chunk_id>` | Полный текст + метаданные чанка |
| `/index list-chunks <name> [--source ...] [--section ...] [--limit N]` | Список чанков с фильтрами |
| `/index delete <name>` | Удалить индекс |
| `/ask "<вопрос>" [--index <name>] [--top-k-retrieve N] [--top-k-after-rerank N] [--reranker threshold\|similarity\|cross_encoder\|none] [--rewrite none\|heuristic\|llm] [--debug]` | RAG-ответ + источники + цитаты (статус ok/unknown) |
| `/ask-raw "<вопрос>" [--debug]` | Ответ LLM без RAG |
| `/ask-compare "<вопрос>" [--index <name>] [--top-k-retrieve N]` | Режимы рядом: no_rag \| rag_no_rerank \| rag |
| `/rerank test "<запрос>" [--index <name>] [--top-k-retrieve N] [--reranker ...]` | Чанки до/после фильтра (без LLM) |
| `/rewrite test "<запрос>" [--provider none\|heuristic\|llm]` | Переписанный запрос (без retrieval) |
| `/eval generate [--index <name>] [--count N] [--out <path>] [--force]` | Генерация контрольных вопросов |
| `/eval run <index_name> [--questions <path>] [--mode <mode>] [--compare-all] [--top-k-retrieve N] [--top-k-after-rerank N] [--check-citations on\|off] [--check-sources on\|off] [--check-quote-match on\|off]` | Прогон вопросов в режимах |
| `/eval report <result_file> [--format console\|markdown\|json]` | Сравнительный отчёт по режимам |
| `/chat new <name> --index <index> [--state-updater heuristic\|llm]` | Новая сессия чата (интерактивный режим) |
| `/chat load <name>` | Загрузить сессию чата |
| `/chat list` / `/chat info <name>` / `/chat delete <name>` | Управление сессиями чата |
| `/chat scenario run <file> [--check-goal] [--check-sources] [--check-citations]` | Прогон длинного сценария |
| `/chat scenario list` | Список сценариев |
| `/debug on\|off` | Отладочный режим (показывает эмбеддинги и стектрейсы) |
| `/exit` | Выход |

### Пример использования

```bash
python -m src.cli.main
> /index build data/documents --strategy fixed --name fixed_idx
> /index build data/documents --strategy structural --name structural_idx
> /index list
> /index stats fixed_idx
> /index stats fixed_idx --format markdown
> /index compare fixed_idx structural_idx --format console
> /index show-chunk fixed_idx intro-fixed_size-0000
> /index list-chunks structural_idx --section "Введение" --limit 5
> /index delete structural_idx
> /exit
```

## Стратегии чанкинга

### A — `fixed_size`

Параметры: `chunk_size` (по умолчанию 800 символов), `overlap` (150).
Рез — по границам предложений/строк (не рвёт слова). Для кода — по строкам.

### B — `structural`

- **Markdown/README** — по заголовкам `#`, `##`, `###`; крупные разделы дробятся `fixed_size`;
- **Python** — по функциям/классам через AST;
- **Прочий код** — эвристика по отступам / `def` / `class` / blank-строкам;
- **Текст** — по абзацам с эвристикой заголовков (пустые строки, ALL CAPS).

Метаданные каждого чанка: `source`, `title`, `section`, `chunk_id`, `strategy`,
`char_count`, `token_count` (эвристика `len(text) // 4`).

## Статистика для сравнения стратегий

Метрики:

- количество документов, чанков, время индексации;
- размеры чанков: min / max / avg / median + гистограмма (ASCII в консоли);
- метаданные: чанков с `section`, с `title`, число уникальных `source`;
- перекрытие (`fixed_size`): перекрывающихся чанков, средний overlap;
- покрытие: суммарная длина чанков vs длина исходных текстов;
- семантическая целостность: доля чанков, начинающихся/заканчивающихся в середине
  предложения, доля с незакрытыми скобками/кавычками; для `structural` — доля
  чанков с ровно одним разделом vs с несколькими разделами.

### Пример (фрагмент, console)

```
===== Статистика индекса: fixed_idx =====
Стратегия: fixed
Документов: 5
Чанков: 23
Время индексации: 12.34 сек.

--- Размеры чанков (символы) ---
min: 120
max: 812
avg: 640.5
median: 655.0
Гистограмма распределения:
  [   120-    189] #####                                    5
  [   189-    258] ##########                              10
  ...
```

### JSON / CSV / Markdown

```bash
> /index stats fixed_idx --format json
> /index stats fixed_idx --format csv
> /index stats fixed_idx --format markdown
```

Файлы отчётов сохраняются рядом с индексом:
`data/index/<index_name>/stats.json|csv|md`. Сравнение — в
`data/index/compare_<a>_vs_<b>.<ext>`.

## Тесты

```bash
python -m unittest discover -s tests -t .
```

Покрытие: загрузка документов, `fixed_size`, `structural`, эмбеддинги
(размерность 768 и нормализация; пропускается, если модель недоступна),
сохранение/загрузка FAISS + SQLite, статистика, retrieval, RAG-пайплайн,
LLM-провайдер (мок), evaluation, CLI-команды.

## RAG-пайплайн (Task 22)

Первый RAG-запрос: `вопрос → retrieval → prompt → LLM → ответ`.

```
вопрос → retrieval (поиск top-k чанков в FAISS) → сборка промта → DeepSeek → ответ (+ источники)
```

### Формат эмбеддингов

Модель `nomic-embed-text-v2-moe` различает типы задач префиксами:

- при индексации документов: `search_document: <текст чанка>`;
- при поиске (запрос пользователя): `search_query: <запрос>`.

В Task 21 префиксы не применялись. В Task 22 формат приведён к единому:

- `/index build` добавляет префикс `search_document:` при расчёте эмбеддингов;
- retrieval всегда эмбеддит запрос с префиксом `search_query:`;
- формат сохраняется в индексе (поле `embedding_format` в SQLite).

Если индекс создан без префиксов (старый индекс Task 21) — при поиске выводится
предупреждение и предложение переиндексировать через `/index build`.

### LLM-провайдер (DeepSeek)

- Модель `deepseek-chat`, endpoint `https://api.deepseek.com`;
- ключ — из переменной окружения `DEEPSEEK_API_KEY` (в коде не хранится);
- установка ключа (PowerShell): `$env:DEEPSEEK_API_KEY="sk-..."`;
- при отсутствии ключа — понятная ошибка с инструкцией.

### Команды запросов

```bash
> /ask "Какие стратегии чанкинга есть?" --index my_index --top-k 5
> /ask-raw "Что такое эмбеддинг?"
> /ask-compare "Какая модель эмбеддингов используется?" --index my_index
```

`/ask` выводит ответ и список источников (source, section, chunk_id, score).
При `/debug on` (или флаге `--debug`) — полный промт и все использованные чанки.

### Контрольные вопросы (questions.yaml)

Файл `data/eval/questions.yaml` — 10 вопросов с ожиданиями и источниками.

```yaml
questions:
  - id: q01
    question: "Какие две стратегии чанкинга поддерживает пайплайн?"
    expected_answer: "fixed_size и structural"
    expected_sources: ["readme_guide.md"]
    notes: "раздел 'Стратегии чанкинга'"
```

Поля: `id`, `question`, `expected_answer`, `expected_sources` (опционально), `notes` (опционально).

### Генерация черновика вопросов

```bash
> /eval generate --index my_index --count 10
```

LLM получает выборку чанков (разные source/section) и формирует вопросы.
Если `data/eval/questions.yaml` уже существует и нет `--force` — результат пишется
в `data/eval/questions.generated.yaml`, чтобы не перезаписать ручные правки.

### Прогон и сравнение режимов

```bash
> /eval run my_index --top-k 5
> /eval report data/eval/results/20260101-120000.json --format console
```

`/eval run` прогоняет каждый вопрос в двух режимах (RAG и No-RAG) и сохраняет
результаты в `data/eval/results/<timestamp>.json` и `.md`.

### Эвристики сравнения (без LLM-as-judge)

Грубая автоматическая оценка (точное сравнение — вручную):

- **keyword_overlap** — пересечение ключевых слов (без стоп-слов) ответа и ожидания;
- **substring_match** — вхождение `expected_answer` как подстроки в ответе;
- **jaccard** — коэффициент Жаккара по множествам токенов;
- **hit** — «попал», если `substring_match` ИЛИ `keyword_overlap >= 0.5`;
- **source hit** (RAG) — пересечение `source`/`source#section` использованных чанков с `expected_sources`;
- сводка — hit rate RAG/No-RAG, среднее время, средние токены.

## Реранкинг и query rewrite (Task 23)

Схема пайплайна:

```
(rewrite) → retrieval (top_k_retrieve) → (reranker) → top_k_after_rerank → prompt → LLM → ответ
```

### Реранкеры

- **threshold** (по умолчанию) — отсекает чанки со score ниже `min_score` (с `min_keep`);
- **similarity** — пересчитывает score = `w1*cosine + w2*keyword_overlap + w3*phrase_match`, сортирует, отсекает;
- **cross_encoder** — отдельная модель `BAAI/bge-reranker-v2-m3` (~2.3 ГБ), по умолчанию выключен;
- **none** — без реранкинга.

Выбор провайдера: `config.yaml` (`reranking.provider`) или флаг `--reranker`.

### Настройка порога

Команда `/rerank test` показывает чанки до/после фильтра на реальных запросах:

```bash
> /rerank test "Что такое ТС ПИоТ" --index my_index --reranker threshold
```

### Query rewrite

- **none** (по умолчанию) — запрос как есть;
- **heuristic** — расширяет запрос ключевыми словами (без стоп-слов);
- **llm** — переписывает запрос через DeepSeek.

Включение: `config.yaml` (`rewrite.provider`) или флаг `--rewrite`. Проверка:

```bash
> /rewrite test "Что такое ТС ПИоТ" --provider heuristic
```

### CrossEncoder: требования к машине

- CPU-режим (GPU не обязателен);
- RAM: 8 GB минимум, 16 GB рекомендуется (с учётом nomic-embed-text-v2-moe в памяти);
- диск: ~8 GB;
- на CPU обработка медленная (несколько секунд на батч).

### Сравнение режимов

```bash
> /eval run my_index --top-k-retrieve 10 --top-k-after-rerank 5        # no_rag, rag_no_rerank, rag
> /eval run my_index --compare-all                                     # все 5 режимов
> /eval report data/eval/results/<timestamp>.json
```

Режимы: `no_rag`, `rag_no_rerank`, `rag`, `rag_no_rewrite`, `rag_full`.

## Цитаты и источники (Task 24)

Схема пайплайна:

```
(rewrite) → retrieval → reranker → проверка контекста → (LLM | «не знаю») → парсинг → верификация
```

### Формат структурированного ответа

LLM обязан вернуть три блока:

```
ОТВЕТ:
<текст ответа>

ИСТОЧНИКИ:
- source=<файл> | chunk_id=<id>

ЦИТАТЫ:
- "<дословный фрагмент>" | source=<файл> | chunk_id=<id>
```

Правила (в system-промте): цитаты — только дословные фрагменты из чанков, не пересказ;
каждая цитата/источник сопровождается source и chunk_id; если контекста недостаточно — ответ
начинается с «Не знаю.»; додумывать нельзя.

### Верификация (анти-галлюцинации)

`src/rag/citations.py` разбирает ответ и проверяет:

- **цитаты** — найдена ли цитата в чанках (по chunk_id или по подстроке); если нет — `unverified`;
- **источники** — есть ли source/chunk_id среди использованных чанков; если нет — `unverified`.

Если структура не распознана (нет блока ОТВЕТ) — `parse_failed=True`, весь ответ считается текстом.

### Режим «не знаю» при слабом контексте

Пороги в `config.yaml`:

```yaml
rag:
  unknown_threshold:
    min_chunks: 1     # после reranker
    min_score: 0.35   # лучший score после reranker
```

Если после reranker 0 чанков ИЛИ лучший score ниже порога — LLM **не вызывается**,
возвращается ответ «Не знаю…» и `answer_status = "unknown"` (экономия токенов и защита от галлюцинаций).
Порог подбирается через `/rerank test`.

### Чтение отчёта

```bash
> /eval run my_index --top-k-retrieve 10 --top-k-after-rerank 5
> /eval report data/eval/results/<timestamp>.json
```

В отчёте по каждому rag-режиму: доля ответов с блоком ИСТОЧНИКИ, с блоком ЦИТАТЫ, доля
полностью подтверждённых цитат/источников, доля «unknown», среднее число цитат/источников.

### Если цитаты unverified

Возможные причины: галлюцинация LLM (цитата не из контекста), плохой парсинг (LLM нарушил формат),
слабый промт. Проверьте сырой ответ через `/ask ... --debug`.

## Мини-чат с RAG + памятью (Task 25)

Схема хода:

```
user → state update → retrieval/rerank/rewrite → prompt (state + history + context) → LLM → парсинг → history + state → ответ
```

### Хранение сессий

```
data/chat/sessions/<session_name>/
├── meta.json           # ChatSessionMeta
├── history.jsonl       # история (append-only, по одному Message на строку)
└── task_state.json     # TaskState (перезаписывается целиком)
```

Сессии загружаются по имени при перезапуске (`/chat load <name>`).

### Память задачи (task state)

Поля: `goal`, `facts`, `constraints`, `terms` (термин→определение), `open_questions`.

Обновление (`chat.state_updater.provider`):

- **heuristic** (по умолчанию) — по маркерам: «цель —», «запомни:», «важно:», «ограничение:», «требование:», «термин: X = Y»;
- **llm** — через DeepSeek (JSON-обновление), при ошибке парсинга state не меняется.

Внутри чата: `/state show`, `/state edit`, `/state clear`, `/history show [N]`, `/history clear`, `/exit`, `/help`.

### Длинные сценарии

```bash
> /chat scenario run scenario_1.yaml --check-goal --check-sources --check-citations
> /chat scenario list
```

Сценарии — YAML (`data/chat/scenarios/*.yaml`) с `name`, `description`, `index_name` и `messages`.
Результаты — `data/chat/scenarios/results/<name>_<timestamp>.json` и `.md`.

### Если ассистент теряет цель

- увеличьте `chat.history_window` (сколько последних сообщений попадает в промт);
- переключите `state_updater.provider` на `llm` (точнее, но медленнее);
- посмотрите логи и `/state show`.

## Ограничения

- LLM-провайдер только DeepSeek; модель эмбеддингов только nomic-embed-text-v2-moe;
- интерфейс — CLI, без web;
- сравнение качества — эвристики, без LLM-as-judge;
- task state — компактная выжимка, возможны потери неявных уточнений (особенно в heuristic-режиме);
- сценарии пишет пользователь вручную.

