# UTUEL — Unified Efficient LLM pipeline

Batch-process multiple JSONL datasets against different Ollama models with:
- **`OllamaLLM.abatch()` + `max_concurrency`** — LangChain semaphore keeps exactly `OLLAMA_NUM_PARALLEL` requests in-flight per model
- **Model-group parallelism** — up to `max_loaded_models` models run concurrently; groups are sequenced to respect VRAM limits
- **Prompt-length sorting** — shortest prompts first to maximise KV-cache slot utilisation
- **Prompt template support** — auto-renders prompts from a template file if the dataset has no prompt column; tabular (`header`+`rows`) data is serialised as CSV automatically
- **Resume support** — re-run after a crash; already-written rows are skipped automatically
- **Multi-run support** — run each dataset N times, each run in its own file
- **Per-run stats** — throughput, success rate, elapsed time
- **Hydra config + CLI overrides** — single `config.yml`, override any value from the command line

---

## Project structure

```
UTUEL/
├── requirements.txt
├── datasets/
│   ├── my_dataset.jsonl            ← input (you provide)
│   └── <name>/<model>/run<N>.jsonl ← outputs (auto-generated, gitignored)
└── prompts_pipeline/               ← Python package (run with -m prompts_pipeline)
    ├── __init__.py
    ├── __main__.py                 ← entry point; sets Ollama env vars, calls runner
    ├── config.yml                  ← all settings (Ollama + pipeline + datasets)
    ├── dataset.py                  ← JSONL loader, prompt template renderer, output paths
    ├── runner.py                   ← orchestrator (group dispatch, abatch, resume, write)
    ├── stats.py                    ← RunStats dataclass
    └── prompt_templates/           ← plain-text prompt template files
        └── lookup_WikiSQL.txt
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama
ollama serve

# 3. Run with defaults from config.yml
python -m prompts_pipeline
```

The pipeline sets `OLLAMA_NUM_PARALLEL`, `OLLAMA_MAX_LOADED_MODELS`, and `OLLAMA_MAX_QUEUE` as environment variables automatically before the first request.

---

## Configuration — `prompts_pipeline/config.yml`

All settings live in one file:

```yaml
ollama:
  num_parallel: 4          # OLLAMA_NUM_PARALLEL       — concurrent request slots per model
  max_loaded_models: 4     # OLLAMA_MAX_LOADED_MODELS  — models kept in VRAM simultaneously
  max_queue: 4096          # OLLAMA_MAX_QUEUE          — server-side request queue depth

pipeline:
  num_runs: 1              # independent passes over every dataset
  ollama_base_url: http://localhost:11434

datasets:
  - name: WikiSQL           # used for output folder name and log labels
    path: datasets/test_lookup_WikiSQL.jsonl
    model: llama3
    prompt_key: prompt      # key that holds (or will hold) the prompt text
    prompt_template: prompt_templates/lookup_WikiSQL.txt  # optional; used if prompt_key is absent
  - name: WikiSQL
    path: datasets/test_lookup_WikiSQL.jsonl
    model: gemma2
    prompt_key: prompt
    prompt_template: prompt_templates/lookup_WikiSQL.txt
```

### How the three `ollama:` values are used

| Key | Env var set | Also controls |
|---|---|---|
| `num_parallel` | `OLLAMA_NUM_PARALLEL` | `max_concurrency` passed to `abatch()` |
| `max_loaded_models` | `OLLAMA_MAX_LOADED_MODELS` | dataset group size — at most this many models dispatched concurrently |
| `max_queue` | `OLLAMA_MAX_QUEUE` | logged in the session header |

### CLI overrides (Hydra dot-notation)

Any config value can be overridden at runtime — no file editing required:

```bash
python -m prompts_pipeline pipeline.num_runs=3
python -m prompts_pipeline ollama.num_parallel=8
python -m prompts_pipeline ollama.max_loaded_models=2
python -m prompts_pipeline pipeline.ollama_base_url=http://gpu-server:11434
```

---

## Input format

Each dataset is a `.jsonl` file — one JSON object per line.

**With a ready-made prompt column:**
```jsonl
{"prompt": "What is the capital of France?", "category": "geo"}
```

**With tabular data (no prompt column yet):**
```jsonl
{"question": "What is Terrence Ross' nationality?", "header": ["Player", "Nationality", ...], "rows": [["Terrence Ross", "United States", ...], ...]}
```
If `prompt_key` is absent from the rows and `prompt_template` is set, the pipeline renders each row through the template and saves the result back into the input file before inference begins.

All keys other than `prompt_key` are preserved verbatim in the output.

---

## Prompt templates

Template files live in `prompts_pipeline/prompt_templates/` and use Python `str.format()` placeholders matching row keys:

```
Task Description: Please look at the table and answer the question.
## Input: {input_data}, Question: {question}
Return the result as JSON: {{"answer": "<YOUR ANSWER>"}}
## Output:
```

If the row has `header` + `rows` fields but no `input_data` key, the pipeline automatically serialises the table as CSV and injects it as `{input_data}` before substitution.

---

## Output structure

```
datasets/
└── WikiSQL/
    ├── llama3/
    │   ├── run1.jsonl
    │   └── run2.jsonl
    └── gemma2/
        └── run1.jsonl
```

Each record:
```jsonl
{"run": 1, "dataset": "WikiSQL", "row_id": 0, "model": "llama3", "prompt": "...", "response": "...", "status": "ok", ...}
```

---

## How parallelism works

```
datasets in config.yml  (e.g. 8 models, max_loaded_models=4)
        │
        ├─ Group 1: llama3, gemma2, tablellm, qwen  ─── asyncio.gather ──▶ run concurrently
        │            wait for all to finish
        └─ Group 2: deepseek, gemma4, TableGPT2, gpt ── asyncio.gather ──▶ run concurrently

Within each model:
  pending prompts (sorted shortest → longest)
        │
        ▼
  OllamaLLM.abatch(prompts, config={"max_concurrency": num_parallel})
        │  asyncio.Semaphore(num_parallel) — at most N HTTP calls in-flight
        ▼
  POST /api/generate ×N ──▶ Ollama server (OLLAMA_NUM_PARALLEL slots)
```

Groups keep VRAM usage bounded; sorting minimises KV-cache padding waste.

---

## Resume after failure

Output files are written **row by row** (append mode).  
If the run crashes, re-run the same command — rows already written are detected by `row_id` and skipped. Filenames are deterministic so the resume logic always finds the right file.

---

## RunStats

`runner.run()` returns a `RunStats` instance:

```python
stats = await runner.run()

stats.succeeded       # int — successful responses
stats.failed          # int — errored responses
stats.skipped_resumed # int — rows skipped (already done)
stats.throughput      # float — responses per second
stats.elapsed_seconds # float
stats.to_dict()       # plain dict for JSON serialisation
print(stats)          # formatted summary table
```

---

## Adding a new dataset

Add entries under `datasets:` in `config.yml` — one entry per model you want to test:

```yaml
datasets:
  - name: MyDataset
    path: datasets/my_data.jsonl
    model: llama3
    prompt_key: prompt
  - name: MyDataset
    path: datasets/my_data.jsonl
    model: gemma2
    prompt_key: prompt
    prompt_template: prompt_templates/my_template.txt  # optional
```

Output goes to `datasets/MyDataset/llama3/run1.jsonl` and `datasets/MyDataset/gemma2/run1.jsonl`.
