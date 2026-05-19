# UTUEL — Unified Efficient LLM pipeline

Batch-process multiple JSONL datasets against different Ollama models with:
- **`OllamaLLM.abatch()` + `max_concurrency`** — LangChain semaphore keeps exactly `OLLAMA_NUM_PARALLEL` requests in-flight
- **Prompt-length sorting** — shortest prompts first to maximise KV-cache slot utilisation
- **Resume support** — re-run after a crash; already-written rows are skipped automatically
- **Multi-run support** — run each dataset N times, each run in its own `<name>_run<N>.jsonl`
- **Per-run stats** — throughput, success rate, elapsed time
- **Hydra config + CLI overrides** — single `config.yml`, override any value from the command line

---

## Project structure

```
UTUEL/
├── requirements.txt
├── datasets/
│   ├── qa.jsonl               ← input (you provide)
│   ├── summaries.jsonl        ← input (you provide)
│   └── *_run*.jsonl           ← outputs (auto-generated, gitignored)
└── prompts_pipeline/          ← Python package (run with -m prompts_pipeline)
    ├── __init__.py
    ├── __main__.py            ← entry point; sets Ollama env vars, calls runner
    ├── config.yml             ← all settings (Ollama + pipeline + datasets)
    ├── dataset.py             ← JSONL loader + output path resolver
    ├── runner.py              ← orchestrator (sort, abatch, resume, write)
    └── stats.py               ← RunStats dataclass
```

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Start Ollama (env vars are also set automatically by the pipeline)
ollama serve

# 3. Run with defaults from config.yml
python -m prompts_pipeline
```

---

## Configuration — `prompts_pipeline/config.yml`

All settings live in one file:

```yaml
ollama:
  num_parallel: 6          # OLLAMA_NUM_PARALLEL       — concurrent slots
  max_loaded_models: 6     # OLLAMA_MAX_LOADED_MODELS  — models kept in VRAM
  max_queue: 2048          # OLLAMA_MAX_QUEUE          — server-side request queue

pipeline:
  num_runs: 1              # independent passes over every dataset
  ollama_timeout: 120.0    # seconds per request
  ollama_base_url: http://localhost:11434

datasets:
  - path: datasets/qa.jsonl
    model: llama3.2
    prompt_key: prompt
  - path: datasets/summaries.jsonl
    model: mistral
    prompt_key: text
```

The three `ollama:` values are exported as environment variables **before** the first
request, so the Ollama server always runs with exactly the capacity declared here.
`num_parallel` also sets `max_concurrency` on `abatch()` so the pipeline never
submits more in-flight requests than the server can execute in parallel.

### CLI overrides (Hydra dot-notation)

Any config value can be overridden at runtime — no file editing required:

```bash
python -m prompts_pipeline pipeline.num_runs=5
python -m prompts_pipeline ollama.num_parallel=12
python -m prompts_pipeline pipeline.ollama_base_url=http://gpu-server:11434
```

---

## Input format

Each dataset is a `.jsonl` file — one JSON object per line:

```jsonl
{"prompt": "What is the capital of France?", "category": "geo"}
{"prompt": "Explain TCP vs UDP.", "category": "networking"}
```

Any key can be the prompt; configure via `prompt_key` in `config.yml`.  
All other keys are preserved verbatim in the output.

## Output format

One file per dataset per run: `<name>_run<N>.jsonl`

```jsonl
{"run": 1, "dataset": "qa.jsonl", "row_id": 0, "model": "llama3.2", "prompt": "...", "response": "...", "status": "ok", "category": "geo"}
```

The `run` field distinguishes records when multiple runs share the same file namespace.

---

## Resume after failure

Output files are written **row by row** (append mode).  
If the run crashes, re-run the same command — rows already written are detected by
`row_id` and skipped. Output filenames are deterministic (`<name>_run<N>.jsonl`) so
the resume logic always finds the right file.

---

## How batching works

```
pending prompts (sorted shortest → longest)
        │
        ▼
OllamaLLM.abatch(prompts, config={"max_concurrency": num_parallel})
        │
        │  asyncio.Semaphore(num_parallel) — at most N coroutines run at once
        │
        ▼
POST /api/generate  ×N  ──▶  Ollama server (OLLAMA_NUM_PARALLEL slots)
```

Sorting by prompt length before dispatch means the server's parallel slots are
occupied by similarly-sized requests, minimising padding waste in the KV-cache.

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

Add one entry under `datasets:` in `config.yml`:

```yaml
datasets:
  - path: datasets/my_new_data.jsonl
    model: gemma3
    prompt_key: question   # whatever key holds the prompt text
```

That's it.
