"""
runner.py
Orchestrates dataset loading, Ollama dispatch via OllamaLLM.abatch(),
resume logic, and result writing for one full pipeline run.

Datasets are processed in groups of max_loaded_models concurrently so
Ollama's VRAM is never over-committed.  Within each group, up to
num_parallel prompts are sent in-flight per model via abatch().
"""

import asyncio
import json
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

from langchain_ollama import OllamaLLM

from .dataset import load_jsonl, load_already_done, output_path_for, apply_prompt_template
from .stats import RunStats


# ── helpers ──────────────────────────────────────────────────────────────────

def _make_timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _append_jsonl(path: Path, obj: dict) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def _progress_bar(done: int, total: int, width: int = 20) -> str:
    filled = int(width * done / total) if total else 0
    return f"[{'█' * filled}{'░' * (width - filled)}] {done}/{total}"


# ── dataset config ────────────────────────────────────────────────────────────

class DatasetConfig:
    """One entry in the pipeline manifest."""

    def __init__(self, path: str, model: str, prompt_key: str = "prompt", name: str = "",
                 prompt_template: str = ""):
        self.path = Path(path)
        self.model = model
        self.prompt_key = prompt_key
        self.name = name or Path(path).stem   # fall back to filename stem if omitted
        self.prompt_template = prompt_template


# ── main runner ───────────────────────────────────────────────────────────────

class PipelineRunner:
    """Runs a collection of (dataset, model) pairs through Ollama."""

    def __init__(
        self,
        datasets: list[DatasetConfig],
        num_runs: int = 1,
        num_parallel: int = 4,
        max_loaded_models: int = 4,
        max_queue: int = 4096,
        ollama_base_url: str = "http://localhost:11434",
    ):
        self.datasets = datasets
        self.num_runs = num_runs
        # num_parallel      — matches OLLAMA_NUM_PARALLEL; controls abatch() concurrency.
        # max_loaded_models — matches OLLAMA_MAX_LOADED_MODELS; datasets are dispatched
        #                     in groups of this size so VRAM is never over-committed.
        # max_queue         — matches OLLAMA_MAX_QUEUE; logged for visibility.
        self.num_parallel      = num_parallel
        self.max_loaded_models = max_loaded_models
        self.max_queue         = max_queue
        self.ollama_base_url   = ollama_base_url

    # ── public entry point ────────────────────────────────────────────────────

    async def run(self) -> RunStats:
        run_id = _make_timestamp() + "_" + uuid.uuid4().hex[:6]
        stats = RunStats(run_id=run_id)

        print(f"\n{'═'*50}")
        print(f"  UTUEL pipeline  —  session {run_id}")
        print(f"  {self.num_runs} run(s) × {len(self.datasets)} dataset(s)  "
              f"[parallel={self.num_parallel}  queue={self.max_queue}]")
        print(f"{'═'*50}\n")

        for run_num in range(1, self.num_runs + 1):
            print(f"── Run {run_num}/{self.num_runs} {'─'*35}")

            # Phase 1 — load rows + apply prompt templates sequentially.
            # Must be sequential: multiple datasets may share the same input
            # file; concurrent writes would corrupt it.
            prepared: list[tuple[DatasetConfig, list[dict]]] = []
            templated_paths: set[Path] = set()
            for ds in self.datasets:
                rows = load_jsonl(ds.path)
                if rows and ds.prompt_key not in rows[0] and ds.prompt_template:
                    if ds.path not in templated_paths:
                        rows = apply_prompt_template(
                            rows, ds.prompt_key, ds.prompt_template, ds.path
                        )
                        templated_paths.add(ds.path)
                    else:
                        # Sibling dataset shares the file — reload after sibling wrote it
                        rows = load_jsonl(ds.path)
                prepared.append((ds, rows))

            # Phase 2 — dispatch datasets concurrently in groups of
            # max_loaded_models so we never send more models to Ollama than
            # it can keep in VRAM simultaneously.  Each group runs in parallel;
            # groups run sequentially so VRAM is never over-committed.
            group_size = self.max_loaded_models
            groups = [prepared[i:i + group_size]
                      for i in range(0, len(prepared), group_size)]
            for g_idx, group in enumerate(groups, start=1):
                models = ", ".join(ds.model for ds, _ in group)
                print(f"\n  ▶ model group {g_idx}/{len(groups)}: {models}")
                await asyncio.gather(*[
                    self._run_dataset(ds, rows, run_num, stats)
                    for ds, rows in group
                ])

        stats.finish()
        print(f"\n{stats}\n")
        return stats

    # ── per-dataset worker ────────────────────────────────────────────────────

    async def _run_dataset(
        self,
        ds: DatasetConfig,
        rows: list[dict],
        run_num: int,
        stats: RunStats,
    ) -> None:
        tag      = f"[{ds.model}]"
        out_path = output_path_for(ds.path, ds.name, ds.model, run_num)
        done_ids = load_already_done(out_path)

        print(f"  {tag} {ds.name}  ({len(rows)} rows, {len(done_ids)} already done)")

        # Collect rows that still need processing
        pending: list[tuple[int, str, dict]] = []
        for row_id, row in enumerate(rows):
            stats.total_requests += 1
            if row_id in done_ids:
                stats.skipped_resumed += 1
                continue
            prompt = row.get(ds.prompt_key, "")
            if not prompt:
                print(f"  {tag} [WARN] row {row_id} has no '{ds.prompt_key}' key — skipping")
                stats.skipped_resumed += 1
                continue
            extra = {k: v for k, v in row.items() if k != ds.prompt_key}
            pending.append((row_id, prompt, extra))

        if not pending:
            print(f"  {tag} (nothing to do)")
            return

        total     = len(pending)
        print(f"  {tag} {total} rows to process  {_progress_bar(0, total)}", flush=True)

        llm = OllamaLLM(
            model=ds.model,
            base_url=self.ollama_base_url,
        )

        # Sort by prompt length so the server processes similarly-sized
        # requests together, minimising padding waste in each KV-cache
        # slot and improving GPU utilisation.
        pending.sort(key=lambda t: len(t[1]))

        # Use a semaphore + individual ainvoke() calls so each result is
        # written to disk immediately as it completes.  This means a crash
        # mid-run loses at most one in-flight request, not the whole batch.
        sem       = asyncio.Semaphore(self.num_parallel)
        successes = 0
        errors    = 0
        done      = 0
        in_flight = 0
        lock      = asyncio.Lock()
        t0        = time.time()

        async def _invoke_and_write(row_id: int, prompt: str, extra: dict) -> None:
            nonlocal successes, errors, done, in_flight
            async with sem:
                async with lock:
                    in_flight += 1
                try:
                    response = await llm.ainvoke(prompt)
                    status, text = "ok", response
                except Exception as exc:
                    status, text = f"error: {exc}", None
                finally:
                    async with lock:
                        in_flight -= 1

            async with lock:
                if status == "ok":
                    successes += 1
                else:
                    errors += 1
                done += 1
                elapsed = time.time() - t0
                rate    = done / elapsed if elapsed > 0 else 0.0
                eta     = (total - done) / rate if rate > 0 else 0.0
                bar     = _progress_bar(done, total)
                # Overwrite the current line in-place for a live progress display
                print(
                    f"\r  {tag} {bar}"
                    f"  in-flight={in_flight}/{self.num_parallel}"
                    f"  ok={successes} err={errors}"
                    f"  {rate:.1f} rows/s  ETA {eta:.0f}s  ",
                    end="", flush=True,
                )

                _append_jsonl(out_path, {
                    "run":      run_num,
                    "dataset":  ds.name,
                    "row_id":   row_id,
                    "model":    ds.model,
                    "prompt":   prompt,
                    "response": text,
                    "status":   status,
                    **extra,
                })

        await asyncio.gather(*[
            _invoke_and_write(row_id, prompt, extra)
            for row_id, prompt, extra in pending
        ])
        elapsed = round(time.time() - t0, 2)

        # Move to next line after the in-place progress bar
        print(flush=True)
        print(f"  {tag} ✓ done  {successes}/{total} ok  {errors} errors  ({elapsed}s)", flush=True)

        stats.record_batch(total, successes)
