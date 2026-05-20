"""
__main__.py
Entry point — settings come from config.yml, overridable with dot-notation:

    python -m prompts_pipeline
    python -m prompts_pipeline pipeline.num_runs=5
    python -m prompts_pipeline ollama.num_parallel=8
    python -m prompts_pipeline pipeline.ollama_base_url=http://gpu-server:11434
"""

import asyncio
import logging
import os
from pathlib import Path

import hydra
from omegaconf import DictConfig

from .runner import PipelineRunner, DatasetConfig

# Suppress httpx / httpcore / LangChain HTTP INFO chatter so only
# the pipeline's own progress lines appear in the terminal.
for _noisy in ("httpx", "httpcore", "langchain", "langchain_core", "ollama"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

@hydra.main(config_path=str(Path(__file__).parent), config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    # Export Ollama server environment variables so they take effect for any
    # subprocess (e.g. a locally spawned ollama serve).
    os.environ["OLLAMA_NUM_PARALLEL"]      = str(cfg.ollama.num_parallel)
    os.environ["OLLAMA_MAX_LOADED_MODELS"] = str(cfg.ollama.max_loaded_models)
    os.environ["OLLAMA_MAX_QUEUE"]         = str(cfg.ollama.max_queue)

    # ── Build dataset list from config ────────────────────────────────────────
    #
    #  Add more entries under `datasets:` in config.yml instead of editing here.
    #
    #  Output files:  datasets/<name>_run<N>.jsonl  (one per dataset per run)
    #
    #  Resume:  if a run is interrupted, just re-run — load_already_done() reads
    #  the existing output file, collects all row_ids already written, and skips
    #  them.  Rows are appended immediately after each abatch() call, so at most
    #  one batch is lost on a crash.
    # ─────────────────────────────────────────────────────────────────────────

    datasets = [
        DatasetConfig(
            path=ds["path"],
            model=ds["model"],
            prompt_key=ds.get("prompt_key", "prompt"),
            name=ds.get("name", ""),
            prompt_template=ds.get("prompt_template", ""),
        )
        for ds in cfg.datasets
    ]

    pipeline = PipelineRunner(
        datasets=datasets,
        num_runs=cfg.pipeline.num_runs,
        num_parallel=cfg.ollama.num_parallel,         # → OLLAMA_NUM_PARALLEL
        max_loaded_models=cfg.ollama.max_loaded_models, # → OLLAMA_MAX_LOADED_MODELS
        max_queue=cfg.ollama.max_queue,               # → OLLAMA_MAX_QUEUE
        ollama_base_url=cfg.pipeline.ollama_base_url,
    )

    asyncio.run(pipeline.run())


if __name__ == "__main__":
    main()
