"""
compile.py
Merges all per-model/per-run output files for a dataset into a single
compiled JSONL for evaluation.

Output schema per record:
  table_id      — from source record
  ground_truth  — from source record's "answers" field (list)
  question      — from source record
  model         — model name (from folder or record)
  run           — run number
  response      — raw model response string
  prediction    — answer extracted from response JSON {"answer": "..."}

Usage (importable):
  from prompts_pipeline.compile import compile_dataset, compile_all

Usage (CLI):
  python -m prompts_pipeline.compile                     # compile all datasets
  python -m prompts_pipeline.compile datasets/test_lookup_WikiSQL
"""

import json
import re
import sys
from pathlib import Path


# ── prediction extraction ─────────────────────────────────────────────────────

def _extract_prediction(response: str | None) -> str | None:
    """
    Parse the model response and extract the value of the "answer" key.

    Handles three common response shapes:
      1. Pure JSON:            {"answer": "French Open"}
      2. JSON after thinking:  <think>...</think>\n{"answer": "French Open"}
      3. JSON embedded in text: "... the answer is {\"answer\": \"French Open\"} ..."

    Returns None when no parseable answer is found.
    """
    if not response:
        return None

    # 1. Try the whole string first (common for well-behaved models)
    try:
        data = json.loads(response.strip())
        if isinstance(data, dict) and "answer" in data:
            return str(data["answer"])
    except (json.JSONDecodeError, ValueError):
        pass

    # 2. Find all {...} blocks and inspect from last to first.
    #    deepseek-r1 puts <think>...</think> before the final JSON answer,
    #    so the last parseable block with "answer" is the right one.
    for m in reversed(list(re.finditer(r'\{[^{}]+\}', response, re.DOTALL))):
        try:
            data = json.loads(m.group())
            if isinstance(data, dict) and "answer" in data:
                return str(data["answer"])
        except (json.JSONDecodeError, ValueError):
            continue

    return None


# ── answer normalisation ────────────────────────────────────────────────────────

def normalize_answer(value) -> str:
    """
    Normalise a ground-truth or prediction value to a comparable string.

    Handles:
      - list   → join all elements with ", " then normalise the joined string
      - str    → lower-case, collapse whitespace, strip leading/trailing spaces
      - other  → convert to str first

    Call this on both sides of a comparison so the equality check is
    robust to capitalisation, extra spaces, and single-vs-list packaging.
    """
    if isinstance(value, list):
        value = ", ".join(str(v) for v in value)
    else:
        value = str(value) if value is not None else ""
    return re.sub(r"\s+", " ", value).strip().lower()


def is_correct(prediction: str | None, ground_truth) -> bool:
    """
    Return True when the normalised prediction matches any normalised
    ground-truth value.

    ground_truth may be a list[str] or a plain str.
    """
    if prediction is None:
        return False
    pred_norm = normalize_answer(prediction)
    candidates = ground_truth if isinstance(ground_truth, list) else [ground_truth]
    return any(pred_norm == normalize_answer(gt) for gt in candidates)


# ── core compiler ─────────────────────────────────────────────────────────────

def compile_dataset(dataset_dir: Path) -> Path:
    """
    Read all  run*.jsonl  files found recursively under dataset_dir and write
    a single compiled JSONL to  <datasets_root>/compiled_<name>/compiled.jsonl.

    Returns the path to the compiled file.
    """
    dataset_dir = Path(dataset_dir).resolve()
    name = dataset_dir.name

    out_dir = dataset_dir.parent / f"compiled_{name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "compiled.jsonl"

    run_files = sorted(dataset_dir.glob("**/*.jsonl"))
    if not run_files:
        print(f"[compile] No run*.jsonl files found under {dataset_dir}")
        return out_path

    total = 0
    with out_path.open("w", encoding="utf-8") as out_f:
        for run_file in run_files:
            with run_file.open(encoding="utf-8") as f:
                for lineno, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError as exc:
                        print(f"[compile] WARN {run_file}:{lineno} — JSON error: {exc}")
                        continue

                    prediction = _extract_prediction(record.get("response"))
                    ground_truth = record.get("answers")  # list[str]
                    parse_ok = prediction is not None
                    compiled = {
                        "table_id":     record.get("table_id"),
                        "ground_truth": ground_truth,
                        "question":     record.get("question"),
                        "model":        record.get("model"),
                        "run":          record.get("run"),
                        "response":     record.get("response"),
                        "prediction":   prediction,
                        "parse_ok":     parse_ok,
                        "correct":      is_correct(prediction, ground_truth),
                    }
                    out_f.write(json.dumps(compiled, ensure_ascii=False) + "\n")
                    total += 1

    print(f"[compile] {name} → {out_path}  ({total} records)")
    return out_path


def compile_all(datasets_root: Path) -> list[Path]:
    """
    Discover every dataset sub-directory under datasets_root (skipping any
    that start with 'compiled_') and compile each one.  Run files are found
    recursively so any nesting depth is supported.

    Returns a list of compiled file paths.
    """
    datasets_root = Path(datasets_root)
    outputs = []
    for candidate in sorted(datasets_root.iterdir()):
        if not candidate.is_dir():
            continue
        if candidate.name.startswith("compiled_"):
            continue
        # Only treat as a dataset folder if it contains at least one run file
        # (search recursively so any sub-folder depth is covered)
        if not any(candidate.glob("**/*.jsonl")):
            continue
        outputs.append(compile_dataset(candidate))
    return outputs


# ── CLI entry point ───────────────────────────────────────────────────────────

if __name__ == "__main__":
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
        if not path.exists():
            print(f"[compile] ERROR: '{path}' does not exist.", file=sys.stderr)
            sys.exit(1)
        compile_dataset(path)
    else:
        # Auto-discover from the default datasets/ folder next to this package
        root = Path(__file__).parent.parent / "datasets"
        if not root.exists():
            print(f"[compile] ERROR: datasets/ folder not found at {root}", file=sys.stderr)
            sys.exit(1)
        results = compile_all(root)
        if not results:
            print("[compile] No dataset folders with run*.jsonl files found.")
