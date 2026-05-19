"""
dataset.py
Loads JSONL datasets and resolves output paths.
Output file name: {stem}_generated_{timestamp}.jsonl
"""

import csv
import io
import json
from pathlib import Path

from langchain_core.prompts import PromptTemplate


def load_jsonl(path: str | Path) -> list[dict]:
    """Read every line of a JSONL file into a list of dicts."""
    path = Path(path)
    rows = []
    with path.open(encoding="utf-8") as f:
        for lineno, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                print(f"[WARN] {path.name}:{lineno} — skipping bad JSON: {exc}")
    return rows


def _table_to_csv(header: list, rows: list) -> str:
    """Render a header + rows structure as a CSV string."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(header)
    writer.writerows(rows)
    return buf.getvalue().strip()


def apply_prompt_template(
    rows: list[dict],
    prompt_key: str,
    template_path: str | Path,
    input_path: str | Path,
) -> list[dict]:
    """
    Render every row through a prompt template and write the result back into
    the input JSONL file (adding prompt_key to each row).

    The template is a plain text file with Python str.format() placeholders
    matching row keys, e.g.  {input_data}, {question}.

    If a row has 'header' and 'rows' fields but no 'input_data', the table is
    automatically serialised as CSV and injected as 'input_data' before
    template substitution.

    The updated file replaces the original so subsequent runs find the prompts
    already present and skip this step.
    """
    template = PromptTemplate.from_template(
        Path(template_path).read_text(encoding="utf-8")
    )
    updated: list[dict] = []
    for row in rows:
        context = dict(row)
        # Auto-build input_data from tabular header+rows if not already present
        if "input_data" not in context and "header" in context and "rows" in context:
            context["input_data"] = _table_to_csv(context["header"], context["rows"])
        # Only pass variables the template actually declares; extra keys are ignored
        template_vars = {k: context[k] for k in template.input_variables if k in context}
        missing = [k for k in template.input_variables if k not in context]
        if missing:
            print(f"    [WARN] template variables {missing} missing in row — prompt left empty")
            prompt = ""
        else:
            prompt = template.format(**template_vars)
        updated.append({**row, prompt_key: prompt})

    p = Path(input_path)
    with p.open("w", encoding="utf-8") as f:
        for row in updated:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"    [INFO] prompt template applied → {p} updated ({len(updated)} rows)")
    return updated


def output_path_for(input_path: str | Path, name: str, model: str, run_num: int) -> Path:
    """
    Build the stable output path for a given dataset name, model and run number.

    Structure:  <datasets_dir>/<name>/<model>/<name>_run<N>.jsonl
    E.g.  name="QA", model="llama3.2", run 2  →  datasets/QA/llama3.2/QA_run2.jsonl

    Directories are created on first use.  The name is deterministic (no
    timestamp) so the resume logic can always locate the file after a crash.
    """
    p = Path(input_path)
    safe_model = model.replace("/", "-")   # guard against namespaced model names
    out_dir = p.parent / name / safe_model
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"{name}_run{run_num}.jsonl"


def load_already_done(output_path: Path) -> set[int]:
    """
    Read an (possibly partial) output file and return the set of row_ids
    already written.  Used for resume logic.
    """
    done: set[int] = set()
    if not output_path.exists():
        return done
    with output_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
                if "row_id" in obj:
                    done.add(int(obj["row_id"]))
            except (json.JSONDecodeError, ValueError):
                pass
    return done
