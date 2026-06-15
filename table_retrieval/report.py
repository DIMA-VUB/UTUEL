"""
report.py
Compile all per-model result JSON files from `results/` into one summary report.

Output
──────
  table_retrieval/results/report.csv      — machine-readable
  table_retrieval/results/report.md       — human-readable Markdown table
  table_retrieval/results/report.html     — styled HTML (sortable)

Run from the repo root:
    python table_retrieval/report.py
    python table_retrieval/report.py --results_dir table_retrieval/results
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

_HERE        = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parent

# Metrics columns shown in the report (in order)
METRIC_COLS = ["MRR", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "Hit@20"]

# TSR sub-spaces reported as extra columns when present (tsr is already the primary MRR/Hit@k)
TSR_SPACES = ["tbl"]


def load_results(results_dir: Path) -> list[dict]:
    """
    Load all per-model result JSON files from results_dir.
    Skips manifest.json, top-k export files, and files with errors.
    """
    rows: list[dict] = []
    for p in sorted(results_dir.glob("*.json")):
        if p.stem in ("manifest",) or "_top" in p.stem:
            continue
        with p.open(encoding="utf-8") as f:
            data = json.load(f)
        if "error" in data:
            print(f"[SKIP] {p.name} — error: {data['error']}")
            continue
        if "metrics" not in data:
            continue
        rows.append(data)
    return rows


def build_report_df(results: list[dict]) -> pd.DataFrame:
    """Convert list of result dicts to a tidy DataFrame."""
    records = []
    for r in results:
        emb_cfg = r.get("embedder_cfg", {})
        record  = {
            "model":        r.get("embedder_label", "?"),
            "type":         emb_cfg.get("type", "?"),
            "dataset":      r.get("dataset", "?"),
            "n_queries":    r.get("n_queries", "?"),
            "n_tables":     r.get("n_tables", "?"),
            "embed_time_s": (
                r.get("embed_time_tables_s", 0)
                + r.get("embed_time_queries_s", 0)
            ),
        }
        tsr_vm   = r.get("tsr_variant_metrics") or {}
        tsr_main = tsr_vm.get("tsr", {})   # primary metrics for UTUEL models
        # For UTUEL models (tsr_variant_metrics present) use the tsr space as
        # the primary MRR/Hit@k columns; otherwise fall back to standard metrics.
        primary  = tsr_main if tsr_main else r.get("metrics", {})
        for col in METRIC_COLS:
            record[col] = primary.get(col, float("nan"))
        # Extra TSR sub-space columns (tbl_MRR, tbl_Hit@k, …)
        # "tbl_tsr" is the legacy key name; "tbl" is the current one.
        _TSR_ALIASES = {"tbl": ["tbl", "tbl_tsr"]}
        for sp in TSR_SPACES:
            aliases = _TSR_ALIASES.get(sp, [sp])
            sp_m = next((tsr_vm[k] for k in aliases if k in tsr_vm), {})
            for col in METRIC_COLS:
                record[f"{sp}_{col}"] = sp_m.get(col, float("nan"))
        records.append(record)

    df = pd.DataFrame(records)
    # Sort by MRR descending
    if "MRR" in df.columns:
        df = df.sort_values("MRR", ascending=False).reset_index(drop=True)
    return df


def format_markdown(df: pd.DataFrame) -> str:
    """Render DataFrame as a Markdown table with 4-decimal metric formatting."""
    display = df.copy()
    tsr_metric_cols = [f"{sp}_{col}" for sp in TSR_SPACES for col in METRIC_COLS]
    for col in METRIC_COLS + tsr_metric_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    if "embed_time_s" in display.columns:
        display["embed_time_s"] = display["embed_time_s"].map(
            lambda v: f"{v:.1f}s" if pd.notna(v) else "—"
        )
    cols  = list(display.columns)
    rows  = display.values.tolist()
    widths = [max(len(str(c)), max((len(str(r)) for r in col_vals), default=0))
               for c, col_vals in zip(cols, zip(*rows) if rows else [[] for _ in cols])]
    sep   = "| " + " | ".join("-" * w for w in widths) + " |"
    header = "| " + " | ".join(str(c).ljust(w) for c, w in zip(cols, widths)) + " |"
    lines  = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(v).ljust(w) for v, w in zip(row, widths)) + " |")
    return "\n".join(lines)


def format_html(df: pd.DataFrame) -> str:
    """Render DataFrame as a styled HTML table."""
    display = df.copy()
    tsr_metric_cols = [f"{sp}_{col}" for sp in TSR_SPACES for col in METRIC_COLS]
    for col in METRIC_COLS + tsr_metric_cols:
        if col in display.columns:
            display[col] = display[col].map(lambda v: f"{v:.4f}" if pd.notna(v) else "—")
    if "embed_time_s" in display.columns:
        display["embed_time_s"] = display["embed_time_s"].map(
            lambda v: f"{v:.1f}s" if pd.notna(v) else "—"
        )

    html = display.to_html(index=False, border=0, classes="retrieval-report")
    style = """
<style>
  .retrieval-report { border-collapse: collapse; font-family: monospace; font-size: 13px; }
  .retrieval-report th { background: #2c3e50; color: #ecf0f1; padding: 8px 12px; text-align: left; }
  .retrieval-report td { padding: 6px 12px; border-bottom: 1px solid #ddd; }
  .retrieval-report tr:nth-child(even) td { background: #f9f9f9; }
  .retrieval-report tr:hover td { background: #eaf3fb; }
</style>
"""
    return style + html


def main(results_dir: str | Path) -> None:
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"[ERROR] Results directory not found: {results_dir}")
        return

    print(f"Loading results from {results_dir} …")
    results = load_results(results_dir)

    if not results:
        print("No result files found. Run `python table_retrieval/evaluate.py` first.")
        return

    df = build_report_df(results)

    # ── CSV ────────────────────────────────────────────────────────────────────
    csv_path = results_dir / "report.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV  → {csv_path}")

    # ── Markdown ───────────────────────────────────────────────────────────────
    md_path = results_dir / "report.md"
    md_text = f"# Table Retrieval — Model Comparison\n\n{format_markdown(df)}\n"
    md_path.write_text(md_text, encoding="utf-8")
    print(f"MD   → {md_path}")

    # ── HTML ───────────────────────────────────────────────────────────────────
    html_path = results_dir / "report.html"
    html_text = (
        "<html><head><meta charset='utf-8'>"
        "<title>Table Retrieval Report</title></head><body>"
        "<h2>Table Retrieval — Model Comparison</h2>"
        f"{format_html(df)}</body></html>"
    )
    html_path.write_text(html_text, encoding="utf-8")
    print(f"HTML → {html_path}")

    # ── Console summary ────────────────────────────────────────────────────────
    print(f"\n{'='*80}")
    print("TABLE RETRIEVAL — MODEL COMPARISON")
    print(f"{'='*80}")
    print(format_markdown(df))
    print(f"{'='*80}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compile table retrieval report")
    parser.add_argument(
        "--results_dir",
        default=str(PROJECT_ROOT / "table_retrieval" / "results"),
        help="Directory containing per-model result JSON files",
    )
    args = parser.parse_args()
    main(args.results_dir)
