"""
check_answer_cell_ids.py
Validate (and optionally correct) the ``answer_cell_id`` field of every question
in ``QUESTIONS_ANSWERS_PER_TABLE/*.json`` against the table's cell descriptions
in ``AdhesiveTable_json_format/*.json``.

Rule
────
For a question, ``answer_cell_id`` must be the string form of the ``id-entry`` of
the cell whose ``content`` equals the question ``answer``.

The script classifies every question and, with ``--apply``, rewrites only the
*unambiguous* fixes (exactly one cell whose content matches the answer) after
writing a ``<file>.json.bak`` backup.  Ambiguous / unmatched cases are reported
for manual review and never touched.

Usage
    python check_answer_cell_ids.py                 # dry-run report
    python check_answer_cell_ids.py --apply         # apply safe fixes (+backups)
    python check_answer_cell_ids.py --arrays questions questions_expert
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

QA_SUBDIR   = "QUESTIONS_ANSWERS_PER_TABLE"
JSON_SUBDIR = "AdhesiveTable_json_format"
DEFAULT_DATA_DIR = Path(
    r"D:\TABLE_DATASET\DATASET_ABOUT_ADHESIVE\ReleasedTableDatasetAdhesive")


def _norm(s: str) -> str:
    """Whitespace-collapsed, lower-cased comparison key (strict)."""
    return re.sub(r"\s+", " ", str(s).strip().lower())


# Unicode / notation variants that show up between answers and cell contents.
_UNI = {
    "±": "+/-", "→": "->", "⟶": "->", "×": "x",
    "³": "3", "²": "2", "—": "-", "–": "-", "‐": "-",
}


def _fuzzy(s: str) -> str:
    """Formatting-insensitive key: unify notation, drop spaces/.-, comparison."""
    s = str(s).strip().lower()
    for k, v in _UNI.items():
        s = s.replace(k, v)
    s = s.replace("-->", "->")
    return re.sub(r"[\s,.\-]", "", s)


def _load_cells(json_path: Path) -> dict[int, str]:
    """Return ``{id-entry: content}`` for a table cell JSON."""
    data = json.loads(json_path.read_text(encoding="utf-8"))
    cells: dict[int, str] = {}
    for c in data.get("Cells", []):
        try:
            cid = int(c["id-entry"])
        except (KeyError, ValueError, TypeError):
            continue
        cells[cid] = str(c.get("content", "")).strip()
    return cells


def _as_int_id(value) -> int | None:
    """Parse an answer_cell_id into an int, or None if it is not a plain id."""
    if value is None:
        return None
    s = str(value).strip()
    if re.fullmatch(r"-?\d+", s):
        return int(s)
    return None


def classify(answer: str, cur_id, cells: dict[int, str]) -> dict:
    """Classify one question's answer_cell_id and propose a correction."""
    ans_n, ans_f = _norm(answer), _fuzzy(answer)
    exact = [cid for cid, c in cells.items() if _norm(c) == ans_n]
    fuzzy = [cid for cid, c in cells.items() if ans_f and _fuzzy(c) == ans_f]

    cur_int = _as_int_id(cur_id)
    cur_content = cells.get(cur_int) if cur_int is not None else None
    cur_f = _fuzzy(cur_content) if cur_content is not None else None
    cur_ok = cur_content is not None and (_norm(cur_content) == ans_n or cur_f == ans_f)
    # answer is a sub-span of the current cell (or vice-versa) → id plausibly right
    cur_partial = bool(cur_f) and (cur_f in ans_f or ans_f in cur_f)

    if cur_ok:
        status, proposed = "ok", str(cur_int)
    elif len(exact) == 1 and exact[0] != cur_int:
        status, proposed = "fix_exact", str(exact[0])
    elif len(fuzzy) == 1 and fuzzy[0] != cur_int:
        status, proposed = "fix_fuzzy", str(fuzzy[0])
    elif len(exact) > 1 or len(fuzzy) > 1:
        status, proposed = "ambiguous", None
    elif cur_partial:
        status, proposed = "ok_partial", str(cur_int)
    else:
        status, proposed = "no_match", None

    return {
        "status": status, "proposed": proposed,
        "exact": exact, "fuzzy": fuzzy,
        "cur_int": cur_int, "cur_content": cur_content,
    }


def process(data_dir: Path, arrays: list[str], apply: bool) -> None:
    qa_dir   = data_dir / QA_SUBDIR
    cell_dir = data_dir / JSON_SUBDIR
    if not qa_dir.is_dir() or not cell_dir.is_dir():
        raise SystemExit(f"Expected {QA_SUBDIR}/ and {JSON_SUBDIR}/ under {data_dir}")

    counts = {"ok": 0, "ok_partial": 0, "fix_exact": 0, "fix_fuzzy": 0,
              "ambiguous": 0, "no_match": 0, "missing_cells": 0}
    fixes: list[str] = []
    ambiguous: list[str] = []
    no_match: list[str] = []
    files_changed = 0

    for qa_file in sorted(qa_dir.glob("*.json")):
        tid = qa_file.stem
        cell_file = cell_dir / f"{tid}.json"
        if not cell_file.is_file():
            counts["missing_cells"] += 1
            print(f"[WARN] no cell file for {tid}")
            continue
        cells = _load_cells(cell_file)

        data = json.loads(qa_file.read_text(encoding="utf-8"))
        file_dirty = False

        for arr in arrays:
            for qi, q in enumerate(data.get(arr, []) or []):
                if "answer_cell_id" not in q:
                    continue
                answer = q.get("answer", "")
                cur_id = q.get("answer_cell_id")
                r = classify(answer, cur_id, cells)
                counts[r["status"]] += 1
                loc = f"{tid} [{arr}#{qi}]"

                if r["status"] in ("fix_exact", "fix_fuzzy"):
                    tag = "exact" if r["status"] == "fix_exact" else "fuzzy"
                    fixes.append(
                        f"[{tag}] {loc}  answer={answer!r}  {cur_id!r} -> {r['proposed']!r}"
                        f"  (cell={cells.get(int(r['proposed']))!r})")
                    if apply:
                        q["answer_cell_id"] = r["proposed"]
                        file_dirty = True
                elif r["status"] == "ambiguous":
                    opts = {cid: cells[cid] for cid in (r["exact"] or r["fuzzy"])}
                    ambiguous.append(f"{loc}  answer={answer!r}  cur={cur_id!r}  matches={opts}")
                elif r["status"] == "no_match":
                    no_match.append(
                        f"{loc}  answer={answer!r}  cur={cur_id!r}  cur_content={r['cur_content']!r}")

        if apply and file_dirty:
            bak = qa_file.with_suffix(".json.bak")
            if not bak.exists():
                bak.write_text(qa_file.read_text(encoding="utf-8"), encoding="utf-8")
            qa_file.write_text(
                json.dumps(data, indent="\t", ensure_ascii=False), encoding="utf-8")
            files_changed += 1

    # ── Report ────────────────────────────────────────────────────────────────
    print("\n================ answer_cell_id audit ================")
    print(f"arrays checked : {arrays}")
    print(f"OK             : {counts['ok']}")
    print(f"OK (partial)   : {counts['ok_partial']}   (answer is a sub-span of the current cell)")
    print(f"FIX (exact)    : {counts['fix_exact']}   (unique exact content match elsewhere)")
    print(f"FIX (fuzzy)    : {counts['fix_fuzzy']}   (unique formatting-variant match elsewhere)")
    print(f"AMBIGUOUS      : {counts['ambiguous']}   (multiple cells match the answer)")
    print(f"NO_MATCH       : {counts['no_match']}   (answer not found in any cell)")
    if counts["missing_cells"]:
        print(f"MISSING CELLS  : {counts['missing_cells']} table(s)")

    def _dump(title: str, rows: list[str], limit: int = 60) -> None:
        if not rows:
            return
        print(f"\n---- {title} ({len(rows)}) ----")
        for row in rows[:limit]:
            print("  " + row)
        if len(rows) > limit:
            print(f"  … {len(rows) - limit} more")

    _dump("FIXES (applied)" if apply else "FIXES (proposed)", fixes)
    _dump("AMBIGUOUS — manual review", ambiguous)
    _dump("NO_MATCH — manual review", no_match)

    if apply:
        print(f"\n[apply] rewrote {files_changed} file(s) with backups (*.json.bak)")
    else:
        print("\n[dry-run] no files modified. Re-run with --apply to write fixes.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    ap.add_argument("--arrays", nargs="+", default=["questions", "questions_expert"],
                    help="Which question arrays to check (default: both).")
    ap.add_argument("--apply", action="store_true",
                    help="Write the unambiguous fixes (creates *.json.bak backups).")
    args = ap.parse_args()
    process(args.data_dir, args.arrays, args.apply)


if __name__ == "__main__":
    main()
