"""
add_stopword_questions.py
Augment every question in the ``QUESTIONS_ANSWERS_PER_TABLE`` files with a
stop-word-removed variant, stored under the ``question_stopword`` key next to
the original ``question``.

Usage
    python add_stopword_questions.py [DATA_DIR]

``DATA_DIR`` defaults to the Adhesive dataset root; the script only touches the
``QUESTIONS_ANSWERS_PER_TABLE`` sub-folder.  Files are rewritten in place as
UTF-8 with the original indentation (tab) so diffs stay minimal.  Re-running is
idempotent and refreshes the SW variant.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

try:
    from .stopwords_util import remove_stopwords
except ImportError:
    from stopwords_util import remove_stopwords

DEFAULT_DATA_DIR = Path(
    r"D:\TABLE_DATASET\DATASET_ABOUT_ADHESIVE\ReleasedTableDatasetAdhesive")
QA_SUBDIR = "QUESTIONS_ANSWERS_PER_TABLE"
SW_KEY = "question_stopword"


def augment_file(path: Path) -> int:
    """Add/refresh the SW variant for every question in *path*; return count."""
    data = json.loads(path.read_text(encoding="utf-8"))
    questions = data.get("questions", []) or []
    n = 0
    for q in questions:
        raw = str(q.get("question", ""))
        if not raw:
            continue
        q[SW_KEY] = remove_stopwords(raw)
        n += 1
    path.write_text(
        json.dumps(data, indent="\t", ensure_ascii=False), encoding="utf-8")
    return n


def main(data_dir: Path) -> None:
    qa_dir = data_dir / QA_SUBDIR
    if not qa_dir.is_dir():
        raise FileNotFoundError(f"QA directory not found: {qa_dir}")

    files = sorted(qa_dir.glob("*.json"))
    total_q = 0
    for i, f in enumerate(files, 1):
        total_q += augment_file(f)
        print(f"\r[sw] {i}/{len(files)} files  {total_q} questions", end="", flush=True)
    print(f"\n[sw] done — added '{SW_KEY}' to {total_q} questions in {len(files)} files")


if __name__ == "__main__":
    _dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_DATA_DIR
    main(_dir)
