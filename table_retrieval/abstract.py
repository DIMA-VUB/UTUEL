"""
abstract.py
Abstract base class for table retrieval embedders.

To add a new model to the benchmark
────────────────────────────────────
1. Subclass TableRetrieverBase.
2. Implement encode_table_corpus() and encode_queries().
3. Register the class with @register_embedder (or add it to EMBEDDER_REGISTRY).
4. Add an entry to table_retrieval/config.yaml:

       embedder:
         type: custom
         class: MyEmbedder          # must match the registered name
         model_name: my-model-label
         <any extra keys your __init__ needs>

5. Run:  python table_retrieval/evaluate.py

The two abstract methods are the only required contract. The rest of the
helpers (record_to_document, l2_norm) are provided for convenience.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


# ── Registry ──────────────────────────────────────────────────────────────────
# Maps class name (str) → class object.
# Use @register_embedder or add entries directly.
EMBEDDER_REGISTRY: dict[str, type["TableRetrieverBase"]] = {}


def register_embedder(cls: type) -> type:
    """
    Class decorator that adds the class to the global embedder registry.

    Usage::

        @register_embedder
        class MyEmbedder(TableRetrieverBase):
            ...
    """
    EMBEDDER_REGISTRY[cls.__name__] = cls
    return cls


# ── Abstract base class ───────────────────────────────────────────────────────

class TableRetrieverBase(ABC):
    """
    Abstract base class for table retrieval embedders.

    Every concrete embedder must implement:
      encode_table_corpus(records)  →  np.ndarray [T, D]   (L2-normalised)
      encode_queries(queries)       →  np.ndarray [Q, D]   (L2-normalised)

    And expose:
      label  : str  — human-readable identifier shown in the report

    Optionally override:
      encode_table_corpus_progress() to add custom progress printing.

    Static utilities (available to all subclasses):
      record_to_document(record)   — default table → text serialisation
      l2_norm(arr)                 — L2-row-normalise a float32 array
    """

    label: str = "unnamed"

    # ── Required interface ────────────────────────────────────────────────────

    @abstractmethod
    def encode_table_corpus(self, records: list[dict]) -> np.ndarray:
        """
        Encode a list of unique table records into a corpus matrix.

        Parameters
        ----------
        records : list[dict]
            One dict per unique table (already deduplicated by table_id).
            Each dict contains at minimum:
              'table_id'  str
              'header'    list[str]
              'rows'      list[list[str]]

        Returns
        -------
        np.ndarray of shape [T, D], dtype float32, each row L2-normalised.
        """

    @abstractmethod
    def encode_queries(self, queries: list[str]) -> np.ndarray:
        """
        Encode a list of natural-language queries.

        Parameters
        ----------
        queries : list[str]

        Returns
        -------
        np.ndarray of shape [Q, D], dtype float32, each row L2-normalised.
        """

    # ── Convenience helpers ───────────────────────────────────────────────────

    @staticmethod
    def record_to_document(record: dict) -> str:
        """
        Default table-to-text serialisation: pipe-separated header + rows.

        Example output::

            Player | No. | Nationality | Position
            Terrence Ross | 31 | United States | Guard
            ...
        """
        sep = " | "
        lines = [sep.join(str(h).strip() for h in record["header"])]
        for row in record["rows"]:
            lines.append(sep.join(str(c).strip() for c in row))
        return "\n".join(lines)

    @staticmethod
    def l2_norm(arr: np.ndarray) -> np.ndarray:
        """Row-wise L2-normalise a 2-D float32 array. Safe against zero vectors."""
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0.0, 1.0, norms)
        return (arr / norms).astype(np.float32)
