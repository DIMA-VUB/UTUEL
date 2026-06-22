"""
embedder.py
Text-based embedding backends for table retrieval evaluation.

Supported backends
──────────────────
  huggingface  — sentence-transformers loaded locally (SentenceTransformer)
  ollama       — langchain_ollama.OllamaEmbeddings (asymmetric embed_documents /
                  embed_query)
  custom       — any class registered in EMBEDDER_REGISTRY (e.g. UTUELTableEmbedder
                  from utuel_embedder.py)

Public API
──────────
    emb = build_embedder(embedder_cfg, project_root)
    emb.encode_table_corpus(unique_records)   # np.ndarray [T, D], L2-normalised
    emb.encode_queries(queries)               # np.ndarray [Q, D], L2-normalised
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np

from abstract import TableRetrieverBase, EMBEDDER_REGISTRY  # type: ignore[import]

# Suppress verbose httpx logging from LangChain OllamaEmbeddings
logging.getLogger("httpx").setLevel(logging.WARNING)


# ── HuggingFace (sentence-transformers) ──────────────────────────────────────

class HuggingFaceEmbedder(TableRetrieverBase):
    """Wraps sentence_transformers.SentenceTransformer."""

    def __init__(self, model_name: str, batch_size: int = 64, device: str | None = None):
        import torch
        from sentence_transformers import SentenceTransformer  # type: ignore[import-untyped]
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device     = device
        self.model      = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size
        self.dim        = self.model.get_sentence_embedding_dimension()
        self.label      = f"hf:{model_name}"
        print(f"  [HuggingFaceEmbedder] using device: {device}")

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

    # Symmetric model — document and query use the same encoding path
    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        return self._encode(texts)

    # TableRetrieverBase required methods
    def encode_table_corpus(self, records: list[dict]) -> np.ndarray:
        texts = [self.record_to_document(r) for r in records]
        return self.encode_documents(texts)


# ── Ollama ───────────────────────────────────────────────────────────────────

class OllamaEmbedder(TableRetrieverBase):
    """
    Asymmetric embedder backed by langchain_ollama.OllamaEmbeddings.

    LangChain's OllamaEmbeddings exposes two separate methods:
      embed_documents(texts)  — passage / document side
      embed_query(text)       — query side

    For models that support asymmetric embedding (e.g. mxbai-embed-large with
    the "Represent this sentence for searching relevant passages:" prefix),
    these two paths produce different vectors, which is the correct behaviour
    for retrieval.  For symmetric models (e.g. nomic-embed-text) both paths
    produce the same vectors.
    """

    def __init__(self, base_url: str, model_name: str, batch_size: int = 64,
                 max_chars: int | str | None = None):
        from langchain_ollama import OllamaEmbeddings  # type: ignore[import-untyped]
        self.model_name = model_name
        self.batch_size = batch_size
        self.dim: Optional[int] = None
        self.label      = f"ollama:{model_name}"
        self._lc = OllamaEmbeddings(
            base_url = base_url.rstrip("/"),
            model    = model_name,
        )
        if max_chars is None or str(max_chars).lower() == "auto":
            self.max_chars = self._auto_max_chars(base_url, model_name)
        else:
            self.max_chars = int(max_chars)

    @staticmethod
    def _auto_max_chars(
        base_url:      str,
        model_name:    str,
        chars_per_tok: float = 3.5,
        safety:        float = 0.9,
        fallback:      int   = 4096,
    ) -> int:
        """
        Query ``/api/show`` for the model's context window, then convert to a
        safe character limit::

            max_chars = floor(context_tokens × chars_per_tok × safety)

        ``chars_per_tok=3.5`` is conservative for English table text.
        ``safety=0.9`` leaves a 10 % headroom below the hard limit.
        Falls back to ``fallback`` chars if the API call fails.
        """
        import json
        import urllib.request
        try:
            url     = base_url.rstrip("/") + "/api/show"
            payload = json.dumps({"name": model_name}).encode()
            req     = urllib.request.Request(
                url, data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                data = json.loads(resp.read().decode())
            info = data.get("model_info", {})
            # context_length key varies by model architecture
            ctx = (
                info.get("llama.context_length")
                or info.get("bert.context_length")
                or info.get("nomic-bert.context_length")
                or info.get("qwen2.context_length")
                or info.get("falcon.context_length")
            )
            if ctx is None:
                # Older Ollama versions expose it as a parameter string
                for line in data.get("parameters", "").splitlines():
                    if line.strip().startswith("num_ctx"):
                        ctx = int(line.split()[-1])
                        break
            if ctx is None:
                raise ValueError("context_length not found in /api/show response")
            chars = int(ctx * chars_per_tok * safety)
            print(f"  [OllamaEmbedder] auto max_chars={chars:,}  "
                  f"(ctx={ctx} tok × {chars_per_tok} ch/tok × {safety} safety)")
            return chars
        except Exception as exc:
            print(f"  [OllamaEmbedder] could not auto-detect context length: {exc}; "
                  f"falling back to {fallback:,} chars")
            return fallback

    @staticmethod
    def _l2_norm(arr: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1.0, norms)
        return arr / norms

    def _to_array(self, vecs: list[list[float]]) -> np.ndarray:
        arr = np.array(vecs, dtype=np.float32)
        if self.dim is None:
            self.dim = arr.shape[1]
        return self._l2_norm(arr)

    def _embed_one(self, text: str, query: bool = False) -> list[float]:
        """Embed a single text, halving length on context-length errors until it fits."""
        limit = self.max_chars
        while limit >= 64:
            t = text[:limit]
            try:
                return self._lc.embed_query(t) if query else self._lc.embed_documents([t])[0]
            except Exception as e:
                if "context length" in str(e).lower():
                    limit //= 2
                else:
                    raise
        raise RuntimeError(f"Text still too long after repeated halving (tried down to 64 chars)")

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Embed a corpus of passages using embed_documents (passage side)."""
        all_vecs: list[list[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t[: self.max_chars] for t in texts[i : i + self.batch_size]]
            try:
                all_vecs.extend(self._lc.embed_documents(batch))
            except Exception as e:
                if "context length" in str(e).lower():
                    # Fall back to one-by-one with adaptive truncation
                    for t in batch:
                        all_vecs.append(self._embed_one(t))
                else:
                    raise
            done = min(i + self.batch_size, len(texts))
            print(f"  Ollama embed_documents: {done}/{len(texts)}", end="\r", flush=True)
        print()
        return self._to_array(all_vecs)

    def encode_queries(self, texts: list[str]) -> np.ndarray:
        """Embed queries using embed_query (query side — asymmetric prefix applied)."""
        all_vecs: list[list[float]] = []
        for i, text in enumerate(texts):
            all_vecs.append(self._embed_one(text, query=True))
            if (i + 1) % 100 == 0 or (i + 1) == len(texts):
                print(f"  Ollama embed_query: {i + 1}/{len(texts)}", end="\r", flush=True)
        print()
        return self._to_array(all_vecs)

    # TableRetrieverBase required method
    def encode_table_corpus(self, records: list[dict]) -> np.ndarray:
        texts = [self.record_to_document(r) for r in records]
        return self.encode_documents(texts)


# ── Factory ───────────────────────────────────────────────────────────────────

def build_embedder(
    embedder_cfg: dict,
    project_root: str | Path = ".",
) -> TableRetrieverBase:
    """
    Instantiate the correct embedder from a config dict.

    Supported types
    ───────────────
    huggingface  : model_name (str)
    ollama       : base_url, model_name (str)
    custom       : class (str) — class name registered in EMBEDDER_REGISTRY.
                   The class receives (embedder_cfg, project_root) as args.
                   Add your own class with @register_embedder and ensure its
                   module is importable (or listed in the auto-import list below).
    """
    etype      = embedder_cfg.get("type", "huggingface")
    batch_size = embedder_cfg.get("batch_size", 64)

    if etype == "huggingface":
        return HuggingFaceEmbedder(
            model_name = embedder_cfg["model_name"],
            batch_size = batch_size,
            device     = embedder_cfg.get("device"),  # None → auto-detect CUDA
        )

    if etype == "ollama":
        return OllamaEmbedder(
            base_url   = embedder_cfg.get("base_url", "http://localhost:11434"),
            model_name = embedder_cfg["model_name"],
            batch_size = batch_size,
            max_chars  = embedder_cfg.get("max_chars"),   # None → auto-detect from /api/show
        )

    if etype == "custom":
        class_name = embedder_cfg.get("class")
        if not class_name:
            raise ValueError("embedder.type=custom requires embedder.class to be set")
        # Auto-import the UTUEL embedder if not yet registered
        if class_name not in EMBEDDER_REGISTRY:
            _here = Path(__file__).resolve().parent
            if str(_here) not in sys.path:
                sys.path.insert(0, str(_here))
            import importlib
            for _mod in ("utuel_embedder",):   # add more module names here as needed
                try:
                    importlib.import_module(_mod)
                except ImportError:
                    pass
        if class_name not in EMBEDDER_REGISTRY:
            raise ValueError(
                f"Custom embedder class '{class_name}' not found in registry. "
                f"Available: {sorted(EMBEDDER_REGISTRY.keys())}.\n"
                f"Decorate your class with @register_embedder and ensure its "
                f"module is importable."
            )
        cls = EMBEDDER_REGISTRY[class_name]
        return cls(embedder_cfg, project_root)

    raise ValueError(f"Unknown embedder type: {etype!r}")
