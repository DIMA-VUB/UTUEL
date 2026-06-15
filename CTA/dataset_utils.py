"""
dataset_utils.py
Shared helpers for the CTA dataset: embedder factory and type_vocab loader.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional


# ── Type vocabulary ───────────────────────────────────────────────────────────

def load_type_vocab(vocab_path: str | Path) -> tuple[dict[str, int], dict[int, str]]:
    """
    Load the type_vocab file (one Freebase type string per line).

    Returns
    -------
    type2idx : dict[str, int]   — type string → class index
    idx2type : dict[int, str]   — class index → type string
    """
    vocab_path = Path(vocab_path)
    type2idx: dict[str, int] = {}
    idx2type: dict[int, str] = {}
    with vocab_path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            # Support both "type_string" (plain) and "<idx>\ttype_string" (tabular) formats
            parts = line.split("\t", 1)
            t = parts[1].strip() if len(parts) == 2 else parts[0]
            if t:
                type2idx[t] = idx
                idx2type[idx] = t
    print(f"[CTA][vocab] loaded {len(type2idx)} types from {vocab_path.name}")
    return type2idx, idx2type


# ── Data path resolver ────────────────────────────────────────────────────────────

_SPLIT_SUFFIXES = {
    "train":      "train.table_col_type.json",
    "dev":        "dev.table_col_type.json",
    "test":       "test.table_col_type.json",
    "type_vocab": "type_vocab.txt",
}


def resolve_data_paths(data_cfg) -> dict[str, str]:
    """
    Resolve the four CTA data paths from a Hydra `data` config node.

    If `data.folder` is set, all four paths are derived automatically:
        <folder>/train.table_col_type
        <folder>/dev.table_col_type
        <folder>/test.table_col_type
        <folder>/type_vocab

    Otherwise the explicit `train_path`, `dev_path`, `test_path`, and
    `type_vocab_path` keys are used as-is.

    Returns a dict with keys: 'train', 'dev', 'test', 'type_vocab'.
    """
    folder = data_cfg.get("folder") if hasattr(data_cfg, "get") else getattr(data_cfg, "folder", None)
    if folder:
        base = Path(folder)
        return {k: str(base / suffix) for k, suffix in _SPLIT_SUFFIXES.items()}
    return {
        "train":      str(data_cfg.train_path),
        "dev":        str(data_cfg.dev_path),
        "test":       str(data_cfg.test_path),
        "type_vocab": str(data_cfg.type_vocab_path),
    }

def get_embedder(
    model_type: str = "huggingface",
    base_url: Optional[str] = None,
    model_name: Optional[str] = None,
    api_key: Optional[str] = None,
):
    """
    Return a LangChain embedder for the requested backend.

    Parameters
    ----------
    model_type : 'openai' | 'huggingface' | <Ollama model name>
    base_url   : Ollama server URL (ignored for other backends)
    model_name : HuggingFace Hub model identifier
    api_key    : API key for OpenAI / HuggingFace Hub
    """
    tag = model_type.strip().lower()

    if tag == "openai":
        from langchain_openai import OpenAIEmbeddings
        return OpenAIEmbeddings(model="text-embedding-3-large", api_key=api_key)

    if tag == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings
        hf_model = model_name or "sentence-transformers/all-MiniLM-L6-v2"
        return HuggingFaceEmbeddings(model_name=hf_model)

    # Default: Ollama
    from langchain_ollama import OllamaEmbeddings
    return OllamaEmbeddings(
        base_url=base_url or "http://localhost:11434/",
        model=tag,
    )
