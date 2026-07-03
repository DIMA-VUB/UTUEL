"""
compute_hit_threshold.py
Compute the HIT (High-informative Token) classification threshold ``alpha_0`` for
a given embedding model, following Fig. 4 of the paper.

Process
───────
1. Each labelled question carries a ``binary_vector`` whose i-th entry is the HIT
   label ``L`` of the i-th whitespace token (1 = HIT, 0 = non-HIT).  Only
   questions whose ``question.split()`` length matches the ``binary_vector``
   length are used (≈625/816 on Adhesive).
2. For every such question, each token is embedded and scored by the cosine
   similarity ``alpha`` to the full-question embedding; the per-question scores
   are **min-max normalised** to ``[0, 1]``.
3. All normalised scores are pooled and split by label.  A Gaussian is fitted to
   the ``L=0`` set and to the ``L=1`` set.
4. The threshold ``alpha_0`` is the intersection of the two Gaussian densities in
   the overlap region between their means.
5. A Fig-4-style plot is saved as ``hit_threshold_<model-slug>.png`` / ``.pdf``.

Usage
    python compute_hit_threshold.py                # uses config.yaml defaults
    python compute_hit_threshold.py --threshold-only
    python compute_hit_threshold.py --model-name sentence-transformers/all-MiniLM-L6-v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.stats import norm

QA_SUBDIR = "QUESTIONS_ANSWERS_PER_TABLE"
_HERE = Path(__file__).resolve().parent


# ── Data loading ──────────────────────────────────────────────────────────────

def load_labelled_questions(data_dir: Path) -> tuple[list[tuple[str, list[str], list[int]]], int, int]:
    """Return ``[(question, tokens, labels), …]`` for questions with a matching
    ``binary_vector``, plus (n_with_binary_vector, n_total_questions)."""
    qa_dir = Path(data_dir) / QA_SUBDIR
    if not qa_dir.is_dir():
        raise FileNotFoundError(f"QA directory not found: {qa_dir}")

    out: list[tuple[str, list[str], list[int]]] = []
    n_total = n_bv = 0
    for f in sorted(qa_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for q in data.get("questions", []) or []:
            n_total += 1
            bv = q.get("binary_vector")
            if not bv:
                continue
            n_bv += 1
            question = str(q.get("question", ""))
            tokens = question.split()
            if len(tokens) != len(bv):
                continue                       # split does not align with labels
            out.append((question, tokens, [int(x) for x in bv]))
    return out, n_bv, n_total


# ── Embedding backend (mirrors dataset._embed_batch_chunk) ────────────────────

def build_embed_fn(
    model_type: str = "huggingface",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    batch_size: int = 64,
    truncate_dim: Optional[int] = None,
) -> Callable[[list[str]], np.ndarray]:
    """Return ``fn(texts) -> [N, d] float32`` for the requested backend.

    ``truncate_dim`` mirrors config ``embedder.embed_dim``: ``None`` keeps the
    model's native width, a set value truncates every embedding to the first N.
    """
    tag = model_type.strip().lower()

    def _cap(arr: np.ndarray) -> np.ndarray:
        arr = np.asarray(arr, dtype=np.float32)
        if truncate_dim is not None and arr.ndim == 2 and arr.shape[1] > truncate_dim:
            return arr[:, :truncate_dim]
        return arr

    if tag == "huggingface" and not api_key:
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer(model_name)

        def _fn(texts: list[str]) -> np.ndarray:
            vecs = st.encode(texts, batch_size=batch_size, show_progress_bar=False)
            return _cap(np.asarray(vecs, dtype=np.float32))
        return _fn

    import requests

    def _fn(texts: list[str]) -> np.ndarray:
        if tag == "huggingface":
            resp = requests.post(
                f"https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}",
                headers={"Authorization": f"Bearer {api_key}"},
                json={"inputs": texts}, timeout=120000)
            resp.raise_for_status()
            return _cap(np.asarray(resp.json(), dtype=np.float32))
        resp = requests.post(
            f"{(base_url or '').rstrip('/')}/api/embed",
            json={"model": model_name or tag, "input": texts}, timeout=120000)
        resp.raise_for_status()
        return _cap(np.asarray(resp.json()["embeddings"], dtype=np.float32))
    return _fn


# ── Core computation ──────────────────────────────────────────────────────────

def _normalised_scores(
    labelled: list[tuple[str, list[str], list[int]]],
    embed_fn: Callable[[list[str]], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Return pooled (alpha_norm, L) arrays and the embedding dimension."""
    # Batch-embed every unique token + question text once.
    idx: dict[str, int] = {}
    for question, tokens, _ in labelled:
        for t in (question, *tokens):
            if t not in idx:
                idx[t] = len(idx)
    texts = [""] * len(idx)
    for t, i in idx.items():
        texts[i] = t
    emb = embed_fn(texts)
    embn = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)

    alphas: list[float] = []
    labels: list[int] = []
    for question, tokens, lab in labelled:
        qv = embn[idx[question]]
        tv = embn[[idx[t] for t in tokens]]
        alpha = tv @ qv                                    # cosine per token
        lo, hi = float(alpha.min()), float(alpha.max())
        if hi - lo < 1e-12:
            continue                                       # degenerate question
        a_norm = (alpha - lo) / (hi - lo)
        alphas.extend(a_norm.tolist())
        labels.extend(lab)
    return np.asarray(alphas), np.asarray(labels), int(emb.shape[1])


def _gaussian_intersection(m0: float, s0: float, m1: float, s1: float) -> float:
    """First crossing of the two Gaussian densities between their means."""
    lo, hi = sorted((m0, m1))
    xs = np.linspace(lo, hi, 20001)
    diff = norm.pdf(xs, m1, s1) - norm.pdf(xs, m0, s0)
    cross = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(cross):
        return float(xs[cross[0]])
    return float(xs[int(np.argmin(np.abs(diff)))])


# ── Plot + public entry point ─────────────────────────────────────────────────

def compute_hit_threshold(
    data_dir: str | Path,
    model_type: str = "huggingface",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    out_dir: str | Path | None = None,
    save: bool = True,
    truncate_dim: Optional[int] = None,
):
    """Compute the HIT threshold for *model_name* and save the Fig-4 plot.

    Returns ``(result_dict, matplotlib_figure)``.  ``result_dict`` holds the
    threshold, the fitted Gaussian parameters, the conditional probabilities and
    the counts, so callers (e.g. a notebook) can display / persist them.
    """
    import matplotlib.pyplot as plt

    labelled, n_bv, n_total = load_labelled_questions(data_dir)
    print(f"[hit_thr] questions with binary_vector: {n_bv}/{n_total}  |  "
          f"usable (split matches labels): {len(labelled)}")
    if not labelled:
        raise ValueError("No usable labelled questions found.")

    embed_fn = build_embed_fn(model_type, model_name, base_url, api_key,
                              truncate_dim=truncate_dim)
    alpha, L, dim = _normalised_scores(labelled, embed_fn)

    a0, a1 = alpha[L == 0], alpha[L == 1]
    m0, s0 = float(a0.mean()), float(a0.std())
    m1, s1 = float(a1.mean()), float(a1.std())
    thr = _gaussian_intersection(m0, s0, m1, s1)

    # Conditional probabilities under the fitted Gaussians.
    p0_lt = float(norm.cdf(thr, m0, s0))          # P(L=0 | alpha < thr)
    p0_gt = 1.0 - p0_lt
    p1_lt = float(norm.cdf(thr, m1, s1))          # P(L=1 | alpha < thr)
    p1_gt = 1.0 - p1_lt

    slug = model_name.replace("/", "-").replace("\\", "-")

    # ── Plot (Fig. 4 style) ────────────────────────────────────────────────
    xs = np.linspace(0.0, 1.0, 1000)
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    ax.plot(xs, norm.pdf(xs, m0, s0), color="tab:blue",
            label=f"L=0 (mean={m0:.2f}, std={s0:.2f})")
    ax.plot(xs, norm.pdf(xs, m1, s1), color="tab:red",
            label=f"L=1 (mean={m1:.2f}, std={s1:.2f})")
    ax.axvline(thr, color="green", linestyle="--", linewidth=1)
    ax.plot([thr], [norm.pdf(thr, m0, s0)], "o", color="green",
            label=f"Threshold: \u03b1\u2080={thr:.2f}")

    ax.text(0.03, 0.78, f"P(L=0|\u03b1<\u03b1\u2080={thr:.2f}) = {p0_lt:.2f}",
            transform=ax.transAxes, color="tab:blue", fontsize=9)
    ax.text(0.03, 0.70, f"P(L=0|\u03b1>\u03b1\u2080={thr:.2f}) = {p0_gt:.2f}",
            transform=ax.transAxes, color="tab:blue", fontsize=9)
    ax.text(0.40, 0.34, f"P(L=1|\u03b1<\u03b1\u2080={thr:.2f}) = {p1_lt:.2f}",
            transform=ax.transAxes, color="tab:red", fontsize=9)
    ax.text(0.40, 0.26, f"P(L=1|\u03b1>\u03b1\u2080={thr:.2f}) = {p1_gt:.2f}",
            transform=ax.transAxes, color="tab:red", fontsize=9)

    ax.set_title(f"Gaussian Fit of L=0 and L=1, norm: min-max, Embed={dim}")
    ax.set_xlabel(r"$\alpha_{\mathrm{norm}}$")
    ax.set_ylabel("density")
    ax.set_xlim(0, 1)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()

    result = {
        "model_name": model_name,
        "model_slug": slug,
        "embed_dim": dim,
        "threshold": round(thr, 4),
        "L0": {"mean": round(m0, 4), "std": round(s0, 4), "n": int((L == 0).sum())},
        "L1": {"mean": round(m1, 4), "std": round(s1, 4), "n": int((L == 1).sum())},
        "P(L=0|a<t)": round(p0_lt, 4), "P(L=0|a>t)": round(p0_gt, 4),
        "P(L=1|a<t)": round(p1_lt, 4), "P(L=1|a>t)": round(p1_gt, 4),
        "n_questions_binary_vector": n_bv,
        "n_questions_total": n_total,
        "n_questions_used": len(labelled),
    }

    if save:
        _out = Path(out_dir) if out_dir else (_HERE / "results")
        _out.mkdir(parents=True, exist_ok=True)
        png = _out / f"hit_threshold_{slug}.png"
        pdf = _out / f"hit_threshold_{slug}.pdf"
        fig.savefig(png, dpi=150)
        fig.savefig(pdf)
        (_out / f"hit_threshold_{slug}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        result["figure_png"] = str(png)
        result["figure_pdf"] = str(pdf)
        print(f"[hit_thr] threshold={thr:.4f}  (embed_dim={dim})  →  {png}")

    return result, fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load_config_defaults() -> dict:
    cfg_path = _HERE / "config.yaml"
    if not cfg_path.is_file():
        return {}
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    emb = cfg.get("embedder", {}) or {}
    data = cfg.get("data", {}) or {}
    return {
        "data_dir": data.get("dir"),
        "model_type": emb.get("model_type", "huggingface"),
        "model_name": emb.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        "base_url": emb.get("base_url"),
        "api_key": emb.get("api_key"),
        "embed_dim": emb.get("embed_dim"),
    }


def main() -> None:
    d = _load_config_defaults()
    ap = argparse.ArgumentParser(description="Compute the HIT threshold for a model.")
    ap.add_argument("--data-dir", default=d.get("data_dir"))
    ap.add_argument("--model-type", default=d.get("model_type", "huggingface"))
    ap.add_argument("--model-name", default=d.get("model_name",
                                                  "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--base-url", default=d.get("base_url"))
    ap.add_argument("--api-key", default=d.get("api_key"))
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--embed-dim", type=int, default=d.get("embed_dim"),
                    help="Truncate embeddings to this many dims (null = native).")
    args = ap.parse_args()

    if not args.data_dir:
        raise SystemExit("--data-dir is required (not found in config.yaml).")

    result, _ = compute_hit_threshold(
        data_dir=args.data_dir, model_type=args.model_type,
        model_name=args.model_name, base_url=args.base_url,
        api_key=args.api_key, out_dir=args.out_dir,
        truncate_dim=args.embed_dim)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
