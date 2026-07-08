"""
query_shift_plot.py
Quantify and visualise the *content-based query distribution shift* between the
three query variants used at evaluation time:

    RAW  — the original question
    SW   — the stop-word-filtered question (``stopwords_util.remove_stopwords``)
    HIT  — the High-informative-Token filtered question
           (``hit_util.high_informative_query``)

For every question (≈816 on Adhesive) the three variants are embedded with the
same model.  The figure has two panels:

* **Left — shift magnitude.**  For each query the cosine similarity between its
  RAW embedding and its SW / HIT embedding is computed, a Gaussian is fitted to
  each set, and the two distributions are drawn as histograms + fitted curves
  with their means marked.  (RAW-vs-RAW is trivially 1.0 and therefore omitted.)
* **Right — shared 2-D PCA.**  RAW, SW and HIT embeddings are stacked and a
  single PCA(2) is fitted so the three spaces share axes.  One tracked query
  (``track_index``, default #0) is circled in each space; the three rings use
  slightly different radii so overlapping ones remain visible.

The figure is saved as ``results/query_shift_<model-slug>.png`` / ``.pdf`` and a
JSON summary alongside it.

Usage
    python query_shift_plot.py                       # uses config.yaml defaults
    python query_shift_plot.py --model-name sentence-transformers/all-mpnet-base-v2
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Optional, Sequence

import numpy as np
from scipy.stats import norm

# Reuse the embedding backend from the HIT-threshold tool (same model / cap).
from compute_hit_threshold import build_embed_fn
from hit_util import high_informative_query
from stopwords_util import remove_stopwords

QA_SUBDIR = "QUESTIONS_ANSWERS_PER_TABLE"
_HERE = Path(__file__).resolve().parent
_EPS = 1e-12


# ── Data loading ──────────────────────────────────────────────────────────────

def load_questions(data_dir: Path, arrays: Sequence[str] = ("questions",)) -> list[str]:
    """Return every non-empty question string from the requested QA arrays."""
    qa_dir = Path(data_dir) / QA_SUBDIR
    if not qa_dir.is_dir():
        raise FileNotFoundError(f"QA directory not found: {qa_dir}")

    out: list[str] = []
    for f in sorted(qa_dir.glob("*.json")):
        data = json.loads(f.read_text(encoding="utf-8"))
        for arr in arrays:
            for q in data.get(arr, []) or []:
                text = str(q.get("question", "")).strip()
                if text:
                    out.append(text)
    return out


# ── Embedding helpers ─────────────────────────────────────────────────────────

def _embed_map(texts: Sequence[str], embed_fn: Callable[[list[str]], np.ndarray]) -> dict:
    """Embed every *unique* text once; return ``{text: vector}``."""
    uniq: list[str] = []
    seen: set[str] = set()
    for t in texts:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    if not uniq:
        return {}
    emb = embed_fn(uniq)
    return {t: np.asarray(emb[i], dtype=np.float32) for i, t in enumerate(uniq)}


def _cosine_rows(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine similarity between two ``[N, d]`` matrices."""
    an = a / (np.linalg.norm(a, axis=1, keepdims=True) + _EPS)
    bn = b / (np.linalg.norm(b, axis=1, keepdims=True) + _EPS)
    return (an * bn).sum(axis=1)


def _pca2(stack: np.ndarray) -> np.ndarray:
    """Project ``[M, d]`` onto its first two principal components (numpy SVD)."""
    centred = stack - stack.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    return centred @ vt[:2].T


# ── Config defaults (HIT threshold per model) ─────────────────────────────────

def _load_config_defaults() -> dict:
    cfg_path = _HERE / "config.yaml"
    if not cfg_path.is_file():
        return {}
    import yaml
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    emb = cfg.get("embedder", {}) or {}
    data = cfg.get("data", {}) or {}
    hit = cfg.get("hit", {}) or {}
    return {
        "data_dir": data.get("dir"),
        "model_type": emb.get("model_type", "huggingface"),
        "model_name": emb.get("model_name", "sentence-transformers/all-MiniLM-L6-v2"),
        "base_url": emb.get("base_url"),
        "api_key": emb.get("api_key"),
        "embed_dim": emb.get("embed_dim"),
        "hit_thresholds": hit.get("thresholds", {}) or {},
        "hit_default_threshold": hit.get("default_threshold", 0.5),
    }


def _resolve_threshold(model_name: str, explicit: Optional[float]) -> float:
    if explicit is not None:
        return float(explicit)
    d = _load_config_defaults()
    return float(d.get("hit_thresholds", {}).get(
        model_name, d.get("hit_default_threshold", 0.5)))


# ── Public entry point ────────────────────────────────────────────────────────

def compute_query_shift(
    data_dir: str | Path,
    model_type: str = "huggingface",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    hit_threshold: Optional[float] = None,
    arrays: Sequence[str] = ("questions",),
    track_index: int = 0,
    out_dir: str | Path | None = None,
    save: bool = True,
    truncate_dim: Optional[int] = None,
):
    """Compute the RAW→SW / RAW→HIT query shift and save the two-panel figure.

    Returns ``(result_dict, matplotlib_figure)``.
    """
    import matplotlib.pyplot as plt

    thr = _resolve_threshold(model_name, hit_threshold)
    questions = load_questions(data_dir, arrays)
    if not questions:
        raise ValueError("No questions found.")
    tokens_per_q = [q.split() for q in questions]
    print(f"[qshift] questions: {len(questions)}  |  HIT threshold: {thr:.4f}  |  "
          f"model: {model_name}")

    embed_fn = build_embed_fn(model_type, model_name, base_url, api_key,
                              truncate_dim=truncate_dim)

    # Phase 1 — embed every question + token once so HIT selection can run.
    base_texts: list[str] = list(questions)
    for toks in tokens_per_q:
        base_texts.extend(toks)
    emap = _embed_map(base_texts, embed_fn)

    # Build the SW and HIT variants of every question.
    import torch
    sw_texts: list[str] = []
    hit_texts: list[str] = []
    for q, toks in zip(questions, tokens_per_q):
        sw_texts.append(remove_stopwords(q))
        if toks:
            tv = torch.from_numpy(np.stack([emap[t] for t in toks]))
            qv = torch.from_numpy(np.asarray(emap[q]))
            hit_texts.append(high_informative_query(toks, tv, qv, thr))
        else:
            hit_texts.append(q)

    # Phase 2 — embed any SW / HIT strings not already covered.
    missing = [t for t in set(sw_texts) | set(hit_texts) if t not in emap]
    emap.update(_embed_map(missing, embed_fn))

    raw_vecs = np.stack([emap[q] for q in questions])
    sw_vecs = np.stack([emap[t] for t in sw_texts])
    hit_vecs = np.stack([emap[t] for t in hit_texts])
    dim = int(raw_vecs.shape[1])

    # ── Left panel data: cosine similarity RAW→SW and RAW→HIT ──────────────
    cos_sw = _cosine_rows(raw_vecs, sw_vecs)
    cos_hit = _cosine_rows(raw_vecs, hit_vecs)
    mu_sw, sd_sw = float(cos_sw.mean()), float(cos_sw.std())
    mu_hit, sd_hit = float(cos_hit.mean()), float(cos_hit.std())

    slug = model_name.replace("/", "-").replace("\\", "-").replace(":", "-")
    disp = model_name.split("/")[-1]

    C_RAW, C_SW, C_HIT = "tab:blue", "tab:orange", "tab:green"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13.5, 5.0))
    # fig.suptitle(f"{disp} — content-based query distribution shift",
                #  fontsize=14, fontweight="bold")

    # ── Left: histograms + fitted Gaussians ────────────────────────────────
    lo = float(min(cos_sw.min(), cos_hit.min()))
    lo = max(0.0, lo - 0.02)
    bins = np.linspace(lo, 1.0, 31)
    bw = bins[1] - bins[0]
    shift_sw = 1.0 - mu_sw          # shift magnitude = 1 - mean cosine similarity
    shift_hit = 1.0 - mu_hit
    axL.hist(cos_sw, bins=bins, color=C_SW, alpha=0.55, edgecolor="none",
             label=f"Stopword (SW) filtered  (\u03bc={mu_sw:.3f}, \u03c3={sd_sw:.3f}, shift={shift_sw:.3f})")
    axL.hist(cos_hit, bins=bins, color=C_HIT, alpha=0.55, edgecolor="none",
             label=f"HIT filtered  (\u03bc={mu_hit:.3f}, \u03c3={sd_hit:.3f}, shift={shift_hit:.3f})")
    xs = np.linspace(lo, 1.0, 400)
    axL.plot(xs, len(cos_sw) * bw * norm.pdf(xs, mu_sw, sd_sw + _EPS), color=C_SW, lw=2)
    axL.plot(xs, len(cos_hit) * bw * norm.pdf(xs, mu_hit, sd_hit + _EPS), color=C_HIT, lw=2)
    axL.axvline(mu_sw, color=C_SW, ls="--", lw=1.5)
    axL.axvline(mu_hit, color=C_HIT, ls="--", lw=1.5)
    axL.set_title("Shift magnitude per filtered space", fontweight="bold")
    axL.set_xlabel("cosine similarity to original query embedding")
    axL.set_ylabel("number of queries")
    axL.legend(loc="upper left", fontsize=12)
    axL.grid(True, ls=":", alpha=0.4)

    # ── Right: shared 2-D PCA of the three spaces ──────────────────────────
    N = len(questions)
    proj = _pca2(np.concatenate([raw_vecs, sw_vecs, hit_vecs], axis=0))
    p_raw, p_sw, p_hit = proj[:N], proj[N:2 * N], proj[2 * N:]

    axR.scatter(p_raw[:, 0], p_raw[:, 1], s=9, color=C_RAW, alpha=0.7, label="Original query")
    axR.scatter(p_sw[:, 0], p_sw[:, 1], s=9, color=C_SW, alpha=0.7, label="Stopword (SW) filtered")
    axR.scatter(p_hit[:, 0], p_hit[:, 1], s=9, color=C_HIT, alpha=0.7, label="HIT filtered")

    ti = int(track_index) if 0 <= int(track_index) < N else 0
    # Slightly different ring radii so overlapping tracked points stay visible.
    axR.plot([p_raw[ti, 0], p_sw[ti, 0], p_hit[ti, 0]],
             [p_raw[ti, 1], p_sw[ti, 1], p_hit[ti, 1]],
             ls=":", color="0.4", lw=1.0, zorder=1)
    for pts, col, size in ((p_raw, C_RAW, 600), (p_sw, C_SW, 950), (p_hit, C_HIT, 1350)):
        axR.scatter(pts[ti, 0], pts[ti, 1], s=size, facecolors="none",
                    edgecolors=col, linewidths=2.2, zorder=5)

    from matplotlib.lines import Line2D
    handles, labels = axR.get_legend_handles_labels()
    handles.append(Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
                          markeredgecolor="0.2", markersize=13, markeredgewidth=2,
                          label=f"tracked query #{ti}"))
    axR.legend(handles=handles, loc="upper left", fontsize=12)
    axR.set_title("Shared 2-D PCA of the three spaces", fontweight="bold")
    axR.set_xlabel("PC 1")
    axR.set_ylabel("PC 2")
    axR.grid(True, ls=":", alpha=0.4)

    fig.tight_layout(rect=(0, 0, 1, 0.96))

    result = {
        "model_name": model_name,
        "model_slug": slug,
        "embed_dim": dim,
        "hit_threshold": round(thr, 4),
        "n_questions": N,
        "tracked_index": ti,
        "raw_to_sw": {"mean": round(mu_sw, 4), "std": round(sd_sw, 4),
                      "shift": round(1.0 - mu_sw, 4)},
        "raw_to_hit": {"mean": round(mu_hit, 4), "std": round(sd_hit, 4),
                       "shift": round(1.0 - mu_hit, 4)},
    }

    if save:
        _out = Path(out_dir) if out_dir else (_HERE / "results")
        _out.mkdir(parents=True, exist_ok=True)
        png = _out / f"query_shift_{slug}.png"
        pdf = _out / f"query_shift_{slug}.pdf"
        fig.savefig(png, dpi=200)
        fig.savefig(pdf)
        (_out / f"query_shift_{slug}.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8")
        result["figure_png"] = str(png)
        result["figure_pdf"] = str(pdf)
        print(f"[qshift] RAW->SW \u03bc={mu_sw:.3f}  RAW->HIT \u03bc={mu_hit:.3f}  \u2192  {png}")

    return result, fig


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    d = _load_config_defaults()
    ap = argparse.ArgumentParser(description="Quantify RAW/SW/HIT query embedding shift.")
    ap.add_argument("--data-dir", default=d.get("data_dir"))
    ap.add_argument("--model-type", default=d.get("model_type", "huggingface"))
    ap.add_argument("--model-name", default=d.get("model_name",
                                                  "sentence-transformers/all-MiniLM-L6-v2"))
    ap.add_argument("--base-url", default=d.get("base_url"))
    ap.add_argument("--api-key", default=d.get("api_key"))
    ap.add_argument("--hit-threshold", type=float, default=None,
                    help="Override the per-model HIT threshold from config.yaml.")
    ap.add_argument("--arrays", nargs="+", default=["questions"])
    ap.add_argument("--track-index", type=int, default=0)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--embed-dim", type=int, default=d.get("embed_dim"),
                    help="Truncate embeddings to this many dims (null = native).")
    args = ap.parse_args()

    if not args.data_dir:
        raise SystemExit("--data-dir is required (not found in config.yaml).")

    result, _ = compute_query_shift(
        data_dir=args.data_dir, model_type=args.model_type,
        model_name=args.model_name, base_url=args.base_url, api_key=args.api_key,
        hit_threshold=args.hit_threshold, arrays=args.arrays,
        track_index=args.track_index, out_dir=args.out_dir,
        truncate_dim=args.embed_dim)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
