"""
visualize_embeddings.py
Embedding-space UMAP plots for the CTA pipeline.

Usage (from repo root):
    %run CTA/visualize_embeddings.py \\
        --pretrain_ckpt  checkpoints/dry_run_pretrain/last.ckpt \\
        --ft_raw_ckpt    CTA/checkpoints/dry_run_finetune_raw/best.ckpt \\
        --ft_enc_ckpt    CTA/checkpoints/dry_run_finetune_enc/best.ckpt \\
        --max_records    20 \\
        --max_rows       6

Two figures are saved / shown:
    1. Node-level UMAP  — shape=node_type (cell ● / pivot ▲),  colour=label
    2. Column-level UMAP — all points are columns; colour=label
"""
from __future__ import annotations

import argparse
import importlib.util as _ilu
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf

# ── Path setup ────────────────────────────────────────────────────────────────
# _CTA and _REPO are used for file paths (config, outputs).
# Only _REPO and _TRL go on sys.path: CTA is a package under _REPO, so
# `from CTA.xxx import` is always unambiguous regardless of kernel state.
_CTA    = Path(__file__).resolve().parent   # …/UTUEL/CTA
_REPO   = _CTA.parent                       # …/UTUEL
_TRL    = _REPO / "TRL-model"              # …/UTUEL/TRL-model
for p in [str(_REPO), str(_TRL)]:
    if p not in sys.path:
        sys.path.insert(0, p)

from CTA.dataset_utils import load_type_vocab, resolve_data_paths
from CTA.dataset       import CTASMPDataset


# ── CLI args (also usable as %run script with args) ──────────────────────────
parser = argparse.ArgumentParser(add_help=False)
parser.add_argument("--pretrain_ckpt",  default=None)
parser.add_argument("--ft_raw_ckpt",   default=None)
parser.add_argument("--ft_enc_ckpt",   default=None)
parser.add_argument("--data_folder",   default=None,
                    help="Path to the CTA data folder (overrides config data.folder)")
parser.add_argument("--max_records",   type=int, default=20)
parser.add_argument("--max_rows",      type=int, default=6)
parser.add_argument("--top_k_labels",  type=int, default=12,
                    help="Number of most-frequent label types to colour (rest → 'other')")
parser.add_argument("--split",         default="test",
                    choices=["train", "dev", "test"])
parser.add_argument("--include_header_emb", action="store_true", default=False,
                    help="Include the column header node embedding in the column-level mean-pool")
parser.add_argument("--output_dir",    default="CTA/outputs/embedding_viz")
args, _unknown = parser.parse_known_args()

OUT_DIR = Path(args.output_dir)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Config & paths ────────────────────────────────────────────────────────────
cfg = OmegaConf.load(_CTA / "config.yaml")
# Apply DRY overrides if the variable exists in the calling namespace
try:
    _dry_str: str = DRY  # type: ignore[name-defined]  # noqa: F821
    # Only apply overrides that don't contain backslashes (Windows paths break dotlist)
    _safe_overrides = [kv for kv in _dry_str.split()
                       if kv.startswith("data.") and "\\" not in kv]
    if _safe_overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(_safe_overrides))
except NameError:
    pass
# --data_folder always wins (avoids backslash parsing issues)
if args.data_folder:
    OmegaConf.update(cfg, "data.folder", args.data_folder, merge=True)

paths     = resolve_data_paths(cfg.data)
type2idx, idx2type = load_type_vocab(paths["type_vocab"])

# ── Load CTASMPDataset for the chosen split ───────────────────────────────────
data_path = paths[args.split]
print(f"[viz] building CTASMPDataset  split={args.split}  max_records={args.max_records}  max_rows={args.max_rows}")
smp_ds = CTASMPDataset(
    data_path=data_path,
    model_type=cfg.embedder.model_type,
    model_name=cfg.embedder.get("model_name"),
    max_records=args.max_records,
    max_rows_per_table=args.max_rows,
    precompute=True,
    cache_embeddings=cfg.embedder.cache_embeddings,
    embed_cache_dir=cfg.embedder.get("embed_cache_dir"),
)
print(f"[viz] {len(smp_ds.records)} tables  {len(smp_ds._samples)} U-paths  "
      f"embed_cache shape={smp_ds._embed_cache.shape}")

# ── Col-type lookup ───────────────────────────────────────────────────────────
raw_records = json.loads(Path(data_path).read_text(encoding="utf-8"))[:args.max_records]
col_types_per_table: dict[str, list[list[str]]] = {
    str(r[0]): (r[7] if len(r) > 7 else []) for r in raw_records
}

# ── Build a-side / b-side U-path indices ─────────────────────────────────────
rec_col_a = defaultdict(list)
rec_col_b = defaultdict(list)
for j, (ri, up) in enumerate(smp_ds._samples):
    rec_col_a[(ri, up.col_idx_a)].append(j)
    if up.col_idx_b != up.col_idx_a:
        rec_col_b[(ri, up.col_idx_b)].append(j)

# ── Identify top-K labels for colouring ───────────────────────────────────────
from collections import Counter
label_counter: Counter = Counter()
for ri, rec in enumerate(smp_ds.records):
    col_types = col_types_per_table.get(rec["table_id"], [])
    for col_idx, types in enumerate(col_types):
        for t in types:
            if t in type2idx:
                label_counter[t] += 1

top_labels: list[str] = [t for t, _ in label_counter.most_common(args.top_k_labels)]
label_to_color_idx: dict[str, int] = {t: i for i, t in enumerate(top_labels)}
cmap = plt.get_cmap("tab20", len(top_labels) + 1)

def _label_color(types_for_col: list[str]) -> int:
    """Return color index for the first top-K label found, else last bucket (other)."""
    for t in types_for_col:
        if t in label_to_color_idx:
            return label_to_color_idx[t]
    return len(top_labels)  # 'other'

# ── Encoder loader ────────────────────────────────────────────────────────────
def _load_encoder(ckpt_path: str | None):
    if not ckpt_path or not Path(ckpt_path).exists():
        return None
    # Dynamic import of TRL-model package
    pkg_name = "trl_model_pkg"
    if pkg_name not in sys.modules:
        pkg_init = _TRL / "model" / "__init__.py"
        spec = _ilu.spec_from_file_location(pkg_name, pkg_init,
                   submodule_search_locations=[str(_TRL / "model")])
        pkg  = _ilu.module_from_spec(spec)
        pkg.__path__ = [str(_TRL / "model")]
        pkg.__package__ = pkg_name
        sys.modules[pkg_name] = pkg
        spec.loader.exec_module(pkg)
    from model import TableEmbedJePA
    from config import TableEmbedJePAConfig
    _hs = cfg.model.hidden_size
    _nh = cfg.model.get("num_heads") or cfg.model.get("num_attention_heads") or 12
    _nh = max(1, min(_nh, _hs // 64))
    while _hs % _nh != 0 and _nh > 1:
        _nh -= 1
    _nl = cfg.model.get("num_layers") or cfg.model.get("num_hidden_layers") or 1
    enc_cfg = TableEmbedJePAConfig(
        num_hidden_layers=_nl,
        hidden_size=_hs,
        num_attention_heads=_nh,
        intermediate_size=cfg.model.intermediate_size or (_hs * 4),
        attention_probs_dropout_prob=cfg.model.attention_dropout,
        hidden_dropout_prob=cfg.model.hidden_dropout,
        layer_norm_eps=cfg.model.layer_norm_eps,
        embedding_dim=int(cfg.embedder.embed_dim),
        tempeture=cfg.model.temperature,
        beta=cfg.model.beta,
    )
    enc = TableEmbedJePA.load_from_checkpoint(ckpt_path, map_location="cpu", config=enc_cfg)
    return enc.eval()

# ── Core extraction — returns (node_embs, node_meta, col_embs, col_meta) ──────
# node_meta / col_meta are dicts: {label_color, node_type ('cell'|'pivot'), col_label_str}

@torch.no_grad()
def extract(encoder) -> tuple[
        np.ndarray, list[dict],   # node level
        np.ndarray, list[dict],   # column level
]:
    node_vecs:  list[np.ndarray] = []
    node_meta:  list[dict]       = []
    col_vecs:   list[np.ndarray] = []
    col_meta:   list[dict]       = []

    def _enc_or_raw(idx_t: torch.Tensor, node_pos: int) -> torch.Tensor:
        """Return [n, d] node reps.

        Matches CTAClassifier.forward(): each raw LLM embedding is treated
        as a single token.  The encoder prepends CLS and runs a 2-token
        sequence; we read back the CLS output.
        If no encoder, return the normalized raw LLM embedding directly.
        """
        raw_embs = smp_ds._embed_cache[smp_ds._smp_idx[idx_t, node_pos]]  # [n, d_llm]
        if encoder is not None:
            x    = raw_embs.unsqueeze(1)                                     # [n, 1, d_llm]
            proj = encoder.input_projection(x)                               # [n, 1, d_model]
            cls_ = encoder.input_projection(
                       encoder.cls_token).expand(len(idx_t), -1, -1)        # [n, 1, d_model]
            seq  = torch.cat([cls_, proj], dim=1)                            # [n, 2, d_model]
            enc_out, _, _ = encoder.transformer_encoder(seq)
            return F.normalize(enc_out[:, 0, :], dim=-1)                     # CLS → [n, d_model]
        return F.normalize(raw_embs, dim=-1)

    for ri, rec in enumerate(smp_ds.records):
        table_id  = rec["table_id"]
        header    = rec["header"]
        col_types = col_types_per_table.get(table_id, [])

        for col_idx, col_hdr in enumerate(header):
            types_for_col = col_types[col_idx] if col_idx < len(col_types) else []
            col_label_str = types_for_col[0] if types_for_col else "<no_label>"
            c_idx = _label_color(types_for_col)

            js_a = rec_col_a.get((ri, col_idx), [])
            js_b = rec_col_b.get((ri, col_idx), [])
            if not js_a and not js_b:
                continue

            node_reps: list[torch.Tensor] = []

            # ── a-side (node_a = this col's cell) ───────────────────────────
            if js_a:
                idx_a = torch.tensor(js_a, dtype=torch.long)
                reps  = _enc_or_raw(idx_a, 1)   # node_a
                for k, j in enumerate(js_a):
                    _, up = smp_ds._samples[j]
                    node_vecs.append(reps[k].numpy())
                    node_meta.append({"color": c_idx, "node_type": "cell",
                                      "label": col_label_str, "col_header": col_hdr,
                                      "cell": up.cell_value_a})
                    # pivot_a (header node — raw LLM, always displayed as pivot shape)
                    piv_raw = smp_ds._embed_cache[smp_ds._smp_idx[idx_a[k:k+1], 0]]
                    piv_rep = F.normalize(piv_raw, dim=-1)
                    node_vecs.append(piv_rep[0].numpy())
                    node_meta.append({"color": c_idx, "node_type": "pivot",
                                      "label": col_label_str, "col_header": col_hdr,
                                      "cell": up.col_header_a})
                node_reps.append(reps)

            # ── b-side (node_b = this col's cell) ───────────────────────────
            if js_b:
                idx_b = torch.tensor(js_b, dtype=torch.long)
                reps  = _enc_or_raw(idx_b, 2)   # node_b
                for k, j in enumerate(js_b):
                    _, up = smp_ds._samples[j]
                    node_vecs.append(reps[k].numpy())
                    node_meta.append({"color": c_idx, "node_type": "cell",
                                      "label": col_label_str, "col_header": col_hdr,
                                      "cell": up.cell_value_b})
                    piv_raw = smp_ds._embed_cache[smp_ds._smp_idx[idx_b[k:k+1], 3]]
                    piv_rep = F.normalize(piv_raw, dim=-1)
                    node_vecs.append(piv_rep[0].numpy())
                    node_meta.append({"color": c_idx, "node_type": "pivot",
                                      "label": col_label_str, "col_header": col_hdr,
                                      "cell": up.col_header_b})
                node_reps.append(reps)

            # ── column-level mean-pool (optionally include header emb) ───────
            all_reps = torch.cat(node_reps, dim=0)   # [n_upaths, d]
            if args.include_header_emb:
                # Header embedding: raw LLM pivot_a (a-side) or pivot_b (b-side)
                if js_a:
                    hdr_raw = smp_ds._embed_cache[smp_ds._smp_idx[idx_a[0:1], 0]]
                else:
                    hdr_raw = smp_ds._embed_cache[smp_ds._smp_idx[idx_b[0:1], 3]]
                hdr_rep = F.normalize(hdr_raw, dim=-1)  # [1, d]
                col_emb = torch.stack([all_reps.mean(dim=0), hdr_rep[0]]).mean(dim=0)
            else:
                col_emb = all_reps.mean(dim=0)
            col_vecs.append(col_emb.numpy())
            col_meta.append({"color": c_idx, "label": col_label_str,
                             "col_header": col_hdr, "n_reps": len(all_reps)})

    return (np.array(node_vecs), node_meta,
            np.array(col_vecs),  col_meta)

# ── Run all four configs ───────────────────────────────────────────────────────
# NOTE: CTALinearClassifier (the finetune model) does NOT store the encoder —
# it trains only a linear head on top of pre-extracted embeddings.  The encoder
# that was used during finetuning IS the pretrain checkpoint; loading ft_enc_ckpt
# as a TableEmbedJePA would fail.  So both "After pretrain" and "Finetune (enc)"
# reuse the same pretrain encoder object.
_enc_pretrain = _load_encoder(args.pretrain_ckpt)
CONFIGS: list[tuple[str, object]] = [
    ("Raw (no encoder)",  None),
    ("After pretrain",    _enc_pretrain),
    ("Finetune (no enc)", None),          # raw embs + separate ft head; encoder unchanged
    ("Finetune (enc)",    _enc_pretrain), # pretrain encoder + ft head; same emb space
]

all_node: list[tuple[str, np.ndarray, list[dict]]] = []
all_col:  list[tuple[str, np.ndarray, list[dict]]] = []
for label, enc in CONFIGS:
    print(f"[viz] extracting  {label} …")
    nv, nm, cv, cm = extract(enc)
    all_node.append((label, nv, nm))
    all_col.append((label,  cv, cm))
    print(f"       node pts={len(nv)}  col pts={len(cv)}")

# ── UMAP dimensionality reduction ─────────────────────────────────────────────
try:
    from umap import UMAP
    _reducer = lambda n: UMAP(n_components=2, random_state=42, n_neighbors=min(15, n-1))
except ImportError:
    from sklearn.manifold import TSNE
    print("[viz] umap-learn not found — falling back to t-SNE")
    _reducer = lambda n: TSNE(n_components=2, random_state=42, perplexity=min(30, max(5, n//5)))

def _reduce(vecs: np.ndarray) -> np.ndarray:
    if len(vecs) < 4:
        return np.zeros((len(vecs), 2))
    return _reducer(len(vecs)).fit_transform(vecs)

# ── Legend helpers ────────────────────────────────────────────────────────────
def _make_legend(ax, meta: list[dict], show_node_type: bool):
    # Label legend
    seen_labels: dict[int, str] = {}
    for m in meta:
        if m["color"] not in seen_labels:
            seen_labels[m["color"]] = m["label"]
    patches = [mpatches.Patch(color=cmap(ci), label=lbl[:35])
               for ci, lbl in sorted(seen_labels.items())]
    if len(top_labels) < len(seen_labels) or any(m["color"] == len(top_labels) for m in meta):
        patches.append(mpatches.Patch(color=cmap(len(top_labels)), label="other"))
    l1 = ax.legend(handles=patches, title="Label", loc="upper left",
                   fontsize=7, title_fontsize=8, framealpha=0.7)
    ax.add_artist(l1)
    # Shape legend
    if show_node_type:
        shape_handles = [
            plt.scatter([], [], marker="o", color="grey", s=30, label="cell"),
            plt.scatter([], [], marker="^", color="grey", s=40, label="pivot/header"),
        ]
        ax.legend(handles=shape_handles, title="Node type", loc="lower right",
                  fontsize=7, title_fontsize=8, framealpha=0.7)

# ── Figure 1: Node-level ──────────────────────────────────────────────────────
fig1, axes1 = plt.subplots(1, 4, figsize=(22, 5))
fig1.suptitle(f"Node-level embeddings — {args.split} split", fontsize=13, fontweight="bold")

for ax, (cfg_label, vecs, meta) in zip(axes1, all_node):
    if len(vecs) == 0:
        ax.set_title(cfg_label); ax.axis("off"); continue
    coords = _reduce(vecs)
    for m, (x, y) in zip(meta, coords):
        marker = "o" if m["node_type"] == "cell" else "^"
        ax.scatter(x, y, c=[cmap(m["color"])], marker=marker,
                   s=28 if m["node_type"] == "cell" else 45,
                   edgecolors="none", alpha=0.75)
    ax.set_title(cfg_label, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    _make_legend(ax, meta, show_node_type=True)

fig1.tight_layout()
out1 = OUT_DIR / f"node_embeddings_{args.split}.png"
fig1.savefig(out1, dpi=150, bbox_inches="tight")
print(f"[viz] saved → {out1}")
plt.show()

# ── Figure 2: Column-level ────────────────────────────────────────────────────
fig2, axes2 = plt.subplots(1, 4, figsize=(22, 5))
fig2.suptitle(f"Column-level embeddings (mean-pooled) — {args.split} split",
              fontsize=13, fontweight="bold")

for ax, (cfg_label, vecs, meta) in zip(axes2, all_col):
    if len(vecs) == 0:
        ax.set_title(cfg_label); ax.axis("off"); continue
    coords = _reduce(vecs)
    colors = [cmap(m["color"]) for m in meta]
    ax.scatter(coords[:, 0], coords[:, 1], c=colors,
               marker="o", s=55, edgecolors="white", linewidths=0.4, alpha=0.85)
    ax.set_title(cfg_label, fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])
    _make_legend(ax, meta, show_node_type=False)

fig2.tight_layout()
out2 = OUT_DIR / f"column_embeddings_{args.split}.png"
fig2.savefig(out2, dpi=150, bbox_inches="tight")
print(f"[viz] saved → {out2}")
plt.show()
