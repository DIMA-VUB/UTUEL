"""
hit_util.py
HIT (High-informative Token) query filtering.

Given a question, each whitespace-split token is embedded and scored by its
cosine similarity to the *full-question* embedding.  The per-question scores are
min-max normalised to ``[0, 1]`` and only tokens whose normalised score is
greater than a per-model threshold are retained.  The retained tokens, joined by
spaces, form the HIT query variant.

The embeddings are supplied by the caller (the dataset already batch-embeds every
text), so this module only holds the pure scoring / selection logic.
"""

from __future__ import annotations

import torch

_EPS = 1e-12


def hit_scores(token_embs: torch.Tensor, q_emb: torch.Tensor) -> torch.Tensor:
    """Min-max normalised cosine similarity of each token to the full question.

    ``token_embs`` is ``[T, d]`` and ``q_emb`` is ``[d]``.  Returns a ``[T]``
    tensor in ``[0, 1]`` (all-ones when every token scores identically).
    """
    te = token_embs / token_embs.norm(dim=-1, keepdim=True).clamp_min(_EPS)
    qe = q_emb / q_emb.norm().clamp_min(_EPS)
    sims = te @ qe                                    # [T] cosine similarity
    lo, hi = sims.min(), sims.max()
    if (hi - lo) < _EPS:
        return torch.ones_like(sims)
    return (sims - lo) / (hi - lo)


def high_informative_query(
    tokens: list[str],
    token_embs: torch.Tensor,
    q_emb: torch.Tensor,
    threshold: float,
) -> str:
    """Return the HIT query: tokens whose min-max cosine score exceeds *threshold*.

    ``tokens`` are the whitespace-split question tokens aligned with the rows of
    ``token_embs``.  If the filter would keep nothing, the original tokens are
    returned so the HIT variant is never empty.
    """
    if not tokens:
        return ""
    scores = hit_scores(token_embs, q_emb)
    kept = [tok for tok, s in zip(tokens, scores.tolist()) if s > threshold]
    return " ".join(kept) if kept else " ".join(tokens)
