"""Learned token embedding front end for TableEmbedJePA."""

from __future__ import annotations

import torch
import torch.nn as nn
from transformers import AutoModel


class ScratchEmbedder(nn.Module):
    """Embed padded cell token ids and mean-pool their non-padding tokens."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        pad_id: int,
        pretrained_model_name: str | None = None,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_id)
        if pretrained_model_name:
            pretrained_embedding = AutoModel.from_pretrained(
                pretrained_model_name).get_input_embeddings().weight
            if tuple(pretrained_embedding.shape) != (vocab_size, embed_dim):
                raise ValueError(
                    "Pretrained embedding shape must match tokenizer vocabulary and embed_dim: "
                    f"expected {(vocab_size, embed_dim)}, got {tuple(pretrained_embedding.shape)}."
                )
            with torch.no_grad():
                self.embedding.weight.copy_(pretrained_embedding)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        token_embeddings = self.embedding(input_ids)
        mask = attention_mask.unsqueeze(-1).to(dtype=token_embeddings.dtype)
        token_counts = mask.sum(dim=-2).clamp_min(1.0)
        return (token_embeddings * mask).sum(dim=-2) / token_counts