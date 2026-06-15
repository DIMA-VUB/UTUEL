"""
config.py
CTAConfig — extends RobertaConfig with CTA-specific hyperparameters.

Column Type Annotation (CTA) task:
  - Input  : pre-computed LLM embeddings of cell values in a column (node_a / node_b)
  - Output : one type label per column (multi-class classification over type_vocab)
  - Model  : column-level aggregation of cell embeddings → transformer → linear head
"""

from transformers import RobertaConfig


class CTAConfig(RobertaConfig):
    """
    Configuration for the CTA column type annotation model.

    Inherits all RobertaConfig fields and adds CTA-specific parameters:

        embedding_dim   — dimensionality of raw LLM cell embeddings (embed_dim_in).
                          A learned linear projection maps this to hidden_size.
        num_classes     — number of column-type classes (len(type_vocab)).
        pool_mode       — how to aggregate cell embeddings into a column embedding:
                          'mean' | 'max' | 'cls' | 'attention'
        pretrain_mode   — objective used during pretraining:
                          'contrastive' | 'masked_cell' | 'none'
        temperature     — InfoNCE softmax temperature (contrastive pretraining).
        label_smoothing — label-smoothing epsilon for CrossEntropy in fine-tuning.
    """

    model_type = "cta_classifier"

    def __init__(
        self,
        vocab_size: int = 50265,
        hidden_size: int = 384,
        num_hidden_layers: int = 2,
        num_attention_heads: int = 8,
        intermediate_size: int = 1536,
        hidden_act: str = "gelu",
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        layer_norm_eps: float = 1e-12,
        pad_token_id: int = 1,
        # ── CTA-specific ────────────────────────────────────────────────────
        embedding_dim: int = 384,
        num_classes: int = 255,       # set automatically from type_vocab at runtime
        pool_mode: str = "mean",      # 'mean' | 'max' | 'cls' | 'attention'
        pretrain_mode: str = "contrastive",  # 'contrastive' | 'masked_cell' | 'none'
        temperature: float = 0.07,
        label_smoothing: float = 0.0,
        classifier_intermediate_size: int | None = None,  # optional hidden layer in classifier head
        **kwargs,
    ):
        super().__init__(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_act=hidden_act,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            layer_norm_eps=layer_norm_eps,
            pad_token_id=pad_token_id,
            **kwargs,
        )
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.pool_mode = pool_mode
        self.pretrain_mode = pretrain_mode
        self.temperature = temperature
        self.label_smoothing = label_smoothing
        self.classifier_intermediate_size = classifier_intermediate_size
