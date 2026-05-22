"""TRL-model — TableEmbedJePA table embedding JEPA."""

try:
    from .config        import TableEmbedJePAConfig
    from .model         import TableEmbedJePA
    from .smp           import (UPath,
                                generate_u_paths_flat, generate_u_paths_from_graph,
                                build_table_graph, walks_to_u_paths,
                                SemanticMetaPath)
    from .dataset       import (TableEmbedJePADataset, TableEmbedJePADataModule,
                                jepa_collate_fn, get_embedder)
except ImportError:
    from config        import TableEmbedJePAConfig
    from model         import TableEmbedJePA
    from smp           import (UPath,
                               generate_u_paths_flat, generate_u_paths_from_graph,
                               build_table_graph, walks_to_u_paths,
                               SemanticMetaPath)
    from dataset       import (TableEmbedJePADataset, TableEmbedJePADataModule,
                               jepa_collate_fn, get_embedder)

__all__ = [
    "TableEmbedJePAConfig",
    "TableEmbedJePA",
    "UPath",
    "generate_u_paths_flat",
    "generate_u_paths_from_graph",
    "build_table_graph",
    "walks_to_u_paths",
    "SemanticMetaPath",
    "TableEmbedJePADataset",
    "TableEmbedJePADataModule",
    "jepa_collate_fn",
    "get_embedder",
]
