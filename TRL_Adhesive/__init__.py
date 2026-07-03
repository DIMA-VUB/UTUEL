"""TRL_Adhesive — TableEmbedJePA on complex / nested Adhesive tables."""

try:
    from .config  import TableEmbedJePAConfig
    from .model   import TableEmbedJePA
    from .smp     import (UPath, load_table_cells, parse_smp_line,
                          upaths_from_smp_line, generate_upaths_for_table)
    from .dataset import (TableEmbedJePADataset, TableEmbedJePADataModule,
                          jepa_collate_fn, get_embedder)
except ImportError:
    from config  import TableEmbedJePAConfig
    from model   import TableEmbedJePA
    from smp     import (UPath, load_table_cells, parse_smp_line,
                         upaths_from_smp_line, generate_upaths_for_table)
    from dataset import (TableEmbedJePADataset, TableEmbedJePADataModule,
                         jepa_collate_fn, get_embedder)
