"""
smp.py — U-path generation from WikiSQL-formatted tables.

Node format in the table graph: "row,col"  (row 0 = header row, rows 1…N = data).

  U-path — 4 nodes, 2 inner-angle cells
  ─────────────────────────────────────
  (0,j) col_header_j        col_header_k (0,k)
        │ down                        ↑ up
  (i,j) cell_a  ──right──►  cell_b (i,k)

  Graph walk: (0,j) down► (i,j) right► (i,k) up► (0,k)
  text:  col_header_j | cell_a | cell_b | col_header_k
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class UPath:
    """
    U-path for a pair of cells in the same data row, different columns.

    Graph walk: (0,j) ─down─► (i,j) ─right─► (i,k) ─up─► (0,k)

      (0,j) col_header_a        col_header_b (0,k)
            |                            |
      (i,j) cell_a  ───right───►  cell_b (i,k)

    text: col_header_a | cell_a | cell_b | col_header_b
    """
    row_idx:      int   # 0-based shared data-row
    col_idx_a:    int   # left column  (col_idx_a < col_idx_b)
    col_idx_b:    int   # right column
    col_header_a: str   # header[col_idx_a]
    cell_value_a: str   # rows[row_idx][col_idx_a]
    cell_value_b: str   # rows[row_idx][col_idx_b]
    col_header_b: str   # header[col_idx_b]

    # derived — graph traceability
    tbl_row:          int = field(default=0,  init=False)  # graph row id (row_idx + 1)
    pivot_a:          str = field(default="", init=False)  # "0,col_idx_a"  top-left
    node_a:           str = field(default="", init=False)  # "tbl_row,col_idx_a"
    node_b:           str = field(default="", init=False)  # "tbl_row,col_idx_b"
    pivot_b:          str = field(default="", init=False)  # "0,col_idx_b"  top-right
    smp_text:         str = field(default="", init=False)  # col_a | cell_a | cell_b | col_b
    reversed_smp_text: str = field(default="", init=False)  # col_b | cell_b | cell_a | col_a
    table_id:         str = field(default="", init=False)  # source table identifier
    record_id:        str = field(default="", init=False)  # record "id" from the JSONL

    def __post_init__(self) -> None:
        self.tbl_row  = self.row_idx + 1
        self.pivot_a  = f"0,{self.col_idx_a}"
        self.node_a   = f"{self.tbl_row},{self.col_idx_a}"
        self.node_b   = f"{self.tbl_row},{self.col_idx_b}"
        self.pivot_b  = f"0,{self.col_idx_b}"
        self.smp_text = (
            f"{self.col_header_a} | {self.cell_value_a}"
            f" | {self.cell_value_b} | {self.col_header_b}"
        )
        self.reversed_smp_text = (
            f"{self.col_header_b} | {self.cell_value_b}"
            f" | {self.cell_value_a} | {self.col_header_a}"
        )


# ── NetworkX graph helpers ────────────────────────────────────────────────────

def build_table_graph(header: list[str], rows: list[list[str]]):
    """
    Build a directed NetworkX graph from a WikiSQL table.

    Node IDs are ``"row,col"`` strings (row 0 = header row).
    Node attributes:
        ``class``  – ``"HEADER"`` for row 0, ``"VALUE"`` for data rows
        ``header`` – True for row 0 nodes, False otherwise
        ``value``  – the cell text
    Edge attributes:
        ``label``  – one of ``"right"``, ``"left"``, ``"down"``, ``"up"``
    """
    import networkx as nx

    G = nx.DiGraph()
    n_cols = len(header)
    n_rows = len(rows)

    # Header row (row 0)
    for j, h in enumerate(header):
        G.add_node(f"0,{j}", value=str(h), **{"class": "HEADER"}, header=True)

    # Data rows (1 … n_rows)
    for i, row in enumerate(rows, start=1):
        for j in range(n_cols):
            val = str(row[j]).strip() if j < len(row) else ""
            G.add_node(f"{i},{j}", value=val, **{"class": "VALUE"}, header=False)

    # Edges
    for i in range(n_rows + 1):
        for j in range(n_cols):
            nid = f"{i},{j}"
            if nid not in G:
                continue
            # Horizontal
            right = f"{i},{j + 1}"
            if j + 1 < n_cols and right in G:
                G.add_edge(nid, right, label="right")
                G.add_edge(right, nid, label="left")
            # Vertical
            down = f"{i + 1},{j}"
            if i + 1 <= n_rows and down in G:
                G.add_edge(nid, down, label="down")
                G.add_edge(down, nid, label="up")

    return G


# ── U-path generation ─────────────────────────────────────────────────────────

def generate_u_paths_flat(
    header: list[str],
    rows: list[list[str]],
) -> list[UPath]:
    """
    Enumerate all U-paths: for every data row, every pair of columns (j < k).

    Each U-path traces:  (0,j) down► (i,j) right► (i,k) up► (0,k)
    text: col_header_j | cell(i,j) | cell(i,k) | col_header_k

    Single-column tables: node_a == node_b and pivot_a == pivot_b
    (the U-path collapses to a self-path on the only column).
    """
    out: list[UPath] = []
    n = len(header)

    if n == 1:
        # Self-path: pivot_a == pivot_b, node_a == node_b
        h = str(header[0]).strip()
        for row_idx, row in enumerate(rows):
            cell = str(row[0]).strip() if row else ""
            out.append(UPath(
                row_idx=row_idx,
                col_idx_a=0,  col_idx_b=0,
                col_header_a=h,
                cell_value_a=cell,
                cell_value_b=cell,
                col_header_b=h,
            ))
        return out

    for row_idx, row in enumerate(rows):
        for j in range(n):
            ca = str(row[j]).strip() if j < len(row) else ""
            for k in range(j + 1, n):
                out.append(UPath(
                    row_idx=row_idx,
                    col_idx_a=j,  col_idx_b=k,
                    col_header_a=str(header[j]).strip(),
                    cell_value_a=ca,
                    cell_value_b=str(row[k]).strip() if k < len(row) else "",
                    col_header_b=str(header[k]).strip(),
                ))
    return out


def walks_to_u_paths(
    walks: list[list[str]],
    header: list[str],
    rows: list[list[str]],
) -> list[UPath]:
    """
    Extract U-paths from graph-walk sequences.

    Scans every consecutive 4-node window for the pattern:
      header(0,j) → data(i,j) → data(i,k) → header(0,k)
    where both data nodes are in the same row and different columns.
    """
    _EDGES = {"down", "up", "right", "left"}
    out: list[UPath] = []
    seen: set[tuple[int, int, int]] = set()   # (row_idx, col_j, col_k)

    for walk in walks:
        nodes = [w for w in walk if w not in _EDGES]
        for i in range(len(nodes) - 3):
            try:
                r0, c0 = map(int, nodes[i    ].split(","))
                r1, c1 = map(int, nodes[i + 1].split(","))
                r2, c2 = map(int, nodes[i + 2].split(","))
                r3, c3 = map(int, nodes[i + 3].split(","))
            except ValueError:
                continue

    # Pattern: header → data_a → data_b → header (same row, diff cols)
            if not (r0 == 0 and r1 > 0 and r2 > 0 and r3 == 0
                    and r1 == r2 and c1 != c2
                    and c0 == c1 and c3 == c2):
                continue

            row_i = r1 - 1
            j, k  = (c1, c2) if c1 < c2 else (c2, c1)
            key   = (row_i, j, k)
            if key in seen:
                continue
            seen.add(key)

            out.append(UPath(
                row_idx=row_i,
                col_idx_a=j,  col_idx_b=k,
                col_header_a=str(header[j]).strip() if j < len(header) else "",
                cell_value_a=str(rows[row_i][j]).strip() if j < len(rows[row_i]) else "",
                cell_value_b=str(rows[row_i][k]).strip() if k < len(rows[row_i]) else "",
                col_header_b=str(header[k]).strip() if k < len(header) else "",
            ))
    return out


def generate_u_paths_from_graph(
    header: list[str],
    rows: list[list[str]],
    num_walks: int = 50,
    chunk_size: int = 1,
) -> list[UPath]:
    """
    Use ``SemanticMetaPath`` graph walks to generate U-paths.

    Falls back to ``generate_u_paths_flat`` when the table is empty or
    the walker produces no usable walks.
    """
    if not rows:
        return generate_u_paths_flat(header, rows)

    if len(header) == 1:
        # Single-column table: graph walks can't form a valid U-path pair;
        # fall back to self-paths directly.
        return generate_u_paths_flat(header, rows)

    G = build_table_graph(header, rows)
    truncate = len(rows) > chunk_size
    smp_walker = SemanticMetaPath(G, chunk_size=chunk_size, truncate=truncate)
    walks  = smp_walker.simulate_walks_fully_structured_data(num_walks=num_walks)
    upaths = walks_to_u_paths(walks, header, rows)
    return upaths if upaths else generate_u_paths_flat(header, rows)


# ── SemanticMetaPath ──────────────────────────────────────────────────────────

import random as _random


class SemanticMetaPath:
    """
    Graph-based Semantic Meta-Path generator.

    Simulates structured random walks on a directed table graph to produce
    SMP and SMP-bar paths.  The graph must use ``"row,col"`` node IDs and
    edge labels ``"right"``, ``"left"``, ``"down"``, ``"up"`` (as built by
    ``build_table_graph``).

    Args:
        nx_G:       directed NetworkX graph of the table
        chunk_size: number of data rows per sub-graph chunk (for large tables)
        truncate:   if True, split the graph into ``chunk_size``-row sub-graphs
                    and merge the resulting walks
    """

    def __init__(self, nx_G, chunk_size: int = 1, truncate: bool = True):
        self.G        = nx_G
        self.truncate = truncate
        if truncate:
            self.chunk_size = chunk_size
            self.subG = self.create_subgraphs_from_chunks(self.G, chunk_size=chunk_size)

    # ── Sub-graph construction ────────────────────────────────────────────────

    def create_subgraphs_from_chunks(self, G=None, chunk_size: int = 1):
        G = G.copy() if G is not None else self.G.copy()
        max_row = max(int(node.split(",")[0]) for node in G.nodes())
        max_col = max(int(node.split(",")[1]) for node in G.nodes())

        node_headers = [f"0,{i}" for i in range(max_col + 1) if f"0,{i}" in G.nodes()]

        chunks = [
            list(range(1 + i * chunk_size, min(1 + (i + 1) * chunk_size, max_row + 1)))
            for i in range((max_row + chunk_size - 1) // chunk_size)
        ]

        nodes_subgraph = []
        for chunk in chunks:
            node_subgraph_ = []
            for i in range(max_col + 1):
                node_subgraph_ += [f"{j},{i}" for j in chunk if f"{j},{i}" in G.nodes()]
            nodes_subgraph.append(node_subgraph_ + node_headers)

        row_bridge_connector = []
        for chunk in chunks:
            row_bridge_connector_ = []
            for i in range(max_col + 1):
                node_connector = "-1"
                for j in chunk:
                    if f"{j},{i}" in G.nodes():
                        node_connector = f"{j},{i}"
                        break
                row_bridge_connector_.append(node_connector)
            row_bridge_connector.append(row_bridge_connector_)

        node_headers_dup = [node_headers] * len(row_bridge_connector)

        down_edges = [
            (h, r, {"label": "down"})
            for header, row in zip(node_headers_dup, row_bridge_connector)
            for h, r in zip(header, row)
            if "-1" not in (h, r)
        ]
        up_edges = [
            (h, r, {"label": "up"})
            for header, row in zip(row_bridge_connector, node_headers_dup)
            for h, r in zip(header, row)
            if "-1" not in (h, r)
        ]
        G.add_edges_from(down_edges + up_edges)

        return [G.subgraph(nodes).copy() for nodes in nodes_subgraph]

    # ── Walk methods ──────────────────────────────────────────────────────────

    def PMP_walk_structured_table_optimized(self, walk: list):
        G = self.G
        walk_ = walk.copy()
        current_node  = walk_[-1]
        previous_node = walk_[-3]

        if G.edges[previous_node, current_node]["label"] == "right":
            directions = ["right", "up"]
        elif G.edges[previous_node, current_node]["label"] == "up":
            directions = ["up"]
        elif G.edges[previous_node, current_node]["label"] == "down":
            directions = ["down", "right"]
        else:
            directions = []

        allowed_neighbors = [
            n for n in G.successors(current_node)
            if G.edges[current_node, n]["label"] in directions
        ]

        if not allowed_neighbors:
            return [walk_]

        allowed_dirs = [G.edges[current_node, n]["label"] for n in allowed_neighbors]
        walk_ = [walk_.copy() for _ in allowed_neighbors]
        for i, (direction, node) in enumerate(zip(allowed_dirs, allowed_neighbors)):
            walk_[i].extend([direction, node])

        output_walks = []
        for w in walk_:
            result = self.PMP_walk_structured_table_optimized(w)
            if not output_walks:
                output_walks.extend(result)
            else:
                output_walks.append(result)

        return self.extract_walks_from_compact_sublist(output_walks)

    def PMP_walk_structured_table(self, walk: list):
        G = self.G
        walk_ = walk.copy()
        while True:
            current_node  = walk_[-1]
            previous_node = walk_[-3]

            label = G.edges[previous_node, current_node]["label"]
            if label == "right":
                directions = ["right", "up"]
            elif label == "up":
                directions = ["up"]
            elif label == "down":
                directions = ["down", "right"]
            else:
                directions = []

            allowed_neighbors = [
                n for n in G.successors(current_node)
                if G.edges[current_node, n]["label"] in directions
            ]
            if not allowed_neighbors:
                break

            next_node = _random.choice(allowed_neighbors)
            walk_.append(G.edges[current_node, next_node]["label"])
            walk_.append(next_node)

        return walk_

    def SMP_walk(self, walk: list):
        G = self.G
        walk_ = walk.copy()
        while True:
            current_node  = walk_[-1]
            previous_node = walk_[-2]

            edge_label = G.edges[previous_node, current_node]["label"]
            node_class = G.nodes[current_node]["class"]

            if edge_label == "right" and "VALUE" not in node_class:
                directions = ["right"]
            elif edge_label == "right" and "VALUE" in node_class:
                directions = ["right", "up"]
            elif edge_label == "up":
                directions = ["up"]
            else:
                directions = []

            allowed_neighbors = [
                n for n in G.successors(current_node)
                if G.edges[current_node, n]["label"] in directions
            ]
            if not allowed_neighbors:
                break

            next_node = _random.choice(allowed_neighbors)
            walk_.append(next_node)

        return walk_

    def PMP_walk_v2(self, walk: list):
        G = self.G
        walk_ = walk.copy()
        while True:
            current_node  = walk_[-1]
            previous_edge = walk_[-2]
            node_class    = G.nodes[current_node]["class"]

            if previous_edge == "right" and "VALUE" not in node_class:
                directions = ["right"]
            elif previous_edge == "right" and "VALUE" in node_class:
                directions = ["right", "up"]
            elif previous_edge == "up":
                directions = ["up"]
            else:
                directions = []

            allowed_neighbors = [
                n for n in G.successors(current_node)
                if G.edges[current_node, n]["label"] in directions
            ]
            if not allowed_neighbors:
                break

            next_node = _random.choice(allowed_neighbors)
            walk_.append(G.edges[current_node, next_node]["label"])
            walk_.append(next_node)

        return walk_

    # ── Walk simulation ───────────────────────────────────────────────────────

    def simulate_walks_v2(self, num_walks: int = 50):
        G = self.G
        walks = []
        row_headers = []

        for node in G.nodes():
            nc = G.nodes[node]["class"]
            if nc in ["DESIGNATION", "TABLE_CAPTION", "PARAGRAPH"] \
               or "VALUE" in nc \
               or not G.nodes[node]["header"]:
                continue
            row_headers.append(node)

        if not row_headers:
            return []

        candidates = []
        for node in row_headers:
            out_labels = [d["label"] for _, _, d in G.out_edges(node, data=True)]

            if not {"up", "left"}.intersection(out_labels):
                pass
            elif not {"up"}.intersection(out_labels):
                continue

            if "left" in out_labels:
                left_neighbors = [v for u, v, d in G.edges(node, data=True)
                                  if d.get("label") == "left"]
                if set(left_neighbors).intersection(row_headers):
                    continue

            candidates.append(node)
            walks_ = self.search_init_walks_v2([node], label=G.nodes[node]["class"])
            if walks_:
                walks.extend(walks_)

        walks = [
            element
            for sublist in walks
            for element in (sublist if isinstance(sublist[0], list) else [sublist])
        ]

        pmp_nodes = []
        for walk in walks:
            if len(walk) < 2:
                continue
            for _ in range(num_walks):
                pmp_nodes.append(self.PMP_walk_v2(walk))

        pmp_nodes = list(set(tuple(w) for w in pmp_nodes))
        pmp_nodes = [list(t) for t in pmp_nodes]

        _flip = {"up": "down", "down": "up", "left": "right", "right": "left"}
        pmp_reverse = [[_flip.get(x, x) for x in w[::-1]] for w in pmp_nodes]

        return [*pmp_nodes, *pmp_reverse]

    def simulate_walks_fully_structured_data_optimized(
        self, num_walks: int = 500, num_row: int = 30,
        list_rows_header: list = None,
    ):
        sub_normal, sub_reverse = [], []
        for subg in self.subG:
            result = self.__class__(subg, chunk_size=self.chunk_size,
                                    truncate=False)\
                         .simulate_walks_fully_structured_data(
                             num_walks=num_walks, num_row=num_row,
                             list_rows_header=list_rows_header,
                         )
            half = len(result) // 2
            sub_normal.extend(result[:half])
            sub_reverse.extend(result[half:])
        return sub_normal + sub_reverse

    def simulate_walks_fully_structured_data(
        self, num_walks: int = 500, num_row: int = 30,
        list_rows_header: list = None,
    ):
        G = self.G
        walks = []

        max_col     = max(int(n.split(",")[1]) for n in G.nodes())
        last_node   = f"0,{max_col}"
        col_headers = [n for n in G.nodes() if G.nodes[n]["header"]]

        if list_rows_header is not None:
            col_headers = [col_headers[i] for i in list_rows_header]

        if len(col_headers) > 1:
            col_headers = [n for n in col_headers if n != last_node]

        for node in col_headers:
            filtered = [
                n for n in G.successors(node)
                if G.edges[node, n]["label"] == "down"
            ]
            if filtered:
                walks.append([node, G.edges[node, filtered[-1]]["label"], filtered[-1]])

        pmp_nodes = []
        for i, walk in enumerate(walks):
            if len(walk) < 3:
                continue
            num_col  = len(walks) - i
            num_step = 2 ** (num_row - 1) * (2 ** num_col - 2) - 2 ** (num_col - 1) + 1
            num_step = max(min(int(num_step), num_walks), 1)

            if self.truncate:
                for _ in range(num_step):
                    pmp_nodes.append(self.PMP_walk_structured_table(walk))
            else:
                pmp_nodes.extend(self.PMP_walk_structured_table_optimized(walk))

        pmp_nodes = list(set(tuple(w) for w in pmp_nodes))
        pmp_nodes = [list(t) for t in pmp_nodes]

        _flip = {"up": "down", "down": "up", "left": "right", "right": "left"}
        pmp_reverse = [[_flip.get(x, x) for x in w[::-1]] for w in pmp_nodes]

        return [*pmp_nodes, *pmp_reverse]

    # ── Helpers ───────────────────────────────────────────────────────────────

    def extract_walks_from_compact_sublist(self, nested):
        walks = []

        def _recurse(item):
            if isinstance(item, list):
                if all(isinstance(x, str) for x in item):
                    walks.append(item)
                else:
                    for sub in item:
                        _recurse(sub)

        _recurse(nested)
        return walks

    def search_init_walks(self, walk: list, label):
        G = self.G
        walks_ = []
        seed = walk[-1]
        allowed = [
            n for n in G.successors(seed)
            if G.nodes[n]["class"] != label
            and G.edges[seed, n]["label"] != "up"
        ]
        for node_ in allowed:
            w = walk.copy()
            w.append(node_)
            if G.edges[seed, node_]["label"] == "right":
                walks_.append(w)
            else:
                walks_.extend(self.search_init_walks(w, label))
        return walks_

    def search_init_walks_v2(self, walk: list, label):
        G = self.G
        walks_ = []
        seed = walk[-1]
        allowed = [
            n for n in G.successors(seed)
            if G.nodes[n]["class"] != label
            and G.edges[seed, n]["label"] not in ["left", "up"]
        ]
        for node_ in allowed:
            w = walk.copy()
            w.append(G.edges[seed, node_]["label"])
            w.append(node_)
            if G.edges[seed, node_]["label"] == "right":
                walks_.append(w)
            else:
                walks_.extend(self.search_init_walks_v2(w, label))
        return walks_

