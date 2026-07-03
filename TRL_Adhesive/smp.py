"""
smp.py — U-path generation from complex / nested Adhesive tables.

Unlike the WikiSQL tables used by ``TRL-model`` (a single header row + data
rows on a regular grid), the Adhesive dataset ships pre-computed *Semantic
Meta-Paths* (SMP) as plain-text walks over a table graph, plus a JSON file that
describes every cell.  This module turns those two artefacts into ``UPath``
objects that are structurally compatible with the ``TableEmbedJePA`` model.

Input artefacts (per table, keyed by the table UUID)
────────────────────────────────────────────────────
  AdhesiveTable_SMP_format/<uuid>.txt   — one walk per line:
        ``id dir id dir id …``   (dir ∈ {right, left, up, down})
        the ints are ``id-entry`` values that index into the JSON cells.

  AdhesiveTable_json_format/<uuid>.json — {"id-table", "Cells": [ … ]}
        each cell: ``id-entry`` (int), ``table-header`` ("Yes"/"No"),
        ``content`` (text), ``Relationships`` (right/left/up/down neighbours).

U-path extraction rules
───────────────────────
A *pivot* is any cell whose ``table-header`` == "Yes".  Because the tables are
nested / row-column oriented, a single walk may pass through several header
cells (multiple pivots).

A *node* is the value cell that sits at a **change of direction** (the elbow of
the L / U shape) — e.g. the walk goes ``right … right`` then turns ``up``; the
cell at the turn is the answer node.  Pass-through cells (same direction before
and after) are ignored.

Path shapes handled
  • L / L-reversed  ``header | node | header``  (row-column oriented matrix)
        e.g.  ``2 right 3 up 1``   →  rowHeader(2) | value(3) | colHeader(1)
              ``1 down 3 left 2``  →  colHeader(1) | value(3) | rowHeader(2)
  • single column   a walk with **no** direction change → treat every value
        cell as its own node paired with the column's header
        (``header | node | header`` with the header repeated).

Model compatibility
────────────────────
``TableEmbedJePA`` consumes a **variable-length, role-masked** SMP sequence — one
embedding slot per cell along the walk, in walk order::

    [a_h1, a_h2, …, node, …, b_h2, b_h1]

with a per-slot binary role mask (``1`` = node, ``0`` = header/pivot).  Because
tables can nest headers, ``pivot_a`` / ``pivot_b`` are *compositions* of one or
more header cells and the node's position shifts with the nesting depth; the
model therefore locates the node via the role mask rather than a fixed index.
``col_header_a`` / ``col_header_b`` still expose the ``" | "``-joined header text
(and ``cell_value_a`` / ``cell_value_b`` alias the single node) so query
composition and evaluation aggregation keep working unchanged.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

_DIRS = {"right", "left", "up", "down"}


# ── U-path dataclass ──────────────────────────────────────────────────────────

@dataclass
class UPath:
    """
    A variable-length ``header … | node | … header`` path from one SMP walk.

    Instead of a fixed 4-slot layout, the path keeps the *ordered* list of nested
    header cells on each side of the single value node, so the model can consume
    the walk as a role-masked variable-length sequence
    ``[a_h1, a_h2, …, node, …, b_h2, b_h1]`` (the node position varies with the
    nesting depth).  Backward-compatible aliases (``col_header_a`` etc.) still
    expose the flattened view used by query composition + evaluation.
    """
    a_header_texts: list[str]                          # ordered a-side (pre-node) header contents
    cell_value:     str                                # the single value node (turning cell)
    b_header_texts: list[str]                          # ordered b-side (post-node) header contents
    a_header_ids:   list[int] = field(default_factory=list)
    b_header_ids:   list[int] = field(default_factory=list)
    node_id:        int = -1                            # source ``id-entry`` of the value node

    table_id:       str = ""                           # source table UUID
    record_id:      str = ""                           # question/record id this sample belongs to

    # ── derived / aliases (keep dataset + train.py aggregation happy) ─────────
    col_header_a: str = field(default="", init=False)  # " | "-joined a-side headers
    col_header_b: str = field(default="", init=False)  # " | "-joined b-side headers
    cell_value_a: str = field(default="", init=False)  # == cell_value
    cell_value_b: str = field(default="", init=False)  # == cell_value
    pivot_a_id:   int = field(default=-1, init=False)   # nearest a-side header id
    pivot_b_id:   int = field(default=-1, init=False)   # nearest b-side header id
    col_idx_a:    int = field(default=0,  init=False)   # grouped by column-header id
    col_idx_b:    int = field(default=0,  init=False)
    tbl_row:      int = field(default=0,  init=False)   # grouped by row-header id
    smp_text:          str = field(default="", init=False)
    reversed_smp_text: str = field(default="", init=False)

    def __post_init__(self) -> None:
        self.col_header_a = " | ".join(self.a_header_texts)
        self.col_header_b = " | ".join(self.b_header_texts)
        self.cell_value_a = self.cell_value
        self.cell_value_b = self.cell_value
        # nearest header to the node on each side (used for dedup + aggregation)
        self.pivot_a_id = self.a_header_ids[-1] if self.a_header_ids else -1
        self.pivot_b_id = self.b_header_ids[0]  if self.b_header_ids else -1
        self.col_idx_a = self.pivot_b_id
        self.col_idx_b = self.pivot_b_id
        self.tbl_row   = self.pivot_a_id
        self.smp_text          = " | ".join(self.seq_texts())
        self.reversed_smp_text = " | ".join(self.seq_texts_bar())

    # ── ordered role-masked sequence views ────────────────────────────────────
    def seq_texts(self) -> list[str]:
        """Ordered walk texts ``[a_h…, node, …b_h]``."""
        return [*self.a_header_texts, self.cell_value, *self.b_header_texts]

    def seq_roles(self) -> list[int]:
        """``1`` at the node position, ``0`` at every header (aligned with ``seq_texts``)."""
        return [0] * len(self.a_header_texts) + [1] + [0] * len(self.b_header_texts)

    def seq_texts_bar(self) -> list[str]:
        """Reversed ordered walk ``[b_h…(rev), node, …a_h(rev)]``."""
        return [*reversed(self.b_header_texts), self.cell_value, *reversed(self.a_header_texts)]

    def seq_roles_bar(self) -> list[int]:
        """Role mask aligned with ``seq_texts_bar``."""
        return [0] * len(self.b_header_texts) + [1] + [0] * len(self.a_header_texts)

    @property
    def seq_len(self) -> int:
        return len(self.a_header_texts) + 1 + len(self.b_header_texts)

    @property
    def all_header_ids(self) -> list[int]:
        return [*self.a_header_ids, *self.b_header_ids]


# ── JSON / SMP parsing ────────────────────────────────────────────────────────

def load_table_cells(json_path: str | Path) -> tuple[dict[int, dict], str]:
    """
    Load a table JSON and return ``(cells, table_id)``.

    ``cells`` maps ``id-entry`` → ``{"id", "header", "content", "label"}``.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cells: dict[int, dict] = {}
    for c in data.get("Cells", []):
        try:
            cid = int(c["id-entry"])
        except (KeyError, ValueError, TypeError):
            continue
        cells[cid] = {
            "id":      cid,
            "header":  str(c.get("table-header", "")).strip().lower() == "yes",
            "content": str(c.get("content", "")).strip(),
            "label":   str(c.get("table-semantic-label", "")).strip(),
        }
    return cells, str(data.get("id-table", ""))


def parse_smp_line(line: str) -> Optional[tuple[list[int], list[str]]]:
    """
    Parse one SMP walk ``"id dir id dir id …"`` into ``(node_ids, dirs)``.

    Tokens alternate node-id / direction.  ``len(dirs) == len(node_ids) - 1``.
    Returns ``None`` for malformed / too-short lines.
    """
    toks = line.split()
    if len(toks) < 3:
        return None
    node_ids: list[int] = []
    dirs: list[str] = []
    for i, tok in enumerate(toks):
        if i % 2 == 0:                       # node id
            try:
                node_ids.append(int(tok))
            except ValueError:
                return None
        else:                                # direction
            dirs.append(tok.strip().lower())
    if len(node_ids) < 2 or len(dirs) != len(node_ids) - 1:
        return None
    return node_ids, dirs


def upaths_from_smp_line(
    node_ids: list[int],
    dirs: list[str],
    cells: dict[int, dict],
) -> list[UPath]:
    """
    Extract ``header | node | header`` U-paths from a single parsed SMP walk.

    • node       = cell at a change of direction (the L/U elbow).
    • pivot_a    = header(s) on the a-side (before the node in walk order).
    • pivot_b    = header(s) on the b-side (after the node in walk order).
    • no turn    = single column → every value cell becomes its own node,
                   paired with the walk's header (repeated on both pivots).
    """
    k = len(node_ids)
    header_flags = [cells.get(nid, {}).get("header", False) for nid in node_ids]
    contents     = [cells.get(nid, {}).get("content", "")   for nid in node_ids]

    # Interior nodes where the direction changes (dirs[i-1] != dirs[i]).
    turns = [i for i in range(1, k - 1) if dirs[i - 1] != dirs[i]]

    out: list[UPath] = []

    if not turns:
        # ── Single-column walk (no direction change) ──────────────────────────
        header_idxs = [i for i, h in enumerate(header_flags) if h]
        if header_idxs:
            h_idx = header_idxs[0]
            h_txt, h_id = contents[h_idx], node_ids[h_idx]
        else:
            h_txt, h_id = "", -1
        for i in range(k):
            if header_flags[i]:
                continue
            out.append(UPath(
                a_header_texts=[h_txt], cell_value=contents[i], b_header_texts=[h_txt],
                a_header_ids=[h_id], b_header_ids=[h_id], node_id=node_ids[i],
            ))
        return out

    # ── L / U walks — one U-path per value turning point ──────────────────────
    for t in turns:
        if header_flags[t]:
            continue  # elbow lands on a header, not a value node → skip
        a_headers = [contents[i] for i in range(t)         if header_flags[i]]
        b_headers = [contents[i] for i in range(t + 1, k)  if header_flags[i]]
        a_ids     = [node_ids[i] for i in range(t)         if header_flags[i]]
        b_ids     = [node_ids[i] for i in range(t + 1, k)  if header_flags[i]]
        out.append(UPath(
            a_header_texts=a_headers,
            cell_value=contents[t],
            b_header_texts=b_headers,
            a_header_ids=a_ids,
            b_header_ids=b_ids,
            node_id=node_ids[t],
        ))
    return out


def generate_upaths_for_table(
    smp_path: str | Path,
    json_path: str | Path,
) -> tuple[list[UPath], str]:
    """
    Build the de-duplicated list of ``UPath`` objects for one table.

    Forward and L-reversed walks describe the same cell from opposite
    orientations; they are collapsed via an unordered ``(node_id, {header ids})``
    key so each physical (node, nested-header-set) path appears once regardless
    of which side the headers were reached from.

    Returns ``(upaths, table_id)``.
    """
    cells, table_id = load_table_cells(json_path)

    out: list[UPath] = []
    seen: set = set()
    text = Path(smp_path).read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed = parse_smp_line(line)
        if parsed is None:
            continue
        node_ids, dirs = parsed
        for up in upaths_from_smp_line(node_ids, dirs, cells):
            key = (up.node_id, frozenset(up.all_header_ids))
            if key in seen:
                continue
            seen.add(key)
            up.table_id = table_id
            out.append(up)
    return out, table_id


# ── JSON-only U-path generation (no pre-computed SMP walk file) ────────────────

def _parse_rect(s) -> tuple[float, float, float, float]:
    """Parse an ``entry-rect`` string ``"(x, y, w, h)"`` → ``(x, y, w, h)``."""
    try:
        nums = [float(v) for v in str(s).strip("() \t").split(",")]
        while len(nums) < 4:
            nums.append(0.0)
        return nums[0], nums[1], nums[2], nums[3]
    except (ValueError, TypeError):
        return 0.0, 0.0, 0.0, 0.0


def load_table_graph(json_path: str | Path) -> tuple[dict[int, dict], str]:
    """
    Load a table JSON into a neighbour graph keyed by ``id-entry``.

    Each cell record carries ``header``/``content``/``label``, geometry
    (``x``/``y``/``w``/``h``) and the four neighbour id-lists
    (``up``/``down``/``left``/``right``) taken from ``Relationships``.
    """
    data = json.loads(Path(json_path).read_text(encoding="utf-8"))
    cells: dict[int, dict] = {}
    for c in data.get("Cells", []):
        try:
            cid = int(c["id-entry"])
        except (KeyError, ValueError, TypeError):
            continue
        rel = c.get("Relationships", {}) or {}

        def _ids(key: str) -> list[int]:
            out_ids: list[int] = []
            for v in rel.get(key, []) or []:
                try:
                    out_ids.append(int(v))
                except (ValueError, TypeError):
                    continue
            return out_ids

        x, y, w, h = _parse_rect(c.get("entry-rect"))
        cells[cid] = {
            "id":      cid,
            "header":  str(c.get("table-header", "")).strip().lower() == "yes",
            "content": str(c.get("content", "")).strip(),
            "label":   str(c.get("table-semantic-label", "")).strip(),
            "x": x, "y": y, "w": w, "h": h,
            "up":    _ids("up"),   "down":  _ids("down"),
            "left":  _ids("left"), "right": _ids("right"),
        }
    return cells, str(data.get("id-table", ""))


def _is_row_column_oriented(cells: dict[int, dict]) -> bool:
    """
    Decide whether the first (left-most) column is a header column.

    A *row-column oriented* (matrix) table has row headers down its first
    column; a *column-oriented* table has data there (headers only on top).
    The first column is the set of cells with no ``left`` neighbour (falling
    back to the smallest ``x`` if relationships are missing).
    """
    if not cells:
        return False
    first_col = [c for c in cells.values() if not c["left"]]
    if not first_col:
        xmin = min(c["x"] for c in cells.values())
        tol  = 1.0 + 0.02 * max((c["w"] for c in cells.values()), default=0.0)
        first_col = [c for c in cells.values() if abs(c["x"] - xmin) <= tol]
    if not first_col:
        return False
    header_frac = sum(1 for c in first_col if c["header"]) / len(first_col)
    return header_frac >= 0.5


def _nested_headers(cell: dict, cells: dict[int, dict], direction: str) -> list[dict]:
    """
    Walk ``direction`` (``"up"`` or ``"left"``) from *cell* through any
    intervening value cells to the first header, then keep collecting the
    consecutive (nested) headers beyond it.

    Returns the header cells **nearest-first** (closest to the node first).
    """
    visited = {cell["id"]}
    cur = cell
    # 1) advance through non-header cells until the first header is reached
    while True:
        nbrs = [n for n in cur[direction] if n in cells and n not in visited]
        if not nbrs:
            return []
        nxt = cells[nbrs[0]]
        visited.add(nxt["id"])
        if nxt["header"]:
            break
        cur = nxt
    # 2) collect the run of nested headers from here onward
    heads = [nxt]
    cur = nxt
    while True:
        nbrs = [n for n in cur[direction]
                if n in cells and n not in visited and cells[n]["header"]]
        if not nbrs:
            break
        h = cells[nbrs[0]]
        visited.add(h["id"])
        heads.append(h)
        cur = h
    return heads


def generate_upaths_from_json(json_path: str | Path) -> tuple[list[UPath], str]:
    """
    Build ``UPath`` objects from the table JSON **alone** (no SMP walk file).

    Steps
      1. Detect the table type from its first column (``_is_row_column_oriented``).
      2. For every value (non-header) cell, gather its nested column headers by
         ascending ``up``, and — for row-column tables — its nested row headers by
         going ``left``.
      3. Emit one path per value cell:
           • column-oriented → **U**  : ``colH… | node | …colH`` (headers mirrored)
           • row-column      → **L-rev**: ``rowH… | node | colH…``
         Nested headers are preserved on both sides via the ordered header lists.

    Returns ``(upaths, table_id)`` de-duplicated on ``(node_id, {header ids})``.
    """
    cells, table_id = load_table_graph(json_path)
    if not cells:
        return [], table_id

    row_col = _is_row_column_oriented(cells)
    out: list[UPath] = []
    seen: set = set()

    for cid, c in cells.items():
        if c["header"]:
            continue  # only value cells become nodes

        col_heads = _nested_headers(c, cells, "up")            # nearest-first
        row_heads = _nested_headers(c, cells, "left") if row_col else []

        if row_col:
            # L-reversed: row headers (a-side) | node | column headers (b-side)
            a_src = list(reversed(row_heads))                  # outer → nearest
            b_src = col_heads                                  # nearest → outer
        else:
            # U: mirror the column headers around the node
            a_src = list(reversed(col_heads))                  # outer → nearest
            b_src = col_heads                                  # nearest → outer

        a_texts = [h["content"] for h in a_src]
        a_ids   = [h["id"]      for h in a_src]
        b_texts = [h["content"] for h in b_src]
        b_ids   = [h["id"]      for h in b_src]
        if not a_texts and not b_texts:
            continue  # value cell with no reachable header → skip

        up = UPath(
            a_header_texts=a_texts, cell_value=c["content"], b_header_texts=b_texts,
            a_header_ids=a_ids, b_header_ids=b_ids, node_id=cid,
        )
        up.table_id = table_id
        key = (up.node_id, frozenset(up.all_header_ids))
        if key in seen:
            continue
        seen.add(key)
        out.append(up)
    return out, table_id
