"""Generate deterministic row and column permutations of WikiSQL-style JSONL."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import random
from enum import Enum
from pathlib import Path
from typing import Iterable


class PermutationKind(str, Enum):
    """Supported physical-table permutations."""

    ROWS = "rows"
    COLUMNS = "columns"
    ROWS_AND_COLUMNS = "rows_columns"


def _table_rng(seed: int, table_id: str, kind: PermutationKind, axis: str) -> random.Random:
    """Create a stable RNG independent of JSONL record order and Python hash randomization."""
    material = f"{seed}:{table_id}:{kind.value}:{axis}".encode("utf-8")
    return random.Random(int.from_bytes(hashlib.sha256(material).digest()[:8], "big"))


def _permutation(length: int, rng: random.Random) -> list[int]:
    order = list(range(length))
    rng.shuffle(order)
    return order


def _new_indices(order: list[int]) -> dict[int, int]:
    return {old_index: new_index for new_index, old_index in enumerate(order)}


def permute_record(
    record: dict,
    *,
    row_order: list[int] | None = None,
    column_order: list[int] | None = None,
) -> dict:
    """Return a copy of one record with its table and coordinate fields remapped."""
    result = copy.deepcopy(record)
    rows = result.get("rows", [])
    header = result.get("header", [])

    if row_order is not None:
        if len(row_order) != len(rows):
            raise ValueError("Row permutation length does not match record rows.")
        row_new_index = _new_indices(row_order)
        result["rows"] = [rows[index] for index in row_order]
        if "target_rows" in result:
            result["target_rows"] = [row_new_index[index] for index in result["target_rows"]]

    if column_order is not None:
        if len(column_order) != len(header):
            raise ValueError("Column permutation length does not match record header.")
        if any(len(row) != len(header) for row in rows):
            raise ValueError("Cannot permute columns in a non-rectangular table.")
        column_new_index = _new_indices(column_order)
        result["header"] = [header[index] for index in column_order]
        result["rows"] = [[row[index] for index in column_order] for row in result["rows"]]
        for field in ("target_column", "target_columns", "condition_columns"):
            if field not in result:
                continue
            if isinstance(result[field], list):
                result[field] = [column_new_index[index] for index in result[field]]
            else:
                result[field] = column_new_index[result[field]]
    return result


def _transform_records(
    records: Iterable[dict], kind: PermutationKind, seed: int,
) -> list[dict]:
    transformed: list[dict] = []
    table_orders: dict[str, tuple[list[int] | None, list[int] | None]] = {}
    for record in records:
        table_id = str(record.get("table_id", ""))
        row_order, column_order = table_orders.get(table_id, (None, None))
        if row_order is None and kind in (PermutationKind.ROWS, PermutationKind.ROWS_AND_COLUMNS):
            row_order = _permutation(len(record.get("rows", [])), _table_rng(seed, table_id, kind, "rows"))
        if column_order is None and kind in (PermutationKind.COLUMNS, PermutationKind.ROWS_AND_COLUMNS):
            column_order = _permutation(len(record.get("header", [])), _table_rng(seed, table_id, kind, "columns"))
        table_orders[table_id] = (row_order, column_order)
        permuted = permute_record(record, row_order=row_order, column_order=column_order)
        permuted["permutation_seed"] = seed
        permuted["permutation_kind"] = kind.value
        transformed.append(permuted)
    return transformed


def generate_permuted_datasets(input_path: str | Path, output_dir: str | Path, seed: int = 42) -> dict[PermutationKind, Path]:
    """Write row, column, and joint permutations of *input_path* into *output_dir*."""
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    records = [json.loads(line) for line in input_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    output_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[PermutationKind, Path] = {}
    for kind in PermutationKind:
        output_path = output_dir / f"{input_path.stem}_{kind.value}.jsonl"
        contents = _transform_records(records, kind, seed)
        with output_path.open("w", encoding="utf-8", newline="\n") as output_file:
            for record in contents:
                output_file.write(json.dumps(record, ensure_ascii=False) + "\n")
        generated[kind] = output_path
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source WikiSQL-style JSONL file.")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "datasets")
    parser.add_argument("--seed", type=int, default=42, help="Deterministic permutation seed.")
    args = parser.parse_args()
    for kind, path in generate_permuted_datasets(args.input, args.output_dir, args.seed).items():
        print(f"[{kind.value}] {path}")


if __name__ == "__main__":
    main()