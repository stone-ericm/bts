"""Deterministic parquet output (spec §3: identical inputs → identical bytes). Tables are built — and so
type-checked — before anything is written."""
from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _sort_key(row: dict, keys: list[str]) -> tuple:
    return tuple("" if row.get(k) is None else str(row.get(k)) for k in keys)


def build_table(rows: list[dict], schema: pa.Schema, *, sort_keys: list[str], name: str) -> pa.Table:
    names = set(schema.names)
    for row in rows:
        extra = set(row) - names
        if extra:
            raise ValueError(f"columns not in schema for {name}: {sorted(extra)}")
    ordered = sorted(rows, key=lambda r: _sort_key(r, sort_keys))
    keys = [_sort_key(r, sort_keys) for r in ordered]
    if len(set(keys)) != len(keys):
        raise ValueError(f"duplicate sort keys in {name}; sort on a unique column")
    return pa.Table.from_pydict({f.name: [r.get(f.name) for r in ordered] for f in schema}, schema=schema)


def write_table(table: pa.Table, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, path, compression="zstd", use_dictionary=False, write_statistics=False)
    return path
