"""park_drag_delta — external park ball-drag regime feature (context/shadow stack).

The table is produced OUTSIDE this repo (~/projects/juiced-ball-analysis,
`build_feature_table.py`): one row per (venue_id, calendar date), where each
row's values are computed only from that venue's games strictly BEFORE the
date. That materialization is what makes the training merge and the serving
lookup read the identical value for the same (venue_id, date) — no off-by-one
between a shift(1) training join and a `.last()`-style serving lookup.

Spec: docs/superpowers/specs/2026-07-07-park-drag-delta-context-feature.md.

Hard requirements enforced here:
- This module must NEVER raise into the pick path: any problem with the
  external artifact (missing, schema drift, duplicate keys, unreadable)
  degrades to an all-NaN feature plus a logged warning.
- Serving values are suppressed (None) when the table's source data is stale
  relative to the prediction date; historical training merges are unaffected
  (old rows are still correct as-of values).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ENV_VAR = "BTS_PARK_DRAG_TABLE"
DEFAULT_TABLE_PATH = Path("data/external/park_drag/park_drag_export.csv")
REQUIRED_COLS = ("venue_id", "date", "park_drag_delta")
# Serving is suppressed when the prediction date is more than this many days
# past the newest source game in the manifest (or, lacking a manifest, the
# newest table row). Regular data flow is a 1-day lag.
STALE_AFTER_DAYS = 4

_UNSET = object()
_CACHE: dict = {}


def _warn(msg: str) -> None:
    print(f"  [park_drag] WARNING: {msg}", file=sys.stderr)


def table_path(path: str | Path | None = None) -> Path:
    if path is not None:
        return Path(path)
    return Path(os.environ.get(ENV_VAR, DEFAULT_TABLE_PATH))


def load_table(path: str | Path | None = None) -> pd.DataFrame | None:
    """Load + validate the export table. Returns None on ANY problem."""
    p = table_path(path)
    try:
        if not p.exists():
            _warn(f"table not found at {p}; park_drag_delta will be NaN")
            return None
        t = pd.read_csv(p)
        missing = [c for c in REQUIRED_COLS if c not in t.columns]
        if missing:
            _warn(f"table {p} missing columns {missing}; ignoring table")
            return None
        t["date"] = pd.to_datetime(t["date"], errors="coerce")
        t = t.dropna(subset=["date", "venue_id"])
        t["venue_id"] = pd.to_numeric(t["venue_id"], errors="coerce")
        t = t.dropna(subset=["venue_id"])
        t["venue_id"] = t["venue_id"].astype("int64")
        if t.duplicated(["venue_id", "date"]).any():
            _warn(f"table {p} has duplicate (venue_id, date) keys; ignoring table")
            return None
        return t
    except Exception as e:  # noqa: BLE001 — never raise into the pick path
        _warn(f"failed to load table {p}: {e}")
        return None


def load_manifest(path: str | Path | None = None) -> dict | None:
    """Load the sibling manifest (freshness metadata). None if unavailable."""
    p = table_path(path).with_name("park_drag_manifest.json")
    try:
        if not p.exists():
            return None
        return json.loads(p.read_text())
    except Exception as e:  # noqa: BLE001
        _warn(f"failed to read manifest {p}: {e}")
        return None


def _reset_cache() -> None:
    _CACHE.clear()


def _get_cached_table() -> pd.DataFrame | None:
    if "table" not in _CACHE:
        _CACHE["table"] = load_table()
    return _CACHE["table"]


def attach_park_drag(df: pd.DataFrame, table: object = _UNSET) -> pd.DataFrame:
    """Left-merge park_drag_delta onto df by (venue_id, date).

    Never raises; on any problem the column is present and all-NaN. Row count
    and order are preserved. Doubleheader rows (same venue_id + date) share
    one value by construction.
    """
    try:
        t = _get_cached_table() if table is _UNSET else table
        if t is None:
            out = df.copy()
            out["park_drag_delta"] = np.nan
            return out
        cols = t[["venue_id", "date", "park_drag_delta"]].drop_duplicates(
            subset=["venue_id", "date"])
        return df.merge(cols, on=["venue_id", "date"], how="left")
    except Exception as e:  # noqa: BLE001 — Codex #3: production path survives
        _warn(f"attach failed ({e}); park_drag_delta set to NaN")
        out = df.copy()
        out["park_drag_delta"] = np.nan
        return out


def _freshness_reference(table: pd.DataFrame, manifest: dict | None) -> pd.Timestamp | None:
    if manifest and manifest.get("max_source_game_date"):
        try:
            return pd.Timestamp(manifest["max_source_game_date"])
        except Exception:  # noqa: BLE001
            pass
    if len(table):
        # export extends past the last source game; back off one day
        return table["date"].max() - pd.Timedelta(days=1)
    return None


def serving_value(table: pd.DataFrame | None, manifest: dict | None,
                  venue_id: int, on_date: pd.Timestamp,
                  stale_after_days: int = STALE_AFTER_DAYS) -> float | None:
    """Value for (venue_id, on_date) from the SAME table training merges read.

    Returns None (-> NaN downstream) when the table is missing, the row is
    absent, the stored value is NaN, or the table's source data is more than
    `stale_after_days` behind on_date (a stale number is worse than no number).
    """
    try:
        if table is None or venue_id is None:
            return None
        on_date = pd.Timestamp(on_date).normalize()
        ref = _freshness_reference(table, manifest)
        if ref is None or (on_date - ref).days > stale_after_days:
            _warn(f"table stale for serving on {on_date.date()} "
                  f"(source through {ref.date() if ref is not None else 'unknown'})")
            return None
        m = table[(table["venue_id"] == int(venue_id)) & (table["date"] == on_date)]
        if not len(m):
            return None
        v = m["park_drag_delta"].iloc[0]
        return None if pd.isna(v) else float(v)
    except Exception as e:  # noqa: BLE001
        _warn(f"serving lookup failed ({e}); returning None")
        return None
