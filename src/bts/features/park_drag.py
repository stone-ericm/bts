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

import functools
import json
import os
import sys
from contextlib import contextmanager
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
        if getattr(t["date"].dt, "tz", None) is not None:
            t["date"] = t["date"].dt.tz_localize(None)
        t["date"] = t["date"].dt.normalize()
        t["venue_id"] = pd.to_numeric(t["venue_id"], errors="coerce")
        t = t.dropna(subset=["venue_id"])
        if (t["venue_id"] % 1 != 0).any():
            _warn(f"table {p} has non-integral venue_id values; ignoring table")
            return None
        t["venue_id"] = t["venue_id"].astype("int64")
        if t.duplicated(["venue_id", "date"]).any():
            _warn(f"table {p} has duplicate (venue_id, date) keys; ignoring table")
            return None
        if t.empty:
            _warn(f"table {p} is empty after normalization; ignoring table")
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
    _WARNED.clear()


_WARNED: set = set()


def _warn_once(key: str, msg: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        _warn(msg)


def _file_sig(p: Path):
    try:
        st = p.stat()
        return (st.st_mtime_ns, st.st_size)
    except OSError:
        return None


def _fingerprint_now() -> str:
    sig = _file_sig(table_path())
    return "absent" if sig is None else f"{sig[0]}:{sig[1]}"


def artifact_fingerprint() -> str:
    """Identity of the external table file (mtime_ns:size, or 'absent').

    Baked into the shadow cache hash so a shadow model trained before the
    table appeared (or against an older table) is never reused after the
    file changes. Pin-aware: stable within a pinned() cycle so the cache
    path computed before run_pipeline and the provenance path computed
    after it always agree."""
    pin = _CACHE.get("pin")
    if pin is not None:
        return pin["fingerprint"]
    return _fingerprint_now()


def _current_table() -> pd.DataFrame | None:
    """mtime/size-invalidated table cache (a days-long daemon picks up the
    daily refresh on the next access after the file changes)."""
    sig = _file_sig(table_path())
    if _CACHE.get("table_sig", "__unset__") != sig:
        _CACHE["table"] = load_table()
        _CACHE["table_sig"] = sig
    return _CACHE.get("table")


def _current_manifest() -> dict | None:
    p = table_path().with_name("park_drag_manifest.json")
    sig = _file_sig(p)
    if _CACHE.get("manifest_sig", "__unset__") != sig:
        _CACHE["manifest"] = load_manifest()
        _CACHE["manifest_sig"] = sig
    return _CACHE.get("manifest")


def get_table() -> pd.DataFrame | None:
    """Table snapshot. Inside a pinned() scope this is frozen for the whole
    scope, so one prediction cycle trains and serves from the SAME artifact
    even if the file is atomically replaced mid-cycle."""
    pin = _CACHE.get("pin")
    if pin is not None:
        return pin["table"]
    return _current_table()


def get_manifest() -> dict | None:
    """Manifest snapshot (pin-aware, see get_table)."""
    pin = _CACHE.get("pin")
    if pin is not None:
        return pin["manifest"]
    return _current_manifest()


@contextmanager
def pinned():
    """Freeze (table, manifest, fingerprint) for the enclosed scope.

    Reentrant: an inner pinned() inherits the outer snapshot, so decorating
    both the scheduler's shadow cycle and run_pipeline is safe."""
    if _CACHE.get("pin") is not None:
        yield
        return
    _CACHE["pin"] = {
        "table": _current_table(),
        "manifest": _current_manifest(),
        "fingerprint": _fingerprint_now(),
    }
    try:
        yield
    finally:
        _CACHE.pop("pin", None)


def with_pinned_artifact(fn):
    """Decorator form of pinned() for whole-cycle functions."""
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with pinned():
            return fn(*args, **kwargs)
    return wrapper


def attach_park_drag(df: pd.DataFrame, table: object = _UNSET) -> pd.DataFrame:
    """Left-merge park_drag_delta onto df by (venue_id, date).

    Never raises; on any problem the column is present and all-NaN. Row count
    and order are preserved. Doubleheader rows (same venue_id + date) share
    one value by construction.
    """
    try:
        t = get_table() if table is _UNSET else table
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
        table_max = table["date"].max()
        if on_date > table_max:
            _warn_once(f"cover-{on_date.date()}",
                       f"table does not cover prediction date {on_date.date()} "
                       f"(table ends {table_max.date()}); serving None for all venues")
            return None
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
