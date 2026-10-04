"""Ranked model surfaces for variables 13–14: served ``bts_slate_v1`` JSON adaptation and per-date admission
(amendment A3; review r1 findings 1 and 9, required change 5).

A served slate is a candidate input, never self-admitting. A date is admitted to the protocol stream only when an
independent witness record binds this exact file (sha256) and evidences every component of the original at-lock /
manifest rule — candidate universe, lineup assumptions, feature computation and a prediction timestamp at or before
the production lock — and the slate is selection-consistent with the locked primary. Selection consistency alone
never admits. With no witness file nothing is admitted: raw rank/probability stay null and their bins
``missing_surface``.

Rank is the 1-based position in the served row order (the writer's order); a batter's rank is his best row, so any
appearance counts for batter top-N coverage. A probability is attached only for a batter with exactly one row, and is
"target bound" for miscalibration only when the consensus voters' game (via the unit→game map) is that row's game.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import pandas as pd

from scripts.audit.benchmark_bridge.core import selection_consistency

SLATE_SCHEMA = "bts_slate_v1"
WITNESS_SCHEMA = "mining87_surface_witness_v1"
WITNESS_COMPONENTS = ("candidate_universe", "lineup_assumptions", "feature_computation", "prediction_timestamp_utc")


class SlateFormatError(ValueError):
    pass


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _int(v, field: str) -> int:
    if isinstance(v, bool) or not isinstance(v, int):
        raise SlateFormatError(f"row {field} is not an integer: {v!r}")
    return v


def parse_slate(raw: bytes, *, expected_date: str) -> dict:
    try:
        doc = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SlateFormatError(f"invalid json: {exc.__class__.__name__}") from exc
    if not isinstance(doc, dict) or doc.get("schema_version") != SLATE_SCHEMA:
        raise SlateFormatError(f"schema_version is not {SLATE_SCHEMA}")
    if doc.get("date") != expected_date:
        raise SlateFormatError(f"slate date {doc.get('date')!r} is not the file date {expected_date}")
    if not isinstance(doc.get("rows"), list):
        raise SlateFormatError("rows is not a list")
    rows, seen = [], set()
    for i, r in enumerate(doc["rows"]):
        if not isinstance(r, dict):
            raise SlateFormatError(f"row {i} is not an object")
        b, g = _int(r.get("batter_id"), "batter_id"), _int(r.get("game_pk"), "game_pk")
        p = r.get("p_game_hit")
        if p is not None and (isinstance(p, bool) or not isinstance(p, (int, float)) or not math.isfinite(p)
                              or not 0.0 <= p <= 1.0):
            raise SlateFormatError(f"row {i} p_game_hit invalid: {p!r}")
        if (b, g) in seen:
            raise SlateFormatError(f"duplicate candidate row (batter {b}, game {g})")
        seen.add((b, g))
        rows.append({"rank": i + 1, "batter_id": b, "game_pk": g, "p_game_hit": None if p is None else float(p)})
    probs = [r["p_game_hit"] for r in rows if r["p_game_hit"] is not None]
    return {"date": expected_date, "written_at": doc.get("written_at"), "n_rows": len(rows), "rows": rows,
            "order_matches_probability": all(a >= b for a, b in zip(probs, probs[1:]))}


def batter_table(rows: list[dict]) -> dict[int, dict]:
    out: dict[int, dict] = {}
    for r in rows:
        entry = out.setdefault(r["batter_id"], {"rank": r["rank"], "rows": []})
        entry["rank"] = min(entry["rank"], r["rank"])
        entry["rows"].append((r["rank"], r["game_pk"], r["p_game_hit"]))
    return out


def load_witnesses(path: Path | None) -> tuple[dict[str, list[dict]], dict]:
    """Witness file (an input; hashed and pinned like every other)::

        {"schema": "mining87_surface_witness_v1",
         "witnesses": [{"date": "YYYY-MM-DD", "surface_sha256": "<sha256 of picks/slates/<date>.json>",
                        "independent_of_served_slate": true, "source": "<the independent record>",
                        "candidate_universe": "<evidence ref>", "lineup_assumptions": "<evidence ref>",
                        "feature_computation": "<evidence ref>",
                        "prediction_timestamp_utc": "<ISO-8601 with offset>"}]}

    One record per date; two records for a date are a conflict and admit nothing."""
    if path is None:
        return {}, {"source": "none_supplied", "n_records": 0}
    doc = json.loads(Path(path).read_text())
    if not isinstance(doc, dict) or doc.get("schema") != WITNESS_SCHEMA or not isinstance(doc.get("witnesses"), list):
        raise ValueError(f"witness file schema is not {WITNESS_SCHEMA}")
    out: dict[str, list[dict]] = {}
    for w in doc["witnesses"]:
        out.setdefault(str(w.get("date")), []).append(w)
    return out, {"source": str(path), "n_records": len(doc["witnesses"])}


def _utc(raw) -> pd.Timestamp | None:
    if not isinstance(raw, str):
        return None
    try:
        t = pd.Timestamp(raw)
    except (ValueError, TypeError):
        return None
    return None if t.tzinfo is None else t.tz_convert("UTC")


def admit_surface(*, date: str, slate, slate_sha256: str | None, witnesses: list[dict],
                  production_primary: dict | None) -> dict:
    """Per-date admission record. ``slate`` is a parsed slate, a SlateFormatError, or None (no file)."""
    out = {"date": date, "admitted": False, "reason": None, "selection_consistency": None,
           "slate_sha256": slate_sha256, "n_witness_records": len(witnesses), "witness_source": None}
    if slate is None:
        return {**out, "reason": "no_served_slate"}
    if isinstance(slate, Exception):
        return {**out, "reason": f"slate_format_invalid:{slate}"}
    if production_primary is not None:
        out["selection_consistency"] = selection_consistency(slate["rows"], production_primary, None)["state"]
    if not witnesses:
        return {**out, "reason": "no_independent_witness"}
    if len(witnesses) > 1:
        return {**out, "reason": "conflicting_witnesses"}
    w = witnesses[0]
    out["witness_source"] = w.get("source")
    out["witness_components"] = {comp: w.get(comp) for comp in WITNESS_COMPONENTS}
    if w.get("surface_sha256") != slate_sha256:
        return {**out, "reason": "witness_surface_hash_mismatch"}
    for comp in WITNESS_COMPONENTS:
        if not isinstance(w.get(comp), str) or not w[comp].strip():
            return {**out, "reason": f"witness_missing_{comp}"}
    if w.get("independent_of_served_slate") is not True or not isinstance(w.get("source"), str) \
            or not w["source"].strip():
        return {**out, "reason": "witness_not_independent"}
    if production_primary is None:
        return {**out, "reason": "no_locked_production_primary"}
    lock = _utc(production_primary.get("locked_at"))
    if lock is None:
        return {**out, "reason": "lock_time_unknown"}
    predicted = _utc(w["prediction_timestamp_utc"])
    if predicted is None:
        return {**out, "reason": "witness_prediction_timestamp_invalid"}
    if predicted > lock:
        return {**out, "reason": "prediction_after_lock"}
    if out["selection_consistency"] != "selection_consistent":
        return {**out, "reason": "selection_inconsistent"}
    return {**out, "admitted": True, "reason": "witness_admitted"}


def consensus_surface(batter_id, batters: dict | None, *, admitted: bool, unit_game: int | None) -> dict:
    """Raw rank/probability for one consensus batter on one date's surface."""
    none = {"rank": None, "p_game_hit": None, "probability_target_bound": False}
    if not admitted or batters is None:
        return {**none, "status": "no_admitted_surface"}
    entry = batters.get(batter_id) if batter_id is not None else None
    if entry is None:
        return {**none, "status": "off_surface"}
    if len(entry["rows"]) > 1:
        return {**none, "rank": entry["rank"], "status": "ambiguous_multiple_rows"}
    _rank, game, p = entry["rows"][0]
    if unit_game is not None and unit_game != game:
        return {**none, "rank": entry["rank"], "status": "row_game_differs_from_consensus_game"}
    return {"rank": entry["rank"], "p_game_hit": p, "status": "unique_row",
            "probability_target_bound": unit_game is not None and p is not None}
