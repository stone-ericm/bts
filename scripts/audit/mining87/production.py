"""The accepted W1.1 season ledger as the bounded, audited projection of locked production slots and their per-slot
settlement (amendment A1; review r1 finding 2, required changes 2 and 4).

Lock (frozen): a ledger ``selection`` row with ``commit_status == committed_evidenced`` and a finalized source
(``finalization`` ``decision`` or ``pick_file_only``). In the ledger that status needs a decision naming the selection
or a confirmed delivery from an agreeing pick file (``season_ledger.rows.commit_status``); previews, conflicting
evidence and unresolved views are never locks and are counted, not dropped silently. ``locked_at`` (scheduler lock
time) is carried but not required: its absence leaves the lock *time* unknown, not the commitment.

Settlement (frozen): the contest's own slot grade, only through a game-bound link — an ``evidenced`` unit capture of
the selected game or an ``inferred`` unique scheduled game — whose contest-slot row carries the same selection, batter
and game. ``hit``/``not_hit`` resolve; ``void`` is void; ungraded, unlinked, ambiguous, other-game and unrecognized
grades are unknown. None is ever a miss. Only the columns this module needs are read: no streaks, no local results.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from scripts.audit.season_ledger.compile import CONTEST_SCHEMA, LEDGER_SCHEMA
from scripts.canonicalize_realized_picks import ALL_REGIMES

LEDGER_FILE = "season_2026_ledger.parquet"
CONTEST_FILE = "season_2026_ledger_contest_slots.parquet"
ACCEPTED_FILE = "ACCEPTED.json"
BUILD_FILE = "season_2026_ledger_build.json"            # upstream manifest identity (identity keys only are read)
BUILD_IDENTITY_KEYS = ("builder_version", "code_sha", "bundle_manifest_sha256", "bundle_acquired_at_utc",
                       "rules_fingerprint")
EXPECTED_RULES_FINGERPRINT = "5e9d74f2f9c3093d66bc7c9ab0a7028e5fb361cdb46bbb1b368cd71e7f297b3f"
SLOT_TO_PICK_NUMBER = {"primary": 1, "double_down": 2}

LEDGER_COLUMNS = ["row_id", "row_kind", "date", "reason", "slot", "selection_id", "batter_id", "batter_name",
                  "game_pk", "p_stated", "projected_lineup", "finalization", "commit_status", "predicted_at",
                  "locked_at", "delivery_confirmed", "entry_status", "match", "match_reason", "round_id", "unit_id",
                  "player_id", "bts_outcome", "bts_outcome_status", "game_eligibility"]
CONTEST_COLUMNS = ["round_id", "unit_id", "player_id", "batter_id", "game_pk", "selection_id"]

ROW_KINDS = {"selection", "contest_only", "skip_day", "unfinalized_day", "unobserved_day"}
FINALIZATIONS = {"decision", "pick_file_only", "unresolved"}
COMMIT_STATUSES = {"committed_evidenced", "conflicted", "unconfirmed"}
OUTCOME_STATUSES = {"graded", "matched_ungraded", "unknown", "match_ambiguous"}
ENTRY_STATUSES = {"confirmed", "unknown"}
LOCKED_FINALIZATIONS = {"decision", "pick_file_only"}
GAME_BOUND_LINKS = {("evidenced", "unit_capture"), ("inferred", "pick_time_team_single_scheduled_game")}
CONTEST_GRADES = {"hit": "hit", "not_hit": "not_hit", "void": "void"}

CONTEXT_COLUMNS = {"batter_skill_prior_pa": "production_batter_skill_prior_pa",
                   "batter_skill_quartile": "production_batter_skill_quartile",
                   "pick_weather_temp": "production_weather_temp", "pick_is_indoor": "production_is_indoor",
                   "is_park_driven": "production_is_park_driven"}

PRODUCTION_UNIT_COLUMNS = [
    "date", "pick_number", "production_slot", "production_selection_id", "production_batter_id",
    "production_batter_name", "production_game_pk", "production_p_game_hit", "production_projected_lineup",
    "production_predicted_at", "production_locked_at", "production_lock_basis", "production_regime",
    "production_game_eligibility", "production_settlement", "production_settlement_reason",
    "production_resolved", "production_hit"]


def load_accepted_ledger(ledger_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Read a published W1.1 build: a parseable ``ACCEPTED.json`` naming the ledger and contest-slot files with the
    predeclared rules fingerprint, and both files' schemas exactly equal to the compiler's. Projection columns only."""
    d = Path(ledger_dir)
    receipt_path = d / ACCEPTED_FILE
    if not receipt_path.exists():
        raise FileNotFoundError(f"{d} has no {ACCEPTED_FILE}: not a published W1.1 build")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("rules_fingerprint") != EXPECTED_RULES_FINGERPRINT:
        raise ValueError(f"ACCEPTED.json rules fingerprint {receipt.get('rules_fingerprint')!r} is not the "
                         "predeclared one")
    if not {LEDGER_FILE, CONTEST_FILE} <= set(receipt.get("files") or []):
        raise ValueError(f"ACCEPTED.json names {receipt.get('files')}, not {LEDGER_FILE} and {CONTEST_FILE}")
    for name, schema in ((LEDGER_FILE, LEDGER_SCHEMA), (CONTEST_FILE, CONTEST_SCHEMA)):
        got = pq.read_schema(d / name).remove_metadata()
        if not got.equals(schema):
            raise ValueError(f"{name} schema differs from the compiler's")
    upstream = {"status": "missing"}
    if (d / BUILD_FILE).exists():
        build = json.loads((d / BUILD_FILE).read_text())
        if build.get("rules_fingerprint") != EXPECTED_RULES_FINGERPRINT:
            raise ValueError(f"{BUILD_FILE} rules fingerprint differs from ACCEPTED.json's predeclared one")
        upstream = {k: build.get(k) for k in BUILD_IDENTITY_KEYS}
    ledger = pq.read_table(d / LEDGER_FILE, columns=LEDGER_COLUMNS).to_pandas(integer_object_nulls=True)
    slots = pq.read_table(d / CONTEST_FILE, columns=CONTEST_COLUMNS).to_pandas(integer_object_nulls=True)
    return ledger, slots, {"run": receipt.get("run"), "rules_fingerprint": receipt["rules_fingerprint"],
                           "accepted_at_utc": receipt.get("accepted_at_utc"), "ledger_rows": int(len(ledger)),
                           "contest_slot_rows": int(len(slots)), "upstream_build": upstream}


def production_regime(predicted_at) -> str | None:
    """The canonicalizer's regime cutoffs, compared as instants (not strings) on the ledger's prediction time."""
    if predicted_at is None or (not isinstance(predicted_at, str) and pd.isna(predicted_at)):
        return None
    t = pd.Timestamp(predicted_at)
    for regime in ALL_REGIMES:
        if t >= pd.Timestamp(regime.cutoff_iso_utc):
            return regime.label
    return None


def projection_sha256(locked: pd.DataFrame) -> str:
    """Content hash of the bounded projection (sorted rows, canonical JSON), independent of input row order."""
    rows = [{k: (None if v is None or (not isinstance(v, str) and pd.isna(v)) else
                 v.item() if hasattr(v, "item") else v) for k, v in r.items()}
            for r in locked.sort_values(["date", "pick_number"]).to_dict("records")]
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def _check_vocab(frame: pd.DataFrame, col: str, allowed: set) -> None:
    bad = sorted({str(v) for v in frame[col].dropna().unique()} - allowed)
    if bad:
        raise ValueError(f"unexpected ledger {col} values {bad}")


def lock_status(row: dict) -> tuple[bool, str]:
    if row["finalization"] not in LOCKED_FINALIZATIONS:
        return False, "finalization_unresolved"
    if row["commit_status"] != "committed_evidenced":
        return False, f"commit_{row['commit_status']}"
    return True, f"{row['finalization']}:committed_evidenced"


def settlement(row: dict, contest_index: dict) -> tuple[str, str]:
    if row["entry_status"] != "confirmed" or row["match"] is None:
        return "unknown", "no_linked_contest_slot"
    if (row["match"], row["match_reason"]) not in GAME_BOUND_LINKS:
        return "unknown", "link_without_game_identity"
    key = (row["round_id"], row["unit_id"], row["player_id"])
    slot = contest_index.get(key)
    if slot is None:
        return "unknown", "contest_slot_missing"
    if (slot["selection_id"], slot["batter_id"], slot["game_pk"]) != (row["selection_id"], row["batter_id"],
                                                                      row["game_pk"]):
        return "unknown", "contest_slot_game_mismatch"
    status = row["bts_outcome_status"]
    if status == "matched_ungraded":
        return "ungraded", "contest_slot_not_graded"
    if status != "graded":
        return "unknown", f"outcome_status_{status}"
    grade = CONTEST_GRADES.get(row["bts_outcome"])
    if grade is None:
        return "unknown", "unrecognized_grade"
    return grade, "contest_void" if grade == "void" else "contest_graded"


def _none(v):
    return None if v is None or (not isinstance(v, str) and pd.isna(v)) else v


def project_locked_slots(ledger: pd.DataFrame, contest_slots: pd.DataFrame, *, window_start: str,
                         window_end: str) -> tuple[pd.DataFrame, dict]:
    """One row per locked production slot in the inclusive window, with settlement; plus counts of everything that
    is not a unit (no outcome values are counted here: hit and not_hit both count as ``resolved``)."""
    _check_vocab(ledger, "row_kind", ROW_KINDS)
    in_window = (ledger["date"] >= window_start) & (ledger["date"] <= window_end)
    frame = ledger[in_window]
    sels = frame[frame["row_kind"] == "selection"]
    for col, allowed in (("slot", set(SLOT_TO_PICK_NUMBER)), ("finalization", FINALIZATIONS),
                         ("commit_status", COMMIT_STATUSES), ("bts_outcome_status", OUTCOME_STATUSES),
                         ("entry_status", ENTRY_STATUSES)):
        _check_vocab(sels, col, allowed)
    contest_index = {(r["round_id"], r["unit_id"], r["player_id"]): r for r in contest_slots.to_dict("records")}
    units, unsupported = [], {}
    for raw in sels.to_dict("records"):
        row = {k: _none(v) for k, v in raw.items()}
        locked, basis = lock_status(row)
        if not locked:
            unsupported[basis] = unsupported.get(basis, 0) + 1
            continue
        status, reason = settlement(row, contest_index)
        units.append({
            "date": row["date"], "pick_number": SLOT_TO_PICK_NUMBER[row["slot"]], "production_slot": row["slot"],
            "production_selection_id": row["selection_id"], "production_batter_id": row["batter_id"],
            "production_batter_name": row["batter_name"], "production_game_pk": row["game_pk"],
            "production_p_game_hit": row["p_stated"], "production_projected_lineup": row["projected_lineup"],
            "production_predicted_at": row["predicted_at"], "production_locked_at": row["locked_at"],
            "production_lock_basis": basis, "production_regime": production_regime(row["predicted_at"]),
            "production_game_eligibility": row["game_eligibility"], "production_settlement": status,
            "production_settlement_reason": reason})
    locked = pd.DataFrame(units, columns=PRODUCTION_UNIT_COLUMNS[:-2])
    dup = locked.duplicated(["date", "pick_number"], keep=False)
    if dup.any():
        raise ValueError(f"duplicate locked production slot keys: "
                         f"{locked.loc[dup, ['date', 'pick_number']].drop_duplicates().to_dict('records')[:5]}")
    locked["production_resolved"] = locked["production_settlement"].isin(["hit", "not_hit"])
    locked["production_hit"] = locked["production_settlement"].map({"hit": 1, "not_hit": 0}).astype("Int64")
    for col in ("production_batter_id", "production_game_pk"):
        locked[col] = locked[col].astype("Int64")
    locked = locked.sort_values(["date", "pick_number"]).reset_index(drop=True)
    days = frame[frame["row_kind"].isin(["skip_day", "unfinalized_day", "unobserved_day"])]
    status_counts = locked["production_settlement"].replace({"hit": "resolved", "not_hit": "resolved"})
    inv = {"window": [window_start, window_end], "ledger_rows_outside_window": int((~in_window).sum()),
           "projection_sha256": projection_sha256(locked),
           "selection_rows_in_window": int(len(sels)), "locked_units": int(len(locked)),
           "locked_units_by_slot": {s: int((locked["production_slot"] == s).sum()) for s in SLOT_TO_PICK_NUMBER},
           "lock_unsupported_selections": dict(sorted(unsupported.items())),
           "lock_time_known": int(locked["production_locked_at"].notna().sum()),
           "day_rows_without_units": dict(sorted((days["row_kind"] + ":" + days["reason"].fillna("none"))
                                                 .value_counts().to_dict().items())),
           "contest_only_rows_in_window": int((frame["row_kind"] == "contest_only").sum()),
           "settlement_status": {k: int(v) for k, v in sorted(status_counts.value_counts().items())},
           "settlement_reason": {k: int(v) for k, v in
                                 sorted(locked["production_settlement_reason"].value_counts().items())}}
    return locked, inv



def attach_production_context(locked: pd.DataFrame, context: pd.DataFrame | None) -> tuple[pd.DataFrame, dict]:
    """Optional pinned context for decomposition variables 6, 7, 10–12, joined on the exact selection identity
    (date, slot, batter, game). Only the named context columns are taken; anything else in the file is ignored."""
    out = locked.copy()
    for target in CONTEXT_COLUMNS.values():
        out[target] = pd.NA
    if context is None:
        return out, {"source": "unavailable", "rows_matched": 0, "columns": []}
    key = ["date", "slot", "batter_id", "game_pk"]
    cols = [c for c in CONTEXT_COLUMNS if c in context.columns]
    ctx = context[key + cols].copy()
    ctx["date"] = pd.to_datetime(ctx["date"]).dt.strftime("%Y-%m-%d")
    if ctx.duplicated(key).any():
        raise ValueError("production context has duplicate (date, slot, batter_id, game_pk) keys")
    ctx = ctx.rename(columns={"slot": "production_slot", "batter_id": "production_batter_id",
                              "game_pk": "production_game_pk", **{c: f"_ctx_{c}" for c in cols}})
    for c in ("production_batter_id", "production_game_pk"):
        ctx[c] = ctx[c].astype("Int64")
    merged = out.merge(ctx, on=["date", "production_slot", "production_batter_id", "production_game_pk"],
                       how="left", indicator=True)
    for c in cols:
        merged[CONTEXT_COLUMNS[c]] = merged.pop(f"_ctx_{c}")
    matched = int((merged.pop("_merge") == "both").sum())
    return merged, {"source": "production_context", "rows_matched": matched, "columns": cols}
