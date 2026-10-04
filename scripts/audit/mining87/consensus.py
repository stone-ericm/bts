"""Public picks → per-cohort legal consensus pairs and conservative consensus settlement (amendment A2; review r1
findings 4–5, required changes 3 and 11).

Votes are distinct users per ``batter_id``; names are display fields only. Slot 1 is the primary mode; slot 2 is the
DD mode among ids other than the chosen primary, with the slot's original vote denominator kept. Ties at the
eligible maximum break to the lowest id and are flagged, including ties exposed by excluding the primary and DDs
whose choice depends on a tied primary. Ids are chosen from vote counts alone, before any outcome is looked at; the
consensus settlement is then computed for the chosen (legal) id's voters only.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

from bts.leaderboard.storage import safe_filename_component

USER_PICK_COLUMNS = ["captured_at", "pick_date", "pick_number", "unit_id", "batter_id", "batter_name", "result"]
SNAPSHOT_COLUMNS = ["captured_at", "tab", "rank", "username", "streak"]
PICK_NUMBERS = (1, 2)
SETTLED = {"hit": "hit", "not_hit": "not_hit", "void": "void"}
PENDING = {"", None}
OBS_IDENTITY = ["batter_id", "unit_id", "result"]          # names are display fields, never an identity


# --- inputs ----------------------------------------------------------------------------------------------------------

def load_public_picks(user_picks_dir: Path, *, window_start: str, window_end: str,
                      capture_end_exclusive: str) -> tuple[pd.DataFrame, dict]:
    """The latest observation per (user, date, slot) in the registered window, captured before the cutoff; the user
    is the file stem (the corpus is username-keyed and has no user column). Each scrape appends a user's whole
    history, so every file is reduced before concatenation. Each excluded row is counted under its first failing
    rule; ambiguous latest observations are counted and cast no vote."""
    files = sorted(Path(user_picks_dir).glob("*.parquet"))
    cutoff = pd.Timestamp(capture_end_exclusive)
    parts, empty = [], 0
    counts = {"rows_read": 0, "rows_outside_window": 0, "rows_after_capture_cutoff": 0, "rows_invalid_pick_number": 0,
              "user_slot_observations": 0, "ambiguous_user_slot_observations": 0}
    ranges = {"pick_date": [None, None], "captured_at": [None, None]}      # of every row read, before filters
    for path in files:
        if pq.read_metadata(path).num_rows == 0:
            empty += 1
            continue
        names = set(pq.read_schema(path).names)
        missing = sorted(set(USER_PICK_COLUMNS) - names)
        if missing:
            raise ValueError(f"{path.name} lacks user-pick columns {missing}")
        frame = pq.read_table(path, columns=USER_PICK_COLUMNS).to_pandas()
        counts["rows_read"] += len(frame)
        frame["pick_date"] = pd.to_datetime(frame["pick_date"]).dt.strftime("%Y-%m-%d")
        frame["captured_at"] = pd.to_datetime(frame["captured_at"])
        for key, col in (("pick_date", "pick_date"), ("captured_at", "captured_at")):
            lo, hi = frame[col].min(), frame[col].max()
            ranges[key] = [lo if ranges[key][0] is None else min(lo, ranges[key][0]),
                           hi if ranges[key][1] is None else max(hi, ranges[key][1])]
        in_window = (frame["pick_date"] >= window_start) & (frame["pick_date"] <= window_end)
        counts["rows_outside_window"] += int((~in_window).sum())
        frame = frame[in_window]
        before = frame["captured_at"] < cutoff
        counts["rows_after_capture_cutoff"] += int((~before).sum())
        frame = frame[before]
        valid_slot = frame["pick_number"].isin(PICK_NUMBERS)
        counts["rows_invalid_pick_number"] += int((~valid_slot).sum())
        frame = frame[valid_slot].copy()
        frame["username"] = path.stem
        latest, inv = latest_observations(frame)
        for k in ("user_slot_observations", "ambiguous_user_slot_observations"):
            counts[k] += inv[k]
        parts.append(latest)
    obs = (pd.concat(parts, ignore_index=True) if parts
           else pd.DataFrame(columns=[*USER_PICK_COLUMNS, "username"]))
    stamp = lambda t: None if t is None else pd.Timestamp(t).isoformat()  # noqa: E731
    return obs, {"user_pick_files": len(files), "empty_user_pick_files": empty, **counts,
                 "users_with_retained_rows": int(obs["username"].nunique()),
                 "pick_date_range_read": ranges["pick_date"],
                 "captured_at_range_read": [stamp(t) for t in ranges["captured_at"]]}


def latest_observations(obs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """The latest capture per (user, date, slot). Rows sharing that latest stamp must agree on id, unit, result and
    name; otherwise the user's slot is ambiguous and casts no vote (counted, never resolved by order)."""
    if obs.empty:
        return obs.copy(), {"user_slot_observations": 0, "ambiguous_user_slot_observations": 0}
    key = ["username", "pick_date", "pick_number"]
    latest_stamp = obs.groupby(key)["captured_at"].transform("max")
    top = obs[obs["captured_at"] == latest_stamp].copy()
    ident = top[OBS_IDENTITY].astype("string").fillna("<null>").agg("\x1f".join, axis=1)
    n_variants = ident.groupby([top[k] for k in key]).transform("nunique")
    ambiguous = top[n_variants > 1].drop_duplicates(key)
    kept = (top[n_variants == 1].sort_values([*key, "batter_name"], na_position="last")
            .drop_duplicates(key).reset_index(drop=True))
    return kept, {"user_slot_observations": int(len(kept) + len(ambiguous)),
                  "ambiguous_user_slot_observations": int(len(ambiguous))}


def load_cohort(snapshot_path: Path, *, tab: str) -> tuple[set[str], dict]:
    """Usernames in ``tab`` of one pinned snapshot file; no 'latest' search."""
    path = Path(snapshot_path)
    if not path.exists():
        raise FileNotFoundError(f"pinned cohort snapshot missing: {path}")
    frame = pq.read_table(path, columns=SNAPSHOT_COLUMNS).to_pandas()
    rows = frame[frame["tab"] == tab]
    usernames = {str(u) for u in rows["username"].dropna()}
    if not usernames:
        raise ValueError(f"cohort snapshot {path.name} has no {tab} rows")
    return usernames, {
        "snapshot_file": path.name, "tab": tab, "n_rows_in_tab": int(len(rows)), "n_usernames": len(usernames),
        "captured_at_values": sorted({pd.Timestamp(t).isoformat() for t in rows["captured_at"].dropna()}),
        "membership_sha256": hashlib.sha256(json.dumps(sorted(usernames)).encode()).hexdigest()}


def cohort_stems(usernames: set[str], *, available_stems: set[str]) -> tuple[set[str], dict]:
    """Map usernames to pick-file stems with the scraper's own sanitizer. Several usernames sharing one stem share
    one file (counted as a collision); a member without a file contributes no votes."""
    by_stem: dict[str, list[str]] = {}
    for u in usernames:
        by_stem.setdefault(safe_filename_component(u), []).append(u)
    present = {s for s in by_stem if s in available_stems}
    return present, {"n_stems": len(by_stem), "stems_with_pick_file": len(present),
                     "usernames_without_pick_file": sum(len(v) for s, v in by_stem.items() if s not in present),
                     "sanitization_collisions": sum(1 for v in by_stem.values() if len(v) > 1)}


# --- legal pair choice -------------------------------------------------------------------------------------------------

def _mode(counts: dict[int, int], exclude: int | None = None) -> tuple[int | None, int, list[int]]:
    eligible = {b: n for b, n in counts.items() if b != exclude}
    if not eligible:
        return None, 0, []
    top = max(eligible.values())
    tied = sorted(b for b, n in eligible.items() if n == top)
    return tied[0], top, tied


def choose_slots(slot1: dict[int, int], slot2: dict[int, int]) -> dict:
    """Vote counts {batter_id: distinct users} → the frozen legal pair (amendment A2)."""
    p_id, p_count, p_tied = _mode(slot1)
    primary = {"batter_id": p_id, "count": p_count, "tied_ids": p_tied, "tie_direct": len(p_tied) > 1,
               "status": "chosen" if p_id is not None else "no_votes"}
    u_id, _u_count, u_tied = _mode(slot2)
    dd = {"batter_id": None, "count": 0, "tied_ids": [], "tie_direct": False, "tie_exposed_by_exclusion": False,
          "tie_dependent_on_primary": False, "unconstrained_batter_id": u_id,
          "legal_differs_from_unconstrained": False}
    if not slot2:
        return {"primary": primary, "dd": {**dd, "status": "no_votes"}}
    if p_id is None:
        return {"primary": primary, "dd": {**dd, "status": "primary_unavailable"}}
    d_id, d_count, d_tied = _mode(slot2, exclude=p_id)
    if d_id is None:
        return {"primary": primary, "dd": {**dd, "status": "no_distinct_legal_id"}}
    dependent = primary["tie_direct"] and any(_mode(slot2, exclude=alt)[0] != d_id for alt in p_tied if alt != p_id)
    dd.update(batter_id=d_id, count=d_count, tied_ids=d_tied, tie_direct=len(d_tied) > 1,
              tie_exposed_by_exclusion=len(d_tied) > 1 and len(u_tied) == 1, tie_dependent_on_primary=dependent,
              legal_differs_from_unconstrained=d_id != u_id, status="chosen")
    return {"primary": primary, "dd": dd}


# --- settlement ------------------------------------------------------------------------------------------------------

def consensus_settlement(results: list, unit_ids: list) -> tuple[str, str, int | None]:
    """Frozen conservative rule for the chosen id's voters: one game identity (unit) or unknown; any unrecognized
    label is unknown; settled labels must agree (hit / not_hit / void) or the slot is unknown; pending-only is
    pending. Popularity never resolves a conflict."""
    units = {int(u) for u in unit_ids if u is not None and not pd.isna(u) and int(u) > 0}
    if len(units) > 1:
        return "unknown", "multiple_units_for_batter", None
    if not units:
        return "unknown", "no_unit_identity", None
    unit = next(iter(units))
    labels = [None if (r is None or (not isinstance(r, str) and pd.isna(r))) else str(r) for r in results]
    if any(lab not in PENDING and lab not in SETTLED for lab in labels):
        return "unknown", "unrecognized_result", unit
    settled = {SETTLED[lab] for lab in labels if lab in SETTLED}
    if len(settled) > 1:
        return "unknown", "conflicting_results", unit
    if not settled:
        return "pending", "no_settled_observation", unit
    value = next(iter(settled))
    return value, "void" if value == "void" else "settled", unit


def _display_name(names: pd.Series) -> str | None:
    vals = names.dropna().astype(str)
    if vals.empty:
        return None
    counts = vals.value_counts()
    return sorted(counts[counts == counts.max()].index)[0]


def consensus_table(latest: pd.DataFrame, *, users: set[str] | None, cohort: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per (date, slot) with any valid vote in this cohort, plus the frozen vote table (counts only)."""
    vote_cols = ["cohort", "date", "pick_number", "batter_id", "n_users", "display_name"]
    if latest.empty:
        return pd.DataFrame(columns=CONSENSUS_COLUMNS), pd.DataFrame(columns=vote_cols)
    frame = latest if users is None else latest[latest["username"].isin(users)]
    frame = frame[frame["batter_id"].notna()].copy()
    frame["batter_id"] = frame["batter_id"].astype("int64")
    frame = frame[frame["batter_id"] > 0]
    vote_rows, rows = [], []
    for date, day in frame.groupby("pick_date", sort=True):
        counts, denom, names = {}, {}, {}
        for slot in PICK_NUMBERS:
            s = day[day["pick_number"] == slot]
            counts[slot] = {int(b): int(g["username"].nunique()) for b, g in s.groupby("batter_id")}
            denom[slot] = int(s["username"].nunique())
            names[slot] = {int(b): _display_name(g["batter_name"]) for b, g in s.groupby("batter_id")}
            vote_rows += [{"cohort": cohort, "date": date, "pick_number": slot, "batter_id": b, "n_users": n,
                           "display_name": names[slot][b]} for b, n in sorted(counts[slot].items())]
        choice = choose_slots(counts[1], counts[2])
        for slot, pick in ((1, choice["primary"]), (2, choice["dd"])):
            if denom[slot] == 0:                 # no valid vote in this slot: the unit builder marks it absent
                continue
            row = {"cohort": cohort, "date": date, "pick_number": slot, "consensus_status": pick["status"],
                   "consensus_available": pick["batter_id"] is not None, "n_public_users": denom[slot],
                   "consensus_batter_id": pick["batter_id"], "consensus_pick_count": pick["count"],
                   "consensus_pick_share": pick["count"] / denom[slot] if pick["batter_id"] is not None else None,
                   "consensus_batter_name": names[slot].get(pick["batter_id"]),
                   "consensus_tie_direct": pick["tie_direct"],
                   "consensus_tie_exposed_by_exclusion": pick.get("tie_exposed_by_exclusion", False),
                   "consensus_tie_dependent": pick.get("tie_dependent_on_primary", False),
                   "consensus_unconstrained_batter_id": pick.get("unconstrained_batter_id") if slot == 2 else pick["batter_id"],
                   "consensus_legal_differs_from_unconstrained": pick.get("legal_differs_from_unconstrained", False)}
            if pick["batter_id"] is not None:
                voters = day[(day["pick_number"] == slot) & (day["batter_id"] == pick["batter_id"])]
                status, reason, unit = consensus_settlement(list(voters["result"]), list(voters["unit_id"]))
            else:
                status, reason, unit = "unavailable", "no_consensus_id", None
            row.update(consensus_settlement=status, consensus_settlement_reason=reason, consensus_unit_id=unit)
            rows.append(row)
    table = pd.DataFrame(rows, columns=CONSENSUS_COLUMNS)
    if not table.empty:
        table["consensus_tie_broken"] = table["consensus_tie_direct"] | table["consensus_tie_dependent"]
        table["consensus_tie_reason"] = [
            "direct_and_dependent" if d and p else "direct" if d else "dependent_on_primary_tie" if p else None
            for d, p in zip(table["consensus_tie_direct"], table["consensus_tie_dependent"])]
        table["consensus_resolved"] = table["consensus_settlement"].isin(["hit", "not_hit"])
        table["consensus_hit"] = table["consensus_settlement"].map({"hit": 1, "not_hit": 0}).astype("Int64")
        table["consensus_batter_id"] = table["consensus_batter_id"].astype("Int64")
        table["consensus_unconstrained_batter_id"] = table["consensus_unconstrained_batter_id"].astype("Int64")
        table["consensus_unit_id"] = table["consensus_unit_id"].astype("Int64")
    votes = pd.DataFrame(vote_rows, columns=vote_cols)
    return table, votes


CONSENSUS_COLUMNS = [
    "cohort", "date", "pick_number", "consensus_status", "consensus_available", "n_public_users",
    "consensus_batter_id", "consensus_pick_count", "consensus_pick_share", "consensus_batter_name",
    "consensus_tie_direct", "consensus_tie_exposed_by_exclusion", "consensus_tie_dependent",
    "consensus_unconstrained_batter_id", "consensus_legal_differs_from_unconstrained", "consensus_settlement",
    "consensus_settlement_reason", "consensus_unit_id", "consensus_tie_broken", "consensus_tie_reason",
    "consensus_resolved", "consensus_hit"]
