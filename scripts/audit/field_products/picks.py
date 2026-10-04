"""The pick data contract (design "Data contract", r1 edit B-E2; code review r1 F4, F7 and conformance notes).

Observations are read as appended (never through ``latest_per_pick_date``, which keeps one row per date and drops a
DD leg), from the exact bytes supplied (``read``), and normalised at the boundary: tz-aware capture times become naive
UTC (the stored schema), pandas nullable integers/``pd.NA`` become plain values, and a null in a non-null pick-schema
column is refused. The analytical identity is (user_id, round_id, pick_number); unit/player identity and provenance
are kept, and every resolved slot carries its representative observation's source, file, file rank, file row and
capture time. Declared revision order: ``captured_at``, then file rank (files sorted by name), then file row.

A round's snapshot is its rows at the latest capture that holds it. Rows of that instant from different files are
separate batches: identical batches are one snapshot (duplicates counted); differing ones are
``competing_equal_time_batches`` — never unioned into one round. Individual slot grades stay usable unless the same
slot conflicts. Round COMPLETENESS (the DD and streak denominators) needs a witness that the snapshot batch is one
whole round response: ``witness[(user, round)] = {file, captured_at, slot_numbers}`` from a verified raw response
(``raw_profile_witness``; only the final grab retains one). Without it a round is ``no_completeness_witness``; a
pending leg that vanished is ``leg_disappeared_unwitnessed``; a settled leg that vanished is ``settled_leg_deleted``
whatever the witness. A round a later capture of the same user no longer shows is kept and flagged
``dropped_from_later_capture`` (a newer omission never erases the older observation, season_ledger
contest.slot_history); the flag establishes neither deletion nor complete follow-up. Only the exact labels ``hit`` /
``not_hit`` are graded; a settlement is never inferred from at_bats/hits, and a later unsettled label is never
replaced by an older settled one.

Composition (team, home/away): stored context comes from capture-time lookups, which are not historical pick-time
evidence, and no independent historical witness exists, so composition is unknown for every slot; equal-time context
disagreements are counted, never resolved by picking a row."""
from __future__ import annotations

import gzip
import io
import json
from collections import namedtuple
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import pandas as pd

GRADED = frozenset({"hit", "not_hit"})
# Labels that settle a slot. "void" is the contest's Pass/void slot label (season_ledger/outcomes.py
# CONTEST_NORMALIZATION); it settles the slot SET of a round but is never a graded slot.
TERMINAL = frozenset({"hit", "not_hit", "void"})
ORDER = ["user_id", "captured_at", "file_rank", "file_row"]
REQUIRED_INT = ["round_id", "pick_number", "unit_id", "bts_player_id", "at_bats", "hits", "streak_after"]
TEXT = ["result", "batter_name", "batter_team", "opponent_team", "home_or_away"]
CONTEXT = ["batter_team", "home_or_away", "opponent_team"]
INCOMPLETE_PRECEDENCE = ("equal_time_conflict", "competing_equal_time_batches", "identity_unresolved",
                         "pick_date_conflict", "missing_primary_slot", "settled_leg_deleted", "unsettled_slot",
                         "leg_disappeared_unwitnessed", "witness_mismatch", "no_completeness_witness")


@dataclass
class Resolution:
    slots: pd.DataFrame
    rounds: pd.DataFrame
    log: dict = field(default_factory=dict)


def _missing(v) -> bool:
    return v is None or (not isinstance(v, str) and pd.isna(v))


def with_provenance(df: pd.DataFrame, *, user_id: int, source: str, file: str, file_rank: int) -> pd.DataFrame:
    """Normalise one file's rows at the input boundary and attach provenance."""
    out = df.reset_index(drop=True).copy()
    for c in REQUIRED_INT:
        if out[c].isna().any():
            raise ValueError(f"{file}: null in non-null pick-schema column {c}")
        out[c] = [int(v) for v in out[c]]
    if out["pick_date"].isna().any():
        raise ValueError(f"{file}: null in non-null pick-schema column pick_date")
    out["pick_date"] = list(pd.to_datetime(out["pick_date"]).dt.date)
    out["batter_id"] = pd.Series([None if _missing(v) else int(v) for v in out["batter_id"]], dtype=object)
    for c in TEXT:
        out[c] = pd.Series([None if _missing(v) else str(v) for v in out[c]], dtype=object)
    cap = pd.to_datetime(out["captured_at"])
    if cap.isna().any():
        raise ValueError(f"{file}: null in non-null pick-schema column captured_at")
    if cap.dt.tz is not None:
        cap = cap.dt.tz_convert("UTC").dt.tz_localize(None)
    out["captured_at"] = cap
    out.insert(0, "file_row", range(len(out)))
    out.insert(0, "file_rank", file_rank)
    out.insert(0, "file", file)
    out.insert(0, "source", source)
    out.insert(0, "user_id", int(user_id))
    return out


def read_observations(paths: list[Path], *, user_id: int, source: str, read=None) -> pd.DataFrame:
    """Every appended row of ``paths`` (one user's files), parsed from ``read(path)`` (the frozen bytes; default: the
    file on disk), with provenance. No dedup of any kind."""
    read = read or (lambda p: Path(p).read_bytes())
    parts = [with_provenance(pd.read_parquet(io.BytesIO(read(p))), user_id=user_id, source=source, file=Path(p).name,
                             file_rank=rank) for rank, p in enumerate(sorted(paths, key=lambda q: Path(q).name))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


def raw_profile_witness(raw: bytes, *, user_id: int, file: str, captured_at) -> dict:
    """Round completeness witnesses from one verified raw profile response (gzipped or plain JSON): for every round
    whose roundPredictions carry distinct slot numbers in {1, 2}, the slot-number set the response held, bound to the
    parsed file and capture instant of that response. Repeated rounds or slot numbers witness nothing."""
    try:
        raw = gzip.decompress(raw)
    except OSError:
        pass
    preds = ((json.loads(raw) or {}).get("success") or {}).get("predictions") or []
    seen: dict[int, list] = {}
    for p in preds:
        if isinstance(p, dict) and isinstance(p.get("roundId"), int):
            seen.setdefault(p["roundId"], []).append(p.get("roundPredictions") or [])
    out = {}
    for rid, lists in seen.items():
        if len(lists) != 1:
            continue
        nums = [rp.get("number") if isinstance(rp, dict) else None for rp in lists[0]]
        if nums and all(isinstance(n, int) and not isinstance(n, bool) and n in (1, 2) for n in nums) \
                and len(set(nums)) == len(nums):
            out[(int(user_id), int(rid))] = {"file": file, "captured_at": pd.Timestamp(captured_at),
                                             "slot_numbers": frozenset(nums)}
    return out


REC_COLS = ["user_id", "round_id", "pick_number", "captured_at", "pick_date", "unit_id", "bts_player_id", "result",
            "at_bats", "hits", "streak_after", "batter_id", "source", "file", "file_rank", "file_row"]
Obs = namedtuple("Obs", REC_COLS)


def _identity(r) -> tuple:
    return int(r.unit_id), int(r.bts_player_id)


def _label(v) -> str:
    return "" if _missing(v) else str(v)


def _content(o: Obs) -> tuple:
    """Identity and outcome fields compared for equal-time conflicts (lookup enrichment excluded)."""
    return (str(o.pick_date), int(o.unit_id), int(o.bts_player_id), _label(o.result), int(o.at_bats), int(o.hits),
            int(o.streak_after))


def _slot(user_id: int, round_id: int, pn: int, hist: list[Obs], snap: list[Obs]) -> dict:
    """One snapshot slot: the declared-last observation at the snapshot instant plus its revision history."""
    rep = snap[-1]
    status = "ok"
    if len({_content(o) for o in snap}) > 1:
        status = "equal_time_conflict"
    elif int(rep.unit_id) == 0 or int(rep.bts_player_id) == 0:
        status = "identity_unresolved"
    label = _label(rep.result) if status != "equal_time_conflict" else None
    snap_ts = rep.captured_at
    earlier_labels = [_label(o.result) for o in hist if o.captured_at < snap_ts]
    first_settled: dict[tuple, object] = {}          # identity -> its first settled capture
    for o in hist:
        if _label(o.result) in TERMINAL:
            first_settled.setdefault(_identity(o), o.captured_at)
    settled_ident_changed = any(t < o.captured_at and ident != _identity(o)
                                for o in hist for ident, t in first_settled.items())
    return {
        "user_id": user_id, "round_id": round_id, "pick_number": pn, "pick_date": rep.pick_date,
        "unit_id": int(rep.unit_id), "bts_player_id": int(rep.bts_player_id), "batter_id": rep.batter_id,
        "label": label, "status": status, "graded": label in GRADED, "usable": status == "ok" and label in GRADED,
        "hit": status == "ok" and label == "hit", "streak_after": int(rep.streak_after),
        "at_bats": int(rep.at_bats), "hits": int(rep.hits),
        "n_obs": len(hist), "first_captured_at": min(o.captured_at for o in hist), "captured_at": snap_ts,
        "identity_changed": len({_identity(o) for o in hist}) > 1,
        "settled_identity_changed": settled_ident_changed,
        "later_unsettled_over_settled": label is not None and label not in GRADED
        and any(x in GRADED for x in earlier_labels),
        "settled_label_changed": label in GRADED and any(x in GRADED and x != label for x in earlier_labels),
        "rep_source": rep.source, "rep_file": rep.file, "rep_file_rank": int(rep.file_rank),
        "rep_file_row": int(rep.file_row), "rep_captured_at": rep.captured_at,
        "files": sorted({o.file for o in hist}),
    }


def _round(user_id: int, round_id: int, g: list[Obs], log: dict, last_capture, witness: dict) -> tuple[dict, list]:
    snap_ts = max(o.captured_at for o in g)
    groups: dict[tuple, list[Obs]] = {}
    for o in g:
        groups.setdefault((int(o.pick_number), o.captured_at), []).append(o)
    for (_pn, ts), b in groups.items():
        n_distinct = len({_content(o) for o in b})
        if n_distinct == 1:
            log["exact_duplicate_rows"] += len(b) - 1
        elif ts != snap_ts:
            log["equal_time_conflicts_superseded"] += 1
    snap = [o for o in g if o.captured_at == snap_ts]
    by_file: dict[str, set] = {}
    for o in snap:
        by_file.setdefault(o.file, set()).add((int(o.pick_number), _content(o)))
    competing = len({frozenset(v) for v in by_file.values()}) > 1
    numbers = sorted({int(o.pick_number) for o in snap})
    slots = [_slot(user_id, round_id, pn, [o for o in g if int(o.pick_number) == pn],
                   [o for o in snap if int(o.pick_number) == pn]) for pn in numbers]
    earlier = [o for o in g if o.captured_at < snap_ts]
    deleted = {int(o.pick_number) for o in earlier} - set(numbers)
    deleted_settled = any(_label(o.result) in TERMINAL for o in earlier if int(o.pick_number) in deleted)
    w = witness.get((user_id, round_id))
    witnessed = (w is not None and pd.Timestamp(w["captured_at"]) == pd.Timestamp(snap_ts) and w["file"] in by_file
                 and frozenset(w["slot_numbers"]) == frozenset(numbers))
    reasons = set()
    if any(s["status"] == "equal_time_conflict" for s in slots):
        reasons.add("equal_time_conflict")
    if competing:
        reasons.add("competing_equal_time_batches")
    if any(s["status"] == "identity_unresolved" for s in slots):
        reasons.add("identity_unresolved")
    if len({str(o.pick_date) for o in snap}) > 1:
        reasons.add("pick_date_conflict")
    if 1 not in numbers:
        reasons.add("missing_primary_slot")
    if deleted_settled:
        reasons.add("settled_leg_deleted")
    if any(s["label"] not in TERMINAL for s in slots):
        reasons.add("unsettled_slot")
    if deleted and not witnessed:
        reasons.add("leg_disappeared_unwitnessed")
    if w is None:
        reasons.add("no_completeness_witness")
    elif not witnessed:
        reasons.add("witness_mismatch")
    reason = next((r for r in INCOMPLETE_PRECEDENCE if r in reasons), None)
    ok_streaks = {s["streak_after"] for s in slots if s["status"] == "ok"}
    rnd = {"user_id": user_id, "round_id": round_id, "pick_date": snap[-1].pick_date,
           "captured_at": snap_ts, "snapshot_files": sorted(by_file), "n_slots": len(slots),
           "slot_numbers": ",".join(str(n) for n in numbers), "labels": ",".join(str(s["label"]) for s in slots),
           "streak_after": ok_streaks.pop() if len(ok_streaks) == 1 else None,
           "streak_conflict": len({s["streak_after"] for s in slots}) > 1,
           "complete": reason is None, "incomplete_reason": reason, "is_dd": reason is None and len(slots) == 2,
           "witnessed": witnessed, "competing_batches": competing,
           "deleted_legs": len(deleted), "settled_leg_deleted": deleted_settled,
           "dropped_from_later_capture": last_capture is not None and snap_ts < last_capture,
           "slots_ok": all(s["status"] == "ok" for s in slots),
           "all_hit": all(s["label"] == "hit" for s in slots),
           "any_not_hit": any(s["label"] == "not_hit" for s in slots),
           "all_graded": all(s["label"] in GRADED for s in slots)}
    return rnd, slots


def resolve(obs: pd.DataFrame, witness: dict | None = None) -> Resolution:
    """Resolve appended observations (any number of users, one source each) into snapshot slots and rounds;
    ``witness`` holds verified whole-round response witnesses (see the module docstring)."""
    witness = witness or {}
    log = {"n_observations": int(len(obs)), "exact_duplicate_rows": 0, "equal_time_conflicts_superseded": 0}
    slot_rows: list[dict] = []
    round_rows: list[dict] = []
    if len(obs):
        ordered = obs.sort_values(ORDER, kind="mergesort")
        groups: dict[tuple, list[Obs]] = {}
        for rec in zip(*(ordered[c].tolist() for c in REC_COLS)):
            o = Obs(*rec)
            groups.setdefault((int(o.user_id), int(o.round_id)), []).append(o)
        last_capture: dict[int, object] = {}
        for (uid, _rid), g in groups.items():
            last_capture[uid] = max(last_capture.get(uid, g[-1].captured_at), g[-1].captured_at)
        for (uid, rid) in sorted(groups):
            rnd, slots = _round(uid, rid, groups[(uid, rid)], log, last_capture[uid], witness)
            round_rows.append(rnd)
            slot_rows += slots
        log["n_users"] = int(obs["user_id"].nunique())
        log["n_captures"] = int(obs[["user_id", "captured_at"]].drop_duplicates().shape[0])
    slots, rounds = pd.DataFrame(slot_rows), pd.DataFrame(round_rows)
    if len(slots):        # keep int/None (pandas would turn a None/int column into float NaN)
        slots["batter_id"] = pd.Series([r["batter_id"] for r in slot_rows], dtype=object)
    ok = slots[slots["status"] == "ok"] if len(slots) else slots
    log.update({
        "n_slots": int(len(slots)), "n_rounds": int(len(rounds)),
        "changed_identity_slots": int(slots["identity_changed"].sum()) if len(slots) else 0,
        "settled_identity_changed": int(slots["settled_identity_changed"].sum()) if len(slots) else 0,
        "equal_time_conflicts": int((slots["status"] == "equal_time_conflict").sum()) if len(slots) else 0,
        "later_unsettled_over_settled": int(slots["later_unsettled_over_settled"].sum()) if len(slots) else 0,
        "settled_label_changed": int(slots["settled_label_changed"].sum()) if len(slots) else 0,
        "deleted_legs": int(rounds["deleted_legs"].sum()) if len(rounds) else 0,
        "settled_legs_deleted_rounds": int(rounds["settled_leg_deleted"].sum()) if len(rounds) else 0,
        "competing_batch_rounds": int(rounds["competing_batches"].sum()) if len(rounds) else 0,
        "witnessed_rounds": int(rounds["witnessed"].sum()) if len(rounds) else 0,
        "rounds_dropped_from_later_capture": int(rounds["dropped_from_later_capture"].sum()) if len(rounds) else 0,
        "label_counts": ok["label"].value_counts().to_dict() if len(ok) else {},
        "status_counts": slots["status"].value_counts().to_dict() if len(slots) else {},
        "incomplete_reasons": rounds["incomplete_reason"].value_counts().to_dict() if len(rounds) else {},
    })
    return Resolution(slots, rounds, log)


def ownership_conflict_users(res: Resolution) -> set[int]:
    """Users whose observations show a settled slot changing identity or a settled leg vanishing: impossible for one
    account after settlement, so the observations cannot be attributed to a single owner (quarantined)."""
    out: set[int] = set()
    if len(res.slots):
        out |= {int(u) for u in res.slots.loc[res.slots["settled_identity_changed"], "user_id"]}
    if len(res.rounds):
        out |= {int(u) for u in res.rounds.loc[res.rounds["settled_leg_deleted"], "user_id"]}
    return out


def _in_window(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if not len(df):
        return df
    d = pd.to_datetime(df["pick_date"]).dt.date
    return df[(d >= start) & (d <= end)]


def dd_summary(rounds: pd.DataFrame, start: date, end: date) -> dict[int, dict]:
    """Per user within [start, end]: observed pick days, complete (witnessed) and DD rounds, DD frequency (complete
    two-slot rounds / complete rounds), incomplete rounds by reason, and unobserved calendar dates (never skips)."""
    out: dict[int, dict] = {}
    cal = (end - start).days + 1
    for uid, g in _in_window(rounds, start, end).groupby("user_id"):
        comp = g[g["complete"]]
        days = int(pd.to_datetime(g["pick_date"]).dt.date.nunique())
        out[int(uid)] = {"rounds": int(len(g)), "pick_days": days, "complete_rounds": int(len(comp)),
                         "dd_rounds": int(comp["is_dd"].sum()), "single_rounds": int((comp["n_slots"] == 1).sum()),
                         "dd_frequency": float(comp["is_dd"].mean()) if len(comp) else None,
                         "incomplete_rounds": int((~g["complete"]).sum()),
                         "incomplete_reasons": g.loc[~g["complete"], "incomplete_reason"].value_counts().to_dict(),
                         "deleted_legs": int(g["deleted_legs"].sum()),
                         "dropped_from_later_capture": int(g["dropped_from_later_capture"].sum()),
                         "calendar_dates": cal, "unobserved_calendar_dates": cal - days}
    return out


def graded_summary(slots: pd.DataFrame, start: date, end: date) -> dict[int, dict]:
    """Per user within [start, end]: usable graded slots (exact hit/not_hit, status ok), hits, rate, graded dates and
    rounds, and every exclusion counted by label or status."""
    out: dict[int, dict] = {}
    for uid, g in _in_window(slots, start, end).groupby("user_id"):
        use = g[g["usable"]]
        bad_status = g[g["status"] != "ok"]
        bad_label = g[(g["status"] == "ok") & ~g["graded"]]
        out[int(uid)] = {"slots": int(len(g)), "graded_slots": int(len(use)), "hits": int(use["hit"].sum()),
                         "hit_rate": float(use["hit"].mean()) if len(use) else None,
                         "graded_dates": int(pd.to_datetime(use["pick_date"]).dt.date.nunique()),
                         "graded_rounds": int(use["round_id"].nunique()),
                         "excluded_by_label": bad_label["label"].value_counts().to_dict(),
                         "excluded_by_status": bad_status["status"].value_counts().to_dict()}
    return out


def composition(obs: pd.DataFrame, slots: pd.DataFrame) -> dict[int, dict]:
    """Home/away and team concentration need an independent historical pick-time context witness; none is stored
    (the pick rows' context comes from capture-time lookups), so every slot is unknown. Equal-time disagreements of
    the stored context for a slot's representative capture are counted (never resolved by choosing a row)."""
    out: dict[int, dict] = {}
    if not len(slots):
        return out
    key = ["user_id", "round_id", "pick_number", "captured_at"]
    ctx = obs[key + CONTEXT].copy()
    for c in CONTEXT:
        ctx[c] = ctx[c].map(lambda v: None if _missing(v) else str(v)).astype(object).fillna("<null>")
    n_ctx = ctx.drop_duplicates().groupby(key).size()
    for uid, g in slots[slots["status"] == "ok"].groupby("user_id"):
        conflicts = sum(1 for s in g.itertuples()
                        if n_ctx.get((s.user_id, s.round_id, s.pick_number, s.captured_at), 1) > 1)
        out[int(uid)] = {"slots": int(len(g)), "witnessed": 0, "unknown": int(len(g)), "home": 0, "away": 0,
                         "distinct_teams": 0, "top_team_share": None, "context_conflicts": int(conflicts),
                         "basis": "no_independent_historical_context_witness"}
    return out
