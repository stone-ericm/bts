"""The pick data contract (design "Data contract", r1 edit B-E2).

Observations are read as appended (never through ``latest_per_pick_date``, which keeps one row per date and drops a
DD leg). The analytical identity is (user_id, round_id, pick_number); unit/player identity and provenance (source,
file, file row, captured_at) are kept. Declared revision order: ``captured_at``, then file rank (files sorted by name),
then the row's position in its file. A round's snapshot is its rows in the latest capture that holds the round;
older legs absent from that snapshot are not retained. A round that a later capture of the same user no longer shows
is kept (a newer omission never erases the older observation, as in season_ledger contest.slot_history) and flagged
``dropped_from_later_capture``. Only the exact labels ``hit`` / ``not_hit`` are graded; a
settlement is never inferred from at_bats/hits, and a later unsettled label is never replaced by an older settled one.
"""
from __future__ import annotations

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
ET = "America/New_York"
INCOMPLETE_PRECEDENCE = ("equal_time_conflict", "identity_unresolved", "pick_date_conflict", "missing_primary_slot",
                         "settled_leg_deleted", "unsettled_slot")


@dataclass
class Resolution:
    slots: pd.DataFrame
    rounds: pd.DataFrame
    log: dict = field(default_factory=dict)


def with_provenance(df: pd.DataFrame, *, user_id: int, source: str, file: str, file_rank: int) -> pd.DataFrame:
    out = df.reset_index(drop=True).copy()
    out.insert(0, "file_row", range(len(out)))
    out.insert(0, "file_rank", file_rank)
    out.insert(0, "file", file)
    out.insert(0, "source", source)
    out.insert(0, "user_id", int(user_id))
    out["captured_at"] = pd.to_datetime(out["captured_at"])
    return out


def read_observations(paths: list[Path], *, user_id: int, source: str) -> pd.DataFrame:
    """Every appended row of ``paths`` (one user's files), with provenance. No dedup of any kind."""
    parts = [with_provenance(pd.read_parquet(p), user_id=user_id, source=source, file=p.name, file_rank=rank)
             for rank, p in enumerate(sorted(paths, key=lambda q: q.name))]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


REC_COLS = ["user_id", "round_id", "pick_number", "captured_at", "pick_date", "unit_id", "bts_player_id", "result",
            "at_bats", "hits", "streak_after", "batter_id", "source", "file"]
Obs = namedtuple("Obs", REC_COLS)


def _identity(r) -> tuple:
    return int(r.unit_id), int(r.bts_player_id)


def _label(v) -> str:
    return "" if v is None or (isinstance(v, float) and pd.isna(v)) else str(v)


def _content(o: Obs) -> tuple:
    """Identity and outcome fields compared for equal-time conflicts (lookup enrichment excluded)."""
    return (str(o.pick_date), int(o.unit_id), int(o.bts_player_id), _label(o.result), int(o.at_bats), int(o.hits),
            int(o.streak_after))


def _int_or_none(v):
    return None if v is None or (isinstance(v, float) and pd.isna(v)) else int(v)


def _slot(user_id: int, round_id: int, pn: int, hist: list[Obs], snap: list[Obs]) -> dict:
    """One snapshot slot: the latest observation plus its revision history over every capture (``hist`` is in
    declared order; ``snap`` its rows at the round's snapshot capture)."""
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
        "unit_id": int(rep.unit_id), "bts_player_id": int(rep.bts_player_id), "batter_id": _int_or_none(rep.batter_id),
        "label": label, "status": status, "graded": label in GRADED, "usable": status == "ok" and label in GRADED,
        "hit": status == "ok" and label == "hit", "streak_after": int(rep.streak_after),
        "at_bats": int(rep.at_bats), "hits": int(rep.hits),
        "n_obs": len(hist), "first_captured_at": min(o.captured_at for o in hist), "captured_at": snap_ts,
        "identity_changed": len({_identity(o) for o in hist}) > 1,
        "settled_identity_changed": settled_ident_changed,
        "later_unsettled_over_settled": label is not None and label not in GRADED
        and any(x in GRADED for x in earlier_labels),
        "settled_label_changed": label in GRADED and any(x in GRADED and x != label for x in earlier_labels),
        "source": ";".join(sorted({o.source for o in hist})), "files": ";".join(sorted({o.file for o in hist})),
    }


def _round(user_id: int, round_id: int, g: list[Obs], log: dict, last_capture=None) -> tuple[dict, list[dict]]:
    snap_ts = max(o.captured_at for o in g)
    batches: dict[tuple, list[Obs]] = {}
    for o in g:
        batches.setdefault((int(o.pick_number), o.captured_at), []).append(o)
    for (_pn, ts), b in batches.items():
        n_distinct = len({_content(o) for o in b})
        if n_distinct == 1:
            log["exact_duplicate_rows"] += len(b) - 1
        elif ts != snap_ts:
            log["equal_time_conflicts_superseded"] += 1
    snap = [o for o in g if o.captured_at == snap_ts]
    numbers = sorted({int(o.pick_number) for o in snap})
    slots = [_slot(user_id, round_id, pn, [o for o in g if int(o.pick_number) == pn], batches[(pn, snap_ts)])
             for pn in numbers]
    earlier = [o for o in g if o.captured_at < snap_ts]
    deleted = {int(o.pick_number) for o in earlier} - set(numbers)
    deleted_settled = any(_label(o.result) in TERMINAL for o in earlier if int(o.pick_number) in deleted)
    reasons = set()
    if any(s["status"] == "equal_time_conflict" for s in slots):
        reasons.add("equal_time_conflict")
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
    reason = next((r for r in INCOMPLETE_PRECEDENCE if r in reasons), None)
    ok_streaks = {s["streak_after"] for s in slots if s["status"] == "ok"}
    rnd = {"user_id": user_id, "round_id": round_id, "pick_date": snap[-1].pick_date,
           "captured_at": snap_ts, "n_slots": len(slots), "slot_numbers": ",".join(str(n) for n in numbers),
           "labels": ",".join(str(s["label"]) for s in slots), "streak_after": ok_streaks.pop() if len(ok_streaks) == 1
           else None, "streak_conflict": len({s["streak_after"] for s in slots}) > 1,
           "complete": reason is None, "incomplete_reason": reason, "is_dd": reason is None and len(slots) == 2,
           "deleted_legs": len(deleted), "settled_leg_deleted": deleted_settled,
           "dropped_from_later_capture": last_capture is not None and snap_ts < last_capture,
           "all_hit": reason is None and all(s["label"] == "hit" for s in slots),
           "any_not_hit": any(s["label"] == "not_hit" for s in slots),
           "all_graded": all(s["label"] in GRADED for s in slots)}
    return rnd, slots


def resolve(obs: pd.DataFrame) -> Resolution:
    """Resolve appended observations (any number of users, one source each) into snapshot slots and rounds."""
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
            rnd, slots = _round(uid, rid, groups[(uid, rid)], log, last_capture[uid])
            round_rows.append(rnd)
            slot_rows += slots
        log["n_users"] = int(obs["user_id"].nunique())
        log["n_captures"] = int(obs[["user_id", "captured_at"]].drop_duplicates().shape[0])
    slots, rounds = pd.DataFrame(slot_rows), pd.DataFrame(round_rows)
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
    """Per user within [start, end]: observed pick days, complete/DD rounds, DD frequency (complete two-slot rounds /
    complete rounds), incomplete rounds by reason, and unobserved calendar dates (never called skips)."""
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
    every exclusion counted by label or status."""
    out: dict[int, dict] = {}
    for uid, g in _in_window(slots, start, end).groupby("user_id"):
        use = g[g["usable"]]
        bad_status = g[g["status"] != "ok"]
        bad_label = g[(g["status"] == "ok") & ~g["graded"]]
        out[int(uid)] = {"slots": int(len(g)), "graded_slots": int(len(use)), "hits": int(use["hit"].sum()),
                         "hit_rate": float(use["hit"].mean()) if len(use) else None,
                         "graded_dates": int(pd.to_datetime(use["pick_date"]).dt.date.nunique()),
                         "excluded_by_label": bad_label["label"].value_counts().to_dict(),
                         "excluded_by_status": bad_status["status"].value_counts().to_dict()}
    return out


def composition(obs: pd.DataFrame, slots: pd.DataFrame) -> dict[int, dict]:
    """Home/away and team concentration on witnessed rows only: an observation of the resolved slot identity captured
    on the pick's own New York date with non-null home/away and team (the lookups were contemporaneous). Later
    captures fill team from the CURRENT player record, which is not historical evidence (B-E4)."""
    o = obs.copy()
    o["cap_et_date"] = o["captured_at"].dt.tz_localize("UTC").dt.tz_convert(ET).dt.date
    o = o[(o["cap_et_date"] == pd.to_datetime(o["pick_date"]).dt.date) & o["home_or_away"].notna()
          & o["batter_team"].notna()].sort_values(ORDER, kind="mergesort")
    key = ["user_id", "round_id", "pick_number", "unit_id", "bts_player_id"]
    latest = o.drop_duplicates(key, keep="last").set_index(key) if len(o) else None
    out: dict[int, dict] = {}
    for uid, g in slots[slots["status"] == "ok"].groupby("user_id"):
        wit = []
        for s in g.itertuples():
            k = (s.user_id, s.round_id, s.pick_number, s.unit_id, s.bts_player_id)
            if latest is not None and k in latest.index:
                wit.append(latest.loc[k])
        teams = pd.Series([w["batter_team"] for w in wit], dtype=object)
        ha = pd.Series([w["home_or_away"] for w in wit], dtype=object)
        out[int(uid)] = {"slots": int(len(g)), "witnessed": len(wit), "unknown": int(len(g) - len(wit)),
                         "home": int((ha == "home").sum()), "away": int((ha == "away").sum()),
                         "distinct_teams": int(teams.nunique()),
                         "top_team_share": float(teams.value_counts().iloc[0] / len(wit)) if wit else None}
    return out
