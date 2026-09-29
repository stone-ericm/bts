"""Closed outcome vocabulary and slot-level comparison (spec §7). HOLD = the slot neither extended
nor broke the streak (a Pass or a voided slot); round labels (`void`, `used_mulligan`, …) are not
part of this table."""
from __future__ import annotations

LOCAL_NORMALIZATION = {"hit": "HIT", "miss": "NO_HIT", "void": "HOLD"}
CONTEST_NORMALIZATION = {"hit": "HIT", "not_hit": "NO_HIT", "void": "HOLD"}
COMPARABLE = frozenset({"HIT", "NO_HIT", "HOLD"})


def normalize_local(raw: str | None) -> str | None:
    return None if raw is None else LOCAL_NORMALIZATION.get(raw, "UNKNOWN")


def normalize_contest(raw: str | None) -> str | None:
    return None if raw is None else CONTEST_NORMALIZATION.get(raw, "UNKNOWN")


def slot_disagreement(local_norm: str | None, contest_norm: str | None) -> bool | None:
    if local_norm not in COMPARABLE or contest_norm not in COMPARABLE:
        return None
    return local_norm != contest_norm


def derived_single_result(pick_row: dict | None) -> tuple[str | None, str | None]:
    """A per-slot value from the day result only for a record that is itself a single pick and has
    no per-slot result, bound to that record (spec §7)."""
    if pick_row is None or pick_row["has_double_down"] or pick_row["slot_result_raw"] is not None:
        return None, None
    if pick_row["day_result_raw"] in LOCAL_NORMALIZATION:
        return pick_row["day_result_raw"], f"day_result_of_single_pick:{pick_row['obs_id']}"
    return None, None
