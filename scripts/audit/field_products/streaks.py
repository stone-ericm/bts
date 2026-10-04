"""Streak/run rules (design W2.1 item 2, r1 edit B-E3; code review r1 F3).

Pick rows carry the SEASONAL reported ``streak_after`` (repeated on DD legs, null-coerced to 0 by the parser) and no
round result, streakIncrease or saver state; no stored record witnesses that the retained rounds are a user's complete
entered-round history (a hidden round can always sit between two observations or before the first). So an exact
within-window maximum, and the start/end dates of a run, are UNAVAILABLE here; only defensible lower bounds are
reported, with their coverage.

Round kinds (from ``picks.resolve``; round completeness is not required): ``H`` every observed slot exactly ``hit``,
no slot conflict or competing batch, and a reported streak at least the slot count; ``M`` every observed slot exactly
graded with at least one ``not_hit`` (a single miss or a mixed DD); ``A`` anything else (pending, Pass/void or other
labels, conflicts, an all-hit round whose streak is null-coerced).

An OBSERVED SEGMENT is a run of retained rounds, each H, where every round's reported streak equals the previous
round's plus its own slot count. A segment never crosses an M or A round and never joins inconsistent values (no
saver or absorption is inferred from endpoints). Its slot sum is a lower bound on the in-window length of the run
reported at its last round, whatever the hidden history: each reported value is the contest's own count of hits since
the last reset, so a hidden reset between two linked rounds must be followed by a hidden rebuild at least as long as
the hits it cut off. The window bound uses in-window rounds only (carried-in streak excluded); ``max(streak_after)``
is never used."""
from __future__ import annotations

from collections import Counter
from datetime import date

import pandas as pd

NO_WITNESS = "no_entered_round_completeness_witness"
NO_WITNESS_TEXT = (f"{NO_WITNESS}: exact maxima and run dates need a complete entered-round history and supported "
                   "transitions; no stored record establishes either")


def annotate(rounds: pd.DataFrame) -> pd.DataFrame:
    """One user's resolved rounds in (pick_date, round_id) order with ``kind``, ``n``, ``streak``, ``segment``
    (H rounds only) and ``split`` (why an H round starts a new segment)."""
    r = rounds.sort_values(["pick_date", "round_id"], kind="mergesort").reset_index(drop=True).copy()
    streak = pd.to_numeric(r["streak_after"], errors="coerce")
    usable = r["slots_ok"] & ~r["competing_batches"] & (r["incomplete_reason"] != "pick_date_conflict")
    is_h = usable & r["all_hit"] & streak.notna() & (streak >= r["n_slots"])
    is_m = usable & r["any_not_hit"] & r["all_graded"]
    r["kind"] = ["H" if h else ("M" if m else "A") for h, m in zip(is_h, is_m)]
    r["n"] = r["n_slots"].astype(int)
    r["streak"] = streak
    segs, splits, seg = [None] * len(r), [None] * len(r), -1
    for i in range(len(r)):
        if r.at[i, "kind"] != "H":
            continue
        prev_kind = r.at[i - 1, "kind"] if i else None
        if prev_kind == "H" and int(r.at[i, "streak"]) == int(r.at[i - 1, "streak"]) + int(r.at[i, "n"]):
            segs[i] = seg
            continue
        seg += 1
        segs[i] = seg
        splits[i] = ("first" if prev_kind is None else "after_miss" if prev_kind == "M"
                     else "after_ambiguous" if prev_kind == "A" else "inconsistent_values")
    r["segment"], r["split"] = segs, splits
    return r


def _dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["pick_date"]).dt.date


def window_summary(rounds: pd.DataFrame, start: date, end: date) -> dict:
    """One user's [start, end]: the longest observed in-window segment as a lower bound on the longest streak built
    inside the window (exact value unavailable), with coverage and segment splits."""
    w = rounds[(_dates(rounds) >= start) & (_dates(rounds) <= end)] if len(rounds) else rounds
    out = {"rounds": int(len(w)), "incomplete_rounds": int((~w["complete"]).sum()) if len(w) else 0,
           "longest_exact": None, "reasons": [NO_WITNESS]}
    if not len(w):
        return {**out, "status": "no_window_rounds", "lower_bound": None, "kinds": {}, "segments": 0, "splits": {}}
    a = annotate(w)
    hs = a[a["kind"] == "H"]
    sums = hs.groupby("segment")["n"].sum()
    return {**out, "status": "lower_bound_only", "kinds": dict(Counter(a["kind"])),
            "lower_bound": int(sums.max()) if len(sums) else 0, "segments": int(len(sums)),
            "splits": dict(Counter(s for s in hs["split"] if s not in (None, "first")))}


def attaining_runs(rounds: pd.DataFrame, best: int | None) -> dict:
    """The board/profile season best ``best`` (kept separate from reconstruction): every H round reporting it, with
    the observed segment ending there as a lower bound. Run start/end dates are unavailable (no completeness or
    transition witness); the reported attainment date is the date of the round whose reported streak equals best."""
    a = annotate(rounds) if len(rounds) else rounds
    hs = a[a["kind"] == "H"] if len(a) else a
    sums = hs.groupby("segment")["n"].sum() if len(hs) else pd.Series(dtype=int)
    base = {"best": best, "rounds": int(len(a)), "kinds": dict(Counter(a["kind"])) if len(a) else {},
            "incomplete_rounds": int((~a["complete"]).sum()) if len(a) else 0,
            "longest_observed_segment": int(sums.max()) if len(sums) else None, "runs": [], "reason": NO_WITNESS_TEXT}
    if best is None:
        return {**base, "status": "best_unavailable"}
    if best == 0:
        return {**base, "status": "best_is_zero"}
    runs = []
    for e in hs[hs["streak"] == best].itertuples():
        seg = hs[(hs["segment"] == e.segment) & (hs.index <= e.Index)]
        runs.append({"recoverable": False, "start_date": None, "end_date": None,
                     "reported_attainment_date": str(e.pick_date),
                     "observed_segment_lower_bound": int(seg["n"].sum()), "n_rounds": int(len(seg)),
                     "n_dd_rounds": int((seg["n"] == 2).sum())})
    return {**base, "status": "dates_unavailable" if runs else "no_settled_round_reports_best", "runs": runs}
