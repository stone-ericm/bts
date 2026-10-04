"""Streak/run rules (design W2.1 item 2, r1 edit B-E3). Pick rows carry the SEASONAL reported ``streak_after``
(repeated on DD legs, null-coerced to 0 by the parser) and no round result, streakIncrease or saver state, so runs are
reconstructed only where the reported values themselves evidence each transition.

Round kinds (complete snapshots from ``picks.resolve``): ``H`` every slot exactly ``hit`` and a reported streak at least
the slot count; ``M`` every slot exactly graded with at least one ``not_hit`` (a single miss or a mixed DD); ``A``
anything else (incomplete, Pass/void or other labels, an all-hit round whose streak is unreported/null-coerced).

Each ``H`` round b is linked to the previous ``H`` round a (rounds in between are M/A, or unobserved):
- ``continuation*``: streak(b) == streak(a) + n(b) — no reset between (``_through_miss`` when an M lies between: the
  saver-consistent case; ``_absorbing`` when only A rounds lie between: they changed nothing);
- ``new_run_*``: streak(b) == n(b) — the streak before b was 0 (entry evidence). ``_after_miss`` when an observed M
  lies between; ``_unexplained`` when none does (the reset happened in unobserved rounds: missing data);
  ``_first`` for the first H round;
- ``carried_unknown`` (first H, start unknown) / ``unresolved`` (neither identity holds).
A miss's own reported 0 is never read as a reset (it may be a null-coerced value). ``max(streak_after)`` is never a
within-window run. Residual assumption: a continuation identity is not produced by an exactly compensating hidden
reset-and-rebuild in unobserved rounds."""
from __future__ import annotations

from collections import Counter
from datetime import date

import pandas as pd

LINK_CONT = frozenset({"continuation", "continuation_through_miss", "continuation_absorbing"})
LINK_NEW = frozenset({"new_run_first", "new_run_after_miss", "new_run_unexplained"})


def annotate(rounds: pd.DataFrame) -> pd.DataFrame:
    """One user's resolved rounds in (pick_date, round_id) order with ``kind``, ``n``, ``link``, ``chain`` (H rounds
    joined by continuation links) and ``absorbed`` (an A round spanned by a continuation link)."""
    r = rounds.sort_values(["pick_date", "round_id"], kind="mergesort").reset_index(drop=True).copy()
    streak = pd.to_numeric(r["streak_after"], errors="coerce")
    is_h = r["complete"] & r["all_hit"] & streak.notna() & (streak >= r["n_slots"])
    is_m = r["complete"] & r["any_not_hit"] & r["all_graded"]
    r["kind"] = ["H" if h else ("M" if m else "A") for h, m in zip(is_h, is_m)]
    r["n"] = r["n_slots"].astype(int)
    r["streak"] = streak
    links, chains = [None] * len(r), [None] * len(r)
    prev, between, chain_id = None, [], -1
    for i in range(len(r)):
        if r.at[i, "kind"] != "H":
            between.append(i)
            continue
        n, s = int(r.at[i, "n"]), int(r.at[i, "streak"])
        kinds_between = {r.at[j, "kind"] for j in between}
        if prev is not None and s == int(r.at[prev, "streak"]) + n:
            link = ("continuation_through_miss" if "M" in kinds_between
                    else "continuation_absorbing" if kinds_between else "continuation")
        elif s == n:
            link = ("new_run_first" if prev is None
                    else "new_run_after_miss" if "M" in kinds_between else "new_run_unexplained")
        else:
            link = "carried_unknown" if prev is None else "unresolved"
        if link not in LINK_CONT:
            chain_id += 1
        links[i], chains[i] = link, chain_id
        prev, between = i, []
    r["link"], r["chain"] = links, chains
    absorbed = [False] * len(r)
    for i in range(len(r)):
        if r.at[i, "kind"] == "A":
            nxt = next((j for j in range(i + 1, len(r)) if r.at[j, "kind"] == "H"), None)
            absorbed[i] = nxt is not None and links[nxt] in LINK_CONT
    r["absorbed"] = absorbed
    return r


def _dates(df: pd.DataFrame) -> pd.Series:
    return pd.to_datetime(df["pick_date"]).dt.date


def window_summary(rounds: pd.DataFrame, start: date, end: date) -> dict:
    """Within [start, end] for one user: the longest run built INSIDE the window (carried-in streak excluded: the
    count starts at the first in-window H round), exact only when every in-window transition is evidenced, else the
    longest evidenced in-window chain as a lower bound. Coverage and every unavailability reason are reported."""
    a = annotate(rounds) if len(rounds) else rounds
    w = a[(_dates(a) >= start) & (_dates(a) <= end)] if len(a) else a
    out = {"rounds": int(len(w)), "kinds": dict(Counter(w["kind"])) if len(w) else {},
           "incomplete_rounds": int((~w["complete"]).sum()) if len(w) else 0, "links": {}, "reasons": []}
    if not len(w):
        return {**out, "status": "no_window_rounds", "longest_exact": None, "lower_bound": None}
    reasons: set[str] = set()
    if ((w["kind"] == "A") & ~w["absorbed"]).any():
        reasons.add("unabsorbed_ambiguous_round")
    hs = w[w["kind"] == "H"]
    links: Counter = Counter()
    count = longest = 0
    for k, row in enumerate(hs.itertuples()):
        if k == 0:
            count = row.n
        else:
            links[row.link] += 1
            if row.link in LINK_CONT:
                count += row.n
            else:
                if row.link != "new_run_after_miss":
                    reasons.add(row.link)
                count = row.n
        longest = max(longest, count)
    lower = int(hs.groupby("chain")["n"].sum().max()) if len(hs) else 0
    return {**out, "links": dict(links), "reasons": sorted(reasons),
            "status": "lower_bound_only" if reasons else "exact",
            "longest_exact": None if reasons else int(longest), "lower_bound": lower}


def attaining_runs(rounds: pd.DataFrame, best: int | None) -> dict:
    """Every run attaining the board/profile season best ``best`` (kept separate from reconstruction): a run is
    recoverable when it ends on an H round reporting ``best``, its chain reaches back through evidenced links to entry
    evidence, and its slot hits sum to ``best``; otherwise its dates are unavailable and the evidenced chain gives an
    observed-segment lower bound."""
    a = annotate(rounds) if len(rounds) else rounds
    hs = a[a["kind"] == "H"] if len(a) else a
    segments = hs.groupby("chain")["n"].sum() if len(hs) else pd.Series(dtype=int)
    base = {"best": best, "rounds": int(len(a)), "kinds": dict(Counter(a["kind"])) if len(a) else {},
            "incomplete_rounds": int((~a["complete"]).sum()) if len(a) else 0,
            "longest_evidenced_segment": int(segments.max()) if len(segments) else None, "runs": []}
    if best is None:
        return {**base, "status": "best_unavailable"}
    if best == 0:
        return {**base, "status": "best_is_zero"}
    runs = []
    for e in hs[hs["streak"] == best].itertuples():
        chain = hs[(hs["chain"] == e.chain) & (hs.index <= e.Index)]
        first = chain.iloc[0]
        total = int(chain["n"].sum())
        rec = first["link"] in LINK_NEW and total == best
        runs.append({"recoverable": bool(rec),
                     "start_date": str(first["pick_date"]) if rec else None,
                     "end_date": str(e.pick_date) if rec else None,
                     "reported_attainment_date": str(e.pick_date),
                     "observed_segment_lower_bound": total, "n_rounds": int(len(chain)),
                     "n_dd_rounds": int((chain["n"] == 2).sum()), "start_link": first["link"],
                     "links": dict(Counter(chain["link"].iloc[1:]))})
    if not runs:
        status = "no_settled_round_reports_best"
    elif all(x["recoverable"] for x in runs):
        status = "recoverable"
    elif any(x["recoverable"] for x in runs):
        status = "partially_recoverable"
    else:
        status = "dates_unavailable"
    return {**base, "status": status, "runs": runs}
