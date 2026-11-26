"""W2.1 items 2–3: the final-leader case series (Cohort A = the top of the final all_season board, from the grab's
cohort.json). Survivor-selected by construction; a capture-as-of listing is not an awarded prize.

Board fields (rank, username, season best) come only from the census gate's QUALIFIED raw rows (code review r2
R2-3), never from the board parquet; an A member without a qualified row has them unavailable, and so is every
best-dependent attainment. Recorded A membership and profile coverage are unchanged.

Per user, from the id-keyed final-grab pick file under the data contract: observed pick days; DD frequency over
rounds completed by the verified raw profile response (``cohort.raw_witness``; incomplete rounds and unobserved
calendar dates reported separately, never as skips); the board season best by stable user id, kept separate from
run reconstruction (``streaks.attaining_runs``: reported attainment and observed-segment lower bounds, run dates
unavailable); and composition, which is unknown without an independent historical context witness (none is stored).
Lineup slot is not stored in the pick schema (omitted). Every file is read through ``read`` (the frozen bytes)."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.audit.field_products import census as C
from scripts.audit.field_products import cohort as K
from scripts.audit.field_products import picks as P
from scripts.audit.field_products import streaks as S

SEASON = (date(2026, 3, 25), date(2026, 9, 27))     # season_ledger SEASON_DATES
LABELS = {"survivor": "Cohort A is selected on the FINAL board: every statistic here is survivor-selected and "
                      "describes winners after the fact; it estimates nothing about the field.",
          "listing": "a capture-as-of board listing is not an awarded prize",
          "composition": "team and home/away need an independent historical pick-time context witness; none is "
                         "stored (pick-row context comes from capture-time lookups), so composition is unknown; "
                         "lineup slot is not stored",
          "runs": "board season best is kept separate from run reconstruction; run start/end dates and exact "
                  "maxima are unavailable (no entered-round completeness witness); lower bounds only",
          "dd": "DD frequency counts only rounds completed by a verified raw profile response"}


def case_series(grab_dir: Path, *, board: pd.DataFrame | None = None, cohort_json: dict | None = None,
                identity: dict | None = None, status: dict | None = None, read=None) -> dict:
    """``board``: the census gate's qualified raw rows (user_id, rank, season_best_streak, username)."""
    grab_dir = Path(grab_dir)
    read = read or (lambda p: Path(p).read_bytes())
    cohort_json = cohort_json or json.loads(read(grab_dir / "cohort.json"))
    identity = identity or json.loads(read(grab_dir / "identity.json"))
    status = status or json.loads(read(grab_dir / "status.json"))
    if board is None:
        board = C.census_gate(C.load_board_receipts(grab_dir, read=read))["qualified"]
    a_ids = [int(x) for x in cohort_json["A"]]
    labels = pd.DataFrame({"order": range(1, len(a_ids) + 1), "user_id": a_ids, "allocation": "A"})
    fg = K.final_grab_status(labels, identity, grab_dir, read=read).set_index("user_id")
    usable = [u for u in a_ids if fg.loc[u, "usable"]]
    parts, witness, wstate = [], {}, {}
    for u in usable:
        o = P.read_observations([grab_dir / fg.loc[u, "parsed_path"]], user_id=u, source="final_grab", read=read)
        w, state = K.raw_witness(grab_dir, status, u, o, read=read)
        witness.update(w)
        wstate[u] = state
        parts.append(o)
    obs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    res = P.resolve(obs, witness=witness)
    dd = P.dd_summary(res.rounds, *SEASON) if len(res.rounds) else {}
    comp = P.composition(obs, res.slots) if len(res.slots) else {}
    by_id = board.drop_duplicates("user_id").set_index("user_id")
    cal = (SEASON[1] - SEASON[0]).days + 1
    rows, run_rows = [], []
    for k, u in enumerate(a_ids, start=1):
        b = by_id.loc[u] if u in by_id.index else None
        best = None if b is None or pd.isna(b["season_best_streak"]) else int(b["season_best_streak"])
        d, c = dd.get(u, {}), comp.get(u, {})
        if u in usable:
            mine = res.rounds[res.rounds["user_id"] == u] if len(res.rounds) else res.rounds
            runs = S.attaining_runs(mine, best)
        else:
            runs = {"status": "history_unavailable", "runs": [], "longest_observed_segment": None}
        rows.append({"a_order": k, "user_id": u, "board_rank": None if b is None else int(b["rank"]),
                     "board_username": None if b is None else b["username"], "board_season_best": best,
                     "history": fg.loc[u, "history"], "fetch_status": fg.loc[u, "fetch_status"],
                     "raw_witness": wstate.get(u, "not_applicable"),
                     "first_pick_date": fg.loc[u, "first_pick_date"], "last_pick_date": fg.loc[u, "last_pick_date"],
                     "rounds": int(d.get("rounds", 0)), "pick_days": int(d.get("pick_days", 0)),
                     "complete_rounds": int(d.get("complete_rounds", 0)), "dd_rounds": int(d.get("dd_rounds", 0)),
                     "dd_frequency": d.get("dd_frequency"), "incomplete_rounds": int(d.get("incomplete_rounds", 0)),
                     "incomplete_reasons": json.dumps(d.get("incomplete_reasons", {}), sort_keys=True),
                     "calendar_dates": cal, "unobserved_calendar_dates": cal - int(d.get("pick_days", 0)),
                     "runs_status": runs["status"], "runs_reported_attainments": len(runs["runs"]),
                     "runs_longest_observed_segment": runs.get("longest_observed_segment"),
                     "composition_slots": int(c.get("slots", 0)), "composition_witnessed": int(c.get("witnessed", 0)),
                     "composition_unknown": int(c.get("unknown", 0)),
                     "composition_context_conflicts": int(c.get("context_conflicts", 0))})
        run_rows += [{"user_id": u, "board_season_best": best, **r} for r in runs["runs"]]
    users = pd.DataFrame(rows)
    use = users[users["history"].isin(["usable", "usable_partial_lookup"])]
    summary = {"n_A": len(a_ids), "survivor_selected": True, "labels": LABELS,
               "season_window": [SEASON[0].isoformat(), SEASON[1].isoformat()],
               "history": users["history"].value_counts().to_dict(),
               "raw_witness": users["raw_witness"].value_counts().to_dict(),
               "pooled_dd": {"complete_rounds": int(use["complete_rounds"].sum()),
                             "dd_rounds": int(use["dd_rounds"].sum()),
                             "dd_frequency": float(use["dd_rounds"].sum() / use["complete_rounds"].sum())
                             if use["complete_rounds"].sum() else None, "users": int(len(use))},
               "pick_days": {"users": int(len(use)), "total": int(use["pick_days"].sum()),
                             "incomplete_rounds": int(use["incomplete_rounds"].sum())},
               "runs_status": users["runs_status"].value_counts().to_dict(),
               "composition": {"slots": int(users["composition_slots"].sum()),
                               "witnessed": int(users["composition_witnessed"].sum()),
                               "unknown": int(users["composition_unknown"].sum()),
                               "context_conflicts": int(users["composition_context_conflicts"].sum()),
                               "basis": "no_independent_historical_context_witness"},
               "revision_log": res.log}
    return {"users": users, "runs": pd.DataFrame(run_rows), "summary": summary, "resolution": res}
