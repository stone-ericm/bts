"""W2.1 items 2–3: the final-leader case series (Cohort A = the top of the final all_season board, from the grab's
cohort.json). Survivor-selected by construction; a capture-as-of listing is not an awarded prize.

Per user, from the id-keyed final-grab pick file under the data contract: observed pick days, DD frequency (complete
two-slot rounds / complete rounds; incomplete rounds and unobserved calendar dates reported separately, never as
skips), the board season best by stable user id kept separate from run reconstruction (``streaks.attaining_runs``),
and home/away/team concentration on witnessed rows only. Lineup slot is not stored in the pick schema (omitted)."""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd

from scripts.audit.field_products import cohort as K
from scripts.audit.field_products import picks as P
from scripts.audit.field_products import streaks as S

SEASON = (date(2026, 3, 25), date(2026, 9, 27))     # season_ledger SEASON_DATES
LABELS = {"survivor": "Cohort A is selected on the FINAL board: every statistic here is survivor-selected and "
                      "describes winners after the fact; it estimates nothing about the field.",
          "listing": "a capture-as-of board listing is not an awarded prize",
          "composition": "home/away and team only on rows captured on the pick's own New York date (contemporaneous "
                         "lookups); later lookup values are not historical team evidence; lineup slot is not stored",
          "runs": "board season best is kept separate from run reconstruction; run dates only where evidenced"}


def _read_json(path: Path, hashes: dict | None, rel: str):
    raw = path.read_bytes()
    if hashes is not None:
        hashes[rel] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def case_series(grab_dir: Path, *, board: pd.DataFrame | None = None, cohort_json: dict | None = None,
                identity: dict | None = None, hashes: dict | None = None) -> dict:
    grab_dir = Path(grab_dir)
    cohort_json = cohort_json or _read_json(grab_dir / "cohort.json", hashes, "cohort.json")
    identity = identity or _read_json(grab_dir / "identity.json", hashes, "identity.json")
    if board is None:
        status = json.loads((grab_dir / "status.json").read_text())
        board = pd.read_parquet(grab_dir / status["artifacts"]["leaderboard_snapshot"]["path"])
    a_ids = [int(x) for x in cohort_json["A"]]
    labels = pd.DataFrame({"order": range(1, len(a_ids) + 1), "user_id": a_ids, "allocation": "A"})
    fg = K.final_grab_status(labels, identity, grab_dir).set_index("user_id")
    usable = [u for u in a_ids if fg.loc[u, "usable"]]
    parts = []
    for u in usable:
        p = grab_dir / fg.loc[u, "parsed_path"]
        if hashes is not None:
            hashes[fg.loc[u, "parsed_path"]] = hashlib.sha256(p.read_bytes()).hexdigest()
        parts.append(P.read_observations([p], user_id=u, source="final_grab"))
    obs = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    res = P.resolve(obs)
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
            runs = {"status": "history_unavailable", "runs": [], "longest_evidenced_segment": None}
        rows.append({"a_order": k, "user_id": u, "board_rank": None if b is None else int(b["rank"]),
                     "board_username": None if b is None else b["username"], "board_season_best": best,
                     "history": fg.loc[u, "history"], "fetch_status": fg.loc[u, "fetch_status"],
                     "first_pick_date": fg.loc[u, "first_pick_date"], "last_pick_date": fg.loc[u, "last_pick_date"],
                     "rounds": int(d.get("rounds", 0)), "pick_days": int(d.get("pick_days", 0)),
                     "complete_rounds": int(d.get("complete_rounds", 0)), "dd_rounds": int(d.get("dd_rounds", 0)),
                     "dd_frequency": d.get("dd_frequency"), "incomplete_rounds": int(d.get("incomplete_rounds", 0)),
                     "incomplete_reasons": json.dumps(d.get("incomplete_reasons", {}), sort_keys=True),
                     "calendar_dates": cal, "unobserved_calendar_dates": cal - int(d.get("pick_days", 0)),
                     "runs_status": runs["status"], "runs_recoverable": sum(r["recoverable"] for r in runs["runs"]),
                     "runs_total": len(runs["runs"]), "runs_longest_evidenced_segment":
                         runs.get("longest_evidenced_segment"),
                     "composition_slots": int(c.get("slots", 0)), "composition_witnessed": int(c.get("witnessed", 0)),
                     "composition_unknown": int(c.get("unknown", 0)), "composition_home": int(c.get("home", 0)),
                     "composition_away": int(c.get("away", 0)),
                     "composition_distinct_teams": int(c.get("distinct_teams", 0)),
                     "composition_top_team_share": c.get("top_team_share")})
        run_rows += [{"user_id": u, "board_season_best": best, **r,
                      "links": json.dumps(r["links"], sort_keys=True)} for r in runs["runs"]]
    users = pd.DataFrame(rows)
    use = users[users["history"].isin(["usable", "usable_partial_lookup"])]
    summary = {"n_A": len(a_ids), "survivor_selected": True, "labels": LABELS,
               "season_window": [SEASON[0].isoformat(), SEASON[1].isoformat()],
               "history": users["history"].value_counts().to_dict(),
               "pooled_dd": {"complete_rounds": int(use["complete_rounds"].sum()),
                             "dd_rounds": int(use["dd_rounds"].sum()),
                             "dd_frequency": float(use["dd_rounds"].sum() / use["complete_rounds"].sum())
                             if use["complete_rounds"].sum() else None, "users": int(len(use))},
               "pick_days": {"users": int(len(use)), "total": int(use["pick_days"].sum()),
                             "incomplete_rounds": int(use["incomplete_rounds"].sum())},
               "runs_status": users["runs_status"].value_counts().to_dict(),
               "composition": {"slots": int(users["composition_slots"].sum()),
                               "witnessed": int(users["composition_witnessed"].sum()),
                               "unknown": int(users["composition_unknown"].sum())},
               "revision_log": res.log}
    return {"users": users, "runs": pd.DataFrame(run_rows), "summary": summary, "resolution": res}
