"""Cold-start rehearsal: predict an early-season date from a data directory that has no
parquet for that season yet.

Usage, from a checkout whose data/processed holds only the earlier seasons:
    .venv/bin/python rehearse.py <YYYY-MM-DD> [--require-cold] [--expect rows|empty]

Real: the season refresh (public schedule and feed reads), the season build, feature
computation, training of the single model and the blend, lineups, prediction, pick
selection and the policy lookup.

Not real, and not evidence of an enterable pick:
- every predicted game is given the status "Scheduled" (for a past date the games are
  Final now), so selection is a counterfactual on eligibility;
- the streak state is supplied, not loaded: streak 0, saver available, doubles allowed,
  an empty picks directory;
- for a past date the feeds are completed games: actual starters, weather and boxscore
  batting orders (substitutes included) stand in for what was known before the game;
- nothing is saved, delivered or posted; no pick is graded or compared with its outcome.

Exit status: 0 when every stage ran and any --expect / --require-cold condition held,
2 when a stage raised or an expectation failed, 3 when --require-cold found a warm state.
"""
import argparse
import math
import sys
import tempfile
import time
import traceback
from dataclasses import asdict, is_dataclass
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("date")
ap.add_argument("--require-cold", action="store_true",
                help="refuse to run unless the season has no parquet, no refresh marker and no raw directory")
ap.add_argument("--expect", choices=("rows", "empty"),
                help="rows: predictions and a selected pick are required; empty: zero prediction rows are required")
args = ap.parse_args()

date = args.date
year = date[:4]
proc = Path("data/processed")
raw = Path("data/raw") / year
failures = []


def parquets():
    return sorted(p.name for p in proc.glob("pa_*.parquet"))


def markers():
    return sorted(p.name for p in proc.glob(".refreshed_*"))


def raw_count():
    return len(list(raw.glob("*.json"))) if raw.exists() else "no directory"


print(f"REHEARSAL date={date} cwd={Path.cwd()}", flush=True)
print("parquets before:", parquets(), flush=True)
print("refresh markers before:", markers(), flush=True)
print(f"json files in data/raw/{year} before:", raw_count(), flush=True)
for label, path in (("probable-pitcher cache", Path("data/models/probable_pitcher_lookup.json")),
                    ("park-drag table", Path("data/external/park_drag/park_drag_export.csv"))):
    print(f"{label} at {path}: {'present' if path.exists() else 'absent'}", flush=True)

if args.require_cold:
    warm = []
    if (proc / f"pa_{year}.parquet").exists():
        warm.append(f"pa_{year}.parquet exists")
    if any(m.startswith(f".refreshed_{year}_") for m in markers()):
        warm.append("a refresh marker for the season exists")
    if raw.exists():
        warm.append(f"data/raw/{year} exists")
    if warm:
        print("NOT COLD: " + "; ".join(warm), flush=True)
        print("RESULT: failed (not a cold state)", flush=True)
        sys.exit(3)
    print("cold state confirmed: no season parquet, no marker, no raw directory for the season", flush=True)

from bts.model.predict import run_pipeline  # noqa: E402
from bts.strategy import (  # noqa: E402
    DecisionContext, _load_mdp, resolve_policy_decision, season_end_date, select_pick,
)

print("season end used for this date:", season_end_date(date), flush=True)

t0 = time.time()
preds = None
try:
    preds = run_pipeline(date, str(proc))
except Exception:
    traceback.print_exc()
    print("PIPELINE: raised", flush=True)
    failures.append("the pipeline raised")
print(f"pipeline seconds: {time.time() - t0:.0f}", flush=True)
print("parquets after:", parquets(), flush=True)
print("refresh markers after:", markers(), flush=True)
print(f"json files in data/raw/{year} after (cached and new):", raw_count(), flush=True)
season_parquet = proc / f"pa_{year}.parquet"
if season_parquet.exists():
    import pandas as pd
    s = pd.read_parquet(season_parquet, columns=["date", "game_pk"])
    print(f"pa_{year}.parquet: {len(s)} rows, {s['game_pk'].nunique()} games, "
          f"dates {s['date'].min()} to {s['date'].max()}", flush=True)

picked = False
if preds is not None:
    n_valid = 0
    if len(preds):
        n_valid = int(sum(1 for v in preds["p_game_hit"]
                          if v is not None and not math.isnan(v) and 0.0 <= v <= 1.0))
    print(f"PIPELINE: returned {len(preds)} rows, {n_valid} with a finite game probability in [0, 1]",
          flush=True)
    if len(preds) and n_valid != len(preds):
        failures.append("some prediction rows lack a valid game probability")
    if len(preds):
        cols = [c for c in ("batter_name", "team", "lineup", "pitcher_name", "game_pk",
                            "game_time", "p_hit_pa", "p_game_hit", "flags") if c in preds.columns]
        print("first 12 rows:", flush=True)
        print(preds.head(12)[cols].to_string(), flush=True)
        print("games with prediction rows:", preds["game_pk"].nunique(), flush=True)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled", "code": "S"}
                    for pk in preds["game_pk"].dropna().unique()}
        with tempfile.TemporaryDirectory() as td:
            try:
                sel = select_pick(preds, date, Path(td), streak=0, saver_available=True,
                                  allow_double=True, game_statuses_detailed=statuses,
                                  require_detailed_statuses=True)
                d = getattr(sel, "decision", None)
                print("SELECTION (statuses substituted, streak 0 supplied): action", sel.action,
                      "| source", sel.source, flush=True)
                if d is not None:
                    print("  decision:", asdict(d) if is_dataclass(d) else d, flush=True)
                pr = sel.pick_result
                if pr is not None:
                    picked = True
                    dd = pr.daily.double_down
                    print("  pick:", pr.daily.pick.batter_name, round(pr.daily.pick.p_game_hit, 4),
                          "| double:", (dd.batter_name, round(dd.p_game_hit, 4)) if dd else None,
                          flush=True)
                else:
                    print("  no pick; primary candidate:", sel.primary_candidate, flush=True)
            except Exception:
                traceback.print_exc()
                print("SELECTION: raised", flush=True)
                failures.append("selection raised")

if args.expect == "rows":
    if preds is None or not len(preds):
        failures.append("expected prediction rows, got none")
    elif not picked:
        failures.append("expected a selected pick, got none")
elif args.expect == "empty":
    if preds is None or len(preds):
        failures.append("expected zero prediction rows")

# The policy on this date for a fresh season, independent of whether lineups existed.
try:
    mdp = _load_mdp()
    if mdp is None:
        print("policy artifacts: neither the base table nor the tail loaded", flush=True)
        failures.append("no policy artifact loaded")
    else:
        print("policy artifacts: base table", "loaded" if mdp.get("policy_table") is not None else "MISSING",
              "| tail", "loaded" if mdp.get("tail") is not None else "MISSING",
              "| base error:", mdp.get("base_error"), "| tail error:", mdp.get("tail_error"), flush=True)
        if mdp.get("policy_table") is None:
            failures.append("the base policy table did not load")
    for p1, p2 in ((0.70, 0.68), (0.76, 0.74), (0.80, 0.78), (0.84, 0.82), (0.88, 0.86)):
        d = resolve_policy_decision(
            DecisionContext(primary_p=p1, second_p=p2, has_diff_game=True, date=date,
                            allow_double=True, mdp=mdp, best_streak=None, best_status=None),
            0, True)
        print(f"  policy at streak 0, p={p1}/{p2}: {d.action} (source {d.source}; "
              f"objective {getattr(d, 'objective', None)})", flush=True)
except Exception:
    traceback.print_exc()
    print("POLICY: raised", flush=True)
    failures.append("the policy lookup raised")

if failures:
    print("RESULT: failed (" + "; ".join(failures) + ")", flush=True)
    sys.exit(2)
print("RESULT: ok", flush=True)
