"""Cold-start rehearsal: predict an early-season date from a data directory that has no
parquet for that season yet.

Usage, from a checkout whose data/processed holds only the earlier seasons:
    .venv/bin/python rehearse.py <YYYY-MM-DD>

Real: the season refresh (public schedule and feed reads), the season build, feature
computation, blend training, lineups, prediction, pick selection and the policy lookup.
Not real: every predicted game is given the status "Scheduled" (for a past date the games
are Final now); the streak state is a fresh season (streak 0, saver available); nothing is
saved, delivered or posted, and no result is read or graded.
"""
import sys
import time
import traceback
import tempfile
from dataclasses import asdict, is_dataclass
from pathlib import Path

date = sys.argv[1]
year = date[:4]
proc = Path("data/processed")
raw = Path("data/raw") / year


def parquets():
    return sorted(p.name for p in proc.glob("pa_*.parquet"))


def raw_count():
    return len(list(raw.glob("*.json"))) if raw.exists() else "no directory"


print(f"REHEARSAL date={date} cwd={Path.cwd()}", flush=True)
print("parquets before:", parquets(), flush=True)
print("raw feeds before:", raw_count(), flush=True)

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
print(f"pipeline seconds: {time.time() - t0:.0f}", flush=True)
print("parquets after:", parquets(), flush=True)
print("refresh markers:", sorted(p.name for p in proc.glob(".refreshed_*")), flush=True)
print("raw feeds after:", raw_count(), flush=True)
season_parquet = proc / f"pa_{year}.parquet"
if season_parquet.exists():
    import pandas as pd
    s = pd.read_parquet(season_parquet, columns=["date", "game_pk"])
    print(f"pa_{year}.parquet: {len(s)} rows, {s['game_pk'].nunique()} games, "
          f"dates {s['date'].min()} to {s['date'].max()}", flush=True)

if preds is not None:
    n_valid = int(preds["p_game_hit"].notna().sum()) if len(preds) else 0
    print(f"PIPELINE: returned {len(preds)} rows, {n_valid} with a game probability", flush=True)
    if len(preds):
        cols = [c for c in ("batter_name", "team", "lineup", "pitcher_name", "game_pk",
                            "game_time", "p_hit_pa", "p_game_hit", "flags") if c in preds.columns]
        print(preds.head(12)[cols].to_string(), flush=True)
        print("games predicted:", preds["game_pk"].nunique(), flush=True)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled", "code": "S"}
                    for pk in preds["game_pk"].dropna().unique()}
        with tempfile.TemporaryDirectory() as td:
            try:
                sel = select_pick(preds, date, Path(td), streak=0, saver_available=True,
                                  allow_double=True, game_statuses_detailed=statuses,
                                  require_detailed_statuses=True)
                d = getattr(sel, "decision", None)
                print("SELECTION: action", sel.action, "| source", sel.source, flush=True)
                if d is not None:
                    print("  decision:", asdict(d) if is_dataclass(d) else d, flush=True)
                pr = sel.pick_result
                if pr is not None:
                    dd = pr.daily.double_down
                    print("  pick:", pr.daily.pick.batter_name, round(pr.daily.pick.p_game_hit, 4),
                          "| double:", (dd.batter_name, round(dd.p_game_hit, 4)) if dd else None,
                          flush=True)
                else:
                    print("  no pick; primary candidate:", sel.primary_candidate, flush=True)
            except Exception:
                traceback.print_exc()
                print("SELECTION: raised", flush=True)

# The policy on this date for a fresh season, independent of whether lineups existed.
try:
    mdp = _load_mdp()
    print("policy loaded:", mdp is not None, flush=True)
    for p1, p2 in ((0.70, 0.68), (0.76, 0.74), (0.80, 0.78), (0.84, 0.82), (0.88, 0.86)):
        d = resolve_policy_decision(
            DecisionContext(primary_p=p1, second_p=p2, has_diff_game=True, date=date,
                            allow_double=True, mdp=mdp, best_streak=None, best_status=None),
            0, True)
        print(f"  policy at streak 0, p={p1}/{p2}: {d.action} ({d.source}; "
              f"objective {getattr(d, 'objective', None)})", flush=True)
except Exception:
    traceback.print_exc()
    print("POLICY: raised", flush=True)
print("REHEARSAL END", flush=True)
