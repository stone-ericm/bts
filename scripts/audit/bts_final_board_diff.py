"""Diff the 9/27-night board capture against the post-cutoff 9/28 capture (C-04 gap closure).

final_grab_20260927 read the board BTS tabulated at 2026-09-27 18:51:54 ET; the 9/28
board-only pass read the board BTS re-tabulated at 2026-09-28 08:11:55 ET — after the
Official Rules 08:00 ET correction cutoff, i.e. the official final board. If every user's
season best and active streak are equal, the 9/27 capture IS the final board and the
C-04 residual is closed; any difference is listed user by user.

Run on the box from ~/projects/bts:
  .venv/bin/python scripts/audit/bts_final_board_diff.py --out /tmp/board_diff.json
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import pandas as pd

ROOT = Path("data/leaderboard")
A, B = "final_grab_20260927", "final_grab_20260928"


def _board(run: str) -> pd.DataFrame:
    date = run[-8:-4] + "-" + run[-4:-2] + "-" + run[-2:]
    return pd.read_parquet(ROOT / run / "leaderboard_snapshots" / f"{date}_all_season_full.parquet")


def _tab_rows(run: str, name: str) -> dict[int, dict]:
    path = next((ROOT / run / "raw" / "tabs").glob(f"*_{name}.json.gz"))
    ranks = json.loads(gzip.open(path).read())["success"]["ranks"]
    return {r["userId"]: r for r in ranks}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    status = {run: json.loads((ROOT / run / "status.json").read_text()) for run in (A, B)}
    old, new = _board(A).set_index("user_id"), _board(B).set_index("user_id")
    only_old, only_new = sorted(set(old.index) - set(new.index)), sorted(set(new.index) - set(old.index))
    both = old.index.intersection(new.index)
    o, n = old.loc[both], new.loc[both]
    best_changed = both[(o["season_best_streak"] != n["season_best_streak"]).to_numpy()]
    active_changed = both[(o["active_streak"] != n["active_streak"]).to_numpy()]
    rank_changed = both[(o["rank"] != n["rank"]).to_numpy()]

    def rows(ids):
        return [{"user_id": int(u), "username": old.at[u, "username"],
                 "season_best": [int(old.at[u, "season_best_streak"]), int(new.at[u, "season_best_streak"])],
                 "active": [int(old.at[u, "active_streak"]), int(new.at[u, "active_streak"])],
                 "rank": [int(old.at[u, "rank"]), int(new.at[u, "rank"])]} for u in ids]

    tabs = {}
    for name in ("active_streak", "all_time", "yesterday"):
        ta, tb = _tab_rows(A, name), _tab_rows(B, name)
        diffs = []
        for uid in sorted(set(ta) | set(tb)):
            ra, rb = ta.get(uid), tb.get(uid)
            ka = None if ra is None else (ra.get("streak"), ra.get("activeStreak"), json.dumps(ra.get("predictions"), sort_keys=True))
            kb = None if rb is None else (rb.get("streak"), rb.get("activeStreak"), json.dumps(rb.get("predictions"), sort_keys=True))
            if ka != kb:
                diffs.append({"user_id": uid, "in_27": ra is not None, "in_28": rb is not None,
                              "streak": [None if ra is None else ra.get("streak"), None if rb is None else rb.get("streak")],
                              "predictions_changed": ra is not None and rb is not None and ka[2] != kb[2]})
        tabs[name] = {"rows_27": len(ta), "rows_28": len(tb), "differing_users": diffs}

    report = {
        "diff": f"{A} (board updatedAt {status[A]['board']['updated_at_values']}) vs {B} (board updatedAt {status[B]['board']['updated_at_values']})",
        "terminal_states": {A: status[A]["terminal_state"], B: status[B]["terminal_state"]},
        "users": {A: int(len(old)), B: int(len(new)), "only_in_27": [int(x) for x in only_old][:50], "only_in_28": [int(x) for x in only_new][:50],
                  "n_only_in_27": len(only_old), "n_only_in_28": len(only_new)},
        "season_best_changed": rows(best_changed),
        "active_streak_changed": {"n": int(len(active_changed)), "rows": rows(active_changed[:200])},
        "rank_changed_n": int(len(rank_changed)),
        "tabs": tabs,
    }
    if a.out:
        a.out.write_text(json.dumps(report, indent=1, default=str) + "\n")
    print(report["diff"])
    print("terminal states:", report["terminal_states"])
    print(f"users: 27={len(old)} 28={len(new)} only27={len(only_old)} only28={len(only_new)}")
    print(f"season best changed: {len(best_changed)} | active streak changed: {len(active_changed)} | rank changed: {len(rank_changed)}")
    for r in report["season_best_changed"][:20]:
        print("   ", r)
    for name, t in tabs.items():
        print(f"tab {name}: rows {t['rows_27']}/{t['rows_28']}, differing users {len(t['differing_users'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
