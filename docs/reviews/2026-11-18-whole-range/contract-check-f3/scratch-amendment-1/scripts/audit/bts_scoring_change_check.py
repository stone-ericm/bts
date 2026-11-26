"""Post-tabulation MLB scoring-change check for one BTS game date (season-wrap W0.6).

Why: the 2026-09-27 final leaderboard grab ran the same evening, against the board
MLB tabulated at 18:51:54 ET. The Official Rules let MLB stat corrections made
before 08:00 ET the next day still change Streak scores (and later ones never do).
This check reads ONLY public MLB game data (statsapi GUMBO feeds; nothing from the
BTS contest) and reports every batter whose BTS outcome — hit / no_hit / pass —
differs between the feed as of the tabulation and the feed as of the cutoff.

BLIND SPOT (found 2026-09-28, C-04): MLB applies late official re-scorings IN PLACE — no
new feed revision, and `?timecode=` then serves the corrected play even at old timecodes
(C-03). This check therefore cannot see such a change, and a "0 changes" result is not
proof. Compare against a copy saved before the cutoff instead:
scripts/audit/bts_scoring_snapshot_check.py.

Mechanism: /feed/live/timestamps lists every feed revision (UTC "YYYYMMDD_HHMMSS");
/feed/live?timecode=T returns the feed as of revision T. For each game: baseline =
latest revision <= tabulated_at, revised = latest revision <= cutoff. Identical
revisions -> nothing could have changed; otherwise diff per-batter outcomes.

Usage:
  python scripts/audit/bts_scoring_change_check.py --date 2026-09-27 \
      --tabulated-at 2026-09-27T18:51:54-04:00 --out docs/audit/2026-09-28-scoring-change-check.json
"""
from __future__ import annotations

import argparse
import json
import sys
import urllib.request
from datetime import date, datetime, time, timedelta, timezone
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
STATSAPI = "https://statsapi.mlb.com/api"
UA = {"User-Agent": "Mozilla/5.0 (bts season-wrap scoring-change check)"}

# Official Rules §6: streak effect of each outcome, and the correction table
# (initial outcome, outcome as revised by 08:00 ET next day) -> streak effect.
EFFECT = {"hit": "increase_1", "no_hit": "end", "pass": "hold"}
CORRECTION_TABLE = {
    ("hit", "pass"): "increase_1",
    ("hit", "no_hit"): "end",
    ("pass", "hit"): "increase_1",
    ("pass", "no_hit"): "end",
    ("no_hit", "pass"): "hold",
    ("no_hit", "hit"): "increase_1",
}


def bts_outcome(batting: dict | None) -> str:
    """Hit = credited with a hit; No Hit = >=1 official at-bat or sacrifice fly
    without one; Pass = anything else (did not play; only BB/HBP/interference/sac bunt)."""
    b = batting or {}
    if (b.get("hits") or 0) >= 1:
        return "hit"
    if (b.get("atBats") or 0) + (b.get("sacFlies") or 0) >= 1:
        return "no_hit"
    return "pass"


def streak_changed(initial: str, revised: str) -> bool:
    if initial == revised:
        return False
    return CORRECTION_TABLE[(initial, revised)] != EFFECT[initial]


def select_timecode(timestamps: list[str], at_utc: datetime) -> str | None:
    """Latest feed revision at or before `at_utc` (timestamps are UTC YYYYMMDD_HHMMSS)."""
    key = at_utc.astimezone(timezone.utc).strftime("%Y%m%d_%H%M%S")
    eligible = [t for t in timestamps if t <= key]
    return max(eligible) if eligible else None


def diff_outcomes(before: dict[int, str], after: dict[int, str]) -> list[dict]:
    """Players whose outcome differs; a player absent from one side is a pass there."""
    out = []
    for pid in sorted(set(before) | set(after)):
        b, a = before.get(pid, "pass"), after.get(pid, "pass")
        if b != a:
            out.append({"player_id": pid, "before": b, "after": a})
    return out


# ------------------------------------------------------------------ network parts
def _get(url: str):
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def _outcomes(feed: dict) -> tuple[dict[int, str], dict[int, dict]]:
    return _box_outcomes(feed["liveData"]["boxscore"])


def _box_outcomes(box: dict) -> tuple[dict[int, str], dict[int, dict]]:
    outcomes, info = {}, {}
    for side in ("away", "home"):
        team = box["teams"][side]
        for p in team["players"].values():
            bat = p.get("stats", {}).get("batting") or {}
            if not bat:
                continue
            pid = p["person"]["id"]
            outcomes[pid] = bts_outcome(bat)
            info[pid] = {"name": p["person"]["fullName"], "team": team["team"].get("abbreviation") or team["team"]["name"],
                         "line": {k: bat.get(k) for k in ("atBats", "hits", "sacFlies", "baseOnBalls", "hitByPitch", "sacBunts")}}
    return outcomes, info


def check_date(game_date: date, tabulated_at: datetime, cutoff: datetime) -> dict:
    sched = _get(f"{STATSAPI}/v1/schedule?sportId=1&date={game_date.isoformat()}&gameType=R")
    games = [g for d in sched.get("dates", []) for g in d["games"]]
    report = {"game_date": game_date.isoformat(), "tabulated_at": tabulated_at.isoformat(),
              "cutoff": cutoff.isoformat(), "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "source": "statsapi GUMBO feed revisions (public MLB data only; no BTS contest requests)",
              "games": []}
    for g in games:
        pk = g["gamePk"]
        matchup = f'{g["teams"]["away"]["team"]["name"]} @ {g["teams"]["home"]["team"]["name"]}'
        rec = {"game_pk": pk, "matchup": matchup, "status": g["status"]["detailedState"]}
        if g["status"].get("codedGameState") != "F":
            rec["note"] = "not a completed game: every batter is a pass; nothing to diff"
            report["games"].append(rec)
            continue
        ts = _get(f"{STATSAPI}/v1.1/game/{pk}/feed/live/timestamps")
        base_tc, rev_tc = select_timecode(ts, tabulated_at), select_timecode(ts, cutoff)
        rec.update(baseline_timecode=base_tc, revised_timecode=rev_tc, last_revision=max(ts) if ts else None,
                   revisions_in_window=[t for t in ts if base_tc and rev_tc and base_tc < t <= rev_tc],
                   revisions_after_cutoff=len([t for t in ts if rev_tc and t > rev_tc]))
        if base_tc is None:
            rec["note"] = "no feed revision at or before the tabulation"
            rec["changes"] = []
            report["games"].append(rec)
            continue
        base_feed = _get(f"{STATSAPI}/v1.1/game/{pk}/feed/live?timecode={base_tc}")
        if base_feed["gameData"]["status"].get("codedGameState") != "F":
            rec["warning"] = "game was not final at the tabulation revision"
        # Second source: the CURRENT stats boxscore. A correction that never produced a
        # feed revision would show up here (timing then unknown -> manual review).
        cur, cur_info = _box_outcomes(_get(f"{STATSAPI}/v1/game/{pk}/boxscore"))
        base_out, base_info = _outcomes(base_feed)
        rec["boxscore_crosscheck"] = [
            {**c, "name": (cur_info.get(c["player_id"]) or base_info.get(c["player_id"]) or {}).get("name")}
            for c in diff_outcomes(base_out, cur)]
        if base_tc == rev_tc:
            rec["changes"] = []
        else:
            before, info_before = base_out, base_info
            after, info = _outcomes(_get(f"{STATSAPI}/v1.1/game/{pk}/feed/live?timecode={rev_tc}"))
            changes = diff_outcomes(before, after)
            for c in changes:
                meta = info.get(c["player_id"]) or info_before.get(c["player_id"]) or {}
                c.update(name=meta.get("name"), team=meta.get("team"),
                         line_before=(info_before.get(c["player_id"]) or {}).get("line"),
                         line_after=(info.get(c["player_id"]) or {}).get("line"),
                         bts_streak_changed=streak_changed(c["before"], c["after"]),
                         revised_effect=CORRECTION_TABLE[(c["before"], c["after"])])
            rec["changes"] = changes
        report["games"].append(rec)
    all_changes = [c for r in report["games"] for c in r.get("changes", [])]
    report["summary"] = {
        "games": len(report["games"]),
        "games_with_revisions_in_window": sum(1 for r in report["games"] if r.get("revisions_in_window")),
        "outcome_changes": len(all_changes),
        "bts_streak_relevant_changes": sum(1 for c in all_changes if c["bts_streak_changed"]),
        "boxscore_crosscheck_discrepancies": sum(len(r.get("boxscore_crosscheck", [])) for r in report["games"]),
    }
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--date", required=True, help="MLB game date YYYY-MM-DD")
    p.add_argument("--tabulated-at", required=True, help="board updatedAt the capture saw (ISO 8601 with offset)")
    p.add_argument("--cutoff", help="default: 08:00 ET the day after --date (Official Rules §6)")
    p.add_argument("--out", help="write the JSON report here")
    a = p.parse_args(argv)
    gd = date.fromisoformat(a.date)
    tab = datetime.fromisoformat(a.tabulated_at)
    cut = datetime.fromisoformat(a.cutoff) if a.cutoff else datetime.combine(gd + timedelta(days=1), time(8, 0), ET)
    if tab.tzinfo is None or cut.tzinfo is None:
        p.error("--tabulated-at/--cutoff need an explicit UTC offset")
    rep = check_date(gd, tab, cut)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(rep, fh, indent=1)
            fh.write("\n")
    s = rep["summary"]
    print(f"{rep['game_date']}: {s['games']} games; revisions between tabulation and cutoff in "
          f"{s['games_with_revisions_in_window']}; outcome changes {s['outcome_changes']}; "
          f"BTS-streak-relevant {s['bts_streak_relevant_changes']}; "
          f"current-boxscore vs tabulation discrepancies {s['boxscore_crosscheck_discrepancies']}")
    for r in rep["games"]:
        for c in r.get("changes", []):
            print(f"  {r['matchup']}: {c['name']} ({c['team']}) {c['before']} -> {c['after']}"
                  f"{'  [streak changes: ' + c['revised_effect'] + ']' if c['bts_streak_changed'] else '  [no streak change]'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
