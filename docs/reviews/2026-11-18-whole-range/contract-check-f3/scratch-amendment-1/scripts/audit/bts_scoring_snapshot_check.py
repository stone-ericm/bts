"""9/27 scoring re-check against PRE-cutoff snapshots (C-04). Public MLB data + our own capture only.

Why this exists: `bts_scoring_change_check.py` looks for feed revisions, but MLB applies
late official re-scorings IN PLACE — no new feed revision, and `feed/live?timecode=` then
serves the corrected play even at old timecodes (C-03: Chandler Simpson's 5/10 and 8/20
singles became fielding errors days later with no revision). So the only reliable
comparison is against a copy saved before the change:

  (b) the box's cached game feeds (`data/raw/2026/<gamePk>.json`, pulled 03:00 ET 9/28,
      i.e. before the 08:00 ET cutoff) vs MLB's current feeds — every batter;
  (a) BTS's own grading of the round-1009 picks held in the final grab (the board as
      tabulated at 18:51:54 ET) vs the 03:00 copies, via the grab's static player lookup;
  (c) statsapi `game/changes?updatedSince=<tabulation>` — which games MLB touched after the
      tabulation — with a full-feed diff of each against its 03:00 copy.

Run on the box from ~/projects/bts:
  .venv/bin/python scripts/audit/bts_scoring_snapshot_check.py --out /tmp/snapshot_check.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import urllib.request
from pathlib import Path

API = "https://statsapi.mlb.com/api"
UA = {"User-Agent": "Mozilla/5.0 (bts scoring snapshot check)"}
GAME_DATE = "2026-09-27"
ROUND_ID = 1009
TABULATED_UTC = "2026-09-27T22:51:54Z"
GRAB_RAW = Path("data/leaderboard/final_grab_20260927/raw")
BTS_TO_OURS = {"hit": "hit", "not_hit": "no_hit"}


def _get(url: str):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
        return json.load(r)


def _outcome(b: dict) -> str:
    if (b.get("hits") or 0) >= 1:
        return "hit"
    if (b.get("atBats") or 0) + (b.get("sacFlies") or 0) >= 1:
        return "no_hit"
    return "pass"


def _lines(feed: dict, pk: int) -> dict[int, dict]:
    out = {}
    for side in ("away", "home"):
        for p in feed["liveData"]["boxscore"]["teams"][side]["players"].values():
            b = p.get("stats", {}).get("batting") or {}
            if b:
                out[p["person"]["id"]] = {"game_pk": pk, "name": p["person"]["fullName"], "hits": b.get("hits"),
                                          "atBats": b.get("atBats"), "sacFlies": b.get("sacFlies"), "outcome": _outcome(b)}
    return out


def _diff(a, b, path, out):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in set(a) | set(b):
            _diff(a.get(k), b.get(k), f"{path}.{k}", out)
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            _diff(x, y, f"{path}[{i}]", out)
    elif a != b:
        out.append(path)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw/2026"))
    ap.add_argument("--out", type=Path)
    a = ap.parse_args(argv)

    sched = _get(f"{API}/v1/schedule?sportId=1&date={GAME_DATE}&gameType=R")
    games = [g["gamePk"] for d in sched["dates"] for g in d["games"] if g["status"].get("codedGameState") == "F"]
    cached, current, cached_feeds = {}, {}, {}
    for pk in games:
        cached_feeds[pk] = json.loads((a.raw_dir / f"{pk}.json").read_text())
        cached.update(_lines(cached_feeds[pk], pk))
        current.update(_lines(_get(f"{API}/v1.1/game/{pk}/feed/live"), pk))
    b_changes = [{"player_id": pid, "cached_0300": cached[pid], "current": current.get(pid)}
                 for pid in cached if cached[pid] != current.get(pid)]

    players = {p["id"]: p for p in json.loads(gzip.open(GRAB_RAW / "static/002_players.json.gz").read())["players"]}
    graded: dict[int, set] = {}

    def take(pred):
        if pred and pred.get("roundId") == ROUND_ID:
            for rp in pred.get("roundPredictions") or []:
                graded.setdefault(rp["playerId"], set()).add((rp.get("result"), rp.get("hits"), rp.get("atBats")))

    for row in json.loads(gzip.open(GRAB_RAW / "tabs/003_yesterday.json.gz").read())["success"]["ranks"]:
        take(row.get("predictions"))
    for pf in glob.glob(str(GRAB_RAW / "profiles/*.json.gz")):
        body = json.loads(gzip.open(pf).read())
        for pred in body.get("success", body).get("predictions") or []:
            take(pred)
    a_checked, a_mismatch, a_not_played, picked_mlb = 0, [], [], set()
    for bid, views in graded.items():
        mlb = (players.get(bid) or {}).get("feedId")
        line = cached.get(mlb)
        if line is None:
            a_not_played.append({"bts_player": (players.get(bid) or {}).get("name"), "bts_views": sorted(map(list, views), key=str)})
            continue
        picked_mlb.add(mlb)
        for res, h, ab in views:
            a_checked += 1
            if res in BTS_TO_OURS and (BTS_TO_OURS[res] != line["outcome"] or h != line["hits"] or ab != line["atBats"]):
                a_mismatch.append({"player": line["name"], "bts": [res, h, ab], "cached_0300": line})

    ch = _get(f"{API}/v1/game/changes?updatedSince={TABULATED_UTC}&sportId=1&gameType=R")
    touched = sorted({g["gamePk"] for d in ch["dates"] for g in d["games"] if g.get("officialDate") == GAME_DATE})
    c_detail = []
    for pk in touched:
        diffs: list[str] = []
        _diff(cached_feeds[pk], _get(f"{API}/v1.1/game/{pk}/feed/live"), "", diffs)
        batters = [pid for pid, ln in cached.items() if ln["game_pk"] == pk and ln["outcome"] != "pass"]
        c_detail.append({"game_pk": pk, "full_feed_fields_differing_since_0300": [d for d in diffs if "metaData" not in d],
                         "batters_with_pa": len(batters), "covered_by_captured_bts_picks": sum(1 for pid in batters if pid in picked_mlb)})

    report = {"check": "C-04 snapshot re-check of 9/27 scoring", "game_date": GAME_DATE, "tabulated_at_utc": TABULATED_UTC,
              "baseline": f"{a.raw_dir} copies pulled 03:00 ET 9/28 (before the 08:00 ET cutoff)",
              "b_all_batters": {"batters": len(cached), "line_changes_since_0300": b_changes},
              "a_captured_bts_picks": {"distinct_players": len(graded), "graded_views_checked": a_checked,
                                       "mismatches_vs_0300": a_mismatch, "did_not_play": a_not_played},
              "c_games_updated_after_tabulation": c_detail}
    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + "\n")
    print(f"(b) {len(cached)} batters: line changes since 03:00 = {len(b_changes)}")
    print(f"(a) {len(graded)} captured players, {a_checked} graded views: mismatches vs 03:00 = {len(a_mismatch)}; "
          f"did not play = {len(a_not_played)}")
    print(f"(c) games updated after the tabulation: {[(c['game_pk'], len(c['full_feed_fields_differing_since_0300']), c['batters_with_pa'], c['covered_by_captured_bts_picks']) for c in c_detail]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
