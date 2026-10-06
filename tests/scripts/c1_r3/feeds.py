"""Synthetic MLB v1.1 live feeds for the rank-3 count build (no real feed is read).

The default game is one complete half-inning per side: away batters 101-109 against home starter 250 (top of 1),
then home batters 201-209 against away starter 150 (bottom of 1).
- **Official totals** (team plateAppearances, the starter's battersFaced, linescore.currentInning) are computed from
  the plays unless overridden. Completed batting turns are production's PA events plus intent_walk and
  batter_interference.
- **Season** is a string, as in the real feeds.
"""
from bts.data.schema import PA_ENDING_EVENTS

COMPLETED_TURN = set(PA_ENDING_EVENTS) | {"intent_walk", "batter_interference"}


def lineup(side_base: int, *, n=9, subs=()):
    """Players dict: starters side_base+1..+n in slots 1..n (codes "100".."900"); subs = (pid, code)."""
    players = {f"ID{side_base + k}": {"person": {"id": side_base + k}, "battingOrder": f"{k}00"} for k in range(1, n + 1)}
    for pid, code in subs:
        players[f"ID{pid}"] = {"person": {"id": pid}, "battingOrder": code}
    players[f"ID{side_base + 50}"] = {"person": {"id": side_base + 50}}            # a pitcher: no battingOrder
    return players


def play(i, half, batter, pitcher, event="single", start="2023-06-01T23:10:00Z", inning=None, complete=True):
    p = {"about": {"atBatIndex": i, "halfInning": half, "inning": inning if inning is not None else 1,
                   "isComplete": complete},
         "matchup": {"batter": {"id": batter}, "pitcher": {"id": pitcher}}, "result": {"eventType": event}}
    if start is not None:
        p["about"]["startTime"] = start
    return p


def default_plays(away_sp=150, home_sp=250):
    plays, i = [], 0
    for k in range(1, 10):
        plays.append(play(i, "top", 100 + k, home_sp, inning=1)); i += 1
    for k in range(1, 10):
        plays.append(play(i, "bottom", 200 + k, away_sp, inning=1)); i += 1
    return plays


def feed(pk=1, date="2023-06-01", *, away=None, home=None, away_pitchers=(150,), home_pitchers=(250,), plays=None,
         resume=None, status="Final", top_pk=None, season=None, totals=None, bf=None, current_inning=None):
    """totals/bf override the official team plateAppearances / starters' battersFaced ({side: n}); by default both
    are computed from the plays."""
    if plays is None:
        plays = default_plays(away_pitchers[0], home_pitchers[0])
    dt = {"officialDate": date}
    if resume:
        dt["resumeDateTime"] = resume
    side_of = lambda p: "home" if p["about"]["halfInning"] == "bottom" else "away"  # noqa: E731
    turns = {s: sum(1 for p in plays if side_of(p) == s and p["result"]["eventType"] in COMPLETED_TURN)
             for s in ("away", "home")}
    sp = {"away": away_pitchers[0], "home": home_pitchers[0]}
    faced = {s: sum(1 for p in plays if side_of(p) != s and p["matchup"]["pitcher"]["id"] == sp[s]
                    and p["result"]["eventType"] in COMPLETED_TURN) for s in ("away", "home")}
    turns.update(totals or {})
    faced.update(bf or {})
    teams = {}
    for s, players in (("away", away if away is not None else lineup(100)), ("home", home if home is not None else lineup(200))):
        players = {k: dict(v) for k, v in players.items()}
        key = f"ID{sp[s]}"
        players.setdefault(key, {"person": {"id": sp[s]}})
        if faced[s] is not None:
            players[key] = {**players[key], "stats": {"pitching": {"battersFaced": faced[s]}}}
        batting = {"plateAppearances": turns[s]} if turns[s] is not None else {}
        teams[s] = {"players": players, "pitchers": list(away_pitchers if s == "away" else home_pitchers),
                    "teamStats": {"batting": batting}}
    last_inning = max((p["about"].get("inning") or 0 for p in plays), default=1)
    return {"gamePk": pk if top_pk is None else top_pk, "metaData": {"timeStamp": "20230602_010203"},
            "gameData": {"game": {"pk": pk, "season": season if season is not None else date[:4], "type": "R"},
                         "datetime": dt, "status": {"detailedState": status},
                         "teams": {"away": {"id": 10}, "home": {"id": 20}}},
            "liveData": {"boxscore": {"teams": teams},
                         "linescore": {"currentInning": current_inning if current_inning is not None else last_inning},
                         "plays": {"allPlays": plays}}}
