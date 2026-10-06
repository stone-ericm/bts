"""Synthetic MLB v1.1 live feeds for the rank-3 count build (no real feed is read)."""


def lineup(side_base: int, *, n=9, subs=()):
    """Players dict: starters side_base+1..+n in slots 1..n (codes "100".."900"); subs = (pid, code)."""
    players = {f"ID{side_base + k}": {"person": {"id": side_base + k}, "battingOrder": f"{k}00"} for k in range(1, n + 1)}
    for pid, code in subs:
        players[f"ID{pid}"] = {"person": {"id": pid}, "battingOrder": code}
    players[f"ID{side_base + 50}"] = {"person": {"id": side_base + 50}}            # a pitcher: no battingOrder
    return players


def play(i, half, batter, pitcher, event="single", start="2023-06-01T23:10:00Z"):
    p = {"about": {"atBatIndex": i, "halfInning": half}, "matchup": {"batter": {"id": batter}, "pitcher": {"id": pitcher}},
         "result": {"eventType": event}}
    if start is not None:
        p["about"]["startTime"] = start
    return p


def feed(pk=1, date="2023-06-01", *, away=None, home=None, away_pitchers=(150,), home_pitchers=(250,), plays=None,
         resume=None):
    """Away batters 101-109 (pitcher 150), home batters 201-209 (pitcher 250). Default plays: one full time through
    each lineup, away batting against home's starter 250 in the top halves."""
    if plays is None:
        plays, i = [], 0
        for k in range(1, 10):
            plays.append(play(i, "top", 100 + k, home_pitchers[0])); i += 1
            plays.append(play(i, "bottom", 200 + k, away_pitchers[0])); i += 1
    dt = {"officialDate": date}
    if resume:
        dt["resumeDateTime"] = resume
    return {"gamePk": pk, "metaData": {"timeStamp": "20230602_010203"},
            "gameData": {"game": {"pk": pk, "season": int(date[:4]), "type": "R"}, "datetime": dt,
                         "teams": {"away": {"id": 10}, "home": {"id": 20}}},
            "liveData": {"boxscore": {"teams": {
                "away": {"players": away if away is not None else lineup(100), "pitchers": list(away_pitchers)},
                "home": {"players": home if home is not None else lineup(200), "pitchers": list(home_pitchers)}}},
                "plays": {"allPlays": plays}}}
