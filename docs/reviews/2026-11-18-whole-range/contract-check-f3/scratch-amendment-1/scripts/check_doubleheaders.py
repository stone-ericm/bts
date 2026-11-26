"""Check for 7-inning doubleheaders and non-standard games."""
import json
import glob

for year in [2021, 2022, 2023, 2024]:
    files = sorted(glob.glob(f"data/raw/{year}/*.json"))
    if not files:
        print(f"\n{year}: no data yet")
        continue

    dh_games = []
    short_games = []
    for f in files:
        feed = json.load(open(f))
        gd = feed.get("gameData", {})
        gi = gd.get("game", {})
        dh = gi.get("doubleHeader", "N")
        scheduled = gi.get("scheduledInnings", 9)
        game_type = gi.get("type", "?")

        if game_type == "R" and (dh not in ("N",) or scheduled < 9):
            pk = gi.get("pk", "?")
            date = gd.get("datetime", {}).get("officialDate", "?")
            plays = feed.get("liveData", {}).get("plays", {}).get("allPlays", [])
            max_inn = max((p["about"]["inning"] for p in plays), default=0)
            n_pa = sum(1 for p in plays if p.get("result", {}).get("eventType"))
            dh_games.append({
                "pk": pk, "date": date, "dh": dh,
                "scheduled": scheduled, "actual": max_inn, "pas": n_pa
            })
            if scheduled < 9:
                short_games.append(pk)

    total_reg = sum(1 for f in files
                    if json.load(open(f)).get("gameData", {}).get("game", {}).get("type") == "R")
    print(f"\n{year}: {total_reg} regular season games, {len(dh_games)} doubleheaders")
    print(f"  7-inning scheduled: {len(short_games)}")
    for g in dh_games[:5]:
        print(f"    {g['date']} pk={g['pk']} DH={g['dh']} sched={g['scheduled']}inn actual={g['actual']}inn PAs={g['pas']}")
    if len(dh_games) > 5:
        print(f"    ... and {len(dh_games) - 5} more")
