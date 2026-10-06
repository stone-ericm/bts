"""Synthetic C1 4a fixtures: archived feeds with a consistent boxscore and linescore, serving witnesses and a serving
contract. Nothing here reads real data."""
import json
from collections import Counter

from bts.data.schema import HIT_EVENTS, PA_ENDING_EVENTS
from scripts.audit.c1_r4a import provenance as PV

T = "2027-04-01T23:30:00Z"
BOX_PA = PA_ENDING_EVENTS | {"intent_walk"}          # the boxscore's PA count includes intentional walks


def box_of(plays):
    """{batter: (side, hits, plate appearances)} of a play list [(batter, event, start, top)] (top: away bats)."""
    side, hits, pa = {}, Counter(), Counter()
    for b, event, _, top in plays:
        side[b] = "away" if top else "home"
        hits[b] += event in HIT_EVENTS
        pa[b] += event in BOX_PA
    return {b: (side[b], hits[b], pa[b]) for b in side}


def feed(pk, plays, *, status="Final", resume=None, box=None, line=None, team=None, bench=(), gap=False,
         complete=True):
    """An archived live feed. The boxscore and linescore default to the plays' own totals (`box_of`)."""
    plays = [p if len(p) == 4 else (*p, True) for p in plays]
    box = box_of(plays) if box is None else box
    ps = [{"about": {"atBatIndex": i + (1 if gap and i else 0), "isComplete": True, "startTime": start,
                     "isTopInning": top},
           "matchup": {"batter": {"id": b}}, "result": {"eventType": event}} for i, (b, event, start, top) in
          enumerate(plays)]
    if ps and not complete:
        ps[-1]["about"]["isComplete"] = False
    players = {"away": {}, "home": {}}
    for b, (side, h, pa) in box.items():
        players[side][f"ID{b}"] = {"person": {"id": b}, "stats": {"batting": {"hits": h, "plateAppearances": pa}}}
    for b, side in bench:
        players[side][f"ID{b}"] = {"person": {"id": b}, "stats": {"batting": {}}}
    team_hits = {s: sum(h for side, h, _ in box.values() if side == s) for s in ("away", "home")}
    team = team_hits if team is None else team
    line = team_hits if line is None else line
    dt = {"officialDate": "2027-04-01"}
    if resume:
        dt["resumeDateTime"] = resume
    return json.dumps({
        "gamePk": pk, "gameData": {"game": {"pk": pk}, "status": {"detailedState": status}, "datetime": dt},
        "liveData": {"plays": {"allPlays": ps},
                     "boxscore": {"teams": {s: {"players": players[s], "teamStats": {"batting": {"hits": team[s]}}}
                                            for s in ("away", "home")}},
                     "linescore": {"teams": {s: {"hits": line[s]} for s in ("away", "home")}}}}).encode()


FILES = {"data/schema.py": "a" * 64, "model/predict.py": "b" * 64}
FUNCS = {"bts.orchestrator:predict_local": "c" * 64}
ENV = {"BTS_LGBM_RANDOM_STATE": None, "BTS_USE_CALIBRATION": None}
PKGS = {"python": "3.12.13", "lightgbm": "4.6.0"}
RECIPE = {"files": FILES, "functions": FUNCS, "sha256": PV.recipe_fingerprint(FILES, FUNCS)}


def witness(d, **over):
    w = {"schema": "bts_serving_witness_v1", "tier_type": "local", "recipe": RECIPE, "env": dict(ENV), "env_names": [],
         "packages": dict(PKGS), "model": {"source": "trained", "file": f"blend_{d}.pkl", "sha256": "e" * 64},
         "inputs": [{"file": "pa_2027.parquet", "bytes": 10, "sha256": "f" * 64}],
         "calibration": {"enabled": False, "applied": False}, "built_at": f"{d}T15:00:00+00:00", "errors": []}
    w.update(over)
    return w


def contract(**over):
    c = {"schema": "c1_r4a_serving_contract_v1", "season": 2027, "witness_schema": "bts_serving_witness_v1",
         "tier": {"name": "local", "type": "local"}, "recipe_sha256": RECIPE["sha256"], "env": dict(ENV),
         "packages": dict(PKGS), "calibration_enabled": False,
         "schedule": "daily: the first local run of a date trains and caches blend_<date>.pkl; later runs load it"}
    c.update(over)
    return json.dumps(c).encode()
