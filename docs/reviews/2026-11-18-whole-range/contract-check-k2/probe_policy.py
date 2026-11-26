import argparse
import hashlib
import json
import socket
import tempfile
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch

from bts import strategy
from bts.health import tail_policy as health
from scripts.audit.season_walk import walk

ROOT = Path(__file__).resolve().parents[2]
STREAKS = (*range(61), 120)
ARTIFACTS = strategy._load_mdp()


def offline(*args, **kwargs):
    raise OSError("contract check: network blocked")


def row(d, streak, saver, p=.8, best=None, status=None, allow=True, partner=True):
    ctx = strategy.DecisionContext(p, p - .01 if partner else None, partner, d, allow,
                                   ARTIFACTS, best, status)
    outcome = strategy.resolve_policy_decision(ctx, streak, saver)
    return asdict(outcome)


def examples():
    out = {"strategy": str(Path(strategy.__file__).relative_to(ROOT)),
           "artifacts": [ARTIFACTS["base_sha256"], ARTIFACTS["tail"].sha256],
           "rates": [ARTIFACTS["tail"].bin_p_hit.tolist(), ARTIFACTS["tail"].bin_p_both.tolist()],
           "examples": [], "walk": [], "health": []}
    for day in (date(2027, 9, 26), date(2027, 9, 27), date(2028, 9, 29)):
        d = day.isoformat()
        for s in STREAKS:
            for saver in (False, True):
                best_cases = [(s, "trusted")]
                if s < 55:
                    best_cases += [(min(57, s + 1), "trusted"), (min(57, s + 2), "trusted"),
                                   (57, "untrusted"), (None, None)]
                if s == 20:
                    best_cases += [(18, "trusted"), (25, "trusted")]
                for best, status in best_cases:
                    for p in (.8,):
                        out["examples"].append({"date": d, "saver": saver, "p": p,
                            "display_bar": strategy.effective_pick_bar(s, d, saver, ARTIFACTS, best, status),
                            **row(d, s, saver, p, best, status)})
        for s in (20, 55, 56, 57, 120):
            for saver in (False, True):
                for best in ((20, 25) if s == 20 else (s,)):
                    st = walk.State("dated", s, best, "active" if saver else "used", .8)
                    out["walk"].append(walk.walk_one(day, st))
                frame = walk._predictions(day, .8)
                statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
                with tempfile.TemporaryDirectory() as tmp:
                    for allow, partner in ((True, True), (False, True), (True, False)):
                        chosen = strategy.select_pick(
                            frame if partner else frame.iloc[:1], d, Path(tmp), s, saver,
                            allow_double=allow, game_statuses_detailed=statuses,
                            best_streak=18 if s == 20 else s, best_status="trusted")
                        out.setdefault("select_pick", []).append(
                            {"date": d, "streak": s, "saver": saver, "allow": allow, "partner": partner,
                             "decision": asdict(chosen.decision),
                             "policy_decision": chosen.pick_result.daily.policy_decision if chosen.pick_result else None})
    for day in (date(2027, 1, 1), date(2027, 3, 24), date(2027, 9, 10), date(2027, 9, 27),
                date(2027, 12, 31), date(2028, 1, 1), date(2028, 9, 29)):
        out["health"].append([day.isoformat(), strategy._days_remaining(day.isoformat()),
                              [asdict(a) for a in health.check(today=day)],
                              [asdict(a) for a in health.season_end_alerts(today=day)]])
    out["walk_problems"] = walk.check([record for record in out["walk"] if record["day"].startswith("2027")])
    print(json.dumps(out, indent=2, default=str))


def grid():
    edges = ARTIFACTS["boundaries"]
    ps = sorted({0., .8, 1., *edges, *[edge - 1e-9 for edge in edges]})
    digest = hashlib.sha256()
    count = 0
    day = date(2027, 1, 1)
    while strategy._days_remaining(d := day.isoformat()) > 0:
        days = strategy._days_remaining(d)
        for streak in STREAKS:
            for saver in (False, True):
                best = streak
                bar = strategy.effective_pick_bar(streak, d, saver, ARTIFACTS, best, "trusted")
                for p in ps:
                    dec = row(d, streak, saver, p, best, "trusted")
                    digest.update(repr((d, streak, saver, p, dec, bar)).encode())
                    count += 1
                if streak + 2 * days < 57:
                    for best, trust in ((streak + 2 * days - 1, "trusted"),
                                        (streak + 2 * days, "trusted"), (57, "untrusted"), (None, None)):
                        bar = strategy.effective_pick_bar(streak, d, saver, ARTIFACTS, best, trust)
                        dec = row(d, streak, saver, .8, best, trust)
                        digest.update(repr((d, streak, saver, dec, bar)).encode())
                        count += 1
        day += timedelta(days=1)
    print(json.dumps({"sha256": digest.hexdigest(), "rows": count, "p": ps,
                      "first": "2027-01-01", "last": (day - timedelta(days=1)).isoformat()}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("examples", "grid"))
    args = parser.parse_args()
    with patch.object(socket.socket, "connect", offline), patch.object(socket, "getaddrinfo", offline):
        (examples if args.mode == "examples" else grid)()
