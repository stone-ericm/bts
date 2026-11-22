"""Exercise late lineup, skip, and calendar changes through the real scheduler."""

from dataclasses import asdict
from datetime import timedelta
import json
import sys
from unittest.mock import patch

import replay

from bts import daily_decision, scheduler


def timeline(case, reuse, day=replay.DAY):
    at = replay.offline.instant
    crossing = case == "rollover"
    check_start = at(day, 23, 20) if crossing else at(day, 18, 25)
    with replay.offline.offline(check_start) as (w, mp):
        w.dated(30)
        games = w.schedule[day]
        if crossing:
            games[:] = [replay.offline.game(101, check_start + timedelta(minutes=70))]
        elif case == "skip":
            games[:] = games[:1]
        elif case == "coalesce":
            games.append(replay.offline.game(103, at(day, 21, 10)))
            mp.setattr(scheduler, "plan_run_times", lambda *a, **kw: [
                {"time_et": at(day, hh, mm), "game_pks": [pk]}
                for pk, hh, mm in [(101, 18, 25), (102, 18, 45), (103, 19, 25)]
            ])
        path = w.picks / f"{day}.json"
        calls = []
        real_check = scheduler.run_single_check
        real_lock = scheduler._lock_decision_from_predictions

        def forecast():
            calls.append(w.now.strftime("%H:%M"))
            if len(calls) == 1:
                w.advance((36 if crossing else 35) * 60)
                return w.frame((0.99, 0.79, 0.77), (True, True, True))
            if case == "skip":
                return w.frame((0.10,), (False,))
            if case == "coalesce" and w.now == at(day, 19, 0):
                frame = w.frame((0.79, 0.99, 0.77), (True, False, True))
                return frame.iloc[[1, 0, 2]]
            return None

        def check(*args, **kwargs):
            captured = []

            def lock(*a, **kw):
                decision = real_lock(*a, **kw)
                captured.append(decision)
                return decision

            with patch.object(scheduler, "_lock_decision_from_predictions", lock):
                result = real_check(*args, **kwargs)
            if len(calls) != 1 or not captured:
                return result
            if case == "rollover":
                w.advance(6 * 60)
                w.dated(30)
            if case == "void_pending":
                games[0]["status"] = {
                    "abstractGameCode": "P", "detailedState": "Postponed", "statusCode": "PO",
                }
            if reuse:
                raw = json.dumps(asdict(result["pick_result"].daily), indent=2).encode()
                assert path.read_bytes() == raw
                result["lock_decision"], = captured
                result["saved_pick_bytes"] = raw
            return result

        if case in {"coalesce", "void_pending"}:
            confirmed = {101, 102} if case == "coalesce" else {101}
            mp.setattr(scheduler, "check_confirmed_lineups", lambda pks: {
                pk: {"away", "home"} for pk in pks if pk in confirmed
            })
        if crossing:
            mp.setattr(scheduler, "plan_run_times", lambda *_a, **_kw: [
                {"time_et": check_start, "game_pks": [101]},
            ])
        w.predictions = forecast
        mp.setattr(scheduler, "run_single_check", check)
        scheduler.run_day(day, w.config)

        state = scheduler.load_state(day, w.picks)
        record = daily_decision.load_decision(day, w.picks) or {}
        facts = {
            "case": case, "reuse": reuse, "day": day, "predict": calls,
            "coalesced": [r["time"] for r in state.runs_completed if r.get("reason") == "coalesced_after_fallback"],
            "action": record.get("action"),
            "source": record.get("source"),
            "pick_dm": [body.splitlines()[0] for body in w.sends("dm") if body.startswith("Today's pick")],
            "health_dm": [body.splitlines()[-1] for body in w.sends("health-dm")],
        }
        print("TIMELINE", json.dumps(facts), flush=True)
        if crossing:
            assert not facts["pick_dm"] and any("[contest_state]" in b for b in facts["health_dm"])
        elif case == "skip":
            assert (facts["action"], facts["source"]) == (("single" if reuse else "skip"), "mdp")
        elif case == "void_pending":
            assert not facts["pick_dm"]
            assert any("[postponed_pick]" in b for b in facts["health_dm"]) is (not reuse)
        elif case == "coalesce":
            assert bool(facts["coalesced"]) is reuse


if __name__ == "__main__":
    case = sys.argv[1]
    for inject in (False, True):
        timeline(case, inject, *sys.argv[2:])
