"""Run the same synthetic regular-season slates against either revision."""

import json
import sys
from datetime import timedelta

from bts import contest_state, picks, scheduler, strategy

from offline import World, at

REAL_GRADE = picks.resolve_daily_slot_results


def automatic(w, streak, best, legacy, season_streak=None):
    day = w.day
    w.profile = {"activeStreak": streak, "seasonBestStreak": best,
                 "predictions": [{"roundId": 1, "result": "hit"}]}
    yesterday = at(day).date() - timedelta(days=1)
    w.rounds = {1: yesterday}
    if not legacy:
        w.switch("enable")
    w.fetch()
    w.invoke("saver-state", "--init", "active", "--season", str(w.now.year))
    if legacy:
        picks.save_streak(streak if season_streak is None else season_streak, w.directory, True)
    else:
        picks.save_streak(streak if season_streak is None else season_streak, w.directory, True, season=w.now.year)


def network_fixture(w, url, **kwargs):
    if "/feed/live" not in url:
        return json.loads(w.schedule_response(url).read())
    pk = int(url.split("/game/")[1].split("/")[0])
    batting = {"hit": dict(hits=1, atBats=4), "walk": dict(hits=0, atBats=0, baseOnBalls=2),
               "miss": dict(hits=0, atBats=4)}[w.results.get(pk, "hit")]
    return {"gameData": {"status": {"abstractGameState": "Final", "abstractGameCode": "F",
                                   "detailedState": "Final"}},
            "liveData": {"boxscore": {"teams": {
                "home": {"players": {}},
                "away": {"players": {f"ID{batter}": {"person": {"id": batter},
                                                    "stats": {"batting": batting}} for batter in (100001, 100002)}}}}}}


def compare(legacy):
    cases = [
        ("double", "2027-07-15", 12, 15, (0.999, 0.999)),
        ("skip", "2027-07-15", 12, 15, (0.10, 0.09)),
        ("postpone", "2027-07-15", 12, 15, (0.999, 0.75)),
        ("doubleheader", "2027-07-15", 12, 15, (0.999, 0.999)),
        ("walk", "2027-09-20", 45, 45, (0.99, 0.85)),
        ("miss-saver", "2027-09-20", 45, 45, (0.99, 0.85)),
        ("finale", "2027-09-26", 56, 56, (0.99, 0.85)),
    ]
    for name, day, streak, best, probs in cases:
        with World(day) as w:
            w.config["health_checks"]["enabled"] = False
            w.game("20:10", 102, pair="A" if name == "doubleheader" else "B")
            w.game("21:10", 103, pair="C")
            w.prediction = lambda: (w.frame(probs), "offline")
            if name == "walk":
                w.results[101] = w.results[102] = "walk"
            if name == "miss-saver":
                w.results[101] = w.results[102] = "miss"
            automatic(w, streak, best, legacy)
            if name == "postpone":
                lock_decision = scheduler._lock_decision_from_predictions
                def lock_then_postpone(*args, **kwargs):
                    decision = lock_decision(*args, **kwargs)
                    w.games[day][0]["status"]["detailedState"] = "Postponed"
                    return decision
                w.patch(scheduler, "_lock_decision_from_predictions", lock_then_postpone)
            w.patch(picks, "resolve_daily_slot_results", REAL_GRADE)
            w.patch(picks, "_fetch_json", lambda url, **opts: network_fixture(w, url, **opts))
            w.record("days_remaining", strategy._days_remaining(day))
            result = w.run(via_cli=True)
            assert result.exit_code == 0, (result.output, result.exception)
            w.snapshot()
            w.dump(f"midseason-{'base' if legacy else 'head'}-{name}")
    compare_scoring(legacy)
    compare_delivery(legacy)
    compare_late_doubleheader(legacy)


def compare_scoring(legacy):
    for name, slot_results in (("hit", {101: "hit", 102: "hit"}),
                               ("walk", {101: "walk", 102: "walk"}),
                               ("mixed", {101: "hit", 102: "walk"})):
        day = "2027-09-20"
        with World(day) as w:
            w.game("20:10", 102, pair="B")
            w.results = slot_results
            w.patch(picks, "resolve_daily_slot_results", REAL_GRADE)
            w.patch(picks, "_fetch_json", lambda url, **opts: network_fixture(w, url, **opts))
            automatic(w, 12, 15, legacy)
            primary, double = (picks.pick_from_row(row) for _, row in w.frame().iterrows())
            daily = picks.DailyPick(day, w.now.isoformat(), primary, double, None,
                                    notification_sent=True, notification_id="offline-confirmed")
            picks.save_pick(daily, w.directory)
            w.now = at("2027-09-21", "01:00")
            w.grade_command(expect=0)
            w.snapshot()
            w.dump(f"scoring-{'base' if legacy else 'head'}-{name}")


def compare_delivery(legacy):
    day = "2027-09-20"
    for kind in ("sole_void", "partial_void"):
        with World(day) as w:
            automatic(w, 45, 45, legacy)
            w.game("20:10", 102, pair="B")
            primary, double = (picks.pick_from_row(row) for _, row in w.frame().iterrows())
            w.now = at(day, "18:35")
            if kind == "sole_void":
                double = None
            daily = picks.DailyPick(day, w.now.isoformat(), primary, double, None)
            if not legacy:
                decision = contest_state.load_decision_streak_state(w.directory, require_contest_state=True)
                daily.chosen_on = picks.chosen_on_record(decision, required=True)
            picks.save_pick(daily, w.directory)
            w.games[day][0]["status"]["detailedState"] = "Postponed"
            state = scheduler.SchedulerState(
                date=day, schedule_fetched_at=w.now.isoformat(), games=[],
                confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            result = scheduler._deliver_and_lock_pick(daily, w.config, w.directory, state, day, "fixture")
            w.record("delivery_result", result)
            if result:
                w.patch(picks, "resolve_daily_slot_results", REAL_GRADE)
                w.patch(picks, "_fetch_json", lambda url, **opts: network_fixture(w, url, **opts))
                w.now = at(day, "22:10")
                w.record("poll_result", scheduler.run_result_polling(101, day, w.directory))
            w.snapshot()
            w.dump(f"delivery-{'base' if legacy else 'head'}-{kind}")


def compare_late_doubleheader(legacy):
    day = "2027-07-15"
    with World(day) as w:
        w.games[day] = []
        w.game("13:10")
        w.game("16:10", 102)
        w.config["health_checks"]["enabled"] = False
        automatic(w, 12, 15, legacy)
        def confirmed(game_pks):
            observed = {pk: {"away", "home"} for pk in game_pks if pk == 101 or w.now >= at(day, "15:35")}
            w.record("lineup_lookup", observed)
            return observed
        w.patch(scheduler, "check_confirmed_lineups", confirmed)
        w.prediction = lambda: (w.frame((0.99, 0.85)), "offline") if w.now >= at(day, "15:35") else (None, None)
        result = w.run(via_cli=True)
        assert result.exit_code == 0, (result.output, result.exception)
        assert picks.load_pick(day, w.directory) is None
        if not legacy:
            w.now = at("2027-07-16", "08:30")
            w.outcome()
        w.snapshot()
        w.dump(f"midseason-{'base' if legacy else 'head'}-late-dh-lineup")


if __name__ == "__main__":
    compare(sys.argv[1:] == ["base"])
