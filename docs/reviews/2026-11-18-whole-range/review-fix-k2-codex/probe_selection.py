import importlib.util
import json
import os
import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from offline import ROOT, install

install()

from bts import orchestrator, picks, posting, scheduler, strategy
from bts.model import predict
from scripts.audit.season_walk import walk

HAS_FLOOR = "min_days" in strategy.DecisionContext.__dataclass_fields__
MAKEUP = date(2027, 9, 27)


def expected(streak, best):
    if streak >= 57:
        return "double", "longest_streak"
    if streak >= 55:
        return ("double" if streak == 55 else "single"), "reach57"
    return ("skip" if best >= streak + 2 else "double"), "emax_season_best"


def decision_tuple(decision):
    return decision.action, decision.objective, decision.days_effective


def fixture(day, probability=.8):
    frame = walk._predictions(day, probability)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
    return frame, statuses


def table():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for day in (MAKEUP, date(2028, 9, 28), date(2028, 9, 29)):
            frame, statuses = fixture(day)
            for streak in (*range(61), 120):
                bests = (streak, 18, 25) if streak == 20 else (streak,)
                for best in bests:
                    for saver in (False, True):
                        result = strategy.select_pick(frame, day.isoformat(), root, streak, saver,
                                                      game_statuses_detailed=statuses,
                                                      best_streak=best, best_status="trusted")
                        want = (*expected(streak, best), 1) if HAS_FLOOR else ("skip", "reach57", 0)
                        assert decision_tuple(result.decision) == want, (day, streak, best, saver, result)
                        if want[0] == "skip" and HAS_FLOOR:
                            assert "max reachable 22" in result.decision.reason
                        if result.pick_result:
                            assert result.pick_result.daily.policy_decision["days_effective"] == want[2]
                        print(day, streak, best, saver, *want)
        day = MAKEUP.isoformat()
        frame, statuses = fixture(MAKEUP)
        for input_frame, allow in ((frame.iloc[:1], True), (frame, False)):
            result = strategy.select_pick(input_frame, day, root, 57, True, allow_double=allow,
                                          game_statuses_detailed=statuses)
            assert result.decision.action == ("single" if HAS_FLOOR else "skip")
        with patch.object(strategy, "_load_mdp", side_effect=RuntimeError("unavailable MDP loader")) as load:
            if HAS_FLOOR:
                result = strategy.select_pick(frame, day, root, 57, True, game_statuses_detailed=statuses)
                assert result.decision.action == "double" and not load.called
            else:
                try:
                    strategy.select_pick(frame, day, root, 57, True, game_statuses_detailed=statuses)
                except RuntimeError as error:
                    assert str(error) == "unavailable MDP loader" and load.called
                else:
                    raise AssertionError("base skipped the MDP load")


def entry_points():
    for day in (MAKEUP, date(2028, 9, 29)):
        key = day.isoformat()
        at = walk.REAL_DATETIME.combine(day, walk.REAL_DATETIME.min.time(), walk.ET).replace(hour=17)
        frame, statuses = fixture(day)
        for streak, best in ((20, 18), (20, 25), (55, 55), (56, 56), (57, 57), (120, 120)):
            for saver in ("active", "used"):
                for entry in ("run_and_pick", "run", "preview", "shadow"):
                    with TemporaryDirectory() as directory, walk.offline(at):
                        root = Path(directory) / "picks"
                        root.mkdir()
                        state = walk.State("dated", streak, max(streak, best), saver, .8)
                        assert walk.install_state(root, day, state) == []
                        walk._set_clock(at)
                        config = walk._config(root)
                        config["orchestrator"].update(heartbeat_path=str(root / "heartbeat"),
                                                      models_dir=str(root / "models"))
                        posts = []
                        want = expected(streak, streak if entry == "shadow" else best)[0] if HAS_FLOOR else "skip"
                        with (patch.object(predict, "run_pipeline", return_value=frame),
                              patch.object(picks, "get_game_statuses_detailed", return_value=statuses),
                              patch.object(posting, "post_to_bluesky",
                                           side_effect=lambda body: posts.append(body) or "at://probe"),
                              patch.object(orchestrator, "run_cascade", return_value=(frame, "probe")),
                              patch.object(scheduler, "heartbeat_watchdog"),
                              patch.object(scheduler, "predict_local_shadow", return_value=frame),
                              patch.object(scheduler, "load_streak", return_value=streak, create=True),
                              patch.object(picks, "load_streak", return_value=streak)):
                            if entry == "run_and_pick":
                                _, selected, _ = orchestrator.run_and_pick(config, key)
                                assert selected.decision.action == want
                                daily = selected.pick_result.daily if selected.pick_result else None
                            elif entry == "shadow":
                                scheduler._run_shadow_prediction(config, key, "irrelevant")
                                daily = picks.load_shadow_pick(key, root)
                            else:
                                code, output = walk._invoke(entry, "--date", key, "--picks-dir", root,
                                                            "--models-dir", root / "models")
                                assert code == 0, output
                                daily = picks.load_pick(key, root)
                                assert ("Skip" in output) == (want == "skip"), output
                                if entry == "run" and want != "skip":
                                    assert len(posts) == 1 and daily.bluesky_posted, (posts, output)
                            if entry != "run":
                                assert not posts
                            assert (daily is None) == (want == "skip"), (entry, key, state, daily, want)
                            if daily:
                                assert bool(daily.double_down) == (want == "double"), (entry, state, daily, want)
                                assert daily.policy_decision["days_effective"] == 1
                            print(key, state.label(), entry, want, "posts", len(posts))


def season_walk():
    archive = os.environ.get("K2_REVIEW_ARCHIVED_WALK")
    subject = walk
    if archive:
        spec = importlib.util.spec_from_file_location("archived_walk", archive)
        subject = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = subject
        spec.loader.exec_module(subject)
        subject.REPO = ROOT
    states = [subject.State("dated", s, best, "used", .8)
              for s, best in ((20, 20), (20, 25), (55, 55), (56, 56), (57, 57), (120, 120))]
    rows = subject.walk(MAKEUP, MAKEUP, states=states)
    for state, row in zip(states, rows, strict=True):
        want = (*expected(state.streak, state.best), 1) if HAS_FLOOR else ("skip", "reach57", 0)
        assert (row["action"], row["objective"], row["days_effective"]) == want, row
        assert row["days_left"] == 0
    assert subject.check(rows) == []
    if HAS_FLOOR:
        empty = subject.walk_one(MAKEUP, states[-1], empty_slate=True)
        assert empty["action"] is None and not subject.check([empty])
    with patch.dict(strategy.SEASON_END_DATES, {2027: "2027-09-26"}):
        wrong_finale = subject.walk_one(date(2027, 9, 26), states[3])
        assert subject.table_boundary_problems()
    problems = subject.check([wrong_finale])
    assert any("days left 0, expected 1" in detail for detail in problems)
    assert (wrong_finale["action"] == "single") == HAS_FLOOR
    if HAS_FLOOR:
        with patch.object(strategy, "_mdp_action_from", return_value=("single", None)):
            wrong_table = subject.walk_one(MAKEUP, states[2])
        assert wrong_table["action"] == "single" and wrong_table["days_effective"] == 1
        assert subject.check([wrong_table]) == []
        print("walk false green: forced 55 to single instead of double", wrong_table["action"])
    print(json.dumps(dict(wrong_finale=wrong_finale["action"], oracle=problems,
                          makeup=[row["action"] for row in rows])))


if __name__ == "__main__":
    table()
    entry_points()
    season_walk()
