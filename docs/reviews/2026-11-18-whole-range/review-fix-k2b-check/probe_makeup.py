import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from run_tests import offline

sys.addaudithook(offline)
import bts
from bts import orchestrator, picks, strategy
from scripts.audit.season_walk import walk


DAY = date(2027, 9, 27)
EXPECTED = Path(sys.argv[1]).resolve() / "bts" / "__init__.py"
assert Path(bts.__file__).resolve() == EXPECTED
print(bts.__file__)
floor = "min_days" in strategy.DecisionContext.__dataclass_fields__
frame = walk._predictions(DAY, .8)
statuses = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
at = walk.REAL_DATETIME.fromisoformat(f"{DAY}T17:00:00-04:00")
for streak, best, action, objective in (
    (20, 20, "double", "emax_season_best"),
    (20, 25, "skip", "emax_season_best"),
    (55, 55, "double", "reach57"),
    (56, 56, "single", "reach57"),
    (57, 57, "double", "longest_streak"),
    (120, 120, "double", "longest_streak"),
):
    wanted = (action, objective, 1) if floor else ("skip", "reach57", 0)
    for saver in (False, True):
        state = walk.State("dated", streak, best, "active" if saver else "used", .8)
        with TemporaryDirectory() as scratch, walk.offline(at):
            root = Path(scratch) / "picks"
            root.mkdir()
            assert walk.install_state(root, DAY, state) == []
            walk._set_clock(at)
            direct = strategy.select_pick(frame, DAY.isoformat(), root, streak, saver,
                                          game_statuses_detailed=statuses,
                                          best_streak=best, best_status="trusted")
            with (patch.object(orchestrator, "run_cascade", return_value=(frame, "fixture")),
                  patch.object(picks, "get_game_statuses_detailed", return_value=statuses)):
                _, from_orchestrator, tier = orchestrator.run_and_pick(walk._config(root), DAY.isoformat())
            assert tier == "fixture" and from_orchestrator.state_source == "contest"
            for result in (direct, from_orchestrator):
                decision = result.decision
                assert (result.action, decision.objective, decision.days_effective) == wanted, result
                assert (result.pick_result is None) == (wanted[0] == "skip")
            assert direct.decision == from_orchestrator.decision
        if floor:
            row = walk.walk_one(DAY, state)
            assert (row["action"], row["objective"], row["days_effective"]) == wanted
            assert row["days_left"] == 0 and walk.check([row]) == []
        print(streak, best, saver, *wanted)
