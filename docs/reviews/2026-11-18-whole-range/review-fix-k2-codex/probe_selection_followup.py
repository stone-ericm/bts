from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from probe_selection import HAS_FLOOR, MAKEUP, entry_points, fixture, season_walk, strategy


def degraded_longest_selection():
    day = MAKEUP.isoformat()
    frame, statuses = fixture(MAKEUP)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        for predictions, allow in ((frame.iloc[:1], True), (frame, False)):
            choice = strategy.select_pick(predictions, day, root, 57, True, allow_double=allow,
                                          game_statuses_detailed=statuses)
            assert choice.decision.action == ("single" if HAS_FLOOR else "skip")
        with patch.object(strategy, "_load_mdp", side_effect=RuntimeError("unavailable MDP loader")) as load:
            try:
                choice = strategy.select_pick(frame, day, root, 57, True,
                                              game_statuses_detailed=statuses)
            except RuntimeError as error:
                assert str(error) == "unavailable MDP loader" and load.called
                assert not HAS_FLOOR
                print("legacy loader exception propagated")
            else:
                assert HAS_FLOOR and load.called
                assert (choice.action, choice.decision.days_effective) == ("single", 1)
                assert "unavailable MDP loader" in choice.decision.degraded_reason
                print("raw-zero longest selection catches loader exception and degrades to a single")


if __name__ == "__main__":
    degraded_longest_selection()
    entry_points()
    season_walk()
