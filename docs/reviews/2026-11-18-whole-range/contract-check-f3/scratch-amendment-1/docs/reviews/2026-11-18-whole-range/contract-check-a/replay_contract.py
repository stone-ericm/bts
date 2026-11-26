"""Run the checked-in offline fixtures from an untracked scratch directory."""

from pathlib import Path
import sys
from tempfile import TemporaryDirectory


SCRATCH = Path(__file__).resolve().parent
REPO = SCRATCH.parents[1]
REVIEWS = REPO / "docs/reviews/2026-11-18-whole-range"


def replay_policy():
    sys.path.insert(0, str(REVIEWS / "slice-2-codex"))
    import offline_fixtures

    offline_fixtures.REPO = REPO
    offline_fixtures.TemporaryDirectory = lambda *, prefix, dir: TemporaryDirectory(
        prefix=prefix, dir=SCRATCH
    )
    offline_fixtures.day_matrix()


def replay_scheduler():
    sys.path.insert(0, str(REVIEWS / "slice-3-codex"))
    import interactions
    import offline
    import refusals

    offline.ROOT = REPO
    offline.HERE = SCRATCH
    probes = {probe.__name__: probe for probe in (
        interactions.fresh_correction_race,
        interactions.overrun_double_cascade,
        interactions.deadline_starved_by_later_check,
        refusals.malformed_schedule,
    )}
    for name in sys.argv[1:]:
        probes[name]()


if __name__ == "__main__":
    family = sys.argv.pop(1)
    if family == "policy":
        replay_policy()
    else:
        replay_scheduler()
