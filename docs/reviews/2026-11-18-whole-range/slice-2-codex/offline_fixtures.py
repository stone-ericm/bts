"""Shared fixture controls for scheduler, deployment, and cron probes."""

from contextlib import contextmanager, ExitStack
from datetime import datetime, timedelta
from pathlib import Path
import shutil
import sys
from unittest.mock import patch

import day_probe as probe


REPO = Path(__file__).resolve().parents[2]
ORIGINAL_SEED = probe.seed
ORIGINAL_ARTIFACT = probe.artifact
ORIGINAL_INSTALLED = probe.installed
ORIGINAL_SNAPSHOT = probe.snapshot


def history(picks, day, streak):
    from bts.picks import DailyPick, Pick, save_pick

    for offset in range(1, streak + 1):
        when = probe.at(day) - timedelta(days=offset)
        if when.year != int(day[:4]):
            break
        day_iso = when.date().isoformat()
        pick = Pick(batter_name="Previous batter", batter_id=700, team="OLD",
                    lineup_position=1, pitcher_name="Previous pitcher", pitcher_id=None, p_game_hit=0.9,
                    flags=[], projected_lineup=False, game_pk=700, game_time=when.isoformat())
        save_pick(DailyPick(date=day_iso, run_time=when.isoformat(), pick=pick,
                            double_down=None, runner_up=None, bluesky_posted=False,
                            bluesky_uri=None, result="hit", slot_results={"pick": "hit"}),
                  picks)


def seed(picks, day, **kwargs):
    ORIGINAL_SEED(picks, day, **kwargs)
    state = picks / "account_state/contest_streak.json"
    observation = probe.read(state)
    observation["source"] = "mlb_bts_profile"
    probe.write(state, observation)
    if not kwargs.get("old"):
        history(picks, day, kwargs.get("model", kwargs.get("streak", 12)))


@contextmanager
def installed(world, root, *, bridge=False, production=False, inclusive=False):
    import bts
    from bts import contest_fetch, dm, entry_receipt, reconcile_receipt, strategy
    from bts.health import alert
    if hasattr(strategy, "SEASON_END_DATES"):
        import bts.season_contract
        import bts.day_outcome

    def production_artifacts(path):
        path.parent.mkdir(parents=True, exist_ok=True)
        for name in ("mdp_policy.npz", "mdp_tail_policy.npz"):
            shutil.copyfile(REPO / "data/models" / name, path.parent / name)

    exclusive_days = strategy._days_remaining
    frozen = None
    try:
        with ExitStack() as scope:
            scope.enter_context(patch.object(probe, "artifact",
                                            production_artifacts if production else ORIGINAL_ARTIFACT))
            scope.enter_context(patch.object(alert, "send_dm", world.send))
            scope.enter_context(ORIGINAL_INSTALLED(world, root, bridge=bridge))
            frozen = probe.datetime_module.datetime
            if inclusive:
                from bts.strategy import season_end_date

                def including_today(day):
                    return max(0, (datetime.fromisoformat(season_end_date(day))
                                   - datetime.fromisoformat(day)).days + 1)

                scope.enter_context(patch.object(strategy, "_days_remaining", including_today))
            yield
    finally:
        if frozen is not None:
            for name, module in tuple(sys.modules.items()):
                if name.startswith("bts.") and getattr(module, "datetime", None) is frozen:
                    module.datetime = datetime
        assert strategy._days_remaining is exclusive_days


def day_matrix():
    """Entrypoint for the day probe with trusted provenance and retained history."""
    import argparse
    import json
    from tempfile import TemporaryDirectory

    parser = argparse.ArgumentParser()
    parser.add_argument("--bridge", action="store_true")
    parser.add_argument("--production", action="store_true")
    parser.add_argument("--inclusive-control", action="store_true")
    parser.add_argument("--case", choices=[case[0] for case in probe.CASES])
    args = parser.parse_args()

    def run_in_world(world, root, *, bridge=False):
        return installed(world, root, bridge=bridge, production=args.production,
                         inclusive=args.inclusive_control)

    # At 07:40 grade/reconcile, then move the read-only day monitor to 08:30.
    def invoke(args):
        if args[0] == "day-outcome":
            active_world.now += timedelta(minutes=50)
        return original_invoke(args)

    original_invoke = probe.invoke
    for case in probe.CASES:
        if args.case and case[0] != args.case:
            continue
        day = case[1]
        with TemporaryDirectory(prefix=f"{case[0]}-", dir=Path(__file__).resolve().parent) as temp:
            def capture(world, root, *, bridge=False):
                nonlocal active_world
                active_world = world
                return run_in_world(world, root, bridge=bridge)

            def snapshots(root):
                return {path: value for path, value in ORIGINAL_SNAPSHOT(root).items()
                        if not (path.startswith("data/picks/2027-") and
                                len(Path(path).stem) == 10 and Path(path).stem < day)}

            active_world = None
            with patch.object(probe, "seed", seed), patch.object(probe, "installed", capture), \
                    patch.object(probe, "snapshot", snapshots), patch.object(probe, "invoke", invoke):
                print(json.dumps(probe.run_case(Path(temp), case, args.bridge)), flush=True)
