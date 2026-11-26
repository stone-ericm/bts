"""Compare raw EOD alerts, captured DMs, and every fixture file against the archive."""

import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import bts
from pytest import MonkeyPatch

from review_support import DAY, NEXT, RECIPIENT, alert, automatic, clock, eod, files, flag, observe, runner, write

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CASES = ("automatic", "expired", "future", "legacy", "missing", "invalid",
         "active", "not_earned_low", "not_earned_zone", "refused", "coverage")


def arrange(picks, scenario):
    if scenario in {"automatic", "coverage", "expired", "future"}:
        automatic(picks, DAY)
    if scenario in {"expired", "future", "invalid", "active", "not_earned_zone", "not_earned_low"}:
        offset = {"expired": -1, "future": 1}.get(scenario, 0)
        saver = "not_earned" if scenario.startswith("not_earned") else "active" if scenario == "active" else "used"
        path = observe(picks, DAY + dt.timedelta(days=offset), saver=saver,
                       streak=9 if scenario.endswith("_low") else 12)
        if scenario == "invalid":
            write(path, {**json.loads(path.read_text()), "saver_state": False})
    elif scenario == "legacy":
        write(picks / "account_state/contest_streak.manual.json", {
            "active_streak": 4, "best_streak": 8, "source_date": str(DAY.date()),
            "override_expires_at": (DAY - dt.timedelta(hours=1)).isoformat(),
        })
    elif scenario == "coverage":
        for offset in (1, 2):
            write(picks / f"{DAY.date() + dt.timedelta(days=offset)}.json", {"result": "hit"})
    elif scenario == "refused":
        automatic(picks, DAY).with_name("season_contract.json").unlink()


def scheduler_snapshot(root, wall, recipient, enabled):
    from bts import dm, orchestrator, scheduler, strategy
    from tests.test_saved_pick_paths_cs5a import prepare_run_day, world

    with MonkeyPatch.context() as changes:
        config, picks, _, _ = world.__wrapped__(root, changes)
        prepare_run_day(changes)
        changes.setattr(strategy, "_mdp_cache", None)
        changes.setattr(scheduler, "notify_ready", Mock())
        changes.setattr(scheduler, "notify_watchdog", Mock())
        changes.setattr(runner, "get_self_pid", lambda: None)
        changes.setattr(runner, "read_systemd_nrestarts", lambda: None)
        changes.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
        config["health_checks"]["enabled"] = enabled
        config["bluesky"]["dm_recipient"] = recipient
        orchestrator.run_cascade.return_value = (None, None)
        observe(picks, saver="not_earned", streak=12)
        flag(picks, "active")
        raw = []
        real = runner.run_all_checks

        def capture(*args, **kwargs):
            rows = real(*args, **kwargs)
            raw.extend(asdict(row) for row in rows)
            return rows

        run = Mock(side_effect=capture)
        changes.setattr(runner, "run_all_checks", run)
        with clock(wall):
            scheduler.run_day(str(DAY.date()), config)
        assert run.call_count == int(enabled)
        if enabled:
            assert run.call_args.kwargs["today"] == DAY.date()
            assert any(row["source"] == "saver_state" for row in raw)
        return raw, [call.args for call in alert.send_dm.call_args_list], [
            call.args for call in dm.send_dm.call_args_list
        ], files(picks.parent / "health_state")


def grid(root):
    result = {}
    for historical, wall in enumerate((DAY, NEXT)):
        for recipient_index, recipient in enumerate((RECIPIENT, None, "")):
            for enabled in (False, True):
                key = f"scheduler/{historical}-{recipient_index}-{enabled}"
                result[key] = scheduler_snapshot(root / key, wall, recipient, enabled)
            for expected in (False, True):
                for scenario in CASES:
                    for state in ("active", "not_earned", "used", "absent"):
                        key = f"{historical}-{recipient_index}-{expected}/{scenario}/{state}"
                        picks = root / key / "picks"
                        picks.mkdir(parents=True)
                        arrange(picks, scenario)
                        if state != "absent":
                            flag(picks, state)
                        alerts, sends = eod(picks, wall=wall, recipient=recipient, expected=expected, existing=(
                            alert.Alert("INFO", "fixture", "an existing diagnostic"),
                            alert.Alert("CRITICAL", "fixture", "an existing digest item"),
                        ))
                        result[key] = [list(map(asdict, alerts)), sends, files(picks.parent / "health_state")]
    return result


if __name__ == "__main__":
    if len(sys.argv) == 3:
        print("bts.__file__ =", bts.__file__)
        Path(sys.argv[2]).write_text(json.dumps(grid(Path(sys.argv[1])), sort_keys=True))
    else:
        with TemporaryDirectory(prefix="f2-pair-", dir="/tmp") as directory:
            work = Path(directory)
            for label, source in (("base", HERE / "base/src"), ("candidate", ROOT / "src")):
                subprocess.run(
                    [sys.executable, __file__, str(work / "fixture"), str(work / f"{label}.json")],
                    env={**os.environ, "PYTHONPATH": f"{source}:{ROOT}"}, timeout=18, check=True,
                )
                (work / "fixture").rename(work / label)
            base, candidate = (json.loads((work / f"{side}.json").read_text()) for side in ("base", "candidate"))
            assert base.keys() == candidate.keys()
            changed = [case for case in base if base[case] != candidate[case]]
            assert not changed, [(case, base[case], candidate[case]) for case in changed[:2]]
            print(f"{len(base)} equal pairs, including 12 scheduler paths; raw order, DMs, status bytes")
