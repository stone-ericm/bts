import datetime as dt
import json
import os
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

from pytest import MonkeyPatch

from probe_support import (
    DAY, RECIPIENT, alert, automatic, bts, clock, eod, flag, health_files, observed,
    put, runner, scratch,
)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
NEXT = (DAY + dt.timedelta(days=1)).replace(hour=0, minute=5)


def arrange(picks, scenario):
    if scenario != "missing":
        coverage = DAY.date() - dt.timedelta(days=2) if scenario == "coverage" else DAY.date()
        if scenario == "future-coverage":
            coverage += dt.timedelta(days=5)
        automatic(picks, DAY, coverage=coverage)
    if scenario in {"active", "not_earned", "not_earned-low", "not_earned-high"}:
        observed(picks, state=scenario.split("-")[0],
                 streak={"not_earned-low": 9, "not_earned-high": 16}.get(scenario, 12))
    elif scenario in {"expired", "invalid", "future-observation"}:
        when = DAY + dt.timedelta(days={"expired": -1, "invalid": 0, "future-observation": 1}[scenario])
        path = observed(picks, when)
        if scenario == "invalid":
            payload = json.loads(path.read_text())
            payload["saver_state"] = True
            path.write_text(json.dumps(payload))
    elif scenario in {"legacy", "legacy-expired"}:
        expiry = DAY + dt.timedelta(hours=1 if scenario == "legacy" else -1)
        put(picks / "account_state", "contest_streak.manual.json", {
            "active_streak": 6, "best_streak": 8, "source_date": str(DAY.date()),
            "override_expires_at": expiry.isoformat(), "recorded_at": DAY.isoformat(),
            "source": "manual_cli",
        })
    elif scenario == "coverage":
        for offset in (0, -1):
            put(picks, f"{DAY.date() + dt.timedelta(days=offset)}.json", {"result": "hit"})
    elif scenario == "refusal":
        (picks / "account_state/season_contract.json").unlink()


def scheduler_case(directory, *, enabled, historical, recipient):
    from bts import dm, orchestrator, scheduler, strategy
    from tests.test_saved_pick_paths_cs5a import prepare_run_day, world

    with MonkeyPatch.context() as patch:
        config, picks, _, _ = world.__wrapped__(directory, patch)
        prepare_run_day(patch)
        patch.setattr(strategy, "_mdp_cache", None)
        patch.setattr(scheduler, "notify_ready", Mock())
        patch.setattr(scheduler, "notify_watchdog", Mock())
        patch.setattr(runner, "get_self_pid", lambda: None)
        patch.setattr(runner, "read_systemd_nrestarts", lambda: None)
        patch.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
        config["health_checks"]["enabled"] = enabled
        config["bluesky"]["dm_recipient"] = recipient
        orchestrator.run_cascade.return_value = (None, None)
        observed(picks, state="not_earned", streak=12)
        flag(picks, "active")
        run = Mock(wraps=runner.run_all_checks)
        raw = []

        def capture(*args, **kwargs):
            alerts = run(*args, **kwargs)
            raw.extend(asdict(row) for row in alerts)
            return alerts

        patch.setattr(runner, "run_all_checks", capture)
        with clock(NEXT if historical else DAY):
            scheduler.run_day(str(DAY.date()), config)
        assert run.call_count == int(enabled)
        if enabled:
            assert run.call_args.kwargs["today"] == DAY.date()
            assert any(row["source"] == "saver_state" for row in raw)
        return {
            "raw": raw, "health_sends": [c.args for c in alert.send_dm.call_args_list],
            "pick_sends": [c.args for c in dm.send_dm.call_args_list],
            "health_files": health_files(picks),
        }


def grid(root):
    results = {}
    mode = {False: DAY, True: NEXT}
    scenarios = ("missing", "automatic", "active", "not_earned", "not_earned-low",
                 "not_earned-high", "expired", "legacy", "legacy-expired", "future-observation",
                 "invalid", "future-coverage", "coverage", "refusal")
    for scenario in scenarios:
        for flag_state in ("active", "not_earned", "used", "absent", "malformed"):
            for historical, clock_at in mode.items():
                for recipient in (RECIPIENT, None):
                    key = f"{scenario}/{flag_state}/{historical}/{recipient}"
                    picks = root / key / "picks"
                    # For truly missing contest state, creation of a flag is independent.
                    picks.mkdir(parents=True)
                    arrange(picks, scenario)
                    if flag_state == "malformed":
                        (picks / "account_state").mkdir(exist_ok=True)
                        (picks / "account_state/saver_state.json").write_text("{")
                    elif flag_state != "absent":
                        flag(picks, flag_state)
                    raw, sends = eod(picks, wall=clock_at, recipient=recipient, existing=(
                        alert.Alert("INFO", "parity_probe", "existing pre-check indicator"),
                        alert.Alert("CRITICAL", "parity_probe", "existing health alarm"),
                    ))
                    results[key] = {
                        "alerts": [asdict(row) for row in raw], "transport": sends,
                        "files": health_files(picks),
                    }
    for historical in mode:
        for enabled, recipient in ((False, RECIPIENT), (False, None), (True, None), (True, RECIPIENT)):
            key = f"scheduler/{historical}/{enabled}/{recipient}"
            results[key] = scheduler_case(root / key, enabled=enabled, historical=historical,
                                          recipient=recipient)
    return results


if __name__ == "__main__" and len(sys.argv) == 3:
    print(f"bts.__file__ = {bts.__file__}", flush=True)
    Path(sys.argv[2]).write_text(json.dumps(grid(Path(sys.argv[1])), sort_keys=True, indent=2))
elif __name__ == "__main__":
    with scratch() as directory:
        fixture = directory / "fixtures"
        for name, source in (("base", HERE / "base/src"), ("candidate", REPO / "src")):
            subprocess.run([sys.executable, __file__, str(fixture), str(directory / f"{name}.json")],
                           env={**os.environ, "PYTHONPATH": f"{source}:{REPO}"},
                           timeout=15, check=True)
            fixture.rename(directory / name)
        base = json.loads((directory / "base.json").read_text())
        candidate = json.loads((directory / "candidate.json").read_text())
        assert base.keys() == candidate.keys()
        different = [key for key in base if base[key] != candidate[key]]
        for key in different:
            print(key, "base:", base[key], "candidate:", candidate[key])
        assert not different
        print(f"{len(base)} paired cases, including 8 scheduler runs: raw alerts, DM calls and status bytes equal")
