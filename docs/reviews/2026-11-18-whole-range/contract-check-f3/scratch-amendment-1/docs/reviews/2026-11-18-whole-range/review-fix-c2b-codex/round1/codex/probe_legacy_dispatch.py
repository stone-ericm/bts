"""Capture real default-route callers and compare behavior with 386aa846."""

import argparse
from contextlib import redirect_stderr
from datetime import timedelta
from io import StringIO
import json
import logging
from pathlib import Path
from traceback import extract_tb

from probe_support import attempt_direct, day_pick, deadline, fixtures, raw, world_at


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.events = []

    def emit(self, record):
        error = record.exc_info
        frames = ([[frame.name, frame.line] for frame in extract_tb(error[2])]
                  if error and error[2] else [])
        self.events.append([
            record.name, record.levelname, record.getMessage(),
            type(error[1]).__name__ if error else None, str(error[1]) if error else None, frames,
        ])


def run_case(root, monkeypatch, opening, case, explicit_false):
    from bts import picks, scheduler
    from bts.health import alert, runner
    from tests.test_pick_chosen_on_cs5a import set_automatic
    from tests.test_saved_pick_paths_cs5a import scheduler_state

    day = "2027-07-15"
    hour = {"eod": "22:15", "cutoff": "18:56"}.get(case, "18:40")
    if case in {"morning_refused", "morning_stale"}:
        hour = "10:00"
    journal = StringIO()
    with world_at(root, monkeypatch, opening, opening.at(day, hour)) as (config, directory, now, statuses, scoped):
        directory = directory.relative_to(root)
        config["orchestrator"]["picks_dir"] = str(directory)
        capture = Capture()
        logger = logging.getLogger("bts.health.alert")
        scoped.setattr(logger, "handlers", [capture])
        scoped.setattr(logger, "propagate", False)
        scoped.setattr(logger, "level", logging.INFO)
        dispatcher = alert.dispatch_dm_for_health_alerts
        checkpoints = []

        def observe(*args, **kwargs):
            if explicit_false:
                kwargs["require_message_id"] = False
            result = dispatcher(*args, **kwargs)
            path = Path(kwargs["status_path"])
            payload = raw(path)
            checkpoints.append([path.name, result, payload.decode("utf-8") if payload is not None else None])
            return result

        scoped.setattr(alert, "dispatch_dm_for_health_alerts", observe)
        scoped.setattr(runner, "dispatch_dm_for_health_alerts", observe)
        daily = day_pick(day)
        result = None
        with redirect_stderr(journal):
            if case == "eod":
                daily.delivery_attempted = True
                picks.save_pick(daily, directory)
                scoped.setattr(runner, "_safe_run", lambda name, check: check() if name == "post_failure" else [])
                scoped.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
                result = [item.source for item in runner.run_all_checks(
                    directory, root / "models", "offline.test", today=now[0].date(),
                )]
            elif case == "4c_stale":
                scheduler._warn_stale_contest_reading({
                    "stale": True, "reason": "old", "recorded_at": "2027-07-12T17:00:00-04:00",
                    "age_hours": 73.67,
                }, config, directory)
            elif case in {"morning_refused", "morning_stale"}:
                if case == "morning_stale":
                    set_automatic(directory, now=now[0] - timedelta(hours=72))
                scheduler._morning_state_check(config, directory, day, now[0], now[0] + timedelta(hours=5))
            elif case in {"cutoff", "void"}:
                if case == "void":
                    statuses[daily.pick.game_pk] = {"abstract": "P", "detailed": "Cancelled", "code": "C"}
                result = attempt_direct(config, directory, daily)
            elif case in {"missed_pick", "missed_pick_unguarded"}:
                config["scheduler"]["contest_state_required"] = case == "missed_pick"
                picks.save_pick(daily, directory)
                state = scheduler_state()
                scheduler._maybe_alert_missed_pick(config, day, directory, 30, None, state)
            else:
                if case == "no_recipient":
                    config["bluesky"].pop("dm_recipient")
                elif case == "missing_id":
                    alert.send_dm.return_value = None
                elif case in {"exception", "exception_retry"}:
                    alert.send_dm.side_effect = [OSError("local fixture transport offline"), "ok"]
                scheduler._alert_contest_state_failure(config, ValueError("same synthetic contest reading error"))
                if case in {"duplicate", "exception_retry"}:
                    scheduler._alert_contest_state_failure(config, ValueError("same synthetic contest reading error"))
        assert checkpoints, case
        if case in {"no_recipient", "duplicate"}:
            assert checkpoints[-1][1] is False
        else:
            assert checkpoints[-1][1] is True
        if case == "exception_retry":
            assert alert.send_dm.call_args.args[1].startswith("⚠️ Previous health alert DM FAILED")
        response = {
            "dispatch_checkpoints": checkpoints,
            "transport": [list(msg.args) for msg in alert.send_dm.call_args_list],
            "log_messages": capture.events,
            "caller_return": result,
            "stderr": journal.getvalue(),
        }
    return response


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("version", choices=["base", "head"])
    parser.add_argument("comparison", type=Path)
    args = parser.parse_args()
    comparison = args.comparison.resolve()
    cases = ("eod", "4c_stale", "morning_refused", "morning_stale", "cutoff", "void",
             "missed_pick", "missed_pick_unguarded", "contest_state_failure", "no_recipient",
             "duplicate", "exception", "missing_id", "exception_retry")
    with deadline(45), fixtures("legacy") as (root, patch, opening):
        results = {}
        for case in cases:
            result = run_case(root / case, patch, opening, case, explicit_false=False)
            results[case] = result
            if args.version == "head":
                explicit = run_case(root / f"explicit-{case}", patch, opening, case, explicit_false=True)
                assert result == explicit, (case, result, explicit)
            print(f"{case}: sent={len(result['transport'])}, "
                  f"returns={[item[1] for item in result['dispatch_checkpoints']]}, "
                  f"logs={len(result['log_messages'])}", flush=True)
    if args.version == "base":
        comparison.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    else:
        base = json.loads(comparison.read_text())
        assert base == results, {key: (base[key], results[key]) for key in results if base[key] != results[key]}
        print("Base, omitted keyword and explicit False match.")


if __name__ == "__main__":
    main()
