"""Run actual scheduler days against archived base and HEAD with identical paths."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import timedelta
from io import StringIO
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from offline_review import offline


REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
MORNING = "contest_morning_check_dm_delivery_status.json"
STALE = "contest_stale_reading_dm_delivery_status.json"
SHARED = "health_dm_delivery_status.json"


class CapturedRestart(BaseException):
    pass


@contextmanager
def sandbox(root):
    from pytest import MonkeyPatch
    from bts import contest_state, heartbeat, orchestrator, picks, scheduler, slate
    from bts.health import runner
    from tests.test_morning_state_cs6b1 import morning
    from tests.test_saved_pick_paths_cs5a import world

    with offline(), MonkeyPatch.context() as patch:
        scenario = morning.__wrapped__(world.__wrapped__(root, patch), patch)
        config, directory, now, _ = scenario
        events = []
        patch.setattr(heartbeat, "datetime", contest_state.datetime)
        patch.setattr(slate, "datetime", contest_state.datetime)
        patch.setattr(picks, "_git_head_sha", lambda *_: "a" * 40)
        patch.setattr(scheduler, "notify_ready", lambda: events.append(["ready", now[0].isoformat()]))
        patch.setattr(scheduler, "notify_watchdog", lambda: events.append(["watchdog", now[0].isoformat()]))
        patch.setattr(runner, "get_self_pid", lambda: None)
        patch.setattr(runner, "read_systemd_nrestarts", lambda: None)
        config["orchestrator"]["data_dir"] = str(root / "processed")

        def watch(module, name, describe):
            original = getattr(module, name)

            def call(*args, **kwargs):
                events.append([name, now[0].isoformat(), describe(*args, **kwargs)])
                return original(*args, **kwargs)

            patch.setattr(module, name, call)

        watch(scheduler, "write_heartbeat", lambda *a, **kw: [
            str(kw["state"]), str(kw.get("sleeping_until")),
        ])
        watch(scheduler, "run_single_check", lambda date, *a, **kw: date)
        watch(scheduler, "_watchdog_ping_sleep", lambda seconds: seconds)
        watch(scheduler, "run_result_polling", lambda pk, day, *a, **kw: [pk, day])
        watch(scheduler, "_idle_until_next_wakeup", lambda when, *_: when)
        watch(runner, "run_all_checks", lambda **kw: kw["today"].isoformat())
        yield patch, config, directory, now, scenario[-1], events


def snapshot(root, events, stderr, stdout):
    return {
        "files": {str(p.relative_to(root)): p.read_bytes().hex()
                  for p in sorted(root.rglob("*")) if p.is_file()},
        "events": events.copy(), "stderr": stderr.getvalue(), "stdout": stdout.getvalue(),
    }


def scenario(root, name):
    from bts import daily_decision, dm, orchestrator, picks, posting, scheduler
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, FIRST, START
    from tests.test_pick_chosen_on_cs5a import daily_pick, record, set_automatic, set_dated
    from tests.test_stale_delivery_cs4c import seed_automatic

    stderr, stdout = StringIO(), StringIO()
    with sandbox(root) as (patch, config, directory, now, (frame, tier), events):
        with redirect_stderr(stderr), redirect_stdout(stdout):
            def predict(*args, **kwargs):
                events.append(["predict", now[0].isoformat(), None])
                return frame, tier

            def health_send(recipient, body):
                events.append(["health dm", now[0].isoformat(), [recipient, body]])
                if name == "eod_failed" and body.startswith("BTS health CRITICAL alert(s):\n- [contest_state] Morning"):
                    raise OSError("captured morning DM failure")

            orchestrator.run_cascade.side_effect = predict
            alert.send_dm.side_effect = health_send
            dm.send_dm.side_effect = lambda *args: events.append(["pick dm", now[0].isoformat(), args]) or "sent"
            posting.post_to_bluesky.side_effect = lambda *args, **kw: events.append(
                ["post", now[0].isoformat(), str(args)]) or "at://post"

            if name in {"dated", "skip", "thin", "delivered_restart", "midday_restart"}:
                set_dated(directory, now=START, streak=30, best=30)
            if name == "automatic":
                set_automatic(directory, now=START, streak=30)
            if name == "stale":
                seed_automatic(directory, (START - timedelta(hours=20)).isoformat(), now=START, streak=30)
            if name == "skip":
                frame = frame.assign(p_game_hit=0.1)
            if name == "off":
                scheduler.fetch_schedule.side_effect = None
            if name == "thin":
                now[0] = FIRST - timedelta(minutes=9, seconds=59)
                first_game = scheduler.fetch_schedule(DAY)[0]
                scheduler.fetch_schedule.side_effect = lambda day: [first_game] if day == DAY else []
                frame = frame.iloc[:1]
            if name in {"refused", "maybe_preview", "eod_failed"}:
                config["health_checks"]["enabled"] = True
            if name == "eod_failed":
                frame, tier = None, None
            if name == "maybe_preview":
                preview = daily_pick()
                preview.chosen_on = record("dated")
                picks.save_pick(preview, directory)
            if name == "midday_restart":
                sleep = scheduler._watchdog_ping_sleep

                def stop_after_first(seconds):
                    if now[0] == FIRST:
                        raise CapturedRestart()
                    return sleep(seconds)

                with patch.context() as first_start:
                    first_start.setattr(scheduler, "_watchdog_ping_sleep", stop_after_first)
                    first_start.setattr(orchestrator.run_cascade, "side_effect", lambda *a, **kw: (None, None))
                    try:
                        scheduler.run_day(DAY, config)
                    except CapturedRestart:
                        assert scheduler.load_state(DAY, directory).runs_completed
                        assert picks.load_pick(DAY, directory) is None
                    else:
                        raise AssertionError("the first scheduler invocation did not stop mid-day")
                now[0] = FIRST + timedelta(minutes=5)
                events.append(["restart", now[0].isoformat(), None])
            if name == "delivered_restart":
                scheduler.run_day(DAY, config)
                assert daily_decision.load_decision(DAY, directory)["scoreable"]
                now[0] = FIRST + timedelta(minutes=5)
                events.append(["restart", now[0].isoformat(), None])

            scheduler.run_day(DAY, config)
            result = daily_decision.load_decision(DAY, directory)
            if name in {"dated", "automatic", "stale", "delivered_restart", "midday_restart", "thin"}:
                assert result and result["scoreable"], (name, stderr.getvalue())
                assert sum(event[0] == "pick dm" for event in events) == 1, (name, events)
            if name == "skip":
                assert result and result["action"] == "skip", stderr.getvalue()
            if name in {"refused", "maybe_preview", "off", "eod_failed"}:
                assert result is None and not dm.send_dm.called, (name, stderr.getvalue())
            if name == "maybe_preview":
                assert any("NO PICK DELIVERED" in call.args[1] for call in alert.send_dm.call_args_list), stderr.getvalue()
            return snapshot(root, events, stderr, stdout)


def cleaned(snapshot, *, is_stale=False, eod_failure=False):
    data = dict(snapshot)
    data["files"] = {key: value for key, value in data["files"].items()
                     if not key.endswith(MORNING) and not (is_stale and key.endswith(STALE))}
    data["events"] = [event for event in data["events"] if not (
        event[0] == "health dm" and ("Morning check for" in event[2][1]
                                     or (is_stale and "pick was delivered using" in event[2][1]))
    )]
    data["stderr"] = "".join(line for line in data["stderr"].splitlines(keepends=True)
                             if not line.startswith(("  Morning state check", "  MORNING STATE CHECK")))
    if eod_failure:
        data["stderr"], matched = re.subn(
            r"^send_dm failed for health alerts; CRITICAL/WARN visibility may be lost: captured morning DM failure\n"
            r".*?^\[health_dm_delivery CRITICAL\] health alert DM delivery failed\n",
            "", data["stderr"], flags=re.MULTILINE | re.DOTALL,
        )
        assert matched == 1
    return data


def compare():
    cases = ["dated", "automatic", "refused", "maybe_preview", "skip", "off", "thin",
             "delivered_restart", "midday_restart", "stale", "eod_failed"]
    with TemporaryDirectory(prefix="cs6b1-day-") as scratch:
        for case in cases:
            root = Path(scratch) / case
            snapshots = []
            for identity, source in (("base", REVIEW / "base" / "src"), ("head", ROOT / "src")):
                root.mkdir()
                result = subprocess.run([sys.executable, __file__, case, str(root)],
                                        cwd=ROOT, capture_output=True, text=True,
                                        env={**os.environ, "PYTHONPATH": f"{source}:{ROOT}"})
                if result.returncode:
                    raise RuntimeError(f"{identity} {case}: {result.stderr}")
                (REVIEW / f"day-{identity}-{case}.json").write_text(result.stdout)
                snapshots.append(json.loads(result.stdout))
                shutil.rmtree(root)
            left = cleaned(snapshots[0], is_stale=(case == "stale"))
            right = cleaned(snapshots[1], is_stale=(case == "stale"), eod_failure=(case == "eod_failed"))
            differences = [key for key in left if left[key] != right[key]]
            assert not differences, (case, differences)
            checks = [event[1] for event in left["events"] if event[0] == "run_single_check"]
            print(f"{case}: matched {len(left['files'])} files, {len(left['events'])} events; checks {checks}", flush=True)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        compare()
    else:
        from bts import scheduler

        assert Path(scheduler.__file__).is_relative_to(os.environ["PYTHONPATH"].split(":")[0])
        print(json.dumps(scenario(Path(sys.argv[2]), sys.argv[1]), indent=2))
