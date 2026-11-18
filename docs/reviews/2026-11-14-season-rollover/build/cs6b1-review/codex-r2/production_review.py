"""Exercise real scheduler startup and full days without live services."""

from contextlib import contextmanager, redirect_stderr, redirect_stdout
from datetime import timedelta
from io import StringIO
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from threading import Event, Thread
from time import sleep

from review_support import REVIEW, ROOT, archive, environment

MORNING = "contest_morning_check_dm_delivery_status.json"
STALE = "contest_stale_reading_dm_delivery_status.json"
HEARTBEAT = "heartbeat"
FAILURE = "  Morning state check failed: "
REFUSAL = "  MORNING STATE CHECK — no pick can be made yet: "
PREFIXES = {
    "refusal_print": REFUSAL,
    "stale_print": "  MORNING STATE CHECK — stale contest reading: ",
    "dated_print": "  Morning state check: dated contest state accepted",
    "fresh_print": "  Morning state check: automatic contest state accepted",
    "due_print": "  Morning state check skipped: first lineup check",
    "empty_print": "  Morning state check skipped: no lineup check",
    "failure_print": FAILURE,
}
STATUS_SITES = {"status", "status_partial", "status_complete", "failed_status"}
SITES = [
    "decision", "pick", "state", "dispatcher", "transport", *PREFIXES,
    "status", "failed_status", "clock", "min", "required", "automatic_reading_record",
    "delivery_marker",
]
SIGNALS = [site for site in SITES if site not in {
    "clock", "min", "required", "automatic_reading_record", "delivery_marker"
}] + ["stalled_transport", "status_partial", "status_complete"]


class FirstPrediction(BaseException):
    pass


class Restart(BaseException):
    pass


class Unprintable(Exception):
    def __str__(self):
        raise RuntimeError("ordinary exception cannot be rendered")


def ordinary_exceptions():
    return [cls("injected ordinary fault") for cls in (
        RuntimeError, OSError, ValueError, TypeError, KeyError, AttributeError,
        RecursionError, MemoryError, UnicodeError,
    )] + [ExceptionGroup("ordinary group", [ValueError("member"), OSError("member")]), Unprintable()]


def files(root):
    return {str(path.relative_to(root)): path.read_bytes().hex()
            for path in sorted(root.rglob("*")) if path.is_file() and not path.is_symlink()}


class Journal(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append((record.getMessage(), self.format(record)))


@contextmanager
def setup(root):
    from pytest import MonkeyPatch

    from bts import contest_state, heartbeat, picks, scheduler, slate
    from bts.health import runner
    from tests.test_morning_state_cs6b1 import morning
    from tests.test_saved_pick_paths_cs5a import world

    journal = Journal()
    logger = logging.getLogger()
    level = logger.level
    logger.addHandler(journal)
    logger.setLevel(logging.INFO)
    stderr, stdout = StringIO(), StringIO()
    with MonkeyPatch.context() as patch, redirect_stderr(stderr), redirect_stdout(stdout):
        try:
            fixture = morning.__wrapped__(world.__wrapped__(root, patch), patch)
            patch.setattr(heartbeat, "datetime", contest_state.datetime)
            patch.setattr(slate, "datetime", contest_state.datetime)
            patch.setattr(picks, "_git_head_sha", lambda *_: "a" * 40)
            patch.setattr(runner, "get_self_pid", lambda: None)
            patch.setattr(runner, "read_systemd_nrestarts", lambda: None)
            patch.setattr(scheduler, "notify_ready", lambda: None)
            patch.setattr(scheduler, "notify_watchdog", lambda: None)
            fixture[0]["orchestrator"]["data_dir"] = str(root / "processed")
            yield patch, fixture, stderr, stdout, journal
        finally:
            logger.removeHandler(journal)
            logger.setLevel(level)


def inject(patch, module, name, fault, predicate=lambda *a, **kw: True):
    real = getattr(module, name)

    def replacement(*args, **kwargs):
        if predicate(*args, **kwargs):
            fault()
        return real(*args, **kwargs)

    patch.setattr(module, name, replacement)


def fault_case(root, site, error, base):
    from bts import contest_state, dm, orchestrator, picks, posting, scheduler
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, FIRST, START
    from tests.test_pick_chosen_on_cs5a import daily_pick, set_automatic, set_dated
    from tests.test_stale_delivery_cs4c import seed_automatic

    with setup(root) as (patch, (config, directory, now, _), stderr, _, journal):
        health_status = root / "health_state" / MORNING
        hits, checks, attempts, sends = [], [], [], []
        armed = False
        use_signal = error is None

        def fault():
            if hits:
                return
            hits.append(now[0].isoformat())
            if use_signal:
                signal.raise_signal(signal.SIGINT)
            else:
                raise error

        start_serial = scheduler.save_state

        def save(*args, **kwargs):
            nonlocal armed
            result = start_serial(*args, **kwargs)
            armed = True
            return result

        patch.setattr(scheduler, "save_state", save)
        if site == "decision":
            decision = directory / DAY / "decision.json"
            decision.parent.mkdir(parents=True)
            decision.write_text(json.dumps({
                "schema_version": "bts_daily_decision_v3", "date": DAY,
                "scoreable": False, "action": "skip",
            }))
        if site in {"pick", "delivery_marker"}:
            picks.save_pick(daily_pick(), directory)
        if site in {"stale_print", "automatic_reading_record"}:
            seed_automatic(directory, (START - timedelta(hours=20)).isoformat(), now=START)
        if site == "dated_print":
            set_dated(directory, now=START)
        if site == "fresh_print":
            set_automatic(directory, now=START)
        if site == "due_print":
            now[0] = FIRST - timedelta(minutes=9, seconds=59)
        if site == "empty_print" and not (base and use_signal):
            patch.setattr(scheduler, "plan_run_times", lambda *a, **kw: [])
        initial = now[0].isoformat()

        def predict(*args, **kwargs):
            checks.append(now[0].isoformat())
            raise FirstPrediction

        orchestrator.run_cascade.side_effect = predict

        def transport(recipient, body):
            attempts.append((recipient, body))
            if site == "stalled_transport" and not base and not hits:
                waiting = Event()

                def interrupt():
                    waiting.wait()
                    sleep(0.05)
                    os.kill(os.getpid(), signal.SIGINT)

                thread = Thread(target=interrupt)
                thread.start()
                try:
                    hits.append(now[0].isoformat())
                    waiting.set()
                    Event().wait(3)
                    raise AssertionError("SIGINT did not interrupt the stalled transport")
                finally:
                    thread.join()
            if site == "transport" and not base:
                fault()
            if site == "failed_status" and not hits:
                raise OSError("transport refused before status write")
            sends.append((recipient, body))
            return "capture"

        alert.send_dm.side_effect = transport
        if base and use_signal:
            inject(patch, scheduler, "_watchdog_ping_sleep", fault)
        elif not base:
            if site in {"decision", "pick"}:
                target = decision if site == "decision" else directory / f"{DAY}.json"
                inject(patch, Path, "read_text", fault, lambda path, *a, **k: path == target and armed)
            if site == "state":
                inject(patch, contest_state, "load_decision_streak_state", fault)
            if site == "dispatcher":
                inject(patch, alert, "dispatch_dm_for_health_alerts", fault)
            if site in PREFIXES:
                if site == "failure_print":
                    def state_read_failure(*args, **kwargs):
                        raise RuntimeError("state read failed")

                    patch.setattr(contest_state, "load_decision_streak_state", state_read_failure)
                inject(patch, stderr, "write", fault, lambda line: line.startswith(PREFIXES[site]))
            if site in STATUS_SITES:
                write = Path.write_text

                def status_write(path, body, *args, **kwargs):
                    if path == health_status and not hits:
                        if site == "status_partial":
                            write(path, body[:len(body) // 2], *args, **kwargs)
                        if site == "status_complete":
                            write(path, body, *args, **kwargs)
                        fault()
                    return write(path, body, *args, **kwargs)

                patch.setattr(Path, "write_text", status_write)
            if site == "clock":
                inject(patch, scheduler, "_now_et", fault, lambda: armed)
            if site == "min":
                import builtins

                def minimum(*args, **kwargs):
                    if armed:
                        fault()
                    return builtins.min(*args, **kwargs)

                patch.setattr(scheduler, "min", minimum, raising=False)
            if site == "required":
                inject(patch, scheduler, "_contest_state_required", fault, lambda *_: armed)
            if site == "automatic_reading_record":
                inject(patch, contest_state, "automatic_reading_record", fault)
            if site == "delivery_marker":
                inject(patch, picks, "pick_was_delivered", fault, lambda *_: armed)

        escaped = None
        try:
            scheduler.run_day(DAY, config)
        except BaseException as caught:
            escaped = type(caught).__name__
        assert not dm.send_dm.called and not posting.post_to_bluesky.called
        snapshot = files(root)
        status = health_status.read_text() if health_status.exists() else None
        result = dict(escaped=escaped, hits=hits.copy(), checks=checks.copy(),
                      attempts=attempts.copy(), sends=sends.copy(), files=snapshot, status=status,
                      first=FIRST.isoformat(), initial=initial, stderr=stderr.getvalue(),
                      logs=journal.records.copy())
        if use_signal and not base and site in STATUS_SITES | {"stalled_transport"}:
            now[0] = START
            sends.clear()
            try:
                scheduler.run_day(DAY, config)
            except FirstPrediction:
                pass
            else:
                raise AssertionError("restart did not reach its first prediction")
            result["restart_sends"] = len(sends)
        return result


def compare_faults(root, base=False):
    series = {}
    for site in SIGNALS:
        path = root / f"signal-{site}"
        series[f"signal-{site}"] = fault_case(path, site, None, base)
    for site in SITES:
        for fault in ordinary_exceptions()[:1] if base else ordinary_exceptions():
            key = f"{site}-{type(fault).__name__}"
            series[key] = fault_case(root / key, site, fault, base)
    return series


def analyze_faults(base, head):
    for key, current in head.items():
        site = key.removeprefix("signal-") if key.startswith("signal-") else key.split("-")[0]
        previous = base[key] if key.startswith("signal-") else base[f"{site}-RuntimeError"]
        expected_files = dict(previous["files"])
        actual_files = dict(current["files"])
        actual_files.pop(f"health_state/{MORNING}", None)
        actual_files.pop(f"health_state/{STALE}", None)
        if key.startswith("signal-"):
            assert previous["hits"] == current["hits"] == [current["initial"]], key
            assert previous["escaped"] == current["escaped"] == "KeyboardInterrupt", key
            assert not previous["checks"] and not current["checks"], key
            assert not previous["attempts"] and not previous["sends"], key
            expected_hb = json.loads(bytes.fromhex(expected_files.pop(HEARTBEAT)))
            actual_hb = json.loads(bytes.fromhex(actual_files.pop(HEARTBEAT)))
            assert expected_hb["state"] == "sleeping" and actual_hb["state"] == "running", (key, expected_hb, actual_hb)
            assert len(current["attempts"]) == (site in STATUS_SITES | {"stalled_transport", "transport"}), key
            assert len(current["sends"]) == (site in {"status", "status_partial", "status_complete"}), key
            if site == "status_complete":
                assert json.loads(current["status"])["status"] == "sent", key
            elif site == "status_partial":
                try:
                    json.loads(current["status"])
                except json.JSONDecodeError:
                    pass
                else:
                    raise AssertionError("partial status unexpectedly parsed")
            else:
                assert current["status"] is None, key
            if "restart_sends" in current:
                assert current["restart_sends"] == (0 if site == "status_complete" else 1), key
        else:
            assert current["hits"] == [current["initial"]] and not previous["hits"], key
            assert current["checks"] == previous["checks"], key
            if site == "empty_print":
                assert current["escaped"] is None and not current["checks"], key
            else:
                assert current["checks"] == [current["first"]], (key, current)
                assert current["escaped"] == previous["escaped"] == "FirstPrediction", key
        assert actual_files == expected_files, (key, sorted(actual_files), sorted(expected_files))
        print(f"{key}: escape={current['escaped']}, checks={current['checks']}, "
              f"attempts={len(current['attempts'])}, sends={len(current['sends'])}, "
              f"restart_sends={current.get('restart_sends')}")


def record_cases(root):
    from bts import contest_state, daily_decision, dm, picks, posting, scheduler
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import DAY, FIRST, MORNING_KEY, START, refusal
    from tests.test_pick_chosen_on_cs5a import daily_pick

    def decision(action, scoreable):
        return json.dumps({"schema_version": "bts_daily_decision_v3", "date": DAY,
                           "action": action, "scoreable": scoreable}).encode()

    shapes = {
        "list-action": decision([], True), "dict-action": decision({}, True),
        "text-scoreable": decision("single", "yes"),
        "nested-20000": b"[" * 20000 + b"]" * 20000,
        "nested-100000": b"[" * 100000 + b"]" * 100000,
        "invalid-utf8": b"\xff\xfe", "directory": None, "symlink-loop": None,
        "empty": b"", "ascii-50MB-zeroes": b"0" * 50_000_000,
        "binary-50MB-zeroes": bytes(50_000_000),
        "null": b"null", "true": b"true", "false": b"false", "integer": b"42",
        "float": b"1.25", "string": b'"text"',
    }
    for kind in ("decision", "pick"):
        for shape in [*shapes, "readable-commit", "failed-read-of-commit"]:
            path = root / f"{kind}-{shape}"
            with setup(path) as (patch, (config, directory, now, _), stderr, _, journal):
                target = (directory / DAY / "decision.json" if kind == "decision"
                          else directory / f"{DAY}.json")
                target.parent.mkdir(parents=True)
                if shape in {"readable-commit", "failed-read-of-commit"}:
                    if kind == "decision":
                        target.write_bytes(decision("single", True))
                        assert daily_decision.is_scoreable_commit(DAY, directory, None)
                    else:
                        daily = daily_pick()
                        daily.notification_sent = True
                        daily.notification_id = "prior"
                        picks.save_pick(daily, directory)
                        assert picks.pick_was_delivered(picks.load_pick(DAY, directory))
                    if shape == "failed-read-of-commit":
                        def unreadable():
                            raise PermissionError("captured denied read of a valid commit")

                        inject(patch, Path, "read_text", unreadable, lambda p, *a, **kw: p == target)
                elif shape == "directory":
                    target.mkdir()
                elif shape == "symlink-loop":
                    target.symlink_to(target.name)
                else:
                    target.write_bytes(shapes[shape])
                reads = []
                inject(patch, contest_state, "load_decision_streak_state", lambda: reads.append(now[0]))

                def stop(*args, **kwargs):
                    assert now[0] == FIRST
                    raise FirstPrediction

                patch.setattr(scheduler, "run_single_check", stop)
                try:
                    scheduler.run_day(DAY, config)
                except FirstPrediction:
                    pass
                else:
                    raise AssertionError("no first check handoff")
                status = path / "health_state" / MORNING
                if shape == "readable-commit":
                    assert not reads and not alert.send_dm.called and not status.exists()
                else:
                    assert reads == [START], (kind, shape)
                    alert.send_dm.assert_called_once_with("offline.test", refusal())
                    assert json.loads(status.read_bytes())["sent_sources"] == [MORNING_KEY]
                    assert sum(line.startswith(REFUSAL) for line in stderr.getvalue().splitlines()) == 1
                assert FAILURE not in stderr.getvalue()
                assert all(FAILURE not in text for _, text in journal.records)
                assert not dm.send_dm.called and not posting.post_to_bluesky.called
                print(f"{kind}/{shape}: state reads={len(reads)}; morning DMs={alert.send_dm.call_count}",
                      file=sys.__stdout__)


def message_cases(root):
    from bts import contest_state, orchestrator, scheduler
    from bts.health import alert
    from tests.test_morning_state_cs6b1 import BANNER, DAY, START, refusal

    output = {}
    for kind in ("missing-switch", "multiline"):
        case = root / kind
        with setup(case) as (patch, (config, directory, now, _), stderr, _, journal):
            if kind == "missing-switch":
                try:
                    contest_state.load_decision_streak_state(directory, require_contest_state=True, now=START)
                except contest_state.ContestStateError as caught:
                    refusal_text = str(caught)
                else:
                    raise AssertionError("expected real missing-switch refusal")
            else:
                refusal_text = "refusal-sentinel-é\n" + "à" * (1040 - 20) + "fin"
                refusal_text = (refusal_text + "界" * 1040)[:1040]
                error = contest_state.ContestStateError(refusal_text)

                def refuse(*args, **kwargs):
                    raise error

                patch.setattr(contest_state, "load_decision_streak_state", refuse)
                assert len(refusal_text) == 1040 and "\n" in refusal_text

            alert.send_dm.side_effect = [OSError("captured transport failure"), "retry-sent"]
            orchestrator.run_cascade.return_value = (None, None)
            status = case / "health_state" / MORNING
            scheduler.run_day(DAY, config)
            failed = json.loads(status.read_bytes())
            first_journal = stderr.getvalue()
            first_logs = journal.records.copy()
            assert failed["status"] == "failed"
            now[0] = START
            scheduler.run_day(DAY, config)
            assert [call.args for call in alert.send_dm.call_args_list] == [
                ("offline.test", refusal()), ("offline.test", BANNER + "\n\n" + refusal()),
            ]
            output[kind] = dict(
                refusal=refusal_text, bound=ascii(refusal_text)[:800],
                journal=first_journal, logs=first_logs, failure=failed,
                retry=json.loads(status.read_bytes()),
            )
    return output


def analyze_messages(old, head):
    for key, current in head.items():
        previous = old[key]
        assert current["failure"] == previous["failure"] and current["retry"] == previous["retry"]
        assert [message for message, _ in current["logs"]] == [message for message, _ in previous["logs"]]
        exact = REFUSAL + current["bound"]
        assert current["journal"].splitlines().count(exact) == 1
        assert current["journal"].replace(exact, "").find(current["bound"]) == -1
        for side, replay in (("round1", previous), ("head", current)):
            logs = "\n".join(text for _, text in replay["logs"])
            context = "During handling of the above exception" in logs
            assert context == (side == "round1"), (key, side)
            assert (replay["refusal"] in logs) == (side == "round1"), (key, side)
            assert "captured transport failure" in logs
        print(f"{key}: {len(current['bound'])} rendered characters; stock failure log, status and retry match")


def one_day(root, name):
    from bts import daily_decision, dm, orchestrator, picks, posting, scheduler
    from bts.health import alert, runner
    from tests.test_morning_state_cs6b1 import DAY, FIRST, START
    from tests.test_pick_chosen_on_cs5a import daily_pick, record, set_automatic, set_dated
    from tests.test_stale_delivery_cs4c import seed_automatic

    with setup(root) as (patch, (config, directory, now, (frame, tier)), stderr, stdout, _):
        events = []

        def event(label, value=None):
            events.append((label, now[0].isoformat(), value))

        def trace(module, attr, description):
            original = getattr(module, attr)

            def recorded(*args, **kwargs):
                event(attr, description(*args, **kwargs))
                return original(*args, **kwargs)

            patch.setattr(module, attr, recorded)

        trace(scheduler, "write_heartbeat", lambda *a, **k: [str(k["state"]), str(k.get("sleeping_until"))])
        trace(scheduler, "run_single_check", lambda **k: k["date"])
        trace(scheduler, "run_result_polling", lambda pk, day, *a, **k: [pk, day])
        trace(runner, "run_all_checks", lambda **k: k["today"].isoformat())
        if name in {"dated", "skip", "thin", "restart-delivered", "restart-midday"}:
            set_dated(directory, now=START, streak=30, best=30)
        if name == "fresh":
            set_automatic(directory, now=START, streak=30)
        if name == "stale":
            seed_automatic(directory, (START - timedelta(hours=20)).isoformat(), now=START, streak=30)
        if name == "skip":
            frame = frame.assign(p_game_hit=0.1)
        if name == "no-games":
            scheduler.fetch_schedule.side_effect = None
        if name == "thin":
            now[0] = FIRST - timedelta(minutes=9, seconds=59)
            game = scheduler.fetch_schedule(DAY)[0]
            scheduler.fetch_schedule.side_effect = lambda date: [game] if date == DAY else []
            frame = frame.iloc[:1]
        if name in {"refused", "refused-preview"}:
            config["health_checks"]["enabled"] = True
        if name == "refused-preview":
            preview = daily_pick()
            preview.chosen_on = record()
            picks.save_pick(preview, directory)

        def predict(*args, **kwargs):
            event("predict")
            return frame, tier

        orchestrator.run_cascade.side_effect = predict
        alert.send_dm.side_effect = lambda *args: event("health dm", args) or "health"
        dm.send_dm.side_effect = lambda *args: event("pick dm", args) or "pick"
        posting.post_to_bluesky.side_effect = lambda *args, **kw: event("post", args) or "post"

        if name == "restart-delivered":
            scheduler.run_day(DAY, config)
            assert daily_decision.load_decision(DAY, directory)["scoreable"]
            now[0] = FIRST + timedelta(minutes=5)
            event("restart")
        if name == "restart-midday":
            sleep = scheduler._watchdog_ping_sleep

            def stop_between_checks(seconds):
                if now[0] == FIRST:
                    raise Restart
                sleep(seconds)

            with patch.context() as first_start:
                first_start.setattr(scheduler, "_watchdog_ping_sleep", stop_between_checks)
                first_start.setattr(orchestrator.run_cascade, "side_effect", lambda *a, **k: (None, None))
                try:
                    scheduler.run_day(DAY, config)
                except Restart:
                    assert scheduler.load_state(DAY, directory).runs_completed
                else:
                    raise AssertionError("restart simulation missed the checkpoint")
            now[0] = FIRST + timedelta(minutes=5)
            event("restart")
        scheduler.run_day(DAY, config)
        result = daily_decision.load_decision(DAY, directory)
        if name in {"refused", "refused-preview", "no-games"}:
            assert result is None and not dm.send_dm.called
        elif name == "skip":
            assert result["action"] == "skip" and not dm.send_dm.called
        else:
            assert result["scoreable"] and dm.send_dm.call_count == 1
        if name == "refused-preview":
            assert any("NO PICK DELIVERED" in call.args[1] for call in alert.send_dm.call_args_list)
        return dict(files=files(root), events=events, stdout=stdout.getvalue(), stderr=stderr.getvalue())


def clean_day(snapshot, name):
    skip_files = [MORNING, STALE] if name == "stale" else [MORNING]
    return {
        **snapshot,
        "files": {path: body for path, body in snapshot["files"].items()
                  if Path(path).name not in skip_files},
        "events": [entry for entry in snapshot["events"] if not (
            entry[0] == "health dm" and (
                "Morning check for " in entry[2][1]
                or (name == "stale" and "The pick was delivered using" in entry[2][1])
            )
        )],
        "stderr": "".join(line for line in snapshot["stderr"].splitlines(keepends=True)
                          if not line.startswith(("  Morning state check", "  MORNING STATE CHECK"))),
    }


def check_day(base, head, name):
    left, right = clean_day(base, name), clean_day(head, name)
    difference = [key for key in left if left[key] != right[key]]
    assert not difference, (name, difference)
    head_morning = [line for line in head["stderr"].splitlines()
                    if line.startswith(("  Morning state check", "  MORNING STATE CHECK"))]
    assert len(head_morning) == (0 if name == "no-games" else 2 if name == "restart-midday" else 1), (
        name, head_morning)
    if name == "stale":
        from datetime import datetime, timezone

        base_status, head_status = [json.loads(bytes.fromhex(x["files"][f"health_state/{STALE}"]))
                                    for x in (base, head)]
        unchanged = {"sent_day", "sent_sources", "status", "critical_count", "warn_attention_count",
                     "body_first_line", "recipient"}
        assert all(base_status[key] == head_status[key] for key in unchanged)
        warnings = [[event for event in snap["events"] if event[0] == "health dm" and
                     event[2][1].startswith("BTS health WARN attention:")] for snap in (base, head)]
        assert [len(entry) for entry in warnings] == [1, 1]
        assert [datetime.fromisoformat(entry[0][1]).hour for entry in warnings] == [16, 10]
        for status, events in zip((base_status, head_status), warnings, strict=True):
            assert datetime.fromisoformat(status["updated_at"]) == datetime.fromisoformat(
                events[0][1]).astimezone(timezone.utc)
    print(f"{name}: {len(left['files'])} identical files, {len(left['events'])} identical events; "
          f"{sum(event[0] == 'run_single_check' for event in left['events'])} checks")


def run(scratch, source, mode, identity, *args):
    log_name = f"{mode}-{identity}"
    case = scratch / "inputs"
    command = [sys.executable, __file__, "child", mode, str(case), identity, *args]
    env = environment(source, log_name)
    completed = subprocess.run(command, cwd=ROOT, env=env, capture_output=True, text=True)
    (REVIEW / f"{log_name}.log").write_text(completed.stderr)
    (REVIEW / f"{log_name}.json").write_text(completed.stdout)
    if case.exists():
        shutil.rmtree(case)
    assert completed.returncode == 0, (log_name, completed.stderr)
    assert not Path(env["CS6B1R2_DENIED"]).exists(), log_name
    return json.loads(completed.stdout) if mode != "records" else completed.stdout


def main():
    assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
    with TemporaryDirectory(prefix="production-", dir=REVIEW) as temp:
        scratch = Path(temp)
        base, old, head = archive("1bf518a2", scratch / "base"), archive(
            "0e0c49a3", scratch / "round1"), ROOT / "src"
        analyze_faults(run(scratch, base, "faults", "base"), run(scratch, head, "faults", "head"))
        print(run(scratch, head, "records", "head"), end="")
        analyze_messages(run(scratch, old, "messages", "round1"), run(scratch, head, "messages", "head"))
        for name in ("dated", "fresh", "stale", "refused", "refused-preview", "skip", "no-games",
                     "thin", "restart-delivered", "restart-midday"):
            check_day(run(scratch, base, name, "base"), run(scratch, head, name, "head"), name)


if __name__ == "__main__":
    if len(sys.argv) == 1:
        main()
    else:
        from bts import scheduler

        assert Path(scheduler.__file__).is_relative_to(os.environ["CS6B1R2_SOURCE"])
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler
        _, _, mode, root, identity, *_ = sys.argv
        root = Path(root)
        if mode == "faults":
            print(json.dumps(compare_faults(root, base=(identity == "base")), indent=2))
        elif mode == "records":
            record_cases(root)
        elif mode == "messages":
            print(json.dumps(message_cases(root), indent=2))
        else:
            print(json.dumps(one_day(root, mode), indent=2))
