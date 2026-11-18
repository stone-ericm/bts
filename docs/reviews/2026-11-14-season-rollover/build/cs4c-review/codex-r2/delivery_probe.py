"""Replay real pre-4c and current scheduler delivery with captured transports."""

import inspect
import io
import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr
from dataclasses import asdict, replace
from datetime import datetime, timedelta
from itertools import product
from pathlib import Path
from unittest.mock import Mock

from pytest_review import CURRENT, REVIEW, ROOT, archive


MODES = ("dm", "public")
READINGS = ("fresh", "old", "future", "unreadable", "dated", "none")
MARKERS = ("unmarked", "delivered", "attempted")
SELECTIONS = ("fresh", "saved-matching", "saved-refused", "none")
FAULTS = ("calculation", "dispatcher", "warning-transport", "warning-stderr",
          "diagnostic-print", "warning-clock", "status-write")
CORE_FAULTS = ("pick-transport", "attempt-save", "confirmed-save", "save-state",
               "record-commit", "capture", "cutoff", "final-cutoff-clock")
FAKE_IDS = (None, "", 0, False, "valid-id", "   ")
WIRE_IDS = (None, "", "missing", "valid-wire-id", 19, False, "   ")
DIAGNOSTIC = "  Stale-reading warning failed:"


class ReviewInterrupt(BaseException):
    pass


class BadString(Exception):
    def __str__(self):
        raise ReviewInterrupt("exception rendering interrupted")


ERRORS = {
    "keyboard": lambda: KeyboardInterrupt("injected"),
    "exit": lambda: SystemExit("injected"),
    "generator": lambda: GeneratorExit("injected"),
    "custom": lambda: ReviewInterrupt("injected"),
    "group": lambda: BaseExceptionGroup("injected", [KeyboardInterrupt()]),
    "bad-string": BadString,
}


def run_case(home, mp, mode, reading="old", required=True, marker="unmarked",
             choice="fresh", fault=None, error_name=None, identifier="sent", wire=False,
             is_base=False, omit_recipient=False):
    from bts import contest_state, daily_decision, dm, picks, posting, scheduler
    from bts.health import alert
    from tests.conftest import stamp_saved_pick
    from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick, set_dated
    from tests.test_saved_pick_paths_cs5a import scheduler_state
    from tests.test_stale_delivery_cs4c import PREFIX, WARNING_FILE, fresh, seed_automatic

    directory = home / "picks"
    config = {
        "orchestrator": {"picks_dir": str(directory)},
        "scheduler": {"contest_state_required": required, "pick_delivery": mode},
        "bluesky": {} if omit_recipient else {"dm_recipient": "operator.test"},
    }

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    for mod in (contest_state, daily_decision, picks, posting, alert, scheduler):
        mp.setattr(mod, "datetime", Clock)
    mp.setattr(scheduler, "_now_et", lambda: NOW)
    mp.setattr(scheduler, "_delivery_game_statuses", Mock(return_value={}))
    mp.setattr(scheduler, "fetch_schedule", Mock(return_value=[]))
    mp.setattr(scheduler, "_trigger_live_forward_capture_on_lock", capture := Mock())
    mp.setattr(alert, "send_dm", warning_transport := Mock(return_value="warn-id"))
    for module, name in ((dm, "get_dm_password"), (posting, "get_bluesky_password")):
        mp.setattr(module, name, lambda: "offline-test-only")

    sends = []
    attempt_records = []
    faults = []
    state = scheduler_state()

    def trip():
        faults.append(fault)
        if fault == "real-sigint":
            signal.raise_signal(signal.SIGINT)
        raise ERRORS[error_name]()

    def log_send(*args):
        sends.append(args)
        attempt = json.loads((directory / f"{DAY}.json").read_bytes())
        assert attempt["delivery_attempted"] is True
        attempt_records.append(attempt.get("automatic_reading"))
        if fault == "pick-transport":
            trip()

    def fake_transport(*args):
        log_send(*args)
        return identifier

    def http(req, **options):
        method = req.full_url.split("/xrpc/")[1].split("?")[0]
        if method == "com.atproto.server.createSession":
            result = {"accessJwt": "offline-token", "did": "did:plc:offline"}
        elif method == "com.atproto.identity.resolveHandle":
            result = {"did": "did:plc:recipient"}
        elif method == "chat.bsky.convo.getConvoForMembers":
            result = {"convo": {"id": "offline-convo"}}
        else:
            terminal = {"dm": "chat.bsky.convo.sendMessage", "public": "com.atproto.repo.createRecord"}
            assert method == terminal[mode] and options["idempotent"] is False
            log_send(method, json.loads(req.data))
            key = "id" if mode == "dm" else "uri"
            result = {} if identifier == "missing" else {key: identifier}
        return io.BytesIO(json.dumps(result).encode())

    if wire:
        mp.setattr(dm, "retry_urlopen", http)
        mp.setattr(posting, "retry_urlopen", http)
    else:
        mp.setattr(dm, "send_dm", fake_transport)
        mp.setattr(posting, "post_to_bluesky", fake_transport)

    if reading == "dated":
        set_dated(directory)
    elif reading != "none":
        times = {"fresh": (NOW - timedelta(hours=3)).isoformat(),
                 "old": (NOW - timedelta(hours=20)).isoformat(),
                 "future": (NOW + timedelta(minutes=20)).isoformat(),
                 "unreadable": "unreadable"}
        seed_automatic(directory, times[reading])

    daily = daily_pick()
    if choice in {"saved-matching", "saved-refused"}:
        if choice == "saved-refused" or reading == "none":
            original = home / "selection-account"
            if reading == "dated":
                seed_automatic(original)
            else:
                set_dated(original)
        else:
            original = directory
        choice_state = contest_state.load_decision_streak_state(original, require_contest_state=True)
        stamp_saved_pick.__wrapped__()(daily, choice_state)
    if marker == "delivered":
        daily.notification_sent = True
        daily.notification_channel = "bluesky_dm"
        daily.notification_id = "prior-dm"
        daily.delivered_at = NOW.isoformat()
    if marker != "unmarked":
        daily.delivery_attempted = True
    if marker != "unmarked" or choice.startswith("saved-"):
        picks.save_pick(daily, directory)
        daily = picks.load_pick(DAY, directory)
    if choice == "fresh":
        selection = fresh(daily)
    elif choice == "saved-matching":
        selection = replace(fresh(daily), decision=None)
    else:
        selection = None

    if fault == "calculation":
        mp.setattr(contest_state, "automatic_reading_record", lambda *_: trip())
    if fault == "dispatcher":
        mp.setattr(alert, "dispatch_dm_for_health_alerts", lambda *a, **k: trip())
    if fault == "warning-transport" or (fault == "real-sigint" and not is_base):
        warning_transport.side_effect = lambda *_: trip()
    if fault == "diagnostic-print":
        def fail_dispatch(*args, **kwargs):
            raise RuntimeError("make scheduler print its failure diagnostic")

        mp.setattr(alert, "dispatch_dm_for_health_alerts", fail_dispatch)
    if fault in {"warning-stderr", "diagnostic-print"}:
        blocked = PREFIX if fault == "warning-stderr" else DIAGNOSTIC

        def print_or_interrupt(*args, **kwargs):
            if str(args[0]).startswith(blocked):
                trip()
            print(*args, **kwargs)

        mp.setattr(scheduler, "print", print_or_interrupt, raising=False)
    if fault == "warning-clock" or fault == "final-cutoff-clock":
        expected_read = 5 if fault == "warning-clock" else 2
        samples = iter(range(1, expected_read + 1))

        def sample():
            if next(samples, None) == expected_read:
                trip()
            return NOW

        mp.setattr(scheduler, "_now_et", sample)
    if fault == "status-write":
        write = Path.write_text

        def write_or_interrupt(path, *args, **kwargs):
            if path.name == WARNING_FILE:
                trip()
            return write(path, *args, **kwargs)

        mp.setattr(Path, "write_text", write_or_interrupt)
    if fault == "attempt-save" or fault == "confirmed-save":
        save = picks.save_pick

        def save_or_interrupt(daily, *args):
            if (daily.delivered_at is not None) == (fault == "confirmed-save"):
                trip()
            return save(daily, *args)

        mp.setattr(picks, "save_pick", save_or_interrupt)
    if fault == "save-state":
        mp.setattr(scheduler, "save_state", lambda *_: trip())
    if fault == "record-commit":
        mp.setattr(scheduler, "_commit_decision_for_pick", lambda *a, **k: trip())
    if fault == "capture":
        capture.side_effect = lambda *_: trip()
    if fault == "cutoff":
        mp.setattr(picks, "submission_cutoff_et", lambda *_: trip())

    traced = fault == "real-sigint" and is_base
    if traced:
        fn = scheduler._deliver_and_lock_pick
        lines, start = inspect.getsourcelines(fn)
        lock_text = "Pick DM sent:" if mode == "dm" else "Posted to Bluesky:"
        lock_index = next(i for i, line in enumerate(lines) if lock_text in line)
        return_index = next(i for i in range(lock_index + 1, len(lines)) if lines[i].strip() == "return True")

        def trace(frame, event, arg):
            if frame.f_code is fn.__code__:
                if event == "line" and frame.f_lineno == start + return_index:
                    trip()
                return trace
            return None

    err = io.StringIO()
    with redirect_stderr(err):
        original_trace = sys.gettrace()
        try:
            if traced:
                sys.settrace(trace)
            result = scheduler._deliver_and_lock_pick(
                daily, config, directory, state, DAY, "review", selection=selection,
            )
        except BaseException as exc:
            result = {"propagated": type(exc).__name__}
        finally:
            if traced:
                sys.settrace(original_trace)

    stamp = None
    files = {}
    status = None
    for path in sorted(home.rglob("*")):
        if not path.is_file():
            continue
        raw = path.read_bytes().replace(os.fsencode(home), b"{case}")
        if path.name == WARNING_FILE:
            status = json.loads(raw)
            continue
        if path == directory / f"{DAY}.json" and not is_base:
            stamp = json.loads(raw)["automatic_reading"]
            offset = raw.rindex(b',\n  "automatic_reading": ')
            assert raw.endswith(b"\n}")
            raw = raw[:offset] + b"\n}"
        files[str(path.relative_to(home))] = raw.hex()
    warning_calls = []
    alerts = []
    for call in warning_transport.call_args_list:
        (warning_calls if call.args[1].startswith("BTS health WARN attention:") else alerts).append(call.args)
    stderr_lines = err.getvalue().replace(str(home), "{case}").splitlines()
    warning_lines = [s for s in stderr_lines if s.startswith(PREFIX)]
    diagnostics = [s for s in stderr_lines if s.startswith(DIAGNOSTIC)]
    return {
        "existing": {
            "result": result, "sends": sends, "files": files, "memory_state": asdict(state),
            "capture": [{"date": call.args[1], "config": json.loads(json.dumps(call.args[0]).replace(str(home), "{case}"))}
                        for call in capture.call_args_list],
            "alerts": alerts, "stderr": [s for s in stderr_lines if not s.startswith((PREFIX, DIAGNOSTIC))],
        },
        "new": {"stamp": stamp, "attempt_records": attempt_records, "status": status,
                "warnings": warning_calls, "lines": warning_lines, "diagnostics": diagnostics},
        "faults": faults,
    }


def child(output, is_base):
    import sitecustomize

    import pytest
    from bts import scheduler

    assert Path(scheduler.__file__).resolve().is_relative_to(Path(os.environ["CS4CR2_SRC"]))
    cases = {}
    with tempfile.TemporaryDirectory(dir=REVIEW, prefix="delivery-data-") as tmp:
        def case(name, mode, **kwargs):
            with pytest.MonkeyPatch.context() as mp:
                cases[name] = run_case(Path(tmp) / str(len(cases)), mp, mode, is_base=is_base, **kwargs)

        for reading, mode, required, marker, selection in product(
                READINGS, (*MODES, "private"), (False, True), MARKERS, SELECTIONS):
            name = f"matrix/{reading}/{mode}/{int(required)}/{marker}/{selection}"
            case(name, mode, reading=reading, required=required, marker=marker, choice=selection)
        for mode in MODES:
            if not is_base:
                for fault, error in product(FAULTS, ERRORS):
                    case(f"fault/{mode}/{fault}/{error}", mode, fault=fault, error_name=error)
            for fault in CORE_FAULTS:
                case(f"core/{mode}/{fault}", mode, fault=fault, error_name="keyboard")
            case(f"sigint/{mode}", mode, fault="real-sigint")
            case(f"no-recipient/{mode}", mode, omit_recipient=True)
            for medium, identifiers in (("fake", FAKE_IDS), ("wire", WIRE_IDS)):
                for i, identifier in enumerate(identifiers):
                    case(f"id/{mode}/{medium}/{i}", mode, identifier=identifier, wire=medium == "wire")
    assert not sitecustomize.blocked
    output.write_text(json.dumps(cases, indent=2))


def record_expectation(reading):
    from tests.test_pick_chosen_on_cs5a import NOW

    timestamps_and_ages = {
        "old": ((NOW - timedelta(hours=20)).isoformat(), 20.0),
        "fresh": ((NOW - timedelta(hours=3)).isoformat(), 3.0),
        "future": ((NOW + timedelta(minutes=20)).isoformat(), -0.33),
        "unreadable": ("unreadable", None),
    }
    if reading not in timestamps_and_ages:
        return None
    timestamp, age = timestamps_and_ages[reading]
    return {"schema": "bts_pick_automatic_reading_v1", "recorded_at": timestamp,
            "age_hours": age, "stale": reading != "fresh", "reason": None if reading == "fresh" else reading}


def assert_warning(result, expected, full=True):
    new = result["new"]
    assert bool(new["lines"]) == expected, new
    assert bool(new["warnings"]) == (expected and full), new
    assert bool(new["status"]) == (expected and full), new
    if expected:
        assert len(new["lines"]) == 1
    if expected and full:
        assert len(new["warnings"]) == 1
        expected_body = "BTS health WARN attention:\n- [contest_state] " + new["lines"][0].removeprefix("  STALE CONTEST READING — ")
        assert new["warnings"][0] == ["operator.test", expected_body]
        assert new["status"]["status"] == "sent"
        assert new["status"]["sent_sources"] == ["WARN_ATTENTION:contest_state:stale_reading"]
        assert new["status"]["sent_day"] == "2027-07-15"


def compare(base, branch):
    from collections import Counter

    count = Counter()
    for name, result in branch.items():
        family, *details = name.split("/")
        count[family] += 1
        new = result["new"]
        baseline = base[name if family != "fault" else f"matrix/old/{details[0]}/1/unmarked/fresh"]
        try:
            if family != "sigint":
                assert result["existing"] == baseline["existing"], (result["existing"], baseline["existing"])
            if family == "matrix":
                reading, _, required, _, _ = details
                sends = len(baseline["existing"]["sends"])
                assert sends <= 1
                stamp = record_expectation(reading) if sends and required == "1" else None
                assert new["stamp"] == stamp
                assert new["attempt_records"] == [stamp] * sends
                warns = bool(stamp and stamp["stale"])
                assert_warning(result, warns)
                count["matrix sends"] += sends
                count["matrix samples"] += stamp is not None
                count["matrix warnings"] += warns
            elif family == "fault":
                _, fault, error = details
                assert result["faults"] == [fault]
                assert new["stamp"] == (None if fault == "calculation" else record_expectation("old"))
                assert new["attempt_records"] == [new["stamp"]]
                assert len(new["warnings"]) == int(fault in {"warning-transport", "warning-stderr", "status-write"})
                assert bool(new["status"]) == (fault == "warning-stderr")
                assert bool(new["lines"]) == (fault not in {"calculation", "warning-stderr", "warning-clock"})
                has_diagnostic = fault != "diagnostic-print" and (
                    error != "bad-string" or fault in {"warning-transport", "status-write"})
                assert bool(new["diagnostics"]) == has_diagnostic
            elif family == "core":
                assert result["faults"] == baseline["faults"] == [details[1]]
                assert result["existing"]["result"] == {"propagated": "KeyboardInterrupt"}
                assert_warning(result, False)
            elif family == "sigint":
                assert result["faults"] == baseline["faults"] == ["real-sigint"]
                assert baseline["existing"]["result"] == {"propagated": "KeyboardInterrupt"}
                assert result["existing"]["result"] is True
                assert result["existing"] | {"result": baseline["existing"]["result"]} == baseline["existing"]
                assert len(new["warnings"]) == len(new["lines"]) == 1 and new["status"] is None
            elif family == "id":
                _, medium, index = details
                identifier = (WIRE_IDS if medium == "wire" else FAKE_IDS)[int(index)]
                assert len(result["existing"]["sends"]) == 1
                missing = medium == "wire" and identifier == "missing"
                assert result["existing"]["result"] is not missing
                assert new["stamp"] == record_expectation("old")
                assert new["attempt_records"] == [new["stamp"]]
                assert_warning(result, bool(isinstance(identifier, str) and identifier and not missing))
            elif family == "no-recipient":
                assert_warning(result, details[0] == "public", full=False)
        except AssertionError as exc:
            raise AssertionError(f"{name}: {exc}\nnew={new}") from exc
    print(dict(count))
    for name, result in branch.items():
        if name.startswith(("id/", "sigint/")):
            print(name, "result=", result["existing"]["result"], "warning=", bool(result["new"]["warnings"]))


def main():
    results = {}
    with tempfile.TemporaryDirectory(dir=REVIEW, prefix="delivery-src-") as tmp:
        base = archive("1c6cbc77", Path(tmp))
        for name, src in (("base", base), ("branch", CURRENT)):
            output = REVIEW / f"delivery-{name}.json"
            env = os.environ | {
                "PYTHONPATH": os.pathsep.join(map(str, [REVIEW, src, ROOT])),
                "CS4CR2_SRC": str(src),
            }
            subprocess.run([sys.executable, __file__, name, str(output)], env=env, cwd=ROOT, check=True)
            results[name] = json.loads(output.read_text())
    compare(results["base"], results["branch"])


if __name__ == "__main__":
    if len(sys.argv) > 1:
        child(Path(sys.argv[2]), sys.argv[1] == "base")
    else:
        main()
