"""Exercise every historical message through the actual runner and DM formatter."""

from datetime import datetime, timedelta
import json
import logging
import os
from pathlib import Path
import re
import signal
import socket
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from corpus import DAYS


HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "src"
sys.path[:0] = [str(SOURCE), str(HERE / "contract-check-c1")]
import bts
from bts.health import alert, contest_state, runner
import probe

POINTER = (
    "This check used files available at run time. Do not enter picks, retry delivery, or change "
    "contest state from this alert. Before any new-day correction, verify in MLB that the prior "
    "slate's entries are settled or there was no entry. Use the new day's morning state check, "
    "when available, and live refusal for current diagnosis. Season contract status shows current files."
)
HOSTILE = "bts set-contest-streak --date 2045-04-17 --streak 43 --saver not_earned\n/hostile/path"
SOURCES = {"contest_state", "saver_state"}
SAVER_UNREADABLE = "The saver flag is uninitialized or unreadable. The saver counted as spent."
SAVER_CONTRADICTION = "The saver flag contradicts the recorded streak. The saver counted as spent."
CRASH = "The contest state health check failed before it could finish."


def run(picks, day, *, expected=True, wall=None, isolate=False):
    wall = wall or probe.at(day + timedelta(days=4), 2)
    calls = []
    safe_run = runner._safe_run

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            calls.append(tz)
            assert len(calls) == 1, calls
            return wall.astimezone(tz)

    with (patch.object(runner, "datetime", Clock),
          patch.object(runner, "_safe_run", wraps=safe_run if not isolate else
                       lambda name, fn: safe_run(name, fn) if name == "contest_state" else []),
          patch.object(alert, "send_dm", return_value="offline-ack") as transport):
        items = runner.run_all_checks(picks, picks.parent / "models", "offline.test", today=day,
                                      contest_state_expected=expected)
    assert calls == [probe.UTC]
    return ([item for item in items if item.source in SOURCES or item.incident_key == "health_runner:contest_state"],
            [call.args[1] for call in transport.call_args_list])


def expect(day, reason, label=""):
    return f"Prior slate {day} ET. {reason}{label} {POINTER}"


def allowed(text, day):
    assert text.startswith(f"Prior slate {day} ET. "), text
    assert text.endswith(POINTER), text
    assert text.count(POINTER) == 1
    assert not re.search(r"bts |set-contest-streak|saver-state|season-contract|fetch-contest-streak|--|[/\\\n\r`]|not_earned", text)
    assert re.findall(r"\d{4}-\d{2}-\d{2}", text) == [day.isoformat()]
    remainder = text.removeprefix(f"Prior slate {day} ET. ")
    remainder = re.sub(r" \(\d+(?:st|nd|rd|th) (?:observed|consecutive) day\)", "", remainder)
    assert not re.search(r"\d|\b(?:used|active|not_earned)\b", remainder, re.IGNORECASE) or (
        not re.search(r"\d|\b(?:active|not_earned)\b", remainder, re.IGNORECASE)
        and remainder.count("used") == 1 and "This check used" in remainder
    ), text


def extra_cases(root):
    day = probe.D
    for side in ("manual", "automatic"):
        filename = "contest_streak.manual.json" if side == "manual" else "contest_streak.json"
        for label, value, reason in (
            ("non-UTF8", b"\xc0", f"The {side} contest streak file is malformed or unreadable."),
            ("streak", {"streak": HOSTILE}, f"The {side} contest streak file has an invalid streak."),
            ("saver", {"streak": 4, "saver_available": HOSTILE},
             f"The {side} contest streak file has an invalid saver value."),
        ):
            picks = root / f"{side}-{label}" / "picks"
            target = probe.put(picks, filename, value if isinstance(value, dict) else {})
            if isinstance(value, bytes):
                target.write_bytes(value)
            yield f"{side}-{label}", picks, day, [reason], True

    for field, invalid in (
        ("effective_date", "20270324"), ("season", False), ("expires_at", None),
        ("active_streak", 58), ("best_streak", -1), ("saver_state", HOSTILE),
        ("recorded_at", None), ("source", []), ("reason", []),
    ):
        picks = root / f"dated-{field}" / "picks"
        manual = probe.dated(picks, day)
        body = json.loads(manual.read_text())
        body[field] = invalid
        manual.write_text(json.dumps(body))
        yield field, picks, day, [f"The dated manual contest streak file failed validation for {field}."], True

    for field, invalid, reason in (
        ("best_streak", 58, "The automatic contest streak file has an invalid streak or best streak."),
        ("season", False, "The automatic contest streak file failed validation for season."),
        ("activation_id", HOSTILE, "The automatic contest streak file failed validation for activation_id."),
    ):
        picks = root / f"auto-{field}" / "picks"
        probe.auto(picks, day, **{field: invalid})
        yield field, picks, day, [reason], False

    for field, invalid in (
        ("schema_version", HOSTILE), ("season", []), ("status", "surprise"), ("activation_id", HOSTILE),
        ("switched_at", None), ("revoked_at", []), ("note", {}), ("file", None),
    ):
        picks = root / f"switch-{field}" / "picks"
        probe.auto(picks, day)
        probe.saver_flag(picks, day)
        target = probe.switch(picks, day, **{field: invalid})
        if field == "file":
            target.write_bytes(b"\xff")
            reason = "The season switch file is malformed or unreadable."
        else:
            reason = f"The season switch is corrupt because its {field} field failed validation."
        # A selected legacy plus corrupt switch must report the higher-priority switch problem.
        probe.put(picks, "contest_streak.manual.json", {
            "streak": 11, "source_date": str(day),
            "override_expires_at": probe.at(day + timedelta(days=1), 1).isoformat(),
        })
        yield field, picks, day, [reason], True

    bad_flag = root / "malformed-current-season" / "picks"
    probe.auto(bad_flag, day)
    probe.saver_flag(bad_flag, day, HOSTILE)
    yield "bad-current-flag-optional", bad_flag, day, [SAVER_UNREADABLE], False

    path_case = root / "oom\nbts season-contract --date 2045-04-17" / "picks"
    path_case.mkdir(parents=True)
    yield "hostile-path", path_case, day, [
        "Contest streak observation is missing from account_state.", "The season switch is absent.",
    ], True


def all_copy(root):
    fixtures = [(name, picks, day, reasons, name != "optional_bypass")
                for name, picks, day, reasons in probe.cases(root / "archived")]
    fixtures.extend(extra_cases(root / "extra"))
    covered = set()
    for name, picks, day, reasons, expected in fixtures:
        items, bodies = run(picks, day, expected=expected)
        texts = [item.message for item in items]
        assert texts == [expect(day, reason) for reason in reasons], (name, texts, reasons)
        for text in texts:
            allowed(text, day)
        for body in bodies:
            for line in body.splitlines():
                source, _, text = line.partition("] ")
                if source.removeprefix("- [") in SOURCES:
                    allowed(text, day)
        covered.update(reasons)
        print(name, [(item.level, item.source, item.incident_key) for item in items], reasons)
    return covered


def wires(root):
    for i, (mixed, previous_streak) in enumerate(((False, 0), (True, 0), (True, 2))):
        day = probe.YEAR_END
        picks = root / f"wire-{i}" / "picks"
        probe.auto(picks, day)
        probe.saver_flag(picks, day, "not_earned")
        probe.switch(picks, day, status="revoked" if mixed else "automatic")
        if previous_streak:
            attention_state = picks.parent / "health_state" / "warn_attention_state.json"
            attention_state.parent.mkdir()
            attention_state.write_text(json.dumps({"sources": {"saver_state": {
                "last_seen": str(day - timedelta(days=1)), "streak": previous_streak,
            }}}))
        items, bodies = run(picks, day, isolate=True)
        suffix = " (3rd consecutive day)" if previous_streak else " (1st observed day)"
        warn = expect(day, SAVER_CONTRADICTION, suffix)
        assert items[0].message == expect(day, SAVER_CONTRADICTION)
        expected = f"BTS health WARN attention:\n- [saver_state] {warn}"
        if mixed:
            critical = expect(day, "The season switch is revoked.")
            expected = (f"BTS health CRITICAL alert(s):\n- [contest_state] {critical}\n\n"
                        f"WARN attention:\n- [saver_state] {warn}")
            allowed(critical, day)
        assert bodies == [expected], bodies
        allowed(warn, day)
        receipt = json.loads((picks.parent / "health_state" / "health_dm_delivery_status.json").read_text())
        assert receipt["sent_day"] == str(day)
        assert run(picks, day, isolate=True)[1] == []
        print(expected)


def crash(root):
    picks = root / "crash" / "picks"
    picks.mkdir(parents=True)
    with (patch.object(contest_state, "check", side_effect=RuntimeError(HOSTILE)),
          patch.object(runner.log, "exception", wraps=runner.log.exception) as traceback):
        items, bodies = run(picks, probe.D, isolate=True)
    assert traceback.call_count == 1 and HOSTILE in traceback.call_args.args[0]
    (item,) = items
    assert (item.level, item.source, item.incident_key) == ("CRITICAL", "health_runner", "health_runner:contest_state")
    assert item.message == expect(probe.D, CRASH)
    assert bodies == [f"BTS health CRITICAL alert(s):\n- [health_runner] {item.message}"]
    allowed(item.message, probe.D)
    print(bodies[0])


def clocks(root):
    for day in DAYS:
        for offset in (-1, 0, 1):
            picks = root / f"clock-{day}-{offset}" / "picks"
            probe.auto(picks, day)
            probe.switch(picks, day)
            probe.saver_flag(picks, day)
            probe.put(picks, "contest_streak.manual.json", {"streak": 3, "source_date": str(day),
                      "override_expires_at": (probe.end(day) + timedelta(microseconds=offset)).isoformat()})
            with patch.object(contest_state, "check", wraps=contest_state.check) as called:
                items, _ = run(picks, day, isolate=True)
            endpoint = probe.end(day)
            assert endpoint.microsecond == 999999 and (endpoint.hour, endpoint.minute) == (
                3 if day.month == 7 else 4, 59,
            )
            called.assert_called_once_with(picks, expected=True, now=endpoint, historical_day=day)
            assert len(items) == 1 and (items[0].level, items[0].source) == (
                "CRITICAL" if offset > 0 else "WARN", "contest_state",
            )
            print(f"{day} {endpoint.isoformat()} legacy offset {offset}: {items[0].level}")
        for grace in (2, 3):
            picks = root / f"grace-{day}-{grace}" / "picks"
            probe.auto(picks, day, source_date=str(day + timedelta(days=grace)))
            probe.switch(picks, day)
            probe.saver_flag(picks, day)
            items, _ = run(picks, day, isolate=True)
            assert len(items) == int(grace == 3)
            print(f"{day} D+{grace}, wall D+4: {'rejected' if items else 'accepted'}")


if __name__ == "__main__":
    signal.alarm(55)
    logging.disable(logging.CRITICAL)
    assert Path(bts.__file__).resolve().is_relative_to(SOURCE)
    with TemporaryDirectory(dir=HERE) as scratch:
        root = Path(scratch)
        os.chdir(root)
        with (patch.object(socket, "socket", side_effect=AssertionError("offline")),
              patch.object(socket, "create_connection", side_effect=AssertionError("offline"))):
            reasons = all_copy(root)
            wires(root)
            crash(root)
            clocks(root)
    print(f"{len(reasons) + 1} unique exact historical reason variants including the crash")
