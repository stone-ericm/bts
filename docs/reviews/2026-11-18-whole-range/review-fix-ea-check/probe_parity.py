"""Compare the previous checkout with HEAD and measure real warning envelopes."""

from datetime import datetime, timedelta
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

from bts import contest_state, dm, picks, scheduler
from bts.health import alert
from scripts.audit.season_walk import walk
from tests.test_pick_chosen_on_cs5a import daily_pick

from probe_contest import WORK, config, day_state, stale, state


ROOT = WORK.parent.parent
NOW = datetime.fromisoformat("2027-07-15T10:00:00-04:00")
FILE = "contest_stale_reading_dm_delivery_status.json"
KEY = "WARN_ATTENTION:contest_state:stale_reading"
OLD = "0001-01-01T00:00:00." + "0" * 102 + "+00:00"
FUTURE = "9999-12-31T00:00:00." + "0" * 102 + "+00:00"


DELIVERY_BEFORE = (
    '            "absolute scheduler picks path. The sent pick and saver flag stay as they were. Write only if "\n'
    '            f"the pick\'s slate and today\'s ET date both equal {warning_day.isoformat()}. "\n'
)
DELIVERY_AFTER = (
    '            "absolute scheduler picks path. The sent pick and saver flag stay as they were. "\n'
    '            f"Write for the {warning_day.isoformat()} slate only on that ET date, with prior entries settled or absent. "\n'
)
STREAK_BEFORE = '    if streak is not None and not no_decision:\n        sentence += f" Streak used at this check: {streak}."'
STREAK_AFTER = (
    '    if streak is not None and not no_decision:\n'
    '        if basis == "decision":\n'
    '            sentence += f" Streak used at this check: {streak}."\n'
    '        else:\n'
    '            sentence += f" Streak recorded with this check: {streak}."'
)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, timeout=5, capture_output=True,
                          check=True, text=True).stdout


def compare_source():
    path = "src/bts/scheduler.py"
    assert git("diff", "--name-only", "1ed57c47", "988708df", "--", "src").splitlines() == [path]
    reference = git("show", f"1ed57c47:{path}")
    expected = reference
    for old, changed in (
        (DELIVERY_BEFORE, DELIVERY_AFTER),
        ('On {warning_day.isoformat()} ET away from midnight,', 'In {warning_day.year:04d} ET away from midnight,'),
        ('On {date} ET away from midnight,', 'In {date[:4]} ET away from midnight,'),
        ('return " ".join(value.splitlines()) if isinstance(value, str) else "?"',
         'return " ".join(value.splitlines()).replace("below the", "under the") if isinstance(value, str) else "?"'),
        ('    no_decision = fields.get("basis") == "no_decision"',
         '    basis = fields.get("basis")\n    no_decision = basis == "no_decision"'),
        (STREAK_BEFORE, STREAK_AFTER),
    ):
        assert expected.count(old) == 1, old
        expected = expected.replace(old, changed)
    assert (ROOT / path).read_text() == expected == git("show", f"988708df:{path}")
    print("Source byte comparison: entire src delta is the six rendering-only replacements.")
    base = ModuleType("bts._ea3_base_scheduler")
    sys.modules[base.__name__] = base
    exec(compile(reference, f"1ed57c47:{path}", "exec"), base.__dict__)
    return base


def timestamp(path, recorded):
    file = path / "account_state/contest_streak.json"
    file.write_text(json.dumps(json.loads(file.read_text()) | {"recorded_at": recorded}))


def morning(module, path, instant):
    module._morning_state_check(config(path), path, instant.date().isoformat(), instant, instant + timedelta(hours=1))


def trace(module, path):
    walk._set_clock(NOW)
    receipts = []
    calls = []
    bodies = []
    snapshot = path.parent / "health_state" / FILE
    dispatch = alert.dispatch_dm_for_health_alerts

    def capture_dispatch(critical, recipient, *, warn_attention, status_path, now_et_date):
        assert critical == [] and recipient == "capture.invalid"
        caution, = warn_attention
        calls.append((caution.level, caution.source, caution.incident_key, status_path.name, now_et_date.isoformat()))
        return dispatch(critical, recipient, warn_attention=warn_attention,
                        status_path=status_path, now_et_date=now_et_date)

    def transport(_, body):
        bodies.append(body)
        if len(bodies) == 1:
            raise RuntimeError("offline first dispatch failure")
        if len(bodies) in (4, 5):
            raise KeyboardInterrupt("offline warning interrupt")
        return "warning"

    def receipts_without_body_hash():
        current = json.loads(snapshot.read_text())
        current.pop("body_sha256")
        receipts.append(current)
        return current

    day = NOW.date().isoformat()
    daily = daily_pick()
    daily.chosen_on = picks.chosen_on_record(state(path), required=True)
    daily.pick.game_time = (NOW + timedelta(hours=8)).isoformat()
    with patch.object(alert, "dispatch_dm_for_health_alerts", side_effect=capture_dispatch), \
            patch.object(alert, "send_dm", side_effect=transport), \
            patch.object(dm, "send_dm", return_value="captured-pick") as pick_send, \
            patch.object(module, "_trigger_live_forward_capture_on_lock"), \
            patch.object(module, "_delivery_game_statuses", return_value={}):
        morning(module, path, NOW)
        assert receipts_without_body_hash()["status"] == "failed"
        assert module._deliver_and_lock_pick(daily, config(path), path, day_state(day), day, "parity")
        assert receipts_without_body_hash()["sent_sources"] == [KEY]
        assert pick_send.call_count == 1
        picked_body = pick_send.call_args.args[1]
        module._warn_stale_contest_reading(daily.automatic_reading, config(path), path)
        assert len(bodies) == 2
        walk._set_clock(NOW + timedelta(days=1))
        module._warn_stale_contest_reading(daily.automatic_reading, config(path), path)
        assert receipts_without_body_hash()["sent_day"] == "2027-07-16"
        assert len(bodies) == 3
        next_day = NOW + timedelta(days=2)
        walk._set_clock(next_day)
        module._warn_stale_contest_reading(daily.automatic_reading, config(path), path)
        assert json.loads(snapshot.read_text())["sent_day"] == "2027-07-16"
        try:
            morning(module, path, next_day)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError("Morning warning swallowed the interrupt")
    assert all(call[:4] == ("WARN", "contest_state", "contest_state:stale_reading", FILE) for call in calls)
    assert all("Previous health alert DM FAILED" not in message for message in bodies)
    files = [(str(file.relative_to(path)), file.read_bytes()) for file in sorted(path.rglob("*.json"))]
    return calls, receipts, picked_body, files, bodies


def longest(path):
    stale(path, NOW)
    calculator = contest_state.automatic_reading_record
    results = {}
    for reason, recorded in (("old", OLD), ("future", FUTURE), ("unreadable", None), ("fresh", NOW.isoformat())):
        timestamp(path, recorded)
        record = calculator(state(path), NOW)
        assert record["reason"] == (None if reason == "fresh" else reason)
        receipt = path.parent / "health_state" / FILE
        bodies = []
        for route in ("delivery", "morning"):
            receipt.unlink(missing_ok=True)
            with patch.object(alert, "send_dm", side_effect=lambda _, body: bodies.append(body)):
                if route == "delivery":
                    scheduler._warn_stale_contest_reading(record, config(path), path)
                else:
                    morning(scheduler, path, NOW)
        expected_head = "BTS health WARN attention:\n- [contest_state] "
        assert all(body.startswith(expected_head) for body in bodies)
        results[reason] = tuple(len(body) for body in bodies)
        print("Body measurements (delivery, morning)", reason, results[reason])
    assert len(OLD) == len(FUTURE) == 128
    assert results["fresh"] == ()
    assert results["old"] == (892, 889)
    assert tuple(max(results[reason][i] for reason in ("old", "future", "unreadable")) for i in (0, 1)) == (892, 889)


if __name__ == "__main__":
    prior = compare_source()
    with TemporaryDirectory(dir=WORK) as temp, walk.offline(NOW):
        root = Path(temp)
        path = root / "prior" / "picks"
        stale(path, NOW)
        current = root / "current" / "picks"
        shutil.copytree(path, current)
        before = trace(prior, path)
        after = trace(scheduler, current)
        assert before[:-1] == after[:-1]
        assert all(old != changed for old, changed in zip(before[-1], after[-1], strict=True))
        print("Equal WARN level/source/key/date/receipt lifecycle, interrupts, pick send and pick/decision/state JSON.")
        longest(root / "max" / "picks")
