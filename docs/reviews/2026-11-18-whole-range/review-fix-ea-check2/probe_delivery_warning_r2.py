"""Check exact copy, alert parity and the operator's midnight and settlement cases."""

from contextlib import redirect_stderr
from datetime import timedelta
from io import StringIO
import json
from pathlib import Path
import re
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

from bts import contest_state, picks, scheduler
from bts.health import alert
from scripts.audit.season_walk import walk
from tests.test_pick_chosen_on_cs5a import daily_pick

from probe_later_slate_r2 import WORK, at, automatic, config, environment, live, scheduler_state, status, write_today


ROOT = WORK.parent.parent
BASE = "988708df"
DOC = "docs/ops/2027-season-start.md"
SOURCE = "src/bts/scheduler.py"
CONTRACT = "docs/reviews/2026-11-18-whole-range/kickoff-fix-ea.md"
KEY = "WARN_ATTENTION:contest_state:stale_reading"
RECEIPT = "health_state/contest_stale_reading_dm_delivery_status.json"


def git_show(file):
    return subprocess.run(["git", "show", f"{BASE}:{file}"], cwd=ROOT, timeout=5,
                          capture_output=True, check=True, text=True).stdout


def copy_audit():
    contract = (ROOT / CONTRACT).read_text().split("\n## Second follow-up", 1)[1]
    previous = git_show(DOC)
    current = (ROOT / DOC).read_text()
    expected = []
    for label in ("Check B1", "Check E1 (B8a half)"):
        line = next(row for row in contract.splitlines() if row.startswith(f"| {label} |"))
        old, new = [field.strip().strip("`") for field in line[2:-2].split(" | ")[1:3]]
        expected.append(new)
        bold_label = "On a warning after a delivery:"
        if old.startswith(bold_label):
            old, new = (value.replace(bold_label, f"**{bold_label}**") for value in (old, new))
        else:
            new = re.sub(r"bts set-contest-streak .*?--picks-dir DIR", r"`\g<0>`", new)
            old = re.sub(r"bts set-contest-streak .*?--picks-dir DIR", r"`\g<0>`", old)
        assert previous.count(old) == 1 and current.count(new) == 1
        previous = previous.replace(old, new)
    assert previous == current
    for sentence in expected:
        print("B8a character-for-character:", sentence)
    entry = next(row for row in contract.splitlines() if row.startswith("12. "))
    previous_warning, new_warning = re.findall(r"`([^`]+)`", entry)
    expression = re.search(r"`(re\.sub\(.+?\)\)\))`", contract)
    assert expression
    return previous_warning, new_warning, expression.group(1)


def source_audit(previous_warning, new_warning, expression):
    previous = git_show(SOURCE)
    current = (ROOT / SOURCE).read_text()
    old_line = f'            f"{previous_warning.replace("{D}", "{warning_day.isoformat()}")} "\n'
    new_line = f'            "{new_warning} "\n'
    assert previous.count(old_line) == current.count(new_line) == 1
    assert expression in current
    normalized = current.replace("import re\n", "", 1).replace(new_line, old_line).replace(
        expression, '" ".join(value.splitlines()).replace("below the", "under the")', 1)
    assert normalized == previous
    changed = subprocess.run(["git", "diff", "--name-only", BASE, "3f725779", "--", "src"], cwd=ROOT,
                             timeout=5, capture_output=True, text=True, check=True).stdout.splitlines()
    assert changed == [SOURCE]
    old = ModuleType("bts._ea4_base_scheduler")
    sys.modules[old.__name__] = old
    exec(compile(previous, f"{BASE}:{SOURCE}", "exec"), old.__dict__)
    print("SOURCE: exact old scheduler restored by undoing import re, warning sentence and display expression.")
    return old


def read_receipt(path):
    status = json.loads((path.parent / RECEIPT).read_text())
    assert (status["status"], status["warn_attention_count"], status["critical_count"]) == ("sent", 1, 0)
    assert status["sent_sources"] == [KEY]
    return status


def emit_delivery(path, before, after):
    walk._set_clock(before)
    candidate = daily_pick()
    candidate.date = before.date().isoformat()
    candidate.pick.game_time = (before + timedelta(minutes=90)).isoformat()
    candidate.chosen_on = picks.chosen_on_record(live(path), required=True)
    with environment(transport=lambda: walk._set_clock(after)) as (sent, warnings):
        assert scheduler._deliver_and_lock_pick(candidate, config(path), path, scheduler_state(candidate.date),
                                                candidate.date, "warning probe")
    locked = picks.load_pick(candidate.date, path)
    assert len(sent) == len(warnings) == 1 and locked.delivered_at == after.isoformat()
    assert read_receipt(path)["sent_day"] == after.date().isoformat()
    print("WARNING SLATE", locked.date, "printed date", after.date().isoformat(), "cutoff", picks.submission_cutoff_et(locked))
    return locked, warnings[0]


def settlement(root, new_warning):
    before, after = at("2027-07-14T23:59:55"), at("2027-07-15T00:00:20")
    path = root / "midnight" / "picks"
    automatic(path, before, streak=12, best=14, saver="active")
    locked, delivered = emit_delivery(path, before, after)
    assert new_warning in delivered and "--date 2027-07-15 " in delivered and locked.date == "2027-07-14"
    assert picks.submission_cutoff_et(locked) > at("2027-07-15T00:05")
    walk._set_clock(at("2027-07-15T00:05"))
    assert "manual: absent" in status(path)
    print("C29: dated warning command skipped (printed day differs from pick); later-slate filing waits for the open, empty 07-14 window.")
    walk._set_clock(at("2027-07-15T09:30"))
    print("MLB confirms absence after closure; authentic MLB state fixture is 13/14/active. Apply TODAY rule.")
    assert "automatic path alone: accepted (streak 12" in status(path)
    write_today(path, 13, 14, "active")
    shown = status(path)
    assert "streak=13 best=14 saver=active in effect=True" in shown
    assert "automatic path alone: accepted (streak 12" in shown
    request_log = path / "account_state/contest_manual_requests.jsonl"
    previous = request_log.read_bytes()
    assert (live(path).streak, live(path).best_streak, live(path).dated_identity["saver_state"]) == (13, 14, "active")
    assert request_log.read_bytes() == previous
    print("DATED EFFECTIVE: compare with dated 13/14/active; no second command even though automatic remains 12.")

    path = root / "same-day" / "picks"
    before = at("2027-07-15T16:00")
    automatic(path, before, streak=12, best=14, saver="active")
    locked, delivered = emit_delivery(path, before, before)
    assert new_warning in delivered and locked.date == before.date().isoformat()
    assert "manual: absent" in status(path)
    print("SAME DAY: 07-14 entry pending in MLB; both B8a versions wait, even with matching D and pick slate.")
    walk._set_clock(before + timedelta(minutes=15))
    print("07-14 settles and differs from status; use D command now.")
    write_today(path, 13, 14, "active")
    assert live(path).observation_kind == "dated" and live(path).streak == 13
    assert picks.load_pick(locked.date, path).notification_id == "captured-pick"


def warning_parity(root, modules, previous_warning, new_warning):
    now = at("2027-07-15T10:00")
    stamp = "0001-01-01T00:00:00." + "0" * 102 + "+00:00"
    source = contest_state.DecisionStreakState(10, False, False, "contest", "fresh", None, None,
                                              observation_kind="automatic", contest_recorded_at=stamp)
    record = contest_state.automatic_reading_record(source, now)
    assert len(stamp) == 128 and record["reason"] == "old"
    receipts, bodies, logs = [], [], []
    for module in modules:
        path = root / module.__name__ / "picks"
        sent, seen, journal = [], [], StringIO()

        def dispatch(*args, **kw):
            payload, = kw["warn_attention"]
            seen.append((payload.level, payload.source, payload.incident_key, kw["now_et_date"].isoformat()))
            assert kw["status_path"] == path.parent / RECEIPT
            return dispatch_real(*args, **kw)

        dispatch_real = alert.dispatch_dm_for_health_alerts
        with (patch.object(module, "_now_et", return_value=now) as clock,
              patch.object(alert, "dispatch_dm_for_health_alerts", side_effect=dispatch),
              patch.object(alert, "send_dm", side_effect=lambda recipient, body: sent.append(body)),
              redirect_stderr(journal)):
            module._warn_stale_contest_reading(record, config(path), path)
            module._warn_stale_contest_reading(record, config(path), path)
            assert len(sent) == 1
            first_receipt = read_receipt(path)
            assert first_receipt["sent_day"] == now.date().isoformat()
            clock.return_value = now + timedelta(days=1)
            module._warn_stale_contest_reading(record, config(path), path)
            assert len(sent) == 2
            assert read_receipt(path)["sent_day"] == "2027-07-16"
        assert seen == [("WARN", "contest_state", "contest_state:stale_reading", day)
                        for day in ("2027-07-15", "2027-07-15", "2027-07-16")]
        with (patch.object(module, "_now_et", return_value=now),
              patch.object(alert, "dispatch_dm_for_health_alerts", side_effect=KeyboardInterrupt),
              redirect_stderr(journal)):
            module._warn_stale_contest_reading(record, config(path), path)
        assert "Stale-reading warning failed:" in journal.getvalue()
        bodies.append(sent)
        logs.append(journal.getvalue())
        receipts.append({key: value for key, value in first_receipt.items()
                         if key not in {"body_sha256", "updated_at"}})
    assert [len(bodies[i][0]) for i in (0, 1)] == [892, 896]
    for i, day in enumerate(("2027-07-15", "2027-07-16")):
        old_text = previous_warning.replace("{D}", day)
        assert bodies[0][i].replace(old_text, new_warning) == bodies[1][i]
        logs[0] = logs[0].replace(old_text, new_warning)
    assert logs[0] == logs[1] and receipts[0] == receipts[1], receipts
    print("PARITY: warning body only, base 892 and head 896 characters. WARN/source/key/receipt/ET day/dedup and interrupt containment matched.")


if __name__ == "__main__":
    old_warning, new_warning, expression = copy_audit()
    previous_scheduler = source_audit(old_warning, new_warning, expression)
    with TemporaryDirectory(dir=WORK) as temp, walk.offline(at("2027-07-15T10:00")):
        root = Path(temp)
        settlement(root, new_warning)
        warning_parity(root, (previous_scheduler, scheduler), old_warning, new_warning)
