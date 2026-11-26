"""Historical contest checks with synthetic files and a captured DM transport."""

from datetime import datetime, time, timedelta
import json
import os
from pathlib import Path
import re
import runpy
import subprocess
import sys
from unittest.mock import patch

import pytest

from bts.health import alert, contest_state, runner


REPO_ROOT = Path(__file__).resolve().parents[1]
REVIEW = REPO_ROOT / "docs/reviews/2026-11-18-whole-range"
POINTER_SENTENCES = (
    "This check used files available at run time.",
    "Do not enter picks, retry delivery, or change contest state from this alert.",
    "Before any new-day correction, verify in MLB that the prior slate's entries are settled or there was no entry.",
    "Use the new day's morning state check, when available, and live refusal for current diagnosis.",
    "Season contract status shows current files.",
)
OPERATOR_ROWS = (
    ("files after the fetch", POINTER_SENTENCES[0]),
    ("new day missing usable state", POINTER_SENTENCES[1]),
    ("pending entry", POINTER_SENTENCES[2]),
    ("skipped morning check", POINTER_SENTENCES[3]),
    ("live ways out before settlement", POINTER_SENTENCES[4]),
)
POINTER = " ".join(POINTER_SENTENCES)
SOURCES = {"contest_state", "saver_state"}


@pytest.fixture
def probe(monkeypatch):
    from types import SimpleNamespace

    monkeypatch.setattr(sys, "path", sys.path.copy())
    return SimpleNamespace(**runpy.run_path(str(REVIEW / "contract-check-c1/probe.py")))


def contest_alerts(alerts):
    return [
        item for item in alerts
        if item.source in SOURCES or item.incident_key == "health_runner:contest_state"
    ]


def run(picks, day, wall, *, expected=True, send=None, inputs=None, isolate=False):
    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return wall.astimezone(tz) if tz is not None else wall.replace(tzinfo=None)

    def other_checks(name, fn):
        return safe_run(name, fn) if name == "contest_state" else []

    safe_run = runner._safe_run
    with (
        patch.object(runner, "datetime", Clock, create=True),
        patch.object(contest_state, "datetime", Clock),
        patch.object(alert, "send_dm", side_effect=send, return_value="offline-ack") as transport,
        patch.object(runner, "_safe_run", other_checks if isolate else safe_run),
    ):
        result = runner.run_all_checks(
            picks, picks.parent / "models", "offline.test",
            thresholds_overrides={
                "warn_attention_state": picks / ".health/warn_attention_state.json",
                "health_dm_delivery_status": picks / ".health/health_dm_delivery_status.json",
            },
            contest_state_expected=expected, **({"today": day} if inputs is None else inputs),
        )
    return contest_alerts(result), [call.args[1] for call in transport.call_args_list]


def exact(day, reason):
    return f"Prior slate {day} ET. {reason} {POINTER}"


def assert_safe(message, day):
    header = f"Prior slate {day} ET."
    assert message.startswith(header)
    assert message.endswith(POINTER)
    assert "\n" not in message and "\r" not in message
    assert not any(token in message for token in (
        "bts ", "set-contest-streak", "saver-state", "season-contract",
        "fetch-contest-streak", "--", "/", "not_earned",
    ))
    assert re.findall(r"\d{4}-\d{2}-\d{2}", message) == [day.isoformat()]


def more_cases(probe, root):
    day = probe.D

    def picks(name):
        path = root / name / "picks"
        path.mkdir(parents=True)
        return path

    for suffix, name in [("", "automatic"), (".manual", "manual")]:
        filename = f"contest_streak{suffix}.json"
        for label, body, reason in [
            ("bad_utf8", None, f"The {name} contest streak file is malformed or unreadable."),
            ("bad_streak", {"streak": False}, f"The {name} contest streak file has an invalid streak."),
            ("bad_saver", {"streak": 4, "saver_available": []},
             f"The {name} contest streak file has an invalid saver value."),
        ]:
            path = picks(f"{name}_{label}")
            file = probe.put(path, filename, body)
            if body is None:
                file.write_bytes(b"\xff")
            yield path, [reason], True
            yield path, [reason], False

    for field, invalid in [
        ("effective_date", "20270324"), ("season", False), ("expires_at", None),
        ("active_streak", 401), ("best_streak", -1), ("saver_state", []),
        ("recorded_at", None), ("source", []), ("reason", []),
    ]:
        path = picks(f"dated_{field}")
        file = probe.dated(path)
        data = json.loads(file.read_text())
        data[field] = invalid
        file.write_text(json.dumps(data))
        yield path, [f"The dated manual contest streak file failed validation for {field}."], True

    for field, invalid, reason in [
        ("best_streak", 3, "The automatic contest streak file has an invalid streak or best streak."),
        ("season", False, "The automatic contest streak file failed validation for season."),
        ("activation_id", "hostile", "The automatic contest streak file failed validation for activation_id."),
    ]:
        path = picks(f"auto_{field}")
        probe.auto(path, **{field: invalid})
        yield path, [reason], True

    path = picks("switch_invalid_json")
    probe.auto(path)
    probe.saver_flag(path)
    probe.switch(path).write_text("{")
    yield path, ["The season switch file is malformed or unreadable."], True
    for field, invalid in [
        ("schema_version", None), ("season", []), ("status", "unknown"),
        ("activation_id", "unknown"), ("switched_at", None), ("revoked_at", []), ("note", []),
    ]:
        path = picks(f"switch_{field}")
        probe.auto(path)
        probe.saver_flag(path)
        probe.switch(path, **{field: invalid})
        yield path, [f"The season switch is corrupt because its {field} field failed validation."], True

    path = picks("same_year_invalid_flag")
    probe.auto(path)
    probe.switch(path)
    probe.saver_flag(path, state="bad")
    yield path, ["The saver flag is uninitialized or unreadable. The saver counted as spent."], True

    path = picks("hostile_old_override")
    probe.put(path, "contest_streak.manual.json", {"streak": 2})
    yield path, ["A legacy or expired manual override file is present."], False
    path = picks("optional_missing")
    yield path, [], False

    path = picks("hostile_exception")
    probe.auto(path)
    probe.switch(path)
    probe.saver_flag(path).write_text('{"season":' + "9" * 5000 + '}')
    yield path, ["The contest state health check failed before it could finish."], True


def test_every_historical_contest_reason_is_exact_and_command_free(tmp_path, probe):
    fixtures = [
        (picks, day, reasons, label != "optional_bypass")
        for label, picks, day, reasons in probe.cases(tmp_path / "driver")
    ]
    fixtures.extend((picks, probe.D, reasons, expected)
                    for picks, reasons, expected in more_cases(probe, tmp_path / "extra"))
    for picks, day, reasons, expected in fixtures:
        with patch.object(contest_state, "datetime", datetime):
            from bts.health.runner import _safe_run
            reference = _safe_run("contest_state", lambda: contest_state.check(
                picks, expected=expected, now=probe.end(day),
            ))
        actual, bodies = run(picks, day, probe.at(day + timedelta(days=1), 2), expected=expected)
        assert [(a.level, a.source, a.incident_key) for a in actual] == [
            (a.level, a.source, a.incident_key) for a in reference
        ], picks
        assert [a.message for a in actual] == [exact(day, reason) for reason in reasons], picks
        for item in actual:
            assert_safe(item.message, day)
        for body in bodies:
            for line in body.splitlines():
                if line.startswith(tuple(f"- [{source}] " for source in SOURCES | {"health_runner"})):
                    assert_safe(line.split("] ", 1)[1], day)


@pytest.mark.parametrize("year_offset", [None, 0, 1])
@pytest.mark.parametrize("future", [2, 3])
def test_historical_contest_uses_et_seasons_and_utc_grace(tmp_path, probe, year_offset, future):
    day = probe.YEAR_END
    probe.auto(tmp_path, day, source_date=(day + timedelta(days=future)).isoformat())
    probe.switch(tmp_path, day, status="automatic" if year_offset is None else "revoked")
    if year_offset is not None:
        probe.saver_flag(tmp_path, day.replace(year=day.year + year_offset))
    else:
        probe.saver_flag(tmp_path, day)
    alerts, _ = run(tmp_path, day, probe.at(day + timedelta(days=4), 2))
    reasons = []
    if future == 3:
        reasons.append("The contest streak file has a coverage date beyond the allowed future grace.")
    if year_offset == 1:
        reasons.append("The saver flag is for another season. The saver counted as spent.")
    if year_offset is not None:
        reasons.append("The season switch is revoked.")
    assert [item.message for item in alerts] == [exact(day, reason) for reason in reasons]


@pytest.mark.parametrize("offset", [-1, 0, 1])
def test_legacy_expiry_uses_last_et_microsecond(tmp_path, probe, offset):
    day = probe.YEAR_END
    expiry = probe.end(day) + timedelta(microseconds=offset)
    probe.put(tmp_path, "contest_streak.manual.json", {
        "streak": 7, "source_date": str(day), "override_expires_at": expiry.isoformat(),
    })
    probe.auto(tmp_path, day)
    probe.switch(tmp_path, day)
    probe.saver_flag(tmp_path, day)
    alerts, _ = run(tmp_path, day, probe.at(day + timedelta(days=1), 2), isolate=True)
    reason = ("A legacy or expired manual override file is present." if offset <= 0
              else "The selected legacy manual override is ineligible for a live decision.")
    assert [item.message for item in alerts] == [exact(day, reason)]


@pytest.mark.parametrize("mixed", [False, True])
def test_historical_warn_pointer_survives_attention_decoration(tmp_path, probe, mixed):
    day = probe.D
    probe.auto(tmp_path)
    probe.switch(tmp_path, status="revoked" if mixed else "automatic")
    probe.saver_flag(tmp_path, state="not_earned")
    wall = probe.at(day + timedelta(days=1), 2)
    result, bodies = run(tmp_path, day, wall, isolate=True)
    reason = "The saver flag contradicts the recorded streak. The saver counted as spent."
    assert result[0].message == exact(day, reason)
    warn_line = f"- [saver_state] {exact(day, reason + ' (1st observed day)')}"
    if mixed:
        critical = exact(day, "The season switch is revoked.")
        assert bodies == [f"BTS health CRITICAL alert(s):\n- [contest_state] {critical}\n\nWARN attention:\n{warn_line}"]
    else:
        assert bodies == [f"BTS health WARN attention:\n{warn_line}"]
    assert run(tmp_path, day, wall, isolate=True)[1] == []


@pytest.mark.parametrize("kind", ["manual", "automatic", "switch", "saver", "coverage", "masked", "directory"])
def test_historical_state_paths_are_bounded(tmp_path, probe, kind):
    day = probe.D
    picks = tmp_path / "picks"
    probe.auto(picks)
    probe.switch(picks)
    probe.saver_flag(picks)
    if kind == "masked":
        probe.dated(picks)
    if kind == "coverage":
        probe.settled(picks, day)
        probe.settled(picks, day - timedelta(days=1))
    paths = {
        "manual": "account_state/contest_streak.manual.json",
        "automatic": "account_state/contest_streak.json",
        "switch": "account_state/season_contract.json",
        "saver": "account_state/saver_state.json",
        "coverage": f"{day}.json",
        "masked": "account_state/contest_streak.json",
        "directory": "account_state/contest_streak.json",
    }
    target = picks / paths[kind]
    target.unlink(missing_ok=True)
    if kind == "directory":
        target.mkdir()
    else:
        os.mkfifo(target)
    args = [sys.executable, str(Path(__file__).resolve()), str(picks), str(target), str(day)]
    try:
        child = subprocess.run(args, check=True, capture_output=True, text=True, timeout=4)
    except subprocess.TimeoutExpired as error:
        assert b"attempted target" in (error.stdout or b"")
        raise AssertionError(f"Historical contest check blocked reading {kind}") from error
    attempt_count = child.stdout.count("attempted target")
    assert attempt_count == {"masked": 0, "coverage": 2}.get(kind, 1)
    reasons = {
        "manual": "The manual contest streak file is malformed or unreadable.",
        "automatic": "The automatic contest streak file is malformed or unreadable.",
        "directory": "The automatic contest streak file is malformed or unreadable.",
        "switch": "The season switch file is malformed or unreadable.",
        "saver": "The saver flag is uninitialized or unreadable. The saver counted as spent.",
        "coverage": "Coverage trails by one settled pick, the expected ledger lag.",
    }
    output = json.loads(child.stdout.splitlines()[-1])
    assert output == ([exact(day, reasons[kind])] if kind in reasons else [])


@pytest.mark.parametrize("kind", ["manual", "automatic"])
@pytest.mark.parametrize("contents", ["[]", '"observation"'], ids=["array", "string"])
def test_historical_state_rejects_non_object_json(tmp_path, probe, kind, contents):
    day = probe.D
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    filename = "contest_streak.manual.json" if kind == "manual" else "contest_streak.json"
    (state_dir / filename).write_text(contents, encoding="utf-8")

    alerts, _ = run(tmp_path, day, probe.at(day + timedelta(days=1), 2), isolate=True)

    assert [(item.level, item.source, item.message) for item in alerts] == [
        ("CRITICAL", "contest_state", exact(day, f"The {kind} contest streak file is malformed or unreadable.")),
    ]


def test_historical_second_manual_read_is_bounded(tmp_path, probe):
    day = probe.D
    probe.put(tmp_path, "contest_streak.manual.json", {
        "streak": 7, "source_date": str(day),
        "override_expires_at": (probe.end(day) + timedelta(days=1)).isoformat(),
    })
    args = [sys.executable, str(Path(__file__).resolve()), "manual-reread", str(tmp_path), str(day)]
    try:
        child = subprocess.run(args, check=True, capture_output=True, text=True, timeout=4)
    except subprocess.TimeoutExpired as error:
        assert b"replaced selected manual file" in (error.stdout or b"")
        raise AssertionError("Historical contest check blocked on the second manual read") from error
    assert json.loads(child.stdout.splitlines()[-1]) == [
        ["WARN", "contest_state", exact(day, "A legacy or expired manual override file is present.")],
    ]


def manual_reread_child(picks, day):
    from zoneinfo import ZoneInfo

    target = picks / "account_state/contest_streak.manual.json"
    select = contest_state._select_contest_streak_state

    def select_then_replace(*args, **kwargs):
        state = select(*args, **kwargs)
        assert state is not None and state.path == target
        target.unlink()
        os.mkfifo(target)
        print("replaced selected manual file", flush=True)
        return state

    wall = datetime.combine(day + timedelta(days=1), time(2), ZoneInfo("America/New_York"))
    with patch.object(contest_state, "_select_contest_streak_state", select_then_replace):
        alerts, _ = run(picks, day, wall, expected=False, isolate=True)
    print(json.dumps([(item.level, item.source, item.message) for item in alerts]))


def fifo_child(picks, target, day):
    from datetime import date
    from zoneinfo import ZoneInfo

    original = os.open
    read_text = Path.read_text

    def mark(path):
        if path == target:
            print("attempted target", flush=True)

    def observe_open(path, *args, **kwargs):
        mark(path)
        return original(path, *args, **kwargs)

    def observe_read(path, *args, **kwargs):
        mark(path)
        return read_text(path, *args, **kwargs)

    wall = datetime.combine(day + timedelta(days=1), time(2), ZoneInfo("America/New_York"))
    # Other EOD checks retain their readers. Isolate the contest slot to confirm
    # both that it reaches the target and that its own read returns.
    with patch.object(os, "open", observe_open), patch.object(Path, "read_text", observe_read):
        alerts, _ = run(picks, day, wall, isolate=True)
    print(json.dumps([item.message for item in alerts]))


@pytest.mark.parametrize("state", [
    "missing_at_0015", "missing_at_0200", "dated_saver", "new_year", "replaced_auto",
    "morning_silent", "no_entry", "entry_uncertain", "past_cutoff",
    "coverage_after_slate",
])
def test_historical_pointer_in_worst_operator_states(tmp_path, probe, state):
    day = probe.YEAR_END if state == "new_year" else probe.D
    following = day + timedelta(days=1)
    wall = probe.at(following, 2)
    if state == "missing_at_0015":
        wall = probe.at(following, 0, 15)
    if state == "dated_saver":
        probe.dated(tmp_path, day, streak=12)
    elif state == "missing_at_0200":
        probe.switch(tmp_path)
    elif state == "coverage_after_slate":
        probe.dated(tmp_path, day)
        probe.settled(tmp_path, following)
        probe.settled(tmp_path, following + timedelta(days=1))
    elif state in {"new_year", "replaced_auto", "morning_silent"}:
        probe.auto(tmp_path, day)
        probe.switch(tmp_path, day, status="automatic" if state == "replaced_auto" else "revoked")
        probe.saver_flag(tmp_path, following if state == "new_year" else day)
        if state == "replaced_auto":
            probe.auto(tmp_path, day, activation_id="b" * 32, recorded_at=probe.at(following, 1, 10).isoformat())
    if state == "past_cutoff":
        wall = probe.at(following, 23)
    if state == "entry_uncertain":
        (tmp_path / f"{day}.json").write_text('{"delivery_attempted":true,"delivery_confirmed":false}')
    if state == "no_entry":
        (tmp_path / f"{day}.json").write_text('{"result":"void"}')
    before = {p.name: p.read_bytes() for p in tmp_path.glob("account_state/*")}
    result, bodies = run(tmp_path, day, wall, isolate=True)
    assert result and len(bodies) == (0 if state == "coverage_after_slate" else 1)
    for line in "\n".join(bodies).splitlines():
        if not line.startswith("- ["):
            continue
        message = line.split("] ", 1)[1]
        assert message.startswith(f"Prior slate {day} ET. ")
        remaining = message[-len(POINTER):]
        for worst_case, sentence in OPERATOR_ROWS:
            assert remaining.startswith(sentence), worst_case
            remaining = remaining.removeprefix(sentence).lstrip(" ")
        assert remaining == ""
        assert_safe(message, day)
    for item in result:
        assert_safe(item.message, day)
    assert {p.name: p.read_bytes() for p in tmp_path.glob("account_state/*")} == before
    if state == "morning_silent":
        from bts import scheduler
        from bts.contest_state import load_decision_streak_state

        morning = probe.at(following, 10)
        config = {"bluesky": {"dm_recipient": "offline.test"}}
        with patch.object(alert, "send_dm", return_value="offline-ack") as transport:
            scheduler._morning_state_check(config, tmp_path, str(following), morning, probe.at(following, 11))
            assert f"Morning check for {following}" in transport.call_args.args[1]
            transport.reset_mock()
            probe.auto(tmp_path, following, source_date=str(day), recorded_at=probe.at(following, 9).isoformat())
            probe.switch(tmp_path, following)
            probe.saver_flag(tmp_path, following)
            scheduler._morning_state_check(config, tmp_path, str(following), morning, probe.at(following, 11))
            transport.assert_not_called()
        assert load_decision_streak_state(
            tmp_path, require_contest_state=True, now=morning,
        ).observation_kind == "automatic"


@pytest.mark.parametrize("mode", ["omitted", "same", "future", "rollover"])
def test_nonhistorical_clock_and_wire_match_current_path(tmp_path, probe, mode):
    day = probe.D
    following = day + timedelta(days=1)
    probe.auto(tmp_path)
    probe.switch(tmp_path, status="revoked")
    probe.saver_flag(tmp_path, state="not_earned")
    evening = probe.at(day, 23, 50)
    wall = probe.at(following, 0, 1) if mode == "rollover" else evening
    reference = contest_state.check(tmp_path, now=wall, expected=True)
    options = {"omitted": {}, "same": {"today": day}, "future": {"today": following}, "rollover": {"today": day}}
    if mode == "rollover":
        class LiveClock(datetime):
            @classmethod
            def now(cls, tz=None):
                return wall.astimezone(tz)

        original = contest_state.check

        def crossed_midnight(*args, **kwargs):
            with patch.object(contest_state, "datetime", LiveClock):
                return original(*args, **kwargs)

        check = patch.object(contest_state, "check", side_effect=crossed_midnight)
    else:
        check = patch.object(contest_state, "check", wraps=contest_state.check)
    with check as called:
        actual, bodies = run(tmp_path, day, evening, inputs=options[mode], isolate=True)
    called.assert_called_once_with(tmp_path, expected=True)
    assert actual == reference
    from bts.health.attention import build_policy_alerts
    _, attention = build_policy_alerts(reference, streaks={})
    assert bodies == [alert.format_health_dm_body(
        [item for item in reference if item.level == "CRITICAL"], attention,
    )]


def test_eod_scheduler_uses_processed_observation(tmp_path):
    subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "scheduler", str(tmp_path)],
        check=True, capture_output=True, text=True, timeout=15,
    )


def scheduler_child(tmp_path):
    sys.path.insert(0, str(REVIEW / "slice-1-carbon"))
    from test_slice1_probes_v3 import RehearsalWorld, _H, _setup_revoked_box, et_dt

    _H.REPO_ROOT = REPO_ROOT
    schedules = _H.default_opening_days_schedules()
    day = "2027-03-24"
    following = "2027-03-25"
    game = schedules[day][0]
    game.start_et = et_dt(2027, 3, 24, 21, 5)
    game.away_confirm_et = et_dt(2027, 3, 24, 18, 45)
    game.home_confirm_et = et_dt(2027, 3, 24, 19, 45)
    game.final_et = et_dt(2027, 3, 25, 0, 15)
    game.away_hits = 0
    with RehearsalWorld(tmp_path, initial_et=et_dt(2027, 3, 24, 9), schedules=schedules) as world:
        _setup_revoked_box(world)
        path_args = ["--picks-dir", str(world.picks_dir)]
        assert world.invoke_cli(["saver-state", "--init", "not_earned", "--season", "2027", *path_args]).exit_code == 0
        (world.park_dir / "park_drag_manifest.json").write_text(json.dumps({
            "max_source_game_date": day, "generated_at": "2027-03-25T00:10:00", "export_rows": 2,
        }))
        world.clock.set_et(et_dt(2027, 3, 24, 9, 30))
        assert world.invoke_cli([
            "set-contest-streak", "--date", day, "--streak", "0",
            "--best-streak", "0", "--saver", "not_earned", *path_args,
        ]).exit_code == 0
        world.clock.set_et(et_dt(2027, 3, 24, 10))
        class RunnerClock(datetime):
            @classmethod
            def now(cls, tz):
                return world.clock.now_et().astimezone(tz)

        disk_warn = alert.Alert(level="WARN", source="disk_fill", message="Synthetic disk usage warning")
        with (
            patch.object(runner, "datetime", RunnerClock, create=True),
            patch.object(runner.disk_fill, "check", return_value=[disk_warn]),
        ):
            world.run_scheduler_day(day, stop_at_et=et_dt(2027, 3, 25, 1))
        (body,) = [dm["text"] for dm in world.dms if dm["text"].startswith("BTS health ")]
        assert "[contest_state]" not in body and "[saver_state]" not in body, body
        assert f"--date {following}" not in body
        status = world.picks_dir.parent / "health_state" / "health_dm_delivery_status.json"
        assert json.loads(status.read_text())["sent_day"] == day


def test_historical_interrupt_propagates_without_contest_write(tmp_path, probe):
    path = probe.auto(tmp_path)
    before = path.read_bytes()
    parse = json.loads

    def interrupt_at_parse(contents, *args, **kwargs):
        if contents == before.decode("utf-8"):
            raise KeyboardInterrupt
        return parse(contents, *args, **kwargs)

    with patch.object(json, "loads", side_effect=interrupt_at_parse):
        with pytest.raises(KeyboardInterrupt):
            run(tmp_path, probe.D, probe.at(probe.D + timedelta(days=1), 2), isolate=True)
    assert path.read_bytes() == before


def test_historical_failed_delivery_can_retry_then_deduplicates(tmp_path, probe):
    day = probe.D
    wall = probe.at(day + timedelta(days=1), 2)
    picks = tmp_path / "picks"
    picks.mkdir()
    status = picks / ".health/health_dm_delivery_status.json"
    failed, first = run(picks, day, wall, send=OSError("offline"), isolate=True)
    assert json.loads(status.read_text())["status"] == "failed"
    retried, second = run(picks, day, wall, isolate=True)
    assert retried == failed and second[0].endswith(first[0])
    assert json.loads(status.read_text())["sent_day"] == str(day)
    assert run(picks, day, wall, isolate=True)[1] == []


if __name__ == "__main__":
    from datetime import date

    if sys.argv[1] == "scheduler":
        scheduler_child(Path(sys.argv[2]))
    elif sys.argv[1] == "manual-reread":
        manual_reread_child(Path(sys.argv[2]), date.fromisoformat(sys.argv[3]))
    else:
        fifo_child(Path(sys.argv[1]), Path(sys.argv[2]), date.fromisoformat(sys.argv[3]))
