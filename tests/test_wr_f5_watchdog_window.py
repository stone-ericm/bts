"""Replay installed watchdog slots through the real CLI with offline transports."""
import gzip
import json
import shlex
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import entry_receipt
from bts.cli import cli
from tests.test_entry_intent import ENTRY_CMD, RESEARCH, REPO, box
from tests.test_wr_f1_wrong_game import RIGHT, WRONG, row

DAY = "2027-03-24"
ET = ZoneInfo("America/New_York")
ENTER = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
OLD = f"*/15 10-23 * * * uv run {ENTRY_CMD} # BTS-HETZNER"
FOREIGN = "0 9 * * * echo foreign"


def watchdog(lines):
    line, = [line for line in lines if ENTRY_CMD in line and not line.startswith("#")]
    return line


@pytest.fixture
def installed(box):
    box.config.write_text(ENTER)
    result = box.run("install")
    assert result.returncode == 0, result.stderr
    return watchdog(box.lines())


@pytest.mark.parametrize("action", ["show", "install"])
def test_m1_wrapped_schedule_and_flag(box, action):
    box.config.write_text(ENTER)
    result = box.run(action)
    assert result.returncode == 0, result.stderr
    lines = result.stdout.splitlines() if action == "show" else box.lines()
    line = watchdog(lines)
    assert line.startswith("*/15 * * * * ")
    assert "/usr/bin/flock -n /home/bts/.check-pick-entered.lock /usr/bin/timeout --verbose -k 30 840 " in line
    assert "--cron" in shlex.split(line)


def test_m4_replace_old_watchdog(box):
    box.cron.write_text(f"{FOREIGN}\n{OLD}\n# {OLD}\n")
    box.config.write_text(ENTER)
    assert box.run("install").returncode == 0
    assert box.lines()[0] == FOREIGN
    assert sum(ENTRY_CMD in line for line in box.lines()) == 1
    assert watchdog(box.lines()).startswith("*/15 * * * * ")


def test_n1_n3_n4_n5_install_remove_and_other_lines(box):
    legacy = OLD.removesuffix(" # BTS-HETZNER")
    box.cron.write_text(f"{FOREIGN}\n{legacy}\n# {OLD}\n")
    box.config.write_text(ENTER)
    assert box.run("install").returncode == 0
    enter = [line for line in box.lines() if line.endswith("# BTS-HETZNER")]
    assert box.run("remove").returncode == 0
    assert box.lines() == [FOREIGN, legacy]
    box.config.write_text(RESEARCH)
    assert box.run("install").returncode == 0
    research = box.lines()[2:]
    assert all(ENTRY_CMD not in line and "--dm-recipient" not in line
               for line in research if "fetch-contest-streak" in line or ENTRY_CMD in line)
    assert research == [
        line.replace(" --dm-recipient stonehengee.bsky.social", "")
        if "fetch-contest-streak" in line else line
        for line in enter if ENTRY_CMD not in line
    ]
    before = box.cron.read_bytes()
    Path(f"{box.cron}.unreadable").touch()
    shown = box.run("show")
    assert shown.returncode == 0 and "(none)" in shown.stdout and "Would install:" in shown.stdout
    for action in ("install", "remove"):
        assert box.run(action).returncode == 1
        assert box.cron.read_bytes() == before


@pytest.mark.parametrize("body", [
    '[scheduler]\npick_delivery = "private"\n',
    '[scheduler\nentry_intent = "enter"\n',
    '[scheduler]\nentry_intent = "enter"\npick_delivery = "private"\n',
    ENTER.replace("contest_state_required = true\n", ""),
])
def test_n2_refuse_bad_intent_without_accessing_crontab(box, body):
    box.config.write_text(body)
    box.cron.write_text(OLD + "\n")
    for action in ("show", "install"):
        assert box.run(action).returncode == 1
        assert not Path(f"{box.cron}.calls").exists()
        assert box.cron.read_text() == OLD + "\n"


def slots(line, day=DAY):
    minute, hour, *_ = line.split()
    step = int(minute.removeprefix("*/"))
    hours = range(24) if hour == "*" else range(10, 24)
    midnight = datetime.fromisoformat(day).replace(tzinfo=ET)
    start = midnight.astimezone(timezone.utc)
    stop = (midnight + timedelta(days=1)).astimezone(timezone.utc)
    while start < stop:
        et = start.astimezone(ET)
        if et.hour in hours:
            yield et.strftime("%H:%M")
        start += timedelta(minutes=step)


@pytest.fixture
def replay(tmp_path, installed, monkeypatch):
    fixture = (REPO / "docs/reviews/2026-11-18-whole-range/closing-test/codex-blind"
               / "evidence/cron-early-entry.json.gz")
    with gzip.open(fixture, "rt") as stream:
        snapshot = next(event[-1] for event in json.load(stream) if event[1] == "snapshot")
    saved = snapshot[f"data/picks/{DAY}.json"]
    decision = snapshot[f"data/picks/{DAY}/decision.json"]
    picks = tmp_path / "picks"
    picks.mkdir()
    config = tmp_path / "config.toml"
    config.write_text(ENTER)
    marker = tmp_path / "health_state" / "pick_entry_check.json"
    args = shlex.split(installed, comments=True)
    args = args[args.index("check-pick-entered"):args.index(">>")]
    args[args.index("--picks-dir") + 1] = str(picks)
    username = args[args.index("--expected-username") + 1]
    recipient = args[args.index("--dm-recipient") + 1]
    calls, dms = [], []
    runner = CliRunner()

    def record(name, response):
        def transport(*args, **kwargs):
            calls.append(name)
            return response
        return transport

    def unexpected(*args, **kwargs):
        pytest.fail("unexpected network access")

    for name in ("connect", "connect_ex", "sendto"):
        monkeypatch.setattr(f"socket.socket.{name}", unexpected)
    monkeypatch.setattr("socket.getaddrinfo", unexpected)
    monkeypatch.setattr("bts.leaderboard.auth.load_session_cookies", record("cookies", {}))
    monkeypatch.setattr("bts.leaderboard.auth.extract_uid", lambda _: "test")
    monkeypatch.setattr("bts.leaderboard.auth.fetch_login_session", record(
        "session", SimpleNamespace(user_id="test", username=username, xsid="test")))
    monkeypatch.setattr("bts.contest_fetch.fetch_profile", record("profile", {"predictions": []}))
    monkeypatch.setattr("bts.cli._fetch_bts_to_mlb", record("players", {1: saved["pick"]["batter_id"]}))
    monkeypatch.setattr("bts.scheduler.fetch_schedule", lambda _: [{"gamePk": 101}])
    monkeypatch.setattr("bts.dm.send_dm", lambda _, msg: dms.append(msg))
    monkeypatch.setattr("time.monotonic", lambda: 1000)

    class Replay:
        def pick(self, pitch, *, day=DAY, deliver="05:10", committed=True, pitch_day=None):
            first = datetime.fromisoformat(f"{pitch_day or day}T{pitch}").replace(tzinfo=ET)
            commit = datetime.fromisoformat(f"{day}T{deliver}").replace(tzinfo=ET).isoformat()
            pick = {**saved, "date": day, "run_time": commit, "delivered_at": commit,
                    "pick": {**saved["pick"], "game_time": first.isoformat()}}
            if not committed:
                pick.update(notification_sent=False, notification_id=None, delivered_at=None,
                            delivery_attempted=False)
            (picks / f"{day}.json").write_text(json.dumps(pick))
            if committed:
                target = picks / day / "decision.json"
                target.parent.mkdir(exist_ok=True)
                target.write_text(json.dumps({**decision, "date": day, "finalized_at": commit}))

        def units(self, unit, stamp="030000"):
            target = entry_receipt.units_snapshot_dir(picks)
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{DAY}T{stamp}Z.json.gz").write_bytes(gzip.compress(
                json.dumps({"units": [unit]}).encode()))

        def marker(self, day=DAY):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"date": day, "status": "confirmed"}))

        def check(self, at, *, day=DAY, present=True, manual=False):
            calls.clear()
            dms.clear()
            before = marker.read_bytes() if marker.exists() else None
            count = len(entry_receipt.discover(picks, day))
            monkeypatch.setattr("bts.contest_fetch.fetch_pending_predictions",
                                record("pending", [row()] if present else []))
            monkeypatch.setattr("bts.cli._fetch_rounds",
                                record("rounds", {1: date.fromisoformat(day)}))
            command = [arg for arg in args if arg != "--cron"] if manual else args
            result = runner.invoke(cli, [*command, "--now-et", f"{day}T{at}"])
            assert result.exit_code in (0, 1), (result.output, result.exception)
            receipts = entry_receipt.discover(picks, day)
            assert len(receipts) == count + 1
            receipt, = [r for r in receipts if r["started_at"].startswith(f"{day}T{at}:00")]
            after = marker.read_bytes() if marker.exists() else None
            return receipt["outcome"], list(calls), list(dms), before != after

        def morning(self):
            dms.clear()
            result = runner.invoke(cli, ["day-outcome", "--date", DAY, "--config", str(config),
                                         "--picks-dir", str(picks), "--expected-username", username,
                                         "--dm-recipient", recipient])
            assert result.exit_code in (0, 1), (result.output, result.exception)
            return result.exit_code, list(dms)

    return Replay()


@pytest.mark.parametrize("pitch, start, end, expected", [
    pytest.param("06:10", "05:00", "06:15", ["05:15", "05:30", "05:45", "06:00"], id="M2"),
    pytest.param("10:05", "09:00", "10:00", ["09:00", "09:15", "09:30", "09:45"], id="M3"),
])
@pytest.mark.parametrize("present", [False, True], ids=["missing", "M7-confirmed"])
def test_early_slate(installed, replay, pitch, start, end, expected, present):
    replay.units(RIGHT)
    if pitch == "10:05":
        replay.pick(pitch, deliver="08:55")
    observed = {}
    for at in slots(installed):
        if start <= at <= end:
            if pitch == "06:10" and at == expected[0]:
                replay.pick(pitch)
            observed[at] = replay.check(at, present=present)
    reads = [at for at, result in observed.items() if "pending" in result[1]]
    assert reads == (expected[:1] if present else expected)
    if pitch == "06:10":
        assert observed["05:00"] == ("no_pick_file", [], [], False)
    assert observed[end] == ("outside_window", [], [], False)
    if present:
        assert observed[expected[0]][0] == "observed"
        for at in expected[1:]:
            assert observed[at] == ("already_confirmed", [], [], False)
        assert replay.morning() == (0, [])


@pytest.mark.parametrize("wrong_game", [True, False], ids=["M5-late-binding", "M6-removed-entry"])
def test_preserve_base_warnings(installed, replay, wrong_game):
    replay.pick("10:50")
    if not wrong_game:
        replay.units(RIGHT)
    observed = {}
    for at in slots(installed):
        if "09:45" <= at <= "10:45":
            if at == "10:00" and wrong_game:
                replay.units(WRONG, "140000")
            observed[at] = replay.check(at, present=at == "09:45" or wrong_game)
    assert observed.get("09:45") == ("base_slot_available", [], [], False)
    assert "pending" in observed["10:00"][1]
    warning = "the entry is recorded in game 102" if wrong_game else "BTS pick NOT entered"
    assert len(observed["10:00"][2]) == 1 and warning in observed["10:00"][2][0]
    assert observed["10:00"][3]
    assert observed["10:45"] == ("outside_window", [], [], False)
    assert sum(len(result[2]) for result in observed.values()) == 3
    code, alerts = replay.morning()
    morning_warning = "ENTRY UNVERIFIABLE" if wrong_game else "NOT ENTERED"
    assert code == 1 and len(alerts) == 1 and morning_warning in alerts[0]


def test_manual_early_check_still_confirms(replay):
    replay.pick("10:50")
    outcome, calls, dms, changed = replay.check("09:45", manual=True)
    assert outcome == "observed" and "pending" in calls and not dms and changed


def test_n6_local_gates(replay):
    assert replay.check("00:00") == ("no_pick_file", [], [], False)
    replay.pick("00:35", day="2027-03-23", pitch_day=DAY, deliver="23:00")
    replay.marker("2027-03-23")
    assert replay.check("00:15") == ("no_pick_file", [], [], False)
    replay.pick("06:10", committed=False, deliver="03:00")
    assert replay.check("03:00") == ("not_committed", [], [], False)
    replay.pick("06:25", deliver="04:00")
    assert replay.check("05:00") == ("outside_window", [], [], False)
    assert "pending" in replay.check("05:15")[1]
    replay.marker()
    assert replay.check("05:30") == ("already_confirmed", [], [], False)
    assert replay.check("09:45") == ("outside_window", [], [], False)


def test_n7_evening_slots(installed, replay):
    replay.pick("19:10")
    replay.units(RIGHT)
    trace = {at: replay.check(at) for at in slots(installed) if "17:45" <= at <= "19:15"}
    assert [at for at, result in trace.items() if result[0] != "outside_window"] == [
        "18:00", "18:15", "18:30", "18:45", "19:00",
    ]
    assert "pending" in trace["18:00"][1]
    assert all(result == ("already_confirmed", [], [], False)
               for at, result in trace.items() if "18:00" < at <= "19:00")


def test_n8_clock_slot_counts(installed, replay):
    all_slots = list(slots(installed))
    assert len(all_slots) == 96 and (all_slots[0], all_slots[-1]) == ("00:00", "23:45")
    spring = list(slots(installed, "2027-03-14"))
    autumn = list(slots(installed, "2027-11-07"))
    assert len(spring) == 92 and "02:00" not in spring
    assert len(autumn) == 100 and autumn.count("01:00") == 2
    assert replay.check("00:00", day="2028-01-01") == ("no_pick_file", [], [], False)
