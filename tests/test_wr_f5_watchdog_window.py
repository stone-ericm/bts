"""Replay installed watchdog slots through the real CLI with offline transports."""
import gzip
import json
import os
import shlex
import subprocess
import sys
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import entry_receipt
from bts.cli import cli
from bts.health import pick_entry
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
        def pick(self, pitch, *, day=DAY, deliver="05:10", committed=True, pitch_day=None, game=None):
            first = datetime.fromisoformat(f"{pitch_day or day}T{pitch}").replace(tzinfo=ET)
            commit = datetime.fromisoformat(f"{day}T{deliver}").replace(tzinfo=ET).isoformat()
            pick = {**saved, "date": day, "run_time": commit, "delivered_at": commit,
                    "pick": {**saved["pick"], "game_time": first.isoformat()}}
            if game is not None:
                pick["pick"]["game_pk"] = game
            if not committed:
                pick.update(notification_sent=False, notification_id=None, delivered_at=None,
                            delivery_attempted=False)
            (picks / f"{day}.json").write_text(json.dumps(pick))
            if committed:
                target = picks / day / "decision.json"
                target.parent.mkdir(exist_ok=True)
                replacement = {**decision, "date": day, "finalized_at": commit}
                if game is not None:
                    replacement["primary"] = {**decision["primary"], "game_pk": game}
                target.write_text(json.dumps(replacement))

        def units(self, unit, stamp="030000"):
            target = entry_receipt.units_snapshot_dir(picks)
            target.mkdir(parents=True, exist_ok=True)
            (target / f"{DAY}T{stamp}Z.json.gz").write_bytes(gzip.compress(
                json.dumps({"units": [unit]}).encode()))

        def marker(self, day=DAY):
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"date": day, "status": "confirmed"}))

        def check(self, at, *, day=DAY, present=True, manual=False, raises=None):
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
            if raises:
                assert isinstance(result.exception, raises), (result.output, result.exception)
                assert not calls and not dms and marker.read_bytes() == before
                return
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

    world = Replay()
    world.marker_path = marker
    world.picks_dir = picks
    return world


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
    receipt, = entry_receipt.discover(replay.picks_dir, DAY)
    assert "early_cron_run" not in receipt
    assert "early_cron_run" not in json.loads(replay.marker_path.read_bytes())


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


@pytest.mark.parametrize("replacement", [False, True], ids=["R1-corrected", "R2-replaced"])
def test_early_confirmation_does_not_stop_original_slots(replay, replacement):
    replay.pick("10:05", deliver="09:40")
    assert "pending" in replay.check("09:45")[1]
    early, = entry_receipt.discover(replay.picks_dir, DAY)
    assert early["qualification"]["slots"][0]["state"] == "unit_unverified"
    if replacement:
        replay.pick("10:50", deliver="09:52", game=102)
        replay.units(RIGHT, "140000")
        observed_game = 101
    else:
        replay.pick("10:50", deliver="09:40")
        replay.units(WRONG, "140000")
        observed_game = 102
    for at in ("10:00", "10:15", "10:30"):
        _, calls, dms, _ = replay.check(at)
        assert "pending" in calls
        assert any(f"entry is recorded in game {observed_game}" in dm for dm in dms)
    code, dms = replay.morning()
    assert code == 1 and "ENTRY UNVERIFIABLE" in dms[0]
    assert early["early_cron_run"] is True
    assert "early_cron_run" not in json.loads(replay.marker_path.read_bytes())


@pytest.mark.parametrize("failed_send", [False, True], ids=["R3-spent-tiers", "failed-send"])
def test_original_slot_sends_its_initial_alert(replay, monkeypatch, failed_send):
    replay.pick("10:05", deliver="09:40")
    if failed_send:
        def fail(*args):
            raise OSError("send failed")
        with monkeypatch.context() as patch:
            patch.setattr("bts.dm.send_dm", fail)
            replay.check("09:45", present=False)
        assert json.loads(replay.marker_path.read_bytes())["status"] == "dm_failed"
    else:
        assert replay.check("09:45", present=False)[2]
    replay.pick("10:06", deliver="09:40")
    assert "BTS pick NOT entered" in replay.check("10:00", present=False)[2][0]


@pytest.mark.parametrize("pitch", ["10:06", "10:15", "10:20"])
def test_r5_r6_sole_original_slot(replay, pitch):
    replay.pick(pitch, deliver="09:40")
    assert replay.check("09:45") == ("base_slot_available", [], [], False)
    replay.units(WRONG, "140000")
    assert "entry is recorded in game 102" in replay.check("10:00")[2][0]
    assert replay.check("10:15") == ("outside_window", [], [], False)
    code, dms = replay.morning()
    assert code == 1 and "ENTRY UNVERIFIABLE" in dms[0]


def test_original_slot_keeps_early_wrong_game_observation(replay):
    replay.pick("10:05", deliver="09:25")
    replay.units(WRONG, "133000")
    assert replay.check("09:30")[2]
    for capture in entry_receipt.units_snapshot_dir(replay.picks_dir).iterdir():
        capture.unlink()
    replay.pick("10:50", deliver="09:25")
    assert replay.check("10:00")[2]
    marker = json.loads(replay.marker_path.read_bytes())
    assert marker["game_unresolved"] is True and marker["reason"] == "wrong_game"


def test_original_slot_sends_one_all_clear_for_early_alert(replay):
    replay.pick("10:05", deliver="09:25")
    assert replay.check("09:30", present=False)[2]
    replay.pick("10:50", deliver="09:25")
    replay.units(RIGHT, "140000")
    assert "entry confirmed" in replay.check("10:00")[2][0]
    assert replay.check("10:15") == ("already_confirmed", [], [], False)
    assert replay.morning() == (0, [])
    receipt, = [r for r in entry_receipt.discover(replay.picks_dir, DAY) if "T10:00:" in r["started_at"]]
    assert "early_cron_run" not in receipt


@pytest.mark.parametrize("present", [True, False])
def test_morning_filters_early_evidence_at_inclusive_window_start(replay, present):
    replay.pick("10:05", deliver="09:40")
    replay.units(RIGHT, "134000")
    assert "pending" in replay.check("09:45", present=present)[1]
    replay.pick("11:15", deliver="09:40")
    code, alerts = replay.morning()
    assert code == 1 and "ENTRY UNVERIFIABLE" in alerts[0]
    assert "observed x1" in alerts[0]


@pytest.mark.parametrize("day, prior", [(DAY, "2027-03-23"), ("2028-01-01", "2027-12-31")])
def test_r4_displaced_marker_health(replay, day, prior):
    replay.pick("19:10", day=prior, deliver="18:00")
    replay.pick("06:10", day=day, deliver="04:55")
    marker = replay.marker_path
    marker.parent.mkdir()
    original = f' {{ "date": "{prior}", "status": "dm_failed" }}\n'.encode()
    marker.write_bytes(original)
    assert "pending" in replay.check("05:00", day=day)[1]
    now = datetime.fromisoformat(f"{day}T05:01").replace(tzinfo=ET)
    alert, = pick_entry.check(replay.picks_dir, date.fromisoformat(prior), now)
    assert alert.level == "CRITICAL" and "operator was never reached" in alert.message
    assert marker.with_name(f"pick_entry_check.{prior}.json").read_bytes() == original


def test_displaced_copy_survives_interrupted_marker_write(replay, monkeypatch):
    prior = "2027-03-23"
    replay.pick("06:10", deliver="04:55")
    marker = replay.marker_path
    marker.parent.mkdir()
    original = json.dumps({"date": prior, "status": "alerted"}).encode()
    marker.write_bytes(original)
    copy = marker.with_name(f"pick_entry_check.{prior}.json")

    def interrupt(path, body):
        assert path == marker
        assert copy.read_bytes() == original
        raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        patch.setattr("bts.cli._atomic_write_json", interrupt)
        replay.check("05:00")
    assert marker.read_bytes() == original
    marker.write_bytes(original + b"\n")
    assert "pending" in replay.check("05:15")[1]
    assert copy.read_bytes() == original


@pytest.mark.parametrize("kind", ["malformed", "non-object", "wrong-date", "directory", "fifo"])
def test_invalid_health_copy_is_ignored(replay, kind):
    prior = "2027-03-23"
    replay.marker()
    copy = replay.marker_path.with_name(f"pick_entry_check.{prior}.json")
    if kind == "directory":
        copy.mkdir()
    elif kind == "fifo":
        os.mkfifo(copy)
    elif kind == "non-object":
        copy.write_text("[]")
    else:
        copy.write_text("[" if kind == "malformed" else json.dumps({"date": DAY, "status": "alerted"}))
    assert pick_entry.check(replay.picks_dir, date.fromisoformat(prior)) == []


def test_malformed_marker_is_replaced_without_copy(replay):
    replay.marker()
    replay.marker_path.write_bytes(b"{")
    replay.pick("06:10", deliver="04:55")
    assert "pending" in replay.check("05:00")[1]
    assert list(replay.marker_path.parent.glob("pick_entry_check.*.json")) == []


def test_health_does_not_block_on_live_fifo(replay):
    marker = replay.marker_path
    marker.parent.mkdir()
    os.mkfifo(marker)
    result = subprocess.run(
        [sys.executable, "-c",
         "from pathlib import Path; import sys; from bts.health.pick_entry import check; "
         "assert check(Path(sys.argv[1])) == []", str(replay.picks_dir)],
        capture_output=True, text=True, timeout=15,
    )
    assert result.returncode == 0, result.stderr


def test_health_live_non_object_marker_raises_as_on_base(replay):
    replay.marker()
    replay.marker_path.write_text("[]")
    with pytest.raises(AttributeError):
        pick_entry.check(replay.picks_dir, date.fromisoformat(DAY))


def test_cron_live_non_object_marker_raises_like_manual(replay):
    replay.marker()
    replay.marker_path.write_text("[]")
    replay.pick("06:10", deliver="04:55")
    replay.check("05:00", manual=True, raises=AttributeError)
    replay.check("05:15", raises=AttributeError)
    assert list(replay.marker_path.parent.glob("pick_entry_check.*.json")) == []
