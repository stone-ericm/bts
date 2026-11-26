"""Replay installed watchdog slots through the real CLI with offline transports."""
import gzip
import json
import os
import shlex
import signal
import subprocess
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import entry_receipt
from bts.cli import cli
from bts.health import pick_entry, runner as health_runner
from tests.test_entry_intent import ENTRY_CMD, RESEARCH, REPO, box
from tests.test_wr_f1_wrong_game import RIGHT, WRONG, row
from tests.test_wr_f1b_lock import _await, _records, _runner

DAY = "2027-03-24"
ET = ZoneInfo("America/New_York")
ENTER = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
OLD = f"*/15 10-23 * * * uv run {ENTRY_CMD} # BTS-HETZNER"
FOREIGN = "0 9 * * * echo foreign"


@pytest.fixture(autouse=True)
def bound_file_checks():
    def timed_out(*_):
        raise TimeoutError("watchdog or reader did not return")

    previous = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(15)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous)


@pytest.fixture(autouse=True)
def bound_installer(monkeypatch):
    from tests import test_entry_intent

    def run(*args, **kwargs):
        return subprocess.run(*args, **{**kwargs, "timeout": 15})

    monkeypatch.setattr(test_entry_intent, "subprocess", SimpleNamespace(run=run))


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

        def check(self, at, *, day=DAY, present=True, manual=False, unit=77, fetch_seconds=0,
                  interrupt=False):
            calls.clear()
            dms.clear()
            before = marker.read_bytes() if marker.exists() else None
            count = len(entry_receipt.discover(picks, day))
            def pending(*args, **kwargs):
                calls.append("pending")
                monkeypatch.setattr("time.monotonic", lambda: 1000 + fetch_seconds)
                return [row(unit)] if present else []

            monkeypatch.setattr("time.monotonic", lambda: 1000)
            monkeypatch.setattr("bts.contest_fetch.fetch_pending_predictions", pending)
            monkeypatch.setattr("bts.cli._fetch_rounds",
                                record("rounds", {1: date.fromisoformat(day)}))
            command = [arg for arg in args if arg != "--cron"] if manual else args
            result = runner.invoke(cli, [*command, "--now-et", f"{day}T{at}"])
            self.result = result
            assert result.exit_code in (0, 1), (result.output, result.exception)
            receipts = entry_receipt.discover(picks, day)
            assert len(receipts) == count + 1
            receipt, = [r for r in receipts if r["started_at"].startswith(f"{day}T{at}:00")]
            if interrupt:
                assert receipt["detail"] == "raised KeyboardInterrupt"
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
    world.early_path = marker.with_name('pick_entry_check.early') / f'{DAY}.json'
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
        assert json.loads(replay.early_path.read_bytes())["status"] == "dm_failed"
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


@pytest.mark.parametrize("replace_first", [False, True])
def test_interrupt_around_early_marker_replace(replay, monkeypatch, replace_first):
    replay.pick("06:10", deliver="04:55")
    replay.check("05:00", present=False)
    prior = replay.early_path.read_bytes()
    replace = os.replace

    def interrupt(src, dst, *args, **kwargs):
        if str(dst) == str(replay.early_path):
            if replace_first:
                replace(src, dst, *args, **kwargs)
            raise KeyboardInterrupt()
        return replace(src, dst, *args, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr("os.replace", interrupt)
        _, calls, dms, _ = replay.check("05:15", interrupt=True)
        assert "pending" in calls and not dms
    assert (replay.early_path.read_bytes() != prior) is replace_first
    assert not replay.marker_path.exists()
    outcome, calls, dms, _ = replay.check("05:30")
    assert outcome == ("already_confirmed" if replace_first else "observed")
    assert ("pending" in calls) is not replace_first
    assert bool(dms) is not replace_first


@pytest.mark.parametrize("kind", ["fifo", "non-object", "bad-encoding", "oversize", "bad-tiers"])
def test_invalid_early_marker_is_no_prior_state(replay, kind):
    replay.pick("06:10", deliver="04:55")
    replay.check("05:00")
    marker = replay.early_path
    saved = marker.read_bytes()
    marker.unlink()
    if kind == "fifo":
        os.mkfifo(marker)
    else:
        marker.write_bytes({
            "non-object": b"[]",
            "bad-encoding": b"\xff",
            "oversize": saved + b" " * 65536,
            "bad-tiers": json.dumps({**json.loads(saved), "status": "dm_failed",
                                     "escalations": 1}).encode(),
        }[kind])
    assert health_at(replay) == []
    assert "pending" in replay.check("05:15")[1]


def test_unavailable_early_decision_falls_back_to_delivery(replay):
    replay.pick("06:10", deliver="04:55")
    decision = replay.picks_dir / DAY / "decision.json"
    decision.unlink()
    os.mkfifo(decision)
    assert "pending" in replay.check("05:00")[1]
    receipt, = entry_receipt.discover(replay.picks_dir, DAY)
    assert receipt["selection"]["commit"]["decision_sha256"] is None
    assert receipt["early_cron_run"] is True


def test_pitch_correction_reopens_confirmation(replay):
    replay.pick("06:10", deliver="04:55")
    replay.check("05:00")
    replay.pick("06:20", deliver="04:55")
    assert "pending" in replay.check("05:15")[1]


def test_pitch_correction_keeps_unresolved_wrong_game(replay):
    replay.pick("06:10", deliver="04:55")
    replay.units(WRONG)
    assert "entry is recorded in game 102" in replay.check("05:00")[2][0]
    replay.pick("06:20", deliver="04:55")
    outcome, calls, dms, _ = replay.check("05:15", unit=78)
    assert outcome == "observed" and "pending" in calls and len(dms) == 1
    assert json.loads(replay.early_path.read_bytes())["game_unresolved"] is True


def test_early_replacement_does_not_inherit_spent_tiers(replay):
    replay.pick("06:10", deliver="04:55")
    assert "BTS pick NOT entered" in replay.check("05:30", present=False)[2][0]
    replay.pick("06:10", deliver="05:35", game=102)
    assert "BTS pick NOT entered" in replay.check("05:45", present=False)[2][0]


def test_early_failed_warning_retries_and_pages_after_cutoff(replay, monkeypatch):
    replay.pick("06:10", deliver="04:55")
    with monkeypatch.context() as patch:
        patch.setattr("bts.dm.send_dm", fail_warning)
        replay.check("05:00", present=False)
    assert health_at(replay, at="06:04") == []
    assert [a.level for a in health_at(replay, at="06:05")] == ["CRITICAL"]
    assert len(replay.check("05:15", present=False)[2]) == 1
    assert [a.level for a in health_at(replay, at="06:05")] == ["WARN"]


def test_both_ledgers_can_warn_and_send_one_all_clear(replay):
    replay.pick("10:05", deliver="09:00")
    assert len(replay.check("09:15", present=False)[2]) == 1
    assert len(replay.check("09:16", manual=True, present=False)[2]) == 1
    replay.units(RIGHT)
    assert "entry confirmed" in replay.check("09:30")[2][0]
    assert "entry confirmed" in replay.check("09:31", manual=True)[2][0]
    assert replay.check("09:45")[2] == []
    assert replay.check("09:46", manual=True)[2] == []


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
    assert marker.read_bytes() == original



def test_early_replacement_is_checked(replay):
    replay.pick("06:10", deliver="04:55")
    assert "pending" in replay.check("05:00")[1]
    replay.pick("06:10", deliver="05:10", game=102)
    replay.units(RIGHT)
    assert "entry is recorded in game 101" in replay.check("05:15")[2][0]


def test_r8_manual_rechecks_after_early_confirmation(replay):
    replay.pick("10:05", deliver="09:40")
    assert "pending" in replay.check("09:45")[1]
    replay.pick("10:50", deliver="09:40")
    replay.units(WRONG, "140000")
    assert "entry is recorded in game 102" in replay.check("10:00", manual=True)[2][0]


def test_r9_early_warning_and_all_clear(replay):
    replay.pick("10:05", deliver="09:10")
    replay.units(WRONG, "131000")
    assert "entry is recorded in game 102" in replay.check("09:15")[2][0]
    replay.units({**RIGHT, "id": 78}, "133000")
    assert "entry confirmed" in replay.check("09:30", unit=78)[2][0]
    assert replay.check("09:45") == ("already_confirmed", [], [], False)
    replay.pick("10:50", deliver="09:10")
    _, calls, dms, _ = replay.check("10:00", unit=78)
    assert "pending" in calls and dms == []


@pytest.mark.parametrize("manual_absence", [False, True])
def test_r10_fifo_final_pick(replay, manual_absence):
    replay.pick("10:05", deliver="09:40")
    replay.check("09:45", present=False)
    if manual_absence:
        replay.check("09:50", manual=True, present=False)
    pick = replay.picks_dir / f"{DAY}.json"
    pick.unlink()
    os.mkfifo(pick)
    status, dms = replay.morning()
    assert status == 1
    assert ("NOT ENTERED" if manual_absence else "ENTRY UNVERIFIABLE") in dms[0]


def test_later_manual_absence_wins_over_early_proof(replay):
    replay.pick("10:05", deliver="09:40")
    replay.units(RIGHT, "134000")
    replay.check("09:45")
    replay.check("09:50", present=False, manual=True)
    status, dms = replay.morning()
    assert status == 1 and "NOT ENTERED" in dms[0]


def test_r11_live_invalid_encoding_raises(replay, monkeypatch, tmp_path):
    replay.marker()
    replay.marker_path.write_bytes(b"\xff")
    with pytest.raises(UnicodeDecodeError):
        pick_entry.check(replay.picks_dir, date.fromisoformat(DAY))
    safe_run = health_runner._safe_run
    monkeypatch.setattr(health_runner, "_safe_run",
                        lambda name, fn: safe_run(name, fn) if name == "pick_entry" else [])
    monkeypatch.setattr(health_runner.realized_calibration, "_current_deploy_iso", lambda _: None)
    messages = []
    monkeypatch.setattr("bts.health.alert.send_dm", lambda _, message: messages.append(message))
    alert, = health_runner.run_all_checks(replay.picks_dir, tmp_path / "models",
                                          today=date.fromisoformat(DAY), dm_recipient="test")
    assert alert.level == "CRITICAL"
    assert len(messages) == 1 and "UnicodeDecodeError" in messages[0]

def fail_warning(*_):
    raise OSError("test DM failure")


def health_at(replay, day=DAY, at="23:59"):
    return pick_entry.check(replay.picks_dir, date.fromisoformat(day),
                            datetime.fromisoformat(f"{day}T{at}").replace(tzinfo=ET))


def test_r7_manual_crosswalk_gap_keeps_live_tier(replay, monkeypatch):
    replay.pick("10:05", deliver="09:40")
    assert replay.check("09:45", present=False)[2]
    replay.pick("10:06", deliver="09:40")
    with monkeypatch.context() as patch:
        patch.setattr("bts.cli._fetch_bts_to_mlb", lambda: {})
        assert replay.check("09:55", manual=True)[2] == []
    assert "1 min to submit" in replay.check("10:00", present=False)[2][0]


def test_r12_manual_recovery_survives_next_days_early_checks(replay, monkeypatch):
    prior, current = "2027-12-31", "2028-01-01"
    replay.pick("19:10", day=prior, deliver="17:00")
    with monkeypatch.context() as patch:
        patch.setattr("bts.dm.send_dm", fail_warning)
        replay.check("18:00", day=prior, present=False)
    replay.pick("06:10", day=current, deliver="04:55")
    replay.check("05:00", day=current)
    replay.check("18:15", day=prior, manual=True)
    restored = replay.marker_path.read_bytes()
    replay.check("05:15", day=current)
    assert replay.marker_path.read_bytes() == restored
    assert health_at(replay, prior) == []


def test_r13_manual_tiers_and_live_health_precedence(replay, monkeypatch):
    replay.pick("10:05", deliver="09:00")
    with monkeypatch.context() as patch:
        patch.setattr("bts.dm.send_dm", fail_warning)
        replay.check("09:15", present=False)
    assert "NOT entered" in replay.check("09:20", manual=True, present=False)[2][0]
    assert "STILL NOT entered" in replay.check("09:30", manual=True, present=False)[2][0]
    replay.check("09:40", manual=True)
    assert health_at(replay) == []


@pytest.mark.parametrize("manual_warning", [False, True])
def test_manual_fix_after_last_early_check(replay, manual_warning):
    replay.pick("10:05", deliver="09:39")
    replay.units(WRONG, "134000")
    assert len(replay.check("09:45")[2]) == 1
    if manual_warning:
        assert len(replay.check("09:46", manual=True)[2]) == 1
    replay.units({**RIGHT, "id": 78}, "134700")
    dms = replay.check("09:50", manual=True, unit=78)[2]
    assert sum("entry confirmed" in dm for dm in dms) == int(manual_warning)
    assert replay.check("10:00")[2] == []


@pytest.mark.parametrize("corrected_pitch", [False, True])
def test_late_early_confirmation(replay, corrected_pitch):
    replay.pick("10:05", deliver="09:39")
    replay.units(RIGHT, "134000")
    assert "pending" in replay.check("09:45", fetch_seconds=901 if not corrected_pitch else 310)[1]
    receipt, = entry_receipt.discover(replay.picks_dir, DAY)
    assert receipt.get("early_cron_run") is True
    if corrected_pitch:
        replay.pick("09:55", deliver="09:39")
    code, alerts = replay.morning()
    assert code == 1 and "ENTERED LATE" in alerts[0]


def test_early_failure_retained_across_dates_and_suppressed_by_live_recovery(replay, monkeypatch):
    tomorrow = "2027-03-25"
    replay.pick("06:10", deliver="04:55")
    with monkeypatch.context() as patch:
        patch.setattr("bts.dm.send_dm", fail_warning)
        replay.check("05:00", present=False)
    replay.pick("06:10", day=tomorrow, deliver="04:55")
    replay.check("05:00", day=tomorrow)
    assert [a.level for a in health_at(replay)] == ["CRITICAL"]
    replay.check("05:20", manual=True)
    replay.pick("10:50", day=tomorrow, deliver="09:50")
    replay.check("10:00", day=tomorrow)
    assert health_at(replay) == []
    yesterday = "2027-03-23"
    replay.pick("19:10", day=yesterday, deliver="17:50")
    replay.check("18:00", day=yesterday, manual=True)
    assert health_at(replay) == []


def test_unavailable_pick_keeps_recorded_early_send_failure(replay, monkeypatch):
    replay.pick("06:10", deliver="04:55")
    with monkeypatch.context() as patch:
        patch.setattr("bts.dm.send_dm", fail_warning)
        replay.check("05:00", present=False)
    pick = replay.picks_dir / f"{DAY}.json"
    pick.unlink()
    os.mkfifo(pick)
    alert, = health_at(replay)
    assert alert.level == "CRITICAL" and "unknown" in alert.message and "05:00" in alert.message
    assert replay.check("05:15") == ("no_pick_file", [], [], False)


def test_invalid_early_marker_destination(replay):
    replay.pick("06:10", deliver="04:55")
    replay.early_path.mkdir(parents=True)
    assert replay.check("05:00", present=False)[2]
    assert replay.result.exit_code == 1 and replay.result.exception.code == 1
    assert "marker" in replay.result.stderr.lower()
    receipt, = entry_receipt.discover(replay.picks_dir, DAY)
    assert "marker" in receipt["detail"].lower()
    assert replay.early_path.is_dir() and not replay.marker_path.exists()


def test_delayed_early_holder_can_block_only_original_slot(installed, box, tmp_path, replay):
    assert "09:45" in list(slots(installed))
    directory = tmp_path / "wrapper"
    directory.mkdir()
    with _runner(installed, box, directory) as (launch, _):
        release = directory / "release"
        holder, trace = launch("hold", str(release), duration="7")
        try:
            assert _await(lambda: dict(_records(trace)).get("ready"))
            ten, trace = launch("quick")
            assert ten.wait(timeout=15) == 1
            assert not dict(_records(trace)).get("ready")
        finally:
            release.touch()
            assert holder.wait(timeout=15) == 0
    replay.pick("10:06", deliver="09:59")
    assert replay.check("10:00", present=False)[2]
    assert replay.check("10:15") == ("outside_window", [], [], False)
