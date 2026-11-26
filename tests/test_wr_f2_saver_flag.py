"""Dated saver disagreements through the CLI, health check and EOD delivery."""

import datetime as dt
import hashlib
import json
import os
import signal
import shlex
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import cli, contest_state, web
from bts.health import alert, contest_state as health, runner
from bts.saver_state import load_saver_state
from tests.test_season_contract_cs4b1 import contract, put
from tests.test_season_contract_cs4b2 import automatic

ET = ZoneInfo("America/New_York")
DATETIME = dt.datetime
DAY = DATETIME(2027, 7, 15, 22, tzinfo=ET)
DECEMBER = DATETIME(2027, 12, 31, 23, 50, tzinfo=ET)
INCIDENT = "saver_state:dated_used_mismatch"
STATUS = "health_dm_delivery_status.json"
MISMATCH_STATUS = STATUS + ".dated_used_mismatch.json"
RECIPIENT = "offline.test"
OLD_B8A = (
    "If MLB shows it not used, the dated observation was the mistaken side and the "
    "flag is right: leave the flag as it is (a dated observation applies only on its own date)."
)
NEW_B8A = (
    "If MLB shows the saver has not been used, compare the current flag and any still-effective "
    "dated observation independently with MLB. Leave the flag as it is only if it agrees with MLB. "
    "If it differs, have Eric verify the current ET season and prior entries and reconcile the "
    "flag before the next automatic pick. For a wrong dated observation still in effect, use "
    "B8a's prior-entry and pick-timing checks before correcting it."
)

# SHA-256 of the ordered alerts, captured transport calls and status JSON
# from the unchanged base. Each pair is (live, historical).
BASE_SNAPSHOTS = {
    "expired": ("c3e171d330ee4df86fe1b9071b3482a47d07d75521eda465f811f05e4801f480",
                "f3f416e5f59823f0f026f71daa89a60d7c96d53a5939417542be950601534bbe"),
    "automatic": ("98d4a68c815ff9789f7d38197df9365c710c6f91d88dfd8da308d66d344af319",
                  "4fa190d6f721e743f0f4e84f07548267323f0edd217c86f3fdee869f856fe5a3"),
    "legacy": ("50cd0198e89937f329602b466941d0d5511293cc002bbaf8889eaa6fd446d63f",
               "d581d29bc886b6ef8a92e7cb9e7544821f92dfe686930d472d0314a545ef2345"),
    "missing": ("7f1fbf09393a44e74c85d6434cce34fc35c55ebd4b9ca667ead4d59490a1aad6",
                "45ef517804ab860656b096b87851f6b667e6854205910730970bb0f0aff542a1"),
    "future": ("bb824052c11fdbd17fd2c471b3a5d291bbada615cd79eaa3f08205b1a2909dbe",
               "c9808abe44b57bc89302632bf72cbcae90807f62ae07ce4f61414567dfef8120"),
    "coverage": ("985d554049fa6f3d14471e0c5f27b742e5bf61f761d351ae1580929bec5fb67f",
                 "897a551dae25d97e1b489f00ef9df4abc3ccfc61012ba9a59c41cf08a55094ba"),
    "invalid": ("a98622b096897beaaee56946a020ee285d86a2ba41a93fbc33b8bf4cc4bf5563",
                "bed6b786d4ccf320733f0ca54c5355c714e788790d0c12ea52e343122c1fdef3"),
}


@contextmanager
def clock(monkeypatch, now):
    class Clock(DATETIME):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    with monkeypatch.context() as patch:
        for module in (dt, cli, alert, health, runner):
            patch.setattr(module, "datetime", Clock)
        yield


def command(monkeypatch, picks, now, *args):
    with clock(monkeypatch, now):
        result = CliRunner().invoke(cli.cli, [*args, "--picks-dir", str(picks)])
    assert result.exit_code == 0, (result.output, result.exception)


def observation(monkeypatch, picks, now=DAY, state="used", streak=14):
    command(monkeypatch, picks, now, "set-contest-streak", "--date", str(now.date()),
            "--streak", str(streak), "--best-streak", str(max(streak, 15)),
            "--saver", state)


def flag(picks, state, year=DAY.year):
    return put(picks, "saver_state.json", {"season": year, "state": state})


def auto(picks, now, streak=14):
    put(picks, "season_contract.json", contract(season=now.year))
    return put(picks, "contest_streak.json", automatic(
        season=now.year, active_streak=streak, best_streak=15,
        source_date=str(now.date() - dt.timedelta(days=1)), recorded_at=now.isoformat(),
    ))


def eod(monkeypatch, picks, *, day=DAY, wall=None, transport=None, existing=(), overrides=None):
    safe_run = runner._safe_run

    def run_check(name, check):
        if name == "contest_state":
            return safe_run(name, check)
        return list(existing) if name == "calibration" else []

    transport = Mock(return_value="sent") if transport is None else transport
    with clock(monkeypatch, wall or day), monkeypatch.context() as patch:
        patch.setattr(alert, "send_dm", transport)
        patch.setattr(runner, "_safe_run", run_check)
        alerts = runner.run_all_checks(picks, picks.parent / "models", RECIPIENT,
                                       today=day.date(), contest_state_expected=True,
                                       thresholds_overrides=overrides)
    return alerts, [call.args for call in transport.call_args_list]


def status_path(picks, name=STATUS):
    return picks.parent / "health_state" / name


def mismatch(alerts):
    return [item for item in alerts if item.incident_key == INCIDENT]


def live_message(day, state):
    return (
        f"Streak Saver on {day.date()} ET. The dated observation says the saver was used. "
        f"The flag for season {day.year} read {state} at this check. "
        "A later automatic decision may count a saver as available. "
        "Verify the current season's saver and prior entries in MLB before any correction. "
        "Use the `bts season-contract status` command with the scheduler's picks directory "
        "as shown in checklist B8a. Follow B8a's saver reconciliation steps for any correction."
    )


@pytest.mark.parametrize(("state", "streak"), [("active", 14), ("not_earned", 14),
                                              ("not_earned", 9)])
def test_m1_m2_live_warning_and_delivery(tmp_path, monkeypatch, state, streak):
    picks = tmp_path / "oom" / "picks"
    observation(monkeypatch, picks, streak=streak)
    flag(picks, state)
    expected = alert.Alert("WARN", "saver_state", live_message(DAY, state), incident_key=INCIDENT)
    assert health.check(picks, now=DAY, expected=True) == [expected]
    alerts, sends = eod(monkeypatch, picks)
    assert alerts == [expected]
    assert len(sends) == 1
    recipient, body = sends[0]
    assert recipient == RECIPIENT
    assert body == f"BTS health WARN attention:\n- [saver_state] {expected.message} (1st observed day)"
    assert not status_path(picks).exists()
    receipt = json.loads(status_path(picks, MISMATCH_STATUS).read_text())
    assert receipt["sent_sources"] == [f"WARN_ATTENTION:{INCIDENT}"]
    assert receipt["sent_day"] == str(DAY.date())


@pytest.mark.parametrize("day", [DAY, DECEMBER])
def test_m3_after_midnight_uses_processed_season_and_current_files(tmp_path, monkeypatch, day):
    picks = tmp_path / "picks"
    tomorrow = day.replace(hour=0, minute=5) + dt.timedelta(days=1)
    observation(monkeypatch, picks, day)
    flag(picks, "active", day.year)
    alerts, sends = eod(monkeypatch, picks, day=day, wall=tomorrow)
    warning, = mismatch(alerts)
    reason = f"The dated observation recorded the saver as used. At this check the flag for season {day.year} read active."
    assert warning.message == f"Prior slate {day.date()} ET. {reason} {health.HISTORICAL_POINTER}"
    recipient, body = sends[0]
    assert recipient == RECIPIENT
    assert body.count(health.HISTORICAL_POINTER) == 1
    assert body.index(reason) < body.index(" (1st observed day)") < body.index(health.HISTORICAL_POINTER)
    assert "`" not in body and str(picks) not in body
    assert json.loads(status_path(picks, MISMATCH_STATUS).read_text())["sent_day"] == str(day.date())
    flag(picks, "active" if tomorrow.year != day.year else "used", tomorrow.year)
    assert not mismatch(eod(monkeypatch, picks, day=day, wall=tomorrow)[0])
    flag(picks, "active", day.year)
    observation(monkeypatch, picks, tomorrow, state="active")
    assert not mismatch(eod(monkeypatch, picks, day=day, wall=tomorrow)[0])


@pytest.mark.parametrize("state", ["active", "not_earned"])
def test_m4_m5_verified_repair_changes_next_automatic_read(tmp_path, monkeypatch, state):
    picks = tmp_path / "picks"
    observation(monkeypatch, picks)
    flag(picks, state)
    manual = picks / "account_state/contest_streak.manual.json"
    original = manual.read_bytes()
    tomorrow = DAY + dt.timedelta(days=1)
    auto(picks, tomorrow, streak=9 if state == "not_earned" else 14)

    def decision(now):
        return contest_state.load_decision_streak_state(picks, now=now, require_contest_state=True)

    assert not decision(DAY).saver_available
    assert decision(tomorrow).saver_available
    alerts, sends = eod(monkeypatch, picks)
    assert mismatch(alerts) and len(sends) == 1
    assert decision(tomorrow).saver_available
    command(monkeypatch, picks, DAY, "saver-state", "--use", "--season", str(DAY.year))
    assert load_saver_state(picks, season=DAY.year).state == "used"
    assert manual.read_bytes() == original
    assert not decision(tomorrow).saver_available
    assert eod(monkeypatch, picks) == ([], [])


@pytest.mark.parametrize("streak", range(10, 16))
def test_k1_existing_dated_warning(tmp_path, monkeypatch, streak):
    picks = tmp_path / "picks"
    observation(monkeypatch, picks, state="not_earned", streak=streak)
    live = health.check(picks, now=DAY, expected=True)
    (picks / "account_state/saver_state.json").write_bytes(decoder_limit())
    live_warning, = live
    arguments = f"bts set-contest-streak --date {DAY.date()} --streak {streak} --best-streak 15"
    directory = f"--picks-dir {shlex.quote(str(picks))}"
    manual = picks / "account_state/contest_streak.manual.json"
    assert live_warning == alert.Alert("WARN", "saver_state", (
        f"Streak Saver not_earned in dated observation {manual}, season {DAY.year}, "
        f"contradicts streak {streak}; counting the saver as spent. "
        "Check MLB and run ONE correction to the observation. Earned and unused: "
        f"`{arguments} --saver active {directory}`. Already used: "
        f"`{arguments} --saver used {directory}`. Not earned: "
        f"`bts set-contest-streak --date {DAY.date()} --streak N --best-streak B "
        f"--saver not_earned {directory}`, replacing N and B with the active and best streaks MLB "
        "shows. Changing the saver flag does not correct this observation."
    ))
    assert health.check(picks, now=DAY, expected=True) == live
    prior = health.check(picks, now=DAY, expected=True, historical_day=DAY.date())
    assert prior == [alert.Alert("WARN", "saver_state", health.historical_message(
        DAY.date(), "The saver state in the dated observation contradicts its streak. The saver counted as spent.",
    ))]


@pytest.mark.parametrize("state", ["used", "uninitialized", "absent", "encoding", "list",
                                  "other-season", "directory", "fifo", "invalid", "integer", "nested"])
def test_k2_k3_unusable_flag_preserves_existing_alerts(tmp_path, monkeypatch, state):
    picks = tmp_path / "picks"
    observation(monkeypatch, picks)
    # Give the contest check a coverage alert before it reaches the saver branch.
    resolved = [DAY.date() + dt.timedelta(days=1)]
    monkeypatch.setattr(health, "latest_resolved_pick_date", lambda _: resolved[0])
    monkeypatch.setattr(health, "_scan_resolved_pick_dates", lambda *a, **kw: resolved)
    monkeypatch.setattr(health, "resolved_pick_settlement_gap", lambda *a: 1)
    modes = (None, DAY.date())
    baseline = [health.check(picks, now=DAY, expected=True, historical_day=day) for day in modes]
    assert all(items and items[0].source == "contest_state" for items in baseline)
    path = picks / "account_state/saver_state.json"
    if state == "fifo":
        os.mkfifo(path)
    elif state == "directory":
        path.mkdir()
    elif state == "other-season":
        flag(picks, "active", DAY.year - 1)
    elif state in ("used", "uninitialized"):
        flag(picks, state)
    elif state != "absent":
        path.write_bytes({
            "encoding": b"\xff", "list": b"[]", "invalid": b"{",
            "integer": b'{"season":' + b"9" * 5000 + b"}",
            "nested": decoder_limit() if state == "nested" else b"",
        }[state])

    for day, expected in zip(modes, baseline):
        with flag_read_deadline():
            assert health.check(picks, now=DAY, expected=True, historical_day=day) == expected


def decoder_limit():
    payload = b"[" * 200_000 + b"]" * 200_000
    with pytest.raises(RecursionError):
        json.loads(payload)
    return payload


@contextmanager
def flag_read_deadline():
    class FlagReadTimeout(BaseException):
        pass

    def timed_out(*_):
        raise FlagReadTimeout("saver flag check blocked")

    previous_handler = signal.signal(signal.SIGALRM, timed_out)
    signal.alarm(15)
    try:
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, previous_handler)


@pytest.mark.parametrize(("state", "streak"), [("active", 14), ("not_earned", 9)])
@pytest.mark.parametrize("flag_state", ["active", "not_earned", "used", "uninitialized"])
def test_k4_no_other_dated_observation_warns(tmp_path, monkeypatch, state, streak, flag_state):
    picks = tmp_path / "picks"
    observation(monkeypatch, picks, state=state, streak=streak)
    flag(picks, flag_state)
    assert health.check(picks, now=DAY, expected=True) == []
    assert health.check(picks, now=DAY, expected=True, historical_day=DAY.date()) == []


@pytest.mark.parametrize("first", ["used", "not_earned"])
def test_same_day_reruns_preserve_both_incidents(tmp_path, monkeypatch, first):
    # The existing message includes the picks path several times. Keep it short
    # enough for the unchanged digest to fit the transport limit.
    monkeypatch.chdir(tmp_path)
    picks = Path("p")
    flag(picks, "active")
    second = "not_earned" if first == "used" else "used"
    for state in (first, second):
        observation(monkeypatch, picks, state=state)
        alerts, sends = eod(monkeypatch, picks)
        warning, = alerts
        assert bool(mismatch(alerts)) == (state == "used")
        assert len(sends) == 1 and warning.message in sends[0][1]
    assert eod(monkeypatch, picks)[1] == []
    assert json.loads(status_path(picks).read_text())["sent_sources"] == ["WARN_ATTENTION:saver_state"]
    assert json.loads(status_path(picks, MISMATCH_STATUS).read_text())["sent_sources"] == [
        f"WARN_ATTENTION:{INCIDENT}",
    ]


def test_existing_digest_sends_first_and_stays_below_chat_limit(tmp_path, monkeypatch):
    picks = tmp_path / "picks"
    observation(monkeypatch, picks)
    flag(picks, "used")
    existing = [alert.Alert("WARN", "pick_entry", "Existing operator alert. " * 36)]

    def capped_transport(_, body):
        # The bodies use one ASCII code point per grapheme except for bullets
        # and other single-code-point punctuation in the formatter.
        if len(body) > 1000:
            raise ValueError("chat length limit")
        return "sent"

    base_alerts, base_sends = eod(monkeypatch, picks, existing=existing, transport=Mock(side_effect=capped_transport))
    base_body = base_sends[0][1]
    base_status = status_path(picks).read_bytes()
    assert 900 < len(base_body) <= 1000
    assert json.loads(base_status)["status"] == "sent"
    status_path(picks).unlink()
    flag(picks, "active")
    alerts, sends = eod(monkeypatch, picks, existing=existing, transport=Mock(side_effect=capped_transport))
    warning, = mismatch(alerts)
    assert [item for item in alerts if item != warning] == base_alerts
    assert sends[0] == base_sends[0] and warning.message in sends[1][1]
    assert len(base_body) + len(sends[1][1]) > 1000
    assert status_path(picks).read_bytes() == base_status
    assert json.loads(status_path(picks, MISMATCH_STATUS).read_text())["status"] == "sent"


def test_failed_send_and_interrupt_before_status_can_retry(tmp_path, monkeypatch):
    picks = tmp_path / "picks"
    observation(monkeypatch, picks)
    path = flag(picks, "active")
    before = path.read_bytes()
    status = status_path(picks, MISMATCH_STATUS)
    transport = Mock(side_effect=RuntimeError("offline"))
    _, failed = eod(monkeypatch, picks, transport=transport)
    assert failed and json.loads(status.read_text())["status"] == "failed"
    transport.reset_mock(side_effect=True)
    transport.side_effect = KeyboardInterrupt()
    with pytest.raises(KeyboardInterrupt):
        eod(monkeypatch, picks, transport=transport)
    assert transport.call_count == 1
    assert json.loads(status.read_text())["status"] == "failed"
    assert path.read_bytes() == before
    _, retried = eod(monkeypatch, picks, transport=Mock(return_value=""))
    assert retried and json.loads(status.read_text())["status"] == "failed"
    assert eod(monkeypatch, picks)[1] == retried
    assert eod(monkeypatch, picks)[1] == []


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_receipt_alias_preserves_saver_flag(tmp_path, monkeypatch, kind):
    picks = tmp_path / "picks"
    observation(monkeypatch, picks)
    path = flag(picks, "active")
    before = path.read_bytes()
    receipt = status_path(picks, MISMATCH_STATUS)
    receipt.parent.mkdir()
    if kind == "symlink":
        receipt.symlink_to(path)
    else:
        os.link(path, receipt)
    alerts, sends = eod(monkeypatch, picks)
    assert mismatch(alerts) and len(sends) == 1
    assert path.read_bytes() == before
    assert receipt.is_symlink() == (kind == "symlink")
    assert receipt.samefile(path)
    assert eod(monkeypatch, picks)[1] == sends


def test_root_receipt_override_preserves_digest_and_sends_mismatch(tmp_path, monkeypatch):
    picks = tmp_path / "picks"
    observation(monkeypatch, picks)
    flag(picks, "used")
    existing = [alert.Alert("CRITICAL", "offline-probe", "Pre-existing critical alert")]
    overrides = {"health_dm_delivery_status": "/"}
    _, baseline = eod(monkeypatch, picks, existing=existing, overrides=overrides)
    assert len(baseline) == 1
    flag(picks, "active")
    alerts, sends = eod(monkeypatch, picks, existing=existing, overrides=overrides)
    warning, = mismatch(alerts)
    assert sends[:1] == baseline
    assert len(sends) == 2 and warning.message in sends[1][1]
    assert json.loads(status_path(picks, MISMATCH_STATUS).read_text())["message_id"] == "sent"


def test_consecutive_days_and_changed_body_are_eligible(tmp_path, monkeypatch):
    picks = tmp_path / "picks"
    for offset in range(3):
        day = DAY + dt.timedelta(days=offset)
        observation(monkeypatch, picks, day)
        flag(picks, "active")
        _, sends = eod(monkeypatch, picks, day=day)
        label = ("1st observed day", "2nd consecutive day", "3rd consecutive day")[offset]
        assert len(sends) == 1 and f"({label})" in sends[0][1]
        assert eod(monkeypatch, picks, day=day)[1] == []
    flag(picks, "not_earned")
    _, sends = eod(monkeypatch, picks, day=day)
    assert len(sends) == 1 and "read not_earned" in sends[0][1]


def test_b8a_saver_reconciliation_text():
    checklist = Path(__file__).resolve().parents[1] / "docs/ops/2027-season-start.md"
    row, = (line for line in checklist.read_text().splitlines() if line.startswith("| B8a |"))
    stale, warning = row.split("if initialization races, reread the flag. ", 1)[1].split("</li>", 1)
    addition, _, _ = stale.partition(" Read the latest fetch status")
    assert addition == (
        "Coordinate saver corrections with anyone using the dashboard. "
        "Immediately before writing, refresh MLB and the scheduler's flag. "
        "If either changed since your verification, stop and reconcile the current state before retrying."
    )
    assert warning.split("</li>", 1)[0] + "</li>" == (
        "<li>**On the end-of-day warning that begins `Streak Saver on D ET. The dated observation "
        "says the saver was used.`** (change set F2): D is the ET date the warning prints. Run "
        "`bts season-contract status --picks-dir DIR` (DIR as above) and compare the dated row's saver "
        "and the saver flag with MLB. If MLB shows this season's saver used, follow **The saver** "
        f"in the bullet above. {NEW_B8A}</li>"
    )


@pytest.mark.parametrize("spelling", ["identical", "dot", "relative", "symlink", "hardlink"])
def test_e1_e2_mismatch_preserves_attention(tmp_path, monkeypatch, spelling):
    monkeypatch.chdir(tmp_path)
    picks = tmp_path / "picks"
    observation(monkeypatch, picks)
    flag(picks, "active")
    digest = status_path(picks, "digest.json")
    receipt = digest.with_name(digest.name + ".dated_used_mismatch.json")
    attention = receipt
    configured = str(attention)
    if spelling == "dot":
        configured = f"{attention.parent}/./{attention.name}"
    elif spelling == "relative":
        configured = str(attention.relative_to(tmp_path))
    elif spelling in {"symlink", "hardlink"}:
        attention = status_path(picks, "attention.json")
        attention.parent.mkdir()
        if spelling == "symlink":
            receipt.symlink_to(attention)
        else:
            # Seed the state for the day before the replay so both links exist.
            attention.write_text("{}")
            os.link(attention, receipt)
        configured = str(attention)
    overrides = {"health_dm_delivery_status": str(digest), "warn_attention_state": configured}
    repeated = alert.Alert("WARN", "calibration_drift", "Calibration drift in both slates")
    dispatch = Mock(wraps=runner.dispatch_dm_for_health_alerts)
    monkeypatch.setattr(runner, "dispatch_dm_for_health_alerts", dispatch)
    alerts, sends = eod(monkeypatch, picks, existing=[repeated], overrides=overrides)
    warning, = mismatch(alerts)
    assert len(sends) == 1 and warning.message in sends[0][1]
    day_one = json.loads(attention.read_text())
    assert day_one["sources"]["calibration_drift"]["streak"] == 1
    assert dispatch.call_args.kwargs["status_path"] is None
    assert eod(monkeypatch, picks, existing=[repeated], overrides=overrides)[1] == sends

    tomorrow = DAY + dt.timedelta(days=1)
    observation(monkeypatch, picks, tomorrow, state="active")
    alerts, sends = eod(monkeypatch, picks, day=tomorrow, existing=[repeated], overrides=overrides)
    assert alerts == [repeated]
    assert json.loads(attention.read_text())["sources"]["calibration_drift"]["streak"] == 2
    assert len(sends) == 1 and f"{repeated.message} (2nd consecutive day)" in sends[0][1]


@pytest.mark.parametrize("custom_attention", [False, True])
def test_e3_e4_separate_attention_preserves_receipt(tmp_path, monkeypatch, custom_attention):
    picks = tmp_path / "picks"
    observation(monkeypatch, picks)
    flag(picks, "active")
    attention = status_path(picks, "elsewhere.json" if custom_attention else "warn_attention_state.json")
    overrides = {"warn_attention_state": str(attention)} if custom_attention else None
    _, sends = eod(monkeypatch, picks, overrides=overrides)
    assert len(sends) == 1
    receipt = status_path(picks, MISMATCH_STATUS)
    before = receipt.read_bytes()
    assert json.loads(before)["message_id"] == "sent"
    assert json.loads(attention.read_text())["sources"][INCIDENT]["streak"] == 1
    assert eod(monkeypatch, picks, overrides=overrides)[1] == []
    assert receipt.read_bytes() == before


def test_e6_b8a_rechecks_flag_after_dashboard_change(tmp_path, monkeypatch):
    picks = tmp_path / "picks"
    day = DAY.replace(month=8, day=19)
    observation(monkeypatch, picks, day, streak=9)
    flag(picks, "active")
    assert "read active" in eod(monkeypatch, picks, day=day)[1][0][1]
    code, details = web.saver_transition_response(
        picks, expected_prior="active", new_state="used", same_origin=True, now=day,
    )
    assert code == 200, details
    with clock(monkeypatch, day):
        status = CliRunner().invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks)])
    assert status.exit_code == 0, status.output
    assert "saver=used" in status.output and "saver: used" in status.output

    # MLB shows the saver is available while both local records say used.
    checklist = Path(__file__).resolve().parents[1] / "docs/ops/2027-season-start.md"
    row, = (line for line in checklist.read_text().splitlines() if line.startswith("| B8a |"))
    assert NEW_B8A in row and OLD_B8A not in row
    code, details = web.saver_transition_response(
        picks, expected_prior="used", new_state="active", same_origin=True, now=day,
    )
    assert code == 200, details
    tomorrow = day + dt.timedelta(days=1)
    auto(picks, tomorrow, streak=9)
    assert contest_state.load_decision_streak_state(
        picks, now=tomorrow, require_contest_state=True,
    ).saver_available


def test_e7_other_b8a_text_matches_round_two():
    checklist = Path(__file__).resolve().parents[1] / "docs/ops/2027-season-start.md"
    row, = (line for line in checklist.read_bytes().splitlines() if line.startswith(b"| B8a |"))
    original = row.replace(NEW_B8A.encode(), OLD_B8A.encode())
    assert hashlib.sha256(original).hexdigest() == (
        "21b6dd612d9019555a4436b0b3f68b5493d5751d466bda8a32dfabf5a156739d"
    )


@pytest.mark.parametrize("historical", [False, True])
def test_e7_rendered_mismatch_matches_round_two(tmp_path, monkeypatch, historical):
    picks = tmp_path / "picks"
    observation(monkeypatch, picks)
    flag(picks, "active")
    wall = DAY + dt.timedelta(hours=3) if historical else None
    _, sends = eod(monkeypatch, picks, wall=wall)
    text = live_message(DAY, "active") + " (1st observed day)"
    if historical:
        text = (
            "Prior slate 2027-07-15 ET. The dated observation recorded the saver as used. "
            "At this check the flag for season 2027 read active. (1st observed day) "
            + health.HISTORICAL_POINTER
        )
    assert sends == [(RECIPIENT, f"BTS health WARN attention:\n- [saver_state] {text}")]


@pytest.mark.parametrize("scenario", ["expired", "automatic", "legacy", "missing",
                                      "future", "coverage", "invalid"])
@pytest.mark.parametrize("historical", [False, True])
def test_k5_k6_k7_alerts_digest_and_status_match_base(tmp_path, monkeypatch, scenario, historical):
    monkeypatch.chdir(tmp_path)
    picks = Path("picks")
    now = DAY
    if scenario == "expired":
        observation(monkeypatch, picks)
        now += dt.timedelta(days=1)
        auto(picks, now)
    elif scenario in ("automatic", "coverage"):
        auto(picks, now)
    elif scenario == "legacy":
        put(picks, "contest_streak.manual.json", {
            "streak": 14, "source_date": str(now.date()),
            "override_expires_at": (now + dt.timedelta(days=1)).isoformat(),
        })
    elif scenario == "future":
        auto(picks, now + dt.timedelta(days=10)).with_name("season_contract.json").unlink()
    elif scenario == "invalid":
        observation(monkeypatch, picks)
        manual = picks / "account_state/contest_streak.manual.json"
        payload = json.loads(manual.read_text())
        payload["saver_state"] = "bad"
        manual.write_text(json.dumps(payload))
    if scenario in ("expired", "automatic", "legacy", "coverage"):
        flag(picks, "not_earned")
    if scenario == "coverage":
        resolved = [now.date(), (now + dt.timedelta(days=1)).date()]
        monkeypatch.setattr(health, "_scan_resolved_pick_dates", lambda *a, **kw: resolved)
        monkeypatch.setattr(health, "latest_resolved_pick_date", lambda _: max(resolved))
        monkeypatch.setattr(health, "resolved_pick_settlement_gap", lambda *a: 2)
        put(picks, "contest_streak.manual.json", {"override_expires_at": None})
    wall = now + dt.timedelta(days=1) if historical else now
    alerts, sends = eod(monkeypatch, picks, day=now, wall=wall)
    assert not status_path(picks, MISMATCH_STATUS).exists()
    recorded = status_path(picks).read_text()
    snapshot = json.dumps([(a.level, a.source, a.message, a.incident_key) for a in alerts])
    payload = snapshot + json.dumps(sends) + recorded
    digest = hashlib.sha256(payload.encode()).hexdigest()
    assert digest == BASE_SNAPSHOTS[scenario][historical], payload
