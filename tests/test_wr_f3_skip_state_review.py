import datetime as dt
import json
from pathlib import Path
import importlib.util
import os
import stat
from types import SimpleNamespace

import pytest
from unittest.mock import patch

from click.testing import CliRunner
from pytest import MonkeyPatch
from bts import cli, contest_state, daily_decision, day_outcome, dm, entry_receipt, orchestrator, scheduler, util
from tests.test_contest_fetch import _mock_fetch
from tests.test_day_outcome import DATE, SHA, _decision, _receipt

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "docs/reviews/2026-11-18-whole-range/closing-test/carbon-slice3/test_slice3_interactions.py"
spec = importlib.util.spec_from_file_location("f3_scheduler_harness", HARNESS)
carbon = importlib.util.module_from_spec(spec)
spec.loader.exec_module(carbon)
carbon.REPO_ROOT = ROOT
ET = carbon.ET
START = dt.datetime(2027, 3, 24, 10, tzinfo=ET)
DAY = START.date().isoformat()
RECIPIENT = "operator.bsky.social"
SKIP = _decision("skip", scoreable=False) | {"streak": 30, "saver_available": False}
CONFIG = {"scheduler": {"pick_delivery": "dm", "entry_intent": "enter"}}
EVIDENCE = {
    "current": "contest_streak.manual.json",
    "archive": "contest_streak.manual.revoked-test.json",
    "switch": "season_contract.json",
}
EXPECTED = {
    "M1": ("dated",), "M2": ("revoke",), "M3": ("dated",), "M4": ("dated", "revoke"),
    "K1": (), "K2a": (), "K2b": (), "K2c": (), "K2d": (),
    "K2c_after_same_day_revoke": (), "K2a_current_outranked_by_archive": (),
    "K2e": ("dated",),
}


def snapshot(root):
    return {path.relative_to(root): path.read_bytes() for path in root.rglob("*")
            if stat.S_ISREG(path.stat().st_mode)}


def review_detail(day, reasons):
    return " ".join([f"The policy recorded a skip for {day} using streak 30 and policy saver availability unavailable.",
                     *reasons, "Check MLB for unsettled entries and verify the current streak, season best and saver."])


def assert_report(outcome, result, sends, day, state, detail, code, pick=None, send_fails=False):
    expected = {"date": day, "state": state, "ok": code == 0, "detail": detail, "pick": pick}
    message = f"BTS day check {day}: {state.replace('_', ' ').upper()}. {detail}"
    if pick:
        message += f" Pick: {pick}."
    print(json.dumps({"state": state, "actual": outcome.as_dict(), "exit": result.exit_code, "DM": sends}))
    assert outcome.as_dict() == expected
    assert outcome.message() == message
    assert result.exit_code == code, (result.output, result.exception)
    assert result.stdout == json.dumps(expected) + "\n"
    assert result.stderr == ("day-outcome: DM failed: OSError\n" if send_fails else "")
    assert sends == ([(RECIPIENT, message)] if code else [])


def evidence(kind, *, day=DATE, **changes):
    if kind == "switch":
        data = {"schema_version": "bts_season_contract_v1", "season": int(day[:4]), "status": "revoked",
                "activation_id": "a" * 32, "switched_at": None, "revoked_at": f"{day}T20:00:00Z"}
    else:
        end = dt.datetime.combine(dt.date.fromisoformat(day), dt.time.max, ET)
        data = {"schema_version": contest_state.DATED_MANUAL_SCHEMA, "season": int(day[:4]),
                "effective_date": day, "expires_at": end.isoformat(), "active_streak": 1,
                "best_streak": 30, "saver_state": "used", "recorded_at": f"{day}T14:00:00Z",
                "source": "manual_cli"}
    return data | changes


@pytest.fixture
def report(tmp_path, monkeypatch):
    picks = tmp_path / "picks"
    account = picks / "account_state"
    account.mkdir(parents=True)
    cfg = tmp_path / "config.toml"
    runner = CliRunner()
    clock = [dt.datetime(2027, 4, 3, 8, 30, tzinfo=ET)]
    sends = []

    def setup(decision=SKIP, *, schedule=({},), intent="enter", day=DATE, send_fails=False):
        clock[0] = dt.datetime.combine(dt.date.fromisoformat(day) + dt.timedelta(days=1), dt.time(8, 30), ET)
        decision_path = daily_decision.decision_path(day, picks)
        decision_path.parent.mkdir(exist_ok=True)
        if decision is not None:
            decision_path.write_text(json.dumps(decision))
        delivery = "private" if intent == "research" else "dm"
        config = {"scheduler": {"entry_intent": intent, "pick_delivery": delivery}}
        cfg.write_text(f'[scheduler]\nentry_intent = "{intent}"\npick_delivery = "{delivery}"\n')

        def fetch(_):
            if isinstance(schedule, Exception):
                raise schedule
            return list(schedule)

        def send(recipient, message):
            sends.append((recipient, message))
            if send_fails:
                raise OSError("offline")
            return "offline"

        monkeypatch.setattr(scheduler, "fetch_schedule", fetch)
        monkeypatch.setattr(dm, "send_dm", send)
        return config

    def command(*args):
        with patch("datetime.datetime", carbon._make_Clock(clock)):
            return runner.invoke(cli.cli, [*args, "--picks-dir", str(picks)])

    def cli_report(day):
        sends.clear()
        return command("day-outcome", "--date", day, "--config", str(cfg),
                       "--expected-username", "stonehengee", "--dm-recipient", RECIPIENT)

    def check(state="deliberate_skip", detail="The policy skipped the day.", *, code=0, day=DATE,
              config=CONFIG, send_fails=False):
        before = snapshot(picks)
        outcome = day_outcome.gather(day, picks, config, expected_username="stonehengee")
        result = cli_report(day)
        assert_report(outcome, result, sends, day, state, detail, code, send_fails=send_fails)
        assert snapshot(picks) == before

    return SimpleNamespace(account=account, picks=picks, cfg=cfg, clock=clock, sends=sends, setup=setup,
                           command=command, cli_report=cli_report, check=check)


@pytest.mark.parametrize("kind", EVIDENCE)
@pytest.mark.parametrize("fault", [
    "absent", "directory", "fifo", "unreadable", "utf8", "malformed", "deep", "long_number", "list",
    "wrong_date", "wrong_season", "wrong_schema", "oversize",
])
def test_K5_bad_optional_evidence(report, monkeypatch, kind, fault):
    report.setup()
    path = report.account / EVIDENCE[kind]
    if fault == "directory":
        path.mkdir()
    elif fault == "fifo":
        os.mkfifo(path)
    elif fault != "absent":
        raw = {
            "utf8": b"\xff",
            "malformed": b'{"x":',
            "deep": b"[" * 2000 + b"0" + b"]" * 2000,
            "long_number": b'{"x":' + b"1" * 5000 + b"}",
            "list": b"[]",
        }.get(fault)
        if raw is None:
            changes = {"wrong_season": {"season": 2028}, "wrong_schema": {"schema_version": "legacy"},
                       "oversize": {"padding": "x" * 65536}}
            raw = json.dumps(evidence(kind, day="2027-04-01" if fault == "wrong_date" else DATE,
                                      **changes.get(fault, {}))).encode()
        path.write_bytes(raw)
    real_open = os.open

    def open_evidence(target, flags, *args, **kwargs):
        if os.fspath(target) == str(path):
            if fault == "unreadable":
                raise PermissionError("evidence inaccessible")
            if fault == "fifo":
                assert flags & os.O_NONBLOCK
        return real_open(target, flags, *args, **kwargs)

    if fault in {"fifo", "unreadable"}:
        monkeypatch.setattr(os, "open", open_evidence)
    report.check()
    other = "archive" if kind == "switch" else "switch"
    (report.account / EVIDENCE[other]).write_text(json.dumps(evidence(other)))
    reason = (f"An archived dated observation for {DATE} records streak 1 and policy saver availability unavailable."
              if other == "archive" else f"The season switch records a revocation on {DATE} at 16:00 ET.")
    report.check("skip_state_review", review_detail(DATE, [reason]), code=1)


@pytest.mark.parametrize("changes", [
    {"date": "2027-04-01"}, {"date": None}, {"scoreable": True},
    {"streak": True}, {"streak": -1}, {"streak": None}, {"saver_available": 0}, {"saver_available": None},
])
def test_K5_base_readable_skip(report, changes):
    report.setup(SKIP | changes)
    for kind in EVIDENCE:
        (report.account / EVIDENCE[kind]).write_text(json.dumps(evidence(kind)))
    report.check()


def test_overflowing_revocation(report):
    report.setup()
    (report.account / EVIDENCE["switch"]).write_text(json.dumps(evidence("switch", revoked_at="0001-01-01T00:00:00Z")))
    report.check()
    (report.account / EVIDENCE["current"]).write_text(json.dumps(evidence("current")))
    reason = f"The installed dated observation for {DATE} records streak 1 and policy saver availability unavailable."
    report.check("skip_state_review", review_detail(DATE, [reason]), code=1)


@pytest.mark.parametrize("kind", ["current", "archive"])
@pytest.mark.parametrize("recorded_at", ["0001-01-01T00:00:00Z", "9999-12-31T23:59:59-12:00"])
def test_overflowing_dated_observation(report, kind, recorded_at):
    report.setup()
    (report.account / EVIDENCE[kind]).write_text(json.dumps(evidence(kind, recorded_at=recorded_at)))
    report.check()


def test_overflowing_matching_archive_preserves_review(report):
    report.setup()
    (report.account / EVIDENCE["switch"]).write_text(json.dumps(evidence("switch")))
    (report.account / EVIDENCE["archive"]).write_text(json.dumps(evidence("archive")))
    matching = evidence("archive", active_streak=30, recorded_at="9999-12-31T23:59:59-12:00")
    (report.account / "contest_streak.manual.revoked-overflow.json").write_text(json.dumps(matching))
    report.check("skip_state_review", review_detail(DATE, [
        f"An archived dated observation for {DATE} records streak 1 and policy saver availability unavailable.",
        f"The season switch records a revocation on {DATE} at 16:00 ET.",
    ]), code=1)


def test_evidence_read_byte_limit(report, monkeypatch):
    report.setup()
    path = report.account / EVIDENCE["current"]
    path.write_text(json.dumps(evidence("current", padding="x" * (256 * 1024))))
    target = path.stat()
    real_read = os.read
    bytes_read = []

    def count_read(fd, size):
        chunk = real_read(fd, size)
        if os.path.samestat(os.fstat(fd), target):
            bytes_read.append(len(chunk))
        return chunk

    with monkeypatch.context() as patches:
        patches.setattr(os, "read", count_read)
        assert day_outcome._read_evidence(path) is None
    assert 0 < sum(bytes_read) <= 64 * 1024 + 1
    report.check()


@pytest.mark.parametrize("source", ["module docstring", "CLAUDE.md"])
def test_review_evidence_documentation(source):
    text = day_outcome.__doc__ if source == "module docstring" else (ROOT / source).read_text()
    text = " ".join(text.split())
    assert ("A deliberate MDP skip can prompt `skip_state_review` if the selected dated observation for that day "
            "records a different streak or policy saver availability. The report uses a valid installed observation "
            "when present and otherwise checks retained archives. In enter mode it also prompts review if the switch "
            "remains revoked for that season, the revocation was on the skip date in ET, and no valid retained dated "
            "observation matches both inputs.") in text
    assert ("Missing or unusable optional contest evidence can leave a deliberate skip clean. The base outcome checks "
            "report missing schedule, decision or entry evidence where required.") in text


def test_K7_year_boundary(report):
    day = "2027-12-31"
    report.setup(SKIP | {"date": day}, day=day)
    report.clock[0] = report.clock[0].replace(hour=0, minute=30)
    result = report.command("season-contract", "revoke", "--season", "2028", "--reason", "restore")
    assert result.exit_code == 0, (result.output, result.exception)
    report.clock[0] = report.clock[0].replace(hour=8)
    report.check(day=day)


@pytest.mark.parametrize("send_fails", [False, True])
def test_K6_problem_exit_and_send(report, send_fails):
    report.setup(None, send_fails=send_fails)
    report.check("no_decision", "1 games were scheduled and the day has no decision record: no pick was "
                 "committed and no skip was decided.", code=1, send_fails=send_fails)


def test_K6_could_not_run(report):
    report.setup()
    report.check()
    report.cfg.write_text("[invalid")
    result = report.cli_report(DATE)
    assert result.exit_code == 2
    assert result.stdout == ""
    assert result.stderr.startswith(f"day-outcome: could not run for {DATE}: TOMLDecodeError:")
    assert report.sends == [(RECIPIENT, f"BTS day check {DATE}: THE CHECK COULD NOT RUN (TOMLDecodeError). "
                                       "The day's end state is unknown.")]


BATTER_ONLY = _receipt(confirmed=False)
BATTER_ONLY["qualification"]["slots"] = [{"state": "unit_unverified"}]


@pytest.mark.parametrize("state,decision,schedule,intent,receipts", [
    ("entered", _decision(), ({},), "enter", [_receipt()]),
    ("entered_game_unverified", _decision(), ({},), "enter", [BATTER_ONLY]),
    ("entered_late", _decision(), ({},), "enter", [_receipt(before_cutoff=False)]),
    ("not_entered", _decision(), ({},), "enter", [_receipt(confirmed=False, reason="no_pick")]),
    ("entry_unverifiable", _decision(), ({},), "enter", []),
    ("no_decision", None, ({},), "enter", []),
    ("no_games", SKIP, (), "enter", []),
    ("no_games", SKIP, ({"status": {"detailedState": "Postponed"}},), "enter", []),
    ("commit_not_scoreable", _decision(scoreable=False), ({},), "enter", []),
    ("research_commit", _decision(delivery_status="private_locked"), ({},), "research", []),
    ("delivered_under_research", _decision(), ({},), "research", []),
    ("contradiction", _decision(), (), "enter", []),
    ("intent_unknown", SKIP, ({},), "broken", []),
    ("schedule_unverifiable", SKIP, OSError("offline"), "enter", []),
    ("skip_not_deliberate", SKIP | {"source": "other"}, ({},), "enter", []),
])
def test_K3_K4_base_outcome(report, monkeypatch, state, decision, schedule, intent, receipts):
    config = report.setup(decision, schedule=schedule, intent=intent)
    monkeypatch.setattr(entry_receipt, "_sha", lambda _: SHA)
    monkeypatch.setattr(entry_receipt, "discover", lambda *_: receipts)
    base = day_outcome.classify_day(
        date=DATE, games=None if isinstance(schedule, Exception) else len(schedule),
        decision=decision, decision_sha256=SHA, receipts=receipts, intent=intent,
        void_games=int(bool(schedule) and isinstance(schedule, tuple) and "status" in schedule[0]),
        expected_username="stonehengee",
    )
    assert base.state == state
    for kind in EVIDENCE:
        (report.account / EVIDENCE[kind]).write_text(json.dumps(evidence(kind)))
    real_open = os.open

    def open_file(path, *args, **kwargs):
        assert os.fspath(path) not in {str(report.account / name) for name in EVIDENCE.values()}
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(os, "open", open_file)
    outcome = day_outcome.gather(DATE, report.picks, config, expected_username="stonehengee")
    result = report.cli_report(DATE)
    assert_report(outcome, result, report.sends, DATE, state, base.detail, int(not base.ok), base.pick)


def test_K5_rejected_base_record(report):
    report.setup(SKIP | {"scoreable": 0})
    (report.account / EVIDENCE["switch"]).write_text(json.dumps(evidence("switch")))
    report.check("no_decision", "1 games were scheduled and the day has no decision record: no pick was "
                 "committed and no skip was decided.", code=1)


@pytest.mark.parametrize("tie,matching_archive", [(True, True), (True, False), (False, True)])
def test_archive_selection(report, tie, matching_archive):
    report.setup()
    for name, data in {
        "contest_streak.manual.revoked-a.json": evidence("archive", active_streak=30 if matching_archive else 2),
        "contest_streak.manual.revoked-b.json": evidence(
            "archive", recorded_at=f"{DATE}T10:{'00' if tie else '01'}:00-04:00"),
        EVIDENCE["switch"]: evidence("switch"),
    }.items():
        (report.account / name).write_text(json.dumps(data))
    if tie and matching_archive:
        report.check()
    else:
        reason = (f"The season switch records a revocation on {DATE} at 16:00 ET." if tie else
                  f"An archived dated observation for {DATE} records streak 1 and policy saver availability unavailable.")
        report.check("skip_state_review", review_detail(DATE, [reason]), code=1)


def test_research_dated_reason_only(report):
    config = report.setup(intent="research")
    for kind in ("current", "switch"):
        (report.account / EVIDENCE[kind]).write_text(json.dumps(evidence(kind)))
    reason = f"The installed dated observation for {DATE} records streak 1 and policy saver availability unavailable."
    report.check("skip_state_review", review_detail(DATE, [reason]), config=config, code=1)


@pytest.mark.parametrize("name", EXPECTED)
def test_scheduler_skip(tmp_path, name):
    root = tmp_path
    with carbon._harness(root, START) as env:
        clock, picks_dir = env["clock"], env["picks_dir"]
        account = picks_dir / "account_state"
        manual = account / "contest_streak.manual.json"
        runner = CliRunner()
        check_times = []
        cfg = root / "config.toml"
        env["confirmed"][101] = {"home", "away"}

        def command(*args, exit_code=0):
            with patch("datetime.datetime", carbon._make_Clock(clock)):
                result = runner.invoke(cli.cli, [*args, "--picks-dir", str(picks_dir)])
            assert result.exit_code == exit_code, (args, result.output, result.exception)
            return result.output

        def dated(streak, saver="used", *, best=30, day=DAY, exit_code=0):
            return command("set-contest-streak", "--date", day, "--streak", str(streak),
                           "--best-streak", str(best), "--saver", saver, exit_code=exit_code)

        def revoke():
            command("season-contract", "revoke", "--season", "2027", "--reason", "restored from snapshot")

        def enable():
            command("season-contract", "enable", "--season", "2027")

        def fetch(streak):
            with MonkeyPatch.context() as mp:
                _mock_fetch(mp, {"activeStreak": streak, "seasonBestStreak": 30,
                                 "predictions": [{"roundId": 1, "result": "hit"}]},
                            {1: START.date()})
                command("fetch-contest-streak", "--expected-username", "stonehengee")

        def interleaved_dated_commands():
            write = util.atomic_write_text
            nested = False

            def delay_before_install(path, text):
                nonlocal nested
                if path == manual and not nested:
                    nested = True
                    clock[0] = START.replace(hour=9, minute=1)
                    dated(1, "not_earned")
                    clock[0] = START.replace(hour=9, minute=2)
                    revoke()
                write(path, text)

            with patch.object(util, "atomic_write_text", side_effect=delay_before_install):
                dated(30)
            live = json.loads(manual.read_text())
            archived, = account.glob("contest_streak.manual.revoked-*.json")
            retired = json.loads(archived.read_text())
            assert live["active_streak"] == 30 and live["recorded_at"] < retired["recorded_at"]
            assert retired["active_streak"] == 1

        carbon._write_auto_obs(picks_dir, START - dt.timedelta(days=1), 30,
                               source_date=DAY, saver_flag="used", recorded_at=START.isoformat())
        clock[0] = START.replace(hour=9)
        if name in {"M3", "K2a", "K2b"}:
            dated(30)
        if name == "K2c":
            (account / "season_contract.json").unlink()
            enable()
            fetch(30)
        if name == "K2d":
            revoke()
            clock[0] = START.replace(hour=9, minute=5)
            dated(30)
        if name == "K2c_after_same_day_revoke":
            revoke()
            clock[0] = START.replace(hour=9, minute=2)
            enable()
            fetch(30)
        if name == "K2e":
            dated(1, "not_earned")
            clock[0] = START.replace(hour=9, minute=1)
            revoke()
            clock[0] = START.replace(hour=9, minute=2)
            enable()
            fetch(30)
            assert not manual.exists()
        if name == "K2a_current_outranked_by_archive":
            interleaved_dated_commands()
        clock[0] = START

        def cascade(_tiers, day):
            check_times.append(clock[0].strftime("%H:%M"))
            if len(check_times) == 1:
                return carbon._pred_frame(day, p1=0.72, p2=None), "local"
            if name == "M2":
                return carbon._pred_frame(day, p1=0.88, p2=None), "local"
            return None, None

        changed = False

        def sleep(secs, *_args, **_kwargs):
            nonlocal changed
            target = clock[0] + dt.timedelta(seconds=max(secs, 0))
            at = START.replace(hour=16, minute=30)
            if check_times and not changed and clock[0] < at <= target:
                env["advance"]((at - clock[0]).total_seconds())
                changed = True
                if name in {"M1", "M4"}:
                    dated(1, "not_earned")
                    env["advance"](60)
                if name in {"M2", "M4"}:
                    revoke()
                if name == "M3":
                    dated(30, "active")
                if name == "K2b":
                    before = manual.read_bytes()
                    with patch.object(util, "atomic_write_text", side_effect=PermissionError("install denied")):
                        message = dated(1, "not_earned", exit_code=1)
                    assert "request was appended" in message and before == manual.read_bytes()
                    log = (account / "contest_manual_requests.jsonl").read_text().splitlines()
                    assert json.loads(log[-1])["active_streak"] == 1
            env["advance"]((target - clock[0]).total_seconds())

        with patch.object(orchestrator, "run_cascade", side_effect=cascade), \
             patch.object(scheduler, "_watchdog_ping_sleep", side_effect=sleep):
            scheduler.run_day(DAY, env["config"])

        decision = daily_decision.load_decision(DAY, picks_dir)
        assert (decision["action"], decision["source"], decision["streak"], decision["saver_available"]) == (
            "skip", "mdp", 30, False), decision
        assert check_times == ["16:10", "17:10", "18:10"] and changed
        assert not (picks_dir / f"{DAY}.json").exists()

        config = {"scheduler": {"pick_delivery": "dm", "entry_intent": "enter"}}
        cfg.write_text('[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n')
        before = snapshot(picks_dir)
        n = len(env["pick_dms"])
        clock[0] = START.replace(day=25, hour=8, minute=30)
        outcome = day_outcome.gather(DAY, picks_dir, config, expected_username="operator")
        result = runner.invoke(cli.cli, ["day-outcome", "--date", DAY, "--config", str(cfg),
                                       "--picks-dir", str(picks_dir), "--expected-username", "operator",
                                       "--dm-recipient", RECIPIENT])
        reasons = EXPECTED[name]
        expected = review_detail(DAY, {
            "M1": [f"The installed dated observation for {DAY} records streak 1 and policy saver availability available."],
            "M2": [f"The season switch records a revocation on {DAY} at 16:30 ET."],
            "M3": [f"The installed dated observation for {DAY} records streak 30 and policy saver availability available."],
            "M4": [f"An archived dated observation for {DAY} records streak 1 and policy saver availability available.",
                   f"The season switch records a revocation on {DAY} at 16:31 ET."],
            "K2e": [f"An archived dated observation for {DAY} records streak 1 and policy saver availability available."],
        }.get(name, [])) if reasons else "The policy skipped the day."
        assert_report(outcome, result, env["pick_dms"][n:], DAY,
                      "skip_state_review" if reasons else "deliberate_skip", expected, int(bool(reasons)))
        assert snapshot(picks_dir) == before
