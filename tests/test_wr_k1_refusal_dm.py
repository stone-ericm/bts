"""Exercise in-day DM text, receipts, and health-runner deduplication."""

import inspect
import contextlib
import io
import os
import json
from datetime import datetime, timedelta
from types import SimpleNamespace
import pytest
from unittest.mock import patch
from pytest import MonkeyPatch

from scripts.audit.season_walk.walk import ET, _invoke, _set_clock, offline
from bts import contest_state as cs, picks, scheduler
from bts.day_outcome import DM_MAX_CODE_POINTS
from bts.health import alert
from bts.season_contract import AUTO_SCHEMA

BOX = "/home/bts/projects/bts/data/picks"
AT = datetime(2027, 3, 24, 16, 1, tzinfo=ET)
DAY = AT.date().isoformat()
GUIDANCE = (
    "See the scheduler journal for this refusal. To inspect current state, "
    "run bts season-contract status with --picks-dir set to the scheduler's absolute picks directory. "
    "For live repair, use the season-start checklist (B8a). Before changing state for a new slate, "
    "verify in MLB that prior entries settled or were absent. "
    "Use a later official pick only while its entry window is open."
)
HEADER = "BTS health CRITICAL alert(s):\n- [contest_state] "
BANNER = "⚠️ Previous health alert DM FAILED to send (1 CRITICAL) — it may have been missed.\n\n"

from bts import contest_state, dm, orchestrator, strategy
from bts.daily_decision import load_decision
from tests.test_pick_chosen_on_cs5a import DAY as CALLER_DAY, daily_pick, set_dated
from tests.test_saved_pick_paths_cs5a import prepare_run_day, scheduler_state, world


def dump(path, data):
    path.write_text(json.dumps(data))


def invoke(*args):
    code, output = _invoke(*args)
    assert code == 0, output
    return output


def base_auto():
    return {"active_streak": 12, "best_streak": 15, "source_date": "2027-03-23"}


def dated():
    return {"schema_version": cs.DATED_MANUAL_SCHEMA, "effective_date": DAY, "season": 2027,
            "expires_at": f"{DAY}T23:59:59.999999-04:00", "active_streak": 12, "best_streak": 15,
            "saver_state": "used", "recorded_at": f"{DAY}T13:00:00-04:00", "source": "MLB"}


def capture(cfg, status, error, prior_count=None, day=DAY):
    if prior_count is not None:
        dump(status, {"status": "failed", "critical_count": prior_count})
    elif status.exists():
        status.unlink()
    attempts = []

    def send(recipient, body):
        attempts.append(body)
        if len(body) > DM_MAX_CODE_POINTS:
            raise RuntimeError("length rejected")
        return "wire"

    kwargs = {"slate_date": day} if "slate_date" in inspect.signature(scheduler._alert_contest_state_failure).parameters else {}
    with patch.object(alert, "send_dm", send):
        scheduler._alert_contest_state_failure(cfg, error, **kwargs)
    assert len(attempts) == 1
    return json.loads(status.read_text())["status"], attempts[0]


def fixtures(case, variant, root, account):
    switch_path = account / "season_contract.json"
    automatic = account / "contest_streak.json"
    manual = account / "contest_streak.manual.json"
    if case == "absent":
        return
    if case == "other_season":
        _set_clock(AT.replace(year=2026))
        invoke("season-contract", "enable", "--season", 2026, "--picks-dir", root)
        _set_clock(AT)
        return
    invoke("season-contract", "enable", "--season", 2027, "--picks-dir", root)
    sw = json.loads(switch_path.read_text())
    if case == "revoked":
        invoke("season-contract", "revoke", "--season", 2027, "--reason", "restore", "--picks-dir", root)
    if case == "switch_corrupt":
        field, value = variant
        if field is None:
            switch_path.write_text("{")
        else:
            sw[field] = value
            dump(switch_path, sw)
    if case == "legacy":
        dump(manual, {**base_auto(), "override_expires_at": f"{DAY}T23:59:00-04:00"})
    if case == "authority":
        dump(automatic, {**base_auto(), "schema_version": AUTO_SCHEMA, "season": 2027,
                         "activation_id": sw["activation_id"], **variant})
    if case.startswith(("auto_", "manual_")):
        dest = automatic if case.startswith("auto_") else manual
        if variant == "bad JSON":
            dest.write_text("{")
        elif variant == "bad UTF8":
            dest.write_bytes(b"\xff")
        elif variant == "directory":
            dest.mkdir()
        else:
            dump(dest, variant)
    if case == "dated":
        field, value = variant
        dump(manual, {**dated(), field: value})
    if case == "model_json":
        (root / "streak.json").write_text("{")
    if case == "model_streak":
        dump(root / "streak.json", {"season": 2027, "streak": False})
    if case == "saved":
        dump(manual, dated())


def refusal(case, variant, root):
    try:
        decision = cs.load_decision_streak_state(
            root, require_contest_state=not case.startswith("model_"), now=AT
        )
    except cs.ContestStateError as error:
        return cs.ContestStateError(str(error).replace(str(root), BOX))
    if case == "saved":
        if variant == "absent":
            chosen_on = None
        else:
            chosen_on = picks.chosen_on_record(decision, required=variant != "preview")
            if variant == "changed":
                chosen_on["dated"]["active_streak"] = 11
                chosen_on["streak"] = 11
        text = picks.saved_pick_refusal(SimpleNamespace(date=DAY, chosen_on=chosen_on), decision)
        assert text
        return cs.ContestStateError(text)
    raise AssertionError((case, variant))


class BrokenString(Exception):
    def __str__(self):
        raise ValueError("broken")


cases = [(key, None) for key in ("absent", "revoked", "no_fetch")]
cases += [(key, None) for key in ("other_season", "legacy", "model_json", "model_streak")]
cases += [("switch_corrupt", v) for v in [
    (None, None), ("schema_version", "bad"), ("season", True), ("status", "bad"),
    ("activation_id", "bad"), ("switched_at", None), ("revoked_at", "bad"), ("note", False),
]]
cases += [("authority", v) for v in [
    {"schema_version": None}, {"schema_version": "obsolete"}, {"season": 2026},
    {"activation_id": "0" * 32},
]]
for kind in ("auto", "manual"):
    cases += [(f"{kind}_parse", v) for v in (
        "bad JSON", "bad UTF8", "directory", [], {"active_streak": True}, {"active_streak": -1},
        {"active_streak": 12, "saver_available": "yes"},
    )]
cases += [("auto_validation", v) for v in (
    {**base_auto(), "active_streak": 401}, {**base_auto(), "best_streak": 11},
    {"active_streak": 12}, {**base_auto(), "schema_version": AUTO_SCHEMA, "season": "2027"},
    {**base_auto(), "schema_version": AUTO_SCHEMA, "season": 2027, "activation_id": "short"},
)]
cases += [("dated", (field, value)) for field, value in [
    ("effective_date", False), ("season", 2026), ("expires_at", "not a timestamp"),
    ("expires_at", f"{DAY}T16:00:00-04:00"), ("active_streak", 401), ("best_streak", 11),
    ("saver_state", "maybe"), ("recorded_at", f"{DAY}T16:00:00"), ("source", 1), ("reason", None),
]]
cases += [("saved", v) for v in ("absent", "preview", "changed")]
cases += [("synthetic", v) for v in ("huge", "empty line", "broken str", "combining", "invalid Unicode")]


@pytest.mark.parametrize("case,variant", cases)
def test_e1_e2_e3_inventory(tmp_path, case, variant):
    with offline(AT):
        root = tmp_path / "data/picks"
        account = root / "account_state"
        account.mkdir(parents=True)
        status = root.parent / "health_state/health_dm_delivery_status.json"
        status.parent.mkdir()
        cfg = {"orchestrator": {"picks_dir": str(root)}, "bluesky": {"dm_recipient": "op.invalid"}}
        if case == "synthetic":
            error = {"huge": cs.ContestStateError("x" * 100000),
                     "empty line": cs.ContestStateError("\nsecond"),
                     "broken str": BrokenString(),
                     "combining": cs.ContestStateError("e\u0301" * 500),
                     "invalid Unicode": cs.ContestStateError("bad \ud800 text")}[variant]
        else:
            fixtures(case, variant, root, account)
            error = refusal(case, variant, root)
        try:
            lines = str(error).splitlines()
            reason = lines[0].strip() if lines else ""
        except ValueError:
            reason = "BrokenString"
        reason = reason.encode("utf-8", "replace").decode() or "(no reason text)"
        if len(reason) > 400:
            reason = reason[:397] + "..."
        bodies = []
        for count in (None, 1):
            result, body = capture(cfg, status, error, prior_count=count)
            assert len(body) < DM_MAX_CODE_POINTS
            body.encode("utf-8")
            assert result == "sent"
            if case == "absent":
                assert "Live decision refused: season 2027 switch is absent." in body
            if case == "revoked":
                assert "Live decision refused: season 2027 switch is revoked." in body
            prefix = BANNER if count else ""
            assert body == f"{prefix}{HEADER}No pick was sent for {DAY} at this check. Reason: {reason}\n{GUIDANCE}"
            bodies.append(body)
        if case in ("absent", "revoked") or variant in ("changed", "huge"):
            print("RENDER", json.dumps([case, variant, bodies], ensure_ascii=False))
        if variant == "huge":
            result, body = capture(cfg, status, error, prior_count=10**700)
            assert result == "sent" and len(body) < DM_MAX_CODE_POINTS
            assert "(999+ CRITICAL)" in body
            result, body = capture(cfg, status, error, day="bad" * 500)
            assert result == "sent" and len(body) < DM_MAX_CODE_POINTS
            assert body.startswith(f"{HEADER}No pick was sent for this slate at this check.")

from bts.contest_state import ContestStateError
from bts.health import runner


def local(day, hm):
    return datetime.fromisoformat(f"{day}T{hm}").replace(tzinfo=ET)


def dms_at(case, root, day, at, cfg):
    if case == "contest_state":
        kwargs = {"slate_date": day} if "slate_date" in inspect.signature(scheduler._alert_contest_state_failure).parameters else {}
        scheduler._alert_contest_state_failure(cfg, ContestStateError(f"switch unreadable at {at:%H:%M}"), **kwargs)
        return
    batter = picks.Pick(batter_name="Test Batter", batter_id=1, team="CLE",
                        lineup_position=1, pitcher_name="Other Player", pitcher_id=2,
                        p_game_hit=.8, flags=[], game_pk=1, projected_lineup=False,
                        game_time=(at + timedelta(minutes=30)).isoformat())
    daily = picks.DailyPick(day, at.isoformat(), batter, None, None)
    if case == "missed_pick":
        picks.save_pick(daily, root)
    state = scheduler.SchedulerState(day, at.isoformat(), [], [], [], False, None, None, None)
    if case == "missed_pick":
        with patch.object(scheduler, "_watchdog_ping_sleep", lambda sec: _set_clock(at + timedelta(seconds=sec))):
            scheduler._maybe_alert_missed_pick(cfg, day, root, 10, None, state)
    elif case == "late_delivery":
        scheduler._refuse_late_delivery(daily, cfg, root, state, day, "fallback", at - timedelta(minutes=1), at)
    else:
        scheduler._refuse_void_game_delivery(daily, cfg, root, state, day, "fallback", {1: "Cancelled"})


@contextlib.contextmanager
def alert_world(tmp_path):
    with offline(local("2027-06-14", "20:00")):
        root = tmp_path / "data/picks"
        root.mkdir(parents=True)
        record = root.parent / "health_state/health_dm_delivery_status.json"
        cfg = {"orchestrator": {"picks_dir": str(root)}, "bluesky": {"dm_recipient": "op.invalid"},
               "scheduler": {"contest_state_required": True}}
        sent = []
        def send(_, body):
            if len(body) > DM_MAX_CODE_POINTS:
                raise ValueError("length rejected")
            sent.append(body)
            return "wire"

        with patch.object(alert, "send_dm", send):
            yield root, record, cfg, sent



@pytest.mark.parametrize("case", ["contest_state", "missed_pick", "late_delivery", "postponed_pick"])
@pytest.mark.parametrize("boundary", ["June", "New Year"])
@pytest.mark.parametrize("row", ["E4-E6", "E7", "E9"])
def test_rollover(tmp_path, case, boundary, row):
    day, next_day = ("2027-06-14", "2027-06-15") if boundary == "June" else ("2027-12-31", "2028-01-01")
    if row == "E4-E6":
        steps = [(day, next_day, "00:02", 1), (next_day, next_day, "08:00", 1)]
    elif row == "E7":
        steps = [(day, day, "23:30", 1), (day, next_day, "00:02", 0), (day, next_day, "00:02", 0)]
    else:
        steps = [(next_day, next_day, "00:01", 1), (day, next_day, "00:02", 1),
                 (next_day, next_day, "08:00", 1)]
    with alert_world(tmp_path) as (root, record, cfg, sent):
        actual = []
        for slate, wall, hm, expected in steps:
            at = local(wall, hm)
            _set_clock(at)
            before = len(sent)
            dms_at(case, root, slate, at, cfg)
            actual.append((len(sent) - before, json.loads(record.read_text())["sent_day"]))
        assert actual == [(expected, slate) for slate, _, _, expected in steps]

def test_cross_midnight_wait(tmp_path):
    with alert_world(tmp_path) as (root, record, cfg, sent):
        at = local("2027-06-14", "23:42")
        _set_clock(at)
        dms_at("missed_pick", root, "2027-06-14", at, cfg)
        status = json.loads(record.read_text())
        assert status["sent_day"] == "2027-06-14"
        assert "00:02" in status["updated_at"] or "04:02" in status["updated_at"]
        print("real wait crossing", status["sent_day"], status["updated_at"])


def test_s1_s2_same_day(tmp_path):
    with alert_world(tmp_path) as (root, record, cfg, sent):
        day = "2027-06-14"
        def daily(case, hm):
            at = local(day, hm)
            _set_clock(at)
            dms_at(case, root, day, at, cfg)

        daily("contest_state", "14:00")
        daily("contest_state", "15:00")
        assert len(sent) == 1  # S1
        daily("missed_pick", "16:00")
        assert len(sent) == 2  # S2
        print("S1/S2", len(sent), json.loads(record.read_text())["sent_sources"])


def test_s4_other_dm_bodies(tmp_path):
    expected = {
        ("missed_pick", True): "BTS health CRITICAL alert(s):\n- [missed_pick] NO PICK DELIVERED. ~10 min to first pitch. See `bts season-contract status`.",
        ("missed_pick", False): "BTS health CRITICAL alert(s):\n- [missed_pick] NO PICK DELIVERED with ~10 min to first pitch (top pick: Test Batter). Deliver manually if you want to keep the streak alive.",
        ("late_delivery", True): "BTS health CRITICAL alert(s):\n- [late_delivery] LATE DELIVERY REFUSED: the 19:59 ET cutoff for Test Batter had been reached. This pick was withheld. A later check may produce another recommendation. Use a later official pick message only while the entry window is open. A manual entry can differ from a later recommendation. The entry check may report a mismatch if it verifies the entry in time.",
        ("postponed_pick", True): "BTS health CRITICAL alert(s):\n- [postponed_pick] DELIVERY REFUSED for 2027-06-14: game 1 is Cancelled. This pick was withheld. A later check may produce another recommendation. Use a later official pick message only while the entry window is open. A manual entry can differ from a later recommendation. The entry check may report a mismatch if it verifies the entry in time.",
    }
    for case in ("missed_pick", "late_delivery", "postponed_pick"):
        for live in (True, False) if case == "missed_pick" else (True,):
            with alert_world(tmp_path / f"{case}-{live}") as (root, record, cfg, sent):
                cfg["scheduler"]["contest_state_required"] = live
                dms_at(case, root, "2027-06-14", local("2027-06-14", "20:00"), cfg)
                assert len(sent) == 1
                assert sent[0] == expected[(case, live)]


def test_s3_s5_s6_failure_and_retry(tmp_path):
    with alert_world(tmp_path) as (root, record, cfg, sent):
        at = local("2027-06-14", "20:00")
        cfg["bluesky"].clear()
        dms_at("contest_state", root, "2027-06-14", at, cfg)
        assert not sent and json.loads(record.read_text())["status"] == "skipped_no_recipient"
        cfg["bluesky"]["dm_recipient"] = "op.invalid"
        with patch.object(alert, "send_dm", side_effect=RuntimeError("line gone")):
            dms_at("contest_state", root, "2027-06-14", at, cfg)
        assert json.loads(record.read_text())["status"] == "failed"
        dms_at("contest_state", root, "2027-06-14", at, cfg)
        assert len(sent) == 1 and sent[0].startswith("⚠️ Previous health alert DM FAILED")
        print("S3/S5/S6 passed")


def test_s7_bad_date(tmp_path):
    for day in ("bad", None, ""):
        with alert_world(tmp_path / repr(day)) as (root, record, cfg, sent):
            dms_at("contest_state", root, day, local("2027-06-14", "20:00"), cfg)
            assert len(sent) == 1 and json.loads(record.read_text())["sent_day"] == "2027-06-14"
            print("S7", repr(day), sent[0].splitlines()[1][:70])


def test_standing_receipts(tmp_path):
    for shape in ("malformed", "FIFO", "directory", "interrupt during send", "after send before write", "no ID"):
        with alert_world(tmp_path / shape) as (root, record, cfg, sent):
            at = local("2027-06-14", "20:00")
            record.parent.mkdir()
            if shape == "malformed":
                record.write_text("{")
            elif shape == "FIFO":
                os.mkfifo(record)
            elif shape == "directory":
                record.mkdir()
            try:
                if shape in ("interrupt during send", "after send before write"):
                    target = "send_dm" if shape == "interrupt during send" else "_persist_health_dm_delivery_status"
                    with patch.object(alert, target, side_effect=KeyboardInterrupt):
                        dms_at("contest_state", root, "2027-06-14", at, cfg)
                    raise AssertionError("interrupt swallowed")
                if shape == "no ID":
                    with patch.object(alert, "send_dm", return_value=None):
                        dms_at("contest_state", root, "2027-06-14", at, cfg)
                    assert json.loads(record.read_text())["status"] == "sent"
            except KeyboardInterrupt:
                assert not record.exists()
            dms_at("contest_state", root, "2027-06-14", at, cfg)
            assert len(sent) == {"after send before write": 2, "no ID": 0}.get(shape, 1)
            print("standing", shape, len(sent))


def test_e8_historical_contest_state(tmp_path):
    with alert_world(tmp_path) as (root, record, cfg, sent):
        day = "2027-06-14"
        at = local("2027-06-15", "00:02")
        _set_clock(at)
        dms_at("contest_state", root, day, at, cfg)
        account = root / "account_state"
        account.mkdir()
        (account / "contest_streak.json").write_text("[]")
        checked = runner._safe_run
        with patch.object(runner, "_safe_run", lambda name, fn: checked(name, fn) if name == "contest_state" else []):
            for _ in range(2):
                alerts = runner.run_all_checks(root, root.parent / "models", "op.invalid",
                                               today=datetime.fromisoformat(day).date(), contest_state_expected=True)
        assert any(a.source == "contest_state" and "Prior slate" in a.message
                   and "contest streak file is malformed" in a.message for a in alerts)
        assert len(sent) == 1


def test_e8_historical_late_delivery(tmp_path):
    with alert_world(tmp_path) as (root, record, cfg, sent):
        day = "2027-06-14"
        at = local("2027-06-15", "00:02")
        _set_clock(at)
        dms_at("late_delivery", root, day, at, cfg)
        # A second ordinary refusal adds distinct detail to the same EOD source.
        state = scheduler.load_state(day, root)
        state.delivery_refusals.append({**state.delivery_refusals[0], "batter": "Second Batter"})
        scheduler.save_state(state, root)
        checked = runner._safe_run
        with patch.object(runner, "_safe_run", lambda name, fn: checked(name, fn) if name == "late_delivery" else []):
            alerts = runner.run_all_checks(root, root.parent / "models", "op.invalid",
                                           today=datetime.fromisoformat(day).date())
        assert any(a.source == "late_delivery" and "Second Batter" in a.message for a in alerts)
        assert len(sent) == 1
        print("EOD real late-delivery detail", len(sent), [a.message for a in alerts if a.source == "late_delivery"])


def command(cmd, directory, *args, at=None):
    with offline(at) if at is not None else contextlib.nullcontext():
        rc, text = _invoke("season-contract", cmd, "--picks-dir", directory, *args)
    assert rc == 0, text
    return text


@pytest.mark.parametrize("case", ["check", "revoked_at_check", "delivery_loader", "saved_changed", "in_loop", "final_midnight"])
def test_s8_real_callers(case, tmp_path):
    with MonkeyPatch.context() as mp, contextlib.redirect_stderr(io.StringIO()) as err:
        directory0 = tmp_path
        cfg, directory, now, statuses = world.__wrapped__(directory0, mp)
        mp.setattr(strategy, "_mdp_cache", None)
        mp.setattr(scheduler, "notify_ready", lambda: None)
        mp.setattr(scheduler, "notify_watchdog", lambda: None)
        prepare_run_day(mp)
        day = CALLER_DAY
        cached = daily_pick()
        sent = []
        routes = []

        def journal_cause():
            rows = err.getvalue().splitlines()
            return [line for line in rows if "CONTEST STATE ERROR" in line or "FALLBACK BLOCKED" in line]

        def send(_, text):
            assert journal_cause()
            sent.append(text)
            if len(text) > 1000:
                raise ValueError("length rejected")
            return "wire"

        dispatch = alert.dispatch_dm_for_health_alerts

        def observe(*args, **kwargs):
            caller = inspect.currentframe().f_back
            if caller.f_code.co_name == "_alert_contest_state_failure":
                parent = caller.f_back
                routes.append((parent.f_code.co_name, parent.f_lineno, kwargs.get("now_et_date")))
            return dispatch(*args, **kwargs)

        mp.setattr(alert, "send_dm", send)
        mp.setattr(alert, "dispatch_dm_for_health_alerts", observe)
        if case == "check":
            scheduler.run_single_check(day, [101, 102], set(), cfg, 0.03)
        elif case == "revoked_at_check":
            now[0] -= timedelta(hours=8)
            command("enable", directory, "--season", 2027, at=now[0])
            now[0] += timedelta(hours=8)
            command("revoke", directory, "--season", 2027, "--reason", "operator rehearsal", at=now[0])
            scheduler.run_single_check(day, [101, 102], set(), cfg, 0.03)
        elif case in ("delivery_loader", "saved_changed"):
            if case == "saved_changed":
                set_dated(directory, streak=12, best=16)
                old = contest_state.load_decision_streak_state(directory, require_contest_state=True)
                cached.chosen_on = picks.chosen_on_record(old, required=True)
                set_dated(directory, streak=13, best=16)
            path = picks.save_pick(cached, directory)
            original = path.read_bytes()
            state = scheduler_state()
            assert not scheduler._deliver_and_lock_pick(cached, cfg, directory, state, day, "lineup")
            assert original == path.read_bytes() and not state.pick_locked and load_decision(day, directory) is None
            dm.send_dm.assert_not_called()
        else:
            cfg["scheduler"]["entry_intent"] = "enter"
            if case == "in_loop":
                set_dated(directory)
                mp.setattr(scheduler, "_lock_decision_from_predictions",
                           lambda *a, **kw: scheduler.LockDecision(False, None, False))
                now[0] -= timedelta(minutes=10)
                frame, tier = orchestrator.run_cascade.return_value

                def predict(*_):
                    if orchestrator.run_cascade.call_count > 1:
                        (directory / "account_state/contest_streak.manual.json").unlink(missing_ok=True)
                    return frame, tier

                orchestrator.run_cascade.side_effect = predict
            else:
                games = scheduler.fetch_schedule(day)
                midnight = datetime.fromisoformat(day).replace(tzinfo=ET) + timedelta(days=1)
                for i, game in enumerate(games):
                    game["gameDate"] = (midnight + timedelta(minutes=40 + 30 * i)).isoformat()
                cached.pick.game_time = games[0]["gameDate"]
                now[0] = midnight - timedelta(minutes=5)
                scheduler.fetch_schedule.side_effect = lambda date: games if date == day else []
                frame, tier = orchestrator.run_cascade.return_value
                orchestrator.run_cascade.side_effect = [(None, None), (None, None), (frame, tier)]

                def advance(seconds):
                    now[0] += timedelta(seconds=seconds)

                mp.setattr(scheduler, "_watchdog_ping_sleep", advance)
            picks.save_pick(cached, directory)
            scheduler.run_day(day, cfg)

        before = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()} if directory.exists() else {}
        status = command("status", directory, at=now[0])
        after = {p.relative_to(directory): p.read_bytes() for p in directory.rglob("*") if p.is_file()} if directory.exists() else {}
        assert after == before
        live, = [line for line in status.splitlines() if line.startswith("live decision now:")]
        assert routes and sent
        if case == "saved_changed":
            assert "chosen on dated" in journal_cause()[0] and "accepted" in live and "streak=13" in live
        else:
            assert "refused" in live
            assert "Alternative dated path" in status and "Alternative automatic path" in status
        if case == "final_midnight":
            assert all(name == "run_day" for name, _, _ in routes)
            assert now[0].date().isoformat() > day
        if case == "final_midnight":
            assert all(d is not None and d.isoformat() == day for _, _, d in routes)

@pytest.mark.parametrize("row", ["E4", "E7"])
def test_real_switch_over_midnight(tmp_path, row):
    day, following = "2027-06-14", "2027-06-15"
    with alert_world(tmp_path) as (root, record, cfg, sent):
        _set_clock(local(day, "23:55"))
        invoke("season-contract", "enable", "--season", 2027, "--picks-dir", root)
        if row == "E4":
            steps = [(day, following, "00:02", "no automatic observation"),
                     (following, following, "08:00", "no automatic observation")]
        else:
            steps = [(day, day, "23:55", "no automatic observation"),
                     (day, following, "00:02", "malformed"),
                     (day, following, "00:02", "malformed")]
        for slate, wall, hm, reason in steps:
            _set_clock(local(wall, hm))
            if reason == "malformed":
                (root / "account_state/contest_streak.json").write_text("[]")
            with pytest.raises(cs.ContestStateError, match=reason) as caught:
                cs.load_decision_streak_state(root, require_contest_state=True)
            scheduler._alert_contest_state_failure(cfg, caught.value, slate_date=slate)
        assert len(sent) == (2 if row == "E4" else 1)
        assert json.loads(record.read_text())["sent_day"] == steps[-1][0]


def test_e8_new_key_sends_then_rerun_dedups(tmp_path):
    day = "2027-06-14"
    at = local("2027-06-15", "00:02")
    with alert_world(tmp_path) as (root, record, cfg, sent):
        _set_clock(at)
        dms_at("postponed_pick", root, day, at, cfg)
        checked = runner._safe_run
        def only_delivery(name, fn):
            return checked(name, fn) if name in ("late_delivery", "postponed_pick") else []

        with patch.object(runner, "_safe_run", only_delivery):
            for _ in range(2):
                alerts = runner.run_all_checks(root, root.parent / "models", "op.invalid",
                                               today=datetime.fromisoformat(day).date())
        assert any(item.source == "late_delivery" for item in alerts)
        assert len(sent) == 2
        assert json.loads(record.read_text())["sent_day"] == day
