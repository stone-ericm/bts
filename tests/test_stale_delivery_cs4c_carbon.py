"""Adversarial regression tests for CS4c (stale automatic reading stamp & warning).

Covers unmasked validator mutations (bool-for-float on non-stale records, float-for-bool,
unbounded/non-ASCII/separator strings on unreadable and old records, negative age bound),
decoupled warning_day vs alert.datetime deduplication, send-without-id + re-entry,
closed stderr independence, BaseException isolation after commit, and real run_day
fresh-to-stale ageing across fallback.
"""

from __future__ import annotations

import io
import json
import sys
from contextlib import nullcontext
from dataclasses import asdict
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import pytest

from bts import contest_state, daily_decision, dm, picks, posting, scheduler, strategy
from bts.health import alert

ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 18, 35, 0, tzinfo=ET)
SCHEMA = "bts_pick_automatic_reading_v1"
KEYS = ("schema", "recorded_at", "age_hours", "stale", "reason")
GOOD_ISO = "2027-07-14T22:35:00-04:00"
WARN_FILE = "contest_stale_reading_dm_delivery_status.json"
WARN_KEY = "WARN_ATTENTION:contest_state:stale_reading"
WARN_PREFIX = "  STALE CONTEST READING — "


def _make_pick() -> picks.DailyPick:
    p1 = picks.Pick(
        "Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037,
        0.82, [], False, 101, f"{DAY}T23:10:00Z",
    )
    return picks.DailyPick(DAY, NOW.isoformat(), p1, None, None)


def _fresh_selection(daily: picks.DailyPick) -> strategy.SelectionResult:
    pol = strategy.PolicyDecision(
        "single", "single", "heuristic", "reach57", 12, 80, None, "missing",
        None, None, None, None, None,
    )
    return strategy.SelectionResult(
        strategy.PickResult(daily), "single", "heuristic",
        scheduler._row_from_daily(daily.pick), None, None, streak=12, decision=pol,
    )


def _seed_covered_auto(picks_dir: Path, *, recorded_at: str, now: datetime = NOW, streak: int = 12) -> None:
    acct = picks_dir / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    (acct / "season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1",
        "season": now.year,
        "status": "automatic",
        "activation_id": "a" * 32,
        "switched_at": now.isoformat(),
        "revoked_at": None,
        "note": None,
    }))
    (acct / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2",
        "season": now.year,
        "activation_id": "a" * 32,
        "active_streak": streak,
        "best_streak": max(streak, 15),
        "source": "mlb_bts_profile",
        "source_date": now.date().isoformat(),
        "recorded_at": recorded_at,
    }))


@pytest.mark.parametrize("bad_record", [
    # Unmasked bool-for-float: reason=None so [-0.08, 14.0] does NOT reject True (1) or False (0)
    {"schema": SCHEMA, "recorded_at": GOOD_ISO, "age_hours": True, "stale": False, "reason": None},
    {"schema": SCHEMA, "recorded_at": GOOD_ISO, "age_hours": False, "stale": False, "reason": None},
    # Float-for-bool: 1.0 == True and 0.0 == False in Python
    {"schema": SCHEMA, "recorded_at": GOOD_ISO, "age_hours": 20.0, "stale": 1.0, "reason": "old"},
    {"schema": SCHEMA, "recorded_at": GOOD_ISO, "age_hours": 2.0, "stale": 0.0, "reason": None},
    # Unmasked non-printable/overlong/surrogate recorded_at on reason="unreadable"
    {"schema": SCHEMA, "recorded_at": "x" * 129, "age_hours": None, "stale": True, "reason": "unreadable"},
    {"schema": SCHEMA, "recorded_at": "\ud800", "age_hours": None, "stale": True, "reason": "unreadable"},
    {"schema": SCHEMA, "recorded_at": "bad\nline", "age_hours": None, "stale": True, "reason": "unreadable"},
    {"schema": SCHEMA, "recorded_at": "bad\x7f", "age_hours": None, "stale": True, "reason": "unreadable"},
    {"schema": SCHEMA, "recorded_at": "badé", "age_hours": None, "stale": True, "reason": "unreadable"},
    # Python 3.12 fromisoformat accepts \n and \ud800 as date-time separators unless bounded ASCII rejects them
    {"schema": SCHEMA, "recorded_at": "2027-07-14\n22:35:00-04:00", "age_hours": 20.0, "stale": True, "reason": "old"},
    {"schema": SCHEMA, "recorded_at": "2027-07-14\ud80022:35:00-04:00", "age_hours": 20.0, "stale": True, "reason": "old"},
    # Negative out-of-range age_hours (kills missing abs() on 87,660,000)
    {"schema": SCHEMA, "recorded_at": GOOD_ISO, "age_hours": -87_660_000.01, "stale": True, "reason": "future"},
])
def test_unmasked_validator_rejections_across_all_loaders(tmp_path, bad_record):
    assert contest_state.valid_automatic_reading(bad_record) is None
    body = asdict(_make_pick())
    body["automatic_reading"] = bad_record
    raw = json.dumps(body).encode("utf-8")
    (tmp_path / f"{DAY}.json").write_bytes(raw)
    (tmp_path / f"{DAY}.shadow.json").write_bytes(raw)
    assert picks.load_pick(DAY, tmp_path).automatic_reading is None
    loaded_b, held = picks.load_pick_bytes(DAY, tmp_path)
    assert held == raw and loaded_b.automatic_reading is None
    assert picks.load_shadow_pick(DAY, tmp_path).automatic_reading is None


def test_negative_max_age_bound_accepted():
    valid_neg = {
        "schema": SCHEMA, "recorded_at": GOOD_ISO,
        "age_hours": -87_660_000.0, "stale": True, "reason": "future",
    }
    assert contest_state.valid_automatic_reading(valid_neg) == valid_neg


def test_warning_passes_explicit_now_et_date_independent_of_alert_clock(tmp_path, monkeypatch):
    class ContestClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    alert_tick = [datetime(2028, 1, 1, 12, 0, tzinfo=ET)]

    class DivergentAlertClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return alert_tick[0].astimezone(tz) if tz else alert_tick[0].replace(tzinfo=None)

    monkeypatch.setattr(contest_state, "datetime", ContestClock)
    monkeypatch.setattr(alert, "datetime", DivergentAlertClock)
    monkeypatch.setattr(scheduler, "_now_et", lambda: NOW)
    monkeypatch.setattr(scheduler, "_delivery_game_statuses", lambda _d: {})
    monkeypatch.setattr(scheduler, "_trigger_live_forward_capture_on_lock", lambda *_: None)
    monkeypatch.setattr(posting, "post_to_bluesky", Mock(return_value="at://1"))
    warn_dm = Mock(return_value="warn-dm")
    monkeypatch.setattr(alert, "send_dm", warn_dm)

    for idx, subname in enumerate(("picks_a", "picks_b")):
        # Advance alert.datetime between the two deliveries so omitted now_et_date would fail dedup
        alert_tick[0] = datetime(2028, 1, 1 + idx, 12, 0, tzinfo=ET)
        pdir = tmp_path / subname
        _seed_covered_auto(pdir, recorded_at=GOOD_ISO)
        daily = _make_pick()
        cfg = {
            "orchestrator": {"picks_dir": str(pdir)},
            "bluesky": {"dm_recipient": "operator.test"},
            "scheduler": {"contest_state_required": True, "pick_delivery": "public",
                          "live_forward_capture_on_lock": False},
        }
        st = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
        assert scheduler._deliver_and_lock_pick(
            daily, cfg, pdir, st, DAY, "dedup", selection=_fresh_selection(daily),
        ) is True

    assert posting.post_to_bluesky.call_count == 2
    assert warn_dm.call_count == 1
    status = json.loads((tmp_path / "health_state" / WARN_FILE).read_text())
    assert status["sent_day"] == DAY
    assert status["sent_sources"] == [WARN_KEY]


@pytest.mark.parametrize("mode", ["dm", "public"])
@pytest.mark.parametrize("empty_id", [None, ""])
def test_send_returning_no_id_and_reentry(tmp_path, monkeypatch, mode, empty_id):
    class ContestClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    pdir = tmp_path / "picks"
    _seed_covered_auto(pdir, recorded_at=GOOD_ISO)
    daily = _make_pick()
    cfg = {
        "orchestrator": {"picks_dir": str(pdir)},
        "bluesky": {"dm_recipient": "operator.test"},
        "scheduler": {"contest_state_required": True, "pick_delivery": mode,
                      "live_forward_capture_on_lock": False},
    }
    monkeypatch.setattr(contest_state, "datetime", ContestClock)
    monkeypatch.setattr(alert, "datetime", ContestClock)
    monkeypatch.setattr(scheduler, "_now_et", lambda: NOW)
    monkeypatch.setattr(scheduler, "_delivery_game_statuses", lambda _d: {})
    monkeypatch.setattr(scheduler, "_trigger_live_forward_capture_on_lock", lambda *_: None)
    monkeypatch.setattr(dm, "send_dm", Mock(return_value=empty_id))
    monkeypatch.setattr(posting, "post_to_bluesky", Mock(return_value=empty_id))
    warn_dm = Mock(return_value="w-1")
    monkeypatch.setattr(alert, "send_dm", warn_dm)

    st = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
    assert scheduler._deliver_and_lock_pick(
        daily, cfg, pdir, st, DAY, "first", selection=_fresh_selection(daily),
    ) is True
    saved = picks.load_pick(DAY, pdir)
    assert saved.automatic_reading is not None and saved.automatic_reading["stale"] is True
    # Round-1 fix (Codex B2): the stamp stays, but with no identifier from the
    # transport the "pick was delivered" warning is withheld. (At the reviewed
    # commit this sent one warning; Carbon recorded that as its note N2.)
    assert warn_dm.call_count == 0

    # Re-entry: public hits already_delivered (True, no alert); DM without notification_id hits crash guard (False, CRITICAL, no stale warn)
    warn_dm.reset_mock()
    st2 = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
    re_ok = scheduler._deliver_and_lock_pick(saved, cfg, pdir, st2, DAY, "reentry")
    if mode == "public":
        assert re_ok is True
        assert not warn_dm.called
    else:
        assert re_ok is False
        assert warn_dm.call_count == 1
        assert "DELIVERY OUTCOME UNKNOWN" in warn_dm.call_args.args[1]


def test_closed_stderr_and_base_exception_after_commit_preserve_delivery(tmp_path, monkeypatch):
    class ContestClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    class CloseOnLocked(io.StringIO):
        def write(self, s: str) -> int:
            n = super().write(s)
            if "LOCKED (" in self.getvalue() and s.endswith("\n") and not self.closed:
                self.close()
            return n

    monkeypatch.setattr(contest_state, "datetime", ContestClock)
    monkeypatch.setattr(daily_decision, "datetime", ContestClock)
    monkeypatch.setattr(alert, "datetime", ContestClock)
    monkeypatch.setattr(scheduler, "_now_et", lambda: NOW)
    monkeypatch.setattr(scheduler, "_delivery_game_statuses", lambda _d: {})
    monkeypatch.setattr(scheduler, "_trigger_live_forward_capture_on_lock", lambda *_: None)
    monkeypatch.setattr(dm, "send_dm", Mock(return_value="dm-1"))

    # 1. Closed stderr after LOCKED line still dispatches warning DM and returns True
    pdir1 = tmp_path / "p1" / "picks"
    _seed_covered_auto(pdir1, recorded_at=GOOD_ISO)
    d1 = _make_pick()
    cfg1 = {
        "orchestrator": {"picks_dir": str(pdir1)},
        "bluesky": {"dm_recipient": "operator.test"},
        "scheduler": {"contest_state_required": True, "pick_delivery": "dm",
                      "live_forward_capture_on_lock": False},
    }
    st1 = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
    warn_dm = Mock(return_value="w-1")
    monkeypatch.setattr(alert, "send_dm", warn_dm)
    stream = CloseOnLocked()
    with monkeypatch.context() as mp:
        mp.setattr(sys, "stderr", stream)
        assert scheduler._deliver_and_lock_pick(
            d1, cfg1, pdir1, st1, DAY, "closed", selection=_fresh_selection(d1),
        ) is True
    assert stream.closed and warn_dm.call_count == 1
    assert picks.pick_was_delivered(picks.load_pick(DAY, pdir1))

    # 2. KeyboardInterrupt inside warning leaves pick delivered, locked, and committed
    pdir2 = tmp_path / "p2" / "picks"
    _seed_covered_auto(pdir2, recorded_at=GOOD_ISO)
    d2 = _make_pick()
    cfg2 = {
        "orchestrator": {"picks_dir": str(pdir2)},
        "bluesky": {"dm_recipient": "operator.test"},
        "scheduler": {"contest_state_required": True, "pick_delivery": "dm",
                      "live_forward_capture_on_lock": False},
    }
    st2 = scheduler.SchedulerState(DAY, NOW.isoformat(), [], [], [], False, None, None, None)
    monkeypatch.setattr(alert, "dispatch_dm_for_health_alerts", Mock(side_effect=KeyboardInterrupt("int")))
    # Round-1 fix (Codex B1): the interrupt no longer escapes the warning work;
    # the delivery returns as it would have without the warning.
    assert scheduler._deliver_and_lock_pick(
        d2, cfg2, pdir2, st2, DAY, "kbi", selection=_fresh_selection(d2),
    ) is True
    saved2 = picks.load_pick(DAY, pdir2)
    assert picks.pick_was_delivered(saved2) and saved2.automatic_reading["stale"] is True
    assert scheduler.load_state(DAY, pdir2).pick_locked is True
    assert daily_decision.load_decision(DAY, pdir2)["delivery_status"] == "delivered"


def test_run_day_reading_ages_past_14h_between_choice_and_fallback(tmp_path, monkeypatch):
    import pandas as pd
    from bts import orchestrator

    pdir = tmp_path / "picks"
    now_box = [datetime(2027, 7, 15, 18, 20, 0, tzinfo=ET)]
    recorded = datetime(2027, 7, 15, 4, 26, 0, tzinfo=ET).isoformat()  # 13h59m at 18:25, 14h09m at 18:35
    _seed_covered_auto(pdir, recorded_at=recorded, now=now_box[0], streak=30)

    class SimClock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now_box[0].astimezone(tz) if tz else now_box[0].replace(tzinfo=None)

    games = [{
        "gamePk": 101, "gameDate": f"{DAY}T23:10:00Z",
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
        "teams": {"away": {"team": {"name": "Away 101"}}, "home": {"team": {"name": "Home 101"}}},
    }]
    preds = pd.DataFrame([{
        "batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
        "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
        "p_hit_pa": 0.28, "p_game_hit": 0.99, "flags": "PROJECTED", "projected_lineup": True,
        "game_pk": 101, "game_time": f"{DAY}T23:10:00Z",
    }])
    calls = 0

    def cascade_fn(*_a, **_kw):
        nonlocal calls
        calls += 1
        if calls == 1:
            return preds, "offline"
        raise RuntimeError("fallback cascade failed")

    monkeypatch.setattr(contest_state, "datetime", SimClock)
    monkeypatch.setattr(daily_decision, "datetime", SimClock)
    monkeypatch.setattr(picks, "datetime", SimClock)
    monkeypatch.setattr(alert, "datetime", SimClock)
    monkeypatch.setattr(scheduler, "_now_et", lambda: now_box[0])
    monkeypatch.setattr(scheduler, "fetch_schedule", lambda d: games if d == DAY else [])
    monkeypatch.setattr(scheduler, "check_confirmed_lineups", lambda *_: {})
    monkeypatch.setattr(scheduler, "_idle_until_next_wakeup", lambda *_: None)
    monkeypatch.setattr(
        scheduler, "_watchdog_ping_sleep",
        lambda secs: now_box.__setitem__(0, now_box[0] + timedelta(seconds=max(0, secs))),
    )
    monkeypatch.setattr(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext())
    monkeypatch.setattr(scheduler, "run_result_polling", lambda *a, **k: "final")
    monkeypatch.setattr(orchestrator, "run_cascade", cascade_fn)
    monkeypatch.setattr(
        picks, "get_game_statuses_detailed",
        lambda *_: {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}},
    )
    monkeypatch.setattr(
        strategy, "get_game_statuses_detailed",
        lambda *_: {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}},
    )
    monkeypatch.setattr(picks, "get_game_statuses", lambda *_: {101: "P"})
    pick_dm = Mock(return_value="pick-dm-1")
    warn_dm = Mock(return_value="warn-dm-1")
    monkeypatch.setattr(dm, "send_dm", pick_dm)
    monkeypatch.setattr(alert, "send_dm", warn_dm)

    cfg = {
        "orchestrator": {
            "picks_dir": str(pdir),
            "heartbeat_path": str(tmp_path / "hb"),
            "models_dir": "data/models",
        },
        "bluesky": {"dm_recipient": "operator.test"},
        "scheduler": {
            "contest_state_required": True, "pick_delivery": "dm",
            "entry_intent": "enter", "live_forward_capture_on_lock": False,
        },
        "health_checks": {"enabled": False},
        "tiers": [],
    }
    scheduler.run_day(date=DAY, config=cfg)
    final = picks.load_pick(DAY, pdir)
    assert pick_dm.call_count == 1 and warn_dm.call_count == 1
    assert final.chosen_on["kind"] == "automatic"
    assert final.automatic_reading == {
        "schema": SCHEMA, "recorded_at": recorded, "age_hours": 14.15, "stale": True, "reason": "old",
    }
    assert "about 14.15 hours ago" in warn_dm.call_args.args[1]
