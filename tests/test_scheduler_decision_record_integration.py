"""Integration tests for the decision-record wiring (Task 3, P0 #4).

Pure-helper tests (test_scheduler_decision_record.py) are not enough: these drive
the REAL control flow — run_day's classification chokepoint + skip-candidate
SET/CLEAR + end-of-day skip write, and the real _deliver_and_lock_pick commit
branches — and assert the authoritative decision.json via load_decision.

run_single_check is mocked (the cascade is hetzner-only and slow), but everything
downstream of it in run_day is the real code under test.
"""
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

from bts.daily_decision import load_decision
from bts.picks import DailyPick, Pick
from bts.strategy import PickResult, SelectionResult
import bts.scheduler as sch

ET = ZoneInfo("America/New_York")
DATE = "2026-04-06"


# --- builders -------------------------------------------------------------

def _cand(bid=1, p=0.78):
    return {"batter_id": bid, "batter_name": "X", "team": "NYM", "game_pk": 9, "p_game_hit": p}


def _sel(action, source, *, primary=None, double=None, streak=0, saver=None):
    return SelectionResult(pick_result=None, action=action, source=source,
                           primary_candidate=primary, double_candidate=double,
                           no_pick_reason=None, streak=streak, saver_available=saver)


def _pick(bid=1, name="Hoerner", team="CHC", gpk=100):
    return Pick(batter_name=name, batter_id=bid, team=team, lineup_position=1,
                pitcher_name="Baz", pitcher_id=2, p_game_hit=0.73, flags=[],
                projected_lineup=False, game_pk=gpk, game_time="2026-04-06T23:05:00Z")


def _daily(delivered=False, double_down=None, **kw):
    d = DailyPick(date=DATE, run_time="2026-04-06T19:29:00+00:00", pick=_pick(),
                  double_down=double_down, runner_up=None)
    if delivered:
        d.bluesky_posted = True
    for k, v in kw.items():
        setattr(d, k, v)
    return d


def _state():
    return sch.SchedulerState(
        date=DATE, schedule_fetched_at="t", games=[], confirmed_game_pks=[],
        runs_completed=[], pick_locked=False, pick_locked_at=None,
        result_status=None, next_wakeup=None,
    )


def _result(**kw):
    base = {"skipped": False, "new_lineups": 0, "should_post": False,
            "pick_result": None, "pick_name": None, "pick_p": None, "selection": None}
    base.update(kw)
    return base


def _game(game_pk, time_et, date=DATE):
    et_dt = datetime.strptime(f"{date} {time_et}", "%Y-%m-%d %H:%M").replace(tzinfo=ET)
    utc_iso = et_dt.astimezone(ZoneInfo("UTC")).isoformat().replace("+00:00", "Z")
    return {"gamePk": game_pk, "gameDate": utc_iso,
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {"away": {"team": {"name": "NYM"}}, "home": {"team": {"name": "ATL"}}}}


def _seq(items):
    """side_effect-style callable: pops successive items, repeats the last."""
    box = list(items)
    def _next(*a, **k):
        return box.pop(0) if len(box) > 1 else box[0]
    return _next


def _drive_run_day(monkeypatch, tmp_path, check_results, *, games=None,
                   pick_delivery="private", mock_post=False, now_et="23:30",
                   poll_fn=None):
    """Run the REAL run_day to completion with run_single_check mocked.

    health_checks disabled and all sleeps/network neutralized so only the
    decision-record control flow under test runs.
    """
    games = games or [_game(100, "19:05")]
    if not isinstance(check_results, (list, tuple)):
        check_results = [check_results]
    monkeypatch.setattr(sch, "fetch_schedule", _seq([games, []]))
    monkeypatch.setattr(sch, "_now_et",
                        lambda: datetime.strptime(f"{DATE} {now_et}", "%Y-%m-%d %H:%M").replace(tzinfo=ET))
    monkeypatch.setattr(sch.time, "sleep", lambda *a, **k: None)
    monkeypatch.setattr(sch, "run_result_polling",
                        poll_fn if poll_fn is not None else (lambda *a, **k: "final"))
    monkeypatch.setattr(sch, "_trigger_live_forward_capture_on_lock", lambda *a, **k: None)
    monkeypatch.setattr(sch, "run_single_check", _seq(list(check_results)))
    if mock_post:
        monkeypatch.setattr("bts.posting.post_to_bluesky", lambda *a, **k: "at://uri")
        monkeypatch.setattr("bts.contest_state.load_decision_streak_state",
                            lambda *a, **k: SimpleNamespace(streak=10, saver_available=True, allow_double=True))
    config = {
        "orchestrator": {"picks_dir": str(tmp_path), "heartbeat_path": str(tmp_path / ".hb")},
        "tiers": [],
        "scheduler": {"pick_delivery": pick_delivery, "early_lock_gap": 0.03,
                      "lineup_check_offset_min": 45, "cluster_min": 10,
                      "doubleheader_recheck_min": 15, "missed_pick_alert_min": 10,
                      "results_poll_interval_min": 15, "results_cap_hour_et": 5},
        "health_checks": {"enabled": False},
    }
    sch.run_day(date=DATE, config=config, dry_run=False)


# --- delivery-branch commit records (real _deliver_and_lock_pick) ----------

def test_public_commit_via_run_day_writes_delivered_scoreable(tmp_path, monkeypatch):
    """run_day threads state + selection into _deliver_and_lock_pick; a public post
    records a scoreable 'delivered' decision with the selection's source."""
    sel = _sel("single", "mdp", primary=_cand())
    _drive_run_day(
        monkeypatch, tmp_path,
        _result(should_post=True, selection=sel, pick_name="Hoerner", pick_p=0.73,
                pick_result=PickResult(daily=_daily(), locked=False)),
        pick_delivery="public", mock_post=True,
    )
    d = load_decision(DATE, tmp_path)
    assert d is not None
    assert d["action"] == "single"
    assert d["source"] == "mdp"
    assert d["delivery_status"] == "delivered"
    assert d["scoreable"] is True


def test_private_commit_writes_private_locked_scoreable(tmp_path, monkeypatch):
    monkeypatch.setattr(sch, "_trigger_live_forward_capture_on_lock", lambda *a, **k: None)
    st = _state()
    sel = _sel("single", "mdp", primary=_cand())
    config = {"scheduler": {"pick_delivery": "private"}}
    ok = sch._deliver_and_lock_pick(_daily(), config, tmp_path, st, DATE, "test",
                                    selection=sel)
    assert ok is True
    d = load_decision(DATE, tmp_path)
    assert d["action"] == "single" and d["source"] == "mdp"
    assert d["delivery_status"] == "private_locked" and d["scoreable"] is True
    assert st.committed_pick_written is True


def test_crash_guard_writes_locked_unconfirmed_scoreable(tmp_path, monkeypatch):
    """delivery_attempted=True but undelivered → daemon crashed mid-send. Lock,
    DON'T re-send, and record a scoreable 'locked_unconfirmed' decision."""
    posted = []
    monkeypatch.setattr("bts.posting.post_to_bluesky", lambda *a, **k: posted.append(1) or "at://uri")
    st = _state()
    sel = _sel("single", "mdp", primary=_cand())
    config = {"scheduler": {"pick_delivery": "public"}}
    ok = sch._deliver_and_lock_pick(_daily(delivery_attempted=True), config, tmp_path,
                                    st, DATE, "test", selection=sel)
    assert ok is False
    assert posted == []  # must NOT re-post
    d = load_decision(DATE, tmp_path)
    assert d["delivery_status"] == "locked_unconfirmed" and d["scoreable"] is True
    assert st.committed_pick_written is True


def test_double_down_commit_records_action_double(tmp_path, monkeypatch):
    """The recorded action reflects daily.double_down even when selection is present."""
    monkeypatch.setattr(sch, "_trigger_live_forward_capture_on_lock", lambda *a, **k: None)
    dd = _pick(bid=2, name="Y", team="KC", gpk=200)
    sel = _sel("double", "mdp", primary=_cand(), double=_cand(bid=2))
    config = {"scheduler": {"pick_delivery": "private"}}
    sch._deliver_and_lock_pick(_daily(double_down=dd), config, tmp_path, _state(), DATE,
                               "test", selection=sel)
    d = load_decision(DATE, tmp_path)
    assert d["action"] == "double"
    assert d["double_down"]["batter_id"] == 2


# --- classification records (run_day chokepoint) --------------------------

def test_delivered_existing_classified_lock_writes_scoreable(tmp_path, monkeypatch):
    """A genuinely delivered existing pick recovered via classification-lock →
    scoreable record (delivery_status=delivered)."""
    delivered = _daily(delivered=True)
    _drive_run_day(
        monkeypatch, tmp_path,
        _result(pick_result=PickResult(daily=delivered, locked=True),
                pick_name="Hoerner", pick_p=0.73, selection=None),
    )
    d = load_decision(DATE, tmp_path)
    assert d is not None
    assert d["action"] == "single"
    assert d["delivery_status"] == "delivered"
    assert d["scoreable"] is True


def test_nondelivered_classified_lock_on_skip_day_writes_skip_not_pick(tmp_path, monkeypatch):
    """The GH #144 case: an MDP skip captured on cycle 1, then a NON-delivered
    stale-preview classified-lock on cycle 2. The classification must write NO
    pick record and must NOT clear the captured skip — end-of-day records the skip."""
    skip = _result(selection=_sel("skip", "mdp", primary=_cand(), streak=7, saver=False))
    classified = _result(pick_result=PickResult(daily=_daily(delivered=False), locked=True),
                         pick_name="Hoerner", pick_p=0.73, selection=None)
    _drive_run_day(
        monkeypatch, tmp_path, [skip, classified],
        games=[_game(100, "16:10"), _game(200, "19:05")],
    )
    d = load_decision(DATE, tmp_path)
    assert d is not None
    assert d["action"] == "skip"        # NOT a pick record
    assert d["source"] == "mdp"
    assert d["scoreable"] is False
    assert d["streak"] == 7


# --- result-polling commit gate (C2 / GH #144) ----------------------------

def test_nondelivered_classified_lock_on_skip_day_skips_result_polling(tmp_path, monkeypatch):
    """C2: a stale, NON-delivered <date>.json gets classification-locked
    (state.pick_locked=True) on an MDP-skip day with no scoreable decision record.
    The result-polling block must NOT run — otherwise run_result_polling would
    score the stale preview via update_streak and corrupt the streak, bypassing
    the check-results scoreable gate entirely."""
    from bts.picks import load_streak, save_pick, save_streak
    # The stale projected-preview pick on disk: undelivered, result=None. This is
    # what load_pick would hand to run_result_polling under the buggy gate.
    save_pick(_daily(delivered=False), tmp_path)
    save_streak(7, tmp_path)

    poll = Mock(return_value="final")
    skip = _result(selection=_sel("skip", "mdp", primary=_cand(), streak=7, saver=False))
    classified = _result(pick_result=PickResult(daily=_daily(delivered=False), locked=True),
                         pick_name="Hoerner", pick_p=0.73, selection=None)
    _drive_run_day(
        monkeypatch, tmp_path, [skip, classified],
        games=[_game(100, "16:10"), _game(200, "19:05")],
        poll_fn=poll,
    )
    poll.assert_not_called()              # polling gated off — update_streak unreachable
    assert load_streak(tmp_path) == 7     # streak untouched
    # And the day still records as the MDP skip (non-scoreable), not the stale pick.
    d = load_decision(DATE, tmp_path)
    assert d is not None and d["action"] == "skip" and d["scoreable"] is False


def test_delivered_pick_enters_result_polling(tmp_path, monkeypatch):
    """Positive companion: a genuinely delivered (scoreable) pick DOES enter result
    polling — the commit gate must not over-block real picks."""
    from bts.picks import save_pick
    # A delivered pick on disk with result still pending.
    save_pick(_daily(delivered=True), tmp_path)

    poll = Mock(return_value="final")
    classified = _result(pick_result=PickResult(daily=_daily(delivered=True), locked=True),
                         pick_name="Hoerner", pick_p=0.73, selection=None)
    _drive_run_day(monkeypatch, tmp_path, classified, poll_fn=poll)

    poll.assert_called_once()
    # Sanity: the classification recorded a scoreable delivered decision.
    d = load_decision(DATE, tmp_path)
    assert d is not None and d["scoreable"] is True


# --- end-of-day skip records ----------------------------------------------

def test_mdp_skip_day_writes_endofday_skip(tmp_path, monkeypatch):
    skip = _result(selection=_sel("skip", "mdp", primary=_cand(), streak=10, saver=True))
    _drive_run_day(monkeypatch, tmp_path, skip)
    d = load_decision(DATE, tmp_path)
    assert d is not None
    assert d["action"] == "skip" and d["source"] == "mdp"
    assert d["scoreable"] is False
    assert d["streak"] == 10
    assert d["saver_available"] is True
    assert d["primary"]["batter_id"] == 1


def test_heuristic_skip_day_writes_no_record(tmp_path, monkeypatch):
    """Only genuine MDP skips are captured; a heuristic skip writes nothing."""
    skip = _result(selection=_sel("skip", "heuristic", primary=_cand(), streak=3))
    _drive_run_day(monkeypatch, tmp_path, skip)
    assert load_decision(DATE, tmp_path) is None


def test_committed_pick_suppresses_endofday_skip(tmp_path, monkeypatch):
    """A pick committed after an earlier MDP-skip cycle suppresses the skip record
    (committed_pick_written), leaving the scoreable commit record as authoritative."""
    skip = _result(selection=_sel("skip", "mdp", primary=_cand(), streak=10, saver=True))
    commit = _result(should_post=True, pick_name="Hoerner", pick_p=0.73,
                     selection=_sel("single", "mdp", primary=_cand()),
                     pick_result=PickResult(daily=_daily(), locked=False))
    _drive_run_day(
        monkeypatch, tmp_path, [skip, commit],
        games=[_game(100, "16:10"), _game(200, "19:05")],
        pick_delivery="public", mock_post=True,
    )
    d = load_decision(DATE, tmp_path)
    assert d is not None
    assert d["action"] == "single"      # the commit, not the earlier skip
    assert d["scoreable"] is True
    assert d["delivery_status"] == "delivered"
