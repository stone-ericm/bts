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
                   pick_delivery="private", mock_post=False, now_et="18:40",
                   poll_fn=None):
    """Run the REAL run_day to completion with run_single_check mocked.

    health_checks disabled and all sleeps/network neutralized so only the
    decision-record control flow under test runs. The clock sits past the
    18:20 lineup check but BEFORE the 19:00 submission cutoff of the 19:05
    game: the delivery guard (2026-08-30) refuses anything later.
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
    monkeypatch.setattr(sch, "_now_et", lambda: datetime(2026, 4, 6, 18, 40, tzinfo=ET))
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
    monkeypatch.setattr(sch, "_now_et", lambda: datetime(2026, 4, 6, 18, 40, tzinfo=ET))
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


def test_nondelivered_classified_lock_on_pick_day_fires_missed_pick_alert(tmp_path, monkeypatch):
    """The 2026-08-13 incident: a real pick candidate (no MDP skip standing) hit a
    NON-delivered classification lock. pick_locked=True suppressed the E3
    missed-pick alert, check-pick-entered keyed off committed_pick_written, and
    the day passed with zero notification. The E3 alert must fire whenever no
    scoreable commit was written and the on-disk pick is undelivered."""
    from bts.picks import save_pick

    alert = Mock()
    monkeypatch.setattr(sch, "_alert_missed_pick", alert)
    save_pick(_daily(delivered=False), tmp_path)  # the undelivered candidate on disk

    classified = _result(pick_result=PickResult(daily=_daily(delivered=False), locked=True),
                         pick_name="Hoerner", pick_p=0.73,
                         selection=_sel("single", "mdp", primary=_cand()))
    _drive_run_day(monkeypatch, tmp_path, classified)

    alert.assert_called_once()


def test_delivered_commit_does_not_fire_missed_pick_alert(tmp_path, monkeypatch):
    """A normally delivered pick (scoreable commit written) must not trip the
    re-gated E3 call site."""
    alert = Mock()
    monkeypatch.setattr(sch, "_alert_missed_pick", alert)

    sel = _sel("single", "mdp", primary=_cand())
    _drive_run_day(
        monkeypatch, tmp_path,
        _result(should_post=True, selection=sel, pick_name="Hoerner", pick_p=0.73,
                pick_result=PickResult(daily=_daily(), locked=False)),
        pick_delivery="public", mock_post=True,
    )
    alert.assert_not_called()


def test_scoreable_commit_short_circuits_cascade_despite_warmup(tmp_path, monkeypatch):
    """Committed means immutable (Codex r2 #2): a scoreable on-disk decision
    (e.g. private_locked, whose pick file carries no delivery flags) must lock
    the pre-cascade short-circuit even when the game status alone (Warmup)
    would classify the pick as refreshable — a same-day restart must not
    reselect over a committed pick."""
    from bts.daily_decision import write_decision
    from bts.picks import save_pick

    save_pick(_daily(delivered=False), tmp_path)
    write_decision(DATE, tmp_path, action="single", source="mdp",
                   primary=_cand(), delivery_status="private_locked", scoreable=True)
    monkeypatch.setattr("bts.picks.get_game_statuses_detailed",
                        lambda date: {100: {"abstract": "L", "detailed": "Warmup"}})
    monkeypatch.setattr("bts.orchestrator.run_and_pick",
                        Mock(side_effect=AssertionError("cascade must not run")))

    config = {"orchestrator": {"picks_dir": str(tmp_path)}}
    result = sch.run_single_check(date=DATE, all_game_pks=[100],
                                  confirmed_sides=set(), config=config,
                                  early_lock_gap=0.03)
    assert result["pick_result"] is not None
    assert result["pick_result"].locked is True


def test_delivery_attempted_classified_lock_finalizes_locked_unconfirmed(tmp_path, monkeypatch):
    """Codex r3 #1: a restart after a crash mid-send classifies the pick locked
    on its durable delivery_attempted marker BEFORE _deliver_and_lock_pick can
    run its unconfirmed-attempt branch. The classification chokepoint must
    finalize the day itself: scoreable locked_unconfirmed decision, E3
    suppressed, result polling enabled."""
    from bts.picks import save_pick

    alert = Mock()
    monkeypatch.setattr(sch, "_alert_missed_pick", alert)
    attempted = _daily(delivered=False, delivery_attempted=True)
    save_pick(attempted, tmp_path)
    classified = _result(pick_result=PickResult(daily=attempted, locked=True),
                         pick_name="Hoerner", pick_p=0.73, selection=None)
    poll = Mock(return_value="final")
    _drive_run_day(monkeypatch, tmp_path, classified, poll_fn=poll)

    d = load_decision(DATE, tmp_path)
    assert d is not None
    assert d["delivery_status"] == "locked_unconfirmed"
    assert d["scoreable"] is True
    alert.assert_not_called()
    poll.assert_called_once()


def test_classified_lock_does_not_clobber_existing_scoreable_decision(tmp_path, monkeypatch):
    """Codex r3 #3: a restart re-classifies a delivered pick as locked and hits
    the chokepoint again. It must not overwrite the authoritative decision
    (source, state provenance, finalized_at) with a source="unknown" record."""
    from bts.daily_decision import write_decision

    write_decision(DATE, tmp_path, action="single", source="mdp",
                   primary=_cand(), delivery_status="delivered", scoreable=True,
                   streak=7, saver_available=True)
    delivered = _daily(delivered=True)
    _drive_run_day(
        monkeypatch, tmp_path,
        _result(pick_result=PickResult(daily=delivered, locked=True),
                pick_name="Hoerner", pick_p=0.73, selection=None),
    )
    d = load_decision(DATE, tmp_path)
    assert d["source"] == "mdp"
    assert d["streak"] == 7


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


# --- fallback honors a genuine skip (C1 / Task D4) ------------------------
#
# These drive the REAL fallback control flow in run_day (run_single_check +
# _refresh_pick_at_fallback_decision mocked, everything between them under test).
# A "projected single" cycle (should_post=False, locked=False, pick_result.daily
# present) leaves the pick unlocked AND clears final_skip_candidate, so the in-loop
# fallback fires; with the default now_et=23:30 the post-loop fallback fires too.


def _projected_single():
    """A cycle that neither locks nor delivers but CLEARS final_skip_candidate
    (sel.action=='single') and exposes a pick_result.daily so the in-loop fallback
    deadline logic engages — the projected→real-flip setup."""
    return _result(should_post=False, pick_name="Hoerner", pick_p=0.73,
                   selection=_sel("single", "mdp", primary=_cand()),
                   pick_result=PickResult(daily=_daily(delivered=False), locked=False))


def test_refresh_carries_selection_on_genuine_skip(tmp_path, monkeypatch):
    """The function change: _refresh_pick_at_fallback_decision carries the
    SelectionResult on the no-fresh-pick path, so the caller can tell a genuine MDP
    skip (selection set) from a cascade/no-predictions (selection None)."""
    from contextlib import nullcontext
    skip_sel = _sel("skip", "mdp", primary=_cand(), streak=8, saver=True)  # pick_result=None
    monkeypatch.setattr(sch, "heartbeat_watchdog", lambda *a, **k: nullcontext())
    monkeypatch.setattr(sch, "run_and_pick", lambda config, date, **k: (None, skip_sel, "local"))
    cached = _daily(delivered=False)
    config = {"orchestrator": {"picks_dir": str(tmp_path), "heartbeat_path": str(tmp_path / ".hb")}}
    refresh = sch._refresh_pick_at_fallback_decision(config, DATE, cached, 0.03)
    assert refresh.daily is cached          # no fresh pick → fall back to cached
    assert refresh.should_post is None
    assert refresh.selection is skip_sel    # genuine skip carried through
    assert sch._refresh_is_genuine_skip(refresh) is True


def test_refresh_no_predictions_is_not_genuine_skip(tmp_path, monkeypatch):
    """Safety-net distinction: when run_and_pick yields no selection (no-predictions
    cascade), selection stays None and _refresh_is_genuine_skip is False — the
    fallback still delivers the cached pick."""
    from contextlib import nullcontext
    monkeypatch.setattr(sch, "heartbeat_watchdog", lambda *a, **k: nullcontext())
    monkeypatch.setattr(sch, "run_and_pick", lambda config, date, **k: (None, None, "local"))
    cached = _daily(delivered=False)
    config = {"orchestrator": {"picks_dir": str(tmp_path), "heartbeat_path": str(tmp_path / ".hb")}}
    refresh = sch._refresh_pick_at_fallback_decision(config, DATE, cached, 0.03)
    assert refresh.selection is None
    assert sch._refresh_is_genuine_skip(refresh) is False


def test_fallback_genuine_skip_recaptures_after_projected_pick_cleared(tmp_path, monkeypatch):
    """C1 + Codex r2: cycle 1 is a projected PICK (clears final_skip_candidate). The
    fallback then re-evaluates to a genuine MDP skip — it must NOT deliver the cached
    <date>.json, must RE-capture the skip candidate, and EOD must record the skip."""
    from bts.picks import save_pick, load_pick, pick_was_delivered
    save_pick(_daily(delivered=False), tmp_path)               # leftover projected preview
    monkeypatch.setattr(sch, "_maybe_alert_missed_pick", lambda *a, **k: None)
    skip_sel = _sel("skip", "mdp", primary=_cand(), streak=7, saver=False)
    monkeypatch.setattr(
        sch, "_refresh_pick_at_fallback_decision",
        lambda cfg, d, cached, gap: sch.FallbackRefreshResult(cached, None, selection=skip_sel),
    )
    _drive_run_day(monkeypatch, tmp_path, _projected_single(), pick_delivery="private")

    cached = load_pick(DATE, tmp_path)
    assert cached is not None and not pick_was_delivered(cached)   # never delivered
    d = load_decision(DATE, tmp_path)
    assert d is not None
    assert d["action"] == "skip" and d["source"] == "mdp"
    assert d["scoreable"] is False
    assert d["streak"] == 7                # re-captured from the fallback selection
    assert d["saver_available"] is False


def test_fallback_cascade_error_still_delivers_cached_safety_net(tmp_path, monkeypatch):
    """Regression guard: a fallback whose refresh returns selection=None (cascade /
    no-predictions) on a genuine PICK day (final_skip_candidate None) STILL delivers
    the cached pick. Only a genuine MDP skip suppresses delivery."""
    from bts.picks import save_pick
    save_pick(_daily(delivered=False), tmp_path)
    monkeypatch.setattr(sch, "_maybe_alert_missed_pick", lambda *a, **k: None)
    monkeypatch.setattr(
        sch, "_refresh_pick_at_fallback_decision",
        lambda cfg, d, cached, gap: sch.FallbackRefreshResult(cached, None),  # selection=None
    )
    _drive_run_day(monkeypatch, tmp_path, _projected_single(), pick_delivery="private")

    d = load_decision(DATE, tmp_path)
    assert d is not None
    assert d["action"] == "single"                 # cached pick committed (safety net intact)
    assert d["scoreable"] is True
    assert d["delivery_status"] == "private_locked"


def test_standing_skip_survives_post_loop_refresh_flake(tmp_path, monkeypatch):
    """Codex r3: the in-loop fallback confirms a genuine skip and continues; the
    post-loop fallback's refresh then FLAKES to selection=None. The standing skip
    (final_skip_candidate set, no commit) must NOT be resurrected into a delivered
    cached pick — EOD still records the skip captured in-loop."""
    from bts.picks import save_pick, load_pick, pick_was_delivered
    save_pick(_daily(delivered=False), tmp_path)
    monkeypatch.setattr(sch, "_maybe_alert_missed_pick", lambda *a, **k: None)
    skip_sel = _sel("skip", "mdp", primary=_cand(), streak=9, saver=True)
    # 1st call (in-loop) → genuine skip; 2nd call (post-loop) → flake (selection=None).
    refresh_mock = Mock(side_effect=[
        sch.FallbackRefreshResult(_daily(delivered=False), None, selection=skip_sel),
        sch.FallbackRefreshResult(_daily(delivered=False), None),
    ])
    monkeypatch.setattr(sch, "_refresh_pick_at_fallback_decision", refresh_mock)
    _drive_run_day(monkeypatch, tmp_path, _projected_single(), pick_delivery="private")

    assert refresh_mock.call_count == 2            # in-loop fired, then post-loop fired
    cached = load_pick(DATE, tmp_path)
    assert cached is not None and not pick_was_delivered(cached)   # flake did NOT resurrect it
    d = load_decision(DATE, tmp_path)
    assert d is not None
    assert d["action"] == "skip" and d["scoreable"] is False
    assert d["streak"] == 9                        # candidate captured in-loop, not lost


# --- standing skip must YIELD to a genuine late pick (post-review Fix 1) ------
#
# The inverse of the r3 case above: an earlier cycle decided an MDP skip
# (final_skip_candidate set), then the fallback refresh resolves to a GENUINE
# single/double (lineups flipped back). The standing skip must be CLEARED and the
# real pick delivered — NOT suppressed into an end-of-day skip.


def _deliver_spy(monkeypatch):
    """Wrap _deliver_and_lock_pick so a test can assert it actually ran."""
    calls = []
    real = sch._deliver_and_lock_pick
    monkeypatch.setattr(sch, "_deliver_and_lock_pick",
                        lambda *a, **k: calls.append(1) or real(*a, **k))
    return calls


def test_standing_skip_yields_to_late_pick_post_loop(tmp_path, monkeypatch):
    """Fix 1 (post-loop site, ~2179): a genuine MDP skip stands going into the
    post-loop fallback; the refresh then returns a genuine single. The standing skip
    must clear — _deliver_and_lock_pick runs and EOD records the PICK, not a skip."""
    from bts.picks import save_pick
    save_pick(_daily(delivered=False), tmp_path)          # leftover preview the fallback re-evaluates
    monkeypatch.setattr(sch, "_maybe_alert_missed_pick", lambda *a, **k: None)
    delivered = _deliver_spy(monkeypatch)
    pick_sel = _sel("single", "mdp", primary=_cand())
    monkeypatch.setattr(
        sch, "_refresh_pick_at_fallback_decision",
        lambda cfg, d, cached, gap: sch.FallbackRefreshResult(cached, True, selection=pick_sel),
    )
    # The day's only cycle is a genuine MDP skip → final_skip_candidate set, unlocked.
    skip = _result(selection=_sel("skip", "mdp", primary=_cand(), streak=7, saver=False))
    _drive_run_day(monkeypatch, tmp_path, skip, pick_delivery="private")

    assert delivered, "_deliver_and_lock_pick must run — the standing skip yields to a real pick"
    d = load_decision(DATE, tmp_path)
    assert d is not None
    assert d["action"] == "single"                        # the late pick, NOT an EOD skip
    assert d["source"] == "mdp"
    assert d["scoreable"] is True
    assert d["delivery_status"] == "private_locked"


def test_standing_skip_yields_to_late_pick_in_loop(tmp_path, monkeypatch):
    """Fix 1 (in-loop site, ~2098): cycle 1 decides an MDP skip (final_skip_candidate
    set); cycle 2 has selection=None (so it neither sets nor clears the candidate) but
    exposes a pick_result.daily that engages the in-loop fallback, whose refresh
    returns a genuine single. The standing skip must yield and deliver in-loop."""
    from bts.picks import save_pick
    save_pick(_daily(delivered=False), tmp_path)
    monkeypatch.setattr(sch, "_maybe_alert_missed_pick", lambda *a, **k: None)
    delivered = _deliver_spy(monkeypatch)
    pick_sel = _sel("single", "mdp", primary=_cand())
    monkeypatch.setattr(
        sch, "_refresh_pick_at_fallback_decision",
        lambda cfg, d, cached, gap: sch.FallbackRefreshResult(cached, True, selection=pick_sel),
    )
    skip = _result(selection=_sel("skip", "mdp", primary=_cand(), streak=7, saver=False))
    # selection=None → candidate survives into the in-loop fallback; pick_result.daily
    # present on the LAST run → the in-loop fallback deadline branch engages.
    unknown = _result(should_post=False, selection=None,
                      pick_result=PickResult(daily=_daily(delivered=False), locked=False))
    _drive_run_day(
        monkeypatch, tmp_path, [skip, unknown],
        games=[_game(100, "16:10"), _game(200, "19:05")], pick_delivery="private",
    )

    assert delivered, "_deliver_and_lock_pick must run in-loop — the standing skip yields"
    d = load_decision(DATE, tmp_path)
    assert d is not None
    assert d["action"] == "single" and d["source"] == "mdp"
    assert d["scoreable"] is True
    assert d["delivery_status"] == "private_locked"


def test_skip_to_pick_flip_clears_skip_summary_unsuppressing_missed_alert(tmp_path, monkeypatch):
    """Fix 2: an early skip sets skip_summary; a later pick cycle (sel.action single)
    must clear it in the run_day result block alongside final_skip_candidate. When
    that pick then fails to deliver, the missed-pick alert (D6) must FIRE — a stale
    skip_summary must not suppress a genuine missed-pick warning."""
    from bts.picks import save_pick
    from bts.contest_state import ContestStateError
    save_pick(_daily(delivered=False), tmp_path)          # the undelivered pick to alert on
    fired = []
    monkeypatch.setattr(sch, "_alert_missed_pick", lambda *a, **k: fired.append(1))
    monkeypatch.setattr(sch, "_watchdog_ping_sleep", lambda *a, **k: None)
    monkeypatch.setattr(sch, "_alert_contest_state_failure", lambda *a, **k: None)
    # Post-loop fallback refresh is blocked → the pick stays undelivered/unlocked, so
    # run_day reaches the missed-pick alert with a genuine undelivered pick on disk.
    monkeypatch.setattr(sch, "_refresh_pick_at_fallback_decision",
                        Mock(side_effect=ContestStateError("blocked")))
    skip = _result(selection=_sel("skip", "mdp", primary=_cand(), streak=5, saver=False),
                   skip_summary={"best_batter": "X", "best_team": "NYM", "best_p": 0.70, "streak": 5})
    # sel.action single clears final_skip_candidate AND (Fix 2) skip_summary; pick_result
    # None → no in-loop fallback and no delivery in the result block here.
    flip = _result(should_post=False, selection=_sel("single", "mdp", primary=_cand()),
                   pick_result=None)
    _drive_run_day(
        monkeypatch, tmp_path, [skip, flip],
        games=[_game(100, "16:10"), _game(200, "19:05")], pick_delivery="private",
    )
    assert fired == [1], "missed-pick alert must fire once skip_summary is cleared on the flip"
