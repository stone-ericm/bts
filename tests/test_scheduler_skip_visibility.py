"""Skip-day visibility: the scheduler must surface a skip (log + DM + dashboard)
instead of returning silently when the MDP declines to pick (best candidate
below the pick bar). Regression guard for the 2026-06-18 "why no suggestion?"
incident, where a legitimate skip looked indistinguishable from a hang."""
import json
from datetime import datetime

import pandas as pd

from bts.scheduler import build_skip_summary, maybe_notify_skip, SchedulerState


def _state(**kw):
    base = dict(
        date="2026-06-18", schedule_fetched_at="t", games=[], confirmed_game_pks=[],
        runs_completed=[], pick_locked=False, pick_locked_at=None,
        result_status=None, next_wakeup=None,
    )
    base.update(kw)
    return SchedulerState(**base)


_DM_CONFIG = {"bluesky": {"dm_recipient": "eric.bsky.social"},
              "scheduler": {"pick_delivery": "dm"}}
_SUMMARY = {"best_batter": "Bo Bichette", "best_team": "NYM", "best_p": 0.75, "streak": 10}


def test_build_skip_summary_reports_top_candidate_and_streak():
    preds = pd.DataFrame([
        {"batter_name": "Bo Bichette", "team": "NYM", "p_game_hit": 0.75, "game_pk": 1},
        {"batter_name": "Bobby Witt Jr.", "team": "KC", "p_game_hit": 0.742, "game_pk": 2},
    ])
    s = build_skip_summary(preds, streak=10)
    assert s["best_batter"] == "Bo Bichette"
    assert s["best_team"] == "NYM"
    assert round(s["best_p"], 3) == 0.75
    assert s["streak"] == 10


def test_build_skip_summary_uses_max_p_even_if_unsorted():
    preds = pd.DataFrame([
        {"batter_name": "Low", "team": "AAA", "p_game_hit": 0.60, "game_pk": 1},
        {"batter_name": "High", "team": "BBB", "p_game_hit": 0.78, "game_pk": 2},
    ])
    s = build_skip_summary(preds, streak=5)
    assert s["best_batter"] == "High"
    assert round(s["best_p"], 2) == 0.78


def test_skip_dm_sent_once_then_suppressed_next_cycle():
    sent = []
    state = _state()
    fired1 = maybe_notify_skip(state, _SUMMARY, _DM_CONFIG,
                               now_iso="2026-06-18T18:00", send=lambda r, t: sent.append((r, t)))
    assert fired1 is True
    assert len(sent) == 1
    assert sent[0][0] == "eric.bsky.social"
    assert "Bo Bichette" in sent[0][1] and "75" in sent[0][1]
    assert state.skip_notified_at == "2026-06-18T18:00"

    # next 10-min cycle: marker already set -> no duplicate DM
    fired2 = maybe_notify_skip(state, _SUMMARY, _DM_CONFIG,
                               now_iso="2026-06-18T18:10", send=lambda r, t: sent.append((r, t)))
    assert fired2 is False
    assert len(sent) == 1


def test_skip_no_dm_when_delivery_mode_not_dm():
    sent = []
    state = _state()
    cfg = {"bluesky": {"dm_recipient": "x"}, "scheduler": {"pick_delivery": "private"}}
    fired = maybe_notify_skip(state, _SUMMARY, cfg, now_iso="t",
                              send=lambda r, t: sent.append(1))
    assert fired is False
    assert sent == []
    assert state.skip_notified_at is None


def test_scheduler_state_loads_without_skip_fields():
    # An old scheduler_state.json (pre-skip-visibility) must still construct,
    # since load_state does SchedulerState(**data).
    old = dict(date="2026-06-18", schedule_fetched_at="t", games=[], confirmed_game_pks=[],
               runs_completed=[], pick_locked=False, pick_locked_at=None,
               result_status=None, next_wakeup=None, analytics_jobs=None)
    st = SchedulerState(**old)
    assert st.skip_summary is None
    assert st.skip_notified_at is None


def test_render_skip_banner_shows_best_and_streak():
    from bts.web import render_skip_banner
    html = render_skip_banner(
        {"best_batter": "Bo Bichette", "best_team": "NYM", "best_p": 0.75, "streak": 10})
    assert "SKIP" in html.upper()
    assert "Bo Bichette" in html
    assert "75" in html        # 0.75 -> 75%
    assert "10" in html        # streak


def test_render_skip_banner_empty_when_no_summary():
    from bts.web import render_skip_banner
    assert render_skip_banner(None) == ""
    assert render_skip_banner({}) == ""


def test_build_skip_summary_tolerates_unknown_streak():
    preds = pd.DataFrame([{"batter_name": "X", "team": "AAA", "p_game_hit": 0.7, "game_pk": 1}])
    s = build_skip_summary(preds, streak=None)
    assert s["streak"] is None
    assert s["best_batter"] == "X"


def test_skip_dm_failure_does_not_mark_notified_so_it_retries():
    def boom(r, t):
        raise RuntimeError("bluesky down")
    state = _state()
    fired = maybe_notify_skip(state, _SUMMARY, _DM_CONFIG, now_iso="t", send=boom)
    assert fired is False
    assert state.skip_notified_at is None  # unmarked -> a later cycle retries


def test_render_page_shows_skip_banner_on_skip_day(tmp_path, monkeypatch):
    """End-to-end: the dashboard page renders a skip banner when scheduler_state
    has a skip_summary (and no f-string break from the {skip_banner} injection)."""
    import bts.web as web
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = tmp_path / today
    day_dir.mkdir(parents=True)
    (day_dir / "scheduler_state.json").write_text(json.dumps({
        "date": today, "pick_locked": False,
        "skip_summary": {"best_batter": "Bo Bichette", "best_team": "NYM",
                         "best_p": 0.75, "streak": 10},
    }))
    monkeypatch.setattr(web, "PICKS_DIR", tmp_path)
    monkeypatch.setattr(web, "fetch_bluesky_posts", lambda *a, **k: [])

    html = web.render_page()

    assert "SKIP TODAY" in html
    assert "Bo Bichette" in html
    assert "75%" in html


def test_run_single_check_surfaces_skip_instead_of_silent_none(tmp_path, monkeypatch):
    """The incident's core regression: when predictions succeed but the policy
    declines (run_and_pick -> pick_result None), run_single_check must return a
    skip_summary, not the old silent {pick_name: None} with no signal."""
    from contextlib import nullcontext
    import bts.scheduler as sch
    import bts.orchestrator as orch
    preds = pd.DataFrame([
        {"batter_name": "Bo Bichette", "team": "NYM", "p_game_hit": 0.75, "game_pk": 1},
        {"batter_name": "Bobby Witt Jr.", "team": "KC", "p_game_hit": 0.742, "game_pk": 2},
    ])
    monkeypatch.setattr(orch, "run_and_pick", lambda config, date, **k: (preds, None, "local"))
    monkeypatch.setattr(sch, "count_new_confirmations", lambda *a, **k: 0)
    monkeypatch.setattr(sch, "heartbeat_watchdog", lambda *a, **k: nullcontext())
    config = {
        "orchestrator": {"picks_dir": str(tmp_path), "heartbeat_path": str(tmp_path / ".hb")},
        "scheduler": {"heartbeat_stall_after_sec": 900},
        "tiers": [],
    }
    result = sch.run_single_check(date="2026-06-18", all_game_pks=[1, 2],
                                  confirmed_sides=set(), config=config, early_lock_gap=0.03)
    assert result["pick_name"] is None
    assert result["skip_summary"]["best_batter"] == "Bo Bichette"
    assert round(result["skip_summary"]["best_p"], 2) == 0.75


def test_skip_notification_survives_scheduler_restart():
    """run_day rebuilds state fresh each startup; the once-per-day skip DM must
    NOT re-fire on restart, so skip fields carry from the prior persisted state."""
    from bts.scheduler import carry_forward_skip_state
    prev = _state(skip_notified_at="2026-06-18T18:00",
                  skip_summary={"best_batter": "Bo Bichette", "best_team": "NYM",
                                "best_p": 0.75, "streak": 10})
    fresh = _state()  # restart: skip fields default to None
    carry_forward_skip_state(fresh, prev)
    assert fresh.skip_notified_at == "2026-06-18T18:00"
    assert fresh.skip_summary["best_batter"] == "Bo Bichette"


def test_carry_forward_skip_state_first_run_of_day():
    from bts.scheduler import carry_forward_skip_state
    fresh = _state()
    carry_forward_skip_state(fresh, None)  # no prior state today
    assert fresh.skip_notified_at is None
    assert fresh.skip_summary is None


def test_build_skip_summary_returns_none_on_all_nan_probs():
    preds = pd.DataFrame([
        {"batter_name": "A", "team": "X", "p_game_hit": float("nan"), "game_pk": 1},
        {"batter_name": "B", "team": "Y", "p_game_hit": float("nan"), "game_pk": 2},
    ])
    assert build_skip_summary(preds, streak=10) is None  # never crash the daemon loop


def test_build_skip_summary_returns_none_on_missing_column():
    preds = pd.DataFrame([{"batter_name": "A", "team": "X", "game_pk": 1}])  # no p_game_hit
    assert build_skip_summary(preds, streak=10) is None


def test_build_skip_summary_returns_json_native_types():
    preds = pd.DataFrame([{"batter_name": "A", "team": "X", "p_game_hit": 0.73, "game_pk": 1}])
    s = build_skip_summary(preds, streak=10)
    assert type(s["best_p"]) is float and type(s["best_batter"]) is str


def test_render_skip_banner_handles_nan_prob():
    from bts.web import render_skip_banner
    html = render_skip_banner({"best_batter": "A", "best_team": "X",
                               "best_p": float("nan"), "streak": 10})
    assert "nan" not in html.lower()  # no "nan%" leaking to the dashboard


def test_skip_dm_is_tentative_not_a_finality_claim():
    from bts.scheduler import format_skip_dm
    text = format_skip_dm("2026-06-18", _SUMMARY)
    assert "Bo Bichette" in text and "75" in text
    low = text.lower()
    assert "no pick today" not in low          # an early cycle may still flip to a pick
    assert "will pick" in low or "yet" in low   # tentative, not a finality claim


def test_carry_forward_skip_state_ignores_different_date():
    from bts.scheduler import carry_forward_skip_state
    prev = _state(date="2026-06-17", skip_notified_at="x", skip_summary={"best_batter": "Y"})
    fresh = _state(date="2026-06-18")
    carry_forward_skip_state(fresh, prev)
    assert fresh.skip_notified_at is None
    assert fresh.skip_summary is None


# --- final_skip_candidate / committed_pick_written carry-forward (#3) -------
# The finalization state now lives on SchedulerState (persisted to
# scheduler_state.json) so a same-day daemon restart (deploys; Restart=always)
# after an MDP skip but before end-of-day no longer loses the captured skip.

_SKIP_CAND = {"primary": {"batter_id": 1, "batter_name": "X", "team": "NYM",
                          "game_pk": 9, "p_game_hit": 0.7},
              "streak": 10, "saver_available": True}


def test_carry_forward_final_skip_candidate_same_date():
    """A rebuilt same-date SchedulerState inherits the captured MDP-skip candidate
    (and the committed flag) so the day's finalization survives a restart."""
    from bts.scheduler import carry_forward_skip_state
    prev = _state(final_skip_candidate=_SKIP_CAND)
    fresh = _state()  # restart: finalization fields default to None/False
    carry_forward_skip_state(fresh, prev)
    assert fresh.final_skip_candidate == _SKIP_CAND
    assert fresh.committed_pick_written is False


def test_carry_forward_final_skip_candidate_ignores_different_date():
    from bts.scheduler import carry_forward_skip_state
    prev = _state(date="2026-06-17", final_skip_candidate=_SKIP_CAND,
                  committed_pick_written=True)
    fresh = _state(date="2026-06-18")
    carry_forward_skip_state(fresh, prev)
    assert fresh.final_skip_candidate is None
    assert fresh.committed_pick_written is False


def test_carried_forward_skip_candidate_still_writes_endofday_skip(tmp_path):
    """#3 end-to-end: a same-day restart rebuilds state fresh; the carried-forward
    candidate must still let _write_endofday_skip record the MDP skip."""
    from bts.scheduler import carry_forward_skip_state, _write_endofday_skip
    from bts.daily_decision import load_decision
    prev = _state(final_skip_candidate=_SKIP_CAND)   # the skip cycle before the restart
    fresh = _state()
    carry_forward_skip_state(fresh, prev)
    _write_endofday_skip(tmp_path, fresh.date, fresh)
    d = load_decision(fresh.date, tmp_path)
    assert d is not None and d["action"] == "skip" and d["source"] == "mdp"
    assert d["scoreable"] is False and d["streak"] == 10


def test_carry_forward_committed_pick_written_suppresses_endofday_skip(tmp_path):
    """A committed pick before a same-day restart carries committed_pick_written
    forward so the rebuilt state does NOT also write an end-of-day skip."""
    from bts.scheduler import carry_forward_skip_state, _write_endofday_skip
    from bts.daily_decision import load_decision
    prev = _state(final_skip_candidate=_SKIP_CAND, committed_pick_written=True)
    fresh = _state()
    carry_forward_skip_state(fresh, prev)
    assert fresh.committed_pick_written is True
    _write_endofday_skip(tmp_path, fresh.date, fresh)
    assert load_decision(fresh.date, tmp_path) is None


# --- flip-day banner gating (2026-07-01): suppression keys on the committed-pick
# --- predicate (decision.json / delivered), NOT on pick-file existence — a stale
# --- provisional file from the projected->real-skip flip must not hide the banner.

def _pick_payload(today, *, delivered):
    return {
        "date": today, "run_time": f"{today}T18:18:45+00:00",
        "pick": {"batter_name": "Luis Arraez", "batter_id": 650333, "team": "SF",
                 "lineup_position": 1, "pitcher_name": "Zac Gallen", "pitcher_id": 668678,
                 "p_game_hit": 0.8097, "flags": ["PROJECTED lineup"], "projected_lineup": True,
                 "game_pk": 825064, "game_time": "2026-07-02T01:40:00Z", "pitcher_team": "AZ"},
        "double_down": None, "runner_up": None,
        "bluesky_posted": False, "bluesky_uri": None,
        "notification_sent": delivered,
        "notification_channel": "dm" if delivered else None,
        "notification_id": "3abc" if delivered else None,
        "delivery_attempted": delivered, "result": None, "slot_results": None,
    }


def _flip_day_page(tmp_path, monkeypatch, *, delivered, decision=None):
    import bts.scorecard as scorecard
    import bts.web as web
    today = datetime.now().strftime("%Y-%m-%d")
    day_dir = tmp_path / today
    day_dir.mkdir(parents=True)
    (day_dir / "scheduler_state.json").write_text(json.dumps({
        "date": today, "pick_locked": False,
        "skip_summary": {"best_batter": "Luis Arraez", "best_team": "SF",
                         "best_p": 0.8097, "streak": 17},
    }))
    (tmp_path / f"{today}.json").write_text(json.dumps(_pick_payload(today, delivered=delivered)))
    if decision is not None:
        (day_dir / "decision.json").write_text(json.dumps({
            "schema_version": "bts_daily_decision_v1", "date": today,
            "finalized_at": f"{today}T22:05:00+00:00", **decision}))
    monkeypatch.setattr(web, "PICKS_DIR", tmp_path)
    monkeypatch.setattr(web, "fetch_bluesky_posts", lambda *a, **k: [])
    monkeypatch.setattr(scorecard, "fetch_live_scorecard", lambda *a, **k: None)
    return web.render_page()


def test_banner_shows_over_stale_provisional_pick_file(tmp_path, monkeypatch):
    """The flip day: an undelivered provisional pick file must NOT suppress the banner."""
    html = _flip_day_page(tmp_path, monkeypatch, delivered=False)
    assert "SKIP TODAY" in html


def test_banner_suppressed_when_pick_actually_delivered(tmp_path, monkeypatch):
    html = _flip_day_page(tmp_path, monkeypatch, delivered=True)
    assert "SKIP TODAY" not in html


def test_banner_shows_when_decision_says_skip(tmp_path, monkeypatch):
    html = _flip_day_page(tmp_path, monkeypatch, delivered=False,
                          decision={"action": "skip", "source": "mdp", "scoreable": False,
                                    "delivery_status": "not_applicable"})
    assert "SKIP TODAY" in html


def test_banner_suppressed_when_decision_scoreable(tmp_path, monkeypatch):
    """decision.json is authoritative: scoreable commit suppresses even if the
    pick file's own delivery flags lag."""
    html = _flip_day_page(tmp_path, monkeypatch, delivered=False,
                          decision={"action": "single", "source": "mdp", "scoreable": True,
                                    "delivery_status": "delivered"})
    assert "SKIP TODAY" not in html


# --- streak-dependent pick bar surfaced in skip messages (2026-07-01) ---

def test_build_skip_summary_carries_pick_bar():
    preds = pd.DataFrame([{"batter_name": "X", "team": "SF", "p_game_hit": 0.8097, "game_pk": 1}])
    s = build_skip_summary(preds, 17, pick_bar=0.8115)
    assert s["bar"] == 0.8115
    s2 = build_skip_summary(preds, 17)
    assert "bar" not in s2


def test_render_skip_banner_shows_streak_bar_when_present():
    from bts.web import render_skip_banner
    html = render_skip_banner({"best_batter": "Luis Arraez", "best_team": "SF",
                               "best_p": 0.8097, "streak": 17, "bar": 0.8115})
    assert "streak-17 bar" in html
    assert "81.2%" in html          # the bar, 0.1% precision
    assert "81.0%" in html          # best_p at matching precision


def test_render_skip_banner_legacy_summary_keeps_old_wording():
    from bts.web import render_skip_banner
    html = render_skip_banner({"best_batter": "Bo Bichette", "best_team": "NYM",
                               "best_p": 0.75, "streak": 10})
    assert "below the" in html and "pick bar" in html
    assert "75%" in html


def test_format_skip_dm_shows_streak_bar_when_present():
    from bts.scheduler import format_skip_dm
    msg = format_skip_dm("2026-07-01", {"best_batter": "Luis Arraez", "best_team": "SF",
                                        "best_p": 0.8097, "streak": 17, "bar": 0.8115})
    assert "streak-17 bar" in msg and "81.2%" in msg and "81.0%" in msg
    assert "~80%" not in msg


def test_format_skip_dm_legacy_summary_keeps_old_wording():
    from bts.scheduler import format_skip_dm
    msg = format_skip_dm("2026-06-18", {"best_batter": "Bo Bichette", "best_team": "NYM",
                                        "best_p": 0.75, "streak": 10})
    assert "~80% bar" in msg


# --- skip-day rendering completeness (2026-07-02): history rows, hero card,
# --- "Waiting for lineups" placeholder, and the live-game section must all
# --- respect the decision record / standing skip, not pick-file existence.

def _render_skip_page(tmp_path, monkeypatch, *, pick_date=None, write_pick=True,
                      delivered=False, decision=None, skip_summary=True,
                      poison_scorecard=False):
    import bts.scorecard as scorecard
    import bts.web as web
    today = datetime.now().strftime("%Y-%m-%d")
    pick_date = pick_date or today
    if skip_summary:
        day_dir = tmp_path / today
        day_dir.mkdir(parents=True, exist_ok=True)
        (day_dir / "scheduler_state.json").write_text(json.dumps({
            "date": today, "pick_locked": False,
            "skip_summary": {"best_batter": "Luis Arraez", "best_team": "SF",
                             "best_p": 0.8097, "streak": 17, "bar": 0.8115},
        }))
    if write_pick:
        (tmp_path / f"{pick_date}.json").write_text(
            json.dumps(_pick_payload(pick_date, delivered=delivered)))
    if decision is not None:
        d_dir = tmp_path / pick_date
        d_dir.mkdir(parents=True, exist_ok=True)
        (d_dir / "decision.json").write_text(json.dumps({
            "schema_version": "bts_daily_decision_v1", "date": pick_date,
            "finalized_at": f"{pick_date}T22:05:00+00:00", **decision}))
    monkeypatch.setattr(web, "PICKS_DIR", tmp_path)
    monkeypatch.setattr(web, "fetch_bluesky_posts", lambda *a, **k: [])
    if poison_scorecard:
        def _boom(*a, **k):
            raise AssertionError("live scorecard must not be fetched on a standing-skip day")
        monkeypatch.setattr(scorecard, "fetch_live_scorecard", _boom)
    else:
        monkeypatch.setattr(scorecard, "fetch_live_scorecard", lambda *a, **k: None)
    return web.render_page()


def test_skip_day_no_waiting_placeholder(tmp_path, monkeypatch):
    html = _render_skip_page(tmp_path, monkeypatch, write_pick=False)
    assert "SKIP TODAY" in html
    assert "Waiting for lineups" not in html


def test_waiting_placeholder_still_shows_before_any_skip(tmp_path, monkeypatch):
    html = _render_skip_page(tmp_path, monkeypatch, write_pick=False, skip_summary=False)
    assert "Waiting for lineups" in html
    assert "SKIP TODAY" not in html


def test_flip_day_hero_card_suppressed(tmp_path, monkeypatch):
    """Standing skip + stale provisional today-file: no hero pick card (the banner
    is the day's status), no live-game fetch."""
    html = _render_skip_page(tmp_path, monkeypatch, write_pick=True, poison_scorecard=True)
    assert "SKIP TODAY" in html
    assert 'class="hero-pct"' not in html   # rendered card, not the CSS rule
    assert "Waiting for lineups" not in html


def test_delivered_day_hero_unchanged(tmp_path, monkeypatch):
    html = _render_skip_page(tmp_path, monkeypatch, write_pick=True, delivered=True,
                             skip_summary=False)
    assert 'class="hero-pct"' in html
    assert "SKIP TODAY" not in html


def test_history_row_shows_skip_not_pending_for_finalized_skip(tmp_path, monkeypatch):
    """Yesterday's never-delivered flip-day file (decision=skip) must not render as
    an eternally-pending pick in the history table."""
    from datetime import timedelta
    yday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    html = _render_skip_page(tmp_path, monkeypatch, pick_date=yday, write_pick=True,
                             skip_summary=False,
                             decision={"action": "skip", "source": "mdp",
                                       "scoreable": False,
                                       "delivery_status": "not_applicable"})
    assert 'class="result-skip"' in html
    assert 'class="result-pending"' not in html


def test_history_row_pending_when_no_decision_yet(tmp_path, monkeypatch):
    """A provisional file with NO decision record keeps the pending dash (mid-day
    normal state) — the SKIP marker only appears once the day finalized as a skip."""
    from datetime import timedelta
    yday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    html = _render_skip_page(tmp_path, monkeypatch, pick_date=yday, write_pick=True,
                             skip_summary=False, decision=None)
    assert 'class="result-pending"' in html
    assert 'class="result-skip"' not in html
