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
