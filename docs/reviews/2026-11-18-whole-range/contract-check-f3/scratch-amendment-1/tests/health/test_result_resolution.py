"""Tests for the result_resolution health source (shadow + production sides).

Backstop for the 2026-07-10 stranded-shadow incident class: the check-results
wait loop settles same-night cases; this check makes the residual stranded
ones visible instead of silent — for BOTH the shadow file and a scoreable
production pick left unresolved past the wait deadline (nothing else revisits
either: nightly crons grade yesterday only, and reconcile skips nonterminal
picks). Codex r2: any-version shadow scan inside a horizon (a version bump
must not hide stranded older-version results), and per-side incident keys.
"""

from datetime import date, timedelta

from bts.health import result_resolution as rr
from bts.picks import Pick, DailyPick, save_pick, save_shadow_pick
from bts.shadow_eval import stamp_shadow_version

TODAY = date(2026, 8, 9)


def _daily(day, *, result=None, delivered=True):
    daily = DailyPick(
        date=day.isoformat(),
        run_time=f"{day.isoformat()}T15:00:00+00:00",
        pick=Pick(
            batter_name="Batter", batter_id=1, team="ATH", lineup_position=1,
            pitcher_name="P", pitcher_id=2, p_game_hit=0.8, flags=[],
            projected_lineup=False, game_pk=1000,
            game_time=f"{day.isoformat()}T23:10:00Z",
        ),
        double_down=None, runner_up=None,
        bluesky_posted=delivered, bluesky_uri="at://x" if delivered else None,
    )
    daily.result = result
    return daily


def _shadow(picks_dir, day, *, result=None, stamped=True):
    picks_dir.mkdir(parents=True, exist_ok=True)
    daily = _daily(day, result=result, delivered=False)
    if stamped:
        daily = stamp_shadow_version(daily)
    save_shadow_pick(daily, picks_dir)


def _production(picks_dir, day, *, result=None, delivered=True):
    picks_dir.mkdir(parents=True, exist_ok=True)
    save_pick(_daily(day, result=result, delivered=delivered), picks_dir)


# --- shadow side ---

def test_no_files_silent(tmp_path):
    assert rr.check(tmp_path / "picks", today=TODAY) == []


def test_yesterday_unresolved_shadow_within_grace(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=1))
    assert rr.check(tmp_path / "picks", today=TODAY) == []


def test_shadow_warns_exactly_at_grace_boundary(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=2))
    alerts = rr.check(tmp_path / "picks", today=TODAY)
    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert alerts[0].source == "result_resolution"
    assert (TODAY - timedelta(days=2)).isoformat() in alerts[0].message


def test_shadow_critical_exactly_at_critical_boundary(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=7))
    alerts = rr.check(tmp_path / "picks", today=TODAY)
    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"


def test_resolved_shadow_quiet(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=5), result="hit")
    assert rr.check(tmp_path / "picks", today=TODAY) == []


def test_legacy_v1_unstamped_shadow_ALERTS_within_horizon(tmp_path):
    """Codex r2 #5: operational monitoring must be version-blind — a v3 bump
    must not silently hide still-stranded v2 (or v1) results."""
    _shadow(tmp_path / "picks", TODAY - timedelta(days=5), stamped=False)
    alerts = rr.check(tmp_path / "picks", today=TODAY)
    assert len(alerts) == 1
    assert alerts[0].level == "WARN"


def test_ancient_unresolved_shadow_outside_horizon_silent(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=40))
    assert rr.check(tmp_path / "picks", today=TODAY) == []


def test_each_overdue_date_gets_own_incident_key(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=3))
    _shadow(tmp_path / "picks", TODAY - timedelta(days=4))
    alerts = rr.check(tmp_path / "picks", today=TODAY)
    assert len(alerts) == 2
    assert len({a.incident_key for a in alerts}) == 2


# --- production side ---

def test_stranded_scoreable_production_warns(tmp_path):
    _production(tmp_path / "picks", TODAY - timedelta(days=3))
    alerts = rr.check(tmp_path / "picks", today=TODAY)
    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert "production" in alerts[0].incident_key


def test_undelivered_production_unscoreable_silent(tmp_path):
    """A stale preview / undelivered pick file (GH #144 territory) is not a
    stranded result — nothing was ever going to score it."""
    _production(tmp_path / "picks", TODAY - timedelta(days=3), delivered=False)
    assert rr.check(tmp_path / "picks", today=TODAY) == []


def test_resolved_production_quiet(tmp_path):
    _production(tmp_path / "picks", TODAY - timedelta(days=3), result="miss")
    assert rr.check(tmp_path / "picks", today=TODAY) == []


def test_both_sides_same_date_two_alerts(tmp_path):
    day = TODAY - timedelta(days=3)
    _production(tmp_path / "picks", day)
    _shadow(tmp_path / "picks", day)
    alerts = rr.check(tmp_path / "picks", today=TODAY)
    assert len(alerts) == 2
    assert len({a.incident_key for a in alerts}) == 2
