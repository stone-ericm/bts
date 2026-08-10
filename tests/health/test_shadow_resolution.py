"""Tests for the shadow_resolution health source.

Backstop for the 2026-07-10 stranded-shadow incident class: the wait loop in
check-results settles same-night cases; this check makes the residual cases
(box down overnight, multi-day suspension, transient API failures exhausting
the deadline) visible instead of silent. Unresolved dates are derived
in-memory from the shadow files (not the status JSON, which is written
non-atomically by another process).
"""

from datetime import date, timedelta

from bts.health import shadow_resolution as sr
from bts.picks import Pick, DailyPick, save_shadow_pick
from bts.shadow_eval import stamp_shadow_version

TODAY = date(2026, 8, 9)


def _shadow(picks_dir, day, *, result=None, stamped=True):
    picks_dir.mkdir(parents=True, exist_ok=True)
    daily = DailyPick(
        date=day.isoformat(),
        run_time=f"{day.isoformat()}T15:00:00+00:00",
        pick=Pick(
            batter_name="Batter", batter_id=1, team="ATH", lineup_position=1,
            pitcher_name="P", pitcher_id=2, p_game_hit=0.8, flags=[],
            projected_lineup=False, game_pk=1000, game_time=f"{day.isoformat()}T23:10:00Z",
        ),
        double_down=None, runner_up=None, bluesky_posted=False, bluesky_uri=None,
    )
    daily.result = result
    if stamped:
        daily = stamp_shadow_version(daily)
    save_shadow_pick(daily, picks_dir)


def test_no_shadow_files_silent(tmp_path):
    assert sr.check(tmp_path / "picks", today=TODAY) == []


def test_yesterday_unresolved_within_grace(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=1))
    assert sr.check(tmp_path / "picks", today=TODAY) == []


def test_stale_unresolved_warns(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=3))
    alerts = sr.check(tmp_path / "picks", today=TODAY)
    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert alerts[0].source == "shadow_resolution"
    assert (TODAY - timedelta(days=3)).isoformat() in alerts[0].message


def test_very_stale_unresolved_critical(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=8))
    alerts = sr.check(tmp_path / "picks", today=TODAY)
    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"


def test_resolved_shadow_quiet(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=5), result="hit")
    assert sr.check(tmp_path / "picks", today=TODAY) == []


def test_legacy_v1_unstamped_excluded(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=5), stamped=False)
    assert sr.check(tmp_path / "picks", today=TODAY) == []


def test_each_overdue_date_gets_own_incident_key(tmp_path):
    _shadow(tmp_path / "picks", TODAY - timedelta(days=3))
    _shadow(tmp_path / "picks", TODAY - timedelta(days=4))
    alerts = sr.check(tmp_path / "picks", today=TODAY)
    assert len(alerts) == 2
    assert len({a.incident_key for a in alerts}) == 2
