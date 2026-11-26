"""pick_entry health source (audit F1): a committed pick whose entry was never
confirmed must leave a visible record once the submission cutoff has passed.

The escalation DMs are the live alarms; this source is the EOD/audit-trail
backstop that a committed pick ended the day alerted (WARN) or with the alert
itself undeliverable (dm_failed -> CRITICAL: unentered AND unreachable)."""
import json
from datetime import date, datetime, timezone

from bts.health.pick_entry import check

TODAY = date(2026, 7, 9)


def _marker(picks_dir, status, d="2026-07-09"):
    hs = picks_dir.parent / "health_state"
    hs.mkdir(parents=True, exist_ok=True)
    (hs / "pick_entry_check.json").write_text(json.dumps({
        "date": d, "status": status, "reason": "no_pick",
        "checked_at": f"{d}T18:00:00-04:00",
    }))


def _pick(picks_dir, game_time_iso, d="2026-07-09"):
    picks_dir.mkdir(parents=True, exist_ok=True)
    (picks_dir / f"{d}.json").write_text(json.dumps({
        "date": d,
        "pick": {"batter_name": "X", "p_game_hit": 0.8, "game_time": game_time_iso},
        "double_down": None,
    }))


def _now(hhmm_utc):
    h, m = hhmm_utc.split(":")
    return datetime(2026, 7, 9, int(h), int(m), tzinfo=timezone.utc)


def test_no_marker_stays_quiet(tmp_path):
    picks = tmp_path / "picks"
    picks.mkdir()
    assert check(picks, today=TODAY, now=_now("23:30")) == []


def test_confirmed_stays_quiet(tmp_path):
    picks = tmp_path / "picks"
    _pick(picks, "2026-07-09T23:10:00+00:00")
    _marker(picks, "confirmed")
    assert check(picks, today=TODAY, now=_now("23:30")) == []


def test_alerted_past_cutoff_warns(tmp_path):
    picks = tmp_path / "picks"
    _pick(picks, "2026-07-09T23:10:00+00:00")  # cutoff 23:05Z
    _marker(picks, "alerted")
    alerts = check(picks, today=TODAY, now=_now("23:30"))
    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert alerts[0].source == "pick_entry"
    assert "never confirmed" in alerts[0].message


def test_alerted_before_cutoff_stays_quiet(tmp_path):
    # Late game still open: the EOD suite must not warn prematurely while the
    # entry can still be made.
    picks = tmp_path / "picks"
    _pick(picks, "2026-07-10T01:45:00+00:00")  # 21:45 ET, cutoff 01:40Z
    _marker(picks, "alerted")
    assert check(picks, today=TODAY, now=_now("23:30")) == []


def test_dm_failed_past_cutoff_is_critical(tmp_path):
    picks = tmp_path / "picks"
    _pick(picks, "2026-07-09T23:10:00+00:00")
    _marker(picks, "dm_failed")
    alerts = check(picks, today=TODAY, now=_now("23:30"))
    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"


def test_stale_marker_from_other_date_stays_quiet(tmp_path):
    picks = tmp_path / "picks"
    _pick(picks, "2026-07-09T23:10:00+00:00")
    _marker(picks, "alerted", d="2026-07-08")
    assert check(picks, today=TODAY, now=_now("23:30")) == []
