import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

from bts.health.fallback_defer import SOURCE, check

ET = ZoneInfo("America/New_York")
AFTER_DELIVERY_WINDOW = datetime(2026, 5, 24, 22, 0, tzinfo=ET)
BEFORE_DELIVERY_WINDOW = datetime(2026, 5, 24, 21, 59, tzinfo=ET)


def _write_archive(picks_dir, date_iso="2026-05-24", *, name="Early Pick", p=0.70):
    day_dir = picks_dir / date_iso
    day_dir.mkdir()
    path = day_dir / "deferred_fallback_20260524T120000-0400.json"
    path.write_text(json.dumps({
        "date": date_iso,
        "pick": {
            "batter_id": 1,
            "batter_name": name,
            "team": "AAA",
            "p_game_hit": p,
            "game_pk": 100,
        },
        "deferred_fallback": {
            "reason": "should_lock_false_future_checks_remain",
            "deferred_at": "2026-05-24T12:00:00-04:00",
        },
    }))
    return path


def _write_final_pick(
    picks_dir,
    date_iso="2026-05-24",
    *,
    name="Later Pick",
    p=0.725,
    delivered=True,
):
    payload = {
        "date": date_iso,
        "pick": {
            "batter_id": 2,
            "batter_name": name,
            "team": "BBB",
            "lineup_position": 1,
            "pitcher_name": "Pitcher",
            "pitcher_id": 9,
            "p_game_hit": p,
            "flags": [],
            "projected_lineup": False,
            "game_pk": 200,
            "game_time": "2026-05-24T23:05:00Z",
        },
        "run_time": "2026-05-24T16:45:00+00:00",
        "double_down": None,
        "runner_up": None,
        "bluesky_posted": False,
        "bluesky_uri": None,
        "notification_sent": delivered,
        "notification_id": "dm-1" if delivered else None,
    }
    (picks_dir / f"{date_iso}.json").write_text(json.dumps(payload))


def test_no_deferred_archives_returns_clean(tmp_path):
    alerts = check(
        tmp_path,
        today=date(2026, 5, 24),
        now=AFTER_DELIVERY_WINDOW,
    )
    assert alerts == []


def test_deferred_archive_with_delivered_pick_returns_info(tmp_path):
    _write_archive(tmp_path)
    _write_final_pick(tmp_path)

    alerts = check(
        tmp_path,
        today=date(2026, 5, 24),
        now=AFTER_DELIVERY_WINDOW,
    )

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.level == "INFO"
    assert alert.source == SOURCE
    assert "fallback defer observed for 2026-05-24" in alert.message
    assert "deferred=Early Pick (AAA) 70.0%" in alert.message
    assert "delivered=Later Pick (BBB) 72.5%" in alert.message
    assert "primary_p_delta=+2.5pp" in alert.message
    assert "never_miss=confirmed" in alert.message


def test_deferred_archive_without_final_pick_is_critical(tmp_path):
    _write_archive(tmp_path)

    alerts = check(
        tmp_path,
        today=date(2026, 5, 24),
        now=AFTER_DELIVERY_WINDOW,
    )

    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"
    assert alerts[0].source == SOURCE
    assert "never-miss validation failed" in alerts[0].message
    assert "final pick file missing" in alerts[0].message


def test_deferred_archive_with_undelivered_final_pick_is_critical(tmp_path):
    _write_archive(tmp_path)
    _write_final_pick(tmp_path, delivered=False)

    alerts = check(
        tmp_path,
        today=date(2026, 5, 24),
        now=AFTER_DELIVERY_WINDOW,
    )

    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"
    assert "no public post or private notification is recorded" in alerts[0].message


def test_deferred_archive_before_delivery_window_does_not_page(tmp_path):
    _write_archive(tmp_path)

    alerts = check(
        tmp_path,
        today=date(2026, 5, 24),
        now=BEFORE_DELIVERY_WINDOW,
    )

    assert alerts == []
