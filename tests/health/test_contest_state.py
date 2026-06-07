import json

from bts.health.contest_state import SOURCE, check


def test_no_alert_when_not_expected_and_missing(tmp_path):
    assert check(tmp_path, expected=False) == []


def test_expected_missing_is_critical(tmp_path):
    alerts = check(tmp_path, expected=True)

    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"
    assert alerts[0].source == SOURCE
    assert "expected but missing" in alerts[0].message


def test_malformed_existing_state_is_critical(tmp_path):
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.manual.json").write_text("{bad")

    alerts = check(tmp_path, expected=False)

    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"
    assert alerts[0].source == SOURCE
    assert "malformed" in alerts[0].message


def test_non_object_existing_state_is_critical(tmp_path):
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.manual.json").write_text("[]")

    alerts = check(tmp_path, expected=False)

    assert len(alerts) == 1
    assert alerts[0].level == "CRITICAL"
    assert alerts[0].source == SOURCE
    assert "expected object" in alerts[0].message


def test_valid_auto_state_is_clean_when_expected(tmp_path):
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 0, "best_streak": 9,
        "source": "mlb_bts_profile", "source_date": "2026-06-06",
    }))

    assert check(tmp_path, expected=True) == []


def test_stale_contest_state_is_critical(tmp_path):
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 0, "best_streak": 9,
        "source": "mlb_bts_profile", "source_date": "2026-06-01",
    }))
    (tmp_path / "2026-06-05.json").write_text(json.dumps({"result": "hit"}))

    alerts = check(tmp_path, expected=True)
    assert any(a.level == "CRITICAL" and "STALE" in a.message for a in alerts), alerts


def test_expected_overnight_lag_is_not_critical(tmp_path):
    """End-of-day: our scheduler settled D=2026-06-07, but the contest account's
    source_date still = D-1 (next-day fetch hasn't run). gap==1 is the EXPECTED
    overnight lag, not genuine staleness — must NOT fire a CRITICAL (alert fatigue
    here could mask a real stale event). Surfaced as INFO for visibility instead."""
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 2, "best_streak": 9,
        "source": "mlb_bts_profile", "source_date": "2026-06-06",
    }))
    (tmp_path / "2026-06-06.json").write_text(json.dumps({"result": "hit"}))
    (tmp_path / "2026-06-07.json").write_text(json.dumps({"result": "hit"}))

    alerts = check(tmp_path, expected=True)
    assert not any(a.level == "CRITICAL" for a in alerts), alerts
    assert any(a.level == "INFO" and a.source == SOURCE for a in alerts), alerts


def test_two_day_gap_is_critical(tmp_path):
    """gap >= 2 is genuine staleness (the week-long-freeze incident class) -> CRITICAL."""
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 2, "best_streak": 9,
        "source": "mlb_bts_profile", "source_date": "2026-06-05",
    }))
    (tmp_path / "2026-06-07.json").write_text(json.dumps({"result": "hit"}))  # gap = 2

    alerts = check(tmp_path, expected=True)
    assert any(a.level == "CRITICAL" and "STALE" in a.message for a in alerts), alerts


def test_future_source_date_is_critical(tmp_path):
    """A source_date in the future is impossible (US contest dates trail UTC); a
    corrupt/fat-fingered future date must not silently pass the freshness check."""
    from datetime import datetime, timezone
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 2, "best_streak": 9,
        "source": "mlb_bts_profile", "source_date": "2099-01-01",
    }))
    (tmp_path / "2026-06-07.json").write_text(json.dumps({"result": "hit"}))

    alerts = check(tmp_path, expected=True, now=datetime(2026, 6, 7, 12, 0, tzinfo=timezone.utc))
    assert any(a.level == "CRITICAL" and "FUTURE" in a.message.upper() for a in alerts), alerts


def test_missing_source_date_is_critical(tmp_path):
    """A contest file with no source_date cannot be freshness-verified -> CRITICAL."""
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 2, "best_streak": 9, "source": "mlb_bts_profile",
    }))  # no source_date
    (tmp_path / "2026-06-07.json").write_text(json.dumps({"result": "hit"}))

    alerts = check(tmp_path, expected=True)
    assert any(a.level == "CRITICAL" for a in alerts), alerts


def test_legacy_manual_present_warns(tmp_path):
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 0, "best_streak": 9, "source": "manual", "source_date": "2026-06-06",
    }))
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 0, "best_streak": 9,
        "source": "mlb_bts_profile", "source_date": "2026-06-06",
    }))

    alerts = check(tmp_path, expected=True)
    assert any(a.level == "WARN" and "legacy" in a.message.lower() for a in alerts), alerts
