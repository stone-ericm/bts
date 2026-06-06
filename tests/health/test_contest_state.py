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
