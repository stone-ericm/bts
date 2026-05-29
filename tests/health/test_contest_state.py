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


def test_valid_state_is_clean_when_expected(tmp_path):
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.manual.json").write_text(json.dumps({
        "active_streak": 7,
        "source": "manual_screenshot",
        "source_date": "2026-05-29",
    }))

    assert check(tmp_path, expected=True) == []
