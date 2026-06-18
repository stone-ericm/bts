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


def test_gap2_coverage_lag_is_warn(tmp_path):
    """A >=2 settled-pick gap is a coverage lag (the predictions array trailing the
    live activeStreak counter), NOT reliable staleness under the snapshot/coverage
    split -> WARN, never a (false) CRITICAL DM."""
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 0, "best_streak": 9,
        "source": "mlb_bts_profile", "source_date": "2026-06-01",
    }))
    # The week-long-freeze incident: picks resolve daily while source_date stays
    # frozen — 4 settled picks newer than source_date == genuine staleness.
    for d in ("2026-06-02", "2026-06-03", "2026-06-04", "2026-06-05"):
        (tmp_path / f"{d}.json").write_text(json.dumps({"result": "hit"}))

    alerts = check(tmp_path, expected=True)
    assert any(a.level == "WARN" and a.source == SOURCE for a in alerts), alerts
    assert not any(a.level == "CRITICAL" for a in alerts), alerts


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

    from datetime import datetime, timezone
    overnight = datetime(2026, 6, 8, 6, 0, tzinfo=timezone.utc)  # 02:00 ET
    alerts = check(tmp_path, expected=True, now=overnight)
    assert not any(a.level == "CRITICAL" for a in alerts), alerts
    assert any(a.level == "INFO" and a.source == SOURCE for a in alerts), alerts


def test_gap1_past_noon_is_info_not_warn(tmp_path):
    """Phase-1 snapshot/coverage split: source_date is derived from the per-round
    predictions array, which trails the live activeStreak counter by ~one round BY
    DESIGN. So a 1-pick gap past noon ET is the normal coverage lag, not a stuck
    contest — it must be INFO, not the old daily-false WARN (and no 'frozen' wording,
    since Phase 1 no longer freezes picks on staleness)."""
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 2, "best_streak": 9,
        "source": "mlb_bts_profile", "source_date": "2026-06-06",
    }))
    (tmp_path / "2026-06-06.json").write_text(json.dumps({"result": "hit"}))
    (tmp_path / "2026-06-07.json").write_text(json.dumps({"result": "hit"}))

    from datetime import datetime, timezone
    afternoon = datetime(2026, 6, 8, 19, 0, tzinfo=timezone.utc)  # 15:00 ET
    alerts = check(tmp_path, expected=True, now=afternoon)
    assert not any(a.level == "WARN" for a in alerts), alerts
    assert not any(a.level == "CRITICAL" for a in alerts), alerts
    infos = [a for a in alerts if a.level == "INFO" and a.source == SOURCE]
    assert infos, alerts
    assert "frozen" not in infos[0].message.lower()
    assert "coverage" in infos[0].message.lower()


def test_offday_gap_with_one_new_pick_is_not_critical(tmp_path):
    """All-Star break / league off-days: a multi-CALENDAR-day gap with only ONE
    settled pick newer than source_date is the expected 1-settlement-step lag,
    not staleness. Gap is measured in settled picks, not calendar days, so this
    must be INFO — not a (false) CRITICAL on the first day back (audit H2)."""
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 5, "best_streak": 9,
        "source": "mlb_bts_profile", "source_date": "2026-07-13",
    }))
    # Last game before the All-Star break (= source_date), a 3-day no-pick gap,
    # then the first game back today: only this one pick is newer than source.
    (tmp_path / "2026-07-13.json").write_text(json.dumps({"result": "hit"}))
    (tmp_path / "2026-07-17.json").write_text(json.dumps({"result": "hit"}))

    from datetime import datetime, timezone
    alerts = check(tmp_path, expected=True,
                   now=datetime(2026, 7, 17, 23, 0, tzinfo=timezone.utc))
    assert not any(a.level == "CRITICAL" for a in alerts), alerts
    # gap==1 (1 settled pick newer than source) is the normal coverage lag -> INFO,
    # never CRITICAL on the first day back (the H2 protection this test pins).
    assert any(a.level == "INFO" and a.source == SOURCE for a in alerts), alerts
    assert not any(a.level == "WARN" for a in alerts), alerts


def test_two_day_gap_is_warn_not_critical(tmp_path):
    """>= 2 settled picks newer than source_date is a coverage gap (the predictions
    array trailing the activeStreak counter), not reliable staleness under the
    snapshot/coverage split -> WARN, not a CRITICAL DM."""
    state_dir = tmp_path / "account_state"
    state_dir.mkdir()
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 2, "best_streak": 9,
        "source": "mlb_bts_profile", "source_date": "2026-06-05",
    }))
    # Two settled picks newer than source_date (06-06 and 06-07): the contest is
    # genuinely >= 2 settlements behind, not just a one-pick overnight lag.
    (tmp_path / "2026-06-06.json").write_text(json.dumps({"result": "hit"}))
    (tmp_path / "2026-06-07.json").write_text(json.dumps({"result": "hit"}))

    alerts = check(tmp_path, expected=True)
    assert any(a.level == "WARN" and a.source == SOURCE for a in alerts), alerts
    assert not any(a.level == "CRITICAL" for a in alerts), alerts


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


def _write_contest(tmp_path, active, best):
    state_dir = tmp_path / "account_state"; state_dir.mkdir(exist_ok=True)
    (state_dir / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": active, "best_streak": best,
        "source": "mlb_bts_profile", "source_date": "2026-06-18",
    }))


def test_saver_flag_uninitialized_in_zone_warns(tmp_path):
    _write_contest(tmp_path, active=12, best=12)   # in the 10-15 zone, no saver_state.json
    saver_alerts = [a for a in check(tmp_path, expected=False) if "Streak Saver" in a.message]
    assert len(saver_alerts) == 1 and saver_alerts[0].level == "WARN"
    assert "saver-state --init" in saver_alerts[0].message


def test_saver_flag_active_in_zone_is_clean(tmp_path):
    from bts.saver_state import transition_saver_state
    _write_contest(tmp_path, active=12, best=12)
    transition_saver_state(tmp_path, expected_prior="uninitialized", new_state="active",
                           season=2026, source="t")
    assert [a for a in check(tmp_path, expected=False) if "Streak Saver" in a.message] == []


def test_saver_flag_uninitialized_below_zone_no_warn(tmp_path):
    _write_contest(tmp_path, active=8, best=9)      # below the zone -> saver irrelevant
    assert [a for a in check(tmp_path, expected=False) if "Streak Saver" in a.message] == []
