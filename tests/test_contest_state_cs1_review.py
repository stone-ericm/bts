# Adopted verbatim from the Carbon review of change set 1, round 1 (marker CS1RG-8c24c71d).
# Source: docs/reviews/2026-11-14-season-rollover/build/cs1-review/carbon-r1/test_cs1_adversarial.py
"""Adversarial regression probes for CS1 (adoptable as pytest regression tests).

Covers gaps and boundaries beyond the author's tests:
- Real unreadable regular files (chmod 0o000 -> PermissionError) across Rows 1, 2, 5, 6, 7.
- Real unexpired manual v2 bounds (13/10, 58/58, 13/58, 0/-1) without disguising as auto schema.
- Automatic bounds for (58, None), (0, -1), and exact boundaries (57, 57) and (0, 0).
- Expired and legacy manual overrides beside Row-3 invalid and Row-4 undated automatic files.
- Malformed primary source_date masking valid as_of (accepted as dated, source_date stays None).
- End-to-end Row 5 optional None diagnostics through DM formatting, decision.json, and subtitle.
"""
from __future__ import annotations

import datetime as dt
import json
import os
from pathlib import Path

import pytest

from bts.contest_state import (
    ContestStateError,
    _has_unconfirmed_miss,
    _resolved_pick_dates,
    load_decision_streak_state,
)
from bts.daily_decision import load_decision, write_decision
from bts.posting import format_post, format_skip_post
from bts.web import _streak_subtitle

NOW = dt.datetime(2027, 3, 24, 16, 0, tzinfo=dt.timezone.utc)


def _auto(picks_dir: Path, active: int = 13, best: int | None = 20,
          source_date: str | None = "2027-03-24",
          recorded_at: str | None = "2027-03-24T16:00:00Z", **extra) -> Path:
    sd = picks_dir / "account_state"
    sd.mkdir(parents=True, exist_ok=True)
    body: dict = {
        "schema_version": "bts_contest_streak_auto_v1",
        "active_streak": active,
        "source": "mlb_bts_profile",
    }
    if best is not None:
        body["best_streak"] = best
    if source_date is not None:
        body["source_date"] = source_date
    if recorded_at is not None:
        body["recorded_at"] = recorded_at
    body.update(extra)
    path = sd / "contest_streak.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


def _manual(picks_dir: Path, active: int = 7, best: int | None = 12,
            source_date: str | None = "2027-03-24",
            expires_at: str | None = "2027-03-25T16:00:00Z", **extra) -> Path:
    sd = picks_dir / "account_state"
    sd.mkdir(parents=True, exist_ok=True)
    body: dict = {
        "schema_version": "bts_contest_streak_manual_v2",
        "active_streak": active,
        "source": "manual_cli",
    }
    if best is not None:
        body["best_streak"] = best
    if source_date is not None:
        body["source_date"] = source_date
    if expires_at is not None:
        body["override_expires_at"] = expires_at
    body.update(extra)
    path = sd / "contest_streak.manual.json"
    path.write_text(json.dumps(body), encoding="utf-8")
    return path


@pytest.mark.skipif(os.geteuid() == 0, reason="root bypasses chmod 000")
def test_chmod000_unreadable_files_across_rows_1_2_5_6_7(tmp_path: Path):
    # Row 1: unreadable pick file before a later miss in sorted order
    _auto(tmp_path, source_date="2027-03-21")
    unreadable_pick = tmp_path / "2027-03-22.json"
    unreadable_pick.write_text('{"result": "hit"}', encoding="utf-8")
    unreadable_pick.chmod(0o000)
    (tmp_path / "2027-03-24.json").write_text('{"result": "miss"}', encoding="utf-8")
    try:
        assert _resolved_pick_dates(tmp_path) == [dt.date(2027, 3, 24)]
        assert _has_unconfirmed_miss(tmp_path, dt.date(2027, 3, 21)) is True
        st = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
        assert (st.streak, st.status) == (13, "stale")
    finally:
        unreadable_pick.chmod(0o644)

    # Row 5: unreadable streak.json does not block valid contest state
    model_path = tmp_path / "streak.json"
    model_path.write_text('{"streak": 4, "saver_available": true}', encoding="utf-8")
    model_path.chmod(0o000)
    try:
        st5 = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
        assert st5.streak == 13
        assert st5.model_streak is None and st5.model_saver_available is None
        assert st5.model_state_error is not None and "PermissionError" in st5.model_state_error

        # Row 2: unreadable contest_streak.json -> ContestStateError
        auto_path = tmp_path / "account_state" / "contest_streak.json"
        auto_path.chmod(0o000)
        try:
            with pytest.raises(ContestStateError, match="malformed"):
                load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
        finally:
            auto_path.chmod(0o644)
            auto_path.unlink()

        # Row 7: unreadable streak.json + missing contest + require_contest_state=True
        with pytest.raises(ContestStateError, match="required but missing"):
            load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)

        # Row 6: unreadable streak.json + missing contest + require_contest_state=False
        with pytest.raises(ContestStateError, match="model state is corrupt"):
            load_decision_streak_state(tmp_path, require_contest_state=False, now=NOW)
    finally:
        model_path.chmod(0o644)


@pytest.mark.parametrize("active,best", [(58, None), (0, -1), (13, 10), (13, 58), (58, 60)])
def test_row3_auto_additional_bounds_rejected_in_both_modes(tmp_path: Path, active, best):
    path = _auto(tmp_path, active=active, best=best)
    for req in (False, True):
        with pytest.raises(ContestStateError) as exc:
            load_decision_streak_state(tmp_path, require_contest_state=req, now=NOW)
        assert str(path) in str(exc.value)
        assert f"active_streak={active}" in str(exc.value)
        assert f"best_streak={best}" in str(exc.value)


@pytest.mark.parametrize("active,best", [(0, 0), (57, 57), (57, None)])
def test_row3_auto_exact_boundaries_accepted(tmp_path: Path, active, best):
    _auto(tmp_path, active=active, best=best)
    st = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert st.streak == active


@pytest.mark.parametrize("active,best,expected_status", [
    (13, 10, "untrusted"),
    (58, 58, "untrusted"),
    (13, 58, "untrusted"),
    (0, -1, "untrusted"),
    (58, None, "missing"),
])
def test_row3_real_unexpired_manual_v2_bounds_remain_accepted(
    tmp_path: Path, active, best, expected_status,
):
    """Unlike renaming an auto_v1 payload, this tests a genuine unexpired manual_v2 schema."""
    _auto(tmp_path, active=2, best=5)
    _manual(tmp_path, active=active, best=best, expires_at="2027-03-25T16:00:00Z")
    st = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert (st.streak, st.best_streak, st.best_status) == (active, best, expected_status)


def test_corrupt_manual_alone_and_beside_corrupt_auto_raises_manual_domain_error(tmp_path: Path):
    sd = tmp_path / "account_state"
    sd.mkdir(parents=True, exist_ok=True)
    manual_path = sd / "contest_streak.manual.json"
    manual_path.write_bytes(b"\xff")
    # Alone (no contest_streak.json beside it)
    with pytest.raises(ContestStateError) as exc_alone:
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert str(manual_path) in str(exc_alone.value)

    # Both corrupt: manual is parsed first and named in the ContestStateError
    (sd / "contest_streak.json").write_bytes(b"\xff")
    with pytest.raises(ContestStateError) as exc_both:
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert str(manual_path) in str(exc_both.value)


def test_fetch_writer_out_of_bounds_best_rejected_and_manual_override_recovers(tmp_path: Path):
    from click.testing import CliRunner
    from bts.cli import cli
    from bts.contest_fetch import build_observation, validate_fetch, ContestFetchError

    profile = {"activeStreak": 0, "seasonBestStreak": 58}
    with pytest.raises(ContestFetchError, match="57"):
        validate_fetch(profile)
    with pytest.raises(ContestFetchError, match="57"):
        build_observation(profile, source_date=dt.date(2027, 3, 24), user_id=1, username="u", recorded_at=NOW)
    obs = {"schema_version": "bts_contest_streak_auto_v1", "active_streak": 0,
           "best_streak": 58, "source_date": "2027-03-24", "source": "mlb_bts_profile"}
    sd = tmp_path / "account_state"
    sd.mkdir(parents=True, exist_ok=True)
    auto_path = sd / "contest_streak.json"
    auto_path.write_text(json.dumps(obs), encoding="utf-8")

    with pytest.raises(ContestStateError, match="active_streak=0, best_streak=58"):
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)

    res = CliRunner().invoke(cli, [
        "set-contest-streak", "--streak", "0", "--best-streak", "57",
        "--source-date", "2027-03-24", "--override-expires-at", "2027-03-25T16:00:00Z",
        "--picks-dir", str(tmp_path),
    ])
    assert res.exit_code == 0, res.output
    recovered = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert (recovered.streak, recovered.best_streak, recovered.best_status) == (0, 57, "trusted")


@pytest.mark.parametrize("manual_expires", [None, "2027-03-23T16:00:00Z"])
def test_expired_or_legacy_manual_does_not_rescue_invalid_or_undated_auto(tmp_path: Path, manual_expires):
    _manual(tmp_path, active=7, best=12, expires_at=manual_expires)
    # Row 3 invalid auto beside expired/legacy manual -> ContestStateError in both modes
    _auto(tmp_path, active=13, best=10)
    for req in (False, True):
        with pytest.raises(ContestStateError, match="automatic contest streak invalid"):
            load_decision_streak_state(tmp_path, require_contest_state=req, now=NOW)

    # Row 4 undated auto beside expired/legacy manual -> accepted when optional, rejected when required
    _auto(tmp_path, active=14, best=18, source_date=None, recorded_at=None)
    st_opt = load_decision_streak_state(tmp_path, require_contest_state=False, now=NOW)
    assert (st_opt.streak, st_opt.status) == (14, "stale")
    with pytest.raises(ContestStateError, match="undated"):
        load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)


def test_row5_optional_diagnostics_flow_cleanly_through_dm_decision_and_subtitle(tmp_path: Path):
    _auto(tmp_path, active=9, best=15)
    (tmp_path / "streak.json").write_bytes(b"\xff")
    st = load_decision_streak_state(tmp_path, require_contest_state=True, now=NOW)
    assert st.model_streak is None and st.model_saver_available is None
    assert st.model_state_error is not None

    post_text = format_post("Steven Kwan", "CLE", "Gerrit Cole", 0.812, st.streak)
    skip_text = format_skip_post("Steven Kwan", "CLE", 0.740, st.streak)
    assert "Streak: 9" in post_text and "None" not in post_text
    assert "Streak holds at 9." in skip_text and "None" not in skip_text
    assert _streak_subtitle(st) == "Contest State"

    rec = write_decision(
        "2027-03-24", tmp_path,
        action="single", source="mdp",
        primary={"batter_id": 1, "batter_name": "Steven Kwan", "team": "CLE", "game_pk": 1, "p_game_hit": 0.812},
        streak=st.streak, saver_available=st.saver_available,
        state_source=st.source, state_status=st.status,
        allow_double=st.allow_double,
        contest_source_date=(st.contest_source_date.isoformat() if st.contest_source_date else None),
        delivery_status="delivered", scoreable=True,
        best_streak=st.best_streak, best_status=st.best_status,
        now=NOW,
    )
    assert rec is not None
    loaded = load_decision("2027-03-24", tmp_path)
    assert loaded is not None and loaded["streak"] == 9 and loaded["state_source"] == "contest"
