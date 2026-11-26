"""Verify real CPython 3.12 RecursionError threshold (200,000 nesting depth) on 2e20fdfc and confirm why 3,000 in test_wr_f2_saver_flag.py:246 was a false green."""

from __future__ import annotations

import datetime as dt
import json
import sys
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

from bts.health import contest_state as health

ET = ZoneInfo("America/New_York")
DAY = dt.datetime(2027, 7, 15, 22, 30, tzinfo=ET)
DEEP_200K = b"[" * 200_000 + b"]" * 200_000
SHALLOW_3K = b"[" * 3000 + b"]" * 3000


def test_3000_nesting_does_not_raise_recursion_error_in_cpython312():
    # Demonstrates why test_wr_f2_saver_flag.py:246 ("nested": b"[" * 3000 + b"]" * 3000) is a false green:
    parsed = json.loads(SHALLOW_3K)
    assert isinstance(parsed, list)
    with pytest.raises(RecursionError):
        json.loads(DEEP_200K)


@pytest.mark.parametrize("historical", [False, True])
def test_real_recursion_error_200k_preserves_alerts_on_dated_used(tmp_path, monkeypatch, historical):
    picks = tmp_path / "picks"
    acct = picks / "account_state"
    acct.mkdir(parents=True)
    exp = dt.datetime.combine(DAY.date(), dt.time.max, ET)
    (acct / "contest_streak.manual.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": "2027-07-15",
        "season": 2027,
        "expires_at": exp.isoformat(),
        "active_streak": 14,
        "best_streak": 15,
        "saver_state": "used",
        "recorded_at": "2027-07-15T20:00:00Z",
        "source": "cli_dated",
    }), encoding="utf-8")

    resolved = [dt.date(2027, 7, 16), dt.date(2027, 7, 17)]
    monkeypatch.setattr(health, "latest_resolved_pick_date", lambda _: max(resolved))
    monkeypatch.setattr(health, "_scan_resolved_pick_dates", lambda *a, **kw: resolved)
    monkeypatch.setattr(health, "resolved_pick_settlement_gap", lambda *a: 2)

    baseline = health.check(picks, expected=True, now=DAY, historical_day=DAY.date() if historical else None)
    assert len(baseline) == 1 and baseline[0].source == "contest_state"

    (acct / "saver_state.json").write_bytes(DEEP_200K)
    actual = health.check(picks, expected=True, now=DAY, historical_day=DAY.date() if historical else None)
    assert actual == baseline


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "-rA"]))
