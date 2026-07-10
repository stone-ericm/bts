"""check_heartbeat.py restart-churn detection (audit F3).

A deterministic startup crash restarts every 30s and refreshes the heartbeat
each cycle — freshness alone scores a crash-loop as healthy. The external
monitor therefore also samples the unit's NRestarts counter: a jump of >=3
within the window means the daemon is churning, which is treated exactly like
a stale heartbeat (fail ping -> healthchecks alert).
"""
from datetime import datetime, timedelta, timezone

from scripts.check_heartbeat import assess_churn

NOW = datetime(2026, 7, 9, 12, 0, tzinfo=timezone.utc)


def _sample(minutes_ago, n):
    return {"ts": (NOW - timedelta(minutes=minutes_ago)).isoformat(), "n": n}


def test_first_sample_no_churn():
    churn, reason, samples = assess_churn([], 1, NOW)
    assert churn is False
    assert samples == [{"ts": NOW.isoformat(), "n": 1}]


def test_small_delta_within_window_no_churn():
    prior = [_sample(15, 1), _sample(10, 2)]
    churn, _, _ = assess_churn(prior, 2, NOW)
    assert churn is False


def test_crash_loop_delta_fires():
    prior = [_sample(15, 1), _sample(10, 3)]
    churn, reason, _ = assess_churn(prior, 9, NOW)
    assert churn is True
    assert "restart churn" in reason
    assert "+8" in reason


def test_old_samples_outside_window_pruned():
    # +8 restarts but spread over 3 hours: baseline inside the window is 8.
    prior = [_sample(180, 1), _sample(15, 8)]
    churn, _, samples = assess_churn(prior, 9, NOW)
    assert churn is False
    assert all(s["n"] >= 8 for s in samples), "stale samples must be pruned"


def test_counter_reset_rebaselines_without_firing():
    # daemon-reload / unit reset drops NRestarts below prior samples.
    prior = [_sample(10, 40)]
    churn, _, samples = assess_churn(prior, 0, NOW)
    assert churn is False
    assert samples == [{"ts": NOW.isoformat(), "n": 0}]


def test_unreadable_nrestarts_skips_quietly():
    prior = [_sample(10, 1)]
    churn, reason, samples = assess_churn(prior, None, NOW)
    assert churn is False
    assert "skipped" in reason
    assert samples == prior, "samples must not be polluted with a None reading"
