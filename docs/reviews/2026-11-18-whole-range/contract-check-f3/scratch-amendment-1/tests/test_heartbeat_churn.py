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


def test_slow_benign_drift_no_churn():
    # +2 over three hours (daily-cycle territory): no window fires.
    prior = [_sample(170, 1), _sample(15, 2)]
    churn, _, _ = assess_churn(prior, 3, NOW)
    assert churn is False


def test_slow_crash_loop_caught_by_hour_window(): 
    # Codex review #3: a ~10-min failure cycle adds only +2 per 20-min window
    # and evaded the single-window check; the 60-min horizon must catch it.
    prior = [_sample(55, 1), _sample(45, 2), _sample(35, 3), _sample(25, 4), _sample(15, 5)]
    churn, reason, _ = assess_churn(prior, 7, NOW)
    assert churn is True
    assert "60 min" in reason


def test_samples_beyond_max_window_pruned():
    prior = [_sample(200, 1), _sample(15, 2)]
    _, _, samples = assess_churn(prior, 2, NOW)
    assert all(s["n"] >= 2 for s in samples), "samples beyond the max window must drop"


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


def test_null_samples_value_handled():
    # Codex review #10: {"samples": null} in the state file must not crash the
    # monitor before it pings.
    from scripts.check_heartbeat import load_churn_samples
    import json, tempfile, pathlib
    d = pathlib.Path(tempfile.mkdtemp())
    f = d / "churn.json"
    f.write_text(json.dumps({"unit": "x", "samples": None}))
    assert load_churn_samples(f) == []


def test_naive_timestamp_sample_does_not_crash():
    prior = [{"ts": "2026-07-09T11:45:00", "n": 1}]  # tz-naive
    churn, _, samples = assess_churn(prior, 1, NOW)
    assert churn is False
