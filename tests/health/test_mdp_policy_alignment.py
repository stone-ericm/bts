"""Tests for MDP policy/probability-scale alignment health check."""

import json
from datetime import date

import numpy as np

from bts.health.mdp_policy_alignment import SOURCE, check, compute_metrics, evaluate


def _write_policy(path, boundaries=(0.80, 0.82, 0.84, 0.86)):
    np.savez_compressed(path, boundaries=np.array(boundaries))


def _write_pick(picks_dir, date_iso, p, dd_p=None):
    body = {
        "date": date_iso,
        "pick": {"batter_name": "X", "p_game_hit": p},
    }
    if dd_p is not None:
        body["double_down"] = {"batter_name": "Y", "p_game_hit": dd_p}
    (picks_dir / f"{date_iso}.json").write_text(json.dumps(body))


def test_no_alert_when_policy_missing(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    for i in range(14):
        _write_pick(picks_dir, f"2026-04-{i + 1:02d}", 0.74)

    alerts = check(picks_dir, tmp_path / "missing.npz", today=date(2026, 4, 14))

    assert alerts == []


def test_no_alert_when_insufficient_recent_picks(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    policy_path = tmp_path / "mdp_policy.npz"
    _write_policy(policy_path)
    for i in range(10):
        _write_pick(picks_dir, f"2026-04-{i + 1:02d}", 0.74)

    alerts = check(picks_dir, policy_path, today=date(2026, 4, 10))

    assert alerts == []


def test_warn_when_all_recent_primary_picks_are_q0(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    policy_path = tmp_path / "mdp_policy.npz"
    _write_policy(policy_path)
    for i in range(21):
        _write_pick(picks_dir, f"2026-04-{i + 1:02d}", 0.70 + i * 0.002, 0.69)

    alerts = check(picks_dir, policy_path, today=date(2026, 4, 21))

    assert len(alerts) == 1
    alert = alerts[0]
    assert alert.level == "WARN"
    assert alert.source == SOURCE
    assert "21/21" in alert.message
    assert "Q0" in alert.message
    assert "21/21 below lowest boundary" in alert.message
    assert "policy boundaries 0.800-0.860" in alert.message
    assert "Double-down maps 21/21 to Q0" in alert.message


def test_warn_when_recent_primary_picks_mostly_one_bin(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    policy_path = tmp_path / "mdp_policy.npz"
    _write_policy(policy_path)
    values = [0.74] * 17 + [0.81, 0.83, 0.85, 0.87]
    for i, p in enumerate(values):
        _write_pick(picks_dir, f"2026-04-{i + 1:02d}", p)

    alerts = check(picks_dir, policy_path, today=date(2026, 4, 21))

    assert len(alerts) == 1
    assert alerts[0].level == "WARN"
    assert "17/21" in alerts[0].message


def test_no_alert_when_recent_picks_use_multiple_bins(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    policy_path = tmp_path / "mdp_policy.npz"
    _write_policy(policy_path)
    values = [0.74, 0.81, 0.83, 0.85, 0.87] * 5
    for i, p in enumerate(values):
        _write_pick(picks_dir, f"2026-04-{i + 1:02d}", p)

    alerts = check(picks_dir, policy_path, today=date(2026, 4, 25))

    assert alerts == []


def test_compute_metrics_uses_recent_pick_limit(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    policy_path = tmp_path / "mdp_policy.npz"
    _write_policy(policy_path)
    for i in range(25):
        p = 0.87 if i < 4 else 0.74
        _write_pick(picks_dir, f"2026-04-{i + 1:02d}", p)

    metrics = compute_metrics(picks_dir, policy_path, today=date(2026, 4, 25))

    assert metrics is not None
    assert metrics.primary.n == 21
    assert metrics.primary.counts[0] == 21
    assert evaluate(metrics)[0].level == "WARN"


def test_compute_metrics_skips_malformed_probability(tmp_path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    policy_path = tmp_path / "mdp_policy.npz"
    _write_policy(policy_path)
    for i in range(14):
        _write_pick(picks_dir, f"2026-04-{i + 1:02d}", 0.74)
    _write_pick(picks_dir, "2026-04-15", "not-a-probability")

    metrics = compute_metrics(picks_dir, policy_path, today=date(2026, 4, 15))

    assert metrics is not None
    assert metrics.primary.n == 14
