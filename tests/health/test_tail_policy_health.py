"""Health source ``tail_policy``: the deployed artifacts must load and pair (2026-09-03).

Codex r2: the deploy canary only checks service state + dashboard HTTP; a
missing/unpaired tail artifact would otherwise surface only as a lazy-loader log
line. Once the season is inside the tail window (<= 28 days left, so SOME streak
is in the tail regime) an invalid tail is CRITICAL; earlier it is a WARN.
"""
from datetime import date

import numpy as np

from bts.health.tail_policy import SOURCE, check
from bts.simulate.tail_policy import (
    MAX_TAIL_DAYS, OBJECTIVE_TAIL, TARGET, TailPolicy, save_tail_policy, sha256_file,
    solve_emax_season_best, tail_manifest,
)

P_HIT, P_BOTH = 2641 / 3600, 1984 / 3600


def _write_base(path):
    np.savez_compressed(path, policy_table=np.zeros((58, 181, 2, 5), np.int8),
                        boundaries=np.array([0.796, 0.811, 0.825, 0.841]),
                        season_length=np.array(180), optimal_p57=np.array(0.08))


def _write_tail(path, base_sha):
    sol = solve_emax_season_best(np.array([1.0]), np.array([P_HIT]), np.array([P_BOTH]))
    save_tail_policy(TailPolicy(objective=OBJECTIVE_TAIL, policy_table=sol.policy, boundaries=[],
                                bin_freq=[1.0], bin_p_hit=[P_HIT], bin_p_both=[P_BOTH],
                                target=TARGET, max_days=MAX_TAIL_DAYS, base_policy_sha256=base_sha,
                                manifest=tail_manifest(n_bins=1, hits=2641, both=1984, late_seed_days=3600),
                                built_at="2026-09-03T00:00:00Z",
                                solver="solve_emax_season_best"), path)


def test_paired_artifacts_are_silent(tmp_path):
    base = tmp_path / "mdp_policy.npz"; tail = tmp_path / "mdp_tail_policy.npz"
    _write_base(base); _write_tail(tail, sha256_file(base))
    assert check(base_path=base, tail_path=tail, today=date(2026, 9, 4)) == []


def test_missing_tail_inside_the_window_is_critical(tmp_path):
    base = tmp_path / "mdp_policy.npz"; _write_base(base)
    alerts = check(base_path=base, tail_path=tmp_path / "nope.npz", today=date(2026, 9, 4))
    assert [a.level for a in alerts] == ["CRITICAL"] and alerts[0].source == SOURCE
    assert "missing" in alerts[0].message


def test_missing_tail_before_the_window_is_a_warning(tmp_path):
    base = tmp_path / "mdp_policy.npz"; _write_base(base)
    alerts = check(base_path=base, tail_path=tmp_path / "nope.npz", today=date(2026, 6, 1))
    assert [a.level for a in alerts] == ["WARN"]


def test_unpaired_tail_is_critical_in_window(tmp_path):
    base = tmp_path / "mdp_policy.npz"; tail = tmp_path / "mdp_tail_policy.npz"
    _write_base(base); _write_tail(tail, "0" * 64)
    alerts = check(base_path=base, tail_path=tail, today=date(2026, 9, 10))
    assert [a.level for a in alerts] == ["CRITICAL"] and "mismatch" in alerts[0].message


def test_missing_base_is_reported_too(tmp_path):
    tail = tmp_path / "mdp_tail_policy.npz"; _write_tail(tail, "0" * 64)
    alerts = check(base_path=tmp_path / "nope.npz", tail_path=tail, today=date(2026, 9, 4))
    assert any("base" in a.message for a in alerts)
