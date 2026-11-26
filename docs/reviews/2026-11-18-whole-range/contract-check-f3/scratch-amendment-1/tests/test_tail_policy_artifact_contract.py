"""Contract test on the COMMITTED artifacts (2026-09-03).

The deploy canary checks only service state and dashboard HTTP, so this is
the gate that proves the shipped pair is what production will load: the base
policy is the pinned bytes, the tail artifact loads under the strict contract
AND is bound to that base, and the first live tail state resolves through it.
Codex r2: seeing a double on 9/04 does not prove best_streak was threaded
(best 0 and best 18 both double at streak 0), so the effective best is asserted
here and must be checked in decision.json on the first live day.
"""
from pathlib import Path

import pytest

from bts.simulate.mdp import DEFAULT_POLICY_PATH
from bts.simulate.tail_policy import (
    DEFAULT_TAIL_POLICY_PATH, MAX_TAIL_DAYS, OBJECTIVE_REACH57, OBJECTIVE_TAIL, load_tail_policy,
    lookup_tail_action, mdp_objective, sha256_file,
)

# The pooled reach-57 policy shipped 2026-06-05 (commit e1ebde9). A base rebuild
# MUST be accompanied by `scripts/rebuild_tail_policy.py` and a new pin here.
SHIPPED_BASE_SHA256 = "66d154717ae51afb3343ee4bec8138c60bd1056e46a3de449043f4e9f76b93b4"

pytestmark = pytest.mark.skipif(
    not (Path(DEFAULT_POLICY_PATH).exists() and Path(DEFAULT_TAIL_POLICY_PATH).exists()),
    reason="committed policy artifacts not present (run from the repo root)")


def test_base_policy_is_the_pinned_artifact():
    assert sha256_file(DEFAULT_POLICY_PATH) == SHIPPED_BASE_SHA256


def test_tail_artifact_loads_and_pairs_with_the_shipped_base():
    tail = load_tail_policy(DEFAULT_TAIL_POLICY_PATH, expected_base_sha=SHIPPED_BASE_SHA256)
    assert tail.objective == OBJECTIVE_TAIL and tail.n_bins == 1 and tail.max_days == MAX_TAIL_DAYS
    assert tail.bin_p_hit == pytest.approx([2641 / 3600]) and tail.bin_p_both == pytest.approx([1984 / 3600])
    assert tail.manifest["late_distinct_dates"] == 150 and tail.manifest["seed_dirs"] == 24


def test_first_live_tail_day_resolves_through_the_committed_tail():
    tail = load_tail_policy(DEFAULT_TAIL_POLICY_PATH)
    # 9/04: streak 0, best 18, 24 days left -> double; 9/19 (9 days) -> the stop rule.
    assert mdp_objective(0, 24) == OBJECTIVE_TAIL and mdp_objective(0, 29) == OBJECTIVE_REACH57
    assert lookup_tail_action(tail, 0, 18, 24, False, 0.72) == "double"
    assert lookup_tail_action(tail, 0, 18, 9, False, 0.72) == "skip"
    assert lookup_tail_action(tail, 0, 18, 10, False, 0.72) == "double"


def test_production_loader_sees_both_artifacts(monkeypatch):
    import bts.strategy as strat
    monkeypatch.setattr(strat, "_mdp_cache", None)
    mdp = strat._load_mdp()
    assert mdp is not None and mdp["policy_table"] is not None and mdp["tail"] is not None
    assert mdp["tail_error"] is None and mdp["base_error"] is None
    dec = strat.resolve_policy_decision(
        strat.DecisionContext(primary_p=0.72, second_p=0.70, has_diff_game=True, date="2026-09-04",
                              allow_double=True, mdp=mdp, best_streak=18, best_status="trusted"),
        streak=0, saver=False)
    assert (dec.action, dec.objective, dec.effective_best) == ("double", OBJECTIVE_TAIL, 18)
    assert dec.tail_sha256 == mdp["tail"].sha256 and dec.degraded_reason is None
