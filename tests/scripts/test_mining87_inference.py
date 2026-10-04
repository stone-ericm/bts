"""#87 adapter: intervals, complete cell families, the five nomination conditions and the tie-excluded sensitivity
(review r1 findings 3, 5, 7, 9; required changes 7, 8, 9, 11)."""
from __future__ import annotations

import pandas as pd
import pytest

from scripts.audit.mining87 import inference as inf
from scripts.leaderboard_mechanism_mining import DECOMPOSITION_VARIABLES

BASE = {"pick_number": 1, "consensus_pick_share_bin": ">=0.25", "production_p_game_hit_bin": "0.74-0.80",
        "agreement_state": "different_batter", "production_batter_skill_quartile": "missing",
        "production_batter_skill_prior_pa_bin": "missing", "production_projected_lineup": "false",
        "production_regime": "post_bpm", "production_is_park_driven": "missing", "production_is_indoor": "missing",
        "production_weather_temp_bin": "indoor_or_missing", "consensus_model_rank_bin": "missing_surface",
        "consensus_model_probability_bin": "missing_surface"}
PARAMS = dict(min_n=15, q_threshold=0.10, mechanism_min_n=30, mechanism_min_lift=0.05)


def units(cohort, deltas, *, start=0, tie=None, **overrides):
    """Resolved disagreement units, one per date, with the given deltas (+1, 0, -1)."""
    rows = []
    for i, d in enumerate(deltas):
        ph = 0 if d > 0 else 1 if d < 0 else 1
        rows.append({**BASE, **overrides, "cohort": cohort, "date": f"2026-{4 + (start + i) // 28:02d}-"
                                                                    f"{1 + (start + i) % 28:02d}",
                     "production_resolved": True, "consensus_available": True, "joint_resolved": True,
                     "resolved_disagreement": overrides.get("agreement_state", "different_batter") == "different_batter",
                     "production_hit": ph, "consensus_hit": ph + d, "delta": d,
                     "consensus_tie_broken": bool(tie[i]) if tie is not None else False})
    return pd.DataFrame(rows)


def cell_of(fr):
    return {v: fr.iloc[0][v] for v in DECOMPOSITION_VARIABLES}


def mech(cell, **kw):
    rec = {"cell": {k: (int(v) if k == "pick_number" else v) for k, v in cell.items()},
           "mechanism": "model underweights confirmed-lineup contact hitters",
           "operational_variables": ["production_projected_lineup"],
           "lock_evidence": {"production_projected_lineup": {"status": "available_at_lock",
                                                             "evidence": "ledger projected_lineup from the pick file"}}}
    rec.update(kw)
    return rec


def stream(frame, records=(), can_nominate=True, name="primary"):
    return inf.evaluate_stream(frame, stream=name, mechanism_records=list(records), can_nominate=can_nominate,
                               **PARAMS)


def fixed_row(result):
    cells = result["cells"]
    return cells[(cells["cohort"] == "fixed_cohort") & cells["testable"]].iloc[0]


# --- intervals -------------------------------------------------------------------------------------------------------

def test_bootstrap_resamples_date_sums_and_counts_so_it_targets_the_unit_weighted_mean():
    frame = pd.DataFrame({"date": ["2026-04-01", "2026-04-01", "2026-04-02"], "delta": [1, 1, -1]})
    out = inf.date_block_bootstrap_ratio(frame, "delta", expected_block_length=7, n_bootstrap=2000, seed=20260510,
                                         return_samples=True)
    assert out["mean"] == pytest.approx(1 / 3) and out["n_units"] == 3 and out["n_dates"] == 2
    samples = out.pop("samples")
    assert any(abs(x - 1 / 3) < 1e-12 for x in samples)                  # a mixed draw is the ratio 1/3 ...
    assert not any(abs(x) < 1e-12 for x in samples)                     # ... never the equal-weight date mean 0
    assert out["seed"] == 20260510 and out["expected_block_length"] == 7 and out["n_bootstrap"] == 2000
    assert "ordered observed dates" in out["block_semantics"]
    again = inf.date_block_bootstrap_ratio(frame, "delta", expected_block_length=7, n_bootstrap=2000, seed=20260510)
    assert (again["ci_lower"], again["ci_upper"]) == (out["ci_lower"], out["ci_upper"])
    assert inf.date_block_bootstrap_ratio(frame.iloc[:2], "delta", expected_block_length=7, n_bootstrap=2000,
                                          seed=1)["ci_lower"] is None   # one date: no interval
    assert inf.date_block_bootstrap_ratio(frame.iloc[:0], "delta", expected_block_length=7, n_bootstrap=2000,
                                          seed=1) is None


# --- complete families -----------------------------------------------------------------------------------------------

def test_every_nonempty_cell_is_inventoried_and_sparse_cells_have_unavailable_p_and_q():
    frame = pd.concat([units("fixed_cohort", [1] * 20), units("fixed_cohort", [1] * 3, start=40,
                                                              production_regime="missing")])
    unresolved = units("fixed_cohort", [0], start=60, production_regime="pre_pooled_mdp")
    unresolved[["joint_resolved", "resolved_disagreement"]] = False
    unresolved["delta"] = pd.NA
    res = stream(pd.concat([frame, unresolved], ignore_index=True))
    cells = res["cells"].set_index("production_regime")
    assert len(cells) == 3 and cells.loc["post_bpm", "testable"] and not cells.loc["missing", "testable"]
    assert cells.loc["missing", "n_resolved_disagreement"] == 3 and pd.isna(cells.loc["missing", "q_BH"])
    assert pd.isna(cells.loc["missing", "p_one_sided_positive"]) and cells.loc["missing", "mean_delta"] == 1.0
    assert cells.loc["pre_pooled_mdp", "n_units"] == 1 and cells.loc["pre_pooled_mdp", "n_resolved_disagreement"] == 0
    assert cells.loc["post_bpm", "test"] == "exact_one_sided_paired_sign_test"
    assert res["summary"]["n_testable_cells"] == 1 and res["summary"]["n_nonempty_cells"] == 3


def test_bh_survivor_that_fails_by_is_labelled_bh_only_exploratory_and_both_cohorts_share_the_family():
    fixed = units("fixed_cohort", [1] * 20 + [-1] * 10)                   # one-sided sign p = 0.0494
    tracked = units("all_tracked", [0] * 15)                              # p = 1, direction non-negative
    res = stream(pd.concat([fixed, tracked], ignore_index=True), records=[mech(cell_of(fixed))])
    row = fixed_row(res)
    assert row["q_BH"] == pytest.approx(0.0987, abs=1e-4) and row["q_BY"] == pytest.approx(0.1481, abs=1e-4)
    assert row["c3_bh"] == "pass" and row["robustness"] == "BH_only_exploratory"
    assert row["nominated"] == True  # noqa: E712
    assert res["summary"]["n_testable_cells"] == 2 and res["summary"]["outcome_state"] == "feature_hypothesis_nominated"


# --- condition 4 -----------------------------------------------------------------------------------------------------

def test_negative_sparse_all_tracked_support_fails_direction_and_blocks_nomination():
    fixed = units("fixed_cohort", [1] * 30)
    tracked = units("all_tracked", [-1] * 10)                              # n < 15: still direction evidence
    res = stream(pd.concat([fixed, tracked], ignore_index=True), records=[mech(cell_of(fixed))])
    row = fixed_row(res)
    assert row["c4_all_tracked_direction"] == "fail" and row["all_tracked_n_resolved_disagreement"] == 10
    assert row["nominated"] == False and res["summary"]["outcome_state"] == "no_actionable_mechanism"  # noqa: E712


def test_absent_all_tracked_support_is_unknown_never_passage():
    fixed = units("fixed_cohort", [1] * 30)
    res = stream(fixed, records=[mech(cell_of(fixed))])
    row = fixed_row(res)
    assert row["c4_all_tracked_direction"] == "unknown" and row["nominated"] == False  # noqa: E712
    assert res["summary"]["fixed_testable_condition_status"]["c4_all_tracked_direction"] == {"unknown": 1}


def test_missing_valued_keys_match_consistently_across_cohorts():
    fixed = units("fixed_cohort", [1] * 30, production_regime="missing")
    tracked = units("all_tracked", [1] * 2, production_regime="missing")
    row = fixed_row(stream(pd.concat([fixed, tracked], ignore_index=True), records=[mech(cell_of(fixed))]))
    assert row["c4_all_tracked_direction"] == "pass" and row["nominated"] == True  # noqa: E712


# --- condition 5 -----------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("record, status", [
    (None, "unknown"),
    ({"mechanism": ""}, "unknown"),
    ({"operational_variables": []}, "unknown"),
    ({"lock_evidence": {}}, "unknown"),
    ({"lock_evidence": {"production_projected_lineup": {"status": "available_at_lock", "evidence": ""}}}, "unknown"),
    ({"lock_evidence": {"production_projected_lineup": {"status": "not_available_at_lock", "evidence": "x"}}}, "fail"),
    ({"operational_variables": ["consensus_pick_share_bin"],
      "lock_evidence": {"consensus_pick_share_bin": {"status": "available_at_lock", "evidence": "x"}}}, "fail"),
    ({}, "pass"),
])
def test_condition_5_needs_an_explicit_lock_available_mechanism_record(record, status):
    fixed = units("fixed_cohort", [1] * 30)
    tracked = units("all_tracked", [1] * 15)
    records = [] if record is None else [mech(cell_of(fixed), **record)]
    row = fixed_row(stream(pd.concat([fixed, tracked], ignore_index=True), records=records))
    assert row["c5_lock_available_mechanism"] == status
    assert (row["c1_min_units"], row["c2_min_lift"], row["c3_bh"], row["c4_all_tracked_direction"]) == ("pass",) * 4
    assert row["nominated"] == (status == "pass")


def test_a_record_for_a_different_cell_does_not_count_and_two_records_conflict():
    fixed = units("fixed_cohort", [1] * 30)
    tracked = units("all_tracked", [1] * 15)
    frame = pd.concat([fixed, tracked], ignore_index=True)
    other = {**cell_of(fixed), "production_regime": "pre_pooled_mdp"}
    assert fixed_row(stream(frame, records=[mech(other)]))["c5_lock_available_mechanism"] == "unknown"
    two = [mech(cell_of(fixed)), mech(cell_of(fixed), mechanism="another story")]
    row = fixed_row(stream(frame, records=two))
    assert row["c5_lock_available_mechanism"] == "unknown" and row["c5_reason"] == "conflicting_mechanism_records"


def test_lift_threshold_is_exact_at_five_points():
    fixed = units("fixed_cohort", [1] * 21 + [-1] * 19)                    # 2/40 = 0.05 exactly
    assert fixed_row(stream(fixed))["c2_min_lift"] == "pass"
    below = units("fixed_cohort", [1] * 20 + [-1] * 19 + [0])              # 1/40
    assert fixed_row(stream(below))["c2_min_lift"] == "fail"


# --- streams ---------------------------------------------------------------------------------------------------------

def test_tie_excluded_sensitivity_recomputes_its_own_family_and_reports_support_losses():
    tie = [True] * 20 + [False] * 10
    fixed = units("fixed_cohort", [1] * 30, tie=tie)
    tracked = units("all_tracked", [1] * 15)
    frame = pd.concat([fixed, tracked], ignore_index=True)
    primary = stream(frame, records=[mech(cell_of(fixed))])
    sens = stream(inf.tie_excluded(frame), records=[mech(cell_of(fixed))], can_nominate=False,
                  name="tie_excluded_sensitivity")
    assert fixed_row(primary)["nominated"] == True  # noqa: E712
    sens_cells = sens["cells"]
    sfixed = sens_cells[sens_cells["cohort"] == "fixed_cohort"].iloc[0]
    assert sfixed["n_resolved_disagreement"] == 10 and not sfixed["testable"] and sfixed["nominated"] == False  # noqa
    assert sens["summary"]["outcome_state"] == "diagnostic_cannot_nominate"
    comp = inf.compare_streams(primary["cells"], sens["cells"])
    assert len(comp) == 1 and comp[0]["passes_sensitivity"] is False
    assert comp[0]["losses"] == ["support"] and comp[0]["primary_nominated"] is True


def test_sensitivity_losses_distinguish_effect_and_q_value():
    fixed = units("fixed_cohort", [1] * 30 + [-1] * 10, tie=[True] * 20 + [False] * 20)
    tracked = units("all_tracked", [1] * 15)
    frame = pd.concat([fixed, tracked], ignore_index=True)
    primary = stream(frame)
    sens = stream(inf.tie_excluded(frame), can_nominate=False, name="tie_excluded_sensitivity")
    comp = inf.compare_streams(primary["cells"], sens["cells"])
    assert comp[0]["losses"] == ["support", "effect", "q_value"]           # 10 +1 / 10 -1 left: n 20, lift 0


def test_an_exploratory_or_sensitivity_stream_never_nominates_even_when_all_five_pass():
    fixed = units("fixed_cohort", [1] * 30)
    tracked = units("all_tracked", [1] * 15)
    res = stream(pd.concat([fixed, tracked], ignore_index=True), records=[mech(cell_of(fixed))], can_nominate=False)
    row = fixed_row(res)
    assert row["all_five_conditions"] == True and row["nominated"] == False  # noqa: E712
    assert res["summary"]["n_nominated"] == 0


def test_no_testable_cells_is_power_limited():
    res = stream(units("fixed_cohort", [1] * 5))
    assert res["summary"]["outcome_state"] == "power_limited_no_testable_cells"


def test_mechanism_record_file_schema(tmp_path):
    import json
    path = tmp_path / "m.json"
    path.write_text(json.dumps({"schema": inf.MECHANISM_SCHEMA, "records": [{"cell": {}}]}))
    records, meta = inf.load_mechanism_records(path)
    assert len(records) == 1 and meta["n_records"] == 1
    assert inf.load_mechanism_records(None) == ([], {"source": "none_supplied", "n_records": 0})
    path.write_text(json.dumps({"schema": "nope", "records": []}))
    with pytest.raises(ValueError, match="schema"):
        inf.load_mechanism_records(path)
