#!/usr/bin/env python3
"""Inventory BTS validation split and lockbox usage.

This is a methodology inventory, not a validator. It records the split axis,
outer/inner temporal split state, lockbox enforcement, and current gating
surface for the main BTS validation and audit paths.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


SCHEMA_VERSION = "nested_cv_lockbox_inventory_v1"
DEFAULT_OUTPUT = Path(
    "data/validation/nested_cv_lockbox_inventory_2026-05-06.json"
)

NEXT_ACTIONS = {
    "compliant",
    "needs-lockbox-application",
    "needs-time-respecting-inner-split",
    "unclear-needs-investigation",
}

REQUIRED_PATH_FIELDS = (
    "id",
    "surface",
    "entrypoint",
    "axis",
    "outer_split_type",
    "inner_split_type",
    "lockbox_state",
    "gating_decision",
    "concerns",
    "next_action",
)


PATH_INVENTORY: list[dict[str, Any]] = [
    {
        "id": "validate_split_manifest",
        "surface": "bts validate split-manifest",
        "entrypoint": "src/bts/cli.py:split_manifest_cmd",
        "axis": "time",
        "outer_split_type": "rolling_origin_forward_chaining",
        "inner_split_type": "none_contract_artifact_only",
        "lockbox_state": "manifest_enforced",
        "gating_decision": "writes_manifest_no_model_gate",
        "concerns": [
            "rolling_origin only; symmetric blocked CV is explicitly deferred",
            "embargo_game_days is recorded but inactive in rolling_origin mode",
        ],
        "next_action": "compliant",
        "evidence": [
            "src/bts/validate/splits.py",
            "tests/validate/test_splits.py",
            "data/validation/split_manifest_conformal_2026-05-06.json",
        ],
    },
    {
        "id": "validate_scorecard_manifest",
        "surface": "bts validate scorecard --manifest",
        "entrypoint": "src/bts/validate/scorecard.py:compute_scorecard_over_manifest",
        "axis": "time",
        "outer_split_type": "manifest_fold_holdout_scorecards",
        "inner_split_type": "none",
        "lockbox_state": "manifest_enforced",
        "gating_decision": "per_fold_scorecards_aggregate_deferred",
        "concerns": [
            "aggregate fold uncertainty is deferred",
            "lockbox certification run has not been produced as a durable artifact",
        ],
        "next_action": "compliant",
        "evidence": [
            "src/bts/validate/scorecard.py",
            "tests/validate/test_scorecard_cli.py",
        ],
    },
    {
        "id": "validate_scorecard_non_manifest",
        "surface": "bts validate scorecard",
        "entrypoint": "src/bts/cli.py:scorecard",
        "axis": "descriptive_all_profiles",
        "outer_split_type": "single_shot_all_loaded_backtests",
        "inner_split_type": "none",
        "lockbox_state": "not_manifest_bound",
        "gating_decision": "descriptive_scorecard_and_optional_diff",
        "concerns": [
            "loads every backtest_*.parquet in profiles-dir",
            "proper_scoring block is descriptive and not lockbox-bound in this mode",
        ],
        "next_action": "needs-lockbox-application",
        "evidence": [
            "src/bts/cli.py",
            "src/bts/validate/scorecard.py",
            "src/bts/validate/proper_scoring.py",
        ],
    },
    {
        "id": "validate_conformal_gate",
        "surface": "bts validate conformal-gate",
        "entrypoint": "src/bts/validate/conformal_gate.py:run_gate_matrix",
        "axis": "time",
        "outer_split_type": "manifest_rolling_origin",
        "inner_split_type": "fit_calibrator_on_fold_train",
        "lockbox_state": "manifest_enforced",
        "gating_decision": "method_alpha_ship_set",
        "concerns": [
            "current durable run has ship_set=[] and NO_PRODUCTION_DEPLOY",
            "binary-y deployment also needs #12 calibration evidence on selectable rows",
        ],
        "next_action": "compliant",
        "evidence": [
            "src/bts/validate/conformal_gate.py",
            "tests/validate/test_conformal_gate.py",
            "data/validation/conformal_gate_v2_2026-05-06.json",
        ],
    },
    {
        "id": "validate_policy_value_eval",
        "surface": "bts validate policy-value-eval",
        "entrypoint": "src/bts/validate/ope_eval.py:evaluate_target_policy_on_manifest",
        "axis": "time",
        "outer_split_type": "manifest_rolling_origin",
        "inner_split_type": "solve_target_policy_on_fold_train",
        "lockbox_state": "manifest_enforced",
        "gating_decision": "per_fold_policy_value_with_sparse_support_flag",
        "concerns": [
            "aggregate fold uncertainty is deferred",
            "prior real-data smoke had universal sparse holdout support",
        ],
        "next_action": "compliant",
        "evidence": [
            "src/bts/validate/ope_eval.py",
            "tests/validate/test_ope_eval.py",
        ],
    },
    {
        "id": "validate_rare_event_ce_is",
        "surface": "bts validate rare-event-ce-is",
        "entrypoint": "src/bts/validate/rare_event_mc_eval.py:evaluate_ceis_on_manifest",
        "axis": "time",
        "outer_split_type": "manifest_rolling_origin",
        "inner_split_type": "learn_theta_on_fold_train",
        "lockbox_state": "manifest_enforced",
        "gating_decision": "per_fold_diagnostic_flags_aggregate_deferred",
        "concerns": [
            "fixed-window estimand is not season P57",
            "aggregate fold uncertainty is deferred",
        ],
        "next_action": "compliant",
        "evidence": [
            "src/bts/validate/rare_event_mc_eval.py",
            "tests/validate/test_rare_event_mc_eval.py",
        ],
    },
    {
        "id": "validate_falsification_harness",
        "surface": "bts validate falsification-harness",
        "entrypoint": "scripts/run_falsification_harness.py:run_harness",
        "axis": "time",
        "outer_split_type": "latest_season_holdout_plus_loso_pipeline",
        "inner_split_type": "fold_local_dependence_and_policy_modes",
        "lockbox_state": "implicit_latest_season_holdout_not_manifest_lockbox",
        "gating_decision": "HEADLINE_DEFENDED_REDUCED_BROKEN_INCONCLUSIVE",
        "concerns": [
            "does not consume the #5 split manifest",
            "latest season holdout is not the same as an untouched lockbox",
            "threshold and dependence choices are harness-specific",
        ],
        "next_action": "needs-lockbox-application",
        "evidence": [
            "src/bts/cli.py",
            "scripts/run_falsification_harness.py",
            "src/bts/validate/ope.py",
        ],
    },
    {
        "id": "simulate_backtest_blend",
        "surface": "bts simulate backtest",
        "entrypoint": "src/bts/simulate/backtest_blend.py:blend_walk_forward",
        "axis": "time",
        "outer_split_type": "daily_walk_forward_within_test_season",
        "inner_split_type": "none_for_hyperparameter_or_feature_selection",
        "lockbox_state": "not_manifest_bound",
        "gating_decision": "writes_backtest_profiles_no_gate",
        "concerns": [
            "fixed model parameters are reused across evaluation seasons",
            "does not reserve a final lockbox segment by itself",
        ],
        "next_action": "needs-time-respecting-inner-split",
        "evidence": [
            "src/bts/simulate/backtest_blend.py",
            "src/bts/simulate/cli.py",
        ],
    },
    {
        "id": "experiment_runner",
        "surface": "bts experiment screen/select",
        "entrypoint": "src/bts/experiment/runner.py",
        "axis": "time",
        "outer_split_type": "walk_forward_over_configured_test_seasons",
        "inner_split_type": "none_for_feature_selection",
        "lockbox_state": "not_manifest_bound",
        "gating_decision": "phase1_phase2_pass_fail_on_test_seasons",
        "concerns": [
            "feature selection can be decided on the same test seasons used for scoring",
            "no time-respecting inner split separates tuning from outer evaluation",
            "no #5 lockbox enforcement in the runner path",
        ],
        "next_action": "needs-time-respecting-inner-split",
        "evidence": [
            "src/bts/experiment/runner.py",
            "src/bts/experiment/cli.py",
            "scripts/audit_driver.py",
        ],
    },
    {
        "id": "pooled_policy_ab",
        "surface": "scripts/pooled_policy_ab.py",
        "entrypoint": "scripts/pooled_policy_ab.py",
        "axis": "seed",
        "outer_split_type": "leave_one_seed_out",
        "inner_split_type": "none_temporal",
        "lockbox_state": "not_temporal_lockbox_applicable",
        "gating_decision": "pooled_vs_production_policy_gap_screen",
        "concerns": [
            "leave-one-out is across seeds, not calendar time",
            "does not answer temporal leakage or lockbox questions",
        ],
        "next_action": "unclear-needs-investigation",
        "evidence": [
            "scripts/pooled_policy_ab.py",
            "scripts/pooled_policy_gap_ci.py",
        ],
    },
]


def _with_evidence_state(entry: dict[str, Any]) -> dict[str, Any]:
    out = dict(entry)
    out["evidence"] = [
        {"path": p, "exists": Path(p).exists()}
        for p in entry.get("evidence", [])
    ]
    return out


def _validate_entry(entry: dict[str, Any]) -> None:
    missing = [field for field in REQUIRED_PATH_FIELDS if field not in entry]
    if missing:
        raise ValueError(f"{entry.get('id', '<unknown>')} missing {missing}")
    if entry["next_action"] not in NEXT_ACTIONS:
        raise ValueError(
            f"{entry['id']} has unsupported next_action={entry['next_action']!r}"
        )
    if not isinstance(entry["concerns"], list) or not entry["concerns"]:
        raise ValueError(f"{entry['id']} must have at least one concern")


def build_inventory(generated_at: str | None = None) -> dict[str, Any]:
    if generated_at is None:
        generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    paths = []
    for entry in PATH_INVENTORY:
        _validate_entry(entry)
        paths.append(_with_evidence_state(entry))

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": generated_at,
        "generator": "scripts/inventory_validation_splits.py",
        "required_path_fields": list(REQUIRED_PATH_FIELDS),
        "paths": paths,
        "summary": {
            "n_paths": len(paths),
            "manifest_lockbox_enforced": sum(
                p["lockbox_state"] == "manifest_enforced" for p in paths
            ),
            "needs_lockbox_application": sum(
                p["next_action"] == "needs-lockbox-application" for p in paths
            ),
            "needs_time_respecting_inner_split": sum(
                p["next_action"] == "needs-time-respecting-inner-split"
                for p in paths
            ),
            "non_temporal_seed_axis": sum(p["axis"] == "seed" for p in paths),
        },
        "notes": [
            "Out-of-fold or leave-one-out evidence is not equivalent to a lockbox.",
            "Seed-axis LOO is recorded separately from temporal CV.",
            "This inventory does not implement fixes; it scopes follow-up work.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=DEFAULT_OUTPUT)
    args = ap.parse_args()

    inventory = build_inventory()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(inventory, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
