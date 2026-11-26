"""Tests for scripts/inventory_validation_splits.py."""
from __future__ import annotations

import importlib.util
import ast
from pathlib import Path


SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


def _load_inventory_module():
    spec = importlib.util.spec_from_file_location(
        "inventory_validation_splits",
        str(SCRIPTS_DIR / "inventory_validation_splits.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_mod = _load_inventory_module()


def test_inventory_has_schema_and_paths():
    inventory = _mod.build_inventory(generated_at="2026-05-06T00:00:00Z")
    assert inventory["schema_version"] == "nested_cv_lockbox_inventory_v1"
    assert inventory["generated_at"] == "2026-05-06T00:00:00Z"
    assert inventory["paths"]


def test_each_path_has_required_fields_and_concerns():
    inventory = _mod.build_inventory(generated_at="2026-05-06T00:00:00Z")
    required = set(inventory["required_path_fields"])
    allowed_next_actions = _mod.NEXT_ACTIONS
    for path in inventory["paths"]:
        assert required <= set(path)
        assert path["next_action"] in allowed_next_actions
        assert path["concerns"]
        assert all("path" in item and "exists" in item for item in path["evidence"])


def test_manifest_bound_validators_are_marked_lockbox_enforced():
    inventory = _mod.build_inventory(generated_at="2026-05-06T00:00:00Z")
    by_id = {path["id"]: path for path in inventory["paths"]}
    for path_id in [
        "validate_split_manifest",
        "validate_scorecard_manifest",
        "validate_conformal_gate",
        "validate_policy_value_eval",
        "validate_rare_event_ce_is",
    ]:
        assert by_id[path_id]["lockbox_state"] == "manifest_enforced"


def test_seed_axis_loo_is_not_treated_as_temporal_cv():
    inventory = _mod.build_inventory(generated_at="2026-05-06T00:00:00Z")
    by_id = {path["id"]: path for path in inventory["paths"]}
    pooled = by_id["pooled_policy_ab"]
    assert pooled["axis"] == "seed"
    assert pooled["outer_split_type"] == "leave_one_seed_out"
    assert pooled["lockbox_state"] == "not_temporal_lockbox_applicable"


def test_summary_counts_match_paths():
    inventory = _mod.build_inventory(generated_at="2026-05-06T00:00:00Z")
    paths = inventory["paths"]
    assert inventory["summary"]["n_paths"] == len(paths)
    assert inventory["summary"]["manifest_lockbox_enforced"] == sum(
        path["lockbox_state"] == "manifest_enforced" for path in paths
    )


def test_python_entrypoint_symbols_exist():
    inventory = _mod.build_inventory(generated_at="2026-05-06T00:00:00Z")
    repo = Path(__file__).parent.parent.parent
    for path in inventory["paths"]:
        entrypoint = path["entrypoint"]
        file_part, sep, symbol = entrypoint.partition(":")
        source_path = repo / file_part
        assert source_path.exists(), f"{path['id']} entrypoint file missing"
        if not sep or not file_part.endswith(".py"):
            continue
        tree = ast.parse(source_path.read_text())
        functions = {
            node.name
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        assert symbol in functions, f"{path['id']} entrypoint symbol missing"
