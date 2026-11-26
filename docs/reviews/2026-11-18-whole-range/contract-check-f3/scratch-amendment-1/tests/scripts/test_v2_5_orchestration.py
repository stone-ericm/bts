"""Smoke tests for v2.5 cloud orchestration scripts.

Tests:
    1. Cell→flag mapping: all 4 ablation cells map to the correct harness flags.
    2. --dry-run mode: v2.5_provision.sh exits 0 with env assertions (subprocess).
    3. v2.5_run_ablations.py --dry-run exits 0 and validates flag mapping (subprocess).
    4. build_harness_cmd: flag strings appear verbatim in the generated command.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import importlib.util

import pytest

SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"

# v2.5_run_ablations.py has a dot in its name so it can't be imported via the
# normal 'import' statement. Use importlib to load it explicitly.
def _load_run_ablations():
    spec = importlib.util.spec_from_file_location(
        "v2_5_run_ablations",
        str(SCRIPTS_DIR / "v2_5_run_ablations.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

_mod = _load_run_ablations()
CELL_FLAGS = _mod.CELL_FLAGS
build_harness_cmd = _mod.build_harness_cmd


# ---- Unit tests: flag mapping ----

EXPECTED_FLAGS = {
    # cell: (params_mode, rho_pair_mode, policy_mode)
    "010": ("pooled",     "per-bin", "global"),
    "001": ("pooled",     "scalar",  "per-fold"),
    "011": ("pooled",     "per-bin", "per-fold"),
    "101": ("fold-local", "scalar",  "per-fold"),
}


def test_all_cells_present():
    """All 4 ablation cells are defined in CELL_FLAGS."""
    assert set(CELL_FLAGS.keys()) == {"010", "001", "011", "101"}


@pytest.mark.parametrize("cell", list(EXPECTED_FLAGS.keys()))
def test_cell_flag_values(cell):
    """Each cell maps to the expected (params_mode, rho_pair_mode, policy_mode) triple."""
    expected = EXPECTED_FLAGS[cell]
    actual = CELL_FLAGS[cell]
    assert actual == expected, (
        f"Cell {cell}: expected {expected}, got {actual}"
    )


@pytest.mark.parametrize("cell", list(EXPECTED_FLAGS.keys()))
def test_build_harness_cmd_contains_flags(cell):
    """build_harness_cmd produces a command string with the correct --mode flags."""
    params_mode, rho_pair_mode, policy_mode = CELL_FLAGS[cell]
    cmd = build_harness_cmd(cell, params_mode, rho_pair_mode, policy_mode)

    assert f"--params-mode {params_mode}" in cmd, (
        f"Cell {cell}: --params-mode {params_mode} not found in cmd"
    )
    assert f"--rho-pair-mode {rho_pair_mode}" in cmd, (
        f"Cell {cell}: --rho-pair-mode {rho_pair_mode} not found in cmd"
    )
    assert f"--policy-mode {policy_mode}" in cmd, (
        f"Cell {cell}: --policy-mode {policy_mode} not found in cmd"
    )


@pytest.mark.parametrize("cell", list(EXPECTED_FLAGS.keys()))
def test_build_harness_cmd_output_path(cell):
    """build_harness_cmd embeds the correct cell label in the output path."""
    params_mode, rho_pair_mode, policy_mode = CELL_FLAGS[cell]
    cmd = build_harness_cmd(cell, params_mode, rho_pair_mode, policy_mode)
    assert f"falsification_harness_v2.5_cell{cell}.json" in cmd


def test_v1_v2_baselines_not_in_cell_flags():
    """Cells 000 (v2 baseline) and 111 (v1 baseline) are intentionally absent."""
    assert "000" not in CELL_FLAGS, "Cell 000 (v2 baseline) should not be in CELL_FLAGS"
    assert "111" not in CELL_FLAGS, "Cell 111 (v1 baseline) should not be in CELL_FLAGS"


# ---- Subprocess smoke tests: --dry-run modes ----

@pytest.mark.slow
def test_run_ablations_dry_run():
    """v2.5_run_ablations.py --dry-run exits 0 and prints cell flag summary."""
    result = subprocess.run(
        [sys.executable, str(SCRIPTS_DIR / "v2_5_run_ablations.py"), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, (
        f"--dry-run failed (rc={result.returncode}):\n{result.stderr}"
    )
    # Should print all 4 cells
    for cell in EXPECTED_FLAGS:
        assert cell in result.stdout, f"Cell {cell} not mentioned in dry-run output"
    assert "OK" in result.stdout, "Expected 'OK' in dry-run output"


@pytest.mark.slow
def test_provision_dry_run(tmp_path):
    """v2.5_provision.sh --dry-run exits 0 without making API calls.

    Note: this test requires the Keychain to be accessible (macOS only) AND
    the vultr-api-token Keychain item to be present and readable. Skip
    cleanly when either precondition is missing — the script's explicit
    ERROR-and-exit path is exercised by direct CLI use, not this test.
    """
    import platform
    if platform.system() != "Darwin":
        pytest.skip("Keychain unavailable on non-macOS")

    # Preflight 1: confirm the Keychain item is actually retrievable. The
    # script's explicit `ERROR: vultr-api-token not found in Keychain` path
    # already covers direct CLI use after the `|| true` fix landed alongside
    # this test. This preflight keeps the success-path smoke from going red
    # when the local Keychain token is absent — without it, the test would
    # observe rc=1 with the explicit ERROR rather than the rc=0 dry-run
    # success it's actually asserting.
    keychain_check = subprocess.run(
        ["security", "find-generic-password", "-a", "claude-cli", "-s", "vultr-api-token", "-w"],
        capture_output=True,
        text=True,
        timeout=5,
    )
    if keychain_check.returncode != 0:
        pytest.skip(
            "vultr-api-token Keychain item not accessible "
            f"(rc={keychain_check.returncode}); see script docstring for setup"
        )

    # Preflight 2: the dry-run path requires data/simulation/*.parquet to
    # exist. The script's explicit "parquets missing" ERROR is exercised by
    # direct CLI use, not this test — the test is asserting dry-run success,
    # so skip when parquets aren't present.
    import glob
    worktree_root = SCRIPTS_DIR.parent
    if not glob.glob(str(worktree_root / "data" / "simulation" / "profiles_seed*_season*.parquet")):
        pytest.skip("data/simulation/ profile parquets not present; dry-run cannot succeed")
    if not glob.glob(str(worktree_root / "data" / "simulation" / "pa_predictions_seed*_season*.parquet")):
        pytest.skip("data/simulation/ pa-prediction parquets not present; dry-run cannot succeed")

    provision_script = SCRIPTS_DIR / "v2_5_provision.sh"
    assert provision_script.exists(), f"{provision_script} not found"

    # Run from the worktree root so relative paths resolve correctly.
    worktree_root = SCRIPTS_DIR.parent
    result = subprocess.run(
        ["bash", str(provision_script), "--dry-run"],
        capture_output=True,
        text=True,
        timeout=30,
        cwd=str(worktree_root),
    )
    assert result.returncode == 0, (
        f"provision --dry-run failed (rc={result.returncode}):\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
    assert "dry-run" in result.stdout.lower(), "Expected dry-run confirmation in output"
