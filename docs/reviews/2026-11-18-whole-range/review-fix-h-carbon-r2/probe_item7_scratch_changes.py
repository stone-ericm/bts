"""Adversarial probe for Item 7: A10 test-diff audit and scratch mutations against the test suite."""
from __future__ import annotations

from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


MUTATIONS: list[tuple[str, str, str, str, list[str]]] = [
    (
        "tie_double_first",
        "src/bts/simulate/tail_policy.py",
        'return ("single", "double", "skip")[values.index(gain)]',
        'return "double" if values[1] == gain else ("single" if values[0] == gain else "skip")',
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "tie_skip_first",
        "src/bts/simulate/tail_policy.py",
        'return ("single", "double", "skip")[values.index(gain)]',
        'return "skip" if values[2] == gain else ("single" if values[0] == gain else "double")',
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "swap_rule_operands",
        "src/bts/simulate/tail_policy.py",
        "values = (p_hit * (1.0 + gain), p_both * (2.0 + gain), gain)",
        "values = (p_both * (2.0 + gain), p_hit * (1.0 + gain), gain)",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_one_bin_check",
        "src/bts/strategy.py",
        '        if tail.n_bins != 1:\n            raise ValueError("tail policy must have one quality bin")\n',
        "",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_positive_horizon_clause",
        "src/bts/strategy.py",
        "if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:",
        "if not isinstance(horizon, int) or isinstance(horizon, bool):",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_horizon_bool_clause",
        "src/bts/strategy.py",
        "if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:",
        "if not isinstance(horizon, int) or horizon < 1:",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_horizon_int_clause",
        "src/bts/strategy.py",
        "if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:",
        "if isinstance(horizon, bool) or horizon < 1:",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_boundaries_list_clause",
        "src/bts/strategy.py",
        '    if not isinstance(boundaries, list):\n        raise ValueError("base policy boundaries must be a list")\n',
        "    boundaries = boundaries if isinstance(boundaries, list) else [0.7, 0.75, 0.8, 0.85]\n",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_ndim_4_clause",
        "src/bts/strategy.py",
        '    if not isinstance(table, np.ndarray) or table.ndim != 4:\n        raise ValueError("base policy table must be a four-dimensional array")\n',
        "",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_table_isinstance_subclause",
        "src/bts/strategy.py",
        "if not isinstance(table, np.ndarray) or table.ndim != 4:",
        "if table.ndim != 4:",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_int_dtype_clause",
        "src/bts/strategy.py",
        '    if table.dtype.kind not in {"i", "u"}:\n        raise ValueError("base policy table must have an integer dtype")\n',
        "",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_streak_57_shape_clause",
        "src/bts/strategy.py",
        "if table.shape[0] < 57 or table.shape[1:] != (horizon + 1, 2, len(boundaries) + 1):",
        "if table.shape[1:] != (horizon + 1, 2, len(boundaries) + 1):",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_day_axis_shape_clause",
        "src/bts/strategy.py",
        "if table.shape[0] < 57 or table.shape[1:] != (horizon + 1, 2, len(boundaries) + 1):",
        "if table.shape[0] < 57 or table.shape[2:] != (2, len(boundaries) + 1):",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_saver_axis_shape_clause",
        "src/bts/strategy.py",
        "if table.shape[0] < 57 or table.shape[1:] != (horizon + 1, 2, len(boundaries) + 1):",
        "if table.shape[0] < 57 or (table.shape[1], table.shape[3]) != (horizon + 1, len(boundaries) + 1):",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_bin_axis_shape_clause",
        "src/bts/strategy.py",
        "if table.shape[0] < 57 or table.shape[1:] != (horizon + 1, 2, len(boundaries) + 1):",
        "if table.shape[0] < 57 or table.shape[1:3] != (horizon + 1, 2):",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_table_min_clause",
        "src/bts/strategy.py",
        "if table.min() < 0 or table.max() > 2:",
        "if table.max() > 2:",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_table_max_clause",
        "src/bts/strategy.py",
        "if table.min() < 0 or table.max() > 2:",
        "if table.min() < 0:",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "use_base_horizon",
        "src/bts/strategy.py",
        "    if mdp_objective(streak, days) == OBJECTIVE_LONGEST:\n        return OBJECTIVE_LONGEST, min(days, LONGEST_STREAK_HORIZON)",
        '    if mdp_objective(streak, days) == OBJECTIVE_LONGEST:\n        return OBJECTIVE_LONGEST, min(days, int((mdp or {}).get("season_length") or 180))',
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "return_skip_when_degraded",
        "src/bts/strategy.py",
        'return "single", reason, None',
        'return "skip", reason, None',
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "drop_new_enum_value",
        "src/bts/daily_decision.py",
        'OBJECTIVES = ("reach57", "emax_season_best", "longest_streak")',
        'OBJECTIVES = ("reach57", "emax_season_best")',
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "load_artifacts_before_regime",
        "src/bts/strategy.py",
        '    if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n        try:\n            mdp = _load_mdp()\n        except Exception as exc:\n            mdp = {"base_error": f"{type(exc).__name__}: {exc}"}\n    else:\n        mdp = _load_mdp()',
        '    try:\n        mdp = _load_mdp()\n    except Exception as exc:\n        if _regime(None, streak, date)[0] != OBJECTIVE_LONGEST:\n            raise\n        mdp = {"base_error": f"{type(exc).__name__}: {exc}"}',
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "make_bar_none",
        "src/bts/strategy.py",
        "        if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n            return 0.0\n",
        "",
        ["tests/test_wr_h_longest.py"],
    ),
    (
        "dispatch_at_zero_days",
        "src/bts/simulate/tail_policy.py",
        "if streak >= LONGEST_STREAK_FROM and days_remaining > 0:",
        "if streak >= LONGEST_STREAK_FROM and days_remaining >= 0:",
        ["tests/simulate/test_tail_policy.py"],
    ),
]


def check_a10_test_diffs() -> None:
    changed = subprocess.run(
        ["git", "diff", "--name-status", "0a2bec10", "84805852", "--", "tests"],
        cwd=REPO, capture_output=True, text=True, check=True, timeout=10,
    ).stdout.strip().splitlines()
    by_file = dict(line.split("\t")[::-1] for line in changed)
    assert set(by_file) == {
        "tests/test_wr_h_longest.py",
        "tests/scripts/season_walk/test_season_walk.py",
        "tests/simulate/test_tail_policy.py",
        "tests/test_wr_g_guard.py",
    }, by_file
    assert by_file["tests/test_wr_h_longest.py"] == "A"

    # test_season_walk.py must only add lines (0 deletions)
    numstat = subprocess.run(
        ["git", "diff", "--numstat", "0a2bec10", "84805852", "--", "tests/scripts/season_walk/test_season_walk.py"],
        cwd=REPO, capture_output=True, text=True, check=True, timeout=10,
    ).stdout.split()
    assert int(numstat[1]) == 0, f"Expected 0 deleted lines in test_season_walk.py, got {numstat}"
    print("OK: A10 test diff file set and zero deletions in test_season_walk.py verified")


def main() -> None:
    check_a10_test_diffs()
    caught = []
    missed = []
    with tempfile.TemporaryDirectory(dir=HERE, prefix="scratch-mut-") as tmp:
        scratch_src = Path(tmp) / "src"
        shutil.copytree(REPO / "src", scratch_src)

        for name, rel_path, target, replacement, test_files in MUTATIONS:
            fpath = Path(tmp) / rel_path
            orig = fpath.read_text()
            assert target in orig, f"{name}: target not found in {rel_path}"
            fpath.write_text(orig.replace(target, replacement, 1))
            try:
                env = dict(
                    **{k: v for k, v in subprocess.os.environ.items()},
                    PYTHONPATH=f"{scratch_src}:{REPO}",
                )
                cmd = [
                    sys.executable, "-m", "pytest", "-q", "--no-header", "-x",
                    "-k", "not test_legacy_parity_against_archived_base",
                    *test_files,
                ]
                proc = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=30)
                if proc.returncode != 0:
                    caught.append(name)
                    print(f"CAUGHT {name}")
                else:
                    missed.append(name)
                    print(f"MISSED {name} (all {test_files} passed!)")
            finally:
                fpath.write_text(orig)

    print(f"\nSummary: caught={len(caught)}/{len(MUTATIONS)}, missed={missed}")


if __name__ == "__main__":
    main()
