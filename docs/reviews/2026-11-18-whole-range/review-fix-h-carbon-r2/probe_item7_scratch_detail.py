"""Detailed scratch-change verification with an unmutated baseline control first + walk.py R4 mutations."""
from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]

from probe_item7_scratch_changes import MUTATIONS


WALK_MUTATIONS = [
    (
        "walk_drop_days_effective_check",
        "scripts/audit/season_walk/walk.py",
        "if (longest or days_effective is not None) and days_effective != min(expected, horizon):",
        "if days_effective is not None and days_effective != min(expected, horizon):",
        ["tests/scripts/season_walk/test_season_walk.py"],
    ),
    (
        "walk_drop_source_check",
        "scripts/audit/season_walk/walk.py",
        'if source != "mdp" and (longest or source is not None):',
        'if source not in (None, "mdp"):',
        ["tests/scripts/season_walk/test_season_walk.py"],
    ),
    (
        "walk_drop_missing_artifacts_check",
        "scripts/audit/season_walk/walk.py",
        '            if not art:\n                problems.append(f"{where}: policy artifacts missing")\n',
        "",
        ["tests/scripts/season_walk/test_season_walk.py"],
    ),
    (
        "walk_drop_longest_action_check",
        "scripts/audit/season_walk/walk.py",
        '            if r.get("action") != want_action:\n                problems.append(f"{where}: longest-streak action {r.get(\'action\')!r}, expected {want_action!r}")\n',
        "",
        ["tests/scripts/season_walk/test_season_walk.py"],
    ),
]


def run_pytest(scratch_root: Path, test_files: list[str]) -> subprocess.CompletedProcess:
    # Override addopts pythonpath=src via -o pythonpath= so scratch_root/src wins over repo/src!
    env = dict(os.environ, PYTHONPATH=f"{scratch_root / 'src'}:{scratch_root}")
    cmd = [
        sys.executable, "-c",
        (
            "import sys; "
            f"sys.path[:0] = [{str(scratch_root / 'src')!r}, {str(scratch_root)!r}]; "
            "import pytest; "
            f"raise SystemExit(pytest.main(['-q', '--tb=line', '-x', '-o', 'pythonpath=', "
            f"'-k', 'not test_legacy_parity_against_archived_base', *{test_files!r}]))"
        ),
    ]
    return subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=30)


def main() -> None:
    with tempfile.TemporaryDirectory(dir=HERE, prefix="scratch-verified-") as tmp:
        sroot = Path(tmp)
        shutil.copytree(REPO / "src", sroot / "src")
        shutil.copytree(REPO / "scripts", sroot / "scripts")

        # 0. Verify unmutated scratch copy passes 100% AND imports from scratch_root!
        verify_imp = subprocess.run(
            [
                sys.executable, "-c",
                f"import sys; sys.path[:0] = [{str(sroot / 'src')!r}, {str(sroot)!r}]; "
                "import bts, scripts.audit.season_walk.walk as w; "
                "print(bts.__file__); print(w.__file__)",
            ],
            cwd=REPO, capture_output=True, text=True, check=True, timeout=10,
        ).stdout.splitlines()
        assert Path(verify_imp[0]).resolve() == (sroot / "src/bts/__init__.py").resolve(), verify_imp
        assert Path(verify_imp[1]).resolve() == (sroot / "scripts/audit/season_walk/walk.py").resolve(), verify_imp

        for baseline_suite in (["tests/test_wr_h_longest.py"], ["tests/simulate/test_tail_policy.py"], ["tests/scripts/season_walk/test_season_walk.py"]):
            base_res = run_pytest(sroot, baseline_suite)
            assert base_res.returncode == 0, f"Baseline failed for {baseline_suite}:\n{base_res.stdout}\n{base_res.stderr}"
        print("OK: Unmutated scratch baseline passed all 3 suites with verified scratch imports")

        caught, missed = [], []
        for name, rel_path, target, replacement, test_files in MUTATIONS + WALK_MUTATIONS:
            fpath = sroot / rel_path
            orig = fpath.read_text()
            assert target in orig, f"{name}: target missing"
            fpath.write_text(orig.replace(target, replacement, 1))
            try:
                proc = run_pytest(sroot, test_files)
                if proc.returncode != 0:
                    first_fail = next((ln for ln in proc.stdout.splitlines() if "FAILED" in ln or "Error" in ln), proc.stdout.splitlines()[-1] if proc.stdout else "")
                    caught.append(name)
                    print(f"CAUGHT {name}: {first_fail[:120]}")
                else:
                    # Also check test_strategy.py and test_daily_decision_v3.py before declaring missed across all 10 suites
                    proc2 = run_pytest(sroot, ["tests/test_strategy.py", "tests/test_daily_decision_v3.py", "tests/test_scheduler_tail_objective.py"])
                    if proc2.returncode != 0:
                        caught.append(name)
                        print(f"CAUGHT (secondary suite) {name}")
                    else:
                        missed.append(name)
                        print(f"MISSED {name} (no test failed!)")
            finally:
                fpath.write_text(orig)

        print(f"\nVerified Summary: caught={len(caught)}/{len(MUTATIONS) + len(WALK_MUTATIONS)}, missed={missed}")


if __name__ == "__main__":
    main()
