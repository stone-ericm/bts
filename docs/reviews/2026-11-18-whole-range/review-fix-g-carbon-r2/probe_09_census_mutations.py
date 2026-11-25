"""Verify census scratch mutations with sitecustomize preloading so pytest imports SCRATCH/scripts ahead of rootdir."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRATCH = HERE / "scratch_census_mut"

CENSUS_MUTATIONS: list[tuple[str, str, str]] = [
    (
        "m16_census_accept_oob_bin",
        '            and 0 <= guard["from_bin"] < guard["bin"] < policy_table.shape[-1]',
        '            and 0 <= guard["from_bin"] < guard["bin"]',
    ),
    (
        "m17_census_accept_raw_not_skip",
        '        if guarded_play and raw_deployed == "skip":',
        "        if guarded_play:",
    ),
    (
        "m18_census_unguarded_play_excluded",
        '        if guarded_play and raw_deployed == "skip":',
        '        if (guarded_play or (guard is None and rec.get("scoreable") is True and rec.get("action") in ("single", "double"))) and raw_deployed == "skip":',
    ),
    (
        "m22_custom_census_weak_schema_check",
        '            and guard.keys() == {"rule", "bin", "from_bin", "table_action"}\n'
        '            and guard["rule"] == "monotone_skip_v1" and guard["table_action"] == "skip"\n'
        '            and type(guard["from_bin"]) is int and type(guard["bin"]) is int\n'
        '            and 0 <= guard["from_bin"] < guard["bin"] < policy_table.shape[-1]',
        '            and isinstance(guard.get("bin"), int) and guard["bin"] < policy_table.shape[-1]',
    ),
    (
        "m25_custom_census_drop_from_bin_lt_bin",
        '            and 0 <= guard["from_bin"] < guard["bin"] < policy_table.shape[-1]',
        '            and 0 <= guard["bin"] < policy_table.shape[-1]',
    ),
    (
        "m26_custom_census_isinstance_int_allows_bool",
        '            and type(guard["from_bin"]) is int and type(guard["bin"]) is int',
        '            and isinstance(guard["from_bin"], int) and isinstance(guard["bin"], int)',
    ),
]


def main() -> None:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    (SCRATCH / "scripts" / "audit").mkdir(parents=True)
    (SCRATCH / "scripts" / "__init__.py").write_text("")
    (SCRATCH / "scripts" / "audit" / "__init__.py").write_text("")
    target_py = SCRATCH / "scripts/audit/boundary_shadow_census.py"
    shutil.copy2(ROOT / "scripts/audit/boundary_shadow_census.py", target_py)

    # Preload scripts.audit from SCRATCH before pytest prepends rootdir to sys.path
    (SCRATCH / "sitecustomize.py").write_text(
        f"import sys\nsys.path.insert(0, {str(SCRATCH)!r})\nimport scripts.audit.boundary_shadow_census\n"
    )
    env = os.environ | {"PYTHONPATH": f"{SCRATCH}:{ROOT / 'src'}:{ROOT}"}

    # Verify boundary_shadow_census.__file__ resolves inside SCRATCH under pytest
    check = subprocess.run(
        [
            sys.executable, "-c",
            "import scripts.audit.boundary_shadow_census as m; print(m.__file__)",
        ],
        cwd=ROOT, check=True, capture_output=True, text=True, timeout=10, env=env,
    )
    assert Path(check.stdout.strip()).resolve() == target_py.resolve(), check.stdout

    original = target_py.read_text()
    results = {}
    for name, target, replacement in CENSUS_MUTATIONS:
        assert target in original, name
        try:
            target_py.write_text(original.replace(target, replacement, 1))
            proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest", "-q", "--tb=no", "--no-header",
                    "tests/test_boundary_shadow_census_g.py",
                    "tests/test_boundary_shadow_census.py",
                ],
                cwd=ROOT, capture_output=True, text=True, timeout=20, env=env,
            )
            results[name] = {
                "killed": proc.returncode != 0,
                "detail": (proc.stdout + proc.stderr).strip().splitlines()[-1],
            }
        finally:
            target_py.write_text(original)

    shutil.rmtree(SCRATCH, ignore_errors=True)
    (HERE / "probe_09_census_mutations.json").write_text(json.dumps(results, indent=2))
    print("PROBE_09_CENSUS_DONE", json.dumps(results))


if __name__ == "__main__":
    main()
