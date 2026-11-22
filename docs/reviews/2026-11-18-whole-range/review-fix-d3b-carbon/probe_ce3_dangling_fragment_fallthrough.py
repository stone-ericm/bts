#!/usr/bin/env python3
"""Item 1 (K12) & Item 2 (The Class): CE3 — not-found legacy unit with a dangling fragment symlink falls through into prove stop's loaded branch and prints verified."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile

REPO = Path(__file__).resolve().parents[2]
SCRATCH_PARENT = REPO / ".codex-review/wr-d3b/tmp"
SPEC = importlib.util.spec_from_file_location("workflow_tests", REPO / "tests/test_deploy_workflow.py")
assert SPEC and SPEC.loader
tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tests)
DEPLOY, RECOVER = tests.scripts_and_steps()[0]


def main() -> None:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    out = {}

    # Exact setup of test_d3b_unproved_legacy_absence_is_unverified (lines 1790-1800 of test_deploy_workflow.py),
    # with scenario="legacy-vanishing" and fragment=True (dangling symlink /etc/systemd/system/bts-scheduler.service -> missing-unit-file):
    for mode, is_recovery in [("deploy", False), ("recovery", True)]:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            box = tests.box.__wrapped__(Path(td))
            tests.d3b_legacy_group(box, 0)
            units = box / "etc/systemd/system"
            units.mkdir(parents=True)
            (units / "bts-scheduler.service").symlink_to(box / "missing-unit-file")
            res, calls = tests.d3b_run_contained(box, is_recovery, "legacy-vanishing")
            out[f"CE3_{mode}_legacy_vanishing_with_dangling_symlink"] = {
                "dangling_symlink_present": (units / "bts-scheduler.service").is_symlink() and not (units / "bts-scheduler.service").exists(),
                "stdout": res.stdout.splitlines(),
                "record": (box / "run/bts-deploy/attempt").read_text().strip(),
            }
            # On 90a9b6f0, prove stop falls through from `if [[ $legacy == not-found ]] && no_fragment` into `elif (( legacy_group_status == 0 )) && check_stopped ...`
            # without checking `[[ $legacy == loaded ]]`, falsely printing `Scheduler stopped: verified.`!
            assert "Scheduler stopped: verified." in res.stdout

    # Verify smallest fix for CE3 (`elif [[ $legacy == loaded ]] && (( legacy_group_status == 0 )) &&`):
    old_clause = "elif (( legacy_group_status == 0 )) && check_stopped"
    new_clause = "elif [[ $legacy == loaded ]] && (( legacy_group_status == 0 )) && check_stopped"
    assert DEPLOY.count(old_clause) == 1 and RECOVER.count(old_clause) == 1
    fixed_deploy = DEPLOY.replace(old_clause, new_clause)
    fixed_recover = RECOVER.replace(old_clause, new_clause)

    for mode, script in [("fixed_deploy", fixed_deploy), ("fixed_recovery", fixed_recover)]:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            box = tests.box.__wrapped__(Path(td))
            tests.d3b_legacy_group(box, 0)
            units = box / "etc/systemd/system"
            units.mkdir(parents=True)
            (units / "bts-scheduler.service").symlink_to(box / "missing-unit-file")
            runtime = tests.d3b_box(box)
            if script is fixed_recover:
                (box / "head").write_text(tests.TARGET)
                (runtime / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n")
                res, _ = tests.run_box_script(box, script, "legacy-vanishing")
            else:
                res, _ = tests.run_box_script(box, script, "legacy-vanishing", fault="reset:17")
            assert "Scheduler stopped: unverified." in res.stdout
            out[mode] = "Scheduler stopped: unverified. (PASS)"

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
