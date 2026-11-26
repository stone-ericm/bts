#!/usr/bin/env python3
"""Item 7 tail: Exact function-name check for the final 2 candidate survivors (own6 and own7), ~20 seconds."""
from __future__ import annotations

import importlib.util
from pathlib import Path
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "mut_targeted", REPO / ".codex-review/wr-d3b/probe_mutations_targeted.py"
)
assert SPEC and SPEC.loader
mod = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mod)


def main() -> None:
    by_name = {name: edits for name, edits, _ in mod.CANDIDATE_SURVIVORS}
    tail_checks = [
        (
            "own6_drop_st_uid_zero_in_lockfile_and_mark",
            by_name["own6_drop_st_uid_zero_in_lockfile_and_mark"],
            "test_round2_recovery_creates_real_lockfile_and_mark or test_invalid_lock_paths_are_refused or test_recovery_does_not_mutate_services_when_mark_fails or test_actual_lock_contender_cannot_deploy",
        ),
        (
            "own7_drop_contained_fields_match_in_publish_readback",
            by_name["own7_drop_contained_fields_match_in_publish_readback"],
            "test_d3b_recovery_reproves_stop_and_marker or test_recovery_rechecks_each_unverified_result or test_recovery_checks_a_fresh_marker_and_keeps_verified_post or test_d3b_surviving_commit_record_is_reported_as_a_record",
        ),
    ]
    for name, edits, k_expr in tail_checks:
        t0 = time.monotonic()
        with tempfile.TemporaryDirectory(dir=mod.SCRATCH_PARENT) as td:
            scratch = mod.make_scratch_repo(Path(td))
            mod.apply_replacements(scratch, edits)
            rc, detail = mod.run_pytest(
                scratch, "tests/test_deploy_workflow.py", ["-m", "not slow", "-k", k_expr], timeout=120
            )
            status = "DETECTED" if rc != 0 else "SURVIVED targeted -k"
            print(f"MUTATION {name}: {status} ({time.monotonic() - t0:.1f}s) [{detail}]", flush=True)


if __name__ == "__main__":
    main()
