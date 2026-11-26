"""Probe 05b: Fast (-x) isolated scratch-copy cron-wrapper mutation checks."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

OUT_DIR = Path(__file__).resolve().parent
if str(OUT_DIR) not in sys.path:
    sys.path.insert(0, str(OUT_DIR))

from probe_05_scratch_mutations import (
    REPO,
    apply_single_replacement,
    build_scratch_repo,
    run_scratch_pytest,
)

WORK_DIR = OUT_DIR / "_work_p05b"


def run_all() -> dict:
    # Clean up any leftover _work_p05 from timed-out un-throttled run
    shutil.rmtree(OUT_DIR / "_work_p05", ignore_errors=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    scratch = WORK_DIR / "scratch_repo"
    orig_cron = (REPO / "scripts" / "cron-setup-hetzner.sh").read_text()

    build_scratch_repo(scratch)
    cron_path = scratch / "scripts" / "cron-setup-hetzner.sh"

    control = run_scratch_pytest(scratch, ["tests/test_wr_f1b_lock.py", "tests/test_wr_f1_game_pk_types.py"])
    assert control["exit_code"] == 0, control

    cron_mutations = [
        (
            "drop_minus_n",
            "/usr/bin/flock -n /home/bts/.check-pick-entered.lock /usr/bin/timeout --verbose -k 30 840",
            "/usr/bin/flock /home/bts/.check-pick-entered.lock /usr/bin/timeout --verbose -k 30 840",
        ),
        (
            "drop_limit",
            "/usr/bin/flock -n /home/bts/.check-pick-entered.lock /usr/bin/timeout --verbose -k 30 840 ",
            "/usr/bin/flock -n /home/bts/.check-pick-entered.lock ",
        ),
        (
            "make_limit_900",
            "/usr/bin/timeout --verbose -k 30 840",
            "/usr/bin/timeout --verbose -k 30 900",
        ),
        (
            "make_limit_870_boundary",
            "/usr/bin/timeout --verbose -k 30 840",
            "/usr/bin/timeout --verbose -k 30 870",
        ),
        (
            "drop_verbose",
            "/usr/bin/timeout --verbose -k 30 840",
            "/usr/bin/timeout -k 30 840",
        ),
        (
            "swap_lock_and_limit",
            "/usr/bin/flock -n /home/bts/.check-pick-entered.lock /usr/bin/timeout --verbose -k 30 840",
            "/usr/bin/timeout --verbose -k 30 840 /usr/bin/flock -n /home/bts/.check-pick-entered.lock",
        ),
    ]

    results = {}
    for name, old, new in cron_mutations:
        cron_path.write_text(orig_cron)
        apply_single_replacement(cron_path, old, new)
        results[name] = run_scratch_pytest(
            scratch,
            ["tests/test_wr_f1b_lock.py", "tests/test_entry_intent.py"],
            extra_args=["-x"],
        )
    cron_path.write_text(orig_cron)
    shutil.rmtree(WORK_DIR, ignore_errors=True)

    summary = {
        "control": control,
        "cron_mutations": results,
        "unnoticed": [k for k, v in results.items() if not v["noticed"]],
    }
    (OUT_DIR / "log_05b_cron_mutations.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
