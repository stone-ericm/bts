"""Probe 05c: Fast (-x) isolated scratch-copy cli.py mutation checks."""
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

WORK_DIR = OUT_DIR / "_work_p05c"


def run_all() -> dict:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    scratch = WORK_DIR / "scratch_repo"
    orig_cli = (REPO / "src" / "bts" / "cli.py").read_text()

    build_scratch_repo(scratch)
    cli_path = scratch / "src" / "bts" / "cli.py"

    cli_mutations = [
        (
            "drop_game_unresolved_alerted",
            'if marker["status"] != "confirmed" and (reason == "wrong_game" or prior_wrong_game):',
            'if marker["status"] not in ("confirmed", "alerted") and (reason == "wrong_game" or prior_wrong_game):',
        ),
        (
            "drop_game_unresolved_dm_failed",
            'if marker["status"] != "confirmed" and (reason == "wrong_game" or prior_wrong_game):',
            'if marker["status"] not in ("confirmed", "dm_failed") and (reason == "wrong_game" or prior_wrong_game):',
        ),
        (
            "drop_game_unresolved_present_unverified",
            'if marker["status"] != "confirmed" and (reason == "wrong_game" or prior_wrong_game):',
            'if marker["status"] not in ("confirmed", "present_unverified") and (reason == "wrong_game" or prior_wrong_game):',
        ),
        (
            "game_pk_isinstance_int_allows_bool",
            'if result["state"] == "wrong_game" and type(slot["game_pk"]) is int]',
            'if result["state"] == "wrong_game" and isinstance(slot["game_pk"], int)]',
        ),
        (
            "game_pk_not_none_in_a1",
            'if result["state"] == "wrong_game" and type(slot["game_pk"]) is int]',
            'if result["state"] == "wrong_game" and slot["game_pk"] is not None]',
        ),
        (
            "game_pk_not_none_in_a3_legs",
            'game = game_pk if type(game_pk) is int else "ID unavailable"',
            'game = game_pk if game_pk is not None else "ID unavailable"',
        ),
        (
            "drop_wrong_game_tier_in_all_clear",
            'if dm_recipient and (was_alerted or "initial" in prior_escalations or "wrong_game" in prior_escalations):',
            'if dm_recipient and (was_alerted or "initial" in prior_escalations):',
        ),
        (
            "drop_escalations_preservation_on_dm_failed_after_wg",
            'if prior_wrong_game or reason == "wrong_game":',
            'if reason == "wrong_game":',
        ),
    ]

    results = {}
    for name, old, new in cli_mutations:
        cli_path.write_text(orig_cli)
        apply_single_replacement(cli_path, old, new)
        results[name] = run_scratch_pytest(
            scratch,
            ["tests/test_wr_f1_game_pk_types.py", "tests/test_wr_f1_review.py", "tests/test_wr_f1_wrong_game.py"],
            extra_args=["-x", "-k", "not fifo"],
        )
    cli_path.write_text(orig_cli)
    shutil.rmtree(WORK_DIR, ignore_errors=True)

    summary = {
        "cli_mutations": results,
        "unnoticed": [k for k, v in results.items() if not v["noticed"]],
    }
    (OUT_DIR / "log_05c_cli_mutations.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
