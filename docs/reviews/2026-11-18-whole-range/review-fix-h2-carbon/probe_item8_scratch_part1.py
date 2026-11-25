"""Item 8 scratch-copy check part 1: restore 57, set 399, and set 401 at each of S1..S6."""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SCRATCH = OUT_DIR / "scratch_src_p1"

FALLBACK_TESTS = [
    "tests/test_contest_dated_cs4a.py",
    "tests/test_contest_dated_cs4a_review.py",
    "tests/test_contest_state_best_trust.py",
    "tests/test_contest_state_cs1_review.py",
    "tests/test_contest_state_defensive_reads.py",
    "tests/test_pick_chosen_on_cs5a.py",
    "tests/test_saver_cs3.py",
    "tests/test_season_contract_cs4b1.py",
    "tests/test_season_contract_cs4b1_edges.py",
    "tests/test_season_contract_cs4b2.py",
    "tests/test_wr_c1_history.py",
    "tests/test_wr_h_longest.py",
    "tests/scripts/season_walk/test_season_walk.py",
]

CHANGES = [
    # Restore 57
    ("s1_check5_restore_57", "bts/contest_state.py", "0 <= active <= MAX_SEASON_STREAK", "0 <= active <= TARGET_STREAK"),
    ("s1_check6_restore_57", "bts/contest_state.py", "active <= best <= MAX_SEASON_STREAK", "active <= best <= TARGET_STREAK"),
    ("s2_active_restore_57", "bts/contest_state.py", "streak > MAX_SEASON_STREAK", "streak > TARGET_STREAK"),
    ("s2_best_restore_57", "bts/contest_state.py", "(best < streak or best > MAX_SEASON_STREAK)", "(best < streak or best > TARGET_STREAK)"),
    ("s3_classify_restore_57", "bts/contest_state.py", "if best < 0 or best < contest.streak or best > MAX_SEASON_STREAK:", "if best < 0 or best < contest.streak or best > TARGET_STREAK:"),
    ("s4_fetch_restore_57", "bts/contest_fetch.py", "if best_streak > MAX_SEASON_STREAK:", "if best_streak > 57:"),
    ("s5_cli_restore_57", "bts/cli.py", "if best_streak > MAX_SEASON_STREAK:", "if best_streak > 57:"),
    ("s6_outer_restore_57", "bts/picks.py", 'if kind in {"dated", "automatic"} and streak > MAX_SEASON_STREAK:', 'if kind in {"dated", "automatic"} and streak > 57:'),
    ("s6_dated_restore_57", "bts/picks.py", "not 0 <= active <= best <= MAX_SEASON_STREAK", "not 0 <= active <= best <= 57"),
    # Bound 399
    ("s1_check5_399", "bts/contest_state.py", "0 <= active <= MAX_SEASON_STREAK", "0 <= active <= 399"),
    ("s1_check6_399", "bts/contest_state.py", "active <= best <= MAX_SEASON_STREAK", "active <= best <= 399"),
    ("s2_active_399", "bts/contest_state.py", "streak > MAX_SEASON_STREAK", "streak > 399"),
    ("s2_best_399", "bts/contest_state.py", "(best < streak or best > MAX_SEASON_STREAK)", "(best < streak or best > 399)"),
    ("s3_classify_399", "bts/contest_state.py", "if best < 0 or best < contest.streak or best > MAX_SEASON_STREAK:", "if best < 0 or best < contest.streak or best > 399:"),
    ("s4_fetch_399", "bts/contest_fetch.py", "if best_streak > MAX_SEASON_STREAK:", "if best_streak > 399:"),
    ("s5_cli_399", "bts/cli.py", "if best_streak > MAX_SEASON_STREAK:", "if best_streak > 399:"),
    ("s6_outer_399", "bts/picks.py", 'if kind in {"dated", "automatic"} and streak > MAX_SEASON_STREAK:', 'if kind in {"dated", "automatic"} and streak > 399:'),
    ("s6_dated_399", "bts/picks.py", "not 0 <= active <= best <= MAX_SEASON_STREAK", "not 0 <= active <= best <= 399"),
    # Bound 401
    ("s1_check5_401", "bts/contest_state.py", "0 <= active <= MAX_SEASON_STREAK", "0 <= active <= 401"),
    ("s1_check6_401", "bts/contest_state.py", "active <= best <= MAX_SEASON_STREAK", "active <= best <= 401"),
    ("s2_active_401", "bts/contest_state.py", "streak > MAX_SEASON_STREAK", "streak > 401"),
    ("s2_best_401", "bts/contest_state.py", "(best < streak or best > MAX_SEASON_STREAK)", "(best < streak or best > 401)"),
    ("s3_classify_401", "bts/contest_state.py", "if best < 0 or best < contest.streak or best > MAX_SEASON_STREAK:", "if best < 0 or best < contest.streak or best > 401:"),
    ("s4_fetch_401", "bts/contest_fetch.py", "if best_streak > MAX_SEASON_STREAK:", "if best_streak > 401:"),
    ("s5_cli_401", "bts/cli.py", "if best_streak > MAX_SEASON_STREAK:", "if best_streak > 401:"),
    ("s6_outer_401", "bts/picks.py", 'if kind in {"dated", "automatic"} and streak > MAX_SEASON_STREAK:', 'if kind in {"dated", "automatic"} and streak > 401:'),
    ("s6_dated_401", "bts/picks.py", "not 0 <= active <= best <= MAX_SEASON_STREAK", "not 0 <= active <= best <= 401"),
]


def run_pytest(test_file: str, env: dict) -> int:
    proc = subprocess.run(
        [str(REPO / ".venv/bin/python"), "-m", "pytest", "-q", "-x", test_file],
        cwd=str(REPO), env=env, capture_output=True, text=True, timeout=30,
    )
    return proc.returncode


def main() -> int:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    shutil.copytree(REPO / "src", SCRATCH)

    env = dict(os.environ, PYTHONPATH=f"{SCRATCH}:{REPO}")
    # Sanity check unmodified scratch copy passes test_wr_h2_caps.py
    assert run_pytest("tests/test_wr_h2_caps.py", env) == 0, "clean scratch failed"

    lines = []
    unnoticed = []
    try:
        for label, rel, target, repl in CHANGES:
            fpath = SCRATCH / rel
            orig = fpath.read_text(encoding="utf-8")
            assert orig.count(target) == 1, f"{label}: target count {orig.count(target)} in {rel}"
            fpath.write_text(orig.replace(target, repl, 1), encoding="utf-8")
            try:
                caught_by = None
                for tf in ["tests/test_wr_h2_caps.py", *FALLBACK_TESTS]:
                    rc = run_pytest(tf, env)
                    if rc != 0:
                        caught_by = tf
                        break
                status = f"CAUGHT by {caught_by}" if caught_by else "UNNOTICED"
                if not caught_by:
                    unnoticed.append(label)
                line = f"{label}: {status}"
                print(line, flush=True)
                lines.append(line)
            finally:
                fpath.write_text(orig, encoding="utf-8")
    finally:
        shutil.rmtree(SCRATCH, ignore_errors=True)

    lines.append(f"total={len(CHANGES)} unnoticed={len(unnoticed)}: {unnoticed}")
    (OUT_DIR / "item8_scratch_part1.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
