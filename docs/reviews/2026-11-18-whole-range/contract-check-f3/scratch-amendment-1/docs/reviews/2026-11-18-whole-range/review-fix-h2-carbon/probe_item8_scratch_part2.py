"""Item 8 scratch-copy check part 2: drop bounds, drop best>=active, use MAX_SEASON_STREAK in policy, and restore 57 in messages."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SCRATCH = OUT_DIR / "scratch_src_p2"

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
    # Drop upper bound at each place
    ("s1_check5_drop_upper", "bts/contest_state.py", "0 <= active <= MAX_SEASON_STREAK", "0 <= active"),
    ("s1_check6_drop_upper", "bts/contest_state.py", "active <= best <= MAX_SEASON_STREAK", "active <= best"),
    ("s2_active_drop_upper", "bts/contest_state.py", "streak > MAX_SEASON_STREAK\n        or ", ""),
    ("s2_best_drop_upper", "bts/contest_state.py", "(best < streak or best > MAX_SEASON_STREAK)", "best < streak"),
    ("s3_classify_drop_upper", "bts/contest_state.py", "if best < 0 or best < contest.streak or best > MAX_SEASON_STREAK:", "if best < 0 or best < contest.streak:"),
    ("s4_fetch_drop_upper", "bts/contest_fetch.py", "if best_streak > MAX_SEASON_STREAK:", "if False:"),
    ("s5_cli_drop_upper", "bts/cli.py", "if best_streak > MAX_SEASON_STREAK:", "if False:"),
    ("s6_outer_drop_upper", "bts/picks.py", 'if kind in {"dated", "automatic"} and streak > MAX_SEASON_STREAK:', "if False:"),
    ("s6_dated_drop_upper", "bts/picks.py", "not 0 <= active <= best <= MAX_SEASON_STREAK", "not 0 <= active <= best"),
    # Drop best >= active
    ("s4_drop_best_ge_active", "bts/contest_fetch.py", "if best_streak < active_streak:", "if False:"),
    ("s1_drop_best_ge_active", "bts/contest_state.py", "not active <= best <= MAX_SEASON_STREAK", "not 0 <= best <= MAX_SEASON_STREAK"),
    ("s2_drop_best_ge_active", "bts/contest_state.py", "(best < streak or best > MAX_SEASON_STREAK)", "best > MAX_SEASON_STREAK"),
    ("s5_drop_best_ge_active", "bts/cli.py", "        if best_streak < streak:\n            raise click.BadParameter(\"best streak must be at least the active streak\",\n                                     param_hint=\"--best-streak\")", "        if False:\n            pass"),
    # Use 400 (MAX_SEASON_STREAK) for policy target / boundaries / effective_best
    ("policy_tail_target_400", "bts/simulate/tail_policy.py", "TARGET = 57", "TARGET = 400"),
    ("policy_longest_from_400", "bts/simulate/tail_policy.py", "LONGEST_STREAK_FROM = 57", "LONGEST_STREAK_FROM = 400"),
    ("policy_effective_best_400", "bts/simulate/tail_policy.py", "def effective_best(streak: int, best: int | None, target: int = TARGET) -> int:", "def effective_best(streak: int, best: int | None, target: int = 400) -> int:"),
    ("policy_skip_reason_400", "bts/strategy.py", "min(57, streak + 2 * d_eff)", "min(400, streak + 2 * d_eff)"),
    # Restore 57 in each of the 4 interpolated messages while keeping 400 bound
    ("msg1_dated_check5_57", "bts/contest_state.py", 'f"must be an integer from 0 to {MAX_SEASON_STREAK}"', '"must be an integer from 0 to 57"'),
    ("msg2_dated_check6_57", "bts/contest_state.py", 'f"must be an integer from active_streak to {MAX_SEASON_STREAK}"', '"must be an integer from active_streak to 57"'),
    ("msg3_fetch_57", "bts/contest_fetch.py", 'f"seasonBestStreak must be <= {MAX_SEASON_STREAK}"', '"seasonBestStreak must be <= 57"'),
    ("msg4_cli_57", "bts/cli.py", 'f"best streak must be at most {MAX_SEASON_STREAK}"', '"best streak must be at most 57"'),
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
    (OUT_DIR / "item8_scratch_part2.log").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(lines[-1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
