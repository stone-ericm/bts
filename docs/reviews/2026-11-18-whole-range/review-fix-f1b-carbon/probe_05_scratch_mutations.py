"""Probe 05: Isolated scratch-copy mutation checks (zero tracked files touched)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
WORK_DIR = OUT_DIR / "_work_p05"

TEST_FILES = [
    "test_entry_intent.py",
    "test_wr_f1b_lock.py",
    "test_wr_f1_game_pk_types.py",
    "test_wr_f1_wrong_game.py",
    "test_wr_f1_review.py",
]


def build_scratch_repo(dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    (dest / "src").mkdir(parents=True)
    shutil.copytree(REPO / "src" / "bts", dest / "src" / "bts")
    (dest / "scripts").mkdir(parents=True)
    shutil.copy2(REPO / "scripts" / "cron-setup-hetzner.sh", dest / "scripts" / "cron-setup-hetzner.sh")
    (dest / "docs" / "ops").mkdir(parents=True)
    shutil.copy2(REPO / "docs" / "ops" / "pick-entry-receipt-v1.md", dest / "docs" / "ops" / "pick-entry-receipt-v1.md")
    (dest / "tests").mkdir(parents=True)
    if (REPO / "tests" / "__init__.py").exists():
        shutil.copy2(REPO / "tests" / "__init__.py", dest / "tests" / "__init__.py")
    for tf in TEST_FILES:
        shutil.copy2(REPO / "tests" / tf, dest / "tests" / tf)


def run_scratch_pytest(scratch: Path, targets: list[str], extra_args: list[str] = ()) -> dict:
    env = {
        **os.environ,
        "PYTHONPATH": f"{scratch / 'src'}:{scratch}",
        "TZ": "America/New_York",
    }
    cmd = [sys.executable, "-m", "pytest", "-q", "--tb=no", *extra_args, *targets]
    res = subprocess.run(cmd, cwd=scratch, env=env, capture_output=True, text=True, timeout=60)
    tail = [ln for ln in res.stdout.splitlines() if ln.strip()][-3:]
    return {
        "exit_code": res.returncode,
        "noticed": res.returncode != 0,
        "summary_lines": tail,
    }


def apply_single_replacement(path: Path, old: str, new: str) -> None:
    text = path.read_text()
    count = text.count(old)
    assert count == 1, f"expected 1 match in {path}, got {count} for {old!r}"
    path.write_text(text.replace(old, new, 1))


def run_all() -> dict:
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    scratch = WORK_DIR / "scratch_repo"
    orig_cron = (REPO / "scripts" / "cron-setup-hetzner.sh").read_text()
    orig_cli = (REPO / "src" / "bts" / "cli.py").read_text()

    build_scratch_repo(scratch)
    cron_path = scratch / "scripts" / "cron-setup-hetzner.sh"
    cli_path = scratch / "src" / "bts" / "cli.py"

    # Verify import isolation inside scratch_repo
    verify = subprocess.run(
        [sys.executable, "-c", "import bts, tests.test_entry_intent as t; print(bts.__file__); print(t.SCRIPT)"],
        cwd=scratch,
        env={**os.environ, "PYTHONPATH": f"{scratch / 'src'}:{scratch}"},
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    v_lines = verify.stdout.strip().splitlines()
    assert str(scratch.resolve()) in str(Path(v_lines[0]).resolve()), v_lines
    assert str(scratch.resolve()) in str(Path(v_lines[1]).resolve()), v_lines

    # Control run (exclude the 20s FIFO subprocess tests during fast mutation sweep)
    control_lock = run_scratch_pytest(scratch, ["tests/test_wr_f1b_lock.py", "tests/test_wr_f1_game_pk_types.py"])
    control_f1 = run_scratch_pytest(
        scratch,
        ["tests/test_wr_f1_wrong_game.py", "tests/test_wr_f1_review.py"],
        extra_args=["-k", "not fifo"],
    )
    assert control_lock["exit_code"] == 0, control_lock
    assert control_f1["exit_code"] == 0, control_f1

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
    ]

    results = {}
    for name, old, new in cron_mutations:
        cron_path.write_text(orig_cron)
        apply_single_replacement(cron_path, old, new)
        results[name] = run_scratch_pytest(scratch, ["tests/test_wr_f1b_lock.py", "tests/test_entry_intent.py"])
    cron_path.write_text(orig_cron)

    for name, old, new in cli_mutations:
        cli_path.write_text(orig_cli)
        apply_single_replacement(cli_path, old, new)
        results[name] = run_scratch_pytest(
            scratch,
            ["tests/test_wr_f1_game_pk_types.py", "tests/test_wr_f1_review.py", "tests/test_wr_f1_wrong_game.py"],
            extra_args=["-k", "not fifo"],
        )
    cli_path.write_text(orig_cli)

    shutil.rmtree(scratch, ignore_errors=True)

    summary = {
        "control_lock": control_lock,
        "control_f1": control_f1,
        "mutations": results,
        "unnoticed": [k for k, v in results.items() if not v["noticed"]],
    }
    (OUT_DIR / "log_05_scratch_mutations.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
