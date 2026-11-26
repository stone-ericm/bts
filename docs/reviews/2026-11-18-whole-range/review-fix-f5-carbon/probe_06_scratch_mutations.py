"""Probe 06: Run isolated one-at-a-time scratch mutations against the test suite on an untracked scratch copy."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent

FOCUSED_TESTS = [
    "tests/test_wr_f5_watchdog_window.py",
    "tests/test_entry_intent.py",
    "tests/test_entry_receipt.py",
    "tests/test_day_outcome.py",
    "-k", "not k2_generated and not k3_ and not interrupted_cli",
    "tests/test_wr_f1b_lock.py",
]

MUTATIONS = [
    (
        "drop_the_gate",
        "src/bts/cli.py",
        (
            "    if cron and now.hour < 10:\n"
            "        first_base_slot = now.replace(hour=10, minute=0, second=0, microsecond=0)\n"
            "        for minute in range(0, 14 * 60, 15):\n"
            "            slot = first_base_slot + timedelta(minutes=minute)\n"
            "            slot_lead = (first_pitch - slot).total_seconds() / 60\n"
            "            if submit_cutoff_min < slot_lead <= window_min:\n"
            '                rec.set_outcome("base_slot_available")\n'
            '                click.echo("check-pick-entered: original schedule has a slot inside the window")\n'
            "                return\n"
        ),
        "",
    ),
    (
        "apply_gate_without_cron",
        "src/bts/cli.py",
        "    if cron and now.hour < 10:\n",
        "    if now.hour < 10:\n",
    ),
    (
        "drop_cron_from_installed_line",
        "scripts/cron-setup-hetzner.sh",
        "bts check-pick-entered --cron --picks-dir",
        "bts check-pick-entered --picks-dir",
    ),
    (
        "change_now_hour_lt_10_to_le_10",
        "src/bts/cli.py",
        "    if cron and now.hour < 10:\n",
        "    if cron and now.hour <= 10:\n",
    ),
    (
        "slot_range_stop_at_23_30",
        "src/bts/cli.py",
        "        for minute in range(0, 14 * 60, 15):\n",
        "        for minute in range(0, 14 * 60 - 15, 15):\n",
    ),
    (
        "slot_range_start_at_10_15",
        "src/bts/cli.py",
        "        for minute in range(0, 14 * 60, 15):\n",
        "        for minute in range(15, 14 * 60, 15):\n",
    ),
    (
        "slot_comparison_inclusive_at_cutoff",
        "src/bts/cli.py",
        "            if submit_cutoff_min < slot_lead <= window_min:\n",
        "            if submit_cutoff_min <= slot_lead <= window_min:\n",
    ),
]


def build_scratch_repo(scratch: Path) -> None:
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True)
    shutil.copytree(REPO / "src", scratch / "src")
    shutil.copytree(REPO / "scripts", scratch / "scripts")
    shutil.copytree(REPO / "tests", scratch / "tests")
    shutil.copy2(REPO / "pyproject.toml", scratch / "pyproject.toml")
    for link_name in ("docs", "data", "config", "ARCHITECTURE.md", "CLAUDE.md", "INCIDENT.md", "README.md"):
        src_p = REPO / link_name
        if src_p.exists():
            os.symlink(src_p, scratch / link_name)


def run_pytest(scratch: Path, test_files: list[str], tmp_base: Path) -> tuple[int, str]:
    if tmp_base.exists():
        shutil.rmtree(tmp_base)
    tmp_base.mkdir(parents=True)
    env = {
        **os.environ,
        "PYTHONPATH": f"{scratch / 'src'}:{scratch}",
        "TZ": "America/New_York",
        "TMPDIR": str(tmp_base),
    }
    # Verify imports come from scratch copy first
    verify = subprocess.run(
        [
            sys.executable, "-c",
            "import bts.cli, tests.test_entry_intent as t; print(bts.cli.__file__); print(t.SCRIPT)",
        ],
        cwd=str(scratch), env=env, capture_output=True, text=True, timeout=10,
    )
    assert verify.returncode == 0, verify.stderr
    lines = verify.stdout.strip().splitlines()
    assert str(scratch.resolve()) in str(Path(lines[0]).resolve()), lines
    assert str(scratch.resolve()) in str(Path(lines[1]).resolve()), lines

    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "-x", "--tb=short",
            f"--basetemp={tmp_base / 'pytest'}",
            *test_files,
        ],
        cwd=str(scratch), env=env, capture_output=True, text=True, timeout=45,
    )
    return proc.returncode, (proc.stdout + proc.stderr)[-800:]


def main() -> None:
    t0 = time.perf_counter()
    work = OUT_DIR / ".work_probe_06"
    scratch = work / "scratch_repo"
    tmp_base = work / "tmp"
    try:
        build_scratch_repo(scratch)
        # Sanity check: unmutated scratch copy passes focused tests
        rc0, out0 = run_pytest(scratch, FOCUSED_TESTS, tmp_base)
        assert rc0 == 0, f"Unmutated scratch copy failed:\n{out0}"

        results = {}
        unnoticed = []
        for name, rel_path, target, replacement in MUTATIONS:
            fpath = scratch / rel_path
            original = fpath.read_text()
            assert original.count(target) == 1, f"Target count != 1 in {rel_path} for {name}"
            try:
                fpath.write_text(original.replace(target, replacement, 1))
                rc, tail = run_pytest(scratch, FOCUSED_TESTS, tmp_base)
                caught = rc != 0
                if not caught:
                    unnoticed.append(name)
                results[name] = {
                    "file": rel_path,
                    "caught_by_existing_tests": caught,
                    "pytest_returncode": rc,
                    "output_tail": tail.strip().splitlines()[-3:],
                }
            finally:
                fpath.write_text(original)

        # Also prove our own probe_02b regression catches slot_range_start_at_10_15 while slot_range_stop_at_23_30 is algebraically equivalent
        cli_path = scratch / "src/bts/cli.py"
        orig_cli = cli_path.read_text()
        try:
            cli_path.write_text(orig_cli.replace("for minute in range(0, 14 * 60, 15):", "for minute in range(15, 14 * 60, 15):", 1))
            env_p2 = {
                **os.environ,
                "PYTHONPATH": str(scratch / "src"),
                "TZ": "America/New_York",
                "TMPDIR": str(tmp_base),
            }
            # Run a quick 1-second check of 10:15 pitch at 09:45 --cron against mutated scratch src
            check_1015 = subprocess.run(
                [
                    sys.executable, "-c",
                    (
                        "import tempfile; from datetime import datetime; from pathlib import Path; from zoneinfo import ZoneInfo; "
                        "from bts import entry_receipt, picks; from bts.cli import _check_pick_entered; from bts.daily_decision import write_decision; "
                        "ET = ZoneInfo('America/New_York'); d = Path(tempfile.mkdtemp()) / 'picks'; d.mkdir(parents=True); "
                        "s = picks.Pick('B', 1010, 'NYY', 1, 'P', 500, 0.8, [], False, 101, '2027-03-24T10:15:00-04:00'); "
                        "picks.save_pick(picks.DailyPick('2027-03-24', '2027-03-24T08:00:00-04:00', s, None, None, notification_sent=True, notification_channel='bluesky_dm', notification_id='1', delivery_attempted=True), d); "
                        "write_decision('2027-03-24', d, action='single', source='mdp', delivery_status='delivered', scoreable=True, primary={'batter_name': 'B'}); "
                        "now = datetime.fromisoformat('2027-03-24T09:45:00').replace(tzinfo=ET); "
                        "rec = entry_receipt.EntryReceipt(et_date='2027-03-24', started_at=now, clock=lambda: now, expected_username='u', picks_dir=d); "
                        "import unittest.mock as m\n"
                        "with m.patch('bts.leaderboard.auth.load_session_cookies', side_effect=RuntimeError('CALLED_AUTH')):\n"
                        "    _check_pick_entered(d, 'u', None, 75, now, rec, cron=True)"
                    ),
                ],
                env=env_p2, capture_output=True, text=True, timeout=10,
            )
            caught_by_1015_probe = check_1015.returncode != 0 and "CALLED_AUTH" in check_1015.stderr
        finally:
            cli_path.write_text(orig_cli)

        elapsed = round(time.perf_counter() - t0, 3)
        report = {
            "status": "PASS",
            "elapsed_sec": elapsed,
            "mutations_tested": len(MUTATIONS),
            "unnoticed_by_existing_tests": unnoticed,
            "slot_range_start_1015_caught_by_1015_probe": caught_by_1015_probe,
            "results": results,
        }
        (OUT_DIR / "log_06_scratch_mutations.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"PASS probe_06 in {elapsed}s: tested={len(MUTATIONS)}, "
            f"unnoticed_by_existing={unnoticed}, 1015_probe_catches_start_1015={caught_by_1015_probe}"
        )
    finally:
        if work.exists():
            shutil.rmtree(work)


if __name__ == "__main__":
    main()
