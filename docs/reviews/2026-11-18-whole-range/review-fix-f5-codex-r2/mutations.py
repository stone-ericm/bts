"""Run the F5 suite with one deliberate behavioral change in an isolated source tree."""
import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

from support import ROOT, legacy

CLI = "src/bts/cli.py"
HEALTH = "src/bts/health/pick_entry.py"
MORNING = "src/bts/day_outcome.py"
RECEIPT = "src/bts/entry_receipt.py"
F5 = "tests/test_wr_f5_watchdog_window.py"
CARBON = ROOT / "docs/reviews/2026-11-18-whole-range/review-fix-f5-carbon/probe_07_extra_regressions.py"
CASES = {
    "write-key-everywhere": [
        (CLI, '        if rec.early_cron_run:\n            marker["early_cron_run"] = True',
         '        marker["early_cron_run"] = rec.early_cron_run'),
        (RECEIPT, '**({"early_cron_run": True} if self.early_cron_run else {}),',
         '"early_cron_run": self.early_cron_run,'),
    ],
    "drop-reset": [(CLI, 'reset_early_marker = cron and now.hour >= 10 and prior.get("early_cron_run") is True',
                    'reset_early_marker = False')],
    "reset-manual": [(CLI, 'reset_early_marker = cron and now.hour >= 10', 'reset_early_marker = now.hour >= 10')],
    "reset-wrong-game": [(CLI, '    if reset_early_marker:\n        prior_escalations = []',
                          '    if reset_early_marker:\n        prior_wrong_game = False\n        prior_escalations = []')],
    "drop-slot-rule": [(MORNING, 'has_original_slot = any(r.get("early_cron_run") is True for r in receipts)'
                        ' and _has_original_slot(date, picks_dir)', 'has_original_slot = False')],
    "inclusive-cutoff": [(MORNING, 'SUBMISSION_CUTOFF_MIN < (first_pitch', 'SUBMISSION_CUTOFF_MIN <= (first_pitch')],
    "drop-copy": [(CLI, '        if displaced:\n', '        if False:\n')],
    "overwrite-copy": [
        (CLI, 'if not copy.exists():', 'if True:'),
        (CLI, 'NamedTemporaryFile(dir=status_path.parent)', 'NamedTemporaryFile(dir=status_path.parent, delete=False)'),
        (CLI, 'os.link(temporary.name, copy)', 'os.replace(temporary.name, copy)'),
    ],
    "copy-with-live-today": [(HEALTH, '    if marker_date != today.isoformat():',
        '    if marker_date == today.isoformat():\n'
        '        copy, _ = read_marker(marker_path.with_name(f"pick_entry_check.{today.isoformat()}.json"))\n'
        '        if isinstance(copy, dict) and copy.get("date") == marker_date:\n'
        '            marker = copy\n'
        '    if marker_date != today.isoformat():')],
    "drop-nonblock": [(HEALTH, 'flags | os.O_NONBLOCK', 'flags')],
    "start-1015": [(CLI, 'range(0, 14 * 60, 15)', 'range(15, 14 * 60, 15)')],
}
FOCUSED = {
    "write-key-everywhere": "manual_early_check_still_confirms",
    "drop-reset": "early_confirmation_does_not_stop_original_slots",
    "reset-wrong-game": "original_slot_keeps_early_wrong_game_observation",
    "drop-slot-rule": "morning_filters_early_evidence",
    "inclusive-cutoff": "early_slate",
    "drop-copy": "r4_displaced_marker_health",
    "overwrite-copy": "displaced_copy_survives_interrupted_marker_write",
    "drop-nonblock": "health_does_not_block_on_live_fifo",
    "start-1015": "r5_r6_sole_original_slot",
}


def tests(tree, env, paths, expr=None):
    argv = [sys.executable, "-m", "pytest", "-p", "no:cacheprovider", "-q", "--tb=short", "-x", *paths]
    if expr:
        argv.extend(["-k", expr])
    completed = subprocess.run(argv, cwd=tree, env=env, timeout=105)
    assert completed.returncode in (0, 1), completed.returncode
    return completed.returncode


if __name__ == "__main__":
    case = sys.argv[1]
    with TemporaryDirectory() as directory:
        tree = Path(directory)
        probe = legacy("mutation_probe")
        probe.ROOT = ROOT
        probe.scratch_copy(tree)
        for name, before, after in CASES.get(case, []):
            path = tree / name
            data = path.read_text()
            assert data.count(before) == 1, (name, before)
            path.write_text(data.replace(before, after))
        (tree / "bts").symlink_to(tree / "src/bts", target_is_directory=True)
        env = {**os.environ, "PYTHONPATH": str(tree / "src")}
        imported = subprocess.check_output([sys.executable, "-c", "import bts;print(bts.__file__)"],
                                           cwd=tree, env=env, text=True, timeout=5).strip()
        assert Path(imported).resolve().is_relative_to((tree / "src").resolve()), imported
        print(case, imported, flush=True)
        expression = FOCUSED.get(case)
        result = tests(tree, env, [F5], expression)
        if case == "control":
            assert result == 0
        if case == "start-1015":
            shutil.copyfile(CARBON, tree / "tests" / CARBON.name)
            carbon = tests(tree, env, [f"tests/{CARBON.name}"])
            assert result == carbon == 1
        if expression and result == 0:
            result = tests(tree, env, [F5])
        print(case, "DETECTED" if result else "SURVIVED", flush=True)
