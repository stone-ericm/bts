"""Capture both historical CLI results before their old clean-state assertions."""
import inspect
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

import probe_scheduler_replay as replay
import probe_carbon_cases as carbon


base = "--base" in sys.argv
dated = carbon.dated_case.test_stale_mdp_skip_not_invalidated_by_dated_correction_silences_all_alarms
revoked = replay.probe_scheduler.carbon.test_item1d_live_refusal_vs_health_and_day_outcome
with TemporaryDirectory() as temp:
    root = Path(temp)
    for check, subdir, variable, marker in (
        (dated, "dated", "r_out", "assert r_out.exit_code"),
        (revoked, "revoked", "cli_do", "assert out2.state"),
    ):
        path = root / subdir
        path.mkdir()
        lines, start = inspect.getsourcelines(check)
        line = start + next(i for i, text in enumerate(lines) if marker in text)
        carbon.capture_archived(check, path, variable, line, base)
    carbon.overwrite_dated_before_report(root / "dated_overwrite", base)
