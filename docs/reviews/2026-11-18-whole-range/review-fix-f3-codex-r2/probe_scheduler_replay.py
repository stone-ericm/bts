"""Rerun the tracked scheduler and Carbon probes from their committed location."""
import inspect
from pathlib import Path
import sys
from tempfile import TemporaryDirectory


repo = Path(__file__).resolve().parents[2]
original = repo / "docs/reviews/2026-11-18-whole-range/review-fix-f3-codex"
sys.path.insert(0, str(original))

import review_support

review_support.REPO = repo
review_support.identify_package()

import probe_scheduler


def run_scheduler(base):
    with TemporaryDirectory() as temp:
        for case in probe_scheduler.CASES:
            if case != "K2d_next_day_replaces_match":
                probe_scheduler.run_case(Path(temp) / case, case, base)
        # Round 1’s complete rerun included the archive reason on this row.
        case = "K2d_next_day_replaces_match"
        probe_scheduler.CASES[case] = ("archived", "revocation")
        probe_scheduler.run_case(Path(temp) / case, case, base)


def run_carbon(base):
    import probe_carbon_cases as carbon

    dated = carbon.dated_case.test_stale_mdp_skip_not_invalidated_by_dated_correction_silences_all_alarms
    revoked = probe_scheduler.carbon.test_item1d_live_refusal_vs_health_and_day_outcome
    with TemporaryDirectory() as temp:
        root = Path(temp)
        for check, subdir, variable, marker in (
            (dated, "dated", "r_out", "assert r_out.exit_code"),
            (revoked, "revoked", "cli_do", "assert cli_do.exit_code"),
        ):
            path = root / subdir
            path.mkdir()
            lines, start = inspect.getsourcelines(check)
            line = start + next(i for i, text in enumerate(lines) if marker in text)
            carbon.capture_archived(check, path, variable, line, base)
        carbon.overwrite_dated_before_report(root / "dated_overwrite", base)


if __name__ == "__main__":
    base = "--base" in sys.argv
    if sys.argv[1] == "scheduler":
        run_scheduler(base)
    elif sys.argv[1] == "carbon":
        run_carbon(base)
    else:
        raise ValueError(sys.argv[1])
