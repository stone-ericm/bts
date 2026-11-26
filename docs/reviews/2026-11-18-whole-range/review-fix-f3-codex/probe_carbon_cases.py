"""Run both archived cases and rerun the dated case after a next-day write."""
import datetime as dt
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from click.testing import CliRunner

import probe_scheduler as scenarios
from review_support import CURRENT, ET, REPO, identify_package


ARCHIVE = REPO / "docs/reviews/2026-11-18-whole-range/closing-test/carbon-slice3"
sys.path.insert(0, str(ARCHIVE))
import test_slice3_base_and_edges as dated_case

dated_case.REPO_ROOT = REPO


def capture_archived(fn, root, local_name, line, base):
    observed = {}

    def trace(frame, event, _arg):
        if event == "line" and frame.f_code is fn.__code__ and frame.f_lineno == line:
            result = frame.f_locals[local_name]
            observed.update({"exit": result.exit_code, **json.loads(result.stdout)})
        return trace

    sys.settrace(trace)
    try:
        fn(root)
    except AssertionError as exc:
        print(f"{fn.__name__} stopped at a historical assertion: {exc}")
    finally:
        sys.settrace(None)
    assert observed.get("state") == ("deliberate_skip" if base else "skip_state_review"), observed
    print(json.dumps({"archived_case": fn.__name__, **observed}))


def overwrite_dated_before_report(root, base):
    d2 = dt.datetime(2027, 3, 25, 8, tzinfo=ET)
    snapshots = 0
    snapshot = scenarios.snapshot

    def write_before_first_snapshot(directory):
        nonlocal snapshots
        snapshots += 1
        if snapshots == 1:
            picks = root / "data" / "picks"
            manual = picks / "account_state" / CURRENT
            assert json.loads(manual.read_bytes())["active_streak"] == 2
            with patch("datetime.datetime", scenarios.carbon._make_Clock([d2])):
                result = CliRunner().invoke(scenarios.cli.cli, [
                    "set-contest-streak", "--picks-dir", str(picks), "--date", "2027-03-25",
                    "--streak", "2", "--best-streak", "2", "--saver", "not_earned",
                ])
            assert result.exit_code == 0, (result.output, result.exception)
            assert json.loads(manual.read_bytes())["effective_date"] == "2027-03-25"
            assert not list(manual.parent.glob("contest_streak.manual.revoked-*.json"))
            print("Carbon dated correction for D was replaced using set-contest-streak on D+1 at 08:00 ET.")
        return snapshot(directory)

    with patch.dict(scenarios.CASES, {"carbon_dated": ()}), \
         patch.object(scenarios, "snapshot", side_effect=write_before_first_snapshot):
        scenarios.run_case(root, "carbon_dated", base)


if __name__ == "__main__":
    identify_package()
    base = "--base" in sys.argv
    with TemporaryDirectory() as temp:
        root = Path(temp)
        (root / "dated").mkdir()
        (root / "revoked").mkdir()
        capture_archived(dated_case.test_stale_mdp_skip_not_invalidated_by_dated_correction_silences_all_alarms,
                         root / "dated", "r_out", 267, base)
        capture_archived(scenarios.carbon.test_item1d_live_refusal_vs_health_and_day_outcome,
                         root / "revoked", "cli_do", 724, base)
        overwrite_dated_before_report(root / "dated_overwrite", base)
