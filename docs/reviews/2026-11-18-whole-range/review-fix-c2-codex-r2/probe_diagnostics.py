"""Recheck containment using a short fixture root and inspect long-path logs."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import sys
from unittest.mock import Mock


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "replay" / "codex"))
import probe_failures
import probe_support as support


def long_path_failure(root, patch, opening, route):
    from bts import dm, picks, scheduler
    from bts.daily_decision import load_decision
    from bts.health import alert

    day = probe_failures.DAY
    root = root / "configured-state-directory" / ("nested-account-state-" * 6)
    with support.world_at(root, patch, opening, opening.at(day, "18:20")) as (config, directory, _, _, scoped):
        daily = support.day_pick(day)
        daily.delivery_attempted = True
        support.prepare_schedules(daily)
        picks.save_pick(daily, directory)
        path = support.receipt(directory, day)
        scoped.setattr(alert, "_replace_receipt", Mock(side_effect=OSError("disk full")))
        stderr = StringIO()
        with redirect_stderr(stderr):
            if route == "direct":
                assert support.attempt_direct(config, directory, daily) is False
            else:
                assert scheduler.run_day(day, config) is None
        alert.send_dm.assert_called_once_with("offline.test", support.status_body(day))
        dm.send_dm.assert_not_called()
        assert scheduler.load_state(day, directory).pick_locked
        assert load_decision(day, directory)["delivery_status"] == "locked_unconfirmed"
        assert not path.exists()
        (line,) = [line for line in stderr.getvalue().splitlines() if line.startswith(support.FAILURE_PREFIX)]
        detail = line.removeprefix(support.FAILURE_PREFIX)
        assert detail.startswith("write receipt at ") and len(detail) == 200
        assert path.name not in detail and "disk full" not in detail
        print(f"{route}: receipt path length {len(str(path))}, omitted {path.name} and the error.")
        print(line)


def main():
    # The archived assertion expects the full path to fit in the log, as it did
    # in the shallower round-one checkout. Only the temporary fixture root moves.
    support.HERE = HERE
    probe_failures.main()
    with support.deadline(20), support.fixtures("diagnostics") as (root, patch, opening):
        for route in ("direct", "classification"):
            long_path_failure(root / route, patch, opening, route)


if __name__ == "__main__":
    main()
