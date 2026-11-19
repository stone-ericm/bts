"""Exercise B8b's exact move with one blocker and two, and trace archive opens."""

import builtins
from datetime import date
from pathlib import Path
import subprocess
from unittest.mock import patch

from probe_common import RUN, examine, receipt, run_against_archives, validate


FIRST = "2027-04-16"
SECOND = "2027-04-17"
ARCHIVE = "set_aside_never_entered.json"


def probes(scratch, revision):
    from bts import daily_decision, picks, reconcile_receipt, scheduler, skip_policy_shadow, web
    from bts.data import backup, lineup_analyze
    from bts.health import analytics_artifacts_missing, fallback_defer, late_delivery, scheduler_state_integrity

    data = {}
    for scenario, blockers in (("single", [FIRST]), ("multiple", [FIRST, SECOND])):
        working = scratch / scenario
        root = working / "data" / "picks"
        root.mkdir(parents=True)
        target = root / "streak.json"
        target.write_bytes(b'{"streak":9,"saver_available":false,"season":2027}')
        (root / "2027-04-18.json").write_text('{"result":"miss"}')
        for day in blockers:
            (root / f"{day}.json").write_text('{"result":null,"synthetic_contest_entry":false}')

        data[scenario + "-before"] = examine(root)
        data[scenario + "-rerun"] = examine(root)
        moved = []
        accessed = []
        for day in blockers:
            aside = root / day / ARCHIVE
            aside.parent.mkdir()
            source = Path("data/picks") / f"{day}.json"
            payload = (working / source).read_bytes()
            subprocess.run(["mv", str(source), str(Path("data/picks") / day / ARCHIVE)],
                           cwd=working, timeout=3, check=True)
            assert not (working / source).exists() and aside.read_bytes() == payload
            moved.append(aside)
            path_open = Path.open
            builtin_open = builtins.open

            def capture(file):
                if isinstance(file, (str, Path)) and Path(file).name == ARCHIVE:
                    accessed.append(str(file))

            def traced_path(file, *args, **kwargs):
                capture(file)
                return path_open(file, *args, **kwargs)

            def traced_builtin(file, *args, **kwargs):
                capture(file)
                return builtin_open(file, *args, **kwargs)

            with patch.object(Path, "open", traced_path), patch.object(builtins, "open", traced_builtin), \
                 patch.object(web, "PICKS_DIR", root):
                result = examine(root)
                data[scenario + "-after-" + day] = result
                assert reconcile_receipt.discover(root, RUN.date().isoformat()) == [receipt(result)]
                d = date.fromisoformat(day)
                assert picks.load_pick(day, root) is None
                assert daily_decision.load_decision_bytes(day, root) == (None, None)
                assert scheduler.load_state(day, root) is None
                assert web.load_scheduler_state(day) == {}
                assert lineup_analyze.backfill_from_scheduler_state(root) == []
                assert skip_policy_shadow.record_pending_skips(root, now=RUN) == []
                assert analytics_artifacts_missing._load_scheduler_state(root, day) == {}
                assert late_delivery.check(root, today=d, now=RUN) == []
                assert scheduler_state_integrity.check(root, today=d) == []
                assert fallback_defer.check(root, today=d, now=RUN) == []
                backup.verify_restored_ops_tree(working)
            assert not accessed and all(path.is_file() for path in moved), accessed
    return data


def check(base, current):
    validate(base, current)
    for scenario in ("single", "multiple"):
        first_run = current[scenario + "-before"]
        assert first_run == current[scenario + "-rerun"]
        assert receipt(first_run)["replay"]["blocked_date"] == FIRST
    for name in ("single-after-" + FIRST, "multiple-after-" + SECOND):
        result = current[name]
        assert result["strict"][1] == 0 and result["stderr"] == ""
        assert receipt(result)["replay"]["state"] == "saved"
    next_run = current["multiple-after-" + FIRST]
    assert receipt(next_run)["replay"]["blocked_date"] == SECOND
    assert next_run["strict"][3] == next_run["strict"][4]
    return {"move_then_save": 2, "move_then_another_blocker": SECOND, "archive_opens": 0}


if __name__ == "__main__":
    run_against_archives(__file__, probes, check)
