"""Check alert containment, bounded diagnostics and the independent EOD backstop."""

from contextlib import redirect_stderr
from io import StringIO
import json
from pathlib import Path
import sys
from unittest.mock import Mock

from probe_support import (
    FAILURE_PREFIX, attempt_direct, day_pick, deadline, fixtures, prepare_schedules, raw,
    receipt, shared, status_body, world_at,
)


DAY = "2027-03-25"


class UnprintableError(Exception):
    def __str__(self):
        raise ValueError("cannot render")


class ReportingUnavailable(StringIO):
    def write(self, message):
        if message.startswith(FAILURE_PREFIX):
            raise ValueError("stderr unavailable during alert")
        return super().write(message)


def failures(root, patch, opening, entry, stage):
    from bts import dm, picks, scheduler
    from bts.daily_decision import load_decision
    from bts.health import alert

    with world_at(root, patch, opening, opening.at(DAY, "18:20")) as (config, directory, _, _, scoped):
        daily = day_pick(DAY)
        daily.delivery_attempted = True
        prepare_schedules(daily)
        pick = picks.save_pick(daily, directory)
        original = pick.read_bytes()
        path = receipt(directory, DAY)
        legacy = shared(directory)
        legacy.parent.mkdir()
        legacy.write_bytes(b"[]\n")
        journal = StringIO()
        error = OSError("first line\nsecond line\rthird line " + "x" * 600)

        if stage == "mkdir":
            original_mkdir = Path.mkdir

            def inaccessible(dirpath, *args, **kwargs):
                if dirpath == path.parent:
                    raise error
                return original_mkdir(dirpath, *args, **kwargs)

            scoped.setattr(Path, "mkdir", inaccessible)
        elif stage in {"format", "dispatch", "read", "write", "write_and_transport", "stderr", "unprintable"}:
            target = {
                "format": "format_health_dm_body", "dispatch": "dispatch_dm_for_health_alerts",
                "read": "_read_receipt", "write": "_replace_receipt",
                "write_and_transport": "_replace_receipt",
                "stderr": "format_health_dm_body", "unprintable": "format_health_dm_body",
            }[stage]
            scoped.setattr(alert, target, Mock(side_effect=UnprintableError() if stage == "unprintable" else error))
            if stage == "stderr":
                journal = ReportingUnavailable()
        if stage in {"transport", "write_and_transport"}:
            alert.send_dm.side_effect = error

        with redirect_stderr(journal):
            assert sys.stderr is journal
            if entry == "direct":
                assert attempt_direct(config, directory, daily) is False
            else:
                assert scheduler.run_day(DAY, config) is None
        state = scheduler.load_state(DAY, directory)
        assert state.pick_locked and state.committed_pick_written
        assert load_decision(DAY, directory)["delivery_status"] == "locked_unconfirmed"
        assert pick.read_bytes() == original and legacy.read_bytes() == b"[]\n"
        dm.send_dm.assert_not_called()
        lines = [line for line in journal.getvalue().splitlines() if line.startswith(FAILURE_PREFIX)]
        assert len(lines) == (0 if stage == "stderr" else 1)
        dm_expected = stage in {"mkdir", "read", "write", "write_and_transport", "transport"}
        assert alert.send_dm.call_count == int(dm_expected)
        if dm_expected:
            alert.send_dm.assert_called_once_with("offline.test", status_body(DAY))
        if lines:
            detail = lines[0].removeprefix(FAILURE_PREFIX)
            assert len(detail) <= 200
            if stage == "unprintable":
                assert "UnprintableError" in detail
            elif stage in {"mkdir", "read", "write", "write_and_transport"}:
                assert str(path) in detail and "receipt" in detail
            else:
                assert r"first line\nsecond line\rthird line" in detail
            assert "second line" not in journal.getvalue().splitlines()
        if stage == "transport":
            assert json.loads(path.read_bytes())["status"] == "failed"
        else:
            assert raw(path) is None
    print(f"{entry}/{stage}: contained; transport={int(dm_expected)}; bounded diagnostic={len(lines)}.")


def eod_backstop(root, patch, opening):
    from bts import dm, picks, scheduler
    from bts.health import alert, runner

    with world_at(root, patch, opening, opening.at(DAY, "22:15")) as (config, directory, _, _, scoped):
        daily = day_pick(DAY)
        daily.delivery_attempted = True
        prepare_schedules(daily)
        picks.save_pick(daily, directory)
        config["health_checks"]["enabled"] = True
        scoped.setattr(runner, "_safe_run", lambda name, check: check() if name == "post_failure" else [])
        scoped.setattr(runner, "get_self_pid", lambda: None)
        scoped.setattr(runner, "read_systemd_nrestarts", lambda: None)
        scoped.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _: None)
        alert.send_dm.side_effect = [None, "eod-id", "retry-id"]
        with redirect_stderr(StringIO()):
            assert scheduler.run_day(DAY, config) is None
            assert alert.send_dm.call_count == 2
            assert alert.send_dm.call_args_list[0].args[1] == status_body(DAY)
            assert "[pick_delivery]" in alert.send_dm.call_args_list[1].args[1]
            assert json.loads(receipt(directory, DAY).read_bytes())["status"] == "failed"
            legacy_checkpoint = shared(directory).read_bytes()
            assert scheduler.run_day(DAY, config) is None
            assert alert.send_dm.call_count == 3
            assert alert.send_dm.call_args.args[1] == status_body(DAY)
            assert json.loads(receipt(directory, DAY).read_bytes())["message_id"] == "retry-id"
            assert shared(directory).read_bytes() == legacy_checkpoint
        dm.send_dm.assert_not_called()
    print("EOD: one unknown attempt and one independent pick_delivery alert; recovery retried only on restart.")


def main():
    with deadline(50), fixtures("failures") as (root, patch, opening):
        for entry in ("direct", "classification"):
            for stage in ("format", "dispatch", "mkdir", "read", "write", "transport",
                          "write_and_transport", "stderr", "unprintable"):
                failures(root / f"{entry}-{stage}", patch, opening, entry, stage)
        eod_backstop(root / "eod", patch, opening)


if __name__ == "__main__":
    main()
