import datetime as dt
import os
import signal
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from bts.health import alert, contest_state as health, runner

from probe_support import DAY, HERE, clock, flag, manual


class ReadBlocked(BaseException):
    pass


def alarm(*_):
    raise ReadBlocked


with TemporaryDirectory(dir=HERE, prefix="special-") as directory:
    root = Path(directory)
    picks = root / "fifo" / "picks"
    manual(picks)
    os.mkfifo(picks / "account_state/saver_state.json")
    signal.signal(signal.SIGALRM, alarm)
    for historical in (False, True):
        signal.alarm(1)
        try:
            result = health.check(picks, now=DAY, expected=True,
                                  historical_day=DAY.date() if historical else None)
            print("historical" if historical else "live", result)
        except ReadBlocked:
            print("live FIFO blocked until alarm")
        finally:
            signal.alarm(0)

    from_candidate = "scratch" in str(Path(health.__file__))
    picks = root / "rollover" / "picks"
    before_midnight = DAY.replace(hour=23, minute=59)
    after_midnight = before_midnight + dt.timedelta(minutes=2)
    manual(picks, before_midnight)
    flag(picks, "active")
    real_check = health.check
    safe_run = runner._safe_run

    def passed_midnight(*args, **kwargs):
        with clock(after_midnight):
            return real_check(*args, **kwargs)

    with (clock(before_midnight), patch.object(health, "check", side_effect=passed_midnight) as check,
          patch.object(alert, "send_dm", Mock(return_value="sent")),
          patch.object(runner, "_safe_run", lambda name, fn: safe_run(name, fn) if name == "contest_state" else [])):
        alerts = runner.run_all_checks(picks, root / "models", "offline.test", today=DAY.date())
    print("clock crossed before check", check.call_args, "alerts", alerts)
    assert check.call_args.kwargs == {"expected": False}
    assert not alerts
    if from_candidate:
        assert len(real_check(picks, now=before_midnight)) == 1
