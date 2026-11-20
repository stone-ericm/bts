"""Render the final proposed copy and check live diagnostics and FIFO evidence."""

from datetime import timedelta
import logging
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import probe


POINTER = (
    "This check used files available at run time. Do not enter picks, retry delivery, or change "
    "contest state from this alert. Before any new-day correction, verify in MLB that the prior "
    "slate's entries are settled or there was no entry. Use the new day's morning state check, "
    "when available, and live refusal for current diagnosis. Season contract status shows current files."
)


def controls(root):
    from bts import scheduler
    from bts.contest_state import ContestStateError, load_decision_streak_state
    from bts.health import alert, contest_state, runner

    day = probe.D
    following = day + timedelta(days=1)
    picks = root / "control" / "picks"
    probe.auto(picks)
    probe.saver_flag(picks, state="not_earned")
    probe.switch(picks, status="revoked")
    evening = probe.at(day, 23, 50)

    class Clock(probe.datetime):
        @classmethod
        def now(cls, tz=None):
            return evening.astimezone(tz)

    def contest_alerts(items):
        return [a for a in items if a.source in ("contest_state", "saver_state")]

    reference = contest_state.check(picks, expected=True, now=evening.astimezone(probe.UTC))
    with patch.object(contest_state, "datetime", Clock), patch.object(alert, "send_dm", return_value="captured"):
        for supplied in ({}, {"today": day}, {"today": following}):
            with patch.object(contest_state, "check", wraps=contest_state.check) as check:
                result = runner.run_all_checks(picks, root / "models", "operator.invalid",
                                               contest_state_expected=True, **supplied)
                check.assert_called_once_with(picks, expected=True)
            assert contest_alerts(result) == reference
    print("Base omitted/same/future controls matched exact contest Alert values at 23:50 ET")

    after_midnight = probe.at(following, 2)
    alerts, _ = probe.real_runner(picks, day, after_midnight)
    assert any(f"--date {following}" in a.message for a in alerts)
    try:
        load_decision_streak_state(picks, require_contest_state=True, now=after_midnight)
    except ContestStateError as exc:
        assert f"--date {following}" in str(exc)
    else:
        raise AssertionError("expected live refusal")

    config = {"bluesky": {"dm_recipient": "operator.invalid"}}
    morning = probe.at(following, 10)
    with patch.object(alert, "send_dm", return_value="captured") as send:
        scheduler._morning_state_check(config, picks, following.isoformat(), morning, probe.at(following, 11))
        send.assert_called_once()
        body = send.call_args.args[1]
        assert f"Morning check for {following}" in body
        print("Base next-day morning DM:", body)
    probe.auto(picks, following, source_date=day.isoformat(), recorded_at=probe.at(following, 9).isoformat())
    probe.switch(picks, following)
    probe.saver_flag(picks, following)
    with patch.object(alert, "send_dm", return_value="captured") as send:
        scheduler._morning_state_check(config, picks, following.isoformat(), morning, probe.at(following, 11))
        send.assert_not_called()
    assert load_decision_streak_state(picks, require_contest_state=True, now=morning).observation_kind == "automatic"
    print("Accepted current state produces no morning DM; historical alert does not settle the current verdict")


def fifo_readiness(root):
    for kind in ("manual", "automatic", "switch", "saver", "coverage"):
        args = [sys.executable, str(probe.HERE / "probe.py"), "fifo-child", kind, str(root)]
        try:
            subprocess.run(args, check=True, timeout=3, capture_output=True)
        except subprocess.TimeoutExpired as exc:
            assert f"FIFO ready: {kind}".encode() in (exc.stdout or b""), exc
        else:
            raise AssertionError(f"FIFO {kind} did not block")
        print(f"Base reached {kind} FIFO and did not return before the deadline")


if __name__ == "__main__":
    signal.alarm(50)
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(dir=probe.HERE) as scratch:
        scratch = Path(scratch)
        os.chdir(scratch)
        with patch.object(socket, "socket", probe.deny_network), patch.object(socket, "create_connection", probe.deny_network):
            if sys.argv[1] == "copy":
                probe.POINTER = POINTER
                probe.messages(scratch)
            else:
                controls(scratch)
                fifo_readiness(scratch)
    signal.alarm(0)
