"""Replay the rendered repair command and the prior-day EOD clock path."""
import datetime as dt
import json
import os
import shlex
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from probe_support import REVIEW, REV, NOW, REAL, ET, auto, dump, invoke, offline, put, switch
from bts import cli, contest_state
from bts.health import alert, contest_state as health, runner
from click.testing import CliRunner


def contradiction(alerts):
    return next(a for a in alerts if a.source == "saver_state" and "flag not_earned" in a.message)


def extract(warning):
    candidates = [part for part in warning.message.split(chr(96))
                  if part.startswith("bts set-contest-streak ")]
    if not candidates:
        return None
    args = shlex.split(candidates[0])
    for option, value in (("--streak", "0"), ("--best-streak", "0"), ("--saver", "not_earned")):
        args[args.index(option)+1] = value
    return args


rows = {}
with TemporaryDirectory(dir=REVIEW, prefix="health-fixtures-") as directory:
    root = Path(directory)
    previous = Path.cwd()
    os.chdir(root)
    try:
        for i, path in enumerate((root / "picks with spaces", root / 'operator\'s "picks"', Path("relative picks"))):
            put(path, "contest_streak.json", auto())
            put(path, "season_contract.json", switch())
            flag = put(path, "saver_state.json", {"season": NOW.year, "state": "not_earned"})
            before = (flag.read_bytes(), flag.stat().st_mtime_ns)
            warning = contradiction(health.check(path, expected=True, now=NOW))
            with offline():
                command = extract(warning)
                if command is None:
                    assert REV == "base" and warning.message.endswith("with bts set-contest-streak")
                    result = CliRunner().invoke(cli.cli, ["set-contest-streak"])
                else:
                    assert command[-1] == str(path)
                    result = CliRunner().invoke(cli.cli, command[1:])
            assert (flag.read_bytes(), flag.stat().st_mtime_ns) == before
            live = contest_state.load_decision_streak_state(path, require_contest_state=True, now=NOW)
            assert (result.exit_code, live.observation_kind, live.streak) == (
                (0, "dated", 0) if REV == "head" else (2, "automatic", 12))
            rows[str(i)] = dict(command=command, exit=result.exit_code, kind=live.observation_kind,
                                message=warning.message)

        before = REAL(2027, 12, 31, 23, 59, 59, tzinfo=ET)
        after = before + dt.timedelta(seconds=2)
        path = root / "midnight"
        put(path, "contest_streak.json", auto(after, source_date="2027-09-26"))
        put(path, "season_contract.json", switch(after))
        flag = put(path, "saver_state.json", {"season": before.year, "state": "not_earned"})
        old = contradiction(health.check(path, now=before))
        flag.write_text(json.dumps({"season": after.year, "state": "not_earned"}))
        original_safe = runner._safe_run
        send = Mock(return_value="ack")
        with (
            offline(after),
            patch.object(health, "datetime", dt.datetime),
            patch.object(runner, "_safe_run", lambda name, fn: original_safe(name, fn)
                         if name == "contest_state" else []),
            patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
            patch.object(alert, "send_dm", send),
        ):
            eod = runner.run_all_checks(path, root / "models", None,
                                        today=before.date(), contest_state_expected=True)
            new = contradiction(eod)
            old_args, new_args = extract(old), extract(new)
            if REV == "head":
                assert "--date 2027-12-31" in old.message and "--date 2028-01-01" in new.message
                stale_command = CliRunner().invoke(cli.cli, old_args[1:])
                current_command = CliRunner().invoke(cli.cli, new_args[1:])
                assert (stale_command.exit_code, current_command.exit_code) == (2, 0)
            for bundle in (False, True):
                status = root / f"dm-{bundle}.json"
                critical = [alert.Alert("CRITICAL", "other", "Unrelated example")] if bundle else []
                assert alert.dispatch_dm_for_health_alerts(
                    critical, "offline.example", warn_attention=[new],
                    status_path=status, now_et_date=before.date(),
                )
                record = json.loads(status.read_text())
                rows[f"dm-{bundle}"] = dict(sources=record["sent_sources"], day=record["sent_day"],
                                            text=send.call_args.args[1])
                assert not alert.dispatch_dm_for_health_alerts(
                    critical, "offline.example", warn_attention=[new],
                    status_path=status, now_et_date=before.date(),
                )
        rows["eod-date"] = dict(slate=before.date().isoformat(), previous=old.message, wall=new.message)
    finally:
        os.chdir(previous)

dump("health", rows)
