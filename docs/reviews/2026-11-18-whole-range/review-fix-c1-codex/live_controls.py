"""Compare exact live alerts, readers, CLI output, and scheduler refusal on identical paths."""

from contextlib import redirect_stderr
from dataclasses import asdict, is_dataclass
from datetime import datetime, timedelta
import datetime as datetime_module
import io
import json
import logging
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch


HERE = Path(__file__).resolve().parent


def result(fn, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as error:
        return {"type": type(error).__name__, "text": str(error)}


def run_tree(repository, root):
    source = HERE / "base/src" if repository == "base" else HERE.parents[1] / "src"
    sys.path[:0] = [str(source), str(HERE / "contract-check-c1")]
    import bts
    from bts import contest_state, saver_state, scheduler, season_contract
    from bts.cli import cli
    from bts.health import alert, contest_state as health, runner
    from bts.picks import DailyPick, Pick, submission_cutoff_et
    from click.testing import CliRunner
    import probe

    assert Path(bts.__file__).resolve().is_relative_to(source)
    print(f"{repository}: {bts.__file__}", file=sys.stderr)
    root.mkdir()
    os.chdir(root)
    output = {}
    day = probe.YEAR_END
    next_day = day + timedelta(days=1)
    evening = probe.at(day, 23, 50)
    crossed = probe.at(next_day, 0, 1)
    sample_calls = []
    check_calls = []

    class ModeClock(datetime):
        @classmethod
        def now(cls, tz=None):
            sample_calls.append(tz)
            return evening.astimezone(tz)

    live = evening

    class LiveClock(datetime):
        @classmethod
        def now(cls, tz=None):
            check_calls.append(tz)
            return live.astimezone(tz) if tz is not None else live.replace(tzinfo=None)

    for mode in ("omitted", "same", "future", "rollover"):
        picks = root / mode / "picks"
        probe.auto(picks, day)
        probe.saver_flag(picks, day, "not_earned")
        probe.switch(picks, day, status="revoked")
        live = crossed if mode == "rollover" else evening
        inputs = {} if mode == "omitted" else {"today": next_day if mode == "future" else day}
        sample_calls.clear()
        check_calls.clear()
        with (patch.object(runner, "datetime", ModeClock, create=True),
              patch.object(health, "datetime", LiveClock),
              patch.object(health, "check", wraps=health.check) as check,
              patch.object(alert, "send_dm", return_value="offline-ack") as transport):
            items = runner.run_all_checks(picks, picks.parent / "models", "offline.test",
                                          contest_state_expected=True, **inputs)
        check.assert_called_once_with(picks, expected=True)
        assert len(check_calls) == 1
        if repository == "head":
            assert len(sample_calls) == int(mode != "omitted")
        dms = [entry.args[1] for entry in transport.call_args_list]
        assert len(dms) == 1 and f"--date {live.date()}" in dms[0] and "--picks-dir" in dms[0]
        output[mode] = {"alerts": items, "dms": dms}

    wall = probe.at(next_day, 0, 30)
    for fixture in ("empty", "automatic", "dated", "revoked", "legacy", "bad switch",
                    "bad automatic", "bad saver", "mismatched", "old reading", "new season"):
        picks = root / fixture / "picks"
        picks.mkdir(parents=True)
        if fixture != "empty":
            auto = probe.auto(picks, next_day, source_date=str(day), recorded_at=wall.isoformat())
            saver = probe.saver_flag(picks, next_day)
            switch = probe.switch(picks, next_day, status="revoked" if fixture == "revoked" else "automatic")
            if fixture == "dated":
                probe.dated(picks, next_day)
            elif fixture == "legacy":
                probe.put(picks, "contest_streak.manual.json", {
                    "streak": 7, "source_date": str(day),
                    "override_expires_at": probe.at(next_day, 19).isoformat(),
                })
            elif fixture == "bad switch":
                switch.write_bytes(b"\xff")
            elif fixture == "bad automatic":
                auto.write_text("{")
            elif fixture == "bad saver":
                saver.write_text('{"state": "surprise"}')
            elif fixture == "mismatched":
                probe.switch(picks, next_day, activation_id="b" * 32)
            elif fixture == "old reading":
                probe.auto(picks, next_day, source_date=str(day), recorded_at=probe.at(day - timedelta(days=2)).isoformat())
            elif fixture == "new season":
                probe.switch(picks, day)
                probe.saver_flag(picks, day)

        selected = result(contest_state.load_contest_streak_state, picks, now=wall, require_dated_auto=True)
        settings = {
            "selected": selected,
            "decision_required": result(contest_state.load_decision_streak_state, picks, now=wall,
                                        require_contest_state=True),
            "decision_optional": result(contest_state.load_decision_streak_state, picks, now=wall),
            "saver": saver_state.load_saver_state(picks, season=next_day.year),
            "switch": season_contract.switch_state(picks, wall),
            "refusal": season_contract.live_automatic_refusal(picks, wall,
                                                             None if isinstance(selected, dict) else selected),
        }

        live = wall
        with patch.object(datetime_module, "datetime", LiveClock):
            cli_result = CliRunner().invoke(cli, ["season-contract", "status", "--picks-dir", str(picks)])
        settings["status"] = (cli_result.exit_code, cli_result.stdout, cli_result.stderr)
        config = {"orchestrator": {"picks_dir": str(picks)}, "bluesky": {"dm_recipient": "offline.test"},
                  "scheduler": {"pick_delivery": "dm"}, "health_checks": {"contest_state_required": True}}
        journal = io.StringIO()
        with redirect_stderr(journal), patch.object(alert, "send_dm", return_value="offline-ack") as transport:
            scheduler._morning_state_check(config, picks, str(next_day), probe.at(next_day, 10), probe.at(next_day, 11))
        settings["morning"] = (journal.getvalue(), [call.args[1] for call in transport.call_args_list])

        if fixture == "revoked":
            pick = Pick("Synthetic Batter", 1, "NYM", 1, "Synthetic Pitcher", 2, 0.8, [], False, 700,
                        probe.at(next_day, 18).isoformat())
            daily = DailyPick(str(next_day), wall.isoformat(), pick, None, None)
            state = scheduler.SchedulerState(str(next_day), wall.isoformat(), [], [], [], False, None, None, None)
            assert wall < submission_cutoff_et(daily)
            journal = io.StringIO()
            with (redirect_stderr(journal), patch.object(alert, "send_dm", return_value="offline-ack") as transport,
                  patch.object(scheduler, "_void_committed_games", return_value=[]),
                  patch.object(scheduler, "_now_et", return_value=wall),
                  patch.object(contest_state, "datetime", LiveClock),
                  patch.object(alert, "datetime", LiveClock)):
                delivered = scheduler._deliver_and_lock_pick(daily, config, picks, state, str(next_day), "review")
            assert not delivered and not state.pick_locked
            assert len(transport.call_args_list) == 1
            assert f"--date {next_day}" in transport.call_args.args[1]
            assert f"--date {next_day}" in cli_result.stdout
            settings["scheduler refusal"] = (delivered, journal.getvalue(), transport.call_args.args[1])
            settings["past cutoff"] = probe.at(next_day, 23) > submission_cutoff_et(daily)
        output[fixture] = settings
    return output


def encode(value):
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, Path):
        return str(value)
    if hasattr(value, "isoformat"):
        return value.isoformat()
    raise TypeError(type(value))


if __name__ == "__main__":
    signal.alarm(55)
    logging.disable(logging.CRITICAL)
    if sys.argv[1] == "compare":
        with TemporaryDirectory(dir=HERE) as scratch:
            root = Path(scratch) / "shared"
            outputs = []
            for revision in ("base", "head"):
                call = subprocess.run([sys.executable, __file__, revision, str(root)],
                                      check=True, stdout=subprocess.PIPE, timeout=22)
                (HERE / f"live-{revision}.json").write_bytes(call.stdout)
                outputs.append(json.loads(call.stdout))
                shutil.rmtree(root)
            diffs = [name for name in outputs[0] if outputs[0][name] != outputs[1][name]]
            print(f"{len(outputs[0])} controls (4 runner routes, 11 file fixtures); differences: {diffs}")
            assert not diffs
    else:
        with patch.object(socket, "socket", side_effect=AssertionError("offline")), \
             patch.object(socket, "create_connection", side_effect=AssertionError("offline")):
            print(json.dumps(run_tree(sys.argv[1], Path(sys.argv[2])), default=encode, sort_keys=True))
