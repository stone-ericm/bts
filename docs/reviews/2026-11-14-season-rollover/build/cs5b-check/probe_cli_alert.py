import datetime as dt
import json
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from probe_run_day_v3 import MonkeyPatch, pin
from click.testing import CliRunner
from bts import cli, contest_state, picks, scheduler
from bts.health import alert
from tests.test_pick_chosen_on_cs5a import DAY, NOW, daily_pick


class Clock(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


def call(*words, **flags):
    return CliRunner().invoke(cli.cli, words, catch_exceptions=False, **flags)


def cli_checks(root, mp):
    mp.setattr(dt, "datetime", Clock)
    mp.setattr(contest_state, "datetime", Clock)
    directory = root / "picks"
    args = ("--picks-dir", str(directory))
    for _ in range(2):
        result = call("set-contest-streak", *args, "--date", DAY,
                      "--streak", "30", "--best-streak", "30", "--saver", "used")
        print("dated", result.exit_code, result.stdout.splitlines(), result.stderr.splitlines())
    path = directory / f"{DAY}.json"
    daily = daily_pick()
    real_loader = contest_state.load_decision_streak_state
    loader = Mock(wraps=real_loader)
    mp.setattr(contest_state, "load_decision_streak_state", loader)
    for label, body in [
        ("absent", None), ("directory", None), ("utf8", b"\xff"),
        ("nonobject", b"[]"), ("otherdate", json.dumps(dict(asdict(daily), date="2027-07-14")).encode()),
        ("newline", json.dumps(dict(asdict(daily), date=f"{DAY}\nextra line")).encode()),
        ("surrogate", json.dumps(dict(asdict(daily), date=f"{DAY}\ud800")).encode()),
    ]:
        if label == "directory":
            path.mkdir()
        elif body is not None:
            path.write_bytes(body)
        old_read = Path.read_text
        reads = []

        def read(p, *a, **k):
            if p == path:
                reads.append(p.name)
            return old_read(p, *a, **k)

        loader.reset_mock()
        with MonkeyPatch.context() as trace:
            trace.setattr(Path, "read_text", read)
            status = call("season-contract", "status", *args, "--season", "2026")
        assert not reads and loader.call_count == 1
        print(label, "status", status.exit_code, status.output.splitlines())
        print("snapshot call", loader.call_args.kwargs)
        try:
            loaded = picks.load_pick(DAY, directory)
            message = (
                picks.saved_pick_refusal(loaded, real_loader(directory, require_contest_state=True, now=NOW))
                if loaded else None
            )
            print("load", repr(loaded.date) if loaded else None, repr(message))
            if message:
                message.encode("utf-8")
        except Exception as exc:
            print("load/render error", type(exc).__name__, repr(str(exc)))
        if path.is_dir():
            path.rmdir()
        elif path.exists():
            path.unlink()

    file_instead = root / "not-a-directory"
    file_instead.write_text("plain file")
    odd = call("season-contract", "status", "--picks-dir", str(file_instead))
    print("pick parent is file", odd.exit_code, odd.output.splitlines())
    print("load below non-directory", picks.load_pick(DAY, file_instead))
    missing_directory = root / "missing"
    cold = call("season-contract", "status", "--picks-dir", str(missing_directory))
    print("cold", cold.exit_code, cold.output.splitlines(), missing_directory.exists())

    # This JSON-backed judge input can parse while repr of its nested date fails.
    valid_pick_json = json.dumps(asdict(daily))
    state = real_loader(directory, require_contest_state=True, now=NOW)
    for depth in range(950, 1010):
        body = valid_pick_json.replace(f'"date": "{DAY}"', '"date": ' + "[" * depth + "0" + "]" * depth)
        try:
            loaded = picks.parse_daily_pick(body)
        except RecursionError:
            continue
        try:
            picks.saved_pick_refusal(loaded, state)
        except RecursionError:
            print("judge raised RecursionError after decoding at depth", depth)
            break


def alert_checks(root, mp):
    mp.setattr(alert, "datetime", Clock)
    transport = Mock(return_value="offline")
    mp.setattr(alert, "send_dm", transport)
    config = {"orchestrator": {"picks_dir": str(root / "picks")},
              "bluesky": {"dm_recipient": "offline.test"}}
    path = root / "health_state" / "health_dm_delivery_status.json"
    scheduler._alert_contest_state_failure(config, RuntimeError("old body"))
    assert transport.call_count == 1
    old_status = json.loads(path.read_text())
    scheduler._alert_contest_state_failure(config, RuntimeError("old body restart line"))
    assert transport.call_count == 1
    print("same-source new body dedup", old_status["sent_sources"])
    path.unlink()
    transport.side_effect = [OSError("offline failure"), "offline"]
    scheduler._alert_contest_state_failure(config, RuntimeError("failed body"))
    scheduler._alert_contest_state_failure(config, RuntimeError("retry body restart line"))
    print("h6", transport.call_args.args[1].splitlines())


if __name__ == "__main__":
    for check in (cli_checks, alert_checks):
        with TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp, MonkeyPatch.context() as mp:
            pin(mp)
            check(Path(tmp), mp)
