"""Compare the real CLI, TOML loader and scheduler around ET midnight."""

import local_guard

from contextlib import ExitStack, nullcontext
from datetime import datetime, timedelta
from io import BytesIO
from itertools import product
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ET = ZoneInfo("America/New_York")
BEFORE = datetime(2027, 7, 15, 23, 59, 59, 900000, tzinfo=ET)
AFTER = datetime(2027, 7, 16, 0, 0, 0, 100000, tzinfo=ET)
OLD, NEW = BEFORE.date().isoformat(), AFTER.date().isoformat()
CONFIG = {
    "required-public": 'pick_delivery = "public"\ncontest_state_required = true\n',
    "expected-dm": 'pick_delivery = "dm"\n[health_checks]\nenabled = false\ncontest_state_expected = true\n',
    "private": 'pick_delivery = "private"\nentry_intent = "research"\n',
    "unchecked-dm": 'pick_delivery = "dm"\n',
    "unchecked-enter": 'pick_delivery = "private"\nentry_intent = "enter"\n',
}
BOUNDARIES = ("before-date", "inside-load", "after-gate", "after-date", "steady")
GAME = {
    "gamePk": 881100,
    "gameDate": "2027-07-16T04:50:00Z",
    "teams": {side: {"team": {"name": side}} for side in ("away", "home")},
    "status": {"abstractGameState": "Preview", "detailedState": "Pre-Game"},
}


def scenario(root, revision, boundary, config_name, dry, explicit):
    import click
    from click.testing import CliRunner
    import tomllib
    from bts import cli, entry_intent, heartbeat, orchestrator, scheduler

    resource = root / "resources"
    root.mkdir()
    path = root / "config.toml"
    body = CONFIG[config_name]
    if "[health_checks]" not in body:
        body += "[health_checks]\nenabled = false\n"
    path.write_text(
        f'tiers = []\n[orchestrator]\npicks_dir = "{resource / "picks"}"\n'
        f'heartbeat_path = "{resource / "health" / "heartbeat.json"}"\n'
        f'models_dir = "{resource / "models"}"\n'
        '[bluesky]\ndm_recipient = "captured.invalid"\n[scheduler]\n' + body
    )

    now = BEFORE
    events, file_access, snapshot = [], [], []
    read_toml, load_config = tomllib.load, orchestrator.load_config
    today, run_day = cli._today_et, scheduler.run_day
    gate = getattr(entry_intent, "live_gate_problem", None)
    convert = click.types.Path.convert
    stat, lstat, mkdir = os.stat, os.lstat, os.mkdir

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.replace(tzinfo=None) if tz is None else now.astimezone(tz)

    def advance(at):
        nonlocal now
        if boundary == at and now == BEFORE:
            events.append(("midnight", at))
            now = AFTER

    def validate(param_type, value, param, ctx):
        result = convert(param_type, value, param, ctx)
        if value == str(path):
            advance("before-date")
        return result

    def parse(stream, *args, **kwargs):
        result = read_toml(stream, *args, **kwargs)
        events.append(("toml-read",))
        advance("inside-load")
        return result

    def load(file):
        events.append(("load-start",))
        result = load_config(file)
        snapshot.append(result)
        events.append(("load-end",))
        return result

    def gate_then_clock(config):
        assert config is snapshot[0]
        problem = gate(config)
        events.append(("gate", problem))
        advance("after-gate")
        return problem

    def read_date():
        result = today()
        events.append(("default-date", result, now.isoformat()))
        advance("after-date")
        return result

    def start_day(*, date, config, dry_run):
        advance("after-gate")  # Base has no gate. Advance at the matching worker boundary.
        advance("after-date")  # For --date there is no default lookup to trigger this case.
        assert config is snapshot[0]
        events.append(("run-day", date, dry_run, now.isoformat()))
        return run_day(date=date, config=config, dry_run=dry_run)

    def fetch(day):
        events.append(("schedule", day))
        return [GAME] if day == OLD else []

    def sleep(seconds):
        nonlocal now
        events.append(("sleep", seconds))
        now += timedelta(seconds=seconds)

    def cascade(tiers, day, *args, **kwargs):
        assert tiers is snapshot[0]["tiers"]
        events.append(("predictions", day))
        return None, None

    def track(label, operation):
        def observed(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                filepath = Path(os.fsdecode(file)).absolute()
                if filepath.is_relative_to(resource):
                    file_access.append((label, str(filepath.relative_to(root))))
            return operation(file, *args, **kwargs)
        return observed

    argv = ["schedule", "--config", str(path)]
    if dry:
        argv.append("--dry-run")
    if explicit:
        argv.extend(("--date", OLD))
    del local_guard.opens[:]
    local_guard.watch[:] = [root]
    try:
        with ExitStack() as stack:
            for module in (cli, heartbeat, scheduler):
                stack.enter_context(patch.object(module, "datetime", Clock))
            patches = [
                (click.types.Path, "convert", validate), (tomllib, "load", parse),
                (orchestrator, "load_config", load), (cli, "_today_et", read_date),
                (scheduler, "run_day", start_day), (scheduler, "fetch_schedule", fetch),
                (scheduler, "notify_ready", lambda: events.append(("ready",))),
                (scheduler, "notify_watchdog", lambda: events.append(("watchdog",))),
                (scheduler, "_watchdog_ping_sleep", sleep),
                (scheduler, "check_confirmed_lineups", lambda pks: {}),
                (scheduler, "heartbeat_watchdog", lambda *args, **kwargs: nullcontext()),
                (orchestrator, "run_cascade", cascade),
                (os, "stat", track("stat", stat)), (os, "lstat", track("lstat", lstat)),
                (os, "mkdir", track("mkdir", mkdir)),
            ]
            if gate is not None:
                patches.append((entry_intent, "live_gate_problem", gate_then_clock))
            for obj, name, replacement in patches:
                stack.enter_context(patch.object(obj, name, replacement))
            result = CliRunner().invoke(cli.cli, argv)
    finally:
        local_guard.watch.clear()

    opens = [str(Path(file).relative_to(root)) for file, *_ in local_guard.opens]
    files = {str(file.relative_to(root)): json.loads(file.read_text()) for file in resource.rglob("*.json")}
    unchecked = config_name.startswith("unchecked")
    is_refused = revision != "09d9b51f" and unchecked and not dry
    if is_refused:
        assert result.exit_code == 2 and result.stderr.startswith("bts schedule: refusing to start: "), result.output
        assert opens == ["config.toml"] and not file_access and not files, (opens, file_access, files)
        assert not any(e[0] in {"default-date", "run-day", "schedule", "ready", "watchdog", "sleep", "predictions"}
                       for e in events), events
    else:
        assert result.exit_code == 0, (result.output, repr(result.exception), events)
        expected = OLD
        early_date = revision == "09d9b51f" or (revision == "HEAD" and dry)
        if not explicit and (boundary == "before-date" or
                             (not early_date and boundary in {"inside-load", "after-gate"})):
            expected = NEW
        assert [e[1] for e in events if e[0] == "run-day"] == [expected], events
        assert [e[1] for e in events if e[0] == "schedule"][0] == expected, events
        assert [e[1] for e in events if e[0] == "default-date"] == ([] if explicit else [expected]), events
        assert result.stderr.startswith("bts schedule: WARNING (a real run would refuse to start): ") == (
            revision != "09d9b51f" and unchecked and dry), result.output
    assert opens.count("config.toml") == 1, opens
    assert not local_guard.network, local_guard.network
    return {"exit": result.exit_code, "stdout": result.stdout, "stderr": result.stderr,
            "events": events, "opens": opens, "path_ops": file_access, "files": files}


def test_help():
    from click.testing import CliRunner
    from bts import cli, orchestrator

    with patch.object(orchestrator, "load_config", side_effect=AssertionError("help read TOML")) as load:
        result = CliRunner().invoke(cli.cli, ["entry-intent", "--help"], terminal_width=100)
        load.assert_not_called()
    assert result.exit_code == 0 and not result.stderr, result.output
    return result.stdout


def archive(revision, dest):
    archive_bytes = subprocess.check_output(["git", "archive", revision, "src"], cwd=ROOT)
    with tarfile.open(fileobj=BytesIO(archive_bytes)) as archive_file:
        archive_file.extractall(dest, filter="data")
    return dest / "src"


def child(revision, src, root):
    sys.path.insert(0, src)
    from bts import cli

    assert Path(cli.__file__).is_relative_to(Path(src)), cli.__file__
    rows = {}
    params = product(BOUNDARIES, CONFIG, (False, True), (False, True))
    for index, (boundary, config, dry, explicit) in enumerate(params):
        key = f"{boundary}/{config}/{('dry' if dry else 'real')}/{('explicit' if explicit else 'default')}"
        rows[key] = scenario(Path(root) / str(index), revision, boundary, config, dry, explicit)
    print(json.dumps({"rows": rows, "help": test_help()}))


def comparable_dry(row):
    stderr = "".join(line for line in row["stderr"].splitlines(keepends=True)
                     if not line.startswith("bts schedule: WARNING "))
    worker = [e for e in row["events"] if e[0] in {"run-day", "schedule", "ready", "watchdog", "sleep"}]
    return row["exit"], row["stdout"], stderr, worker, row["files"], row["opens"], row["path_ops"]


def main():
    data = {}
    with TemporaryDirectory(prefix="schedule-runs-", dir=HERE) as folder:
        temp = Path(folder)
        # Identical input paths across revisions make stdout and filesystem traces comparable.
        import shutil

        for revision in ("09d9b51f", "cad7c5ec", "HEAD"):
            src = ROOT / "src" if revision == "HEAD" else archive(revision, temp / revision)
            scratch = temp / "scenarios"
            scratch.mkdir()
            result = subprocess.run([sys.executable, str(__file__), "child", revision, str(src), str(scratch)],
                                    capture_output=True, text=True, cwd=ROOT)
            assert result.returncode == 0, result.stderr
            data[revision] = json.loads(result.stdout)
            print(f"{revision}: {len(data[revision]['rows'])} scenarios")
            shutil.rmtree(scratch)
    (HERE / "schedule-traces.json").write_text(json.dumps(data, indent=2) + "\n")
    base, old, head = (data[rev]["rows"] for rev in ("09d9b51f", "cad7c5ec", "HEAD"))
    keys = [key for key in base if "/dry/" in key]
    for key in keys:
        assert comparable_dry(head[key]) == comparable_dry(base[key]), key
    print(f"HEAD matches all {len(keys)} base dry runs (excluding the required warning).")
    print("Round 1 dry-run divergences:", [key for key in keys if comparable_dry(old[key]) != comparable_dry(base[key])])
    assert data["09d9b51f"]["help"] == data["cad7c5ec"]["help"]
    print(data["HEAD"]["help"])


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(*sys.argv[2:])
    else:
        main()
