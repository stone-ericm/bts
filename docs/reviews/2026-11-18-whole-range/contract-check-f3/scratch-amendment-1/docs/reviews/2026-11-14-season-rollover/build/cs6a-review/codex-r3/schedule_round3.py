"""Record the real schedule command and worker at each midnight boundary."""

from contextlib import ExitStack, contextmanager
from datetime import datetime, timedelta
from itertools import count, product
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from unittest.mock import patch

from schedule_round2 import AFTER, BEFORE, CONFIG, GAME, HERE, NEW, OLD, ROOT, archive
import local_guard


BASE, PRIOR, HEAD = "09d9b51f", "87c118f6", "HEAD"
REVISIONS = BASE, PRIOR, HEAD
BOUNDARIES = ("before-date", "inside-load", "after-gate", "steady")
CONFIGS = {
    **{name: "[scheduler]\n" + values for name, values in CONFIG.items()},
    "unchecked-public": "[scheduler]\ncontest_state_required = false\n",
    "unchecked-both": '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n',
    "invalid-delivery": '[scheduler]\npick_delivery = "unknown"\n',
    "invalid-scheduler": "scheduler = 3\n",
    "invalid-health": 'health_checks = false\n[scheduler]\npick_delivery = "private"\n',
    "invalid-toml": "[scheduler",
}
REASONS = {
    "unchecked-dm": "scheduler delivery 'dm' requires contest state.",
    "unchecked-public": "scheduler delivery 'public' requires contest state.",
    "unchecked-enter": "scheduler.entry_intent = 'enter' requires contest state.",
    "unchecked-both": "scheduler delivery 'dm' and scheduler.entry_intent = 'enter' require contest state.",
}
FIX = (
    " Optional reads can use a contest observation without the season-coverage check."
    " Set contest_state_required = true under [scheduler] and"
    " contest_state_expected = true under [health_checks]."
)
PROBLEMS = {name: reason + FIX for name, reason in REASONS.items()} | {
    "invalid-scheduler": "[scheduler] is not a table",
    "invalid-health": "[health_checks] is not a table",
}
REFUSAL = "bts schedule: refusing to start: "
WARNING = "bts schedule: WARNING (a real run would refuse to start): "
AUDIT_PATHS = {"open": 1, "os.mkdir": 1, "os.listdir": 1, "os.scandir": 1,
               "os.remove": 1, "os.rmdir": 1, "os.rename": 2}
active_trace = None


def audit(event, args):
    if active_trace is not None and event in AUDIT_PATHS:
        active_trace(event, *args[:AUDIT_PATHS[event]], details=args[1:] if event == "open" else ())
    if active_trace is not None and event in {"subprocess.Popen", "os.system", "os.exec", "os.posix_spawn"}:
        active_trace(event)
        raise PermissionError(f"schedule probe blocked {event}")


sys.addaudithook(audit)


def scenario(root, revision, boundary, config_name, dry, explicit):
    import click
    from click.testing import CliRunner
    import tomllib
    from bts import cli, entry_intent, heartbeat, orchestrator, scheduler
    from bts.health import alert

    global active_trace
    root.mkdir()
    config_path = root / "config.toml"
    resources = root / "resources"
    body = CONFIGS[config_name]
    if "health_checks" not in body:
        body = "[health_checks]\nenabled = false\n" + body if config_name not in {
            "invalid-scheduler", "invalid-toml",
        } else body
    source_paths = [Path(cli.__file__).parents[1], ROOT / "src"]
    config_path.write_text(
        f'tiers = []\n{body}\n[orchestrator]\npicks_dir = "{resources / "picks"}"\n'
        f'heartbeat_path = "{resources / "heartbeat" / "beat.json"}"\n'
        f'models_dir = "{resources / "models"}"\n'
    )
    now = BEFORE
    events, imports, snapshot = [], [], []
    current_accesses = []
    read_toml, load_config = tomllib.load, orchestrator.load_config
    today, run_day = cli._today_et, scheduler.run_day
    gate = getattr(entry_intent, "live_gate_problem", None)
    convert = click.types.Path.convert

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
        if value == str(config_path):
            advance("before-date")
        return result

    def parse(stream, *args, **kwargs):
        try:
            return read_toml(stream, *args, **kwargs)
        finally:
            events.append(("toml-read",))
            advance("inside-load")

    def load(path):
        events.append(("load-start",))
        result = load_config(path)
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
        return result

    def start_day(*, date, config, dry_run):
        advance("after-gate")  # Match the post-gate worker boundary at the base.
        assert config is snapshot[0]
        events.append(("run-day", date, dry_run, now.isoformat()))
        return run_day(date=date, config=config, dry_run=dry_run)

    def fetch(day):
        events.append(("schedule", day))
        return [GAME] if day == OLD else []

    def lineups(pks):
        events.append(("lineups", *pks))
        return {}

    def sleep(seconds):
        nonlocal now
        events.append(("sleep", seconds))
        now += timedelta(seconds=seconds)

    def cascade(tiers, day, *args, **kwargs):
        assert tiers is snapshot[0]["tiers"]
        events.append(("predictions", day))
        return None, None

    @contextmanager
    def beat_during_cascade(path, **kwargs):
        events.append(("cascade-watchdog", str(path.relative_to(root)), kwargs["date"]))
        yield

    def path_label(path):
        if not isinstance(path, (str, bytes, os.PathLike)):
            return "<fd>"
        candidate = Path(os.fsdecode(path)).absolute()
        if candidate.is_relative_to(root):
            return str(candidate.relative_to(root))
        for src in source_paths:
            if candidate.is_relative_to(src):
                return "src/" + str(candidate.relative_to(src))
        return str(candidate)

    def trace(label, *paths, details=()):
        names = tuple(path_label(path) for path in paths)
        record = (label, *names, *details)
        if any(name.endswith((".py", ".pyc", ".so", ".dylib")) for name in names):
            imports.append(record)
        else:
            current_accesses.append(record)

    def watch_path(label, operation):
        def observed(file, *args, **kwargs):
            if isinstance(file, (str, bytes, os.PathLike)):
                filepath = Path(os.fsdecode(file)).absolute()
                # Python's import machinery calls stat internally. Record app and
                # Click file operations, including any path outside the fixture.
                frame = sys._getframe(1)
                while frame and frame.f_globals.get("__name__") in {"genericpath", "pathlib", "posixpath"}:
                    frame = frame.f_back
                module = frame.f_globals.get("__name__", "") if frame else ""
                if filepath.is_relative_to(root) or module.startswith(("bts.", "click.")):
                    trace(label, file)
            return operation(file, *args, **kwargs)
        return observed

    argv = ["schedule", "--config", str(config_path)]
    if explicit:
        argv += ["--date", OLD]
    if dry:
        argv.append("--dry-run")
    temps = (f"probe-{number}" for number in count())
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
            (scheduler, "check_confirmed_lineups", lineups),
            (scheduler, "heartbeat_watchdog", beat_during_cascade),
            (orchestrator, "run_cascade", cascade),
            (alert, "dispatch_dm_for_health_alerts", lambda *a, **kw: events.append(("alert",))),
            (tempfile, "_get_candidate_names", lambda: temps),
        ]
        for name in ("stat", "lstat", "access"):
            patches.append((os, name, watch_path(name, getattr(os, name))))
        if gate is not None:
            patches.append((entry_intent, "live_gate_problem", gate_then_clock))
        for obj, name, replacement in patches:
            stack.enter_context(patch.object(obj, name, replacement))
        active_trace = trace
        try:
            result = CliRunner().invoke(cli.cli, argv)
        finally:
            active_trace = None

    files = {str(file.relative_to(root)): json.loads(file.read_text()) for file in resources.rglob("*.json")}
    assert not local_guard.network, local_guard.network
    exception = result.exception
    return {"exit": result.exit_code, "stdout": result.stdout, "stderr": result.stderr,
            "exception": type(exception).__name__ if exception is not None else None,
            "events": events, "accesses": current_accesses, "imports": imports, "files": files}


def child(revision, src, root):
    sys.path.insert(0, src)
    from bts import cli, contest_state, daily_decision, heartbeat, orchestrator, picks, scheduler, strategy
    from bts.health import alert

    assert Path(cli.__file__).is_relative_to(Path(src)), cli.__file__
    rows = {}
    for index, params in enumerate(product(BOUNDARIES, CONFIGS, (False, True), (False, True))):
        boundary, config, dry, explicit = params
        key = f"{boundary}/{config}/{'dry' if dry else 'real'}/{'explicit' if explicit else 'default'}"
        rows[key] = scenario(Path(root) / str(index), revision, *params)
    print(json.dumps(rows))


def compare_worker(row, message="", *, ignore_date_lookup=False):
    ignored = {"gate", "default-date"} if ignore_date_lookup else {"gate"}
    events = [event for event in row["events"] if event[0] not in ignored]
    return (row["exit"], row["exception"], row["stdout"], row["stderr"].removeprefix(message),
            events, row["accesses"], row["files"])


def validate(data):
    accepted = warned = refused = load_errors = 0
    prior_shift = []
    for key, base in data[BASE].items():
        boundary, config, mode, date_option = key.split("/")
        problem = PROBLEMS.get(config)
        dry = mode == "dry"
        explicit = date_option == "explicit"
        expected_stderr_prefix = WARNING + problem + "\n" if dry and problem else ""
        if config == "invalid-toml":
            load_errors += 1
        elif problem and dry:
            warned += 1
        elif problem:
            refused += 1
        else:
            accepted += 1

        for rev in REVISIONS:
            row = data[rev][key]
            events = row["events"]
            dates = [event[1] for event in events if event[0] == "default-date"]
            workers = [event[1] for event in events if event[0] == "run-day"]
            gates = [event[1] for event in events if event[0] == "gate"]
            assert gates == ([] if rev == BASE or config == "invalid-toml" else [problem]), (rev, key, row)
            assert [event for event in row["accesses"] if event[0] == "open" and
                    event[1] == "config.toml"] == [["open", "config.toml", "r", os.O_CLOEXEC]], (rev, key, row)
            if rev != BASE and problem and not dry:
                assert (row["exit"], row["stdout"], row["stderr"]) == (2, "", REFUSAL + problem + "\n"), row
                assert not workers and not row["files"] and not row["imports"], (rev, key, row)
                assert all(event[0] in {"midnight", "default-date", "load-start", "toml-read", "load-end", "gate"}
                           for event in events), (rev, key, row)
                assert all(event[1] == "config.toml" and event[0] in {"open", "stat", "access"}
                           for event in row["accesses"]), (rev, key, row)
                assert len(dates) == (0 if explicit or rev == PRIOR else 1), (rev, key, row)
            else:
                if rev != BASE:
                    if expected_stderr_prefix:
                        assert row["stderr"].startswith(expected_stderr_prefix), (rev, key, row)
                    prior_late = rev == PRIOR and not dry and not explicit
                    if not prior_late:
                        assert compare_worker(row, expected_stderr_prefix) == compare_worker(base), (rev, key, row, base)
                    elif row["exception"] != "TOMLDecodeError" and boundary in {"inside-load", "after-gate"}:
                        assert workers == [NEW] and [event[1] for event in base["events"] if event[0] == "run-day"] == [OLD]
                        prior_shift.append(key)
                    else:
                        assert compare_worker(row, ignore_date_lookup=True) == compare_worker(
                            base, ignore_date_lookup=True
                        ), (rev, key, row, base)
                if config == "invalid-toml":
                    assert row["exception"] == "TOMLDecodeError" and not workers, (rev, key, row)
                else:
                    expected_date = OLD
                    if not explicit and (boundary == "before-date" or (
                        rev == PRIOR and not dry and boundary in {"inside-load", "after-gate"}
                    )):
                        expected_date = NEW
                    assert workers == [expected_date], (rev, key, row)
                    assert dates == ([] if explicit else [expected_date]), (rev, key, row)
                    if not config.startswith("invalid-"):
                        assert row["exit"] == 0, (rev, key, row)
                        schedules = [event[1] for event in events if event[0] == "schedule"]
                        assert schedules[0] == expected_date, (rev, key, row)
                        assert row["files"] and any(event[0] == "ready" for event in events), (rev, key, row)
                        if expected_date == OLD and not dry:
                            assert any(event[0] == "predictions" for event in events), (rev, key, row)
            if rev in {BASE, HEAD} and not explicit:
                assert [event[0] for event in events].index("default-date") < [
                    event[0] for event in events
                ].index("load-start"), (rev, key, row)
    print(f"HEAD matched base on {accepted} gate-accepted cases, {warned} dry warnings apart from their prefix, "
          f"and {load_errors} loader errors. {refused} real cases refused exactly.")
    print(f"Prior revision selected the next day on {len(prior_shift)} post- or mid-load real default dates:")
    print(*prior_shift, sep="\n")
    for key in ("inside-load/required-public/real/default", "after-gate/private/real/default",
                "before-date/unchecked-dm/real/default", "inside-load/unchecked-both/dry/default"):
        for rev in REVISIONS:
            item = data[rev][key]
            details = [event for event in item["events"] if event[0] not in {"gate", "watchdog"}]
            print(rev, key, "exit", item["exit"], details)


def main():
    data = {}
    with tempfile.TemporaryDirectory(prefix="schedule-runs-", dir=HERE) as folder:
        temp = Path(folder)
        for rev in REVISIONS:
            src = ROOT / "src" if rev == HEAD else archive(rev, temp / rev)
            scratch = temp / "scenarios"
            scratch.mkdir()
            result = subprocess.run([sys.executable, __file__, "child", rev, str(src), str(scratch)],
                                    capture_output=True, text=True, cwd=ROOT, timeout=90)
            assert result.returncode == 0, result.stderr
            data[rev] = json.loads(result.stdout)
            print(f"{rev}: {len(data[rev])} scenarios", flush=True)
            shutil.rmtree(scratch)
    (HERE / "schedule-traces.json").write_text(json.dumps(data, indent=2) + "\n")
    validate(data)


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "child":
        child(*sys.argv[2:])
    else:
        main()
