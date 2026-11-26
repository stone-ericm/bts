"""Run both guard implementations with local, pinned fixtures and stand-in commands."""

from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
import datetime as datetime_module
import fcntl
import hashlib
from importlib import import_module
from importlib.util import module_from_spec, spec_from_file_location
from io import BytesIO, StringIO
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
import time
import traceback
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pytest

import bts
from bts import orchestrator, scheduler
from bts.cli import cli
from tests.test_contest_dated_cs4a import observation, write
from tests.test_deploy_hold_tool import load_guard
from tests.test_web_saver import _setup, _saver_state, NOW as SAVER_NOW

from run_review import GUARD, HEAD, HERE, LEGACY, MONITOR


REV = sys.argv[1]
ROOT = Path.cwd()
ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime.fromisoformat(DAY + "T18:00:00").replace(tzinfo=ET)
MIDNIGHT = datetime(2027, 7, 16, tzinfo=ET).timestamp()
RUN, FOLLOWUP = "1800000000-review", "1800000001-followup"
PREVIOUS, TARGET = "a" * 40, "b" * 40
GUARD_BIN = "/usr/local/bin/bts-deploy-hold"
CONDITION = f"/usr/bin/python3 -I {GUARD_BIN} check-clear --scheduler"
MONITOR_CMD = "/usr/bin/python3 scripts/check_heartbeat.py --heartbeat-path data/.heartbeat --deploy-hold /var/lib/bts-deploy/hold"
CLI = import_module("bts.cli")
RUNNER = CliRunner()


def capture(callback):
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            code = callback()
        except SystemExit as exc:
            code = exc.code
    return code, out.getvalue(), err.getvalue()


def fixture(place):
    place.mkdir(parents=True, exist_ok=True)
    root = place / "guard"
    root.mkdir(mode=0o755)
    (place / "data").mkdir()
    for name in ("state.lock", "transaction.lock", "writers.lock"):
        (root / name).touch(mode=0o644)
    guard = load_guard(root)
    guard.PROC_LOCKS = place / "locks"
    guard.PROC_LOCKS.write_text("")
    guard.os = SimpleNamespace(**vars(os))
    guard.os.path = SimpleNamespace(**vars(os.path))
    guard.os.path.lexists = lambda p: str(p).startswith(str(place)) and os.path.lexists(p)
    guard.time = SimpleNamespace(time=NOW.timestamp, monotonic=time.monotonic, sleep=lambda _: None)
    return guard


def hold(state="failed", *, step="sync", when=NOW.timestamp()):
    timestamp = int(when) - 10
    return {"schema": "bts_deploy_hold_v1", "state": state, "step": step, "deploy_id": RUN,
            "owner_unit": f"bts-deploy-{RUN}.service", "target": TARGET, "previous": PREVIOUS,
            "armed_at": timestamp, "updated_at": timestamp}


def inventory():
    commands = ["check-results", "reconcile", "reconcile", *(["fetch-contest-streak"] * 4), "preview"]
    writers = []
    for index, command in enumerate(commands):
        job = "nightly-preview" if command == "preview" else command
        writers.append(f"{index} 3 * * * {GUARD_BIN} run {job} -- /bin/sh -c "
                       f"'uv run bts {command} --fixture' >> /home/bts/logs/cron.log 2>&1 # BTS-HETZNER")
    return writers + [f"*/5 * * * * cd /home/bts/projects/bts && {MONITOR_CMD}"]


def structured(command, path=None):
    executable = path if path is not None else command.split()[0]
    return f"{{ path={executable} ; argv[]={command} ; ignore_errors=no ; start_time=[n/a] ; }}"


def preflight_shim(guard, cron=None, overrides=None):
    installer = (ROOT / GUARD).read_bytes()
    (guard.ROOT / "guard-install.sha256").write_text(hashlib.sha256(installer).hexdigest() + "\n")
    requested = []

    def call(argv, *, limit=10):
        if "/usr/bin/crontab" in argv:
            return 0, ("\n".join(inventory() if cron is None else cron) + "\n").encode()
        if "/usr/bin/systemctl" in argv and "show" in argv:
            if "--user" not in argv:
                return 0, b"not-found\n"
            name = argv[argv.index("show") + 1]
            schedule = name == "bts-scheduler.service"
            args = f"bts schedule --config {guard.CONFIG}" if schedule else "python3 -m bts.web"
            props = {"LoadState": "loaded", "NeedDaemonReload": "no", "ExecStart": structured(f"{guard.UV} run --no-sync {args}"),
                     "ExecCondition": structured(CONDITION), "SuccessExitStatus": "143", "WorkingDirectory": guard.PROJECT,
                     "Environment": "UV_CACHE_DIR=/tmp/uv-cache PATH=/home/bts/.local/bin:/usr/local/bin:/usr/bin:/bin",
                     "EnvironmentFiles": f"{guard.PROJECT}/.env (ignore_errors=no)"}
            if schedule and overrides:
                props.update(overrides)
            keys = [argv[i + 1] for i, arg in enumerate(argv) if arg == "-p"]
            if schedule:
                requested.extend(keys)
            return 0, ("\n".join(props[key] if "--value" in argv else f"{key}={props[key]}" for key in keys) + "\n").encode()
        if "/usr/bin/git" in argv:
            if "rev-parse" in argv:
                return 0, PREVIOUS.encode() + b"\n"
            if "cat-file" in argv:
                return 0, b"commit\n"
            if "show" in argv:
                return 0, installer
            if "config" in argv or "fetch" in argv:
                return 0, b""
        raise AssertionError(f"unexpected subprocess {argv}")

    guard.call = call
    return requested


def verify(guard):
    try:
        guard.preflight(1000, TARGET, fetch=False)
    except guard.Refused as exc:
        return exc.step, exc.previous
    return "accepted", PREVIOUS


def cron_and_units(place):
    guard = fixture(place)
    canonical = inventory()
    cron_cases = {
        "canonical": canonical,
        "quoted": canonical + ["0 4 * * * uv run 'bts' 'reconcile'"],
        "module": canonical + ["0 4 * * * .venv/bin/python -m bts.cli reconcile"],
        "wrapped": canonical + ["0 4 * * * /bin/sh -c 'uv run bts reconcile'"],
        "trailing-command": [canonical[0].replace(" # BTS-HETZNER", "; uv run bts check-results # BTS-HETZNER"), *canonical[1:]],
        "monitor-commented": [*canonical[:-1], "# " + canonical[-1]],
        "monitor-short-circuited": [*canonical[:-1], f"*/5 * * * * /usr/bin/true || {MONITOR_CMD}"],
        "writer-name-in-diagnostic": [canonical[0].replace("uv run bts", "echo bts"), *canonical[1:]],
        "escaped-token-parse-only": canonical + [r"0 4 * * * uv run bts rec\oncile"],
    }
    for label, rows in cron_cases.items():
        preflight_shim(guard, cron=rows)
        print("F1", label, *verify(guard))

    # Check the last two semantic cases on an inert program. Nothing launches bts.
    escaped = subprocess.run(["/bin/sh", "-c", r"printf '%s\n' rec\oncile"], cwd=place,
                             capture_output=True, text=True, timeout=4, check=True)
    no_heartbeat = subprocess.run(["/bin/sh", "-c", "true || printf monitor"], cwd=place,
                                  capture_output=True, text=True, timeout=4, check=True)
    print("F1 shell controls", escaped.stdout.strip(), repr(no_heartbeat.stdout))

    units = {
        "canonical": {},
        "wrong-condition-path": {"ExecCondition": structured(CONDITION, "/usr/bin/true")},
        "condition-exit-1-success": {"SuccessExitStatus": "143 1"},
        "alternate-working-directory": {"WorkingDirectory": "/srv/alternate"},
        "alternate-environment-file": {"EnvironmentFiles": "/srv/alternate/.env (ignore_errors=no)"},
        "alternate-root": {"RootDirectory": "/srv/restored-image"},
        "alternate-data-mount": {"BindPaths": "/srv/restore/data:/home/bts/projects/bts/data"},
    }
    for label, props in units.items():
        requested = preflight_shim(guard, overrides=props)
        outcome = verify(guard)
        print("F3", label, *outcome, "requested", sorted(set(props) & set(requested)))

    preflight_shim(guard, overrides={"NeedDaemonReload": "yes"})
    guard.bts_uid = lambda: 1000
    with guard.directory() as fd:
        rc, stdout, _ = capture(lambda: guard.owned_deploy(fd, RUN, TARGET, ""))
    print("F14", rc, " ".join(stdout.splitlines()))

    classified = f"deploy canary {NOW.isoformat()}: live contest state accepted (dated, streak 12)\n".encode()
    for name, exit_code, output in (
        ("classified-success", 0, classified),
        ("classified-success-with-warning", 0, b"warning: local runtime diagnostic\n" + classified),
        ("classified-refusal", 1, classified.replace(b"live contest state accepted (dated, streak 12)", b"contest state refused")),
    ):
        guard.call = lambda *args, **kwargs: (exit_code, output)

        def classify():
            try:
                return guard.canary()
            except guard.Refused as exc:
                return exc.step

        print("F14 canary output", name, capture(classify))


def liveness(place):
    guard = fixture(place)
    info = (guard.ROOT / "transaction.lock").stat()
    record = hold("activating", step="activation")

    def reply(argv, **kwargs):
        assert argv[:2] == ["/usr/bin/systemctl", "show"] and kwargs["timeout"] == 3
        output = f"ActiveState={state}\nMainPID={pid}\n" if "MainPID" in argv else state + "\n"
        return SimpleNamespace(returncode=0, stdout=output.encode())

    guard.subprocess = SimpleNamespace(run=reply, DEVNULL=subprocess.DEVNULL, TimeoutExpired=subprocess.TimeoutExpired)
    guard.os.execv = lambda *args: (_ for _ in ()).throw(AssertionError("writer reached exec"))
    with guard.directory() as fd, guard.lock(fd, "transaction.lock", fcntl.LOCK_EX):
        guard.publish_json(fd, "hold", record)
        for label, state, pid, holder, inode in (
            ("current-owner", "active", 42, 42, info.st_ino),
            ("old-deactivating-owner-new-lock-holder", "deactivating", 42, 43, info.st_ino),
            ("active-but-wrong-lock-holder", "active", 42, 43, info.st_ino),
            ("active-but-wrong-inode", "active", 42, 42, info.st_ino + 1),
            ("stale-active-no-lock", "active", 42, 42, None),
        ):
            row = "" if inode is None else f"3: FLOCK ADVISORY WRITE {holder} {os.major(info.st_dev):x}:{os.minor(info.st_dev):x}:{inode} 0 EOF\n"
            guard.PROC_LOCKS.write_text(row)
            print("F2", label, guard.clear(fd))
        state, pid = "active", 42
        guard.PROC_LOCKS.write_text(f"3: FLOCK ADVISORY WRITE 42 {os.major(info.st_dev):x}:{os.minor(info.st_dev):x}:{info.st_ino} 0 EOF\n")
        guard.publish_json(fd, "hold", hold("pending", step="arm"))
        with patch.object(sys, "argv", ["bts-deploy-hold", "check-clear", "--scheduler"]):
            print("F12 pending", guard.main(), [capture(lambda: guard.writer(job, ["/usr/bin/true"]))[0] for job in guard.JOBS])

        exited = subprocess.Popen([sys.executable, "-I", "-c", "pass"], stdin=subprocess.DEVNULL)
        assert exited.wait(timeout=5) == 0
        state, pid = "active", exited.pid
        row = f"3: FLOCK ADVISORY WRITE {pid} {os.major(info.st_dev):x}:{os.minor(info.st_dev):x}:{info.st_ino} 0 EOF\n"
        guard.PROC_LOCKS.write_text(row)
        guard.publish_json(fd, "hold", record)
        print("F2 simulated stale manager and inherited-lock owner PID", pid, "exited", exited.returncode, "clear", guard.clear(fd))


def invoke_canary(config, now=NOW):
    with patch.object(CLI, "_now_et", return_value=now):
        result = RUNNER.invoke(cli, ["deploy-canary", "--config", str(config)])
    assert result.output.startswith(f"deploy canary {now.isoformat()}:"), (result.output, result.exception)
    return result


def fail_if_called(*args, **kwargs):
    raise AssertionError("an unmodeled side effect was called")


def canaries(place):
    picks = place / "data/picks"
    write(picks, observation(DAY))
    (place / "regular-file").write_text("not a directory")
    example = (ROOT / "config/orchestrator.example.toml").read_text()
    dm = example.replace('pick_delivery = "public"', 'pick_delivery = "dm"')
    research = dm.replace('pick_delivery = "dm"', 'pick_delivery = "private"').replace('entry_intent = "enter"', 'entry_intent = "research"')
    local = dm.split("[[tiers]]")[0] + '[[tiers]]\nname = "local"\ntype = "local"\n'
    configs = {
        "example-public": example, "dm": dm, "research": research, "local-tier": local,
        "private-mode-alias": research.replace('pick_delivery = "private"', 'private_mode = true'),
        "bad-budget-dm": dm.replace("[scheduler]", '[scheduler]\ncascade_budget_min = "forty"'),
        "bad-budget-public": example.replace("[scheduler]", '[scheduler]\ncascade_budget_min = "forty"'),
        "bad-budget-research": research.replace("[scheduler]", '[scheduler]\ncascade_budget_min = "forty"'),
        "missing-tiers": dm.split("[[tiers]]")[0],
        "missing-recipient": re.sub(r"^dm_recipient = .*\n", "", dm, flags=re.M),
        "heartbeat-parent-not-directory": dm.replace("[orchestrator]", '[orchestrator]\nheartbeat_path = "regular-file/.heartbeat"'),
        "explicit-linux-tier": dm.replace("timeout_min = 5", 'timeout_min = 5\nplatform = "linux"', 1),
    }
    previous = Path.cwd()
    try:
        os.chdir(place)
        for label, contents in configs.items():
            config = Path(label + ".toml")
            config.write_text(contents)
            before = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in Path.cwd().rglob("*") if p.is_file()}
            with patch.object(scheduler, "run_day", side_effect=fail_if_called):
                result = invoke_canary(config)
            after = {p: (p.stat().st_mtime_ns, p.read_bytes()) for p in Path.cwd().rglob("*") if p.is_file()}
            assert before == after
            print("F5 canary", label, result.exit_code, result.output.strip())
            if label.startswith("bad-budget") or label == "heartbeat-parent-not-directory":
                with patch.object(scheduler, "notify_ready") as ready, patch.object(scheduler, "fetch_schedule", side_effect=fail_if_called):
                    start = RUNNER.invoke(cli, ["schedule", "--config", str(config), "--date", DAY])
                    assert not ready.called
                print("F5 real startup", label, type(start.exception).__name__)
            elif label == "missing-tiers":
                try:
                    orchestrator.run_and_pick(orchestrator.load_config(config), DAY)
                except KeyError as err:
                    print("F5 real selection", label, repr(err))
            elif label == "explicit-linux-tier":
                called = []

                def ssh(argv, **kwargs):
                    called.append((argv[0], argv[-2], kwargs["timeout"]))
                    return SimpleNamespace(returncode=0, stdout='[{"batter_id": 7}]', stderr="")

                with patch.object(orchestrator.subprocess, "run", side_effect=ssh):
                    predictions, tier = orchestrator.run_cascade(orchestrator.load_config(config)["tiers"], DAY)
                assert predictions.iloc[0]["batter_id"] == 7
                print("F5 real cascade", tier, called)
            elif label == "missing-recipient":
                from bts import contest_state, picks as pick_module
                daily = SimpleNamespace(delivery_attempted=False)
                with patch.object(scheduler, "_void_committed_games", return_value={}), \
                     patch.object(scheduler, "_now_et", return_value=NOW), \
                     patch.object(scheduler, "_format_pick_delivery_text", return_value="fixture"), \
                     patch.object(pick_module, "pick_was_delivered", return_value=False), \
                     patch.object(pick_module, "submission_cutoff_et", return_value=NOW + timedelta(minutes=30)), \
                     patch.object(pick_module, "saved_pick_refusal", return_value=None), \
                     patch.object(contest_state, "load_decision_streak_state", return_value=SimpleNamespace(streak=12)):
                    outcome = scheduler._deliver_and_lock_pick(daily, orchestrator.load_config(config), picks, None, DAY, "fixture")
                print("F5 real delivery", outcome, "attempted", daily.delivery_attempted)
        boundary = Path("example-public.toml")
        for day in (DAY, "2027-12-31"):
            write(picks, observation(day, recorded_at=day + "T17:00:00Z"))
            before = datetime.fromisoformat(day + "T23:59:59").replace(tzinfo=ET)
            after = before + timedelta(seconds=2)
            print("F6 dated CLI", day, invoke_canary(boundary, before).exit_code, invoke_canary(boundary, after).exit_code)
        operator_docs(place, dm)
    finally:
        os.chdir(previous)


def operator_docs(place, dm):
    incident = (ROOT / "INCIDENT.md").read_text()
    configured = place / "data/verified-account"
    configured.mkdir()
    config = place / "recovery.toml"
    config.write_text(dm.replace('picks_dir = "data/picks"', 'picks_dir = "data/verified-account"'))
    argv = ["set-contest-streak", "--date", DAY, "--streak", "12", "--best-streak", "14", "--saver", "not_earned"]
    status = ["season-contract", "status"]
    if REV == HEAD:
        matches = re.findall(r"`(uv run --no-sync bts (?:season-contract status|set-contest-streak) [^`]+)`", incident)
        replacements = {"<the configured picks_dir>": shlex.quote(str(configured)), "<today ET>": DAY,
                        "<active>": "12", "<best>": "14", "<state>": "not_earned"}
        commands = []
        for line in matches:
            for key, value in replacements.items():
                line = line.replace(key, value)
            commands.append(shlex.split(line)[4:])
        status, argv = commands[:2]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

    with patch.object(datetime_module, "datetime", Clock), patch.object(CLI, "_now_et", return_value=NOW):
        updated, readback = RUNNER.invoke(cli, argv), RUNNER.invoke(cli, status)
    assert updated.exit_code == 0 and readback.exit_code == 0, (updated.output, readback.output)
    recorded = (configured / "account_state/contest_streak.manual.json").exists()
    print("F7 configured observation", recorded, "canary", invoke_canary(config).exit_code,
          "same guard protocol", "same guard protocol" in incident)


def module(name, source):
    spec = spec_from_file_location(name, source)
    result = module_from_spec(spec)
    sys.modules[name] = result
    spec.loader.exec_module(result)
    return result


def dashboard_case(place):
    old_web = module("legacy_web", HERE / "trees" / LEGACY / "src/bts/web.py")
    for original_running in (False, True):
        work = place / ("old-running" if original_running else "old-stopped")
        guard = fixture(work)
        picks = work / "picks"
        _setup(picks, "active")
        running = original_running
        statuses = []

        class Clock(datetime):
            @classmethod
            def now(cls, tz=None):
                return SAVER_NOW.astimezone(tz) if tz else SAVER_NOW.replace(tzinfo=None)

        def post():
            request = object.__new__(old_web.Handler)
            form = b"expected_prior=active&new_state=used"
            request.rfile, request.wfile = BytesIO(form), BytesIO()
            request.path = "/saver/transition"
            request.headers = {"Content-Length": str(len(form)), "Host": "localhost"}
            request.client_address = ("127.0.0.1", 0)
            request.send_response = statuses.append
            request.send_header = request.end_headers = lambda *args: None
            with patch.object(old_web, "PICKS_DIR", picks), patch.object(old_web, "datetime", Clock):
                request.do_POST()

        def stop_dashboard(uid):
            nonlocal running
            with guard.directory() as fd:
                assert guard.read_hold(fd)["state"] == "pending"
                with pytest.raises(BlockingIOError), guard.lock(fd, "writers.lock", fcntl.LOCK_SH):
                    pass
            running = False
            return True

        clock = SimpleNamespace(now=NOW.timestamp())
        txn(guard, clock)
        guard.verified_stop_dashboard = stop_dashboard
        require = guard.require

        def stage(argv, step, expected=None, **kwargs):
            if step == "reset" and running:
                post()
            return require(argv, step, expected, **kwargs)

        guard.require = stage
        with guard.directory() as fd:
            rc, _, _ = capture(lambda: guard.owned_deploy(fd, RUN, TARGET, ""))
        print("F4 old dashboard at reset", original_running, "deploy", rc, "POST", statuses, "saver", _saver_state(picks))


def txn(guard, clock):
    events = []
    guard.time.time = lambda: clock.now
    guard.bts_uid = lambda: 1000
    guard.preflight = guard.verified_target = lambda *args, **kwargs: (PREVIOUS, ["fixture-git"])
    guard.require = lambda argv, step, expected=None, **kwargs: expected or b""
    guard.call = fail_if_called
    guard.verified_stop = lambda uid: (events.append("stop"), True)[1]
    guard.window = lambda *args, **kwargs: None
    guard.dashboard = lambda uid: None
    guard.activate_scheduler = lambda uid, at: events.append("start")
    guard.canary = lambda: datetime.fromtimestamp(clock.now, ET).date().isoformat()
    guard.page = lambda hold, stopped: (events.append(("page", stopped)), 204)[1]
    guard.write_stamp = lambda: True
    return events


def lifecycle(place):
    for label in ("normal", "during-start", "during-marker-fsync", "midnight-before-unlink", "same-day-state-disappears"):
        guard = fixture(place / label)
        clock = SimpleNamespace(now=MIDNIGHT - (3600 if label == "normal" else 1))
        events = txn(guard, clock)
        picks = Path(guard.PROJECT) / "data/picks"
        state = write(picks, observation(DAY))
        config = Path(guard.PROJECT) / "orchestrator.toml"
        config.write_text(f'[orchestrator]\npicks_dir = {json.dumps(str(picks))}\n'
                          '[scheduler]\nentry_intent="enter"\npick_delivery="public"\ncontest_state_required=true\n'
                          '[[tiers]]\nname="local"\ntype="local"\n')

        def canary():
            tick = datetime.fromtimestamp(clock.now, ET)
            result = invoke_canary(config, tick)
            events.append(("canary", tick.isoformat(), result.exit_code))
            if result.exit_code:
                raise guard.Refused("canary-contest-state")
            return tick.date().isoformat()

        def start(uid, authorized):
            events.append("start")
            clock.now = MIDNIGHT + 1 if label == "during-start" else authorized + (30 if label == "normal" else 0.1)
            if label == "same-day-state-disappears":
                state.unlink()

        publish = guard.publish_json

        def record(fd, name, data):
            answer = publish(fd, name, data)
            if name == "last-release" and label == "during-marker-fsync":
                clock.now = MIDNIGHT + 1
            return answer

        def unlink(name, *args, **kwargs):
            if name == "hold" and label == "midnight-before-unlink":
                clock.now = MIDNIGHT + 1
            return os.unlink(name, *args, **kwargs)

        guard.canary, guard.activate_scheduler, guard.publish_json, guard.os.unlink = canary, start, record, unlink
        with guard.directory() as fd:
            rc, output, _ = capture(lambda: guard.owned_deploy(fd, RUN, TARGET, ""))
            final = guard.read_hold(fd)
            marker = guard.read_json(fd, "last-release")
        print("F6/F8", label, "rc", rc, "hold", final and (final["state"], final["step"]), "marker", marker,
              "events", events, "activation failure text", " ".join(line for line in output.splitlines() if line.startswith("Activation failed")))


def incidents(place):
    for label in ("stale-target", "uid-failure", "page-claim", "late-hook"):
        guard = fixture(place / label)
        clock = SimpleNamespace(now=NOW.timestamp())
        events = txn(guard, clock)
        if label == "stale-target":
            def refuse(*args, **kwargs):
                events.append("target")
                raise guard.Refused("guard-stale", PREVIOUS)
            guard.verified_target = refuse
        elif label == "uid-failure":
            def bad_uid():
                raise guard.Refused("id")
            guard.bts_uid = bad_uid
        elif label == "late-hook":
            original_lock = guard.lock

            def contended(fd, name, operation, timeout=0):
                if name == "transaction.lock":
                    events.append(("wait", timeout))
                    raise BlockingIOError("stand-in contention")
                return original_lock(fd, name, operation, timeout)

            guard.lock = contended

        with guard.directory() as fd:
            guard.publish_json(fd, "hold", hold("failed" if label == "page-claim" else "activating"))
            if label == "page-claim":
                claim = "claimed" if REV == HEAD else "attempted"
                guard.publish_json(fd, guard.run_name(RUN, "evidence"), {"stop": "verified", "page": claim})
            if label == "late-hook":
                def release_then_uid():
                    # Stand-in interleaving: the current owner clears the old hold after the hook reads it.
                    assert guard.read_hold(fd)["deploy_id"] == RUN
                    os.unlink("hold", dir_fd=fd)
                    events.append("replacement-started")
                    return 1000
                guard.bts_uid = release_then_uid
            try:
                if label == "stale-target":
                    rc, _, _ = capture(lambda: guard.owned_deploy(fd, FOLLOWUP, TARGET, ""))
                else:
                    rc, _, _ = capture(lambda: guard.reconcile(fd, RUN))
            except (guard.Refused, BlockingIOError) as error:
                rc = type(error).__name__
            final = guard.read_hold(fd)
            result = guard.read_json(fd, guard.run_name(RUN, "result"))
            evidence = guard.read_json(fd, guard.run_name(RUN, "evidence"))
        print("F2/F11", label, rc, final and final["state"], events, result, evidence)

    guard = fixture(place / "busy-page-evidence")
    clock = SimpleNamespace(now=NOW.timestamp())
    events = txn(guard, clock)
    with guard.directory() as fd:
        guard.publish_json(fd, "hold", hold())
        with guard.lock(fd, "state.lock", fcntl.LOCK_SH):
            ticks = iter((0, 3, 4, 7, 8, 11, 12, 15))
            guard.time.monotonic = lambda: next(ticks)
            capture(lambda: guard.reconcile(fd, RUN))
            capture(lambda: guard.reconcile(fd, RUN))
        print("F11 simultaneous monitor read lock before next preflight", events,
              guard.read_json(fd, guard.run_name(RUN, "evidence")))


def hand_release(place):
    for label, state, age, wake in (
        ("old-sleep", "sleeping", 3600, -1800), ("old-running", "running", 301, None),
        ("current-running", "running", 30, None), ("current-waiting", "waiting_for_games", 60, None),
        ("wake-in-29s", "sleeping", 3, 29), ("wake-in-31s", "sleeping", 3, 31),
        ("fresh-stalled", "stalled", 10, None), ("missing", None, 0, None),
    ):
        case = place / label
        guard = fixture(case)
        clock = SimpleNamespace(now=NOW.timestamp())
        window = guard.window
        txn(guard, clock)
        guard.window = window
        guard.preflight = lambda *args, **kwargs: (TARGET, ["fixture-git"])
        if state:
            row = {"state": state, "timestamp": (NOW - timedelta(seconds=age)).isoformat()}
            if wake is not None:
                row["sleeping_until"] = (NOW + timedelta(seconds=wake)).isoformat()
            (case / "data/.heartbeat").write_text(json.dumps(row))
        with guard.directory() as fd:
            guard.publish_json(fd, "hold", hold())
            code, stdout, _ = capture(lambda: guard.release_by_hand(fd, "review fixture"))
            still_held = guard.read_hold(fd) is not None
        print("F9", label, code, "held", still_held, stdout.splitlines()[0])


def monitors(place):
    monitor = module("round_monitor", ROOT / MONITOR)
    legacy = module("legacy_monitor", HERE / "trees" / LEGACY / MONITOR)
    guard = fixture(place)
    heartbeat = place / "data/.heartbeat"
    hold_path = guard.ROOT / "hold"
    churn = place / "churn.json"
    now = NOW.astimezone(timezone.utc)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    def run(tool, path=None, launch=fail_if_called):
        pings = []
        if churn.exists():
            churn.unlink()
        argv = ["check_heartbeat.py", "--heartbeat-path", str(heartbeat), "--ping-url", "https://fixture.invalid/ping",
                "--churn-state", str(churn)]
        if path:
            argv += ["--deploy-hold", str(path)]
        with patch.object(sys, "argv", argv), patch.object(tool, "datetime", Clock), \
             patch.object(tool, "ping", side_effect=lambda url, suffix="": pings.append(("GET", suffix))), \
             patch.object(tool, "ping_deploy_failure", create=True, side_effect=lambda url, reason: pings.append(("POST", "/fail", reason))), \
             patch.object(tool, "read_nrestarts", return_value=0), \
             patch.object(tool, "subprocess", SimpleNamespace(run=launch, TimeoutExpired=subprocess.TimeoutExpired)):
            output = capture(tool.main)
        return (*output, pings, churn.read_bytes() if churn.exists() else None)

    for label, row in {
        "fresh-sleep": {"state": "sleeping", "timestamp": (now - timedelta(seconds=60)).isoformat(),
                        "sleeping_until": (now + timedelta(hours=4)).isoformat()},
        "fresh-run": {"state": "running", "timestamp": (now - timedelta(seconds=60)).isoformat()},
        "stale-run": {"state": "running", "timestamp": (now - timedelta(seconds=600)).isoformat()},
        "missing": None,
    }.items():
        if row is None:
            heartbeat.unlink(missing_ok=True)
        else:
            heartbeat.write_text(json.dumps(row))
        for layout, path in (("present-empty-guard", hold_path), ("absent-guard", place / "absent/hold")):
            current = run(monitor, path)
            original = run(legacy)
            print("F10", label, layout, "identical", current == original, current[:4])

    for label, heartbeat_delta, stamp_delta, metadata in (
        ("first-sleep-heartbeat", -600, -601 if REV == HEAD else -571, ""), ("later-stale-heartbeat", -600, -900, ""),
        ("equal-time", -600, -600, ""), ("later-large-heartbeat", -60, -600, "x" * 8200),
    ):
        hb_state = "sleeping" if label == "first-sleep-heartbeat" else "running"
        row = {"state": hb_state, "timestamp": (now + timedelta(seconds=heartbeat_delta)).isoformat()}
        if hb_state == "sleeping":
            row["sleeping_until"] = (now + timedelta(hours=4)).isoformat()
        if metadata:
            row["diagnostics"] = metadata
        heartbeat.write_text(json.dumps(row))
        (guard.ROOT / "last-release").write_text(json.dumps({"kind": "deploy", "released_at": int(now.timestamp()) + stamp_delta}))
        current = run(monitor, hold_path)
        print("F8", label, "identical", current == run(legacy), current[:4])

    (guard.ROOT / "last-release").unlink()
    with guard.directory() as fd:
        guard.publish_json(fd, "hold", hold("pending", step="arm"))
    # Round 1's production status call is a stand-in error on the custom path.
    def unavailable_status(*args, **kwargs):
        return SimpleNamespace(returncode=1, stdout=b'{"state":"unreadable"}')

    launch = unavailable_status if REV != HEAD else fail_if_called
    print("F13 custom record", run(monitor, hold_path, launch)[:4])

    hold_path.unlink()
    (guard.ROOT / "state.lock").unlink()
    print("F10 damaged guard", run(monitor, hold_path)[:4])


def dashboard_address(place):
    guard = fixture(place)
    fake_proc = SimpleNamespace(read_text=lambda: "0::/fixture/bts-dashboard.service\n")
    guard.Path = lambda path: fake_proc if path == "/proc/456/cgroup" else Path(path)
    guard.os.readlink = lambda path: guard.PROJECT if path == "/proc/456/cwd" else fail_if_called(path)
    calls = []

    def call(argv, *, limit=10):
        if "/usr/bin/tailscale" in argv:
            return 1, b""
        if "ActiveState" in argv:
            return 0, b"active\n"
        if "ControlGroup" in argv:
            return 0, b"/fixture/bts-dashboard.service\n"
        if argv[0] == "/usr/bin/ss":
            return 0, b'LISTEN 0 5 127.0.0.1:3003 0.0.0.0:* users:(("python3",pid=456,fd=3))\n'
        if argv[0] == "/usr/bin/curl":
            calls.append(argv[-1])
            return 0, b"200"
        raise AssertionError(argv)

    guard.call = call
    try:
        guard.dashboard(1000)
    except guard.Refused as exc:
        print("F15", exc.step, calls)
    else:
        print("F15 accepted loopback", calls)


if __name__ == "__main__":
    assert os.geteuid() != 0 and Path(bts.__file__).is_relative_to(ROOT / "src")
    suites = (cron_and_units, liveness, canaries, dashboard_case, lifecycle, incidents,
              hand_release, monitors, dashboard_address)
    errors = False
    with tempfile.TemporaryDirectory(prefix=f"compare-{REV}-", dir=HERE) as temp:
        for suite in suites:
            folder = Path(temp) / suite.__name__
            folder.mkdir()
            try:
                suite(folder)
            except Exception:
                errors = True
                traceback.print_exc()
    raise SystemExit(int(errors))
