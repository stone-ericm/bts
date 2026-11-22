"""D3 deploy workflow contract. The YAML block is the executable artifact."""

import fcntl
from concurrent.futures import ThreadPoolExecutor
import difflib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time

import pytest
import yaml


WORKFLOW = Path(__file__).resolve().parents[1] / ".github/workflows/deploy.yml"
ACTION = "appleboy/ssh-action@0ff4204d59e8e51228ff73bce53f80d53301dee2"
PREDICATE = "${{ always() && (steps.deploy.outcome == 'failure' || steps.deploy.outcome == 'cancelled') }}"
OLD = "a" * 40
TARGET = "b" * 40
COMMANDS = {
    "bash", "bts", "cat", "curl", "cut", "date", "env", "flock", "git", "grep", "head",
    "id", "install", "mv", "pgrep", "pkill", "python3", "rm", "sleep", "stat", "sudo",
    "systemctl", "tailscale", "timeout", "uv",
}
DIRECT_PATH = re.compile(r"(?<![\w.$/])/(?:[\w.-]+/)*[\w.-]+")


def scripts_and_steps(source=None):
    source = WORKFLOW.read_text() if source is None else source
    steps = [step for step in yaml.safe_load(source)["jobs"]["deploy"]["steps"]
             if "script" in step.get("with", {})]
    literals = re.findall(r"^          script: \|\n((?:^ {12}[^\n]*\n|^\n)*)", source, re.MULTILINE)
    scripts = [re.sub(r"^ {12}", "", literal, flags=re.MULTILINE).rstrip("\n") + "\n"
               for literal in literals]
    assert scripts == [step["with"]["script"] for step in steps]
    return scripts, steps


def test_deploy_and_recovery_shape():
    scripts, steps = scripts_and_steps()
    assert len(scripts) == 2
    deploy, recovery = steps
    assert deploy["id"] == "deploy"
    assert recovery["if"] == PREDICATE
    for step, deadline in zip(steps, ("20m", "35m"), strict=True):
        assert step["uses"] == ACTION
        assert step["with"]["command_timeout"] == deadline
        assert "request_pty" not in step["with"]
        assert set(step["with"]["envs"].split(",")) == {"TARGET_SHA", "RUN_ID", "RUN_ATTEMPT"}
        assert step["env"] == {
            "TARGET_SHA": "${{ github.sha }}",
            "RUN_ID": "${{ github.run_id }}",
            "RUN_ATTEMPT": "${{ github.run_attempt }}",
        }
    for key in ("host", "username", "key"):
        assert deploy["with"][key] == recovery["with"][key]


SHIM = r'''
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import sys
import time

command = Path(sys.argv[0]).name
args = sys.argv[1:]
root = Path(os.environ["BOX"])
scenario = os.environ["SCENARIO"]
with (root / "calls").open("a") as log:
    print(json.dumps([command, *args]), file=log)

def phase():
    if command == "systemctl" and args:
        user = args[0] == "--user"
        operation, *rest = args[1:] if user else args
        if user and operation in {"disable", "enable", "restart"}:
            return operation
        if operation == "is-enabled" and user:
            return "check-enable" if (root / "enabled").read_text() == "enabled" else "check-disable"
        if operation == "show" and user and "ExecMainStatus" in rest:
            return "scheduler-state"
        if operation == "show" and user and "LoadState" in rest:
            return "user-units"
        if operation == "show" and not user and "LoadState" in rest:
            return "legacy-units"
        if operation == "show" and user and "bts-dashboard" in rest and "ActiveState" in rest:
            return "dashboard-state"
        if not user and operation in {"disable", "stop", "daemon-reload"}:
            return "legacy-" + operation
    if command == "git" and args:
        if args[0] == "rev-parse" and (root / "head").read_text().startswith("b"):
            return "head"
        return args[0]
    if command == "id":
        return "uid"
    if command in {"uv", "rm", "pkill", "sleep", "bts", "curl"}:
        return {"uv": "sync", "rm": "legacy-rm", "pkill": "legacy-pkill",
                "sleep": "wait", "bts": "contest-state",
                "curl": "post" if "--data-raw" in args else "dashboard-http"}[command]
    if command == "python3" and len(args) > 2:
        return {"stage": "stamp-stage", "stamp": "stamp"}.get(args[2])
    if command == "mv":
        return "publish-" + Path(args[-2]).read_text().split()[5]
    if command == "bash" and any("CANARY PASSED" in arg for arg in args[3:]):
        return "log-pass"
    return None

at = phase()
if command == "python3" and len(args) > 3 and args[2] == "stage-record":
    at = "stage-" + args[-1].split()[5]
fault = os.environ.get("FAULT", "").split(":")
if at and fault[0] == at and not (root / "fault-fired").exists():
    (root / "fault-fired").touch()
    if fault[1] in {"17", "124", "137"}:
        raise SystemExit(int(fault[1]))
    if fault[1] == "mismatch":
        print("c" * 40)
        raise SystemExit(0)
    if fault[1] == "hold":
        end = time.monotonic() + 10
        while not (root / "release").exists() and time.monotonic() < end:
            time.sleep(0.01)
    else:
        if len(fault) > 2 and fault[2].startswith("forge-"):
            # D3b K17: a valid record appears in place of this invocation's write.
            _, field, state = fault[2].split("-")
            values = dict(run="123", attempt=os.environ["RUN_ATTEMPT"], target="b" * 40, old="a" * 40)
            values[field] = {"run": "999", "attempt": "9", "target": "c" * 40, "old": "d" * 40}.get(field, values.get(field))
            (root / "run/bts-deploy/attempt").write_text(
                f"bts_deploy_attempt_v1 {values['run']} {values['attempt']} {values['target']} {values['old']} {state} - - - -\n")
        os.kill(int((root / "owner.pid").read_text()), getattr(signal, "SIG" + fault[1]))
        if len(fault) > 2 and (fault[2] == "unpublished" or fault[2].startswith("forge-")):
            raise SystemExit(17)

def dispatch(words):
    if words[0].startswith("/") and words[0] != "/bin/bash" and not Path(words[0]).is_relative_to(root):
        raise SystemExit(99)
    program = Path(words[0]).name
    shim = root / "shims" / program
    if not shim.is_file():
        raise SystemExit(98)
    os.execv(shim, [str(shim), *words[1:]])

if command == "sudo":
    args = args[args.index("bts") + 1:]
    while args and "=" in args[0] and not args[0].startswith("/"):
        key, value = args.pop(0).split("=", 1)
        os.environ[key] = value
    dispatch(args)
elif command == "env":
    while args and "=" in args[0] and not args[0].startswith("/"):
        key, value = args.pop(0).split("=", 1)
        os.environ[key] = value
    dispatch(args)
elif command == "timeout":
    dispatch(args[3:])
elif command == "install":
    Path(args[-1]).mkdir(mode=0o700, parents=True, exist_ok=True)
elif command == "stat":
    p = Path(args[-1])
    owner = 1000 if scenario == "bad-owner" else 0
    print(f"{owner} {p.stat().st_mode & 0o777:o}")
elif command == "flock":
    fd = int(args[-1])
    # Keep the real advisory lock and compress the 1500s timeout for tests.
    wait = min(float(args[1]), 1) if args[0] == "-w" else 0
    deadline = time.monotonic() + wait
    while True:
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            break
        except BlockingIOError:
            if time.monotonic() >= deadline:
                raise SystemExit(1)
            (root / "waiting-for-lock").touch()
            time.sleep(0.01)
elif command == "python3":
    op = args[2] if len(args) > 2 else None
    code = args[1]
    sys.argv = ["-c", *args[2:]]
    if op in {"lockfile", "mark"}:
        # Emulate only the root UID, leaving creation and verification to fs_code.
        original_fstat = os.fstat

        def root_fstat(fd):
            metadata = original_fstat(fd)
            if metadata.st_uid != os.geteuid():
                return metadata
            fields = list(metadata)
            fields[4] = 0
            return os.stat_result(fields)

        os.fstat = root_fstat
        if op == "mark" and scenario == "mark-error":
            def denied_open(*args, **kwargs):
                raise PermissionError("simulated cancellation mark creation failure")
            os.open = denied_open
    if op == "clock" and (root / "fail-containment-clock").exists():
        if not scenario.endswith("-every"):
            (root / "fail-containment-clock").unlink()
        (root / "clock-failed").touch()
        raise SystemExit(124)
    if op == "read" and scenario.startswith("recovery-clock-") and not (root / "recovery-armed").exists():
        (root / "recovery-armed").touch()
        (root / "fail-containment-clock").touch()
    if op == "clock" and scenario == "late-signal":
        if (root / "exhausted-phase").exists():
            print(time.monotonic_ns() // 1_000_000 + 805_000)
            raise SystemExit(0)
        if (root / "next-clock").exists():
            print(time.monotonic_ns() // 1_000_000 + 790_000)
            raise SystemExit(0)
    if op == "clock" and (root / "clock-at-795").exists():
        print(time.monotonic_ns() // 1_000_000 + 795_000)
        raise SystemExit(0)
    if op == "read" and scenario == "beginning-read-126" and (root / "fault-fired").exists():
        if not (root / "read-failed").exists():
            (root / "read-failed").touch()
            raise SystemExit(126)
    if op == "read" and scenario in {"commit-read-124", "commit-read-137", "commit-read-126"}:
        record = root / "run/bts-deploy/attempt"
        if record.exists() and " committed " in record.read_text():
            raise SystemExit(int(scenario.rsplit("-", 1)[1]))
    if op == "read" and scenario == "commit-read-126-contain-fail" and not (root / "commit-read-failed").exists():
        record = root / "run/bts-deploy/attempt"
        if record.exists() and " committed " in record.read_text():
            (root / "commit-read-failed").touch()
            raise SystemExit(126)
    if op == "read" and scenario == "record-read-126" and (root / "exhausted-phase").exists() and not (root / "stopped").exists():
        if not (root / "read-failed").exists():
            (root / "read-failed").touch()
            raise SystemExit(126)
    if op == "clock" and scenario == "canary-budget" and (root / "advance-clock").exists():
        print(time.monotonic_ns() // 1_000_000 + 1_000_000)
        raise SystemExit(0)
    if op == "marker" and scenario == "marker-unwritable":
        raise SystemExit(17)
    if op == "mark" and scenario == "legacy-unloads-markfail":
        raise SystemExit(1)
    if op == "stamp" and scenario == "stamp-directory-sync":
        def fail_fsync(fd):
            raise OSError("simulated directory sync failure")
        os.fsync = fail_fsync
    if op == "stage" and scenario == "partial-stamp":
        import tempfile
        original = tempfile.NamedTemporaryFile
        def partially_written(*args, **kwargs):
            file = original(*args, **kwargs)
            write = file.write
            def partial_write(data):
                write(data[:2])
                raise OSError("simulated short stage write")
            file.write = partial_write
            return file
        tempfile.NamedTemporaryFile = partially_written
    exec(compile(code, "<workflow helper>", "exec"), {"__name__": "__main__"})
    if op == "stamp" and scenario == "signal-after-stamp":
        (root / "clock-at-795").touch()
        os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
elif command == "mv":
    if scenario == "commit-error" and at == "publish-committed":
        raise SystemExit(17)
    if scenario in {"commit-read-126-contain-fail", "contain-rename-fails"} and at == "publish-contained":
        raise SystemExit(17)
    os.replace(args[-2], args[-1])
    if scenario == "published-with-error" and at == "publish-begun":
        raise SystemExit(17)
    if scenario == "committed-with-error" and at == "publish-committed":
        raise SystemExit(17)
elif command == "git":
    branch = root / "head"
    if at == "head" and scenario.startswith("deploy-clock-"):
        (root / "fail-containment-clock").touch()
        raise SystemExit(17)
    if args[0] == "fetch" and scenario == "fetch":
        raise SystemExit(17)
    if args[0] == "rev-parse":
        print(branch.read_text()[:7] if "--short" in args else branch.read_text())
    elif args[0] == "reset":
        if scenario == "closed-mutator-fd":
            try:
                os.fstat(9)
            except OSError:
                (root / "mutator-closed-fd").touch()
            else:
                raise SystemExit(17)
        branch.write_text(args[-1])
    elif args[0] == "cat-file":
        print("commit")
elif command == "id":
    print(1000)
elif command == "systemctl":
    user = args[0] == "--user"
    if user:
        args.pop(0)
    if user and args[0] in {"disable", "enable"}:
        (root / "enabled").write_text("disabled" if args[0] == "disable" else "enabled")
    if user and args[0] == "restart":
        (root / "stopped").unlink(missing_ok=True)
        (root / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    if args[0] == "is-enabled" and not user and scenario == "legacy-still-enabled":
        print("enabled")
        raise SystemExit(0)
    if args[0] == "is-enabled":
        enabled = root / "enabled"
        state = enabled.read_text() if enabled.exists() else "enabled"
        print(state)
        if scenario == "extra-disabled-line" and state == "disabled":
            print()
        raise SystemExit(state == "disabled")
    if args[0] == "is-active" and "--quiet" not in args:
        print("activating" if os.environ["SCENARIO"] == "auto-restart" else "active")
    if args[0] == "show":
        if user and "ExecMainStatus" in args and scenario in {"late-signal", "record-read-126"}:
            (root / "exhausted-phase").touch()
            os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
        active = "active" if user and not (root / "stopped").exists() else "inactive"
        states = dict(Id=args[1], LoadState="loaded" if user or scenario in {"legacy-present", "legacy-survivor", "legacy-blank-cgroup"} else "not-found", UnitFileState="enabled",
                      ActiveState=active, SubState="running", Result="success", ExecMainStatus="0",
                      MainPID="0", ControlGroup="/user-sched" if user else "")
        if os.environ["SCENARIO"] == "auto-restart" and "bts-scheduler" in args:
            states.update(ActiveState="activating", SubState="auto-restart")
        if "ControlGroup" in args and user and scenario == "descendant-cgroup":
            states["ControlGroup"] = "/sched"
        if "ControlGroup" in args and user and scenario == "blank-cgroup":
            states["ControlGroup"] = "" if (root / "stopped").exists() else "/sched"
        if "ControlGroup" in args and user and scenario == "unknown-cgroup":
            states["ControlGroup"] = ""
        if "ControlGroup" in args and not user and scenario == "legacy-blank-cgroup" and not (root / "legacy-stopped").exists():
            states["ControlGroup"] = "/sched"
        if not user and args[1] == "bts-scheduler.service" and scenario.startswith("legacy-vanishing"):
            # D3b K11/K12: loaded at containment, reported not-found after its stop.
            if not (root / "legacy-stopped").exists():
                states.update(LoadState="loaded", ActiveState="active", MainPID="1234",
                              ControlGroup="" if scenario.endswith("nogroup") else "/legacy-orphan")
        if not user and args[1] == "bts-scheduler.service" and scenario.startswith("legacy-unloads"):
            # Carbon B2: loaded until phase 2's stop, rm and daemon-reload; then not-found and blank.
            if not (root / "legacy-stopped").exists():
                states.update(LoadState="loaded", ActiveState="active", MainPID="1234", ControlGroup="/legacy-orphan")
        if not user and args[1] == "bts-scheduler.service" and scenario == "legacy-groupfail-post":
            # Carbon N1 own1: the pre-stop group query fails; after the stop a path is reported.
            if (root / "legacy-stopped").exists():
                states["ControlGroup"] = "/legacy-post"
            else:
                states.update(LoadState="loaded", ActiveState="active", MainPID="1234")
                if args[2:] == ["-p", "ControlGroup"]:
                    raise SystemExit(1)
        if not user and args[1] == "bts-scheduler.service" and scenario == "legacy-late-group":
            # Carbon N1 own2: absent and blank at the first sample, a populated path at the proof.
            if args[2:] == ["-p", "ControlGroup"]:
                if (root / "legacy-group-seen").exists():
                    states["ControlGroup"] = "/legacy-late"
                (root / "legacy-group-seen").touch()
        if not user and args[1] == "bts-scheduler.service" and scenario == "legacy-still-enabled":
            # Carbon N1 own4: loaded and stopped, but its boot link stays enabled.
            states.update(LoadState="loaded", ControlGroup="/la")
        if not user and args[1] == "bts-scheduler.service" and scenario == "legacy-ghost-group":
            # D3b review round 1: not-found, inactive, PID 0, no jobs, yet a cgroup is reported.
            states["ControlGroup"] = "/legacy-group"
        if not user and args[1] == "bts-scheduler.service" and scenario == "legacy-fragment-present":
            # D3b review round 1: loaded, then not-found after its stop while the fragment file stays.
            if (root / "legacy-stopped").exists():
                states["ControlGroup"] = "/legacy-post"
            else:
                states.update(LoadState="loaded", ControlGroup="/legacy-pre")
        if "ControlGroup" in args and user and scenario.startswith("moved-user"):
            states["ControlGroup"] = "/b" if (root / "stopped").exists() else "/a"
        if not user and args[1] == "bts-scheduler.service" and scenario.startswith("moved-legacy"):
            states.update(LoadState="loaded", ControlGroup="/lb" if (root / "legacy-stopped").exists() else "/la")
        if (not user and args[1] == "bts-scheduler.service" and scenario == "legacy-vanishing-queryfail"
                and (root / "legacy-stopped").exists()):
            raise SystemExit(1)
        if user and scenario == "user-group-fail" and args[1:] == ["bts-scheduler", "-p", "ControlGroup"]:
            raise SystemExit(1)
        if "ControlGroup" in args and user and scenario == "main-pid":
            states["MainPID"] = "4321"
        if user and "UnitFileState" in args and scenario in {"masked", "linked", "static"}:
            states["UnitFileState"] = scenario
        if user and "LoadState" in args and scenario == "unloaded":
            states["LoadState"] = "not-found"
        if not user and scenario == "legacy-survivor":
            states.update(ActiveState="active", MainPID="4321")
        if user and scenario == "dashboard-inactive" and "bts-dashboard" in args:
            states["ActiveState"] = "inactive"
        if user and scenario == "crash-loop" and "ExecMainStatus" in args:
            states.update(ActiveState="activating", SubState="auto-restart", Result="exit-code", ExecMainStatus="1")
        if user and scenario == "malformed-query" and "ExecMainStatus" in args:
            states["ActiveState"] = "bad value"
        for key in (arg for arg in args if arg in states):
            print(states[key] if "--value" in args else f"{key}={states[key]}")
        if user and scenario == "duplicate-query" and "ExecMainStatus" in args:
            print("ActiveState=active")
        if user and scenario == "canary-budget" and "ExecMainStatus" in args:
            (root / "advance-clock").touch()
    if user and args[0] == "stop":
        if scenario in {"stop-error", "all-errors"}:
            raise SystemExit(17)
        if scenario == "second-signal":
            owner = int((root / "owner.pid").read_text())
            os.kill(owner, signal.SIGINT)
            os.kill(owner, signal.SIGPIPE)
        (root / "stopped").touch()
    if not user and args[0] == "stop":
        (root / "legacy-stopped").touch()
        if scenario == "legacy-vanishing-stopfail":
            raise SystemExit(17)
    if user and args[0] == "list-jobs" and scenario == "pending-job":
        print("17 bts-scheduler.service start waiting")
    if not user and args[0] == "list-jobs" and scenario == "legacy-vanishing-job" and (root / "legacy-stopped").exists():
        print("18 bts-scheduler.service stop waiting")
elif command == "tailscale":
    print("100.64.0.1")
    if scenario == "tailnet-error":
        raise SystemExit(17)
elif command == "curl":
    if "--data-raw" in args and scenario.startswith("post-"):
        status, code = scenario.removeprefix("post-").split("-")
        print(code, end="")
        raise SystemExit(int(status))
    print("503" if scenario == "http" else "200", end="")
    if scenario == "http-transport" and "--data-raw" not in args:
        raise SystemExit(28)
elif command == "head":
    print(sys.stdin.readline(), end="")
elif command == "grep":
    raise SystemExit(0 if re.search(args[-1], sys.stdin.read()) else 1)
elif command == "cut":
    print(sys.stdin.read()[:7])
elif command == "bash" and args[0] == "-c" and args[1].startswith("date -u "):
    (root / "home/projects/bts/data/.last_deploy_iso").write_text("2026-11-21T13:00:00Z\n")
elif command == "bash" and args[0] == "-c" and args[1].startswith("printf "):
    print(*args[3:], sep="\n")
elif command == "bash" and args[0] == "-c" and args[1].startswith("exec 9>&-"):
    os.close(9)
    dispatch(args[3:])
elif command == "bts":
    if scenario == "contest-nul":
        sys.stdout.buffer.write(b"Deploy canary: not a live configu\x00ration (delivery private); contest state not checked.\n")
        raise SystemExit(0)
    if scenario == "contest-refused":
        print("Deploy canary: REFUSED — invalid state")
        raise SystemExit(1)
    elif scenario == "contest-multiline":
        print("Deploy canary: live contest state accepted (")
        print("UNEXPECTED OUTPUT")
    elif scenario == "contest-error":
        print("Deploy canary: ERROR — ValueError")
        raise SystemExit(2)
    else:
        print("Deploy canary: not a live configuration (delivery private); contest state not checked.")
elif command in {"pgrep", "pkill"}:
    if command == "pgrep" and scenario == "residual":
        raise SystemExit(0)
    raise SystemExit(1)
elif command == "rm":
    for operand in (arg for arg in args if not arg.startswith("-")):
        Path(operand).unlink(missing_ok=True)
elif command in {"sleep", "uv"}:
    if command == "sleep" and scenario == "late-signal":
        (root / "next-clock").touch()
    if command == "sleep" and scenario == "signal-during-wait":
        (root / "clock-at-795").touch()
        os.kill(int((root / "owner.pid").read_text()), signal.SIGHUP)
    if command == "uv" and os.environ["SCENARIO"] == "sync":
        raise SystemExit(17)
else:
    raise SystemExit(97)
'''


@pytest.fixture
def box(tmp_path):
    shim_dir = tmp_path / "shims"
    shim_dir.mkdir()
    for name in COMMANDS:
        shim = shim_dir / name
        shim.write_text(f"#!{sys.executable}\n" + SHIM)
        shim.chmod(0o700)
    (tmp_path / "home/projects/bts/data").mkdir(parents=True)
    (tmp_path / "sys/fs/cgroup").mkdir(parents=True)
    (tmp_path / "sys/fs/cgroup/cgroup.controllers").touch()
    (tmp_path / "head").write_text(OLD)
    return tmp_path


def run_box_script(box, script, scenario, *, fault="", run_id="123", attempt="1", target=TARGET, closed_output=False,
                   limit=15):
    # Documented injection points: checkout and its bts home, deployed state,
    # systemd fragments, system Python, the null sink, and the uv executable.
    # The frame uses the host /bin/bash explicitly. All PATH lookups use shims.
    for absolute, path in {
        "/home/bts/.local/bin/uv": "shims/uv",
        "/home/bts/projects/bts/.venv/bin/bts": "shims/bts",
        "/home/bts": "home",
        "/run/bts-deploy": "run/bts-deploy",
        "/run/user": "run/user",
        "/sys/fs/cgroup": "sys/fs/cgroup",
        "/etc/systemd/system": "etc/systemd/system",
        "/usr/bin/python3": "shims/python3",
        "/tmp/uv-cache": "uv-cache",
        "/dev/null": "null",
    }.items():
        script = script.replace(absolute, str(box / path))
    assert_direct_paths(box, script)
    argv = [
        "/usr/bin/env", "-i", f"PATH={box / 'shims'}", f"BOX={box}", f"FAULT={fault}",
        f"SCENARIO={scenario}", f"TARGET_SHA={target}", f"RUN_ID={run_id}",
        f"RUN_ATTEMPT={attempt}", "LC_ALL=C", "/bin/bash", "-c", script,
    ]
    # The shared wrapper starts pytest as a background job, which can inherit
    # ignored SIGINT. Give the test's noninteractive SSH shell default signals.
    launcher = (
        "import os, signal, sys\n"
        "for sig in (signal.SIGHUP, signal.SIGINT, signal.SIGTERM, signal.SIGPIPE):\n"
        "    signal.signal(sig, signal.SIG_DFL)\n"
        "os.execv(sys.argv[1], sys.argv[1:])\n"
    )
    writer = subprocess.PIPE
    if closed_output:
        reader, writer = os.pipe()
        os.close(reader)
    try:
        process = subprocess.Popen(
            [sys.executable, "-c", launcher, *argv],
            cwd=box, stdout=writer, stderr=subprocess.PIPE, text=True,
        )
    finally:
        if closed_output:
            os.close(writer)
    with process:
        (box / "owner.pid").write_text(str(process.pid))
        try:
            stdout, stderr = process.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            process.kill()
            process.communicate(timeout=2)
            raise
        result = subprocess.CompletedProcess(argv[:2], process.returncode, stdout or "", stderr)
    log = box / "calls"
    calls = [json.loads(line) for line in log.read_text().splitlines()] if log.exists() else []
    return result, calls


def assert_direct_paths(box, script):
    for match in DIRECT_PATH.finditer(script):
        path = Path(match.group())
        if path == Path("/fail") and script[match.start() - 4 : match.end() + 2] == ' + "/fail")':
            continue
        assert path == Path("/bin/bash") or path.is_relative_to(box), path


@pytest.mark.parametrize("recovery", [False, True])
@pytest.mark.parametrize("frequency", ["once", "every"])
def test_round1_clock_failure_does_not_skip_containment(box, recovery, frequency):
    scripts, _ = scripts_and_steps()
    data = box / "home/projects/bts/data"
    (data.parent / ".env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/a\n")
    if recovery:
        root = box / "run/bts-deploy"
        root.mkdir(mode=0o700, parents=True)
        (root / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {TARGET} {OLD} begun - - - -\n")
    scenario = f"{'recovery' if recovery else 'deploy'}-clock-{frequency}"
    result, calls = run_box_script(box, scripts[int(recovery)], scenario)
    assert result.returncode == 1
    assert (box / "clock-failed").exists()
    assert ["systemctl", "--user", "stop", "bts-scheduler"] in calls
    assert any(call[0] == "curl" and "--data-raw" in call for call in calls)
    assert json.loads((data / ".heartbeat").read_text())["state"] == "deploy_failed"
    assert " contained " in (box / "run/bts-deploy/attempt").read_text()


@pytest.mark.parametrize("scenario", ["late-signal", "record-read-126"])
def test_round1_late_signal_uses_confirmed_begin(box, scenario):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[0], scenario)
    assert result.returncode == 1
    assert "DEPLOY FAILED at interrupted" in result.stdout
    assert "DEPLOY NOT STARTED" not in result.stdout
    assert ["systemctl", "--user", "stop", "bts-scheduler"] in calls
    assert not (box / "read-failed").exists()


@pytest.mark.parametrize(("scenario", "fault"), [("blank-cgroup", "head:17"), ("legacy-blank-cgroup", "sync:17")])
def test_round1_blank_control_group_retains_populated_subtree(box, scenario, fault):
    scripts, _ = scripts_and_steps()
    group = box / "sys/fs/cgroup/sched"
    group.mkdir()
    (group / "cgroup.events").write_text("populated 1\n")
    result, calls = run_box_script(box, scripts[0], scenario, fault=fault)
    assert "Scheduler stopped: unverified." in result.stdout
    assert any(call[0] == "python3" and call[-1] == "/sched" for call in calls)


def test_round1_unknown_control_group_cannot_verify_stop(box):
    scripts, _ = scripts_and_steps()
    group = box / "sys/fs/cgroup/sched"
    group.mkdir()
    (group / "cgroup.events").write_text("populated 1\n")
    result, _ = run_box_script(box, scripts[0], "unknown-cgroup", fault="head:17")
    assert "Scheduler stopped: unverified." in result.stdout


@pytest.mark.parametrize("endpoint", ["{other}", "{one,two}", "[1-2]"])
def test_round1_post_rejects_curl_globs(box, endpoint):
    scripts, _ = scripts_and_steps()
    (box / "home/projects/bts/.env").write_text(f"BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/{endpoint}\n")
    result, calls = run_box_script(box, scripts[0], "active", fault="head:17")
    assert not any(call[0] == "curl" and "--data-raw" in call for call in calls)
    assert "Failure ping received HTTP 2xx: unverified." in result.stdout


def test_round1_curl_disables_globs(box):
    scripts, _ = scripts_and_steps()
    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/a\n")
    result, calls = run_box_script(box, scripts[0], "http")
    assert result.returncode == 1
    curls = [call for call in calls if call[0] == "curl"]
    assert len(curls) == 2
    assert all("--globoff" in call for call in curls)


@pytest.mark.parametrize("command", ["stage-committed", "publish-committed"])
@pytest.mark.parametrize(("status", "logged"), [(17, "17"), (124, "timeout")])
def test_round1_commit_status_survives_failed_publication(box, command, status, logged):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], "active", fault=f"{command}:{status}")
    assert (box / "fault-fired").exists()
    assert f"Check commit-record failed ({logged})." in result.stdout
    assert (box / "home/projects/bts/data/.last_deploy_iso").is_file()


def test_round1_commit_readback_overrides_failed_rename_status(box):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], "committed-with-error")
    assert result.returncode == 0
    assert "CANARY PASSED" in result.stdout
    assert " committed " in (box / "run/bts-deploy/attempt").read_text()


def test_round1_activation_edits_follow_the_boot_hold():
    checklist = (WORKFLOW.parents[2] / "docs/ops/2027-season-start.md").read_text().splitlines()
    a5 = next(line for line in checklist if line.startswith("| A5 |"))
    b8a = next(line for line in checklist if line.startswith("| B8a |"))
    assert a5.split("|")[2].strip() == "Follow B8a's activation order; do not edit delivery, intent or cron before B8a says so."
    stop = b8a.index("confirm the scheduler is stopped and its local boot link disabled")
    for command in ('pick_delivery = "dm"', 'entry_intent = "enter"', "cron-setup-hetzner.sh"):
        assert stop < b8a.index(command)


def test_round1_incident_shortcuts_point_to_checked_deploy():
    incident = (WORKFLOW.parents[2] / "INCIDENT.md").read_text()
    sections = incident.split("- **Dashboard down, picks fine:**", 1)[1].split("## 5.", 1)[0]
    dashboard, restore = sections.split("- **Box/filesystem loss", 1)
    for advice in (dashboard, restore):
        assert "§4" in advice and "§5" in advice and "checked deploy" in advice


def test_round1_canary_rejects_a_nul_in_the_prefix(box):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], "contest-nul")
    assert result.returncode == 1
    assert "Check contest-state failed (query)." in result.stdout
    assert "CANARY PASSED" not in result.stdout


@pytest.mark.parametrize("scenario", ["signal-during-wait", "signal-after-stamp"])
def test_round2_signal_at_795_seconds_contains(box, scenario):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[0], scenario)
    assert (box / "clock-at-795").exists()
    assert result.returncode == 1
    assert "DEPLOY FAILED at interrupted" in result.stdout
    assert "DEPLOY NOT STARTED" not in result.stdout
    assert (box / "head").read_text() == TARGET
    assert (box / "enabled").read_text() == "disabled"
    assert ["systemctl", "--user", "stop", "bts-scheduler"] in calls
    assert (box / "home/projects/bts/data/.last_deploy_iso").exists() == (scenario == "signal-after-stamp")


def test_round2_uncertain_commit_signal_has_a_fresh_read_budget(box):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[0], "active", fault="publish-committed:TERM")
    assert result.returncode == 0
    assert " committed " in (box / "run/bts-deploy/attempt").read_text()
    reads = [call for call in calls if call[0] == "timeout" and call[-2] == "read"]
    assert reads[-1][:3] == ["timeout", "-k", "5"]
    assert 9 <= int(reads[-1][3]) <= 10


def test_round2_endpoint_rejects_carbon_bracket_range(box):
    scripts, _ = scripts_and_steps()
    (box / "home/projects/bts/.env").write_text(
        "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.invalid/[1-1]/uuid\n"
    )
    result, calls = run_box_script(box, scripts[0], "active", fault="head:17")
    assert "Failure ping received HTTP 2xx: unverified." in result.stdout
    assert not any(call[0] == "curl" for call in calls)


@pytest.mark.parametrize("timeout", [124, 137])
@pytest.mark.parametrize("operation", ["stage-committed", "publish-committed", "read"])
def test_round2_all_commit_timeouts_keep_timeout_status(box, operation, timeout):
    scripts, _ = scripts_and_steps()
    scenario = f"commit-read-{timeout}" if operation == "read" else "active"
    fault = "" if operation == "read" else f"{operation}:{timeout}"
    result, _ = run_box_script(box, scripts[0], scenario, fault=fault)
    assert result.returncode == 1
    assert "Check commit-record failed (timeout)." in result.stdout
    assert " contained commit-record " in (box / "run/bts-deploy/attempt").read_text()


def test_round2_different_run_with_same_attempt_cannot_recover(box):
    scripts, _ = scripts_and_steps()
    lock_dir = box / "run/bts-deploy"
    lock_dir.mkdir(mode=0o700, parents=True)
    record = f"bts_deploy_attempt_v1 456 1 {TARGET} {OLD} begun - - - -\n"
    (lock_dir / "attempt").write_text(record)
    result, calls = run_box_script(box, scripts[1], "active", run_id="123", attempt="1")
    assert "a different attempt owns the box record" in result.stdout
    assert (lock_dir / "attempt").read_text() == record
    assert (lock_dir / "cancelled.123.1").is_file()
    assert not any(call[0] == "systemctl" for call in calls)


def test_round2_failure_post_never_follows_redirects(box):
    scripts, _ = scripts_and_steps()
    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/a\n")
    result, calls = run_box_script(box, scripts[0], "post-0-301", fault="head:17")
    posts = [call for call in calls if call[0] == "curl" and "--data-raw" in call]
    assert len(posts) == 1
    assert all(option not in posts[0] for option in ("-L", "--location", "--location-trusted"))
    assert "Failure ping received HTTP 2xx: unverified." in result.stdout


def test_round2_recovery_creates_real_lockfile_and_mark(box):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[1], "active")
    assert "no phase-2 record is present" in result.stdout
    lock_dir = box / "run/bts-deploy"
    for path in (lock_dir / "lock", lock_dir / "cancelled.123.1"):
        assert path.is_file()
        assert path.stat().st_mode & 0o777 == 0o600
        assert path.read_bytes() == b""


def test_round2_recovery_waits_for_a_held_lock(box):
    scripts, _ = scripts_and_steps()
    lock_dir = box / "run/bts-deploy"
    lock_dir.mkdir(mode=0o700, parents=True)
    with (lock_dir / "lock").open("a") as holder, ThreadPoolExecutor(max_workers=1) as executor:
        os.chmod(holder.name, 0o600)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        invocation = executor.submit(run_box_script, box, scripts[1], "active")
        try:
            deadline = time.monotonic() + 5
            while not (box / "waiting-for-lock").exists() and not invocation.done() and time.monotonic() < deadline:
                time.sleep(0.01)
            assert (box / "waiting-for-lock").exists()
            assert not invocation.done()
            assert not (lock_dir / "cancelled.123.1").exists()
        finally:
            fcntl.flock(holder, fcntl.LOCK_UN)
        result, calls = invocation.result(timeout=8)
    assert ["flock", "-w", "1500", "9"] in calls
    assert "no phase-2 record is present" in result.stdout
    assert (lock_dir / "cancelled.123.1").is_file()


def test_round2_wrap_points_to_checked_activation():
    claude = (WORKFLOW.parents[2] / "CLAUDE.md").read_text()
    wrap = claude.split("## Season 2026 wrap", 1)[1].split("\n\n", 1)[0]
    assert "follow checklist B8a's activation order and deploy inside an approved sleep window" in wrap


def test_external_commands_resolve_in_the_shim_directory(box):
    names = {item.name for item in (box / "shims").iterdir()}
    assert names == COMMANDS
    for name in names:
        assert shutil.which(name, path=str(box / "shims")) == str(box / "shims" / name)
    assert shutil.which("ssh", path=str(box / "shims")) is None
    with pytest.raises(AssertionError):
        assert_direct_paths(box, "/usr/bin/ssh")


@pytest.mark.parametrize("scenario", ["active", "auto-restart"])
def test_healthy_scheduler_passes(box, scenario):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], scenario)
    assert result.returncode == 0, result
    assert "CANARY PASSED" in result.stdout


def test_healthy_wait_deadline(box):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[0], "active")
    assert result.returncode == 0, result
    assert ["timeout", "-k", "2", "33", "sleep", "30"] in calls


def test_http_failure_leaves_candidate_checkout(box):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[0], "http")
    assert result.returncode == 1, result
    assert ["git", "reset", "--hard", TARGET] in calls
    assert ["git", "reset", "--hard", OLD] not in calls
    assert (box / "head").read_text() == TARGET
    assert "DEPLOY FAILED at dashboard-http" in result.stdout


def test_preflight_failure_leaves_checkout_and_services_alone(box):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[0], "fetch")
    assert result.returncode == 1, result
    assert "DEPLOY NOT STARTED — fetch failed. This deploy invocation" in result.stdout
    assert not any(call[0] == "systemctl" for call in calls)
    assert (box / "head").read_text() == OLD


def test_sync_failure_takes_the_common_containment_path(box):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[0], "sync")
    assert result.returncode == 1, result
    assert "DEPLOY FAILED at sync" in result.stdout
    assert "Check sync failed (17)." in result.stdout
    assert sum(call[:3] == ["systemctl", "--user", "stop"] for call in calls) == 1
    for result_label in ("Scheduler stopped:", "Failure heartbeat written:", "Failure ping received HTTP 2xx:"):
        assert result.stdout.count(result_label) == 1
    assert ["git", "reset", "--hard", OLD] not in calls


def test_stamp_precedes_pass(box):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[0], "active")
    assert result.returncode == 0, result
    assert "CANARY PASSED — bbbbbbb passed the deploy checks" in result.stdout
    assert (box / "home/projects/bts/data/.last_deploy_iso").is_file()
    record = (box / "run/bts-deploy/attempt").read_text()
    assert record == f"bts_deploy_attempt_v1 123 1 {TARGET} {OLD} committed - - - -\n"


def test_recovery_first_cancels_late_deploy(box):
    scripts, _ = scripts_and_steps()
    assert len(scripts) == 2
    result, calls = run_box_script(box, scripts[1], "active")
    assert result.returncode == 1, result
    assert "no phase-2 record is present" in result.stdout
    assert not any(call[0] == "systemctl" for call in calls)
    assert (box / "run/bts-deploy/cancelled.123.1").read_bytes() == b""
    result, calls = run_box_script(box, scripts[0], "active")
    assert result.returncode == 1, result
    assert "DEPLOY NOT STARTED — cancelled failed." in result.stdout
    assert (box / "head").read_text() == OLD
    assert not any(call[0] == "systemctl" for call in calls)


def test_operator_documentation_records_recovery_eligibility():
    agreed = "Recovery is eligible after a failed or cancelled deploy step."
    for name in ("INCIDENT.md", "CLAUDE.md"):
        assert agreed in (WORKFLOW.parents[2] / name).read_text()


def test_deploy_and_recovery_share_containment_and_helpers():
    scripts, _ = scripts_and_steps()
    deploy, recovery = scripts
    assert deploy.split("\nnot_started() {", 1)[0] == recovery.split("\nrecovery_exit() {", 1)[0]


def test_actual_lock_contender_cannot_deploy(box):
    scripts, _ = scripts_and_steps()
    lock_dir = box / "run/bts-deploy"
    lock_dir.mkdir(mode=0o700, parents=True)
    with (lock_dir / "lock").open("a") as holder:
        os.chmod(holder.name, 0o600)
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        result, calls = run_box_script(box, scripts[0], "active")
    assert result.returncode == 1
    assert "DEPLOY NOT STARTED — lock failed." in result.stdout
    assert not any(call[0] == "systemctl" for call in calls)


@pytest.mark.parametrize("case", ["duplicate", "different-target", "invalid-record", "committed"])
def test_attempt_record_controls_authorization(box, case):
    scripts, _ = scripts_and_steps()
    assert len(scripts) == 2
    lock_dir = box / "run/bts-deploy"
    lock_dir.mkdir(mode=0o700, parents=True)
    attempt = lock_dir / "attempt"
    target = "c" * 40 if case == "different-target" else TARGET
    record = f"bts_deploy_attempt_v1 123 1 {target} {OLD} committed - - - -\n"
    if case == "invalid-record":
        record += "extra\n"
    attempt.write_text(record)
    result, calls = run_box_script(box, scripts[int(case != "duplicate")], "active")
    assert result.returncode == 1
    if case == "duplicate":
        assert "DEPLOY NOT STARTED — attempt-record failed." in result.stdout
    elif case == "committed":
        assert "this attempt's record says committed" in result.stdout
    else:
        assert "the attempt record is unreadable" in result.stdout
    assert not any(call[0] == "systemctl" for call in calls)
    assert attempt.read_text() == record


@pytest.mark.parametrize("case", ["dangling-directory", "dangling-lock", "mode", "fifo"])
def test_invalid_lock_paths_are_refused(box, case):
    scripts, _ = scripts_and_steps()
    lock_dir = box / "run/bts-deploy"
    lock_dir.parent.mkdir()
    if case == "dangling-directory":
        lock_dir.symlink_to(box / "absent")
    else:
        lock_dir.mkdir(mode=0o700)
        lock = lock_dir / "lock"
        if case == "dangling-lock":
            lock.symlink_to(box / "absent")
        elif case == "fifo":
            os.mkfifo(lock)
        else:
            lock.touch(mode=0o644)
    result, calls = run_box_script(box, scripts[0], "active")
    assert result.returncode == 1
    assert "DEPLOY NOT STARTED — lock failed." in result.stdout
    assert not any(call[0] == "systemctl" for call in calls)


@pytest.mark.parametrize(
    ("scenario", "step", "status"),
    [
        ("extra-disabled-line", "disable", "1"),
        ("duplicate-query", "scheduler-state", "query"),
        ("http-transport", "dashboard-http", "28"),
        ("contest-refused", "contest-state", "1"),
        ("contest-error", "contest-state", "2"),
        ("contest-multiline", "contest-state", "query"),
        ("commit-error", "commit-record", "17"),
    ],
)
def test_failed_checks_are_reported_and_contained(box, scenario, step, status):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], scenario)
    assert result.returncode == 1, result
    assert f"DEPLOY FAILED at {step}" in result.stdout
    assert result.stdout.count(f"Check {step} failed ({status}).") == 1
    assert "CANARY PASSED" not in result.stdout
    assert "UNEXPECTED OUTPUT" not in result.stdout
    assert (box / "home/projects/bts/data/.last_deploy_iso").exists() == (scenario == "commit-error")
    marker = json.loads((box / "home/projects/bts/data/.heartbeat").read_text())
    assert (marker["state"], marker["step"], marker["target_sha"], marker["old_sha"]) == (
        "deploy_failed", step, TARGET[:7], OLD[:7],
    )


@pytest.mark.parametrize("scenario", ["tailnet-error", "published-with-error", "legacy-present"])
def test_passing_controls_for_legacy_fallback_and_publication(box, scenario):
    scripts, _ = scripts_and_steps()
    if scenario == "legacy-present":
        fragments = box / "etc/systemd/system"
        fragments.mkdir(parents=True)
        for name in ("bts-scheduler.service", "bts-dashboard.service"):
            (fragments / name).touch()
    result, calls = run_box_script(box, scripts[0], scenario)
    assert result.returncode == 0, result
    assert "CANARY PASSED" in result.stdout
    if scenario == "tailnet-error":
        assert any(call[0] == "curl" and call[-1] == "http://127.0.0.1:3003/" for call in calls)
    if scenario == "legacy-present":
        for name in ("bts-scheduler.service", "bts-dashboard.service"):
            assert ["systemctl", "stop", name] in calls
            assert not (fragments / name).exists()


def test_recovery_checks_a_fresh_marker_and_keeps_verified_post(box):
    scripts, _ = scripts_and_steps()
    record_dir = box / "run/bts-deploy"
    record_dir.mkdir(mode=0o700, parents=True)
    (record_dir / "attempt").write_text(
        f"bts_deploy_attempt_v1 123 1 {TARGET} {OLD} contained contest-state verified verified verified\n"
    )
    marker = box / "home/projects/bts/data/.heartbeat"
    marker.write_text('{"state":"running"}')
    result, calls = run_box_script(box, scripts[1], "active")
    assert result.returncode == 1
    assert "Failure heartbeat written: verified." in result.stdout
    assert "Failure ping received HTTP 2xx: verified." in result.stdout
    assert not any(call[0] == "curl" for call in calls)
    assert json.loads(marker.read_text())["step"] == "contest-state"


def test_recovery_does_not_mutate_services_when_mark_fails(box):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[1], "mark-error")
    assert result.returncode == 1
    assert "cancellation mark could not be verified" in result.stdout
    assert not any(call[0] == "systemctl" for call in calls)


@pytest.mark.parametrize("url", [
    '"https://example.invalid/token/"',
    "https://example.invalid/token?",
    "https://example.invalid/token#",
    "https://example.invalid:bad/token",
    "https://user@example.invalid/token",
])
def test_post_only_uses_a_valid_literal_https_endpoint(box, url):
    scripts, _ = scripts_and_steps()
    (box / "home/projects/bts/.env").write_text(f"BTS_SCHEDULER_HEARTBEAT_PING_URL={url}\n")
    result, calls = run_box_script(box, scripts[0], "sync")
    assert result.returncode == 1
    assert "example.invalid" not in result.stdout + result.stderr
    posts = [call for call in calls if call[0] == "curl" and "--data-raw" in call]
    if url.startswith('"'):
        assert len(posts) == 1
        assert posts[0][-1] == "https://example.invalid/token/fail"
        assert "Failure ping received HTTP 2xx: verified." in result.stdout
    else:
        assert not posts
        assert "Failure ping received HTTP 2xx: unverified." in result.stdout


PHASE_TWO = {
    "disable": "disable", "check-disable": "disable", "reset": "reset", "sync": "sync",
    "legacy-disable": "legacy-units", "legacy-stop": "legacy-units", "legacy-rm": "legacy-units",
    "legacy-daemon-reload": "legacy-units", "legacy-pkill": "legacy-units",
    "restart": "restart", "head": "head", "wait": "wait",
}
PHASE_THREE = ("scheduler-state", "dashboard-state", "dashboard-http", "contest-state")
PHASE_FOUR = {
    "stamp-stage": "stamp-stage", "enable": "enable", "check-enable": "enable",
    "stamp": "stamp", "publish-committed": "commit-record",
}


@pytest.mark.parametrize("status", [17, 124])
@pytest.mark.parametrize(("command", "step"), [*PHASE_TWO.items(), *PHASE_FOUR.items()])
def test_failing_phase_command_contains_first_failure(box, command, step, status):
    scripts, _ = scripts_and_steps()
    scenario = "legacy-present" if command.startswith("legacy-") else "active"
    result, calls = run_box_script(box, scripts[0], scenario, fault=f"{command}:{status}")
    expected = "timeout" if status == 124 else "1" if command.startswith("check-") else str(status)
    assert result.returncode == 1
    assert (box / "fault-fired").exists()
    assert f"DEPLOY FAILED at {step}" in result.stdout
    assert result.stdout.count(f"Check {step} failed ({expected}).") == 1
    assert ["git", "reset", "--hard", OLD] not in calls


def test_head_mismatch_contains_without_rollback(box):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], "active", fault="head:mismatch")
    assert "DEPLOY FAILED at head" in result.stdout
    assert "Check head failed (query)." in result.stdout


@pytest.mark.slow
@pytest.mark.parametrize("signal_name", ["HUP", "INT", "TERM", "PIPE"])
@pytest.mark.parametrize("command", [*PHASE_TWO, *PHASE_THREE, *PHASE_FOUR])
def test_owner_signal_at_each_phase_boundary(box, command, signal_name):
    scripts, _ = scripts_and_steps()
    scenario = "legacy-present" if command.startswith("legacy-") else "active"
    fault = f"{command}:{signal_name}"
    if command == "publish-committed":
        # Keep the rename unpublished here. The publication-boundary tests
        # separately verify signals after a successful atomic rename.
        fault += ":unpublished"
    result, calls = run_box_script(box, scripts[0], scenario, fault=fault)
    assert (box / "fault-fired").exists()
    assert result.returncode == 1, result
    assert result.stdout.count("DEPLOY FAILED at interrupted") == 1
    assert result.stdout.count("Check interrupted failed (130).") == 1
    assert ["git", "reset", "--hard", OLD] not in calls


@pytest.mark.parametrize("path_type", ["directory", "symlink-to-directory", "old-file"])
def test_stamp_failure_preserves_previous_entry(box, path_type):
    scripts, _ = scripts_and_steps()
    stamp = box / "home/projects/bts/data/.last_deploy_iso"
    if path_type == "directory":
        stamp.mkdir()
    elif path_type == "symlink-to-directory":
        destination = box / "destination"
        destination.mkdir()
        stamp.symlink_to(destination)
    else:
        stamp.write_bytes(b"old stamp\n")
    fault = "stamp:17" if path_type == "old-file" else ""
    result, _ = run_box_script(box, scripts[0], "active", fault=fault)
    assert result.returncode == 1
    assert "DEPLOY FAILED at stamp" in result.stdout
    assert "CANARY PASSED" not in result.stdout
    if path_type == "old-file":
        assert stamp.read_bytes() == b"old stamp\n"
    else:
        assert stamp.is_dir()
        assert stamp.is_symlink() == (path_type == "symlink-to-directory")


@pytest.mark.parametrize(("scenario", "step", "status"), [
    ("malformed-query", "scheduler-state", "query"),
    ("crash-loop", "scheduler-state", "1"),
    ("dashboard-inactive", "dashboard-state", "1"),
])
def test_malformed_and_unhealthy_canary_states(box, scenario, step, status):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], scenario)
    assert result.returncode == 1
    assert f"Check {step} failed ({status})." in result.stdout


def test_canary_checks_continue_after_first_failure(box):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], "http", fault="scheduler-state:17")
    assert result.returncode == 1
    assert result.stdout.count("Check scheduler-state failed (query).") == 1
    assert result.stdout.count("Check dashboard-http failed (503).") == 1
    assert "DEPLOY FAILED at scheduler-state" in result.stdout


@pytest.mark.parametrize("scenario", [
    "stop-error", "main-pid", "pending-job", "descendant-cgroup", "legacy-survivor",
    "residual", "all-errors",
])
def test_stop_verification_requires_all_evidence(box, scenario):
    scripts, _ = scripts_and_steps()
    data = box / "home/projects/bts/data"
    if scenario == "descendant-cgroup":
        scope = box / "sys/fs/cgroup/sched"
        scope.mkdir()
        (scope / "cgroup.procs").touch()
        (scope / "cgroup.events").write_text("populated 1\n")
    if scenario == "all-errors":
        (data / ".heartbeat").mkdir()
    else:
        (data.parent / ".env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://example.invalid/token\n")
    result, calls = run_box_script(box, scripts[0], scenario, fault="sync:17")
    assert result.returncode == 1
    assert "Scheduler stopped: unverified." in result.stdout
    assert ("Failure heartbeat written: verified." in result.stdout) == (scenario != "all-errors")
    assert ("Failure ping received HTTP 2xx: verified." in result.stdout) == (scenario != "all-errors")
    if scenario == "legacy-survivor":
        assert ["systemctl", "stop", "bts-scheduler.service"] in calls


@pytest.mark.parametrize("kind", ["directory", "symlink", "fifo", "old-marker", "unwritable"])
def test_marker_replacement_and_verification(box, kind):
    scripts, _ = scripts_and_steps()
    marker = box / "home/projects/bts/data/.heartbeat"
    destination = box / "other-heartbeat"
    if kind == "directory":
        marker.mkdir()
    elif kind == "symlink":
        destination.write_bytes(b"outside data")
        marker.symlink_to(destination)
    elif kind == "fifo":
        os.mkfifo(marker)
    elif kind == "old-marker":
        marker.write_text('{"state":"deploy_failed","timestamp":"old","step":"sync","target_sha":"bbbbbbb","old_sha":"aaaaaaa"}')
    result, _ = run_box_script(box, scripts[0], "marker-unwritable" if kind == "unwritable" else "active",
                               fault="sync:17")
    assert result.returncode == 1
    verified = kind not in {"directory", "unwritable"}
    assert ("Failure heartbeat written: verified." in result.stdout) == verified
    if verified:
        assert not marker.is_symlink()
        assert json.loads(marker.read_text())["timestamp"] != "old"
    if kind == "symlink":
        assert destination.read_bytes() == b"outside data"


@pytest.mark.parametrize("scenario", ["post-28-200", "post-0-503", "post-0-301", "post-7-200"])
def test_post_requires_exit_zero_and_2xx(box, scenario):
    scripts, _ = scripts_and_steps()
    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://example.invalid/check\n")
    result, calls = run_box_script(box, scripts[0], scenario, fault="sync:17")
    assert any(call[0] == "curl" and "--data-raw" in call for call in calls)
    assert "Failure ping received HTTP 2xx: unverified." in result.stdout


@pytest.mark.parametrize("kind", ["directory", "fifo", "oversized", "no-assignment", "two-assignments", "plain-http"])
def test_invalid_env_never_posts(box, kind):
    scripts, _ = scripts_and_steps()
    path = box / "home/projects/bts/.env"
    if kind == "directory":
        path.mkdir()
    elif kind == "fifo":
        os.mkfifo(path)
    else:
        line = "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://example.invalid/check\n"
        path.write_text({"oversized": " " * 65537, "no-assignment": "FOO=bar\n",
                         "two-assignments": line * 2, "plain-http": line.replace("https://", "http://")}[kind])
    result, calls = run_box_script(box, scripts[0], "active", fault="sync:17")
    assert not any(call[0] == "curl" and "--data-raw" in call for call in calls)
    assert "Failure ping received HTTP 2xx: unverified." in result.stdout


@pytest.mark.parametrize("signal_name", ["HUP", "INT", "TERM", "PIPE"])
@pytest.mark.parametrize(("fault_pattern", "step"), [
    ("publish-begun:{}:unpublished", None),
    ("publish-begun:{}", "interrupted"),
    ("publish-committed:{}", None),
    ("log-pass:{}", None),
])
def test_signals_at_publication_and_after_commit(box, fault_pattern, step, signal_name):
    scripts, _ = scripts_and_steps()
    fault = fault_pattern.format(signal_name)
    result, calls = run_box_script(box, scripts[0], "active", fault=fault)
    assert (box / "fault-fired").exists()
    if "unpublished" in fault:
        assert result.returncode == 1
        assert "DEPLOY NOT STARTED" in result.stdout
        assert ["git", "reset", "--hard", TARGET] not in calls
    elif step:
        assert result.returncode == 1
        assert f"DEPLOY FAILED at {step}" in result.stdout
    else:
        assert result.returncode == 0
        assert "DEPLOY FAILED" not in result.stdout


def test_second_signal_does_not_reenter_containment(box):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[0], "second-signal", fault="sync:TERM")
    assert result.returncode == 1
    assert result.stdout.count("DEPLOY FAILED at interrupted") == 1
    assert calls.count(["systemctl", "--user", "stop", "bts-scheduler"]) == 1


def test_committed_deploy_survives_a_failed_final_log(box):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], "active", fault="log-pass:17")
    assert result.returncode == 0
    assert " committed " in (box / "run/bts-deploy/attempt").read_text()


def test_owner_death_does_not_release_lock_while_mutator_is_live(box):
    scripts, _ = scripts_and_steps()
    with ThreadPoolExecutor(max_workers=1) as executor:
        owner = executor.submit(run_box_script, box, scripts[0], "active", fault="reset:hold")
        try:
            deadline = time.monotonic() + 8
            while not (box / "fault-fired").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert (box / "fault-fired").exists()
            os.kill(int((box / "owner.pid").read_text()), signal.SIGKILL)
            contender, _ = run_box_script(box, scripts[1], "active")
            assert contender.returncode == 1
            assert "lock unavailable" in contender.stdout
        finally:
            (box / "release").touch()
        assert owner.result(timeout=15)[0].returncode == -signal.SIGKILL
    recovered, _ = run_box_script(box, scripts[1], "active")
    assert "DEPLOY FAILED at deploy-step-incomplete" in recovered.stdout


@pytest.mark.parametrize(("command", "enabled"), [
    ("disable", "enabled"),
    *((command, "disabled") for command in PHASE_TWO if command != "disable"),
    *((command, "disabled") for command in PHASE_THREE),
    ("stamp-stage", "disabled"), ("enable", "disabled"), ("check-enable", "enabled"),
    ("stamp", "enabled"), ("publish-committed", "enabled"),
])
def test_boot_link_and_record_loss_at_boundaries(box, command, enabled):
    scripts, _ = scripts_and_steps()
    scenario = "legacy-present" if command.startswith("legacy-") else "active"
    result, _ = run_box_script(box, scripts[0], scenario, fault=f"{command}:KILL:unpublished")
    assert result.returncode == -signal.SIGKILL
    link = box / "enabled"
    assert (link.read_text() if link.exists() else "enabled") == enabled
    shutil.rmtree(box / "run")
    result, _ = run_box_script(box, scripts[1], "active")
    assert "no phase-2 record is present" in result.stdout


def test_older_recovery_does_not_mutate_newer_commit(box):
    scripts, _ = scripts_and_steps()
    deployed, _ = run_box_script(box, scripts[0], "active", attempt="2")
    assert deployed.returncode == 0
    (box / "calls").unlink()
    recovered, calls = run_box_script(box, scripts[1], "active", attempt="1")
    assert "a different attempt owns the box record" in recovered.stdout
    assert not any(call[0] == "systemctl" for call in calls)
    assert (box / "run/bts-deploy/cancelled.123.1").is_file()


def test_new_attempt_can_deploy_after_containment(box):
    scripts, _ = scripts_and_steps()
    failed, _ = run_box_script(box, scripts[0], "sync", attempt="1")
    assert failed.returncode == 1
    assert (box / "enabled").read_text() == "disabled"
    passed, _ = run_box_script(box, scripts[0], "active", attempt="2")
    assert passed.returncode == 0
    assert (box / "enabled").read_text() == "enabled"
    assert "CANARY PASSED" in passed.stdout
    assert json.loads((box / "home/projects/bts/data/.heartbeat").read_text())["state"] == "running"


@pytest.fixture(scope="module")
def baseline_script():
    baseline = subprocess.run(
        ["git", "show", "24019a9a:.github/workflows/deploy.yml"],
        cwd=WORKFLOW.parents[2], capture_output=True, text=True, check=True, timeout=5,
    ).stdout
    scripts, _ = scripts_and_steps(baseline)
    old, = scripts
    return old


@pytest.mark.parametrize("legacy", [False, True], ids=["legacy-absent", "legacy-present"])
def test_passing_transcript_order_delta(box, baseline_script, legacy):
    scripts, _ = scripts_and_steps()
    fragments = box / "etc/systemd/system"
    data = box / "home/projects/bts/data"
    mode = "legacy-present" if legacy else "active"

    def prepare():
        (box / "head").write_text(OLD)
        (box / "calls").unlink(missing_ok=True)
        if legacy:
            fragments.mkdir(parents=True, exist_ok=True)
            for unit in ("bts-scheduler.service", "bts-dashboard.service"):
                (fragments / unit).touch()

    prepare()
    base, old_calls = run_box_script(box, baseline_script, mode)
    prepare()
    current, new_calls = run_box_script(box, scripts[0], mode)
    assert (base.returncode, current.returncode) == (0, 0)
    assert "is live and healthy" in base.stdout
    assert "passed the deploy checks" in current.stdout
    assert (data / ".last_deploy_iso").is_file()
    old = [json.dumps(call, ensure_ascii=True) for call in old_calls]
    new = [json.dumps(call, ensure_ascii=True) for call in new_calls]
    delta = list(difflib.SequenceMatcher(a=old, b=new, autojunk=False).get_opcodes())
    replay = [line for tag, i, j, k, count in delta
              for line in (old[i:j] if tag == "equal" else new[k:count])]
    assert replay == new
    (box / "baseline-transcript.json").write_text(json.dumps({"argv": old_calls, "stdout": base.stdout}, indent=2))
    (box / "candidate-transcript.json").write_text(json.dumps({"argv": new_calls, "stdout": current.stdout}, indent=2))
    (box / "delta.json").write_text(json.dumps(delta, indent=2))
    ordered = [
        ["git", "fetch", "origin"], ["systemctl", "--user", "disable", "bts-scheduler"],
        ["git", "reset", "--hard", TARGET], ["uv", "sync", "--locked", "--extra", "model"],
        ["systemctl", "--user", "restart", "bts-scheduler", "bts-dashboard"],
        ["timeout", "-k", "2", "33", "sleep", "30"],
        ["systemctl", "--user", "enable", "bts-scheduler"],
    ]
    positions = [new_calls.index(call) for call in ordered]
    assert positions == sorted(positions)
    if legacy:
        assert ["systemctl", "disable", "--now", "bts-scheduler.service", "bts-dashboard.service"] in old_calls
        assert ["systemctl", "stop", "bts-scheduler.service"] in new_calls
    else:
        assert ["systemctl", "stop", "bts-scheduler.service"] not in new_calls


@pytest.mark.parametrize(("scenario", "fault", "step"), [
    ("active", "uid:17", "uid"),
    ("active", "rev-parse:17", "old-sha"),
    ("active", "config:17", "git-config"),
    ("active", "cat-file:17", "target"),
    ("active", "user-units:17", "user-units"),
    ("active", "legacy-units:17", "legacy-units"),
    ("masked", "", "user-units"),
    ("linked", "", "user-units"),
    ("static", "", "user-units"),
    ("unloaded", "", "user-units"),
    ("bad-owner", "", "lock"),
])
def test_preflight_checks_refuse_before_phase_two(box, scenario, fault, step):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[0], scenario, fault=fault)
    assert result.returncode == 1
    assert f"DEPLOY NOT STARTED — {step} failed." in result.stdout
    assert (box / "head").read_text() == OLD
    assert not any("disable" in call or "restart" in call or "stop" in call for call in calls)


@pytest.mark.parametrize(("kwargs", "step"), [
    ({"target": "A" * 40}, "target"), ({"run_id": "0"}, "attempt-id"),
    ({"attempt": "1;false"}, "attempt-id"), ({"attempt": "1" * 21}, "attempt-id"),
])
def test_bad_inputs_are_refused_in_both_scripts(box, kwargs, step):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], "active", **kwargs)
    assert f"{step} failed" in result.stdout
    result, _ = run_box_script(box, scripts[1], "active", **kwargs)
    assert "invalid attempt inputs" in result.stdout
    assert not (box / "run").exists()


def test_literal_epipe_does_not_change_deploy_outcome(box):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], "active", closed_output=True)
    assert result.returncode == 0
    assert " committed " in (box / "run/bts-deploy/attempt").read_text()


@pytest.mark.parametrize("unverified", ["stop", "marker", "page"])
def test_recovery_rechecks_each_unverified_result(box, unverified):
    scripts, _ = scripts_and_steps()
    results = " ".join("unverified" if name == unverified else "verified" for name in ("stop", "marker", "page"))
    root = box / "run/bts-deploy"
    root.mkdir(mode=0o700, parents=True)
    (root / "attempt").write_text(
        f"bts_deploy_attempt_v1 123 1 {TARGET} {OLD} contained sync {results}\n"
    )
    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://example.invalid/check\n")
    result, calls = run_box_script(box, scripts[1], "active")
    assert result.returncode == 1
    assert "containment re-checked" in result.stdout
    assert (root / "attempt").read_text().endswith("sync verified verified verified\n")
    assert any(call[0] == "curl" and "--data-raw" in call for call in calls) == (unverified == "page")


@pytest.mark.parametrize("step", PHASE_THREE)
def test_canary_timeout_is_reported(box, step):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], "active", fault=f"{step}:124")
    assert result.returncode == 1
    assert f"Check {step} failed (timeout)." in result.stdout
    assert f"DEPLOY FAILED at {step}" in result.stdout


@pytest.mark.parametrize("scenario", ["partial-stamp", "stamp-directory-sync"])
def test_stamp_write_and_dirsync_faults_keep_complete_bytes(box, scenario):
    scripts, _ = scripts_and_steps()
    data = box / "home/projects/bts/data"
    stamp = data / ".last_deploy_iso"
    stamp.write_bytes(b"previous\n")
    result, _ = run_box_script(box, scripts[0], scenario)
    expected = "stamp-stage" if scenario == "partial-stamp" else "stamp"
    assert result.returncode == 1
    assert f"DEPLOY FAILED at {expected}" in result.stdout
    assert "CANARY PASSED" not in result.stdout
    assert not list(data.glob(".deploy.*"))
    if scenario == "partial-stamp":
        assert stamp.read_bytes() == b"previous\n"
    else:
        assert re.fullmatch(rb"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ\n", stamp.read_bytes())


def test_preflight_refuses_a_missing_checkout(box):
    scripts, _ = scripts_and_steps()
    shutil.rmtree(box / "home")
    result, calls = run_box_script(box, scripts[0], "active")
    assert result.returncode == 1
    assert "DEPLOY NOT STARTED — cd failed." in result.stdout
    assert not any(call[0] in {"systemctl", "sudo", "flock"} for call in calls)


def test_missing_candidate_venv_fails_contest_check(box):
    scripts, _ = scripts_and_steps()
    (box / "shims/bts").unlink()
    result, calls = run_box_script(box, scripts[0], "active")
    assert result.returncode == 1
    assert "DEPLOY FAILED at contest-state" in result.stdout
    expected = str(box / "home/projects/bts/.venv/bin/bts")
    assert any(call[:2] == ["env", "PYTHONDONTWRITEBYTECODE=1"] and expected in call
               for call in calls)


def test_expired_shared_canary_budget_skips_remaining_commands(box):
    scripts, _ = scripts_and_steps()
    result, calls = run_box_script(box, scripts[0], "canary-budget")
    assert result.returncode == 1
    for step in ("dashboard-state", "dashboard-http", "contest-state"):
        assert f"Check {step} failed (timeout)." in result.stdout
    assert "DEPLOY FAILED at dashboard-state" in result.stdout
    assert not any(call[0] in {"tailscale", "curl", "bts"} for call in calls)


def test_mutator_does_not_inherit_lock_descriptor(box):
    scripts, _ = scripts_and_steps()
    result, _ = run_box_script(box, scripts[0], "closed-mutator-fd")
    assert result.returncode == 0
    assert (box / "mutator-closed-fd").exists()


def test_old_recovery_during_ownership_transfer(box):
    scripts, _ = scripts_and_steps()
    root = box / "run/bts-deploy"
    root.mkdir(mode=0o700, parents=True)
    (root / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {TARGET} {OLD} begun - - - -\n")
    with ThreadPoolExecutor(max_workers=1) as executor:
        owner = executor.submit(run_box_script, box, scripts[0], "active", fault="reset:hold", attempt="2")
        try:
            deadline = time.monotonic() + 8
            while not (box / "fault-fired").exists() and time.monotonic() < deadline:
                time.sleep(0.02)
            assert (box / "fault-fired").exists()
            assert (root / "attempt").read_text().split()[2] == "2"
            older, calls = run_box_script(box, scripts[1], "active", attempt="1")
            assert "lock unavailable" in older.stdout
            assert not any(call[0] == "systemctl" and "stop" in call for call in calls)
        finally:
            (box / "release").touch()
        assert owner.result(timeout=15)[0].returncode == 0
    older, _ = run_box_script(box, scripts[1], "active", attempt="1")
    assert "a different attempt owns the box record" in older.stdout


# ---------------------------------------------------------------------------
# D3b (contract amendment 4, rulings R56 and R57). The class: recovery claims a
# state it did not prove. Row numbers K9 to K20 refer to the amendment's
# "must still catch" table.

D3B_PRIORS = {
    "other-run-committed": f"bts_deploy_attempt_v1 101 1 {OLD} {'c' * 40} committed - - - -\n",
    "earlier-attempt-contained": f"bts_deploy_attempt_v1 123 1 {TARGET} {OLD} contained sync verified verified verified\n",
    "other-run-begun": f"bts_deploy_attempt_v1 101 1 {'c' * 40} {OLD} begun - - - -\n",
}


def d3b_box(box, prior=None):
    runtime = box / "run/bts-deploy"
    runtime.mkdir(mode=0o700, parents=True)
    (box / "enabled").write_text("enabled")
    (box / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/d3b\n")
    if prior:
        (runtime / "attempt").write_text(D3B_PRIORS[prior])
    return runtime


def posts(calls):
    return [call for call in calls if call[0] == "curl" and "--data-raw" in call]


def service_mutations(calls):
    return [call for call in calls if call[0] == "systemctl" and any(
        word in call for word in ("disable", "enable", "stop", "restart", "daemon-reload"))]


@pytest.mark.parametrize("signal_name", ["HUP", "INT", "TERM", "PIPE"])
@pytest.mark.parametrize("where", ["stage-begun", "publish-begun"])
@pytest.mark.parametrize("prior", sorted(D3B_PRIORS))
def test_d3b_signal_before_begin_is_published_never_contains(box, prior, where, signal_name):
    """K9 (round-2 finding 1, both reviewers): rule S6."""
    scripts, _ = scripts_and_steps()
    runtime = d3b_box(box, prior)
    attempt = "2" if prior == "earlier-attempt-contained" else "1"
    result, calls = run_box_script(box, scripts[0], "active", fault=f"{where}:{signal_name}:unpublished", attempt=attempt)
    assert (box / "fault-fired").exists()
    assert result.returncode == 1
    assert "DEPLOY NOT STARTED — attempt-record failed." in result.stdout
    assert "DEPLOY FAILED" not in result.stdout
    assert not service_mutations(calls) and not posts(calls)
    assert (runtime / "attempt").read_text() == D3B_PRIORS[prior]
    assert (box / "home/projects/bts/data/.heartbeat").read_text() == '{"state":"running"}'
    assert (box / "enabled").read_text() == "enabled"
    assert (box / "head").read_text() == OLD


@pytest.mark.parametrize("prior", [None, *sorted(D3B_PRIORS)])
def test_d3b_signal_after_begin_is_published_contains(box, prior):
    """K10 control (rule S7): the begin was renamed before the signal."""
    scripts, _ = scripts_and_steps()
    runtime = d3b_box(box, prior)
    attempt = "2" if prior == "earlier-attempt-contained" else "1"
    result, calls = run_box_script(box, scripts[0], "active", fault="publish-begun:TERM", attempt=attempt)
    assert result.returncode == 1
    assert sum(call[0] == "mv" and call[-1] == str(runtime / "attempt") for call in calls) >= 2
    assert result.stdout.count("DEPLOY FAILED at interrupted") == 1
    assert "DEPLOY NOT STARTED" not in result.stdout
    assert ["systemctl", "--user", "stop", "bts-scheduler"] in [call[:4] for call in calls]
    assert f" {attempt} {TARGET} {OLD} contained interrupted " in (runtime / "attempt").read_text()


@pytest.mark.parametrize("prior", sorted(D3B_PRIORS))
def test_d3b_unreadable_record_while_beginning_contains(box, prior):
    """K10 control (rule S9): a failed fresh read cannot exclude publication."""
    scripts, _ = scripts_and_steps()
    d3b_box(box, prior)
    attempt = "2" if prior == "earlier-attempt-contained" else "1"
    result, calls = run_box_script(box, scripts[0], "beginning-read-126", fault="stage-begun:TERM:unpublished",
                                   attempt=attempt)
    assert (box / "read-failed").exists()
    assert result.returncode == 1
    assert result.stdout.count("DEPLOY FAILED at interrupted") == 1
    assert "DEPLOY NOT STARTED" not in result.stdout


def d3b_legacy_group(box, populated):
    group = box / "sys/fs/cgroup/legacy-orphan"
    group.mkdir()
    (group / "cgroup.procs").touch()
    (group / "cgroup.events").write_text(f"populated {populated}\n")


def d3b_run_contained(box, recovery, scenario):
    scripts, _ = scripts_and_steps()
    runtime = d3b_box(box)
    if recovery:
        (box / "head").write_text(TARGET)
        (runtime / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {TARGET} {OLD} begun - - - -\n")
        return run_box_script(box, scripts[1], scenario)
    return run_box_script(box, scripts[0], scenario, fault="reset:17")


@pytest.mark.parametrize("recovery", [False, True], ids=["deploy", "recovery"])
@pytest.mark.parametrize(("populated", "verified"), [(1, False), (0, True)])
def test_d3b_recorded_legacy_group_stays_in_the_proof(box, recovery, populated, verified):
    """K11 (round-2 finding 2) and its K12 control."""
    d3b_legacy_group(box, populated)
    result, calls = d3b_run_contained(box, recovery, "legacy-vanishing")
    assert result.returncode == 1
    assert ["systemctl", "stop", "bts-scheduler.service"] in calls
    assert any(call[0] == "python3" and call[3:5] == ["cgroup", str(box / "sys/fs/cgroup")]
               and call[-1] == "/legacy-orphan" for call in calls)
    assert ("Scheduler stopped: verified." in result.stdout) == verified
    assert ("Scheduler stopped: unverified." in result.stdout) == (not verified)


@pytest.mark.parametrize("recovery", [False, True], ids=["deploy", "recovery"])
def test_d3b_loaded_legacy_unit_without_a_known_group_is_unverified(box, recovery):
    """K12 (new case of the class): seen loaded, no cgroup path known, then not-found."""
    result, calls = d3b_run_contained(box, recovery, "legacy-vanishing-nogroup")
    assert result.returncode == 1
    assert ["systemctl", "stop", "bts-scheduler.service"] in calls
    assert "Scheduler stopped: unverified." in result.stdout


@pytest.mark.parametrize("recovery", [False, True], ids=["deploy", "recovery"])
def test_d3b_legacy_absent_throughout_needs_no_legacy_stop(box, recovery):
    """K12 control: never loaded, so nothing to stop and the proof can pass."""
    result, calls = d3b_run_contained(box, recovery, "active")
    assert result.returncode == 1
    assert not any(call[:2] == ["systemctl", "stop"] for call in calls)
    assert "Scheduler stopped: verified." in result.stdout


@pytest.mark.parametrize("scope", ["user", "legacy"])
@pytest.mark.parametrize(("before", "after"), [(1, 0), (0, 1), (0, 0)])
def test_d3b_every_known_cgroup_path_is_in_the_proof(box, scope, before, after):
    """K16 (round-2 Carbon N3 b): distinct pre-stop and post-stop paths in both scopes."""
    scripts, _ = scripts_and_steps()
    names = ("a", "b") if scope == "user" else ("la", "lb")
    for name, populated in zip(names, (before, after)):
        group = box / "sys/fs/cgroup" / name
        group.mkdir()
        (group / "cgroup.events").write_text(f"populated {populated}\n")
    d3b_box(box)
    result, calls = run_box_script(box, scripts[0], f"moved-{scope}", fault="reset:17")
    checked = [call[-1] for call in calls if call[0] == "python3" and call[3:4] == ["cgroup"]]
    if before or after:
        assert "Scheduler stopped: unverified." in result.stdout
        assert f"/{names[0 if before else 1]}" in checked
    else:
        assert "Scheduler stopped: verified." in result.stdout
        assert {f"/{names[0]}", f"/{names[1]}"} <= set(checked)


def test_d3b_commit_readback_keeps_its_own_status(box):
    """K15 (round-2 note): a read-back exiting 126 after the rename is logged as 126."""
    scripts, _ = scripts_and_steps()
    d3b_box(box)
    result, _ = run_box_script(box, scripts[0], "commit-read-126")
    assert result.returncode == 1
    assert "Check commit-record failed (126)." in result.stdout


def shell_function(script, name):
    match = re.search(rf"^{name}\(\) {{\n(.*?)^}}\n", script, re.MULTILINE | re.DOTALL)
    assert match, name
    return match.group(1)


def test_d3b_containment_windows_reserve_the_residual_check():
    """K14 (M2): the windows in both scripts; marker and POST windows as before."""
    window = r"deadline=\$\(\(containment_deadline - (\d+)\)\)"
    for script in scripts_and_steps()[0]:
        contain, proof = shell_function(script, "contain"), shell_function(script, "prove")
        assert [int(n) for n in re.findall(window, contain)] == [190000, 100000, 40000, 26000, 10000, 2000]
        assert [int(n) for n in re.findall(window, proof)] == [75000, 60000]
        residual = proof.index("pgrep -u bts -f 'bts schedule'")
        assert proof.index("containment_deadline - 75000") < proof.index("containment_deadline - 60000") < residual
        assert contain.index("containment_deadline - 40000") < contain.index("\"$fs_code\" marker")
        assert contain.index("containment_deadline - 26000") < contain.index("\"$fs_code\" endpoint")
        assert contain.index("\"$fs_code\" endpoint") < contain.index("containment_deadline - 10000))") < contain.index("--data-raw")
        assert contain.index("\"$fs_code\" marker") < contain.index('rm -f -- "$staged"')
        assert contain.index("containment_deadline - 2000))") < contain.index("publish contained")


RESULT_LINES = ("Scheduler stopped:", "Failure heartbeat written:", "Failure ping received HTTP 2xx:",
                "containment re-checked")


def test_d3b_only_the_proof_function_makes_or_prints_a_verified_result():
    """K19 (M1): one proof function, byte-identical in both scripts (RD8)."""
    scripts, _ = scripts_and_steps()
    proofs = {shell_function(script, "prove") for script in scripts}
    assert len(proofs) == 1
    proof = proofs.pop()
    for script in scripts:
        outside = script.replace(proof, "")
        for line in RESULT_LINES:
            assert line in proof
            assert line not in outside
        assert not re.search(r"\b(stop|marker|page)=verified\b", outside)
        assert re.findall(r"\b(stop|marker|page)=verified\b", proof)


def test_d3b_signal_handler_has_one_named_rule():
    """K17 / M3: on_signal acts only on signal_rule's answer; the rule is shared."""
    scripts, _ = scripts_and_steps()
    deploy = scripts[0]
    handler = shell_function(deploy, "on_signal")
    assert "signal_rule" in handler
    assert "read_record" not in handler and "publication" not in handler
    rule = shell_function(deploy, "signal_rule")
    for answer in ("success", "not-started", "contain"):
        assert f"answer={answer}" in rule


def timed_shim():
    old = 'elif command == "timeout":\n    dispatch(args[3:])'
    assert SHIM.count(old) == 1
    shim = SHIM.replace(old, '''
elif command == "timeout":
    word = args[3]
    if word.startswith("/") and word != "/bin/bash" and not Path(word).is_relative_to(root):
        raise SystemExit(99)
    executable = root / "shims" / Path(word).name
    if not executable.is_file():
        raise SystemExit(98)
    os.execv("/usr/bin/timeout", ["timeout", *args[:3], str(executable), *args[4:]])''')
    delays = '''
    proof = (root / "stopped").exists() and (root / "legacy-stopped").exists()
    if (root / "clock-failed").exists():
        if args[0] == "stop":
            time.sleep(52.9)
        elif args[0] in {"show", "list-jobs", "is-enabled"} and proof:
            time.sleep(13.7)
        elif args[0] in {"show", "list-jobs", "disable"}:
            time.sleep(3.9)
    if user and args[0] in {"disable", "enable"}:'''
    old = '    if user and args[0] in {"disable", "enable"}:'
    assert shim.count(old) == 1
    shim = shim.replace(old, delays)
    old = '    if op == "marker" and scenario == "marker-unwritable":'
    assert shim.count(old) == 1
    shim = shim.replace(old, """    if op == "endpoint" and (root / "clock-failed").exists():
        time.sleep(12)
""" + old)
    old = '        if "ControlGroup" in args and user and scenario == "main-pid":'
    shim = shim.replace(old, '''        if not user and args[1] == "bts-scheduler.service" and scenario == "recovery-clock-every":
            states.update(LoadState="loaded", ControlGroup="/legacy-sched")
''' + old)
    return shim


@pytest.mark.slow
@pytest.mark.skipif(not Path("/usr/bin/timeout").exists(), reason="needs a real timeout(1)")
def test_d3b_residual_check_is_launched_after_a_slow_proof(box):
    """K13 (round-2 finding 3): slow stops and proof queries, no clock; about 4 minutes."""
    shim = timed_shim()
    for file in (box / "shims").iterdir():
        file.write_text(file.read_text().split("\n", 1)[0] + "\n" + shim)
    scripts, _ = scripts_and_steps()
    runtime = d3b_box(box)
    (box / "head").write_text(TARGET)
    (runtime / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {TARGET} {OLD} begun - - - -\n")
    started = time.monotonic()
    result, calls = run_box_script(box, scripts[1], "recovery-clock-every", limit=300)
    elapsed = time.monotonic() - started
    assert result.returncode == 1 and elapsed < 280
    residual = [call for call in calls if call[0] == "timeout" and "pgrep" in call]
    assert residual, "the residual check was never launched"
    assert int(residual[0][3]) >= 5
    post = [call for call in calls if call[0] == "timeout" and "curl" in call and "--data-raw" in call]
    assert post and int(post[0][3]) >= 10
    assert "Scheduler stopped: unverified." in result.stdout
    assert json.loads((box / "home/projects/bts/data/.heartbeat").read_text())["state"] == "deploy_failed"
    assert len(posts(calls)) == 1
    assert " contained " in (runtime / "attempt").read_text()


def test_d3b_operator_notes():
    """K20: the restore instruction and B8a's picks directory for every saver-state command."""
    repo = WORKFLOW.parents[2]
    incident = (repo / "INCIDENT.md").read_text()
    restore = incident.split("- **Box/filesystem loss", 1)[1].split("\n- ", 1)[0]
    assert "enable + start" not in restore and "systemctl --user restart" not in restore
    hold = restore.index("first confirm the scheduler is stopped and its local boot link")
    assert hold < restore.index("install restic") < restore.index("restic restore") < restore.index("install-systemd-hetzner.sh")
    assert "activate the" in restore and "scheduler through §5's checked deploy" in restore
    b8a = next(line for line in (repo / "docs/ops/2027-season-start.md").read_text().splitlines()
               if line.startswith("| B8a |"))
    assert "Use the scheduler's configured picks directory for status, revoke, enable, fetch, dated observations and every `saver-state` invocation." in b8a
    savers = re.findall(r"`bts saver-state [^`]*`", b8a)
    assert savers and all("--season <ET year>" in c and "--picks-dir <scheduler picks dir>" in c for c in savers)
    for text in ("`Scheduler stopped: verified` reports the snapshot checks",
                 "A commit record can survive a failed containment rewrite."):
        assert text in incident


def test_d3b_prior_begun_owner_keeps_its_own_recovery(box):
    """K9 (check text): after B's unpublished begin over A's begun record, B's recovery and a later
    attempt run as separate invocations; A's record survives B."""
    scripts, _ = scripts_and_steps()
    runtime = d3b_box(box, "other-run-begun")
    first, calls = run_box_script(box, scripts[0], "active", fault="stage-begun:TERM:unpublished")
    assert "DEPLOY NOT STARTED" in first.stdout and not service_mutations(calls)
    (box / "calls").unlink()
    (box / "fault-fired").unlink()
    recovered, calls = run_box_script(box, scripts[1], "active")
    assert "a different attempt owns the box record" in recovered.stdout
    assert not service_mutations(calls) and (runtime / "cancelled.123.1").is_file()
    assert (runtime / "attempt").read_text() == D3B_PRIORS["other-run-begun"]
    (box / "calls").unlink()
    later, _ = run_box_script(box, scripts[0], "active", run_id="124")
    assert later.returncode == 0 and "CANARY PASSED" in later.stdout
    assert " 124 1 " in (runtime / "attempt").read_text()


@pytest.mark.parametrize(("scenario", "fragment"), [
    ("legacy-vanishing-job", False), ("legacy-vanishing-queryfail", False), ("active", True),
])
def test_d3b_unproved_legacy_absence_is_unverified(box, scenario, fragment):
    """K12 (check text): a pending job, a failed absence query, a dangling fragment link."""
    d3b_legacy_group(box, 0)
    if fragment:
        units = box / "etc/systemd/system"
        units.mkdir(parents=True)
        (units / "bts-scheduler.service").symlink_to(box / "missing-unit-file")
    result, _ = d3b_run_contained(box, False, scenario)
    assert result.returncode == 1
    assert "Scheduler stopped: unverified." in result.stdout


def test_d3b_failed_user_group_query_is_unverified(box):
    """M1 rule (check text): a failed pre-stop cgroup query, unlike a blank one, leaves the scope unverified."""
    scripts, _ = scripts_and_steps()
    d3b_box(box)
    result, _ = run_box_script(box, scripts[0], "user-group-fail", fault="reset:17")
    assert "Scheduler stopped: unverified." in result.stdout


@pytest.mark.parametrize("field", ["run", "attempt", "target", "old", "none"])
def test_d3b_success_needs_this_attempts_exact_commit(box, field):
    """K17 (S3, S4): a valid committed record appears while committing; only the exact one is success."""
    scripts, _ = scripts_and_steps()
    d3b_box(box)
    result, calls = run_box_script(box, scripts[0], "active", fault=f"publish-committed:TERM:forge-{field}-committed")
    assert (box / "fault-fired").exists()
    if field == "none":
        assert result.returncode == 0 and "DEPLOY FAILED" not in result.stdout
    else:
        assert result.returncode == 1
        assert "DEPLOY FAILED at interrupted" in result.stdout


@pytest.mark.parametrize("fault", ["forge-none-committed", "forge-target-committed", "forge-old-begun"])
def test_d3b_same_ids_while_beginning_contains(box, fault):
    """K17 (S8, synthetic: preflight refuses these ids under the lock): contain, never success."""
    scripts, _ = scripts_and_steps()
    d3b_box(box)
    result, _ = run_box_script(box, scripts[0], "active", fault=f"stage-begun:TERM:{fault}")
    assert (box / "fault-fired").exists()
    assert result.returncode == 1
    assert "DEPLOY FAILED at interrupted" in result.stdout


@pytest.mark.parametrize("failure", [None, "stop", "marker", "record"])
def test_d3b_recovery_reproves_stop_and_marker(box, failure):
    """K18: an earlier 2xx is carried, stop and marker are proved again, re-checked only when both hold."""
    scripts, _ = scripts_and_steps()
    runtime = d3b_box(box)
    record = f"bts_deploy_attempt_v1 123 1 {TARGET} {OLD} contained sync verified verified verified\n"
    (runtime / "attempt").write_text(record)
    scenario = {None: "active", "stop": "main-pid", "marker": "active", "record": "contain-rename-fails"}[failure]
    if failure == "marker":
        heartbeat = box / "home/projects/bts/data/.heartbeat"
        heartbeat.unlink()
        heartbeat.mkdir()
    result, calls = run_box_script(box, scripts[1], scenario)
    assert result.returncode == 1 and not posts(calls)
    assert "Failure ping received HTTP 2xx: verified." in result.stdout
    assert ("Scheduler stopped: verified." in result.stdout) == (failure != "stop")
    assert ("Failure heartbeat written: verified." in result.stdout) == (failure != "marker")
    assert ("containment re-checked" in result.stdout) == (failure in {None, "record"})
    if failure == "record":
        assert (runtime / "attempt").read_text() == record
    if failure in {None, "stop"}:
        assert json.loads((box / "home/projects/bts/data/.heartbeat").read_text())["state"] == "deploy_failed"


def test_d3b_surviving_commit_record_is_reported_as_a_record(box):
    """Appended case: a commit rename, a failed read-back and a failed contained rename."""
    scripts, _ = scripts_and_steps()
    runtime = d3b_box(box)
    deployed, _ = run_box_script(box, scripts[0], "commit-read-126-contain-fail")
    assert deployed.returncode == 1 and "Check commit-record failed (126)." in deployed.stdout
    assert " committed " in (runtime / "attempt").read_text()
    (box / "calls").unlink()
    recovered, calls = run_box_script(box, scripts[1], "commit-read-126-contain-fail")
    assert ("DEPLOY RECOVERY — this attempt's record says committed. Check the deploy log and verify "
            "current state via INCIDENT.md §4.") in recovered.stdout
    assert not service_mutations(calls)


# D3b review round 1 (Codex WRD3BREV1-80afbadd): three more cases of the class.

@pytest.mark.parametrize("recovery", [False, True], ids=["deploy", "recovery"])
@pytest.mark.parametrize("url", [
    "https://checks.invalid/configured/../different-check",
    "https://checks.invalid/a/./b",
    "https://checks.invalid/a/..",
    "https://checks.invalid/%2e%2e/other",
])
def test_d3b_r1_page_url_with_dot_segments_or_escapes_never_posts(box, recovery, url):
    """Finding 1: curl would rewrite the path and a 2xx from another check would read as verified."""
    scripts, _ = scripts_and_steps()
    runtime = d3b_box(box)
    (box / "home/projects/bts/.env").write_text(f"BTS_SCHEDULER_HEARTBEAT_PING_URL={url}\n")
    if recovery:
        (box / "head").write_text(TARGET)
        (runtime / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {TARGET} {OLD} begun - - - -\n")
        result, calls = run_box_script(box, scripts[1], "active")
    else:
        result, calls = run_box_script(box, scripts[0], "active", fault="reset:17")
    assert result.returncode == 1 and not posts(calls)
    assert "Failure ping received HTTP 2xx: unverified." in result.stdout
    assert (runtime / "attempt").read_text().endswith(" unverified\n")


def test_d3b_r1_failure_post_sends_the_path_as_written(box):
    """Finding 1: every failure POST carries --path-as-is (and still --globoff)."""
    scripts, _ = scripts_and_steps()
    d3b_box(box)
    result, calls = run_box_script(box, scripts[0], "active", fault="reset:17")
    sent = posts(calls)
    assert len(sent) == 1 and "--path-as-is" in sent[0] and "--globoff" in sent[0]
    assert sent[0][-1] == "https://checks.invalid/d3b/fail"
    for script in scripts:
        assert script.count("--path-as-is") == 1


@pytest.mark.parametrize("recovery", [False, True], ids=["deploy", "recovery"])
@pytest.mark.parametrize(("populated", "verified"), [(1, False), (0, True)])
def test_d3b_r1_a_cgroup_reported_by_an_absent_unit_is_checked(box, recovery, populated, verified):
    """Finding 2a: both absence samples read the cgroup; a reported path is checked, never skipped."""
    group = box / "sys/fs/cgroup/legacy-group"
    group.mkdir()
    (group / "cgroup.events").write_text(f"populated {populated}\n")
    result, calls = d3b_run_contained(box, recovery, "legacy-ghost-group")
    assert result.returncode == 1
    assert any(call[0] == "python3" and call[3:4] == ["cgroup"] and call[-1] == "/legacy-group" for call in calls)
    assert ("Scheduler stopped: verified." in result.stdout) == verified
    assert ("Scheduler stopped: unverified." in result.stdout) == (not verified)


@pytest.mark.parametrize("recovery", [False, True], ids=["deploy", "recovery"])
def test_d3b_r1_not_found_with_a_fragment_is_not_a_loaded_proof(box, recovery):
    """Finding 2b: after its stop the unit reads not-found while its fragment stays; neither proof applies."""
    for name in ("legacy-pre", "legacy-post"):
        group = box / "sys/fs/cgroup" / name
        group.mkdir()
        (group / "cgroup.events").write_text("populated 0\n")
    units = box / "etc/systemd/system"
    units.mkdir(parents=True)
    (units / "bts-scheduler.service").write_text("[Service]\nType=oneshot\nExecStart=/bin/false\n")
    result, calls = d3b_run_contained(box, recovery, "legacy-fragment-present")
    assert result.returncode == 1
    assert ["systemctl", "stop", "bts-scheduler.service"] in calls
    assert "Scheduler stopped: unverified." in result.stdout


def test_d3b_r1_final_lines_are_one_bounded_print():
    """Finding 3: the result lines and the re-checked line leave in one print with a one-second grace."""
    for script in scripts_and_steps()[0]:
        proof = shell_function(script, "prove")
        assert proof.count("timeout -k 1 1 /bin/bash") == 1
        assert proof.count("printf") == 1
        assert proof.index("containment re-checked") < proof.index("timeout -k 1 1 /bin/bash")
        assert "prove recheck" not in script and "prove report" in shell_function(script, "contain")


def test_d3b_r1_repeat_recovery_prints_recheck_with_the_results(box):
    """Finding 3 control: the re-checked line still appears after a fully verified repeat recovery."""
    scripts, _ = scripts_and_steps()
    runtime = d3b_box(box)
    (runtime / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {TARGET} {OLD} contained sync verified verified unverified\n")
    result, calls = run_box_script(box, scripts[1], "active")
    lines = result.stdout.splitlines()
    assert lines[-5:] == ["Scheduler stopped: verified.", "Failure heartbeat written: verified.",
                          "Failure ping received HTTP 2xx: verified.", "Recovery procedure: INCIDENT.md §4.",
                          "DEPLOY RECOVERY — containment re-checked."]
    assert len(posts(calls)) == 1


# D3b review round 1 (Carbon WRD3BCARB1-0133d0ea): B2 and the surviving mutants N1 own1 to own5.

def d3b_unloading_box(box, populated):
    group = box / "sys/fs/cgroup/legacy-orphan"
    group.mkdir()
    (group / "cgroup.events").write_text(f"populated {populated}\n")
    units = box / "etc/systemd/system"
    units.mkdir(parents=True)
    for unit in ("bts-scheduler.service", "bts-dashboard.service"):
        (units / unit).write_text("[Service]\n")
    return d3b_box(box)


@pytest.mark.parametrize("populated", [1, 0])
def test_d3b_r1_phase_two_legacy_cleanup_marks_the_attempt(box, populated):
    """Carbon B2: phase 2 stops, deletes and unloads a loaded legacy scheduler, then a later step fails.
    Neither the deploy's containment nor this attempt's recovery may call the legacy scope absent."""
    scripts, _ = scripts_and_steps()
    runtime = d3b_unloading_box(box, populated)
    deployed, calls = run_box_script(box, scripts[0], "legacy-unloads", fault="restart:17")
    assert deployed.returncode == 1
    assert ["systemctl", "daemon-reload"] in calls
    assert (runtime / "legacy.123.1").is_file()
    assert "Scheduler stopped: unverified." in deployed.stdout
    (box / "calls").unlink()
    recovered, _ = run_box_script(box, scripts[1], "legacy-unloads")
    assert "Scheduler stopped: unverified." in recovered.stdout
    assert "containment re-checked" not in recovered.stdout


def test_d3b_r1_mark_failure_contains_before_the_legacy_stop(box):
    """Carbon B2: the mark is written and checked before any legacy mutation."""
    scripts, _ = scripts_and_steps()
    d3b_unloading_box(box, 0)
    result, calls = run_box_script(box, scripts[0], "legacy-unloads-markfail")
    assert result.returncode == 1
    assert "DEPLOY FAILED at legacy-units" in result.stdout
    assert ["systemctl", "daemon-reload"] not in calls


def test_d3b_r1_a_passing_deploy_with_legacy_units_still_passes(box):
    """Carbon B2 control: the mark changes nothing on the success path."""
    scripts, _ = scripts_and_steps()
    runtime = d3b_unloading_box(box, 0)
    result, _ = run_box_script(box, scripts[0], "legacy-unloads")
    assert result.returncode == 0 and "CANARY PASSED" in result.stdout
    assert (runtime / "legacy.123.1").is_file()


@pytest.mark.parametrize("recovery", [False, True], ids=["deploy", "recovery"])
@pytest.mark.parametrize("scenario", [
    "legacy-groupfail-post",      # own1: failed pre-stop query, empty post-stop path
    "legacy-late-group",          # own2: blank first sample, populated path at the proof
    "legacy-vanishing-stopfail",  # own3: the legacy stop failed
    "legacy-still-enabled",       # own4: loaded and stopped with its boot link enabled
])
def test_d3b_r1_each_legacy_stop_clause_is_required(box, recovery, scenario):
    """Carbon N1: each clause of rule 2 decides the result on its own."""
    for name, populated in (("legacy-post", 0), ("legacy-late", 1), ("legacy-orphan", 0), ("la", 0)):
        group = box / "sys/fs/cgroup" / name
        group.mkdir()
        (group / "cgroup.events").write_text(f"populated {populated}\n")
    result, _ = d3b_run_contained(box, recovery, scenario)
    assert result.returncode == 1
    assert "Scheduler stopped: unverified." in result.stdout


def test_d3b_r1_rounding_allowance_is_in_both_scripts():
    """Carbon N1 own5: the one-second rounding allowance inside containment and the signal read."""
    for script in scripts_and_steps()[0]:
        assert shell_function(script, "run").count("[[ -z ${budget_start:-} ]] || available=$((available - 1))") == 1
