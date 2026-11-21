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
fault = os.environ.get("FAULT", "").split(":")
if at and fault[0] == at and not (root / "fault-fired").exists():
    (root / "fault-fired").touch()
    if fault[1] in {"17", "124"}:
        raise SystemExit(int(fault[1]))
    if fault[1] == "mismatch":
        print("c" * 40)
        raise SystemExit(0)
    if fault[1] == "hold":
        end = time.monotonic() + 10
        while not (root / "release").exists() and time.monotonic() < end:
            time.sleep(0.01)
    else:
        os.kill(int((root / "owner.pid").read_text()), getattr(signal, "SIG" + fault[1]))
        if len(fault) > 2 and fault[2] == "unpublished":
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
    try:
        fcntl.flock(int(args[-1]), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(1)
elif command == "python3":
    op = args[2] if len(args) > 2 else None
    if op in {"lockfile", "mark"}:
        import stat
        path = Path(args[3])
        try:
            fd = os.open(path, os.O_APPEND | os.O_CREAT | os.O_NOFOLLOW | os.O_NONBLOCK, 0o600)
            with os.fdopen(fd, "ab") as f:
                metadata = os.fstat(f.fileno())
                if not stat.S_ISREG(metadata.st_mode) or stat.S_IMODE(metadata.st_mode) != 0o600:
                    raise SystemExit(1)
                if op == "mark" and (metadata.st_size or scenario == "mark-error"):
                    raise SystemExit(1)
        except OSError:
            raise SystemExit(1)
    else:
        code = args[1]
        sys.argv = ["-c", *args[2:]]
        if op == "clock" and scenario == "canary-budget" and (root / "advance-clock").exists():
            print(time.monotonic_ns() // 1_000_000 + 1_000_000)
            raise SystemExit(0)
        if op == "marker" and scenario == "marker-unwritable":
            raise SystemExit(17)
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
elif command == "mv":
    if scenario == "commit-error" and at == "publish-committed":
        raise SystemExit(17)
    os.replace(args[-2], args[-1])
    if scenario == "published-with-error" and at == "publish-begun":
        raise SystemExit(17)
elif command == "git":
    branch = root / "head"
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
        active = "active" if user and not (root / "stopped").exists() else "inactive"
        states = dict(Id=args[1], LoadState="loaded" if user or scenario in {"legacy-present", "legacy-survivor"} else "not-found", UnitFileState="enabled",
                      ActiveState=active, SubState="running", Result="success", ExecMainStatus="0",
                      MainPID="0", ControlGroup="")
        if os.environ["SCENARIO"] == "auto-restart" and "bts-scheduler" in args:
            states.update(ActiveState="activating", SubState="auto-restart")
        if "ControlGroup" in args and user and scenario == "descendant-cgroup":
            states["ControlGroup"] = "/sched"
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
    if user and args[0] == "list-jobs" and scenario == "pending-job":
        print("17 bts-scheduler.service start waiting")
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


def run_box_script(box, script, scenario, *, fault="", run_id="123", attempt="1", target=TARGET, closed_output=False):
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
            stdout, stderr = process.communicate(timeout=15)
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
    assert "this attempt has no phase-2 record on this boot" in result.stdout
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
        assert "this attempt committed on the box" in result.stdout
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
        ("commit-error", "commit-record", "1"),
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
PHASE_FOUR = {"stamp-stage": "stamp-stage", "enable": "enable", "check-enable": "enable", "stamp": "stamp"}


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
    result, calls = run_box_script(box, scripts[0], scenario, fault=f"{command}:{signal_name}")
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
    assert "this attempt has no phase-2 record on this boot" in result.stdout


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
