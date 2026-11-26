"""Offline base replay of the deploy shell, rendered cron, and heartbeat readers."""

import importlib.util
import json
import os
import re
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from textwrap import dedent


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = "619abda7"
TARGET = "acbd1234"
APP = "/home/bts/projects/bts"
LEGACY_UNIT = "/etc/systemd/system/bts-scheduler.service"
CASES = {
    "pass", "clean_exit", "cd", "old_sha", "config", "fetch", "reset", "sync",
    "legacy_reload", "legacy_disable", "restart", "new_sha", "sleep", "id",
    "crash", "dashboard_inactive", "http_503", "http_timeout", "http_error_200",
    "scheduler_query_error", "show_query_error", "no_tailnet", "stamp",
    "rollback_reset", "hup", "term",
}


def read_local(path):
    if not path.is_file():
        raise AssertionError(f"Expected a regular local file: {path}")
    return path.read_text()


def extract_workflow():
    import yaml

    source = read_local(REPO / ".github/workflows/deploy.yml")
    lines = source.splitlines(keepends=True)
    headers = [index for index, line in enumerate(lines) if re.fullmatch(r" +script: \|\n", line)]
    assert len(headers) == 1, headers
    start = headers[0]
    parent_indent = len(lines[start]) - len(lines[start].lstrip())
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.strip() and len(line) - len(line.lstrip()) <= parent_indent:
            break
        end += 1
    payload = dedent("".join(lines[start + 1:end]))
    payload = payload.rstrip("\n") + "\n"
    step_payloads = [step["with"]["script"] for step in yaml.safe_load(source)["jobs"]["deploy"]["steps"]
                     if "script" in step.get("with", {})]
    assert step_payloads == [payload]
    return payload


def shim():
    executable, *args = sys.argv[2:]
    executable = Path(executable).name
    root = Path(os.environ["PROBE_ROOT"])
    case = os.environ["PROBE_CASE"]
    with (root / "calls.jsonl").open("a") as calls:
        print(json.dumps([executable, *args]), file=calls)
    if case in {"enter", "research"}:
        if executable == "uv" and args[:3] == ["run", "bts", "entry-intent"]:
            print(case)
            return 0
        if executable == "crontab" and args == ["-l"]:
            return 0
        raise AssertionError((executable, args))

    sha_path = root / "sha"
    state_path = root / "scheduler"
    current = read_local(sha_path)
    if executable == "id":
        if case == "id":
            return 3
        print("1000")
        return 0
    if executable == "sleep":
        if args == ["30"] and case in {"hup", "term"}:
            os.kill(os.getppid(), {"hup": signal.SIGHUP, "term": signal.SIGTERM}[case])
        return 5 if case == "sleep" else 0
    if executable == "curl":
        if case in {"http_timeout", "http_error_200"}:
            print("200" if case == "http_error_200" else "000", end="")
            return 28
        print("503" if case in {"http_503", "rollback_reset"} else "200", end="")
        return 0
    if executable == "systemctl" and args == ["daemon-reload"]:
        return 1 if case == "legacy_reload" else 0
    if executable == "systemctl" and args[0] == "disable":
        return 1 if case == "legacy_disable" else 0
    if executable in {"rm", "pkill"}:
        return 0
    if executable == "sudo" and "git" in args:
        command = args[args.index("git") + 1]
        if command == "rev-parse":
            if case in {"old_sha", "new_sha"} and (case == "old_sha") == ("--short" not in args):
                return 6
            print(current[:7] if "--short" in args else current)
        elif command == "config":
            (root / "global-config-touched").touch()
            return int(case == "config")
        elif command == "fetch":
            if case == "fetch":
                (root / "fetched-metadata-touched").touch()
                return 1
        elif command == "reset":
            if case == "reset":
                (root / "partial-tree").touch()
                return 1
            if case == "rollback_reset" and args[-1] == OLD:
                return 1
            sha_path.write_text(args[-1])
        else:
            raise AssertionError(args)
        return 0
    if executable == "sudo" and "sync" in args:
        return int(case == "sync")
    if executable == "sudo" and "tailscale" in args:
        if case == "no_tailnet":
            return 1
        print("100.64.0.8")
        return 0
    if executable == "sudo" and "systemctl" in args:
        verb = args[args.index("systemctl") + 2]
        if verb == "restart":
            if case == "restart":
                state_path.write_text("candidate-refused")
                return 2
            state_path.write_text(current + "-running")
            return 0
        if verb == "is-active":
            if "--quiet" in args:
                return int(case == "dashboard_inactive" and "bts-scheduler" not in args)
            if case in {"clean_exit", "crash", "show_query_error"}:
                print("activating")
                return 3
            print("active")
            return 1 if case == "scheduler_query_error" else 0
        if verb == "show":
            prop = args[args.index("-p") + 1]
            print({"ActiveState": "activating", "SubState": "auto-restart",
                   "Result": "exit-code" if case == "crash" else "success",
                   "ExecMainStatus": "2" if case == "crash" else "0"}[prop])
            return int(case == "show_query_error")
    if executable == "sudo" and "bash" in args and ".last_deploy_iso" in args[-1]:
        if case == "stamp":
            return 1
        (root / "data" / ".last_deploy_iso").write_text("2027-01-01T00:00:00Z\n")
        return 0
    raise AssertionError((executable, args))


def make_shims(directory, executables):
    directory.mkdir(parents=True)
    interpreter = shlex.quote(sys.executable)
    probe = shlex.quote(str(Path(__file__).resolve()))
    for executable in executables:
        path = directory / executable
        path.write_text(f'#!/bin/sh\nexec {interpreter} {probe} --shim "$0" "$@"\n')
        path.chmod(0o700)


def bash(script, root, case, path, *args):
    env = {"HOME": str(root), "PATH": f"{path}:/usr/bin:/bin", "TARGET_SHA": TARGET,
           "PROBE_ROOT": str(root), "PROBE_CASE": case, "LC_ALL": "C",
           "HEALTHCHECKS_PING_URL": "https://box.invalid/ping", "PYTHONDONTWRITEBYTECODE": "1"}
    return subprocess.run(["/bin/bash", *script, *args], cwd=root, env=env,
                          capture_output=True, text=True, timeout=5)


def workflow(temp):
    payload = extract_workflow()
    # Supply a fake cwd and legacy-unit predicate, without rewriting the YAML body.
    prelude = f'''cd() {{
    builtin [ "$1" = {APP} ] || return 97
    builtin [ "$PROBE_CASE" != cd ] || return 1
    builtin cd "$PROBE_ROOT"
}}
function [ () {{
    if builtin [ "$1" = -f ] && builtin [ "$2" = {LEGACY_UNIT} ]; then
        case "$PROBE_CASE" in legacy_*) return 0 ;; *) return 1 ;; esac
    fi
    builtin [ "$@"
}}
'''
    root_bin = temp / "bin"
    make_shims(root_bin, ("sudo", "systemctl", "curl", "sleep", "id", "rm", "pkill"))
    result = {}
    for case in sorted(CASES):
        root = temp / case
        (root / "data").mkdir(parents=True)
        (root / "calls.jsonl").touch()
        (root / "sha").write_text(OLD)
        (root / "scheduler").write_text(OLD + "-running")
        finished = bash(["-c", prelude + payload], root, case, root_bin)
        result[case] = {
            "exit": finished.returncode,
            "sha": read_local(root / "sha"),
            "scheduler_model": read_local(root / "scheduler"),
            "stamp": (root / "data" / ".last_deploy_iso").exists(),
            "side_effects": sorted(p.name for p in root.glob("*-touched")) +
                            ["partial-tree"] * (root / "partial-tree").exists(),
            "stdout": finished.stdout.splitlines(),
            "stderr": finished.stderr.splitlines(),
            "calls": [json.loads(line) for line in read_local(root / "calls.jsonl").splitlines()],
        }
    for case, status in (("sync", 1), ("restart", 2)):
        assert result[case]["exit"] == status and result[case]["sha"] == TARGET, result[case]
        assert not any(call[0] in {"sleep", "curl"} for call in result[case]["calls"]), result[case]
    assert result["http_503"]["sha"] == OLD and result["http_503"]["exit"] == 1
    assert result["pass"]["exit"] == result["clean_exit"]["exit"] == 0
    assert result["hup"]["exit"] == -signal.SIGHUP and result["term"]["exit"] == -signal.SIGTERM
    return result


def cron(temp):
    result = {}
    for intent, count in (("enter", 19), ("research", 18)):
        root = temp / intent
        (root / "projects" / "bts").mkdir(parents=True)
        make_shims(root / ".local" / "bin", ("uv", "crontab"))
        finished = bash([str(REPO / "scripts" / "cron-setup-hetzner.sh")], root, intent,
                        root / ".local" / "bin", "show")
        lines = [line for line in finished.stdout.splitlines() if line.endswith(" # BTS-HETZNER")]
        assert finished.returncode == 0 and len(lines) == count, (finished, lines)
        result[intent] = [line.replace(str(root), "$HOME") for line in lines]
    return result


def heartbeat(temp):
    from bts.web import health_check

    checker_path = REPO / "scripts" / "check_heartbeat.py"
    spec = importlib.util.spec_from_file_location("heartbeat_checker", checker_path)
    checker = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(checker)
    from datetime import datetime, timezone

    hb = temp / ".heartbeat"
    hb.write_text(json.dumps({"state": "deploy_failed", "timestamp": datetime.now(timezone.utc).isoformat(),
                              "step": "sync", "target_sha": TARGET[:7], "old_sha": OLD[:7]}))
    monitor = checker.is_stale(hb)
    dashboard = health_check(hb)
    assert monitor == (True, "unknown state: deploy_failed") and dashboard[0] == 200
    hb.unlink()
    hb.mkdir()
    directory = checker.is_stale(hb)
    hb.rmdir()
    os.mkfifo(hb)
    try:
        subprocess.run([sys.executable, str(checker_path), "--heartbeat-path", str(hb), "--no-churn"],
                       capture_output=True, text=True, timeout=0.5)
    except subprocess.TimeoutExpired:
        fifo = "blocked until test timeout"
    else:
        raise AssertionError("FIFO read did not block on base")
    return {"deploy_failed_monitor": monitor, "deploy_failed_dashboard": dashboard,
            "directory_monitor": directory, "fifo_monitor": fifo}


if __name__ == "__main__":
    if sys.argv[1:] and sys.argv[1] == "--shim":
        sys.exit(shim())
    with TemporaryDirectory(prefix="contract-probe-", dir=HERE) as workspace:
        temp = Path(workspace)
        results = {"workflow": workflow(temp), "cron": cron(temp), "heartbeat": heartbeat(temp)}
        print(json.dumps(results, indent=2))
