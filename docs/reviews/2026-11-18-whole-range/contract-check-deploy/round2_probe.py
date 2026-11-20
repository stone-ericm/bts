"""Offline draft-2 fault model and real cross-revision cron writer replay."""

import io
import json
import os
import shlex
import signal
import subprocess
import sys
import tarfile
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
OLD = "619abda7"
TARGET = "fef05b5b"

# This models the specified ordering, not an implementation or systemd itself.
SHELL = """
fail() {
    trap '' HUP INT TERM
    printf 'DEPLOY FAILED at %s\n' "$1"
    systemctl --user disable bts-scheduler
    systemctl --user stop bts-scheduler
    systemctl --user is-active bts-scheduler
    systemctl --user is-enabled bts-scheduler
    fixture mark
    exit 1
}
trap 'fail interrupted' HUP INT TERM
fixture before-disable
systemctl --user disable bts-scheduler || fail disable
git reset --hard "$TARGET_SHA" || fail reset
uv sync --locked --extra model || fail sync
fixture legacy-cleanup || fail legacy-cleanup
systemctl --user restart bts-scheduler bts-dashboard || fail restart
fixture head || fail head
sleep 30 || fail sleep
curl dashboard || fail canary
systemctl --user enable bts-scheduler || fail enable
stamp || { printf 'WARNING: deploy succeeded but data/.last_deploy_iso was not written\n'; exit 1; }
printf 'CANARY PASSED — %s is live and healthy\n' "$TARGET_SHA"
"""


def load(path):
    return json.loads(path.read_text())


def dump(path, value):
    path.write_text(json.dumps(value))


def shim():
    name, *args = sys.argv[2:]
    name = Path(name).name
    root = Path(os.environ["DRAFT2_CASE"])
    path = root / "state.json"
    state = load(path)
    state["calls"].append([name, *args])
    event = name
    status = 0
    if name == "fixture":
        event = args[0]
        if event == "boot":
            state["active"] = (state["unit_loadable"] and
                               bool(set(state["links"]) | set(state["persistent_edges"])))
            state["legacy_active"] = state["legacy_enabled"]
        if event == "legacy-cleanup" and state["legacy_enabled"]:
            if "ignore-legacy-error" not in state["faults"]:
                state["legacy_enabled"] = state["legacy_active"] = False
        if event == "mark":
            state["heartbeat"] = "deploy_failed"
    elif name == "systemctl":
        event = args[1]
        if not state["manager"]:
            status = 1
        elif event == "disable":
            state["links"] = []
            state["unit_loadable"] = not state["outside_unit"]
            status = int("disable-reload-failed" in state["faults"])
        elif event == "restart":
            status = int(not state["unit_loadable"])
            state["active"] = state["unit_loadable"]
            if state["active"]:
                state["heartbeat"] = "running"
        elif event == "enable":
            state["links"] = ["default.target.wants"]
            status = int("enable-reload-failed" in state["faults"])
            if "--now" in args:
                status = int("gate-refuses" in state["faults"])
                state["active"] = not status
        elif event == "stop":
            if "stop-failed" in state["faults"]:
                status = 1
            else:
                state["active"] = False
        elif event == "is-active":
            print("active" if state["active"] else "inactive")
            status = 0 if state["active"] else 3
        elif event == "is-enabled":
            print("enabled" if state["links"] else "disabled")
            status = 0 if state["links"] else 1
        else:
            raise AssertionError(args)
    elif name == "git":
        assert args == ["reset", "--hard", TARGET], args
        event = "reset"
        status = int("reset-failed" in state["faults"])
        if not status:
            state["head"] = TARGET
    elif name == "uv":
        event = "sync"
        status = int("sync-failed" in state["faults"])
        if "manager-lost" in state["faults"]:
            state["manager"] = False
    elif name == "curl":
        event = "canary"
        status = int("canary-failed" in state["faults"])
    elif name == "stamp":
        status = int("stamp-failed" in state["faults"])
        if not status:
            state["stamp"] = "new"
    elif name != "sleep":
        raise AssertionError((name, args))
    killed = state["interrupt"] and event == state["interrupt"][0]
    if killed:
        signum = state.pop("interrupt")[1]
        state["interrupt"] = None
    dump(path, state)
    if killed:
        os.kill(os.getppid(), signum)
    return status


def run(argv, *, cwd=None, env=None, timeout=12):
    return subprocess.run(argv, cwd=cwd, env=env, capture_output=True, timeout=timeout)


def systemd_cases(work):
    bin_dir = work / "bin"
    bin_dir.mkdir()
    interpreter = shlex.quote(sys.executable)
    probe = shlex.quote(str(Path(__file__).resolve()))
    for name in ("fixture", "systemctl", "git", "uv", "sleep", "curl", "stamp"):
        path = bin_dir / name
        path.write_text(f'#!/bin/sh\nexec {interpreter} {probe} --shim "$0" "$@"\n')
        path.chmod(0o700)

    def evaluate(name, shell=SHELL, **overrides):
        current = work / name
        current.mkdir()
        state_path = current / "state.json"
        default = dict(active=True, manager=True, links=["default.target.wants"],
                       persistent_edges=[], unit_loadable=True, outside_unit=False,
                       legacy_enabled=False, legacy_active=False, stamp="old", head=OLD,
                       heartbeat="sleeping", faults=[], interrupt=None, calls=[])
        dump(state_path, {**default, **overrides})
        env = {"PATH": f"{bin_dir}:/usr/bin:/bin", "DRAFT2_CASE": str(current), "HOME": str(current),
               "TARGET_SHA": TARGET, "PYTHONDONTWRITEBYTECODE": "1"}
        finished = run(["/bin/bash", "-c", shell], cwd=current, env=env)
        immediate = load(state_path)
        boot = run([str(bin_dir / "fixture"), "boot"], cwd=current, env=env)
        assert boot.returncode == 0, boot.stderr
        rebooted = load(state_path)
        return {"exit": finished.returncode, "stdout": finished.stdout.decode().splitlines(),
                "stderr": finished.stderr.decode().splitlines(), "before_reboot": immediate,
                "boot_user": rebooted["active"], "boot_legacy": rebooted["legacy_active"]}

    scenarios = {
        "pass": evaluate("pass"),
        "from_failed": evaluate("from_failed", active=False, links=[], heartbeat="deploy_failed"),
        "known_extra_wants_link": evaluate("known_extra_wants_link", faults=["canary-failed"],
                                           links=["default.target.wants", "another.target.wants"]),
        "direct_wants": evaluate("direct_wants", faults=["canary-failed"],
                                persistent_edges=["boot-reachable unit Wants="]),
        "global_enable": evaluate("global_enable", faults=["canary-failed"],
                                  persistent_edges=["global user unit enable link"]),
        "linked_only": evaluate("linked_only", outside_unit=True),
        "no_manager": evaluate("no_manager", manager=False),
        "manager_lost": evaluate("manager_lost", faults=["manager-lost", "sync-failed"]),
        "disable_reload_failed": evaluate("disable_reload_failed", faults=["disable-reload-failed"]),
        "legacy_early": evaluate("legacy_early", faults=["sync-failed"], legacy_enabled=True,
                                 legacy_active=True),
        "legacy_ignored": evaluate("legacy_ignored", faults=["ignore-legacy-error", "canary-failed"],
                                   legacy_enabled=True, legacy_active=True),
        "stop_failed": evaluate("stop_failed", faults=["canary-failed", "stop-failed"]),
        "enable_failed": evaluate("enable_failed", faults=["enable-reload-failed"]),
        "stamp_failed": evaluate("stamp_failed", faults=["stamp-failed"]),
        "manual_refusal": evaluate("manual_refusal", "systemctl --user enable --now bts-scheduler",
                                   active=False, links=[], faults=["gate-refuses"]),
    }
    for step in ("before-disable", "disable", "reset", "sync", "legacy-cleanup", "restart", "head", "sleep",
                 "canary", "enable", "stamp"):
        name = "hard-loss-after-" + step
        scenarios[name] = evaluate(name, interrupt=[step, signal.SIGKILL])
    for sig in (signal.SIGHUP, signal.SIGTERM):
        for step in ("reset", "sync", "restart", "sleep"):
            name = f"{sig.name}-in-{step}"
            scenarios[name] = evaluate(name, interrupt=[step, sig])
            assert scenarios[name]["stdout"] == ["DEPLOY FAILED at interrupted"]
            assert not scenarios[name]["before_reboot"]["active"]
    assert scenarios["pass"]["boot_user"] and scenarios["from_failed"]["boot_user"]
    assert not scenarios["known_extra_wants_link"]["boot_user"]
    assert scenarios["direct_wants"]["boot_user"] and scenarios["global_enable"]["boot_user"]
    assert scenarios["linked_only"]["exit"] == 1 and not scenarios["linked_only"]["boot_user"]
    assert scenarios["no_manager"]["before_reboot"]["head"] == OLD
    assert scenarios["legacy_early"]["boot_legacy"] and scenarios["legacy_ignored"]["boot_legacy"]
    assert scenarios["hard-loss-after-enable"]["boot_user"]
    assert scenarios["hard-loss-after-enable"]["before_reboot"]["stamp"] == "old"
    assert scenarios["hard-loss-after-restart"]["before_reboot"]["active"]
    assert scenarios["stamp_failed"]["exit"] == 1 and scenarios["stamp_failed"]["boot_user"]
    assert scenarios["manual_refusal"]["exit"] == 1 and scenarios["manual_refusal"]["boot_user"]
    return scenarios


def cron_child(action, roots):
    sys.path.insert(0, str(REPO / "docs/reviews/2026-11-18-whole-range/slice-2-codex"))
    import day_probe
    import deploy_probe
    import offline_fixtures

    results = {}
    for path in roots:
        root = Path(path)
        picks = root / "committed" / "data" / "picks"
        if action == "seed":
            age = 9 if root.name == "failure-b" else 8
            offline_fixtures.seed(picks, "2027-07-16", age=age, best=17 if age == 9 else 18)
        if action == "fetch":
            results[root.name] = deploy_probe.rollforward_fetch(root, inspect=False)
        else:
            from bts import contest_state

            world = day_probe.World("2027-07-16", "normal")
            with offline_fixtures.installed(world, root / "committed"):
                try:
                    contest_state.load_decision_streak_state(picks, require_contest_state=True)
                    status = "accepted"
                except contest_state.ContestStateError as exc:
                    status = str(exc)
            results[root.name] = {"observation": load(picks / "account_state" / "contest_streak.json"),
                                  "acceptance": status}
    print(json.dumps(results))


def replay_cron_writer(work):
    archive = run(["/usr/bin/git", "archive", OLD, "src/bts"], cwd=REPO)
    assert archive.returncode == 0, archive.stderr
    old = work / "snapshot"
    old.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as files:
        files.extractall(old, filter="data")
    cases = [work / name for name in ("failure-a", "failure-b", "success")]

    def trial(source, action, selected):
        env = {**os.environ, "PYTHONPATH": str(source / "src"), "PYTHONDONTWRITEBYTECODE": "1"}
        result = run([sys.executable, __file__, "--cron", action, *map(str, selected)], cwd=work, env=env)
        assert result.returncode == 0, result.stderr.decode()
        return json.loads(result.stdout)

    initial = trial(REPO, "seed", cases)
    assert all(record["acceptance"] == "accepted" for record in initial.values()), initial
    after_failed_reset = trial(old, "fetch", cases[:2])
    after_success = trial(REPO, "fetch", cases[2:])
    fixed_code_no_fetch_yet = trial(REPO, "inspect", cases)
    assert initial["failure-a"]["observation"] != initial["failure-b"]["observation"]
    assert after_failed_reset["failure-a"]["after"] == after_failed_reset["failure-b"]["after"]
    assert after_failed_reset["failure-a"]["after"]["schema_version"] == "bts_contest_streak_auto_v1"
    assert fixed_code_no_fetch_yet["success"]["acceptance"] == "accepted"
    assert "schema_version" in fixed_code_no_fetch_yet["failure-a"]["acceptance"]
    return {"initial": initial, "failed_reset_cron_old_code": after_failed_reset,
            "successful_deploy_cron_correct_code": after_success,
            "correct_code_after_both_timelines": fixed_code_no_fetch_yet}


if __name__ == "__main__":
    if sys.argv[1:2] == ["--shim"]:
        sys.exit(shim())
    if sys.argv[1:2] == ["--cron"]:
        cron_child(sys.argv[2], sys.argv[3:])
        sys.exit(0)
    with TemporaryDirectory(prefix="round2-probe-", dir=HERE) as temp:
        work = Path(temp)
        print(json.dumps({"systemd_model": systemd_cases(work), "cron_writer": replay_cron_writer(work)}, indent=2))
