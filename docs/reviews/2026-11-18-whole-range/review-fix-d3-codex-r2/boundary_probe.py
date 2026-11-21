"""Try publication, legacy proof, numeric read-back and recovery lock boundaries."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from harness import DEPLOY, HERE, RECOVER, invoke, make_box, posts, replace_one, source_at, tests


def initialize(box):
    runtime = box / "run/bts-deploy"
    runtime.mkdir(mode=0o700, parents=True)
    (box / "enabled").write_text("enabled")
    (box / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/literal\n")
    return runtime


def publication():
    earlier, _ = tests.scripts_and_steps(source_at("d3d05abd", ".github/workflows/deploy.yml"))[0]
    prior = f"bts_deploy_attempt_v1 101 1 {tests.OLD} {'c' * 40} committed - - - -\n"
    for label, script, initial, fault in [
        ("r1-existing-commit-stage-unpublished", earlier, prior, "stage-begun:TERM:unpublished"),
        ("r2-existing-commit-stage-unpublished", DEPLOY, prior, "stage-begun:TERM:unpublished"),
        ("r2-existing-commit-rename-unpublished", DEPLOY, prior, "publish-begun:TERM:unpublished"),
        ("r2-no-prior-record-stage-unpublished", DEPLOY, None, "stage-begun:TERM:unpublished"),
        ("r2-actual-begin-published-signal", DEPLOY, prior, "publish-begun:TERM"),
    ]:
        with TemporaryDirectory(prefix="publication-", dir=HERE) as name:
            box = make_box(Path(name), tests.SHIM)
            runtime = initialize(box)
            if initial:
                (runtime / "attempt").write_text(initial)
            result = invoke(box, label, script, fault=fault)
            should_stop = label in {"r2-existing-commit-stage-unpublished", "r2-existing-commit-rename-unpublished",
                                    "r2-actual-begin-published-signal"}
            assert result["rc"] == 1 and result["stopped"] == should_stop
            assert result["head"] == tests.OLD and not any(call[:2] == ["git", "reset"] for call in result["calls"])
            assert result["enabled"] == ("disabled" if should_stop else "enabled")
            assert bool(posts(result)) == should_stop
            if should_stop:
                assert "contained interrupted" in result["attempt"] and '"state": "deploy_failed"' in result["heartbeat"]
            else:
                assert result["attempt"] == initial and any("DEPLOY NOT STARTED" in line for line in result["stdout"])


def legacy():
    old = '        if "ControlGroup" in args and user and scenario == "main-pid":'
    new = '''        if not user and args[1] == "bts-scheduler.service" and scenario.startswith("legacy-loaded-"):
            if not (root / "legacy-stopped").exists():
                states.update(LoadState="loaded", ActiveState="active", MainPID="1234", ControlGroup="/legacy-orphan")
            elif scenario == "legacy-loaded-still-visible":
                states.update(LoadState="loaded", ControlGroup="")
            with (root / "calls").open("a") as log:
                print(json.dumps(["probe", states["LoadState"], states["ControlGroup"]]), file=log)
'''
    shim = replace_one(tests.SHIM, old, new + old)
    for scenario, populated, verified in [
        ("legacy-loaded-then-absent", 1, True),
        ("legacy-loaded-still-visible", 1, False),
        ("legacy-loaded-then-absent", 0, True),
    ]:
        with TemporaryDirectory(prefix="legacy-", dir=HERE) as name:
            box = make_box(Path(name), shim)
            runtime = initialize(box)
            (box / "head").write_text(tests.TARGET)
            (runtime / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n")
            group = box / "sys/fs/cgroup/legacy-orphan"
            group.mkdir()
            (group / "cgroup.events").write_text(f"populated {populated}\n")
            (group / "cgroup.procs").touch()
            result = invoke(box, f"{scenario}-{populated}", RECOVER, scenario)
            assert result["rc"] == 1 and result["stopped"]
            assert ["systemctl", "stop", "bts-scheduler.service"] in result["calls"]
            assert ["probe", "loaded", "/legacy-orphan"] in result["calls"]
            assert ("Scheduler stopped: verified." in result["stdout"]) == verified
            checked = any(call[0:2] == ["python3", "cgroup"] and call[-1] == "/legacy-orphan" for call in result["calls"])
            assert checked == (scenario == "legacy-loaded-still-visible")


def readback():
    old = '    if op == "read" and scenario in {"commit-read-124", "commit-read-137"}:'
    shim = replace_one(tests.SHIM, old, old.replace('"commit-read-137"', '"commit-read-137", "commit-read-126"'))
    for code, logged in ((126, "2"), (124, "timeout"), (137, "timeout")):
        with TemporaryDirectory(prefix="readback-", dir=HERE) as name:
            box = make_box(Path(name), shim)
            initialize(box)
            result = invoke(box, f"commit-read-returned-{code}", DEPLOY, scenario=f"commit-read-{code}")
            assert result["rc"] == 1 and result["stopped"] and result["stamp"]
            assert f"Check commit-record failed ({logged})." in result["stdout"]


def locks():
    for release in (True, False):
        with TemporaryDirectory(prefix="lock-", dir=HERE) as name:
            box = make_box(Path(name), tests.SHIM)
            runtime = initialize(box)
            lock = runtime / "lock"
            lock.write_bytes(b"persist")
            lock.chmod(0o600)
            with lock.open("ab") as holder, ThreadPoolExecutor(max_workers=1) as executor:
                fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
                work = executor.submit(invoke, box, "release-holder" if release else "hold-past-shim-deadline", RECOVER)
                try:
                    end = time.monotonic() + 6
                    while not (box / "waiting-for-lock").exists() and not work.done() and time.monotonic() < end:
                        time.sleep(0.01)
                    assert not work.done() and (box / "waiting-for-lock").exists()
                    assert not (runtime / "cancelled.123.1").exists()
                    if release:
                        fcntl.flock(holder, fcntl.LOCK_UN)
                    result = work.result(timeout=8)
                finally:
                    fcntl.flock(holder, fcntl.LOCK_UN)
            assert result["rc"] == 1 and not result["stopped"]
            assert ["flock", "-w", "1500", "9"] in result["calls"]
            assert (runtime / "cancelled.123.1").exists() == release
            assert lock.read_bytes() == b"persist"


parser = argparse.ArgumentParser()
parser.add_argument("family", choices=("publication", "legacy", "readback", "locks"))
globals()[parser.parse_args().family]()
