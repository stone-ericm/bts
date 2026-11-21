"""Run an authorized recovery near its real proof deadline using timed shims."""

import argparse
import inspect
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import time

from harness import HERE, RECOVER, invoke, make_box, posts, replace_one, tests


shim = replace_one(tests.SHIM, 'elif command == "timeout":\n    dispatch(args[3:])', '''
elif command == "timeout":
    word = args[3]
    if word.startswith("/") and word != "/bin/bash" and not Path(word).is_relative_to(root):
        raise SystemExit(99)
    executable = root / "shims" / Path(word).name
    if not executable.is_file():
        raise SystemExit(98)
    os.execv("/usr/bin/timeout", ["timeout", *args[:3], str(executable), *args[4:]])''')

shim = replace_one(shim, 'def phase():', '''
def delay(label, seconds):
    base = root / "containment-start"
    if not base.exists():
        return
    if (root / "slow-marker-mode").exists():
        if label != "failure-marker":
            return
        seconds = 30
    now = time.monotonic() - float(base.read_text())
    with (root / "progress").open("a") as log:
        print(json.dumps([label, round(now, 3), seconds]), file=log)
    time.sleep(seconds)

def phase():''')

shim = replace_one(shim, '        (root / "clock-failed").touch()', '''
        if not (root / "containment-start").exists():
            (root / "containment-start").write_text(str(time.monotonic()))
        (root / "clock-failed").touch()''')

shim = replace_one(shim, '    if op == "marker" and scenario == "marker-unwritable":', '''
    if op == "cgroup":
        delay("cgroup-" + args[-1], 13.7)
    if op == "marker":
        delay("failure-marker", 0)
    if op == "marker" and scenario == "marker-unwritable":''')

shim = replace_one(shim, 'elif command == "id":\n    print(1000)', '''
elif command == "id":
    delay("get-uid", 3.9)
    print(1000)''')

shim = replace_one(shim, '    if user and args[0] in {"disable", "enable"}:', '''
    proof = (root / "stopped").exists() and (root / "legacy-stopped").exists()
    scope = "user" if user else "system"
    if args[0] == "show":
        if args[-2:] == ["-p", "ControlGroup"] and "ActiveState" not in args:
            delay(scope + "-capture-group", 3.9)
        elif "ActiveState" in args:
            delay(scope + "-show-proof" if proof else scope + "-discover", 13.7 if proof else 3.9)
    elif args[0] == "list-jobs":
        delay(scope + "-jobs-proof" if proof else scope + "-jobs-discovery", 13.7 if proof else 3.9)
    elif args[0] == "disable":
        delay(scope + "-disable", 3.9)
    elif args[0] == "stop":
        delay(scope + "-stop", 52.9)
    elif args[0] == "is-enabled":
        delay(scope + "-enablement", 13.7)
    if user and args[0] in {"disable", "enable"}:''')

shim = replace_one(shim, '        if "ControlGroup" in args and user and scenario == "main-pid":', '''
        if not user and args[1] == "bts-scheduler.service":
            states.update(LoadState="loaded", ControlGroup="/legacy-sched")
        if "ControlGroup" in args and user and scenario == "main-pid":''')

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=("slow-proof", "slow-marker"))
mode = parser.parse_args().mode
limit = 45 if mode == "slow-marker" else 305
runner_source = replace_one(inspect.getsource(tests.run_box_script), "communicate(timeout=15)", f"communicate(timeout={limit})")
namespace = vars(tests).copy()
exec(compile(runner_source, "<timing adapter>", "exec"), namespace)

with TemporaryDirectory(prefix="budget-", dir=HERE) as name:
    box = make_box(Path(name), shim)
    runtime = box / "run/bts-deploy"
    runtime.mkdir(mode=0o700, parents=True)
    (runtime / "attempt").write_text(f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n")
    (box / "head").write_text(tests.TARGET)
    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/monitor\n")
    if mode == "slow-marker":
        (box / "slow-marker-mode").touch()
    launched = time.monotonic()
    result = invoke(box, mode + "-with-fallback", RECOVER, "recovery-clock-every", runner=namespace["run_box_script"])
    elapsed = time.monotonic() - launched
    timeline = [json.loads(line) for line in (box / "progress").read_text().splitlines()]
    print(json.dumps(dict(elapsed=round(elapsed, 3), timeline=timeline)), flush=True)
    assert result["rc"] == 1 and result["stopped"] and (box / "legacy-stopped").exists()
    assert len(posts(result)) == 1 and "contained" in result["attempt"]
    if mode == "slow-marker":
        assert elapsed < 40 and result["heartbeat"] is None
        assert "Failure heartbeat written: unverified." in result["stdout"]
    else:
        assert elapsed < 280 and '"state": "deploy_failed"' in result["heartbeat"]
        assert any(label == "system-jobs-proof" for label, _, _ in timeline)
        assert all(call[0] != "pgrep" for call in result["calls"]), result
