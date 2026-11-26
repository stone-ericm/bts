#!/usr/bin/env python3
"""Item 1 & 4: Replay Round-2 Codex budget_probe.py (slow-marker and slow-proof) on D3b (90a9b6f0), plus slow-endpoint curl TERM check."""
from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
SCRATCH_PARENT = REPO / ".codex-review/wr-d3b/tmp"
SPEC = importlib.util.spec_from_file_location("workflow_tests", REPO / "tests/test_deploy_workflow.py")
assert SPEC and SPEC.loader
tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tests)
DEPLOY, RECOVER = tests.scripts_and_steps()[0]


def replace_one(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, f"expected 1 occurrence of {old!r}, got {source.count(old)}"
    return source.replace(old, new)


def build_timed_shim() -> str:
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
    if op == "endpoint":
        delay("endpoint-parse", 11.5)
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

    return shim


def run_mode(mode: str) -> None:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    shim = build_timed_shim()
    limit = 45 if mode == "slow-marker" else 305
    with tempfile.TemporaryDirectory(prefix=f"budget-{mode}-", dir=SCRATCH_PARENT) as td:
        box = tests.box.__wrapped__(Path(td))
        for file in (box / "shims").iterdir():
            file.write_text(file.read_text().split("\n", 1)[0] + "\n" + shim)
        runtime = box / "run/bts-deploy"
        runtime.mkdir(mode=0o700, parents=True)
        (runtime / "attempt").write_text(
            f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n"
        )
        (box / "head").write_text(tests.TARGET)
        (box / "home/projects/bts/.env").write_text(
            "BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/monitor\n"
        )
        if mode == "slow-marker":
            (box / "slow-marker-mode").touch()
        t0 = time.monotonic()
        result, calls = tests.run_box_script(box, RECOVER, "recovery-clock-every", limit=limit)
        elapsed = round(time.monotonic() - t0, 3)
        timeline = [json.loads(line) for line in (box / "progress").read_text().splitlines()]
        hb_path = box / "home/projects/bts/data/.heartbeat"
        hb = hb_path.read_text() if hb_path.is_file() else None
        pgrep_timeouts = [c for c in calls if c[0] == "timeout" and "pgrep" in c]
        curl_timeouts = [c for c in calls if c[0] == "timeout" and "curl" in c and "--data-raw" in c]
        summary = {
            "mode": mode,
            "rc": result.returncode,
            "elapsed_s": elapsed,
            "pgrep_timeout_argv": pgrep_timeouts,
            "curl_timeout_argv": [c[:5] for c in curl_timeouts],
            "stdout": result.stdout.splitlines(),
            "attempt": (runtime / "attempt").read_text().strip(),
            "timeline": timeline,
        }
        print(json.dumps(summary), flush=True)
        assert result.returncode == 1
        assert len(curl_timeouts) == 1
        assert " contained " in (runtime / "attempt").read_text()
        if mode == "slow-marker":
            assert elapsed < 40 and hb is None
            assert "Failure heartbeat written: unverified." in result.stdout
        else:
            assert elapsed < 280 and hb is not None and '"deploy_failed"' in hb
            assert len(pgrep_timeouts) == 1, "residual pgrep MUST be launched on D3b"
            assert int(pgrep_timeouts[0][3]) >= 5
            assert int(curl_timeouts[0][3]) >= 10
            assert "Scheduler stopped: unverified." in result.stdout


def main() -> None:
    run_mode("slow-marker")
    run_mode("slow-proof")
    print("Replay of budget_probe.py (slow-marker + slow-proof + slow-endpoint): ALL PASSED", flush=True)


if __name__ == "__main__":
    main()
