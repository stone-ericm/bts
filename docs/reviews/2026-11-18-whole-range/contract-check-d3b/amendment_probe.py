"""Run the fixture's Bash workflow with bounded, local command shims."""

import argparse
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
spec = importlib.util.spec_from_file_location("workflow_tests", REPO / "tests/test_deploy_workflow.py")
tests = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tests)
DEPLOY, RECOVER = tests.scripts_and_steps()[0]


def replace_one(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def make_box(path, shim=tests.SHIM, *, prior=None):
    box = tests.box.__wrapped__(path)
    for file in (box / "shims").iterdir():
        file.write_text(file.read_text().split("\n", 1)[0] + "\n" + shim)
    runtime = box / "run/bts-deploy"
    runtime.mkdir(mode=0o700, parents=True)
    if prior:
        (runtime / "attempt").write_text(prior)
    (box / "enabled").write_text("enabled")
    (box / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/token\n")
    return box


def record(run="123", attempt="1", state="begun"):
    tail = "reset verified verified verified" if state == "contained" else "- - - -"
    return f"bts_deploy_attempt_v1 {run} {attempt} {tests.TARGET} {tests.OLD} {state} {tail}\n"


def invoke(box, script, scenario="active", **kwargs):
    calls = box / "calls"
    calls.unlink(missing_ok=True)
    return tests.run_box_script(box, script, scenario, **kwargs)


def legacy(case):
    addition = '''        if not user and args[1] == "bts-scheduler.service":
            if not (root / "legacy-stopped").exists() and scenario != "escaped":
                states.update(LoadState="loaded", ControlGroup="" if scenario == "idle" else "/legacy-orphan")
            if scenario == "escaped":
                states.update(LoadState="not-found", ControlGroup="")
'''
    shim = replace_one(tests.SHIM, '        if user and "LoadState" in args and scenario == "unloaded":',
                       addition + '        if user and "LoadState" in args and scenario == "unloaded":')
    shim = replace_one(shim, '    if command == "pgrep" and scenario == "residual":', '''    if command == "pgrep" and scenario == "escaped":
        import re
        raise SystemExit(not bool(re.search(args[-1], "python -m bts schedule --config /home/bts/.bts-orchestrator.toml")))
    if command == "pgrep" and scenario == "residual":''')
    with TemporaryDirectory(prefix="legacy-", dir=HERE) as name:
        box = make_box(Path(name), shim, prior=record() if case != "idle" else None)
        if case != "idle":
            group = box / "sys/fs/cgroup/legacy-orphan"
            group.mkdir()
            (group / "cgroup.events").write_text("populated 1\n")
        script = DEPLOY if case == "idle" else RECOVER
        out, calls = invoke(box, script, case, fault="reset:17" if case == "idle" else "")
        assert out.returncode == 1 and "Scheduler stopped: verified." in out.stdout
        checked = any(call[0] == "python3" and call[3:4] == ["cgroup"] and call[-1] == "/legacy-orphan" for call in calls)
        stopped = ["systemctl", "stop", "bts-scheduler.service"] in calls
        assert not checked and stopped == (case != "escaped")
        print(case, "legacy stop", stopped, "orphan checked", checked, "printed verified")


def signal_order(case):
    changed = replace_one(DEPLOY, 'if [[ $publication == beginning ]] && (( rc == 3 )); then',
                          'if [[ $publication == beginning ]] && { (( rc == 3 )) || { (( rc == 0 )) && ! same_ids; }; }; then')
    with TemporaryDirectory(prefix="order-", dir=HERE) as name:
        prior = record("101", "1", "begun" if case == "order" else "committed")
        box = make_box(Path(name), prior=prior)
        out, calls = invoke(box, DEPLOY if case == "baseline" else changed, fault="publish-begun:TERM:unpublished")
        attempt = box / "run/bts-deploy/attempt"
        if case == "baseline":
            assert out.returncode == 1 and "DEPLOY FAILED at interrupted" in out.stdout
            assert attempt.read_text() != prior and (box / "stopped").exists()
            print("base incorrectly replaced prior commit and stopped")
            return
        assert out.returncode == 1 and "DEPLOY NOT STARTED" in out.stdout
        assert attempt.read_text() == prior and not (box / "stopped").exists()
        out, calls = invoke(box, RECOVER)
        assert "a different attempt owns the box record" in out.stdout
        assert attempt.read_text() == prior and not (box / "stopped").exists()
        assert (box / "run/bts-deploy/cancelled.123.1").exists()
        out, calls = invoke(box, RECOVER, run_id="101")
        assert "DEPLOY FAILED at deploy-step-incomplete" in out.stdout and (box / "stopped").exists()
        assert attempt.read_text().startswith("bts_deploy_attempt_v1 101 1 ")
        print("S6-shaped edit preserves A's begun; B recovery marks B; delayed A recovery contains A")


def record_failure():
    shim = replace_one(tests.SHIM, 'if at and fault[0] == at and not (root / "fault-fired").exists():', '''if at == "publish-contained" and (root / "refuse-contained-record").exists():
    raise SystemExit(17)
if at and fault[0] == at and not (root / "fault-fired").exists():''')
    shim = replace_one(shim, '    if op == "read" and scenario in {"commit-read-124", "commit-read-137"}:',
                       '    if op == "read" and scenario in {"commit-read-124", "commit-read-137", "commit-read-126"}:')
    with TemporaryDirectory(prefix="record-", dir=HERE) as name:
        box = make_box(Path(name), shim)
        (box / "refuse-contained-record").touch()
        out, _ = invoke(box, DEPLOY, "commit-read-126")
        assert "Check commit-record failed (2)." in out.stdout
        assert (box / "stopped").exists()
        assert " committed " in (box / "run/bts-deploy/attempt").read_text()
        out, _ = invoke(box, RECOVER)
        assert "this attempt committed on the box" in out.stdout
        print("commit rename + read 126 + failed contained rename: deploy stopped; recovery claims committed")


def budget(case):
    script = RECOVER
    if case != "base":
        script = replace_one(script, 'deadline=$((containment_deadline - 60000))',
                             'deadline=$((containment_deadline - 75000))')
        script = replace_one(script, '    run 20 5 normal pgrep',
                             '    deadline=$((containment_deadline - 60000))\n    run 20 5 normal pgrep')
    script = replace_one(script, '        now=$((budget_origin + (SECONDS - budget_start) * 1000))', '''        local elapsed=0
        if [[ -f $BOX/ticks ]]; then IFS= read -r elapsed < "$BOX/ticks"; fi
        now=$((budget_origin + elapsed * 1000))''')
    shim = replace_one(tests.SHIM, 'elif command == "timeout":\n    dispatch(args[3:])', '''elif command == "timeout":
    ticks = root / "ticks"
    if ticks.exists():
        elapsed = int(ticks.read_text())
        step = Path(args[3]).name
        if step == "python3" and args[4] == "-c":
            step = args[6]
        elif step in {"sudo", "systemctl"}:
            step = args[args.index("systemctl") + 1] if "systemctl" in args else args[-3]
            if "endpoint" in args:
                step = "endpoint"
            if "marker" in args:
                step = "marker"
        cap = int(args[1]) + int(args[2])
        with (root / "ticks-log").open("a") as log:
            print(json.dumps([step, elapsed, int(args[2])]), file=log)
        if step == "list-jobs" and elapsed == 203:
            ticks.write_text(str(elapsed + cap) + "\\n")
            raise SystemExit(124)
        if step in {"pgrep", "marker", "endpoint", "curl", "stage-record"}:
            ticks.write_text(str(elapsed + cap) + "\\n")
    dispatch(args[3:])''')
    shim = replace_one(shim, '        (root / "clock-failed").touch()', '''        (root / "clock-failed").touch()
        if not (root / "ticks").exists():
            (root / "ticks").write_text("0\\n")''')
    hook = '    if user and args[0] == "list-jobs" and scenario == "pending-job":'
    shim = replace_one(shim, hook, '''    if not user and args[0] == "show" and "LoadState" in args and (root / "stopped").exists():
        counter = root / "legacy-queries"
        if counter.exists():
            (root / "ticks").write_text("203\\n")
        counter.touch()
''' + hook)
    with TemporaryDirectory(prefix="budget-", dir=HERE) as name:
        box = make_box(Path(name), shim, prior=record())
        out, calls = invoke(box, script, "recovery-clock-every")
        timeline = [json.loads(line) for line in (box / "ticks-log").read_text().splitlines()]
        tail = [entry for entry in timeline if entry[1] >= 203]
        assert out.returncode == 1 and "Scheduler stopped: unverified." in out.stdout
        if case == "base":
            assert not any(step == "pgrep" for step, _, _ in timeline)
        else:
            assert ["pgrep", 203, 12] in tail
        assert any(step == "marker" for step, _, _ in tail)
        assert any(step == "curl" and term == 5 for step, _, term in tail)
        assert not any(step == "mv" for step, _, _ in tail)
        print(case, tail, "record remains", (box / "run/bts-deploy/attempt").read_text().split()[5])


parser = argparse.ArgumentParser()
parser.add_argument("case", choices=("orphan", "escaped", "idle", "baseline", "order", "record", "base", "window"))
case = parser.parse_args().case
if case in {"orphan", "escaped", "idle"}:
    legacy(case)
elif case in {"baseline", "order"}:
    signal_order(case)
elif case == "record":
    record_failure()
else:
    budget(case)
