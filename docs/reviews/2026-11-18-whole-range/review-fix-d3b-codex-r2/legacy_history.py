"""Exercise legacy marker ordering, stale ownership and proof-time state changes."""
import json
from shutil import rmtree
from subprocess import run
import sys

from shared import ROOT, DEPLOY, RECOVER, box, group, inject_shim, invoke, record, replace, tests

RUNTIME = "run/bts-deploy"
LEGACY = "bts-scheduler.service"
CGROUP = "sys/fs/cgroup/legacy-orphan"


def set_up_legacy(path, population):
    fragments = path / "etc/systemd/system"
    fragments.mkdir(parents=True)
    for name in (LEGACY, "bts-dashboard.service"):
        (fragments / name).write_text("[Service]\n")
    group(path, "/legacy-orphan", population)


def stop_verified(state):
    return "Scheduler stopped: verified." in state["out"]


def run_deploy(path, label, fault, owner="123", attempt="1", scenario="legacy-unloads"):
    (path / "fault-fired").unlink(missing_ok=True)
    return invoke(path, label, DEPLOY, scenario, fault=fault, run_id=owner, attempt=attempt)


def boundary():
    for fault, scenario in (("", "legacy-unloads"), ("", "legacy-unloads-markfail"), ("legacy-stop:TERM", "legacy-unloads")):
        with box() as path:
            set_up_legacy(path, 0)
            state, calls = run_deploy(path, f"mark authority {scenario=} {fault=}", fault, scenario=scenario)
            mark = path / RUNTIME / "legacy.123.1"
            mark_call = next(i for i, c in enumerate(calls) if c[0] == "python3" and c[3:4] == ["mark"])
            commands = [i for i, c in enumerate(calls) if c[0] == "systemctl" and c[1:2] in (["disable"], ["stop"]) and c[-1] == LEGACY]
            assert commands and all(i > mark_call for i in commands)
            if scenario.endswith("markfail"):
                assert not mark.exists() and "DEPLOY FAILED at legacy-units" in state["out"]
                assert not any(c[:2] == ["systemctl", "daemon-reload"] for c in calls)
            else:
                assert mark.is_file() and mark.read_bytes() == b"" and mark.stat().st_mode & 0o777 == 0o600
                if fault:
                    assert state["rc"] == 1 and "DEPLOY FAILED at interrupted" in state["out"]
                else:
                    assert state["rc"] == 0 and "CANARY PASSED" in state["out"]


def cross(population):
    with box() as path:
        set_up_legacy(path, int(population))
        a, calls = run_deploy(path, f"A marks, unloads, then canary fails, group {population=}", "contest-state:17")
        assert a["rc"] == 1 and not stop_verified(a)
        assert ["systemctl", "daemon-reload"] in calls
        runtime = path / RUNTIME
        mark = runtime / "legacy.123.1"
        assert mark.exists()
        rec_a, _ = invoke(path, "A matching recovery with mark", RECOVER, "legacy-unloads")
        assert not stop_verified(rec_a) and "containment re-checked" not in rec_a["out"]
        b, calls = run_deploy(path, "B rerun reset failure on the same boot", "reset:17", attempt="2")
        assert b["rc"] == 1 and stop_verified(b) and "/legacy-orphan" not in b["groups"]
        assert mark.is_file() and not (runtime / "legacy.123.2").exists()
        assert not any(c[:2] == ["systemctl", "stop"] for c in calls)
        assert (path / CGROUP / "cgroup.events").read_text() == f"populated {population}\n"
        stale, _ = invoke(path, "A recovery arrives again after B takes ownership", RECOVER, "legacy-unloads")
        assert not stale["writes"] and not stale["posts"] and "a different attempt owns" in stale["out"]
        rec_b, _ = invoke(path, "B matching recovery", RECOVER, "legacy-unloads", attempt="2")
        assert stop_verified(rec_b) and "containment re-checked" in rec_b["out"]
        c, _ = run_deploy(path, "C distinct run reset failure on the same boot", "reset:17", owner="708")
        assert stop_verified(c) and (path / CGROUP / "cgroup.events").is_file()
        rmtree(path / "run")
        # A reboot also destroys the old process and its cgroup.
        rmtree(path / CGROUP)
        after, _ = invoke(path, "C recovery after reboot clears old attempt/mark and orphan", RECOVER,
                          "legacy-unloads", run_id="708")
        assert "no phase-2 record is present" in after["out"] and not after["writes"]
        newer, _ = run_deploy(path, "D failure after reboot with no orphan", "reset:17", owner="809")
        assert stop_verified(newer) and not mark.exists()


RACE = inject_shim(r'''
if command == "systemctl" and args[:2] == ["show", "bts-scheduler.service"]:
    scoped_stop = (root / "legacy-stopped").exists()
    changes_state = scenario == "split-proof-mixed" and scoped_stop and "Id" not in args
    absent = changes_state
    states = dict(Id=args[1], LoadState="not-found" if absent else "loaded",
                  ActiveState="inactive" if absent or scenario == "split-proof-stable" else "active",
                  MainPID="0" if absent or scenario == "split-proof-stable" else "4242",
                  ControlGroup="/legacy-orphan")
    with (root / "manager-samples").open("a") as log:
        print(json.dumps(dict(args=args, fields=states, after_stop=scoped_stop)), file=log)
    for flag in args:
        if flag in states:
            print(f"{flag}={states[flag]}")
    raise SystemExit(0)
if command == "systemctl" and args == ["is-enabled", "bts-scheduler.service"]:
    print("disabled")
    raise SystemExit(1)
''')


def mixed_proof(variant="original"):
    def script_for(recovery):
        script = RECOVER if recovery else DEPLOY
        if variant == "same-sample":
            old = 'output=$("$command" 20 normal show "$name" -p ActiveState -p MainPID -p ControlGroup) || return 1'
            new = '''output=$("$command" 20 normal show "$name" -p ActiveState -p MainPID -p ControlGroup -p LoadState) || return 1
    if [[ $command == system_unit ]]; then
        [[ $(field LoadState "$output") == loaded ]] || return 1
    fi'''
            script = replace(script, old, new)
        return script

    for recovery in (False, True):
        for scenario in ("split-proof-mixed", "split-proof-stable"):
            with box(record() if recovery else None, shim=RACE) as path:
                set_up_legacy(path, 0)
                state, _ = invoke(path, f"{scenario=} {recovery=} {variant=}", script_for(recovery), scenario,
                                  fault="" if recovery else "reset:17")
                samples = [json.loads(line) for line in (path / "manager-samples").read_text().splitlines()]
                print(json.dumps(samples), flush=True)
                expected = variant == "original" or scenario == "split-proof-stable"
                assert stop_verified(state) == expected
                assert ("/legacy-orphan" in state["groups"]) == expected
                if scenario == "split-proof-mixed":
                    post = [item for item in samples if item["after_stop"]]
                    assert [s["fields"]["LoadState"] for s in post] == ["loaded", "not-found"]
                    assert post[0]["fields"]["MainPID"] == "4242"
                    assert ("LoadState" in post[1]["args"]) == (variant == "same-sample")
                    assert (path / "etc/systemd/system" / LEGACY).is_file()


KILL_AT_PROOF = inject_shim(r'''
if (scenario == "legacy-vanishing" and command == "python3" and args[2] == "cgroup"
        and (root / "legacy-stopped").exists() and not (root / "owner-killed").exists()):
    (root / "owner-killed").touch()
    os.kill(int((root / "owner.pid").read_text()), signal.SIGKILL)
''')


def repeated_stop():
    for lost_owner in (False, True):
        with box(shim=KILL_AT_PROOF if lost_owner else None) as path:
            group(path, "/legacy-orphan", 1)
            state, _ = invoke(path, f"reset fails before phase-2 legacy cleanup, {lost_owner=}",
                              scenario="legacy-vanishing", fault="reset:17")
            assert (path / "legacy-stopped").exists()
            assert " begun " in state["attempt"] if lost_owner else (
                " contained reset unverified verified verified" in state["attempt"] and not stop_verified(state))
            assert not (path / RUNTIME / "legacy.123.1").exists()
            after, calls = invoke(path, f"same-attempt recovery after stopped legacy vanished, {lost_owner=}",
                                  RECOVER, "legacy-vanishing")
            assert stop_verified(after) and "/legacy-orphan" not in after["groups"]
            assert (path / CGROUP / "cgroup.events").read_text() == "populated 1\n"
            assert not any(c[:2] == ["systemctl", "stop"] for c in calls)
            assert ("containment re-checked" in after["out"]) == (not lost_owner)


def mark_checks():
    first = replace(RECOVER, "no_fragment && ! legacy_cleaned; then", "no_fragment; then")
    proof = replace(RECOVER, "if ! legacy_cleaned && post_group=", "if post_group=")
    for label, script, scenario, expected_proof, expected_stop in (
        ("first sample current", RECOVER, "active", False, True),
        ("first sample mutated", first, "active", False, False),
        ("proof current", RECOVER, "legacy-ghost-group", False, True),
        ("proof mutated", proof, "legacy-ghost-group", True, True),
    ):
        with box(record()) as path:
            mark = path / RUNTIME / "legacy.123.1"
            mark.touch(mode=0o600)
            group(path, "/legacy-group", 0)
            state, calls = invoke(path, label, script, scenario)
            assert stop_verified(state) == expected_proof
            assert (["systemctl", "stop", LEGACY] in calls) == expected_stop
    start = DEPLOY.index('\n        if [[ $name == "$scheduler" ]]; then\n', DEPLOY.index("publication=begun")) + 1
    end = DEPLOY.index('\n        system_unit 20 mutation disable "$name"', start) + 1
    marker = DEPLOY[start:end]
    # Match the stop and its check including the original YAML-derived indentation.
    delimiter = '        system_unit 20 mutation stop "$name" >/dev/null\n        step_status legacy-units "$?"\n'
    delayed = replace(DEPLOY[:start] + DEPLOY[end:], delimiter, delimiter + marker)
    with box() as path:
        set_up_legacy(path, 0)
        state, calls = invoke(path, "delayed mark, original phase stop runs first", delayed, "legacy-unloads-markfail")
        stop = calls.index(["systemctl", "stop", LEGACY])
        mark = next(i for i, c in enumerate(calls) if c[0] == "python3" and c[3:4] == ["mark"])
        assert stop < mark and "DEPLOY FAILED at legacy-units" in state["out"]


def blank_controls():
    shim = inject_shim(r'''
if (scenario == "absent-property-error" and command == "systemctl"
        and args == ["show", "bts-scheduler.service", "-p", "ControlGroup"]):
    raise SystemExit(1)
''')
    for script in (DEPLOY, RECOVER):
        for scenario in ("active", "absent-property-error"):
            with box(record() if script == RECOVER else None, shim=shim) as path:
                state, calls = invoke(path, f"empty legacy scope {scenario=} {script == RECOVER=}", script, scenario,
                                      fault="" if script == RECOVER else "reset:17")
                assert stop_verified(state) == (scenario == "active")
                assert (["systemctl", "stop", LEGACY] in calls) == (scenario != "active")


PHASE_TIME = inject_shim(r'''
elapsed = root / "virtual-phase-ms"
if command == "python3" and args[2] == "clock":
    print(100_000_000 + (int(elapsed.read_text()) if elapsed.exists() else 0))
    raise SystemExit(0)
if at == "publish-begun" and not elapsed.exists():
    elapsed.write_text("0")
if command == "timeout":
    os.environ["VIRTUAL_TERM"] = args[2]
    os.environ["VIRTUAL_GRACE"] = args[1]
if elapsed.exists():
    delays = {
        "disable": 13, "check-disable": 13, "reset": 36, "sync": 470,
        "legacy-disable": 8, "legacy-stop": 8, "legacy-rm": 8,
        "legacy-daemon-reload": 8, "legacy-pkill": 8, "restart": 68,
        "head": 4, "wait": 30, "scheduler-state": 12, "dashboard-state": 12,
        "dashboard-http": 8, "contest-state": 50, "stamp-stage": 4,
        "enable": 13, "check-enable": 13, "stamp": 4, "stage-committed": 2,
    }
    label = "legacy-mark" if command == "python3" and args[2] == "mark" else at
    delta = 4 if label == "legacy-mark" else delays.get(label, 0)
    if delta:
        at_ms = int(elapsed.read_text())
        term, grace = (int(os.environ[name]) for name in ("VIRTUAL_TERM", "VIRTUAL_GRACE"))
        timed_out = delta >= term
        end = at_ms + 1000 * (term + grace if timed_out else delta)
        elapsed.write_text(str(end))
        with (root / "phase-timeline").open("a") as log:
            print(json.dumps([label, at_ms, end, term, timed_out]), file=log)
        if timed_out:
            raise SystemExit(124)
''')


def phase_budget():
    previous = run(["git", "show", "4733a298:.github/workflows/deploy.yml"], cwd=ROOT,
                   check=True, capture_output=True, text=True, timeout=5).stdout
    old = tests.scripts_and_steps(previous)[0][0]
    for script in (old, DEPLOY):
        with box(shim=PHASE_TIME) as path:
            set_up_legacy(path, 0)
            state, _ = invoke(path, "current deadline model" if script == DEPLOY else "pre-mark deadline model",
                              script, "legacy-unloads")
            phases = [json.loads(line) for line in (path / "phase-timeline").read_text().splitlines()]
            print(json.dumps(phases), flush=True)
            assert next(row[2] for row in phases if row[0] == "stamp") == (794000 if script == DEPLOY else 790000)
            assert next(row[-1] for row in phases if row[0] == "stage-committed") == (script == DEPLOY)
            if script == DEPLOY:
                assert state["rc"] == 1 and "Check commit-record failed (timeout)." in state["out"]
            else:
                assert state["rc"] == 0 and "CANARY PASSED" in state["out"]


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
