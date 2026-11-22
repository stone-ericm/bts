"""Run the must-still-catch inputs independently of pytest assertions."""
import fcntl
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path
import json
import re
import sys
import time

from review_box import (ROOT, DEPLOY, RECOVER, SIGNALS, PRIORS, OLD, TARGET, box, cgroups, contained,
                        group, inject_shim, invoke, record, shell_function)


def clock():
    for recovery, frequency in product((False, True), ("once", "every")):
        with box(record() if recovery else None) as path:
            if recovery:
                (path / "head").write_text(TARGET)
            result, calls = invoke(path, f"K1 {recovery=} {frequency=}", RECOVER if recovery else DEPLOY,
                                   ("recovery" if recovery else "deploy") + "-clock-" + frequency)
            assert (path / "clock-failed").exists()
            contained(result, "deploy-step-incomplete" if recovery else "head")
            assert ["systemctl", "--user", "stop", "bts-scheduler"] in calls


LATE = inject_shim(r'''
if scenario.startswith("late-independent-"):
    _, _, seconds, failpoint, signame = scenario.split("-")
    timefile = root / "late-time"
    if at == failpoint and not timefile.exists():
        timefile.write_text(seconds)
        os.kill(int((root / "owner.pid").read_text()), getattr(signal, "SIG" + signame))
    if command == "python3" and args[2] == "read" and timefile.exists() and not (root / "stopped").exists():
        (root / "handler-read").touch()
        raise SystemExit(126)
    if command == "python3" and args[2] == "clock" and timefile.exists():
        print(time.monotonic_ns() // 1_000_000 + int(timefile.read_text()) * 1000)
        raise SystemExit(0)
''')


def late():
    for seconds, point, sig in ((790, "wait", "HUP"), (795, "stamp", "TERM"), (805, "scheduler", "PIPE"),
                                (795, "enable", "INT"), (795, "restart", "TERM")):
        selected = "scheduler-state" if point == "scheduler" else point
        shim = LATE.replace("at == failpoint", f"at == {selected!r}")
        with box(shim=shim) as path:
            state, _ = invoke(path, f"K2 {seconds=} {point=} {sig=}",
                              scenario=f"late-independent-{seconds}-{point}-{sig}")
            contained(state, "interrupted")
            assert not (path / "handler-read").exists()
            assert (path / "late-time").read_text() == str(seconds)


def blanks():
    shim = inject_shim(r'''
if scenario == "all-blank-system" and command == "systemctl" and args[:2] == ["show", "bts-scheduler.service"]:
    states = dict(Id=args[1], LoadState="loaded", ActiveState="inactive", MainPID="0", ControlGroup="")
    for flag in args:
        if flag in states:
            print(f"{flag}={states[flag]}")
    raise SystemExit(0)
''')
    cases = (("blank-cgroup", "head:17", True), ("legacy-blank-cgroup", "sync:17", True),
             ("unknown-cgroup", "reset:17", False), ("all-blank-system", "reset:17", False))
    for scenario, fault, known in cases:
        with box(shim=shim) as path:
            group(path, "/sched", 1)
            state, calls = invoke(path, "K3 " + scenario, scenario=scenario, fault=fault)
            contained(state)
            assert "Scheduler stopped: unverified." in state["out"]
            assert ("/sched" in cgroups(calls)) == known


def url():
    for suffix in ("{other}", "{one,two}", "[1-1]", "[1-2]", "literal"):
        for script in (DEPLOY, RECOVER):
            recovery = script == RECOVER
            with box(record() if recovery else None, url=f"https://checks.invalid/{suffix}/route") as path:
                state, calls = invoke(path, f"K4 {suffix=} {recovery=}", script,
                                      scenario="active" if recovery else "contest-refused")
                contained(state, paged=suffix == "literal")
                assert all("--globoff" in c for c in calls if c[0] == "curl")
                assert "Failure ping received HTTP 2xx: " + ("verified." if suffix == "literal" else "unverified.") in state["out"]


def commit():
    cases = [(f"{where}-committed:{status}", "active", str(status) if status == 17 else "timeout")
             for where, status in product(("stage", "publish"), (17, 124, 137))]
    cases += [("", f"commit-read-{status}", "126" if status == 126 else "timeout") for status in (124, 137, 126)]
    for fault, scenario, status in cases:
        row = "K15" if status == "126" else "K5"
        with box() as path:
            state, _ = invoke(path, f"{row} {fault=} {scenario=}", scenario=scenario, fault=fault)
            contained(state, "commit-record")
            assert f"Check commit-record failed ({status})." in state["out"]
            assert re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z\n", state["stamp"])
    with box() as path:
        state, _ = invoke(path, "K5 rename performed, wrapper exits 17", scenario="committed-with-error")
        assert state["rc"] == 0 and "CANARY PASSED" in state["out"]
        assert state["attempt"] == record("committed")


def nul():
    with box() as path:
        state, _ = invoke(path, "K7 zero-exit canary with raw NUL", scenario="contest-nul")
        contained(state, "contest-state")
        assert "Check contest-state failed (query)." in state["out"]
        assert "Deploy canary:" not in state["out"]


def lock():
    with box(record("begun", owner="412")) as path:
        state, calls = invoke(path, "K8 foreign run same attempt", RECOVER)
        assert "different attempt owns" in state["out"] and not state["writes"] and state["posts"] == 0
        runtime = path / "run/bts-deploy"
        assert (runtime / "cancelled.123.1").stat().st_mode & 0o777 == 0o600
        assert (runtime / "cancelled.123.1").read_bytes() == b""
        assert (runtime / "lock").stat().st_mode & 0o777 == 0o600
        assert state["attempt"] == record("begun", owner="412")
    with box(record()) as path:
        state, calls = invoke(path, "K8 valid POST endpoint with redirect response", RECOVER, scenario="post-0-301")
        assert state["posts"] == 1 and "HTTP 2xx: unverified." in state["out"]
        post = next(c for c in calls if c[0] == "curl")
        assert not {"-L", "--location", "--location-trusted"}.intersection(post)
    for failure in ("lockfile", "mark"):
        shim = inject_shim(f'''
if command == "python3" and args[2] == "{failure}":
    raise SystemExit(17)
''')
        with box(shim=shim) as path:
            state, _ = invoke(path, "K8 independent helper failure " + failure, RECOVER)
            assert not state["writes"]
            assert ("lock unavailable" if failure == "lockfile" else "mark could not be verified") in state["out"]
    with box() as path, ThreadPoolExecutor(max_workers=1) as pool:
        runtime = path / "run/bts-deploy"
        lockfile = runtime / "lock"
        lockfile.write_bytes(b"preserved")
        lockfile.chmod(0o600)
        with lockfile.open("ab") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            future = pool.submit(invoke, path, "K8 real inode held then released", RECOVER)
            try:
                stop = time.monotonic() + 6
                while time.monotonic() < stop and not (path / "waiting-for-lock").exists() and not future.done():
                    time.sleep(.01)
                assert (path / "waiting-for-lock").exists() and not future.done()
                assert not (runtime / "cancelled.123.1").exists()
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)
            state, calls = future.result(timeout=14)
        assert ["flock", "-w", "1500", "9"] in calls
        assert (runtime / "cancelled.123.1").exists() and lockfile.read_bytes() == b"preserved"
        assert not state["writes"]


def publication(signal):
    for prior in (*PRIORS, None):
        initial = record(prior, owner="103" if prior != "contained" else "123") if prior else None
        attempt = "2" if prior == "contained" else "1"
        for stage, mode in (("stage-begun", "before"), ("publish-begun", "before"),
                            ("publish-begun", "after"), ("publish-begun", "unreadable")):
            if prior is None and mode == "unreadable":
                continue
            with box(initial) as path:
                fault = f"{stage}:{signal}" + (":unpublished" if mode != "after" else "")
                state, calls = invoke(path, f"{'K9' if mode == 'before' else 'K10'} {prior=} {fault=} {mode=}",
                                      scenario="beginning-read-126" if mode == "unreadable" else "active",
                                      fault=fault, attempt=attempt)
                assert (path / "fault-fired").exists()
                if mode == "before":
                    assert state["rc"] == 1 and "DEPLOY NOT STARTED — attempt-record failed." in state["out"]
                    assert not state["writes"] and not state["posts"] and "DEPLOY FAILED" not in state["out"]
                    assert (state["attempt"], state["marker"], state["link"], state["head"]) == (initial, '{"state":"running"}', "enabled", OLD)
                    if prior == "committed":
                        print("K17 foreign committed beginning gave not-started", flush=True)
                else:
                    contained(state, "interrupted")
                    assert state["out"].count("DEPLOY FAILED at interrupted") == 1
                    if mode == "after":
                        assert any(c[0] == "mv" and ".attempt." in c[-2] for c in calls)
                        assert sum(c[0] == "mv" for c in calls) >= 2
                    else:
                        assert (path / "read-failed").exists()


def legacy():
    cases = (
        ("legacy-vanishing", 1, False, True, "K11"), ("legacy-vanishing", 0, True, True, "K12"),
        ("legacy-vanishing-nogroup", 0, False, False, "K12"), ("active", 0, True, False, "K12"),
        ("legacy-vanishing-job", 0, False, False, "K12"),
        ("legacy-vanishing-queryfail", 0, False, False, "K12"), ("dangling", 0, False, False, "K12"),
    )
    for recovery, (scenario, pop, verified, checked, row) in product((False, True), cases):
        with box(record() if recovery else None) as path:
            group(path, "/legacy-orphan", pop)
            if scenario == "dangling":
                fragment = path / "etc/systemd/system/bts-scheduler.service"
                fragment.parent.mkdir(parents=True)
                fragment.symlink_to(path / "missing")
            state, calls = invoke(path, f"{row} {recovery=} {scenario=} {pop=}", RECOVER if recovery else DEPLOY,
                                  "active" if scenario == "dangling" else scenario,
                                  fault="" if recovery else "reset:17")
            contained(state, "deploy-step-incomplete" if recovery else "reset")
            assert ("Scheduler stopped: verified." in state["out"]) == verified
            assert ("/legacy-orphan" in cgroups(calls)) == checked
            assert (["systemctl", "stop", "bts-scheduler.service"] in calls) == (scenario != "active")


MOVED = inject_shim(r'''
if scenario == "inverse-system-enabled" and command == "systemctl" and args[0] == "is-enabled":
    print("disabled")
    raise SystemExit(1)
''').replace('scenario == "inverse-system-enabled"', 'scenario.startswith("moved-legacy")')


def moved():
    for scope, (pre, post) in product(("user", "legacy"), ((1, 0), (0, 1), (0, 0))):
        with box(shim=MOVED) as path:
            prefix = "l" if scope == "legacy" else ""
            before, after = "/" + prefix + "a", "/" + prefix + "b"
            group(path, before, pre)
            group(path, after, post)
            state, calls = invoke(path, f"K16 {scope=} {pre=} {post=}", scenario=f"moved-{scope}", fault="reset:17")
            contained(state)
            assert ("Scheduler stopped: verified." in state["out"]) == (pre == post == 0)
            assert before in cgroups(calls) and (after in cgroups(calls)) == (not pre)


def choices():
    for field in ("run", "attempt", "target", "old", "none"):
        with box() as path:
            state, _ = invoke(path, f"K17 committing read with mismatch {field}", fault=f"publish-committed:HUP:forge-{field}-committed")
            if field == "none":
                assert state["rc"] == 0 and state["link"] == "enabled" and not state["stopped"]
            else:
                contained(state, "interrupted")
    for change in ("none-committed", "target-committed", "old-begun"):
        with box() as path:
            state, _ = invoke(path, f"K17 beginning same IDs control {change}", fault=f"publish-begun:INT:forge-{change}")
            contained(state, "interrupted")


def recheck():
    for case in ("active", "residual", "marker-unwritable", "contain-rename-fails"):
        initial = record("contained", tail="head unverified unverified verified")
        with box(initial) as path:
            state, calls = invoke(path, f"K18 {case}", RECOVER, case)
            contained(state, None, paged=False, fresh_marker=case != "marker-unwritable")
            assert bool(cgroups(calls))
            assert ("Scheduler stopped: verified." in state["out"]) == (case != "residual")
            assert ("Failure heartbeat written: verified." in state["out"]) == (case != "marker-unwritable")
            assert ("containment re-checked." in state["out"]) == (case in ("active", "contain-rename-fails"))
            assert (state["attempt"] == initial) == (case == "contain-rename-fails")
    with box() as path:
        state, _ = invoke(path, "K21 commit succeeds, read 126, contained rename fails", scenario="commit-read-126-contain-fail")
        contained(state)
        assert state["attempt"] == record("committed") and "Check commit-record failed (126)." in state["out"]
        state, _ = invoke(path, "K21 recovery of surviving commit", RECOVER)
        assert "this attempt's record says committed." in state["out"] and "INCIDENT.md §4" in state["out"]
        assert not state["writes"] and not state["posts"]


def source():
    incident = (ROOT / "INCIDENT.md").read_text()
    checklist = (ROOT / "docs/ops/2027-season-start.md").read_text()
    a5 = next(line for line in checklist.splitlines() if line.startswith("| A5 "))
    b8a = next(line for line in checklist.splitlines() if line.startswith("| B8a "))
    assert "Follow B8a" in a5 and "do not edit" in a5.lower()
    assert b8a.index("stopped and its local boot link disabled") < b8a.index('pick_delivery = "dm"')
    assert b8a.index("confirm the scheduler is stopped", b8a.index("Let `bts fetch")) < b8a.index("season-contract enable")
    print("K6 A5 edit deferred, B8a checks boot hold before initial and automatic-cutover edits")
    left, right = (shell_function(script, "prove") for script in (DEPLOY, RECOVER))
    assert left == right
    for script in (DEPLOY, RECOVER):
        outer = script.replace(left, "")
        assert not re.search(r"\b(stop|marker|page)=verified\b", outer)
        assert re.findall(r"\b(?:stop|marker|page)=verified\b", left) == ["stop=verified", "marker=verified", "page=verified"]
        assert all(message in left and message not in outer for message in (
            "Scheduler stopped:", "Failure heartbeat written:", "Failure ping received HTTP 2xx:", "containment re-checked."))
    wire = shell_function(DEPLOY, "on_signal")
    assert "signal_rule" in wire and "read_record" not in wire and "publication" not in wire
    assert "signal_rule()" not in RECOVER
    print("K19 single identical proof body; handler delegates its decision, recovery has no signal decision")
    restore = incident.split("**Box/filesystem loss", 1)[1].split("## 5.", 1)[0]
    assert "enable + start" not in restore and "activate the\n  scheduler through §5's checked deploy" in restore
    for argv in ("--show", "--init <MLB state>"):
        assert f"bts saver-state {argv} --season <ET year> --picks-dir <scheduler picks dir>" in b8a
    print("K20 restore goes through checked deploy, both saver examples have season and picks dir")


if __name__ == "__main__":
    globals()[sys.argv[1]](*sys.argv[2:])
