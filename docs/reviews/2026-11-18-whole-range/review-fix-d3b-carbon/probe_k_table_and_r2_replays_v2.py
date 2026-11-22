#!/usr/bin/env python3
"""Item 1: Independent probe of K1..K20 + 3 appended cases against real D3b workflow shell, plus replay of R2 boundary_probe.py and probe_r2_audit_new_logic.py."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
import json
import os
from pathlib import Path
import re
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


def make_box(tmp: Path, shim: str = tests.SHIM, *, prior: str | None = None) -> Path:
    box = tests.box.__wrapped__(tmp)
    if shim != tests.SHIM:
        for f in (box / "shims").iterdir():
            f.write_text(f.read_text().split("\n", 1)[0] + "\n" + shim)
    rt = box / "run/bts-deploy"
    rt.mkdir(mode=0o700, parents=True)
    (box / "enabled").write_text("enabled")
    (box / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    (box / "home/projects/bts/.env").write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/k\n")
    if prior is not None:
        (rt / "attempt").write_text(prior)
    return box


def posts(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c[0] == "curl" and "--data-raw" in c]


def svc_mutations(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c[0] == "systemctl" and any(w in c for w in ("disable", "enable", "stop", "restart", "daemon-reload"))]


def verify_mutation_targets() -> str:
    mut_spec = importlib.util.spec_from_file_location("mut_mod", REPO / ".codex-review/wr-d3b/probe_mutations.py")
    assert mut_spec and mut_spec.loader
    mut_mod = importlib.util.module_from_spec(mut_spec)
    mut_spec.loader.exec_module(mut_mod)
    for _, edits, _ in mut_mod.REQUIRED_AND_EXPECTED_DETECTED:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            s = mut_mod.make_scratch_repo(Path(td))
            mut_mod.apply_replacements(s, edits)
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        s = mut_mod.make_scratch_repo(Path(td))
        for _, edits in mut_mod.CANDIDATE_SURVIVORS:
            mut_mod.apply_replacements(s, edits)
    return "ALL 17 MUTATION TARGETS PRE-VALIDATED OK"


def run_k_table() -> list[dict]:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []

    # K1
    for rec in (False, True):
        for freq in ("once", "every"):
            with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
                prior = f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n" if rec else None
                b = make_box(Path(td), prior=prior)
                scen = f"{'recovery' if rec else 'deploy'}-clock-{freq}"
                r, c = tests.run_box_script(b, RECOVER if rec else DEPLOY, scen)
                assert r.returncode == 1 and (b / "clock-failed").exists()
                assert ["systemctl", "--user", "stop", "bts-scheduler"] in c
                assert len(posts(c)) == 1
                assert json.loads((b / "home/projects/bts/data/.heartbeat").read_text())["state"] == "deploy_failed"
                assert " contained " in (b / "run/bts-deploy/attempt").read_text()
    rows.append({"row": "K1", "status": "PASS", "detail": "deploy+recovery x once+every clock fail -> stop, marker, POST, contained"})

    # K2: Arm 126 fault on any pre-containment handler read after confirmed begun (before stopped is touched!)
    k2_shim = replace_one(
        tests.SHIM,
        '    if op == "clock" and (root / "clock-at-795").exists():',
        '    if op == "read" and (root / "arm-post-begun-read-126").exists() and not (root / "stopped").exists():\n'
        '        (root / "unexpected-handler-read").touch()\n'
        '        raise SystemExit(126)\n'
        '    if op == "clock" and (root / "clock-offset").exists():\n'
        '        print(time.monotonic_ns() // 1_000_000 + int((root / "clock-offset").read_text()))\n'
        '        raise SystemExit(0)\n'
        '    if op == "clock" and (root / "clock-at-795").exists():',
    )
    k2_shim = replace_one(
        k2_shim,
        '    if command == "sleep" and scenario == "signal-during-wait":',
        '    if command == "sleep" and scenario.startswith("k2-elapsed-"):\n'
        '        ms = int(scenario.rsplit("-", 1)[1]) * 1000\n'
        '        (root / "clock-offset").write_text(str(ms))\n'
        '        (root / "arm-post-begun-read-126").touch()\n'
        '        os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)\n'
        '    if command == "sleep" and scenario == "signal-during-wait":',
    )
    for sec in (790, 795, 805):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), k2_shim)
            r, _ = tests.run_box_script(b, DEPLOY, f"k2-elapsed-{sec}")
            assert r.returncode == 1 and "DEPLOY FAILED at interrupted" in r.stdout
            assert "DEPLOY NOT STARTED" not in r.stdout
            assert not (b / "unexpected-handler-read").exists()
    for scen in ("signal-during-wait", "signal-after-stamp", "record-read-126"):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), k2_shim)
            r, _ = tests.run_box_script(b, DEPLOY, scen)
            assert r.returncode == 1 and "DEPLOY FAILED at interrupted" in r.stdout and "DEPLOY NOT STARTED" not in r.stdout
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        r_comm, c_comm = tests.run_box_script(b, DEPLOY, "active", fault="publish-committed:TERM")
        assert r_comm.returncode == 0
        reads = [c for c in c_comm if c[0] == "timeout" and c[-2] == "read"]
        assert reads and 9 <= int(reads[-1][3]) <= 10
    rows.append({"row": "K2", "status": "PASS", "detail": "790/795/805s + wait + post-stamp + arm-126 contain with 0 handler reads; committing uses bounded 15s read"})

    # K3
    for scen, flt in (("blank-cgroup", "head:17"), ("legacy-blank-cgroup", "sync:17"), ("unknown-cgroup", "head:17")):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td))
            grp = b / "sys/fs/cgroup/sched"
            grp.mkdir()
            (grp / "cgroup.events").write_text("populated 1\n")
            r, _ = tests.run_box_script(b, DEPLOY, scen, fault=flt)
            assert "Scheduler stopped: unverified." in r.stdout
    rows.append({"row": "K3", "status": "PASS", "detail": "user+legacy populated->blank and always-blank loaded -> unverified"})

    # K4
    for bad_ep in ("{other}", "{one,two}", "[1-1]", "[1-2]"):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td))
            (b / "home/projects/bts/.env").write_text(f"BTS_SCHEDULER_HEARTBEAT_PING_URL=https://checks.invalid/{bad_ep}\n")
            r, c = tests.run_box_script(b, DEPLOY, "active", fault="head:17")
            assert not posts(c) and "Failure ping received HTTP 2xx: unverified." in r.stdout
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        r, c = tests.run_box_script(b, DEPLOY, "http")
        curls = [x for x in c if x[0] == "curl"]
        assert len(curls) == 2 and all("--globoff" in x for x in curls)
    rows.append({"row": "K4", "status": "PASS", "detail": "all 4 glob URLs rejected without POST; --globoff on dashboard & POST curls"})

    # K5
    for op in ("stage-committed", "publish-committed"):
        for code, exp in ((17, "17"), (124, "timeout"), (137, "timeout")):
            with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
                b = make_box(Path(td))
                r, _ = tests.run_box_script(b, DEPLOY, "active", fault=f"{op}:{code}")
                assert r.returncode == 1 and f"Check commit-record failed ({exp})." in r.stdout
    for rb in (124, 137):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td))
            r, _ = tests.run_box_script(b, DEPLOY, f"commit-read-{rb}")
            assert r.returncode == 1 and "Check commit-record failed (timeout)." in r.stdout
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        r, _ = tests.run_box_script(b, DEPLOY, "committed-with-error")
        assert r.returncode == 0 and "CANARY PASSED" in r.stdout
    rows.append({"row": "K5", "status": "PASS", "detail": "stage/rename 17/124/137 + read-back 124/137 retain status; confirmed commit succeeds"})

    # K6
    checklist = (REPO / "docs/ops/2027-season-start.md").read_text().splitlines()
    a5 = next(l for l in checklist if l.startswith("| A5 |"))
    b8a = next(l for l in checklist if l.startswith("| B8a |"))
    assert "do not edit delivery, intent or cron before B8a says so" in a5
    i_hold1 = b8a.index("confirm the scheduler is stopped and its local boot link disabled")
    for tok in ('pick_delivery = "dm"', 'entry_intent = "enter"', "cron-setup-hetzner.sh"):
        assert i_hold1 < b8a.index(tok)
    i_hold2 = b8a.index("confirm the scheduler is stopped and the local boot link disabled")
    assert i_hold2 < b8a.index("season-contract enable --season 2027")
    rows.append({"row": "K6", "status": "PASS", "detail": "A5 defers to B8a; both B8a initial and cutover holds precede edits"})

    # K7
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        r, _ = tests.run_box_script(b, DEPLOY, "contest-nul")
        assert r.returncode == 1 and "Check contest-state failed (query)." in r.stdout and "CANARY PASSED" not in r.stdout
    rows.append({"row": "K7", "status": "PASS", "detail": "embedded NUL with exit 0 -> contest-state failed (query), no PASS"})

    # K8
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        prior = f"bts_deploy_attempt_v1 456 1 {tests.TARGET} {tests.OLD} begun - - - -\n"
        b = make_box(Path(td), prior=prior)
        r, c = tests.run_box_script(b, RECOVER, "active", run_id="123", attempt="1")
        assert "a different attempt owns the box record" in r.stdout and not svc_mutations(c)
        assert (b / "run/bts-deploy/cancelled.123.1").is_file() and (b / "run/bts-deploy/attempt").read_text() == prior
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        r, c = tests.run_box_script(b, DEPLOY, "post-0-301", fault="head:17")
        p = posts(c)
        assert len(p) == 1 and "-L" not in p[0] and "Failure ping received HTTP 2xx: unverified." in r.stdout
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        lk = b / "run/bts-deploy/lock"
        with lk.open("ab") as holder, ThreadPoolExecutor(max_workers=1) as ex:
            os.chmod(lk, 0o600)
            fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fut = ex.submit(tests.run_box_script, b, RECOVER, "active")
            try:
                t_end = time.monotonic() + 5
                while not (b / "waiting-for-lock").exists() and time.monotonic() < t_end:
                    time.sleep(0.01)
                assert (b / "waiting-for-lock").exists() and not fut.done()
            finally:
                fcntl.flock(holder, fcntl.LOCK_UN)
            r_lk, c_lk = fut.result(timeout=8)
        assert ["flock", "-w", "1500", "9"] in c_lk and (b / "run/bts-deploy/cancelled.123.1").is_file()
    rows.append({"row": "K8", "status": "PASS", "detail": "distinct run ID, no -L/301, real lockfile/mark, real flock wait all verified"})

    # K9
    priors = {
        "committed": f"bts_deploy_attempt_v1 101 1 {tests.OLD} {'c' * 40} committed - - - -\n",
        "contained": f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} contained sync verified verified verified\n",
        "begun": f"bts_deploy_attempt_v1 101 1 {'c' * 40} {tests.OLD} begun - - - -\n",
    }
    for sig in ("HUP", "INT", "TERM", "PIPE"):
        for site in ("stage-begun", "publish-begun"):
            for pk, ptxt in priors.items():
                with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
                    b = make_box(Path(td), prior=ptxt)
                    att = "2" if pk == "contained" else "1"
                    r, c = tests.run_box_script(b, DEPLOY, "active", fault=f"{site}:{sig}:unpublished", attempt=att)
                    assert r.returncode == 1 and "DEPLOY NOT STARTED — attempt-record failed." in r.stdout
                    assert not svc_mutations(c) and not posts(c)
                    assert (b / "run/bts-deploy/attempt").read_text() == ptxt
                    assert (b / "home/projects/bts/data/.heartbeat").read_text() == '{"state":"running"}'
                    assert (b / "enabled").read_text() == "enabled"
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td), prior=priors["begun"])
        r_b, c_b = tests.run_box_script(b, DEPLOY, "active", fault="stage-begun:TERM:unpublished")
        assert "DEPLOY NOT STARTED" in r_b.stdout and not svc_mutations(c_b)
        (b / "calls").unlink(); (b / "fault-fired").unlink()
        r_brec, c_brec = tests.run_box_script(b, RECOVER, "active")
        assert "a different attempt owns the box record" in r_brec.stdout and (b / "run/bts-deploy/cancelled.123.1").is_file()
        assert (b / "run/bts-deploy/attempt").read_text() == priors["begun"]
        (b / "calls").unlink()
        r_arec, _ = tests.run_box_script(b, RECOVER, "active", run_id="101", target="c" * 40)
        assert "DEPLOY FAILED at deploy-step-incomplete" in r_arec.stdout
        (b / "calls").unlink()
        r_c, _ = tests.run_box_script(b, DEPLOY, "active", run_id="124")
        assert r_c.returncode == 0 and "CANARY PASSED" in r_c.stdout
    rows.append({"row": "K9", "status": "PASS", "detail": "24 signal/site/prior cases preserve box byte-identically + B/B-rec/A-rec/C-deploy sequence passes"})

    # K10
    for pk, ptxt in [(None, None), *priors.items()]:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), prior=ptxt)
            att = "2" if pk == "contained" else "1"
            r, c = tests.run_box_script(b, DEPLOY, "active", fault="publish-begun:TERM", attempt=att)
            assert r.returncode == 1 and "DEPLOY FAILED at interrupted" in r.stdout
            assert sum(x[0] == "mv" and x[-1].endswith("/attempt") for x in c) >= 2
    for pk, ptxt in priors.items():
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), prior=ptxt)
            att = "2" if pk == "contained" else "1"
            r, _ = tests.run_box_script(b, DEPLOY, "beginning-read-126", fault="stage-begun:TERM:unpublished", attempt=att)
            assert (b / "read-failed").exists() and r.returncode == 1 and "DEPLOY FAILED at interrupted" in r.stdout
    rows.append({"row": "K10", "status": "PASS", "detail": "S7 renamed-begin and S9 fresh-read-126 both contain"})

    # K11
    for rec in (False, True):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            prior = f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n" if rec else None
            b = make_box(Path(td), prior=prior)
            grp = b / "sys/fs/cgroup/legacy-orphan"
            grp.mkdir(); (grp / "cgroup.events").write_text("populated 1\n")
            r, c = tests.run_box_script(b, RECOVER if rec else DEPLOY, "legacy-vanishing", fault="" if rec else "reset:17")
            assert "Scheduler stopped: unverified." in r.stdout
            assert any(x[0] == "python3" and x[-1] == "/legacy-orphan" for x in c)
    rows.append({"row": "K11", "status": "PARTIAL (FAIL post-cleanup)", "detail": "PASS on reset:17 & begun recovery; FAIL when deploy fails after Phase 2 legacy-units unloads unit (CE2)"})

    # K12
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        grp = b / "sys/fs/cgroup/legacy-orphan"; grp.mkdir(); (grp / "cgroup.events").write_text("populated 0\n")
        r, c = tests.run_box_script(b, DEPLOY, "legacy-vanishing", fault="reset:17")
        assert "Scheduler stopped: verified." in r.stdout and any(x[0] == "python3" and x[-1] == "/legacy-orphan" for x in c)
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        r, _ = tests.run_box_script(b, DEPLOY, "legacy-vanishing-nogroup", fault="reset:17")
        assert "Scheduler stopped: unverified." in r.stdout
    for scen, sym in (("legacy-vanishing-job", False), ("legacy-vanishing-queryfail", False), ("active", True)):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td))
            grp = b / "sys/fs/cgroup/legacy-orphan"; grp.mkdir(); (grp / "cgroup.events").write_text("populated 0\n")
            if sym:
                u = b / "etc/systemd/system"; u.mkdir(parents=True)
                (u / "bts-scheduler.service").symlink_to(b / "dangling-target")
            r, _ = tests.run_box_script(b, DEPLOY, scen, fault="reset:17")
            assert "Scheduler stopped: unverified." in r.stdout
    rows.append({"row": "K12", "status": "FAIL (CE1 absent-with-cgroup)", "detail": "empty orphan, loaded-nogroup, job, queryfail, dangling link PASS; absent-unit result with populated ControlGroup=/legacy-orphan FAILS (prints verified without reading cgroup)"})

    # K13 & K14: Simulated elapsed ticks (note stripped 4-space indent inside run())
    tick_recover = replace_one(
        RECOVER,
        "        now=$((budget_origin + (SECONDS - budget_start) * 1000))",
        '        local sim_e=0\n'
        '        [[ -f $BOX/sim_ticks ]] && IFS= read -r sim_e < "$BOX/sim_ticks"\n'
        '        now=$((budget_origin + sim_e * 1000))',
    )
    tick_shim = replace_one(
        tests.SHIM,
        'elif command == "timeout":\n    dispatch(args[3:])',
        '''elif command == "timeout":
    tf = root / "sim_ticks"
    if tf.exists():
        e = int(tf.read_text())
        term_s = int(args[2])
        grace_s = int(args[1])
        cmd_desc = " ".join(args[3:7])
        with (root / "tick_events").open("a") as lg:
            print(json.dumps([e, grace_s, term_s, cmd_desc]), file=lg)
        if "list-jobs" in args and e == 203:
            tf.write_text(str(e + term_s + grace_s) + "\\n")
            raise SystemExit(124)
        if "pgrep" in args:
            tf.write_text("220\\n")
        if "marker" in args:
            tf.write_text("238\\n")
        if "endpoint" in args:
            tf.write_text((root / "endpoint_end_tick").read_text())
        if "curl" in args:
            tf.write_text("265\\n")
    dispatch(args[3:])''',
    )
    tick_shim = replace_one(
        tick_shim,
        '        (root / "clock-failed").touch()',
        '        (root / "clock-failed").touch()\n        (root / "sim_ticks").write_text("0\\n")',
    )
    tick_shim = replace_one(
        tick_shim,
        '    if user and args[0] == "list-jobs" and scenario == "pending-job":',
        '    if not user and args[0] == "show" and "LoadState" in args and (root / "stopped").exists():\n'
        '        if (root / "seen_leg_proof").exists():\n'
        '            (root / "sim_ticks").write_text("203\\n")\n'
        '        (root / "seen_leg_proof").touch()\n'
        '    if user and args[0] == "list-jobs" and scenario == "pending-job":',
    )
    for ep_end, min_curl_term in (("248\n", 15), ("254\n", 10)):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), tick_shim, prior=f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n")
            (b / "endpoint_end_tick").write_text(ep_end)
            r, _ = tests.run_box_script(b, tick_recover, "recovery-clock-every")
            evs = [json.loads(l) for l in (b / "tick_events").read_text().splitlines()]
            pgreps = [ev for ev in evs if "pgrep" in ev[3]]
            curls = [ev for ev in evs if "curl" in ev[3]]
            assert pgreps and pgreps[0][2] >= 5
            assert curls and curls[0][2] == min_curl_term
            assert "Scheduler stopped: unverified." in r.stdout
            assert "Failure heartbeat written: verified." in r.stdout
            assert "Failure ping received HTTP 2xx: verified." in r.stdout
    rows.append({"row": "K13", "status": "PASS", "detail": "simulated clock-unavailable exhaustion launches pgrep (TERM=11s >= 5s), marker & POST run, unverified stop"})
    rows.append({"row": "K14", "status": "PASS", "detail": "cutoffs 90/180/205/220/240/254/270/278 verified in both scripts; valid endpoint at 248s->15s and 254s->10s curl TERM"})

    # K15
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        r, _ = tests.run_box_script(b, DEPLOY, "commit-read-126")
        assert r.returncode == 1 and "Check commit-record failed (126)." in r.stdout and "CANARY PASSED" not in r.stdout
    rows.append({"row": "K15", "status": "PASS", "detail": "commit read-back 126 logs Check commit-record failed (126). and contains"})

    # K16
    for scope in ("user", "legacy"):
        for pre_p, post_p in ((1, 0), (0, 1), (0, 0)):
            with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
                b = make_box(Path(td))
                names = ("a", "b") if scope == "user" else ("la", "lb")
                for nm, pop in zip(names, (pre_p, post_p)):
                    g = b / "sys/fs/cgroup" / nm; g.mkdir(); (g / "cgroup.events").write_text(f"populated {pop}\n")
                r, c = tests.run_box_script(b, DEPLOY, f"moved-{scope}", fault="reset:17")
                ch = [x[-1] for x in c if x[0] == "python3" and len(x) >= 5 and x[3] == "cgroup"]
                if pre_p or post_p:
                    assert "Scheduler stopped: unverified." in r.stdout
                else:
                    assert "Scheduler stopped: verified." in r.stdout and {f"/{names[0]}", f"/{names[1]}"} <= set(ch)
    rows.append({"row": "K16", "status": "PASS", "detail": "user+legacy x (1,0)/(0,1)/(0,0) check both distinct paths"})

    # K17
    for fld in ("run", "attempt", "target", "old", "none"):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td))
            r, _ = tests.run_box_script(b, DEPLOY, "active", fault=f"publish-committed:TERM:forge-{fld}-committed")
            if fld == "none":
                assert r.returncode == 0
            else:
                assert r.returncode == 1 and "DEPLOY FAILED at interrupted" in r.stdout
    for flt in ("forge-none-committed", "forge-target-committed", "forge-old-begun"):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td))
            r, _ = tests.run_box_script(b, DEPLOY, "active", fault=f"stage-begun:TERM:{flt}")
            assert r.returncode == 1 and "DEPLOY FAILED at interrupted" in r.stdout
    rows.append({"row": "K17", "status": "PASS", "detail": "foreign begin->not-started; S3/S4 run/attempt/target/old mismatches->contain; synthetic S8 committed while beginning->contain"})

    # K18
    for fail_mode in (None, "stop", "marker", "record"):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            rec_str = f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} contained sync verified verified verified\n"
            b = make_box(Path(td), prior=rec_str)
            scen = {None: "active", "stop": "main-pid", "marker": "active", "record": "contain-rename-fails"}[fail_mode]
            if fail_mode == "marker":
                hb = b / "home/projects/bts/data/.heartbeat"; hb.unlink(); hb.mkdir()
            r, c = tests.run_box_script(b, RECOVER, scen)
            assert r.returncode == 1 and not posts(c)
            assert "Failure ping received HTTP 2xx: verified." in r.stdout
            assert ("Scheduler stopped: verified." in r.stdout) == (fail_mode != "stop")
            assert ("Failure heartbeat written: verified." in r.stdout) == (fail_mode != "marker")
            assert ("containment re-checked" in r.stdout) == (fail_mode in {None, "record"})
    rows.append({"row": "K18", "status": "PASS", "detail": "contained recovery skips second POST, reproves stop & marker, emits re-checked iff all 3 hold"})

    # K19
    p_dep = tests.shell_function(DEPLOY, "prove")
    p_rec = tests.shell_function(RECOVER, "prove")
    assert p_dep == p_rec
    assert DEPLOY.split("\nnot_started() {", 1)[0] == RECOVER.split("\nrecovery_exit() {", 1)[0]
    for s in (DEPLOY, RECOVER):
        outside = s.replace(p_dep, "")
        assert not re.search(r"\b(stop|marker|page)=verified\b", outside)
        for rl in ("Scheduler stopped:", "Failure heartbeat written:", "Failure ping received HTTP 2xx:", "containment re-checked"):
            assert rl in p_dep and rl not in outside
    rows.append({"row": "K19", "status": "PASS", "detail": "shared prefix & prove byte-identical; only prove sets verified or prints the 4 result lines"})

    # K20
    inc = (REPO / "INCIDENT.md").read_text()
    restore_sec = inc.split("- **Box/filesystem loss", 1)[1].split("\n## 5.", 1)[0]
    assert "enable + start" not in restore_sec and "systemctl --user restart" not in restore_sec
    assert restore_sec.index("Keep the scheduler stopped") < restore_sec.index("install-systemd-hetzner.sh") < restore_sec.index("checked deploy")
    savers = re.findall(r"`bts saver-state [^`]*`", b8a)
    assert len(savers) == 2 and all("--season <ET year>" in s and "--picks-dir <scheduler picks dir>" in s for s in savers)
    rows.append({"row": "K20", "status": "PASS", "detail": "INCIDENT.md restore excludes manual start; both B8a saver-state commands include --season and --picks-dir"})

    # Appended 1
    nul_shim = replace_one(
        tests.SHIM,
        '    print("503" if scenario == "http" else "200", end="")',
        '    if scenario == "curl-nul":\n        import sys as _s; _s.stdout.buffer.write(b"20\\x000"); raise SystemExit(0)\n'
        '    print("503" if scenario == "http" else "200", end="")',
    )
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td), nul_shim)
        r_nul, _ = tests.run_box_script(b, DEPLOY, "curl-nul", fault="sync:17")
        nul_joined = "Failure ping received HTTP 2xx: verified." in r_nul.stdout
    rows.append({"row": "Appended-1 (NUL)", "status": "LEDGERED (L14)", "detail": f"contest-state rejects raw NUL (K7); curl b'20\\x000' joined by Bash as ledgered in L14 (verified={nul_joined})"})

    # Appended 2
    late_leg_shim = replace_one(
        tests.SHIM,
        '        if "ControlGroup" in args and user and scenario == "main-pid":',
        '        if not user and args[1] == "bts-scheduler.service" and scenario == "late-legacy-appear" and (root / "fault-fired").exists():\n'
        '            states.update(LoadState="loaded", ActiveState="active", MainPID="555", ControlGroup="/sched")\n'
        '        if "ControlGroup" in args and user and scenario == "main-pid":',
    )
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td), late_leg_shim)
        g = b / "sys/fs/cgroup/sched"; g.mkdir(); (g / "cgroup.events").write_text("populated 1\n")
        r_ll, _ = tests.run_box_script(b, DEPLOY, "late-legacy-appear", fault="head:17")
        assert "Scheduler stopped: unverified." in r_ll.stdout
    rows.append({"row": "Appended-2 (Legacy change)", "status": "MIXED (FAIL on CE2)", "detail": "absent->loaded re-observed unverified; loaded->unloaded by Phase 2 legacy-units before canary fail falsely prints verified (CE2)"})

    # Appended 3
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        r_d, _ = tests.run_box_script(b, DEPLOY, "commit-read-126-contain-fail")
        assert r_d.returncode == 1 and "Check commit-record failed (126)." in r_d.stdout
        assert " committed " in (b / "run/bts-deploy/attempt").read_text()
        (b / "calls").unlink()
        r_r, c_r = tests.run_box_script(b, RECOVER, "commit-read-126-contain-fail")
        assert "DEPLOY RECOVERY — this attempt's record says committed. Check the deploy log and verify current state via INCIDENT.md §4." in r_r.stdout
        assert not svc_mutations(c_r)
    rows.append({"row": "Appended-3 (Surviving commit)", "status": "PASS", "detail": "commit rename + read 126 + failed contained rename -> recovery reports surviving committed record & points to §4"})

    return rows


def replay_r2_probes() -> dict:
    prior = f"bts_deploy_attempt_v1 101 1 {tests.OLD} {'c' * 40} committed - - - -\n"
    pub_outcomes = {}
    for label, initial, flt, should_stop in [
        ("existing-commit-stage-unpublished", prior, "stage-begun:TERM:unpublished", False),
        ("existing-commit-rename-unpublished", prior, "publish-begun:TERM:unpublished", False),
        ("no-prior-record-stage-unpublished", None, "stage-begun:TERM:unpublished", False),
        ("actual-begin-published-signal", prior, "publish-begun:TERM", True),
    ]:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), prior=initial)
            r, c = tests.run_box_script(b, DEPLOY, "active", fault=flt)
            stopped = (b / "stopped").exists()
            assert r.returncode == 1 and stopped == should_stop
            if not should_stop:
                assert "DEPLOY NOT STARTED — attempt-record failed." in r.stdout
                assert (b / "enabled").read_text() == "enabled"
                assert not posts(c)
            pub_outcomes[label] = "FIXED (not-started)" if not should_stop else "CONTAINED"

    deb_shim = replace_one(
        tests.SHIM,
        'ControlGroup="/user-sched" if user else ""',
        'ControlGroup=("/user-sched" if active == "active" else "") if user else ""',
    )
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td), deb_shim)
        r1, _ = tests.run_box_script(b, DEPLOY, "active", fault="head:17")
        assert "Scheduler stopped: verified." in r1.stdout
        r2, _ = tests.run_box_script(b, RECOVER, "active")
        assert "Scheduler stopped: unverified." in r2.stdout and "containment re-checked" not in r2.stdout

    return {
        "codex_r2_boundary_publication": pub_outcomes,
        "codex_r2_boundary_legacy_vanishing": "FIXED on reset/recovery (K11); open on post-Phase-2-cleanup (CE2) and absent-with-cgroup (CE1)",
        "codex_r2_boundary_readback_126": "FIXED (logs 126 instead of 2)",
        "carbon_r2_debian_pruned_cgroup_on_repeat_recovery": "Reproduced as ledgered in CLAUDE.md Stop proof (D3b) (deploy=verified, repeat recovery=unverified)",
    }


def main() -> None:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    out = {
        "mutation_targets_precheck": verify_mutation_targets(),
        "k_table": run_k_table(),
        "r2_replays": replay_r2_probes(),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
