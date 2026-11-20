#!/usr/bin/env python3
"""Item 7 probe (7b): hand release refusals, exact output, fsync-before-unlink durability order, and adversarial window deadlock."""
from __future__ import annotations

import fcntl
import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
INSTALLER = REPO / "scripts" / "install-deploy-guard-hetzner.sh"

OLD_SHA = "a" * 40
TARGET_SHA = "b" * 40
RUN_ID = "1795197580-hand01"


def extract_guard_source() -> str:
    lines = INSTALLER.read_text().splitlines(keepends=True)
    start, = (i for i, ln in enumerate(lines) if "<<'BTS_DEPLOY_HOLD_PYTHON'" in ln)
    end, = (i for i, ln in enumerate(lines) if ln.rstrip("\n") == "BTS_DEPLOY_HOLD_PYTHON")
    return "".join(lines[start + 1 : end])


def snapshot_dir(root: Path) -> dict[str, str]:
    snap = {}
    for p in sorted(root.iterdir()):
        if p.is_symlink():
            snap[p.name] = f"symlink:{os.readlink(p)}"
        elif p.is_fifo():
            snap[p.name] = "fifo"
        elif p.is_dir():
            snap[p.name] = "dir"
        else:
            snap[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    return snap


def make_env(work_dir: Path, name: str):
    box = work_dir / name
    guard = box / "guard"
    proj = box / "project"
    (proj / "data").mkdir(parents=True)
    guard.mkdir(parents=True, mode=0o755)
    for lk in ("writers.lock", "transaction.lock", "state.lock"):
        (guard / lk).touch(mode=0o644)

    mod = ModuleType(f"guard_{name}")
    exec(compile(extract_guard_source(), str(INSTALLER), "exec"), mod.__dict__)
    mod.ROOT = guard
    mod.UID = os.geteuid()
    mod.PROJECT = str(proj)
    now_ts = int(time.time())
    mod.time = SimpleNamespace(time=lambda: now_ts, monotonic=time.monotonic, sleep=lambda s: None)
    return box, guard, proj, mod, now_ts


def write_hold(mod: ModuleType, rfd: int, now_ts: int, state: str = "failed", step: str = "sync") -> dict:
    rec = {
        "schema": "bts_deploy_hold_v1",
        "deploy_id": RUN_ID,
        "owner_unit": f"bts-deploy-{RUN_ID}.service",
        "state": state,
        "step": step,
        "target": TARGET_SHA,
        "previous": OLD_SHA,
        "armed_at": now_ts - 60,
        "updated_at": now_ts - 30,
    }
    mod.publish_json(rfd, "hold", rec)
    return rec


def main() -> int:
    work_dir = Path(tempfile.mkdtemp(prefix="p7b_", dir=Path(__file__).resolve().parent))
    try:
        refusal_results = {}
        refusal_cases = [
            "no_hold", "state_pending", "state_activating", "hold_fifo", "hold_corrupt",
            "target_mismatch", "dirty_tree", "uv_exit_1", "uv_no_check_flag_exit_2",
            "uv_missing", "canary_fails", "reason_empty", "reason_spaces",
            "reason_newline", "reason_control_char", "reason_201_chars", "running_transaction",
            "audit_intent_fails",
        ]

        for case in refusal_cases:
            _, guard, proj, mod, now_ts = make_env(work_dir, f"ref_{case}")
            (proj / "data" / ".heartbeat").write_text(json.dumps({
                "state": "sleeping",
                "timestamp": datetime.fromtimestamp(now_ts - 10, timezone.utc).isoformat(),
                "sleeping_until": datetime.fromtimestamp(now_ts + 3600, timezone.utc).isoformat(),
            }))
            mod.bts_uid = lambda: 1000
            mod.preflight = lambda uid, target, *, fetch: (OLD_SHA if case == "target_mismatch" else TARGET_SHA, ["git"])
            if case == "canary_fails":
                mod.canary = lambda: (_ for _ in ()).throw(mod.Refused("canary-contest-state"))
            else:
                mod.canary = lambda: None

            def fake_call(argv, *, limit=10):
                if "status" in argv:
                    return (0, b" M src/bts/cli.py\n") if case == "dirty_tree" else (0, b"")
                if "sync" in argv:
                    if case == "uv_missing":
                        raise FileNotFoundError("uv missing")
                    if case == "uv_exit_1":
                        return 1, b"unsynced"
                    if case == "uv_no_check_flag_exit_2":
                        return 2, b"error: unexpected argument '--check' found"
                    return 0, b""
                raise AssertionError(argv)

            mod.call = fake_call
            if case == "audit_intent_fails":
                mod.release_audit = lambda *a, **kw: (_ for _ in ()).throw(OSError("disk full"))

            with mod.directory() as rfd:
                if case == "no_hold":
                    pass
                elif case == "state_pending":
                    write_hold(mod, rfd, now_ts, state="pending")
                elif case == "state_activating":
                    write_hold(mod, rfd, now_ts, state="activating")
                elif case == "hold_fifo":
                    os.mkfifo(guard / "hold")
                elif case == "hold_corrupt":
                    (guard / "hold").write_text("{corrupt")
                else:
                    write_hold(mod, rfd, now_ts, state="failed")

                before = snapshot_dir(guard)
                reason = {
                    "reason_empty": "",
                    "reason_spaces": "   ",
                    "reason_newline": "ok\nline2",
                    "reason_control_char": "bad\x07char",
                    "reason_201_chars": "r" * 201,
                }.get(case, "verified state and window")

                buf = io.StringIO()
                if case == "running_transaction":
                    with (guard / "transaction.lock").open("rb") as lf:
                        fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        with redirect_stdout(buf):
                            rc = mod.release_by_hand(rfd, reason)
                else:
                    with redirect_stdout(buf):
                        rc = mod.release_by_hand(rfd, reason)

                after = snapshot_dir(guard)
                assert rc == 1, f"{case} returned {rc}!"
                assert before == after, f"{case} mutated guard dir: {before} -> {after}"
                out = buf.getvalue().strip()
                if case.startswith("uv_"):
                    assert out == f"hand release refused: the environment could not be verified as synced for {TARGET_SHA[:7]}; use a deploy", out
                refusal_results[case] = out

        tool_script = work_dir / "bts-deploy-hold"
        tool_script.write_text(extract_guard_source())
        _, g_nr, _, mod_nr, now_nr = make_env(work_dir, "nonroot_cli")
        with mod_nr.directory() as rfd:
            write_hold(mod_nr, rfd, now_nr)
        before_nr = snapshot_dir(g_nr)
        nr_res = subprocess.run(
            [sys.executable, "-I", str(tool_script), "release", "--by-hand", "--reason", "test"],
            env={"BTS_TEST_DEPLOY_GUARD_DIR": str(g_nr)},
            stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
        )
        assert nr_res.returncode == 1 and snapshot_dir(g_nr) == before_nr

        _, g_ok, p_ok, mod_ok, now_ok = make_env(work_dir, "success_case")
        (p_ok / "data" / ".heartbeat").write_text(json.dumps({
            "state": "sleeping",
            "timestamp": datetime.fromtimestamp(now_ok - 10, timezone.utc).isoformat(),
            "sleeping_until": datetime.fromtimestamp(now_ok + 3600, timezone.utc).isoformat(),
        }))
        mod_ok.bts_uid = lambda: 1000
        mod_ok.preflight = lambda uid, target, *, fetch: (TARGET_SHA, ["git"])
        mod_ok.canary = lambda: None
        mod_ok.call = lambda argv, *, limit=10: (0, b"")

        syscall_trace = []
        real_fsync = os.fsync
        real_unlink = os.unlink
        real_replace = os.replace

        def traced_fsync(fd):
            hold_exists = (g_ok / "hold").exists()
            audit_lines = (g_ok / "releases.log").read_text().splitlines() if (g_ok / "releases.log").exists() else []
            syscall_trace.append(("fsync", f"hold_exists={hold_exists}", f"audit_events={[json.loads(x)['event'] for x in audit_lines]}"))
            return real_fsync(fd)

        def traced_unlink(path, *a, **kw):
            if path == "hold":
                audit_lines = (g_ok / "releases.log").read_text().splitlines()
                syscall_trace.append(("unlink_hold", f"audit_events={[json.loads(x)['event'] for x in audit_lines]}", f"last_release_exists={(g_ok / 'last-release').exists()}"))
            return real_unlink(path, *a, **kw)

        def traced_replace(src, dst, *a, **kw):
            syscall_trace.append(("replace", dst))
            return real_replace(src, dst, *a, **kw)

        os_ns = SimpleNamespace(**{k: getattr(os, k) for k in dir(os) if not k.startswith("__")})
        os_ns.fsync = traced_fsync
        os_ns.unlink = traced_unlink
        os_ns.replace = traced_replace
        mod_ok.os = os_ns

        with mod_ok.directory() as rfd:
            write_hold(mod_ok, rfd, now_ok, state="failed", step="sync")
            out_ok = io.StringIO()
            with redirect_stdout(out_ok):
                rc_ok = mod_ok.release_by_hand(rfd, "verified all checks")
        assert rc_ok == 0
        lines_ok = out_ok.getvalue().splitlines()
        assert len(lines_ok) == 2
        assert lines_ok[0].startswith("Deploy hold released by hand at ") and lines_ok[0].endswith(
            f"(deploy {RUN_ID}, failed at sync, target {TARGET_SHA[:7]}). Reason: verified all checks."
        )
        assert lines_ok[1] == "Cron and boot activation are now permitted. This command did not request a scheduler start."
        unlink_idx, = (i for i, ev in enumerate(syscall_trace) if ev[0] == "unlink_hold")
        assert syscall_trace[unlink_idx] == ("unlink_hold", "audit_events=['intent']", "last_release_exists=True")
        prior_fsyncs = [ev for ev in syscall_trace[:unlink_idx] if ev[0] == "fsync" and "['intent']" in ev[2]]
        assert len(prior_fsyncs) >= 2

        adv_window_cases = {}
        for label, hb_state in [
            ("absent_heartbeat", None),
            ("stale_sleeping_expired_15m_ago_while_held", {
                "state": "sleeping",
                "timestamp": datetime.fromtimestamp(now_ok - 3600, timezone.utc).isoformat(),
                "sleeping_until": datetime.fromtimestamp(now_ok - 900, timezone.utc).isoformat(),
            }),
            ("fresh_idle_end_of_day_all_games_over", {
                "state": "idle_end_of_day",
                "timestamp": datetime.fromtimestamp(now_ok - 60, timezone.utc).isoformat(),
            }),
            ("fresh_sleeping_20min_ahead_despite_30s_arg", {
                "state": "sleeping",
                "timestamp": datetime.fromtimestamp(now_ok - 10, timezone.utc).isoformat(),
                "sleeping_until": datetime.fromtimestamp(now_ok + 1200, timezone.utc).isoformat(),
            }),
        ]:
            _, _, p_adv, mod_adv, now_adv = make_env(work_dir, f"adv_{label}")
            if hb_state is not None:
                (p_adv / "data" / ".heartbeat").write_text(json.dumps(hb_state))
            mod_adv.bts_uid = lambda: 1000
            mod_adv.preflight = lambda uid, target, *, fetch: (TARGET_SHA, ["git"])
            mod_adv.canary = lambda: None
            mod_adv.call = lambda argv, *, limit=10: (0, b"")
            with mod_adv.directory() as rfd:
                write_hold(mod_adv, rfd, now_adv)
                buf_adv = io.StringIO()
                with redirect_stdout(buf_adv):
                    rc_adv = mod_adv.release_by_hand(rfd, "break-glass recovery")
            adv_window_cases[label] = {"rc": rc_adv, "out": buf_adv.getvalue().strip()}

        report = {
            "refusal_cases_verified_unchanged": len(refusal_results),
            "success_exact_lines": lines_ok,
            "syscall_durability_trace": syscall_trace,
            "adversarial_hand_release_window_deadlock": adv_window_cases,
        }
        out_path = Path(__file__).resolve().parent / "probe_7b_hand_release.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
