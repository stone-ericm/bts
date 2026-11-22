#!/usr/bin/env python3
"""Round 2 Items 3 & 4: Adversarial audit of what 428cfa93 changed (legacy mark orders, stricter absence rule, URL rule, single final print, Python 3.11) + operator sentence verification."""
from __future__ import annotations

from ctypes import CDLL, POINTER, byref, c_char_p, c_int, c_uint, c_void_p
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

REPO = Path(__file__).resolve().parents[3]
SCRATCH_PARENT = REPO / ".codex-review/wr-d3b/r2/tmp"
PY311 = Path("/Users/eric/.local/share/uv/python/cpython-3.11.18-macos-aarch64-none/bin/python3.11")
SPEC = importlib.util.spec_from_file_location("workflow_tests", REPO / "tests/test_deploy_workflow.py")
assert SPEC and SPEC.loader
tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tests)
DEPLOY, RECOVER = tests.scripts_and_steps()[0]


def replace_one(source: str, old: str, new: str) -> str:
    assert source.count(old) == 1, f"expected 1 occurrence of {old!r}, got {source.count(old)}"
    return source.replace(old, new)


def inject_shim(fragment: str) -> str:
    return replace_one(tests.SHIM, 'if command == "sudo":', fragment + '\nif command == "sudo":')


def make_box(tmp: Path, *, shim: str | None = None, url: str = "https://hc-ping.com/550e8400-e29b-41d4-a716-446655440000") -> Path:
    path = tests.box.__wrapped__(tmp)
    if shim is not None:
        for exe in (path / "shims").iterdir():
            exe.write_text(exe.read_text().split("\n", 1)[0] + "\n" + shim)
    runtime = path / "run/bts-deploy"
    runtime.mkdir(mode=0o700, parents=True)
    (path / "enabled").write_text("enabled")
    (path / "home/projects/bts/data/.heartbeat").write_text('{"state":"running"}')
    (path / "home/projects/bts/.env").write_text(f"BTS_SCHEDULER_HEARTBEAT_PING_URL={url}\n")
    return path


def add_group(root: Path, group_path: str, population: int) -> None:
    loc = root / "sys/fs/cgroup" / group_path.lstrip("/")
    loc.mkdir(parents=True, exist_ok=True)
    (loc / "cgroup.events").write_text(f"populated {population}\n")


def posts(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c[0] == "curl" and "--data-raw" in c]


def svc_mutations(calls: list[list[str]]) -> list[list[str]]:
    return [c for c in calls if c[0] == "systemctl" and any(w in c for w in ("disable", "enable", "stop", "restart", "daemon-reload"))]


def curl_path(url: str, preserve: bool = False) -> str:
    lib = CDLL("/usr/lib/libcurl.4.dylib")
    lib.curl_url.restype = c_void_p
    lib.curl_url_cleanup.argtypes = [c_void_p]
    lib.curl_free.argtypes = [c_void_p]
    lib.curl_url_set.argtypes = [c_void_p, c_int, c_char_p, c_uint]
    lib.curl_url_get.argtypes = [c_void_p, c_int, POINTER(c_char_p), c_uint]
    handle = lib.curl_url()
    val = c_char_p()
    try:
        assert lib.curl_url_set(handle, 0, url.encode(), (1 << 4) if preserve else 0) == 0
        assert lib.curl_url_get(handle, 7, byref(val), 0) == 0
        return val.value.decode()
    finally:
        if val:
            lib.curl_free(val)
        lib.curl_url_cleanup(handle)


def probe_legacy_mark_orders() -> dict:
    out = {}

    # 1. Multi-run / re-run / recovery / reboot matrix
    # M1: Deploy (123,1) cleans legacy scheduler, fails at contest-state -> Recovery (123,1) -> Re-run (123,2) passes -> Late Recovery (123,1)
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        add_group(b, "/legacy-orphan", 1)
        frags = b / "etc/systemd/system"; frags.mkdir(parents=True)
        for u in ("bts-scheduler.service", "bts-dashboard.service"):
            (frags / u).write_text("[Service]\n")

        dep1, _ = tests.run_box_script(b, DEPLOY, "legacy-unloads", run_id="123", attempt="1", fault="contest-state:17")
        assert dep1.returncode == 1 and (b / "run/bts-deploy/legacy.123.1").is_file()
        assert "Scheduler stopped: unverified." in dep1.stdout

        (b / "calls").unlink(); (b / "fault-fired").unlink()
        rec1, _ = tests.run_box_script(b, RECOVER, "legacy-unloads", run_id="123", attempt="1")
        assert "Scheduler stopped: unverified." in rec1.stdout

        # Re-run attempt 2 on same run_id=123 passes cleanly despite legacy.123.1 remaining in /run/bts-deploy
        (b / "calls").unlink()
        dep2, _ = tests.run_box_script(b, DEPLOY, "legacy-unloads", run_id="123", attempt="2")
        assert dep2.returncode == 0 and "CANARY PASSED" in dep2.stdout
        assert not (b / "run/bts-deploy/legacy.123.2").exists()

        # Late recovery of attempt 1 does NOT act on attempt 2 despite legacy.123.1 existing
        (b / "calls").unlink()
        late_rec1, late_calls = tests.run_box_script(b, RECOVER, "legacy-unloads", run_id="123", attempt="1")
        assert "a different attempt owns the box record" in late_rec1.stdout
        assert not svc_mutations(late_calls)
        assert (b / "run/bts-deploy/cancelled.123.1").is_file()
        out["M1_deploy1_recover1_rerun2_pass_laterecover1"] = "PASS (attempt 2 passes; late recovery 1 fenced, no service mutation)"

    # M2: What happens if Attempt 1 (123,1) unloaded a legacy scheduler leaving an un-pgrep-able orphan in /legacy-orphan
    # (and reported `Scheduler stopped: unverified.`), AND a second attempt (123,2) or second run (124,1) is launched anyway
    # without Eric checking SSH per INCIDENT.md §4, and (123,2) fails at reset:17?
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        add_group(b, "/legacy-orphan", 1)
        frags = b / "etc/systemd/system"; frags.mkdir(parents=True)
        (frags / "bts-scheduler.service").write_text("[Service]\n")
        dep1, _ = tests.run_box_script(b, DEPLOY, "legacy-unloads", run_id="123", attempt="1", fault="contest-state:17")
        assert "Scheduler stopped: unverified." in dep1.stdout
        (b / "calls").unlink(); (b / "fault-fired").unlink(); (b / "head").write_text(tests.OLD)
        dep2_fail, _ = tests.run_box_script(b, DEPLOY, "legacy-unloads", run_id="123", attempt="2", fault="reset:17")
        out["M2_second_attempt_after_unverified_unload"] = {
            "attempt1_stop": "unverified",
            "attempt2_has_legacy_mark": (b / "run/bts-deploy/legacy.123.2").exists(),
            "attempt2_stop": "verified" if "Scheduler stopped: verified." in dep2_fail.stdout else "unverified",
            "note": "Attempt 2 began with legacy unit already unloaded before preflight (within L8/L13; Attempt 1 paged unverified requiring SSH check per §4)",
        }

    # M3: Reboot clears /run/bts-deploy (both attempt and legacy.123.1) before Recovery (123,1)
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        add_group(b, "/legacy-orphan", 0)
        frags = b / "etc/systemd/system"; frags.mkdir(parents=True)
        (frags / "bts-scheduler.service").write_text("[Service]\n")
        dep1, _ = tests.run_box_script(b, DEPLOY, "legacy-unloads", fault="restart:17")
        assert (b / "run/bts-deploy/legacy.123.1").is_file()
        shutil.rmtree(b / "run/bts-deploy")
        (b / "calls").unlink()
        rec_boot, calls_boot = tests.run_box_script(b, RECOVER, "legacy-unloads")
        assert "DEPLOY RECOVERY — no phase-2 record is present. Check the deploy log and the host's boot time." in rec_boot.stdout
        assert not svc_mutations(calls_boot)
        assert (b / "run/bts-deploy/cancelled.123.1").is_file()
        out["M3_reboot_clears_run_before_recovery"] = "PASS (no service change; cancelled.123.1 written)"

    # M4: Dashboard-only legacy unit (bts-dashboard.service loaded, bts-scheduler.service not-found)
    dash_only_shim = inject_shim(r'''
if command == "systemctl" and args[:2] == ["show", "bts-dashboard.service"] and scenario == "legacy-dash-only":
    loaded = "not-found" if (root / "dash-removed").exists() else "loaded"
    for k, v in dict(Id="bts-dashboard.service", LoadState=loaded, ActiveState="inactive", MainPID="0").items():
        if k in args:
            print(f"{k}={v}")
    raise SystemExit(0)
if command == "rm" and scenario == "legacy-dash-only" and args[-1].endswith("bts-dashboard.service"):
    (root / "dash-removed").touch()
''')
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td), shim=dash_only_shim)
        frags = b / "etc/systemd/system"; frags.mkdir(parents=True)
        (frags / "bts-dashboard.service").write_text("[Service]\n")
        res_dash, calls_dash = tests.run_box_script(b, DEPLOY, "legacy-dash-only", fault="restart:17")
        assert ["systemctl", "daemon-reload"] in calls_dash
        assert not (b / "run/bts-deploy/legacy.123.1").exists()
        assert "Scheduler stopped: verified." in res_dash.stdout
        out["M4_dashboard_only_legacy_unit"] = "PASS (cleans legacy dashboard without marking legacy scheduler; stop=verified)"

    # M5: Signal during Phase-2 mark write; corrupt pre-existing mark entries; idempotent valid pre-existing mark
    sig_mark_loaded_shim = inject_shim(r'''
if command == "python3" and len(args) > 2 and args[2] == "mark" and scenario.startswith("signal-on-legacy-mark") and not (root / "sig-sent").exists():
    (root / "sig-sent").touch()
    os.kill(int((root / "owner.pid").read_text()), signal.SIGTERM)
if command == "systemctl" and args[:2] == ["show", "bts-scheduler.service"] and scenario.startswith("signal-on-legacy-mark"):
    stopped = (root / "legacy-stopped").exists()
    fields = dict(Id="bts-scheduler.service", LoadState="loaded",
                  ActiveState="inactive" if stopped else "active",
                  MainPID="0" if stopped else "1234", ControlGroup="/legacy-orphan")
    for k in args:
        if k in fields:
            print(f"{k}={fields[k]}")
    raise SystemExit(0)
if command == "systemctl" and args == ["is-enabled", "bts-scheduler.service"] and scenario.startswith("signal-on-legacy-mark"):
    print("disabled")
    raise SystemExit(1)
''')
    for pop, exp_ver in ((0, True), (1, False)):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), shim=sig_mark_loaded_shim)
            add_group(b, "/legacy-orphan", pop)
            frags = b / "etc/systemd/system"; frags.mkdir(parents=True)
            (frags / "bts-scheduler.service").write_text("[Service]\n")
            res_s, calls_s = tests.run_box_script(b, DEPLOY, f"signal-on-legacy-mark-{pop}")
            assert res_s.returncode == 1 and "DEPLOY FAILED at interrupted" in res_s.stdout
            assert (b / "run/bts-deploy/legacy.123.1").is_file()
            assert ["systemctl", "daemon-reload"] not in calls_s
            assert ("Scheduler stopped: verified." in res_s.stdout) == exp_ver
    out["M5_signal_during_mark_write"] = "PASS (mark finishes under supervise, daemon-reload not run, loaded proof checks /legacy-orphan)"

    # M6: Pre-existing corrupt vs valid legacy.123.1 before Phase 2
    for kind in ("dangling-symlink", "fifo", "bad-mode", "non-empty", "valid-empty"):
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td))
            add_group(b, "/legacy-orphan", 0)
            frags = b / "etc/systemd/system"; frags.mkdir(parents=True)
            (frags / "bts-scheduler.service").write_text("[Service]\n")
            mk = b / "run/bts-deploy/legacy.123.1"
            if kind == "dangling-symlink":
                mk.symlink_to(b / "nonexistent")
            elif kind == "fifo":
                os.mkfifo(mk)
            elif kind == "bad-mode":
                mk.touch(mode=0o644)
                os.chmod(mk, 0o644)
            elif kind == "non-empty":
                mk.write_bytes(b"x")
                os.chmod(mk, 0o600)
            else:
                mk.touch(mode=0o600)
                os.chmod(mk, 0o600)
            res_m, calls_m = tests.run_box_script(b, DEPLOY, "legacy-unloads")
            if kind == "valid-empty":
                assert res_m.returncode == 0 and "CANARY PASSED" in res_m.stdout
            else:
                assert res_m.returncode == 1 and "DEPLOY FAILED at legacy-units" in res_m.stdout
                assert ["systemctl", "daemon-reload"] not in calls_m
    out["M6_preexisting_mark_entries"] = "PASS (corrupt symlink/fifo/bad-mode/non-empty fail closed before legacy mutation; valid 0600 empty mark is idempotent and passes)"

    return out


def probe_stricter_absence_rule() -> dict:
    out = {}

    # 1. Ordinary box states that NOW give `unverified` where `verified` is actually true on the box:
    # State U1: Clean October-snapshot legacy cleanup (populated 0, no processes left), followed by canary failure (e.g. contest-state:1)
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        add_group(b, "/legacy-orphan", 0)
        frags = b / "etc/systemd/system"; frags.mkdir(parents=True)
        for u in ("bts-scheduler.service", "bts-dashboard.service"):
            (frags / u).write_text("[Service]\n")
        dep_u1, _ = tests.run_box_script(b, DEPLOY, "legacy-unloads", fault="contest-state:17")
        (b / "calls").unlink()
        rec_u1, _ = tests.run_box_script(b, RECOVER, "legacy-unloads")
        assert "Scheduler stopped: unverified." in dep_u1.stdout
        assert "Scheduler stopped: unverified." in rec_u1.stdout
        out["U1_clean_legacy_cleanup_then_canary_fail"] = "Confirmed conservative unverified (was verified on 90a9b6f0; scheduler is genuinely stopped with populated 0, marked unverified because unloaded unit's cgroup was not proved before daemon-reload)"

    # State U2: Modern box (no legacy unit ever installed), transient 1-shot failure of first-sample ControlGroup query in contain
    transient_cg_shim = inject_shim(r'''
if command == "systemctl" and args == ["show", "bts-scheduler.service", "-p", "ControlGroup"] and scenario == "transient-first-legacy-cg":
    if not (root / "first-cg-failed").exists():
        (root / "first-cg-failed").touch()
        raise SystemExit(1)
''')
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td), shim=transient_cg_shim)
        res_u2, _ = tests.run_box_script(b, DEPLOY, "transient-first-legacy-cg", fault="reset:17")
        assert "Scheduler stopped: unverified." in res_u2.stdout
        out["U2_absent_unit_transient_first_group_query_error"] = "Confirmed conservative unverified (first-sample ControlGroup query failure blocks absence shortcut, and post-stop blank ControlGroup has no path)"

    # State U3: Real-Debian systemctl exit codes (disable exits 1, stop exits 5 on not-found unit) when not-found unit reports empty cgroup
    real_debian_shim = inject_shim(r'''
if command == "systemctl" and args[:2] == ["show", "bts-scheduler.service"] and scenario == "debian-ghost-empty":
    for k, v in dict(Id="bts-scheduler.service", LoadState="not-found", ActiveState="inactive", MainPID="0", ControlGroup="/legacy-group").items():
        if k in args:
            print(f"{k}={v}")
    raise SystemExit(0)
if command == "systemctl" and args == ["disable", "bts-scheduler.service"] and scenario == "debian-ghost-empty":
    raise SystemExit(1)
if command == "systemctl" and args == ["stop", "bts-scheduler.service"] and scenario == "debian-ghost-empty":
    raise SystemExit(5)
''')
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td), shim=real_debian_shim)
        add_group(b, "/legacy-group", 0)
        res_u3, _ = tests.run_box_script(b, DEPLOY, "debian-ghost-empty", fault="reset:17")
        assert "Scheduler stopped: unverified." in res_u3.stdout
        out["U3_real_debian_disable_stop_rc_on_not_found_with_empty_group"] = "Confirmed conservative unverified (legacy_absent=1 requires legacy_disabled$legacy_stopped==00, whereas real systemctl returns 1 and 5 on a nonexistent unit)"

    # 2. Reverse direction check: attempt to forge ANY false-verified with root cgroup "/", dangling legacy mark link, or state flip
    reverse_shim = inject_shim(r'''
if command == "systemctl" and args[:2] == ["show", "bts-scheduler.service"] and scenario == "root-cgroup":
    for k, v in dict(Id="bts-scheduler.service", LoadState="not-found", ActiveState="inactive", MainPID="0", ControlGroup="/").items():
        if k in args:
            print(f"{k}={v}")
    raise SystemExit(0)
''')
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td), shim=reverse_shim)
        res_rev1, _ = tests.run_box_script(b, DEPLOY, "root-cgroup", fault="reset:17")
        assert "Scheduler stopped: unverified." in res_rev1.stdout

    # Dangling symlink at legacy.123.1 during recovery of begun record still triggers legacy_cleaned (-L) -> unverified
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        (b / "head").write_text(tests.TARGET)
        (b / "run/bts-deploy/attempt").write_text(f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} begun - - - -\n")
        (b / "run/bts-deploy/legacy.123.1").symlink_to(b / "missing")
        res_rev2, _ = tests.run_box_script(b, RECOVER, "active")
        assert "Scheduler stopped: unverified." in res_rev2.stdout

    out["reverse_false_verified_search"] = "NONE FOUND (root cgroup '/', dangling legacy mark symlink, fragment/state mismatches all give unverified)"
    return out


def probe_url_rule_and_py311() -> dict:
    valid_urls = [
        ("hc_uuid", "https://hc-ping.com/550e8400-e29b-41d4-a716-446655440000", "/550e8400-e29b-41d4-a716-446655440000/fail"),
        ("hc_uuid_slash", "https://hc-ping.com/550e8400-e29b-41d4-a716-446655440000/", "/550e8400-e29b-41d4-a716-446655440000/fail"),
        ("hc_quoted", '"https://hc-ping.com/550e8400-e29b-41d4-a716-446655440000"', "/550e8400-e29b-41d4-a716-446655440000/fail"),
        ("hc_slug", "https://hc-ping.com/AbCdEfGh1234567890_xyz/bts-scheduler-heartbeat", "/AbCdEfGh1234567890_xyz/bts-scheduler-heartbeat/fail"),
        ("self_hosted_dots_in_segment_and_port", "https://hc.example.org:8443/v1.2/ping/bts.scheduler~1", "/v1.2/ping/bts.scheduler~1/fail"),
        ("bare_host", "https://hc.example.org", "/fail"),
    ]
    refused_urls = [
        ("percent_space_slug", "https://hc.example.org/ping/bts%20scheduler"),
        ("percent_tilde", "https://hc.example.org/ping/%7ebts"),
        ("percent_dot", "https://hc.example.org/%2e/check"),
        ("single_dot_seg", "https://hc.example.org/./check"),
        ("trailing_dot_seg", "https://hc.example.org/check/."),
        ("double_dot_seg", "https://hc.example.org/a/../b"),
        ("trailing_double_dot", "https://hc.example.org/a/.."),
    ]

    accepted_checked = []
    for name, u, expected_path in valid_urls:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), url=u)
            res, calls = tests.run_box_script(b, DEPLOY, "active", fault="reset:17")
            (argv,) = posts(calls)
            target_url = argv[-1]
            p_preserve = curl_path(target_url, preserve=True)
            p_normal = curl_path(target_url, preserve=False)
            assert p_preserve == p_normal == expected_path, (name, target_url, p_preserve, p_normal)
            assert "Failure ping received HTTP 2xx: verified." in res.stdout
            accepted_checked.append(name)

    refused_checked = []
    for name, u in refused_urls:
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            b = make_box(Path(td), url=u)
            res, calls = tests.run_box_script(b, DEPLOY, "active", fault="reset:17")
            assert not posts(calls), (name, u)
            assert "Failure ping received HTTP 2xx: unverified." in res.stdout
            refused_checked.append(name)

    # Also run inline fs_code under real Python 3.11.18
    m_start = DEPLOY.index("fs_code='\n") + len("fs_code='\n")
    m_end = DEPLOY.index("\n'\n\nclock()", m_start)
    fs_code = DEPLOY[m_start:m_end]
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        env_p = Path(td) / ".env"
        env_p.write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.com/550e8400-e29b-41d4-a716-446655440000\n")
        r311 = subprocess.run([str(PY311), "-c", fs_code, "endpoint", str(env_p)], capture_output=True, text=True, timeout=10)
        assert r311.returncode == 0 and r311.stdout.strip().endswith("/fail")
        env_p.write_text("BTS_SCHEDULER_HEARTBEAT_PING_URL=https://hc-ping.com/a/../b\n")
        r311_bad = subprocess.run([str(PY311), "-c", fs_code, "endpoint", str(env_p)], capture_output=True, text=True, timeout=10)
        assert r311_bad.returncode == 1

    # Also check closed output (EPIPE) on single final print in repeat recovery
    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
        b = make_box(Path(td))
        (b / "head").write_text(tests.TARGET)
        (b / "run/bts-deploy/attempt").write_text(
            f"bts_deploy_attempt_v1 123 1 {tests.TARGET} {tests.OLD} contained sync verified verified unverified\n"
        )
        res_epipe, calls_epipe = tests.run_box_script(b, RECOVER, "active", closed_output=True)
        assert res_epipe.returncode == 1
        assert len(posts(calls_epipe)) == 1
        assert (b / "run/bts-deploy/attempt").read_text().endswith("sync verified verified verified\n")

    return {
        "accepted_healthchecks_urls": accepted_checked,
        "refused_urls": refused_checked,
        "note_on_refused_percent": "Standard healthchecks.io UUID and ping-key+slug URLs use only [A-Za-z0-9_-] and are all accepted; only custom reverse-proxy paths containing percent-escapes (%XX) are refused.",
        "python311_inline_endpoint_check": "PASS",
        "single_print_epipe_repeat_recovery": "PASS",
    }


def probe_operator_sentences() -> list[dict]:
    inc = (REPO / "INCIDENT.md").read_text()
    claude = (REPO / "CLAUDE.md").read_text()
    season = (REPO / "docs/ops/2027-season-start.md").read_text()
    inst_sys = (REPO / "scripts/install-systemd-hetzner.sh").read_text()

    # Confirm install-systemd-hetzner.sh neither enables nor starts/restarts units
    active_lines = [l.strip() for l in inst_sys.splitlines() if l.strip() and not l.strip().startswith("#")]
    assert not any("enable" in l or ("restart" in l and not l.startswith("echo")) for l in active_lines)

    restore = inc.split("- **Box/filesystem loss", 1)[1].split("\n## 5.", 1)[0]
    assert restore.index("first confirm the scheduler is stopped and its local boot link") < restore.index("install restic")
    assert "whenever the attempt record does not hold a verified 2xx (a real earlier 2xx is sent again if its record rewrite failed)" in inc
    assert "- **L10.** The box can commit after Actions fails or cancels. Recovery then reports that this attempt's record says committed." in claude
    b8c = next(l for l in season.splitlines() if l.startswith("| B8c |"))
    assert "with the scheduler's local boot link disabled through the switch from maintenance to normal boot." in b8c

    return [
        {"loc": "INCIDENT.md:120-121", "verified": True},
        {"loc": "INCIDENT.md:131-133", "verified": True},
        {"loc": "INCIDENT.md:205", "verified": True},
        {"loc": "CLAUDE.md:135", "verified": True},
        {"loc": "docs/ops/2027-season-start.md:330 (B8c)", "verified": True},
    ]


def main() -> None:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    result = {
        "item3_legacy_mark_orders": probe_legacy_mark_orders(),
        "item3_stricter_absence_rule": probe_stricter_absence_rule(),
        "item3_url_and_single_print": probe_url_rule_and_py311(),
        "item4_operator_sentences": probe_operator_sentences(),
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
