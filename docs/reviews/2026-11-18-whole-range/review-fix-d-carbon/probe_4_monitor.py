#!/usr/bin/env python3
"""Item 4 probe: check_heartbeat.py byte-identity vs b128854d, hold/release states, and oldest Python (/usr/bin/python3 3.9.6)."""
from __future__ import annotations

import fcntl
import io
import json
import os
import shutil
import subprocess
import sys
import tempfile
from contextlib import redirect_stderr, redirect_stdout
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import ModuleType, SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
BASE_MONITOR = Path(__file__).resolve().parent / "base" / "scripts" / "check_heartbeat.py"
HEAD_MONITOR = REPO / "scripts" / "check_heartbeat.py"
INSTALLER = REPO / "scripts" / "install-deploy-guard-hetzner.sh"
OLDEST_PYTHON = "/usr/bin/python3"


def extract_guard_source() -> str:
    lines = INSTALLER.read_text().splitlines(keepends=True)
    start, = (i for i, ln in enumerate(lines) if "<<'BTS_DEPLOY_HOLD_PYTHON'" in ln)
    end, = (i for i, ln in enumerate(lines) if ln.rstrip("\n") == "BTS_DEPLOY_HOLD_PYTHON")
    return "".join(lines[start + 1 : end])


def load_monitor(path: Path, now_utc: datetime) -> tuple[ModuleType, list]:
    mod = ModuleType(f"mon_{path.parent.name}")
    exec(compile(path.read_text(), str(path), "exec"), mod.__dict__)

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return now_utc.astimezone(tz) if tz else now_utc.replace(tzinfo=None)

    mod.datetime = Clock
    pings = []

    def fake_ping(url: str, suffix: str = "") -> None:
        pings.append(("GET", url + suffix))

    mod.ping = fake_ping
    if hasattr(mod, "ping_deploy_failure"):
        def fake_post_fail(url: str, reason: str) -> None:
            pings.append(("POST", url + "/fail", reason))
            print("Failure ping received HTTP 2xx: verified.")
        mod.ping_deploy_failure = fake_post_fail
    mod.read_nrestarts = lambda unit, run=None: mod._sim_nrestarts
    mod._sim_nrestarts = 2
    return mod, pings


def invoke_monitor(mod: ModuleType, argv: list[str], env_guard_dir: str | None = None) -> tuple[int, str, str]:
    out_buf, err_buf = io.StringIO(), io.StringIO()
    old_argv = sys.argv
    old_env = os.environ.get("BTS_TEST_DEPLOY_GUARD_DIR")
    if env_guard_dir is not None:
        os.environ["BTS_TEST_DEPLOY_GUARD_DIR"] = env_guard_dir
    else:
        os.environ.pop("BTS_TEST_DEPLOY_GUARD_DIR", None)
    sys.argv = argv
    try:
        with redirect_stdout(out_buf), redirect_stderr(err_buf):
            try:
                mod.main()
                rc = 0
            except SystemExit as exc:
                rc = int(exc.code) if exc.code is not None else 0
    finally:
        sys.argv = old_argv
        if old_env is None:
            os.environ.pop("BTS_TEST_DEPLOY_GUARD_DIR", None)
        else:
            os.environ["BTS_TEST_DEPLOY_GUARD_DIR"] = old_env
    return rc, out_buf.getvalue(), err_buf.getvalue()


def main() -> int:
    work_dir = Path(tempfile.mkdtemp(prefix="p4_", dir=Path(__file__).resolve().parent))
    try:
        now = datetime(2027, 7, 15, 18, 0, 0, tzinfo=timezone.utc)
        guard_dir = work_dir / "guard"
        guard_dir.mkdir(mode=0o755)
        for n in ("writers.lock", "transaction.lock", "state.lock"):
            (guard_dir / n).touch(mode=0o644)

        # 1. Parity corpus: no hold, no last-release -> byte-identical stdout, stderr, rc, and pings vs b128854d!
        iso = lambda delta_s: (now + timedelta(seconds=delta_s)).isoformat()
        corpus = [
            ("missing", None, 2, []),
            ("corrupt_json", "{bad", 2, []),
            ("running_fresh_0s", {"state": "running", "timestamp": iso(0)}, 2, []),
            ("running_boundary_300s", {"state": "running", "timestamp": iso(-300)}, 2, []),
            ("running_stale_301s", {"state": "running", "timestamp": iso(-301)}, 2, []),
            ("waiting_fresh_600s", {"state": "waiting_for_games", "timestamp": iso(-600)}, 2, []),
            ("waiting_stale_601s", {"state": "waiting_for_games", "timestamp": iso(-601)}, 2, []),
            ("sleeping_future_2h", {"state": "sleeping", "timestamp": iso(-3600), "sleeping_until": iso(7200)}, 2, []),
            ("sleeping_overrun_600s", {"state": "sleeping", "timestamp": iso(-7200), "sleeping_until": iso(-600)}, 2, []),
            ("sleeping_overrun_601s", {"state": "sleeping", "timestamp": iso(-7200), "sleeping_until": iso(-601)}, 2, []),
            ("sleeping_missing_wake", {"state": "sleeping", "timestamp": iso(0)}, 2, []),
            ("sleeping_bad_wake", {"state": "sleeping", "timestamp": iso(0), "sleeping_until": "nope"}, 2, []),
            ("idle_fresh_5400s", {"state": "idle_end_of_day", "timestamp": iso(-5400)}, 2, []),
            ("idle_stale_5401s", {"state": "idle_end_of_day", "timestamp": iso(-5401)}, 2, []),
            ("stalled", {"state": "stalled", "timestamp": iso(-5), "stage": "computing_features", "stalled_for_s": 950}, 2, []),
            ("unknown_state", {"state": "teleporting", "timestamp": iso(0)}, 2, []),
            ("bad_ts", {"state": "running", "timestamp": "not-a-ts"}, 2, []),
            ("churn_20m_fires", {"state": "running", "timestamp": iso(-10)}, 6, [{"ts": iso(-600), "n": 2}]),
            ("churn_60m_fires", {"state": "sleeping", "timestamp": iso(-600), "sleeping_until": iso(3600)}, 5, [{"ts": iso(-3000), "n": 2}]),
            ("churn_none_nrestarts", {"state": "running", "timestamp": iso(-10)}, None, [{"ts": iso(-300), "n": 1}]),
        ]

        total_parity_runs = 0
        for name, hb_payload, nrest, prior_samples in corpus:
            for ping_url in (None, "https://hc-ping.com/token123"):
                for no_churn in (False, True):
                    total_parity_runs += 1
                    sub = work_dir / f"c_{total_parity_runs}"
                    sub.mkdir()
                    for side in ("base", "head"):
                        sdir = sub / side
                        sdir.mkdir()
                        hb = sdir / ".heartbeat"
                        if isinstance(hb_payload, str):
                            hb.write_text(hb_payload)
                        elif isinstance(hb_payload, dict):
                            hb.write_text(json.dumps(hb_payload))
                        churn_file = sdir / "churn.json"
                        if prior_samples:
                            churn_file.write_text(json.dumps({"unit": "bts-scheduler", "samples": prior_samples}))

                    base_mod, base_pings = load_monitor(BASE_MONITOR, now)
                    head_mod, head_pings = load_monitor(HEAD_MONITOR, now)
                    base_mod._sim_nrestarts = nrest
                    head_mod._sim_nrestarts = nrest

                    def build_argv(sdir):
                        a = ["check_heartbeat.py", "--heartbeat-path", str(sdir / ".heartbeat"),
                             "--churn-state", str(sdir / "churn.json")]
                        if ping_url:
                            a += ["--ping-url", ping_url]
                        if no_churn:
                            a.append("--no-churn")
                        return a

                    b_res = invoke_monitor(base_mod, build_argv(sub / "base"), None)
                    h_res = invoke_monitor(head_mod, build_argv(sub / "head"), str(guard_dir))
                    assert b_res == h_res, f"Mismatch on {name} ping={ping_url} no_churn={no_churn}: {b_res} vs {h_res}"
                    assert base_pings == head_pings, f"Ping mismatch on {name}: {base_pings} vs {head_pings}"

        # 2. Hold states & release states (with fresh sleeping heartbeat!)
        tool_file = work_dir / "bts-deploy-hold"
        tool_file.write_text(extract_guard_source())
        tool_file.chmod(0o755)
        hb_fresh = work_dir / "fresh_sleep.json"
        hb_fresh.write_text(json.dumps({
            "state": "sleeping",
            "timestamp": (now - timedelta(seconds=10)).isoformat(),
            "sleeping_until": (now + timedelta(hours=2)).isoformat(),
        }))

        def run_hold_case(hold_setup, *, lock_held: bool = False, unit_state: bytes = b"inactive\n"):
            h_mod, h_pings = load_monitor(HEAD_MONITOR, now)
            h_mod.GUARD_TOOL = str(tool_file)
            orig_sub_run = h_mod.subprocess.run

            def shimmed_sub_run(argv, **kw):
                if argv[0] == "/usr/bin/python3" and argv[2] == str(tool_file):
                    return orig_sub_run([sys.executable, "-I", str(tool_file), *argv[3:]], **kw)
                if argv[0] == "/usr/bin/systemctl":
                    return SimpleNamespace(returncode=0, stdout=unit_state)
                raise AssertionError(argv)

            h_mod.subprocess.run = shimmed_sub_run
            # Patch time.time inside child guard invocation if needed via real current epoch for armed_at tests
            hold_setup(guard_dir)
            if lock_held:
                with (guard_dir / "transaction.lock").open("rb") as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    rc, out, _ = invoke_monitor(
                        h_mod,
                        ["check_heartbeat.py", "--heartbeat-path", str(hb_fresh),
                         "--ping-url", "https://hc-ping.com/token123", "--no-churn"],
                        str(guard_dir),
                    )
            else:
                rc, out, _ = invoke_monitor(
                    h_mod,
                    ["check_heartbeat.py", "--heartbeat-path", str(hb_fresh),
                     "--ping-url", "https://hc-ping.com/token123", "--no-churn"],
                    str(guard_dir),
                )
            return rc, out.strip(), h_pings

        real_epoch = int(datetime.now(timezone.utc).timestamp())
        live_now = datetime.fromtimestamp(real_epoch, timezone.utc)

        def write_valid_hold(state: str, age_s: int):
            def _setup(gdir: Path):
                (gdir / "hold").unlink(missing_ok=True)
                if (gdir / "hold").is_dir():
                    shutil.rmtree(gdir / "hold")
                armed = real_epoch - age_s
                rec = {
                    "schema": "bts_deploy_hold_v1",
                    "deploy_id": "1795197580-monprobe",
                    "owner_unit": "bts-deploy-1795197580-monprobe.service",
                    "state": state,
                    "step": "sync" if state != "activating" else "activation",
                    "target": "b" * 40,
                    "previous": "a" * 40,
                    "armed_at": armed,
                    "updated_at": max(armed, min(real_epoch, armed + 1)),
                }
                (gdir / "hold").write_text(json.dumps(rec))
            return _setup

        # Re-bind load_monitor to live_now so child guard tool's `updated_at <= int(time.time())` agrees with monitor's `now`!
        orig_now = now
        now = live_now
        hb_fresh.write_text(json.dumps({
            "state": "sleeping",
            "timestamp": (now - timedelta(seconds=10)).isoformat(),
            "sleeping_until": (now + timedelta(hours=2)).isoformat(),
        }))

        hold_matrix = {
            "young_pending_live_active": run_hold_case(write_valid_hold("pending", 30), lock_held=True, unit_state=b"active\n"),
            "young_pending_1380s_deactivating": run_hold_case(write_valid_hold("pending", 1380), lock_held=True, unit_state=b"deactivating\n"),
            "young_activating_live_active": run_hold_case(write_valid_hold("activating", 30), lock_held=True, unit_state=b"active\n"),
            "old_pending_1381s_even_if_alive": run_hold_case(write_valid_hold("pending", 1381), lock_held=True, unit_state=b"active\n"),
            "old_activating_1381s_even_if_alive": run_hold_case(write_valid_hold("activating", 1381), lock_held=True, unit_state=b"active\n"),
            "pending_dead_unlock": run_hold_case(write_valid_hold("pending", 10), lock_held=False, unit_state=b"active\n"),
            "activating_foreign_lock_dead_unit": run_hold_case(write_valid_hold("activating", 10), lock_held=True, unit_state=b"inactive\n"),
            "failed_hold": run_hold_case(write_valid_hold("failed", 10), lock_held=False, unit_state=b"inactive\n"),
        }

        assert hold_matrix["young_pending_live_active"][0] == 0 and hold_matrix["young_pending_live_active"][2] == []
        assert hold_matrix["young_pending_1380s_deactivating"][0] == 0 and hold_matrix["young_pending_1380s_deactivating"][2] == []
        assert hold_matrix["young_activating_live_active"][0] == 0 and hold_matrix["young_activating_live_active"][2] == []
        for k in ("old_pending_1381s_even_if_alive", "old_activating_1381s_even_if_alive",
                  "pending_dead_unlock", "activating_foreign_lock_dead_unit", "failed_hold"):
            assert hold_matrix[k][0] == 1 and hold_matrix[k][2][0][:2] == ("POST", "https://hc-ping.com/token123/fail"), (k, hold_matrix[k])

        # Adversarial notes on monitor:
        # (a) --deploy-hold <custom_dir>/hold WITHOUT BTS_TEST_DEPLOY_GUARD_DIR ignores <custom_dir>/hold in GUARD_TOOL status --json!
        custom_dir = work_dir / "custom_guard"
        custom_dir.mkdir(mode=0o755)
        for n in ("writers.lock", "transaction.lock", "state.lock"):
            (custom_dir / n).touch(mode=0o644)
        write_valid_hold("failed", 10)(custom_dir)
        (guard_dir / "hold").unlink(missing_ok=True)
        h_mod_custom, _ = load_monitor(HEAD_MONITOR, now)
        h_mod_custom.GUARD_TOOL = str(tool_file)
        orig_sr = h_mod_custom.subprocess.run
        h_mod_custom.subprocess.run = lambda argv, **kw: orig_sr([sys.executable, "-I", str(tool_file), *argv[3:]], **kw)
        # Set BTS_TEST_DEPLOY_GUARD_DIR to guard_dir (clear!), pass --deploy-hold custom_dir/hold (failed!):
        rc_custom, out_custom, _ = invoke_monitor(
            h_mod_custom,
            ["check_heartbeat.py", "--deploy-hold", str(custom_dir / "hold"),
             "--heartbeat-path", str(hb_fresh), "--no-churn"],
            str(guard_dir),
        )

        # (b) With permanent last-release from weeks ago, a normal stalled/stale scheduler loses its real staleness reason!
        (guard_dir / "hold").unlink(missing_ok=True)
        (guard_dir / "last-release").write_text(json.dumps({
            "released_at": real_epoch - 86400 * 20, "kind": "deploy", "deploy_id": "1790000000-old",
        }))
        hb_stalled = work_dir / "stalled_weeks_later.json"
        hb_stalled.write_text(json.dumps({
            "state": "stalled", "timestamp": (now - timedelta(seconds=5)).isoformat(),
            "stage": "computing_features", "stalled_for_s": 1200,
        }))
        h_mod_stalled, _ = load_monitor(HEAD_MONITOR, now)
        rc_stalled, out_stalled, _ = invoke_monitor(
            h_mod_stalled,
            ["check_heartbeat.py", "--heartbeat-path", str(hb_stalled), "--no-churn"],
            str(guard_dir),
        )

        # (c) Microsecond race: heartbeat written 1ms AFTER datetime.now(timezone.utc) sampled at main() line 407!
        hb_1ms_after = work_dir / "hb_1ms_after.json"
        hb_1ms_after.write_text(json.dumps({
            "state": "running", "timestamp": (now + timedelta(milliseconds=1)).isoformat(),
        }))
        h_mod_race, _ = load_monitor(HEAD_MONITOR, now)
        rc_race, out_race, _ = invoke_monitor(
            h_mod_race,
            ["check_heartbeat.py", "--heartbeat-path", str(hb_1ms_after), "--no-churn"],
            str(guard_dir),
        )

        # 3. Run real check_heartbeat.py subprocess under oldest Python (/usr/bin/python3 = Python 3.9.6)!
        ver_proc = subprocess.run([OLDEST_PYTHON, "--version"], stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5)
        oldest_ver = ver_proc.stdout.strip()
        (guard_dir / "last-release").unlink(missing_ok=True)

        def run_39(hb_file: Path):
            return subprocess.run(
                [OLDEST_PYTHON, str(HEAD_MONITOR), "--heartbeat-path", str(hb_file), "--no-churn"],
                env={"BTS_TEST_DEPLOY_GUARD_DIR": str(guard_dir)},
                stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=5,
            )

        py39_clear = run_39(hb_fresh)
        assert py39_clear.returncode == 0 and "stale=False  reason=fresh sleeping" in py39_clear.stdout

        os.mkfifo(guard_dir / "hold")
        py39_fifo = run_39(hb_fresh)
        assert py39_fifo.returncode == 1 and "Deploy hold present and unreadable" in py39_fifo.stdout
        (guard_dir / "hold").unlink()

        (guard_dir / "last-release").write_text(json.dumps({
            "released_at": real_epoch - 5, "kind": "hand", "deploy_id": "1795197580-39",
        }))
        py39_older_hb = run_39(hb_fresh)  # hb_fresh is real_epoch - 10s (< released_at!)
        assert py39_older_hb.returncode == 1 and "(hand); no scheduler heartbeat since." in py39_older_hb.stdout

        hb_newer = work_dir / "newer_39.json"
        hb_newer.write_text(json.dumps({
            "state": "sleeping",
            "timestamp": datetime.fromtimestamp(real_epoch - 2, timezone.utc).isoformat(),
            "sleeping_until": (now + timedelta(hours=2)).isoformat(),
        }))
        py39_newer_hb = run_39(hb_newer)
        assert py39_newer_hb.returncode == 0 and "stale=False  reason=fresh sleeping" in py39_newer_hb.stdout

        # Also check Python 3.9's Z-timestamp incompatibility when last-release is present vs base!
        hb_z = work_dir / "hb_z.json"
        hb_z.write_text(json.dumps({
            "state": "running",
            "timestamp": datetime.fromtimestamp(real_epoch - 2, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        }))
        py39_z = run_39(hb_z)

        now = orig_now
        report = {
            "parity_corpus_runs_passed": total_parity_runs,
            "hold_matrix_summary": {k: {"rc": v[0], "out": v[1]} for k, v in hold_matrix.items()},
            "oldest_python": {
                "binary": OLDEST_PYTHON,
                "version": oldest_ver,
                "clear_rc": py39_clear.returncode,
                "fifo_hold_rc": py39_fifo.returncode,
                "post_release_older_hb_rc": py39_older_hb.returncode,
                "post_release_newer_hb_rc": py39_newer_hb.returncode,
                "py39_trailing_Z_timestamp_rc": py39_z.returncode,
            },
            "adversarial_findings": {
                "custom_deploy_hold_flag_misreads_guard_tool": {"rc": rc_custom, "out": out_custom.strip()},
                "stalled_cascade_20_days_after_release_masks_reason": {"rc": rc_stalled, "out": out_stalled.strip()},
                "heartbeat_1ms_newer_than_sampled_now_falsely_fails": {"rc": rc_race, "out": out_race.strip()},
            },
        }
        out_path = Path(__file__).resolve().parent / "probe_4_monitor.json"
        out_path.write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return 0
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())
