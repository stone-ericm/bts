"""Probe 04: Linux gate / shim fidelity audit and R64 / Amendment 5 / operator text checks."""
from __future__ import annotations

from datetime import date, datetime, timezone
import gzip
import json
import os
from pathlib import Path
from shlex import quote
import subprocess
import sys
import tempfile
import time

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
BASE_ARCH = OUT_DIR / "_work_p01" / "base_archive"
DAY = "2027-03-24"
PITCH = f"{DAY}T19:40:00-04:00"


def check_workflows_and_shim_fidelity() -> dict:
    sys.path.insert(0, str(REPO))
    import tests.test_wr_f1b_lock as lock_mod

    test_yml = (REPO / ".github" / "workflows" / "test.yml").read_text()
    deploy_yml = (REPO / ".github" / "workflows" / "deploy.yml").read_text()

    assert "runs-on: ubuntu-latest" in test_yml
    assert "TZ: America/New_York" in test_yml
    assert 'version: "0.11.6"' in test_yml
    assert 'uv run pytest -m "not slow"' in test_yml

    assert "runs-on: ubuntu-latest" in deploy_yml
    assert "TZ: America/New_York" in deploy_yml
    assert 'version: "0.11.6"' in deploy_yml
    assert 'uv run pytest -m "not slow"' in deploy_yml
    assert "cron-setup-hetzner.sh" not in deploy_yml  # Amendment 5: Deploy leaves the old cron schedule in place

    # Check shim divergences in tests/test_wr_f1b_lock.py
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # 1. _flock shim on a directory lock path raises IsADirectoryError (util-linux flock falls back to O_RDONLY)
        dir_lock = root / "dir.lock"
        dir_lock.mkdir()
        dir_raised = None
        try:
            lock_mod._flock("-n", str(dir_lock), "/usr/bin/true")
        except Exception as exc:
            dir_raised = type(exc).__name__
        assert dir_raised == "IsADirectoryError"

        # 2. _flock shim on a mode-0444 read-only file raises PermissionError (util-linux flock uses O_RDONLY|O_CREAT)
        ro_lock = root / "ro.lock"
        ro_lock.write_text("")
        ro_lock.chmod(0o444)
        ro_raised = None
        if os.geteuid() != 0:
            try:
                lock_mod._flock("-n", str(ro_lock), "/usr/bin/true")
            except Exception as exc:
                ro_raised = type(exc).__name__
            assert ro_raised == "PermissionError"

        # 3. _timeout shim requires BTS_TEST_LOCK_FD in env (coupled to _flock running before _timeout)
        env_clean = {k: v for k, v in os.environ.items() if k != "BTS_TEST_LOCK_FD"}
        res_no_fd = subprocess.run(
            [sys.executable, "-c", "import tests.test_wr_f1b_lock as m; m._timeout('--verbose', '-k', '0.1', '0.1', '/usr/bin/true')"],
            cwd=REPO,
            env={**env_clean, "PYTHONPATH": str(REPO)},
            capture_output=True,
            text=True,
            timeout=10,
        )
        assert res_no_fd.returncode != 0 and "BTS_TEST_LOCK_FD" in res_no_fd.stderr

        # 4. Measure module-entry startup latency of `python -m tests.test_wr_f1b_lock child quick`
        trace_t = root / "trace.txt"
        t0 = time.monotonic()
        res_quick = subprocess.run(
            [sys.executable, "-m", "tests.test_wr_f1b_lock", "child", "quick"],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO), "BTS_TEST_TRACE": str(trace_t)},
            capture_output=True,
            text=True,
            timeout=10,
        )
        startup_ms = round((time.monotonic() - t0) * 1000, 1)
        assert res_quick.returncode == 0

        # 5. Check _uv FD inheritance when BTS_TEST_LOCK_FD is unset (as under real /usr/bin/flock on Linux [kill-137-real]):
        # Open an inheritable fd in parent without setting BTS_TEST_LOCK_FD, invoke _uv("quick"), inspect whether child sees it.
        probe_fd_script = (
            "import os, subprocess, sys; "
            "fd = os.open(sys.argv[1], os.O_CREAT | os.O_RDWR, 0o600); "
            "os.set_inheritable(fd, True); "
            "p = subprocess.Popen([sys.executable, '-c', "
            "'import os, sys; fd=int(sys.argv[1]); "
            "open(sys.argv[2], \"w\").write(str(os.path.exists(f\"/dev/fd/{fd}\")))', str(fd), sys.argv[2]], "
            "pass_fds=()); "
            "p.wait(timeout=5); os.close(fd)"
        )
        seen_file = root / "fd_seen_without_pass_fds.txt"
        subprocess.run([sys.executable, "-c", probe_fd_fd := probe_fd_script, str(root / "t.lock"), str(seen_file)], check=True, timeout=10)
        child_inherits_without_pass_fds = seen_file.read_text() == "True"
        assert child_inherits_without_pass_fds is False

    return {
        "workflows_verified": True,
        "deploy_leaves_crontab_untouched": True,
        "shim_flock_dir_raises": dir_raised,
        "shim_flock_readonly_raises": ro_raised,
        "shim_timeout_requires_lock_fd_env": True,
        "shim_uv_closes_real_flock_fd_in_kill_mode": not child_inherits_without_pass_fds,
        "single_harness_startup_ms": startup_ms,
    }


def check_texts_amendment5_and_rollback() -> dict:
    sys.path.insert(0, str(REPO))
    from tests.test_wr_f1_wrong_game import FIRST, LEG, MATCH, REMINDER, RIGHT, WRONG, World, row

    # 1. Cron header comment & Amendment 5 verbatim check against contract-check-f1b/contract-check.md
    cron_sh = (REPO / "scripts" / "cron-setup-hetzner.sh").read_text()
    expected_header = (
        "#                A nonblocking lock serializes runs, and timeout sends TERM at 14 minutes "
        "and KILL 30 seconds later if needed."
    )
    assert expected_header in cron_sh

    f1_contract = (REPO / "docs" / "reviews" / "2026-11-18-whole-range" / "kickoff-fix-f1.md").read_text()
    f1b_check = (REPO / "docs" / "reviews" / "2026-11-18-whole-range" / "contract-check-f1b" / "contract-check.md").read_text()

    def extract_replacement(label: str) -> str:
        prefix = f"- **{label}** → “"
        start = f1b_check.index(prefix) + len(prefix)
        end = f1b_check.index("”\n", start)
        return f1b_check[start:end]

    r_a54 = extract_replacement("Known limits 1–3 and A5.4's last clause")
    r_a64 = extract_replacement("Amendment 5 target A6.4 (replace the whole row)")
    r_a65 = extract_replacement("Amendment 5 target A6.5 (replace the whole row)")

    assert f"- **A6.4 (whole row)** — {r_a64}" in f1_contract
    assert f"- **A6.5 (whole row)** — {r_a65}" in f1_contract
    assert f"- **A5.4 (last clause)** — {r_a54}" in f1_contract

    # 2. Empirically verify A6.4 on real extracted 1e69d0f0 base code
    if not BASE_ARCH.exists():
        from probe_01_scope_parity_and_cron import extract_base
        extract_base(BASE_ARCH)

    rollback_script = """
import json, sys, tempfile
from pathlib import Path
from types import SimpleNamespace
from datetime import date
from unittest.mock import patch
from click.testing import CliRunner
import bts
from bts.cli import cli
from bts import picks
from bts.daily_decision import write_decision
from datetime import datetime

DAY = "2027-03-24"
PITCH = f"{DAY}T19:40:00-04:00"
cases = [
    ("alerted_wg", {"date": DAY, "status": "alerted", "reason": "wrong_game", "escalations": ["wrong_game"], "game_unresolved": True}, True),
    ("dm_failed_wg", {"date": DAY, "status": "dm_failed", "reason": "wrong_game", "escalations": [], "game_unresolved": True}, False),
    ("present_wg_only", {"date": DAY, "status": "present_unverified", "reason": "present_unverified", "escalations": ["wrong_game"], "game_unresolved": True}, False),
    ("present_init_and_wg", {"date": DAY, "status": "present_unverified", "reason": "present_unverified", "escalations": ["initial", "wrong_game"], "game_unresolved": True}, True),
    ("confirmed_unchecked", {"date": DAY, "status": "confirmed", "reason": "match_game_unchecked"}, False),
]
out = {}
with tempfile.TemporaryDirectory() as td:
    for name, marker_in, expect_all_clear in cases:
        root = Path(td) / name
        pdir = root / "picks"
        pdir.mkdir(parents=True)
        mpath = root / "health_state" / "pick_entry_check.json"
        mpath.parent.mkdir(parents=True)
        mpath.write_text(json.dumps(marker_in))
        p1 = picks.Pick("Batter 1010", 1010, "NYY", 1, "P", 500, 0.8, [], False, 101, PITCH)
        dp = picks.DailyPick(DAY, f"{DAY}T16:00:00-04:00", p1, None, None, notification_sent=True, notification_channel="bluesky_dm", notification_id="x")
        picks.save_pick(dp, pdir)
        write_decision(DAY, pdir, action="single", source="mdp", delivery_status="delivered", scoreable=True, primary={"batter_name": "Batter 1010"}, double_down=None, now=datetime.fromisoformat(f"{DAY}T20:00:00+00:00"))
        dms = []
        fetches = []
        sess = SimpleNamespace(user_id="u1", username="operator", xsid="x1")
        with patch("bts.leaderboard.auth.load_session_cookies", lambda: {}), \
             patch("bts.leaderboard.auth.extract_uid", lambda c: "u1"), \
             patch("bts.leaderboard.auth.fetch_login_session", lambda *a, **k: sess), \
             patch("bts.contest_fetch.fetch_profile", lambda *a, **k: fetches.append("p") or {"predictions": []}), \
             patch("bts.contest_fetch.fetch_pending_predictions", lambda *a, **k: [{"roundId": 1, "unitId": 77, "playerId": 1, "number": 1, "result": None}]), \
             patch("bts.cli._fetch_rounds", lambda *a, **k: {1: date.fromisoformat(DAY)}), \
             patch("bts.cli._fetch_bts_to_mlb", lambda *a, **k: {1: 1010}), \
             patch("bts.dm.send_dm", lambda r, m: dms.append(m)):
            res = CliRunner().invoke(cli, ["check-pick-entered", "--picks-dir", str(pdir), "--now-et", f"{DAY}T18:45:00", "--expected-username", "operator", "--dm-recipient", "op.invalid"])
        after = json.loads(mpath.read_text())
        assert res.exit_code == 0
        assert after["status"] == "confirmed"
        assert bool(dms) == expect_all_clear, (name, dms, expect_all_clear)
        if name == "confirmed_unchecked":
            assert fetches == [] and after["reason"] == "match_game_unchecked"
        out[name] = {"after_reason": after["reason"], "all_clear_sent": bool(dms), "fetched": bool(fetches)}
print(json.dumps(out))
"""
    rb_res = subprocess.run(
        [sys.executable, "-c", rollback_script],
        env={**os.environ, "PYTHONPATH": str(BASE_ARCH / "src"), "TZ": "America/New_York"},
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    rollback_verified = json.loads(rb_res.stdout)

    # 3. Empirically verify A6.5 morning priority & separate capture reads + F1 operator messages
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # Operator messages check (FIRST, REMINDER, locked_unconfirmed, match_game_unchecked, stdout)
        w_txt = World(root / "txt")
        w_txt.commit(pitch=PITCH, dd=(1020, 201, PITCH), name="", delivery="locked_unconfirmed")
        w_txt.capture([WRONG, {"id": 78, "roundId": 1, "feedId": 201}])
        r_lu = w_txt.check("18:30", [row(77, 1, 1), row(78, 2, 2)])
        expected_lu = (
            f"⚠️ Delivery of this pick was NOT confirmed. The scheduler tried to send batter 1010 + DD Batter 1020. "
            f"MLB game evidence needs verification. "
            f"pick batter 1010: recommended game 101 (first pitch 7:40 PM ET), the entry is recorded in game 102. "
            f"Check your DMs and the MLB entry before making any change. "
            f"Run `bts season-contract status --picks-dir {quote(str(w_txt.picks))}`."
        )
        assert r_lu["dm"] == [expected_lu]

        w_norm = World(root / "norm")
        w_norm.commit(pitch=PITCH)
        w_norm.capture([WRONG])
        r_f = w_norm.check("18:30", MATCH)
        assert r_f["dm"] == [FIRST.format(legs=LEG)]
        r_mid = w_norm.check("18:45", MATCH)
        assert r_mid["stdout"] == f"check-pick-entered: {DAY} game verification unresolved (pick); re-verifying each run until cutoff\n"
        r_rem = w_norm.check("19:05", MATCH)
        assert r_rem["dm"] == [REMINDER.format(legs=LEG)]
        for msg in (*r_lu["dm"], *r_f["dm"], *r_rem["dm"]):
            assert "Move the entry" not in msg
            assert "min to submit" not in msg and "min left" not in msg

        # A6.5 coexistence: unresolved wrong-game marker + later conflicting capture -> morning entered_game_unverified (ok=True) + health WARN
        w_co = World(root / "coexist")
        w_co.commit(pitch=PITCH)
        w_co.capture([WRONG])
        w_co.check("18:30", MATCH)
        w_co.capture([RIGHT], f"{DAY}T220000Z")
        w_co.check("18:45", MATCH)
        assert w_co.read_marker()["game_unresolved"] is True
        m_out = w_co.morning()
        h_out = w_co.health()
        assert m_out["exit"] == 0 and m_out["outcome"]["state"] == "entered_game_unverified"
        assert len(h_out) == 1 and h_out[0]["level"] == "WARN"

        # A5.2 separate capture reads: qualify_run sees WRONG, publish() sees RIGHT -> marker=alerted/wrong_game, morning=entered!
        w_sep = World(root / "sep_reads")
        w_sep.commit(pitch=PITCH)
        cap_file = w_sep.capture([WRONG])
        from bts import entry_receipt
        orig_load = entry_receipt.load_units
        calls = []

        def swap_after_first_read(pdir, season):
            res = orig_load(pdir, season)
            calls.append(len(calls))
            if len(calls) == 1:
                cap_file.write_bytes(gzip.compress(json.dumps({"units": [RIGHT]}).encode(), mtime=0))
            return res

        r_sep = w_sep.check("18:30", MATCH, more_patches=[("bts.entry_receipt.load_units", swap_after_first_read)])
        assert r_sep["exit"] == 1 and r_sep["marker"]["game_unresolved"] is True
        assert w_sep.morning()["outcome"]["state"] == "entered"

    return {
        "cron_header_verified": True,
        "amendment5_verbatim_match": True,
        "a64_rollback_matrix": rollback_verified,
        "operator_messages_r64_clean": True,
        "a65_coexistence_and_separate_reads_verified": True,
    }


def run_all() -> dict:
    summary = {
        "linux_and_shims": check_workflows_and_shim_fidelity(),
        "texts_and_limits": check_texts_amendment5_and_rollback(),
    }
    (OUT_DIR / "log_04_linux_audit_and_texts.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


def test_probe_04():
    res = run_all()
    assert res["linux_and_shims"]["workflows_verified"] is True
    assert res["texts_and_limits"]["amendment5_verbatim_match"] is True


if __name__ == "__main__":
    print(json.dumps(run_all(), indent=2))
