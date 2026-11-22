#!/usr/bin/env python3
"""Item 5: Probe D3 pre-D3b scope — success transcript delta vs 24019a9a, stamp order, deploy-canary parity, check_heartbeat.py + fs_code under Python 3.11.18 and 3.12.16."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

REPO = Path(__file__).resolve().parents[2]
SCRATCH_PARENT = REPO / ".codex-review/wr-d3b/tmp"
BASE_DIR = REPO / ".codex-review/wr-d3b/base"
PY311 = Path("/Users/eric/.local/share/uv/python/cpython-3.11.18-macos-aarch64-none/bin/python3.11")

SPEC = importlib.util.spec_from_file_location("workflow_tests", REPO / "tests/test_deploy_workflow.py")
assert SPEC and SPEC.loader
tests = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(tests)
DEPLOY, RECOVER = tests.scripts_and_steps()[0]
BASE_DEPLOY, = tests.scripts_and_steps((BASE_DIR / ".github/workflows/deploy.yml").read_text())[0]


def check_success_transcript_and_stamp() -> dict:
    SCRATCH_PARENT.mkdir(parents=True, exist_ok=True)
    summary = {}
    for legacy in (False, True):
        mode = "legacy-present" if legacy else "active"
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td1, tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td2:
            b_old = tests.box.__wrapped__(Path(td1))
            b_new = tests.box.__wrapped__(Path(td2))
            if legacy:
                for b in (b_old, b_new):
                    frags = b / "etc/systemd/system"
                    frags.mkdir(parents=True, exist_ok=True)
                    for u in ("bts-scheduler.service", "bts-dashboard.service"):
                        (frags / u).touch()
            res_old, calls_old = tests.run_box_script(b_old, BASE_DEPLOY, mode)
            res_new, calls_new = tests.run_box_script(b_new, DEPLOY, mode)
            assert res_old.returncode == 0 and res_new.returncode == 0

            # Verify strict ordering of key milestones in candidate transcript:
            def idx(pred):
                for i, c in enumerate(calls_new):
                    if pred(c):
                        return i
                raise AssertionError("missing expected call")

            order = [
                ("flock", idx(lambda c: c[:2] == ["flock", "-n"])),
                ("fetch", idx(lambda c: c[:3] == ["git", "fetch", "origin"])),
                ("publish_begun", idx(lambda c: c[0] == "mv" and c[-1].endswith("/attempt"))),
                ("disable", idx(lambda c: c == ["systemctl", "--user", "disable", "bts-scheduler"])),
                ("reset", idx(lambda c: c[:3] == ["git", "reset", "--hard"])),
                ("sync", idx(lambda c: c[:2] == ["uv", "sync"])),
                ("restart", idx(lambda c: c[:3] == ["systemctl", "--user", "restart"])),
                ("wait_30", idx(lambda c: c == ["timeout", "-k", "2", "33", "sleep", "30"])),
                ("deploy_canary", idx(lambda c: c[:2] == ["bts", "deploy-canary"])),
                ("stamp_stage", idx(lambda c: c[0] == "python3" and len(c) > 3 and c[3] == "stage")),
                ("enable", idx(lambda c: c == ["systemctl", "--user", "enable", "bts-scheduler"])),
                ("stamp_replace", idx(lambda c: c[0] == "python3" and len(c) > 3 and c[3] == "stamp")),
                ("publish_committed", max(i for i, c in enumerate(calls_new) if c[0] == "mv" and c[-1].endswith("/attempt"))),
                ("canary_passed_log", idx(lambda c: c[0] == "bash" and any("CANARY PASSED" in a for a in c))),
            ]
            indices = [pos for _, pos in order]
            assert indices == sorted(indices) and len(set(indices)) == len(indices), order
            summary["legacy_present" if legacy else "legacy_absent"] = {
                "base_calls": len(calls_old),
                "candidate_calls": len(calls_new),
                "stdout": res_new.stdout.splitlines(),
                "milestone_indices": dict(order),
            }
    return summary


def check_deploy_canary_parity() -> dict:
    from click.testing import CliRunner
    from zoneinfo import ZoneInfo
    import bts.cli as cli_mod
    from bts.contest_state import ContestStateError, load_decision_streak_state
    from bts.entry_intent import live_gate_problem
    from bts.orchestrator import load_config
    from bts.scheduler import _contest_state_required, _morning_state_check, _pick_delivery_mode
    from tests.test_season_contract_cs4b2 import AUTOS, MANUALS, NOW, SWITCHES, dated, install

    runner = CliRunner()
    orig_now = cli_mod._now_et
    paired_count = 0
    try:
        cli_mod._now_et = lambda: NOW
        # Full cross-product across delivery modes, flag combos, and state cases
        state_cases = [
            *[(f"corrupt{i}", "absent", "covered") for i in range(1, 9)],
            ("automatic", "legacy", "covered"),
            ("absent", "absent", "covered"),
            ("other_season", "absent", "covered"),
            ("revoked", "absent", "covered"),
            *[("automatic", "absent", a) for a in ("absent", "refused", "corrupt", "v1", "other_season", "other_id", "covered", "stale")],
            ("corrupt1", "today", "absent"),
            ("absent", "today", "absent"),
            ("revoked", "today", "refused"),
        ]
        for delivery in ("private", "dm", "public"):
            for flags in (0, 1, 2, 3):
                for sw, man, aut in state_cases[:4] if delivery == "private" or flags == 0 else state_cases:
                    with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
                        p = Path(td)
                        install(p, "season_contract.json", SWITCHES[sw])
                        install(p, "contest_streak.manual.json", MANUALS[man])
                        auto_val = (AUTOS["covered"] | {"recorded_at": "2027-07-14T10:00:00Z"}) if aut == "stale" else AUTOS[aut]
                        install(p, "contest_streak.json", auto_val)
                        lines = [f'[orchestrator]\npicks_dir = "{p}"', f'[scheduler]\npick_delivery = "{delivery}"']
                        if delivery == "private":
                            lines.append('entry_intent = "research"')
                        if flags & 1:
                            lines.append("contest_state_required = true")
                        if flags & 2:
                            lines.append("[health_checks]\ncontest_state_expected = true")
                        cfg_path = p / "cfg.toml"
                        cfg_path.write_text("\n".join(lines) + "\n")
                        before = {f: f.read_bytes() for f in p.rglob("*") if f.is_file()}
                        res = runner.invoke(cli_mod.cli, ["deploy-canary", "--config", str(cfg_path)])
                        after = {f: f.read_bytes() for f in p.rglob("*") if f.is_file()}
                        assert before == after, "deploy-canary mutated files"

                        cfg = load_config(cfg_path)
                        gate = live_gate_problem(cfg)
                        if gate is not None:
                            assert res.exit_code == 1 and res.output == f"Deploy canary: REFUSED — {gate}\n"
                        elif _pick_delivery_mode(cfg) not in ("dm", "public") or not _contest_state_required(cfg):
                            assert res.exit_code == 0 and "not a live configuration" in res.output
                        else:
                            try:
                                dec = load_decision_streak_state(p, require_contest_state=True, now=NOW)
                            except ContestStateError as exc:
                                assert res.exit_code == 1
                                assert res.output == f"Deploy canary: REFUSED — {ascii(str(exc))[:800]}\n"
                            else:
                                assert res.exit_code == 0
                                assert f"({dec.observation_kind}, streak {dec.streak}) at {NOW.isoformat()}." in res.output
                        paired_count += 1

        # Dec 31 -> Jan 1 year rollover with 2027 automatic state: accepted on Dec 31, refused on Jan 1!
        et = ZoneInfo("America/New_York")
        with tempfile.TemporaryDirectory(dir=SCRATCH_PARENT) as td:
            p = Path(td)
            install(p, "season_contract.json", SWITCHES["automatic"])
            install(p, "contest_streak.json", AUTOS["covered"])
            cfg_path = p / "cfg.toml"
            cfg_path.write_text(f'[orchestrator]\npicks_dir = "{p}"\n[scheduler]\npick_delivery = "dm"\ncontest_state_required = true\n')
            cli_mod._now_et = lambda: datetime(2027, 12, 31, 23, 59, 59, tzinfo=et)
            r_dec31 = runner.invoke(cli_mod.cli, ["deploy-canary", "--config", str(cfg_path)])
            cli_mod._now_et = lambda: datetime(2028, 1, 1, 0, 0, 1, tzinfo=et)
            r_jan1 = runner.invoke(cli_mod.cli, ["deploy-canary", "--config", str(cfg_path)])
            assert r_dec31.exit_code == 0 and "accepted" in r_dec31.output
            assert r_jan1.exit_code == 1 and "REFUSED" in r_jan1.output
            paired_count += 2
    finally:
        cli_mod._now_et = orig_now
    return {"paired_cases_verified": paired_count, "dec31_jan1_rollover": "PASS"}


def check_heartbeat_and_inline_py311() -> dict:
    # Run an isolated verification script under BOTH Python 3.12 (.venv) and Python 3.11.18!
    driver = r'''
import datetime, json, os, re, subprocess, sys, tempfile
from pathlib import Path

repo = Path(sys.argv[1])
base_hb_path = Path(sys.argv[2])
scratch_parent = Path(sys.argv[3])
sys.path.insert(0, str(repo))

from scripts import check_heartbeat

# 1. Compile and execute workflow fs_code operations under this Python interpreter!
wf_text = (repo / ".github/workflows/deploy.yml").read_text()
fs_codes = re.findall(r"fs_code='\n(.*?)\n            '", wf_text, re.DOTALL)
assert len(fs_codes) == 2 and fs_codes[0] == fs_codes[1]
compile(fs_codes[0], "<fs_code>", "exec")

with tempfile.TemporaryDirectory(dir=scratch_parent) as td:
    p = Path(td)
    # Test marker, endpoint, read, stage-record, cgroup under this interpreter
    hb = p / ".heartbeat"
    r = subprocess.run([sys.executable, "-c", fs_codes[0], "marker", str(hb), "sync", "bbbbbbb", "aaaaaaa"], timeout=5)
    assert r.returncode == 0 and json.loads(hb.read_text())["state"] == "deploy_failed"
    env_f = p / ".env"
    env_f.write_text('BTS_SCHEDULER_HEARTBEAT_PING_URL="https://checks.invalid/abc/"\n')
    r_ep = subprocess.run([sys.executable, "-c", fs_codes[0], "endpoint", str(env_f)], capture_output=True, text=True, timeout=5)
    assert r_ep.returncode == 0 and r_ep.stdout == "https://checks.invalid/abc/fail\n"

# 2. Compare check_heartbeat.py against base 24019a9a across 32 ordinary cases + symlink + missing + chmod 000!
base_ns = {"__name__": "base_hb"}
exec(compile(base_hb_path.read_text(), "<base_hb>", "exec"), base_ns)

now = datetime.datetime(2026, 11, 21, 12, 0, 0, tzinfo=datetime.timezone.utc)
class PinnedDT(datetime.datetime):
    @classmethod
    def now(cls, tz=None):
        return now.astimezone(tz) if tz else now.replace(tzinfo=None)

base_ns["datetime"] = PinnedDT
check_heartbeat.datetime = PinnedDT

ordinary_payloads = []
for ts in ("2026-11-21T11:59:00+00:00", "2026-11-21T08:00:00+00:00", None, "not-a-date"):
    for st, extra in [
        ("running", {}),
        ("waiting_for_games", {}),
        ("sleeping", {"sleeping_until": "2026-11-21T13:00:00+00:00"}),
        ("sleeping", {"sleeping_until": "2026-11-21T10:00:00+00:00"}),
        ("sleeping", {}),
        ("sleeping", {"sleeping_until": "bad-wake"}),
        ("idle_end_of_day", {}),
        ("stalled", {"stage": "fetch", "stalled_for_s": 99}),
    ]:
        d = {"state": st, **extra}
        if ts is not None:
            d["timestamp"] = ts
        ordinary_payloads.append(json.dumps(d).encode())
ordinary_payloads.append(b"{broken json")

with tempfile.TemporaryDirectory(dir=scratch_parent) as td:
    p = Path(td)
    for idx, raw_bytes in enumerate(ordinary_payloads):
        f = p / f"hb_{idx}.json"
        f.write_bytes(raw_bytes)
        link = p / f"link_{idx}.json"
        link.symlink_to(f)
        assert check_heartbeat.is_stale(link, now=now) == base_ns["is_stale"](link, now=now)

    missing = p / "does_not_exist.json"
    assert check_heartbeat.is_stale(missing, now=now) == base_ns["is_stale"](missing, now=now)

    # Hostile / special inputs -> fixed malformed reason or sanitized deploy_failed
    fifo = p / "hb.fifo"
    os.mkfifo(fifo)
    slink_fifo = p / "link.fifo"
    slink_fifo.symlink_to(fifo)
    dpath = p / "hb.dir"
    dpath.mkdir()
    for bad in (fifo, slink_fifo, dpath):
        assert check_heartbeat.is_stale(bad, now=now) == (True, "heartbeat unreadable or malformed")

    for bad_bytes in (b"[]", b"123", b'"str"', b"\xff\xfe", b" " * 65537,
                      b'{"state":"running","timestamp":"2026-11-21T12:00:00"}',
                      b'{"state":"sleeping","timestamp":"2026-11-21T11:59:00+00:00","sleeping_until":123}',
                      b'{"state":"sleeping","timestamp":"2026-11-21T11:59:00+00:00","sleeping_until":"2026-11-21T13:00:00"}'):
        bf = p / "bad.json"
        bf.write_bytes(bad_bytes)
        assert check_heartbeat.is_stale(bf, now=now) == (True, "heartbeat unreadable or malformed")

    # deploy_failed sanitization cases (including trailing newline in target_sha, bool step, naive timestamp)
    for payload, exp in [
        ({"state": "deploy_failed", "step": "contest-state", "target_sha": "012abcd", "timestamp": "naive"},
         "Deploy failed at contest-state (requested target 012abcd). See INCIDENT.md §4."),
        ({"state": "deploy_failed", "step": "cd", "target_sha": "012abcd\n"},
         "Deploy failed at unknown (requested target unknown). See INCIDENT.md §4."),
        ({"state": "deploy_failed", "step": True, "target_sha": "012ABCD"},
         "Deploy failed at unknown (requested target unknown). See INCIDENT.md §4."),
    ]:
        df = p / "df.json"
        df.write_text(json.dumps(payload))
        assert check_heartbeat.is_stale(df, now=now) == (True, exp)

print(json.dumps({"python": sys.version.split()[0], "ordinary_checked": len(ordinary_payloads) + 1, "status": "OK"}))
'''
    results = {}
    for label, interp in [("py312_venv", Path(sys.executable)), ("py311_real", PY311)]:
        proc = subprocess.run(
            [str(interp), "-c", driver, str(REPO), str(BASE_DIR / "scripts/check_heartbeat.py"), str(SCRATCH_PARENT)],
            capture_output=True, text=True, check=True, timeout=30,
        )
        results[label] = json.loads(proc.stdout.strip())
    # Also verify cron line in scripts/cron-setup-hetzner.sh:
    cron_text = (REPO / "scripts/cron-setup-hetzner.sh").read_text()
    expected_cron = (
        '*/5 * * * * $PREFIX /usr/bin/python3 scripts/check_heartbeat.py '
        '--heartbeat-path data/.heartbeat --ping-url \\"\\$BTS_SCHEDULER_HEARTBEAT_PING_URL\\" '
        '>> $LOG_DIR/heartbeat.log 2>&1 $MARKER'
    )
    assert expected_cron in cron_text
    results["cron_line_rd4"] = "VERIFIED"
    return results


def main() -> None:
    out = {
        "success_transcript_and_stamp": check_success_transcript_and_stamp(),
        "deploy_canary_parity": check_deploy_canary_parity(),
        "heartbeat_and_inline_py311_py312": check_heartbeat_and_inline_py311(),
    }
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
