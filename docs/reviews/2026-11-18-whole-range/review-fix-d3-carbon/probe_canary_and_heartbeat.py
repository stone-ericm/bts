#!/usr/bin/env python3
"""Probes for Review Items 5 and 6: `bts deploy-canary` parity/safety and `check_heartbeat.py` 3.11/base parity."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import urllib.request
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import requests.sessions
import yaml

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

import bts.cli as cli_mod
from bts.contest_state import ContestStateError, load_decision_streak_state
from bts.entry_intent import live_gate_problem
import bts.orchestrator as orchestrator
import bts.picks as picks_mod
import bts.scheduler as scheduler_mod
from scripts import check_heartbeat
from tests.test_season_contract_cs4b2 import AUTOS, MANUALS, NOW, SWITCHES, dated, install

PY311 = Path("/Users/eric/.local/share/uv/python/cpython-3.11.18-macos-aarch64-none/bin/python3.11")
ET = ZoneInfo("America/New_York")


def invoke_canary(cfg: Path):
    runner = CliRunner(mix_stderr=False)
    return runner.invoke(cli_mod.cli, ["deploy-canary", "--config", str(cfg)])


def check_item5() -> None:
    # 1. Delivery modes & aliases x contest flags x entry_intent
    private_modes = [
        'pick_delivery = "private"',
        'pick_delivery = "none"',
        'pick_delivery = "off"',
        'pick_delivery = "local"',
        'private_mode = true',
    ]
    live_modes = [
        ('pick_delivery = "dm"', "dm"),
        ('pick_delivery = "bluesky_dm"', "dm"),
        ('pick_delivery = "direct_message"', "dm"),
        ('pick_delivery = "public"', "public"),
        ('pick_delivery = "bluesky"', "public"),
        ('pick_delivery = "feed"', "public"),
        ('pick_delivery = "post"', "public"),
        ('', "public"),  # omitted pick_delivery defaults to public
    ]
    flag_combos = [
        ("", False),
        ("contest_state_required = true\n", True),
        ("[health_checks]\ncontest_state_expected = true\n", True),
        ("contest_state_required = true\n[health_checks]\ncontest_state_expected = true\n", True),
    ]

    orig_now = cli_mod._now_et
    cli_mod._now_et = lambda: NOW
    try:
        for pm in private_modes:
            for flags_toml, _ in flag_combos:
                with tempfile.TemporaryDirectory() as td:
                    p = Path(td)
                    cfg = p / "c.toml"
                    cfg.write_text(f'[orchestrator]\npicks_dir = "{p}"\n[scheduler]\nentry_intent = "research"\n{pm}\n{flags_toml}')
                    prob = live_gate_problem(orchestrator.load_config(cfg))
                    res = invoke_canary(cfg)
                    assert prob is None and res.exit_code == 0 and res.stderr == ""
                    assert res.stdout == "Deploy canary: not a live configuration (delivery private); contest state not checked.\n"

        # enter + private -> refused by startup gate
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            cfg = p / "c.toml"
            cfg.write_text(f'[orchestrator]\npicks_dir = "{p}"\n[scheduler]\nentry_intent = "enter"\npick_delivery = "private"\n')
            prob = live_gate_problem(orchestrator.load_config(cfg))
            res = invoke_canary(cfg)
            assert prob is not None and res.exit_code == 1
            assert res.stdout == f"Deploy canary: REFUSED — {prob}\n"

        # Live modes with and without flags
        for lm_toml, resolved_mode in live_modes:
            for flags_toml, has_flag in flag_combos:
                with tempfile.TemporaryDirectory() as td:
                    p = Path(td)
                    install(p, "contest_streak.manual.json", dated())
                    cfg = p / "c.toml"
                    cfg.write_text(f'[orchestrator]\npicks_dir = "{p}"\n[scheduler]\n{lm_toml}\n{flags_toml}')
                    prob = live_gate_problem(orchestrator.load_config(cfg))
                    res = invoke_canary(cfg)
                    assert res.stderr == ""
                    if not has_flag:
                        assert prob is not None and res.exit_code == 1
                        assert res.stdout == f"Deploy canary: REFUSED — {prob}\n"
                    else:
                        assert prob is None and res.exit_code == 0
                        assert res.stdout == f"Deploy canary: live contest state accepted (dated, streak 7) at {NOW.isoformat()}.\n"

        # Legacy shadow_mode = true refusal -> ValueError -> ERROR exit 2 (with contest_state_required=true)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            cfg = p / "c.toml"
            cfg.write_text(f'[orchestrator]\npicks_dir = "{p}"\n[scheduler]\nshadow_mode = true\ncontest_state_required = true\n')
            res = invoke_canary(cfg)
            assert res.exit_code == 2 and res.stdout == "Deploy canary: ERROR — ValueError\n"

        # Corrupt streak.json (model diagnostics logger silenced -> zero stderr!)
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            (p / "streak.json").write_text("{corrupt json")
            install(p, "contest_streak.manual.json", dated())
            cfg = p / "c.toml"
            cfg.write_text(f'[orchestrator]\npicks_dir = "{p}"\n[scheduler]\npick_delivery = "dm"\ncontest_state_required = true\n')
            # Add a stream handler to root logger to verify logger.disabled suppresses warning!
            import io
            buf = io.StringIO()
            handler = logging.StreamHandler(buf)
            logging.getLogger().addHandler(handler)
            try:
                res = invoke_canary(cfg)
            finally:
                logging.getLogger().removeHandler(handler)
            assert res.exit_code == 0 and res.stderr == "" and buf.getvalue() == "", f"leaked log: {buf.getvalue()!r}"

        # Relative picks_dir resolved from CWD identically to scheduler
        with tempfile.TemporaryDirectory() as td:
            cwd_root = Path(td)
            rel_picks = cwd_root / "data/picks"
            rel_picks.mkdir(parents=True)
            install(rel_picks, "contest_streak.manual.json", dated())
            cfg = cwd_root / "c.toml"
            cfg.write_text('[orchestrator]\npicks_dir = "data/picks"\n[scheduler]\npick_delivery = "dm"\ncontest_state_required = true\n')
            old_cwd = os.getcwd()
            os.chdir(cwd_root)
            try:
                res = invoke_canary(cfg)
            finally:
                os.chdir(old_cwd)
            assert res.exit_code == 0 and "accepted (dated, streak 7)" in res.stdout

        # Full refusal + acceptance parity matrix including future & unreadable automatic recorded_at
        extra_autos = {
            **AUTOS,
            "stale-old": AUTOS["covered"] | {"recorded_at": "2027-07-14T16:00:00Z"},
            "stale-future": AUTOS["covered"] | {"recorded_at": "2027-07-15T18:00:00Z"},
            "stale-unreadable": AUTOS["covered"] | {"recorded_at": "not-a-timestamp"},
        }
        cases = [
            *[(f"corrupt{i}", "absent", "covered") for i in range(1, 9)],
            ("automatic", "legacy", "covered"),
            ("absent", "absent", "covered"),
            ("other_season", "absent", "covered"),
            ("revoked", "absent", "covered"),
            *[("automatic", "absent", k) for k in extra_autos],
            ("corrupt1", "today", "corrupt"),
            ("absent", "today", "absent"),
        ]
        for sw, man, au in cases:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td)
                install(p, "season_contract.json", SWITCHES[sw])
                install(p, "contest_streak.manual.json", MANUALS[man])
                install(p, "contest_streak.json", extra_autos[au])
                cfg = p / "c.toml"
                cfg.write_text(f'[orchestrator]\npicks_dir = "{p}"\n[scheduler]\npick_delivery = "public"\ncontest_state_required = true\n')
                before = {f: f.read_bytes() for f in p.rglob("*") if f.is_file()}
                res = invoke_canary(cfg)
                try:
                    dec = load_decision_streak_state(p, require_contest_state=True, now=NOW)
                except ContestStateError as exc:
                    assert res.exit_code == 1
                    assert res.stdout == f"Deploy canary: REFUSED — {ascii(str(exc))[:800]}\n"
                else:
                    assert res.exit_code == 0
                    assert res.stdout == f"Deploy canary: live contest state accepted ({dec.observation_kind}, streak {dec.streak}) at {NOW.isoformat()}.\n"
                assert {f: f.read_bytes() for f in p.rglob("*") if f.is_file()} == before

        # ET midnight and Dec 31 -> Jan 1 year rollover with AUTOMATIC 2027 switch + observation!
        for t_before, t_after in [
            (datetime(2027, 7, 15, 23, 59, 59, tzinfo=ET), datetime(2027, 7, 16, 0, 0, 0, tzinfo=ET)),
            (datetime(2027, 12, 31, 23, 59, 59, tzinfo=ET), datetime(2028, 1, 1, 0, 0, 0, tzinfo=ET)),
        ]:
            with tempfile.TemporaryDirectory() as td:
                p = Path(td)
                if t_before.month == 12:
                    # Automatic 2027 switch + 2027 automatic observation: accepted on 2027-12-31, refused on 2028-01-01!
                    install(p, "season_contract.json", SWITCHES["automatic"])
                    install(p, "contest_streak.json", AUTOS["covered"])
                else:
                    install(p, "contest_streak.manual.json", dated("2027-07-15", recorded_at=t_before.isoformat()))
                cfg = p / "c.toml"
                cfg.write_text(f'[orchestrator]\npicks_dir = "{p}"\n[scheduler]\npick_delivery = "dm"\ncontest_state_required = true\n')
                cli_mod._now_et = lambda: t_before
                r1 = invoke_canary(cfg)
                cli_mod._now_et = lambda: t_after
                r2 = invoke_canary(cfg)
                assert r1.exit_code == 0 and r2.exit_code == 1, (t_before, r1.stdout, r2.stdout)
                if t_before.month == 12:
                    assert "season 2028 switch is other_season (file season 2027)" in r2.stdout

        # Symlink-to-FIFO config: bounded non-blocking regular-file check refuses without blocking
        with tempfile.TemporaryDirectory() as td:
            p = Path(td)
            fifo = p / "f"
            os.mkfifo(fifo)
            link = p / "link.toml"
            link.symlink_to(fifo)
            sub = subprocess.run(
                [sys.executable, "-c", "from bts.cli import cli; cli()", "deploy-canary", "--config", str(link)],
                capture_output=True, text=True, timeout=4,
            )
            assert sub.returncode == 2 and sub.stdout == "Deploy canary: ERROR — ValueError\n"
    finally:
        cli_mod._now_et = orig_now
    print("Item 5: all deploy-canary parity, boundary, alias, logger, and safety checks -> OK")


def check_item6() -> None:
    assert PY311.is_file(), f"Missing Python 3.11 at {PY311}"
    base_hb = REPO / ".codex-review/wr-d3/base/scripts/check_heartbeat.py"
    cand_hb = REPO / "scripts/check_heartbeat.py"

    # 1. Verify Python 3.11 compiles & executes check_heartbeat.py AND deploy.yml's inline Python snippets!
    wf = yaml.safe_load((REPO / ".github/workflows/deploy.yml").read_text())
    deploy_script = wf["jobs"]["deploy"]["steps"][3]["with"]["script"]
    m = re.search(r"fs_code='\n(.*?)\n            '", deploy_script, re.DOTALL)
    assert m, "could not locate fs_code"
    fs_code = "\n".join(l[12:] if l.startswith("            ") else l for l in m.group(1).splitlines())

    r311 = subprocess.run(
        [str(PY311), "-c", fs_code, "clock"],
        capture_output=True, text=True, timeout=5,
    )
    assert r311.returncode == 0 and r311.stdout.strip().isdigit(), r311

    # Also test sanitizer one-liner under Python 3.11
    sanitizer = 'import sys; print("".join(c if c.isprintable() else ascii(c)[1:-1] for c in sys.argv[1])[:900])'
    r311_s = subprocess.run(
        [str(PY311), "-c", sanitizer, "Deploy canary: REFUSED — bad\x07line"],
        capture_output=True, text=True, timeout=5,
    )
    assert r311_s.returncode == 0 and r311_s.stdout == "Deploy canary: REFUSED — bad\\x07line\n"

    # 2. Run check_heartbeat.py under real Python 3.11 across ordinary states, deploy_failed, FIFO, oversized, etc.
    driver_311 = r'''
import json, os, subprocess, sys, tempfile
from datetime import datetime, timezone
from pathlib import Path

base_src = Path(sys.argv[1]).read_text()
cand_path = Path(sys.argv[2])
cand_ns = {"__name__": "cand", "__file__": str(cand_path)}
base_ns = {"__name__": "base", "__file__": sys.argv[1]}
exec(compile(base_src, sys.argv[1], "exec"), base_ns)
exec(compile(cand_path.read_text(), str(cand_path), "exec"), cand_ns)

now = datetime(2026, 11, 21, 12, 0, 0, tzinfo=timezone.utc)
class PinnedDT(datetime):
    @classmethod
    def now(cls, tz=None):
        return now.astimezone(tz) if tz else now.replace(tzinfo=None)

base_ns["datetime"] = PinnedDT
cand_ns["datetime"] = PinnedDT

ordinary_payloads = [
    None,  # missing file
    b"{invalid json",
    *(json.dumps(d).encode() for ts in ("2026-11-21T11:59:00Z", "2026-11-20T10:00:00Z") for d in [
        {"state": "running", "timestamp": ts},
        {"state": "waiting_for_games", "timestamp": ts},
        {"state": "sleeping", "timestamp": ts, "sleeping_until": "2026-11-21T13:00:00Z"},
        {"state": "sleeping", "timestamp": ts, "sleeping_until": "2026-11-21T11:00:00Z"},
        {"state": "sleeping", "timestamp": ts},
        {"state": "sleeping", "timestamp": ts, "sleeping_until": "bad-wake"},
        {"state": "idle_end_of_day", "timestamp": ts},
        {"state": "stalled", "timestamp": ts, "stage": "predict", "stalled_for_s": 45},
        {"state": "weird", "timestamp": ts},
        {"state": "running"},
        {"state": "running", "timestamp": "bad-ts"},
    ]),
]

import io
from contextlib import redirect_stdout, redirect_stderr

def run_ns(ns, hb_path):
    pings = []
    ns["ping"] = lambda *a: pings.append(a)
    sys.argv = ["check_heartbeat.py", "--heartbeat-path", str(hb_path), "--no-churn", "--ping-url", "https://hc.invalid/u"]
    out, err = io.StringIO(), io.StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        try:
            ns["main"]()
            rc = 0
        except SystemExit as e:
            rc = e.code
    return rc, out.getvalue().encode(), err.getvalue().encode(), pings

with tempfile.TemporaryDirectory() as td:
    root = Path(td)
    for i, payload in enumerate(ordinary_payloads):
        hb = root / f"hb_{i}.json"
        if payload is not None:
            hb.write_bytes(payload)
        b_res = run_ns(base_ns, hb)
        c_res = run_ns(cand_ns, hb)
        assert b_res == c_res, (i, payload, b_res, c_res)
        if payload is not None:
            link = root / f"link_{i}.json"
            link.symlink_to(hb)
            assert run_ns(base_ns, link) == run_ns(cand_ns, link)

    # Check deploy_failed with odd fields, naive/missing ts, sleeping_until, etc.
    for step, target, exp_s, exp_t in [
        ("sync", "0123abc", "sync", "0123abc"),
        ("cd", "0123abc", "unknown", "0123abc"),  # preflight token not in DEPLOY_FAILURE_STEPS!
        ("sync", "0123ab", "sync", "unknown"),
        ("sync", "0123abcd", "sync", "unknown"),
        ("sync", "0123abc\n", "sync", "unknown"),
        (True, 1234567, "unknown", "unknown"),
        (None, None, "unknown", "unknown"),
    ]:
        hb = root / "df.json"
        hb.write_text(json.dumps({"state": "deploy_failed", "step": step, "target_sha": target, "timestamp": "naive"}))
        rc, out, err, pings = run_ns(cand_ns, hb)
        assert rc == 1 and pings == [("https://hc.invalid/u", "/fail")]
        expected = f"Deploy failed at {exp_s} (requested target {exp_t}). See INCIDENT.md §4."
        assert expected.encode() in out, (out, expected)

    # Check hostile/special inputs -> heartbeat unreadable or malformed
    for bad in [
        b"[]", b"123", b'"str"', b"null", b"\xff", b" " * 65537,
        b'{"state":"running","timestamp":"2026-11-21T12:00:00"}',
        b'{"state":"sleeping","timestamp":"2026-11-21T12:00:00Z","sleeping_until":"2026-11-21T13:00:00"}',
        b'{"state":"sleeping","timestamp":"2026-11-21T12:00:00Z","sleeping_until":123}',
        b'{"a":' + b"1" * 5000 + b"}",
    ]:
        hb = root / "bad.json"
        hb.write_bytes(bad)
        assert cand_ns["is_stale"](hb, now=now) == (True, "heartbeat unreadable or malformed"), bad[:40]

    # Directory & FIFO & symlink-to-FIFO
    d = root / "dir_hb"
    d.mkdir()
    assert cand_ns["is_stale"](d, now=now) == (True, "heartbeat unreadable or malformed")
    fifo = root / "fifo_hb"
    os.mkfifo(fifo)
    assert cand_ns["is_stale"](fifo, now=now) == (True, "heartbeat unreadable or malformed")
    s_fifo = root / "sym_fifo"
    s_fifo.symlink_to(fifo)
    assert cand_ns["is_stale"](s_fifo, now=now) == (True, "heartbeat unreadable or malformed")

print("Python", sys.version.split()[0], "all checks OK")
'''
    import re as _re  # noqa: F401
    for interp in (Path(sys.executable), PY311):
        proc = subprocess.run(
            [str(interp), "-c", driver_311, str(base_hb), str(cand_hb)],
            capture_output=True, text=True, timeout=15,
        )
        assert proc.returncode == 0, (interp, proc.stdout, proc.stderr)
        print(f"Item 6 ({interp.name}): {proc.stdout.strip()}")


if __name__ == "__main__":
    import re
    check_item5()
    check_item6()
