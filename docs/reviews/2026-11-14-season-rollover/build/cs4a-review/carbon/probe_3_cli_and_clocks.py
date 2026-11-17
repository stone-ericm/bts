"""Adversarial Probe 3: CLI Command (Item 4) & Clocks/Midnight (Item 5)."""
from __future__ import annotations

import builtins
import datetime as dt
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO_ROOT = Path(__file__).resolve().parents[3]
REVIEW_DIR = REPO_ROOT / ".codex-review" / "cs4a-review"
BASE_SRC = REVIEW_DIR / "base_tree" / "src"
BRANCH_SRC = REPO_ROOT / "src"

ET = ZoneInfo("America/New_York")
SCHEMA = "bts_contest_streak_manual_v3"


def install_audit_hooks(fs_events: list[tuple]) -> None:
    forbidden = {
        "socket.connect",
        "socket.connect_ex",
        "socket.getaddrinfo",
        "socket.getnameinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
        "socket.sendto",
    }

    def hook(event: str, args: tuple) -> None:
        if event in forbidden:
            raise RuntimeError(f"Network access forbidden in probe: {event} {args}")
        if event == "open" and args:
            fs_events.append(("open", str(args[0]), str(args[1]) if len(args) > 1 else ""))
        elif event == "os.replace" and len(args) >= 2:
            fs_events.append(("replace", str(args[0]), str(args[1])))
        elif event == "os.mkdir" and args:
            fs_events.append(("mkdir", str(args[0]), ""))

    sys.addaudithook(hook)


def make_clock(times: list[dt.datetime]):
    class TickingDateTime(dt.datetime):
        calls = 0

        @classmethod
        def now(cls, tz=None):
            val = times[min(cls.calls, len(times) - 1)]
            cls.calls += 1
            return val.astimezone(tz) if tz else val.replace(tzinfo=None)

    return TickingDateTime


def test_cli_dated_mode(fs_events: list[tuple]) -> dict:
    import click
    from click.testing import CliRunner
    import bts.contest_state as cs
    from bts.cli import cli
    import bts.util as util

    failures: list[str] = []
    now = dt.datetime(2027, 7, 15, 12, 30, 45, 123456, tzinfo=ET)
    runner = CliRunner()

    # 1. Option combinations & exact read/write order + sentinels
    combos_tested = 0
    with tempfile.TemporaryDirectory() as tmp:
        for reason_opt in (None, "", "Opening check", "MLB — verified"):
            for streak, best in ((0, 0), (0, 57), (12, 14), (57, 57)):
                for saver in ("not_earned", "active", "used"):
                    combos_tested += 1
                    pdir = Path(tmp) / f"c_{combos_tested}"
                    sdir = pdir / "account_state"
                    sdir.mkdir(parents=True)
                    for s_file in ("saver_state.json", "saver_transitions.jsonl", "contest_streak.json"):
                        (sdir / s_file).write_bytes(f"sentinel:{s_file}".encode())
                    hdir = pdir.parent / f"health_{combos_tested}"
                    # Check standard fetch status location relative to picks_dir.parent
                    fetch_status = pdir.parent / "health_state" / "contest_streak_fetch_status.json"
                    fetch_status.parent.mkdir(exist_ok=True)
                    fetch_status.write_bytes(b"sentinel:fetch")

                    clk = make_clock([now])
                    args = [
                        "set-contest-streak", "--picks-dir", str(pdir),
                        "--date", "2027-7-15" if combos_tested % 2 == 0 else "2027-07-15",
                        "--streak", str(streak), "--best-streak", str(best), "--saver", saver,
                    ]
                    if reason_opt is not None:
                        args.extend(["--reason", reason_opt])

                    fs_events.clear()
                    with patch.object(dt, "datetime", clk), patch.object(cs, "datetime", clk):
                        res = runner.invoke(cli, args)
                    if res.exit_code != 0:
                        failures.append(f"Combo failed {args}: {res.output}")
                        continue
                    if clk.calls != 1:
                        failures.append(f"Expected 1 clock call, got {clk.calls}")

                    # Filter fs_events relevant to pdir
                    p_str = str(pdir)
                    rel_events = [(ev, Path(a).name, b) for ev, a, b in fs_events if a.startswith(p_str)]
                    # Order must be: read manual -> open requests.jsonl 'a' -> mkstemp/open tmp -> replace tmp -> manual
                    manual_read_idx = next((i for i, e in enumerate(rel_events) if e[0] == "open" and e[1] == "contest_streak.manual.json" and "r" in e[2]), -1)
                    log_append_idx = next((i for i, e in enumerate(rel_events) if e[0] == "open" and e[1] == "contest_manual_requests.jsonl" and "a" in e[2]), -1)
                    replace_idx = next((i for i, e in enumerate(rel_events) if e[0] == "replace" and Path(e[2]).name == "contest_streak.manual.json"), -1)
                    if not (0 <= manual_read_idx < log_append_idx < replace_idx):
                        failures.append(f"Wrong syscall order: {rel_events}")

                    # Verify last line and JSON contents
                    lines = res.output.splitlines()
                    exp_last = f"contest_streak: {streak} (best {best}, saver {saver}) for 2027-07-15, in effect today"
                    if lines[-1] != exp_last:
                        failures.append(f"Wrong last line: {lines[-1]!r} != {exp_last!r}")

                    written = json.loads((sdir / "contest_streak.manual.json").read_text(encoding="utf-8"))
                    if ("reason" in written) != (reason_opt is not None):
                        failures.append(f"Reason presence mismatch: {written}")
                    if reason_opt is not None and written["reason"] != reason_opt:
                        failures.append(f"Reason value mismatch: {written}")
                    if written["recorded_at"] != "2027-07-15T16:30:45.123456Z":
                        failures.append(f"Wrong recorded_at: {written['recorded_at']}")

                    # Re-run identical -> Row 4 no-op (zero writes!)
                    fs_events.clear()
                    clk2 = make_clock([now + dt.timedelta(minutes=5)])
                    with patch.object(dt, "datetime", clk2), patch.object(cs, "datetime", clk2):
                        res2 = runner.invoke(cli, args)
                    if res2.exit_code != 0 or "unchanged" not in res2.output or res2.output.splitlines()[-1] != exp_last:
                        failures.append(f"No-op output wrong: {res2.output}")
                    write_syscalls = [
                        e for e in fs_events
                        if e[1].startswith(p_str) and (e[0] in ("replace", "mkdir") or any(m in e[2] for m in ("w", "a", "+")))
                    ]
                    if write_syscalls:
                        failures.append(f"No-op performed write syscalls: {write_syscalls}")

                    # Sentinels untouched
                    for s_file in ("saver_state.json", "saver_transitions.jsonl", "contest_streak.json"):
                        if (sdir / s_file).read_bytes() != f"sentinel:{s_file}".encode():
                            failures.append(f"Sentinel {s_file} modified!")
                    if fetch_status.read_bytes() != b"sentinel:fetch":
                        failures.append("fetch_status sentinel modified!")

    # 2. Row 2 rejections + audit hook proof of ZERO reads under picks_dir!
    rejection_cases = [
        (["--streak", "1", "--saver", "active"], "--saver"),
        (["--date", "2027-07-15", "--streak", "1", "--saver", "active"], "--best-streak"),
        (["--date", "2027-07-15", "--streak", "1", "--best-streak", "1"], "--saver"),
        (["--date", "2027-07-15", "--best-streak", "1", "--saver", "active"], "--streak"),
        (["--date", "2027-07-15", "--streak", "1", "--best-streak", "1", "--saver", "active", "--source-date", "2027-07-15"], "--source-date"),
        (["--date", "2027-07-15", "--streak", "1", "--best-streak", "1", "--saver", "active", "--source", "manual_cli"], "--source"),
        (["--date", "2027-07-15", "--streak", "1", "--best-streak", "1", "--saver", "active", "--username", ""], "--username"),
        (["--date", "2027-07-15", "--streak", "1", "--best-streak", "1", "--saver", "active", "--ttl-hours", "24"], "--ttl-hours"),
        (["--date", "2027-07-15", "--streak", "1", "--best-streak", "1", "--saver", "active", "--ttl-hours", "24.0"], "--ttl-hours"),
        (["--date", "2027-07-15", "--streak", "1", "--best-streak", "1", "--saver", "active", "--override-expires-at", "2099-01-01T00:00:00Z"], "--override-expires-at"),
        (["--date", "2027-07-15", "--streak", "1", "--best-streak", "1", "--saver", "active", "--saver-available"], "--saver-available/--saver-unavailable"),
        (["--date", "2027-07-15", "--streak", "1", "--best-streak", "1", "--saver", "active", "--saver-unavailable"], "--saver-available/--saver-unavailable"),
        (["--date", "2027-07-14", "--streak", "1", "--best-streak", "1", "--saver", "active"], "--date"),
        (["--date", "2027-07-16", "--streak", "1", "--best-streak", "1", "--saver", "active"], "--date"),
        (["--date", "2026-07-15", "--streak", "1", "--best-streak", "1", "--saver", "active"], "--date"),
        (["--date", "20270715", "--streak", "1", "--best-streak", "1", "--saver", "active"], "--date"),
        (["--date", "2027-W28-4", "--streak", "1", "--best-streak", "1", "--saver", "active"], "--date"),
        (["--date", "2027-07-15T00:00:00", "--streak", "1", "--best-streak", "1", "--saver", "active"], "--date"),
        (["--date", "2027-07-15", "--streak", "-1", "--best-streak", "1", "--saver", "active"], "--streak"),
        (["--date", "2027-07-15", "--streak", "5", "--best-streak", "4", "--saver", "active"], "--best-streak"),
        (["--date", "2027-07-15", "--streak", "5", "--best-streak", "58", "--saver", "active"], "--best-streak"),
        (["--date", "2027-07-15", "--streak", "58", "--best-streak", "58", "--saver", "active"], "--best-streak"),
        (["--date", "2027-07-15", "--streak", "1", "--best-streak", "1", "--saver", "ACTIVE"], "--saver"),
        (["--date", "2027-07-15", "--streak", "1", "--best-streak", "1", "--saver", "uninitialized"], "--saver"),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        rdir = Path(tmp) / "rej_picks"
        (rdir / "account_state").mkdir(parents=True)
        # Put an existing valid observation on disk so any premature Row 4 read would open it!
        (rdir / "account_state" / "contest_streak.manual.json").write_text("{}", encoding="utf-8")
        r_str = str(rdir)
        for cli_args, expected_opt in rejection_cases:
            clk = make_clock([now])
            fs_events.clear()
            with patch.object(dt, "datetime", clk), patch.object(cs, "datetime", clk):
                res = runner.invoke(cli, ["set-contest-streak", "--picks-dir", r_str, *cli_args])
            if res.exit_code != 2:
                failures.append(f"Rejection {cli_args} exited {res.exit_code}, expected 2: {res.output}")
            if expected_opt not in res.output:
                failures.append(f"Rejection {cli_args} did not name {expected_opt!r}: {res.output}")
            if cli_args[:2] == ["--date", "2027-07-14"] and "2027-07-15" not in res.output:
                failures.append(f"Wrong-date rejection did not name today 2027-07-15: {res.output}")
            opens_in_rdir = [e for e in fs_events if e[0] == "open" and e[1].startswith(r_str)]
            if opens_in_rdir:
                failures.append(f"Premature open under picks_dir on rejection {cli_args}: {opens_in_rdir}")

    # 3. Real OS permission failures in Row 3: read-only directory (0o555) WITHOUT and WITH existing writable log!
    with tempfile.TemporaryDirectory() as tmp:
        # 3a: Read-only account_state WITHOUT existing log -> step (a) open fails with PermissionError -> exit 1!
        ro1 = Path(tmp) / "ro_no_log"
        s1 = ro1 / "account_state"
        s1.mkdir(parents=True)
        existing_obs = {
            "schema_version": SCHEMA, "effective_date": "2027-07-15", "season": 2027,
            "expires_at": "2027-07-15T23:59:59.999999-04:00", "active_streak": 3,
            "best_streak": 5, "saver_state": "not_earned", "recorded_at": "2027-07-15T15:00:00Z",
            "source": "manual_cli",
        }
        (s1 / "contest_streak.manual.json").write_text(json.dumps(existing_obs), encoding="utf-8")
        s1.chmod(0o555)
        try:
            clk = make_clock([now])
            with patch.object(dt, "datetime", clk), patch.object(cs, "datetime", clk):
                res_ro1 = runner.invoke(cli, [
                    "set-contest-streak", "--picks-dir", str(ro1),
                    "--date", "2027-07-15", "--streak", "4", "--best-streak", "5", "--saver", "not_earned",
                ])
            if res_ro1.exit_code != 1 or "not written" not in res_ro1.output or "partial" not in res_ro1.output:
                failures.append(f"RO dir without log unexpected result: {res_ro1.exit_code} {res_ro1.output}")
            if json.loads((s1 / "contest_streak.manual.json").read_text(encoding="utf-8"))["active_streak"] != 3:
                failures.append("RO dir without log modified existing manual file!")
        finally:
            s1.chmod(0o755)

        # 3b: Read-only account_state WITH pre-existing writable contest_manual_requests.jsonl ->
        # step (a) SUCCEEDS (appends line!), step (b) mkstemp FAILS with PermissionError -> exit 1 ("NOT saved; its request was appended")!
        ro2 = Path(tmp) / "ro_with_log"
        s2 = ro2 / "account_state"
        s2.mkdir(parents=True)
        (s2 / "contest_streak.manual.json").write_text(json.dumps(existing_obs), encoding="utf-8")
        (s2 / "contest_manual_requests.jsonl").write_text("", encoding="utf-8")
        s2.chmod(0o555)
        try:
            clk = make_clock([now])
            with patch.object(dt, "datetime", clk), patch.object(cs, "datetime", clk):
                res_ro2 = runner.invoke(cli, [
                    "set-contest-streak", "--picks-dir", str(ro2),
                    "--date", "2027-07-15", "--streak", "4", "--best-streak", "5", "--saver", "not_earned",
                ])
            if res_ro2.exit_code != 1 or "NOT saved" not in res_ro2.output or "appended" not in res_ro2.output:
                failures.append(f"RO dir with writable log unexpected: {res_ro2.exit_code} {res_ro2.output}")
            log_lines = (s2 / "contest_manual_requests.jsonl").read_text(encoding="utf-8").splitlines()
            if len(log_lines) != 1 or json.loads(log_lines[0])["active_streak"] != 4:
                failures.append(f"RO dir with writable log did not append request: {log_lines}")
            if json.loads((s2 / "contest_streak.manual.json").read_text(encoding="utf-8"))["active_streak"] != 3:
                failures.append("RO dir with writable log changed manual file!")
        finally:
            s2.chmod(0o755)
        # Retry after restoring permissions -> succeeds, 2 log lines total, manual file updated to 4!
        clk = make_clock([now])
        with patch.object(dt, "datetime", clk), patch.object(cs, "datetime", clk):
            res_retry = runner.invoke(cli, [
                "set-contest-streak", "--picks-dir", str(ro2),
                "--date", "2027-07-15", "--streak", "4", "--best-streak", "5", "--saver", "not_earned",
            ])
        if res_retry.exit_code != 0:
            failures.append(f"Retry after RO restore failed: {res_retry.output}")
        if len((s2 / "contest_manual_requests.jsonl").read_text(encoding="utf-8").splitlines()) != 2:
            failures.append("Retry after RO restore did not append second log line!")
        if json.loads((s2 / "contest_streak.manual.json").read_text(encoding="utf-8"))["active_streak"] != 4:
            failures.append("Retry after RO restore did not update manual file!")

    # 4. Replacing weird existing files: dangling symlink, 5000-digit int, deep nesting, UTF-8 BOM
    with tempfile.TemporaryDirectory() as tmp:
        for weird_kind in ("dangling_symlink", "5000_digits", "deep_nesting", "utf8_bom"):
            wdir = Path(tmp) / weird_kind
            ws = wdir / "account_state"
            ws.mkdir(parents=True)
            mp = ws / "contest_streak.manual.json"
            if weird_kind == "dangling_symlink":
                mp.symlink_to(ws / "missing_target.json")
            elif weird_kind == "5000_digits":
                mp.write_text('{"x":' + ("9" * 5000) + "}", encoding="utf-8")
            elif weird_kind == "deep_nesting":
                mp.write_text("[" * 2000 + "]" * 2000, encoding="utf-8")
            elif weird_kind == "utf8_bom":
                mp.write_bytes(b"\xef\xbb\xbf{}")
            clk = make_clock([now])
            with patch.object(dt, "datetime", clk), patch.object(cs, "datetime", clk):
                res_w = runner.invoke(cli, [
                    "set-contest-streak", "--picks-dir", str(wdir),
                    "--date", "2027-07-15", "--streak", "2", "--best-streak", "2", "--saver", "active",
                ])
            if res_w.exit_code != 0 or mp.is_symlink() or json.loads(mp.read_text(encoding="utf-8"))["active_streak"] != 2:
                failures.append(f"Failed replacing {weird_kind}: {res_w.exit_code} {res_w.output}")

    return {
        "combos_tested": combos_tested,
        "rejections_tested": len(rejection_cases),
        "failures": failures,
    }


def test_legacy_cli_parity_vs_base() -> dict:
    """Compare legacy set-contest-streak (no --date, no --saver) byte-for-byte against Base 22e3ebf9."""
    worker = """
import datetime as dt, json, sys, tempfile
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
from click.testing import CliRunner
from bts.cli import cli
import bts.saver_state as flags

ET = ZoneInfo("America/New_York")
ticks = [
    dt.datetime(2027, 7, 15, 16, 0, 1, tzinfo=dt.timezone.utc),
    dt.datetime(2027, 7, 15, 16, 0, 2, tzinfo=dt.timezone.utc),
    dt.datetime(2027, 7, 15, 16, 0, 3, tzinfo=dt.timezone.utc),
    dt.datetime(2027, 7, 15, 16, 0, 4, tzinfo=dt.timezone.utc),
    dt.datetime(2027, 7, 15, 16, 0, 5, tzinfo=dt.timezone.utc),
]

invocations = [
    ["--streak", "0"],
    ["--streak", "7", "--best-streak", "9", "--saver-available", "--username", "alice", "--reason", "manual check", "--ttl-hours", "12"],
    ["--streak", "12", "--best-streak", "12", "--saver-unavailable", "--source", "ops", "--source-date", "2027-07-14"],
    ["--streak", "15", "--best-streak", "20", "--source-date", "2026-09-28"],
    ["--streak", "58", "--best-streak", "60", "--override-expires-at", "2027-07-16T04:00:00Z"],
    ["--streak", "3", "--override-expires-at", "2027-07-16T04:00:00"],
    ["--streak", "-1"],
    ["--streak", "5", "--best-streak", "4"],
    ["--streak", "5", "--source-date", "bad-date"],
    ["--streak", "5", "--override-expires-at", "bad-iso"],
]

out = []
runner = CliRunner()
for idx, args in enumerate(invocations):
    with tempfile.TemporaryDirectory() as tmp:
        p = Path(tmp) / "picks"
        class Clk(dt.datetime):
            i = 0
            @classmethod
            def now(cls, tz=None):
                v = ticks[min(cls.i, len(ticks) - 1)]
                cls.i += 1
                return v.astimezone(tz) if tz else v.replace(tzinfo=None)
        with patch.object(dt, "datetime", Clk), patch.object(flags, "datetime", Clk):
            res = runner.invoke(cli, ["set-contest-streak", "--picks-dir", str(p), *args])
        files = {}
        if p.exists():
            for fp in sorted(p.rglob("*")):
                if fp.is_file() and fp.suffix != ".lock":
                    files[str(fp.relative_to(p))] = fp.read_text(encoding="utf-8")
        out.append({
            "args": args,
            "exit_code": res.exit_code,
            "output": res.output.replace(str(p), "<PICKS>"),
            "clock_calls": Clk.i,
            "files": files,
        })
print(json.dumps(out))
"""
    base_res = json.loads(subprocess.check_output(
        [sys.executable, "-c", worker],
        env=dict(os.environ, PYTHONPATH=str(BASE_SRC), PYTHONDONTWRITEBYTECODE="1"),
        text=True,
    ))
    branch_res = json.loads(subprocess.check_output(
        [sys.executable, "-c", worker],
        env=dict(os.environ, PYTHONPATH=str(BRANCH_SRC), PYTHONDONTWRITEBYTECODE="1"),
        text=True,
    ))
    diffs = [
        {"index": i, "base": b, "branch": br}
        for i, (b, br) in enumerate(zip(base_res, branch_res))
        if b != br
    ]
    return {"invocations_compared": len(base_res), "differences": diffs}


def test_clocks_and_midnight() -> dict:
    import click
    from click.testing import CliRunner
    import bts.contest_state as cs
    import bts.health.contest_state as health
    from bts.cli import cli

    failures: list[str] = []
    dates_to_check = [
        "2027-07-15",  # ordinary EDT day
        "2027-03-14",  # spring-forward DST (23h day: -05:00 at start, -04:00 at end)
        "2027-11-07",  # fall-back DST (25h day: -04:00 at start, -05:00 at end)
        "2027-12-31",  # New Year's Eve
        "2028-01-01",  # New Year's Day
        "2028-02-29",  # Leap day
    ]

    boundary_checks = 0
    with tempfile.TemporaryDirectory() as tmp:
        for d_str in dates_to_check:
            pdir = Path(tmp) / d_str
            sdir = pdir / "account_state"
            sdir.mkdir(parents=True)
            d_obj = dt.date.fromisoformat(d_str)
            eod = dt.datetime.combine(d_obj, dt.time.max, ET)
            sod = dt.datetime.combine(d_obj, dt.time.min, ET)
            # Put a conflicting 2026/2027 saver flag & future settled pick on disk
            (sdir / "saver_state.json").write_text('{"state":"used","season":' + ("9" * 5000) + "}", encoding="utf-8")
            (pdir / "2029-01-01.json").write_text('{"result":"miss"}', encoding="utf-8")
            (sdir / "contest_streak.manual.json").write_text(json.dumps({
                "schema_version": SCHEMA,
                "effective_date": d_str,
                "season": d_obj.year,
                "expires_at": eod.isoformat(),
                "active_streak": 12,
                "best_streak": 20,
                "saver_state": "not_earned",
                "recorded_at": sod.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                "source": "manual_cli",
            }), encoding="utf-8")

            # Construct exact instants in UTC so +/- 1 microsecond crosses true wall-time boundaries accurately across DST!
            sod_utc = sod.astimezone(dt.timezone.utc)
            eod_utc = eod.astimezone(dt.timezone.utc)
            instants = [
                ("before_1us", (sod_utc - dt.timedelta(microseconds=1)).astimezone(ET), False),
                ("start_00_00", sod, True),
                ("end_23_59_999999", eod, True),
                ("after_1us", (eod_utc + dt.timedelta(microseconds=1)).astimezone(ET), False),
                ("rollback_to_start", sod, True),
            ]
            # On 2027-11-07, also test BOTH 01:30 fold=0 (EDT) and 01:30 fold=1 (EST)!
            if d_str == "2027-11-07":
                instants.append(("nov_fold0", dt.datetime(2027, 11, 7, 1, 30, tzinfo=ET, fold=0), True))
                instants.append(("nov_fold1", dt.datetime(2027, 11, 7, 1, 30, tzinfo=ET, fold=1), True))

            for label, instant, exp_in_effect in instants:
                boundary_checks += 1
                r = cs.load_contest_streak_state(pdir, now=instant)
                if (r is not None) != exp_in_effect:
                    failures.append(f"{d_str} {label}: reader expected {exp_in_effect}, got {r}")
                if exp_in_effect:
                    ds = cs.load_decision_streak_state(pdir, require_contest_state=True, now=instant)
                    if ds.status != "fresh" or ds.best_status != "trusted" or ds.saver_flag_state != "not_earned" or ds.saver_available is not False:
                        failures.append(f"{d_str} {label}: decision wrong {ds}")
                    alerts = health.check(pdir, expected=False, now=instant)
                    # Note: streak=12 not_earned -> 1 saver_state WARN, 0 legacy/expired WARN!
                    if len(alerts) != 1 or alerts[0].source != "saver_state" or f"season {d_obj.year}" not in alerts[0].message:
                        failures.append(f"{d_str} {label}: health wrong {alerts}")
                else:
                    # Without observation in effect and no auto file:
                    # require_contest_state=True raises ContestStateError; health(expected=False) returns [] without reading flag!
                    try:
                        cs.load_decision_streak_state(pdir, require_contest_state=True, now=instant)
                        failures.append(f"{d_str} {label}: expected missing ContestStateError")
                    except cs.ContestStateError:
                        pass
                    if health.check(pdir, expected=False, now=instant) != []:
                        failures.append(f"{d_str} {label}: health expected [], got non-empty")

        # Advancing clock across midnight & New Year (now=None) for all 3 functions + CLI command starting at 23:59:59.9
        ny_dir = Path(tmp) / "ny_cross"
        (ny_dir / "account_state").mkdir(parents=True)
        eod_dec31 = dt.datetime(2027, 12, 31, 23, 59, 59, 999999, tzinfo=ET)
        jan1_after = dt.datetime(2028, 1, 1, 0, 0, 0, 500000, tzinfo=ET)
        (ny_dir / "account_state" / "saver_state.json").write_text('{"state":"used","season":' + ("9" * 5000) + "}", encoding="utf-8")
        (ny_dir / "account_state" / "contest_streak.manual.json").write_text(json.dumps({
            "schema_version": SCHEMA,
            "effective_date": "2027-12-31",
            "season": 2027,
            "expires_at": eod_dec31.isoformat(),
            "active_streak": 8,
            "best_streak": 8,
            "saver_state": "not_earned",
            "recorded_at": "2027-12-31T20:00:00Z",
            "source": "manual_cli",
        }), encoding="utf-8")

        read_counts = {}
        for fn_name, fn in [
            ("load_contest_streak_state", lambda: cs.load_contest_streak_state(ny_dir)),
            ("load_decision_streak_state", lambda: cs.load_decision_streak_state(ny_dir)),
            ("health_check", lambda: health.check(ny_dir, expected=True)),
        ]:
            clk = make_clock([eod_dec31, jan1_after, jan1_after])
            with patch.object(dt, "datetime", clk), patch.object(cs, "datetime", clk), patch.object(health, "datetime", clk):
                out_val = fn()
            read_counts[fn_name] = clk.calls
            if clk.calls != 1:
                failures.append(f"{fn_name} read clock {clk.calls} times instead of 1!")
            if fn_name == "load_decision_streak_state":
                if out_val.source != "contest" or out_val.best_status != "trusted" or out_val.saver_available is not True:
                    failures.append(f"New Year cross decision wrong: {out_val}")
            elif fn_name == "health_check":
                if out_val != []:
                    failures.append(f"New Year cross health expected [], got {out_val}")

        # Command starting at 23:59:59.900000 ET on 2027-12-31 and finishing after midnight (2028-01-01 00:00:00.400000 ET)
        cmd_dir = Path(tmp) / "cmd_midnight"
        t_start = dt.datetime(2027, 12, 31, 23, 59, 59, 900000, tzinfo=ET)
        t_after = dt.datetime(2028, 1, 1, 0, 0, 0, 400000, tzinfo=ET)
        clk_cmd = make_clock([t_start, t_after, t_after])
        # Even if wall clock would return t_after on any 2nd read, command must read once and write/print 2027-12-31!
        with patch.object(dt, "datetime", clk_cmd), patch.object(cs, "datetime", clk_cmd):
            res_cmd = CliRunner().invoke(cli, [
                "set-contest-streak", "--picks-dir", str(cmd_dir),
                "--date", "2027-12-31", "--streak", "4", "--best-streak", "9", "--saver", "active",
            ])
        read_counts["dated_cli_command"] = clk_cmd.calls
        if res_cmd.exit_code != 0 or clk_cmd.calls != 1:
            failures.append(f"Midnight CLI failed or multi-clock: code={res_cmd.exit_code}, calls={clk_cmd.calls}, out={res_cmd.output}")
        cmd_written = json.loads((cmd_dir / "account_state" / "contest_streak.manual.json").read_text(encoding="utf-8"))
        cmd_last = res_cmd.output.splitlines()[-1]
        if (cmd_written["effective_date"] != "2027-12-31"
                or cmd_written["season"] != 2027
                or cmd_written["recorded_at"] != "2028-01-01T04:59:59.900000Z"
                or cmd_last != "contest_streak: 4 (best 9, saver active) for 2027-12-31, in effect today"):
            failures.append(f"Midnight CLI wrong payload/output: {cmd_written} / {cmd_last}")

    return {
        "boundary_checks": boundary_checks,
        "clock_read_counts": read_counts,
        "midnight_cli_written": cmd_written,
        "midnight_cli_output": res_cmd.output.replace(str(cmd_dir), "<CMD_DIR>"),
        "failures": failures,
    }


def main() -> None:
    fs_events: list[tuple] = []
    install_audit_hooks(fs_events)
    sys.path.insert(0, str(BRANCH_SRC))

    cli_dated = test_cli_dated_mode(fs_events)
    cli_legacy = test_legacy_cli_parity_vs_base()
    clocks = test_clocks_and_midnight()

    report = {
        "cli_dated": cli_dated,
        "cli_legacy_vs_base": cli_legacy,
        "clocks_and_midnight": clocks,
    }
    (REVIEW_DIR / "probe3_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if cli_dated["failures"] or cli_legacy["differences"] or clocks["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
