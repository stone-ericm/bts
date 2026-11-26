"""Probe 02b: Early gate exhaustive 1,440-minute sweep, doubles, boundaries, window_min, commit changes, and markers."""
from __future__ import annotations

import gzip
import json
import os
import shlex
import shutil
import socket
import stat
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from click.testing import CliRunner  # noqa: E402
from bts import entry_receipt, picks  # noqa: E402
from bts.cli import _check_pick_entered, cli  # noqa: E402
from bts.daily_decision import write_decision  # noqa: E402

ET = ZoneInfo("America/New_York")
DAY = "2027-03-24"
ENTER_CFG = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'
XW = {1: 1010, 2: 1020}
RIGHT_U1 = {"id": 77, "roundId": 1, "feedId": 101}
WRONG_U1 = {"id": 77, "roundId": 1, "feedId": 102}


def _forbid_network() -> None:
    def _deny(*a, **kw):
        raise AssertionError(f"Unexpected network access: {a}")

    socket.socket.connect = _deny  # type: ignore[assignment]
    socket.socket.connect_ex = _deny  # type: ignore[assignment]
    socket.socket.sendto = _deny  # type: ignore[assignment]
    socket.getaddrinfo = _deny  # type: ignore[assignment]


def _exe(path: Path, body: str) -> None:
    path.write_text(body)
    path.chmod(path.stat().st_mode | stat.S_IXUSR)


def install_real_watchdog_line(work_dir: Path) -> tuple[str, list[str]]:
    home = work_dir / "home"
    (home / "projects" / "bts").mkdir(parents=True)
    (home / "projects" / "bts" / ".env").write_text("")
    (home / ".local" / "bin").mkdir(parents=True)
    _exe(
        home / ".local" / "bin" / "uv",
        '#!/bin/sh\n[ "$1 $2 $3" = "run bts entry-intent" ] || exit 97\n'
        f'shift 2\nPYTHONPATH="{REPO / "src"}" exec "{sys.executable}" -c "from bts.cli import cli; cli()" "$@"\n',
    )
    shims = work_dir / "shims"
    shims.mkdir()
    crontab = work_dir / "crontab.txt"
    _exe(
        shims / "crontab",
        f'#!/bin/sh\nf="{crontab}"\ncase "$1" in\n'
        '  -l) [ -f "$f" ] && cat "$f" || { echo "no crontab for user" >&2; exit 1; } ;;\n'
        '  -) cat > "$f.new" && mv "$f.new" "$f" ;;\n  *) exit 98 ;;\nesac\n',
    )
    (home / ".bts-orchestrator.toml").write_text(ENTER_CFG)
    (work_dir / "tmp").mkdir()
    env = {
        "HOME": str(home),
        "PATH": f"{shims}:/usr/bin:/bin",
        "HEALTHCHECKS_PING_URL": "https://hc-ping.invalid/probe02b",
        "TMPDIR": str(work_dir / "tmp"),
    }
    res = subprocess.run(
        ["bash", str(REPO / "scripts/cron-setup-hetzner.sh"), "install"],
        env=env, capture_output=True, text=True, timeout=20,
    )
    assert res.returncode == 0, res.stderr
    lines = crontab.read_text().splitlines()
    wd_lines = [ln for ln in lines if "bts check-pick-entered" in ln and not ln.startswith("#")]
    assert len(wd_lines) == 1
    line = wd_lines[0]
    tokens = shlex.split(line, comments=True)
    cli_args = tokens[tokens.index("check-pick-entered"):tokens.index(">>")]
    assert "--cron" in cli_args
    return line, cli_args


def cron_slots(line: str, day: str = DAY) -> list[str]:
    minute_field, hour_field, *_ = line.split()
    step = int(minute_field.removeprefix("*/"))
    hours = range(24) if hour_field == "*" else range(10, 24)
    midnight = datetime.fromisoformat(day).replace(tzinfo=ET)
    cur = midnight.astimezone(timezone.utc)
    stop = (midnight + timedelta(days=1)).astimezone(timezone.utc)
    out = []
    while cur < stop:
        et = cur.astimezone(ET)
        if et.hour in hours:
            out.append(et.strftime("%H:%M"))
        cur += timedelta(minutes=step)
    return out


def make_slot(batter_id: int, game_pk: int, gt_iso: str) -> picks.Pick:
    return picks.Pick(
        batter_name=f"Batter {batter_id}",
        batter_id=batter_id,
        team="NYY",
        lineup_position=1,
        pitcher_name="Pitcher",
        pitcher_id=500,
        p_game_hit=0.80,
        flags=[],
        projected_lineup=False,
        game_pk=game_pk,
        game_time=gt_iso,
    )


def write_committed_pick(pdir: Path, gt1_iso: str, *, gt2_iso: str | None = None, day: str = DAY,
                         committed: bool = True, commit_iso: str | None = None) -> None:
    commit_iso = commit_iso or f"{day}T00:01:00-04:00"
    p1 = make_slot(1010, 101, gt1_iso)
    p2 = make_slot(1020, 202, gt2_iso) if gt2_iso else None
    daily = picks.DailyPick(
        day, commit_iso, p1, p2, None,
        notification_sent=committed,
        notification_channel="bluesky_dm" if committed else None,
        notification_id="msg-1" if committed else None,
        delivery_attempted=committed,
        delivered_at=commit_iso if committed else None,
    )
    picks.save_pick(daily, pdir)
    dec_path = pdir / day / "decision.json"
    if committed:
        write_decision(
            day, pdir,
            action="double" if p2 else "single",
            source="mdp",
            delivery_status="delivered",
            scoreable=True,
            primary={"batter_name": p1.batter_name},
            double_down={"batter_name": p2.batter_name} if p2 else None,
            now=datetime.fromisoformat(commit_iso),
        )
    elif dec_path.exists():
        dec_path.unlink()


def sweep_all_1440_minutes(installed_line: str, work_dir: Path) -> dict:
    """Test every single first-pitch minute 00:00..23:59 against the real _check_pick_entered logic."""
    all_96 = cron_slots(installed_line, DAY)
    assert len(all_96) == 96
    base_56 = [s for s in all_96 if s >= "10:00"]
    assert len(base_56) == 56

    sdir = work_dir / "sweep"
    pdir = sdir / "picks"
    pdir.mkdir(parents=True)
    marker = sdir / "health_state" / "pick_entry_check.json"

    calls = []

    def fake_login(*a, **kw):
        calls.append("login")
        return SimpleNamespace(user_id="u1", username="stonehengee", xsid="x1")

    no_base_slot_pitches = []
    shared_window_deferred_pitches = []
    zero_slot_pitches = []
    total_invocations = 0

    with ExitStack() as stack:
        stack.enter_context(patch("bts.leaderboard.auth.load_session_cookies", lambda: {}))
        stack.enter_context(patch("bts.leaderboard.auth.extract_uid", lambda _: "u1"))
        stack.enter_context(patch("bts.leaderboard.auth.fetch_login_session", fake_login))
        stack.enter_context(patch("bts.contest_fetch.fetch_profile", lambda *a, **k: {"predictions": []}))
        stack.enter_context(patch("bts.contest_fetch.fetch_pending_predictions", lambda *a, **k: []))
        stack.enter_context(patch("bts.cli._fetch_rounds", lambda: {1: date.fromisoformat(DAY)}))
        stack.enter_context(patch("bts.cli._fetch_bts_to_mlb", lambda: dict(XW)))
        stack.enter_context(patch("bts.dm.send_dm", lambda *a, **k: None))
        stack.enter_context(patch("click.echo", lambda *a, **k: None))
        stack.enter_context(patch("time.monotonic", lambda: 1000.0))

        for m_of_day in range(1440):
            hh, mm = divmod(m_of_day, 60)
            pitch_hm = f"{hh:02d}:{mm:02d}"
            pitch_dt = datetime.fromisoformat(f"{DAY}T{pitch_hm}:00").replace(tzinfo=ET)
            write_committed_pick(pdir, pitch_dt.isoformat())

            if mm % 5 == 0 or (8 * 60 + 30 <= m_of_day <= 11 * 60 + 30):
                slots_to_run = all_96
            else:
                always = {"00:00", "08:45", "09:00", "09:15", "09:30", "09:45", "10:00", "10:15", "23:30", "23:45"}
                near = {
                    s for s in all_96
                    if -15 <= (pitch_dt - datetime.fromisoformat(f"{DAY}T{s}:00").replace(tzinfo=ET)).total_seconds() / 60 <= 90
                }
                slots_to_run = [s for s in all_96 if s in (always | near)]

            base_reads = []
            for s in slots_to_run:
                if s < "10:00":
                    continue
                if marker.exists():
                    marker.unlink()
                calls.clear()
                now_dt = datetime.fromisoformat(f"{DAY}T{s}:00").replace(tzinfo=ET)
                rec = entry_receipt.EntryReceipt(
                    et_date=DAY, started_at=now_dt, clock=lambda: now_dt,
                    expected_username="stonehengee", picks_dir=pdir,
                )
                try:
                    _check_pick_entered(pdir, "stonehengee", "op.bsky.social", 75, now_dt, rec, cron=False)
                except SystemExit:
                    pass
                if calls:
                    base_reads.append(s)

            f5_reads = []
            f5_deferred = []
            for s in slots_to_run:
                if marker.exists():
                    marker.unlink()
                calls.clear()
                now_dt = datetime.fromisoformat(f"{DAY}T{s}:00").replace(tzinfo=ET)
                rec = entry_receipt.EntryReceipt(
                    et_date=DAY, started_at=now_dt, clock=lambda: now_dt,
                    expected_username="stonehengee", picks_dir=pdir,
                )
                try:
                    _check_pick_entered(pdir, "stonehengee", "op.bsky.social", 75, now_dt, rec, cron=True)
                except SystemExit:
                    pass
                total_invocations += 1
                if rec.outcome == "base_slot_available":
                    assert not calls, f"Account read during base_slot_available at {s} for pitch {pitch_hm}"
                    assert not marker.exists(), f"Marker written during base_slot_available at {s} for pitch {pitch_hm}"
                    f5_deferred.append(s)
                elif calls:
                    f5_reads.append(s)

            if base_reads:
                assert f5_reads == base_reads, f"Pitch {pitch_hm}: F5 changed reads on base-covered slate: {f5_reads} vs {base_reads}"
                if f5_deferred:
                    shared_window_deferred_pitches.append((pitch_hm, f5_deferred, base_reads))
            else:
                assert f5_deferred == [], f"Pitch {pitch_hm}: gate wrongly deferred slate with NO base slot: {f5_deferred}"
                if f5_reads:
                    assert all(s < "10:00" for s in f5_reads)
                    no_base_slot_pitches.append((pitch_hm, f5_reads))
                else:
                    zero_slot_pitches.append(pitch_hm)

    assert zero_slot_pitches == ["00:00", "00:01", "00:02", "00:03", "00:04", "00:05"], zero_slot_pitches
    assert (no_base_slot_pitches[0][0], no_base_slot_pitches[-1][0], len(no_base_slot_pitches)) == ("00:06", "10:05", 600)
    assert (shared_window_deferred_pitches[0][0], shared_window_deferred_pitches[-1][0], len(shared_window_deferred_pitches)) == (
        "10:06", "11:00", 55
    )

    shutil.rmtree(sdir)
    return {
        "minutes_checked": 1440,
        "total_watchdog_invocations": total_invocations,
        "zero_slot_pitches": zero_slot_pitches,
        "early_only_pitch_range": [no_base_slot_pitches[0][0], no_base_slot_pitches[-1][0], len(no_base_slot_pitches)],
        "shared_window_deferred_range": [
            shared_window_deferred_pitches[0][0],
            shared_window_deferred_pitches[-1][0],
            len(shared_window_deferred_pitches),
        ],
        "sample_10_05": dict(no_base_slot_pitches)["10:05"],
        "sample_10_06": {
            "deferred": dict((p, d) for p, d, _ in shared_window_deferred_pitches)["10:06"],
            "reads": dict((p, r) for p, _, r in shared_window_deferred_pitches)["10:06"],
        },
        "sample_10_15": {
            "deferred": dict((p, d) for p, d, _ in shared_window_deferred_pitches)["10:15"],
            "reads": dict((p, r) for p, _, r in shared_window_deferred_pitches)["10:15"],
        },
    }


def check_doubles_boundaries_and_sequences(cli_args: list[str], installed_line: str, work_dir: Path) -> dict:
    """Real CliRunner + real os.fsync checks for doubles, boundaries, 10:15 wrong-game, window-min, and markers."""
    runner = CliRunner()
    all_slots = cron_slots(installed_line, DAY)

    def run_case(name: str):
        cdir = work_dir / name
        pdir = cdir / "picks"
        pdir.mkdir(parents=True)
        cfg = cdir / "config.toml"
        cfg.write_text(ENTER_CFG)
        marker = cdir / "health_state" / "pick_entry_check.json"
        units_dir = entry_receipt.units_snapshot_dir(pdir)
        units_dir.mkdir(parents=True)
        args = list(cli_args)
        args[args.index("--picks-dir") + 1] = str(pdir)
        calls, dms = [], []

        def invoke(now_iso: str, *, pending_rows=(), manual: bool = False, window_min: int | None = None):
            calls.clear()
            dms.clear()
            before_m = marker.read_bytes() if marker.exists() else None
            before_ids = {r["attempt_id"] for r in entry_receipt.discover(pdir, DAY)}
            cmd = [a for a in args if a != "--cron"] if manual else list(args)
            if window_min is not None:
                cmd.extend(["--window-min", str(window_min)])
            cmd.extend(["--now-et", now_iso])

            with ExitStack() as stack:
                stack.enter_context(patch("bts.leaderboard.auth.load_session_cookies", lambda: {}))
                stack.enter_context(patch("bts.leaderboard.auth.extract_uid", lambda _: "u1"))
                stack.enter_context(patch(
                    "bts.leaderboard.auth.fetch_login_session",
                    lambda *a, **k: calls.append("session") or SimpleNamespace(user_id="u1", username="stonehengee", xsid="x1"),
                ))
                stack.enter_context(patch("bts.contest_fetch.fetch_profile", lambda *a, **k: calls.append("profile") or {"predictions": []}))
                stack.enter_context(patch("bts.contest_fetch.fetch_pending_predictions", lambda *a, **k: calls.append("pending") or list(pending_rows)))
                stack.enter_context(patch("bts.cli._fetch_rounds", lambda: calls.append("rounds") or {1: date.fromisoformat(DAY)}))
                stack.enter_context(patch("bts.cli._fetch_bts_to_mlb", lambda: calls.append("xw") or dict(XW)))
                stack.enter_context(patch("bts.dm.send_dm", lambda r, m: dms.append(m)))
                stack.enter_context(patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"))
                stack.enter_context(patch("time.monotonic", lambda: 1000.0))
                res = runner.invoke(cli, cmd)

            after_m = marker.read_bytes() if marker.exists() else None
            new_recs = [r for r in entry_receipt.discover(pdir, DAY) if r["attempt_id"] not in before_ids]
            assert len(new_recs) == 1, f"Expected 1 new sealed receipt, got {len(new_recs)}"
            latest = new_recs[0]
            return {
                "exit_code": res.exit_code,
                "stdout": res.stdout,
                "outcome": latest["outcome"],
                "calls": list(calls),
                "dms": list(dms),
                "marker_changed": before_m != after_m,
                "receipt": latest,
            }

        def morning():
            dms.clear()
            with ExitStack() as stack:
                stack.enter_context(patch("bts.scheduler.fetch_schedule", lambda _: [{"gamePk": 101}]))
                stack.enter_context(patch("bts.dm.send_dm", lambda r, m: dms.append(m)))
                res = runner.invoke(cli, [
                    "day-outcome", "--date", DAY, "--config", str(cfg), "--picks-dir", str(pdir),
                    "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
                ])
            return res.exit_code, json.loads(res.stdout), list(dms)

        return cdir, pdir, marker, units_dir, invoke, morning

    # 1. Doubles with legs in both orders across early-only (09:35 + 19:10) and base-covered (10:15 + 19:10)
    double_results = {}
    for label, g1, g2, exp_deferred, exp_reads in (
        ("dd_early_first", "09:35", "19:10", [], ["08:30", "08:45", "09:00", "09:15"]),
        ("dd_early_second", "19:10", "09:35", [], ["08:30", "08:45", "09:00", "09:15"]),
        ("dd_sole_1000_first", "10:15", "19:10", ["09:00", "09:15", "09:30", "09:45"], ["10:00"]),
        ("dd_sole_1000_second", "19:10", "10:15", ["09:00", "09:15", "09:30", "09:45"], ["10:00"]),
    ):
        cdir, pdir, _, _, invoke, _ = run_case(label)
        write_committed_pick(pdir, f"{DAY}T{g1}:00-04:00", gt2_iso=f"{DAY}T{g2}:00-04:00")
        deferred, reads = [], []
        for s in all_slots:
            if "08:15" <= s <= "10:30" or s in ("18:00", "18:30"):
                r = invoke(f"{DAY}T{s}:00")
                if r["outcome"] == "base_slot_available":
                    deferred.append(s)
                elif "pending" in r["calls"]:
                    reads.append(s)
        assert deferred == exp_deferred, (label, deferred, exp_deferred)
        assert reads == exp_reads, (label, reads, exp_reads)
        double_results[label] = {"deferred": deferred, "reads": reads}
        shutil.rmtree(cdir)

    # 2. Critical regression for 10:15 pitch (where 10:00 is the ONLY base slot — kills "start at 10:15" mutant!)
    cdir, pdir, marker, units_dir, invoke, morning = run_case("sole_1000_wrong_game_1015")
    write_committed_pick(pdir, f"{DAY}T10:15:00-04:00")
    row_match = [{"roundId": 1, "unitId": 77, "playerId": 1, "number": 1, "result": None}]
    for early_s in ("09:00", "09:15", "09:30", "09:45"):
        r = invoke(f"{DAY}T{early_s}:00", pending_rows=row_match)
        assert r["outcome"] == "base_slot_available" and not r["calls"] and not r["dms"] and not r["marker_changed"]
        assert "original schedule has a slot inside the window" in r["stdout"]
    (units_dir / f"{DAY}T140000Z.json.gz").write_bytes(gzip.compress(json.dumps({"units": [WRONG_U1]}).encode(), mtime=0))
    r_1000 = invoke(f"{DAY}T10:00:00", pending_rows=row_match)
    assert r_1000["outcome"] == "observed" and r_1000["exit_code"] == 1
    assert len(r_1000["dms"]) == 1 and "the entry is recorded in game 102" in r_1000["dms"][0]
    assert json.loads(marker.read_text())["game_unresolved"] is True
    m_code, m_body, m_dms = morning()
    assert m_code == 1 and m_body["state"] == "entry_unverifiable" and len(m_dms) == 1
    shutil.rmtree(cdir)

    # 3. Exact boundaries: 09:59:59 vs 10:00:00; 10:05:00 vs 10:05:01; 75-min boundary; window-min other than 75
    cdir, pdir, marker, units_dir, invoke, _ = run_case("boundaries")
    (units_dir / f"{DAY}T010000Z.json.gz").write_bytes(gzip.compress(json.dumps({"units": [RIGHT_U1]}).encode(), mtime=0))
    write_committed_pick(pdir, f"{DAY}T10:50:00-04:00")
    r_095959 = invoke(f"{DAY}T09:59:59")
    assert r_095959["outcome"] == "base_slot_available" and not r_095959["calls"]
    r_100000 = invoke(f"{DAY}T10:00:00")
    assert r_100000["outcome"] == "observed" and "pending" in r_100000["calls"]

    if marker.exists():
        marker.unlink()
    write_committed_pick(pdir, f"{DAY}T10:05:00-04:00")
    assert invoke(f"{DAY}T09:45:00")["outcome"] == "observed"
    if marker.exists():
        marker.unlink()
    write_committed_pick(pdir, f"{DAY}T10:05:01-04:00")
    assert invoke(f"{DAY}T09:45:00")["outcome"] == "base_slot_available"
    assert invoke(f"{DAY}T10:00:00")["outcome"] == "observed"

    # Exact 75-min boundary at 09:45 (11:00:00 = 75.0m inside -> base_slot_available; 11:01:00 = 76.0m -> outside_window)
    if marker.exists():
        marker.unlink()
    write_committed_pick(pdir, f"{DAY}T11:00:00-04:00")
    assert invoke(f"{DAY}T09:45:00")["outcome"] == "base_slot_available"
    write_committed_pick(pdir, f"{DAY}T11:01:00-04:00")
    assert invoke(f"{DAY}T09:45:00")["outcome"] == "outside_window"

    # Custom --window-min values with --cron
    if marker.exists():
        marker.unlink()
    write_committed_pick(pdir, f"{DAY}T10:05:00-04:00")
    assert invoke(f"{DAY}T09:45:00", window_min=20)["outcome"] == "observed"
    if marker.exists():
        marker.unlink()
    write_committed_pick(pdir, f"{DAY}T10:15:00-04:00")
    assert invoke(f"{DAY}T09:55:00", window_min=20)["outcome"] == "base_slot_available"
    write_committed_pick(pdir, f"{DAY}T23:55:00-04:00")
    assert invoke(f"{DAY}T09:45:00", window_min=900)["outcome"] == "base_slot_available"
    shutil.rmtree(cdir)

    # 4. Pick committed/changed between early run and 10:00, and already-confirmed / alerted markers
    cdir, pdir, marker, units_dir, invoke, morning = run_case("sequences_and_markers")
    (units_dir / f"{DAY}T010000Z.json.gz").write_bytes(gzip.compress(json.dumps({"units": [RIGHT_U1]}).encode(), mtime=0))
    write_committed_pick(pdir, f"{DAY}T06:10:00-04:00", committed=False)
    assert invoke(f"{DAY}T05:30:00")["outcome"] == "not_committed"
    write_committed_pick(pdir, f"{DAY}T10:50:00-04:00", committed=True, commit_iso=f"{DAY}T09:40:00-04:00")
    assert invoke(f"{DAY}T09:45:00", pending_rows=row_match)["outcome"] == "base_slot_available"
    assert not marker.exists()
    write_committed_pick(pdir, f"{DAY}T13:10:00-04:00", committed=True, commit_iso=f"{DAY}T09:55:00-04:00")
    assert invoke(f"{DAY}T10:00:00", pending_rows=row_match)["outcome"] == "outside_window"
    r_1200 = invoke(f"{DAY}T12:00:00", pending_rows=row_match)
    assert r_1200["outcome"] == "observed" and json.loads(marker.read_text())["status"] == "confirmed"
    shutil.rmtree(cdir)

    cdir, pdir, marker, units_dir, invoke, morning = run_case("early_escalation_and_manual_confirm")
    (units_dir / f"{DAY}T010000Z.json.gz").write_bytes(gzip.compress(json.dumps({"units": [RIGHT_U1]}).encode(), mtime=0))
    write_committed_pick(pdir, f"{DAY}T10:05:00-04:00", commit_iso=f"{DAY}T08:50:00-04:00")
    r_0900 = invoke(f"{DAY}T09:00:00", pending_rows=[])
    r_0915 = invoke(f"{DAY}T09:15:00", pending_rows=[])
    r_0930 = invoke(f"{DAY}T09:30:00", pending_rows=[])
    r_0945 = invoke(f"{DAY}T09:45:00", pending_rows=row_match)
    assert (len(r_0900["dms"]), len(r_0915["dms"]), len(r_0930["dms"]), len(r_0945["dms"])) == (1, 0, 1, 1)
    assert "✅ BTS pick entry confirmed" in r_0945["dms"][0]
    assert morning()[0] == 0
    shutil.rmtree(cdir)

    cdir, pdir, marker, units_dir, invoke, morning = run_case("manual_confirm_then_cron")
    (units_dir / f"{DAY}T010000Z.json.gz").write_bytes(gzip.compress(json.dumps({"units": [RIGHT_U1]}).encode(), mtime=0))
    write_committed_pick(pdir, f"{DAY}T10:50:00-04:00", commit_iso=f"{DAY}T09:35:00-04:00")
    r_man = invoke(f"{DAY}T09:40:00", pending_rows=row_match, manual=True)
    assert r_man["outcome"] == "observed" and json.loads(marker.read_text())["status"] == "confirmed"
    r_0945_cron = invoke(f"{DAY}T09:45:00", pending_rows=row_match, manual=False)
    assert r_0945_cron["outcome"] == "base_slot_available" and not r_0945_cron["marker_changed"]
    r_1000_cron = invoke(f"{DAY}T10:00:00", pending_rows=row_match, manual=False)
    assert r_1000_cron["outcome"] == "already_confirmed" and r_1000_cron["receipt"]["references"]["confirmed_by"] == r_man["receipt"]["attempt_id"]
    assert morning()[0] == 0
    shutil.rmtree(cdir)

    return {
        "doubles": double_results,
        "sole_1000_wrong_game_1015": "PASS",
        "boundaries_and_window_min": "PASS",
        "mid_morning_commit_changes_and_markers": "PASS",
    }


def main() -> None:
    t0 = time.perf_counter()
    _forbid_network()
    work = OUT_DIR / ".work_probe_02b"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        installed_line, cli_args = install_real_watchdog_line(work / "inst")
        sweep = sweep_all_1440_minutes(installed_line, work)
        edges = check_doubles_boundaries_and_sequences(cli_args, installed_line, work)
        elapsed = round(time.perf_counter() - t0, 3)
        report = {
            "status": "PASS",
            "elapsed_sec": elapsed,
            "installed_line": installed_line,
            "minute_by_minute_sweep": sweep,
            "doubles_boundaries_and_sequences": edges,
        }
        (OUT_DIR / "log_02b_early_gate_and_doubles.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"PASS probe_02b in {elapsed}s: 1440 mins ({sweep['total_watchdog_invocations']} runs), "
            f"early-only={sweep['early_only_pitch_range']}, deferred={sweep['shared_window_deferred_range']}"
        )
    finally:
        if work.exists():
            shutil.rmtree(work)


if __name__ == "__main__":
    main()
