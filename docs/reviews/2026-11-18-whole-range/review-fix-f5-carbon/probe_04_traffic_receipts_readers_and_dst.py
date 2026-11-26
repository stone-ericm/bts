"""Probe 04: Extra-slot traffic & receipts (00:00..09:45 x 8 states), reader compatibility, F1b lock/timeout & DST."""
from __future__ import annotations

import gzip
import json
import os
import re
import shlex
import shutil
import socket
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
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "src"))

from click.testing import CliRunner  # noqa: E402
from bts import day_outcome, entry_receipt, picks, receipt_io  # noqa: E402
from bts.cli import cli  # noqa: E402
from bts.daily_decision import load_decision_bytes, write_decision  # noqa: E402
from bts.data import backup as bts_backup  # noqa: E402
from bts.health import pick_entry as health_pick_entry  # noqa: E402
from tests.test_entry_intent import ENTRY_CMD, _exe  # noqa: E402

ET = ZoneInfo("America/New_York")
DAY = "2027-03-24"
PREV_DAY = "2027-03-23"
ENTER_CFG = '[scheduler]\nentry_intent = "enter"\npick_delivery = "dm"\ncontest_state_required = true\n'


def _forbid_network() -> None:
    def _deny(*a, **kw):
        raise AssertionError(f"Unexpected network access: {a}")

    socket.socket.connect = _deny  # type: ignore[assignment]
    socket.socket.connect_ex = _deny  # type: ignore[assignment]
    socket.socket.sendto = _deny  # type: ignore[assignment]
    socket.getaddrinfo = _deny  # type: ignore[assignment]


def _make_pick(pdir: Path, pitch_iso: str, *, day: str = DAY, committed: bool = True) -> None:
    slot = picks.Pick(
        batter_name="Batter 1010", batter_id=1010, team="NYY", lineup_position=1,
        pitcher_name="Pitcher", pitcher_id=500, p_game_hit=0.81, flags=[], projected_lineup=False,
        game_pk=101, game_time=pitch_iso,
    )
    commit_iso = f"{day}T00:01:00-04:00"
    daily = picks.DailyPick(
        day, commit_iso, slot, None, None,
        notification_sent=committed,
        notification_channel="bluesky_dm" if committed else None,
        notification_id="m1" if committed else None,
        delivery_attempted=committed,
        delivered_at=commit_iso if committed else None,
    )
    picks.save_pick(daily, pdir)
    if committed:
        write_decision(
            day, pdir, action="single", source="mdp", delivery_status="delivered", scoreable=True,
            primary={"batter_name": "Batter 1010", "batter_id": 1010, "team": "NYY", "game_pk": 101, "p_game_hit": 0.81},
            now=datetime.fromisoformat(commit_iso),
        )


def check_extra_40_slots_traffic_and_receipts(work_dir: Path) -> dict:
    """Test all 40 extra slots (00:00..09:45) across 8 ordinary local states (320 runs)."""
    runner = CliRunner()
    extra_slots = [f"{h:02d}:{m:02d}" for h in range(10) for m in (0, 15, 30, 45)]
    assert len(extra_slots) == 40 and (extra_slots[0], extra_slots[-1]) == ("00:00", "09:45")

    states = [
        "no_pick_file",
        "yesterday_only_confirmed",
        "today_preview_0610",
        "today_preview_1910",
        "today_committed_1910",
        "today_committed_1050",
        "today_committed_1015",
        "today_committed_0610_already_confirmed",
    ]

    outcome_summary = {}
    sample_base_slot_receipt = None

    def fail_transport(*a, **kw):
        raise AssertionError("Unexpected authenticated/public transport or DM call on local-only extra slot!")

    for st_idx, state in enumerate(states):
        sdir = work_dir / f"st_{st_idx}"
        pdir = sdir / "picks"
        pdir.mkdir(parents=True)
        marker = sdir / "health_state" / "pick_entry_check.json"

        if state == "yesterday_only_confirmed":
            _make_pick(pdir, f"{PREV_DAY}T19:10:00-04:00", day=PREV_DAY, committed=True)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"date": PREV_DAY, "status": "confirmed", "receipt": "a" * 32}))
        elif state == "today_preview_0610":
            _make_pick(pdir, f"{DAY}T06:10:00-04:00", committed=False)
        elif state == "today_preview_1910":
            _make_pick(pdir, f"{DAY}T19:10:00-04:00", committed=False)
        elif state == "today_committed_1910":
            _make_pick(pdir, f"{DAY}T19:10:00-04:00", committed=True)
        elif state == "today_committed_1050":
            _make_pick(pdir, f"{DAY}T10:50:00-04:00", committed=True)
        elif state == "today_committed_1015":
            _make_pick(pdir, f"{DAY}T10:15:00-04:00", committed=True)
        elif state == "today_committed_0610_already_confirmed":
            _make_pick(pdir, f"{DAY}T06:10:00-04:00", committed=True)
            marker.parent.mkdir(parents=True, exist_ok=True)
            marker.write_text(json.dumps({"date": DAY, "status": "confirmed", "receipt": "b" * 32}))

        before_marker = marker.read_bytes() if marker.exists() else None
        counts: dict[str, int] = {}

        with ExitStack() as stack:
            stack.enter_context(patch("bts.leaderboard.auth.load_session_cookies", fail_transport))
            stack.enter_context(patch("bts.leaderboard.auth.fetch_login_session", fail_transport))
            stack.enter_context(patch("bts.contest_fetch.fetch_profile", fail_transport))
            stack.enter_context(patch("bts.contest_fetch.fetch_pending_predictions", fail_transport))
            stack.enter_context(patch("bts.cli._fetch_rounds", fail_transport))
            stack.enter_context(patch("bts.cli._fetch_bts_to_mlb", fail_transport))
            stack.enter_context(patch("bts.dm.send_dm", fail_transport))
            stack.enter_context(patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"))
            if st_idx >= 1:
                stack.enter_context(patch("os.fsync", lambda fd: None))

            for hm in extra_slots:
                res = runner.invoke(cli, [
                    "check-pick-entered", "--cron", "--picks-dir", str(pdir),
                    "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
                    "--now-et", f"{DAY}T{hm}:00",
                ])
                assert res.exit_code == 0, (state, hm, res.output)
                after_marker = marker.read_bytes() if marker.exists() else None
                assert after_marker == before_marker, f"Marker mutated in {state} at {hm}"

        recs = entry_receipt.discover(pdir, DAY)
        assert len(recs) == 40, (state, len(recs))
        # Verify each JSON also has a .sealed companion
        rec_dir = entry_receipt.receipts_root(pdir) / DAY
        assert len(list(rec_dir.glob("*.json.sealed"))) == 40
        for r in recs:
            counts[r["outcome"]] = counts.get(r["outcome"], 0) + 1
            if r["outcome"] == "base_slot_available" and sample_base_slot_receipt is None:
                sample_base_slot_receipt = r
        outcome_summary[state] = counts
        shutil.rmtree(sdir)

    assert outcome_summary["no_pick_file"] == {"no_pick_file": 40}
    assert outcome_summary["yesterday_only_confirmed"] == {"no_pick_file": 40}
    assert outcome_summary["today_preview_0610"] == {"not_committed": 40}
    assert outcome_summary["today_preview_1910"] == {"not_committed": 40}
    assert outcome_summary["today_committed_1910"] == {"outside_window": 40}
    assert outcome_summary["today_committed_1050"] == {"outside_window": 39, "base_slot_available": 1}
    assert outcome_summary["today_committed_1015"] == {"outside_window": 36, "base_slot_available": 4}
    assert outcome_summary["today_committed_0610_already_confirmed"] == {"outside_window": 35, "already_confirmed": 5}

    assert sample_base_slot_receipt is not None
    r = sample_base_slot_receipt
    assert r["schema"] == "bts_pick_entry_receipt_v1"
    assert r["outcome"] == "base_slot_available"
    assert r["detail"] is None and r["exit_code"] == 0 and r["degraded"] == []
    assert r["selection"] is not None and r["cutoff_at"] is not None and r["minutes_to_pitch"] == 65.0
    for none_key in ("account", "observation", "verifier", "qualification", "references", "error", "failed_at", "marker_status"):
        assert r[none_key] is None, (none_key, r[none_key])

    return {"runs_checked": 320, "outcomes_by_state": outcome_summary}


def check_all_readers(work_dir: Path) -> dict:
    """Check day_outcome, health/pick_entry.py, backup ops set, and scripts/audit/."""
    rdir = work_dir / "readers"
    pdir = rdir / "data" / "picks"
    pdir.mkdir(parents=True)
    _make_pick(pdir, f"{DAY}T10:50:00-04:00", committed=True)
    dec, dec_raw = load_decision_bytes(DAY, pdir)
    dec_sha = entry_receipt._sha(dec_raw)

    def mk_rec(outcome: str, **extra) -> dict:
        base = {
            "schema": "bts_pick_entry_receipt_v1",
            "attempt_id": "1" * 32,
            "et_date": DAY,
            "season": 2027,
            "outcome": outcome,
            "selection": {"commit": {"decision_sha256": dec_sha}},
            "degraded": [],
        }
        base.update(extra)
        return base

    obs_confirmed = mk_rec(
        "observed",
        account={"username": "stonehengee"},
        observation={"before_cutoff": True, "response_completed_at": f"{DAY}T10:00:02-04:00"},
        verifier={"ok": True, "reason": "match"},
        qualification={"all_confirmed": True, "slots": [{"state": "confirmed"}]},
    )
    obs_absent = mk_rec(
        "observed",
        account={"username": "stonehengee"},
        observation={"before_cutoff": True, "response_completed_at": f"{DAY}T10:00:02-04:00"},
        verifier={"ok": False, "reason": "no_pick"},
        qualification={"all_confirmed": False, "slots": [{"state": "missing"}]},
    )

    # 1. day_outcome: 96 receipts with base_slot_available + 1 confirmed
    full_96_ok = (
        [mk_rec("no_pick_file")] * 39
        + [mk_rec("base_slot_available")]
        + [obs_confirmed]
        + [mk_rec("already_confirmed")] * 2
        + [mk_rec("outside_window")] * 53
    )
    assert len(full_96_ok) == 96
    out_ok = day_outcome.classify_day(
        date=DAY, games=12, decision=dec, decision_sha256=dec_sha,
        receipts=full_96_ok, intent="enter", expected_username="stonehengee",
    )
    assert out_ok.ok and out_ok.state == "entered"
    assert out_ok.detail == "The entry was confirmed in 1 of 96 checks before the cutoff."

    # Adversarial shape: base_slot_available dressed with confirmed fields MUST NEVER count as confirmation or absence!
    dressed_bsa = obs_confirmed | {"outcome": "base_slot_available"}
    dressed_bsa_absent = obs_absent | {"outcome": "base_slot_available"}
    out_unverif = day_outcome.classify_day(
        date=DAY, games=12, decision=dec, decision_sha256=dec_sha,
        receipts=[dressed_bsa, dressed_bsa_absent, mk_rec("outside_window")],
        intent="enter", expected_username="stonehengee",
    )
    assert not out_unverif.ok and out_unverif.state == "entry_unverifiable"
    assert "base_slot_available x2, outside_window x1" in out_unverif.detail

    # 2. health.pick_entry: publish real base_slot_available + outside_window receipts on disk and verify behavior
    rec_path = entry_receipt.receipts_root(pdir) / DAY / "20270324T094500000000-" + ("2" * 32) + ".json"
    receipt_io.publish(rec_path, json.dumps(mk_rec("base_slot_available")).encode())
    marker_path = pdir.parent / "health_state" / "pick_entry_check.json"
    assert health_pick_entry.check(pdir, today=date.fromisoformat(DAY), now=datetime.fromisoformat(f"{DAY}T23:00:00+00:00")) == []
    marker_path.write_text(json.dumps({"date": DAY, "status": "confirmed", "reason": "match"}))
    assert health_pick_entry.check(pdir, today=date.fromisoformat(DAY), now=datetime.fromisoformat(f"{DAY}T23:00:00+00:00")) == []
    marker_path.write_text(json.dumps({"date": DAY, "status": "alerted", "reason": "no_pick"}))
    alerts = health_pick_entry.check(pdir, today=date.fromisoformat(DAY), now=datetime.fromisoformat(f"{DAY}T23:00:00+00:00"))
    assert len(alerts) == 1 and alerts[0].level == "WARN"

    # 3. bts.data.backup.run_backup("ops", ...): includes data/picks and data/health_state (with receipts + seals)
    restic_calls = []

    def fake_restic_runner(cmd, **kw):
        restic_calls.append(list(cmd))
        if "backup" in cmd:
            out = json.dumps({"message_type": "summary", "snapshot_id": "snap1234", "files_new": 3, "data_added": 512}) + "\n"
            return SimpleNamespace(returncode=0, stdout=out, stderr="")
        return SimpleNamespace(returncode=0, stdout="[]\n", stderr="")

    b_status = bts_backup.run_backup(
        "ops", rdir,
        env={"RESTIC_PASSWORD": "pw", "BTS_RESTIC_REPOSITORY": "/local/repo", "BTS_RESTIC_BIN": "/usr/bin/restic"},
        runner=fake_restic_runner,
    )
    assert b_status["ok"] is True and b_status["snapshot_id"] == "snap1234"
    backup_cmd = restic_calls[0]
    assert "data/picks" in backup_cmd and "data/health_state" in backup_cmd

    # 4. Scan all scripts/audit/**/*.py to confirm none reads pick_entry_receipts or base_slot_available
    audit_files = sorted((REPO / "scripts" / "audit").rglob("*.py"))
    audit_hits = [
        str(p.relative_to(REPO)) for p in audit_files
        if "pick_entry_receipt" in p.read_text() or "base_slot_available" in p.read_text()
    ]
    assert audit_hits == [], audit_hits

    shutil.rmtree(rdir)
    return {
        "day_outcome_96_ok": out_ok.detail,
        "day_outcome_dressed_bsa_rejected": out_unverif.detail,
        "health_pick_entry": "PASS",
        "backup_ops_set": "PASS",
        "audit_scripts_scanned": len(audit_files),
        "audit_receipt_consumers": audit_hits,
    }


def check_f1b_lock_dst_and_minute_sharing(work_dir: Path) -> dict:
    """Verify F1b lock contention at 05:15 --cron, cron minute sharing, and synthetic DST wall-time behavior."""
    # 1. Inspect every cron line generated by scripts/cron-setup-hetzner.sh under enter
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
    shims.mkdir(parents=True, exist_ok=True)
    crontab = work_dir / "crontab.txt"
    _exe(shims / "crontab", f'#!/bin/sh\n[ "$1" = "-" ] && cat > "{crontab}" || cat "{crontab}" 2>/dev/null || {{ echo "no crontab for user" >&2; exit 1; }}\n')
    (home / ".bts-orchestrator.toml").write_text(ENTER_CFG)
    (work_dir / "tmp").mkdir(parents=True, exist_ok=True)
    res = subprocess.run(
        ["bash", str(REPO / "scripts/cron-setup-hetzner.sh"), "install"],
        env={"HOME": str(home), "PATH": f"{shims}:/usr/bin:/bin", "HEALTHCHECKS_PING_URL": "https://hc.invalid/x", "TMPDIR": str(work_dir / "tmp")},
        capture_output=True, text=True, timeout=20,
    )
    assert res.returncode == 0
    lines = [ln for ln in crontab.read_text().splitlines() if ln and not ln.startswith("#")]
    fetch_schedules = [" ".join(ln.split()[:5]) for ln in lines if "fetch-contest-streak" in ln]
    assert fetch_schedules == ["10 1 * * *", "10 2 * * *", "30 10 * * *", "30 13 * * *"]
    wd_line, = [ln for ln in lines if ENTRY_CMD in ln]

    # 2. Real OS subprocess lock contention + real /usr/bin/timeout on an early 05:15 --cron run!
    lock_file = work_dir / "early.lock"
    ready_flag = work_dir / "holder_ready"
    release_flag = work_dir / "holder_release"
    log_file = home / "logs" / "cron.log"
    log_file.parent.mkdir(parents=True, exist_ok=True)
    log_file.touch()

    shell_body = wd_line.split(maxsplit=5)[5]
    flock_cmd = [sys.executable, "-m", "tests.test_wr_f1b_lock", "flock"]
    holder_py = (
        f"import time; from pathlib import Path; Path({str(ready_flag)!r}).write_text('1'); "
        f"deadline = time.monotonic() + 8\n"
        f"while not Path({str(release_flag)!r}).exists() and time.monotonic() < deadline: time.sleep(0.02)\n"
        f"print('early-holder-done', flush=True)"
    )
    holder_shell = (
        shell_body
        .replace("/usr/bin/flock", shlex.join(flock_cmd), 1)
        .replace("/home/bts/.check-pick-entered.lock", shlex.quote(str(lock_file)), 1)
        .replace("/usr/bin/timeout --verbose -k 30 840", "/usr/bin/timeout --verbose -k 1 6", 1)
    )
    holder_shell = re.sub(
        r"\S+/uv run bts check-pick-entered .*? >>",
        lambda _: f"{shlex.quote(sys.executable)} -c {shlex.quote(holder_py)} >>",
        holder_shell,
        count=1,
    )
    env_sub = {**os.environ, "PYTHONPATH": str(REPO), "BTS_TEST_TRACE": str(work_dir / "trace")}
    p_holder = subprocess.Popen(["/bin/sh", "-c", holder_shell], env=env_sub)
    try:
        deadline = time.monotonic() + 5
        while not ready_flag.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert ready_flag.exists()
        # Contender at 05:15 must exit 1 immediately without writing to cron.log
        p_contender = subprocess.run(["/bin/sh", "-c", holder_shell], env={**env_sub, "BTS_TEST_TRACE": str(work_dir / "trace2")}, timeout=5)
        assert p_contender.returncode == 1
        release_flag.write_text("1")
        assert p_holder.wait(timeout=5) == 0
        assert "early-holder-done" in log_file.read_text()
    finally:
        if p_holder.poll() is None:
            p_holder.kill()
            p_holder.wait(timeout=5)

    # 3. Verify synthetic DST wall-time behavior matches Amendment 1's Known Limit 1 truthfully
    runner = CliRunner()
    dst_dir = work_dir / "dst_wall"
    pdir = dst_dir / "picks"
    pdir.mkdir(parents=True)
    # Spring forward 2027-03-14: 02:00 jumps to 03:00. A synthetic 03:05 EDT pitch at 01:45 EST is 20 real min away,
    # but wall-time subtraction (same ZoneInfo singleton) computes 80 min -> outside_window!
    _make_pick(pdir, "2027-03-14T03:05:00-04:00", day="2027-03-14", committed=True)
    with patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"), patch("os.fsync", lambda fd: None):
        r_spring = runner.invoke(cli, [
            "check-pick-entered", "--cron", "--picks-dir", str(pdir), "--now-et", "2027-03-14T01:45:00",
        ])
    assert r_spring.exit_code == 0 and "outside window (80 min to pitch)" in r_spring.stdout

    return {
        "fetch_schedules": fetch_schedules,
        "early_slot_lock_contention": "PASS",
        "dst_wall_time_limit_verified": r_spring.stdout.strip(),
    }


def main() -> None:
    t0 = time.perf_counter()
    _forbid_network()
    work = OUT_DIR / ".work_probe_04"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        traffic = check_extra_40_slots_traffic_and_receipts(work / "traffic")
        readers = check_all_readers(work / "readers")
        f1b_dst = check_f1b_lock_dst_and_minute_sharing(work / "f1b_dst")
        elapsed = round(time.perf_counter() - t0, 3)
        report = {
            "status": "PASS",
            "elapsed_sec": elapsed,
            "extra_slots_traffic_and_receipts": traffic,
            "readers": readers,
            "f1b_lock_dst_and_sharing": f1b_dst,
        }
        (OUT_DIR / "log_04_traffic_receipts_readers_and_dst.json").write_text(json.dumps(report, indent=2) + "\n")
        print(f"PASS probe_04 in {elapsed}s: 320 extra-slot runs local-only, all readers compatible, F1b/DST verified")
    finally:
        if work.exists():
            shutil.rmtree(work)


if __name__ == "__main__":
    main()
