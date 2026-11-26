"""Probe 05: Operator texts (R64 audit + repo scan + 2025 game-time check) and Standing Rows (interrupts, bad files, midnight/NYE, failed/unconfirmed sends, restarts)."""
from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import socket
import subprocess
import sys
import time
from contextlib import ExitStack
from datetime import date, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO / "src"))

from click.testing import CliRunner  # noqa: E402
from bts import entry_receipt, picks, receipt_io  # noqa: E402
from bts.cli import cli  # noqa: E402
from bts.daily_decision import write_decision  # noqa: E402

ET = ZoneInfo("America/New_York")
DAY = "2027-03-24"
PREV_DAY = "2027-03-23"


def _forbid_network() -> None:
    def _deny(*a, **kw):
        raise AssertionError(f"Unexpected network access: {a}")

    socket.socket.connect = _deny  # type: ignore[assignment]
    socket.socket.connect_ex = _deny  # type: ignore[assignment]
    socket.socket.sendto = _deny  # type: ignore[assignment]
    socket.getaddrinfo = _deny  # type: ignore[assignment]


def audit_texts_and_2025_feeds() -> dict:
    # 1. Cron script syntax and header check
    bash_n = subprocess.run(["bash", "-n", str(REPO / "scripts/cron-setup-hetzner.sh")], capture_output=True, text=True, timeout=10)
    assert bash_n.returncode == 0, bash_n.stderr

    cron_text = (REPO / "scripts/cron-setup-hetzner.sh").read_text()
    assert "10-23" not in cron_text, "Stale 10-23 found in scripts/cron-setup-hetzner.sh"
    assert "#   */15 *     — Every 15 minutes, all day ET, the entry watchdog can DM about a missing," in cron_text

    # 2. ARCHITECTURE.md check
    arch_text = (REPO / "ARCHITECTURE.md").read_text()
    assert "10-23" not in arch_text, "Stale 10-23 found in ARCHITECTURE.md"
    assert "(cron every 15 minutes, all day ET)" in arch_text

    # 3. docs/ops/pick-entry-receipt-v1.md check
    rec_doc = (REPO / "docs/ops/pick-entry-receipt-v1.md").read_text()
    assert "56 slots" not in rec_doc and "10-23" not in rec_doc
    assert "96 nominal slots on an ordinary day, 92 on the spring clock-change day, and 100 on the fall clock-change day." in rec_doc
    assert "| `base_slot_available` | no attempt |" in rec_doc

    # 4. CLI --help and stdout line check
    runner = CliRunner()
    help_res = runner.invoke(cli, ["check-pick-entered", "--help"])
    assert help_res.exit_code == 0
    assert "--cron" in help_res.stdout
    assert "Defer early checks when the original schedule has a slot" in " ".join(help_res.stdout.split())

    # 5. Scan current non-review docs/config/scripts/src for stale `10-23` or `56 slots`
    stale_hits = []
    for target in [REPO / "CLAUDE.md", REPO / "README.md", REPO / "INCIDENT.md",
                   *(REPO / "docs/ops").glob("*.md"), *(REPO / "deploy").rglob("*"), *(REPO / "config").rglob("*")]:
        if not target.is_file():
            continue
        txt = target.read_text(errors="replace")
        for pat in ("*/15 10-23", "10-23 * * *", "56 slots"):
            if pat in txt:
                stale_hits.append(f"{target.relative_to(REPO)}:{pat}")
    assert stale_hits == [], f"Unexpected stale schedule strings in live docs/configs: {stale_hits}"

    # 6. Verify 2025 regular-season feed game times against contract limits (no starts 00:00..05:59 ET, latest <= 22:40 ET)
    gt_path = REPO / "docs/reviews/2026-11-18-whole-range/contract-check-f5/game-times-2025.json"
    records = json.loads(gt_path.read_text())
    et_hm_list = []
    non_zero_sec = []
    cross_date_games = []
    for item in records:
        et_hms, et_iso, off_date, *_ = item
        dt_et = datetime.fromisoformat(et_iso)
        et_hm_list.append(dt_et.strftime("%H:%M"))
        if dt_et.second != 0 or not et_hms.endswith(":00"):
            non_zero_sec.append(et_iso)
        if off_date and dt_et.date().isoformat() != off_date:
            cross_date_games.append((off_date, dt_et.isoformat()))
    assert len(et_hm_list) == 2425, len(et_hm_list)
    earliest_2025, latest_2025 = min(et_hm_list), max(et_hm_list)
    assert earliest_2025 >= "06:00", earliest_2025
    assert latest_2025 <= "22:40", latest_2025
    assert non_zero_sec == [], non_zero_sec

    return {
        "cron_bash_n": "PASS",
        "live_docs_stale_hits": stale_hits,
        "feeds_2025_checked": len(et_hm_list),
        "feeds_2025_earliest_et": earliest_2025,
        "feeds_2025_latest_et": latest_2025,
        "feeds_2025_nonzero_seconds": len(non_zero_sec),
        "feeds_2025_cross_official_date": len(cross_date_games),
    }


def _save_pick(pdir: Path, day: str, pitch_iso: str, *, delivery: str = "delivered", committed: bool = True) -> None:
    slot = picks.Pick(
        batter_name="Batter 1010", batter_id=1010, team="NYY", lineup_position=1,
        pitcher_name="Pitcher", pitcher_id=500, p_game_hit=0.80, flags=[], projected_lineup=False,
        game_pk=101, game_time=pitch_iso,
    )
    commit_iso = f"{day}T00:01:00-04:00"
    daily = picks.DailyPick(
        day, commit_iso, slot, None, None,
        notification_sent=(committed and delivery == "delivered"),
        notification_channel="bluesky_dm" if committed else None,
        notification_id="m1" if (committed and delivery == "delivered") else None,
        delivery_attempted=committed,
        delivered_at=commit_iso if (committed and delivery == "delivered") else None,
    )
    picks.save_pick(daily, pdir)
    if committed:
        write_decision(
            day, pdir, action="single", source="mdp", delivery_status=delivery, scoreable=True,
            primary={"batter_name": "Batter 1010", "batter_id": 1010, "team": "NYY", "game_pk": 101, "p_game_hit": 0.80},
            now=datetime.fromisoformat(commit_iso),
        )


def check_standing_rows(work_dir: Path) -> dict:
    runner = CliRunner()

    # 1. Interrupt mid-step during base_slot_available receipt publication (temp write vs post-rename pre-seal)
    for stage in ("tmp_write", "before_seal"):
        idir = work_dir / f"int_{stage}"
        pdir = idir / "picks"
        pdir.mkdir(parents=True)
        _save_pick(pdir, f"{DAY}T10:50:00-04:00")
        marker = idir / "health_state" / "pick_entry_check.json"

        real_replace = os.replace
        real_open = open

        def boom_open(file, mode="r", *a, **kw):
            if stage == "tmp_write" and "pick_entry_receipts" in str(file) and str(file).endswith(".tmp"):
                raise OSError("simulated interrupt during receipt temp write")
            return real_open(file, mode, *a, **kw)

        def boom_replace(src, dst):
            if stage == "before_seal" and str(dst).endswith(".sealed"):
                raise OSError("simulated interrupt before seal rename")
            return real_replace(src, dst)

        with patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"), \
             patch("builtins.open", boom_open), \
             patch("os.replace", boom_replace):
            res = runner.invoke(cli, [
                "check-pick-entered", "--cron", "--picks-dir", str(pdir), "--now-et", f"{DAY}T09:45:00",
            ])
        assert res.exit_code == 0
        assert "entry receipt unavailable" in res.stderr
        assert not marker.exists()
        # No unsealed receipt may ever be discoverable!
        assert entry_receipt.discover(pdir, DAY) == []
        shutil.rmtree(idir)

    # 2. Malformed or non-regular files at early --cron runs (05:15 and 09:45)
    mdir = work_dir / "malformed"
    pdir = mdir / "picks"
    pdir.mkdir(parents=True)
    # Corrupt pick JSON -> load_pick_bytes returns (None, raw) -> no_pick_file, exit 0
    (pdir / f"{DAY}.json").write_text("{bad-json")
    with patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"), patch("os.fsync", lambda fd: None):
        r_bad_pick = runner.invoke(cli, ["check-pick-entered", "--cron", "--picks-dir", str(pdir), "--now-et", f"{DAY}T05:15:00"])
    assert r_bad_pick.exit_code == 0
    assert entry_receipt.discover(pdir, DAY)[-1]["outcome"] == "no_pick_file"

    # Pick path is a directory -> no_pick_file, exit 0
    (pdir / f"{DAY}.json").unlink()
    (pdir / f"{DAY}.json").mkdir()
    with patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"), patch("os.fsync", lambda fd: None):
        r_dir_pick = runner.invoke(cli, ["check-pick-entered", "--cron", "--picks-dir", str(pdir), "--now-et", f"{DAY}T05:15:00"])
    assert r_dir_pick.exit_code == 0
    shutil.rmtree(pdir / f"{DAY}.json")

    # Corrupt decision.json with delivered pick -> falls back to pick_was_delivered, early gate still works!
    _save_pick(pdir, f"{DAY}T10:50:00-04:00")
    (pdir / DAY / "decision.json").write_text("{corrupt-decision")
    with patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"), patch("os.fsync", lambda fd: None):
        r_bad_dec = runner.invoke(cli, ["check-pick-entered", "--cron", "--picks-dir", str(pdir), "--now-et", f"{DAY}T09:45:00"])
    assert r_bad_dec.exit_code == 0
    rec_bad_dec = entry_receipt.discover(pdir, DAY)[-1]
    assert rec_bad_dec["outcome"] == "base_slot_available" and rec_bad_dec["selection"]["commit"]["decision_valid"] is False
    shutil.rmtree(mdir)

    # 3. Midnight (00:00..00:45) and Year Boundary (2027-12-31 23:45 -> 2028-01-01 00:00..00:45)
    ndir = work_dir / "midnight_nye"
    pdir = ndir / "picks"
    pdir.mkdir(parents=True)
    # 2027-03-23 slate with an after-midnight 2027-03-24T00:35 pitch:
    # checked at 2027-03-23T23:45 (50m before pitch), NOT checked at 2027-03-24T00:00..00:45 (reads 2027-03-24)!
    _save_pick(pdir, f"{DAY}T00:35:00-04:00", day=PREV_DAY)
    udir = entry_receipt.units_snapshot_dir(pdir)
    udir.mkdir(parents=True)
    (udir / "2027-03-23T200000Z.json.gz").write_bytes(
        gzip.compress(json.dumps({"units": [{"id": 77, "roundId": 1, "feedId": 101}]}).encode(), mtime=0)
    )
    row_match = [{"roundId": 1, "unitId": 77, "playerId": 1, "number": 1, "result": None}]
    with ExitStack() as stack:
        stack.enter_context(patch("bts.leaderboard.auth.load_session_cookies", lambda: {}))
        stack.enter_context(patch("bts.leaderboard.auth.extract_uid", lambda _: "u1"))
        stack.enter_context(patch("bts.leaderboard.auth.fetch_login_session", lambda *a, **k: SimpleNamespace(user_id="u1", username="stonehengee", xsid="x1")))
        stack.enter_context(patch("bts.contest_fetch.fetch_profile", lambda *a, **k: {"predictions": []}))
        stack.enter_context(patch("bts.contest_fetch.fetch_pending_predictions", lambda *a, **k: row_match))
        stack.enter_context(patch("bts.cli._fetch_rounds", lambda: {1: date.fromisoformat(PREV_DAY)}))
        stack.enter_context(patch("bts.cli._fetch_bts_to_mlb", lambda: {1: 1010}))
        stack.enter_context(patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"))
        stack.enter_context(patch("os.fsync", lambda fd: None))
        r_2345 = runner.invoke(cli, ["check-pick-entered", "--cron", "--picks-dir", str(pdir), "--now-et", f"{PREV_DAY}T23:45:00"])
        assert r_2345.exit_code == 0 and entry_receipt.discover(pdir, PREV_DAY)[-1]["outcome"] == "observed"
        for hm in ("00:00", "00:15", "00:30", "00:45"):
            r_mid = runner.invoke(cli, ["check-pick-entered", "--cron", "--picks-dir", str(pdir), "--now-et", f"{DAY}T{hm}:00"])
            assert r_mid.exit_code == 0
        assert [r["outcome"] for r in entry_receipt.discover(pdir, DAY)] == ["no_pick_file"] * 4

    # NYE / Jan 1 boundary: 2027-12-31T23:45 (season 2027) -> 2028-01-01T00:00..00:45 (season 2028)
    with patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"), patch("os.fsync", lambda fd: None):
        runner.invoke(cli, ["check-pick-entered", "--cron", "--picks-dir", str(pdir), "--now-et", "2027-12-31T23:45:00"])
        for hm in ("00:00", "00:15", "00:30", "00:45"):
            runner.invoke(cli, ["check-pick-entered", "--cron", "--picks-dir", str(pdir), "--now-et", f"2028-01-01T{hm}:00"])
    rec_2027, = entry_receipt.discover(pdir, "2027-12-31")
    recs_2028 = entry_receipt.discover(pdir, "2028-01-01")
    assert rec_2027["season"] == 2027 and rec_2027["outcome"] == "no_pick_file"
    assert len(recs_2028) == 4 and all(r["season"] == 2028 and r["outcome"] == "no_pick_file" for r in recs_2028)
    shutil.rmtree(ndir)

    # 4. Failed / unconfirmed send & restart on an early-only 06:10 slate with --cron
    fdir = work_dir / "failed_unconfirmed"
    pdir = fdir / "picks"
    pdir.mkdir(parents=True)
    marker = fdir / "health_state" / "pick_entry_check.json"
    _save_pick(pdir, f"{DAY}T06:10:00-04:00", delivery="locked_unconfirmed", committed=True)
    sent_dms = []
    fail_first = [True]

    def fake_send(rec, msg):
        if fail_first[0]:
            fail_first[0] = False
            raise RuntimeError("transient DM outage at 05:15")
        sent_dms.append(msg)

    with ExitStack() as stack:
        stack.enter_context(patch("bts.leaderboard.auth.load_session_cookies", lambda: {}))
        stack.enter_context(patch("bts.leaderboard.auth.extract_uid", lambda _: "u1"))
        stack.enter_context(patch("bts.leaderboard.auth.fetch_login_session", lambda *a, **k: SimpleNamespace(user_id="u1", username="stonehengee", xsid="x1")))
        stack.enter_context(patch("bts.contest_fetch.fetch_profile", lambda *a, **k: {"predictions": []}))
        stack.enter_context(patch("bts.contest_fetch.fetch_pending_predictions", lambda *a, **k: []))
        stack.enter_context(patch("bts.cli._fetch_rounds", lambda: {1: date.fromisoformat(DAY)}))
        stack.enter_context(patch("bts.cli._fetch_bts_to_mlb", lambda: {1: 1010}))
        stack.enter_context(patch("bts.dm.send_dm", fake_send))
        stack.enter_context(patch("bts.picks._git_head_sha", lambda *a, **k: "fixed-head"))
        stack.enter_context(patch("os.fsync", lambda fd: None))
        stack.enter_context(patch("time.monotonic", lambda: 1000.0))

        # 05:15: DM fails -> marker status is dm_failed, retryable!
        r1 = runner.invoke(cli, [
            "check-pick-entered", "--cron", "--picks-dir", str(pdir),
            "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
            "--now-et", f"{DAY}T05:15:00",
        ])
        assert r1.exit_code == 1 and json.loads(marker.read_text())["status"] == "dm_failed"
        assert sent_dms == []

        # 05:30: retries and sends W7 cautious locked_unconfirmed message (never "Fix it now!")
        r2 = runner.invoke(cli, [
            "check-pick-entered", "--cron", "--picks-dir", str(pdir),
            "--expected-username", "stonehengee", "--dm-recipient", "stonehengee.bsky.social",
            "--now-et", f"{DAY}T05:30:00",
        ])
        assert r2.exit_code == 1 and json.loads(marker.read_text())["status"] == "alerted"
        assert len(sent_dms) == 1
        assert "Delivery of this pick was NOT confirmed" in sent_dms[0]
        assert "Fix it now!" not in sent_dms[0]

    shutil.rmtree(fdir)
    return {
        "interrupt_mid_publication": "PASS",
        "malformed_and_non_regular_files": "PASS",
        "midnight_and_year_boundary": "PASS",
        "failed_dm_and_locked_unconfirmed_restart": "PASS",
    }


def main() -> None:
    t0 = time.perf_counter()
    _forbid_network()
    work = OUT_DIR / ".work_probe_05"
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    try:
        texts = audit_texts_and_2025_feeds()
        standing = check_standing_rows(work)
        elapsed = round(time.perf_counter() - t0, 3)
        report = {
            "status": "PASS",
            "elapsed_sec": elapsed,
            "texts_and_2025_feeds": texts,
            "standing_rows": standing,
        }
        (OUT_DIR / "log_05_texts_and_standing_rows.json").write_text(json.dumps(report, indent=2) + "\n")
        print(
            f"PASS probe_05 in {elapsed}s: 2025 feeds={texts['feeds_2025_checked']} "
            f"({texts['feeds_2025_earliest_et']}..{texts['feeds_2025_latest_et']} ET), texts & standing rows green"
        )
    finally:
        if work.exists():
            shutil.rmtree(work)


if __name__ == "__main__":
    main()
