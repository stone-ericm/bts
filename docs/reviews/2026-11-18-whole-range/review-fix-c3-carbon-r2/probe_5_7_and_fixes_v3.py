"""Probe v3 for:
- Item 5: `bts reconcile` end-to-end verification (cutoff, wording, exit codes, atomic rebuild,
  corrupt streak preservation, bounded special files).
- Item 1 & Item 2 smallest-fix demonstrations + healthy-box directory-scan benchmark.
- Item 7: `dry_compare_619abda7.py` verification on synthetic `data/picks` histories.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
import signal
import socket
import stat
import subprocess
import sys
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(HERE))

import bts.cli as cli_mod
from bts.contest_state import DecisionStreakState
from bts.daily_decision import decision_path, parse_decision, write_decision
import bts.picks as picks_mod
import bts.reconcile_receipt as rr_mod
from click.testing import CliRunner


ET = ZoneInfo("America/New_York")


def _slot(day: str, game_pk: int = 101) -> picks_mod.Pick:
    return picks_mod.Pick(
        batter_name="Shohei Ohtani", batter_id=660271, team="LAD", lineup_position=1,
        pitcher_name="Gerrit Cole", pitcher_id=543037, p_game_hit=0.81, flags=[],
        projected_lineup=False, game_pk=game_pk, game_time=f"{day}T23:10:00Z", pitcher_team="NYY",
    )


def _plant(
    root: Path, day: str, result: str | None = None, *, delivered: bool = True,
    attempted: bool | None = None, preview: bool = False,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    m = DecisionStreakState(0, True, True, "model", "fresh", 0, True)
    daily = picks_mod.DailyPick(
        date=day, run_time=f"{day}T18:00:00+00:00", pick=_slot(day), double_down=None, runner_up=None,
        notification_sent=delivered, notification_id=f"dm-{day}" if delivered else None,
        delivery_attempted=delivered if attempted is None else attempted, result=result,
        slot_results={"pick": result} if result in ("hit", "miss", "void") else None,
        chosen_on=picks_mod.chosen_on_record(m, required=False) if preview else None,
    )
    return picks_mod.save_pick(daily, root)


def _run_reconcile(root: Path, now_et: datetime, *, feed: str = "miss"):
    real_rr = rr_mod.ReconcileReceipt
    clock = now_et if callable(now_et) else (lambda: now_et)
    with patch.object(picks_mod, "resolve_daily_slot_results", lambda *a, **k: {"pick": feed}), patch(
        "bts.reconcile_receipt.ReconcileReceipt",
        side_effect=lambda **kw: real_rr(clock=clock, lookback_days=kw["lookback_days"]),
    ):
        return CliRunner().invoke(cli_mod.cli, ["reconcile", "--picks-dir", str(root)])


def _test_item5_e2e(scratch: Path) -> dict:
    out: dict = {}

    for label, dt_et, old_r, new_r, expect_change in [
        ("pre_0200_hit_to_miss", datetime(2027, 4, 20, 2, 0, 0, tzinfo=ET), "hit", "miss", True),
        ("pre_075959_miss_to_hit", datetime(2027, 4, 20, 7, 59, 59, tzinfo=ET), "miss", "hit", True),
        ("at_080000_cutoff", datetime(2027, 4, 20, 8, 0, 0, tzinfo=ET), "hit", "miss", False),
    ]:
        d = scratch / label
        _plant(d, "2027-04-19", old_r, delivered=True)
        picks_mod.save_streak(1 if old_r == "hit" else 0, d, saver_available=True, season=2027)
        res = _run_reconcile(d, dt_et, feed=new_r)
        assert res.exit_code == 0, res.output
        p_after = picks_mod.load_pick("2027-04-19", d).result
        st_after = picks_mod.load_streak(d, season=2027)
        if expect_change:
            assert p_after == new_r
            assert f"{old_r} -> {new_r}" in res.stdout
            assert f"Streak recalculated: {1 if new_r == 'hit' else 0}" in res.stdout
        else:
            assert p_after == old_r
            assert f"No scoring changes detected. Streak: {st_after}" in res.stdout
        out[label] = {"exit": res.exit_code, "pick_result": p_after, "streak": st_after, "stdout": res.stdout.strip()}

    d_blk_real = scratch / "blocked_real_delivered_no_decision"
    _plant(d_blk_real, "2027-04-16", None, delivered=True, attempted=True)
    _plant(d_blk_real, "2027-04-19", "hit", delivered=True)
    picks_mod.save_streak(14, d_blk_real, saver_available=False, season=2027)
    sp_blk_real = d_blk_real / "streak.json"
    before_blk_real = sp_blk_real.read_bytes()
    res_blk_real = _run_reconcile(d_blk_real, datetime(2027, 4, 20, 2, 0, tzinfo=ET), feed="miss")
    assert res_blk_real.exit_code == 0
    assert "Season replay unavailable for 2027-04-16 (delivery attempt without a decision or result)" in res_blk_real.stderr
    assert "Streak retained: 14" in res_blk_real.stdout and "recalculated" not in res_blk_real.stdout
    assert sp_blk_real.read_bytes() == before_blk_real

    d_blk = scratch / "blocked_with_correction"
    _plant(d_blk, "2027-04-16", None, delivered=True, attempted=False)
    _plant(d_blk, "2027-04-19", "hit", delivered=True)
    picks_mod.save_streak(14, d_blk, saver_available=False, season=2027)
    sp_blk = d_blk / "streak.json"
    before_blk = sp_blk.read_bytes()
    res_blk = _run_reconcile(d_blk, datetime(2027, 4, 20, 2, 0, tzinfo=ET), feed="miss")
    assert res_blk.exit_code == 0
    assert "Season replay unavailable for 2027-04-16 (committed pick awaiting a result)" in res_blk.stderr
    assert "Streak retained: 14" in res_blk.stdout and "recalculated" not in res_blk.stdout
    assert sp_blk.read_bytes() == before_blk
    out["blocked_with_correction"] = {
        "real_delivered_no_decision_stderr": res_blk_real.stderr.strip(),
        "delivered_no_attempt_stderr": res_blk.stderr.strip(),
        "stdout": res_blk.stdout.strip(),
    }

    corrupt_payloads = {
        "trunc": "{bad", "array": "[]",
        "str_streak": '{"streak":"3","season":2027}', "bool_streak": '{"streak":true,"season":2027}',
    }
    out["corrupt_streak_blocked"] = {}
    for cname, payload in corrupt_payloads.items():
        dc = scratch / f"corrupt_blk_{cname}"
        _plant(dc, "2027-04-16", None, delivered=True)
        spc = dc / "streak.json"
        spc.write_text(payload)
        rc = _run_reconcile(dc, datetime(2027, 4, 20, 2, 0, tzinfo=ET))
        assert rc.exit_code == 1
        assert "Streak unavailable" in rc.output and "Traceback" not in rc.output
        assert spc.read_text() == payload
        out["corrupt_streak_blocked"][cname] = {"exit": rc.exit_code, "preserved": True}

    da = scratch / "atomic_rebuild"
    _plant(da, "2027-04-01", "hit", delivered=True)
    _plant(da, "2027-04-02", None, delivered=False, preview=True)
    spa = da / "streak.json"
    spa.write_text("{corrupt")
    old_inode = spa.stat().st_ino
    with spa.open("rb") as open_fh:
        ra = _run_reconcile(da, datetime(2027, 4, 20, 2, 0, tzinfo=ET), feed="hit")
        assert ra.exit_code == 0
        old_bytes_via_fd = open_fh.read()
    new_inode = spa.stat().st_ino
    rebuilt = json.loads(spa.read_text())
    assert old_bytes_via_fd == b"{corrupt" and new_inode != old_inode
    assert (rebuilt["streak"], rebuilt["saver_available"], rebuilt["season"]) == (1, True, 2027)
    out["atomic_rebuild"] = {
        "exit": ra.exit_code,
        "inode_replaced": new_inode != old_inode,
        "rebuilt": [rebuilt["streak"], rebuilt["saver_available"], rebuilt["season"]],
    }

    class _SpecTimeout(BaseException):
        pass

    signal.signal(signal.SIGALRM, lambda *_a: (_ for _ in ()).throw(_SpecTimeout("special file hung")))
    special_cases = {}
    for skind in ("old_pick_fifo", "recent_pick_fifo", "decision_fifo", "streak_fifo", "receipt_root_fifo"):
        ds = scratch / f"spec_{skind}" / "picks"
        ds.mkdir(parents=True)
        _plant(ds, "2027-04-01", "hit", delivered=True)
        picks_mod.save_streak(1, ds, saver_available=True, season=2027)
        if skind == "old_pick_fifo":
            os.mkfifo(ds / "2027-04-10.json")
        elif skind == "recent_pick_fifo":
            os.mkfifo(ds / "2027-04-19.json")
        elif skind == "decision_fifo":
            (ds / "2027-04-10").mkdir()
            os.mkfifo(ds / "2027-04-10" / "decision.json")
        elif skind == "streak_fifo":
            (ds / "streak.json").unlink()
            os.mkfifo(ds / "streak.json")
        elif skind == "receipt_root_fifo":
            rpath = rr_mod.receipts_root(ds)
            rpath.parent.mkdir(parents=True, exist_ok=True)
            os.mkfifo(rpath)
        t0 = time.perf_counter()
        signal.alarm(2)
        try:
            rs = _run_reconcile(ds, datetime(2027, 4, 20, 2, 0, tzinfo=ET), feed="hit")
        finally:
            signal.alarm(0)
        elapsed_ms = round((time.perf_counter() - t0) * 1000, 2)
        special_cases[skind] = {"exit": rs.exit_code, "elapsed_ms": elapsed_ms, "stderr": rs.stderr.strip()[:120]}
    out["special_files"] = special_cases
    return out


def _fixed_past_pick_blocker_picks_only(
    picks_dir: Path, day: str, daily: picks_mod.DailyPick, decision: dict | None, raw_data: dict,
) -> str | None:
    if decision is not None and decision["scoreable"]:
        return "committed pick awaiting a result"
    if any(value is not None and not isinstance(value, str) for value in (
        daily.bluesky_uri, daily.delivered_at,
    )):
        return "unreadable or invalid pick file"
    delivery_signal = any((
        daily.bluesky_posted, daily.bluesky_uri, daily.notification_sent,
        daily.notification_id, daily.delivery_attempted, daily.delivered_at,
        raw_data.get("automatic_reading") is not None,
    ))
    if delivery_signal:
        if decision is not None:
            return "skip record contradicts a delivery"
        if picks_mod.pick_was_delivered(daily) or daily.bluesky_uri or daily.notification_sent or daily.notification_id or daily.delivered_at:
            return "committed pick awaiting a result"
        return "delivery attempt without a decision or result"

    try:
        day_dir = picks_dir / day
        try:
            dir_mode = day_dir.lstat().st_mode
        except FileNotFoundError:
            pass
        else:
            if not stat.S_ISDIR(dir_mode):
                raise picks_mod.NonRegularReplayFile(f"{day_dir} is not a directory")
            with os.scandir(day_dir) as it:
                for entry in it:
                    if entry.name.startswith("scheduler_state.json.corrupt-"):
                        raise ValueError("quarantined scheduler state")
        text = picks_mod._read_reconcile_file(day_dir / "scheduler_state.json")
        if text is not None:
            state = json.loads(text)
            if (not isinstance(state, dict) or state.get("date") != day
                    or type(state.get("pick_locked")) is not bool):
                raise ValueError("invalid scheduler state")
            if state["pick_locked"]:
                return "scheduler state marks the pick locked"
    except picks_mod._REPLAY_READ_ERRORS:
        return "unreadable or invalid scheduler state"

    if decision is None:
        if raw_data.get("preview_only") is not True or daily.chosen_on is None or daily.chosen_on["required"]:
            return "no decision or preview stamp"
        return None
    if daily.chosen_on is not None and daily.chosen_on["required"]:
        return "live-selected pick beside a skip record"
    return None


def _fixed_fold_season(picks_dir: Path, season: int, today: str, markers: list[str]):
    prefix = f"{season}-"
    dates: set[str] = set()
    try:
        with os.scandir(picks_dir) as it:
            for entry in it:
                name = entry.name
                if not name.startswith(prefix):
                    continue
                stem = name[:-5] if name.endswith(".json") else name
                if picks_mod._ISO_DATE_RE.fullmatch(stem):
                    dates.add(stem)
    except OSError as exc:
        raise picks_mod.NonRegularReplayFile(f"cannot scan {picks_dir}: {exc}") from exc

    streak, saver = 0, True
    for day in sorted(dates):
        if day > today:
            continue
        try:
            data = picks_mod._read_reconcile_pick_data(picks_dir, day)
            daily = None
            if data is not None:
                result = data.get("result")
                if result == "hit":
                    daily = picks_mod._daily_pick_from_data(data)
                    streak, saver = picks_mod._apply_streak_day(
                        streak, saver, True, picks_mod.streak_increment_for_resolved_hit(daily)
                    )
                    continue
                if result == "miss":
                    streak, saver = picks_mod._apply_streak_day(streak, saver, False, 0)
                    continue
                if result == "void":
                    continue
                daily = picks_mod._validate_reconcile_pick(data, day)
        except picks_mod._REPLAY_READ_ERRORS:
            return None, (day, "unreadable or invalid pick file")
        try:
            day_dir = picks_dir / day
            try:
                dir_mode = day_dir.lstat().st_mode
            except FileNotFoundError:
                text = None
            else:
                if not stat.S_ISDIR(dir_mode):
                    raise picks_mod.NonRegularReplayFile(f"{day_dir} is not a directory")
                with os.scandir(day_dir) as _it:
                    for _ in _it:
                        break
                text = picks_mod._read_reconcile_file(decision_path(day, picks_dir))
            decision = None if text is None else parse_decision(text)
            if text is not None and (decision is None or decision["date"] != day):
                raise ValueError("invalid decision record")
        except picks_mod._REPLAY_READ_ERRORS:
            return None, (day, "unreadable or invalid decision record")

        if daily is None:
            if decision is not None and decision["scoreable"]:
                return None, (day, "committed pick file missing")
        elif day in markers:
            return None, (day, "applied-date marker without a result")
        elif day < today:
            assert data is not None
            reason = _fixed_past_pick_blocker_picks_only(picks_dir, day, daily, decision, data)
            if reason is not None:
                return None, (day, reason)
    return (streak, saver), None


def _test_fixes_and_benchmark(scratch: Path) -> dict:
    import probe_1_and_2_v5 as p12
    import bts.contest_state as contest_state
    import bts.daily_decision as daily_decision
    import bts.dm as dm_mod
    import bts.entry_intent as entry_intent
    from bts.model import predict
    import bts.orchestrator as orchestrator
    import bts.posting as posting
    import bts.scheduler as scheduler
    import bts.strategy as strategy

    mods = {
        "cli": cli_mod, "contest_state": contest_state, "daily_decision": daily_decision,
        "dm": dm_mod, "entry_intent": entry_intent, "orchestrator": orchestrator,
        "picks": picks_mod, "posting": posting, "predict": predict, "rr": rr_mod,
        "scheduler": scheduler, "strategy": strategy,
    }

    orig_cwd = Path.cwd()
    wdir = scratch / "fix_verify"
    (wdir / "data").mkdir(parents=True)
    (wdir / "data" / "models").symlink_to(REPO / "data" / "models")

    fix_scenarios = {}
    fix_carbon = {}
    try:
        os.chdir(wdir)
        with patch.object(picks_mod, "_fold_season", _fixed_fold_season):
            for sc in (
                "N1_lost_ack_then_fallback_skip_auto",
                "N2_lost_ack_then_fallback_skip_dated",
                "N3_manual_run_crash_with_daemon_eod_skip",
                "N4_live_private_preview_lock_restart",
                "N5_private_research_fresh_pick_restart",
                "N6_live_private_preview_lock_quarantined",
                "N7_orchestrate_not_live_post_crash",
                "N8_preview_overwrites_unmarked_entered_pick",
            ):
                res = p12._build_scenario(sc, wdir / sc / "picks", mods)
                assert res["after_overturn_streak_saver"] == [10, True], f"{sc} failed under fix: {res}"
                assert res["after_true_10_false_streak_saver"] == [10, False], f"{sc} true10F failed under fix: {res}"
                fix_scenarios[sc] = {
                    "after_overturn": res["after_overturn_streak_saver"],
                    "after_true_10_false": res["after_true_10_false_streak_saver"],
                    "stderr": res["stderr"],
                }

            fix_carbon = p12._run_carbon_dir_states(mods, wdir / "carbon")
            assert fix_carbon["a_chmod000_decision_dir_no_pick"]["streak_after"] == [9, False]
            assert fix_carbon["b1_symlink_skip_dir_over_delivered_pick"]["streak_after"] == [9, False]
            assert fix_carbon["b2_symlink_skip_dir_over_unmarked_live_pick"]["streak_after"] == [9, False]
            assert fix_carbon["b3_fifo_date_dir_no_pick"]["streak_after"] == [9, False]
            assert fix_carbon["b3b_fifo_date_dir_with_pick"]["streak_after"] == [9, False]
            assert fix_carbon["c_chmod0300_picks_dir_lib"]["raised"] == "NonRegularReplayFile"
            assert fix_carbon["c_chmod0300_picks_dir_lib"]["streak_after"] == [10, False]

            hp = wdir / "healthy_preview" / "picks"
            _plant(hp, "2027-04-01", "hit", delivered=True)
            pp = _plant(hp, "2027-04-16", None, delivered=False, preview=True)
            pdata = json.loads(pp.read_text())
            pdata["preview_only"] = True
            pp.write_text(json.dumps(pdata))
            picks_mod.save_streak(9, hp, saver_available=False, season=2027)
            rhp = _run_reconcile(hp, datetime(2027, 4, 20, 2, 0, tzinfo=ET), feed="hit")
            assert rhp.exit_code == 0 and not rhp.stderr.strip() and picks_mod.load_streak(hp, season=2027) == 1

            hs = wdir / "healthy_skip" / "picks"
            _plant(hs, "2027-04-01", "hit", delivered=True)
            _plant(hs, "2027-04-16", None, delivered=False, preview=True)
            write_decision("2027-04-16", hs, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False)
            picks_mod.save_streak(9, hs, saver_available=False, season=2027)
            rhs = _run_reconcile(hs, datetime(2027, 4, 20, 2, 0, tzinfo=ET), feed="hit")
            assert rhs.exit_code == 0 and not rhs.stderr.strip() and picks_mod.load_streak(hs, season=2027) == 1
    finally:
        os.chdir(orig_cwd)

    bench_dir = scratch / "bench_180"
    bench_dir.mkdir()
    start_d = date(2026, 3, 26)
    for i in range(180):
        di = (start_d + timedelta(days=i)).isoformat()
        _plant(bench_dir, di, "hit" if i % 3 else "miss", delivered=True)
        write_decision(di, bench_dir, action="single", source="mdp", delivery_status="delivered", scoreable=True)
        (bench_dir / f"{di}.shadow.json").write_text("{}")

    iters = 100
    t0 = time.perf_counter()
    for _ in range(iters):
        r_cur = picks_mod._fold_season(bench_dir, 2026, "2026-12-31", [])
    cur_ms = (time.perf_counter() - t0) * 1000 / iters

    t1 = time.perf_counter()
    for _ in range(iters):
        r_fix = _fixed_fold_season(bench_dir, 2026, "2026-12-31", [])
    fix_ms = (time.perf_counter() - t1) * 1000 / iters
    assert r_cur == r_fix

    return {
        "fix_scenarios": fix_scenarios,
        "fix_carbon_dir_states": fix_carbon,
        "healthy_controls_ok": True,
        "healthy_180_day_benchmark_ms": {
            "current_glob_fold_ms": round(cur_ms, 3),
            "scandir_lstat_fold_ms": round(fix_ms, 3),
            "delta_ms": round(fix_ms - cur_ms, 3),
        },
    }


def _test_item7_dry_compare(scratch: Path) -> dict:
    script = str(HERE / "dry_compare_619abda7.py")
    out = {}

    d_ok = scratch / "dry_ok"
    for i in range(12):
        _plant(d_ok, (date(2026, 5, 1) + timedelta(days=i)).isoformat(), "miss" if i == 10 else "hit", delivered=True)
    picks_mod.save_streak(11, d_ok, saver_available=False, season=2026)
    p_ok = subprocess.run(
        [sys.executable, script, str(d_ok), "--today", "2026-12-31", "--repo", str(REPO)],
        capture_output=True, text=True, timeout=25,
    )
    j_ok = json.loads(p_ok.stdout)
    assert p_ok.returncode == 0 and j_ok["first_diff_or_block"] is None and j_ok["619abda7"]["value"] == [11, False]
    out["identical_terminal"] = {"exit": p_ok.returncode, "result": j_ok["aa61bb43"]["value"]}

    d_div = scratch / "dry_div"
    _plant(d_div, "2026-05-01", "hit", delivered=True)
    _plant(d_div, "2026-05-05", None, delivered=False, preview=True)
    _plant(d_div, "2026-05-06", "hit", delivered=True)
    p_div = subprocess.run(
        [sys.executable, script, str(d_div), "--today", "2026-12-31", "--repo", str(REPO)],
        capture_output=True, text=True, timeout=25,
    )
    j_div = json.loads(p_div.stdout)
    assert p_div.returncode == 2 and j_div["first_diff_or_block"]["date"] == "2026-05-05"
    out["preview_divergence"] = {"exit": p_div.returncode, "first_diff_or_block": j_div["first_diff_or_block"]}

    d_blk = scratch / "dry_blk"
    _plant(d_blk, "2026-05-01", "hit", delivered=True)
    _plant(d_blk, "2026-05-08", None, delivered=True)
    p_blk = subprocess.run(
        [sys.executable, script, str(d_blk), "--today", "2026-12-31", "--repo", str(REPO)],
        capture_output=True, text=True, timeout=25,
    )
    j_blk = json.loads(p_blk.stdout)
    assert p_blk.returncode == 3 and j_blk["first_diff_or_block"]["date"] == "2026-05-08"
    out["both_blocked"] = {"exit": p_blk.returncode, "first_diff_or_block": j_blk["first_diff_or_block"]}

    d_l02 = scratch / "dry_l02"
    _plant(d_l02, "2026-05-01", "hit", delivered=True)
    _plant(d_l02, "2026-05-02", "hit", delivered=True)
    p_l02 = subprocess.run(
        [sys.executable, script, str(d_l02), "--today", "2026-05-02", "--repo", str(REPO)],
        capture_output=True, text=True, timeout=25,
    )
    j_l02 = json.loads(p_l02.stdout)
    assert p_l02.returncode == 2 and j_l02["first_diff_or_block"]["date"] == "2026-05-02"
    out["l02_same_day_horizon"] = {"exit": p_l02.returncode, "first_diff_or_block": j_l02["first_diff_or_block"]}

    return out


def main() -> int:
    def _deny_net(*args, **kwargs):
        raise AssertionError("Network access forbidden")

    socket.socket.connect = _deny_net  # type: ignore[method-assign]
    socket.getaddrinfo = _deny_net  # type: ignore[assignment]

    with TemporaryDirectory(dir=HERE, prefix="_p57_v3_") as td:
        scratch = Path(td)
        res5 = _test_item5_e2e(scratch / "item5")
        res_fix = _test_fixes_and_benchmark(scratch / "fixes")
        res7 = _test_item7_dry_compare(scratch / "item7")

    payload = {"item5": res5, "fixes_and_benchmark": res_fix, "item7": res7}
    (HERE / "item5_7_and_fixes_results.json").write_text(json.dumps(payload, indent=2) + "\n")
    print("=== ITEM 5 E2E: OK ===")
    print("Blocked wording:", json.dumps(res5["blocked_with_correction"], indent=2))
    print("Special files:", json.dumps(res5["special_files"], indent=2))
    print("=== FIXES & BENCHMARK: OK ===")
    print("Fix scenarios:", json.dumps(res_fix["fix_scenarios"], indent=2))
    print("Benchmark (180-day season):", json.dumps(res_fix["healthy_180_day_benchmark_ms"], indent=2))
    print("=== ITEM 7 DRY COMPARE: OK ===")
    print(json.dumps(res7, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
