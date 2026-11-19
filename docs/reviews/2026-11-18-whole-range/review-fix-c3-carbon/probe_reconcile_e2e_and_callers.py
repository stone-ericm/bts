"""Items 3, 4, 5 & 8 probe: skip/block matrix, CLI e2e, special files (subprocess-bounded),
concurrent reconciles, interrupt+retry, adversarial crash sequences, and other callers parity.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo


REVIEW_DIR = Path(__file__).resolve().parent
REPO_ROOT = REVIEW_DIR.parents[1]
ET = ZoneInfo("America/New_York")
RUN_ET = datetime(2027, 4, 20, 2, 0, tzinfo=ET)


def _install_no_network() -> None:
    def _deny(*args, **kwargs):
        raise AssertionError("Network access is forbidden")

    socket.socket = _deny  # type: ignore[assignment]
    socket.create_connection = _deny  # type: ignore[assignment]


def _make_pick(picks_mod, root: Path, day: str, result=None, *, delivered_dm=False, posted=False, attempted=False, dd=False, partial_void=False):
    def _slot(gpk, bid, bname):
        return picks_mod.Pick(
            batter_name=bname,
            batter_id=bid,
            team="LAD",
            lineup_position=1,
            pitcher_name="Gerrit Cole",
            pitcher_id=543037,
            p_game_hit=0.81,
            flags=[],
            projected_lineup=False,
            game_pk=gpk,
            game_time=f"{day}T23:10:00Z",
            pitcher_team="NYY",
        )

    if partial_void and result in ("hit", "miss", "void"):
        sr = {"pick": "void", "double_down": result}
    elif result in ("hit", "miss", "void"):
        sr = {"pick": result, **({"double_down": result} if dd else {})}
    else:
        sr = None

    daily = picks_mod.DailyPick(
        date=day,
        run_time=f"{day}T18:00:00+00:00",
        pick=_slot(745100, 660271, "Shohei Ohtani"),
        double_down=_slot(745101, 605141, "Mookie Betts") if (dd or partial_void) else None,
        runner_up={"batter_name": "Freddie Freeman", "p_game_hit": 0.77},
        bluesky_posted=posted,
        bluesky_uri="at://post/1" if posted else None,
        notification_sent=delivered_dm,
        notification_channel="bluesky_dm" if delivered_dm else None,
        notification_id=f"dm-{day}" if delivered_dm else None,
        delivery_attempted=(attempted or delivered_dm or posted),
        result=result,
        slot_results=sr,
    )
    return picks_mod.save_pick(daily, root)


def _invoke_reconcile(cli_mod, picks_mod, rr_mod, root: Path, *, clock_fn=lambda: RUN_ET, feed_map=None, lookback=8):
    from click.testing import CliRunner

    feed_map = feed_map or {}

    def _resolve(daily, day, **kwargs):
        if day not in feed_map:
            raise AssertionError(f"Unexpected resolve_daily_slot_results for {day}")
        return feed_map[day]

    with patch.object(picks_mod, "resolve_daily_slot_results", _resolve), patch(
        "bts.reconcile_receipt.ReconcileReceipt",
        side_effect=lambda **kw: rr_mod.ReconcileReceipt(clock=clock_fn, lookback_days=kw["lookback_days"]),
    ):
        return CliRunner().invoke(cli_mod.cli, ["reconcile", "--picks-dir", str(root), "--lookback", str(lookback)])


# ---------------------------------------------------------------------------
# Subprocess helper modes for bounded special-file & concurrency probes
# ---------------------------------------------------------------------------

def _subproc_special_probe(target_root: str, clock_iso: str, feed_hit: str) -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import bts.cli as cli_mod
    import bts.picks as picks_mod
    import bts.reconcile_receipt as rr_mod
    from click.testing import CliRunner

    _install_no_network()
    fixed = datetime.fromisoformat(clock_iso)

    def _resolve(daily, day, **kwargs):
        return {"pick": feed_hit} if feed_hit else None

    with patch.object(picks_mod, "resolve_daily_slot_results", _resolve), patch(
        "bts.reconcile_receipt.ReconcileReceipt",
        side_effect=lambda **kw: rr_mod.ReconcileReceipt(clock=lambda: fixed, lookback_days=kw["lookback_days"]),
    ):
        res = CliRunner().invoke(cli_mod.cli, ["reconcile", "--picks-dir", target_root])
    out = {
        "exit_code": res.exit_code,
        "stdout": res.stdout,
        "stderr": res.stderr,
        "exc_type": type(res.exception).__name__ if res.exception else None,
    }
    sys.stdout.write(json.dumps(out) + "\n")
    sys.stdout.flush()
    return 0


def _subproc_concurrent_worker(target_root: str, hold_ms: int, new_res: str) -> int:
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import bts.cli as cli_mod
    import bts.picks as picks_mod
    import bts.reconcile_receipt as rr_mod
    from click.testing import CliRunner

    _install_no_network()
    real_lock = picks_mod.scoring_lock

    @contextmanager
    def _slow_lock(pdir):
        with real_lock(pdir):
            if hold_ms > 0:
                time.sleep(hold_ms / 1000.0)
            yield

    with patch.object(picks_mod, "scoring_lock", _slow_lock), patch.object(
        picks_mod, "resolve_daily_slot_results", lambda *a, **k: {"pick": new_res}
    ), patch(
        "bts.reconcile_receipt.ReconcileReceipt",
        side_effect=lambda **kw: rr_mod.ReconcileReceipt(
            clock=lambda: datetime(2027, 4, 20, 2, 0, tzinfo=ET) + timedelta(milliseconds=hold_ms),
            lookback_days=kw["lookback_days"],
        ),
    ):
        res = CliRunner().invoke(cli_mod.cli, ["reconcile", "--picks-dir", target_root])
    sys.stdout.write(json.dumps({"exit_code": res.exit_code, "stdout": res.stdout, "stderr": res.stderr}) + "\n")
    sys.stdout.flush()
    return 0


def _subproc_callers_snapshot(src_dir: str, work_dir: str) -> int:
    sys.path.insert(0, src_dir)
    import bts.cli as cli_mod
    from bts.contest_state import DecisionStreakState
    from bts.daily_decision import write_decision
    import bts.health.contest_state as h_cs
    import bts.health.streak_validation as h_sv
    import bts.picks as picks_mod
    import bts.scheduler as sched_mod
    import bts.web as web_mod
    from click.testing import CliRunner

    _install_no_network()
    w = Path(work_dir)
    results: dict = {}

    # 1. update_streak & save_streak exact file bytes under pinned UTC clock
    u_dir = w / "update_streak_bytes"
    u_dir.mkdir()
    fixed_utc = datetime(2027, 5, 10, 6, 0, 0, tzinfo=timezone.utc)

    class _PinnedDT(datetime):
        @classmethod
        def now(cls, tz=None):
            return fixed_utc if tz is not None else fixed_utc.replace(tzinfo=None)

    snapshots = []
    with patch.object(picks_mod, "datetime", _PinnedDT):
        for i in range(1, 11):
            d = f"2027-05-{i:02d}"
            val = picks_mod.update_streak([True], u_dir, date=d)
            snapshots.append((f"hit_{i}", val, (u_dir / "streak.json").read_text()))
        # Duplicate date idempotency
        val_dup = picks_mod.update_streak([True], u_dir, date="2027-05-10")
        snapshots.append(("dup_hit_10", val_dup, (u_dir / "streak.json").read_text()))
        # Miss at streak 10 consumes saver
        val_saver = picks_mod.update_streak([False], u_dir, date="2027-05-11")
        snapshots.append(("saver_miss_11", val_saver, (u_dir / "streak.json").read_text()))
        # Dateless update preserves season=2027 and applied_dates
        val_dateless = picks_mod.update_streak([True, True], u_dir)
        snapshots.append(("dateless_dd", val_dateless, (u_dir / "streak.json").read_text()))
        # Second miss at streak 12 resets to 0
        val_reset = picks_mod.update_streak([False], u_dir, date="2027-05-12")
        snapshots.append(("reset_miss_12", val_reset, (u_dir / "streak.json").read_text()))
    results["update_streak_snapshots"] = snapshots

    # 2. parse_daily_pick & load_pick_bytes
    p_dir = w / "parse_pick"
    p_dir.mkdir()
    _make_pick(picks_mod, p_dir, "2027-05-01", "hit", delivered_dm=True, partial_void=True)
    loaded, raw_bytes = picks_mod.load_pick_bytes("2027-05-01", p_dir)
    results["parse_daily_pick"] = {
        "date": loaded.date,
        "result": loaded.result,
        "slot_results": loaded.slot_results,
        "inc": picks_mod.streak_increment_for_resolved_hit(loaded),
        "raw_len": len(raw_bytes),
    }

    # 3. check-results CLI (committed hit, refused preview not-scoring, skip not-scoring)
    cr_dir = w / "check_results"
    cr_dir.mkdir()
    picks_mod.save_streak(2, cr_dir, saver_available=True, season=2027)
    _make_pick(picks_mod, cr_dir, "2027-04-19", None, delivered_dm=True)
    p_prev = _make_pick(picks_mod, cr_dir, "2027-04-18", None)
    d_prev = picks_mod.load_pick("2027-04-18", cr_dir)
    d_prev.chosen_on = picks_mod.chosen_on_record(
        DecisionStreakState(0, True, True, "model", "fresh", 0, True), required=False
    )
    picks_mod.save_pick(d_prev, cr_dir)
    _make_pick(picks_mod, cr_dir, "2027-04-17", None)
    write_decision("2027-04-17", cr_dir, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN_ET)

    with patch.object(cli_mod, "_now_et", lambda: datetime(2027, 4, 20, 1, 0, tzinfo=ET)), patch.object(
        picks_mod, "resolve_daily_slot_results", lambda daily, d, **kw: {"pick": "hit"}
    ):
        r_hit = CliRunner().invoke(cli_mod.cli, ["check-results", "--date", "2027-04-19", "--picks-dir", str(cr_dir)])
        r_prev = CliRunner().invoke(cli_mod.cli, ["check-results", "--date", "2027-04-18", "--picks-dir", str(cr_dir)])
        r_skip = CliRunner().invoke(cli_mod.cli, ["check-results", "--date", "2027-04-17", "--picks-dir", str(cr_dir)])

    results["check_results"] = {
        "hit": [r_hit.exit_code, r_hit.output, picks_mod.load_pick("2027-04-19", cr_dir).result],
        "preview": [r_prev.exit_code, r_prev.output, picks_mod.load_pick("2027-04-18", cr_dir).result],
        "skip": [r_skip.exit_code, r_skip.output, picks_mod.load_pick("2027-04-17", cr_dir).result],
        "streak": [picks_mod.load_streak(cr_dir, season=2027), picks_mod.load_saver_available(cr_dir, season=2027)],
    }

    # 4. scheduler.run_result_polling (final hit + saver miss)
    rp_dir = w / "result_polling"
    rp_dir.mkdir()
    picks_mod.save_streak(10, rp_dir, saver_available=True, season=2027)
    _make_pick(picks_mod, rp_dir, "2027-04-19", None, delivered_dm=True)
    with patch.object(sched_mod, "_now_et", lambda: datetime(2027, 4, 19, 23, 0, tzinfo=ET)), patch.object(
        sched_mod, "poll_game_result", lambda pk: "final"
    ), patch.object(picks_mod, "get_game_statuses_detailed", lambda d: {745100: {"abstract": "F", "detailed": "Final"}}), patch.object(
        picks_mod, "resolve_daily_slot_results", lambda daily, d, **kw: {"pick": "miss"}
    ):
        poll_status = sched_mod.run_result_polling(745100, "2027-04-19", rp_dir)
    results["result_polling"] = {
        "status": poll_status,
        "pick_result": picks_mod.load_pick("2027-04-19", rp_dir).result,
        "streak": picks_mod.load_streak(rp_dir, season=2027),
        "saver": picks_mod.load_saver_available(rp_dir, season=2027),
    }

    # 5. Dashboard & Health checks
    with patch.object(web_mod, "PICKS_DIR", cr_dir), patch.object(web_mod, "_now_et", lambda: RUN_ET, create=True):
        dash_streak = web_mod.load_streak()
        dash_picks = len(web_mod.load_all_picks())
    sv_alerts = [(a.level, a.source, a.message) for a in h_sv.check(cr_dir)]
    cs_alerts = [(a.level, a.source) for a in h_cs.check(cr_dir, expected=False, now=RUN_ET.astimezone(timezone.utc))]
    results["dashboard_and_health"] = {
        "dash_streak": dash_streak,
        "dash_picks": dash_picks,
        "sv_alerts": sv_alerts,
        "cs_alerts": cs_alerts,
    }

    sys.stdout.write(json.dumps(results) + "\n")
    sys.stdout.flush()
    return 0


def main() -> int:
    if len(sys.argv) >= 2 and sys.argv[1] == "--special":
        return _subproc_special_probe(sys.argv[2], sys.argv[3], sys.argv[4])
    if len(sys.argv) >= 2 and sys.argv[1] == "--concurrent":
        return _subproc_concurrent_worker(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    if len(sys.argv) >= 2 and sys.argv[1] == "--callers":
        return _subproc_callers_snapshot(sys.argv[2], sys.argv[3])

    sys.path.insert(0, str(REPO_ROOT / "src"))
    import bts.cli as cli_mod
    from bts.contest_state import DecisionStreakState
    from bts.daily_decision import decision_path, write_decision
    import bts.picks as picks_mod
    import bts.reconcile_receipt as rr_mod
    import bts.scheduler as sched_mod

    _install_no_network()
    report: dict = {"item3": {}, "item4": {}, "item5": {}, "adversarial": {}}

    with TemporaryDirectory(dir=REVIEW_DIR, prefix="_e2e_run_") as scratch_str:
        scratch = Path(scratch_str)

        # ===================================================================
        # ITEM 3: Every skip and block case + exact CLI stderr/stdout lines
        # ===================================================================
        def _fresh(name: str) -> Path:
            p = scratch / f"i3_{name}"
            p.mkdir()
            _make_pick(picks_mod, p, "2027-04-01", "hit", delivered_dm=True)
            picks_mod.save_streak(9, p, saver_available=False, season=2027)
            return p

        # 3.1 Refused preview
        p = _fresh("refused_preview")
        _make_pick(picks_mod, p, "2027-04-16", None)
        dp = picks_mod.load_pick("2027-04-16", p)
        dp.chosen_on = picks_mod.chosen_on_record(
            DecisionStreakState(0, True, True, "model", "fresh", 0, True), required=False
        )
        picks_mod.save_pick(dp, p)
        r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
        assert r.exit_code == 0 and r.stderr == "" and "No scoring changes detected. Streak: 1" in r.stdout
        assert json.loads((p / "streak.json").read_text())["streak"] == 1
        report["item3"]["refused_preview"] = {"action": "skipped", "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}

        # 3.2 Private-mode committed with result vs without result
        p = _fresh("private_with_result")
        _make_pick(picks_mod, p, "2027-04-16", "hit")
        write_decision("2027-04-16", p, action="single", source="mdp", delivery_status="private_locked", scoreable=True, now=RUN_ET)
        r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
        assert r.exit_code == 0 and r.stderr == "" and "Streak: 2" in r.stdout
        report["item3"]["private_with_result"] = {"action": "folded", "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}

        p = _fresh("private_without_result")
        _make_pick(picks_mod, p, "2027-04-16", None)
        write_decision("2027-04-16", p, action="single", source="mdp", delivery_status="private_locked", scoreable=True, now=RUN_ET)
        r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
        assert r.exit_code == 0 and r.stderr.strip() == "Season replay unavailable for 2027-04-16 (committed pick awaiting a result)"
        assert "Streak retained: 9" in r.stdout
        report["item3"]["private_without_result"] = {"action": "blocked", "stdout": r.stdout.strip(), "stderr": r.stderr.strip()}

        # 3.3 locked_unconfirmed without result
        p = _fresh("locked_unconfirmed_pending")
        _make_pick(picks_mod, p, "2027-04-16", None, attempted=True)
        write_decision("2027-04-16", p, action="single", source="mdp", delivery_status="locked_unconfirmed", scoreable=True, now=RUN_ET)
        r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
        assert r.exit_code == 0 and r.stderr.strip() == "Season replay unavailable for 2027-04-16 (committed pick awaiting a result)"
        report["item3"]["locked_unconfirmed_pending"] = {"action": "blocked", "stderr": r.stderr.strip()}

        # 3.4 Bare delivery_attempted without decision
        p = _fresh("bare_attempt")
        _make_pick(picks_mod, p, "2027-04-16", None, attempted=True)
        r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
        assert r.exit_code == 0 and r.stderr.strip() == "Season replay unavailable for 2027-04-16 (delivery attempt without a decision or result)"
        report["item3"]["bare_attempt"] = {"action": "blocked", "stderr": r.stderr.strip()}

        # 3.5 Delivered pick awaiting result (None, suspended, unresolved)
        for sub_res in (None, "suspended", "unresolved"):
            p = _fresh(f"delivered_{sub_res}")
            _make_pick(picks_mod, p, "2027-04-16", sub_res, delivered_dm=True)
            r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
            assert r.exit_code == 0 and r.stderr.strip() == "Season replay unavailable for 2027-04-16 (committed pick awaiting a result)"
            report["item3"][f"delivered_{sub_res}"] = {"action": "blocked", "stderr": r.stderr.strip()}

        # 3.6 Skip day's leftover pick (and skip decision without pick)
        p = _fresh("skip_leftover")
        _make_pick(picks_mod, p, "2027-04-16", None)
        write_decision("2027-04-16", p, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN_ET)
        r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
        assert r.exit_code == 0 and r.stderr == "" and "Streak: 1" in r.stdout
        report["item3"]["skip_leftover"] = {"action": "skipped", "stdout": r.stdout.strip()}

        # 3.7 Scoreable decision with pick file missing
        p = _fresh("missing_pick_file")
        write_decision("2027-04-16", p, action="single", source="mdp", delivery_status="delivered", scoreable=True, now=RUN_ET)
        r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
        assert r.exit_code == 0 and r.stderr.strip() == "Season replay unavailable for 2027-04-16 (committed pick file missing)"
        report["item3"]["missing_pick_file"] = {"action": "blocked", "stderr": r.stderr.strip()}

        # 3.8 Applied-date marker on unresolved past day (same season, tagless, foreign season)
        for m_name, m_tag, should_block in [("same_season", 2027, True), ("tagless", None, True), ("foreign_season", 2026, False)]:
            p = _fresh(f"past_marker_{m_name}")
            _make_pick(picks_mod, p, "2027-04-16", None)
            picks_mod.save_streak(9, p, saver_available=False, applied_date="2027-04-16", season=m_tag)
            r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
            assert r.exit_code == 0
            if should_block:
                assert r.stderr.strip() == "Season replay unavailable for 2027-04-16 (applied-date marker without a result)"
            else:
                assert r.stderr == "" and "Streak: 1" in r.stdout
            report["item3"][f"past_marker_{m_name}"] = {"action": "blocked" if should_block else "skipped", "stderr": r.stderr.strip(), "stdout": r.stdout.strip()}

        # 3.9 Today's cases (pp1 hold same season, pp1 hold tagless, foreign marker, unmarked pending commit, terminal hit, malformed)
        for t_name, t_res, t_tag, t_block_reason, exp_streak in [
            ("pp1_same_season", None, 2027, "applied-date marker without a result", 9),
            ("pp1_tagless", None, None, "applied-date marker without a result", 0),
            ("foreign_marker", None, 2026, None, 1),
            ("unmarked_committed_pending", None, "no_marker", None, 1),
            ("terminal_hit", "hit", 2027, None, 2),
        ]:
            p = _fresh(f"today_{t_name}")
            _make_pick(picks_mod, p, "2027-04-20", t_res, delivered_dm=True)
            if t_tag != "no_marker":
                picks_mod.save_streak(9, p, saver_available=False, applied_date="2027-04-20", season=t_tag)
            r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
            assert r.exit_code == 0
            if t_block_reason:
                assert r.stderr.strip() == f"Season replay unavailable for 2027-04-20 ({t_block_reason})"
                assert f"Streak retained: {exp_streak}" in r.stdout
            else:
                assert r.stderr == "" and f"Streak: {exp_streak}" in r.stdout
            report["item3"][f"today_{t_name}"] = {"stderr": r.stderr.strip(), "stdout": r.stdout.strip()}

        # 3.10 Invalid pick and invalid decision exact reasons
        p = _fresh("invalid_pick_reason")
        (p / "2027-04-16.json").write_text("[]")
        r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
        assert r.stderr.strip() == "Season replay unavailable for 2027-04-16 (unreadable or invalid pick file)"
        report["item3"]["invalid_pick"] = {"stderr": r.stderr.strip()}

        p = _fresh("invalid_decision_reason")
        _make_pick(picks_mod, p, "2027-04-16", "hit", delivered_dm=True)
        decision_path("2027-04-16", p).parent.mkdir(parents=True, exist_ok=True)
        decision_path("2027-04-16", p).write_text("{}")
        r = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p)
        assert r.stderr.strip() == "Season replay unavailable for 2027-04-16 (unreadable or invalid decision record)"
        report["item3"]["invalid_decision"] = {"stderr": r.stderr.strip()}

        # 3.11 Adversarial sequence: crash mid-DM send -> restart -> locked_unconfirmed -> 02:00 reconcile
        p_adv = scratch / "adv_crash_mid_dm"
        p_adv.mkdir()
        _make_pick(picks_mod, p_adv, "2027-04-18", "hit", delivered_dm=True)
        picks_mod.save_streak(1, p_adv, saver_available=True, season=2027)
        # Simulate crash after delivery_attempted=True saved and before confirmation:
        _make_pick(picks_mod, p_adv, "2027-04-19", None, attempted=True)
        # Simulate scheduler restart classification lock:
        st_adv = sched_mod.SchedulerState(date="2027-04-19")
        dp_adv = picks_mod.load_pick("2027-04-19", p_adv)
        sched_mod._write_classification_decision(
            p_adv, "2027-04-19", action="single", delivered=False, attempted=True,
            primary=sched_mod._row_from_daily(dp_adv.pick), double_down=None, state=st_adv, daily=dp_adv,
        )
        r_adv = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p_adv)
        assert r_adv.exit_code == 0
        assert "Season replay unavailable for 2027-04-19 (committed pick awaiting a result)" in r_adv.stderr
        assert "Streak retained: 1" in r_adv.stdout
        report["adversarial"]["crash_mid_dm_restart_reconcile"] = "blocked_cleanly"

        # 3.12 Adversarial: chmod 000 decision subdirectory WITH vs WITHOUT root pick file
        if os.geteuid() != 0:
            for with_root in (True, False):
                p_sub = _fresh(f"chmod000_dec_dir_root_{with_root}")
                if with_root:
                    _make_pick(picks_mod, p_sub, "2027-04-16", "hit", delivered_dm=True)
                write_decision("2027-04-16", p_sub, action="single", source="mdp", delivery_status="delivered", scoreable=True, now=RUN_ET)
                dec_dir = p_sub / "2027-04-16"
                os.chmod(dec_dir, 0o000)
                try:
                    r_sub = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p_sub)
                    report["adversarial"][f"chmod000_decision_dir_with_root_{with_root}"] = {
                        "exit_code": r_sub.exit_code,
                        "stdout": r_sub.stdout.strip(),
                        "stderr": r_sub.stderr.strip(),
                    }
                finally:
                    os.chmod(dec_dir, 0o700)

        # ===================================================================
        # ITEM 4: CLI e2e (cutoffs, wording invariants, corrupt streak, FIFOs, concurrency, interrupt)
        # ===================================================================
        # 4.1 Cutoff inside (07:59:59 ET) vs outside (08:00:00 ET) in both directions + blocked vs unblocked wording
        for direction, before_res, after_res, exp_recalc in [
            ("hit_to_miss", "hit", "miss", 0),
            ("miss_to_hit", "miss", "hit", 2),
        ]:
            # Inside cutoff, replay succeeds -> "Streak recalculated" (and NOT "retained")
            p_in = scratch / f"i4_in_{direction}"
            p_in.mkdir()
            _make_pick(picks_mod, p_in, "2027-04-18", "hit", delivered_dm=True)
            _make_pick(picks_mod, p_in, "2027-04-19", before_res, delivered_dm=True)
            picks_mod.save_streak(5, p_in, saver_available=True, season=2027)
            r_in = _invoke_reconcile(
                cli_mod, picks_mod, rr_mod, p_in,
                clock_fn=lambda: datetime(2027, 4, 20, 7, 59, 59, tzinfo=ET),
                feed_map={"2027-04-19": {"pick": after_res}},
            )
            assert r_in.exit_code == 0 and r_in.stderr == ""
            assert f"2027-04-19: Shohei Ohtani — {before_res} -> {after_res}" in r_in.stdout
            assert f"Streak recalculated: {exp_recalc}" in r_in.stdout
            assert "Streak retained" not in r_in.stdout

            # At/past 08:00 ET cutoff -> no fetch, no correction, "No scoring changes detected. Streak:"
            p_cut = scratch / f"i4_cut_{direction}"
            p_cut.mkdir()
            _make_pick(picks_mod, p_cut, "2027-04-18", "hit", delivered_dm=True)
            _make_pick(picks_mod, p_cut, "2027-04-19", before_res, delivered_dm=True)
            picks_mod.save_streak(5, p_cut, saver_available=True, season=2027)
            r_cut = _invoke_reconcile(
                cli_mod, picks_mod, rr_mod, p_cut,
                clock_fn=lambda: datetime(2027, 4, 20, 8, 0, 0, tzinfo=ET),
                feed_map={},  # asserts resolve_daily_slot_results is NOT called!
            )
            assert r_cut.exit_code == 0 and r_cut.stderr == ""
            assert picks_mod.load_pick("2027-04-19", p_cut).result == before_res
            assert "CORRECTIONS FOUND" not in r_cut.stdout
            assert "Streak recalculated" not in r_cut.stdout
            assert "Streak retained" not in r_cut.stdout

            # Inside cutoff WITH an earlier blocked date -> correction applied, "Streak retained: 5", NEVER "recalculated"
            p_blk = scratch / f"i4_blk_{direction}"
            p_blk.mkdir()
            _make_pick(picks_mod, p_blk, "2027-04-15", None, delivered_dm=True)
            _make_pick(picks_mod, p_blk, "2027-04-19", before_res, delivered_dm=True)
            picks_mod.save_streak(5, p_blk, saver_available=True, season=2027)
            r_blk = _invoke_reconcile(
                cli_mod, picks_mod, rr_mod, p_blk,
                clock_fn=lambda: datetime(2027, 4, 20, 2, 0, 0, tzinfo=ET),
                feed_map={"2027-04-19": {"pick": after_res}},
            )
            assert r_blk.exit_code == 0
            assert "Season replay unavailable for 2027-04-15 (committed pick awaiting a result)" in r_blk.stderr
            assert f"2027-04-19: Shohei Ohtani — {before_res} -> {after_res}" in r_blk.stdout
            assert "Streak retained: 5" in r_blk.stdout
            assert "Streak recalculated" not in r_blk.stdout
            assert picks_mod.load_pick("2027-04-19", p_blk).result == after_res

        report["item4"]["cutoff_and_wording_matrix"] = "all_passed"

        # 4.2 Unreadable/malformed streak.json with blocked replay vs complete replay
        bad_streak_payloads = [
            ("truncated", b"{bad json"),
            ("array", b"[]"),
            ("string_streak", b'{"season": 2027, "streak": "five"}'),
            ("bool_streak", b'{"season": 2027, "streak": true}'),
            ("giant_int", b'{"season": 2027, "streak": ' + (b"9" * 5000) + b"}"),
        ]
        for b_name, b_bytes in bad_streak_payloads:
            # Blocked replay -> exit 1, no traceback, preserves exact bytes
            p_b = scratch / f"i4_badstreak_blocked_{b_name}"
            p_b.mkdir()
            _make_pick(picks_mod, p_b, "2027-04-16", None, delivered_dm=True)
            sp = p_b / "streak.json"
            sp.write_bytes(b_bytes)
            r_b = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p_b)
            assert r_b.exit_code == 1, f"{b_name}: expected exit 1, got {r_b.exit_code} ({r_b.output})"
            assert "Traceback" not in r_b.output and "Traceback" not in r_b.stderr
            assert "Streak unavailable" in r_b.stderr or "Streak unavailable" in r_b.output
            assert sp.read_bytes() == b_bytes

            # Complete replay -> exit 0, rebuilds valid JSON atomically
            p_c = scratch / f"i4_badstreak_complete_{b_name}"
            p_c.mkdir()
            for idx in range(10):
                _make_pick(picks_mod, p_c, (datetime(2027, 4, 1).date() + timedelta(days=idx)).isoformat(), "hit", delivered_dm=True)
            _make_pick(picks_mod, p_c, "2027-04-11", "miss", delivered_dm=True)
            sp_c = p_c / "streak.json"
            sp_c.write_bytes(b_bytes)
            r_c = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p_c)
            assert r_c.exit_code == 0
            rebuilt = json.loads(sp_c.read_text())
            assert (rebuilt["streak"], rebuilt["saver_available"], rebuilt["season"]) == (10, False, 2027)

        report["item4"]["bad_streak_blocked_and_rebuilt"] = "all_5_payloads_passed"

        # 4.3 Subprocess-bounded special-file probes (FIFO, directory, symlink)
        special_cases: dict[str, dict] = {}

        def _run_special_subproc(case_name: str, setup_fn, feed_hit: str = "", timeout_sec: float = 4.0):
            base = scratch / f"sp_{case_name}"
            base.mkdir()
            pdir = setup_fn(base)
            t0 = time.monotonic()
            try:
                proc = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--special", str(pdir), RUN_ET.isoformat(), feed_hit],
                    capture_output=True,
                    text=True,
                    timeout=timeout_sec,
                )
                elapsed = round(time.monotonic() - t0, 3)
                parsed = json.loads(proc.stdout.strip().splitlines()[-1]) if proc.returncode == 0 and proc.stdout.strip() else {"raw_rc": proc.returncode, "raw_err": proc.stderr[-300:]}
                special_cases[case_name] = {"hung": False, "elapsed_s": elapsed, **parsed}
            except subprocess.TimeoutExpired:
                special_cases[case_name] = {"hung": True, "elapsed_s": round(time.monotonic() - t0, 3)}

        # Pick files (recent & older): FIFO, dir, symlink
        for loc, day in [("recent_pick", "2027-04-19"), ("older_pick", "2027-04-10")]:
            for kind in ("fifo", "dir", "symlink"):
                def _s(base: Path, d=day, k=kind):
                    p = base / "picks"
                    p.mkdir()
                    picks_mod.save_streak(3, p, saver_available=True, season=2027)
                    target = p / f"{d}.json"
                    if k == "fifo":
                        os.mkfifo(target)
                    elif k == "dir":
                        target.mkdir()
                    else:
                        real = base / "outside.json"
                        real.write_text("{}")
                        target.symlink_to(real)
                    return p

                _run_special_subproc(f"{loc}_{kind}", _s, feed_hit="hit")
                sc = special_cases[f"{loc}_{kind}"]
                assert not sc["hung"] and sc["exit_code"] == 0 and f"Season replay unavailable for {day} (unreadable or invalid pick file)" in sc["stderr"]

        # Decision files: FIFO, dir, symlink, and parent date path as FIFO / regular file
        for kind in ("fifo", "dir", "symlink", "parent_fifo", "parent_regfile"):
            def _sd(base: Path, k=kind):
                p = base / "picks"
                p.mkdir()
                _make_pick(picks_mod, p, "2027-04-10", "hit", delivered_dm=True)
                picks_mod.save_streak(3, p, saver_available=True, season=2027)
                if k == "parent_fifo":
                    os.mkfifo(p / "2027-04-10")
                elif k == "parent_regfile":
                    (p / "2027-04-10").write_text("not a dir")
                else:
                    ddir = p / "2027-04-10"
                    ddir.mkdir()
                    target = ddir / "decision.json"
                    if k == "fifo":
                        os.mkfifo(target)
                    elif k == "dir":
                        target.mkdir()
                    else:
                        real = base / "outside_dec.json"
                        real.write_text("{}")
                        target.symlink_to(real)
                return p

            _run_special_subproc(f"decision_{kind}", _sd)
            sc = special_cases[f"decision_{kind}"]
            assert not sc["hung"] and sc["exit_code"] == 0 and "Season replay unavailable for 2027-04-10 (unreadable or invalid decision record)" in sc["stderr"]

        # streak.json: FIFO, dir, symlink
        for kind in ("fifo", "dir", "symlink"):
            def _ss(base: Path, k=kind):
                p = base / "picks"
                p.mkdir()
                _make_pick(picks_mod, p, "2027-04-10", "hit", delivered_dm=True)
                target = p / "streak.json"
                if k == "fifo":
                    os.mkfifo(target)
                elif k == "dir":
                    target.mkdir()
                else:
                    real = base / "outside_streak.json"
                    real.write_text('{"streak": 1, "season": 2027}')
                    target.symlink_to(real)
                return p

            _run_special_subproc(f"streak_{kind}", _ss)
            sc = special_cases[f"streak_{kind}"]
            assert not sc["hung"] and sc["exit_code"] == 1 and "streak.json is not a regular file" in sc["stderr"] and "Traceback" not in sc["stderr"]

        # picks_dir itself as FIFO or regular file; receipts_root as FIFO or regular file
        for kind in ("picks_dir_fifo", "picks_dir_regfile", "receipts_root_fifo", "receipts_root_dir_ok"):
            def _sp(base: Path, k=kind):
                if k == "picks_dir_fifo":
                    p = base / "picks"
                    os.mkfifo(p)
                    return p
                if k == "picks_dir_regfile":
                    p = base / "picks"
                    p.write_text("file")
                    return p
                p = base / "picks"
                p.mkdir()
                _make_pick(picks_mod, p, "2027-04-10", "hit", delivered_dm=True)
                if k == "receipts_root_fifo":
                    hs = base / "health_state"
                    hs.mkdir()
                    os.mkfifo(hs / "reconcile_receipts")
                return p

            _run_special_subproc(kind, _sp)
            assert not special_cases[kind]["hung"]

        # Also check untouched `.scoring.lock` (dir vs FIFO) and `lineup_evolution_*.jsonl` (dir vs FIFO on correction)
        for kind in ("scoring_lock_dir", "scoring_lock_fifo", "lineup_evo_dir", "lineup_evo_fifo"):
            def _su(base: Path, k=kind):
                p = base / "picks"
                p.mkdir()
                _make_pick(picks_mod, p, "2027-04-19", "miss", delivered_dm=True)
                picks_mod.save_streak(0, p, saver_available=True, season=2027)
                if k == "scoring_lock_dir":
                    (p / ".scoring.lock").mkdir()
                elif k == "scoring_lock_fifo":
                    os.mkfifo(p / ".scoring.lock")
                elif k == "lineup_evo_dir":
                    lp = p / "lineup_evolution_2027-04-19.jsonl"
                    lp.unlink(missing_ok=True)
                    lp.mkdir()
                elif k == "lineup_evo_fifo":
                    lp = p / "lineup_evolution_2027-04-19.jsonl"
                    lp.unlink(missing_ok=True)
                    os.mkfifo(lp)
                return p

            _run_special_subproc(kind, _su, feed_hit="hit", timeout_sec=2.0)

        report["item4"]["special_files"] = special_cases

        # 4.4 Two reconciles at once (concurrent subprocesses contending on scoring_lock)
        p_conc = scratch / "i4_concurrent" / "picks"
        p_conc.mkdir(parents=True)
        _make_pick(picks_mod, p_conc, "2027-04-18", "hit", delivered_dm=True)
        _make_pick(picks_mod, p_conc, "2027-04-19", "miss", delivered_dm=True)
        picks_mod.save_streak(1, p_conc, saver_available=True, season=2027)

        proc_a = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--concurrent", str(p_conc), "350", "hit"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        time.sleep(0.05)
        proc_b = subprocess.Popen(
            [sys.executable, str(Path(__file__).resolve()), "--concurrent", str(p_conc), "0", "hit"],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        )
        out_a, err_a = proc_a.communicate(timeout=10)
        out_b, err_b = proc_b.communicate(timeout=10)
        assert proc_a.returncode == 0 and proc_b.returncode == 0, (err_a, err_b)
        res_a, res_b = json.loads(out_a), json.loads(out_b)
        assert res_a["exit_code"] == 0 and res_b["exit_code"] == 0
        final_streak_doc = json.loads((p_conc / "streak.json").read_text())
        assert (final_streak_doc["streak"], final_streak_doc["saver_available"], final_streak_doc["season"]) == (2, True, 2027)
        published_receipts = rr_mod.discover(p_conc, "2027-04-20")
        assert len(published_receipts) == 2
        report["item4"]["concurrent_reconciles"] = {
            "proc_a": res_a["stdout"].strip(),
            "proc_b": res_b["stdout"].strip(),
            "receipts_published": len(published_receipts),
            "final_streak": final_streak_doc["streak"],
        }

        # 4.5 Interrupt between correction writes and streak write, then retry
        p_int = scratch / "i4_interrupt" / "picks"
        p_int.mkdir(parents=True)
        _make_pick(picks_mod, p_int, "2027-04-16", None)  # uncommitted preview (skipped)
        _make_pick(picks_mod, p_int, "2027-04-18", "hit", delivered_dm=True)
        _make_pick(picks_mod, p_int, "2027-04-19", "miss", delivered_dm=True)
        picks_mod.save_streak(99, p_int, saver_available=False, season=2027)
        before_bytes = (p_int / "streak.json").read_bytes()

        with patch.object(picks_mod, "_write_streak", side_effect=KeyboardInterrupt("simulated mid-run interrupt")):
            r_int = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p_int, feed_map={"2027-04-19": {"pick": "hit"}})
        assert r_int.exit_code != 0
        assert picks_mod.load_pick("2027-04-19", p_int).result == "hit"
        assert (p_int / "streak.json").read_bytes() == before_bytes
        int_receipts = rr_mod.discover(p_int, "2027-04-20")
        assert len(int_receipts) == 1 and int_receipts[0]["outcome"] == "raised" and int_receipts[0]["error"] == "KeyboardInterrupt"

        # Retry: 0 new corrections, yet refolds season and repairs streak.json from 99 -> 2!
        r_retry = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p_int, feed_map={"2027-04-19": {"pick": "hit"}})
        assert r_retry.exit_code == 0 and "No scoring changes detected. Streak: 2" in r_retry.stdout
        rebuilt_int = json.loads((p_int / "streak.json").read_text())
        assert (rebuilt_int["streak"], rebuilt_int["saver_available"], rebuilt_int["season"]) == (2, True, 2027)
        report["item4"]["interrupt_and_retry"] = "passed"

        # ===================================================================
        # ITEM 5: Other callers parity (Base c2039541 vs HEAD c70effe9)
        # ===================================================================
        base_extract = scratch / "base_c2039541"
        base_extract.mkdir()
        with subprocess.Popen(["git", "archive", "c2039541", "src"], cwd=REPO_ROOT, stdout=subprocess.PIPE) as arch:
            assert arch.stdout is not None
            with tarfile.open(fileobj=arch.stdout, mode="r|") as tf:
                tf.extractall(base_extract, filter="data")
            assert arch.wait() == 0

        w_base = scratch / "callers_base"
        w_head = scratch / "callers_head"
        w_base.mkdir()
        w_head.mkdir()
        p_base_callers = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--callers", str(base_extract / "src"), str(w_base)],
            capture_output=True, text=True, check=True, timeout=15,
        )
        p_head_callers = subprocess.run(
            [sys.executable, str(Path(__file__).resolve()), "--callers", str(REPO_ROOT / "src"), str(w_head)],
            capture_output=True, text=True, check=True, timeout=15,
        )
        snap_base = json.loads(p_base_callers.stdout)
        snap_head = json.loads(p_head_callers.stdout)
        assert snap_base == snap_head, f"Caller parity mismatch!\nBase: {snap_base}\nHead: {snap_head}"
        report["item5"]["caller_parity_base_vs_head"] = {
            "equal": True,
            "checked": list(snap_head.keys()),
            "update_streak_steps_byte_identical": len(snap_head["update_streak_snapshots"]),
        }

    out_file = REVIEW_DIR / "e2e_and_callers_results.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n")
    print("ALL E2E & CALLERS PROBES PASSED")
    print(json.dumps({
        "item3_cases": len(report["item3"]),
        "special_files_hung": {k: v["hung"] for k, v in report["item4"]["special_files"].items() if v["hung"]},
        "adversarial": report["adversarial"],
        "item5": report["item5"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
