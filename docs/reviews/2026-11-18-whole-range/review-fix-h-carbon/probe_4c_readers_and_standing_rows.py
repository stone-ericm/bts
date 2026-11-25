"""Adversarial probe 4c: objective readers (Item 4 / A5) and standing rows + ordinary sequences (Item 5)."""
from __future__ import annotations

from contextlib import nullcontext
from datetime import date, datetime, time, timezone
import gzip
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock, patch

from click.testing import CliRunner
import pytest

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SCRATCH = OUT_DIR / "scratch"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bts import (
    cli as cli_mod,
    contest_fetch,
    contest_state,
    daily_decision,
    day_outcome,
    dm,
    entry_receipt,
    orchestrator,
    picks,
    scheduler,
    skip_policy_shadow,
    strategy,
    web,
)
from bts.health import mdp_policy_alignment, post_failure, tail_policy as health_tail
from bts.leaderboard import auth
from bts.simulate import mdp, tail_policy
from scripts.audit import boundary_shadow_census as census
from scripts.audit.season_ledger.sources import day_records as ledger_day_records
from scripts.audit.season_ledger.sources import pick_files as ledger_pick_files
from scripts.audit.season_walk import walk


class SimulatedCrash(BaseException):
    """Simulates SIGKILL / power loss / alarm interrupt (not caught by except Exception)."""


def _write_units_snapshot(picks_dir: Path, season: int, mapping: list[tuple[int, int, int]]) -> None:
    snap_dir = entry_receipt.units_snapshot_dir(picks_dir)
    snap_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "units": [{"id": uid, "feedId": game_pk, "roundId": round_id} for uid, game_pk, round_id in mapping]
    }
    raw = gzip.compress(json.dumps(payload).encode())
    (snap_dir / f"{season}0926T120000Z.json.gz").write_bytes(raw)


def check_all_readers(tmp_dir: Path) -> dict:
    strategy._mdp_cache = None
    artifact = strategy._load_mdp()
    tail_sha = artifact["tail"].sha256

    pdir = tmp_dir / "readers_picks"
    pdir.mkdir(parents=True, exist_ok=True)
    cfg = walk._config(pdir)
    cfg["orchestrator"]["heartbeat_path"] = str(pdir / ".heartbeat")
    cfg["bluesky"] = {"dm_recipient": "operator-did"}

    for d_str, want_act in (("2027-07-15", "single"), ("2027-09-26", "double")):
        day = date.fromisoformat(d_str)
        frame = walk._predictions(day, 0.96).head(2)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
        noon = datetime.combine(day, time(12), walk.ET)
        with walk.offline(noon), pytest.MonkeyPatch.context() as mp:
            mp.setattr(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe"))
            mp.setattr(picks, "get_game_statuses_detailed", lambda *a: statuses)
            mp.setattr(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext())
            mp.setattr(scheduler, "count_new_confirmations", lambda *a: 4)
            mp.setattr(scheduler, "_now_et", lambda: noon)
            mp.setattr(scheduler, "_void_committed_games", lambda *a: {})
            mp.setattr(scheduler, "_trigger_live_forward_capture_on_lock", lambda *a, **k: None)
            mp.setattr(dm, "send_dm", Mock(return_value=f"dm-{d_str}"))
            assert walk.install_state(pdir, day, walk.State("dated", 57, 57, "used", 0.96)) == []

            res = scheduler.run_single_check(d_str, list(statuses), set(), cfg, 0.01)
            daily = picks.load_pick(d_str, pdir)

            pre_align = mdp_policy_alignment.compute_metrics(pdir, REPO / "data/models/mdp_policy.npz", today=day)
            assert pre_align.primary.n == 1

            st = scheduler.SchedulerState(d_str, noon.isoformat(), [], [], [], False, None, None, None)
            assert scheduler._deliver_and_lock_pick(
                daily, cfg, pdir, st, d_str, "lineup", selection=res["selection"]
            )

        rec = daily_decision.load_decision(d_str, pdir)
        assert rec["objective"] == "longest_streak" and rec["action"] == want_act
        assert daily_decision.decision_objective(rec) == "longest_streak"
        assert daily_decision.is_reach57_mdp_skip(rec) is False

        post_align = mdp_policy_alignment.compute_metrics(pdir, REPO / "data/models/mdp_policy.npz", today=day)
        assert post_align.primary.n == 0 and post_align.double_down.n == 0
        assert mdp_policy_alignment.check(pdir, REPO / "data/models/mdp_policy.npz", today=day) == []

    rec_single = daily_decision.load_decision("2027-07-15", pdir)
    assert rec_single["double_down"] is not None
    assert day_outcome._pick_text(rec_single) == "Batter 0"
    rec_double = daily_decision.load_decision("2027-09-26", pdir)
    assert day_outcome._pick_text(rec_double) == "Batter 0 + Batter 1"

    skip_dir = tmp_dir / "shadow_skip"
    skip_dir.mkdir(parents=True, exist_ok=True)
    now_utc = datetime(2027, 7, 20, 12, tzinfo=timezone.utc)
    daily_decision.write_decision(
        "2027-07-19", skip_dir, action="skip", source="mdp", objective="longest_streak",
        primary={"batter_id": 100, "batter_name": "Batter 0", "team": "T0", "game_pk": 70000, "p_game_hit": 0.96},
        delivery_status="not_applicable", scoreable=False, now=now_utc,
    )
    assert skip_policy_shadow.record_skip_from_decision("2027-07-19", skip_dir, now=now_utc) is None
    assert skip_policy_shadow.record_pending_skips(skip_dir, now=now_utc) == []
    (skip_dir / "2027-07-19.policy_shadow.json").write_text(json.dumps({"date": "2027-07-19", "divergent": True}))
    assert skip_policy_shadow.prune_superseded(skip_dir, now=now_utc) == ["2027-07-19"]
    status_doc = skip_policy_shadow.write_status(skip_dir, skip_dir / "status.json", now=now_utc)
    assert status_doc["counts"]["divergent_days"] == 0

    ledger_path = tmp_dir / "census_ledger.jsonl"
    ledger_path.write_text("")
    census_out = tmp_dir / "census_out.json"
    census_rep = census.run_census(
        picks_dir=pdir,
        ledger_path=ledger_path,
        policy_path=REPO / "data/models/mdp_policy.npz",
        saver_active=False,
        start="2027-07-01",
        end="2027-09-30",
        as_of="2027-09-30",
        output=census_out,
    )
    assert census_rep["boundaries"]["primary"]["n"] == 0
    assert census_rep["boundaries"]["s2_flat_season"]["n"] == 2
    assert [r["state_source"] for r in census_rep["rows"]] == [
        "excluded_tail_objective", "excluded_tail_objective"
    ]
    assert all(r["policy_identity"] == "verified" and r["diffs"] == {} for r in census_rep["rows"])

    _write_units_snapshot(pdir, 2027, [(501, 70000, 99), (502, 70001, 99)])
    pending_rows = [
        {"roundId": 99, "unitId": 501, "playerId": 1000, "number": 1, "result": None},
        {"roundId": 99, "unitId": 502, "playerId": 1001, "number": 2, "result": None},
    ]
    crosswalk = {1000: 100, 1001: 101}
    session = SimpleNamespace(user_id=4242, username=walk.ACCOUNT, xsid="xsid")
    with patch.object(auth, "load_session_cookies", lambda: {}), \
            patch.object(auth, "extract_uid", lambda _c: 4242), \
            patch.object(auth, "fetch_login_session", lambda **_k: session), \
            patch.object(contest_fetch, "fetch_profile", lambda *_a: {"predictions": []}), \
            patch.object(contest_fetch, "fetch_pending_predictions", lambda *_a: pending_rows), \
            patch.object(cli_mod, "_fetch_rounds", lambda: {99: date(2027, 9, 26)}), \
            patch.object(cli_mod, "_fetch_bts_to_mlb", lambda: crosswalk):
        res_entry = CliRunner().invoke(
            cli_mod.cli,
            [
                "check-pick-entered",
                "--picks-dir", str(pdir),
                "--expected-username", walk.ACCOUNT,
                "--now-et", "2027-09-26T18:35:00",
            ],
        )
    assert res_entry.exit_code == 0, res_entry.output
    receipts = entry_receipt.discover(pdir, "2027-09-26")
    assert len(receipts) == 1
    assert receipts[0]["outcome"] == "observed"
    assert receipts[0]["qualification"]["all_confirmed"] is True
    assert len(receipts[0]["qualification"]["slots"]) == 2

    outcome_double = day_outcome.gather(
        "2027-09-26",
        pdir,
        cfg,
        expected_username=walk.ACCOUNT,
        fetch_schedule=lambda _d: [{"status": {"detailedState": "Final"}}],
    )
    assert outcome_double.ok is True
    assert outcome_double.state == "entered"
    assert outcome_double.pick == "Batter 0 + Batter 1"

    picks.save_streak(57, pdir, saver_available=False, season=2027)
    with walk.offline(datetime(2027, 7, 16, 1, 0, tzinfo=walk.ET)), \
            patch.object(picks, "resolve_daily_slot_results", lambda _d, _dt: {"pick": "hit"}), \
            patch.object(cli_mod, "_now_et", lambda: datetime(2027, 7, 16, 1, 0, tzinfo=walk.ET)):
        r_grade1 = CliRunner().invoke(
            cli_mod.cli, ["check-results", "--date", "2027-07-15", "--picks-dir", str(pdir)]
        )
    assert r_grade1.exit_code == 0, r_grade1.output
    assert picks.load_streak(pdir, season=2027) == 58

    with walk.offline(datetime(2027, 9, 27, 1, 0, tzinfo=walk.ET)), \
            patch.object(picks, "resolve_daily_slot_results", lambda _d, _dt: {"pick": "hit", "double_down": "hit"}), \
            patch.object(cli_mod, "_now_et", lambda: datetime(2027, 9, 27, 1, 0, tzinfo=walk.ET)):
        r_grade2 = CliRunner().invoke(
            cli_mod.cli, ["check-results", "--date", "2027-09-26", "--picks-dir", str(pdir)]
        )
    assert r_grade2.exit_code == 0, r_grade2.output
    assert picks.load_streak(pdir, season=2027) == 60

    replay_val, blocker = picks._replay_season_streak_detail(pdir, 2027, "2027-09-27")
    assert blocker is None and replay_val == (3, True)

    assert post_failure.check(pdir, today=date(2027, 9, 26)) == []
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(web, "PICKS_DIR", pdir)
        assert web._decision_action("2027-07-15") == "single"
        assert web._decision_action("2027-09-26") == "double"
        assert len(web.load_all_picks()) == 2

    raw_dec_bytes = daily_decision.decision_path("2027-09-26", pdir).read_bytes()
    parsed_dec = ledger_day_records.parse_decision("picks/2027-09-26/decision.json", raw_dec_bytes)
    assert parsed_dec.quarantined == [] and parsed_dec.rows[0]["objective"] == "unknown"
    assert parsed_dec.rows[0]["objective_raw"] == "longest_streak"

    raw_pick_bytes = (pdir / "2027-09-26.json").read_bytes()
    parsed_pick = ledger_pick_files.parse_pick_file("picks/2027-09-26.json", raw_pick_bytes)
    assert parsed_pick.quarantined == [] and len(parsed_pick.rows) == 2
    assert all(r["pick_policy_objective"] == "longest_streak" for r in parsed_pick.rows)

    return {"readers_verified": 11, "graded_streak_after_single_and_double": 60, "tail_sha_prefix": tail_sha[:12]}


def check_standing_rows_and_ordinary_sequences(tmp_dir: Path) -> dict:
    strategy._mdp_cache = None
    artifact = strategy._load_mdp()
    tail_sha = artifact["tail"].sha256

    # --- Sequence A: Failed send followed by retry success, then separate mid-send crash (unconfirmed) ---
    pdir_send = tmp_dir / "seq_send"
    pdir_send.mkdir(parents=True, exist_ok=True)
    cfg = walk._config(pdir_send)
    cfg["bluesky"] = {"dm_recipient": "operator-did"}
    day = date(2027, 9, 25)
    d_str = day.isoformat()
    noon = datetime.combine(day, time(12), walk.ET)
    frame = walk._predictions(day, 0.96).head(2)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}

    with walk.offline(noon), pytest.MonkeyPatch.context() as mp:
        mp.setattr(orchestrator, "run_cascade", lambda *a: (frame.copy(), "probe"))
        mp.setattr(picks, "get_game_statuses_detailed", lambda *a: statuses)
        mp.setattr(scheduler, "_now_et", lambda: noon)
        mp.setattr(scheduler, "_void_committed_games", lambda *a: {})
        mp.setattr(scheduler, "_trigger_live_forward_capture_on_lock", lambda *a, **k: None)
        assert walk.install_state(pdir_send, day, walk.State("automatic", 57, 57, "used", 0.96)) == []

        _, sel, _ = orchestrator.run_and_pick(cfg, d_str)
        daily = sel.pick_result.daily
        picks.save_pick(daily, pdir_send)
        st = scheduler.SchedulerState(d_str, noon.isoformat(), [], [], [], False, None, None, None)

        mp.setattr(dm, "send_dm", Mock(side_effect=OSError("502 Bad Gateway")))
        assert scheduler._deliver_and_lock_pick(daily, cfg, pdir_send, st, d_str, "lineup", selection=sel) is False
        assert picks.load_pick(d_str, pdir_send).delivery_attempted is False
        assert daily_decision.load_decision(d_str, pdir_send) is None
        assert st.pick_locked is False

        mp.setattr(dm, "send_dm", Mock(return_value="dm-retry-ok"))
        assert scheduler._deliver_and_lock_pick(daily, cfg, pdir_send, st, d_str, "retry", selection=sel) is True
        rec_retry = daily_decision.load_decision(d_str, pdir_send)
        assert rec_retry["delivery_status"] == "delivered" and rec_retry["objective"] == "longest_streak"

        d2 = "2027-09-26"
        frame2 = walk._predictions(date(2027, 9, 26), 0.96).head(2)
        mp.setattr(orchestrator, "run_cascade", lambda *a: (frame2.copy(), "probe"))
        mp.setattr(scheduler, "_now_et", lambda: datetime(2027, 9, 26, 12, tzinfo=walk.ET))
        walk._set_clock(datetime(2027, 9, 26, 12, tzinfo=walk.ET))
        _, sel2, _ = orchestrator.run_and_pick(cfg, d2)
        daily2 = sel2.pick_result.daily
        st2 = scheduler.SchedulerState(d2, "2027-09-26T12:00:00", [], [], [], False, None, None, None)
        mp.setattr(dm, "send_dm", Mock(side_effect=SimulatedCrash("power cut mid-HTTP")))
        with pytest.raises(SimulatedCrash):
            scheduler._deliver_and_lock_pick(daily2, cfg, pdir_send, st2, d2, "lineup", selection=sel2)
        reloaded2 = picks.load_pick(d2, pdir_send)
        assert reloaded2.delivery_attempted is True and not picks.pick_was_delivered(reloaded2)
        assert daily_decision.load_decision(d2, pdir_send) is None

        resend_spy = Mock(return_value="should-not-run")
        unknown_alert_spy = Mock()
        mp.setattr(dm, "send_dm", resend_spy)
        mp.setattr(scheduler, "_alert_delivery_outcome_unknown", unknown_alert_spy)
        assert scheduler._deliver_and_lock_pick(reloaded2, cfg, pdir_send, st2, d2, "recovery", selection=None) is False
        assert resend_spy.call_count == 0
        assert unknown_alert_spy.call_count == 1
        rec_unconf = daily_decision.load_decision(d2, pdir_send)
        assert rec_unconf["delivery_status"] == "locked_unconfirmed"
        assert rec_unconf["objective"] == "longest_streak"
        assert rec_unconf["tail_policy_sha256"] == tail_sha

    # --- Sequence B: Finale (2027-09-26) vs Day After (2027-09-27) & Dec 31 vs Jan 1 ---
    ctx_fin = strategy.DecisionContext(0.96, 0.95, True, "2027-09-26", True, artifact, 57, "trusted")
    ctx_post = strategy.DecisionContext(0.96, 0.95, True, "2027-09-27", True, artifact, 57, "trusted")
    ctx_post_no_base_hi = strategy.DecisionContext(0.90, 0.85, True, "2027-09-27", True, None, 57, "trusted")
    ctx_post_no_base_lo = strategy.DecisionContext(0.70, 0.65, True, "2027-09-27", True, None, 57, "trusted")
    assert strategy.resolve_policy_decision(ctx_fin, 57, False).action == "double"
    dec_post = strategy.resolve_policy_decision(ctx_post, 57, False)
    assert dec_post.action == "skip" and dec_post.objective == "reach57" and dec_post.pick_bar is None
    dec_post_hi = strategy.resolve_policy_decision(ctx_post_no_base_hi, 57, False)
    assert dec_post_hi.action == "single" and dec_post_hi.source == "heuristic"
    dec_post_lo = strategy.resolve_policy_decision(ctx_post_no_base_lo, 57, False)
    assert dec_post_lo.action == "skip" and dec_post_lo.source == "heuristic" and dec_post_lo.pick_bar == 0.80

    pdir_ny = tmp_dir / "seq_newyear"
    pdir_ny.mkdir(parents=True, exist_ok=True)
    with walk.offline(datetime(2027, 12, 31, 20, tzinfo=walk.ET)):
        assert walk.install_state(pdir_ny, date(2027, 12, 31), walk.State("automatic", 57, 57, "used", 0.96)) == []
        ctx_dec31 = strategy.DecisionContext(0.96, 0.95, True, "2027-12-31", True, artifact, 57, "trusted")
        dec_dec31 = strategy.resolve_policy_decision(ctx_dec31, 57, False)
        assert dec_dec31.action == "skip" and dec_dec31.objective == "reach57" and dec_dec31.days_effective == 0

        at_2359 = datetime(2027, 12, 31, 23, 59, tzinfo=walk.ET)
        walk._set_clock(at_2359)
        st_2359 = contest_state.load_decision_streak_state(pdir_ny, require_contest_state=True, now=at_2359)
        assert st_2359.streak == 57

        at_0001 = datetime(2028, 1, 1, 0, 1, tzinfo=walk.ET)
        walk._set_clock(at_0001)
        with pytest.raises(contest_state.ContestStateError):
            contest_state.load_decision_streak_state(pdir_ny, require_contest_state=True, now=at_0001)

        assert walk.install_state(pdir_ny, date(2028, 1, 1), walk.State("automatic", 57, 57, "used", 0.96)) == []
        frame_2028 = walk._predictions(date(2028, 1, 1), 0.96).head(2)
        statuses_2028 = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame_2028["game_pk"]}
        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(orchestrator, "run_cascade", lambda *a: (frame_2028.copy(), "probe"))
            mp.setattr(picks, "get_game_statuses_detailed", lambda *a: statuses_2028)
            _, sel_2028, _ = orchestrator.run_and_pick(walk._config(pdir_ny), "2028-01-01")
        assert strategy._days_remaining("2028-01-01") == 271
        assert sel_2028.action == "single"
        assert sel_2028.decision.objective == "longest_streak" and sel_2028.decision.days_effective == 180
        warns_2028 = health_tail.season_end_alerts(today=date(2028, 1, 1))
        assert len(warns_2028) == 1 and "not configured" in warns_2028[0].message

    # --- Sequence C: Limits 1, 2, 6, 11 ---
    pdir_lim = tmp_dir / "seq_limits"
    pdir_lim.mkdir(parents=True, exist_ok=True)
    cfg_lim = walk._config(pdir_lim)
    cfg_lim["orchestrator"]["heartbeat_path"] = str(pdir_lim / ".heartbeat")
    cfg_lim["bluesky"] = {"dm_recipient": "operator-did"}
    with walk.offline(datetime(2027, 7, 15, 12, tzinfo=walk.ET)), pytest.MonkeyPatch.context() as mp:
        assert walk.install_state(pdir_lim, date(2027, 7, 15), walk.State("automatic", 57, 57, "used", 0.96)) == []
        with walk._fetch_stubs(58, 58, date(2027, 7, 14)):
            code_58, _ = walk._invoke(
                "fetch-contest-streak", "--picks-dir", pdir_lim, "--expected-username", walk.ACCOUNT
            )
        assert code_58 != 0
        assert contest_state.load_decision_streak_state(pdir_lim, require_contest_state=True).streak == 57

        frame_lim = walk._predictions(date(2027, 7, 15), 0.96).head(2)
        started_statuses = {int(pk): {"abstract": "L", "detailed": "In Progress"} for pk in frame_lim["game_pk"]}
        mp.setattr(orchestrator, "run_cascade", lambda *a: (frame_lim.copy(), "probe"))
        mp.setattr(picks, "get_game_statuses_detailed", lambda *a: started_statuses)
        mp.setattr(scheduler, "heartbeat_watchdog", lambda *a, **k: nullcontext())
        mp.setattr(scheduler, "count_new_confirmations", lambda *a: 0)
        no_elig = scheduler.run_single_check("2027-07-15", list(started_statuses), set(), cfg_lim, 0.01)
        assert no_elig["selection"].no_pick_reason == "no_eligible"
        assert no_elig["skip_summary"]["bar"] == 0.0
        assert "below the streak-57 bar of 0.0%" in scheduler.format_skip_dm("2027-07-15", no_elig["skip_summary"])

        sched_statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame_lim["game_pk"]}
        mp.setattr(picks, "get_game_statuses_detailed", lambda *a: sched_statuses)
        _, sel_57, _ = orchestrator.run_and_pick(cfg_lim, "2027-07-15")
        saved_57 = sel_57.pick_result.daily
        picks.save_pick(saved_57, pdir_lim)

        walk._set_clock(datetime(2027, 7, 15, 13, tzinfo=walk.ET))
        with walk._fetch_stubs(0, 57, date(2027, 7, 14)):
            code_0, out_0 = walk._invoke(
                "fetch-contest-streak", "--picks-dir", pdir_lim, "--expected-username", walk.ACCOUNT
            )
        assert code_0 == 0, out_0
        st_now_0 = contest_state.load_decision_streak_state(pdir_lim, require_contest_state=True)
        assert st_now_0.streak == 0
        assert picks.saved_pick_refusal(saved_57, st_now_0) is None

        other_frame = walk._predictions(date(2027, 7, 15), 0.70).head(1)
        other_frame["batter_id"] = [900]
        other_frame["batter_name"] = ["Batter 9"]
        other_statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in other_frame["game_pk"]}
        sent_lim_dms: list[str] = []
        mp.setattr(orchestrator, "run_cascade", lambda *a: (other_frame.copy(), "probe"))
        mp.setattr(picks, "get_game_statuses_detailed", lambda *a: other_statuses)
        mp.setattr(strategy, "_load_mdp", lambda: None)
        mp.setattr(scheduler, "_now_et", lambda: datetime(2027, 7, 15, 13, tzinfo=walk.ET))
        mp.setattr(scheduler, "_void_committed_games", lambda *a: {})
        mp.setattr(scheduler, "_trigger_live_forward_capture_on_lock", lambda *a, **k: None)
        mp.setattr(dm, "send_dm", lambda _r, txt: (sent_lim_dms.append(txt) or "dm-lim6"))
        refresh_lim = scheduler._refresh_pick_at_fallback_decision(cfg_lim, "2027-07-15", saved_57, 0.01)
        assert not scheduler._refresh_is_genuine_skip(refresh_lim)
        st_lim = scheduler.SchedulerState("2027-07-15", "2027-07-15T13:00:00", [], [], [], False, None, None, None)
        assert scheduler._deliver_and_lock_pick(
            refresh_lim.daily, cfg_lim, pdir_lim, st_lim, "2027-07-15", "final fallback", selection=refresh_lim.selection
        )
        assert len(sent_lim_dms) == 1 and "Batter 0" in sent_lim_dms[0]
        rec_lim6 = daily_decision.load_decision("2027-07-15", pdir_lim)
        assert (rec_lim6["objective"], rec_lim6["source"], rec_lim6["primary"]["batter_id"]) == (
            "reach57", "heuristic", 900
        )

    # --- Limit 7 (outside outer mp so real strategy._load_mdp runs!): unwritten FIFO at either path blocks ---
    for attr_target, attr_name in ((mdp, "DEFAULT_POLICY_PATH"), (tail_policy, "DEFAULT_TAIL_POLICY_PATH")):
        fifo_p = tmp_dir / f"fifo_{attr_name}"
        os.mkfifo(fifo_p)
        with pytest.MonkeyPatch.context() as mp_fifo:
            mp_fifo.setattr(attr_target, attr_name, fifo_p)
            mp_fifo.setattr(strategy, "_mdp_cache", None)
            old_h = signal.signal(
                signal.SIGALRM, lambda *_: (_ for _ in ()).throw(SimulatedCrash("blocked_on_fifo"))
            )
            try:
                signal.setitimer(signal.ITIMER_REAL, 0.12)
                with pytest.raises(SimulatedCrash, match="blocked_on_fifo"):
                    strategy._load_mdp()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
                signal.signal(signal.SIGALRM, old_h)

    # --- Sequence D: Rollback read & write under archived base 0a2bec10 ---
    base_dir = tmp_dir / "rollback_0a2bec10"
    base_dir.mkdir(parents=True, exist_ok=True)
    archive = subprocess.run(
        ["git", "archive", "0a2bec10", "src"], cwd=REPO, capture_output=True, check=True, timeout=15
    ).stdout
    subprocess.run(["tar", "-x", "-C", str(base_dir)], input=archive, check=True, timeout=15)

    rb_picks = tmp_dir / "rollback_picks"
    rb_picks.mkdir(parents=True, exist_ok=True)
    fin_rec = daily_decision.write_decision(
        "2027-09-25", rb_picks, action="double", source="mdp", objective="longest_streak",
        primary={"batter_id": 100, "batter_name": "Batter 0", "team": "T0", "game_pk": 70000, "p_game_hit": 0.96},
        double_down={"batter_id": 101, "batter_name": "Batter 1", "team": "T1", "game_pk": 70001, "p_game_hit": 0.95},
        delivery_status="delivered", scoreable=True, tail_policy_sha256=tail_sha,
    )
    assert fin_rec is not None
    uncomm_frame = walk._predictions(date(2027, 9, 26), 0.96)
    uncomm_daily = picks.DailyPick(
        "2027-09-26", "2027-09-26T16:00:00Z", picks.pick_from_row(uncomm_frame.iloc[0]),
        picks.pick_from_row(uncomm_frame.iloc[1]), None,
    )
    uncomm_daily.policy_decision = {
        "objective": "longest_streak", "source": "mdp", "action": "double", "policy_action": "double",
        "tail_sha256": tail_sha, "best_supplied": 57, "best_status": "trusted", "effective_best": None,
        "degraded_reason": None, "guard": None,
    }
    picks.save_pick(uncomm_daily, rb_picks)

    rb_code = """
import json, sys
from pathlib import Path
import bts
from bts import daily_decision, day_outcome, picks, scheduler

base_src = Path(sys.argv[1]).resolve()
pdir = Path(sys.argv[2]).resolve()
assert Path(bts.__file__).resolve() == base_src / "bts" / "__init__.py"

rec = daily_decision.load_decision("2027-09-25", pdir)
assert rec is not None
assert daily_decision.decision_objective(rec) == "unknown"
assert daily_decision.is_reach57_mdp_skip(rec) is False
assert daily_decision.is_scoreable_commit("2027-09-25", pdir, None) is True
out = day_outcome.classify_day(
    date="2027-09-25", games=5, decision=rec, decision_sha256="abc",
    receipts=[], intent="research", expected_username=None,
)
assert out.state == "delivered_under_research" and out.pick == "Batter 0 + Batter 1"

saved = picks.load_pick("2027-09-26", pdir)
assert saved is not None and saved.policy_decision["objective"] == "longest_streak"
st = scheduler.SchedulerState("2027-09-26", "2027-09-26T16:00:00", [], [], [], False, None, None, None)
scheduler._commit_decision_for_pick(pdir, "2027-09-26", saved, selection=None, delivery_status="delivered", state=st)
assert daily_decision.load_decision("2027-09-26", pdir) is None
print(json.dumps({"rollback_read_ok": True, "rollback_write_rejected_as_documented": True}))
"""
    env = os.environ | {"PYTHONPATH": str(base_dir / "src")}
    proc = subprocess.run(
        [sys.executable, "-c", rb_code, str(base_dir / "src"), str(rb_picks)],
        cwd=REPO, env=env, capture_output=True, text=True, check=True, timeout=30,
    )
    rb_res = json.loads(proc.stdout.strip().splitlines()[-1])
    assert rb_res["rollback_read_ok"] and rb_res["rollback_write_rejected_as_documented"]
    return {"standing_sequences_verified": 7, **rb_res}


def test_probe_4c_all(tmp_path):
    r1 = check_all_readers(tmp_path)
    r2 = check_standing_rows_and_ordinary_sequences(tmp_path)
    assert r1["readers_verified"] == 11
    assert r2["standing_sequences_verified"] == 7


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp_dir = SCRATCH / "p4c"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    summary = {
        "readers": check_all_readers(tmp_dir),
        "standing_rows": check_standing_rows_and_ordinary_sequences(tmp_dir),
    }
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    out_log = OUT_DIR / "probe_4_readers_and_standing_rows.log"
    out_log.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
