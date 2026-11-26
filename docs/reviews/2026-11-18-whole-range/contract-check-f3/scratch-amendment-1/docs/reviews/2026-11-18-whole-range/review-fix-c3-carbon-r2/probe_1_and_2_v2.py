"""Adversarial probe v2 for Items 1 and 2 across Base (c2039541), R1 (c70effe9), and R2 (aa61bb43).

Runs real scheduler, CLI, orchestrator, preview, check-results, and reconcile code paths
with synthetic state, pinned ET clocks, and network socket access blocked.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
ET = ZoneInfo("America/New_York")
ENTERED = "2027-04-18"
GRADED = "2027-04-19"
RUN_ET = datetime(2027, 4, 20, 2, 0, tzinfo=ET)
DELIVERY_ET = datetime(2027, 4, 18, 18, 25, tzinfo=ET)
FALLBACK_ET = datetime(2027, 4, 18, 18, 35, tzinfo=ET)


def _archive_src(rev: str, dest: Path) -> Path:
    raw = subprocess.check_output(["git", "archive", rev, "src"], cwd=REPO, timeout=20)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(raw)) as tf:
        tf.extractall(dest, filter="data")
    return dest / "src"


def _slot(day: str, game_pk: int = 101, p: float = 0.82) -> dict:
    return {
        "batter_name": "Shohei Ohtani",
        "batter_id": 660271,
        "team": "LAD",
        "lineup_position": 1,
        "pitcher_name": "Gerrit Cole",
        "pitcher_id": 543037,
        "p_game_hit": p,
        "flags": [],
        "projected_lineup": False,
        "game_pk": game_pk,
        "game_time": f"{day}T23:10:00Z",  # 19:10 ET
        "pitcher_team": "NYY",
    }


def _pick_dict(day: str, result: str | None = None, **extra) -> dict:
    base = {
        "date": day,
        "run_time": f"{day}T22:25:00+00:00",
        "pick": _slot(day),
        "double_down": None,
        "runner_up": None,
        "result": result,
    }
    if result in ("hit", "miss", "void"):
        base["slot_results"] = {"pick": result}
    base.update(extra)
    return base


def _seed_prior_hits_and_streak(picks_mod, root: Path, *, streak: int, saver: bool) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for offset in range(9):
        d = (date(2027, 4, 1) + timedelta(days=offset)).isoformat()
        daily = picks_mod.parse_daily_pick(json.dumps(_pick_dict(d, "hit", bluesky_posted=True)))
        picks_mod.save_pick(daily, root)
    sp = root / "streak.json"
    sp.write_text(json.dumps({
        "streak": streak,
        "saver_available": saver,
        "applied_dates": [],
        "updated": "2027-04-20T05:00:00+00:00",
        "season": 2027,
    }))
    return sp


def _invoke_reconcile(cli_mod, picks_mod, rr_mod, root: Path, *, feed_result: str = "miss"):
    from click.testing import CliRunner

    real_rr = rr_mod.ReconcileReceipt

    def _resolve(daily, day, **kwargs):
        return {"pick": feed_result}

    with patch.object(picks_mod, "resolve_daily_slot_results", _resolve), patch(
        "bts.reconcile_receipt.ReconcileReceipt",
        side_effect=lambda **kw: real_rr(clock=lambda: RUN_ET, lookback_days=kw["lookback_days"]),
    ):
        res = CliRunner().invoke(cli_mod.cli, ["reconcile", "--picks-dir", str(root)])
    recs = []
    rroot = rr_mod.receipts_root(root)
    if rroot.exists():
        for ddir in sorted(rroot.iterdir()):
            if ddir.is_dir():
                recs.extend(rr_mod.discover(root, ddir.name))
    return res, (recs[-1]["replay"] if recs else None)


def _invoke_check_results(cli_mod, picks_mod, root: Path, day: str, *, hit_val: str = "hit"):
    from click.testing import CliRunner

    with patch.object(cli_mod, "_now_et", return_value=RUN_ET), patch.object(
        picks_mod, "get_game_statuses_detailed", lambda *a, **k: {}
    ), patch.object(picks_mod, "check_hit", lambda *a, **k: hit_val), patch.object(
        picks_mod, "resolve_daily_slot_results", lambda *a, **k: {"pick": hit_val}
    ):
        return CliRunner().invoke(cli_mod.cli, ["check-results", "--date", day, "--picks-dir", str(root)])


def _build_scenario(name: str, root: Path, modules: dict) -> dict:
    """Build on-disk state using REAL producer functions."""
    import pandas as pd

    cli_mod = modules["cli"]
    contest_state = modules["contest_state"]
    daily_decision = modules["daily_decision"]
    dm_mod = modules["dm"]
    entry_intent = modules["entry_intent"]
    orchestrator = modules["orchestrator"]
    picks_mod = modules["picks"]
    posting = modules["posting"]
    predict = modules["predict"]
    scheduler = modules["scheduler"]
    strategy = modules["strategy"]
    from click.testing import CliRunner

    sp = _seed_prior_hits_and_streak(picks_mod, root, streak=9, saver=True)
    entered_evidence = []

    auto_obs = contest_state.DecisionStreakState(
        9, True, True, "contest", "fresh", 9, True,
        observation_kind="automatic",
        contest_recorded_at="2027-04-18T15:00:00+00:00",
    )
    dated_obs = contest_state.DecisionStreakState(
        9, True, True, "contest", "fresh", 9, True,
        observation_kind="dated",
        dated_identity={
            "effective_date": ENTERED,
            "active_streak": 9,
            "best_streak": 9,
            "saver_state": "not_earned",
        },
    )
    model_obs = contest_state.DecisionStreakState(9, True, True, "model", "fresh", 0, True)

    hi_df = pd.DataFrame([{**_slot(ENTERED, 101, 0.84), "lineup": 1}])
    lo_df = pd.DataFrame([{**_slot(ENTERED, 101, 0.55), "lineup": 1}])
    detailed_pre = {101: {"abstract": "P", "detailed": "Scheduled"}}
    tiers = [{"name": "local", "type": "local"}]

    if name in ("codex_lost_ack", "codex_skip_conflict", "codex_private_crash"):
        local = picks_mod.parse_daily_pick(json.dumps(_pick_dict(ENTERED)))
        st = scheduler.SchedulerState(ENTERED, DELIVERY_ET.isoformat(), [], [], [], False, None, None, None)
        is_priv = name == "codex_private_crash"
        cfg = {
            "scheduler": {"pick_delivery": "private" if is_priv else "dm"},
            "bluesky": {"dm_recipient": "synthetic.invalid"},
        }
        if name == "codex_skip_conflict":
            st.final_skip_candidate = {"source": "mdp", "primary": {"batter_id": 660271}}

        def _send(rec, body):
            entered_evidence.append("dm_accepted_and_entered")
            if name == "codex_lost_ack":
                raise TimeoutError("lost ack after remote accept")
            return "dm-id-1"

        def _cap(*_a):
            if is_priv:
                entered_evidence.append("private_locked_and_entered")
                raise KeyboardInterrupt()

        with patch.object(scheduler, "_now_et", return_value=DELIVERY_ET), patch.object(
            scheduler, "_void_committed_games", return_value={}
        ), patch.object(scheduler, "_trigger_live_forward_capture_on_lock", _cap), patch.object(
            contest_state, "load_decision_streak_state", return_value=model_obs
        ), patch.object(dm_mod, "send_dm", _send), patch.object(
            daily_decision, "write_decision", return_value=None
        ):
            try:
                scheduler._deliver_and_lock_pick(local, cfg, root, st, ENTERED, "fallback")
            except KeyboardInterrupt:
                pass
        if name == "codex_skip_conflict":
            scheduler._write_endofday_skip(root, ENTERED, st)

    elif name == "codex_manual_run_crash":
        daily = picks_mod.parse_daily_pick(json.dumps(_pick_dict(ENTERED)))
        selected = SimpleNamespace(pick_result=SimpleNamespace(daily=daily, locked=False), decision=True)

        def _post(body):
            entered_evidence.append("post_accepted_and_entered")
            raise KeyboardInterrupt()

        with patch.object(contest_state, "DecisionStreakState", return_value=auto_obs), patch.object(
            predict, "run_pipeline", return_value=hi_df
        ), patch.object(contest_state, "load_decision_streak_state", return_value=auto_obs), patch.object(
            picks_mod, "get_game_statuses_detailed", return_value=detailed_pre
        ), patch.object(picks_mod, "attach_provenance"), patch.object(
            strategy, "select_pick", return_value=selected
        ), patch.object(posting, "should_post_now", return_value=True), patch.object(
            posting, "post_to_bluesky", _post
        ):
            CliRunner().invoke(cli_mod.cli, ["run", "--date", ENTERED, "--picks-dir", str(root)])

    elif name in ("N1_lost_ack_then_fallback_skip_auto", "N2_lost_ack_then_fallback_skip_dated"):
        obs = auto_obs if name.endswith("_auto") else dated_obs
        cfg = {
            "orchestrator": {
                "picks_dir": str(root),
                "models_dir": str(root.parent / "models"),
                "heartbeat_path": str(root.parent / ".hb"),
            },
            "tiers": tiers,
            "scheduler": {
                "pick_delivery": "dm",
                "entry_intent": "enter",
                "contest_state_required": True,
                "live_forward_capture_on_lock": False,
            },
            "health_checks": {"contest_state_expected": True, "enabled": False},
            "bluesky": {"dm_recipient": "operator.invalid"},
        }
        assert entry_intent.check_entry_intent(cfg, include_live_gate=True).ok

        st = scheduler.SchedulerState(
            ENTERED, DELIVERY_ET.isoformat(),
            [{"game_pk": 101, "game_time_et": "2027-04-18T19:10:00-04:00",
              "lineup_confirmed": True, "is_doubleheader_game2": False}],
            [101], [], False, None, None, None,
        )
        scheduler.save_state(st, root)

        def _send_timeout(rec, body):
            entered_evidence.append("dm_accepted_and_entered")
            raise TimeoutError("lost HTTP response after DM accepted")

        with patch.object(scheduler, "_now_et", return_value=DELIVERY_ET), patch.object(
            scheduler, "count_new_confirmations", return_value=1
        ), patch.object(orchestrator, "run_cascade", return_value=(hi_df, "local")), patch.object(
            contest_state, "load_decision_streak_state", return_value=obs
        ), patch.object(picks_mod, "get_game_statuses_detailed", return_value=detailed_pre), patch.object(
            picks_mod, "attach_provenance"
        ), patch.object(scheduler, "_void_committed_games", return_value={}), patch.object(
            dm_mod, "send_dm", _send_timeout
        ):
            chk = scheduler.run_single_check(ENTERED, [101], {(101, "away"), (101, "home")}, cfg, 0.03, set())
            assert chk["should_post"] is True and chk["pick_result"] is not None
            ok = scheduler._deliver_and_lock_pick(
                chk["pick_result"].daily, cfg, root, st, ENTERED, "lineup", selection=chk["selection"]
            )
            assert ok is False and st.pick_locked is False

        with patch.object(scheduler, "_now_et", return_value=FALLBACK_ET), patch.object(
            orchestrator, "run_cascade", return_value=(lo_df, "local")
        ), patch.object(contest_state, "load_decision_streak_state", return_value=obs), patch.object(
            picks_mod, "get_game_statuses_detailed", return_value=detailed_pre
        ):
            cached = picks_mod.load_pick(ENTERED, root)
            refresh = scheduler._refresh_pick_at_fallback_decision(cfg, ENTERED, cached, 0.03, set())
            assert scheduler._refresh_is_genuine_skip(refresh)
            scheduler._capture_fallback_skip(st, refresh)
            scheduler.save_state(st, root)

        scheduler._write_endofday_skip(root, ENTERED, st)

    elif name == "N3_manual_run_crash_with_daemon_eod_skip":
        st = scheduler.SchedulerState(
            ENTERED, "2027-04-18T12:25:00-04:00", [], [], [], False, None, None, None,
            final_skip_candidate={"source": "mdp", "primary": {"batter_id": 660271}},
        )
        scheduler.save_state(st, root)

        def _post(body):
            entered_evidence.append("manual_post_accepted_and_entered")
            raise KeyboardInterrupt()

        with patch.object(predict, "run_pipeline", return_value=hi_df), patch.object(
            contest_state, "load_decision_streak_state", return_value=auto_obs
        ), patch.object(picks_mod, "get_game_statuses_detailed", return_value=detailed_pre), patch.object(
            picks_mod, "attach_provenance"
        ), patch.object(posting, "should_post_now", return_value=True), patch.object(
            posting, "post_to_bluesky", _post
        ):
            r = CliRunner().invoke(cli_mod.cli, ["run", "--date", ENTERED, "--picks-dir", str(root)])
            assert r.exit_code == 1

        scheduler._write_endofday_skip(root, ENTERED, st)

    elif name in ("N4_live_private_preview_lock_restart", "N6_live_private_preview_lock_quarantined"):
        with patch.object(predict, "run_pipeline", return_value=hi_df), patch.object(
            contest_state, "load_decision_streak_state", return_value=model_obs
        ), patch.object(picks_mod, "get_game_statuses_detailed", return_value=detailed_pre), patch.object(
            picks_mod, "attach_provenance"
        ):
            pr = CliRunner().invoke(
                cli_mod.cli,
                ["preview", "--date", ENTERED, "--picks-dir", str(root), "--models-dir", str(root.parent / "m")],
            )
            assert pr.exit_code == 0, pr.output
        assert picks_mod.load_pick(ENTERED, root).chosen_on["required"] is False

        cfg = {
            "orchestrator": {
                "picks_dir": str(root),
                "models_dir": str(root.parent / "m"),
                "heartbeat_path": str(root.parent / ".hb"),
            },
            "tiers": tiers,
            "scheduler": {
                "pick_delivery": "private",
                "entry_intent": "research",
                "contest_state_required": True,
                "live_forward_capture_on_lock": True,
            },
            "health_checks": {"contest_state_expected": True, "enabled": False},
        }
        assert entry_intent.check_entry_intent(cfg, include_live_gate=True).ok

        st = scheduler.SchedulerState(ENTERED, DELIVERY_ET.isoformat(), [], [], [], False, None, None, None)

        def _cap_crash(*_a):
            assert scheduler.load_state(ENTERED, root).pick_locked is True
            entered_evidence.append("private_preview_locked_and_entered")
            raise KeyboardInterrupt()

        with patch.object(scheduler, "_now_et", return_value=FALLBACK_ET), patch.object(
            orchestrator, "run_cascade", return_value=(None, None)
        ), patch.object(scheduler, "_void_committed_games", return_value={}), patch.object(
            scheduler, "_trigger_live_forward_capture_on_lock", _cap_crash
        ):
            cached = picks_mod.load_pick(ENTERED, root)
            refresh = scheduler._refresh_pick_at_fallback_decision(cfg, ENTERED, cached, 0.03, set())
            assert refresh.daily.chosen_on["required"] is False
            try:
                scheduler._deliver_and_lock_pick(
                    refresh.daily, cfg, root, st, ENTERED, "final fallback", selection=refresh.selection
                )
            except KeyboardInterrupt:
                pass

        assert daily_decision.load_decision(ENTERED, root) is None
        assert picks_mod.load_pick(ENTERED, root).chosen_on["required"] is False

        if name == "N4_live_private_preview_lock_restart":
            sched_games = [{
                "gamePk": 101,
                "gameDate": f"{ENTERED}T23:10:00Z",
                "teams": {"away": {"team": {"name": "LAD"}}, "home": {"team": {"name": "NYY"}}},
            }]
            restart_et = datetime(2027, 4, 18, 18, 36, tzinfo=ET)
            with patch.object(scheduler, "_now_et", return_value=restart_et), patch.object(
                scheduler, "fetch_schedule", side_effect=lambda d: sched_games if d == ENTERED else []
            ), patch.object(scheduler, "count_new_confirmations", return_value=0), patch.object(
                orchestrator, "run_cascade", return_value=(None, None)
            ), patch.object(
                contest_state, "load_decision_streak_state",
                side_effect=contest_state.ContestStateError("automatic fetch stale on restart"),
            ), patch.object(scheduler, "_alert_contest_state_failure"), patch.object(
                scheduler, "_maybe_alert_missed_pick"
            ), patch.object(scheduler, "_idle_until_next_wakeup"), patch.object(
                scheduler, "notify_ready"
            ), patch.object(scheduler, "notify_watchdog"):
                scheduler.run_day(ENTERED, cfg)
            assert scheduler.load_state(ENTERED, root).pick_locked is False
        else:
            (root / ENTERED / "scheduler_state.json").write_text("{torn-write")
            assert scheduler.load_state(ENTERED, root) is None
            assert list((root / ENTERED).glob("scheduler_state.json.corrupt-*"))
            assert not (root / ENTERED / "scheduler_state.json").exists()

    elif name == "N5_private_research_fresh_pick_restart":
        cfg = {
            "orchestrator": {
                "picks_dir": str(root),
                "models_dir": str(root.parent / "m"),
                "heartbeat_path": str(root.parent / ".hb"),
            },
            "tiers": tiers,
            "scheduler": {
                "pick_delivery": "private",
                "entry_intent": "research",
                "contest_state_required": False,
                "live_forward_capture_on_lock": True,
            },
            "health_checks": {"contest_state_expected": False, "enabled": False},
        }
        assert entry_intent.check_entry_intent(cfg, include_live_gate=True).ok

        st = scheduler.SchedulerState(ENTERED, DELIVERY_ET.isoformat(), [], [], [], False, None, None, None)

        def _cap_crash(*_a):
            assert scheduler.load_state(ENTERED, root).pick_locked is True
            entered_evidence.append("private_fresh_locked_and_entered")
            raise KeyboardInterrupt()

        with patch.object(scheduler, "_now_et", return_value=DELIVERY_ET), patch.object(
            scheduler, "count_new_confirmations", return_value=1
        ), patch.object(orchestrator, "run_cascade", return_value=(hi_df, "local")), patch.object(
            contest_state, "load_decision_streak_state", return_value=model_obs
        ), patch.object(picks_mod, "get_game_statuses_detailed", return_value=detailed_pre), patch.object(
            picks_mod, "attach_provenance"
        ), patch.object(scheduler, "_void_committed_games", return_value={}), patch.object(
            scheduler, "_trigger_live_forward_capture_on_lock", _cap_crash
        ):
            chk = scheduler.run_single_check(ENTERED, [101], {(101, "away"), (101, "home")}, cfg, 0.03, set())
            assert chk["pick_result"].daily.chosen_on["required"] is False
            try:
                scheduler._deliver_and_lock_pick(
                    chk["pick_result"].daily, cfg, root, st, ENTERED, "lineup", selection=chk["selection"]
                )
            except KeyboardInterrupt:
                pass

        sched_games = [{
            "gamePk": 101,
            "gameDate": f"{ENTERED}T23:10:00Z",
            "teams": {"away": {"team": {"name": "LAD"}}, "home": {"team": {"name": "NYY"}}},
        }]
        with patch.object(scheduler, "_now_et", return_value=datetime(2027, 4, 18, 18, 20, tzinfo=ET)), patch.object(
            scheduler, "fetch_schedule", return_value=sched_games
        ), patch.object(scheduler, "_watchdog_ping_sleep", side_effect=SystemExit(143)), patch.object(
            scheduler, "notify_ready"
        ), patch.object(scheduler, "notify_watchdog"):
            try:
                scheduler.run_day(ENTERED, cfg)
            except SystemExit:
                pass
        assert scheduler.load_state(ENTERED, root).pick_locked is False

    elif name == "N7_orchestrate_not_live_post_crash":
        cfg = {
            "orchestrator": {"picks_dir": str(root), "models_dir": str(root.parent / "m")},
            "tiers": tiers,
            "bluesky": {"dm_recipient": "operator.invalid"},
        }

        def _post(body):
            entered_evidence.append("orchestrate_post_accepted_and_entered")
            raise KeyboardInterrupt()

        with patch.object(orchestrator, "run_cascade", return_value=(hi_df, "local")), patch.object(
            contest_state, "load_decision_streak_state", return_value=model_obs
        ), patch.object(picks_mod, "get_game_statuses_detailed", return_value=detailed_pre), patch.object(
            picks_mod, "attach_provenance"
        ), patch.object(posting, "should_post_now", return_value=True), patch.object(
            posting, "post_to_bluesky", _post
        ):
            try:
                orchestrator.orchestrate(Path("unused.toml"), ENTERED, config=cfg)
            except KeyboardInterrupt:
                pass

    elif name == "N8_preview_overwrites_unmarked_entered_pick":
        daily = picks_mod.parse_daily_pick(json.dumps(_pick_dict(
            ENTERED, chosen_on=picks_mod.chosen_on_record(auto_obs, required=True)
        )))
        picks_mod.save_pick(daily, root)
        entered_evidence.append("dm_lost_ack_entered_then_preview_overwrote")
        with patch.object(predict, "run_pipeline", return_value=hi_df), patch.object(
            contest_state, "load_decision_streak_state", return_value=model_obs
        ), patch.object(picks_mod, "get_game_statuses_detailed", return_value=detailed_pre), patch.object(
            picks_mod, "attach_provenance"
        ):
            pr = CliRunner().invoke(
                cli_mod.cli,
                ["preview", "--date", ENTERED, "--picks-dir", str(root), "--models-dir", str(root.parent / "m")],
            )
            assert pr.exit_code == 0

    else:
        raise ValueError(f"Unknown scenario {name}")

    chk_entered = _invoke_check_results(cli_mod, picks_mod, root, ENTERED, hit_val="hit")
    assert chk_entered.exit_code == 0 and "not scoring" in chk_entered.stdout, chk_entered.output
    assert picks_mod.load_pick(ENTERED, root).result is None

    p19 = picks_mod.parse_daily_pick(json.dumps(_pick_dict(GRADED, None, bluesky_posted=True)))
    picks_mod.save_pick(p19, root)
    chk_19 = _invoke_check_results(cli_mod, picks_mod, root, GRADED, hit_val="hit")
    assert chk_19.exit_code == 0 and "Streak: 10" in chk_19.stdout, chk_19.output

    before_bytes = sp.read_bytes()
    res, rep = _invoke_reconcile(cli_mod, picks_mod, modules["rr"], root, feed_result="miss")
    after_overturn_bytes = sp.read_bytes()
    after_data = json.loads(sp.read_text())
    p18_raw = json.loads((root / f"{ENTERED}.json").read_text())
    dec18_path = root / ENTERED / "decision.json"
    st18_path = root / ENTERED / "scheduler_state.json"

    # Also test starting from ALREADY-RIGHT streak.json = (10, False)
    sp.write_text(json.dumps({
        "streak": 10,
        "saver_available": False,
        "applied_dates": [GRADED],
        "updated": "2027-04-20T01:05:00+00:00",
        "season": 2027,
    }))
    res_true, _ = _invoke_reconcile(cli_mod, picks_mod, modules["rr"], root, feed_result="miss")
    after_true = json.loads(sp.read_text())

    return {
        "evidence": entered_evidence,
        "pick_chosen_on_required": (p18_raw.get("chosen_on") or {}).get("required"),
        "pick_has_auto_reading": p18_raw.get("automatic_reading") is not None,
        "decision_scoreable": json.loads(dec18_path.read_text())["scoreable"] if dec18_path.exists() else None,
        "sched_state_locked": json.loads(st18_path.read_text())["pick_locked"] if st18_path.exists() else None,
        "quarantine_files": [p.name for p in (root / ENTERED).glob("scheduler_state.json.corrupt-*")]
        if (root / ENTERED).exists() else [],
        "exit_code": res.exit_code,
        "stderr": res.stderr.strip(),
        "stdout": res.stdout.strip(),
        "bytes_preserved": after_overturn_bytes == before_bytes,
        "after_overturn_streak_saver": [after_data["streak"], after_data["saver_available"]],
        "after_true_10_false_streak_saver": [after_true["streak"], after_true["saver_available"]],
        "true_fold": [10, False],
        "replay_receipt": rep,
    }


def _run_carbon_dir_states(modules: dict, scratch: Path) -> dict:
    cli_mod = modules["cli"]
    daily_decision = modules["daily_decision"]
    picks_mod = modules["picks"]
    rr_mod = modules["rr"]
    out = {}

    def _plant(root: Path, day: str, result: str | None = None, *, delivered: bool = True, req_true: bool = False):
        extra = {}
        if delivered:
            extra.update(notification_sent=True, notification_id=f"dm-{day}", delivery_attempted=True)
        if req_true:
            obs = modules["contest_state"].DecisionStreakState(
                9, True, True, "contest", "fresh", 9, True, observation_kind="automatic"
            )
            extra["chosen_on"] = picks_mod.chosen_on_record(obs, required=True)
        d = picks_mod.parse_daily_pick(json.dumps(_pick_dict(day, result, **extra)))
        return picks_mod.save_pick(d, root)

    # (a) chmod 000 on 2027-04-16/ with scoreable decision and missing 2027-04-16.json
    p1 = scratch / "c_a" / "picks"
    p1.mkdir(parents=True)
    _plant(p1, "2027-04-01", "hit")
    daily_decision.write_decision(
        "2027-04-16", p1, action="single", source="mdp", delivery_status="delivered", scoreable=True, now=RUN_ET
    )
    picks_mod.save_streak(9, p1, saver_available=False, season=2027)
    os.chmod(p1 / "2027-04-16", 0o000)
    try:
        r1, _ = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p1, feed_result="hit")
        st1 = json.loads((p1 / "streak.json").read_text())
        out["a_chmod000_decision_dir_no_pick"] = {
            "exit_code": r1.exit_code,
            "stderr": r1.stderr.strip(),
            "streak_after": [st1["streak"], st1["saver_available"]],
        }
    finally:
        os.chmod(p1 / "2027-04-16", 0o700)

    # (b1) symlink decision dir over DELIVERED pending pick
    ext = scratch / "ext_skip"
    daily_decision.write_decision(
        "2027-04-16", ext, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN_ET
    )
    p2 = scratch / "c_b1" / "picks"
    p2.mkdir(parents=True)
    _plant(p2, "2027-04-01", "hit")
    _plant(p2, "2027-04-16", None, delivered=True)
    (p2 / "2027-04-16").symlink_to(ext / "2027-04-16")
    picks_mod.save_streak(9, p2, saver_available=False, season=2027)
    r2, _ = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p2, feed_result="hit")
    st2 = json.loads((p2 / "streak.json").read_text())
    out["b1_symlink_skip_dir_over_delivered_pick"] = {
        "exit_code": r2.exit_code,
        "stderr": r2.stderr.strip(),
        "streak_after": [st2["streak"], st2["saver_available"]],
    }

    # (b2) symlink decision dir over UNMARKED live pick (chosen_on.required=True, no delivery flag)
    p2b = scratch / "c_b2" / "picks"
    p2b.mkdir(parents=True)
    _plant(p2b, "2027-04-01", "hit")
    _plant(p2b, "2027-04-16", None, delivered=False, req_true=True)
    (p2b / "2027-04-16").symlink_to(ext / "2027-04-16")
    picks_mod.save_streak(9, p2b, saver_available=False, season=2027)
    r2b, _ = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p2b, feed_result="hit")
    st2b = json.loads((p2b / "streak.json").read_text())
    out["b2_symlink_skip_dir_over_unmarked_live_pick"] = {
        "exit_code": r2b.exit_code,
        "stderr": r2b.stderr.strip(),
        "streak_after": [st2b["streak"], st2b["saver_available"]],
    }

    # (b3) FIFO at 2027-04-16 without root pick, and with root pick
    p3 = scratch / "c_b3" / "picks"
    p3.mkdir(parents=True)
    _plant(p3, "2027-04-01", "hit")
    os.mkfifo(p3 / "2027-04-16")
    picks_mod.save_streak(9, p3, saver_available=False, season=2027)
    r3, _ = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p3, feed_result="hit")
    st3 = json.loads((p3 / "streak.json").read_text())
    out["b3_fifo_date_dir_no_pick"] = {
        "exit_code": r3.exit_code,
        "stderr": r3.stderr.strip(),
        "streak_after": [st3["streak"], st3["saver_available"]],
    }

    p3b = scratch / "c_b3b" / "picks"
    p3b.mkdir(parents=True)
    _plant(p3b, "2027-04-01", "hit")
    _plant(p3b, "2027-04-16", None, delivered=False)
    os.mkfifo(p3b / "2027-04-16")
    picks_mod.save_streak(9, p3b, saver_available=False, season=2027)
    r3b, _ = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p3b, feed_result="hit")
    st3b = json.loads((p3b / "streak.json").read_text())
    out["b3b_fifo_date_dir_with_pick"] = {
        "exit_code": r3b.exit_code,
        "stderr": r3b.stderr.strip(),
        "streak_after": [st3b["streak"], st3b["saver_available"]],
    }

    # (c) chmod 0300 on picks_dir (unreadable directory listing)
    p4 = scratch / "c_c"
    p4.mkdir()
    for i in range(1, 12):
        _plant(p4, f"2027-04-{i:02d}", "miss" if i == 11 else "hit")
    picks_mod.save_streak(10, p4, saver_available=False, season=2027)
    os.chmod(p4, 0o300)
    try:
        try:
            picks_mod.reconcile_results(p4, clock=lambda: RUN_ET)
            raised_lib = None
        except Exception as exc:  # noqa: BLE001
            raised_lib = type(exc).__name__
        st4 = json.loads((p4 / "streak.json").read_text())
        out["c_chmod0300_picks_dir_lib"] = {
            "raised": raised_lib,
            "streak_after": [st4["streak"], st4["saver_available"]],
        }
    finally:
        os.chmod(p4, 0o700)

    p4_cli = scratch / "c_c_cli"
    p4_cli.mkdir()
    for i in range(1, 12):
        _plant(p4_cli, f"2027-04-{i:02d}", "miss" if i == 11 else "hit")
    picks_mod.save_streak(10, p4_cli, saver_available=False, season=2027)
    os.chmod(p4_cli, 0o300)
    try:
        r4c, _ = _invoke_reconcile(cli_mod, picks_mod, rr_mod, p4_cli, feed_result="hit")
        st4c = json.loads((p4_cli / "streak.json").read_text())
        out["c_chmod0300_picks_dir_cli"] = {
            "exit_code": r4c.exit_code,
            "stderr": r4c.stderr.strip(),
            "streak_after": [st4c["streak"], st4c["saver_available"]],
        }
    finally:
        os.chmod(p4_cli, 0o700)

    return out


def _worker(src_dir: Path) -> int:
    sys.path.insert(0, str(src_dir))
    import bts.cli as cli_mod
    import bts.contest_state as contest_state
    import bts.daily_decision as daily_decision
    import bts.dm as dm_mod
    import bts.entry_intent as entry_intent
    from bts.model import predict
    import bts.orchestrator as orchestrator
    import bts.picks as picks_mod
    import bts.posting as posting
    import bts.reconcile_receipt as rr_mod
    import bts.scheduler as scheduler
    import bts.strategy as strategy

    def _deny_net(*args, **kwargs):
        raise AssertionError(f"Unexpected network access: {args}")

    socket.socket.connect = _deny_net  # type: ignore[method-assign]
    socket.getaddrinfo = _deny_net  # type: ignore[assignment]

    mods = {
        "cli": cli_mod,
        "contest_state": contest_state,
        "daily_decision": daily_decision,
        "dm": dm_mod,
        "entry_intent": entry_intent,
        "orchestrator": orchestrator,
        "picks": picks_mod,
        "posting": posting,
        "predict": predict,
        "rr": rr_mod,
        "scheduler": scheduler,
        "strategy": strategy,
    }

    scenarios = [
        "codex_lost_ack",
        "codex_skip_conflict",
        "codex_private_crash",
        "codex_manual_run_crash",
        "N1_lost_ack_then_fallback_skip_auto",
        "N2_lost_ack_then_fallback_skip_dated",
        "N3_manual_run_crash_with_daemon_eod_skip",
        "N4_live_private_preview_lock_restart",
        "N5_private_research_fresh_pick_restart",
        "N6_live_private_preview_lock_quarantined",
        "N7_orchestrate_not_live_post_crash",
        "N8_preview_overwrites_unmarked_entered_pick",
    ]
    out = {"file": picks_mod.__file__, "scenarios": {}, "carbon_dir_states": {}}
    orig_cwd = Path.cwd()
    with TemporaryDirectory(dir=HERE, prefix="_w_item12_") as td:
        tpath = Path(td)
        try:
            os.chdir(tpath)
            for sc in scenarios:
                out["scenarios"][sc] = _build_scenario(sc, tpath / sc / "picks", mods)
            out["carbon_dir_states"] = _run_carbon_dir_states(mods, tpath / "carbon")
        finally:
            os.chdir(orig_cwd)
    print(json.dumps(out))
    return 0


def main() -> int:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        return _worker(Path(sys.argv[2]))

    report = {}
    with TemporaryDirectory(dir=HERE, prefix="_arch_item12_") as td:
        tpath = Path(td)
        revs = {
            "base_c2039541": _archive_src("c2039541", tpath / "c2039541"),
            "r1_c70effe9": _archive_src("c70effe9", tpath / "c70effe9"),
            "r2_aa61bb43": REPO / "src",
        }
        for label, src in revs.items():
            env = os.environ | {
                "PYTHONPATH": str(src),
                "PYTHONDONTWRITEBYTECODE": "1",
                "TZ": "America/New_York",
                "MPLCONFIGDIR": "/tmp",
            }
            proc = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--worker", str(src)],
                cwd=REPO,
                env=env,
                text=True,
                capture_output=True,
                timeout=90,
            )
            if proc.returncode != 0:
                sys.stderr.write(f"Worker {label} failed:\n{proc.stdout}\n{proc.stderr}\n")
                return proc.returncode
            report[label] = json.loads(proc.stdout)

    out_file = HERE / "item1_and_2_results.json"
    out_file.write_text(json.dumps(report, indent=2) + "\n")

    print("=== SCENARIO SUMMARY (true fold = [10, False]) ===")
    for sc in report["r2_aa61bb43"]["scenarios"]:
        b = report["base_c2039541"]["scenarios"][sc]
        r1 = report["r1_c70effe9"]["scenarios"][sc]
        r2 = report["r2_aa61bb43"]["scenarios"][sc]
        print(
            f"{sc:44s} | Base: {str(b['after_overturn_streak_saver']):12s} (true10F->{b['after_true_10_false_streak_saver']}) "
            f"| R1: {str(r1['after_overturn_streak_saver']):12s} "
            f"| R2: {str(r2['after_overturn_streak_saver']):12s} (true10F->{r2['after_true_10_false_streak_saver']}) "
            f"| R2 stderr: {r2['stderr']!r}"
        )

    print("\n=== CARBON DIR STATES SUMMARY ===")
    for k in report["r2_aa61bb43"]["carbon_dir_states"]:
        b = report["base_c2039541"]["carbon_dir_states"][k]
        r1 = report["r1_c70effe9"]["carbon_dir_states"][k]
        r2 = report["r2_aa61bb43"]["carbon_dir_states"][k]
        print(
            f"{k:46s} | Base: {b['streak_after']} | R1: {r1['streak_after']} | R2: {r2['streak_after']} "
            f"| R2 err: {r2.get('stderr', r2.get('raised'))!r}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
