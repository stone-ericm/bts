"""Carbon's C3 round-two writer sequences, checked against the C3v base replay."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date, datetime, timedelta
import json
from pathlib import Path
import signal
import socket
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pytest

from tests.test_wr_c3v_reconcile import PENDING, base_modules, compare_reconcile, only_receipt, warning


ET = ZoneInfo("America/New_York")
ENTERED = "2027-04-18"
GRADED = "2027-04-19"
DELIVERY_ET = datetime(2027, 4, 18, 18, 25, tzinfo=ET)
FALLBACK_ET = datetime(2027, 4, 18, 18, 35, tzinfo=ET)
RUN_ET = datetime(2027, 4, 20, 2, 0, tzinfo=ET)


@contextmanager
def _deadline(seconds: int = 5):
    class _TimedOut(BaseException):
        pass

    prev = signal.signal(signal.SIGALRM, lambda *_a: (_ for _ in ()).throw(_TimedOut()))
    signal.alarm(seconds)
    try:
        yield
    except _TimedOut:
        pytest.fail("Timed out in entered-pick blocker test")
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, prev)


def _slot(day: str, game_pk: int = 101, p: float = 0.84) -> dict:
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
        "game_time": f"{day}T23:10:00Z",
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


def _seed_nine_hits(picks_mod, root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for offset in range(9):
        d = (date(2027, 4, 1) + timedelta(days=offset)).isoformat()
        picks_mod.save_pick(picks_mod.parse_daily_pick(json.dumps(_pick_dict(d, "hit", bluesky_posted=True))), root)
    sp = root / "streak.json"
    sp.write_text(json.dumps({
        "streak": 9,
        "saver_available": True,
        "applied_dates": [],
        "updated": "2027-04-20T05:00:00+00:00",
        "season": 2027,
    }))
    return sp


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail(f"Unexpected network access: {args}")

    monkeypatch.setattr(socket.socket, "connect", unexpected)
    monkeypatch.setattr(socket, "getaddrinfo", unexpected)


@pytest.mark.parametrize("scenario", [
    "lost_ack_then_fallback_skip_auto",
    "lost_ack_then_fallback_skip_dated",
    "manual_run_crash_with_daemon_eod_skip",
    "live_private_preview_lock_restart",
    "private_research_fresh_pick_restart",
    "live_private_preview_lock_quarantined",
    "orchestrate_not_live_post_crash",
    "preview_overwrites_unmarked_entered_pick",
], ids=[f"N{i}" for i in range(1, 9)])
def test_entered_pick_never_skipped_by_season_replay(tmp_path, monkeypatch, base_modules, scenario):
    try:
        import pandas as pd
    except (ImportError, OSError):
        pytest.skip("pandas unavailable")

    from bts import cli, contest_state, dm, entry_intent, orchestrator, picks, posting, scheduler, strategy
    from bts.model import predict

    repo_root = Path(__file__).resolve().parents[1]
    work = tmp_path / "work"
    (work / "data").mkdir(parents=True)
    (work / "data" / "models").symlink_to(repo_root / "data" / "models")
    monkeypatch.chdir(work)
    monkeypatch.setattr(strategy, "_mdp_cache", None)

    root = work / "picks"
    sp = _seed_nine_hits(picks, root)

    auto_obs = contest_state.DecisionStreakState(
        9, True, True, "contest", "fresh", 9, True,
        observation_kind="automatic", contest_recorded_at="2027-04-18T15:00:00+00:00",
    )
    dated_obs = contest_state.DecisionStreakState(
        9, True, True, "contest", "fresh", 9, True,
        observation_kind="dated",
        dated_identity={
            "effective_date": ENTERED, "active_streak": 9, "best_streak": 9, "saver_state": "not_earned",
        },
    )
    model_obs = contest_state.DecisionStreakState(9, True, True, "model", "fresh", 0, True)
    hi_df = pd.DataFrame([{**_slot(ENTERED, 101, 0.84), "lineup": 1}])
    lo_df = pd.DataFrame([{**_slot(ENTERED, 101, 0.55), "lineup": 1}])
    detailed_pre = {101: {"abstract": "P", "detailed": "Scheduled"}}
    tiers = [{"name": "local", "type": "local"}]

    with _deadline():
        if scenario in ("lost_ack_then_fallback_skip_auto", "lost_ack_then_fallback_skip_dated"):
            obs = auto_obs if scenario.endswith("_auto") else dated_obs
            cfg = {
                "orchestrator": {"picks_dir": str(root), "models_dir": str(repo_root / "data" / "models"),
                                 "heartbeat_path": str(work / ".hb")},
                "tiers": tiers,
                "scheduler": {"pick_delivery": "dm", "entry_intent": "enter", "contest_state_required": True,
                              "live_forward_capture_on_lock": False},
                "health_checks": {"contest_state_expected": True, "enabled": False},
                "bluesky": {"dm_recipient": "operator.invalid"},
            }
            assert entry_intent.check_entry_intent(cfg, include_live_gate=True).ok
            st = scheduler.SchedulerState(ENTERED, DELIVERY_ET.isoformat(), [], [], [], False, None, None, None)

            def _send_timeout(rec, body):
                raise TimeoutError("lost response after DM accepted and entered")

            with monkeypatch.context() as mp:
                mp.setattr(scheduler, "_now_et", lambda: DELIVERY_ET)
                mp.setattr(scheduler, "count_new_confirmations", lambda *a: 1)
                mp.setattr(orchestrator, "run_cascade", lambda *a, **k: (hi_df, "local"))
                mp.setattr(contest_state, "load_decision_streak_state", lambda *a, **k: obs)
                mp.setattr(picks, "get_game_statuses_detailed", lambda *a, **k: detailed_pre)
                mp.setattr(picks, "attach_provenance", lambda d, **k: d)
                mp.setattr(scheduler, "_void_committed_games", lambda *a: {})
                mp.setattr(dm, "send_dm", _send_timeout)
                chk = scheduler.run_single_check(ENTERED, [101], {(101, "away"), (101, "home")}, cfg, 0.03, set())
                scheduler._deliver_and_lock_pick(
                    chk["pick_result"].daily, cfg, root, st, ENTERED, "lineup", selection=chk["selection"]
                )

            with monkeypatch.context() as mp:
                mp.setattr(scheduler, "_now_et", lambda: FALLBACK_ET)
                mp.setattr(orchestrator, "run_cascade", lambda *a, **k: (lo_df, "local"))
                mp.setattr(contest_state, "load_decision_streak_state", lambda *a, **k: obs)
                mp.setattr(picks, "get_game_statuses_detailed", lambda *a, **k: detailed_pre)
                refresh = scheduler._refresh_pick_at_fallback_decision(
                    cfg, ENTERED, picks.load_pick(ENTERED, root), 0.03, set()
                )
                assert scheduler._refresh_is_genuine_skip(refresh)
                scheduler._capture_fallback_skip(st, refresh)
                scheduler.save_state(st, root)
            scheduler._write_endofday_skip(root, ENTERED, st)

        elif scenario == "manual_run_crash_with_daemon_eod_skip":
            st = scheduler.SchedulerState(
                ENTERED, "2027-04-18T12:25:00-04:00", [], [], [], False, None, None, None,
                final_skip_candidate={"source": "mdp", "primary": {"batter_id": 660271}},
            )
            scheduler.save_state(st, root)
            with monkeypatch.context() as mp:
                mp.setattr(predict, "run_pipeline", lambda *a, **k: hi_df)
                mp.setattr(contest_state, "load_decision_streak_state", lambda *a, **k: auto_obs)
                mp.setattr(picks, "get_game_statuses_detailed", lambda *a, **k: detailed_pre)
                mp.setattr(picks, "attach_provenance", lambda d, **k: d)
                mp.setattr(posting, "should_post_now", lambda *a: True)
                mp.setattr(posting, "post_to_bluesky", lambda _b: (_ for _ in ()).throw(KeyboardInterrupt()))
                r = CliRunner().invoke(cli.cli, ["run", "--date", ENTERED, "--picks-dir", str(root)])
                assert r.exit_code == 1
            scheduler._write_endofday_skip(root, ENTERED, st)

        elif scenario in ("live_private_preview_lock_restart", "live_private_preview_lock_quarantined"):
            with monkeypatch.context() as mp:
                mp.setattr(predict, "run_pipeline", lambda *a, **k: hi_df)
                mp.setattr(contest_state, "load_decision_streak_state", lambda *a, **k: model_obs)
                mp.setattr(picks, "get_game_statuses_detailed", lambda *a, **k: detailed_pre)
                mp.setattr(picks, "attach_provenance", lambda d, **k: d)
                pr = CliRunner().invoke(
                    cli.cli, ["preview", "--date", ENTERED, "--picks-dir", str(root), "--models-dir", str(work / "m")]
                )
                assert pr.exit_code == 0

            cfg = {
                "orchestrator": {"picks_dir": str(root), "models_dir": str(work / "m"),
                                 "heartbeat_path": str(work / ".hb")},
                "tiers": tiers,
                "scheduler": {"pick_delivery": "private", "entry_intent": "research", "contest_state_required": True,
                              "live_forward_capture_on_lock": True},
                "health_checks": {"contest_state_expected": True, "enabled": False},
            }
            assert entry_intent.check_entry_intent(cfg, include_live_gate=True).ok
            st = scheduler.SchedulerState(ENTERED, DELIVERY_ET.isoformat(), [], [], [], False, None, None, None)

            with monkeypatch.context() as mp:
                mp.setattr(scheduler, "_now_et", lambda: FALLBACK_ET)
                mp.setattr(orchestrator, "run_cascade", lambda *a, **k: (None, None))
                mp.setattr(scheduler, "_void_committed_games", lambda *a: {})
                mp.setattr(
                    scheduler, "_trigger_live_forward_capture_on_lock",
                    lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()),
                )
                refresh = scheduler._refresh_pick_at_fallback_decision(
                    cfg, ENTERED, picks.load_pick(ENTERED, root), 0.03, set()
                )
                with pytest.raises(KeyboardInterrupt):
                    scheduler._deliver_and_lock_pick(
                        refresh.daily, cfg, root, st, ENTERED, "final fallback", selection=refresh.selection
                    )

            if scenario == "live_private_preview_lock_restart":
                sched_games = [{
                    "gamePk": 101, "gameDate": f"{ENTERED}T23:10:00Z",
                    "teams": {"away": {"team": {"name": "LAD"}}, "home": {"team": {"name": "NYY"}}},
                }]
                with monkeypatch.context() as mp:
                    mp.setattr(scheduler, "_now_et", lambda: datetime(2027, 4, 18, 18, 36, tzinfo=ET))
                    mp.setattr(scheduler, "fetch_schedule", lambda d: sched_games if d == ENTERED else [])
                    mp.setattr(scheduler, "count_new_confirmations", lambda *a: 0)
                    mp.setattr(picks, "get_game_statuses_detailed", lambda *a, **k: detailed_pre)
                    mp.setattr(orchestrator, "run_cascade", lambda *a, **k: (hi_df, "local"))
                    mp.setattr(
                        contest_state, "load_decision_streak_state",
                        lambda *a, **k: (_ for _ in ()).throw(contest_state.ContestStateError("stale")),
                    )
                    mp.setattr(scheduler, "_alert_contest_state_failure", lambda *a: None)
                    mp.setattr(scheduler, "_maybe_alert_missed_pick", lambda *a: None)
                    mp.setattr(scheduler, "_idle_until_next_wakeup", lambda *a: None)
                    mp.setattr(scheduler, "notify_ready", lambda: None)
                    mp.setattr(scheduler, "notify_watchdog", lambda: None)
                    scheduler.run_day(ENTERED, cfg)
            else:
                (root / ENTERED / "scheduler_state.json").write_text("{torn")
                assert scheduler.load_state(ENTERED, root) is None

        elif scenario == "private_research_fresh_pick_restart":
            cfg = {
                "orchestrator": {"picks_dir": str(root), "models_dir": str(work / "m"),
                                 "heartbeat_path": str(work / ".hb")},
                "tiers": tiers,
                "scheduler": {"pick_delivery": "private", "entry_intent": "research", "contest_state_required": False,
                              "live_forward_capture_on_lock": True},
                "health_checks": {"contest_state_expected": False, "enabled": False},
            }
            assert entry_intent.check_entry_intent(cfg, include_live_gate=True).ok
            st = scheduler.SchedulerState(ENTERED, DELIVERY_ET.isoformat(), [], [], [], False, None, None, None)

            with monkeypatch.context() as mp:
                mp.setattr(scheduler, "_now_et", lambda: DELIVERY_ET)
                mp.setattr(scheduler, "count_new_confirmations", lambda *a: 1)
                mp.setattr(orchestrator, "run_cascade", lambda *a, **k: (hi_df, "local"))
                mp.setattr(contest_state, "load_decision_streak_state", lambda *a, **k: model_obs)
                mp.setattr(picks, "get_game_statuses_detailed", lambda *a, **k: detailed_pre)
                mp.setattr(picks, "attach_provenance", lambda d, **k: d)
                mp.setattr(scheduler, "_void_committed_games", lambda *a: {})
                mp.setattr(
                    scheduler, "_trigger_live_forward_capture_on_lock",
                    lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()),
                )
                chk = scheduler.run_single_check(ENTERED, [101], {(101, "away"), (101, "home")}, cfg, 0.03, set())
                with pytest.raises(KeyboardInterrupt):
                    scheduler._deliver_and_lock_pick(
                        chk["pick_result"].daily, cfg, root, st, ENTERED, "lineup", selection=chk["selection"]
                    )

            sched_games = [{
                "gamePk": 101, "gameDate": f"{ENTERED}T23:10:00Z",
                "teams": {"away": {"team": {"name": "LAD"}}, "home": {"team": {"name": "NYY"}}},
            }]
            with monkeypatch.context() as mp:
                mp.setattr(scheduler, "_now_et", lambda: datetime(2027, 4, 18, 18, 20, tzinfo=ET))
                mp.setattr(scheduler, "fetch_schedule", lambda _d: sched_games)
                mp.setattr(scheduler, "_watchdog_ping_sleep", lambda _s: (_ for _ in ()).throw(SystemExit(143)))
                mp.setattr(scheduler, "notify_ready", lambda: None)
                mp.setattr(scheduler, "notify_watchdog", lambda: None)
                with pytest.raises(SystemExit):
                    scheduler.run_day(ENTERED, cfg)

        elif scenario == "orchestrate_not_live_post_crash":
            cfg = {
                "orchestrator": {"picks_dir": str(root), "models_dir": str(work / "m")},
                "tiers": tiers,
                "bluesky": {"dm_recipient": "operator.invalid"},
            }
            with monkeypatch.context() as mp:
                mp.setattr(orchestrator, "run_cascade", lambda *a, **k: (hi_df, "local"))
                mp.setattr(contest_state, "load_decision_streak_state", lambda *a, **k: model_obs)
                mp.setattr(picks, "get_game_statuses_detailed", lambda *a, **k: detailed_pre)
                mp.setattr(picks, "attach_provenance", lambda d, **k: d)
                mp.setattr(posting, "should_post_now", lambda *a: True)
                mp.setattr(posting, "post_to_bluesky", lambda _b: (_ for _ in ()).throw(KeyboardInterrupt()))
                with pytest.raises(KeyboardInterrupt):
                    orchestrator.orchestrate(Path("unused.toml"), ENTERED, config=cfg)

        elif scenario == "preview_overwrites_unmarked_entered_pick":
            daily = picks.parse_daily_pick(json.dumps(_pick_dict(
                ENTERED, chosen_on=picks.chosen_on_record(auto_obs, required=True)
            )))
            picks.save_pick(daily, root)
            with monkeypatch.context() as mp:
                mp.setattr(predict, "run_pipeline", lambda *a, **k: hi_df)
                mp.setattr(contest_state, "load_decision_streak_state", lambda *a, **k: model_obs)
                mp.setattr(picks, "get_game_statuses_detailed", lambda *a, **k: detailed_pre)
                mp.setattr(picks, "attach_provenance", lambda d, **k: d)
                preview = CliRunner().invoke(
                    cli.cli, ["preview", "--date", ENTERED, "--picks-dir", str(root), "--models-dir", str(work / "m")]
                )
                assert preview.exit_code == 0, preview.output
                assert picks.load_pick(ENTERED, root).chosen_on["required"] is False

        # Verify check-results leaves 2027-04-18 unresolved, 2027-04-19 scores to 10, and reconcile holds!
        with monkeypatch.context() as mp:
            mp.setattr(cli, "_now_et", lambda: RUN_ET)
            mp.setattr(picks, "get_game_statuses_detailed", lambda *a, **k: {})
            mp.setattr(picks, "resolve_daily_slot_results", lambda *a, **k: {"pick": "hit"})
            refused = CliRunner().invoke(cli.cli, ["check-results", "--date", ENTERED, "--picks-dir", str(root)])
            assert refused.exit_code == 0
            assert picks.load_pick(ENTERED, root).result is None

            picks.save_pick(picks.parse_daily_pick(json.dumps(_pick_dict(GRADED, None, bluesky_posted=True))), root)
            scored = CliRunner().invoke(cli.cli, ["check-results", "--date", GRADED, "--picks-dir", str(root)])
            assert scored.exit_code == 0 and "Streak: 10" in scored.output

        before = sp.read_bytes()
        result, fold = compare_reconcile(
            root, monkeypatch, base_modules, clock=lambda: RUN_ET, lookback=8,
            result_feed=lambda *args, **kwargs: {"pick": "miss"},
        )
        assert fold is None
        assert result.exit_code == 0, result.output
        assert result.stderr == warning(ENTERED, PENDING)
        assert result.stdout.splitlines()[-3:] == [
            "CORRECTIONS FOUND (1):",
            f"  {GRADED}: Shohei Ohtani — hit -> miss",
            "Streak retained: 10",
        ]
        assert only_receipt(root)["replay"] == {
            "state": "unavailable", "blocked_date": ENTERED, "reason": PENDING,
        }
        assert sp.read_bytes() == before


@_deadline()
def test_b2_symlinked_decision_parent_keeps_unmarked_live_pick(tmp_path, monkeypatch, base_modules):
    from bts import contest_state, daily_decision, picks

    ext = tmp_path / "external"
    assert daily_decision.write_decision(
        "2027-04-16", ext, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN_ET,
    )

    root = tmp_path / "current" / "picks"
    root.mkdir(parents=True)
    picks.save_pick(picks.parse_daily_pick(json.dumps(_pick_dict("2027-04-01", "hit", bluesky_posted=True))), root)
    obs = contest_state.DecisionStreakState(
        9, True, True, "contest", "fresh", 9, True, observation_kind="automatic",
    )
    picks.save_pick(
        picks.parse_daily_pick(json.dumps(_pick_dict("2027-04-16", None, chosen_on=picks.chosen_on_record(obs, required=True)))),
        root,
    )
    (root / "2027-04-16").symlink_to(ext / "2027-04-16")
    picks.save_streak(9, root, saver_available=False, season=2027)
    sp = root / "streak.json"
    before = sp.read_bytes()

    result, fold = compare_reconcile(
        root, monkeypatch, base_modules, clock=lambda: RUN_ET, lookback=8,
        result_feed=lambda *args, **kwargs: {"pick": "hit"},
    )
    assert fold is None
    assert result.exit_code == 0, result.output
    assert result.stderr == warning("2027-04-16", PENDING)
    assert result.stdout.endswith("No scoring changes detected. Streak retained: 9\n")
    assert only_receipt(root)["replay"] == {
        "state": "unavailable", "blocked_date": "2027-04-16", "reason": PENDING,
    }
    assert sp.read_bytes() == before
