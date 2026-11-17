"""Fast standalone caller comparison probe v4 (Item 9: base vs branch, zero sleep, zero network)."""
from __future__ import annotations

import sys

sys.dont_write_bytecode = True

DENIED_NET = []
_NET_EVENTS = frozenset({
    "socket.getaddrinfo",
    "socket.getnameinfo",
    "socket.gethostbyname",
    "socket.gethostbyaddr",
    "socket.connect",
    "socket.connect_ex",
    "socket.sendto",
})


def _hook(event: str, args: tuple) -> None:
    if event in _NET_EVENTS:
        DENIED_NET.append((event, repr(args)))
        raise RuntimeError(f"network forbidden by audit hook: {event}")


sys.addaudithook(_hook)

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

os.environ.setdefault("MPLCONFIGDIR", tempfile.gettempdir())

import pandas as pd
from click.testing import CliRunner

ET = ZoneInfo("America/New_York")
REAL_DT = dt.datetime
NOW = REAL_DT(2027, 7, 15, 12, 0, 0, tzinfo=ET)
ID = "a" * 32


def _put_json(picks: Path, name: str, obj) -> Path:
    p = picks / "account_state" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj), encoding="utf-8")
    return p


def _sample_predictions(date_str: str = "2027-07-15") -> pd.DataFrame:
    return pd.DataFrame([
        {
            "batter_id": 680757,
            "batter_name": "Steven Kwan",
            "team": "CLE",
            "lineup": 1,
            "pitcher_id": 543037,
            "pitcher_name": "Gerrit Cole",
            "opp": "NYY",
            "p_hit_pa": 0.24,
            "p_game_hit": 0.82,
            "flags": "",
            "projected_lineup": False,
            "game_pk": 101,
            "game_time": f"{date_str}T23:10:00Z",
        },
        {
            "batter_id": 665742,
            "batter_name": "Juan Soto",
            "team": "NYM",
            "lineup": 2,
            "pitcher_id": 605483,
            "pitcher_name": "Blake Snell",
            "opp": "LAD",
            "p_hit_pa": 0.23,
            "p_game_hit": 0.81,
            "flags": "",
            "projected_lineup": False,
            "game_pk": 102,
            "game_time": f"{date_str}T23:40:00Z",
        },
    ])


def run_worker(src_dir: Path, out_path: Path) -> None:
    sys.path.insert(0, str(src_dir.resolve()))
    import bts.cli as cli
    import bts.contest_state as cs
    import bts.daily_decision as dd
    import bts.dm as dm_mod
    import bts.health.alert as halert
    import bts.orchestrator as orch
    import bts.picks as picks_mod
    import bts.posting as posting_mod
    import bts.scheduler as sch
    import bts.strategy as strat_mod

    runner = CliRunner()
    date_s = "2027-07-15"
    df = _sample_predictions(date_s)
    statuses_detailed = {
        101: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
        102: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
    }
    statuses_coarse = {101: "P", 102: "P"}
    out = {}

    def seed_uncovered(picks: Path) -> None:
        _put_json(picks, "contest_streak.json", {
            "schema_version": "bts_contest_streak_auto_v1",
            "active_streak": 0,
            "best_streak": 0,
            "source": "mlb_bts_profile",
            "source_date": "2027-07-14",
            "recorded_at": "2027-07-15T10:00:00Z",
        })
        _put_json(picks, "saver_state.json", {"season": 2027, "state": "not_earned"})

    def seed_covered(picks: Path, streak: int = 0) -> None:
        _put_json(picks, "season_contract.json", {
            "schema_version": "bts_season_contract_v1",
            "season": 2027,
            "status": "automatic",
            "activation_id": ID,
            "switched_at": "2027-07-15T10:00:00Z",
            "revoked_at": None,
            "note": None,
        })
        _put_json(picks, "contest_streak.json", {
            "schema_version": "bts_contest_streak_auto_v2",
            "season": 2027,
            "activation_id": ID,
            "active_streak": streak,
            "best_streak": max(streak, 9),
            "source": "mlb_bts_profile",
            "source_date": "2027-07-14",
            "recorded_at": "2027-07-15T10:00:00Z",
        })
        _put_json(picks, "saver_state.json", {"season": 2027, "state": "not_earned"})

    class FakeDT(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return NOW.astimezone(tz) if tz is not None else NOW.replace(tzinfo=None)

    with (
        patch.object(dt, "datetime", FakeDT),
        patch.object(sch, "_now_et", return_value=NOW),
        patch.object(sch, "_watchdog_ping_sleep", side_effect=AssertionError("unexpected sleep")),
        patch.object(sch, "_void_committed_games", return_value={}),
        patch.object(sch, "_trigger_live_forward_capture_on_lock", return_value=None),
        patch.object(picks_mod, "get_game_statuses_detailed", return_value=statuses_detailed),
        patch.object(picks_mod, "get_game_statuses", return_value=statuses_coarse),
        patch.object(strat_mod, "get_game_statuses_detailed", return_value=statuses_detailed),
        patch.object(strat_mod, "get_game_statuses", return_value=statuses_coarse),
        patch.object(picks_mod, "attach_provenance", return_value=None),
    ):
        # 1. orchestrator.run_and_pick
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            cfg = {
                "orchestrator": {"picks_dir": str(picks)},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm"},
                "tiers": [],
            }
            with patch.object(orch, "run_cascade", return_value=(pd.DataFrame(), "local")):
                _, sel_empty, _ = orch.run_and_pick(cfg, date_s)
                out["run_and_pick_empty_predictions"] = {"sel_is_none": sel_empty is None}
            with patch.object(orch, "run_cascade", return_value=(df, "local")):
                try:
                    _, sel, _ = orch.run_and_pick(cfg, date_s)
                    out["run_and_pick_uncovered"] = {"accepted": True, "action": sel.action}
                except cs.ContestStateError as exc:
                    out["run_and_pick_uncovered"] = {"accepted": False, "err": str(exc).splitlines()[0]}

        # 2. orchestrator.orchestrate initial uncovered state
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            cfg_obj = {
                "orchestrator": {"picks_dir": str(picks)},
                "bluesky": {"dm_recipient": "eric.test"},
                "scheduler": {"contest_state_required": True},
                "tiers": [],
            }
            dms, posts = [], []
            with (
                patch.object(orch, "load_config", return_value=cfg_obj),
                patch.object(orch, "run_cascade", return_value=(df, "local")),
                patch.object(dm_mod, "send_dm", side_effect=lambda r, t: dms.append((r, t)) or "dm-1"),
                patch.object(posting_mod, "post_to_bluesky", side_effect=lambda t: posts.append(t) or "uri-1"),
                patch.object(posting_mod, "should_post_now", return_value=True),
            ):
                ok = orch.orchestrate(Path(tmp) / "orch.toml", date_s)
            out["orchestrate_initial_uncovered"] = {
                "returned": ok,
                "pick_saved": (picks / f"{date_s}.json").exists(),
                "dms_count": len(dms),
                "posts_count": len(posts),
            }

        # 3. orchestrator.orchestrate later reads: post-save, skip, and locked catch-up
        for sub_mode in ("post_save", "skip", "locked_catchup"):
            with tempfile.TemporaryDirectory() as tmp:
                picks = Path(tmp) / "picks"
                picks.mkdir()
                seed_covered(picks, streak=18 if sub_mode == "skip" else 0)
                if sub_mode == "locked_catchup":
                    dp_l = picks_mod.DailyPick(
                        date=date_s, run_time=NOW.isoformat(),
                        pick=picks_mod.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 101, f"{date_s}T23:10:00Z", "NYY"),
                        double_down=None, runner_up=None, bluesky_posted=False, delivery_attempted=True,
                    )
                    picks_mod.save_pick(dp_l, picks)
                cfg_obj = {
                    "orchestrator": {"picks_dir": str(picks)},
                    "bluesky": {"dm_recipient": "eric.test"},
                    "scheduler": {"contest_state_required": True},
                    "tiers": [],
                }
                df_use = df.assign(p_game_hit=0.60) if sub_mode == "skip" else df
                posts = []
                real_rap = orch.run_and_pick

                def rap_then_revoke(*a, **kw):
                    res = real_rap(*a, **kw)
                    _put_json(picks, "season_contract.json", {
                        "schema_version": "bts_season_contract_v1",
                        "season": 2027,
                        "status": "revoked",
                        "activation_id": ID,
                        "switched_at": "2027-07-15T10:00:00Z",
                        "revoked_at": "2027-07-15T12:00:00Z",
                        "note": "mid-orchestrate",
                    })
                    return res

                with (
                    patch.object(orch, "load_config", return_value=cfg_obj),
                    patch.object(orch, "run_cascade", return_value=(df_use, "local")),
                    patch.object(orch, "run_and_pick", side_effect=rap_then_revoke),
                    patch.object(posting_mod, "post_to_bluesky", side_effect=lambda t: posts.append(t) or "uri-1"),
                    patch.object(posting_mod, "should_post_now", return_value=True),
                ):
                    try:
                        ok = orch.orchestrate(Path(tmp) / "orch.toml", date_s)
                        out[f"orchestrate_mid_revoke_{sub_mode}"] = {"raised": False, "returned": ok, "posts": len(posts)}
                    except cs.ContestStateError as exc:
                        out[f"orchestrate_mid_revoke_{sub_mode}"] = {
                            "raised": True,
                            "err": str(exc).splitlines()[0],
                            "pick_exists": (picks / f"{date_s}.json").exists(),
                            "posts": len(posts),
                        }

        # 4. scheduler.run_single_check on uncovered state
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            cfg = {
                "orchestrator": {"picks_dir": str(picks), "heartbeat_path": str(Path(tmp) / ".hb")},
                "bluesky": {"dm_recipient": "eric.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm"},
                "tiers": [],
            }
            dms = []
            with (
                patch.object(sch, "count_new_confirmations", return_value=2),
                patch.object(orch, "run_cascade", return_value=(df, "local")),
                patch.object(strat_mod, "should_lock", return_value=SimpleNamespace(should_lock=True, should_lock_ungated=True, block_reason=None, contender_game_pk=None, contender_game_pks=())),
                patch.object(halert, "send_dm", side_effect=lambda r, t: dms.append((r, t)) or "dm-h"),
                patch.object(dm_mod, "send_dm", side_effect=lambda r, t: dms.append((r, t)) or "dm-d"),
            ):
                chk = sch.run_single_check(
                    date=date_s,
                    all_game_pks=[101, 102],
                    confirmed_sides={(101, "away"), (101, "home"), (102, "away"), (102, "home")},
                    config=cfg,
                    early_lock_gap=0.03,
                )
            out["scheduler_run_single_check_uncovered"] = {
                "should_post": chk["should_post"],
                "pick_result_none": chk["pick_result"] is None,
                "selection_none": chk["selection"] is None,
                "pick_file_exists": (picks / f"{date_s}.json").exists(),
                "decision_exists": (picks / date_s / "decision.json").exists(),
                "alert_dms": len(dms),
            }

        # 5. scheduler._deliver_and_lock_pick across dm, public, private, delivered, crash
        for mode in ("dm", "public", "private", "delivered", "crash"):
            with tempfile.TemporaryDirectory() as tmp:
                picks = Path(tmp) / "picks"
                picks.mkdir()
                seed_uncovered(picks)
                cfg = {
                    "orchestrator": {"picks_dir": str(picks)},
                    "bluesky": {"dm_recipient": "eric.test"},
                    "scheduler": {
                        "contest_state_required": True,
                        "pick_delivery": "private" if mode == "private" else ("public" if mode == "public" else "dm"),
                    },
                    "tiers": [],
                }
                dp = picks_mod.DailyPick(
                    date=date_s,
                    run_time=NOW.isoformat(),
                    pick=picks_mod.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 101, f"{date_s}T23:10:00Z", "NYY"),
                    double_down=None,
                    runner_up=None,
                )
                if mode == "delivered":
                    dp.notification_sent = True
                    dp.notification_channel = "bluesky_dm"
                    dp.notification_id = "already"
                    dp.delivered_at = NOW.isoformat()
                elif mode == "crash":
                    dp.delivery_attempted = True
                picks_mod.save_pick(dp, picks)
                st = sch.SchedulerState(
                    date=date_s, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
                    runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
                )
                pick_dms, alert_dms, posts = [], [], []
                with (
                    patch.object(dm_mod, "send_dm", side_effect=lambda r, t: pick_dms.append((r, t)) or "dm-p"),
                    patch.object(halert, "send_dm", side_effect=lambda r, t: alert_dms.append((r, t)) or "dm-a"),
                    patch.object(posting_mod, "post_to_bluesky", side_effect=lambda t: posts.append(t) or "uri-p"),
                ):
                    ok = sch._deliver_and_lock_pick(dp, cfg, picks, st, date_s, mode)
                out[f"deliver_and_lock_{mode}"] = {
                    "returned": ok,
                    "pick_locked": st.pick_locked,
                    "delivery_attempted": dp.delivery_attempted,
                    "pick_dms": len(pick_dms),
                    "alert_dms": len(alert_dms),
                    "posts": len(posts),
                    "decision_exists": (picks / date_s / "decision.json").exists(),
                }

        # 6. scheduler._refresh_pick_at_fallback_decision on uncovered state
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            cfg = {
                "orchestrator": {"picks_dir": str(picks), "heartbeat_path": str(Path(tmp) / ".hb")},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm"},
                "tiers": [],
            }
            cached = picks_mod.DailyPick(
                date=date_s,
                run_time=NOW.isoformat(),
                pick=picks_mod.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 101, f"{date_s}T23:10:00Z", "NYY"),
                double_down=None,
                runner_up=None,
            )
            with (
                patch.object(orch, "run_cascade", return_value=(df, "local")),
                patch.object(strat_mod, "should_lock", return_value=SimpleNamespace(should_lock=True, should_lock_ungated=True, block_reason=None, contender_game_pk=None, contender_game_pks=())),
            ):
                try:
                    ref = sch._refresh_pick_at_fallback_decision(cfg, date_s, cached, 0.03)
                    out["fallback_refresh_uncovered"] = {"raised": False, "batter": ref.daily.pick.batter_name}
                except cs.ContestStateError as exc:
                    out["fallback_refresh_uncovered"] = {"raised": True, "err": str(exc).splitlines()[0]}

        # 7. Cross-path probe: bts preview (optional loader) + fallback cascade failure (7a uncovered, 7b dated 56)
        with tempfile.TemporaryDirectory() as tmp:
            picks_7a = Path(tmp) / "picks_7a"
            picks_7a.mkdir()
            seed_uncovered(picks_7a)
            models_dir = Path(tmp) / "models"
            models_dir.mkdir()
            with patch("bts.model.predict.run_pipeline", return_value=df):
                assert runner.invoke(
                    cli.cli,
                    ["preview", "--date", date_s, "--picks-dir", str(picks_7a), "--models-dir", str(models_dir)],
                ).exit_code == 0
            preview_daily = picks_mod.load_pick(date_s, picks_7a)
            preview_was_double = preview_daily.double_down is not None

            cfg_7a = {
                "orchestrator": {"picks_dir": str(picks_7a), "heartbeat_path": str(Path(tmp) / ".hb1")},
                "bluesky": {"dm_recipient": "eric.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm"},
                "tiers": [],
            }
            pick_dms_a, alert_dms_a = [], []
            st_a = sch.SchedulerState(
                date=date_s, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            with (
                patch.object(orch, "run_cascade", return_value=(None, None)),
                patch.object(dm_mod, "send_dm", side_effect=lambda r, t: pick_dms_a.append((r, t)) or "dm-p"),
                patch.object(halert, "send_dm", side_effect=lambda r, t: alert_dms_a.append((r, t)) or "dm-a"),
            ):
                ref_a = sch._refresh_pick_at_fallback_decision(cfg_7a, date_s, preview_daily, 0.03)
                ok_a = sch._deliver_and_lock_pick(ref_a.daily, cfg_7a, picks_7a, st_a, date_s, "final fallback", selection=ref_a.selection)

            picks_7b = Path(tmp) / "picks_7b"
            picks_7b.mkdir()
            seed_uncovered(picks_7b)
            with patch("bts.model.predict.run_pipeline", return_value=df):
                assert runner.invoke(
                    cli.cli,
                    ["preview", "--date", date_s, "--picks-dir", str(picks_7b), "--models-dir", str(models_dir)],
                ).exit_code == 0
            d56 = {
                "schema_version": "bts_contest_streak_manual_v3",
                "effective_date": date_s,
                "season": 2027,
                "expires_at": REAL_DT.combine(dt.date(2027, 7, 15), dt.time.max, ET).isoformat(),
                "active_streak": 56,
                "best_streak": 56,
                "saver_state": "used",
                "recorded_at": NOW.isoformat(),
                "source": "manual_cli",
            }
            _put_json(picks_7b, "contest_streak.manual.json", d56)
            cfg_7b = {
                "orchestrator": {"picks_dir": str(picks_7b), "heartbeat_path": str(Path(tmp) / ".hb2")},
                "bluesky": {"dm_recipient": "eric.test"},
                "scheduler": {"contest_state_required": True, "pick_delivery": "dm"},
                "tiers": [],
            }
            pick_dms_b = []
            st_b = sch.SchedulerState(
                date=date_s, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            preview_daily_b = picks_mod.load_pick(date_s, picks_7b)
            with (
                patch.object(orch, "run_cascade", return_value=(None, None)),
                patch.object(dm_mod, "send_dm", side_effect=lambda r, t: pick_dms_b.append((r, t)) or "dm-p"),
            ):
                ref_b = sch._refresh_pick_at_fallback_decision(cfg_7b, date_s, preview_daily_b, 0.03)
                ok_b = sch._deliver_and_lock_pick(ref_b.daily, cfg_7b, picks_7b, st_b, date_s, "final fallback", selection=ref_b.selection)

            out["preview_plus_fallback_cascade_failure"] = {
                "preview_created_double_from_uncovered_streak_0": preview_was_double,
                "still_uncovered_dm_delivered": ok_a,
                "still_uncovered_pick_dms": len(pick_dms_a),
                "after_dated_56_cascade_fail_delivered_cached_double": ok_b and (preview_daily_b.double_down is not None),
                "after_dated_56_dm_first_line": pick_dms_b[0][1].splitlines()[0] if pick_dms_b else None,
            }

        # 8. Optional-loader live posting path: bts run (non-dry-run) on uncovered state
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            models_dir = Path(tmp) / "models"
            models_dir.mkdir()
            posts = []
            with (
                patch("bts.model.predict.run_pipeline", return_value=df),
                patch.object(posting_mod, "should_post_now", return_value=True),
                patch.object(posting_mod, "post_to_bluesky", side_effect=lambda t: posts.append(t) or "uri-run"),
            ):
                r_run = runner.invoke(
                    cli.cli,
                    ["run", "--date", date_s, "--picks-dir", str(picks), "--models-dir", str(models_dir)],
                )
            out["bts_run_optional_loader_uncovered"] = {
                "exit_code": r_run.exit_code,
                "pick_saved": (picks / f"{date_s}.json").exists(),
                "posts_sent": len(posts),
            }

        # 9. Non-blocked paths on refused state: check-results, run_result_polling, EOD skip, day-outcome
        with tempfile.TemporaryDirectory() as tmp:
            picks = Path(tmp) / "picks"
            picks.mkdir()
            seed_uncovered(picks)
            dp = picks_mod.DailyPick(
                date=date_s,
                run_time=NOW.isoformat(),
                pick=picks_mod.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 101, f"{date_s}T23:10:00Z", "NYY"),
                double_down=None,
                runner_up=None,
                notification_sent=True,
                notification_channel="bluesky_dm",
                notification_id="dm-ok",
                delivered_at=NOW.isoformat(),
            )
            picks_mod.save_pick(dp, picks)
            dd.write_decision(
                date_s, picks, action="single", source="mdp", streak=0, saver_available=True,
                state_source="contest", state_status="fresh", delivery_status="delivered", scoreable=True,
            )
            with (
                patch.object(cli, "_now_et", return_value=NOW),
                patch.object(picks_mod, "resolve_daily_slot_results", return_value={"primary": "hit"}),
            ):
                r_chk = runner.invoke(cli.cli, ["check-results", "--date", date_s, "--picks-dir", str(picks)])
            streak_after_chk = json.loads((picks / "streak.json").read_text(encoding="utf-8"))["streak"]

            picks_poll = Path(tmp) / "picks_poll"
            picks_poll.mkdir()
            seed_uncovered(picks_poll)
            dp2 = picks_mod.DailyPick(
                date=date_s, run_time=NOW.isoformat(),
                pick=picks_mod.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.82, [], False, 101, f"{date_s}T23:10:00Z", "NYY"),
                double_down=None, runner_up=None, notification_sent=True, notification_channel="bluesky_dm",
                notification_id="dm-ok2", delivered_at=NOW.isoformat(),
            )
            picks_mod.save_pick(dp2, picks_poll)
            with (
                patch.object(sch, "poll_game_result", return_value="final"),
                patch.object(picks_mod, "get_game_statuses_detailed", return_value={101: {"abstract": "F", "detailed": "Final", "code": "F"}}),
                patch.object(picks_mod, "resolve_daily_slot_results", return_value={"primary": "hit"}),
            ):
                poll_res = sch.run_result_polling(101, date_s, picks_poll, poll_interval_min=1, cap_hour_et=5)
            streak_after_poll = json.loads((picks_poll / "streak.json").read_text(encoding="utf-8"))["streak"]

            picks_skip = Path(tmp) / "picks_skip"
            picks_skip.mkdir()
            seed_uncovered(picks_skip)
            st_skip = sch.SchedulerState(
                date=date_s, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
                runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            st_skip.final_skip_candidate = {
                "source": "mdp",
                "primary": {"batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "game_pk": 101, "p_game_hit": 0.72},
                "double": None,
                "streak": 18,
                "saver_available": False,
                "state_source": "contest",
                "state_status": "fresh",
                "allow_double": True,
                "contest_source_date": "2027-07-14",
            }
            sch._write_endofday_skip(picks_skip, date_s, st_skip)
            cfg_toml = Path(tmp) / "do.toml"
            cfg_toml.write_text(
                f'[orchestrator]\npicks_dir = "{picks_skip}"\n[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n',
                encoding="utf-8",
            )
            dms_do = []
            with (
                patch.object(sch, "fetch_schedule", return_value=[{"gamePk": 101, "status": {"detailedState": "Final"}}]),
                patch.object(dm_mod, "send_dm", side_effect=lambda r, t: dms_do.append((r, t)) or "dm"),
            ):
                r_do = runner.invoke(
                    cli.cli,
                    [
                        "day-outcome", "--date", date_s, "--config", str(cfg_toml),
                        "--picks-dir", str(picks_skip), "--expected-username", "stonehengee",
                        "--dm-recipient", "eric.test",
                    ],
                )
            out["non_blocked_grading_skip_day_outcome"] = {
                "check_results_exit": r_chk.exit_code,
                "streak_after_check_results": streak_after_chk,
                "poll_status": poll_res,
                "streak_after_poll": streak_after_poll,
                "eod_skip_written": (picks_skip / date_s / "decision.json").exists(),
                "day_outcome_exit": r_do.exit_code,
                "day_outcome_output": r_do.output.strip(),
                "day_outcome_dms": len(dms_do),
            }

    out["denied_network"] = DENIED_NET
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--src", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()

    repo = Path(__file__).resolve().parents[2]
    work = Path(__file__).resolve().parent
    if args.worker:
        run_worker(args.src, args.out)
        return

    base_p = work / "callers_base.json"
    branch_p = work / "callers_branch.json"
    script = Path(__file__).resolve()
    for src, out_p in [(work / "base/src", base_p), (repo / "src", branch_p)]:
        proc = subprocess.run(
            [sys.executable, str(script), "--worker", "--src", str(src), "--out", str(out_p)],
            capture_output=True,
            text=True,
            check=False,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"worker failed ({src}):\n{proc.stdout}\n{proc.stderr}")

    base_c = json.loads(base_p.read_text(encoding="utf-8"))
    branch_c = json.loads(branch_p.read_text(encoding="utf-8"))
    assert base_c["denied_network"] == []
    assert branch_c["denied_network"] == []

    assert base_c["run_and_pick_uncovered"] == {"accepted": True, "action": "double"}
    assert branch_c["run_and_pick_uncovered"]["accepted"] is False
    assert base_c["orchestrate_initial_uncovered"] == {"returned": True, "pick_saved": True, "dms_count": 0, "posts_count": 1}
    assert branch_c["orchestrate_initial_uncovered"] == {"returned": False, "pick_saved": False, "dms_count": 1, "posts_count": 0}
    for sub in ("post_save", "skip", "locked_catchup"):
        assert branch_c[f"orchestrate_mid_revoke_{sub}"]["raised"] is True
        assert branch_c[f"orchestrate_mid_revoke_{sub}"]["posts"] == 0
    assert base_c["scheduler_run_single_check_uncovered"]["pick_file_exists"] is True
    assert branch_c["scheduler_run_single_check_uncovered"] == {
        "should_post": False,
        "pick_result_none": True,
        "selection_none": True,
        "pick_file_exists": False,
        "decision_exists": False,
        "alert_dms": 1,
    }
    assert base_c["deliver_and_lock_dm"]["returned"] is True and base_c["deliver_and_lock_dm"]["pick_dms"] == 1
    assert branch_c["deliver_and_lock_dm"]["returned"] is False and branch_c["deliver_and_lock_dm"]["pick_dms"] == 0
    assert base_c["deliver_and_lock_public"]["returned"] is True and base_c["deliver_and_lock_public"]["posts"] == 1
    assert branch_c["deliver_and_lock_public"]["returned"] is False and branch_c["deliver_and_lock_public"]["posts"] == 0
    assert branch_c["deliver_and_lock_private"]["returned"] is True
    assert branch_c["deliver_and_lock_delivered"]["returned"] is True
    assert branch_c["deliver_and_lock_crash"]["returned"] is False and branch_c["deliver_and_lock_crash"]["pick_locked"] is True
    assert base_c["fallback_refresh_uncovered"]["raised"] is False
    assert branch_c["fallback_refresh_uncovered"]["raised"] is True
    assert branch_c["non_blocked_grading_skip_day_outcome"]["check_results_exit"] == 0
    assert branch_c["non_blocked_grading_skip_day_outcome"]["streak_after_check_results"] == 1
    assert branch_c["non_blocked_grading_skip_day_outcome"]["poll_status"] == "final"
    assert branch_c["non_blocked_grading_skip_day_outcome"]["streak_after_poll"] == 1
    assert branch_c["non_blocked_grading_skip_day_outcome"]["eod_skip_written"] is True
    assert branch_c["non_blocked_grading_skip_day_outcome"]["day_outcome_exit"] == 0
    print(json.dumps({"base": base_c, "branch": branch_c}, indent=2))


if __name__ == "__main__":
    main()
