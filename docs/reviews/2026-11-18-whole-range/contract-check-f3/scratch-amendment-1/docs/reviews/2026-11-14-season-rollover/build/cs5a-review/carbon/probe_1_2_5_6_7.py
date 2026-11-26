"""Adversarial probe for CS5a items 1, 2, 5, 6, 7 (base 4cab4212 vs branch 30956cdd).

Runs in isolated worker subprocesses with network/DNS forbidden and transports captured.
"""

from __future__ import annotations

import copy
import datetime as dt_mod
import io
import json
import os
import subprocess
import sys
import tempfile
from contextlib import ExitStack, contextmanager, nullcontext, redirect_stderr
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

REAL_DATETIME = dt_mod.datetime
ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = REAL_DATETIME.fromisoformat("2027-07-15T18:35:00-04:00")


def install_no_network_hook() -> None:
    """Forbid DNS lookups and non-loopback socket connects in probe workers."""
    blocked_events = {
        "socket.getaddrinfo",
        "socket.getnameinfo",
        "socket.gethostbyname",
        "socket.gethostbyaddr",
    }

    def hook(event: str, args: tuple) -> None:
        if event in blocked_events:
            raise RuntimeError(f"Forbidden network DNS event: {event} {args!r}")
        if event == "socket.connect":
            addr = args[1] if len(args) > 1 else None
            host = addr[0] if isinstance(addr, tuple) and addr else None
            if host not in {"127.0.0.1", "::1", "localhost"}:
                raise RuntimeError(f"Forbidden outward connect: {addr!r}")

    sys.addaudithook(hook)


def write_account(picks_dir: Path, name: str, payload: dict) -> Path:
    p = picks_dir / "account_state" / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(payload, indent=2))
    return p


def set_uncovered_2026_auto(picks_dir: Path, streak: int = 0) -> None:
    write_account(picks_dir, "contest_streak.json", {
        "schema_version": "bts_contest_streak_auto_v2",
        "season": 2026,
        "activation_id": "a" * 32,
        "active_streak": streak,
        "best_streak": streak,
        "source": "mlb_bts_profile",
        "source_date": "2026-09-27",
        "recorded_at": "2026-09-28T06:00:00Z",
    })


def set_covered_2027_auto(picks_dir: Path, now: dt_mod.datetime = NOW, streak: int = 5,
                          activation_id: str = "b" * 32) -> None:
    day = now.astimezone(ET).date()
    write_account(picks_dir, "contest_streak.json", {
        "schema_version": "bts_contest_streak_auto_v2",
        "season": day.year,
        "activation_id": activation_id,
        "active_streak": streak,
        "best_streak": max(9, streak),
        "source": "mlb_bts_profile",
        "source_date": day.isoformat(),
        "recorded_at": now.astimezone(dt_mod.timezone.utc).isoformat(),
    })
    write_account(picks_dir, "season_contract.json", {
        "schema_version": "bts_season_contract_v1",
        "season": day.year,
        "status": "automatic",
        "activation_id": activation_id,
        "switched_at": now.astimezone(dt_mod.timezone.utc).isoformat(),
        "revoked_at": None,
        "note": None,
    })


def set_dated_obs(picks_dir: Path, now: dt_mod.datetime = NOW, streak: int = 56,
                  best: int = 56, saver: str = "used",
                  reason: str | None = None,
                  recorded_at: str | None = None) -> None:
    day = now.astimezone(ET).date()
    end_of_day = REAL_DATETIME.combine(day, dt_mod.time.max, ET)
    body = {
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": day.isoformat(),
        "season": day.year,
        "expires_at": end_of_day.isoformat(),
        "active_streak": streak,
        "best_streak": best,
        "saver_state": saver,
        "recorded_at": recorded_at or now.astimezone(dt_mod.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "manual_cli",
    }
    if reason is not None:
        body["reason"] = reason
    write_account(picks_dir, "contest_streak.manual.json", body)


@contextmanager
def offline_harness(tmp_dir: Path, initial_now: dt_mod.datetime = NOW):
    import pandas as pd
    from bts import cli, contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
    from bts.health import alert
    from bts.model import predict

    clock_box = [initial_now]

    class PinnedDatetime(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            cur = clock_box[0]
            return cur.astimezone(tz) if tz is not None else cur.replace(tzinfo=None)

    picks_dir = tmp_dir / "picks"
    picks_dir.mkdir(parents=True, exist_ok=True)
    statuses = {
        101: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
        102: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
    }
    frame = pd.DataFrame([
        {
            "batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
            "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
            "p_hit_pa": 0.24, "p_game_hit": 0.82, "flags": "", "projected_lineup": False,
            "game_pk": 101, "game_time": f"{DAY}T23:10:00Z",
        },
        {
            "batter_id": 665742, "batter_name": "Juan Soto", "team": "NYM", "lineup": 2,
            "pitcher_id": 605483, "pitcher_name": "Blake Snell", "opp": "LAD",
            "p_hit_pa": 0.23, "p_game_hit": 0.81, "flags": "", "projected_lineup": False,
            "game_pk": 102, "game_time": f"{DAY}T23:40:00Z",
        },
    ])
    games = [
        {
            "gamePk": 101, "gameDate": f"{DAY}T23:10:00Z",
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {"away": {"team": {"name": "Away 101"}}, "home": {"team": {"name": "Home 101"}}},
        },
        {
            "gamePk": 102, "gameDate": f"{DAY}T23:40:00Z",
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {"away": {"team": {"name": "Away 102"}}, "home": {"team": {"name": "Home 102"}}},
        },
    ]
    config = {
        "orchestrator": {
            "picks_dir": str(picks_dir),
            "heartbeat_path": str(tmp_dir / "heartbeat"),
            "models_dir": str(tmp_dir / "models"),
        },
        "bluesky": {"dm_recipient": "operator.test"},
        "scheduler": {
            "contest_state_required": True,
            "pick_delivery": "dm",
            "live_forward_capture_on_lock": False,
        },
        "health_checks": {"enabled": False},
        "tiers": [],
    }

    pick_dm = Mock(return_value="dm://pick")
    health_dm = Mock(return_value="dm://health")
    post_bsky = Mock(return_value="at://post/1")
    should_post = Mock(return_value=True)
    pipeline = Mock(return_value=frame)
    cascade = Mock(return_value=(frame, "offline"))

    with ExitStack() as stack:
        for p in (
            patch.object(dt_mod, "datetime", PinnedDatetime),
            patch.object(contest_state, "datetime", PinnedDatetime),
            patch.object(daily_decision, "datetime", PinnedDatetime),
            patch.object(alert, "datetime", PinnedDatetime),
            patch.object(scheduler, "_now_et", side_effect=lambda: clock_box[0].astimezone(ET)),
            patch.object(cli, "_now_et", side_effect=lambda: clock_box[0].astimezone(ET)),
            patch.object(predict, "run_pipeline", pipeline),
            patch.object(orchestrator, "run_cascade", cascade),
            patch.object(picks, "get_game_statuses_detailed", side_effect=lambda *_a: statuses),
            patch.object(strategy, "get_game_statuses_detailed", side_effect=lambda *_a: statuses),
            patch.object(picks, "get_game_statuses", side_effect=lambda *_a: {k: v["abstract"] for k, v in statuses.items()}),
            patch.object(strategy, "get_game_statuses", side_effect=lambda *_a: {k: v["abstract"] for k, v in statuses.items()}),
            patch.object(scheduler, "fetch_schedule", side_effect=lambda d: games if d == DAY else []),
            patch.object(scheduler, "check_confirmed_lineups", side_effect=lambda *_a: {}),
            patch.object(scheduler, "heartbeat_watchdog", side_effect=lambda *_a, **_k: nullcontext()),
            patch.object(scheduler, "_idle_until_next_wakeup", side_effect=lambda *_a: None),
            patch.object(scheduler, "_watchdog_ping_sleep", side_effect=lambda *_a: None),
            patch.object(scheduler, "run_result_polling", Mock(return_value="final")),
            patch.object(scheduler, "retry_urlopen", Mock(side_effect=AssertionError("unexpected retry_urlopen"))),
            patch.object(dm, "send_dm", pick_dm),
            patch.object(alert, "send_dm", health_dm),
            patch.object(posting, "post_to_bluesky", post_bsky),
            patch.object(posting, "should_post_now", should_post),
        ):
            stack.enter_context(p)
        yield SimpleNamespace(
            config=config,
            picks_dir=picks_dir,
            clock=clock_box,
            statuses=statuses,
            frame=frame,
            games=games,
            pick_dm=pick_dm,
            health_dm=health_dm,
            post_bsky=post_bsky,
            should_post=should_post,
            pipeline=pipeline,
            cascade=cascade,
        )


def write_toml(config: dict, path: Path) -> Path:
    lines = ["tiers = []"]
    for sec, vals in config.items():
        if sec == "tiers":
            continue
        lines.append(f"[{sec}]")
        for k, v in vals.items():
            lines.append(f"{k} = {json.dumps(v)}")
    path.write_text("\n".join(lines) + "\n")
    return path


def make_sched_state(date: str = DAY):
    from bts import scheduler
    return scheduler.SchedulerState(
        date=date, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
        runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None,
        next_wakeup=None,
    )


def run_item_1_hole(target: str) -> dict:
    """Reproduce the hole on base and verify closure + recovery on branch across scheduler, run, orchestrate."""
    from click.testing import CliRunner
    from bts import cli, picks, scheduler
    from bts.daily_decision import load_decision

    runner = CliRunner()
    out: dict = {}

    # 1A: Exact Why reproduction (streak 56, saver used) across scheduler, bts run, bts orchestrate
    for channel in ("scheduler", "run_warmup", "run_locked", "orchestrate_warmup", "orchestrate_locked"):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with offline_harness(root) as h:
                set_uncovered_2026_auto(h.picks_dir, streak=0)
                r_prev = runner.invoke(cli.cli, [
                    "preview", "--date", DAY,
                    "--picks-dir", str(h.picks_dir),
                    "--models-dir", h.config["orchestrator"]["models_dir"],
                ], catch_exceptions=False)
                assert r_prev.exit_code == 0, r_prev.output
                pick_file = h.picks_dir / f"{DAY}.json"
                preview_bytes = pick_file.read_bytes()
                cached = picks.load_pick(DAY, h.picks_dir)
                assert cached.double_down is not None

                # Operator supplies dated observation via real CLI command
                r_set = runner.invoke(cli.cli, [
                    "set-contest-streak", "--date", DAY,
                    "--streak", "56", "--best-streak", "56", "--saver", "used",
                    "--picks-dir", str(h.picks_dir),
                ], catch_exceptions=False)
                assert r_set.exit_code == 0, r_set.output

                if channel == "scheduler":
                    h.cascade.side_effect = RuntimeError("cascade down")
                    ref = scheduler._refresh_pick_at_fallback_decision(h.config, DAY, cached, 0.03)
                    assert ref.daily is cached and ref.selection is None
                    st = make_sched_state()
                    err_buf = io.StringIO()
                    with redirect_stderr(err_buf):
                        ok = scheduler._deliver_and_lock_pick(
                            ref.daily, h.config, h.picks_dir, st, DAY, "fallback", selection=ref.selection
                        )
                    dec = load_decision(DAY, h.picks_dir)
                    reloaded = picks.load_pick(DAY, h.picks_dir)
                    out[channel] = {
                        "returned": ok,
                        "pick_dm_calls": [c.args[1] for c in h.pick_dm.call_args_list],
                        "post_calls": [c.args[0] for c in h.post_bsky.call_args_list],
                        "health_dm_calls": [c.args[1] for c in h.health_dm.call_args_list],
                        "bytes_identical": pick_file.read_bytes() == preview_bytes,
                        "delivery_attempted": reloaded.delivery_attempted,
                        "notification_sent": reloaded.notification_sent,
                        "pick_locked": st.pick_locked,
                        "decision_status": dec["delivery_status"] if dec else None,
                        "stderr": err_buf.getvalue(),
                    }
                else:
                    is_locked = channel.endswith("_locked")
                    st_val = ({"abstract": "L", "detailed": "In Progress", "code": "I"}
                              if is_locked else {"abstract": "P", "detailed": "Warmup", "code": "PW"})
                    h.statuses[101] = st_val
                    h.statuses[102] = st_val
                    if channel.startswith("run_"):
                        res = runner.invoke(cli.cli, [
                            "run", "--date", DAY,
                            "--picks-dir", str(h.picks_dir),
                            "--models-dir", h.config["orchestrator"]["models_dir"],
                        ])
                    else:
                        cfg_path = write_toml(h.config, root / "cfg.toml")
                        res = runner.invoke(cli.cli, [
                            "orchestrate", "--date", DAY, "--config", str(cfg_path),
                        ])
                    reloaded = picks.load_pick(DAY, h.picks_dir)
                    out[channel] = {
                        "exit_code": res.exit_code,
                        "post_calls": [c.args[0] for c in h.post_bsky.call_args_list],
                        "bluesky_posted": reloaded.bluesky_posted,
                        "chosen_on": getattr(reloaded, "chosen_on", "ABSENT"),
                        "stderr": res.stderr,
                    }

    # 1B: Refusal then successful replacement on streak 30 (p=0.99 -> fresh single) across run_day, run, orchestrate
    if target == "branch":
        recovery = {}
        for channel in ("run_day", "run", "orchestrate"):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                with offline_harness(root) as h:
                    set_uncovered_2026_auto(h.picks_dir, streak=0)
                    assert runner.invoke(cli.cli, [
                        "preview", "--date", DAY,
                        "--picks-dir", str(h.picks_dir),
                        "--models-dir", h.config["orchestrator"]["models_dir"],
                    ], catch_exceptions=False).exit_code == 0
                    pick_file = h.picks_dir / f"{DAY}.json"
                    preview_bytes = pick_file.read_bytes()
                    set_dated_obs(h.picks_dir, streak=30, best=30, saver="used")
                    single_frame = h.frame.assign(p_game_hit=[0.99, 0.81])

                    if channel == "run_day":
                        h.cascade.return_value = (None, None)
                        err1 = io.StringIO()
                        with redirect_stderr(err1):
                            scheduler.run_day(date=DAY, config=h.config)
                        after_fail_bytes = pick_file.read_bytes()
                        after_fail_locked = scheduler.load_state(DAY, h.picks_dir).pick_locked
                        after_fail_dec = load_decision(DAY, h.picks_dir)
                        after_fail_dms = len(h.pick_dm.call_args_list)
                        after_fail_alerts = [c.args[1] for c in h.health_dm.call_args_list]

                        # Now cascade recovers and chooses again
                        h.cascade.return_value = (single_frame, "offline")
                        scheduler.run_day(date=DAY, config=h.config)
                        final_pick = picks.load_pick(DAY, h.picks_dir)
                        final_dec = load_decision(DAY, h.picks_dir)
                        recovery[channel] = {
                            "after_fail_bytes_identical": after_fail_bytes == preview_bytes,
                            "after_fail_locked": after_fail_locked,
                            "after_fail_dec": after_fail_dec,
                            "after_fail_dms": after_fail_dms,
                            "after_fail_alerts_count": len(after_fail_alerts),
                            "final_delivered": picks.pick_was_delivered(final_pick),
                            "final_single": final_pick.double_down is None,
                            "final_chosen_on": final_pick.chosen_on,
                            "final_dec_status": final_dec["delivery_status"],
                            "final_dm_text": h.pick_dm.call_args.args[1],
                        }
                    else:
                        # First run in Warmup -> refuses reuse
                        h.statuses[101] = {"abstract": "P", "detailed": "Warmup", "code": "PW"}
                        h.statuses[102] = {"abstract": "P", "detailed": "Warmup", "code": "PW"}
                        h.pipeline.return_value = single_frame
                        h.cascade.return_value = (single_frame, "offline")
                        cfg_path = write_toml(h.config, root / "cfg.toml")
                        args = (["run", "--date", DAY, "--picks-dir", str(h.picks_dir),
                                 "--models-dir", h.config["orchestrator"]["models_dir"]]
                                if channel == "run"
                                else ["orchestrate", "--date", DAY, "--config", str(cfg_path)])
                        r1 = runner.invoke(cli.cli, args)
                        mid_pick = picks.load_pick(DAY, h.picks_dir)
                        assert r1.exit_code == 1 and len(h.post_bsky.call_args_list) == 0
                        assert mid_pick.chosen_on["required"] is False

                        # Second run with Scheduled -> fresh choice -> stamps required=True and posts!
                        h.statuses[101] = {"abstract": "P", "detailed": "Scheduled", "code": "S"}
                        h.statuses[102] = {"abstract": "P", "detailed": "Scheduled", "code": "S"}
                        r2 = runner.invoke(cli.cli, args)
                        final_pick = picks.load_pick(DAY, h.picks_dir)
                        recovery[channel] = {
                            "r1_exit": r1.exit_code,
                            "r1_stderr": r1.stderr,
                            "r2_exit": r2.exit_code,
                            "post_count": len(h.post_bsky.call_args_list),
                            "post_text": h.post_bsky.call_args.args[0],
                            "final_single": final_pick.double_down is None,
                            "final_chosen_on": final_pick.chosen_on,
                        }
        out["recovery"] = recovery
    return out


def run_item_2_attacks() -> dict:
    """Branch-only deep attacks on 'saved' definition and all 4 select_pick reuse returns across all paths."""
    from click.testing import CliRunner
    from bts import cli, contest_state, orchestrator, picks, scheduler, strategy
    from bts.daily_decision import load_decision

    runner = CliRunner()
    results = {}

    # Part 1: Attacking the saved-pick predicate in _deliver_and_lock_pick
    for case in (
        "same_cycle_fresh",
        "reused_same_object",
        "decision_set_copied_daily",
        "decision_set_reloaded_daily",
        "earlier_cycle_selection",
        "decision_set_on_skip",
    ):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with offline_harness(root) as h:
                set_dated_obs(h.picks_dir, streak=7, best=9, saver="used")
                _, sel1, _ = orchestrator.run_and_pick(h.config, DAY)
                assert sel1.decision is not None and sel1.pick_result is not None
                daily = sel1.pick_result.daily
                picks.save_pick(daily, h.picks_dir)

                # Change live dated observation so saved_pick_refusal WILL refuse if called
                set_dated_obs(h.picks_dir, streak=7, best=11, saver="used")

                if case == "same_cycle_fresh":
                    passed_daily, passed_sel = daily, sel1
                elif case == "reused_same_object":
                    sel1.decision = None
                    passed_daily, passed_sel = daily, sel1
                elif case == "decision_set_copied_daily":
                    copied = copy.copy(daily)
                    passed_daily, passed_sel = copied, sel1
                elif case == "decision_set_reloaded_daily":
                    reloaded = picks.load_pick(DAY, h.picks_dir)
                    passed_daily, passed_sel = reloaded, sel1
                elif case == "earlier_cycle_selection":
                    # Cycle 2 produces its own pick object on streak 0 (unrecorded or old), paired with sel1
                    other = picks.load_pick(DAY, h.picks_dir)
                    passed_daily, passed_sel = other, sel1
                elif case == "decision_set_on_skip":
                    # Real skip selection (decision is not None, pick_result is None) paired with cached daily
                    set_dated_obs(h.picks_dir, streak=56, best=56, saver="used")
                    h.cascade.return_value = (h.frame.assign(p_game_hit=[0.10, 0.09]), "offline")
                    _, skip_sel, _ = orchestrator.run_and_pick(h.config, DAY)
                    assert skip_sel.decision is not None and skip_sel.pick_result is None
                    passed_daily, passed_sel = daily, skip_sel

                st = make_sched_state()
                with redirect_stderr(io.StringIO()):
                    ok = scheduler._deliver_and_lock_pick(
                        passed_daily, h.config, h.picks_dir, st, DAY, "attack", selection=passed_sel
                    )
                results[case] = {
                    "delivered": ok,
                    "locked": st.pick_locked,
                    "dm_called": h.pick_dm.called,
                    "alert_called": h.health_dm.called,
                }

    # Part 2: Overrun branch in run_day: match, mismatch + failed final, and mismatch + fresh final replacement
    for ov_case in ("match", "mismatch_failed_final", "mismatch_fresh_final"):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with offline_harness(root) as h:
                set_dated_obs(h.picks_dir, streak=7, best=9, saver="used")
                proj_frame = h.frame.copy()
                proj_frame.loc[1, "flags"] = "PROJECTED"
                if ov_case == "mismatch_failed_final":
                    h.cascade.side_effect = [(proj_frame, "offline"), (None, None)]
                else:
                    h.cascade.return_value = (proj_frame, "offline")
                orig_check = scheduler.run_single_check

                def wrap_check(*a, **kw):
                    r = orig_check(*a, **kw)
                    if r["pick_result"] is not None:
                        r["lock_decision"] = scheduler._lock_decision_from_predictions(
                            proj_frame, r["pick_result"].daily, DAY, 0.03
                        )
                        if ov_case != "match":
                            set_dated_obs(h.picks_dir, streak=7, best=12, saver="used")
                    return r

                with patch.object(scheduler, "run_single_check", side_effect=wrap_check), redirect_stderr(io.StringIO()):
                    scheduler.run_day(date=DAY, config=h.config)
                final_loaded = picks.load_pick(DAY, h.picks_dir)
                results[f"overrun_{ov_case}"] = {
                    "delivered": h.pick_dm.called,
                    "locked": scheduler.load_state(DAY, h.picks_dir).pick_locked,
                    "alerted": h.health_dm.called,
                    "chosen_best": final_loaded.chosen_on["dated"]["best_streak"],
                }

    # Part 3: Drive select_pick's 4 reuse returns into EVERY reachable delivery path with matching & non-matching
    reuse_matrix = {}

    # 3A: Lineup path via real run_single_check -> _deliver_and_lock_pick (unavailable reuse 546 -> Scheduled on lock check)
    for matches in (True, False):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with offline_harness(root) as h:
                set_dated_obs(h.picks_dir, streak=7, best=9, saver="used")
                st_live = contest_state.load_decision_streak_state(h.picks_dir, now=NOW, require_contest_state=True)
                seed = picks.DailyPick(
                    date=DAY, run_time=NOW.isoformat(),
                    pick=picks.pick_from_row(h.frame.iloc[0]),
                    double_down=None, runner_up=None,
                    chosen_on=picks.chosen_on_record(st_live, required=matches),
                )
                picks.save_pick(seed, h.picks_dir)

                warmup = {101: {"abstract": "P", "detailed": "Warmup", "code": "PW"},
                          102: {"abstract": "P", "detailed": "Warmup", "code": "PW"}}
                sched = {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
                         102: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}
                # Calls in run_single_check:
                # 1. pre-cascade classify_pick_lock_state -> warmup (unlocked)
                # 2. run_and_pick -> warmup -> select_pick hits line 546 (unavailable, unlocked, decision=None)!
                # 3. _lock_decision_from_predictions -> sched -> should_lock=True!
                seq = Mock(side_effect=[warmup, warmup, sched, sched])
                with (
                    patch.object(picks, "get_game_statuses_detailed", seq),
                    patch.object(strategy, "get_game_statuses_detailed", seq),
                    redirect_stderr(io.StringIO()),
                ):
                    chk = scheduler.run_single_check(DAY, [101, 102], set(), h.config, 0.03)
                    assert chk["should_post"] is True
                    assert chk["selection"].decision is None
                    assert chk["selection"].pick_result.daily.pick.batter_id == seed.pick.batter_id
                    st = make_sched_state()
                    ok = scheduler._deliver_and_lock_pick(
                        chk["pick_result"].daily, h.config, h.picks_dir, st, DAY, "lineup",
                        selection=chk["selection"],
                    )
                reuse_matrix[f"lineup_unavailable_match_{matches}"] = {
                    "ok": ok, "dm": h.pick_dm.called, "locked": st.pick_locked, "alert": h.health_dm.called
                }

    # 3B: In-loop & final fallback via real _refresh_pick_at_fallback_decision for locked(508), coarse(517), unavailable(546)
    for reuse_kind in ("locked_508", "coarse_517", "unavailable_546"):
        for fb_site in ("in_loop", "final"):
            for matches in (True, False):
                with tempfile.TemporaryDirectory() as td:
                    root = Path(td)
                    with offline_harness(root) as h:
                        set_dated_obs(h.picks_dir, streak=7, best=9, saver="used")
                        st_live = contest_state.load_decision_streak_state(h.picks_dir, now=NOW, require_contest_state=True)
                        seed = picks.DailyPick(
                            date=DAY, run_time=NOW.isoformat(),
                            pick=picks.pick_from_row(h.frame.iloc[0]),
                            double_down=None, runner_up=None,
                            chosen_on=picks.chosen_on_record(st_live, required=matches),
                        )
                        picks.save_pick(seed, h.picks_dir)
                        cached = picks.load_pick(DAY, h.picks_dir)

                        warmup = {101: {"abstract": "P", "detailed": "Warmup", "code": "PW"},
                                  102: {"abstract": "P", "detailed": "Warmup", "code": "PW"}}
                        sched = {101: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
                                 102: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}

                        if reuse_kind == "locked_508":
                            # run_and_pick detailed -> sched; classify_pick_lock_state detailed+coarse fail -> locked=True(508);
                            # _lock_decision_from_predictions detailed -> sched
                            det_mock = Mock(side_effect=[sched, OSError("det down"), sched, sched])
                            coarse_picks = Mock(side_effect=OSError("coarse down"))
                            ctx_patches = (
                                patch.object(picks, "get_game_statuses_detailed", det_mock),
                                patch.object(picks, "get_game_statuses", coarse_picks),
                            )
                        elif reuse_kind == "coarse_517":
                            # run_and_pick detailed raises; classify_pick_lock_state detailed -> sched (unlocked);
                            # select_pick coarse raises -> locked=True(517); _lock_decision detailed -> sched
                            det_mock = Mock(side_effect=[OSError("det down"), sched, sched, sched])
                            coarse_strat = Mock(side_effect=OSError("coarse down"))
                            ctx_patches = (
                                patch.object(picks, "get_game_statuses_detailed", det_mock),
                                patch.object(strategy, "get_game_statuses", coarse_strat),
                            )
                        else:
                            det_mock = Mock(return_value=warmup)
                            ctx_patches = (
                                patch.object(picks, "get_game_statuses_detailed", det_mock),
                                patch.object(strategy, "get_game_statuses_detailed", det_mock),
                            )

                        with ctx_patches[0], ctx_patches[1], redirect_stderr(io.StringIO()):
                            ref = scheduler._refresh_pick_at_fallback_decision(h.config, DAY, cached, 0.03)
                            assert ref.selection is not None and ref.selection.decision is None
                            assert ref.daily is ref.selection.pick_result.daily
                            assert ref.selection.pick_result.locked is (reuse_kind != "unavailable_546")
                            if fb_site == "in_loop":
                                plan = scheduler.plan_fallback_action(
                                    now=NOW, cutoff=picks.submission_cutoff_et(ref.daily),
                                    should_post=ref.should_post, should_post_ungated=ref.should_post_ungated,
                                    block_reason=ref.block_reason, contender_game_pk=ref.contender_game_pk,
                                    contender_game_pks=ref.contender_game_pks, remaining_runs=[],
                                    confirmed_sides=set(), budget_min=15.0, operator_reserve_min=10.0,
                                )
                                assert plan.action == "deliver"
                            st = make_sched_state()
                            ok = scheduler._deliver_and_lock_pick(
                                ref.daily, h.config, h.picks_dir, st, DAY, fb_site, selection=ref.selection
                            )
                        reuse_matrix[f"{fb_site}_{reuse_kind}_match_{matches}"] = {
                            "ok": ok, "dm": h.pick_dm.called, "locked": st.pick_locked, "alert": h.health_dm.called
                        }

    # 3C: Catch-up in bts run & bts orchestrate when pick was earlier DM'd (notification_sent=True, bluesky_posted=False)
    for cmd in ("run", "orchestrate"):
        for matches in (True, False):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                with offline_harness(root) as h:
                    set_dated_obs(h.picks_dir, streak=7, best=9, saver="used")
                    st_live = contest_state.load_decision_streak_state(h.picks_dir, now=NOW, require_contest_state=True)
                    seed = picks.DailyPick(
                        date=DAY, run_time=NOW.isoformat(),
                        pick=picks.pick_from_row(h.frame.iloc[0]),
                        double_down=None, runner_up=None,
                        notification_sent=True, notification_channel="bluesky_dm",
                        notification_id="dm://prior", delivered_at=NOW.isoformat(),
                        bluesky_posted=False,
                        chosen_on=picks.chosen_on_record(st_live, required=True),
                    )
                    picks.save_pick(seed, h.picks_dir)
                    if not matches:
                        set_dated_obs(h.picks_dir, streak=8, best=9, saver="used")
                    cfg_path = write_toml(h.config, root / "cfg.toml")
                    args = (["run", "--date", DAY, "--picks-dir", str(h.picks_dir),
                             "--models-dir", h.config["orchestrator"]["models_dir"]]
                            if cmd == "run" else ["orchestrate", "--date", DAY, "--config", str(cfg_path)])
                    res = runner.invoke(cli.cli, args)
                    reuse_matrix[f"dm_then_{cmd}_catchup_match_{matches}"] = {
                        "exit_code": res.exit_code,
                        "posted": h.post_bsky.called,
                        "stderr": res.stderr,
                    }

    results["reuse_matrix"] = reuse_matrix
    return results


def run_item_5_after_refusal() -> dict:
    """Branch-only checks for everything a refusal leaves behind, alert failure/resurface, and Row 10."""
    from click.testing import CliRunner
    from bts import cli, picks, scheduler
    from bts.daily_decision import load_decision

    runner = CliRunner()
    out: dict = {}

    # 5A: Alert DM transient failure then resurface on next refusal; then dedup on third refusal
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with offline_harness(root) as h:
            set_dated_obs(h.picks_dir, streak=12, best=12, saver="active")
            unrecorded = picks.DailyPick(
                date=DAY, run_time=NOW.isoformat(),
                pick=picks.pick_from_row(h.frame.iloc[0]), double_down=None, runner_up=None,
            )
            pfile = picks.save_pick(unrecorded, h.picks_dir)
            raw_before = pfile.read_bytes()
            st = make_sched_state()

            h.health_dm.side_effect = [RuntimeError("dm network error"), "dm://ok-2", "dm://unexpected-3"]
            rets = []
            stderrs = []
            for _ in range(3):
                buf = io.StringIO()
                with redirect_stderr(buf):
                    rets.append(scheduler._deliver_and_lock_pick(unrecorded, h.config, h.picks_dir, st, DAY, "fallback"))
                stderrs.append(buf.getvalue())
            out["alert_resurface_and_dedup"] = {
                "returns": rets,
                "health_dm_attempts": h.health_dm.call_count,
                "second_dm_has_resurface_banner": "Previous health alert DM FAILED" in h.health_dm.call_args_list[1].args[1],
                "all_stderr_have_blocked": all("CONTEST STATE ERROR — pick delivery blocked:" in s for s in stderrs),
                "bytes_identical": pfile.read_bytes() == raw_before,
                "locked": st.pick_locked,
                "decision": load_decision(DAY, h.picks_dir),
            }

    # 5B: Non-object health status files ("[]", "null", "123", '"str"')
    non_obj_results = {}
    for bad_json in ("[]", "null", "123", '"str"'):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with offline_harness(root) as h:
                set_dated_obs(h.picks_dir, streak=7, best=9, saver="used")
                st_path = h.picks_dir.parent / "health_state" / "health_dm_delivery_status.json"
                st_path.parent.mkdir(parents=True, exist_ok=True)
                st_path.write_text(bad_json)
                d = picks.DailyPick(
                    date=DAY, run_time=NOW.isoformat(),
                    pick=picks.pick_from_row(h.frame.iloc[0]), double_down=None, runner_up=None,
                )
                pf = picks.save_pick(d, h.picks_dir)
                raw = pf.read_bytes()
                st = make_sched_state()
                buf = io.StringIO()
                with redirect_stderr(buf):
                    ret = scheduler._deliver_and_lock_pick(d, h.config, h.picks_dir, st, DAY, "fallback")
                non_obj_results[bad_json] = {
                    "ret": ret,
                    "bytes_unchanged": pf.read_bytes() == raw,
                    "status_unchanged": st_path.read_text() == bad_json,
                    "locked": st.pick_locked,
                    "decision": load_decision(DAY, h.picks_dir),
                    "stderr_reported": "Contest-state alert failed:" in buf.getvalue() or bad_json == "null",
                }
    out["non_object_status"] = non_obj_results

    # 5C: Row 10 _alert_missed_pick across live settings and positive / zero / sub-half / negative minutes
    missed_cases = {}
    for setting in ("scheduler_only", "health_only", "both_private", "not_live"):
        for mins in (12.4, 0.2, 0.0, -0.1, -5.0):
            with tempfile.TemporaryDirectory() as td:
                root = Path(td)
                with offline_harness(root) as h:
                    h.config["scheduler"]["contest_state_required"] = setting in {"scheduler_only", "both_private"}
                    h.config["health_checks"]["contest_state_expected"] = setting in {"health_only", "both_private"}
                    if setting == "both_private":
                        h.config["scheduler"]["pick_delivery"] = "private"
                    d = picks.DailyPick(
                        date=DAY, run_time=NOW.isoformat(),
                        pick=picks.pick_from_row(h.frame.iloc[0]), double_down=None, runner_up=None,
                    )
                    scheduler._alert_missed_pick(h.config, d, mins)
                    missed_cases[f"{setting}_{mins}"] = h.health_dm.call_args.args[1]
    out["missed_pick_messages"] = missed_cases

    # 5D: Next morning check-results + check-pick-entered + day-outcome after refusal
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        with offline_harness(root) as h:
            h.config["scheduler"]["entry_intent"] = "enter"
            set_uncovered_2026_auto(h.picks_dir, streak=0)
            runner.invoke(cli.cli, [
                "preview", "--date", DAY, "--picks-dir", str(h.picks_dir),
                "--models-dir", h.config["orchestrator"]["models_dir"],
            ], catch_exceptions=False)
            set_dated_obs(h.picks_dir, streak=20, best=20, saver="active")
            h.cascade.return_value = (None, None)
            with redirect_stderr(io.StringIO()):
                scheduler.run_day(date=DAY, config=h.config)

            # check-pick-entered with --dm-recipient set: must NOT DM batter names!
            r_ent = runner.invoke(cli.cli, [
                "check-pick-entered", "--picks-dir", str(h.picks_dir),
                "--now-et", f"{DAY}T18:45:00", "--expected-username", "offline",
                "--dm-recipient", "operator.test",
            ], catch_exceptions=False)

            # check-results next morning: must NOT score uncommitted pick!
            h.clock[0] = REAL_DATETIME.fromisoformat("2027-07-16T01:05:00-04:00")
            r_chk = runner.invoke(cli.cli, [
                "check-results", "--date", DAY, "--picks-dir", str(h.picks_dir),
            ], catch_exceptions=False)

            # day-outcome next morning at 08:30 ET: must report no_decision!
            h.clock[0] = REAL_DATETIME.fromisoformat("2027-07-16T08:30:00-04:00")
            cfg_p = write_toml(h.config, root / "do.toml")
            r_out = runner.invoke(cli.cli, [
                "day-outcome", "--date", DAY, "--config", str(cfg_p),
                "--picks-dir", str(h.picks_dir), "--expected-username", "offline",
            ], catch_exceptions=False)

            out["next_morning"] = {
                "entry_exit": r_ent.exit_code,
                "entry_stdout": r_ent.stdout,
                "entry_dm_called": h.pick_dm.called,
                "check_results_exit": r_chk.exit_code,
                "check_results_stdout": r_chk.stdout,
                "streak_file_exists": (h.picks_dir / "streak.json").exists(),
                "day_outcome_exit": r_out.exit_code,
                "day_outcome_state": json.loads(r_out.stdout)["state"],
            }

    return out


def run_items_6_7_parity_and_controls(target: str) -> dict:
    """Run identical scenarios on base and branch for Row 8, Row 11, loader matrix, and R33 controls."""
    from click.testing import CliRunner
    from bts import cli, contest_state, picks, scheduler
    from bts.daily_decision import load_decision

    runner = CliRunner()
    summary: dict = {}

    # 6A: Row 8 unchanged paths in _deliver_and_lock_pick (with unrecorded / mismatched pick)
    row8 = {}
    for scenario in ("delivered_dm", "delivered_public", "crash_attempted", "private",
                     "optional_dm", "optional_public", "all_void", "already_late", "loader_error"):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with offline_harness(root) as h:
                if scenario != "loader_error":
                    set_dated_obs(h.picks_dir, streak=15, best=15, saver="active")
                if scenario == "private":
                    h.config["scheduler"]["pick_delivery"] = "private"
                elif scenario == "optional_dm":
                    h.config["scheduler"]["contest_state_required"] = False
                    h.config["scheduler"]["pick_delivery"] = "dm"
                elif scenario == "optional_public":
                    h.config["scheduler"]["contest_state_required"] = False
                    h.config["scheduler"]["pick_delivery"] = "public"
                elif scenario == "delivered_public":
                    h.config["scheduler"]["pick_delivery"] = "public"

                d = picks.DailyPick(
                    date=DAY, run_time=NOW.isoformat(),
                    pick=picks.pick_from_row(h.frame.iloc[0]), double_down=None, runner_up=None,
                )
                if scenario == "delivered_dm":
                    d.notification_sent = True
                    d.notification_channel = "bluesky_dm"
                    d.notification_id = "dm://already"
                    d.delivered_at = NOW.isoformat()
                elif scenario == "delivered_public":
                    d.bluesky_posted = True
                    d.bluesky_uri = "at://already"
                    d.delivered_at = NOW.isoformat()
                elif scenario == "crash_attempted":
                    d.delivery_attempted = True
                elif scenario == "already_late":
                    h.clock[0] = REAL_DATETIME.fromisoformat(f"{DAY}T19:08:00-04:00")

                picks.save_pick(d, h.picks_dir)
                st = make_sched_state()

                extra_ctx = (
                    patch.object(scheduler, "_void_committed_games", side_effect=lambda *_a: {101: "Postponed"})
                    if scenario == "all_void" else nullcontext()
                )
                judge_ctx = (
                    patch.object(picks, "saved_pick_refusal", side_effect=AssertionError("judge called on row 8!"))
                    if target == "branch" else nullcontext()
                )
                with extra_ctx, judge_ctx, redirect_stderr(io.StringIO()):
                    ret = scheduler._deliver_and_lock_pick(d, h.config, h.picks_dir, st, DAY, scenario)

                dec = load_decision(DAY, h.picks_dir)
                persisted = picks.load_pick(DAY, h.picks_dir)
                pdict = asdict(persisted) if persisted else None
                if pdict is not None:
                    pdict.pop("chosen_on", None)
                    pdict.pop("delivered_at", None)
                if dec is not None:
                    dec = {k: v for k, v in dec.items() if k not in {"recorded_at"}}
                row8[scenario] = {
                    "ret": ret,
                    "dm_sends": [c.args[1] for c in h.pick_dm.call_args_list],
                    "post_sends": [c.args[0] for c in h.post_bsky.call_args_list],
                    "health_alerts": [
                        c.args[1].replace(str(h.picks_dir), "<PICKS_DIR>")
                        for c in h.health_dm.call_args_list
                    ],
                    "locked": st.pick_locked,
                    "decision": dec,
                    "pick_exists": persisted is not None,
                    "pick_flags": (
                        {
                            "bluesky_posted": pdict["bluesky_posted"],
                            "notification_sent": pdict["notification_sent"],
                            "delivery_attempted": pdict["delivery_attempted"],
                        } if pdict else None
                    ),
                }
    summary["row8"] = row8

    # 6B: load_decision_streak_state matrix over switch x manual x auto x require x saver_flag
    loader_cells = {}
    for sw in ("absent", "automatic", "revoked"):
        for man in ("absent", "dated_today", "dated_yesterday", "legacy_unexpired", "legacy_expired", "invalid_dated"):
            for aut in ("absent", "covered_v2", "uncovered_v2", "v1"):
                for req in (False, True):
                    for sflag in ("absent", "active", "used"):
                        key = f"{sw}|{man}|{aut}|req={req}|sf={sflag}"
                        with tempfile.TemporaryDirectory() as td:
                            pdir = Path(td)
                            if sw != "absent":
                                write_account(pdir, "season_contract.json", {
                                    "schema_version": "bts_season_contract_v1",
                                    "season": 2027,
                                    "status": sw,
                                    "activation_id": "b" * 32,
                                    "switched_at": NOW.isoformat(),
                                    "revoked_at": NOW.isoformat() if sw == "revoked" else None,
                                    "note": "rev" if sw == "revoked" else None,
                                })
                            if man == "dated_today":
                                set_dated_obs(pdir, now=NOW, streak=11, best=14, saver="not_earned")
                            elif man == "dated_yesterday":
                                set_dated_obs(pdir, now=NOW - dt_mod.timedelta(days=1), streak=4, best=9, saver="active")
                            elif man == "legacy_unexpired":
                                write_account(pdir, "contest_streak.manual.json", {
                                    "schema_version": "bts_contest_streak_manual_v2",
                                    "active_streak": 6,
                                    "best_streak": 12,
                                    "source": "manual_cli",
                                    "source_date": DAY,
                                    "recorded_at": NOW.isoformat(),
                                    "override_expires_at": (NOW + dt_mod.timedelta(hours=5)).isoformat(),
                                })
                            elif man == "legacy_expired":
                                write_account(pdir, "contest_streak.manual.json", {
                                    "schema_version": "bts_contest_streak_manual_v2",
                                    "active_streak": 6,
                                    "source": "manual_cli",
                                    "source_date": DAY,
                                    "recorded_at": NOW.isoformat(),
                                    "override_expires_at": (NOW - dt_mod.timedelta(hours=5)).isoformat(),
                                })
                            elif man == "invalid_dated":
                                write_account(pdir, "contest_streak.manual.json", {
                                    "schema_version": "bts_contest_streak_manual_v3",
                                    "effective_date": DAY,
                                    "season": 2027,
                                })
                            if aut == "covered_v2":
                                write_account(pdir, "contest_streak.json", {
                                    "schema_version": "bts_contest_streak_auto_v2",
                                    "season": 2027,
                                    "activation_id": "b" * 32,
                                    "active_streak": 9,
                                    "best_streak": 15,
                                    "source": "mlb_bts_profile",
                                    "source_date": DAY,
                                    "recorded_at": NOW.isoformat(),
                                })
                            elif aut == "uncovered_v2":
                                write_account(pdir, "contest_streak.json", {
                                    "schema_version": "bts_contest_streak_auto_v2",
                                    "season": 2027,
                                    "activation_id": "f" * 32,
                                    "active_streak": 9,
                                    "best_streak": 15,
                                    "source": "mlb_bts_profile",
                                    "source_date": DAY,
                                    "recorded_at": NOW.isoformat(),
                                })
                            elif aut == "v1":
                                write_account(pdir, "contest_streak.json", {
                                    "schema_version": "bts_contest_streak_auto_v1",
                                    "active_streak": 3,
                                    "source": "mlb_bts_profile",
                                    "source_date": DAY,
                                    "recorded_at": NOW.isoformat(),
                                })
                            if sflag != "absent":
                                write_account(pdir, "saver_state.json", {
                                    "season": 2027, "state": sflag, "source": "cli", "updated_at": NOW.isoformat()
                                })
                            try:
                                st = contest_state.load_decision_streak_state(
                                    pdir, require_contest_state=req, now=NOW
                                )
                                d_st = asdict(st)
                                obs_kind = d_st.pop("observation_kind", None)
                                dated_id = d_st.pop("dated_identity", None)
                                if d_st.get("contest_source_date") is not None:
                                    d_st["contest_source_date"] = d_st["contest_source_date"].isoformat()
                                loader_cells[key] = {"ok": True, "fields": d_st, "kind": obs_kind, "dated": dated_id}
                            except contest_state.ContestStateError as exc:
                                import shlex
                                err_txt = (
                                    str(exc)
                                    .replace(shlex.quote(str(pdir)), "<PICKS_DIR>")
                                    .replace(str(pdir), "<PICKS_DIR>")
                                )
                                loader_cells[key] = {"ok": False, "err": err_txt}
    summary["loader_cells"] = loader_cells

    # 7: Ordinary-day R33 controls through real run_day (must match base!)
    r33 = {}
    for ctl in ("gap_fetch_fail_restart", "saver_flag_edit", "revoke_enable_fetch"):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            with offline_harness(root) as h:
                set_covered_2027_auto(h.picks_dir, now=NOW, streak=5)
                proj = h.frame.copy()
                proj.loc[1, "flags"] = "PROJECTED"
                h.cascade.return_value = (proj, "offline")
                with redirect_stderr(io.StringIO()):
                    chk = scheduler.run_single_check(DAY, [101, 102], set(), h.config, 0.03)
                assert not chk["should_post"] and not h.pick_dm.called

                if ctl == "gap_fetch_fail_restart":
                    set_covered_2027_auto(h.picks_dir, now=NOW, streak=18)
                elif ctl == "saver_flag_edit":
                    write_account(h.picks_dir, "saver_state.json", {
                        "season": 2027, "state": "active", "source": "cli", "updated_at": NOW.isoformat()
                    })
                elif ctl == "revoke_enable_fetch":
                    assert runner.invoke(cli.cli, [
                        "season-contract", "revoke", "--season", "2027",
                        "--reason", "drill", "--picks-dir", str(h.picks_dir),
                    ], catch_exceptions=False).exit_code == 0
                    assert runner.invoke(cli.cli, [
                        "season-contract", "enable", "--season", "2027",
                        "--picks-dir", str(h.picks_dir),
                    ], catch_exceptions=False).exit_code == 0
                    new_act = json.loads((h.picks_dir / "account_state" / "season_contract.json").read_text())["activation_id"]
                    set_covered_2027_auto(h.picks_dir, now=NOW, streak=14, activation_id=new_act)

                h.cascade.return_value = (None, None)
                with redirect_stderr(io.StringIO()):
                    scheduler.run_day(date=DAY, config=h.config)
                dec = load_decision(DAY, h.picks_dir)
                r33[ctl] = {
                    "dm_sends": [c.args[1] for c in h.pick_dm.call_args_list],
                    "alerts": [c.args[1] for c in h.health_dm.call_args_list],
                    "locked": scheduler.load_state(DAY, h.picks_dir).pick_locked,
                    "dec_status": dec["delivery_status"] if dec else None,
                }
    summary["r33"] = r33
    return summary


def worker_main(target: str) -> None:
    install_no_network_hook()
    payload = {
        "target": target,
        "item_1": run_item_1_hole(target),
        "items_6_7": run_items_6_7_parity_and_controls(target),
    }
    if target == "branch":
        payload["item_2"] = run_item_2_attacks()
        payload["item_5"] = run_item_5_after_refusal()
    print(json.dumps(payload))


def run_worker_subprocess(target: str, repo_root: Path) -> dict:
    env = os.environ.copy()
    env["TZ"] = "America/New_York"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    if target == "base":
        base_root = repo_root / ".codex-review" / "cs5a-review" / "scratch" / "base_4cab4212"
        env["PYTHONPATH"] = f"{base_root / 'src'}:{base_root}"
    else:
        env["PYTHONPATH"] = f"{repo_root / 'src'}:{repo_root}"
    proc = subprocess.run(
        [str(repo_root / ".venv" / "bin" / "python"), __file__, "--worker", target],
        cwd=str(repo_root),
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(proc.stdout)


def verify_all(base: dict, branch: dict) -> None:
    # 1. Verify hole on base vs closed on branch
    b_sched = base["item_1"]["scheduler"]
    assert b_sched["returned"] is True
    assert len(b_sched["pick_dm_calls"]) == 1 and "Streak: 56" in b_sched["pick_dm_calls"][0]
    assert "Steven Kwan" in b_sched["pick_dm_calls"][0] and "Juan Soto" in b_sched["pick_dm_calls"][0]
    assert b_sched["pick_locked"] is True and b_sched["decision_status"] == "delivered"

    br_sched = branch["item_1"]["scheduler"]
    assert br_sched["returned"] is False
    assert br_sched["pick_dm_calls"] == [] and br_sched["post_calls"] == []
    assert br_sched["bytes_identical"] is True
    assert br_sched["delivery_attempted"] is False and br_sched["notification_sent"] is False
    assert br_sched["pick_locked"] is False and br_sched["decision_status"] is None
    assert len(br_sched["health_dm_calls"]) == 1
    alert_msg = br_sched["health_dm_calls"][0]
    assert "Saved pick for 2027-07-15 was not delivered" in alert_msg
    assert "chosen without the live check (for example by the preview)" in alert_msg
    assert "dated (effective date 2027-07-15, active streak 56, best streak 56, saver used)" in alert_msg
    assert "A later scheduled check or fallback refresh may choose a replacement, if one remains." in alert_msg

    for ch in ("run_warmup", "run_locked", "orchestrate_warmup", "orchestrate_locked"):
        assert base["item_1"][ch]["exit_code"] == 0
        assert len(base["item_1"][ch]["post_calls"]) == 1
        assert "Streak: 56" in base["item_1"][ch]["post_calls"][0]
        assert base["item_1"][ch]["bluesky_posted"] is True

        assert branch["item_1"][ch]["exit_code"] == 1
        assert branch["item_1"][ch]["post_calls"] == []
        assert branch["item_1"][ch]["bluesky_posted"] is False
        assert branch["item_1"][ch]["chosen_on"]["required"] is False
        prefix = "bts run: no pick: " if ch.startswith("run_") else "bts orchestrate: no pick: "
        assert prefix in branch["item_1"][ch]["stderr"]

    # Recovery checks
    rec = branch["item_1"]["recovery"]
    assert rec["run_day"]["after_fail_bytes_identical"] is True
    assert rec["run_day"]["after_fail_locked"] is False
    assert rec["run_day"]["after_fail_dec"] is None
    assert rec["run_day"]["after_fail_dms"] == 0
    assert rec["run_day"]["after_fail_alerts_count"] == 2  # 1 contest_state + 1 missed_pick
    assert rec["run_day"]["final_delivered"] is True and rec["run_day"]["final_single"] is True
    assert rec["run_day"]["final_chosen_on"]["required"] is True
    assert rec["run_day"]["final_chosen_on"]["dated"]["active_streak"] == 30
    assert "Streak: 30" in rec["run_day"]["final_dm_text"]
    for ch in ("run", "orchestrate"):
        assert rec[ch]["r1_exit"] == 1 and rec[ch]["r2_exit"] == 0
        assert rec[ch]["post_count"] == 1 and "Streak: 30" in rec[ch]["post_text"]
        assert rec[ch]["final_single"] is True and rec[ch]["final_chosen_on"]["required"] is True

    # 2. Attacks on saved definition
    i2 = branch["item_2"]
    assert i2["same_cycle_fresh"] == {"delivered": True, "locked": True, "dm_called": True, "alert_called": False}
    for saved_case in (
        "reused_same_object",
        "decision_set_copied_daily",
        "decision_set_reloaded_daily",
        "earlier_cycle_selection",
        "decision_set_on_skip",
    ):
        assert i2[saved_case] == {"delivered": False, "locked": False, "dm_called": False, "alert_called": True}, saved_case
    assert i2["overrun_match"] == {"delivered": True, "locked": True, "alerted": False, "chosen_best": 9}
    assert i2["overrun_mismatch_failed_final"] == {"delivered": False, "locked": False, "alerted": True, "chosen_best": 9}
    assert i2["overrun_mismatch_fresh_final"] == {"delivered": True, "locked": True, "alerted": True, "chosen_best": 12}

    rm = i2["reuse_matrix"]
    for key, val in rm.items():
        expected_ok = key.endswith("_True")
        if "catchup" in key:
            assert val["exit_code"] == (0 if expected_ok else 1), (key, val)
            assert val["posted"] is expected_ok, (key, val)
        else:
            assert val == {"ok": expected_ok, "dm": expected_ok, "locked": expected_ok, "alert": not expected_ok}, (key, val)

    # 3. Item 5 after-refusal checks
    i5 = branch["item_5"]
    ard = i5["alert_resurface_and_dedup"]
    assert ard["returns"] == [False, False, False]
    assert ard["health_dm_attempts"] == 2  # 1st failed -> 2nd retried & succeeded -> 3rd deduplicated!
    assert ard["second_dm_has_resurface_banner"] is True
    assert ard["all_stderr_have_blocked"] is True
    assert ard["bytes_identical"] is True and ard["locked"] is False and ard["decision"] is None

    for bad_json, info in i5["non_object_status"].items():
        assert info["ret"] is False and info["bytes_unchanged"] is True
        assert info["locked"] is False and info["decision"] is None

    for key, msg in i5["missed_pick_messages"].items():
        if key.startswith("not_live_"):
            assert "Steven Kwan" in msg and "Deliver manually" in msg
        else:
            assert "Steven Kwan" not in msg and "Deliver manually" not in msg
            assert "bts season-contract status" in msg
            mins = float(key.rsplit("_", 1)[1])
            if mins >= 0:
                assert f"~{mins:.0f} min to first pitch" in msg
            else:
                assert "first pitch has passed" in msg

    nm = i5["next_morning"]
    assert nm["entry_exit"] == 0 and "not committed/locked" in nm["entry_stdout"]
    assert nm["entry_dm_called"] is False
    assert nm["check_results_exit"] == 0 and "not a committed pick" in nm["check_results_stdout"]
    assert nm["streak_file_exists"] is False
    assert nm["day_outcome_exit"] == 1 and nm["day_outcome_state"] == "no_decision"

    # 4. Items 6 & 7 exact parity between base and branch
    assert base["items_6_7"]["row8"] == branch["items_6_7"]["row8"]
    assert base["items_6_7"]["r33"] == branch["items_6_7"]["r33"]
    for ctl_name, ctl_val in branch["items_6_7"]["r33"].items():
        assert len(ctl_val["dm_sends"]) == 1 and ctl_val["alerts"] == []
        assert ctl_val["locked"] is True and ctl_val["dec_status"] == "delivered", ctl_name

    base_loader = base["items_6_7"]["loader_cells"]
    branch_loader = branch["items_6_7"]["loader_cells"]
    assert set(base_loader) == set(branch_loader)
    for k in base_loader:
        assert base_loader[k]["ok"] == branch_loader[k]["ok"], k
        if not base_loader[k]["ok"]:
            assert base_loader[k]["err"] == branch_loader[k]["err"], k
        else:
            assert base_loader[k]["fields"] == branch_loader[k]["fields"], k
            assert branch_loader[k]["kind"] in {"dated", "automatic", "legacy_manual", "model"}, k
            assert (branch_loader[k]["dated"] is not None) is (branch_loader[k]["kind"] == "dated"), k


def main() -> None:
    if len(sys.argv) == 3 and sys.argv[1] == "--worker":
        worker_main(sys.argv[2])
        return
    repo_root = Path(__file__).resolve().parents[2]
    base = run_worker_subprocess("base", repo_root)
    branch = run_worker_subprocess("branch", repo_root)
    verify_all(base, branch)
    out_path = Path(__file__).with_name("probe_1_2_5_6_7_results.json")
    out_path.write_text(json.dumps({
        "row8_cells": len(branch["items_6_7"]["row8"]),
        "loader_cells": len(branch["items_6_7"]["loader_cells"]),
        "r33_controls": list(branch["items_6_7"]["r33"].keys()),
        "reuse_cells": len(branch["item_2"]["reuse_matrix"]),
        "missed_pick_samples": {
            k: branch["item_5"]["missed_pick_messages"][k]
            for k in ("scheduler_only_12.4", "scheduler_only_0.0", "scheduler_only_-0.1", "not_live_0.0")
        },
        "status": "ALL_ASSERTIONS_PASSED",
    }, indent=2))
    print(f"probe_1_2_5_6_7 PASSED: {len(branch['items_6_7']['loader_cells'])} loader cells, "
          f"{len(branch['items_6_7']['row8'])} row-8 cells, "
          f"{len(branch['item_2']['reuse_matrix'])} reuse cells, "
          f"3 R33 controls, hole reproduced on base & closed on branch.")


if __name__ == "__main__":
    main()
