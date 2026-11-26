"""Slice 3 probe family 1: cross-change-set interactions on real scheduler and CLI code.

Adoptable as a pytest regression suite and runnable standalone via:
    nice -n 19 .venv/bin/python .carbon-review/whole-range/test_probe_interactions.py
No module-level import of bts.model.predict, no leaked process-wide state, no network.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
from datetime import date, datetime, time as dt_time, timedelta, timezone
import gzip
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from click.testing import CliRunner

from bts import (
    cli,
    contest_fetch,
    contest_state,
    daily_decision,
    day_outcome,
    dm,
    entry_receipt,
    orchestrator,
    picks,
    posting,
    saver_state,
    scheduler,
    season_contract,
    skip_policy_shadow,
    strategy,
)
from bts.data import build as data_build
from bts.features import compute as feat_compute, park_drag
from bts.health import (
    alert as health_alert,
    contest_state as health_contest_state,
    mdp_policy_alignment,
    pick_entry as health_pick_entry,
    post_failure as health_post_failure,
    probable_pitcher_cache as health_pp_cache,
    runner as health_runner,
)

ET = ZoneInfo("America/New_York")
REAL_DATETIME = datetime


def _make_clock(box_now: list[datetime]):
    class PinnedDatetime(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            current = box_now[0]
            if tz is None:
                return current.replace(tzinfo=None)
            return current.astimezone(tz)

    return PinnedDatetime


@contextmanager
def block_network():
    def _deny(*args, **kwargs):
        raise AssertionError(f"unexpected network call: {args[:2]}")

    with ExitStack() as stack:
        for method in ("connect", "connect_ex", "sendto"):
            stack.enter_context(patch.object(socket.socket, method, _deny))
        stack.enter_context(patch.object(socket, "getaddrinfo", _deny))
        yield


def make_predictions(day_iso: str, p1: float = 0.86, p2: float | None = 0.84,
                     gt1: str | None = None, gt2: str | None = None) -> pd.DataFrame:
    gt1 = gt1 or f"{day_iso}T23:10:00Z"
    gt2 = gt2 or f"{day_iso}T23:40:00Z"
    rows = [{
        "batter_id": 680757,
        "batter_name": "Steven Kwan",
        "team": "CLE",
        "lineup": 1,
        "pitcher_id": 543037,
        "pitcher_name": "Gerrit Cole",
        "opp": "NYY",
        "p_hit_pa": 0.28,
        "p_game_hit": p1,
        "flags": "",
        "projected_lineup": False,
        "game_pk": 101,
        "game_time": gt1,
    }]
    if p2 is not None:
        rows.append({
            "batter_id": 665742,
            "batter_name": "Juan Soto",
            "team": "NYM",
            "lineup": 2,
            "pitcher_id": 605483,
            "pitcher_name": "Blake Snell",
            "opp": "LAD",
            "p_hit_pa": 0.27,
            "p_game_hit": p2,
            "flags": "",
            "projected_lineup": False,
            "game_pk": 102,
            "game_time": gt2,
        })
    return pd.DataFrame(rows)


def make_schedule_games(day_iso: str, gt1: str | None = None, gt2: str | None = None) -> list[dict]:
    gt1 = gt1 or f"{day_iso}T23:10:00Z"
    gt2 = gt2 or f"{day_iso}T23:40:00Z"
    return [
        {
            "gamePk": 101,
            "gameDate": gt1,
            "officialDate": day_iso,
            "doubleHeader": "N",
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {
                "away": {"team": {"id": 114, "name": "Guardians"}},
                "home": {"team": {"id": 147, "name": "Yankees"}},
            },
        },
        {
            "gamePk": 102,
            "gameDate": gt2,
            "officialDate": day_iso,
            "doubleHeader": "N",
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
            "teams": {
                "away": {"team": {"id": 121, "name": "Mets"}},
                "home": {"team": {"id": 119, "name": "Dodgers"}},
            },
        },
    ]


def write_covered_auto(picks_dir: Path, now_dt: datetime, streak: int = 12,
                       best: int | None = None, source_date: str | None = None,
                       recorded_at: str | None = None, saver_flag: str = "used") -> str:
    year = now_dt.astimezone(ET).year
    act_id = "a" * 32
    acct = picks_dir / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    (acct / "season_contract.json").write_text(json.dumps({
        "schema_version": season_contract.SCHEMA,
        "season": year,
        "status": "automatic",
        "activation_id": act_id,
        "switched_at": now_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "revoked_at": None,
        "note": "probe",
    }))
    src_day = source_date or (now_dt.astimezone(ET).date() - timedelta(days=1)).isoformat()
    rec_at = recorded_at or now_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    (acct / "contest_streak.json").write_text(json.dumps({
        "schema_version": season_contract.AUTO_SCHEMA,
        "season": year,
        "activation_id": act_id,
        "active_streak": streak,
        "best_streak": best if best is not None else max(streak, 15),
        "source": "mlb_bts_profile",
        "source_date": src_day,
        "recorded_at": rec_at,
        "username": "stonehengee",
        "user_id": 1,
    }))
    (acct / "saver_state.json").write_text(json.dumps({
        "season": year,
        "state": saver_flag,
        "updated_at": rec_at,
        "source": "probe",
    }))
    return act_id


def write_dated_obs(picks_dir: Path, now_dt: datetime, streak: int = 12,
                    best: int | None = None, saver: str = "used",
                    day_iso: str | None = None) -> None:
    d = date.fromisoformat(day_iso) if day_iso else now_dt.astimezone(ET).date()
    end = REAL_DATETIME.combine(d, dt_time.max, ET)
    acct = picks_dir / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    (acct / "contest_streak.manual.json").write_text(json.dumps({
        "schema_version": contest_state.DATED_MANUAL_SCHEMA,
        "season": d.year,
        "effective_date": d.isoformat(),
        "expires_at": end.isoformat(),
        "active_streak": streak,
        "best_streak": best if best is not None else max(streak, 15),
        "saver_state": saver,
        "recorded_at": now_dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "manual_cli",
    }))


@contextmanager
def offline_harness(root: Path, initial_et: datetime):
    picks_dir = root / "data" / "picks"
    models_dir = root / "data" / "models"
    picks_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    clock = [initial_et]
    clock_cls = _make_clock(clock)
    day_iso = initial_et.date().isoformat()
    statuses = {
        101: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
        102: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
    }
    confirmed: dict[int, set[str]] = {101: {"away", "home"}, 102: {"away", "home"}}
    games_by_date: dict[str, list[dict]] = {day_iso: make_schedule_games(day_iso)}
    pick_dms: list[tuple[str, str]] = []
    health_dms: list[tuple[str, str]] = []
    posts: list[str] = []
    preds_holder = [make_predictions(day_iso)]

    config = {
        "orchestrator": {
            "picks_dir": str(picks_dir),
            "models_dir": str(models_dir),
            "heartbeat_path": str(root / "data" / ".heartbeat"),
        },
        "bluesky": {"dm_recipient": "operator.bsky.social"},
        "scheduler": {
            "pick_delivery": "dm",
            "entry_intent": "enter",
            "contest_state_required": True,
            "live_forward_capture_on_lock": False,
            "shadow_model_enabled": False,
        },
        "health_checks": {
            "enabled": False,
            "contest_state_expected": True,
        },
        "tiers": [],
    }
    toml_path = root / "orch.toml"
    toml_path.write_text(
        f'[orchestrator]\npicks_dir = "{picks_dir}"\nmodels_dir = "{models_dir}"\n'
        f'heartbeat_path = "{root / "data" / ".heartbeat"}"\n'
        '[bluesky]\ndm_recipient = "operator.bsky.social"\n'
        '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\n'
        'contest_state_required = true\nlive_forward_capture_on_lock = false\n'
        '[health_checks]\nenabled = false\ncontest_state_expected = true\n'
    )

    def _advance(secs: float, *_a, **_k):
        if secs > 0:
            clock[0] = clock[0] + timedelta(seconds=secs)

    def _check_confirmed(game_pks):
        return {pk: set(confirmed.get(pk, set())) for pk in game_pks if confirmed.get(pk)}

    def _send_pick_dm(recipient: str, text: str):
        pick_dms.append((recipient, text))
        return f"dm-{len(pick_dms)}"

    def _send_health_dm(recipient: str, text: str):
        health_dms.append((recipient, text))
        return f"hdm-{len(health_dms)}"

    def _post_bsky(text: str):
        posts.append(text)
        return f"at://post/{len(posts)}"

    def _cascade(_tiers, d):
        frame = preds_holder[0]
        return (None, None) if frame is None else (frame, "offline")

    with block_network(), ExitStack() as stack:
        stack.enter_context(patch("datetime.datetime", clock_cls))
        for mod in (contest_state, daily_decision, picks, strategy, cli, scheduler,
                    health_alert, health_contest_state, health_runner, saver_state):
            if hasattr(mod, "datetime"):
                stack.enter_context(patch.object(mod, "datetime", clock_cls))
        stack.enter_context(patch.object(scheduler, "_now_et", lambda: clock[0]))
        stack.enter_context(patch.object(cli, "_now_et", lambda: clock[0]))
        stack.enter_context(patch.object(scheduler, "_watchdog_ping_sleep", _advance))
        stack.enter_context(patch.object(scheduler, "_idle_until_next_wakeup", lambda *_a, **_k: None))
        stack.enter_context(patch.object(scheduler, "heartbeat_watchdog", lambda *_a, **_k: nullcontext()))
        stack.enter_context(patch.object(scheduler, "notify_ready", lambda: None))
        stack.enter_context(patch.object(scheduler, "notify_watchdog", lambda: None))
        stack.enter_context(patch.object(scheduler, "fetch_schedule", lambda d: list(games_by_date.get(d, []))))
        stack.enter_context(patch.object(scheduler, "check_confirmed_lineups", _check_confirmed))
        stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", lambda _d: dict(statuses)))
        stack.enter_context(patch.object(scheduler, "run_result_polling", lambda *_a, **_k: "final"))
        stack.enter_context(patch.object(picks, "get_game_statuses_detailed", lambda *_a, **_k: dict(statuses)))
        stack.enter_context(patch.object(strategy, "get_game_statuses_detailed", lambda *_a, **_k: dict(statuses)))
        stack.enter_context(patch.object(picks, "get_game_statuses",
                                         lambda *_a, **_k: {k: v["abstract"] for k, v in statuses.items()}))
        stack.enter_context(patch.object(picks, "_git_head_sha", lambda *_a, **_k: "0" * 40))
        stack.enter_context(patch.object(orchestrator, "run_cascade", _cascade))
        stack.enter_context(patch.object(dm, "send_dm", _send_pick_dm))
        stack.enter_context(patch.object(health_alert, "send_dm", _send_health_dm))
        stack.enter_context(patch.object(posting, "post_to_bluesky", _post_bsky))
        stack.enter_context(patch.object(posting, "should_post_now", lambda *_a, **_k: True))
        yield {
            "root": root,
            "picks_dir": picks_dir,
            "models_dir": models_dir,
            "config": config,
            "toml_path": toml_path,
            "clock": clock,
            "statuses": statuses,
            "confirmed": confirmed,
            "games_by_date": games_by_date,
            "pick_dms": pick_dms,
            "health_dms": health_dms,
            "posts": posts,
            "preds_holder": preds_holder,
        }


def test_5a_saved_pick_refusal_vs_fallbacks_and_downstream(tmp_path: Path):
    """5a saved-pick refusal against in-loop fallback, final fallback, EOD post_failure, reconcile, and day-outcome."""
    from bts.model import predict as model_predict

    day = "2027-07-15"
    start = REAL_DATETIME(2027, 7, 15, 3, 0, tzinfo=ET)
    with offline_harness(tmp_path, start) as env:
        picks_dir = env["picks_dir"]
        runner = CliRunner()
        # 1) 03:00 preview writes uncommitted {day}.json with chosen_on.required=False
        with patch.object(model_predict, "run_pipeline", return_value=make_predictions(day, 0.88, 0.85)):
            res = runner.invoke(cli.cli, [
                "preview", "--date", day,
                "--picks-dir", str(picks_dir), "--models-dir", str(env["models_dir"]),
            ])
        assert res.exit_code == 0, res.output
        preview_saved = picks.load_pick(day, picks_dir)
        assert preview_saved is not None and preview_saved.chosen_on["required"] is False

        # 2) Operator supplies dated observation for today, then scheduler wakes at 18:36 ET (at fallback)
        #    while cascades fail -> both fallback and final fallback refuse the 03:00 preview.
        env["clock"][0] = REAL_DATETIME(2027, 7, 15, 18, 36, tzinfo=ET)
        write_dated_obs(picks_dir, env["clock"][0], streak=30, best=30, saver="used")
        env["preds_holder"][0] = None
        scheduler.run_day(day, env["config"])

        assert env["pick_dms"] == []
        assert daily_decision.load_decision(day, picks_dir) is None
        assert not scheduler.load_state(day, picks_dir).pick_locked
        # {day}.json remains on disk -> EOD post_failure (>=22:00 ET) flags it AND next-day reconcile blocks on C3!
        assert (picks_dir / f"{day}.json").exists()
        pf_alerts = health_post_failure.check(
            picks_dir, today=date(2027, 7, 15), now=REAL_DATETIME(2027, 7, 15, 22, 30, tzinfo=ET)
        )
        assert len(pf_alerts) == 1 and pf_alerts[0].level == "CRITICAL"

        # Deduped contest_state CRITICAL DM + missed_pick CRITICAL DM (naming no batter)
        bodies = [msg for _, msg in env["health_dms"]]
        assert sum("[contest_state]" in b for b in bodies) == 1
        assert sum("[missed_pick]" in b and "Steven Kwan" not in b for b in bodies) == 1

        # Next morning day-outcome classifies as no_decision (exit 1)
        out = day_outcome.gather(day, picks_dir, env["config"], expected_username="stonehengee")
        assert out.state == "no_decision" and not out.ok

        # Next morning 02:00 reconcile hits C3 blocker on the leftover preview file
        picks.save_streak(5, picks_dir, saver_available=True, season=2027)
        env["clock"][0] = REAL_DATETIME(2027, 7, 16, 2, 0, tzinfo=ET)
        rec_res = runner.invoke(cli.cli, ["reconcile", "--picks-dir", str(picks_dir)])
        assert rec_res.exit_code == 0
        assert "Season replay unavailable for 2027-07-15" in rec_res.stderr
        assert picks.load_streak(picks_dir, season=2027) == 5

        # 3) Now test mid-day dated correction where in-loop fallback fails cascade (refuses old pick)
        #    and final fallback succeeds (re-picks on corrected dated streak and delivers!).
        day2 = "2027-07-16"
        env["games_by_date"][day2] = make_schedule_games(day2)
        env["clock"][0] = REAL_DATETIME(2027, 7, 16, 18, 10, tzinfo=ET)
        write_dated_obs(picks_dir, env["clock"][0], streak=10, best=15, saver="used", day_iso=day2)
        # Lineup 102 unconfirmed so 18:10 check defers early lock and saves {day2}.json on streak=10
        env["confirmed"][102] = set()
        f1 = make_predictions(day2, 0.85, 0.845)
        f1.loc[1, "flags"] = "PROJECTED"
        f1.loc[1, "projected_lineup"] = True
        calls = [0]

        def _staged_cascade(_tiers, _d):
            calls[0] += 1
            if calls[0] == 1:
                # 18:10 check chooses on streak=10, then operator corrects dated observation to streak=30!
                res_frame = f1
                return res_frame, "offline"
            if calls[0] == 2:
                # In-loop fallback at 18:35: operator corrected dated streak=30; cascade flakes!
                write_dated_obs(picks_dir, env["clock"][0], streak=30, best=30, saver="used", day_iso=day2)
                return None, None
            # Final fallback at step 5: cascade recovers -> chooses fresh single on streak=30!
            return make_predictions(day2, 0.96, 0.81), "offline"

        env["pick_dms"].clear()
        env["health_dms"].clear()
        with patch.object(orchestrator, "run_cascade", _staged_cascade):
            scheduler.run_day(day2, env["config"])
        assert calls[0] == 3
        assert len(env["pick_dms"]) == 1 and "Streak: 30" in env["pick_dms"][0][1]
        assert daily_decision.load_decision(day2, picks_dir)["delivery_status"] == "delivered"


def test_4c_stale_reading_stamp_vs_dated_and_bts_run_reuse(tmp_path: Path):
    """4c stale-reading stamp against failed send + dated switch + bts run / orchestrate reuse."""
    from bts.model import predict as model_predict

    day = "2027-07-15"
    now_dt = REAL_DATETIME(2027, 7, 15, 18, 0, tzinfo=ET)
    with offline_harness(tmp_path, now_dt) as env:
        picks_dir = env["picks_dir"]
        stale_rec = (now_dt - timedelta(hours=20)).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
        write_covered_auto(picks_dir, now_dt, streak=7, recorded_at=stale_rec)

        _, sel, _ = orchestrator.run_and_pick(env["config"], day)
        daily = sel.pick_result.daily
        picks.save_pick(daily, picks_dir)
        st = scheduler.SchedulerState(day, now_dt.isoformat(), [], [], [], False, None, None, None)

        # Failed DM send stamps automatic_reading (stale=True) onto disk even after clearing delivery_attempted!
        with patch.object(dm, "send_dm", side_effect=RuntimeError("transient network drop")):
            assert scheduler._deliver_and_lock_pick(
                daily, env["config"], picks_dir, st, day, "lineup", selection=sel
            ) is False
        on_disk = picks.load_pick(day, picks_dir)
        assert not on_disk.delivery_attempted
        assert on_disk.automatic_reading is not None and on_disk.automatic_reading["stale"] is True

        # Interaction 4c x bts run reuse: suppose fetch-contest-streak now refreshes contest_streak.json
        # (age=0h, NOT stale!), games enter Warmup so select_pick reuses on_disk via R33 automatic reuse,
        # and `bts run` posts publicly: `bts run` preserves the stale automatic_reading from the failed DM!
        write_covered_auto(picks_dir, now_dt, streak=7, recorded_at=now_dt.astimezone(timezone.utc).isoformat())
        env["statuses"][101] = {"abstract": "P", "detailed": "Warmup", "code": "PW"}
        env["statuses"][102] = {"abstract": "P", "detailed": "Warmup", "code": "PW"}
        runner = CliRunner()
        with patch.object(model_predict, "run_pipeline", return_value=make_predictions(day)):
            run_res = runner.invoke(cli.cli, [
                "run", "--date", day,
                "--picks-dir", str(picks_dir), "--models-dir", str(env["models_dir"]),
            ])
        assert run_res.exit_code == 0, run_res.output
        assert len(env["posts"]) == 1
        after_run = picks.load_pick(day, picks_dir)
        assert after_run.bluesky_posted is True
        # Stale automatic_reading from failed scheduler attempt survived on disk onto a fresh-state bts run post!
        assert after_run.automatic_reading is not None and after_run.automatic_reading["stale"] is True

        # Conversely, if operator switches to a dated observation and games are Scheduled, scheduler re-selects
        # and clears automatic_reading to None on delivery.
        day2 = "2027-07-16"
        env["clock"][0] = REAL_DATETIME(2027, 7, 16, 18, 0, tzinfo=ET)
        env["statuses"][101] = {"abstract": "P", "detailed": "Scheduled", "code": "S"}
        env["statuses"][102] = {"abstract": "P", "detailed": "Scheduled", "code": "S"}
        write_dated_obs(picks_dir, env["clock"][0], streak=8, best=15, saver="used", day_iso=day2)
        env["preds_holder"][0] = make_predictions(day2)
        _, sel2, _ = orchestrator.run_and_pick(env["config"], day2)
        st2 = scheduler.SchedulerState(day2, env["clock"][0].isoformat(), [], [], [], False, None, None, None)
        assert scheduler._deliver_and_lock_pick(
            sel2.pick_result.daily, env["config"], picks_dir, st2, day2, "lineup", selection=sel2
        ) is True
        assert picks.load_pick(day2, picks_dir).automatic_reading is None


def test_6b1_morning_check_vs_same_day_restarts(tmp_path: Path):
    """6b-1 morning state check against same-day restart after delivered pick and mid-day restart."""
    day = "2027-07-15"
    morning = REAL_DATETIME(2027, 7, 15, 10, 0, tzinfo=ET)
    with offline_harness(tmp_path, morning) as env:
        picks_dir = env["picks_dir"]
        write_dated_obs(picks_dir, morning, streak=12, best=15, saver="used")
        scheduler.run_day(day, env["config"])
        assert len(env["pick_dms"]) == 1
        assert scheduler.load_state(day, picks_dir).pick_locked is True

        # Restart 1: normal same-day restart after delivered pick -> morning check not run
        # Break contest state to prove morning check does NOT read it after lock!
        (picks_dir / "account_state" / "contest_streak.manual.json").unlink()
        env["clock"][0] = REAL_DATETIME(2027, 7, 15, 11, 0, tzinfo=ET)
        scheduler.run_day(day, env["config"])
        assert env["health_dms"] == []

        # Restart 2: scheduler_state.json deleted/quarantined, but delivered pick & scoreable decision exist
        (picks_dir / day / "scheduler_state.json").unlink()
        scheduler.run_day(day, env["config"])
        assert env["health_dms"] == []

        # Restart 3: mid-day restart when first check (12:10 ET) is in the past and 18:10 ET check is ahead
        day2 = "2027-07-16"
        env["games_by_date"][day2] = make_schedule_games(
            day2, gt1=f"{day2}T17:10:00Z", gt2=f"{day2}T23:10:00Z"
        )
        env["clock"][0] = REAL_DATETIME(2027, 7, 16, 14, 0, tzinfo=ET)
        scheduler.run_day(day2, env["config"])
        # Morning check skipped because earliest planned target (12:10 ET) is < now + 10m; live check alerts instead
        assert not (picks_dir.parent / "health_state" / "contest_morning_check_dm_delivery_status.json").exists()
        assert any("[contest_state]" in msg for _, msg in env["health_dms"])


def test_cs2_streak_season_vs_reconcile_and_new_year(tmp_path: Path):
    """CS2 season-scoped streak.json against 02:00 reconcile and check-results across Day 1 and New Year."""
    runner = CliRunner()
    # Part A: Season's first day (2027-03-24) with restored tagless 2026 streak.json
    p1 = tmp_path / "day1" / "picks"
    p1.mkdir(parents=True)
    (p1 / "streak.json").write_text(json.dumps({
        "streak": 14, "saver_available": False, "applied_dates": ["2026-09-26"]
    }))
    day1 = "2027-03-24"
    dp1 = picks.DailyPick(
        date=day1, run_time="2027-03-24T18:00:00-04:00",
        pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Cole", 543037, 0.85, [], False, 101,
                        "2027-03-24T23:10:00Z"),
        double_down=None, runner_up=None,
        notification_sent=True, notification_channel="bluesky_dm", notification_id="m1",
    )
    picks.save_pick(dp1, p1)
    daily_decision.write_decision(day1, p1, action="single", source="mdp",
                                  primary={"batter_id": 680757, "batter_name": "Steven Kwan"},
                                  delivery_status="delivered", scoreable=True)
    with block_network(), \
         patch.object(cli, "_now_et", lambda: REAL_DATETIME(2027, 3, 25, 1, 0, tzinfo=ET)), \
         patch.object(picks, "resolve_daily_slot_results", lambda *_a, **_k: {"primary": "hit"}):
        r = runner.invoke(cli.cli, ["check-results", "--date", day1, "--picks-dir", str(p1)])
    assert r.exit_code == 0, r.output
    st_after = json.loads((p1 / "streak.json").read_text())
    assert (st_after["season"], st_after["streak"], st_after["saver_available"]) == (2027, 1, True)

    # Part B: Across New Year (2027-12-31 -> 2028-01-01)
    # B1: Offseason Jan 1 02:00 reconcile wipes end-of-2027 streak.json to season=2028, streak=0!
    p2 = tmp_path / "ny_offseason" / "picks"
    p2.mkdir(parents=True)
    dp_sep = picks.DailyPick(
        date="2027-09-26", run_time="2027-09-26T18:00:00-04:00",
        pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Cole", 543037, 0.85, [], False, 101,
                        "2027-09-26T19:10:00Z"),
        double_down=None, runner_up=None, result="hit", slot_results={"primary": "hit"},
        notification_sent=True, notification_channel="bluesky_dm", notification_id="m1",
    )
    picks.save_pick(dp_sep, p2)
    (p2 / "streak.json").write_text(json.dumps({
        "season": 2027, "streak": 18, "saver_available": False, "applied_dates": ["2027-09-26"]
    }))
    jan1_2am = REAL_DATETIME(2028, 1, 1, 2, 0, tzinfo=ET)
    picks.reconcile_results(p2, clock=lambda: jan1_2am)
    wiped = json.loads((p2 / "streak.json").read_text())
    assert wiped["season"] == 2028 and wiped["streak"] == 0 and wiped["saver_available"] is True

    # B2: Concurrent/late Dec 31 grading after 02:00 ET reconcile loses the entire 2027 streak!
    p3 = tmp_path / "ny_concurrent" / "picks"
    p3.mkdir(parents=True)
    (p3 / "streak.json").write_text(json.dumps({
        "season": 2027, "streak": 25, "saver_available": False, "applied_dates": ["2027-12-30"]
    }))
    dp_dec31 = picks.DailyPick(
        date="2027-12-31", run_time="2027-12-31T18:00:00-05:00",
        pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Cole", 543037, 0.85, [], False, 101,
                        "2027-12-31T23:10:00Z"),
        double_down=None, runner_up=None,
        notification_sent=True, notification_channel="bluesky_dm", notification_id="m1",
    )
    picks.save_pick(dp_dec31, p3)
    daily_decision.write_decision("2027-12-31", p3, action="single", source="mdp",
                                  primary={"batter_id": 680757, "batter_name": "Steven Kwan"},
                                  delivery_status="delivered", scoreable=True)
    # 02:00 ET reconcile runs while check-results --wait-deadline-et 06:00 is still waiting on extra innings
    picks.reconcile_results(p3, clock=lambda: jan1_2am)
    assert json.loads((p3 / "streak.json").read_text())["season"] == 2028
    # At 02:15 ET Dec 31 game finishes as a hit -> check-results scores it, resetting 2027 streak 25 -> 1!
    with block_network(), \
         patch.object(cli, "_now_et", lambda: REAL_DATETIME(2028, 1, 1, 2, 15, tzinfo=ET)), \
         patch.object(picks, "resolve_daily_slot_results", lambda *_a, **_k: {"primary": "hit"}):
        r3 = runner.invoke(cli.cli, ["check-results", "--date", "2027-12-31", "--picks-dir", str(p3)])
    assert r3.exit_code == 0, r3.output
    corrupted_2027 = json.loads((p3 / "streak.json").read_text())
    assert corrupted_2027["season"] == 2027
    assert corrupted_2027["streak"] == 1  # Lost 25-game 2027 streak and resurrected spent saver!
    assert corrupted_2027["saver_available"] is True


def test_bullpen_and_probable_cache_cold_start(tmp_path: Path):
    """Bullpen same-day fix + probable-pitcher cache repair on cold-start before first 2027 game."""
    raw_dir = tmp_path / "data" / "raw"
    proc_dir = tmp_path / "data" / "processed"
    models_dir = tmp_path / "data" / "models"
    (raw_dir / "2027").mkdir(parents=True)
    models_dir.mkdir(parents=True)
    # Spring-training game in 2027 raw dir -> build_season returns None (no-op)
    (raw_dir / "2027" / "700001.json").write_text(json.dumps({
        "gameData": {
            "game": {"type": "S"},
            "teams": {"away": {"id": 114}, "home": {"id": 147}},
            "probablePitchers": {"away": {"id": 543037}, "home": {"id": 605483}},
        }
    }))
    assert data_build.build_season(raw_dir, proc_dir / "pa_2027.parquet", 2027) is None
    assert not (proc_dir / "pa_2027.parquet").exists()

    # Cache with one invalid entry + raw feed repair -> emits status record surfaced by health check
    cache_path = models_dir / "probable_pitcher_lookup.json"
    cache_path.write_text(json.dumps({"700001": "corrupt_non_dict", "99": {"away": 21, "home": 12, "away_tid": 51, "home_tid": 52}}))
    with block_network(), \
         patch.object(feat_compute, "_probable_pitcher_cache_path", cache_path), \
         patch.object(park_drag, "get_table", return_value=None):
        lookup = feat_compute._build_probable_pitcher_lookup(str(raw_dir))
        assert lookup[700001]["away"] == 543037
        dates = pd.date_range("2026-09-10", periods=10)
        rows = [
            {
                "pa_id": i, "date": d, "game_pk": 1000 + i, "batter_id": 11, "pitcher_id": 22,
                "pitch_hand": "R", "venue_id": 31, "is_home": True, "is_hit": i % 2,
                "pitch_calls": ["X"], "pitch_types": ["FF"], "pitch_speeds": [93.0],
                "pitch_spin_rates": [2300.0], "pitch_extensions": [6.0],
                "pitch_break_vertical": [15.0], "pitch_break_horizontal": [5.0],
                "pitch_px": [0.0], "pitch_pz": [2.5], "sz_top": 3.5, "sz_bottom": 1.5,
                "final_count_balls": 0, "final_count_strikes": 0, "launch_speed": 91.0, "launch_angle": 10.0,
            }
            for i, d in enumerate(dates)
        ]
        # Cold-start target row on 2027-03-24
        rows.append(dict(rows[-1], pa_id=10, date=pd.Timestamp("2027-03-24"), game_pk=2001, pitcher_id=21))
        prob_map = {1000 + i: {"away": 21, "away_tid": 51} for i in range(10)} | {2001: {"away": 21, "away_tid": 51}}
        with patch.object(feat_compute, "_build_probable_pitcher_lookup", return_value=prob_map):
            feats = feat_compute.compute_all_features(pd.DataFrame(rows)).set_index("pa_id")
        assert np.isclose(feats.loc[10, "opp_bullpen_hr_30g"], 0.5)

    today_et = datetime.now(timezone.utc).astimezone(ET).date()
    pp_alerts = health_pp_cache.check(tmp_path / "data" / "health_state" / "probable_pitcher_cache", today=today_et)
    assert len(pp_alerts) == 1 and pp_alerts[0].level == "WARN"


def test_watchdog_f2_f3_f4_g_h_h2_end_to_end(tmp_path: Path):
    """Entry watchdog (F1/F5b), saver-mismatch (F2), day-outcome reviews (F3/F4), and G/H/H2 policy."""
    day = "2027-09-26"
    noon = REAL_DATETIME(2027, 9, 26, 12, 0, tzinfo=ET)
    with offline_harness(tmp_path, noon) as env:
        picks_dir = env["picks_dir"]
        runner = CliRunner()

        # 1) H + H2: streak=58, best=60 on finale (2027-09-26, 1 day left) -> longest_streak double!
        write_dated_obs(picks_dir, noon, streak=58, best=60, saver="used", day_iso=day)
        # Set saver flag to 'active' to ALSO trigger F2 saver_state:dated_used_mismatch at EOD!
        (picks_dir / "account_state" / "saver_state.json").write_text(json.dumps({
            "season": 2027, "state": "active", "updated_at": noon.isoformat(), "source": "probe"
        }))
        env["config"]["health_checks"]["enabled"] = True
        scheduler.run_day(day, env["config"])
        assert len(env["pick_dms"]) == 1
        saved = picks.load_pick(day, picks_dir)
        assert saved.double_down is not None
        assert saved.policy_decision["objective"] == "longest_streak"
        dec = daily_decision.load_decision(day, picks_dir)
        assert dec["objective"] == "longest_streak" and dec["action"] == "double"
        assert picks.saved_pick_refusal(saved, contest_state.load_decision_streak_state(picks_dir, require_contest_state=True)) is None

        # Verify F2 mismatch DM fired on its separate receipt
        mismatch_receipt = picks_dir.parent / "health_state" / "health_dm_delivery_status.json.dated_used_mismatch.json"
        assert mismatch_receipt.exists()
        assert any("The dated observation says the saver was used" in msg for _, msg in env["health_dms"])

        # Verify reach57 readers exclude longest_streak record
        assert skip_policy_shadow.record_skip_from_decision(day, picks_dir, now=noon) is None
        from bts.simulate import mdp as sim_mdp

        assert mdp_policy_alignment.compute_metrics(
            picks_dir, sim_mdp.DEFAULT_POLICY_PATH, today=date(2027, 9, 26)
        ).primary.n == 0

        # 2) Entry watchdog F5b early deferral + F1 wrong_game doubleheader check + day-outcome F4 broken streak.json
        units_dir = entry_receipt.units_snapshot_dir(picks_dir)
        units_dir.mkdir(parents=True, exist_ok=True)
        (units_dir / "2027-09-26T150000Z.json.gz").write_bytes(gzip.compress(json.dumps({
            "units": [
                {"id": 77, "roundId": 1, "feedId": 999},  # wrong game (recommended 101)
                {"id": 78, "roundId": 1, "feedId": 102},  # right game
            ]
        }).encode()))
        session = SimpleNamespace(user_id="u1", username="stonehengee", xsid="x1")
        pending = [
            {"roundId": 1, "unitId": 77, "playerId": 1, "number": 1, "result": None},
            {"roundId": 1, "unitId": 78, "playerId": 2, "number": 2, "result": None},
        ]
        with patch("bts.leaderboard.auth.load_session_cookies", return_value={}), \
             patch("bts.leaderboard.auth.extract_uid", return_value="u1"), \
             patch("bts.leaderboard.auth.fetch_login_session", return_value=session), \
             patch.object(contest_fetch, "fetch_profile", return_value={"predictions": []}), \
             patch.object(contest_fetch, "fetch_pending_predictions", return_value=pending), \
             patch.object(cli, "_fetch_rounds", return_value={1: date(2027, 9, 26)}), \
             patch.object(cli, "_fetch_bts_to_mlb", return_value={1: 680757, 2: 665742}):
            # Early --cron run at 09:45 for a 10:50 pitch defers via base_slot_available
            saved_1050 = picks.load_pick(day, picks_dir)
            saved_1050.pick.game_time = "2027-09-26T14:50:00Z"
            saved_1050.double_down.game_time = "2027-09-26T14:55:00Z"
            picks.save_pick(saved_1050, picks_dir)
            early_res = runner.invoke(cli.cli, [
                "check-pick-entered", "--cron", "--picks-dir", str(picks_dir),
                "--expected-username", "stonehengee", "--dm-recipient", "operator.bsky.social",
                "--now-et", "2027-09-26T09:45:00",
            ])
            assert early_res.exit_code == 0 and "deferred" in early_res.stdout

            # Original-hours run at 10:00 catches wrong_game on leg 1!
            live_res = runner.invoke(cli.cli, [
                "check-pick-entered", "--cron", "--picks-dir", str(picks_dir),
                "--expected-username", "stonehengee", "--dm-recipient", "operator.bsky.social",
                "--now-et", "2027-09-26T10:00:00",
            ])
            assert live_res.exit_code == 1
            assert any("the entry is recorded in game 999" in msg for _, msg in env["pick_dms"])

        pe_alerts = health_pick_entry.check(
            picks_dir, today=date(2027, 9, 26), now=REAL_DATETIME(2027, 9, 26, 20, 0, tzinfo=ET)
        )
        assert len(pe_alerts) == 1 and "reason=wrong_game" in pe_alerts[0].message

        # Corrupt streak.json -> day-outcome reports BOTH entry_unverifiable AND streak_problem (F4)
        (picks_dir / "streak.json").write_bytes(b"[]")
        out_bad = day_outcome.gather(day, picks_dir, env["config"], expected_username="stonehengee")
        assert out_bad.state == "entry_unverifiable"
        assert out_bad.streak_problem == "not a JSON object"
        assert len(out_bad.message()) <= 1000

        # 3) Guarded pick (G) + grading + F3 skip_state_review on another day
        day_g = "2027-03-24"
        env["clock"][0] = REAL_DATETIME(2027, 3, 24, 12, 0, tzinfo=ET)
        write_dated_obs(picks_dir, env["clock"][0], streak=50, best=50, saver="used", day_iso=day_g)
        env["preds_holder"][0] = make_predictions(day_g, 0.96, 0.85)
        _, sel_g, _ = orchestrator.run_and_pick(env["config"], day_g)
        assert sel_g.action == "single" and sel_g.decision.guard is not None
        assert sel_g.decision.guard["rule"] == "monotone_skip_v1"

        # F3 skip_state_review: deliberate skip on streak=30, then dated observation changed to streak=12
        day_s = "2027-04-10"
        env["clock"][0] = REAL_DATETIME(2027, 4, 10, 12, 0, tzinfo=ET)
        (picks_dir / "streak.json").write_text(json.dumps({"season": 2027, "streak": 12, "saver_available": False}))
        daily_decision.write_decision(
            day_s, picks_dir, action="skip", source="mdp", objective="reach57",
            delivery_status="not_applicable", scoreable=False, streak=30, saver_available=False,
        )
        write_dated_obs(picks_dir, env["clock"][0], streak=12, best=30, saver="used", day_iso=day_s)
        env["games_by_date"][day_s] = make_schedule_games(day_s)
        out_f3 = day_outcome.gather(day_s, picks_dir, env["config"], expected_username="stonehengee")
        assert out_f3.state == "skip_state_review" and not out_f3.ok


if __name__ == "__main__":
    import tempfile

    checks = [
        test_5a_saved_pick_refusal_vs_fallbacks_and_downstream,
        test_4c_stale_reading_stamp_vs_dated_and_bts_run_reuse,
        test_6b1_morning_check_vs_same_day_restarts,
        test_cs2_streak_season_vs_reconcile_and_new_year,
        test_bullpen_and_probable_cache_cold_start,
        test_watchdog_f2_f3_f4_g_h_h2_end_to_end,
    ]
    results = []
    for fn in checks:
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
            results.append({"test": fn.__name__, "status": "PASS"})
    out_dir = Path(__file__).resolve().parent / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_interactions.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"probe_interactions: {len(results)}/{len(checks)} PASS")
