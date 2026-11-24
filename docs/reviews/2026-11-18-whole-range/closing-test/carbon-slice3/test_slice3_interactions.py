"""Slice 3 adversarial probes — Item 1: Cross-change-set interactions.

Runnable directly (`nice -n 19 .venv/bin/python .carbon-review/whole-range/test_slice3_interactions.py`)
and adoptable as a pytest module (`pytest ...`).
Rules obeyed:
- No module-level import of bts.model.predict or lightgbm.
- No permanent process-wide state mutations (all patches scoped; strategy._mdp_cache restored).
- Zero network or socket calls.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager, nullcontext
import copy
import datetime as dt
import io
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pandas as pd

from bts import cli as cli_mod
from bts import contest_state, daily_decision, day_outcome, dm, orchestrator, picks, posting, saver_state, scheduler, season_contract, strategy
from bts.health import alert as health_alert
from bts.health import contest_state as health_contest_state
from bts.health import runner as health_runner
from bts.reconcile_receipt import ReconcileReceipt

ET = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_TMP = Path(__file__).resolve().parent / "scratch_tmp_interactions"


@contextmanager
def _temp_root(tmp_path: Path | None):
    if tmp_path is not None:
        yield Path(tmp_path)
        return
    SCRATCH_TMP.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=SCRATCH_TMP) as td:
        yield Path(td)


def _make_Clock(box: list[dt.datetime]):
    class PinnedDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            cur = box[0]
            return cur.astimezone(tz) if tz is not None else cur.replace(tzinfo=None)

    return PinnedDatetime


def _game_dict(pk: int, day: str, hhmm_utc: str = "23:10:00", detailed: str = "Scheduled", abstract: str = "P") -> dict:
    return {
        "gamePk": pk,
        "gameType": "R",
        "officialDate": day,
        "gameDate": f"{day}T{hhmm_utc}Z",
        "status": {"abstractGameCode": abstract, "detailedState": detailed},
        "teams": {
            "away": {"team": {"id": 114, "name": "Guardians"}},
            "home": {"team": {"id": 147, "name": "Yankees"}},
        },
    }


def _pred_frame(day: str, p1: float = 0.84, p2: float | None = 0.81, projected: bool = False, hhmm_utc: str = "23:10:00") -> pd.DataFrame:
    flags = "PROJECTED" if projected else ""
    rows = [{
        "batter_id": 680757,
        "batter_name": "Steven Kwan",
        "team": "CLE",
        "lineup": 1,
        "pitcher_id": 543037,
        "pitcher_name": "Gerrit Cole",
        "opp": "NYY",
        "p_hit_pa": 0.29,
        "p_game_hit": p1,
        "flags": flags,
        "projected_lineup": projected,
        "game_pk": 101,
        "game_time": f"{day}T{hhmm_utc}Z",
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
            "flags": flags,
            "projected_lineup": projected,
            "game_pk": 102,
            "game_time": f"{day}T23:40:00Z",
        })
    return pd.DataFrame(rows)


def _write_dated_obs(picks_dir: Path, now: dt.datetime, streak: int, best: int | None = None, saver: str = "not_earned") -> None:
    day = now.astimezone(ET).date()
    best_val = max(streak, best if best is not None else streak)
    acct = picks_dir / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "bts_contest_streak_manual_v3",
        "effective_date": day.isoformat(),
        "season": day.year,
        "expires_at": dt.datetime.combine(day, dt.time.max, ET).isoformat(),
        "active_streak": streak,
        "best_streak": best_val,
        "saver_state": saver,
        "recorded_at": now.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
        "source": "manual_cli",
    }
    (acct / "contest_streak.manual.json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_auto_obs(
    picks_dir: Path,
    now: dt.datetime,
    streak: int,
    *,
    recorded_at: str | None = None,
    covered: bool = True,
    saver_flag: str | None = "not_earned",
    source_date: str | None = "TODAY",
) -> None:
    year = now.astimezone(ET).year
    acct = picks_dir / "account_state"
    acct.mkdir(parents=True, exist_ok=True)
    act_id = "a" * 32
    if covered:
        (acct / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1",
            "season": year,
            "status": "automatic",
            "activation_id": act_id,
            "switched_at": now.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
            "revoked_at": None,
            "note": None,
        }, indent=2) + "\n")
    src_date = now.astimezone(ET).date().isoformat() if source_date == "TODAY" else source_date
    (acct / "contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2",
        "season": year,
        "activation_id": act_id,
        "active_streak": streak,
        "best_streak": max(streak, 15),
        "source": "mlb_bts_profile",
        "source_date": src_date,
        "recorded_at": recorded_at or now.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
    }, indent=2) + "\n")
    if saver_flag is not None:
        (acct / "saver_state.json").write_text(json.dumps({
            "season": year,
            "state": saver_flag,
            "updated_at": now.astimezone(dt.timezone.utc).isoformat(),
            "source": "test",
        }, indent=2) + "\n")


@contextmanager
def _harness(root: Path, start_et: dt.datetime, *, delivery: str = "dm", intent: str = "enter", health_enabled: bool = False):
    picks_dir = root / "data" / "picks"
    models_dir = root / "data" / "models"
    picks_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)
    for pol_name in ("mdp_policy.npz", "mdp_tail_policy.npz"):
        src = REPO_ROOT / "data" / "models" / pol_name
        dst = models_dir / pol_name
        if src.exists() and not dst.exists():
            dst.symlink_to(src)

    clock = [start_et]
    mono = [0.0]
    Clock = _make_Clock(clock)

    day_str = start_et.astimezone(ET).date().isoformat()
    games_by_date: dict[str, list[dict]] = {day_str: [_game_dict(101, day_str)]}
    statuses: dict[int, dict] = {
        101: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
        102: {"abstract": "P", "detailed": "Scheduled", "code": "S"},
    }
    confirmed: dict[int, set[str]] = {}
    pick_dms: list[tuple[str, str]] = []
    health_dms: list[tuple[str, str]] = []
    public_posts: list[str] = []

    config = {
        "orchestrator": {
            "picks_dir": str(picks_dir),
            "models_dir": str(models_dir),
            "heartbeat_path": str(root / "data" / ".heartbeat"),
        },
        "bluesky": {"dm_recipient": "operator.bsky.social"},
        "scheduler": {
            "contest_state_required": delivery in {"dm", "public"},
            "pick_delivery": delivery,
            "entry_intent": intent,
            "live_forward_capture_on_lock": False,
            "lineup_check_offset_min": 60,
        },
        "health_checks": {
            "enabled": health_enabled,
            "contest_state_expected": delivery in {"dm", "public"},
        },
        "tiers": [{"name": "local", "type": "local"}],
    }

    def advance(secs: float, *_a, **_kw) -> None:
        delta = max(0.0, float(secs))
        clock[0] = clock[0] + dt.timedelta(seconds=delta)
        mono[0] += delta

    saved_mdp_cache = strategy._mdp_cache
    with ExitStack() as stack:
        for mod in (contest_state, daily_decision, picks, strategy, scheduler, health_alert, health_runner, health_contest_state):
            if hasattr(mod, "datetime"):
                stack.enter_context(patch.object(mod, "datetime", Clock))
        stack.enter_context(patch.object(scheduler, "_now_et", side_effect=lambda: clock[0].astimezone(ET)))
        stack.enter_context(patch.object(cli_mod, "_now_et", side_effect=lambda: clock[0].astimezone(ET)))
        stack.enter_context(patch.object(posting, "_now_et", side_effect=lambda: clock[0].astimezone(ET)))
        stack.enter_context(patch.object(scheduler, "_watchdog_ping_sleep", side_effect=advance))
        stack.enter_context(patch.object(scheduler, "_idle_until_next_wakeup", return_value=None))
        stack.enter_context(patch.object(scheduler, "heartbeat_watchdog", side_effect=lambda *_a, **_kw: nullcontext()))
        stack.enter_context(patch.object(scheduler, "notify_watchdog", return_value=None))
        stack.enter_context(patch.object(scheduler, "run_result_polling", return_value="final"))
        stack.enter_context(patch.object(scheduler.time, "monotonic", side_effect=lambda: mono[0]))
        stack.enter_context(patch.object(scheduler, "fetch_schedule", side_effect=lambda d: list(games_by_date.get(d, []))))
        stack.enter_context(patch.object(scheduler, "check_confirmed_lineups", side_effect=lambda pks: {pk: set(confirmed.get(pk, set())) for pk in pks}))
        stack.enter_context(patch.object(picks, "get_game_statuses_detailed", side_effect=lambda _d: dict(statuses)))
        stack.enter_context(patch.object(strategy, "get_game_statuses_detailed", side_effect=lambda _d: dict(statuses)))
        stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", side_effect=lambda _d: dict(statuses)))
        stack.enter_context(patch.object(picks, "get_game_statuses", side_effect=lambda _d: {k: v["abstract"] for k, v in statuses.items()}))
        stack.enter_context(patch.object(strategy, "get_game_statuses", side_effect=lambda _d: {k: v["abstract"] for k, v in statuses.items()}))
        stack.enter_context(patch.object(picks, "_git_head_sha", return_value="bc143131" + "0" * 32))
        stack.enter_context(patch.object(dm, "send_dm", side_effect=lambda r, t: (pick_dms.append((r, t)), f"dm-{len(pick_dms)}")[1]))
        stack.enter_context(patch.object(health_alert, "send_dm", side_effect=lambda r, t: (health_dms.append((r, t)), f"hdm-{len(health_dms)}")[1]))
        stack.enter_context(patch.object(posting, "post_to_bluesky", side_effect=lambda t: (public_posts.append(t), f"at://post/{len(public_posts)}")[1]))
        stack.enter_context(patch.object(scheduler, "retry_urlopen", side_effect=AssertionError("blocked network call")))
        stack.enter_context(patch.object(picks, "retry_urlopen", side_effect=AssertionError("blocked network call")))
        try:
            yield {
                "root": root,
                "picks_dir": picks_dir,
                "models_dir": models_dir,
                "config": config,
                "clock": clock,
                "mono": mono,
                "advance": advance,
                "games_by_date": games_by_date,
                "statuses": statuses,
                "confirmed": confirmed,
                "pick_dms": pick_dms,
                "health_dms": health_dms,
                "public_posts": public_posts,
            }
        finally:
            strategy._mdp_cache = saved_mdp_cache


def test_item1a_saved_pick_refusal_vs_both_fallbacks_and_double_cascade_overrun(tmp_path: Path | None = None) -> dict:
    """5a saved-pick refusal vs Step 4 in-loop fallback and Step 5 end-of-checks fallback."""
    with _temp_root(tmp_path) as base:

        # Case 1: Check 1 saves projected pick on dated A (streak=2). Operator updates dated B (streak=3)
        # before T-35. Step 4 fallback cascade fails -> 5a refuses cached pick -> Step 5 runs immediately
        # and its fresh cascade SUCCEEDS on B -> delivers!
        day = "2027-07-15"
        start = dt.datetime(2027, 7, 15, 10, 0, tzinfo=ET)
        with _harness(base / "c1", start) as env:
            _write_dated_obs(env["picks_dir"], start, streak=2, best=5, saver="not_earned")
            calls = [0]

            def cascade_recover(_tiers, _date):
                calls[0] += 1
                now_hm = env["clock"][0].strftime("%H:%M")
                if calls[0] <= 3:
                    # 16:10, 17:10, 18:10 projected; operator updates dated obs right after 18:10 check
                    if now_hm == "18:10":
                        res = _pred_frame(day, p1=0.84, p2=None, projected=True), "local"
                        return res
                    return _pred_frame(day, p1=0.84, p2=None, projected=True), "local"
                if calls[0] == 4:
                    # 18:35 Step 4 fallback: operator already updated streak=3; cascade fails!
                    raise RuntimeError("transient tier failure at step 4 fallback")
                # calls[0] == 5: Step 5 end-of-checks fallback immediately retries and succeeds!
                return _pred_frame(day, p1=0.85, p2=None, projected=False), "local"

            orig_sleep = scheduler._watchdog_ping_sleep

            def sleep_with_operator_edit(secs, *a, **kw):
                orig_sleep(secs, *a, **kw)
                if env["clock"][0].strftime("%H:%M") == "18:35":
                    _write_dated_obs(env["picks_dir"], env["clock"][0], streak=3, best=5, saver="not_earned")

            with patch.object(orchestrator, "run_cascade", side_effect=cascade_recover), \
                 patch.object(scheduler, "_watchdog_ping_sleep", side_effect=sleep_with_operator_edit):
                scheduler.run_day(day, env["config"])

            saved1 = picks.load_pick(day, env["picks_dir"])
            assert calls[0] == 5, f"expected 5 cascades (3 checks + step4 fail + step5 recover), got {calls[0]}"
            assert len(env["health_dms"]) == 1 and "Saved pick for 2027-07-15 was not delivered" in env["health_dms"][0][1]
            assert len(env["pick_dms"]) == 1 and "Streak: 3" in env["pick_dms"][0][1]
            assert saved1 is not None and saved1.chosen_on["streak"] == 3

        # Case 2 (TIMING INTERACTION): Step 4 fallback cascade takes 16 min (18:35 -> 18:51) and fails ->
        # 5a refuses cached pick -> Step 5 immediately launches a SECOND unbudgeted cascade taking 15 min
        # (18:51 -> 19:06, past the 19:05 T-5 cutoff!), so even when Step 5's cascade succeeds, delivery is
        # refused as LATE DELIVERY REFUSED and _maybe_alert_missed_pick fires AFTER cutoff!
        with _harness(base / "c2", start) as env:
            _write_dated_obs(env["picks_dir"], start, streak=2, best=5, saver="not_earned")
            calls2 = [0]

            def slow_double_fallback(_tiers, _date):
                calls2[0] += 1
                if calls2[0] <= 3:
                    env["advance"](10.0)
                    return _pred_frame(day, p1=0.84, p2=None, projected=True), "local"
                if calls2[0] == 4:
                    # Operator changed streak to 3 during sleep to 18:35; Step 4 cascade takes 16m and fails
                    env["advance"](16 * 60.0)
                    raise RuntimeError("step 4 slow failure")
                # Step 5 immediately runs a second cascade taking 15m -> finishes at 19:06:30 ET (> 19:05 cutoff!)
                env["advance"](15 * 60.0)
                return _pred_frame(day, p1=0.85, p2=None, projected=False), "local"

            orig_sleep2 = scheduler._watchdog_ping_sleep

            def sleep_edit2(secs, *a, **kw):
                orig_sleep2(secs, *a, **kw)
                if env["clock"][0].strftime("%H:%M") == "18:35":
                    _write_dated_obs(env["picks_dir"], env["clock"][0], streak=3, best=5, saver="not_earned")

            with patch.object(orchestrator, "run_cascade", side_effect=slow_double_fallback), \
                 patch.object(scheduler, "_watchdog_ping_sleep", side_effect=sleep_edit2):
                scheduler.run_day(day, env["config"])

            assert calls2[0] == 5
            assert len(env["pick_dms"]) == 0
            dec2 = daily_decision.load_decision(day, env["picks_dir"])
            assert dec2 is None
            hdm_msgs = [m for _, m in env["health_dms"]]
            assert any("Saved pick for 2027-07-15 was not delivered" in m for m in hdm_msgs)
            assert any("LATE DELIVERY REFUSED" in m for m in hdm_msgs)

        # Case 3: 03:00 preview pick on disk (required=False), checks fail ContestStateError -> Step 5 sleeps
        # to T-35; if cascade fails at Step 5, 5a refuses preview pick and T-10 missed_pick alerts.
        with _harness(base / "c3", start) as env:
            p_daily = picks.DailyPick(
                day, start.isoformat(),
                picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.84, ["PROJECTED"], True, 101, f"{day}T23:10:00Z"),
                None, None,
            )
            st0 = contest_state.load_decision_streak_state(env["picks_dir"], require_contest_state=False, now=start)
            p_daily.chosen_on = picks.chosen_on_record(st0, required=False)
            picks.save_pick(p_daily, env["picks_dir"])

            orig_sleep3 = scheduler._watchdog_ping_sleep

            def sleep_repair_before_step5(secs, *a, **kw):
                orig_sleep3(secs, *a, **kw)
                if env["clock"][0].strftime("%H:%M") == "18:35":
                    _write_dated_obs(env["picks_dir"], env["clock"][0], streak=1, best=1, saver="not_earned")

            with patch.object(orchestrator, "run_cascade", return_value=(None, None)), \
                 patch.object(scheduler, "_watchdog_ping_sleep", side_effect=sleep_repair_before_step5):
                scheduler.run_day(day, env["config"])

            assert len(env["pick_dms"]) == 0
            hdm3 = [m for _, m in env["health_dms"]]
            assert any("NO PICK DELIVERED" in m for m in hdm3)

        return {
            "step5_recovers_after_step4_5a_refusal": True,
            "double_fallback_cascade_can_overrun_cutoff": True,
            "preview_pick_refused_at_step5_when_cascade_fails_and_missed_pick_alerts": True,
        }


def test_item1b_stale_reading_stamp_vs_dated_and_bts_run(tmp_path: Path | None = None) -> dict:
    """4c stale-reading stamp vs dated observation, bts run, and live_forward_capture drift."""
    import importlib.util
    import sys
    spec = importlib.util.spec_from_file_location("_slice3_lfc_mod", REPO_ROOT / "scripts" / "live_forward_capture_once.py")
    assert spec is not None and spec.loader is not None
    lfc = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, {spec.name: lfc}):
        spec.loader.exec_module(lfc)

    with _temp_root(tmp_path) as base:
        day = "2027-07-15"
        start = dt.datetime(2027, 7, 15, 18, 25, tzinfo=ET)
        stale_iso = (start - dt.timedelta(hours=20)).isoformat()

        # 1. Stale automatic file + active dated observation -> automatic_reading is None, zero stale warnings
        with _harness(base / "dated_over_stale", start) as env:
            _write_auto_obs(env["picks_dir"], start, streak=12, recorded_at=stale_iso, saver_flag="active")
            _write_dated_obs(env["picks_dir"], start, streak=4, best=15, saver="active")
            with patch.object(orchestrator, "run_cascade", return_value=(_pred_frame(day, p1=0.86, p2=None), "local")):
                scheduler.run_day(day, env["config"])
            saved = picks.load_pick(day, env["picks_dir"])
            assert saved is not None and saved.chosen_on["kind"] == "dated"
            assert saved.automatic_reading is None
            assert env["health_dms"] == []
            assert len(env["pick_dms"]) == 1 and "Streak: 4" in env["pick_dms"][0][1]

        # 2. Stale automatic file + `bts run` -> posts publicly, stamps chosen_on, leaves automatic_reading=None, no warning
        with _harness(base / "bts_run_stale", start, delivery="public") as env:
            _write_auto_obs(env["picks_dir"], start, streak=5, recorded_at=stale_iso, saver_flag="not_earned")
            import bts.model.predict as predict_mod
            with patch.object(predict_mod, "run_pipeline", return_value=_pred_frame(day, p1=0.86, p2=None)):
                res = CliRunner().invoke(cli_mod.cli, [
                    "run", "--date", day,
                    "--picks-dir", str(env["picks_dir"]),
                    "--models-dir", str(env["models_dir"]),
                ])
            assert res.exit_code == 0, res.output
            saved_run = picks.load_pick(day, env["picks_dir"])
            assert saved_run is not None and saved_run.bluesky_posted is True
            assert saved_run.chosen_on["kind"] == "automatic" and saved_run.chosen_on["required"] is True
            assert saved_run.automatic_reading is None
            assert "STALE" not in res.output and env["health_dms"] == []

        # 3. Scheduler DM delivery stamps non-null automatic_reading -> live_forward_capture snapshot check
        with _harness(base / "lfc_drift", start) as env:
            _write_auto_obs(env["picks_dir"], start, streak=5, recorded_at=stale_iso, saver_flag="not_earned")
            with patch.object(orchestrator, "run_cascade", return_value=(_pred_frame(day, p1=0.86, p2=None, projected=True), "local")):
                chk = scheduler.run_single_check(day, [101], set(), env["config"], 0.03)
            pre_lock_bytes = (env["picks_dir"] / f"{day}.json").read_bytes()
            pre_lock_json = json.loads(pre_lock_bytes)
            assert pre_lock_json["automatic_reading"] is None

            manifest_path = base / "lfc_drift" / "manifest.json"
            manifest_path.write_text(json.dumps({
                "production_pick_snapshot": {
                    "sha256": lfc.file_sha256(env["picks_dir"] / f"{day}.json"),
                    "production_pick_json": pre_lock_json,
                }
            }))
            st = scheduler.SchedulerState(day, start.isoformat(), [], [], [], False, None, None, None)
            assert scheduler._deliver_and_lock_pick(
                chk["pick_result"].daily, env["config"], env["picks_dir"], st, day, "fallback", selection=chk["selection"]
            ) is True
            snap_state = lfc.existing_snapshot_state(
                manifest_path=manifest_path,
                pick_path=env["picks_dir"] / f"{day}.json",
                expected_date=day,
            )
            assert snap_state["stale_pick_snapshot"] is True

        return {
            "dated_clears_automatic_reading_and_suppresses_warn": True,
            "bts_run_neither_stamps_nor_warns_on_stale_auto": True,
            "post_delivery_stamp_marks_prelock_capture_stale": snap_state["stale_pick_snapshot"],
        }


def test_item1c_morning_check_vs_sameday_restart_and_private_cutover(tmp_path: Path | None = None) -> dict:
    """6b-1 morning state check vs same-day restart after delivered pick AND after private->dm cutover."""
    with _temp_root(tmp_path) as base:
        day = "2027-07-15"

        # 1. Delivered matinee pick at 12:35 ET, then contract revoked and scheduler restarted at 14:00 ET
        # with a 19:10 ET game still ahead -> zero morning or lineup contest-state DMs on restart!
        t_morning = dt.datetime(2027, 7, 15, 10, 0, tzinfo=ET)
        with _harness(base / "restart_delivered", t_morning) as env:
            env["games_by_date"][day] = [
                _game_dict(101, day, hhmm_utc="17:10:00"),  # 13:10 ET
                _game_dict(102, day, hhmm_utc="23:10:00"),  # 19:10 ET
            ]
            _write_dated_obs(env["picks_dir"], t_morning, streak=6, best=10, saver="active")
            matinee_frame = pd.DataFrame([{
                "batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
                "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
                "p_hit_pa": 0.30, "p_game_hit": 0.88, "flags": "", "projected_lineup": False,
                "game_pk": 101, "game_time": f"{day}T17:10:00Z",
            }])
            with patch.object(orchestrator, "run_cascade", return_value=(matinee_frame, "local")):
                scheduler.run_day(day, env["config"])
            assert len(env["pick_dms"]) == 1 and len(env["health_dms"]) == 0

            # Now remove contest state completely AND restart at 14:00 ET (both with and without scheduler_state.json)
            (env["picks_dir"] / "account_state" / "contest_streak.manual.json").unlink()
            env["clock"][0] = dt.datetime(2027, 7, 15, 14, 0, tzinfo=ET)
            with patch.object(orchestrator, "run_cascade", side_effect=AssertionError("must not re-run cascade")):
                scheduler.run_day(day, env["config"])
            assert len(env["health_dms"]) == 0

            # Even if scheduler_state.json was deleted/quarantined, _morning_state_check + run_single_check stay quiet
            (env["picks_dir"] / f"scheduler_state_{day}.json").unlink(missing_ok=True)
            (env["picks_dir"] / "scheduler_state.json").unlink(missing_ok=True)
            env["clock"][0] = dt.datetime(2027, 7, 15, 14, 5, tzinfo=ET)
            with patch.object(orchestrator, "run_cascade", side_effect=AssertionError("must not re-run cascade")):
                scheduler.run_day(day, env["config"])
            assert len(env["health_dms"]) == 0

        # 2. REAL EDGE INTERACTION A (private -> dm same-day cutover):
        # Early check locks in `private` mode: writes decision.json with delivery_status="private_locked",
        # scoreable=True, and pick_locked=True while `{date}.json` is UNDELIVERED. If the operator cuts over
        # config to `dm` + `enter` on that SAME game day and restarts scheduler before a 19:10 ET game:
        #   - run_day skips _morning_state_check AND skips the 18:10 cascade ("Pick already committed"),
        #     so NO live pick is ever delivered, while `day-outcome` under `intent="enter"` flags the
        #     private_locked unentered pick as `entry_unverifiable` / `not_entered`!
        # REAL EDGE INTERACTION B (non-delivered classification lock -> restart):
        # Undelivered 03:00 preview for 13:10 ET game is seen after game start (or previous_state.pick_locked=True
        # with scoreable=False), operator removes the stale preview and restarts at 14:00 ET before 18:10 ET check:
        # `previous_state.pick_locked == True` suppresses `_morning_state_check` at 14:00 ET!
        with _harness(base / "private_cutover", t_morning, delivery="private", intent="research") as env:
            env["games_by_date"][day] = [
                _game_dict(101, day, hhmm_utc="17:10:00"),  # 13:10 ET
                _game_dict(102, day, hhmm_utc="23:10:00"),  # 19:10 ET
            ]
            matinee_frame = pd.DataFrame([{
                "batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
                "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY",
                "p_hit_pa": 0.30, "p_game_hit": 0.88, "flags": "", "projected_lineup": False,
                "game_pk": 101, "game_time": f"{day}T17:10:00Z",
            }])
            with patch.object(orchestrator, "run_cascade", return_value=(matinee_frame, "local")):
                scheduler.run_day(day, env["config"])
            prev_st = scheduler.load_state(day, env["picks_dir"])
            assert prev_st is not None and prev_st.pick_locked is True
            assert daily_decision.is_scoreable_commit(day, env["picks_dir"], None) is True
            assert not picks.pick_was_delivered(picks.load_pick(day, env["picks_dir"]))

            # Cut over to DM + enter at 14:00 ET: scheduler treats private_locked as already committed,
            # never delivers for 19:10 ET game, and day-outcome flags entry_unverifiable!
            env["config"]["scheduler"]["pick_delivery"] = "dm"
            env["config"]["scheduler"]["entry_intent"] = "enter"
            env["config"]["scheduler"]["contest_state_required"] = True
            env["clock"][0] = dt.datetime(2027, 7, 15, 14, 0, tzinfo=ET)
            scheduler.run_day(day, env["config"])
            assert env["pick_dms"] == []
            cutover_outcome = day_outcome.gather(
                day, env["picks_dir"], env["config"],
                expected_username="stonehengee", fetch_schedule=scheduler.fetch_schedule,
            )
            assert cutover_outcome.state == "entry_unverifiable" and cutover_outcome.ok is False

        # Now test Interaction B: previous_state.pick_locked=True with NO scoreable commit (e.g. classification
        # lock on started undelivered preview, then operator deletes preview & restarts at 14:00 ET):
        with _harness(base / "prior_lock_no_commit", dt.datetime(2027, 7, 15, 14, 0, tzinfo=ET)) as env:
            env["games_by_date"][day] = [_game_dict(102, day, hhmm_utc="23:10:00")]
            prior = scheduler.SchedulerState(day, t_morning.isoformat(), [], [], [], True, t_morning.isoformat(), None, None)
            scheduler.save_state(prior, env["picks_dir"])
            dm_times: list[str] = []
            health_alert.send_dm.side_effect = lambda r, t: (
                dm_times.append(env["clock"][0].strftime("%H:%M")),
                env["health_dms"].append((r, t)),
                "hdm",
            )[-1]
            night_frame = pd.DataFrame([{
                "batter_id": 665742, "batter_name": "Juan Soto", "team": "NYM", "lineup": 2,
                "pitcher_id": 605483, "pitcher_name": "Blake Snell", "opp": "LAD",
                "p_hit_pa": 0.28, "p_game_hit": 0.86, "flags": "", "projected_lineup": False,
                "game_pk": 102, "game_time": f"{day}T23:10:00Z",
            }])
            with patch.object(orchestrator, "run_cascade", return_value=(night_frame, "local")):
                scheduler.run_day(day, env["config"])
            morning_skipped_on_prior_noncommit_lock = ("14:00" not in dm_times and "16:10" in dm_times)

        # 3. Check shipped MDP policy sensitivity to uninitialized saver_state (saver_available=False vs True)
        # while _morning_state_check reports "accepted" without warning on uninitialized saver_state!
        mdp_diffs: list[tuple[str, int, float, str, str]] = []
        old_cache = strategy._mdp_cache
        try:
            strategy._mdp_cache = None
            mdp = strategy._load_mdp()
            for test_day in ("2027-03-24", "2027-07-15"):
                for s in range(0, 16):
                    for p_cand in (0.68, 0.72, 0.76, 0.80, 0.84):
                        ctx = strategy.DecisionContext(
                            primary_p=p_cand, second_p=p_cand, has_diff_game=True,
                            date=test_day, allow_double=True, mdp=mdp,
                            best_streak=s, best_status="trusted",
                        )
                        a_true = strategy.resolve_policy_decision(ctx, s, True).action
                        a_false = strategy.resolve_policy_decision(ctx, s, False).action
                        if a_true != a_false:
                            mdp_diffs.append((test_day, s, p_cand, a_true, a_false))
        finally:
            strategy._mdp_cache = old_cache

        return {
            "delivered_pick_restart_stays_quiet": True,
            "private_to_dm_sameday_cutover_blocks_live_pick_as_already_committed": True,
            "private_to_dm_cutover_day_outcome": cutover_outcome.state,
            "prior_noncommit_lock_skips_morning_check": morning_skipped_on_prior_noncommit_lock,
            "first_dm_after_prior_noncommit_lock_at": dm_times,
            "mdp_action_diffs_when_saver_uninitialized_count": len(mdp_diffs),
            "mdp_action_diffs_sample": mdp_diffs[:8],
        }


def test_item1d_live_refusal_vs_health_and_day_outcome(tmp_path: Path | None = None) -> dict:
    """4b-2 live refusal vs health.contest_state (same-day + historical) and bts day-outcome."""
    with _temp_root(tmp_path) as base:
        day = "2027-07-15"
        start = dt.datetime(2027, 7, 15, 10, 0, tzinfo=ET)

        # 1. Pure 4b-2 refusal all day (uncovered auto observation -> ContestStateError)
        with _harness(base / "pure_refusal", start, health_enabled=True) as env:
            _write_auto_obs(env["picks_dir"], start, streak=5, covered=False)
            with patch.object(orchestrator, "run_cascade", return_value=(_pred_frame(day), "local")):
                scheduler.run_day(day, env["config"])

            same_day_alerts = health_contest_state.check(env["picks_dir"], expected=True, now=env["clock"][0])
            assert any(a.level == "CRITICAL" and a.source == "contest_state" for a in same_day_alerts)

            # Post-midnight C1 historical mode also flags uncovered auto as CRITICAL
            env["clock"][0] = dt.datetime(2027, 7, 16, 0, 30, tzinfo=ET)
            eod_alerts = health_runner.run_all_checks(
                env["picks_dir"], env["models_dir"], None,
                today=dt.date(2027, 7, 15), contest_state_expected=True,
            )
            assert any(a.level == "CRITICAL" and a.source == "contest_state" for a in eod_alerts)

            out1 = day_outcome.gather(
                day, env["picks_dir"], env["config"],
                expected_username="stonehengee", fetch_schedule=scheduler.fetch_schedule,
            )
            assert out1.state == "no_decision" and out1.ok is False

        # 2. REAL INTERACTION: Check 1 (16:10 ET) under automatic streak=30 gets MDP skip (p=0.62).
        # At 16:30 ET operator runs `bts season-contract revoke` (real CLI) because automatic streak=30
        # was wrong, forgetting to enter today's dated observation. Check 2 (17:10) and Check 3 (18:10)
        # have a 0.86 batter that WOULD be picked at streak=0, yet raise ContestStateError (4b-2).
        # EOD writes decision.json action="skip", source="mdp" from the revoked Check-1 state,
        # and next-morning `bts day-outcome` reports `deliberate_skip` (ok=True, exit 0, NO DM)!
        with _harness(base / "skip_then_revoke", start) as env:
            _write_auto_obs(env["picks_dir"], start, streak=30, covered=True, saver_flag="used")
            check_num = [0]

            def cascade_skip_then_strong(_tiers, _date):
                check_num[0] += 1
                if check_num[0] == 1:
                    return _pred_frame(day, p1=0.62, p2=0.60, projected=True), "local"
                return _pred_frame(day, p1=0.86, p2=0.85, projected=False), "local"

            orig_sleep = scheduler._watchdog_ping_sleep

            def sleep_and_revoke(secs, *a, **kw):
                orig_sleep(secs, *a, **kw)
                if env["clock"][0].strftime("%H:%M") == "17:10" and check_num[0] == 1:
                    with patch.object(season_contract, "ET", ET):
                        r_rev = CliRunner().invoke(cli_mod.cli, [
                            "season-contract", "revoke",
                            "--picks-dir", str(env["picks_dir"]),
                            "--season", "2027",
                            "--reason", "streak 30 was stale/wrong",
                        ])
                        assert r_rev.exit_code == 0, r_rev.output

            # Patch datetime in cli season-contract revoke via PinnedDatetime
            Clock = _make_Clock(env["clock"])
            with patch("datetime.datetime", Clock) if False else nullcontext(), \
                 patch.object(orchestrator, "run_cascade", side_effect=cascade_skip_then_strong), \
                 patch.object(scheduler, "_watchdog_ping_sleep", side_effect=sleep_and_revoke):
                # Note: CLI season_contract_revoke imports datetime from datetime; current real year is 2026 in summary?
                # Wait: check real wall year vs 2027! Let's invoke revoke with patched season_contract_revoke or --season matching!
                pass

        # Re-run subcase 2 cleanly with `cli_mod` clock-independent revoke (patching datetime inside bts.season_contract / writing via real CLI when year matches or via helper):
        with _harness(base / "skip_then_revoke_clean", start) as env:
            _write_auto_obs(env["picks_dir"], start, streak=30, covered=True, saver_flag="used")
            check_num2 = [0]

            def cascade_fn2(_tiers, _date):
                check_num2[0] += 1
                if check_num2[0] == 1:
                    return _pred_frame(day, p1=0.62, p2=0.60, projected=True), "local"
                return _pred_frame(day, p1=0.86, p2=0.85, projected=False), "local"

            orig_sleep2 = scheduler._watchdog_ping_sleep

            def sleep_and_revoke2(secs, *a, **kw):
                orig_sleep2(secs, *a, **kw)
                if env["clock"][0].strftime("%H:%M") == "17:10" and check_num2[0] == 1:
                    # Exact state written by `bts season-contract revoke --season 2027 --reason ...`
                    acct = env["picks_dir"] / "account_state"
                    (acct / "season_contract.json").write_text(json.dumps({
                        "schema_version": "bts_season_contract_v1",
                        "season": 2027,
                        "status": "revoked",
                        "activation_id": "a" * 32,
                        "switched_at": start.astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                        "revoked_at": env["clock"][0].astimezone(dt.timezone.utc).isoformat().replace("+00:00", "Z"),
                        "note": "streak 30 was wrong",
                    }, indent=2) + "\n")
                    (acct / "saver_state.json").rename(acct / "saver_state.revoked.json")

            with patch.object(orchestrator, "run_cascade", side_effect=cascade_fn2), \
                 patch.object(scheduler, "_watchdog_ping_sleep", side_effect=sleep_and_revoke2):
                scheduler.run_day(day, env["config"])

            dec = daily_decision.load_decision(day, env["picks_dir"])
            out2 = day_outcome.gather(
                day, env["picks_dir"], env["config"],
                expected_username="stonehengee", fetch_schedule=scheduler.fetch_schedule,
            )
            cfg_toml = base / "skip_then_revoke_clean" / "orch.toml"
            cfg_toml.write_text(
                f'[orchestrator]\npicks_dir = {json.dumps(str(env["picks_dir"]))}\n'
                '[bluesky]\ndm_recipient = "operator.bsky.social"\n'
                '[scheduler]\npick_delivery = "dm"\nentry_intent = "enter"\ncontest_state_required = true\n'
            )
            with patch.object(scheduler, "fetch_schedule", return_value=[_game_dict(101, day)]):
                cli_do = CliRunner().invoke(cli_mod.cli, [
                    "day-outcome", "--date", day, "--config", str(cfg_toml),
                    "--picks-dir", str(env["picks_dir"]),
                    "--expected-username", "stonehengee",
                    "--dm-recipient", "operator.bsky.social",
                ])

            assert dec is not None and dec["action"] == "skip" and dec["source"] == "mdp" and dec["streak"] == 30
            assert out2.state == "deliberate_skip" and out2.ok is True
            assert cli_do.exit_code == 0
            # Afternoon refusal DM DID fire once at 17:10 ET, while next-morning day-outcome reports clean deliberate_skip!
            assert len(env["health_dms"]) == 1 and "revoked" in env["health_dms"][0][1]

        return {
            "pure_refusal_day_outcome": out1.state,
            "skip_then_revoke_writes_revoked_skip_decision": True,
            "skip_then_revoke_day_outcome_state": out2.state,
            "skip_then_revoke_afternoon_dm_fired": len(env["health_dms"]) == 1,
        }


def test_item1e_cs2_streak_season_vs_reconcile_and_check_results_new_year_and_opener(tmp_path: Path | None = None) -> dict:
    """CS2 season-scoped streak.json vs 02:00 reconcile and 01:00 check-results across New Year & Opener."""
    with _temp_root(tmp_path) as base:

        # 1. Across New Year (2027-12-31 -> 2028-01-01):
        # 2027-12-31 late game still in progress at 01:00 ET (check-results --wait-deadline-et 06:00).
        # At 02:00 ET, `bts reconcile` runs, sees 0 files matching `2028-*.json`, and overwrites
        # streak.json with {"streak": 0, "saver_available": true, "season": 2028, "applied_dates": []}!
        # At 02:15 ET, `check-results --date 2027-12-31` (and/or `run_result_polling`) scores 2027-12-31:
        #   - On a HIT: streak becomes 1 instead of 13!
        #   - On a MISS in saver zone (streak=12, saver=True): resets to 0 instead of holding 12!
        ny_hit_dir = base / "ny_hit"
        ny_hit_dir.mkdir(parents=True)
        picks.save_streak(12, ny_hit_dir, saver_available=True, applied_date="2027-12-30", season=2027)
        dec31_pick = picks.DailyPick(
            date="2027-12-31",
            run_time="2027-12-31T23:00:00+00:00",
            pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Gerrit Cole", 543037, 0.84, [], False, 991, "2027-12-31T23:10:00Z"),
            double_down=None,
            runner_up=None,
            bluesky_posted=True,
            bluesky_uri="at://did:plc:test/post/dec31",
            notification_sent=True,
            delivery_attempted=True,
        )
        picks.save_pick(dec31_pick, ny_hit_dir)

        clock_ny = [dt.datetime(2028, 1, 1, 1, 0, tzinfo=ET)]
        attempt_idx = [0]

        def resolve_after_0200(_daily, _date):
            attempt_idx[0] += 1
            if clock_ny[0] < dt.datetime(2028, 1, 1, 2, 15, tzinfo=ET):
                return None  # still in extra innings at 01:00..02:00 ET!
            return {"pick": "hit"}

        def sleep_interleave_reconcile(secs: float):
            prev = clock_ny[0]
            clock_ny[0] = clock_ny[0] + dt.timedelta(seconds=secs)
            t_0200 = dt.datetime(2028, 1, 1, 2, 0, tzinfo=ET)
            if prev < t_0200 <= clock_ny[0]:
                # Cron fires 02:00 `bts reconcile` while check-results is sleeping!
                with patch("bts.reconcile_receipt.ReconcileReceipt",
                           side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_0200, lookback_days=kw["lookback_days"])):
                    r_rec = CliRunner().invoke(cli_mod.cli, ["reconcile", "--picks-dir", str(ny_hit_dir)])
                    assert r_rec.exit_code == 0, r_rec.output

        with patch.object(cli_mod, "_now_et", side_effect=lambda: clock_ny[0]), \
             patch.object(cli_mod, "_sleep", side_effect=sleep_interleave_reconcile), \
             patch.object(picks, "resolve_daily_slot_results", side_effect=resolve_after_0200), \
             patch.object(picks, "get_game_statuses_detailed", return_value={}):
            r_cr = CliRunner().invoke(cli_mod.cli, [
                "check-results", "--date", "2027-12-31",
                "--picks-dir", str(ny_hit_dir),
                "--wait-deadline-et", "06:00",
            ])
        assert r_cr.exit_code == 0, r_cr.output
        after_ny_hit = json.loads((ny_hit_dir / "streak.json").read_text())
        # Demonstrates 02:00 reconcile clobbered streak.json to season=2028, so check-results scored 2027-12-31 as 1 (not 13)!
        assert after_ny_hit["season"] == 2027 and after_ny_hit["streak"] == 1

        # 2. Opening Night 2027-03-24 from restored 2026 untagged streak.json (streak=13, saver_available=false):
        opener_dir = base / "opener"
        opener_dir.mkdir(parents=True)
        (opener_dir / "streak.json").write_text(json.dumps({
            "streak": 13, "saver_available": False, "applied_dates": ["2026-09-27"],
        }))
        (opener_dir / "2026-09-27.json").write_text(json.dumps({"date": "2026-09-27", "result": "hit"}))

        # 02:00 reconcile on 2027-03-24 (with 03:00 preview NOT yet run) cleanly initializes 2027 to (0, True)
        t_open_rec = dt.datetime(2027, 3, 24, 2, 0, tzinfo=ET)
        with patch("bts.reconcile_receipt.ReconcileReceipt",
                   side_effect=lambda **kw: ReconcileReceipt(clock=lambda: t_open_rec, lookback_days=kw["lookback_days"])), \
             patch.object(picks, "get_game_statuses_detailed", return_value={}):
            r_open = CliRunner().invoke(cli_mod.cli, ["reconcile", "--picks-dir", str(opener_dir)])
        assert r_open.exit_code == 0, r_open.output
        open_doc = json.loads((opener_dir / "streak.json").read_text())
        assert (open_doc["streak"], open_doc["saver_available"], open_doc["season"]) == (0, True, 2027)

        return {
            "new_year_0200_reconcile_clobbers_pending_dec31_streak_12_to_1": (after_ny_hit["streak"] == 1),
            "check_results_dec31_output": r_cr.output.strip().splitlines()[-1],
            "opener_0200_reconcile_resets_2026_13_to_2027_0": True,
        }


def test_item1f_bullpen_asof_and_probable_cache_vs_cold_start(tmp_path: Path | None = None) -> dict:
    """Bullpen merge_asof fix + probable_pitcher_cache (W15) vs cold-start path."""
    from bts.data.build import build_season
    from bts.features import compute, probable_pitcher_cache
    import bts.model.predict as predict_mod

    with _temp_root(tmp_path) as base:
        raw_dir = base / "data" / "raw"
        proc_dir = base / "data" / "processed"
        models_dir = base / "data" / "models"
        status_dir = base / "data" / "health_state" / "probable_pitcher_cache"
        raw_dir.mkdir(parents=True)
        proc_dir.mkdir(parents=True)
        models_dir.mkdir(parents=True)

        # Build synthetic 2026 regular-season games with reliever PAs + 2027 Spring Training games
        def make_feed(pk: int, season: int, day_str: str, gtype: str = "R", reliever_hit: bool = False) -> dict:
            return {
                "gameData": {
                    "game": {"pk": pk, "type": gtype, "season": str(season)},
                    "datetime": {"officialDate": day_str, "dateTime": f"{day_str}T23:10:00Z"},
                    "status": {"detailedState": "Final", "abstractGameCode": "F"},
                    "teams": {
                        "away": {"id": 114, "name": "Guardians", "abbreviation": "CLE"},
                        "home": {"id": 147, "name": "Yankees", "abbreviation": "NYY"},
                    },
                    "probablePitchers": {
                        "away": {"id": 4001, "fullName": "Away Starter"},
                        "home": {"id": 5001, "fullName": "Home Starter"},
                    },
                    "venue": {"id": 3313, "name": "Yankee Stadium"},
                    "weather": {"temp": "72", "wind": "5 mph, Out To CF", "condition": "Clear"},
                },
                "liveData": {
                    "plays": {
                        "allPlays": [
                            {
                                "result": {"eventType": "single" if reliever_hit else "field_out", "event": "Single" if reliever_hit else "Flyout"},
                                "about": {"inning": 8, "halfInning": "top", "isTopInning": True, "atBatIndex": 0},
                                "matchup": {
                                    "batter": {"id": 680757, "fullName": "Steven Kwan"},
                                    "batSide": {"code": "L"},
                                    "pitcher": {"id": 5999, "fullName": "Home Reliever"},  # != 5001 probable starter!
                                    "pitchHand": {"code": "R"},
                                },
                                "playEvents": [{
                                    "isPitch": True,
                                    "details": {"type": {"code": "FF"}, "call": {"code": "X"}},
                                    "pitchData": {
                                        "startSpeed": 95.0,
                                        "coordinates": {"pX": 0.0, "pZ": 2.5},
                                        "strikeZoneTop": 3.4,
                                        "strikeZoneBottom": 1.6,
                                    },
                                    "hitData": {
                                        "launchSpeed": 99.0,
                                        "launchAngle": 14.0,
                                        "trajectory": "line_drive",
                                        "hardness": "hard",
                                        "totalDistance": 285.0,
                                    },
                                }],
                            }
                        ]
                    },
                    "boxscore": {
                        "officials": [{"officialType": "Home Plate", "official": {"id": 9001}}],
                        "teams": {"away": {"players": {}}, "home": {"players": {}}},
                    },
                },
            }

        (raw_dir / "2026").mkdir()
        for idx in range(1, 16):
            # Days 1..14 reliever_hit=False (bp_hr = 0.0); Day 15 (last 2026 game) reliever_hit=True (bp_hr = 1.0)
            f26 = make_feed(2600 + idx, 2026, f"2026-09-{idx:02d}", "R", reliever_hit=(idx == 15))
            (raw_dir / "2026" / f"{2600 + idx}.json").write_text(json.dumps(f26))

        (raw_dir / "2027").mkdir()
        f27_spring = make_feed(2701, 2027, "2027-03-21", "S", reliever_hit=True)
        # Include {"away": null} in spring training probablePitchers to test W15 scan resilience!
        f27_spring_null = copy.deepcopy(f27_spring)
        f27_spring_null["gameData"]["game"]["pk"] = 2702
        f27_spring_null["gameData"]["probablePitchers"] = {"away": None, "home": {"id": 5001}}
        (raw_dir / "2027" / "2701.json").write_text(json.dumps(f27_spring))
        (raw_dir / "2027" / "2702.json").write_text(json.dumps(f27_spring_null))

        df26 = build_season(raw_dir, proc_dir / "pa_2026.parquet", 2026)
        assert df26 is not None and len(df26) == 15

        # 1. CLI `bts data build --seasons 2027` is a clean no-op exit 0 before first regular-season game
        r_build = CliRunner().invoke(cli_mod.cli, [
            "data", "build", "--raw-dir", str(raw_dir), "--out-dir", str(proc_dir), "--seasons", "2027",
        ])
        assert r_build.exit_code == 0 and not (proc_dir / "pa_2027.parquet").exists()

        # 2. _refresh_season_data("2027-03-24") pulls March 20..23 schedule, sees 0 regular games, leaves no marker
        pull_calls: list[tuple[str, str]] = []
        with patch("bts.data.pull.pull_feeds", side_effect=lambda s, e, r, **kw: (pull_calls.append((s, e)), [])[1]):
            predict_mod._refresh_season_data("2027-03-24", raw_dir=str(raw_dir), processed_dir=str(proc_dir))
            predict_mod._refresh_season_data("2027-03-24", raw_dir=str(raw_dir), processed_dir=str(proc_dir))
        assert len(pull_calls) == 2  # No .refreshed_2027_through_2027-03-23 written on cold start -> runs each uncached cascade!

        # 3. Bullpen feature + probable_pitcher_cache on cold start (including partial cache + archived raw!)
        cache_path = models_dir / "probable_pitcher_lookup.json"
        with patch.object(compute, "_probable_pitcher_cache_path", cache_path):
            lookup = probable_pitcher_cache.build(cache_path, raw_dir)
            assert 2601 in lookup and 2701 in lookup and 2702 not in lookup

            # Remove half of 2026 raw files AND remove 2614 from cache to test mixed int/None _pitcher_team_id in merge_asof!
            for f in (raw_dir / "2026").glob("*.json"):
                f.unlink()
            partial_cache = json.loads(cache_path.read_text())
            partial_cache.pop("2614", None)
            cache_path.write_text(json.dumps(partial_cache))

            pa = pd.read_parquet(proc_dir / "pa_2026.parquet")
            with patch.object(compute, "_build_probable_pitcher_lookup",
                              side_effect=lambda *_a: probable_pitcher_cache.build(cache_path, raw_dir)):
                feat = compute.compute_all_features(pa.copy())

            lookups = predict_mod._build_feature_lookups(feat)
            # Days 1..13 all had bp_hr=0.0; Day 15 (2026-09-15) had bp_hr=1.0, yet Day 15's row gets strictly-prior 0.0 (M3 lag)!
            bp_served = lookups["bullpen"][147]
            assert bp_served == 0.0

        return {
            "cli_data_build_2027_noop_exit_0": True,
            "cold_start_refresh_repeats_without_marker": len(pull_calls) == 2,
            "partial_cache_mixed_team_id_merge_asof_clean": True,
            "bullpen_serving_lags_final_historical_game_m3": (bp_served == 0.0),
        }


def run_all() -> dict:
    results = {
        "1a_5a_vs_fallbacks": test_item1a_saved_pick_refusal_vs_both_fallbacks_and_double_cascade_overrun(),
        "1b_4c_vs_dated_and_bts_run": test_item1b_stale_reading_stamp_vs_dated_and_bts_run(),
        "1c_6b1_vs_restart_and_cutover": test_item1c_morning_check_vs_sameday_restart_and_private_cutover(),
        "1d_4b2_vs_health_and_day_outcome": test_item1d_live_refusal_vs_health_and_day_outcome(),
        "1e_cs2_vs_reconcile_and_check_results": test_item1e_cs2_streak_season_vs_reconcile_and_check_results_new_year_and_opener(),
        "1f_bullpen_and_cache_vs_cold_start": test_item1f_bullpen_asof_and_probable_cache_vs_cold_start(),
    }
    return results


if __name__ == "__main__":
    import shutil
    try:
        out = run_all()
        print(json.dumps(out, indent=2, default=str))
    finally:
        shutil.rmtree(SCRATCH_TMP, ignore_errors=True)
