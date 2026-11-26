"""Slice 3 probe family 2: clocks and years across 2027-03-24, midnight ET, 2027-09-26/27, 2027-12-31, and 2028.

Adoptable as a pytest regression suite and runnable standalone via:
    nice -n 19 .venv/bin/python .carbon-review/whole-range/test_probe_clocks_and_years.py
No module-level import of bts.model.predict, no leaked process-wide state, no network.
"""
from __future__ import annotations

from contextlib import ExitStack, contextmanager
from datetime import date, datetime, time as dt_time, timedelta, timezone
import json
from pathlib import Path
import socket
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import (
    cli,
    contest_fetch,
    contest_state,
    daily_decision,
    day_outcome,
    dm,
    orchestrator,
    picks,
    saver_state,
    scheduler,
    season_contract,
    skip_policy_shadow,
    strategy,
    web,
)
from bts.features import park_drag_producer
from bts.health import (
    alert as health_alert,
    contest_state as health_contest_state,
    pick_entry as health_pick_entry,
    runner as health_runner,
    tail_policy as health_tail_policy,
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


def test_2027_03_24_opening_night_boundaries(tmp_path: Path):
    """Check 2027-03-24 Opening Night across strategy, tail health, thin-slate schedule, and _refresh_season_data."""
    from bts.model import predict as model_predict
    from bts.data import pull as data_pull

    day = "2027-03-24"
    assert strategy.season_end_date(day) == "2027-09-27"
    assert strategy._days_remaining(day) == 187
    assert picks.season_of_date(day) == 2027
    from bts.simulate import mdp as sim_mdp, tail_policy as sim_tp

    # Shipped artifacts healthy on 2027-03-24
    assert health_tail_policy.check(
        base_path=sim_mdp.DEFAULT_POLICY_PATH, tail_path=sim_tp.DEFAULT_TAIL_POLICY_PATH, today=date(2027, 3, 24)
    ) == []

    # 1-game Opening Night gets thin-slate early checks in plan_run_times
    games = [{
        "gamePk": 101,
        "gameDate": "2027-03-25T00:05:00Z",  # 20:05 ET on 2027-03-24
        "officialDate": day,
        "doubleHeader": "N",
        "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
    }]
    runs = scheduler.plan_run_times(games, now=REAL_DATETIME(2027, 3, 24, 10, 0, tzinfo=ET))
    assert len(runs) >= 2  # Thin-slate adds early check(s) ahead of regular T-60 check

    # _refresh_season_data("2027-03-24"): yesterday="2027-03-23" >= "2027-03-20", handles empty 2027 cleanly
    raw_dir = tmp_path / "raw"
    proc_dir = tmp_path / "proc"
    raw_dir.mkdir()
    proc_dir.mkdir()
    with block_network(), patch.object(data_pull, "pull_feeds", return_value=[]):
        model_predict._refresh_season_data(day, raw_dir=str(raw_dir), processed_dir=str(proc_dir))
    assert not (proc_dir / "pa_2027.parquet").exists()


def test_midnight_et_in_day_health_dm_dedup_spends_tomorrow_budget(tmp_path: Path):
    """Demonstrates post-midnight in-day scheduler CRITICAL DMs stamping D+1 into health_dm_delivery_status.json."""
    picks_dir = tmp_path / "data" / "picks"
    models_dir = tmp_path / "data" / "models"
    picks_dir.mkdir(parents=True)
    models_dir.mkdir(parents=True)
    status_file = tmp_path / "data" / "health_state" / "health_dm_delivery_status.json"

    # Slate D = 2027-07-15 has a late rain-delayed / West Coast game at 2027-07-16 00:15 ET (04:15 UTC)
    slate_d = "2027-07-15"
    slate_d1 = "2027-07-16"
    post_midnight = [REAL_DATETIME(2027, 7, 16, 0, 2, tzinfo=ET)]
    clock_cls = _make_clock(post_midnight)
    sent_health: list[tuple[str, str]] = []

    cfg = {
        "orchestrator": {"picks_dir": str(picks_dir), "models_dir": str(models_dir)},
        "bluesky": {"dm_recipient": "operator.bsky.social"},
        "scheduler": {"pick_delivery": "dm", "entry_intent": "enter", "contest_state_required": True},
        "health_checks": {"enabled": True, "contest_state_expected": True},
    }
    late_daily = picks.DailyPick(
        date=slate_d,
        run_time="2027-07-15T22:00:00-04:00",
        pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Cole", 543037, 0.85, [], False, 101,
                        "2027-07-16T04:15:00Z"),
        double_down=None,
        runner_up=None,
    )

    with block_network(), \
         patch.object(health_alert, "datetime", clock_cls), \
         patch.object(health_runner, "datetime", clock_cls), \
         patch.object(contest_state, "datetime", clock_cls), \
         patch.object(scheduler, "_now_et", lambda: post_midnight[0]), \
         patch.object(health_alert, "send_dm", lambda r, b: sent_health.append((r, b)) or "hdm"):
        # 1) At 00:02 ET on 2027-07-16 (processing slate 2027-07-15!), contest_state refusal + missed_pick fire:
        scheduler._alert_contest_state_failure(cfg, contest_state.ContestStateError("slate D refusal"))
        scheduler._alert_missed_pick(cfg, late_daily, mins_to_game=10.0)

        assert len(sent_health) == 2
        saved_status = json.loads(status_file.read_text())
        # BUG: Both stamped wall date 2027-07-16 (D+1) instead of processed slate 2027-07-15!
        assert saved_status["sent_day"] == slate_d1
        assert set(saved_status["sent_sources"]) == {"CRITICAL:contest_state", "CRITICAL:missed_pick"}

        # 2) At 01:10 ET cron fetch heals contest_state (and operator sets saver); EOD run_all_checks for 2027-07-15
        #    emits 0 alerts -> status_file is NOT rewritten, keeping sent_day="2027-07-16"!
        acct = picks_dir / "account_state"
        acct.mkdir(parents=True, exist_ok=True)
        act_id = "b" * 32
        (acct / "season_contract.json").write_text(json.dumps({
            "schema_version": season_contract.SCHEMA, "season": 2027, "status": "automatic",
            "activation_id": act_id, "switched_at": "2027-07-15T12:00:00Z", "revoked_at": None, "note": None,
        }))
        (acct / "contest_streak.json").write_text(json.dumps({
            "schema_version": season_contract.AUTO_SCHEMA, "season": 2027, "activation_id": act_id,
            "active_streak": 5, "best_streak": 10, "source": "mlb_bts_profile",
            "source_date": slate_d, "recorded_at": "2027-07-15T23:00:00Z",
        }))
        (acct / "saver_state.json").write_text(json.dumps({"season": 2027, "state": "used"}))
        (picks_dir / "streak.json").write_text(json.dumps({"season": 2027, "streak": 5, "saver_available": False}))

        safe_run = health_runner._safe_run
        with patch.object(health_runner, "_safe_run",
                          lambda name, fn: safe_run(name, fn) if name == "contest_state" else []):
            eod_alerts = health_runner.run_all_checks(
                picks_dir, models_dir, dm_recipient="operator.bsky.social",
                today=date(2027, 7, 15), contest_state_expected=True,
            )
        assert eod_alerts == []
        assert json.loads(status_file.read_text())["sent_day"] == slate_d1

        # 3) On 2027-07-16 at 16:10 ET, contest_state breaks (e.g. revoked) AND missed_pick fires for 2027-07-16!
        post_midnight[0] = REAL_DATETIME(2027, 7, 16, 16, 10, tzinfo=ET)
        sent_health.clear()
        scheduler._alert_contest_state_failure(cfg, contest_state.ContestStateError("slate D+1 refusal"))
        d1_daily = picks.DailyPick(
            date=slate_d1, run_time="2027-07-16T16:00:00-04:00",
            pick=late_daily.pick, double_down=None, runner_up=None,
        )
        scheduler._alert_missed_pick(cfg, d1_daily, mins_to_game=10.0)
        # BOTH CRITICAL DMs on 2027-07-16 are SILENTLY SUPPRESSED (lost alerts)!
        assert sent_health == []

        # 4) Also verify _refuse_late_delivery across midnight fails to dedup against same-slate EOD!
        status_file.unlink(missing_ok=True)
        post_midnight[0] = REAL_DATETIME(2027, 7, 16, 0, 12, tzinfo=ET)
        st_late = scheduler.SchedulerState(slate_d, post_midnight[0].isoformat(), [], [], [], False, None, None, None)
        cutoff_d = REAL_DATETIME(2027, 7, 16, 0, 10, tzinfo=ET)
        scheduler._refuse_late_delivery(
            late_daily, cfg, picks_dir, st_late, slate_d, "fallback", cutoff_d, post_midnight[0]
        )
        assert len(sent_health) == 1
        assert json.loads(status_file.read_text())["sent_day"] == slate_d1
        # When slate D's EOD runs at 00:20 ET, now_et_date=2027-07-15 != 2027-07-16 -> sends duplicate CRITICAL DM!
        with patch.object(health_runner, "_safe_run",
                          lambda name, fn: safe_run(name, fn) if name == "late_delivery" else []):
            health_runner.run_all_checks(
                picks_dir, models_dir, dm_recipient="operator.bsky.social",
                today=date(2027, 7, 15), contest_state_expected=True,
            )
        assert len(sent_health) == 2


def test_midnight_et_watchdog_post_midnight_pitch_blind_spot(tmp_path: Path):
    """Demonstrates check-pick-entered --cron missing entries/escalations after 00:00 ET for a 00:35 ET game."""
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    slate_d = "2027-07-15"
    # Game on 2027-07-15 slate with first pitch at 2027-07-16 00:35 ET (cutoff 00:30 ET)
    dp = picks.DailyPick(
        date=slate_d,
        run_time="2027-07-15T23:20:00-04:00",
        pick=picks.Pick("Steven Kwan", 680757, "CLE", 1, "Cole", 543037, 0.85, [], False, 101,
                        "2027-07-16T00:35:00-04:00"),
        double_down=None, runner_up=None,
        delivered_at="2027-07-15T23:20:00-04:00",
        notification_sent=True, notification_channel="bluesky_dm", notification_id="dm1",
    )
    picks.save_pick(dp, picks_dir)
    daily_decision.write_decision(
        slate_d, picks_dir, action="single", source="mdp",
        primary={"batter_id": 680757, "batter_name": "Steven Kwan", "game_pk": 101},
        delivery_status="delivered", scoreable=True,
    )

    runner = CliRunner()
    session = SimpleNamespace(user_id="u1", username="stonehengee", xsid="x1")
    dms: list[str] = []
    pending_rows: list[dict] = []

    with block_network(), \
         patch("bts.leaderboard.auth.load_session_cookies", return_value={}), \
         patch("bts.leaderboard.auth.extract_uid", return_value="u1"), \
         patch("bts.leaderboard.auth.fetch_login_session", return_value=session), \
         patch.object(contest_fetch, "fetch_profile", return_value={"predictions": []}), \
         patch.object(contest_fetch, "fetch_pending_predictions", side_effect=lambda *_a, **_k: list(pending_rows)), \
         patch.object(cli, "_fetch_rounds", return_value={1: date(2027, 7, 15)}), \
         patch.object(cli, "_fetch_bts_to_mlb", return_value={1: 680757}), \
         patch.object(picks, "_git_head_sha", lambda *_a, **_k: "0" * 40), \
         patch.object(dm, "send_dm", lambda _r, msg: dms.append(msg) or "id"):
        # 23:30 ET and 23:45 ET: not yet entered -> sends initial alert, records no_pick receipts
        r1 = runner.invoke(cli.cli, [
            "check-pick-entered", "--cron", "--picks-dir", str(picks_dir),
            "--expected-username", "stonehengee", "--dm-recipient", "op",
            "--now-et", "2027-07-15T23:30:00",
        ])
        assert r1.exit_code == 1 and len(dms) == 1
        # At 23:55 ET (35 min BEFORE 00:30 ET cutoff!), operator enters the pick in MLB!
        pending_rows.append({"roundId": 1, "unitId": 77, "playerId": 1, "number": 1, "result": None})
        # 00:00 ET and 00:15 ET cron runs: look for 2027-07-16.json instead of 2027-07-15.json!
        for slot in ("2027-07-16T00:00:00", "2027-07-16T00:15:00"):
            r_post = runner.invoke(cli.cli, [
                "check-pick-entered", "--cron", "--picks-dir", str(picks_dir),
                "--expected-username", "stonehengee", "--dm-recipient", "op",
                "--now-et", slot,
            ])
            assert r_post.exit_code == 0 and "no pick for 2027-07-16" in r_post.stdout

    # Consequence: EOD pick_entry WARNs unconfirmed AND 08:30 day-outcome falsely reports NOT ENTERED!
    pe = health_pick_entry.check(
        picks_dir, today=date(2027, 7, 15), now=REAL_DATETIME(2027, 7, 16, 1, 0, tzinfo=ET)
    )
    assert len(pe) == 1 and pe[0].level == "WARN"
    out = day_outcome.gather(
        slate_d, picks_dir,
        {"scheduler": {"entry_intent": "enter", "pick_delivery": "dm"}},
        expected_username="stonehengee",
        fetch_schedule=lambda _d: [{"gamePk": 101, "status": {"detailedState": "Final"}}],
    )
    assert out.state == "not_entered" and not out.ok


def test_2027_09_26_and_2027_09_27_finale_and_makeup_day(tmp_path: Path):
    """Check 2027-09-26 (last scheduled game day) and 2027-09-27 (makeup day = 0 days remaining)."""
    import pandas as pd

    pred_df = pd.DataFrame([
        {"batter_id": 1, "batter_name": "B1", "team": "CLE", "lineup": 1, "pitcher_id": 10,
         "pitcher_name": "P1", "opp": "NYY", "p_hit_pa": 0.30, "p_game_hit": 0.96,
         "flags": "", "projected_lineup": False, "game_pk": 101, "game_time": "2027-09-26T19:10:00Z"},
        {"batter_id": 2, "batter_name": "B2", "team": "NYM", "lineup": 1, "pitcher_id": 11,
         "pitcher_name": "P2", "opp": "LAD", "p_hit_pa": 0.29, "p_game_hit": 0.95,
         "flags": "", "projected_lineup": False, "game_pk": 102, "game_time": "2027-09-26T19:15:00Z"},
    ])
    statuses = {101: {"abstract": "P", "detailed": "Scheduled"}, 102: {"abstract": "P", "detailed": "Scheduled"}}

    # On 2027-09-26: days_remaining == 1 -> streak 56 plays double (reach57), streak 57 plays double (longest_streak)
    assert strategy._days_remaining("2027-09-26") == 1
    with block_network():
        s56 = strategy.select_pick(pred_df, "2027-09-26", tmp_path, streak=56, saver_available=False,
                                   game_statuses_detailed=statuses)
        s57 = strategy.select_pick(pred_df, "2027-09-26", tmp_path, streak=57, saver_available=False,
                                   game_statuses_detailed=statuses)
    assert s56.decision.objective == "reach57" and s56.action in ("single", "double")
    assert s57.decision.objective == "longest_streak" and s57.action == "double"

    # On 2027-09-27 (e.g. a Monday makeup game if Sept 26 was rained out): _days_remaining == 0!
    # Every streak (including 56 and 57!) resolves to reach57 at d=0 -> SKIP!
    assert strategy._days_remaining("2027-09-27") == 0
    pred_27 = pred_df.assign(game_time=["2027-09-27T17:10:00Z", "2027-09-27T17:15:00Z"])
    with block_network():
        for st in (0, 30, 56, 57, 60):
            sel_27 = strategy.select_pick(pred_27, "2027-09-27", tmp_path, streak=st, saver_available=True,
                                           best_streak=st, best_status="trusted",
                                           game_statuses_detailed=statuses)
            assert sel_27.action == "skip" and sel_27.decision.objective == "reach57"


def test_2027_12_31_and_2028_unconfigured(tmp_path: Path):
    """Check 2027-12-31 (including 20:00 ET when UTC is 2028-01-01) and 2028 with nothing reconfigured."""
    picks_dir = tmp_path / "data" / "picks"
    picks_dir.mkdir(parents=True)
    runner = CliRunner()

    # 1) 2027-12-31 20:00 ET (= 2028-01-01 01:00 UTC): ET year is still 2027 while UTC year is 2028!
    nye_evening = [REAL_DATETIME(2027, 12, 31, 20, 0, tzinfo=ET)]
    clock_cls = _make_clock(nye_evening)
    with block_network(), \
         patch.object(cli, "datetime", clock_cls, create=True), \
         patch("datetime.datetime", clock_cls):
        res_en = runner.invoke(cli.cli, ["season-contract", "enable", "--season", "2027", "--picks-dir", str(picks_dir)])
        assert res_en.exit_code == 0, res_en.output
        res_dated = runner.invoke(cli.cli, [
            "set-contest-streak", "--date", "2027-12-31", "--streak", "5", "--best-streak", "10",
            "--saver", "used", "--picks-dir", str(picks_dir),
        ])
        assert res_dated.exit_code == 0, res_dated.output
        assert contest_state.load_decision_streak_state(
            picks_dir, now=nye_evening[0], require_contest_state=True
        ).streak == 5

    # 2) Roll 1 microsecond past midnight ET into 2028-01-01 00:00:00 ET with NOTHING reconfigured:
    jan_2028 = REAL_DATETIME(2028, 1, 1, 0, 0, 1, tzinfo=ET)
    # 2027 dated observation expired; 2027 season_contract is now 'other_season' -> live refusal!
    with pytest.raises(contest_state.ContestStateError, match="other_season"):
        contest_state.load_decision_streak_state(picks_dir, now=jan_2028, require_contest_state=True)

    # Saver flag for 2027 reads as uninitialized in 2028
    (picks_dir / "account_state" / "saver_state.json").write_text(json.dumps({"season": 2027, "state": "active"}))
    sv_2028 = saver_state.load_saver_state(picks_dir, season=2028)
    assert sv_2028.state == "uninitialized" and sv_2028.season == 2027

    # Dashboard saver context in 2028 shows uninitialized
    ctx_2028 = web.saver_dashboard_context(picks_dir, now=jan_2028)
    assert ctx_2028.state == "uninitialized" and ctx_2028.button is None

    from bts.simulate import mdp as sim_mdp, tail_policy as sim_tp

    # Strategy & tail_policy health in 2028 unconfigured: fallback 2028-09-28 + daily WARN
    assert strategy.season_end_date("2028-04-01") == "2028-09-28"
    tp_alerts = health_tail_policy.check(
        base_path=sim_mdp.DEFAULT_POLICY_PATH, tail_path=sim_tp.DEFAULT_TAIL_POLICY_PATH, today=date(2028, 4, 1)
    )
    assert any("season end for 2028 is not configured" in a.message for a in tp_alerts)

    # park_drag_producer in March 2028 with 2027 store raises ProducerError loudly
    import gzip

    pd_dir = tmp_path / "park_drag"
    prod_dir = pd_dir / "producer"
    prod_dir.mkdir(parents=True)
    (prod_dir / "pitches_current.csv.gz").write_bytes(gzip.compress(b"game_date\n2027-09-26\n"))
    (prod_dir / "games_meta.csv").write_text("game_pk\n")
    (prod_dir / "game_level_static.csv").write_text("game_pk\n")
    summary = park_drag_producer.refresh(pd_dir, today=date(2028, 3, 25))
    assert summary["ok"] is False and "store season 2027 is behind 2028" in summary["error"]


if __name__ == "__main__":
    import tempfile

    checks = [
        test_2027_03_24_opening_night_boundaries,
        test_midnight_et_in_day_health_dm_dedup_spends_tomorrow_budget,
        test_midnight_et_watchdog_post_midnight_pitch_blind_spot,
        test_2027_09_26_and_2027_09_27_finale_and_makeup_day,
        test_2027_12_31_and_2028_unconfigured,
    ]
    results = []
    for fn in checks:
        with tempfile.TemporaryDirectory() as td:
            fn(Path(td))
            results.append({"test": fn.__name__, "status": "PASS"})
    out_dir = Path(__file__).resolve().parent / "logs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "probe_clocks_and_years.json").write_text(json.dumps(results, indent=2) + "\n")
    print(f"probe_clocks_and_years: {len(results)}/{len(checks)} PASS")
