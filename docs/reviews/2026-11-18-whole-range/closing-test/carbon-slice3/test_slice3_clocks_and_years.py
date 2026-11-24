"""Slice 3 adversarial probes — Item 2: Everything that reads a clock or a year.

Verifies every changed clock/year reader across:
1. 2027-03-24 (Opening Night singleton slate)
2. Midnight ET on a game day (23:59:58 ET -> 00:00:02 ET straddle, plus 20:30 ET = 00:30Z UTC-next-day)
3. 2027-12-31 (including 20:30 ET = 2028-01-01T01:30Z UTC-next-year!)
4. 2028 with nothing reconfigured (2028-02-29 leap day, 2028-03-24, 2028-09-01 tail window)

Rules obeyed:
- No module-level import of bts.model.predict or lightgbm.
- No permanent process-wide state mutations.
- Zero network or socket calls.
"""
from __future__ import annotations

from contextlib import ExitStack, nullcontext
import datetime as dt
import json
from pathlib import Path
import tempfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pandas as pd

from bts import cli as cli_mod
from bts import contest_fetch, contest_state, daily_decision, dm, orchestrator, picks, posting, probable_pitcher_status, saver_state, scheduler, season_contract, strategy
from bts.features import probable_pitcher_cache
from bts.health import alert as health_alert
from bts.health import attention as health_attention
from bts.health import contest_state as health_contest_state
from bts.health import probable_pitcher_cache as health_pp_cache
from bts.health import runner as health_runner
from bts.health import tail_policy as health_tail_policy
from bts.reconcile_receipt import ReconcileReceipt

ET = ZoneInfo("America/New_York")
REPO_ROOT = Path(__file__).resolve().parents[2]
SCRATCH_TMP = Path(__file__).resolve().parent / "scratch_tmp_clocks"


def _temp_dir(tmp_path: Path | None):
    if tmp_path is not None:
        return nullcontext(Path(tmp_path))
    SCRATCH_TMP.mkdir(parents=True, exist_ok=True)
    return tempfile.TemporaryDirectory(dir=SCRATCH_TMP)


def _make_Clock(box: list[dt.datetime]):
    class PinnedDatetime(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            cur = box[0]
            return cur.astimezone(tz) if tz is not None else cur.replace(tzinfo=None)

    return PinnedDatetime


def _pred_frame(day: str, game_time_utc: str, p1: float = 0.85) -> pd.DataFrame:
    return pd.DataFrame([{
        "batter_id": 680757,
        "batter_name": "Steven Kwan",
        "team": "CLE",
        "lineup": 1,
        "pitcher_id": 543037,
        "pitcher_name": "Gerrit Cole",
        "opp": "NYY",
        "p_hit_pa": 0.29,
        "p_game_hit": p1,
        "flags": "",
        "projected_lineup": False,
        "game_pk": 101,
        "game_time": game_time_utc,
    }])


def test_opening_night_2027_03_24(tmp_path: Path | None = None) -> dict:
    """Verify all clock/year readers on 2027-03-24 (including 20:30 ET = 2027-03-25T00:30Z)."""
    with _temp_dir(tmp_path) as td:
        base = Path(td)
        picks_dir = base / "data" / "picks"
        picks_dir.mkdir(parents=True)

        assert strategy.season_end_date("2027-03-24") == "2027-09-27"
        assert strategy._days_remaining("2027-03-24") == 187
        assert health_tail_policy._days_remaining(dt.date(2027, 3, 24)) == 187
        assert health_tail_policy.check(
            base_path=REPO_ROOT / "data/models/mdp_policy.npz",
            tail_path=REPO_ROOT / "data/models/mdp_tail_policy.npz",
            today=dt.date(2027, 3, 24),
        ) == []

        # Singleton 19:10 ET opener -> thin-slate 3 checks [16:10, 17:10, 18:10] ET
        games = [{
            "gamePk": 101,
            "officialDate": "2027-03-24",
            "gameDate": "2027-03-24T23:10:00Z",
            "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
        }]
        runs = scheduler.plan_run_times(
            games, offset_min=60,
            now=dt.datetime(2027, 3, 24, 10, 0, tzinfo=ET),
        )
        run_hm = [r["time_et"].strftime("%H:%M") for r in runs]
        assert run_hm == ["16:10", "17:10", "18:10"], run_hm

        # At 20:30 ET (00:30Z on March 25 UTC!), ET date must still be 2027-03-24 everywhere
        t_utc_tomorrow = dt.datetime(2027, 3, 25, 0, 30, tzinfo=dt.timezone.utc)
        assert t_utc_tomorrow.astimezone(ET).date().isoformat() == "2027-03-24"
        clock = [t_utc_tomorrow]
        Clock = _make_Clock(clock)
        with patch.object(contest_state, "datetime", Clock), \
             patch.object(probable_pitcher_cache, "datetime", Clock):
            cli_mod._write_dated_contest_observation(
                picks_dir, effective_date=dt.date(2027, 3, 24),
                streak=0, best_streak=0, saver="not_earned", reason="opener", now=t_utc_tomorrow,
            )
            ds = contest_state.load_decision_streak_state(picks_dir, require_contest_state=True, now=t_utc_tomorrow)
            assert ds.observation_kind == "dated" and ds.streak == 0

            status_dir = base / "data" / "health_state" / "probable_pitcher_cache"
            pp_path = base / "data" / "models" / "pp.json"
            pp_path.parent.mkdir(parents=True, exist_ok=True)
            pp_path.write_text("{corrupt_at_utc_tomorrow")
            raw_dir = base / "data" / "raw"
            (raw_dir / "2027").mkdir(parents=True)
            probable_pitcher_cache.build(pp_path, raw_dir)
            status_files = list(status_dir.glob("2027-03-24.unreadable_cache.*.json"))
            assert len(status_files) == 1
            alerts_pp = health_pp_cache.check(status_dir, today=dt.date(2027, 3, 24))
            assert len(alerts_pp) == 1 and alerts_pp[0].level == "WARN"
            _, pp_attn = health_attention.apply_warn_attention_policy(
                alerts_pp, state_path=base / "pp_warn_state.json", today=dt.date(2027, 3, 24),
            )
            assert len(pp_attn) == 1 and pp_attn[0].source == "probable_pitcher_cache"

        return {"days_remaining": 187, "thin_slate_runs": run_hm, "utc_tomorrow_et_today_ok": True}


def test_midnight_et_on_game_day(tmp_path: Path | None = None) -> dict:
    """Verify midnight ET straddles on a game day (23:59:58 ET -> 00:00:05 ET)."""
    with _temp_dir(tmp_path) as td:
        base = Path(td)
        day = "2027-07-15"

        # Scenario A: Operator overrides wrong automatic streak=30 with dated observation streak=2 for 2027-07-15.
        # Case A1: Pick selected at 23:59:58 ET (dated streak=2), delivery runs at 00:00:02 ET (2027-07-16):
        #   load_decision_streak_state falls back to automatic (streak=30) after dated expires at 23:59:59.999999 ET,
        #   and CS5a saved_pick_refusal BLOCKS delivery (dated != automatic)!
        pdir_a1 = base / "a1" / "data" / "picks"
        pdir_a1.mkdir(parents=True)
        t_pre = dt.datetime(2027, 7, 15, 23, 59, 58, tzinfo=ET)
        t_post = dt.datetime(2027, 7, 16, 0, 0, 2, tzinfo=ET)

        # Seed covered automatic streak=30 AND dated 2027-07-15 streak=2
        acct1 = pdir_a1 / "account_state"
        acct1.mkdir(parents=True)
        (acct1 / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1", "season": 2027, "status": "automatic",
            "activation_id": "b" * 32, "switched_at": "2027-04-01T12:00:00Z", "revoked_at": None, "note": None,
        }))
        (acct1 / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": "b" * 32,
            "active_streak": 30, "best_streak": 30, "source": "mlb_bts_profile",
            "source_date": "2027-07-14", "recorded_at": "2027-07-15T18:00:00Z",
        }))
        (acct1 / "saver_state.json").write_text(json.dumps({
            "season": 2027, "state": "used", "updated_at": "2027-07-01T00:00:00Z", "source": "cli",
        }))
        cli_mod._write_dated_contest_observation(
            pdir_a1, effective_date=dt.date(2027, 7, 15),
            streak=2, best_streak=30, saver="used", reason="override wrong auto 30", now=t_pre,
        )

        clock = [t_pre]
        Clock = _make_Clock(clock)
        cfg1 = {
            "orchestrator": {"picks_dir": str(pdir_a1), "models_dir": str(REPO_ROOT / "data/models")},
            "bluesky": {"dm_recipient": "operator.bsky.social"},
            "scheduler": {"contest_state_required": True, "pick_delivery": "dm", "live_forward_capture_on_lock": False},
            "tiers": [{"name": "local", "type": "local"}],
        }
        late_game_utc = "2027-07-16T04:30:00Z"  # 00:30 ET late/delayed game on 2027-07-15 slate
        sent_pick_a1: list[str] = []
        sent_alert_a1: list[str] = []
        with ExitStack() as stack:
            for mod in (contest_state, daily_decision, picks, strategy, scheduler, health_alert):
                if hasattr(mod, "datetime"):
                    stack.enter_context(patch.object(mod, "datetime", Clock))
            stack.enter_context(patch.object(scheduler, "_now_et", side_effect=lambda: clock[0]))
            stack.enter_context(patch.object(orchestrator, "run_cascade", return_value=(_pred_frame(day, late_game_utc, p1=0.75), "local")))
            stack.enter_context(patch.object(picks, "get_game_statuses_detailed", return_value={101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}))
            stack.enter_context(patch.object(strategy, "get_game_statuses_detailed", return_value={101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}))
            stack.enter_context(patch.object(scheduler, "_delivery_game_statuses", return_value={101: {"abstract": "P", "detailed": "Scheduled", "code": "S"}}))
            stack.enter_context(patch.object(dm, "send_dm", side_effect=lambda _r, txt: (sent_pick_a1.append(txt), "dm-1")[1]))
            stack.enter_context(patch.object(health_alert, "send_dm", side_effect=lambda _r, txt: (sent_alert_a1.append(txt), "hdm-1")[1]))

            # 1) Select at 23:59:58 ET -> uses dated streak=2 (picks single at p=0.75)
            _, sel_pre, _ = orchestrator.run_and_pick(cfg1, day)
            assert sel_pre is not None and sel_pre.pick_result is not None and sel_pre.streak == 2

            # 2) Deliver at 00:00:02 ET -> dated expired, falls back to auto=30 -> CS5a blocks!
            clock[0] = t_post
            st_a1 = scheduler.SchedulerState(day, t_pre.isoformat(), [], [], [], False, None, None, None)
            ok_a1 = scheduler._deliver_and_lock_pick(
                sel_pre.pick_result.daily, cfg1, pdir_a1, st_a1, day, "lineup", selection=sel_pre,
            )
            assert ok_a1 is False and sent_pick_a1 == []
            assert len(sent_alert_a1) == 1 and "chosen on dated" in sent_alert_a1[0]

            # Case A2: BOTH run_and_pick("2027-07-15") AND _deliver_and_lock_pick run at 00:00:05 ET!
            # Because load_decision_streak_state checks wall clock ET date (2027-07-16) instead of slate `date`
            # (2027-07-15), the 2027-07-15 dated override is expired/ignored, and BOTH selection and delivery
            # silently use the wrong automatic streak=30 (delivering "Streak: 30" at p=0.86)!
            pdir_a2 = base / "a2" / "data" / "picks"
            pdir_a2.mkdir(parents=True)
            for f in acct1.iterdir():
                (pdir_a2 / "account_state").mkdir(exist_ok=True)
                (pdir_a2 / "account_state" / f.name).write_bytes(f.read_bytes())
            cfg2 = {**cfg1, "orchestrator": {**cfg1["orchestrator"], "picks_dir": str(pdir_a2)}}
            sent_pick_a2: list[str] = []
            stack.enter_context(patch.object(dm, "send_dm", side_effect=lambda _r, txt: (sent_pick_a2.append(txt), "dm-2")[1]))
            stack.enter_context(patch.object(orchestrator, "run_cascade", return_value=(_pred_frame(day, late_game_utc, p1=0.86), "local")))

            clock[0] = dt.datetime(2027, 7, 16, 0, 0, 5, tzinfo=ET)
            _, sel_post, _ = orchestrator.run_and_pick(cfg2, day)
            st_a2 = scheduler.SchedulerState(day, t_pre.isoformat(), [], [], [], False, None, None, None)
            ok_a2 = scheduler._deliver_and_lock_pick(
                sel_post.pick_result.daily, cfg2, pdir_a2, st_a2, day, "lineup", selection=sel_post,
            )
            assert ok_a2 is True and len(sent_pick_a2) == 1 and "Streak: 30" in sent_pick_a2[0]

        return {
            "pre_midnight_select_post_midnight_deliver_blocked_by_5a": (not ok_a1),
            "post_midnight_select_and_deliver_falls_back_from_slate_dated_to_auto": ok_a2,
        }


def test_dec_31_2027_and_utc_new_year(tmp_path: Path | None = None) -> dict:
    """Verify all clock/year sites on 2027-12-31, especially at 20:30 ET (2028-01-01T01:30Z UTC!)."""
    with _temp_dir(tmp_path) as td:
        base = Path(td)
        picks_dir = base / "data" / "picks"
        picks_dir.mkdir(parents=True)

        # 2027-12-31 20:30 ET == 2028-01-01 01:30 UTC!
        # Any code that uses `now.year` (UTC = 2028) instead of `now.astimezone(ET).year` (ET = 2027) will break here!
        t_dec31_et = dt.datetime(2028, 1, 1, 1, 30, tzinfo=dt.timezone.utc)
        assert t_dec31_et.year == 2028
        assert t_dec31_et.astimezone(ET).date() == dt.date(2027, 12, 31)

        assert strategy._days_remaining("2027-12-31") == 0
        assert health_tail_policy._days_remaining(dt.date(2027, 12, 31)) == 0
        assert health_tail_policy.check(
            base_path=REPO_ROOT / "data/models/mdp_policy.npz",
            tail_path=REPO_ROOT / "data/models/mdp_tail_policy.npz",
            today=dt.date(2027, 12, 31),
        ) == []

        # Seed covered automatic 2027 state + saver 2027 + streak 2027
        acct = picks_dir / "account_state"
        acct.mkdir(parents=True)
        (acct / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1", "season": 2027, "status": "automatic",
            "activation_id": "c" * 32, "switched_at": "2027-12-31T15:00:00Z", "revoked_at": None, "note": None,
        }))
        (acct / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": "c" * 32,
            "active_streak": 12, "best_streak": 20, "source": "mlb_bts_profile",
            "source_date": "2027-12-30", "recorded_at": "2027-12-31T23:00:00Z",
        }))
        (acct / "saver_state.json").write_text(json.dumps({
            "season": 2027, "state": "active", "updated_at": "2027-12-31T23:00:00Z", "source": "cli",
        }))
        picks.save_streak(12, picks_dir, saver_available=True, applied_date="2027-12-30", season=2027)

        ds = contest_state.load_decision_streak_state(picks_dir, require_contest_state=True, now=t_dec31_et)
        assert ds.streak == 12
        assert ds.saver_available is True
        assert ds.best_streak == 20 and ds.best_status == "trusted", (ds.best_streak, ds.best_status)
        assert ds.model_streak == 12

        hc_alerts = health_contest_state.check(picks_dir, expected=True, now=t_dec31_et)
        assert [a for a in hc_alerts if a.level == "CRITICAL"] == []

        # Dated observation at 2027-12-31 20:30 ET (UTC 2028-01-01)
        cli_mod._write_dated_contest_observation(
            picks_dir, effective_date=dt.date(2027, 12, 31),
            streak=12, best_streak=20, saver="active", reason="nye", now=t_dec31_et,
        )
        ds_dated = contest_state.load_decision_streak_state(picks_dir, require_contest_state=True, now=t_dec31_et)
        assert ds_dated.observation_kind == "dated" and ds_dated.best_status == "trusted"

        return {
            "utc_2028_et_2027_dec31_contract_and_saver_accepted": True,
            "best_streak_trusted_across_utc_new_year": ds.best_status == "trusted",
        }


def test_year_2028_nothing_reconfigured(tmp_path: Path | None = None) -> dict:
    """Verify 2028 behavior when 2027 files & SEASON_END_DATES are left untouched."""
    with _temp_dir(tmp_path) as td:
        base = Path(td)
        picks_dir = base / "data" / "picks"
        acct = picks_dir / "account_state"
        acct.mkdir(parents=True)

        # Leave end-of-2027 automatic contract, contest_streak, saver_state, and streak.json untouched
        (acct / "season_contract.json").write_text(json.dumps({
            "schema_version": "bts_season_contract_v1", "season": 2027, "status": "automatic",
            "activation_id": "d" * 32, "switched_at": "2027-04-10T12:00:00Z", "revoked_at": None, "note": None,
        }))
        (acct / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": "d" * 32,
            "active_streak": 14, "best_streak": 25, "source": "mlb_bts_profile",
            "source_date": "2027-09-26", "recorded_at": "2027-09-27T06:10:00Z",
        }))
        (acct / "saver_state.json").write_text(json.dumps({
            "season": 2027, "state": "used", "updated_at": "2027-08-01T00:00:00Z", "source": "cli",
        }))
        picks.save_streak(14, picks_dir, saver_available=False, applied_date="2027-09-26", season=2027)

        t_2028 = dt.datetime(2028, 3, 24, 12, 0, tzinfo=ET)

        # 1. Live contest-state read fails closed (2027 switch != 2028)
        refusal_err = None
        try:
            contest_state.load_decision_streak_state(picks_dir, require_contest_state=True, now=t_2028)
        except contest_state.ContestStateError as exc:
            refusal_err = str(exc)
        assert refusal_err is not None and "2027" in refusal_err and "2028" in refusal_err

        # 2. Preview / non-required read does NOT carry 2027 streak (14) or spent saver into 2028!
        # WAIT: CHECK WHAT `load_decision_streak_state(picks_dir, require_contest_state=False, now=t_2028)` RETURNS!!
        ds_preview_2028 = contest_state.load_decision_streak_state(picks_dir, require_contest_state=False, now=t_2028)

        # 3. Saver state for 2028 is uninitialized; model streak for 2028 is 0, saver True
        assert saver_state.load_saver_state(picks_dir, season=2028).state == "uninitialized"
        assert picks.load_streak(picks_dir, season=2028) == 0
        assert picks.load_saver_available(picks_dir, season=2028) is True
        assert picks.season_of_date("2028-02-29") == 2028  # leap day

        # 4. Calendar fallback & tail_policy health WARN + attention promotion in 2028
        assert strategy.season_end_date("2028-03-24") == "2028-09-28"
        assert strategy._days_remaining("2028-02-29") == (dt.date(2028, 9, 28) - dt.date(2028, 2, 29)).days
        tp_alerts = health_tail_policy.check(
            base_path=REPO_ROOT / "data/models/mdp_policy.npz",
            tail_path=REPO_ROOT / "data/models/mdp_tail_policy.npz",
            today=dt.date(2028, 3, 24),
        )
        assert len(tp_alerts) == 1 and tp_alerts[0].level == "WARN" and "2028 is not configured" in tp_alerts[0].message
        _, warn_attn = health_attention.apply_warn_attention_policy(
            tp_alerts, state_path=base / "warn_state.json", today=dt.date(2028, 3, 24),
        )
        assert len(warn_attn) == 1 and warn_attn[0].source == "tail_policy"

        return {
            "live_2028_refused_with_2027_switch": True,
            "preview_2028_with_2027_auto_v2_file": {
                "streak": ds_preview_2028.streak,
                "source": ds_preview_2028.source,
                "saver_available": ds_preview_2028.saver_available,
                "best_streak": ds_preview_2028.best_streak,
                "best_status": ds_preview_2028.best_status,
            },
            "tail_policy_2028_warns_and_pages_attention": True,
        }


def run_all() -> dict:
    return {
        "2027_03_24": test_opening_night_2027_03_24(),
        "midnight_et": test_midnight_et_on_game_day(),
        "2027_12_31": test_dec_31_2027_and_utc_new_year(),
        "2028_unreconfigured": test_year_2028_nothing_reconfigured(),
    }


if __name__ == "__main__":
    import shutil
    try:
        print(json.dumps(run_all(), indent=2, default=str))
    finally:
        shutil.rmtree(SCRATCH_TMP, ignore_errors=True)
