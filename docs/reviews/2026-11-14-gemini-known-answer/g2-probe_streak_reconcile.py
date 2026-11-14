"""Adversarial probes for L02 + L04 streak & reconcile interactions."""
from __future__ import annotations

import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from bts.cli import cli
from bts.daily_decision import write_decision
from bts.picks import (
    APPLIED_DATES_KEPT,
    DailyPick,
    Pick,
    load_applied_dates,
    load_pick,
    load_saver_available,
    load_streak,
    reconcile_results,
    save_pick,
    save_streak,
    update_streak,
)

ET = ZoneInfo("America/New_York")


def _make_daily(date: str, batter_id: int = 100, game_pk: int = 7001, result: str | None = None,
                dd_id: int | None = None, dd_pk: int | None = None) -> DailyPick:
    pick = Pick(
        batter_name=f"Batter {batter_id}", batter_id=batter_id, team="NYY",
        lineup_position=1, pitcher_name="Opponent Pitcher", pitcher_id=999,
        p_game_hit=0.78, flags=[], projected_lineup=False,
        game_pk=game_pk, game_time=f"{date}T23:10:00Z",
    )
    dd = None
    if dd_id is not None:
        dd = Pick(
            batter_name=f"Batter {dd_id}", batter_id=dd_id, team="LAD",
            lineup_position=2, pitcher_name="Opponent Pitcher 2", pitcher_id=998,
            p_game_hit=0.77, flags=[], projected_lineup=False,
            game_pk=dd_pk or (game_pk + 1), game_time=f"{date}T23:10:00Z",
        )
    return DailyPick(
        date=date, run_time="18:00 ET", pick=pick, double_down=dd, runner_up=None,
        delivered_at=f"{date}T18:05:00-04:00",
        notification_sent=True, notification_channel="bluesky_dm",
        result=result,
    )


def _feed(game_pk: int, batter_id: int, hits: int, ab: int = 4, sf: int = 0) -> dict:
    return {
        "gameData": {
            "game": {"pk": game_pk},
            "status": {"abstractGameCode": "F", "detailedState": "Final"},
            "datetime": {},
        },
        "liveData": {
            "boxscore": {
                "teams": {
                    "away": {"players": {f"ID{batter_id}": {
                        "person": {"id": batter_id, "fullName": f"Batter {batter_id}"},
                        "stats": {"batting": {"hits": hits, "atBats": ab, "sacFlies": sf}},
                    }}},
                    "home": {"players": {}},
                }
            },
            "plays": {"allPlays": []},
        },
    }


def probe_crash_then_same_day_reconcile_hit():
    """Crash between update_streak and save_pick, then same-day reconcile, then cron check-results."""
    with tempfile.TemporaryDirectory() as tmp:
        picks = Path(tmp) / "picks"
        picks.mkdir()
        d1 = _make_daily("2027-05-01", batter_id=101, game_pk=7001, result="hit")
        save_pick(d1, picks)
        save_streak(1, picks, saver_available=True, applied_date="2027-05-01")

        # Day 2 locked pick, scored by daemon BEFORE crash right after update_streak:
        d2 = _make_daily("2027-05-02", batter_id=102, game_pk=7002, result=None)
        save_pick(d2, picks)
        write_decision("2027-05-02", picks, action="single", source="mdp",
                       delivery_status="delivered", scoreable=True)
        # Step 1: update_streak succeeds -> streak=2, applied_dates=['2027-05-01', '2027-05-02']
        s_after_update = update_streak([True], picks, date="2027-05-02")
        # Daemon crashes BEFORE save_pick(d2 with result="hit")!

        # Step 2: Same-day reconcile runs at 23:30 ET on 2027-05-02 (or with lookback_days=2)
        now_reconcile = datetime(2027, 5, 2, 23, 30, tzinfo=ET)
        with patch("bts.picks._fetch_json", return_value=_feed(7001, 101, 1)), \
             patch("bts.picks.get_game_statuses_detailed", return_value={7001: {"abstract": "F", "detailed": "Final"}}):
            reconcile_results(picks, lookback_days=2, clock=lambda: now_reconcile)
        s_after_reconcile = load_streak(picks)
        applied_after_reconcile = load_applied_dates(picks)

        # Step 3: Cron check-results runs at 01:00 ET on 2027-05-03 for --date 2027-05-02
        with patch("bts.picks._fetch_json", return_value=_feed(7002, 102, 1)), \
             patch("bts.picks.get_game_statuses_detailed", return_value={7002: {"abstract": "F", "detailed": "Final"}}), \
             patch("bts.cli._now_et", return_value=datetime(2027, 5, 3, 1, 0, tzinfo=ET)), \
             patch("bts.skip_policy_shadow.reconcile_pending", return_value=0):
            res = CliRunner().invoke(cli, ["check-results", "--date", "2027-05-02", "--picks-dir", str(picks)])
        s_after_cron = load_streak(picks)
        d2_final = load_pick("2027-05-02", picks)
        print("[PROBE 1A crash+same-day-reconcile+check-results (HIT)]")
        print(f"  after update_streak: streak={s_after_update}")
        print(f"  after same-day reconcile: streak={s_after_reconcile}, applied_dates={applied_after_reconcile}")
        print(f"  after cron check-results: exit={res.exit_code}, daily.result={d2_final.result}, streak={s_after_cron}")


def probe_crash_then_same_day_reconcile_saver_miss():
    """Same scenario at streak=10 with saver=True and a MISS."""
    with tempfile.TemporaryDirectory() as tmp:
        picks = Path(tmp) / "picks"
        picks.mkdir()
        for day in range(1, 11):
            ds = f"2027-05-{day:02d}"
            save_pick(_make_daily(ds, batter_id=100 + day, game_pk=7000 + day, result="hit"), picks)
        save_streak(10, picks, saver_available=True, applied_date="2027-05-10")

        d11 = _make_daily("2027-05-11", batter_id=111, game_pk=7011, result=None)
        save_pick(d11, picks)
        write_decision("2027-05-11", picks, action="single", source="mdp",
                       delivery_status="delivered", scoreable=True)
        update_streak([False], picks, date="2027-05-11")
        saver_after_update = load_saver_available(picks)

        now_reconcile = datetime(2027, 5, 11, 23, 30, tzinfo=ET)
        with patch("bts.picks._fetch_json", return_value=_feed(7010, 110, 1)), \
             patch("bts.picks.get_game_statuses_detailed", return_value={7010: {"abstract": "F", "detailed": "Final"}}):
            reconcile_results(picks, lookback_days=1, clock=lambda: now_reconcile)
        saver_after_reconcile = load_saver_available(picks)

        with patch("bts.picks._fetch_json", return_value=_feed(7011, 111, 0, ab=4)), \
             patch("bts.picks.get_game_statuses_detailed", return_value={7011: {"abstract": "F", "detailed": "Final"}}), \
             patch("bts.cli._now_et", return_value=datetime(2027, 5, 12, 1, 0, tzinfo=ET)), \
             patch("bts.skip_policy_shadow.reconcile_pending", return_value=0):
            CliRunner().invoke(cli, ["check-results", "--date", "2027-05-11", "--picks-dir", str(picks)])
        saver_after_cron = load_saver_available(picks)
        d11_final = load_pick("2027-05-11", picks)
        print("[PROBE 1B crash+same-day-reconcile+check-results (SAVER MISS)]")
        print(f"  saver after update_streak={saver_after_update}, after reconcile={saver_after_reconcile}, after cron={saver_after_cron}, d11.result={d11_final.result}")


def probe_after_midnight_with_earlier_unresolved_and_backfill():
    """What if yesterday had an unresolved suspended pick (replay=None), AND same-day crash occurred,
    OR --allow-stale-scoring backfills after 30 days?"""
    with tempfile.TemporaryDirectory() as tmp:
        picks = Path(tmp) / "picks"
        picks.mkdir()
        save_pick(_make_daily("2027-05-01", batter_id=101, game_pk=7001, result="hit"), picks)
        save_streak(1, picks, saver_available=True, applied_date="2027-05-01")
        for i in range(1, 31):
            d = (datetime(2027, 5, 1) + timedelta(days=i)).strftime("%Y-%m-%d")
            update_streak([True], picks, date=d)
        kept = load_applied_dates(picks)
        print("[PROBE 1C 30-date cap in applied_dates]")
        print(f"  len(applied_dates)={len(kept)} (cap={APPLIED_DATES_KEPT}), first={kept[0]}, last={kept[-1]}, '2027-05-01' still present? {'2027-05-01' in kept}")

    with tempfile.TemporaryDirectory() as tmp:
        picks = Path(tmp) / "picks"
        picks.mkdir()
        # Scenario: 2027-05-01 had hit (streak=1), 2027-05-02 had crash between update_streak (streak=2,
        # applied=['2027-05-01','2027-05-02']) and save_pick, same-day reconcile rolled streak back to 1
        # while keeping '2027-05-02' in applied_dates, AND 2027-05-01 was actually unresolved? Wait: even
        # simpler, on 2027-05-03 at 01:00 cron check-results runs AND picks/decision use load_streak BEFORE 02:00 reconcile.
        # What if 2027-05-02's 01:00 cron failed because game wasn't final, OR 2027-05-03 game was unresolved so
        # _replay_season_streak returned None forever?
        save_pick(_make_daily("2027-04-30", batter_id=100, game_pk=7000, result="hit"), picks)
        save_pick(_make_daily("2027-05-01", batter_id=101, game_pk=7001, result="hit"), picks)
        save_pick(_make_daily("2027-05-02", batter_id=102, game_pk=7002, result=None), picks)
        write_decision("2027-05-02", picks, action="single", source="mdp",
                       delivery_status="delivered", scoreable=True)
        save_streak(2, picks, saver_available=True, applied_date="2027-05-01")
        update_streak([True], picks, date="2027-05-02")  # streak=3, applied_dates has 05-02
        # Suppose same-day reconcile at 23:30 on 05-02 rolls streak back to 2 while leaving 05-02 in applied_dates!
        with patch("bts.picks._fetch_json", return_value=_feed(7001, 101, 1)), \
             patch("bts.picks.get_game_statuses_detailed", return_value={7001: {"abstract": "F", "detailed": "Final"}}):
            reconcile_results(picks, lookback_days=1, clock=lambda: datetime(2027, 5, 2, 23, 30, tzinfo=ET))
        # Now cron scores 05-02 as hit -> remains streak=2!
        with patch("bts.picks._fetch_json", return_value=_feed(7002, 102, 1)), \
             patch("bts.picks.get_game_statuses_detailed", return_value={7002: {"abstract": "F", "detailed": "Final"}}), \
             patch("bts.cli._now_et", return_value=datetime(2027, 5, 3, 1, 0, tzinfo=ET)), \
             patch("bts.skip_policy_shadow.reconcile_pending", return_value=0):
            CliRunner().invoke(cli, ["check-results", "--date", "2027-05-02", "--picks-dir", str(picks)])
        # Now an earlier pick (05-03 suspended) or manual correction/incomplete file makes replay return None:
        save_pick(_make_daily("2027-05-03", batter_id=103, game_pk=7003, result="suspended"), picks)
        with patch("bts.picks._fetch_json", return_value=_feed(7002, 102, 1)), \
             patch("bts.picks.get_game_statuses_detailed", return_value={7002: {"abstract": "F", "detailed": "Final"}}):
            reconcile_results(picks, lookback_days=1, clock=lambda: datetime(2027, 5, 4, 2, 0, tzinfo=ET))
        print("[PROBE 1D stranded low streak when subsequent pick is suspended]")
        print(f"  expected_streak=3, actual_streak={load_streak(picks)}")


def probe_void_correction_to_hit_in_reconcile():
    """What if a double-down was void+void, then DD corrected void->hit in reconcile?"""
    with tempfile.TemporaryDirectory() as tmp:
        picks = Path(tmp) / "picks"
        picks.mkdir()
        save_pick(_make_daily("2027-05-01", batter_id=101, game_pk=7001, result="hit"), picks)
        save_pick(_make_daily("2027-05-02", batter_id=102, game_pk=7002, result="void",
                              dd_id=103, dd_pk=7003), picks)
        save_streak(1, picks, saver_available=True, applied_date="2027-05-01")
        now_reconcile = datetime(2027, 5, 3, 2, 0, tzinfo=ET)

        def fake_fetch(url, timeout=15):
            if "7001" in url:
                return _feed(7001, 101, 1)
            if "7002" in url:
                return _feed(7002, 102, 0, ab=0, sf=0)
            return _feed(7003, 103, 1)

        with patch("bts.picks._fetch_json", side_effect=fake_fetch), \
             patch("bts.picks.get_game_statuses_detailed", return_value={
                 7001: {"abstract": "F", "detailed": "Final"},
                 7002: {"abstract": "F", "detailed": "Final"},
                 7003: {"abstract": "F", "detailed": "Final"},
             }):
            corr = reconcile_results(picks, lookback_days=2, clock=lambda: now_reconcile)
        d2 = load_pick("2027-05-02", picks)
        print("[PROBE 1E double-down partial void -> hit correction in reconcile]")
        print(f"  corrections={corr}, streak={load_streak(picks)}, d2.result={d2.result}, d2.slot_results={d2.slot_results}")


if __name__ == "__main__":
    probe_crash_then_same_day_reconcile_hit()
    probe_crash_then_same_day_reconcile_saver_miss()
    probe_after_midnight_with_earlier_unresolved_and_backfill()
    probe_void_correction_to_hit_in_reconcile()
