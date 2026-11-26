"""Adversarial probes for b205ae0 postponed delivery guard + health checks + day-outcome."""
from __future__ import annotations

import tempfile
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from bts.day_outcome import gather
from bts.health.late_delivery import check as late_delivery_check
from bts.health.runner import run_all_checks
from bts.picks import DailyPick, Pick, load_pick
from bts.scheduler import SchedulerState, _deliver_and_lock_pick, load_state

ET = ZoneInfo("America/New_York")


def _make_daily(d: str, pk: int = 8001, batter_id: int = 500, name: str = "Aaron Judge") -> DailyPick:
    return DailyPick(
        date=d,
        run_time="18:00 ET",
        pick=Pick(
            batter_name=name, batter_id=batter_id, team="NYY",
            lineup_position=2, pitcher_name="Opposing Pitcher", pitcher_id=999,
            p_game_hit=0.80, flags=[], projected_lineup=False,
            game_pk=pk, game_time=f"{d}T23:10:00Z",
        ),
        double_down=None,
        runner_up=None,
        generated_at=None if False else None,  # unused field check skipped
    ) if False else DailyPick(
        date=d,
        run_time="18:00 ET",
        pick=Pick(
            batter_name=name, batter_id=batter_id, team="NYY",
            lineup_position=2, pitcher_name="Opposing Pitcher", pitcher_id=999,
            p_game_hit=0.80, flags=[], projected_lineup=False,
            game_pk=pk, game_time=f"{d}T23:10:00Z",
        ),
        double_down=None,
        runner_up=None,
    )


def probe_void_refusal_triggers_false_late_delivery_critical():
    """Run _deliver_and_lock_pick on a postponed game, then run late_delivery.check & run_all_checks."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        picks_dir = root / "picks"
        models_dir = root / "models"
        picks_dir.mkdir()
        models_dir.mkdir()
        d = "2027-05-10"
        daily = _make_daily(d, pk=8001, name="Aaron Judge")
        state = SchedulerState(
            date=d,
            schedule_fetched_at=f"{d}T10:00:00-04:00",
            games=[{"game_pk": 8001, "game_time_et": f"{d}T19:10:00-04:00",
                    "lineup_confirmed": True, "is_doubleheader_game2": False}],
            confirmed_game_pks=[],
            runs_completed=[],
            pick_locked=False,
            pick_locked_at=None,
            result_status=None,
            next_wakeup=None,
        )
        config = {
            "orchestrator": {"picks_dir": str(picks_dir), "models_dir": str(models_dir)},
            "scheduler": {"pick_delivery": "dm"},
            "bluesky": {"dm_recipient": "op.bsky.social"},
        }
        sent_dms: list[tuple[str, str]] = []

        def fake_send_dm(recipient, text):
            sent_dms.append((recipient, text))
            return "dm-1"

        with patch("bts.scheduler._now_et", return_value=datetime(2027, 5, 10, 18, 45, tzinfo=ET)), \
             patch("bts.scheduler._delivery_game_statuses",
                   return_value={8001: {"abstract": "P", "detailed": "Postponed", "code": "PPD"}}), \
             patch("bts.health.alert.send_dm", side_effect=fake_send_dm):
            ok = _deliver_and_lock_pick(daily, config, picks_dir, state, d, "fallback")

        loaded = load_state(d, picks_dir)
        alerts = late_delivery_check(picks_dir, today=date(2027, 5, 10))
        with patch("bts.health.alert.send_dm", side_effect=fake_send_dm):
            all_alerts = run_all_checks(
                picks_dir=picks_dir,
                models_dir=models_dir,
                today=date(2027, 5, 10),
                dm_recipient="op.bsky.social",
                live_forward_capture_enabled=False,
                shadow_model_enabled=False,
                contest_state_expected=False,
            )
        rel_alerts = [(a.level, a.source, a.message) for a in all_alerts
                      if a.source in ("late_delivery", "postponed_pick", "pick_delivery")]

        print("[PROBE 2A void-refusal vs health.late_delivery]")
        print(f"  _deliver_and_lock_pick returned={ok}, pick_on_disk={load_pick(d, picks_dir)}")
        print(f"  state.delivery_refusals={loaded.delivery_refusals}")
        print(f"  late_delivery.check alerts ({len(alerts)}):")
        for a in alerts:
            print(f"    - [{a.level}/{a.source}] {a.message}")
        print(f"  immediate DMs dispatched ({len(sent_dms)}):")
        for _, m in sent_dms:
            print(f"    - {m}")
        print(f"  run_all_checks delivery/refusal alerts ({len(rel_alerts)}):")
        for item in rel_alerts:
            print(f"    - {item}")


def probe_postponed_only_day_triggers_day_outcome_no_decision():
    """On a day whose only game(s) are Postponed (e.g. L03 refused it), what does day_outcome.gather return?"""
    with tempfile.TemporaryDirectory() as tmp:
        picks_dir = Path(tmp) / "picks"
        picks_dir.mkdir()
        d = "2027-05-10"
        sched_games = [{
            "gamePk": 8001,
            "gameType": "R",
            "officialDate": d,
            "gameDate": f"{d}T23:10:00Z",
            "status": {"abstractGameCode": "F", "detailedState": "Postponed", "statusCode": "PPD"},
        }]
        config = {"scheduler": {"entry_intent": "enter", "pick_delivery": "dm"}}
        with patch("bts.scheduler.fetch_schedule", return_value=sched_games):
            outcome = gather(d, picks_dir, config, expected_username="stonehengee")
        print("[PROBE 2B postponed-only day in day_outcome.gather]")
        print(f"  state={outcome.state}, ok={outcome.ok}, detail={outcome.detail}")
        print(f"  message={outcome.message()}")


if __name__ == "__main__":
    probe_void_refusal_triggers_false_late_delivery_critical()
    probe_postponed_only_day_triggers_day_outcome_no_decision()
