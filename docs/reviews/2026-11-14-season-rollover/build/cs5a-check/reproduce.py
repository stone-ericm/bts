"""Offline reproduction of Carbon N2 on main."""

import sys

sys.dont_write_bytecode = True
network_attempts = []


def forbid_network(event, args):
    if event.startswith("socket.") and event != "socket.__new__":
        network_attempts.append(event)
        raise AssertionError(f"network forbidden: {event}")


sys.addaudithook(forbid_network)

from datetime import datetime
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pandas as pd

from bts import cli, contest_state, daily_decision, dm, orchestrator, picks, posting, scheduler
from bts.health import alert
from bts.model import predict


ET = ZoneInfo("America/New_York")
DAY = "2027-07-15"
NOW = datetime(2027, 7, 15, 18, 35, tzinfo=ET)
STATUS = {pk: {"abstract": "P", "detailed": "Scheduled", "code": "S"} for pk in (101, 102)}
PREDICTIONS = pd.DataFrame([
    {"batter_id": 680757, "batter_name": "Steven Kwan", "team": "CLE", "lineup": 1,
     "pitcher_id": 543037, "pitcher_name": "Gerrit Cole", "opp": "NYY", "p_hit_pa": 0.24,
     "p_game_hit": 0.82, "flags": "", "projected_lineup": False, "game_pk": 101,
     "game_time": f"{DAY}T23:10:00Z"},
    {"batter_id": 665742, "batter_name": "Juan Soto", "team": "NYM", "lineup": 2,
     "pitcher_id": 605483, "pitcher_name": "Blake Snell", "opp": "LAD", "p_hit_pa": 0.23,
     "p_game_hit": 0.81, "flags": "", "projected_lineup": False, "game_pk": 102,
     "game_time": f"{DAY}T23:40:00Z"},
])


class PinnedDateTime(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


def main(root):
    picks_dir = root / "picks"
    account = picks_dir / "account_state"
    account.mkdir(parents=True)
    config = {
        "orchestrator": {"picks_dir": str(picks_dir), "heartbeat_path": str(root / "heartbeat")},
        "bluesky": {"dm_recipient": "local.invalid"},
        "scheduler": {"contest_state_required": True, "pick_delivery": "dm",
                      "live_forward_capture_on_lock": False},
        "tiers": [],
    }
    auto = {
        "schema_version": "bts_contest_streak_auto_v2", "season": 2026,
        "activation_id": "a" * 32, "active_streak": 0, "best_streak": 0,
        "source": "mlb_bts_profile", "source_date": "2026-09-27",
        "recorded_at": "2026-09-28T10:00:00Z",
    }
    (account / "contest_streak.json").write_text(json.dumps(auto))
    state = scheduler.SchedulerState(
        date=DAY, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
        runs_completed=[], pick_locked=False, pick_locked_at=None, result_status=None,
        next_wakeup=None,
    )
    with (
        patch.object(contest_state, "datetime", PinnedDateTime),
        patch.object(scheduler, "_now_et", return_value=NOW),
        patch.object(predict, "run_pipeline", return_value=PREDICTIONS) as prediction,
        patch.object(orchestrator, "run_cascade", return_value=(None, None)) as refresh_prediction,
        patch.object(picks, "get_game_statuses_detailed", return_value=STATUS),
        patch.object(picks, "get_game_statuses", return_value=dict.fromkeys(STATUS, "P")),
        patch.object(scheduler, "fetch_schedule", return_value=[]) as schedule,
        patch.object(dm, "send_dm", return_value="captured-pick-dm") as dm_capture,
        patch.object(alert, "send_dm", return_value="captured-alert") as alert_capture,
        patch.object(posting, "post_to_bluesky", return_value="captured-public-post") as post_capture,
    ):
        optional = contest_state.load_decision_streak_state(picks_dir, require_contest_state=False)
        assert optional.streak == 0 and optional.allow_double, optional
        try:
            contest_state.load_decision_streak_state(picks_dir, require_contest_state=True)
        except contest_state.ContestStateError:
            pass
        else:
            raise AssertionError("last season's automatic read was accepted live")
        result = CliRunner().invoke(cli.cli, [
            "preview", "--date", DAY, "--picks-dir", str(picks_dir), "--models-dir", str(root / "models"),
        ], catch_exceptions=False)
        assert result.exit_code == 0, result.output
        cached = picks.load_pick(DAY, picks_dir)
        assert cached and cached.double_down, result.output

        dated = {
            "schema_version": "bts_contest_streak_manual_v3", "effective_date": DAY,
            "season": 2027, "expires_at": f"{DAY}T23:59:59.999999-04:00",
            "active_streak": 56, "best_streak": 56, "saver_state": "used",
            "recorded_at": NOW.isoformat(), "source": "manual_cli",
        }
        (account / "contest_streak.manual.json").write_text(json.dumps(dated))
        current = contest_state.load_decision_streak_state(picks_dir, require_contest_state=True)
        assert current.streak == 56 and not current.saver_available
        refresh = scheduler._refresh_pick_at_fallback_decision(config, DAY, cached, 0.03)
        assert refresh.daily is cached and refresh.selection is None
        delivered = scheduler._deliver_and_lock_pick(
            refresh.daily, config, picks_dir, state, DAY, "final fallback", selection=refresh.selection,
        )
        delivery = picks.load_pick(DAY, picks_dir)
        decision = daily_decision.load_decision(DAY, picks_dir)
        assert delivered and state.pick_locked and delivery.delivery_attempted
        assert delivery.notification_sent and decision["delivery_status"] == "delivered"
        assert dm_capture.call_count == prediction.call_count == refresh_prediction.call_count == 1
        assert not alert_capture.called and not post_capture.called and not network_attempts
        print(json.dumps({
            "preview_streak": optional.streak, "live_streak": current.streak,
            "dm": dm_capture.call_args.args[1], "locked": state.pick_locked,
            "decision": decision["delivery_status"], "schedule_calls": schedule.call_count,
            "denied_network": network_attempts,
        }, indent=2))


if __name__ == "__main__":
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as directory:
        main(Path(directory))
