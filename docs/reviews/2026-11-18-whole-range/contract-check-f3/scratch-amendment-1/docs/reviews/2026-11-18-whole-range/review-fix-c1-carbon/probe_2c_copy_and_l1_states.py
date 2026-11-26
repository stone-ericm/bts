"""Item 2 probe (v3): render every historical branch, verify exact H3/H5/L1 rules under hostile inputs, and exercise L1 worst states."""

from __future__ import annotations

import datetime as _dt_mod
from datetime import date, timedelta
import io
import json
import logging
from pathlib import Path
import signal
import socket
from tempfile import TemporaryDirectory
from unittest.mock import patch

from probe_2b_copy_and_l1_states import (
    D,
    DEC31,
    EXPECTED_POINTER,
    HERE,
    MATCH_ID,
    at_et,
    deny_network,
    run_historical,
    verify_all_h3_branches_and_hostile_inputs,
)


def verify_l1_worst_states_v2(root: Path) -> None:
    from click.testing import CliRunner
    from bts.cli import cli
    from bts.contest_state import ContestStateError, load_decision_streak_state
    from bts import picks as picks_mod, scheduler
    from bts.health import alert

    # 1. Row: `Prior slate <D> ET.` — D = Dec 31, alert arrives Jan 1 00:30 ET
    p_ny = root / "l1_ny" / "picks"
    (p_ny / "account_state").mkdir(parents=True)
    (p_ny / "account_state/season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1", "season": 2027, "activation_id": MATCH_ID,
        "status": "revoked", "switched_at": at_et(DEC31, 8).isoformat(), "revoked_at": at_et(DEC31, 9).isoformat(),
    }))
    alerts_ny, bodies_ny = run_historical(p_ny, DEC31, at_et(date(2028, 1, 1), 0, 30))
    assert all(a.message.startswith("Prior slate 2027-12-31 ET. ") for a in alerts_ny)
    assert "2028" not in bodies_ny[0]

    # 2. Row: `This check used files available at run time.` — 01:10 fetch replaced auto before 02:00 health run
    p_fetch = root / "l1_fetch" / "picks"
    (p_fetch / "account_state").mkdir(parents=True)
    (p_fetch / "account_state/season_contract.json").write_text(json.dumps({
        "schema_version": "bts_season_contract_v1", "season": 2027, "activation_id": MATCH_ID,
        "status": "automatic", "switched_at": at_et(D, 8).isoformat(), "revoked_at": None,
    }))
    (p_fetch / "account_state/contest_streak.json").write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2", "season": 2027, "activation_id": "d" * 32,
        "active_streak": 3, "best_streak": 3, "source_date": D.isoformat(),
        "recorded_at": at_et(D + timedelta(days=1), 1, 10).isoformat(),
    }))
    (p_fetch / "account_state/saver_state.json").write_text(json.dumps({"season": 2027, "state": "not_earned"}))
    alerts_f, _ = run_historical(p_fetch, D, at_et(D + timedelta(days=1), 2))
    assert "This check used files available at run time." in alerts_f[0].message

    # 3. Rows: D+1 genuinely has no usable state; morning check skipped in `run_day` (<10m lead) vs active;
    # live refusal at lineup check; `season-contract status` at 00:30 prints D+1 ways out; cutoff passed.
    p_d1 = root / "l1_d1" / "picks"
    (p_d1 / "account_state").mkdir(parents=True)
    next_day = D + timedelta(days=1)

    # 3a. Skipped morning check via real `scheduler.run_day` when first lineup check is 5 min away (10:00 wake, 10:05 check):
    stderr_buf = io.StringIO()
    game_start_utc = at_et(next_day, 10, 50).astimezone(_dt_mod.timezone.utc).isoformat().replace("+00:00", "Z")
    with (
        patch.object(scheduler, "fetch_schedule", side_effect=[
            [{"gamePk": 1, "gameType": "R", "gameDate": game_start_utc,
              "teams": {"away": {"team": {"name": "A"}}, "home": {"team": {"name": "B"}}}}],
            [],
        ]),
        patch.object(scheduler, "_now_et", return_value=at_et(next_day, 10, 0)),
        patch.object(scheduler, "run_single_check", side_effect=KeyboardInterrupt),
        patch.object(alert, "send_dm", return_value="ack") as send_mock,
        patch("sys.stderr", stderr_buf),
    ):
        try:
            scheduler.run_day(
                date=next_day.isoformat(),
                config={
                    "orchestrator": {"picks_dir": str(p_d1)},
                    "bluesky": {"dm_recipient": "op"},
                    "scheduler": {
                        "pick_delivery": "dm", "entry_intent": "enter",
                        "contest_state_required": True, "lineup_check_offset_min": 45,
                    },
                    "health_checks": {"enabled": False, "contest_state_expected": True},
                    "tiers": [],
                },
            )
        except KeyboardInterrupt:
            pass
    send_mock.assert_not_called()
    assert "Morning state check skipped: first lineup check is due within 10 minutes." in stderr_buf.getvalue()

    # 3b. Live refusal at lineup check still refuses & surfaces same-day ways out:
    try:
        load_decision_streak_state(p_d1, require_contest_state=True, now=at_et(next_day, 10, 5))
    except ContestStateError as exc:
        assert f"bts set-contest-streak --date {next_day.isoformat()}" in str(exc)
    else:
        raise AssertionError("Expected live refusal on D+1")

    # 3c. Morning state check when available (>=10m lead) sends D+1 morning DM when state missing, silent when accepted:
    with patch.object(alert, "send_dm", return_value="ack") as send_mock2:
        scheduler._morning_state_check(
            {"bluesky": {"dm_recipient": "op"}}, p_d1, next_day.isoformat(),
            at_et(next_day, 10, 0), at_et(next_day, 11, 0),
        )
        send_mock2.assert_called_once()
        assert f"Morning check for {next_day.isoformat()}" in send_mock2.call_args.args[1]

    # 3d. Status at 00:30 ET on D+1 shows current files AND prints live refusal's ways out (the documented residual):
    real_dt = _dt_mod.datetime

    class CliClock(real_dt):
        @classmethod
        def now(cls, tz=None):
            t = at_et(next_day, 0, 30)
            return t.astimezone(tz) if tz is not None else t.replace(tzinfo=None)

    with patch.object(_dt_mod, "datetime", CliClock):
        res = CliRunner().invoke(cli, ["season-contract", "status", "--picks-dir", str(p_d1)])
    assert res.exit_code == 0, res.output
    assert "live decision now: refused;" in res.output
    assert f"bts set-contest-streak --date {next_day.isoformat()}" in res.output

    # 3e. Past cutoff on D+1: historical alert arriving after submission_cutoff_et leaves pick file untouched
    game = at_et(next_day, 14, 0)
    pick = picks_mod.Pick(
        batter_name="Test", batter_id=1, team="CLE", lineup_position=1,
        pitcher_name="Pitcher", pitcher_id=2, p_game_hit=0.65, flags=[],
        projected_lineup=False, game_pk=99, game_time=game.isoformat(),
    )
    daily = picks_mod.DailyPick(
        date=next_day.isoformat(), run_time=(game - timedelta(hours=1)).isoformat(),
        pick=pick, double_down=None, runner_up=None, delivered_at=(game - timedelta(hours=1)).isoformat(),
    )
    saved = picks_mod.save_pick(daily, p_d1)
    saved_bytes = saved.read_bytes()
    late_wall = at_et(next_day, 15, 0)
    assert late_wall > picks_mod.submission_cutoff_et(daily)
    alerts_late, _ = run_historical(p_d1, D, late_wall)
    assert alerts_late and saved.read_bytes() == saved_bytes


def main() -> None:
    signal.alarm(50)
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(dir=HERE) as tmp:
        with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
            count = verify_all_h3_branches_and_hostile_inputs(Path(tmp))
            verify_l1_worst_states_v2(Path(tmp))
            print(f"[PASS] Item 2: {count} historical branches + H4 streak/mixed DM + hostile injections + L1 worst states verified")
    signal.alarm(0)


if __name__ == "__main__":
    main()
