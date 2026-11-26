"""Check literal advice after B, including the Carbon scheduler reproduction."""

from contextlib import redirect_stderr
from datetime import datetime, timedelta
import io
import json
import logging
from pathlib import Path
import signal
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

from probe import D, HERE, POINTER, ROOT, SLICES, YEAR_END, at, auto, dated, end, put, real_runner, saver_flag, settled, switch


def full_scheduler(root):
    sys.path.insert(0, str(SLICES / "slice-1-carbon"))
    from test_slice1_probes_v3 import RehearsalWorld, _H, _setup_revoked_box, et_dt
    from bts.contest_state import load_decision_streak_state

    _H.REPO_ROOT = ROOT
    schedules = _H.default_opening_days_schedules()
    game = schedules[D.isoformat()][0]
    game.start_et = et_dt(2027, 3, 24, 21, 5)
    game.away_confirm_et = et_dt(2027, 3, 24, 18, 45)
    game.home_confirm_et = et_dt(2027, 3, 24, 19, 45)
    game.final_et = et_dt(2027, 3, 25, 0, 15)
    game.away_hits = 0

    with RehearsalWorld(root, initial_et=at(D, 9), schedules=schedules) as world:
        _setup_revoked_box(world)
        path_args = ["--picks-dir", str(world.picks_dir)]

        def cli(*args):
            result = world.invoke_cli([*args, *path_args])
            assert result.exit_code == 0, (args, result.output)
            return result.stdout

        cli("saver-state", "--init", "not_earned", "--season", "2027")
        (world.park_dir / "park_drag_manifest.json").write_text(json.dumps({
            "max_source_game_date": "2027-03-24", "generated_at": "2027-03-25T00:10:00", "export_rows": 2,
        }))
        world.clock.set_et(at(D, 9, 30))
        cli("set-contest-streak", "--date", D.isoformat(), "--streak", "0", "--best-streak", "0", "--saver", "not_earned")

        tomorrow = D + timedelta(days=1)
        world.clock.set_et(at(D, 10))
        with redirect_stderr(io.StringIO()):
            world.run_scheduler_day(D.isoformat(), stop_at_et=at(tomorrow, 1))
        (dm,) = [item for item in world.dms if item["text"].startswith("BTS health ")]
        assert "Live decision refused: season 2027 switch is revoked." in dm["text"]
        assert "contradicts streak 12" in dm["text"]
        assert f"bts set-contest-streak --date {tomorrow}" in dm["text"]
        print("Post-B Carbon scheduler EOD:", dm["at_et"], dm["text"], sep="\n")
        status_file = world.picks_dir.parent / "health_state" / "health_dm_delivery_status.json"
        assert json.loads(status_file.read_text())["sent_day"] == D.isoformat()

        world.clock.set_et(at(tomorrow, 0, 30))
        status_before = cli("season-contract", "status")
        assert "live decision now: refused" in status_before
        cli("set-contest-streak", "--date", tomorrow.isoformat(), "--streak", "12", "--best-streak", "12", "--saver", "not_earned")
        world.clock.set_et(at(tomorrow, 10))
        live = load_decision_streak_state(world.picks_dir, require_contest_state=True, now=at(tomorrow, 10))
        assert (live.observation_kind, live.streak) == ("dated", 12)
        print("00:30 repair example accepted stale synthetic count 12 and still selects it at 10:00")

        cli("set-contest-streak", "--streak", "0")
        cli("season-contract", "enable", "--season", "2027")
        cli("fetch-contest-streak", "--expected-username", "stonehengee")
        after = cli("season-contract", "status")
        assert "the selected legacy manual override is no longer accepted for a live pick" in after
        print("Carbon's independent legacy-command follow-through still masks enabled/fetched auto")


def extra_states(root):
    from bts.health import contest_state, runner
    from bts.health.alert import Alert, format_health_dm_body
    from bts.health.attention import build_policy_alerts

    def render(picks, day, wall, reason):
        state, _ = real_runner(picks, day, wall)
        reference, _ = real_runner(picks, day, end(day))
        print(picks.parent.name, "wall", [(a.level, a.source, a.message) for a in state])
        print(picks.parent.name, "end", [(a.level, a.source, a.message) for a in reference])
        if reference and reason:
            first = reference[0]
            historical = Alert(first.level, first.source, f"Prior slate {day} ET. {reason} {POINTER}", first.incident_key)
            _, attention = build_policy_alerts([historical], streaks={})
            complete = format_health_dm_body([historical] if first.level == "CRITICAL" else [], attention)
            print("Candidate", complete or historical.message)
        return reference

    next_day = D + timedelta(days=1)
    picks = root / "scheduled_early_already_resolved" / "picks"
    dated(picks)
    settled(picks, next_day)
    settled(picks, next_day + timedelta(days=1))
    coverage = render(picks, D, at(next_day + timedelta(days=2), 18),
                      "Coverage trails by more than one settled pick. The predictions array can trail the live counter.")
    assert [(a.level, a.source) for a in coverage] == [("WARN", "contest_state")]

    picks = root / "legacy_concurrent_edit" / "picks"
    dated(picks)
    selection = contest_state.load_contest_streak_state

    def replace_after_select(*args, **kwargs):
        result = selection(*args, **kwargs)
        put(picks, "contest_streak.manual.json", {"streak": 10})
        return result

    with patch.object(contest_state, "load_contest_streak_state", replace_after_select):
        alert, = contest_state.check(picks, expected=True, now=end(D))
        assert "archive it" in alert.message
        print("Dated selection, then concurrent replacement: ", alert.message)

    picks = root / "rollover" / "picks"
    dated(picks, YEAR_END)
    january = YEAR_END + timedelta(days=1)
    sample = [at(YEAR_END, 23, 59)]

    class Clock(datetime):
        @classmethod
        def now(cls, tz=None):
            return sample[0].astimezone(tz)

    def calibrate(*args, **kwargs):
        sample[0] = at(january, 0, 1)
        return []

    with patch.object(runner.calibration, "check", calibrate), patch.object(contest_state, "datetime", Clock):
        with patch.object(contest_state, "check", wraps=contest_state.check) as check:
            alerts = runner.run_all_checks(picks, root / "models", None, today=YEAR_END, contest_state_expected=True)
        check.assert_called_once_with(picks, expected=True)
        assert any("--date 2028-01-01" in a.message for a in alerts)
    print("Base rollover: date passed through for dedup, no contest now or mode; January template after calibration")

    picks = root / "0200_live_change" / "picks"
    switch(picks, status="revoked")
    old, _ = real_runner(picks, D, at(next_day, 2))
    assert any("switch is revoked" in a.message for a in old)
    switch(picks, next_day)
    auto(picks, next_day, source_date=D.isoformat())
    saver_flag(picks, next_day)
    from bts.contest_state import load_decision_streak_state
    live = load_decision_streak_state(picks, require_contest_state=True, now=at(next_day, 10))
    assert live.observation_kind == "automatic"
    print("02:00 refusal, 10:00 real decision accepts intervening fetch/switch; no repair required")

    picks = root / "overlong_saver" / "picks"
    auto(picks)
    switch(picks)
    path = saver_flag(picks)
    path.write_text('{"season":' + '9' * 5000 + '}', encoding="utf-8")
    alert, = render(picks, D, at(next_day, 2), "The contest state health check failed before it could finish.")
    assert (alert.level, alert.source, alert.incident_key) == ("CRITICAL", "health_runner", "health_runner:contest_state")


if __name__ == "__main__":
    signal.alarm(50)
    logging.disable(logging.CRITICAL)
    with TemporaryDirectory(dir=HERE) as scratch:
        scratch = Path(scratch)
        if sys.argv[1] == "scheduler":
            full_scheduler(scratch)
        else:
            import os
            import socket
            from probe import deny_network

            os.chdir(scratch)
            with patch.object(socket, "socket", deny_network), patch.object(socket, "create_connection", deny_network):
                extra_states(scratch)
    signal.alarm(0)
