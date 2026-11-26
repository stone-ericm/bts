"""Run the U7 operator scenarios and compare C2 with the real entry watchdog."""

from contextlib import ExitStack, redirect_stderr
from datetime import date, timedelta
from importlib import import_module
from io import StringIO
from unittest.mock import Mock, patch

from probe_support import (
    attempt_direct, day_pick, deadline, fixtures, prepare_schedules, status_body, world_at,
)


class PickSendInterrupted(BaseException):
    pass


def inspect_status(opening, directory):
    def snapshot():
        return {file.relative_to(directory.parent): file.read_bytes()
                for file in directory.parent.rglob("*") if file.is_file()}

    before = snapshot()
    result = opening.invoke("season-contract", "status", "--picks-dir", directory)
    assert snapshot() == before
    return result.output


def watchdog(opening, directory, moment, daily, app_entries, expected_exit):
    from bts import cli, contest_fetch, dm

    day = daily.date
    rounds = {345: date.fromisoformat(day)}
    pending = [
        {"roundId": 345, "unitId": daily.pick.game_pk, "number": 0,
         "playerId": player, "result": "pending"}
        for player in app_entries.get(day, ())
    ]
    with ExitStack() as stack:
        opening.auth_patches(stack, lambda: {"predictions": []}, rounds)
        stack.enter_context(patch.object(contest_fetch, "fetch_pending_predictions", lambda *args: pending))
        stack.enter_context(patch.object(cli, "_fetch_bts_to_mlb", lambda: {daily.pick.batter_id: daily.pick.batter_id}))
        mail = stack.enter_context(patch.object(dm, "send_dm", Mock(return_value="watchdog-ack")))
        result = opening.invoke(
            "check-pick-entered", "--expected-username", opening.ACCOUNT,
            "--dm-recipient", opening.DM, "--picks-dir", directory,
            "--now-et", moment.strftime("%Y-%m-%dT%H:%M:%S"), expected=expected_exit,
        )
        return result.output, [call.args[1] for call in mail.call_args_list]


def execute_recovery(route, config, directory, daily):
    from bts import picks, scheduler

    for _ in range(2):
        if route == "direct":
            assert attempt_direct(config, directory, daily) is False
        else:
            picks.save_pick(daily, directory)
            assert scheduler.run_day(daily.date, config) is None


def scenario(root, patcher, opening, version, label, slate, wall_day, at):
    from bts import dm, picks, scheduler
    from bts.health import alert
    from bts.daily_decision import load_decision

    moment = opening.at(wall_day, at)
    with world_at(root, patcher, opening, moment) as (config, directory, now, _, scoped):
        config["scheduler"]["contest_state_required"] = False
        daily = day_pick(slate)
        prepare_schedules(daily)
        health = alert.send_dm
        outward = dm.send_dm
        remote = []
        app_entries = {}
        fresh_private = label == "fresh_private"
        switched = label == "switched_private"
        pick_did_land = label == "accepted_and_entered"
        if fresh_private:
            assert label == "fresh_private"
            config["scheduler"].update(pick_delivery="private", entry_intent="research")
            assert attempt_direct(config, directory, daily) is True
            assert scheduler.run_day(slate, config) is None
            assert not picks.load_pick(slate, directory).delivery_attempted
            health.assert_not_called()
            outward.assert_not_called()
            print(f"{version}/{label}: no attempt marker, no recovery DM.")
            return

        def pick_transport(_, body):
            assert picks.load_pick(slate, directory).delivery_attempted
            if pick_did_land:
                remote.append(body)
                app_entries[slate] = (daily.pick.batter_id,)
            raise PickSendInterrupted()

        now[0] = opening.at(slate, "18:19")
        outward.side_effect = pick_transport
        try:
            attempt_direct(config, directory, daily)
        except PickSendInterrupted:
            pass
        else:
            raise AssertionError("recommendation transport wasn't reached")
        assert bool(remote) == pick_did_land and bool(app_entries) == pick_did_land
        daily = picks.load_pick(slate, directory)
        assert daily.delivery_attempted and not picks.pick_was_delivered(daily)
        assert load_decision(slate, directory) is None
        outward.reset_mock(side_effect=True)
        if switched:
            config["scheduler"].update(pick_delivery="private", entry_intent="research")
        now[0] = moment
        health.side_effect = [None, "alert-ack"]
        execute_recovery(version, config, directory, daily)
        outward.assert_not_called()
        assert [call.args[1] for call in health.call_args_list] == [status_body(slate)] * 2
        cutoff = picks.submission_cutoff_et(daily)
        eligible = slate == wall_day and moment < cutoff
        assert eligible is (label in {"never_left", "accepted_and_entered", "switched_private"})
        status = inspect_status(opening, directory) if eligible else ""
        if eligible:
            assert f"ET date: {slate}" in status
        assert daily.pick.batter_name not in status_body(slate)
        if label == "never_left":
            output, notes = watchdog(opening, directory, moment, daily, app_entries, 1)
            assert len(notes) == 1 and "Delivery of this pick was NOT confirmed." in notes[0]
            assert "Check your DMs and MLB entry before entering anything" in notes[0]
            assert "Fix it now!" not in notes[0]
            assert "bts season-contract status" in notes[0]
            assert "dm sent" in output.lower()
            assert "bts season-contract enable" in status and "bts fetch-contest-streak" in status
            print("  Refused status output contains pre-existing conditional enable/fetch instructions.")
        elif pick_did_land:
            output, notes = watchdog(opening, directory, moment, daily, app_entries, 0)
            assert "pick entered (match)" in output and not notes
        elif label in {"past_cutoff", "past_first_pitch"}:
            output, notes = watchdog(opening, directory, moment, daily, app_entries, 0)
            assert "outside window" in output and not notes
        elif label in {"midnight", "new_year"}:
            output, notes = watchdog(opening, directory, moment, daily, app_entries, 0)
            assert f"no pick for {wall_day}" in output and not notes
    print(f"{version}/{label}: slate={slate}, wall={moment.isoformat()}, "
          f"status read={eligible}, entry={bool(app_entries)}, two exact recovery bodies.")


def corrected(root, opening):
    from bts import picks, scheduler
    from bts.health import alert

    edges = import_module("opening_edges")
    day = opening.DAYS[1]
    attempts = opening.Season(root / "interrupted")
    attempts.root.mkdir(parents=True)
    attempts.write_config("dm", "enter", True)
    with attempts.run_offline():
        evidence = edges.restart_at_send(attempts, "corrected_after_marker")
    old = evidence["boundary"][0]["pick"]
    observation = evidence["snapshot"]["account"]["contest_streak.manual.json"]
    fresh = opening.Season(root / "fresh")
    fresh.root.mkdir()
    fresh.write_config("dm", "enter", True)
    with fresh.run_offline():
        fresh.clock.move(opening.at(day, "09:30"))
        fresh.dated(day, 46, saver="used")
        fresh.clock.move(opening.at(day, "10:00"))
        fresh.start(day)
        new = fresh.snapshot(day)["pick"]
    assert old["chosen_on"]["streak"] == 1 and old["double_down"] is not None
    assert new["chosen_on"]["streak"] == 46 and new["double_down"] is None
    assert old["pick"]["batter_id"] == new["pick"]["batter_id"] == 27250
    assert observation["active_streak"] == 46
    for route in ("direct", "classification"):
        season = opening.Season(root / route)
        season.root.mkdir()
        season.write_config("dm", "enter", True)
        opening.write_json(season.picks / f"{day}.json", old)
        opening.write_json(season.account_dir / "contest_streak.manual.json", observation)
        season.clock.move(opening.at(day, "15:26"))
        with season.run_offline():
            config = {
                "scheduler": {"pick_delivery": "dm", "entry_intent": "enter",
                              "contest_state_required": True, "live_forward_capture_on_lock": False},
                "orchestrator": {"picks_dir": str(season.picks),
                                 "heartbeat_path": str(season.root / "beat")},
                "bluesky": {"dm_recipient": opening.DM}, "health_checks": {"enabled": False},
            }
            with patch.object(alert, "send_dm", Mock(side_effect=[None, "ack"])) as transport, \
                    patch.object(scheduler, "run_result_polling", return_value="final"):
                daily = picks.load_pick(day, season.picks)
                execute_recovery(route, config, season.picks, daily)
                assert [call.args[1] for call in transport.call_args_list] == [status_body(day)] * 2
                assert not edges.picks_sent(season)
                assert "streak=46" in inspect_status(opening, season.picks)
        print(f"{route}/corrected: same primary 27250, old double versus current single, "
              "two exact recovery bodies; inspection read the corrected streak.")


def delivered_label_watchdog(root, patcher, opening):
    from bts import dm, picks, scheduler
    from bts.daily_decision import load_decision
    from bts.health import alert

    day = "2027-03-25"
    moment = opening.at(day, "18:20")
    with world_at(root, patcher, opening, moment) as (config, directory, _, _, _):
        daily = day_pick(day)
        prepare_schedules(daily)
        config["scheduler"]["contest_state_required"] = False
        dm.send_dm.return_value = None
        assert attempt_direct(config, directory, daily) is True
        assert load_decision(day, directory)["delivery_status"] == "delivered"
        assert daily.delivery_attempted and not picks.pick_was_delivered(daily)
        assert scheduler.run_day(day, config) is None
        alert.send_dm.assert_called_once_with("offline.test", status_body(day))
        _, notices = watchdog(opening, directory, moment + timedelta(minutes=2), daily, {}, 1)
        assert len(notices) == 1 and "Fix it now!" in notices[0]
        print(f"Existing delivered-label watchdog conflict: {notices[0]}")


def main():
    day = "2027-03-25"
    with deadline(50), fixtures("operator") as (root, patcher, opening), redirect_stderr(StringIO()):
        for route in ("direct", "classification"):
            for label, slate, wall, hour in [
                ("never_left", day, day, "18:20"),
                ("accepted_and_entered", day, day, "18:20"),
                ("past_cutoff", day, day, "18:56"),
                ("past_first_pitch", day, day, "19:04"),
                ("midnight", day, "2027-03-26", "00:01"),
                ("new_year", "2027-12-31", "2028-01-01", "00:01"),
                ("fresh_private", day, day, "18:20"),
                ("switched_private", day, day, "18:20"),
            ]:
                scenario(root / f"{route}-{label}", patcher, opening, route, label, slate, wall, hour)
        corrected(root / "corrected", opening)
        delivered_label_watchdog(root / "delivered-label-watchdog", patcher, opening)


if __name__ == "__main__":
    main()
