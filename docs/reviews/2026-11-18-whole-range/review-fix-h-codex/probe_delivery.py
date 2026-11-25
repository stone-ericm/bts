"""Synthetic lineup, delivery, fallback, restart and ET rollover sequences."""
from contextlib import ExitStack, contextmanager, nullcontext
from datetime import date, datetime, time, timedelta
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from bts import contest_state, daily_decision as dd, orchestrator, picks, scheduler, strategy
from bts.simulate import tail_policy as tp
from scripts.audit.season_walk import walk


def clock(day, hour=12, minute=0):
    return datetime.combine(day, time(hour, minute), walk.ET)


def state(text):
    return scheduler.SchedulerState(text, None, [], [], [], False, None, None, None)


@contextmanager
def world(root, day, kind, streak=57, best=57):
    root.mkdir(exist_ok=True)
    with walk.offline(clock(day)), ExitStack() as stack:
        cfg = walk._config(root) | {"bluesky": {"dm_recipient": "offline-review"}}
        cfg["health_checks"]["enabled"] = False
        cfg["orchestrator"]["heartbeat_path"] = str(root / ".heartbeat")
        frame = walk._predictions(day, .84).head(2)
        statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
        replacements = [
            (orchestrator, "run_cascade", lambda *args, **kwargs: (frame, "review")),
            (picks, "get_game_statuses_detailed", lambda *args: statuses),
            (strategy, "get_game_statuses_detailed", lambda *args: statuses),
            (scheduler, "heartbeat_watchdog", lambda *args, **kwargs: nullcontext()),
            (scheduler, "count_new_confirmations", lambda *args: 6),
            (scheduler, "_lock_decision_from_predictions", lambda *args, **kwargs:
             scheduler.LockDecision(True, None, True)),
            (scheduler, "_void_committed_games", lambda *args: {}),
            (scheduler, "_trigger_live_forward_capture_on_lock", Mock()),
            (scheduler, "_alert_delivery_outcome_unknown", Mock()),
            (scheduler, "_alert_contest_state_failure", Mock()),
            (scheduler, "notify_ready", Mock()),
            (scheduler, "notify_watchdog", Mock()),
        ]
        for mod, attr, new in replacements:
            stack.enter_context(patch.object(mod, attr, new))
        assert not walk.install_state(root, day, walk.State(kind, streak, best, "used", .84))
        yield cfg, frame, statuses


def chosen(root, sel):
    daily = sel.pick_result.daily
    obs = contest_state.load_decision_streak_state(root, require_contest_state=True)
    assert picks._valid_chosen_on(daily.chosen_on) == picks.chosen_on_record(obs, required=True)
    return daily


def checked_record(root, text, saved, status):
    rec = dd.load_decision(text, root)
    assert rec and rec["delivery_status"] == status and rec["scoreable"]
    assert rec["action"] == ("double" if saved.double_down else "single")
    meta = saved.policy_decision
    for field, storage in (("source", "source"), ("objective", "objective"), ("best_supplied", "best_streak"),
                           ("best_status", "best_status"), ("effective_best", "effective_best"),
                           ("tail_sha256", "tail_policy_sha256"), ("degraded_reason", "degraded_reason"),
                           ("guard", "policy_guard")):
        assert rec[storage] == meta[field], (storage, rec[storage], meta[field])
    assert rec["primary"]["batter_id"] == saved.pick.batter_id
    if saved.double_down:
        assert rec["double_down"]["batter_id"] == saved.double_down.batter_id
    return rec


def deliver(saved, cfg, root, selection=None):
    return scheduler._deliver_and_lock_pick(
        saved, cfg, root, state(saved.date), saved.date, "review",
        selection=selection)


def h5(root):
    for text, kind in product(("2027-07-15", "2027-09-24", "2027-09-25", "2027-09-26"),
                              ("dated", "automatic")):
        day = date.fromisoformat(text)
        folder = root / f"{text}-{kind}"
        with world(folder, day, kind) as (cfg, frame, statuses):
            _, sel, _ = orchestrator.run_and_pick(cfg, text)
            want = "single" if day.month == 7 else "double"
            assert sel.action == want and sel.decision.objective == "longest_streak"
            original = chosen(folder, sel)
            check = scheduler.run_single_check(text, list(statuses), set(), cfg, .01)
            saved = picks.load_pick(text, folder)
            assert saved.chosen_on == original.chosen_on and saved.policy_decision == original.policy_decision
            assert [p.batter_id for p in (saved.pick, saved.double_down) if p] == (
                [100] if want == "single" else [100, 101])

            def send(recipient, message):
                attempted = picks.load_pick(text, folder)
                assert attempted.delivery_attempted and not picks.pick_was_delivered(attempted)
                assert recipient == "offline-review" and "Batter 0" in message
                assert ("Batter 1" in message) == (want == "double")
                return "captured"

            with patch("bts.dm.send_dm", side_effect=send) as captured:
                assert deliver(saved, cfg, folder, check["selection"])
                captured.assert_called_once()
            rec = checked_record(folder, text, saved, "delivered")
            assert rec["streak"] == 57 and rec["tail_policy_sha256"] == strategy._load_mdp()["tail"].sha256
            assert scheduler.load_state(text, folder).committed_pick_written
            print("fresh", text, kind, want, rec["objective"], rec["delivery_status"])


def send_and_restart(root):
    day = date(2027, 9, 26)
    text = str(day)
    for mode in ("cached", "known-failure", "interrupted-send", "classification-restart"):
        folder = root / mode
        with world(folder, day, "automatic") as (cfg, frame, statuses):
            scheduler.run_single_check(text, list(statuses), set(), cfg, .01)
            disk = picks.load_pick(text, folder)
            assert disk.policy_decision["objective"] == "longest_streak" and disk.double_down
            assert dd.load_decision(text, folder) is None
            with patch.object(orchestrator, "run_cascade", return_value=(frame.iloc[0:0], "empty")):
                refresh = scheduler._refresh_pick_at_fallback_decision(cfg, text, disk, .01)
            assert refresh.daily is disk and refresh.selection is None
            if mode == "known-failure":
                with patch("bts.dm.send_dm", side_effect=OSError("known rejected send")) as sent:
                    assert not deliver(disk, cfg, folder, refresh.selection)
                    sent.assert_called_once()
                disk = picks.load_pick(text, folder)
                assert not disk.delivery_attempted and dd.load_decision(text, folder) is None
                print(mode, "attempt cleared and no decision written")
            if mode in {"interrupted-send", "classification-restart"}:
                with patch("bts.dm.send_dm", side_effect=KeyboardInterrupt("transport interrupted")):
                    try:
                        deliver(disk, cfg, folder, refresh.selection)
                    except KeyboardInterrupt:
                        pass
                    else:
                        raise AssertionError("transport interrupt was swallowed")
                disk = picks.load_pick(text, folder)
                assert disk.delivery_attempted and dd.load_decision(text, folder) is None
                with patch("bts.dm.send_dm", side_effect=AssertionError("unexpected second send")) as sent:
                    if mode == "interrupted-send":
                        assert not deliver(disk, cfg, folder, refresh.selection)
                    else:
                        games = [
                            {"gamePk": int(row.game_pk), "gameDate": row.game_time,
                             "status": {"abstractGameCode": "P", "detailedState": "Scheduled"},
                             "teams": {"away": {"team": {"name": row.team}},
                                       "home": {"team": {"name": row.pitcher_team}}}}
                            for row in frame.itertuples()
                        ]
                        walk._set_clock(clock(day, 18, 40))
                        with (patch.object(scheduler, "fetch_schedule", side_effect=[games, []]),
                              patch.object(scheduler, "_watchdog_ping_sleep"),
                              patch.object(scheduler, "_idle_until_next_wakeup"),
                              patch.object(scheduler, "run_result_polling", return_value="final"),
                              patch.object(scheduler, "retry_urlopen", side_effect=OSError("offline"))):
                            scheduler.run_day(text, cfg)
                    sent.assert_not_called()
                status = "locked_unconfirmed"
            else:
                with patch("bts.dm.send_dm", return_value="captured-retry") as sent:
                    assert deliver(disk, cfg, folder, refresh.selection)
                    sent.assert_called_once()
                status = "delivered"
            rec = checked_record(folder, text, disk, status)
            assert rec["streak"] is None and rec["saver_available"] is None
            print(mode, status, rec["objective"], rec["tail_policy_sha256"][:12])


def refresh_obs(root, active, best, day):
    with walk._fetch_stubs(active, best, day - timedelta(days=1)):
        return walk._invoke("fetch-contest-streak", "--picks-dir", root,
                            "--expected-username", walk.ACCOUNT)


def reuse_and_rollover(root):
    day = date(2027, 9, 26)
    text = str(day)
    for before, after, objective in ((56, 57, "reach57"), (57, 0, "longest_streak")):
        folder = root / f"{before}-{after}"
        with world(folder, day, "automatic", before) as (cfg, frame, statuses):
            scheduler.run_single_check(text, list(statuses), set(), cfg, .01)
            saved = picks.load_pick(text, folder)
            assert saved.policy_decision["objective"] == objective
            walk._set_clock(clock(day, 14))
            code, output = refresh_obs(folder, after, 57, day)
            assert code == 0, output
            if after == 0:
                real = scheduler._refresh_pick_at_fallback_decision(cfg, text, saved, .01)
                assert scheduler._refresh_is_genuine_skip(real)
            with patch.object(orchestrator, "run_cascade", return_value=(frame.iloc[:0], "empty")):
                empty = scheduler._refresh_pick_at_fallback_decision(cfg, text, saved, .01)
            with patch("bts.dm.send_dm", return_value="reuse-id") as send:
                assert deliver(saved, cfg, folder, empty.selection)
                assert f"streak: {after}" in send.call_args.args[1].lower(), send.call_args
            assert checked_record(folder, text, saved, "delivered")["objective"] == objective
            print("R33", before, "to", after, "saved objective", objective)
    folder = root / "above-cap"
    with world(folder, day, "automatic"):
        code, output = refresh_obs(folder, 58, 58, day)
        assert code != 0 and "57" in output, output
        obs = contest_state.load_decision_streak_state(folder, require_contest_state=True)
        assert obs.streak == obs.best_streak == 57
        print("58/58 fetch refused with error and retained 57/57", output.strip().splitlines()[0])
    for text, want in (("2027-09-26", "double"), ("2027-09-27", "skip")):
        day = date.fromisoformat(text)
        with world(root / text, day, "dated") as (cfg, _, _):
            _, sel, _ = orchestrator.run_and_pick(cfg, text)
            assert sel.action == want
            print("finale", text, sel.action, sel.decision.days_effective)
    december = date(2027, 12, 31)
    january = date(2028, 1, 1)
    folder = root / "rollover"
    with world(folder, december, "automatic") as (cfg, _, _):
        walk._set_clock(clock(december, 23, 59))
        old = contest_state.load_decision_streak_state(folder, require_contest_state=True)
        assert old.streak == 57 and strategy._days_remaining(str(december)) == 0
        walk._set_clock(clock(january, 0, 1))
        try:
            contest_state.load_decision_streak_state(folder, require_contest_state=True)
        except contest_state.ContestStateError as exc:
            print("ET rollover refused old season", str(exc).splitlines()[0])
        else:
            raise AssertionError("old automatic switch accepted after ET midnight")
    with world(folder, january, "automatic") as (cfg, _, _):
        _, sel, _ = orchestrator.run_and_pick(cfg, str(january))
        assert (sel.action, sel.decision.days_effective, strategy._days_remaining(str(january))) == (
            "single", 180, 271)
        print("ET rollover accepted 2028 state", sel.action, "raw 271 effective 180")


def main():
    with TemporaryDirectory() as temp:
        root = Path(temp)
        h5(root)
        send_and_restart(root)
        reuse_and_rollover(root)


if __name__ == "__main__":
    main()
