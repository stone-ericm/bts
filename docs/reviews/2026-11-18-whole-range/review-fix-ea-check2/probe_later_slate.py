"""Replay later-slate instructions on pinned ET clocks and captured transports."""

from contextlib import contextmanager, nullcontext
from datetime import datetime, timedelta
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import contest_state, dm, orchestrator, picks, scheduler
from bts.daily_decision import load_decision
from bts.health import alert
from scripts.audit.season_walk import walk


WORK = Path(__file__).resolve().parent
DAY = "2027-07-17"


def at(iso):
    return datetime.fromisoformat(iso).replace(tzinfo=walk.ET)


def today(clock):
    return at(f"{DAY}T{clock}")


def cli(path, *args, expected=0):
    code, result = walk._invoke(*args, "--picks-dir", path)
    assert code == expected, (code, result)
    print(f"CLI {code} {' '.join(map(str, args))}\n{result.strip()}")
    return result


def status(path):
    return cli(path, "season-contract", "status")


def write_today(path, streak, best, saver):
    day = scheduler._now_et().date().isoformat()
    assert "in effect today" in cli(path, "set-contest-streak", "--date", day,
                                    "--streak", streak, "--best-streak", best, "--saver", saver)


def live(path):
    return contest_state.load_decision_streak_state(path, require_contest_state=True)


def automatic(path, now, streak=0, best=15, saver="used"):
    path.mkdir(parents=True)
    assert walk.install_state(path, now.date(), walk.State("automatic", streak, best, saver, .99)) == []
    walk._set_clock(now)
    reading = path / "account_state/contest_streak.json"
    raw = json.loads(reading.read_text())
    raw["recorded_at"] = (now - timedelta(hours=20)).isoformat()
    reading.write_text(json.dumps(raw))


def config(path):
    settings = walk._config(path)
    settings["orchestrator"]["heartbeat_path"] = str(path.parent / "heartbeat")
    settings["scheduler"]["live_forward_capture_on_lock"] = False
    settings["health_checks"]["enabled"] = False
    settings["bluesky"] = {"dm_recipient": "capture.invalid"}
    return settings


def games(day=DAY, count=2):
    return [{"gamePk": 70000 + i, "gameDate": f"{day}T23:{5 + 5*i:02}:00Z",
             "teams": {side: {"team": {"name": f"{side}{i}"}} for side in ("home", "away")}}
            for i in range(count)]


def rows(date, count=2, probability=.80):
    frame = walk._predictions(date, probability).iloc[:count].copy()
    frame["p_game_hit"] = probability
    return frame


@contextmanager
def environment(frame=None, cascade=None, transport=None):
    scheduled = {pk: {"abstract": "P", "detailed": "Scheduled", "code": "S"}
                 for pk in range(70000, 70003)}
    messages = []
    warnings = []

    def capture(recipient, body):
        if transport is not None:
            transport()
        messages.append((scheduler._now_et().isoformat(), body))
        print("PICK TRANSPORT", messages[-1])
        return "captured-pick"

    def capture_warning(recipient, body):
        warnings.append(body)
        print("HEALTH TRANSPORT", body)
        return "captured-warning"

    with (patch.object(picks, "get_game_statuses_detailed", return_value=scheduled),
          patch.object(scheduler, "_delivery_game_statuses", return_value=scheduled),
          patch.object(orchestrator, "run_cascade", side_effect=cascade,
                       return_value=(frame, "pinned")),
          patch.object(scheduler, "count_new_confirmations", return_value=3),
          patch.object(scheduler, "heartbeat_watchdog", side_effect=lambda *a, **kw: nullcontext()),
          patch.object(dm, "send_dm", side_effect=capture),
          patch.object(alert, "send_dm", side_effect=capture_warning)):
        yield messages, warnings


def scheduler_state(day=DAY):
    return scheduler.SchedulerState(day, "pinned", [], [], [], False, None, None, None)


def check(path, now, frame, *, transport=None):
    walk._set_clock(now)
    day = now.date().isoformat()
    with environment(frame, transport=transport) as (sent, warnings):
        result = scheduler.run_single_check(day, [70000, 70001], set(), config(path), .03)
        if result["should_post"]:
            assert scheduler._deliver_and_lock_pick(result["pick_result"].daily, config(path), path,
                                                    scheduler_state(day), day, "lineup",
                                                    selection=result["selection"])
    print("CHECK", now.isoformat(), result["selection"] and result["selection"].action, len(sent))
    return result, sent, warnings


def c28(root):
    print("C28: 07-15 warning, 07-16 suspended; Pass at 16:08 on 07-17")
    first, second, *_ = scheduler.plan_run_times(games(), now=today("10:00"))
    assert [first["time_et"], second["time_et"]] == [today("16:05"), today("17:05")]
    slate = rows(today("16:05").date())
    for rule in ("old", "new", "before_first", "matching"):
        path = root / rule / "picks"
        july15 = at("2027-07-15T18:35")
        automatic(path, july15, streak=15 if rule == "matching" else 0)
        _, sent, warnings = check(path, july15, rows(july15.date(), probability=.99))
        assert len(sent) == len(warnings) == 1 and "--date 2027-07-15 " in warnings[0]
        walk._set_clock(today("09:30"))
        assert "manual: absent" in status(path)
        if rule == "before_first":
            walk._set_clock(today("15:50"))  # All earlier entries have already settled in this variant.
            write_today(path, 15, 15, "used")
        first_result, sent, _ = check(path, first["time_et"], slate.iloc[:0])
        assert first_result["pick_result"] is None and not sent and picks.load_pick(DAY, path) is None
        walk._set_clock(today("16:08"))
        if rule == "new":
            assert "automatic path alone: accepted (streak 0" in status(path)
            write_today(path, 15, 15, "used")
        elif rule == "matching":
            assert "automatic path alone: accepted (streak 15" in status(path)
            assert live(path).best_streak == 15 and live(path).saver_state == "used"
        result, sent, _ = check(path, second["time_et"], slate)
        expected = (0, "double", 1) if rule == "old" else (15, "skip", 0)
        assert (result["selection"].streak, result["selection"].action, len(sent)) == expected
        if rule == "matching":
            assert live(path).observation_kind == "automatic"


def fallback_day(path, rule, *, fail_refresh, file_at="18:22"):
    """Drive run_day with the real thin-slate plan, fallback and delivery guard."""
    automatic(path, today("10:00"), streak=12, best=14, saver="used")
    frame = rows(today("16:05").date(), count=1, probability=.99)
    # The only competing lineup is still projected, so the first checks save a candidate.
    frame["flags"] = "PROJECTED"
    one_game = games(count=1)
    planned = [run["time_et"].strftime("%H:%M") for run in scheduler.plan_run_times(one_game, now=today("10:00"))]
    assert planned == ["16:05", "17:05", "18:20"]
    saw_retry = []

    def cascade(*args):
        now = scheduler._now_et()
        if now >= today("18:30"):
            saw_retry.append(now)
            if fail_refresh:
                raise RuntimeError("synthetic prediction provider unavailable at fallback")
        return frame.copy(), "pinned"

    filed = []

    def sleep(seconds):
        start = scheduler._now_et()
        end = start + timedelta(seconds=seconds)
        filing = today(file_at)
        if not filed and start <= filing < end:
            walk._set_clock(filing)
            pending = picks.load_pick(DAY, path)
            assert pending and not pending.delivery_attempted and not picks.pick_was_delivered(pending)
            assert not scheduler.load_state(DAY, path).pick_locked
            assert "automatic path alone: accepted (streak 12" in status(path)
            assert live(path).best_streak == 14 and live(path).saver_state == "used"
            print("Settled MLB differs only in best: 12/15/used. Verified pending", pending.pick.batter_name)
            if rule == "new":
                write_today(path, 12, 15, "used")
            filed.append(filing)
        walk._set_clock(end)

    with (environment(cascade=cascade) as (sent, warnings),
          patch.object(scheduler, "fetch_schedule", side_effect=lambda day: one_game if day == DAY else []),
          patch.object(scheduler, "_watchdog_ping_sleep", side_effect=sleep),
          patch.object(scheduler, "run_result_polling", return_value="probe ends at delivery"),
          patch.object(scheduler, "_idle_until_next_wakeup")):
        scheduler.run_day(DAY, config(path))
    assert filed and saw_retry
    daily = picks.load_pick(DAY, path)
    decision = load_decision(DAY, path)
    if rule == "new" and fail_refresh:
        assert not sent and not picks.pick_was_delivered(daily) and not scheduler.load_state(DAY, path).pick_locked
        assert decision is None
        assert any("Saved pick for" in warning and "not delivered" in warning for warning in warnings)
    else:
        assert len(sent) == 1 and picks.pick_was_delivered(daily)
        assert decision["action"] == "single" and decision["delivery_status"] == "delivered"
        assert daily.pick.batter_name == "Batter 0" and "Streak: 12" in sent[0][1]
        if rule == "new":
            assert daily.chosen_on["dated"]["best_streak"] == 15
    print("THIN SLATE", rule, "failed refresh", fail_refresh, "pick sends", len(sent),
          "chosen-on", daily.chosen_on, "final decision", decision and decision["delivery_status"])


def lock_and_send_controls(root):
    path = root / "in-flight" / "picks"
    now = today("16:05")
    automatic(path, now, streak=0)
    slate = rows(now.date())

    def in_flight():
        pending = picks.load_pick(DAY, path)
        assert pending.delivery_attempted and not picks.pick_was_delivered(pending)
        assert not (scheduler.load_state(DAY, path) and scheduler.load_state(DAY, path).pick_locked)
        walk._set_clock(today("16:05:05"))
        write_today(path, 15, 15, "used")        # Deliberately outside the between-checks condition.

    _, sent, _ = check(path, now, slate, transport=in_flight)
    assert len(sent) == 1 and "Streak: 0" in sent[0][1]
    assert scheduler.load_state(DAY, path).pick_locked
    walk._set_clock(today("16:08"))
    assert "streak=15 best=15 saver=used in effect=True" in status(path)
    print("IN FLIGHT: CLI cannot recall the old-state DM. The send was still within the 16:05 check.")

    path = root / "already-locked" / "picks"
    automatic(path, now, streak=0)
    _, sent, _ = check(path, now, slate)
    assert sent and scheduler.load_state(DAY, path).pick_locked
    raw_pick = (path / f"{DAY}.json").read_bytes()
    walk._set_clock(today("16:08"))
    assert "manual: absent" in status(path)
    assert live(path).streak == 0 and (path / f"{DAY}.json").read_bytes() == raw_pick
    print("LOCKED: verify with MLB before acting on old-state pick, no write, pick stays committed.")

    path = root / "expired" / "picks"
    automatic(path, now)
    one_game = games(count=1)
    last = scheduler.plan_run_times(one_game, now=today("10:00"))[-1]["time_et"]
    result, sent, _ = check(path, last, rows(last.date()).iloc[:0])
    assert result["pick_result"] is None and not sent and picks.load_pick(DAY, path) is None
    walk._set_clock(today("19:02"))
    assert "manual: absent" in status(path)
    print("NO CHECK OR FALLBACK LEFT: no pending pick, no write, no scheduler restart (accepted limit).")


if __name__ == "__main__":
    with TemporaryDirectory(dir=WORK) as temp, walk.offline(today("10:00")):
        root = Path(temp)
        c28(root)
        lock_and_send_controls(root)
        for rule, fail, time in (("old", True, "18:22"), ("new", True, "18:22"), ("new", False, "16:08")):
            fallback_day(root / f"fallback-{rule}-{fail}" / "picks", rule, fail_refresh=fail, file_at=time)
