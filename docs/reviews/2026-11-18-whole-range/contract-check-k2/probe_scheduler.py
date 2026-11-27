import io
import json
import os
import socket
import tempfile
from contextlib import ExitStack, redirect_stderr
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from bts import scheduler as sch, strategy
from bts.daily_decision import load_decision
from bts.health import alert
from bts.picks import load_pick, save_pick
from scripts.audit.season_walk import walk

D = "2027-09-27"
STATUS = "season_boundary_dm_delivery_status.json"


def offline(*args, **kwargs):
    raise OSError("contract check: network blocked")


def clock(when):
    return datetime.fromisoformat(when).replace(tzinfo=sch.ET)


def game(d, n=0, kind="R", detailed="Scheduled", official=None):
    return {"gamePk": 70000 + n, "gameType": kind, "officialDate": official or d,
            "gameDate": clock(d + "T19:05").isoformat(),
            "teams": {"away": {"team": {"name": "Away" + str(n)}},
                      "home": {"team": {"name": "Home" + str(n)}}},
            "status": {"abstractGameCode": "P", "detailedState": detailed}}


class FirstCheck(BaseException):
    pass


def fixture(root, *, mode="private", required=False, recipient=True):
    picks = root / "picks"
    picks.mkdir(exist_ok=True, parents=True)
    return {"orchestrator": {"picks_dir": str(picks), "heartbeat_path": str(root / "heartbeat")},
            "scheduler": {"pick_delivery": mode, "contest_state_required": required},
            "health_checks": {"enabled": False}, "bluesky": {"dm_recipient": "test" if recipient else None}}


def start(config, d, games, *, at=None, behavior="ok", dry=False, lag=0,
          check_result=None, fail_journal=False):
    at = at or d + "T10:00"
    now = clock(at)
    events = []
    bodies = []
    stderr = io.StringIO()

    def set_time(seconds):
        nonlocal now
        now += timedelta(seconds=seconds)

    def send(recipient, body):
        events.append(["dm", now.isoformat()])
        bodies.append(body)
        if lag and "season_boundary" in body:
            set_time(lag)
        if behavior == "interrupt":
            raise KeyboardInterrupt
        if behavior == "fail":
            raise OSError("transport timed out without confirmation")
        if behavior == "no-id":
            return None
        return "captured-id"

    def check(**kwargs):
        events.append(["lineup", now.isoformat()])
        if check_result is None:
            raise FirstCheck
        return check_result

    def idle(*args):
        events.append(["idle", now.isoformat()])

    def fetch(url, **kwargs):
        if "&hydrate=probablePitcher" not in url:
            raise AssertionError(url)
        if isinstance(games, Exception):
            raise games
        slate = games if f"date={d}&" in url else []
        return io.BytesIO(json.dumps({"dates": [{"date": d, "games": slate}]}).encode())

    class Journal(io.StringIO):
        def write(self, line):
            if "SEASON BOUNDARY BEHIND SCHEDULE" in line:
                raise OSError("journal unavailable")
            return super().write(line)

    if fail_journal:
        stderr = Journal()
    result = "returned"
    with ExitStack() as stack:
        stack.enter_context(redirect_stderr(stderr))
        for name, substitute in {"_now_et": lambda: now, "retry_urlopen": fetch,
                                 "_watchdog_ping_sleep": set_time, "run_single_check": check,
                                 "_idle_until_next_wakeup": idle, "notify_ready": lambda: None,
                                 "notify_watchdog": lambda: None, "run_result_polling": lambda *a, **k: "final",
                                 "_trigger_live_forward_capture_on_lock": lambda *a, **k: None}.items():
            stack.enter_context(patch.object(sch, name, substitute))
        stack.enter_context(patch.object(alert, "send_dm", send))
        stack.enter_context(patch("bts.picks.get_game_statuses_detailed", return_value={}))
        try:
            sch.run_day(d, config, dry_run=dry)
        except FirstCheck:
            result = "first lineup"
        except KeyboardInterrupt:
            result = "interrupted"
        except Exception as error:
            result = f"{type(error).__name__}: {error}"
    picks = Path(config["orchestrator"]["picks_dir"])
    status_path = picks.parent / "health_state" / STATUS
    if status_path.is_file():
        try:
            status = json.loads(status_path.read_text())
        except ValueError:
            status = "bad JSON"
    elif status_path.exists():
        status = "non-regular"
    else:
        status = None
    return {"result": result, "events": events, "bodies": bodies, "status": status,
            "journal": stderr.getvalue().splitlines(), "decision_json": load_decision(d, picks)}


def selection(d, root, streak, best):
    day = datetime.fromisoformat(d).date()
    frame = walk._predictions(day, .8)
    statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame.game_pk}
    sel = strategy.select_pick(frame, d, root, streak, False, game_statuses_detailed=statuses,
                               best_streak=best, best_status="trusted")
    return {"skipped": False, "new_lineups": 1, "should_post": sel.pick_result is not None,
            "pick_result": sel.pick_result, "selection": sel}


def main(root):
    out = {}
    daily = fixture(root / "restart")
    out["E6_initial"] = start(daily, D, [game(D)])
    out["E6_restart"] = start(daily, D, [game(D)])
    d2 = "2027-09-28"
    out["two_makeup_days"] = start(daily, d2, [game(d2), game(d2, 1)])
    out["back_to_earlier_slate"] = start(daily, D, [game(D)], at=d2 + "T00:01")
    out["midnight_same_slate"] = start(daily, D, [game(D)], at=d2 + "T00:01")
    for key, d, games, options in (
        ("S3", "2027-09-26", [game("2027-09-26")], {}),
        ("S4", D, [], {}),
        ("S4_nonregular", D, [game(D, kind="F"), game(D, 1, kind="S")], {}),
        ("S6", D, [game(D)], {"dry": True}),
        ("after_midnight", D, [game(D)], {"at": d2 + "T00:01"}),
        ("fallback_year", "2028-09-29", [game("2028-09-29")], {}),
        ("dec31", "2027-12-31", [], {}),
        ("jan01", "2028-01-01", [], {}),
        ("postponed", D, [game(D, detailed="Postponed")], {}),
        ("resume", D, [game(D, detailed="Suspended", official="2027-09-26")], {}),
        ("unreadable", D, OSError("schedule failed"), {}),
        ("journal_error", D, [game(D)], {"fail_journal": True}),
        ("bad_date", "oops", [game(D)], {"at": D + "T10:00"}),
    ):
        out[key] = start(fixture(root / key), d, games, **options)

    for mode in ("private", "dm", "public"):
        out["mode_" + mode] = start(fixture(root / mode, mode=mode), D, [game(D)])
    out["no_recipient"] = start(fixture(root / "no_recipient", recipient=False), D, [game(D)])
    for kind in ("no-id", "fail", "interrupt", "bad-json", "directory", "fifo"):
        config = fixture(root / kind)
        path = root / kind / "health_state" / STATUS
        path.parent.mkdir()
        if kind == "bad-json":
            path.write_text("{")
        elif kind == "directory":
            path.mkdir()
        elif kind == "fifo":
            os.mkfifo(path)
        behavior = kind if kind in {"no-id", "fail", "interrupt"} else "ok"
        out[kind] = [start(config, D, [game(D)], behavior=behavior), start(config, D, [game(D)])]

    for key, at, duration in (("S5_ample", D + "T10:00", 0),
                              ("S5_lost_morning", D + "T15:54:55", 11),
                              ("S5_first_check_late", D + "T16:04:59", 21)):
        out[key] = start(fixture(root / key, mode="dm", required=True), D, [game(D)], at=at, lag=duration)

    for key, d, streak, best in (("S7_floored_pick", D, 57, 57),
                                  ("S7_floored_skip", D, 20, 25),
                                  ("S7_regular_pick", "2027-09-26", 57, 57)):
        config = fixture(root / key)
        picks = Path(config["orchestrator"]["picks_dir"])
        result = selection(d, picks, streak, best)
        out[key] = start(config, d, [game(d), game(d, 1), game(d, 2)], at=d + "T18:30", check_result=result)
        saved = load_pick(d, picks)
        out[key]["pick_policy"] = saved.policy_decision if saved is not None else None

    sent = fixture(root / "already_sent")
    picks = Path(sent["orchestrator"]["picks_dir"])
    existing = selection("2027-09-26", picks, 57, 57)["pick_result"].daily
    existing.date = D
    existing.bluesky_posted = True
    save_pick(existing, picks)
    out["already_sent"] = start(sent, D, [game(D)], at=D + "T19:30")
    print(json.dumps(out, indent=2, default=str))


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as tmp, patch.object(socket.socket, "connect", offline), \
            patch.object(socket, "getaddrinfo", offline):
        main(Path(tmp))
