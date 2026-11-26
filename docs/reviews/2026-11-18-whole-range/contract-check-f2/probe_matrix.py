import datetime as dt
import json
import os
import signal
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock

from bts.health import contest_state as health

from probe_support import DAY, ET, HERE, REAL_DATETIME, auto, end_of_day, flag, manual, put


def alarm(*_):
    raise TimeoutError("flag read exceeded one second")


def run(picks, now, historical):
    signal.signal(signal.SIGALRM, alarm)
    signal.alarm(1)
    try:
        alerts = health.check(picks, now=now, expected=True,
                              historical_day=now.date() if historical else None)
        return [(a.level, a.source, a.message.replace(str(picks), "PICKS")) for a in alerts]
    except (TimeoutError, ValueError, RecursionError) as exc:
        return type(exc).__name__
    finally:
        signal.alarm(0)


with TemporaryDirectory(dir=HERE, prefix="matrix-") as directory:
    root = Path(directory)
    for case, observation, streak, state in (
        ("K1 minimum", "not_earned", 10, "active"), ("K1 maximum", "not_earned", 15, "active"),
        ("K2", "used", 14, "used"), ("K3 uninitialized", "used", 14, "uninitialized"),
        ("K3 absent", "used", 14, None), ("K3 bad encoding", "used", 14, "bad encoding"),
        ("K3 list", "used", 14, "list"), ("K3 stale season", "used", 14, "stale season"),
        ("K3 directory", "used", 14, "directory"), ("K3 FIFO", "used", 14, "FIFO"),
        ("K3 long number", "used", 14, "long number"), ("K3 deeply nested", "used", 14, "nested"),
        ("K4 active", "active", 14, "not_earned"), ("K4 not earned", "not_earned", 9, "used"),
        ("K5 expired", "used", 14, "not_earned"), ("K5 auto", None, 14, "not_earned"),
        ("K5 legacy", "legacy", 14, "not_earned"), ("K5 no file", None, 14, None),
        ("K6 future and refusal", None, 14, None), ("K7 invalid", "bad", 14, "active"),
        ("M1", "used", 14, "active"), ("M2", "used", 14, "not_earned"),
        ("M2 below 10", "used", 9, "not_earned"),
    ):
        picks = root / case / "picks"
        path = picks / "account_state/saver_state.json"
        path.parent.mkdir(parents=True)
        if state == "bad encoding":
            path.write_bytes(b"\xff")
        elif state == "list":
            path.write_text("[]")
        elif state == "stale season":
            flag(picks, "active", season=2026)
        elif state == "directory":
            path.mkdir()
        elif state == "FIFO":
            os.mkfifo(path)
        elif state == "long number":
            path.write_text('{"season":' + "9" * 5000 + "}")
        elif state == "nested":
            path.write_text("[" * 3000 + "]" * 3000)
        elif state is not None:
            flag(picks, state)

        now = DAY if case != "K5 expired" else DAY + dt.timedelta(days=1)
        if case in {"K5 expired", "K5 auto"}:
            auto(picks, now)
        elif case == "K6 future and refusal":
            auto_path = auto(picks, now + dt.timedelta(days=10))
            auto_path.with_name("season_contract.json").unlink()
        if observation == "legacy":
            put(picks, "contest_streak.manual.json", {
                "streak": streak, "source_date": str(DAY.date()),
                "override_expires_at": (DAY + dt.timedelta(days=1)).isoformat(),
            })
        elif observation == "bad":
            manual(picks)
            observation_path = path.with_name("contest_streak.manual.json")
            data = json.loads(observation_path.read_text())
            data["saver_state"] = "bad"
            observation_path.write_text(json.dumps(data))
        elif observation is not None:
            manual(picks, state=observation, streak=streak)
        morning = run(picks, now, historical=False)
        prior = run(picks, now, historical=True)
        print(json.dumps([case, morning, prior]))

    dec = REAL_DATETIME(2027, 12, 31, 23, 50, tzinfo=ET)
    jan = dec + dt.timedelta(minutes=15)
    picks = root / "year" / "picks"
    manual(picks, dec)
    flag(picks, "active", season=dec.year)
    print("retained at midnight", end_of_day(picks, dec.date(), jan, Mock(return_value="sent")))
    flag(picks, "active", season=jan.year)
    print("replaced flag at midnight", end_of_day(picks, dec.date(), jan, Mock(return_value="sent")))
    manual(picks, jan, state="active")
    print("replaced observation at midnight", end_of_day(picks, dec.date(), jan, Mock(return_value="sent")))
