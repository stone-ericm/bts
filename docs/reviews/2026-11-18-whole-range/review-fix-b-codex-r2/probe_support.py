"""Offline fixtures shared by the independent review probes."""
import datetime as dt
import json
import os
import signal
import socket
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]
REV = sys.argv[1] if len(sys.argv) > 1 else "head"
sys.path[:0] = [str((ROOT if REV == "head" else REVIEW / "scratch" / REV) / "src"), str(ROOT)]

from click.testing import CliRunner
from bts import cli, contest_fetch, dm
from bts.leaderboard import auth

REAL = dt.datetime
ET = ZoneInfo("America/New_York")
NOW = REAL(2027, 7, 15, 12, tzinfo=ET)
AID = "a" * 32


class Timeout(SystemExit):
    pass


@contextmanager
def bounded(seconds=0.35):
    def fail(*_):
        raise Timeout("blocked on read")
    previous = signal.signal(signal.SIGALRM, fail)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


@contextmanager
def offline(now=NOW):
    import pandas
    from unittest.mock import Mock

    class Clock(REAL):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    def no_network(*a, **kw):
        raise AssertionError("uncaptured transport")

    send = Mock(return_value="ack-1")
    profile = Mock(return_value=dict(activeStreak=7, seasonBestStreak=9,
                                    predictions=[dict(roundId=1, result="hit")]))
    with (
        patch.object(dt, "datetime", Clock),
        patch.object(socket, "create_connection", no_network),
        patch.object(socket, "getaddrinfo", no_network),
        patch.object(socket.socket, "connect", no_network),
        patch.object(dm, "send_dm", send),
        patch.object(auth, "load_session_cookies", return_value={}),
        patch.object(auth, "extract_uid", return_value="uid"),
        patch.object(auth, "fetch_login_session", return_value=SimpleNamespace(
            user_id="uid", username="offline", xsid="x")),
        patch.object(contest_fetch, "fetch_profile", profile),
        patch.object(contest_fetch, "fetch_pending_predictions", return_value=[]),
        patch.object(cli, "_fetch_rounds", return_value={1: now.date() - dt.timedelta(days=1),
                                                        2: now.date()}),
        patch.object(cli, "_fetch_bts_to_mlb", return_value={88: 592450, 99: -1}),
    ):
        yield send, profile


def invoke(picks, *args):
    return CliRunner().invoke(cli.cli, [*args, "--picks-dir", str(picks)])


def fetch(picks, *args):
    return invoke(picks, "fetch-contest-streak", "--expected-username", "offline",
                  "--dm-recipient", "offline.example", *args)


def put(picks, name, value):
    path = picks / "account_state" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, bytes):
        path.write_bytes(value)
    else:
        path.write_text(json.dumps(value), encoding="utf-8")
    return path


def switch(now=NOW, **updates):
    return dict(schema_version="bts_season_contract_v1", season=now.year, status="automatic",
                activation_id=AID, switched_at=now.isoformat(), revoked_at=None, note=None) | updates


def auto(now=NOW, **updates):
    return dict(schema_version="bts_contest_streak_auto_v2", active_streak=12,
                best_streak=13, season=now.year, activation_id=AID, source="mlb_bts_profile",
                source_date=(now.date()-dt.timedelta(days=1)).isoformat(),
                recorded_at=now.isoformat()) | updates


def dated(now=NOW, **updates):
    return dict(schema_version="bts_contest_streak_manual_v3", source="operator",
                effective_date=now.date().isoformat(), active_streak=1, best_streak=1,
                season=now.year, saver_state="not_earned", recorded_at=now.isoformat(),
                expires_at=REAL.combine(now.date(), dt.time.max, ET).isoformat()) | updates


def snapshot(root):
    import stat
    data = {}
    for p in root.rglob("*"):
        st = p.lstat()
        body = p.read_bytes() if stat.S_ISREG(st.st_mode) else None
        data[str(p.relative_to(root))] = (st.st_mode, st.st_mtime_ns, body)
    return data


def dump(family, rows):
    blob = json.dumps(rows, indent=2, ensure_ascii=False) + "\n"
    (REVIEW / f"{family}-{REV}.json").write_text(blob, encoding="utf-8")
    print(f"{family} {REV}: {len(rows)} fixture groups")
