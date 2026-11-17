"""Independent offline review helpers; never import the author's test fixtures."""
import base64
import contextlib
import datetime as dt
import json
import os
import stat
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent.parent
REAL_DT, REAL_DATE = dt.datetime, dt.date
ET = ZoneInfo("America/New_York")
NOW = REAL_DT(2027, 7, 15, 12, tzinfo=ET)
IDENT = "0123456789abcdef" * 2
TRACE = []
ACTIVE = None


def audit(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
                 "socket.connect", "socket.connect_ex"}:
        raise AssertionError("NETWORK FORBIDDEN: " + event)
    if ACTIVE is not None and event in {"open", "os.mkdir", "os.rename", "os.remove", "os.rmdir", "os.chmod"}:
        paths = [str(a) for a in args if isinstance(a, (str, bytes, Path))]
        if any(str(ACTIVE) in p for p in paths):
            TRACE.append([event, *[str(a).replace(str(ACTIVE), "<case>") for a in args]])


sys.addaudithook(audit)


def contract(**kw):
    return dict(schema_version="bts_season_contract_v1", season=2027, status="automatic",
                activation_id=IDENT, switched_at="2027-07-15T16:00:00Z", revoked_at=None,
                note=None) | kw


def auto(**kw):
    return dict(schema_version="bts_contest_streak_auto_v2", active_streak=7, best_streak=9,
                saver_available=None, source="mlb_bts_profile", source_date="2027-07-14",
                recorded_at="2027-07-15T16:00:00Z", user_id=1, username="audit",
                season=2027, activation_id=IDENT) | kw


def put(picks, name, value):
    path = picks / "account_state" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())
    return path


def snapshot(root):
    out = {}
    if not root.exists():
        return out
    for p in sorted(root.rglob("*")):
        name = str(p.relative_to(root))
        if stat.S_ISLNK(os.lstat(p).st_mode):
            out[name] = ["link", os.readlink(p)]
        elif p.is_dir():
            out[name] = ["dir", p.stat().st_mode & 0o777]
        else:
            try:
                out[name] = ["file", base64.b64encode(p.read_bytes()).decode(), p.stat().st_mode & 0o777]
            except OSError as exc:
                out[name] = ["unreadable", str(exc)]
    return out


def decode_files(tree):
    return {k: base64.b64decode(v[1]) for k, v in tree.items() if v[0] == "file"}


class Harness:
    def __init__(self, source):
        sys.path.insert(0, str(source))
        import pandas  # initialize C extensions before replacing datetime
        import bts.cli as cli
        import bts.contest_fetch as cf
        import bts.contest_state as cs
        import bts.leaderboard.auth as auth
        import bts.saver_state as saver
        import bts.dm as dm
        import bts.health.contest_state as health
        import httpx
        from click.testing import CliRunner
        assert Path(cli.__file__).is_relative_to(Path(source)), cli.__file__
        self.cli, self.cf, self.cs = cli, cf, cs
        self.auth, self.saver, self.dm, self.health = auth, saver, dm, health
        self.runner = CliRunner()
        self.current = NOW
        self.phase = None
        self.cross = None
        self.events, self.dms, self.earn_calls = [], [], []
        self.profile = {"activeStreak": 7, "seasonBestStreak": 9,
                        "predictions": [{"roundId": 1, "result": "hit"}]}
        self.rounds = {1: REAL_DATE(2027, 7, 14)}
        self.auth_error = None
        self.stack = contextlib.ExitStack()
        harness = self

        class Clock(REAL_DT):
            @classmethod
            def now(cls, tz=None):
                harness.events.append(["clock", harness.current.isoformat()])
                return harness.current.astimezone(tz) if tz else harness.current.replace(tzinfo=None)

        self.stack.enter_context(patch.object(dt, "datetime", Clock))
        self.stack.enter_context(patch.object(httpx, "get", side_effect=AssertionError("uncaptured HTTP")))
        def cookies():
            self.event("auth")
            if self.auth_error:
                raise self.auth_error
            return {}
        self.stack.enter_context(patch.object(auth, "load_session_cookies", cookies))
        self.stack.enter_context(patch.object(auth, "extract_uid", lambda _: "uid"))
        self.stack.enter_context(patch.object(auth, "fetch_login_session", lambda **_: SimpleNamespace(user_id=1, username="audit", xsid="x")))
        def profile(*_):
            self.event("profile")
            if isinstance(self.profile, BaseException):
                raise self.profile
            return self.profile
        def rounds():
            self.event("rounds")
            if isinstance(self.rounds, BaseException):
                raise self.rounds
            return self.rounds
        self.stack.enter_context(patch.object(cf, "fetch_profile", profile))
        self.stack.enter_context(patch.object(cli, "_fetch_rounds", rounds))
        self.stack.enter_context(patch.object(dm, "send_dm", lambda *a: (self.events.append(["DM", *a]), self.dms.append(a))))
        old_writer = cli._atomic_write_json
        def writer(path, value):
            self.event("write:" + path.name)
            return old_writer(path, value)
        self.stack.enter_context(patch.object(cli, "_atomic_write_json", writer))
        old_earn = saver.maybe_auto_earn_saver
        def earn(*args, **kw):
            self.event("saver")
            self.earn_calls.append(kw)
            return old_earn(*args, **kw)
        self.stack.enter_context(patch.object(saver, "maybe_auto_earn_saver", earn))

    def event(self, name):
        self.events.append([name])
        if self.cross == name:
            self.current = self.phase

    def invoke(self, root, picks, args):
        global ACTIVE
        self.events.clear()
        self.dms.clear()
        self.earn_calls.clear()
        before = snapshot(root)
        TRACE.clear()
        ACTIVE = root
        try:
            result = self.runner.invoke(self.cli.cli, [*args, "--picks-dir", str(picks)])
        finally:
            ACTIVE = None
        trace = list(TRACE)
        return dict(exit=result.exit_code, output=result.output.replace(str(root), "<case>"),
                    exception=None if result.exception is None else type(result.exception).__name__ + ": " + str(result.exception).replace(str(root), "<case>"),
                    before=before, after=snapshot(root), trace=trace,
                    events=list(self.events), dms=[list(x) for x in self.dms], earn=list(self.earn_calls))

    def fetch(self, root, picks, dry=False):
        return self.invoke(root, picks, ["fetch-contest-streak", "--dm-recipient", "audit.invalid", *(["--dry-run"] if dry else [])])


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")

