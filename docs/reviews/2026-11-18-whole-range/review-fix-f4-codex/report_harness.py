"""Drive day-outcome and its real DM formatter without external requests."""

import datetime as dt
import json
from contextlib import ExitStack
from io import BytesIO
from pathlib import Path
from urllib.error import HTTPError
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from bts import dm, scheduler
from bts.cli import cli
from tests.test_day_outcome import DATE, _bound_receipt, _box, _publish_receipt, _write_decision

ET = ZoneInfo("America/New_York")
WALL = dt.datetime(2027, 4, 3, 8, 30, tzinfo=ET)
ROOT = Path(__file__).resolve().parents[2]


class Clock(dt.datetime):
    wall = WALL

    @classmethod
    def now(cls, tz=None):
        return cls.wall.astimezone(tz) if tz else cls.wall.replace(tzinfo=None)


class Chat:
    def __init__(self, *, count=len, failure=None):
        self.count = count
        self.failure = failure
        self.attempted = []
        self.accepted = []

    def response(self, request, **kwargs):
        method = request.full_url.split("/xrpc/")[1].split("?")[0]
        if method == "com.atproto.server.createSession":
            return BytesIO(b'{"accessJwt":"offline"}')
        if method == "chat.bsky.convo.getConvoForMembers":
            return BytesIO(b'{"convo":{"id":"offline"}}')
        assert method == "chat.bsky.convo.sendMessage", request.full_url
        assert kwargs["idempotent"] is False
        body = json.loads(request.data)["message"]["text"]
        self.attempted.append(body)
        if self.count(body) > 1000:
            raise HTTPError(request.full_url, 400, "maxGraphemes", {}, None)
        if self.failure == "timeout":
            raise TimeoutError("response did not arrive")
        self.accepted.append(body)
        if self.failure == "no-id":
            return BytesIO(b"{}")
        return BytesIO(b'{"id":"offline-confirmed"}')


def report(picks, config, games, *, date=DATE, wall=WALL, chat=None, recipient=True):
    chat = chat if chat is not None else Chat()

    def schedule(requested):
        assert requested == (date or wall.date().isoformat()), (requested, date, wall)
        if isinstance(games, Exception):
            raise games
        return [{}] * games if isinstance(games, int) else games

    args = ["day-outcome", "--picks-dir", str(picks), "--config", str(config),
            "--expected-username", "stonehengee"]
    if date is not None:
        args += ["--date", date]
    if recipient:
        args += ["--dm-recipient", "did:plc:offline"]
    with ExitStack() as stack:
        stack.enter_context(patch.object(Clock, "wall", wall))
        stack.enter_context(patch.object(dt, "datetime", Clock))
        stack.enter_context(patch.object(scheduler, "fetch_schedule", schedule))
        stack.enter_context(patch.object(dm, "get_dm_password", return_value="offline"))
        stack.enter_context(patch.object(dm, "retry_urlopen", chat.response))
        result = CliRunner().invoke(cli, args)
    return result, chat


def observed(picks, **fields):
    record = _bound_receipt(picks, **fields)
    _publish_receipt(picks, record)
    return record


def snapshot(root):
    result = {}
    for path in root.rglob("*"):
        name = path.relative_to(root)
        if path.is_symlink():
            result[name] = ("symlink", str(path.readlink()))
        elif path.is_file():
            info = path.stat()
            result[name] = ("file", info.st_mode, info.st_mtime_ns,
                            path.read_bytes() if info.st_mode & 0o400 else info.st_size)
        else:
            result[name] = ("special", path.lstat().st_mode)
    return result
