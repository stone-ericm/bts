"""Local review helpers. All synthetic fixtures and scratch trees stay beside this file."""

from contextlib import contextmanager
from datetime import date, datetime, timedelta
from io import BytesIO
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory
from unittest.mock import patch
from zoneinfo import ZoneInfo


HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
BASE = "c2039541"
HEAD = "c70effe9"
ET = ZoneInfo("America/New_York")
DAY = "2027-04-19"
RUN = datetime(2027, 4, 20, 2, tzinfo=ET)
STREAK = "streak.json"


def scratch(prefix):
    return TemporaryDirectory(prefix=f"{prefix}-", dir=HERE)


def archive(revision, dest, *paths):
    data = subprocess.check_output(["git", "archive", revision, *paths], cwd=REPO, timeout=20)
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=BytesIO(data)) as stream:
        stream.extractall(dest, filter="data")
    return dest


def python_env(src):
    return os.environ | {"PYTHONPATH": str(src), "PYTHONDONTWRITEBYTECODE": "1", "TZ": ET.key}


def python(src, *args, timeout=15):
    return subprocess.run([sys.executable, *map(str, args)], cwd=REPO, env=python_env(src),
                          text=True, capture_output=True, timeout=timeout, check=True)


@contextmanager
def offline():
    def unexpected(*args, **kwargs):
        raise AssertionError(f"Unexpected network call: {args[1:]}")

    with patch.object(socket.socket, "connect", unexpected), patch.object(socket, "getaddrinfo", unexpected):
        yield


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))
    return path


def load(path):
    return json.loads(path.read_text())


def slot(day, game=101):
    return dict(batter_name="Synthetic batter", batter_id=100001, team="PIT", lineup_position=1,
                pitcher_name="Synthetic pitcher", pitcher_id=100002, p_game_hit=0.8, flags=[],
                projected_lineup=False, game_pk=game, game_time=f"{day}T23:00:00Z")


def pick_data(day, result=None, **extra):
    return dict(date=day, run_time=f"{day}T19:00:00Z", pick=slot(day), double_down=None,
                runner_up=None, result=result) | extra


def plant(root, day, result=None, **extra):
    from bts import picks

    daily = picks.parse_daily_pick(json.dumps(pick_data(day, result, **extra)))
    return picks.save_pick(daily, root)


def decision(root, day, *, scoreable, status=None, schema=None):
    from bts.daily_decision import decision_path, write_decision

    status = status or ("delivered" if scoreable else "not_applicable")
    record = write_decision(day, root, action="single" if scoreable else "skip", source="probe",
                            delivery_status=status, scoreable=scoreable, now=RUN)
    assert record is not None
    path = decision_path(day, root)
    if schema is not None:
        dump(path, record | {"schema_version": schema})
    return path


def seed_streak(root, *, streak=21, saver=False, markers=(), season=2027):
    data = dict(streak=streak, saver_available=saver, applied_dates=list(markers),
                updated="2027-04-20T05:00:00+00:00")
    return dump(root / STREAK, data if season is None else data | {"season": season})


def run_cli(root, now=RUN, *, lookback=8):
    from click.testing import CliRunner
    from bts.cli import cli
    from bts.reconcile_receipt import ReconcileReceipt

    clock = now if callable(now) else lambda: now
    with patch("bts.reconcile_receipt.ReconcileReceipt",
               lambda **kw: ReconcileReceipt(clock=clock, lookback_days=kw["lookback_days"])):
        return CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(root), "--lookback", str(lookback)])


def stagger(day, offset):
    return (date.fromisoformat(day) + timedelta(days=offset)).isoformat()
