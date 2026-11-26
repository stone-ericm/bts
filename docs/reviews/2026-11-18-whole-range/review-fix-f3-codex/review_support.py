import datetime as dt
import json
import os
from pathlib import Path
import stat
from unittest.mock import patch
from zoneinfo import ZoneInfo

import bts
from bts import cli, contest_state, daily_decision, dm, scheduler
from click.testing import CliRunner
from tests.test_day_outcome import DATE, _decision


ET = ZoneInfo("America/New_York")
SKIP = _decision("skip", scoreable=False) | {"streak": 30, "saver_available": False}
RECIPIENT = "operator.invalid"
CURRENT = "contest_streak.manual.json"
ARCHIVE = "contest_streak.manual.revoked-probe.json"
SWITCH = "season_contract.json"
REQUESTS = "contest_manual_requests.jsonl"
REPO = Path(__file__).resolve().parents[2]


def identify_package():
    print(f"bts.__file__={bts.__file__}", flush=True)
    expected_root = Path(os.environ["PYTHONPATH"].split(os.pathsep)[0]).resolve()
    assert Path(bts.__file__).resolve().is_relative_to(expected_root)


def dated(day=DATE, *, streak=1, saver="used", stamp=None):
    day_obj = dt.date.fromisoformat(day)
    return {"schema_version": contest_state.DATED_MANUAL_SCHEMA, "season": day_obj.year,
            "effective_date": day, "active_streak": streak, "best_streak": max(streak, 30),
            "saver_state": saver, "source": "manual_cli",
            "expires_at": dt.datetime.combine(day_obj, dt.time.max, ET).isoformat(),
            "recorded_at": stamp or f"{day}T14:00:00Z"}


def switch(day=DATE, *, stamp=None, status="revoked"):
    return {"schema_version": "bts_season_contract_v1", "season": int(day[:4]), "status": status,
            "activation_id": "a" * 32, "switched_at": f"{day}T13:00:00Z",
            "revoked_at": stamp or f"{day}T20:00:00Z"}


def put(path, data):
    path.write_text(json.dumps(data))


def setup(root, decision=SKIP, *, day=DATE, intent="enter"):
    picks = root / "picks"
    account = picks / "account_state"
    account.mkdir(parents=True, exist_ok=True)
    path = daily_decision.decision_path(day, picks)
    path.parent.mkdir(exist_ok=True)
    if decision is None:
        path.unlink(missing_ok=True)
    else:
        put(path, decision)
    config = root / "config.toml"
    delivery = "private" if intent == "research" else "dm"
    config.write_text(f'[scheduler]\nentry_intent = "{intent}"\npick_delivery = "{delivery}"\n')
    return config, picks, account


def snapshot(root):
    captured = {}
    for path in sorted(root.rglob("*")):
        mode = path.lstat().st_mode
        if stat.S_ISREG(mode):
            captured[str(path.relative_to(root))] = (mode, path.read_bytes())
        elif stat.S_ISLNK(mode):
            captured[str(path.relative_to(root))] = (mode, os.readlink(path))
        else:
            captured[str(path.relative_to(root))] = (mode, None)
    return captured


def command(config, picks, *, day=DATE, schedule=({},), transport="confirmed", use_clock=True):
    sends = []
    now = dt.datetime.combine(dt.date.fromisoformat(day) + dt.timedelta(days=1), dt.time(8, 30), ET)

    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    def fetch(requested_day):
        assert requested_day == day
        if isinstance(schedule, Exception):
            raise schedule
        return list(schedule)

    def send(recipient, message):
        sends.append((recipient, message))
        if transport == "error":
            raise OSError("test transport failed")
        return "offline-receipt" if transport == "confirmed" else None

    clock = Clock if use_clock else dt.datetime
    with patch.object(scheduler, "fetch_schedule", side_effect=fetch), patch.object(dm, "send_dm", side_effect=send), \
         patch("datetime.datetime", clock):
        result = CliRunner().invoke(cli.cli, ["day-outcome", "--date", day, "--config", str(config),
                                              "--picks-dir", str(picks), "--expected-username", "stonehengee",
                                              "--dm-recipient", RECIPIENT])
    return {"exit": result.exit_code, "stdout": result.stdout, "stderr": result.stderr,
            "sends": sends, "exception": type(result.exception).__name__ if result.exception else None}


def state(result):
    return json.loads(result["stdout"])["state"] if result["stdout"] else None
