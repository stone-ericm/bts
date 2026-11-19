"""Capture real fetch CLI exit, send and write order for status-file edges."""
import datetime as dt
import json
import os
import stat
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import httpx
from probe_support import REVIEW, REV, NOW, auto, bounded, dump, fetch, offline, put, switch
from bts import cli, saver_state
from bts.leaderboard import auth

PATH = "health_state/contest_streak_fetch_status.json"
STAMP = "2027-07-15T15:00:00Z"
CONFORMING = {
    "map": {"last_error": "old", "last_alert_at": STAMP,
            "last_alert_at_by_category": {"actionable": STAMP}},
    "legacy": {"last_error": "old", "last_alert_at": STAMP},
}
EXTREME = "9999-12-31T23:59:59-01:00"


def inspect(file):
    if not file.exists():
        return None
    if not stat.S_ISREG(file.stat().st_mode):
        return "directory" if file.is_dir() else "fifo"
    return file.read_bytes().hex()


rows = {}
with TemporaryDirectory(dir=REVIEW, prefix="fetch-fixtures-") as directory:
    root = Path(directory)

    def case(label, body, failure=False, *, sequence=False, switched=True):
        picks = root / label / "picks"
        if switched:
            put(picks, "season_contract.json", switch())
        initial_auto = put(picks, "contest_streak.json", auto())
        status = picks.parent / PATH
        status.parent.mkdir()
        if body == "fifo":
            os.mkfifo(status)
        elif body == "directory":
            status.mkdir()
        elif body == "unwritable":
            status.write_text("{}")
            status.parent.chmod(0o555)
        elif isinstance(body, bytes):
            status.write_bytes(body)
        elif body is not None:
            status.write_text(json.dumps(body))
        ledger = initial_auto.with_name("contest_ledger.jsonl")
        flag = initial_auto.with_name("saver_state.json")
        original_writer, original_open = cli._atomic_write_json, Path.open
        original_earn = saver_state.maybe_auto_earn_saver
        events = []

        def write(path, value):
            events.append("try:" + path.name)
            original_writer(path, value)
            events.append("wrote:" + path.name)

        def opening(path, mode="r", *args, **kwargs):
            if path == ledger and mode == "a":
                events.append("append:" + path.name)
            return original_open(path, mode, *args, **kwargs)

        def saver(*args, **kwargs):
            events.append("saver")
            return original_earn(*args, **kwargs)

        good = dict(activeStreak=7, seasonBestStreak=9, predictions=[dict(roundId=1, result="hit")])
        bad = good | {"predictions": {}}
        phases = [
            ("failed", NOW, bad, ()),
            ("success", NOW+dt.timedelta(hours=1), good, ()),
            ("still-cooled", NOW+dt.timedelta(hours=3), bad, ()),
            ("after-six", NOW+dt.timedelta(hours=6), bad, ()),
        ] if sequence else [("single", NOW, bad if failure else good, ())]
        if label.startswith("dry-"):
            phases = [(name, now, profile, ("--dry-run",)) for name, now, profile, _ in phases]
        results = []
        try:
            for name, now, profile_body, args in phases:
                events.clear()
                with (
                    offline(now) as (send, profile),
                    patch.object(cli, "_atomic_write_json", write),
                    patch.object(Path, "open", opening),
                    patch.object(saver_state, "maybe_auto_earn_saver", saver),
                    bounded(),
                ):
                    profile.return_value = profile_body
                    send.side_effect = lambda *a: events.append("DM") or "ack"
                    if label == "unconfirmed":
                        send.side_effect = lambda *a: events.append("DM-attempt")
                    if label == "raised":
                        send.side_effect = httpx.ReadTimeout("offline lost reply")
                    result = fetch(picks, *args)
                    results.append(dict(phase=name, exit=result.exit_code,
                                        output=result.output.replace(str(picks.parent), "FIXTURE"),
                                        exception=type(result.exception).__name__ if result.exception else None,
                                        events=events[:], dm=[c.args[1] for c in send.call_args_list],
                                        auto=inspect(initial_auto), status=inspect(status),
                                        ledger=inspect(ledger), saver=inspect(flag)))
        finally:
            if body == "unwritable":
                status.parent.chmod(0o755)
        rows[label] = results

    bad_bodies = {
        "list": b"[]", "null": b"null", "utf8": b"\xff",
        "malformed-map": {"last_alert_at": STAMP,
                          "last_alert_at_by_category": {"actionable": 42, "transient": STAMP}},
        "malformed-legacy": {"last_alert_at": "2027-07-15T15:00:00"},
        "fifo": "fifo", "directory": "directory", "unwritable": "unwritable",
    }
    for name, content in bad_bodies.items():
        for fail in (False, True):
            case(f"{name}-{'failure' if fail else 'success'}", content, fail)
    for name, value in CONFORMING.items():
        case(f"conforming-{name}", value, sequence=True)
    case("extreme-legacy", {"last_alert_at": EXTREME}, True)
    case("extreme-map", {"last_alert_at_by_category": {"actionable": EXTREME}}, True)
    case("unconfirmed", None, True)
    case("raised", None, True)
    case("unswitched", None, switched=False)
    case("dry-success", None)
    case("dry-failure", None, True)

dump("fetch", rows)
