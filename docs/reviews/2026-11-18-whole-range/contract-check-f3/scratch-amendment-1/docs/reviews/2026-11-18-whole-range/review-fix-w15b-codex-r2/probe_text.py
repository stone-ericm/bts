"""Sweep real W15 text producers and both unchanged OOM matchers."""
import contextlib
import hashlib
import io
import json
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import bts
from bts.features import compute
from bts.health import attention


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
import probe_health as fixture


HOSTILE = (
    "oom-kill", "OOM", "room", "out of memory", "status=137", "code=killed status=9/KILL",
    "b" + chr(96), "with\nnewline", "with\rcr", "tab\tx", "escape" + chr(27),
    "separator" + chr(0x2028), "zero" + chr(1), "back" + chr(92), "café",
)
GENUINE = (
    "systemd[1]: process of this unit killed by the OOM killer.",
    "kernel: Out of memory: Killed process 123 (python)",
    "systemd[1]: Main process exited, code=killed, status=9/KILL",
    "systemd[1]: Failed with result 'oom-kill'.",
    "child exited: status=137",
    "systemd[1]: code=killed signal=kill",
)


class Capture(logging.Handler):
    def __init__(self):
        super().__init__()
        self.messages = []

    def emit(self, record):
        msg = record.getMessage()
        if record.name == fixture.health.__name__ or msg.startswith(f"[{fixture.SOURCE} WARN]"):
            self.messages.append(msg)


def put_row(status, label, *, with_copy, reason="invalid_entries", separator="T", cache_path="cache"):
    day = fixture.DAY
    identity = hashlib.sha256(label.encode()).hexdigest()
    row = dict(
        reason=reason, cache_path=cache_path, original_sha256="b" * 64,
        preserved="probable_pitcher_lookup.20261231T140000Z.cccccccccccc.json" if with_copy else None,
        dropped=1, recovered_from_cache=0, rescanned=0, unusable_feed=0, without_feed=1,
        schema_version="bts_probable_pitcher_cache_status_v1", recorded_at=fixture.AT.replace("T", separator),
        recorded_date_et=str(day), game_ids=[-10 ** 80], identity=identity,
    )
    if reason == "unreadable_cache":
        row.update(dropped=None, recovered_from_cache=None, rescanned=2, unusable_feed=None, without_feed=None, game_ids=[])
    name = f"{day}.{reason}.{identity[:12]}.json"
    path = status / name
    path.write_text(json.dumps(row))
    return path


def run_reader(root):
    root_logger = logging.getLogger()
    handler = Capture()
    root_logger.addHandler(handler)
    try:
        with fixture.captured() as (packets, rejected, _):
            (alert,) = fixture.run(picks=root / "picks")
        assert len(packets) == 1 and not rejected
        return handler.messages, alert.message, packets[0]
    finally:
        root_logger.removeHandler(handler)


def health_messages():
    data = Path("room") / ("config" + chr(96) + "\r\n")
    status = data / "health_state" / fixture.SOURCE
    status.mkdir(parents=True)
    before = {}
    for i, bad in enumerate(HOSTILE):
        day = fixture.DAY - timedelta(days=i % 2)
        file = status / f"{day}.{bad}.json"
        file.write_bytes(b"{")
        before[file] = b"{"
    for i, separator in enumerate(("T", chr(96), "\n", "\r", "\t", chr(0x2028), chr(0), "🦉", " ")):
        reason = "invalid_entries" if i % 2 == 0 else "unreadable_cache"
        record = put_row(status, str(i), separator=separator, with_copy=i % 3 == 0, reason=reason,
                         cache_path="other/" + "/".join(HOSTILE) + chr(0) + "/models/cache.json")
        before[record] = record.read_bytes()
    for candidate in (
        f"{fixture.DAY}.invalid_entries.{'a' * 12}.json",
        f"{fixture.DAY}.unreadable_cache.{'f' * 12}.json",
        f"{fixture.DAY}.invalid_entries.{'A' * 12}.json",
        f"{fixture.DAY}.invalid_entries.{'a' * 12}.json.extra.json",
    ):
        path = status / candidate
        path.write_bytes(b"{")
        before[path] = b"{"
    logs, summary, wire = run_reader(data)
    assert all(path.read_bytes() == raw for path, raw in before.items())
    assert sum("whose name is not a status record name" in text for text in logs) == len(HOSTILE) + 2
    assert sum("status unreadable: record" in text for text in logs) == 2
    assert sum(fixture.AT in text for text in logs) == 9
    assert "28 observations" in summary
    print("Hostile reader records:", len(before), "messages:", len(logs), "wire graphemes:", len(wire))
    unreadable_root = Path("out of memory") / ("blocked" + chr(96) + "\n")
    unreadable = unreadable_root / "health_state" / fixture.SOURCE
    unreadable.parent.mkdir(parents=True)
    os.mkfifo(unreadable)
    blocked_logs, blocked_summary, blocked_wire = run_reader(unreadable_root)
    assert unreadable.is_fifo() and any(fixture.health.FOLDER in text for text in blocked_logs)
    print("Unreadable custom folder:", blocked_logs[0], "wire graphemes:", len(blocked_wire))
    return [*logs, summary, *wire.splitlines(), *blocked_logs, blocked_summary, *blocked_wire.splitlines()]


def producer_messages():
    raw = Path("raw/2026")
    raw.mkdir(parents=True)
    (raw / "100.json").write_text("{}")
    (raw / "101.json").write_bytes(b"{")
    cache = Path("oom") / ("room" + chr(96) + "\n") / "models" / ("café" + chr(27) + ".json")
    cache.parent.mkdir(parents=True)
    cache.write_bytes(b'{"100":null,"101":true,"102":[]}')
    rejected = cache.parent / "rejected"
    rejected.touch()
    status = cache.parent.parent / "health_state" / fixture.SOURCE
    status.parent.mkdir(parents=True)
    status.touch()
    output = io.StringIO()
    fixture.Clock.at = datetime.fromisoformat(fixture.AT)
    with (
        contextlib.redirect_stderr(output),
        patch.object(fixture.producer, "datetime", fixture.Clock),
        patch.object(compute, "_probable_pitcher_cache_path", cache),
    ):
        lookup = compute._build_probable_pitcher_lookup("raw")
    assert list(lookup) == [100] and cache.read_bytes() == b'{"100":null,"101":true,"102":[]}'
    lines = output.getvalue().split("\n")
    assert len(lines) == 4 and lines[-1] == ""
    lines.pop()
    assert lines[0].startswith("WARNING: probable-pitcher cache: dropped 3 invalid entries")
    assert lines[1] == "WARNING: could not preserve probable-pitcher cache evidence (NotADirectoryError)"
    assert lines[2].startswith(f"WARNING: could not write probable-pitcher cache status record {fixture.DAY}.invalid_entries.")
    print("Producer warnings:", repr(lines))
    return lines


def journal_scan(lines):
    def journal(command, **options):
        assert command[0] == "journalctl" and options["timeout"] == 5
        return SimpleNamespace(returncode=0, stdout="\n".join(lines) + "\n")
    with patch.object(fixture.analytics_artifacts_missing.subprocess, "run", journal):
        return fixture.analytics_artifacts_missing._fatal_scheduler_journal_line(
            "bts-scheduler.service", "2026-12-31T21:00:00-05:00")


def matchers(texts):
    for text in texts:
        assert text and all(char.isprintable() and char != chr(96) for char in text), repr(text)
        assert not attention.OOM_EVIDENCE_RE.search(f"{fixture.SOURCE} {text}"), repr(text)
    lines = [f"Dec 31 22:35:00 box bts[1]: {text}" for text in texts]
    assert journal_scan(lines) is None
    alerts = [fixture.alert.Alert("WARN", fixture.SOURCE, text) for text in texts]
    assert attention.build_policy_alerts(alerts, streaks={})[0] == []
    pairs = 0
    for fatal in GENUINE:
        line = f"Dec 31 22:36:00 box {fatal}"
        for item in lines:
            assert journal_scan([item, line]) == journal_scan([line, item]) == line
            pairs += 2
        assert journal_scan([*lines, line]) == journal_scan([line, *lines]) == line
        genuine = [fixture.alert.Alert("WARN", "analytics_artifacts_missing", fatal),
                   fixture.alert.Alert("WARN", "memory_growth", "working set grew")]
        expected = attention.build_policy_alerts(genuine, streaks={})[0]
        assert attention.build_policy_alerts(alerts + genuine, streaks={})[0] == expected
        assert attention.build_policy_alerts(genuine + alerts, streaks={})[0] == expected
    line = lines[0].replace(" box ", " room-node ", 1)
    assert journal_scan([line]) == line[:240]
    print("Matcher sweep:", len(lines), "messages;", pairs, "before/after pairs; source-safe text; "
          "unchanged matcher can classify external hostname metadata:", repr(line[:240]))


def main():
    root = HERE.parents[2]
    assert Path(bts.__file__).resolve() == root / "src/bts/__init__.py"
    before = Path.cwd()
    with TemporaryDirectory(dir=HERE, prefix="text-") as work:
        os.chdir(work)
        try:
            matchers(health_messages() + producer_messages())
        finally:
            os.chdir(before)


if __name__ == "__main__":
    main()
