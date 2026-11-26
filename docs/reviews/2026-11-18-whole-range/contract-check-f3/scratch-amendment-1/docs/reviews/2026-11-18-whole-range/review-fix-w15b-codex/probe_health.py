import hashlib
import itertools
import json
import logging
import os
import sys
from contextlib import contextmanager
from datetime import date, datetime, timedelta, timezone
from io import BytesIO, StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError

import bts
from bts import dm
from bts.daily_decision import DECISION_SCHEMA
from bts.features import probable_pitcher_cache as producer
from bts.health import alert, analytics_artifacts_missing, attention, probable_pitcher_cache as health, runner


HERE = Path(__file__).resolve().parent
DAY = date(2026, 12, 31)
NEXT = DAY + timedelta(days=1)
SOURCE = "probable_pitcher_cache"
AT = "2027-01-01T04:59:00+00:00"
TICK = chr(96)
BACKSLASH = chr(92)
STATUS = Path("data/health_state") / SOURCE
PICKS = Path("data/picks")
CACHE = Path("data/models/probable_pitcher_lookup.json")


class Clock(datetime):
    at = datetime.fromisoformat(AT)

    @classmethod
    def now(cls, tz=None):
        return cls.at.astimezone(tz)


def write_record(folder, *, timestamp=AT, cache_path=str(CACHE), preserved=None):
    folder.mkdir(parents=True, exist_ok=True)
    identity = "a" * 64
    path = folder / f"{DAY}.invalid_entries.{identity[:12]}.json"
    path.write_text(json.dumps(dict(
        reason="invalid_entries", cache_path=cache_path, original_sha256="b" * 64, preserved=preserved,
        dropped=1, recovered_from_cache=0, rescanned=0, unusable_feed=0, without_feed=1,
        schema_version="bts_probable_pitcher_cache_status_v1", recorded_at=timestamp,
        recorded_date_et=str(DAY), game_ids=[100], identity=identity,
    )))
    return path


def summary_key(day, *parts):
    digest = hashlib.sha256(json.dumps(sorted(parts), separators=(",", ":")).encode()).hexdigest()[:16]
    return f"{SOURCE}:{day}:{digest}"


@contextmanager
def captured(*, sources=(SOURCE,)):
    packets, rejected = [], []
    logs = StringIO()
    handler = logging.StreamHandler(logs)
    root = logging.getLogger()
    root.addHandler(handler)
    safe_run = runner._safe_run
    Clock.at = datetime(2027, 1, 1, 5, 1, tzinfo=timezone.utc)

    def api(request, **kwargs):
        url = request.full_url
        if url.endswith("createSession"):
            return BytesIO(b'{"accessJwt":"offline"}')
        if "getConvoForMembers" in url:
            assert url.endswith("did:plc:fake")
            return BytesIO(b'{"convo":{"id":"offline"}}')
        assert url.endswith("chat.bsky.convo.sendMessage")
        assert kwargs == dict(timeout=15, idempotent=False)
        body = json.loads(request.data)["message"]["text"]
        # Every wire fixture is ASCII with LF line endings, so each character is one grapheme.
        assert body.isascii() and "\r" not in body, repr(body)
        if len(body) > 1000:
            rejected.append(body)
            raise HTTPError(url, 400, "synthetic 1000-grapheme cap", {}, None)
        packets.append(body)
        return BytesIO(b'{"id":"offline"}')

    try:
        with (
            patch.object(runner, "_safe_run", lambda name, call: safe_run(name, call) if name in sources else []),
            patch.object(runner.realized_calibration, "_current_deploy_iso", return_value=None),
            patch.object(runner, "datetime", Clock),
            patch.object(alert, "datetime", Clock),
            patch.object(dm, "get_dm_password", return_value="offline"),
            patch.object(dm, "retry_urlopen", api),
        ):
            yield packets, rejected, logs
    finally:
        root.removeHandler(handler)


def run(day=DAY, picks=PICKS, *, shadow=False, receipt=None):
    overrides = {"health_dm_delivery_status": receipt} if receipt is not None else {}
    return runner.run_all_checks(
        picks, picks.parent / "models", "did:plc:fake", today=day,
        shadow_model_enabled=shadow, shadow_unit=None, thresholds_overrides=overrides,
    )


def oom_paths(prior):
    write_record(STATUS)
    expected = ("oom/data", "OOM-killed/data", "out of memory/data", "x/oom_kill/data", "x/out-of-memory/data")
    for index, name in enumerate(expected):
        alias = Path(name)
        alias.parent.mkdir(parents=True, exist_ok=True)
        alias.symlink_to(PICKS.parent.resolve())
        assert (alias / "health_state" / SOURCE).samefile(STATUS)
        with captured() as (packets, rejected, _):
            result = run(picks=alias / "picks", receipt=Path(f"receipt-{index}.json"))
        if prior:
            assert [a.source for a in result] == [SOURCE, "analytics_job_oom"] and not packets
            assert len(rejected) == 1
        else:
            assert [a.source for a in result] == [SOURCE] and not rejected and len(packets) == 1
        print("OOM alias", name, "sources", [a.source for a in result], "accepted", list(map(len, packets)),
              "rejected", list(map(len, rejected)))


def backticks(prior):
    status = Path(f"c{TICK}ustom{BACKSLASH}data/health_state") / SOURCE
    picks = status.parent.parent / "picks"
    evidence = "probable_pitcher_lookup.20261231T140000Z.bbbbbbbbbbbb.json"
    cache = f"else{TICK}where{BACKSLASH}data/models/probable_pitcher_lookup.json"
    record = write_record(status, cache_path=cache, preserved=evidence)
    bad = status / f"{DAY}.broken{TICK}{BACKSLASH}x.json"
    bad.write_bytes(b"{")
    with captured() as (packets, rejected, logs):
        result, = run(picks=picks)
        details = logs.getvalue()
        assert result.incident_key == summary_key(DAY, "a" * 64, bad.name)
        assert len(packets) == 1 and not rejected
        assert (TICK in packets[0]) == prior and (TICK in details) == prior
        assert json.loads(record.read_bytes())["cache_path"] == cache and bad.read_bytes() == b"{"
        print("Backtick/path replay", len(packets[0]), repr(details))
    bad.unlink()
    record.unlink()
    status.rmdir()
    os.mkfifo(status)
    with captured() as (packets, rejected, logs):
        result, = run(picks=picks, receipt=Path("fifo-receipt.json"))
        assert status.is_fifo() and result.incident_key == summary_key(DAY, status.name)
        assert len(packets) == 1 and not rejected and (TICK in logs.getvalue()) == prior
        print("Custom unreadable directory", result.message, repr(logs.getvalue()))


def timestamp_and_log_controls(prior):
    row = write_record(STATUS, timestamp=AT.replace("T", TICK),
                       cache_path="vendor\nnext/models/cache.json",
                       preserved="probable_pitcher_lookup.20261231T140000Z.bbbbbbbbbbbb.json")
    parsed = health.read_record(row)
    assert TICK in parsed["recorded_at"]
    with captured() as (packets, rejected, logs):
        result, = run()
        assert TICK in logs.getvalue() and "vendor\nnext" in logs.getvalue()
        assert len(packets) == 1 and not rejected and TICK not in packets[0]
        print("Accepted noncanonical timestamp and literal line break", repr(logs.getvalue()))


def journal_feedback(prior):
    STATUS.mkdir(parents=True)
    bad = STATUS / f"{DAY}.oom-kill.json"
    bad.write_bytes(b"{")
    picked = PICKS / str(DAY)
    picked.mkdir(parents=True)
    (picked / "decision.json").write_text(json.dumps(dict(
        schema_version=DECISION_SCHEMA, date=str(DAY), action="single",
        source="mdp", delivery_status="delivered", scoreable=True,
    )))
    (picked / "scheduler_state.json").write_text(json.dumps(dict(
        date=str(DAY), pick_locked=True, analytics_jobs={"shadow": {
            "status": "dispatched", "dispatched_at": "2026-12-31T22:45:00-04:00",
        }},
    )))
    with captured() as (_, _, journal):
        run(receipt=Path("first-receipt.json"))
        lines = [line for line in journal.getvalue().splitlines() if "status unreadable at" in line]
        assert len(lines) == 1 and str(bad) in lines[0]
    journal_text = "Dec 31 22:46:00 synthetic bts[1]: " + lines[0] + "\n"

    def capture_journal(args, **kwargs):
        assert args[:4] == ["journalctl", "--user", "-u", "bts-scheduler.service"], args
        assert kwargs["timeout"] == 5 and kwargs["capture_output"]
        return SimpleNamespace(returncode=0, stdout=journal_text)

    with captured(sources=(SOURCE, analytics_artifacts_missing.SOURCE)) as (packets, rejected, _):
        with patch.object(analytics_artifacts_missing.subprocess, "run", capture_journal):
            result = run(shadow=True, receipt=Path("replay-receipt.json"))
    critical, = [a for a in result if a.level == "CRITICAL"]
    assert critical.source == "analytics_artifacts_missing" and str(bad) in critical.message
    assert not packets and len(rejected) == 1
    assert json.loads(Path("replay-receipt.json").read_text())["status"] == "failed"
    print("W15 detail fed to later real journal classifier", repr(critical.message), "wire cap rejection", len(rejected[0]))

    with captured(sources=(SOURCE, analytics_artifacts_missing.SOURCE)) as (packets, rejected, _):
        with patch.object(analytics_artifacts_missing.subprocess, "run", capture_journal):
            journal_text = ""
            result = run(shadow=True, receipt=Path("negative-control.json"))
    assert not any(a.level == "CRITICAL" for a in result) and not rejected
    print("Identical shadow state with no journal evidence: no CRITICAL; accepted length", list(map(len, packets)))


def policy_and_bound(prior):
    genuine_warn = alert.Alert("WARN", "analytics_artifacts_missing", "Result=oom-kill")
    genuine_critical = alert.Alert("CRITICAL", "analytics_artifacts_missing", "Result=oom-kill")
    memory = alert.Alert("WARN", "memory_growth", "memory grew")
    samples = {
        "genuine warning": [genuine_warn], "warning+memory": [genuine_warn, memory],
        "genuine critical": [genuine_critical], "critical+memory": [genuine_critical, memory],
        "restart": [alert.Alert("CRITICAL", "restart_spike", "journal/OOM/watchdog")],
        "spelled out": [alert.Alert("WARN", "analytics_artifacts_missing", "Out of memory")],
        "room": [alert.Alert("WARN", "analytics_artifacts_missing", "room")], "none": [],
    }
    for root in ("data", "oom/data", "x/out-of-memory"):
        status = Path(root) / "health_state" / SOURCE
        write_record(status)
        w15, = health.check(status, DAY)
        for name, inputs in samples.items():
            baseline = attention.build_policy_alerts(inputs, streaks={})[0]
            for position, combined in (("first", [w15, *inputs]), ("last", [*inputs, w15])):
                added = attention.build_policy_alerts(combined, streaks={})[0]
                if not prior:
                    assert added == baseline
                if added != baseline:
                    print("Policy difference", root, name, position,
                          [(a.source, "quotes W15" if w15.message in a.message else a.message) for a in added],
                          "genuine-only", [a.source for a in baseline])
    if not prior:
        labels = ["status unreadable", *health.KINDS.values()]
        max_len = 0
        for size in range(1, 4):
            for kinds in itertools.combinations(labels, size):
                for count in range(1, 1001):
                    row = health.summarize(DAY, ["a"] * count, set(kinds))
                    assert not attention.OOM_EVIDENCE_RE.search(f"{row.source} {row.message}")
                    assert TICK not in row.message and len(row.message) <= 600
                    max_len = max(max_len, len(row.message))
        print("7000 fixed-summary cases", max_len, "maximum length; no OOM match or backtick")


def sequences(prior):
    CACHE.parent.mkdir(parents=True)
    raw = Path("data/raw")
    raw.mkdir()
    CACHE.write_bytes(b'{"100":null}')
    feed = raw / "2026/100.json"
    feed.parent.mkdir()
    clock = datetime.fromisoformat(AT)
    Clock.at = clock
    with patch.object(producer, "datetime", Clock):
        producer.build(CACHE, raw)
        feed.write_text('{"gameData":{"teams":{"away":{"id":51},"home":{"id":52}},"probablePitchers":{"away":{"id":1},"home":{"id":22}}}}')
        producer.build(CACHE, raw)
        CACHE.write_bytes(b'{"100":null}')
        Clock.at = clock + timedelta(minutes=2)
        producer.build(CACHE, raw)
    rows = [json.loads(path.read_bytes()) for path in sorted(STATUS.iterdir())]
    assert [row["recorded_date_et"] for row in rows] == [str(DAY), str(DAY), str(NEXT)]
    assert {row["rescanned"] for row in rows[:2]} == {0, 1}
    with captured() as (packets, rejected, logs):
        yesterday, = run()
        assert "2 observations" in yesterday.message and rows[2]["recorded_at"] not in logs.getvalue()
        assert run() == [yesterday] and len(packets) == 1
        next_day, = run(None)
        assert "3 observations" in next_day.message and rows[2]["recorded_at"] in logs.getvalue()
        assert len(packets) == 2 and not rejected
        print("Same day outcome change and late D after D+1", [a.incident_key for a in (yesterday, next_day)],
              "accepted", list(map(len, packets)))


def main():
    source = Path(sys.argv[1]).resolve()
    assert Path(bts.__file__).resolve() == source / "bts/__init__.py"
    prior = source == HERE / "r2/src"
    print("Imported", bts.__file__)
    initial = Path.cwd()
    with TemporaryDirectory(dir=HERE, prefix="health-") as work:
        for probe in (oom_paths, backticks, timestamp_and_log_controls, journal_feedback, policy_and_bound, sequences):
            folder = Path(work) / probe.__name__
            folder.mkdir()
            os.chdir(folder)
            try:
                probe(prior)
            finally:
                os.chdir(initial)


if __name__ == "__main__":
    main()
