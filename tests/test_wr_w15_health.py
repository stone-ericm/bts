"""Dated observation discovery and delivery through the health runner."""

import hashlib
import json
import os
import subprocess
import sys
from datetime import date, timedelta
from pathlib import Path

import pytest

from bts.health import alert, probable_pitcher_cache, runner
from tests.test_wr_w15_corpus import corpus_env, RECOVERY, feed_payload
from tests.test_wr_w15_incidents import AT, build, child_env, clock, freeze_clock, incident, records, seed


DAY = date(2026, 12, 31)
SOURCE = "probable_pitcher_cache"


@pytest.fixture
def health(tmp_path, monkeypatch):
    sent = []
    monkeypatch.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _root: None)
    safe_run = runner._safe_run
    monkeypatch.setattr(
        runner, "_safe_run",
        lambda source, check: safe_run(source, check) if source == SOURCE else [],
    )

    def send(recipient, message):
        assert recipient == "offline"
        sent.append(message)
        return "fake-delivery"

    monkeypatch.setattr(alert, "send_dm", send)
    freeze_clock(monkeypatch, "2027-01-01T05:01:00+00:00", module=runner)
    freeze_clock(monkeypatch, "2027-01-01T05:01:00+00:00", module=alert)
    folder = tmp_path / "data/health_state" / SOURCE

    def run(day=DAY):
        return runner.run_all_checks(
            tmp_path / "data/picks", tmp_path / "data/models", "offline", today=day,
        )

    return folder, run, sent


def envelope(day=DAY, reason="invalid_entries"):
    identity = "a" * 64 if reason == "invalid_entries" else "b" * 64
    invalid = reason == "invalid_entries"
    row = dict(
        schema_version="bts_probable_pitcher_cache_status_v1",
        reason=reason, recorded_at=f"{day}T14:00:00+00:00", recorded_date_et=str(day),
        cache_path="data/models/probable_pitcher_lookup.json",
        dropped=1 if invalid else None, recovered_from_cache=0 if invalid else None,
        rescanned=0, unusable_feed=0 if invalid else None, without_feed=1 if invalid else None,
        game_ids=[100] if invalid else [], identity=identity, preserved=None, original_sha256="c" * 64,
    )
    return f"{day}.{reason}.{identity[:12]}.json", row


def put(folder, day=DAY, reason="invalid_entries"):
    folder.mkdir(parents=True, exist_ok=True)
    name, row = envelope(day, reason)
    target = folder / name
    target.write_text(json.dumps(row))
    return target


def summary_key(day, *identities):
    payload = json.dumps(sorted(identities), separators=(",", ":")).encode()
    return f"{SOURCE}:{day}:{hashlib.sha256(payload).hexdigest()[:16]}"


def unreadable_summary(result, name, day=DAY):
    assert result.level == "WARN"
    assert " (status unreadable) recorded today or yesterday;" in result.message
    assert result.incident_key == summary_key(day, name)


def test_missing_status_is_silent(health):
    _, run, sent = health
    assert run() == []
    assert sent == []


def test_nonmatching_status_names_are_ignored(health):
    folder, run, sent = health
    folder.mkdir(parents=True)
    for name in (f"{DAY}.json", f"{DAY}.anything.txt", "old.json"):
        (folder / name).write_text("{")
    assert run() == []
    assert sent == []


def test_late_d_health_can_read_d_after_d_plus_one(health):
    folder, run, sent = health
    for offset in (-2, -1, 0, 1):
        put(folder, DAY + timedelta(days=offset))
    result, = run()
    assert result.incident_key == summary_key(DAY, "a" * 64, "a" * 64)
    assert "2 observations" in result.message
    assert len(sent) == 1


def test_attention_delivery_both_reasons_and_processed_day_dedup(health):
    folder, run, sent = health
    put(folder)
    put(folder, reason="unreadable_cache")
    for day in [DAY, DAY + timedelta(days=1)]:
        before = len(sent)
        result, = run(day)
        assert result.incident_key == summary_key(day, "a" * 64, "b" * 64)
        assert len(result.message) <= 600
        assert len(run(day)) == 1
        assert len(sent) == before + 1
    assert all("2 observations (invalid entries, unreadable cache)" in body for body in sent)
    assert all(len(body) < 1000 and "That run left the active cache unchanged." not in body for body in sent)


def test_operator_text_describes_the_observation_and_recovery_route(health, caplog):
    folder, run, _ = health
    path = put(folder)
    row = json.loads(path.read_text())
    row.update(
        dropped=4, recovered_from_cache=1, rescanned=1, unusable_feed=1, without_feed=1,
        preserved="probable_pitcher_lookup.20261231T140000Z.cccccccccccc.json",
    )
    path.write_text(json.dumps(row))
    invalid, = run()
    assert invalid.message.startswith("Probable-pitcher cache: 1 observation (invalid entries)")
    assert (
        "Probable-pitcher cache at 2026-12-31T14:00:00+00:00 had 4 invalid entries (first game IDs 100). "
        "Of those entries, 1 had a valid cache alias and 1 had a usable raw feed. "
        "No usable lookup was found for 1 with unusable or unverified feeds and 1 with missing feeds. "
        "That run recorded an evidence copy at "
        "data/models/rejected/probable_pitcher_lookup.20261231T140000Z.cccccccccccc.json. "
        "This describes the cache at the recorded time; health may show this observation again on the next processed day. "
        "Keep the active cache, raw feeds, any named evidence copy and other available copies intact for review. "
        "Send this alert promptly to the supervised BTS lead Claude session (bts-lead), or to Eric if the session is unavailable. "
        "Ask them to review the latest preview and any pending pick before the next scheduled selection. "
        "If a pick was already sent or entered, report that and preserve its records."
    ) in [record.getMessage() for record in caplog.records]
    path.unlink()
    path = put(folder, reason="unreadable_cache")
    row = json.loads(path.read_text())
    row["rescanned"] = 1
    path.write_text(json.dumps(row))
    unreadable, = run()
    assert unreadable.message.startswith("Probable-pitcher cache: 1 observation (unreadable cache)")
    assert (
        "Probable-pitcher cache was unreadable at 2026-12-31T14:00:00+00:00. "
        "That run started from an empty lookup and built usable entries from raw feeds for 1 game. "
        "Keep the active cache, raw feeds, any named evidence copy and other available copies intact for review. "
        "Other games may have been missing from that run's lookup. "
        "This observation names no evidence copy. That run left the active cache unchanged. "
        "Send this alert immediately to the supervised BTS lead Claude session (bts-lead), or to Eric if the session is unavailable. "
        "Ask them to preserve and verify any usable cache copy in the current or previous R2 manifest before the next successful sync. "
        "An unreferenced content-addressed cache object can be pruned once it is at least seven days old."
    ) in [record.getMessage() for record in caplog.records]
    assert all(chr(96) not in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("field,value", [
    ("schema_version", "other"), ("reason", "unknown"), ("recorded_at", "2027-01-01T05:01:00+00:00"),
    ("recorded_at", "2026-12-31T15:00:00+01:00"), ("recorded_date_et", "2027-01-01"),
    ("cache_path", ""), ("dropped", True), ("dropped", 0), ("recovered_from_cache", -1),
    ("rescanned", "0"), ("unusable_feed", 0.0), ("without_feed", 0),
    ("game_ids", []), ("game_ids", [True]), ("game_ids", [101, 100]),
    ("game_ids", [100, 100]), ("game_ids", [100, 101]),
    ("identity", "A" * 64), ("identity", "b" * 64),
    ("preserved", "../probable_pitcher_lookup.20270101T045900Z.0123456789ab.json"),
    ("original_sha256", "bad"),
])
def test_malformed_fields_warn(health, field, value, caplog):
    folder, run, _ = health
    path = put(folder)
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    result, = run()
    unreadable_summary(result, path.name)
    assert (
        f"Probable-pitcher cache status unreadable at {path}. "
        "Send this alert to the supervised BTS lead Claude session (bts-lead), "
        "or to Eric if the session is unavailable."
    ) in [record.getMessage() for record in caplog.records]


@pytest.mark.parametrize("field,value", [
    ("dropped", 0), ("recovered_from_cache", 0), ("rescanned", None),
    ("unusable_feed", 0), ("without_feed", 0), ("game_ids", [100]),
])
def test_malformed_unreadable_envelope_warns(health, field, value):
    folder, run, _ = health
    path = put(folder, reason="unreadable_cache")
    row = json.loads(path.read_text())
    row[field] = value
    path.write_text(json.dumps(row))
    result, = run()
    unreadable_summary(result, path.name)


@pytest.mark.parametrize("kind", ["encoding", "oversized", "directory", "symlink", "bad-name"])
def test_malformed_status_files_warn(health, kind):
    folder, run, _ = health
    path = put(folder)
    if kind == "encoding":
        path.write_bytes(b"\xff")
    elif kind == "oversized":
        path.write_bytes(b" " * (64 * 1024 + 1))
    elif kind == "directory":
        path.unlink()
        path.mkdir()
    elif kind == "symlink":
        target = path.with_suffix(".original")
        path.rename(target)
        path.symlink_to(target)
    else:
        path = path.rename(folder / f"{DAY}.bad.json")
    result, = run()
    unreadable_summary(result, path.name)


@pytest.mark.parametrize("kind", ["file", "symlink", "fifo"])
def test_bad_status_directory_warns(health, kind):
    folder, run, _ = health
    folder.parent.mkdir(parents=True)
    if kind == "file":
        folder.write_text("{}")
    elif kind == "symlink":
        target = folder.with_name("other")
        target.mkdir()
        folder.symlink_to(target)
    else:
        os.mkfifo(folder)
    result, = run()
    unreadable_summary(result, folder.name)


def test_fifo_status_file_is_bounded(health):
    folder, _, _ = health
    path = put(folder)
    path.unlink()
    os.mkfifo(path)
    script = (
        "import json, sys, pytest; from datetime import date; from pathlib import Path; "
        "from tests.test_wr_w15_health import health; "
        "_, run, _ = health.__wrapped__(Path(sys.argv[1]), pytest.MonkeyPatch()); "
        "print(json.dumps([a.incident_key for a in run(date(2026,12,31))]))"
    )
    result = subprocess.run(
        [sys.executable, "-c", script, str(folder.parents[2])],
        env=child_env(),
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == [summary_key(DAY, path.name)]


def test_same_day_changed_incident_remains_visible(health, incident, caplog):
    _, run, sent = health
    cache, raw = incident
    seed(cache, {"100": None})
    assert build() == {}
    run()
    assert len(sent) == 1
    raw.write_text(feed_payload(RECOVERY))
    assert build() == {100: RECOVERY}
    rows = records(cache)
    assert len(rows) == 2
    assert {row["recorded_at"] for row in rows} == {AT}
    result, = run()
    assert len(sent) == 2
    assert result.incident_key == summary_key(DAY, *(row["identity"] for row in rows))
    assert "2 observations (invalid entries)" in sent[-1]
    assert "That run recorded an evidence copy at" in caplog.text
    assert "This observation names no evidence copy." in caplog.text
    run()
    assert len(sent) == 2


def test_producer_midnight_and_year_boundary(health, incident, monkeypatch):
    _, run, sent = health
    cache, _ = incident
    seed(cache, {"100": None})
    build()
    freeze_clock(monkeypatch, "2027-01-01T05:01:00+00:00")
    build()
    rows = records(cache)
    assert [(row["recorded_at"], row["recorded_date_et"]) for row in rows] == [
        (AT, "2026-12-31"), ("2027-01-01T05:01:00+00:00", "2027-01-01"),
    ]
    first, = run(DAY)
    second, = run(DAY + timedelta(days=1))
    assert "1 observation (invalid entries)" in first.message
    assert "2 observations (invalid entries)" in second.message
    assert len(sent) == 2 and all(len(body) < 1000 for body in sent)

@pytest.mark.parametrize("long_path", [False, True])
def test_sixteen_records_send_one_bounded_warning(tmp_path, monkeypatch, long_path):
    monkeypatch.chdir(tmp_path)
    folder = Path(SOURCE)
    if long_path:
        folder = Path("x" * 130) / ("y" * 130) / SOURCE
    folder.mkdir(parents=True)
    identities = []
    for number in range(15):
        _, row = envelope(reason="invalid_entries" if number % 2 else "unreadable_cache")
        identity = f"{number:012x}" + "a" * 52
        identities.append(identity)
        row["identity"] = identity
        (folder / f"{DAY}.{row['reason']}.{identity[:12]}.json").write_text(json.dumps(row))
    bad = f"{DAY}.bad.json"
    (folder / bad).write_text("{")
    result, = probable_pitcher_cache.check(folder, DAY)
    # W15b (amendment 3): the summary names no path, long or short.
    assert result.message == (
        "Probable-pitcher cache: 16 observations (invalid entries, status unreadable, unreadable cache) "
        "recorded today or yesterday; details in this health run's log and in "
        "health_state/probable_pitcher_cache beside the picks directory. "
        "Keep the active cache, raw feeds, evidence copies and other copies intact. "
        "Send this alert promptly to the supervised BTS lead Claude session (bts-lead), or to Eric if it is unavailable, "
        "to review the latest preview and any pending pick before the next selection; for an unreadable cache, "
        "to secure a usable R2 copy before the next successful sync."
    )
    assert result.incident_key == summary_key(DAY, *identities, bad)
    assert len(result.message) <= 600
    body = alert.format_health_dm_body([], [result])
    assert len(body) < 1000 and chr(96) not in body
