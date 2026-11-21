"""W15b round 1 (amendment 4): no W15 text carries text read from disk or configuration.

Every W15 line that can reach the scheduler's journal or stderr (the health check's logged texts, its one
sent summary, and W15's own producer warnings) is built from fixed words, integers, re-rendered timestamps,
ISO dates and names that match the producer's own strict record and evidence patterns. So neither the
attention policy's OOM pattern nor the shadow check's journal scan can read a W15 line as OOM evidence,
and no backtick or line break from disk can reach an operator's text. The one exception is the base's
existing unreadable-cache stderr line (P2, byte for byte the base; ledgered as inherited).
"""

import json
import logging
import subprocess
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from bts.features import compute
from bts.health import analytics_artifacts_missing as shadow_check
from bts.health import attention, probable_pitcher_cache
from tests.test_wr_w15_corpus import RECOVERY, corpus_env, feed_payload  # noqa: F401 (fixture)
from tests.test_wr_w15_health import DAY, SOURCE, envelope, summary_key
from tests.test_wr_w15_incidents import AT, build, clock, freeze_clock, incident, records, seed  # noqa: F401

TICK = chr(96)
HOSTILE_NAMES = ["oom-kill", "OOM", "out of memory", f"b{TICK}t", "a\nOOM", "room", "x\rz", "tab\tx"]
GENUINE = "Nov 21 06:00:00 box systemd[1]: bts-scheduler.service: A process of this unit has been killed by the OOM killer."


def journal_scan(lines):
    """The shadow check's real journal scan over the given lines (journalctl replaced)."""
    output = "\n".join(lines) + "\n"
    original = subprocess.run

    def fake_run(args, **kwargs):
        assert args[0] == "journalctl"
        return SimpleNamespace(returncode=0, stdout=output, stderr="")

    shadow_check.subprocess.run = fake_run
    try:
        return shadow_check._fatal_scheduler_journal_line("bts-scheduler.service", "2026-12-31T10:00:00+00:00")
    finally:
        shadow_check.subprocess.run = original


def assert_safe(texts):
    assert texts
    for text in texts:
        lower = text.lower()
        assert "oom" not in lower and "out of memory" not in lower, text
        assert TICK not in text, text
        assert all(ch.isprintable() for ch in text), repr(text)
        assert not attention.OOM_EVIDENCE_RE.search(f"{SOURCE} {lower}"), text
    lines = [line for text in texts for line in text.splitlines()]
    assert journal_scan(lines) is None
    assert journal_scan([*lines, GENUINE]) == GENUINE.strip()[:240]
    assert journal_scan([GENUINE, *lines]) == GENUINE.strip()[:240]


def health_texts(caplog, alerts):
    return [r.getMessage() for r in caplog.records if r.name == probable_pitcher_cache.__name__] + [
        a.message for a in alerts]


def put(folder, row, name=None):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (name or f"{row['recorded_date_et']}.{row['reason']}.{row['identity'][:12]}.json")
    target.write_text(json.dumps(row))
    return target


@pytest.mark.parametrize("bad", HOSTILE_NAMES)
def test_badly_named_status_entry_supplies_no_journal_evidence(tmp_path, caplog, bad):
    """Codex round 1, finding 1 (RED on 284dc97a for every name containing an OOM word)."""
    caplog.set_level(logging.WARNING)
    status = tmp_path / SOURCE
    status.mkdir()
    entry = status / f"{DAY}.{bad}.json"
    entry.write_bytes(b"{")
    alerts = probable_pitcher_cache.check(status, DAY)
    texts = health_texts(caplog, alerts)
    assert_safe(texts)
    assert alerts[0].incident_key == summary_key(DAY, entry.name)
    assert entry.read_bytes() == b"{"
    assert (f"Probable-pitcher cache status unreadable: an entry for {DAY} whose name is not a status "
            "record name. Send this alert to the supervised BTS lead Claude session (bts-lead), or to Eric "
            "if the session is unavailable.") in texts


def test_validly_named_bad_record_is_named(tmp_path, caplog):
    caplog.set_level(logging.WARNING)
    status = tmp_path / SOURCE
    status.mkdir()
    name = f"{DAY}.invalid_entries.{'a' * 12}.json"
    (status / name).write_text("{")
    alerts = probable_pitcher_cache.check(status, DAY)
    assert (f"Probable-pitcher cache status unreadable: record {name}. Send this alert to the supervised BTS "
            "lead Claude session (bts-lead), or to Eric if the session is unavailable.") in health_texts(caplog, alerts)


@pytest.mark.parametrize("folder", ["oom/x", f"b{TICK}t/x", "a\nOOM/x"])
def test_unreadable_status_directory_names_no_path(tmp_path, caplog, folder):
    caplog.set_level(logging.WARNING)
    status = tmp_path / folder / SOURCE
    status.parent.mkdir(parents=True)
    status.write_text("not a directory")
    alerts = probable_pitcher_cache.check(status, DAY)
    texts = health_texts(caplog, alerts)
    assert_safe(texts)
    assert ("Probable-pitcher cache status unreadable: the status folder (health_state/probable_pitcher_cache "
            "beside the picks directory). Send this alert to the supervised BTS lead Claude session (bts-lead), "
            "or to Eric if the session is unavailable.") in texts


@pytest.mark.parametrize("root", ["x/oom", f"other{TICK}data", "a\nOOM", "out of memory"])
def test_record_cache_path_is_not_displayed(tmp_path, caplog, root):
    caplog.set_level(logging.WARNING)
    status = tmp_path / root / "health_state" / SOURCE
    _, row = envelope()
    row["cache_path"] = f"{tmp_path / root}/models/probable_pitcher_lookup.json"
    row["preserved"] = "probable_pitcher_lookup.20261231T140000Z.cccccccccccc.json"
    target = put(status, row)
    alerts = probable_pitcher_cache.check(status, DAY)
    texts = health_texts(caplog, alerts)
    assert_safe(texts)
    assert any("That run recorded an evidence copy named probable_pitcher_lookup.20261231T140000Z.cccccccccccc.json "
               "in the rejected folder beside the cache." in text for text in texts)
    assert json.loads(target.read_text())["cache_path"] == row["cache_path"]


def test_timestamp_is_rendered_from_its_parsed_value(tmp_path, caplog):
    """Codex round 1, finding 2: a backtick separator in recorded_at (accepted by fromisoformat)."""
    caplog.set_level(logging.WARNING)
    status = tmp_path / SOURCE
    _, row = envelope()
    row["recorded_at"] = f"{DAY}{TICK}14:00:00+00:00"
    put(status, row)
    alerts = probable_pitcher_cache.check(status, DAY)
    texts = health_texts(caplog, alerts)
    assert_safe(texts)
    assert any(text.startswith(f"Probable-pitcher cache at {DAY}T14:00:00+00:00 had 1 invalid entry") for text in texts)


def producer_root(tmp_path, monkeypatch, name):
    root = tmp_path / name
    cache = root / "data/models/probable_pitcher_lookup.json"
    cache.parent.mkdir(parents=True)
    (root / "data/raw/2025").mkdir(parents=True)
    monkeypatch.chdir(root)
    monkeypatch.setattr(compute, "_probable_pitcher_cache_path", cache)
    freeze_clock(monkeypatch, AT)
    return root, cache


def stderr_lines(capsys):
    return [line for line in capsys.readouterr().err.splitlines() if line]


@pytest.mark.parametrize("name", ["oom", f"w{TICK}d", "out of memory", "a\nOOM"])
def test_producer_dropped_line_names_no_path(tmp_path, monkeypatch, capsys, name):
    root, cache = producer_root(tmp_path, monkeypatch, name)
    (root / "data/raw/2025/100.json").write_text(feed_payload(RECOVERY))
    seed(cache, {"100": None})
    build()
    lines = stderr_lines(capsys)
    assert lines == [
        "WARNING: probable-pitcher cache: dropped 1 invalid entry (0 recovered from valid cache entries, "
        "1 rescanned from raw feeds, 0 with unusable feeds, 0 without feeds); first game ids: 100"
    ]
    assert_safe(lines)
    row, = records(cache)
    assert row["cache_path"] == str(cache)


@pytest.mark.parametrize("name", ["oom", f"w{TICK}d"])
def test_producer_failure_lines_name_no_path(tmp_path, monkeypatch, capsys, name):
    root, cache = producer_root(tmp_path, monkeypatch, name)
    (root / "data/raw/2025/100.json").write_text(feed_payload(RECOVERY))
    seed(cache, {"100": None})
    (cache.parent / "rejected").write_text("not a directory")
    (root / "data/health_state").mkdir(parents=True)
    (root / "data/health_state/probable_pitcher_cache").write_text("not a directory")
    build()
    lines = stderr_lines(capsys)
    assert len(lines) == 3, lines
    assert lines[1] == "WARNING: could not preserve probable-pitcher cache evidence (NotADirectoryError)"
    assert lines[2].startswith("WARNING: could not write probable-pitcher cache status record 2026-12-31.invalid_entries.")
    assert lines[2].endswith(".json (NotADirectoryError)")
    assert_safe(lines)


def test_stream_without_errors_attribute_still_gets_the_line(tmp_path, monkeypatch):
    """Codex round 1 note: a descriptor proxy with no `errors` attribute suppressed the line."""
    root, cache = producer_root(tmp_path, monkeypatch, "plain")
    seed(cache, {"100": None})
    reader = tmp_path / "stderr.txt"

    class Proxy:
        encoding = "utf-8"

        def __init__(self, handle):
            self.handle = handle

        def fileno(self):
            return self.handle.fileno()

        def flush(self):
            self.handle.flush()

        def write(self, text):
            return self.handle.write(text)

    with reader.open("w") as handle:
        monkeypatch.setattr("sys.stderr", Proxy(handle))
        build()
    assert "WARNING: probable-pitcher cache: dropped 1 invalid entry" in reader.read_text()


def test_genuine_journal_evidence_still_fires_beside_every_w15_text(tmp_path, caplog):
    """Must still catch: the scan and the policy are unchanged, so genuine evidence wins wherever W15 lines sit."""
    caplog.set_level(logging.WARNING)
    status = tmp_path / "oom" / SOURCE
    _, row = envelope()
    row["cache_path"] = f"{tmp_path}/oom/models/probable_pitcher_lookup.json"
    put(status, row)
    (status / f"{DAY}.oom-kill.json").write_text("{")
    alerts = probable_pitcher_cache.check(status, DAY)
    assert_safe(health_texts(caplog, alerts))
    for genuine in ("kernel: Out of memory: Killed process 123 (python)",
                    "bts-scheduler.service: Main process exited, code=killed, status=9/KILL",
                    "bts-scheduler.service: Failed with result 'oom-kill'.",
                    "worker exited status=137"):
        lines = [*health_texts(caplog, alerts), genuine]
        assert journal_scan(lines) == genuine.strip()[:240]
