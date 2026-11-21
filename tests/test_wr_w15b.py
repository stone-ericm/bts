"""W15b: no W15 text supplies OOM evidence, and displayed disk paths carry no backticks.

Amendment 3 of kickoff-fix-w15.md (the two round-2 findings of STOP-W15).
"""

import itertools
import json
import os
import subprocess
import sys
from datetime import date
from pathlib import Path

import pytest

from bts.health import alert, attention, probable_pitcher_cache, runner
from bts.health.alert import Alert
from tests.test_wr_w15_corpus import RECOVERY, corpus_env, feed_payload  # noqa: F401 (fixture)
from tests.test_wr_w15_health import DAY, SOURCE, envelope, summary_key
from tests.test_wr_w15_incidents import AT, build, child_env, clock, freeze_clock, incident, records, seed  # noqa: F401

TICK = chr(96)
KIND_LABELS = ("invalid entries", "status unreadable", "unreadable cache")
SUMMARY_TAIL = (
    "Keep the active cache, raw feeds, evidence copies and other copies intact. "
    "Send this alert promptly to the supervised BTS lead Claude session (bts-lead), or to Eric if it is unavailable, "
    "to review the latest preview and any pending pick before the next selection; for an unreadable cache, "
    "to secure a usable R2 copy before the next successful sync."
)
DETAILS = (
    "details in this health run's log and in health_state/probable_pitcher_cache beside the picks directory. "
)


def expected_summary(count, labels):
    noun = "observation" if count == 1 else "observations"
    return (
        f"Probable-pitcher cache: {count} {noun} ({', '.join(sorted(labels))}) recorded today or yesterday; "
        + DETAILS + SUMMARY_TAIL
    )


@pytest.fixture
def health_at(tmp_path, monkeypatch):
    """The W15 health fixture, with the data directory chosen by the test."""
    sent = []
    monkeypatch.setattr(runner.realized_calibration, "_current_deploy_iso", lambda _root: None)
    safe_run = runner._safe_run
    monkeypatch.setattr(
        runner, "_safe_run",
        lambda source, check: safe_run(source, check) if source == SOURCE else [],
    )

    def send(recipient, message):
        sent.append(message)
        return "fake-delivery"

    monkeypatch.setattr(alert, "send_dm", send)
    freeze_clock(monkeypatch, "2027-01-01T05:01:00+00:00", module=runner)
    freeze_clock(monkeypatch, "2027-01-01T05:01:00+00:00", module=alert)

    def run(data, day=DAY):
        return runner.run_all_checks(data / "picks", data / "models", "offline", today=day)

    return run, sent


def put_row(folder, row, name=None):
    folder.mkdir(parents=True, exist_ok=True)
    target = folder / (name or f"{row['recorded_date_et']}.{row['reason']}.{row['identity'][:12]}.json")
    target.write_text(json.dumps(row))
    return target


# --- (a) no W15 text supplies OOM evidence -------------------------------------------------


@pytest.mark.parametrize("parts", [("oom", "data"), ("x", "OOM-killed", "data"), ("out of memory", "data"),
                                   ("x", "oom_kill", "data")])
def test_oom_named_data_directory_gives_no_oom_critical(tmp_path, health_at, parts):
    """The round-2 reviewer's input: RED on d7c66234 (an analytics_job_oom CRITICAL copying the summary)."""
    run, sent = health_at
    data = tmp_path.joinpath(*parts)
    _, row = envelope()
    put_row(data / "health_state" / SOURCE, row)
    result = run(data)
    assert [(a.level, a.source) for a in result] == [("WARN", SOURCE)]
    assert result[0].message == expected_summary(1, {"invalid entries"})
    assert len(sent) == 1 and len(sent[0]) < 1000


@pytest.mark.parametrize("folder", ["probable_pitcher_cache", f"a{TICK}b/oom/{'z' * 200}/{'y' * 200}/{'x' * 200}/probable_pitcher_cache"])
def test_summary_names_no_path(tmp_path, folder):
    status = tmp_path / folder
    _, row = envelope()
    put_row(status, row)
    result, = probable_pitcher_cache.check(status, DAY)
    assert result.message == expected_summary(1, {"invalid entries"})
    assert str(tmp_path) not in result.message
    assert result.incident_key == summary_key(DAY, row["identity"])


def test_unreadable_status_directory_summary_names_no_path(tmp_path):
    status = tmp_path / "oom" / SOURCE
    status.parent.mkdir()
    status.write_text("not a directory")
    result, = probable_pitcher_cache.check(status, DAY)
    assert result.message == expected_summary(1, {"status unreadable"})
    assert result.incident_key == summary_key(DAY, status.name)


def test_every_summary_is_oom_free_backtick_free_and_bounded():
    """The summary's only variable parts are a count and a set of fixed labels: check them all."""
    for size in range(1, len(KIND_LABELS) + 1):
        for labels in itertools.combinations(KIND_LABELS, size):
            for count in range(1, 1001):
                identities = [f"{n:064x}" for n in range(count)]
                text = probable_pitcher_cache.summarize(DAY, identities, set(labels)).message
                assert text == expected_summary(count, labels)
                assert not attention.OOM_EVIDENCE_RE.search(f"{SOURCE} {text}".lower())
                assert TICK not in text
                assert len(text) <= 600


GENUINE = {
    "warn_alone": [Alert("WARN", "analytics_artifacts_missing", "shadow missing Result=oom-kill")],
    "critical_alone": [Alert("CRITICAL", "analytics_artifacts_missing", "capture Result=oom-kill")],
    "memory_and_warn": [Alert("WARN", "memory_growth", "scheduler RSS 3350 MB"),
                        Alert("WARN", "analytics_artifacts_missing", "shadow missing Result=oom-kill")],
    "memory_and_critical": [Alert("WARN", "memory_growth", "scheduler RSS 3350 MB"),
                            Alert("CRITICAL", "analytics_artifacts_missing", "capture Result=oom-kill")],
    "out_of_memory_text": [Alert("WARN", "analytics_artifacts_missing", "worker: Out of memory")],
    "restart_spike": [Alert("CRITICAL", "restart_spike", "NRestarts spiked; inspect journal/OOM/watchdog/crash evidence.")],
    "innocent": [Alert("WARN", "analytics_artifacts_missing", "shadow worker room unavailable")],
    "none": [],
}


def policy(alerts):
    return [(a.level, a.source, a.message) for a in attention.build_policy_alerts(alerts, streaks={})[0]]


@pytest.mark.parametrize("name", sorted(GENUINE))
@pytest.mark.parametrize("where", ["first", "last"])
@pytest.mark.parametrize("folder", ["data/health_state", "oom/data/health_state", "x/out-of-memory/health_state"])
def test_must_still_catch_policy_is_unchanged_by_a_w15_summary(tmp_path, monkeypatch, name, where, folder):
    """Every genuine OOM input gives the same policy CRITICALs with a W15 summary present, wherever
    the status directory is; a W15 summary alone gives none. RED on d7c66234 for the OOM-named folders
    (a short relative path, so d7c66234's over-600 fallback does not hide the path)."""
    monkeypatch.chdir(tmp_path)
    status = Path(folder) / SOURCE
    _, row = envelope()
    put_row(status, row)
    summary, = probable_pitcher_cache.check(status, DAY)
    genuine = GENUINE[name]
    mixed = [summary, *genuine] if where == "first" else [*genuine, summary]
    assert policy(mixed) == policy(genuine)


# --- (b) backticks: superseded by tests/test_wr_w15b_r1.py (amendment 4: nothing from disk is displayed) ---


def test_unreadable_cache_line_keeps_the_base_text(tmp_path, monkeypatch, capsys):
    """P2: the existing unreadable diagnostic is byte for byte the base's (the one documented exception)."""
    root = tmp_path / f"w{TICK}d"
    cache = root / "data/models/probable_pitcher_lookup.json"
    cache.parent.mkdir(parents=True)
    (root / "data/raw").mkdir(parents=True)
    monkeypatch.chdir(root)
    from bts.features import compute
    monkeypatch.setattr(compute, "_probable_pitcher_cache_path", cache)
    freeze_clock(monkeypatch, AT)
    cache.write_text("{")
    build()
    assert capsys.readouterr().err == (
        f"WARNING: unreadable probable-pitcher cache {cache} (JSONDecodeError); rebuilding it from raw feeds\n"
    )


# --- STOP-W15 note: the stream's own error handler -----------------------------------------


def test_ascii_stderr_keeps_backslash_spelling_of_a_non_ascii_path(tmp_path):
    """Round 1 and the base printed caf\\xe9 on an ASCII stderr; d7c66234 printed caf? (RED there)."""
    cache = tmp_path / "data/models/café.json"
    cache.parent.mkdir(parents=True)
    (tmp_path / "data/raw").mkdir(parents=True)
    cache.write_text("{")
    command = (
        "import sys; from pathlib import Path; from bts.features import compute; "
        "compute._probable_pitcher_cache_path = Path(sys.argv[1]); "
        "compute._build_probable_pitcher_lookup('data/raw')"
    )
    child = subprocess.run(
        [sys.executable, "-c", command, str(cache)], cwd=tmp_path,
        env={**child_env(), "PYTHONIOENCODING": "ascii"}, capture_output=True, timeout=30,
    )
    assert child.returncode == 0, child.stderr
    assert b"caf\\xe9.json" in child.stderr and b"caf?.json" not in child.stderr
