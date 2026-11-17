"""CS4b-2: required reads, operator recovery, and ordered quarantine.

Synthetic files and captured fetch transports only. Each operation pins its clock.
"""
import datetime as dt
import json
import os
import re
import shlex
from dataclasses import asdict
from pathlib import Path
from unittest.mock import Mock

import pytest
from click.testing import CliRunner

import bts.cli as cli
import bts.contest_state as cs
import bts.season_contract as sc
import bts.util as util
from bts.health import contest_state as health
from bts.saver_state import load_saver_state
from tests.test_season_contract_cs4b1 import (
    ID, NOW, ET, REAL_DATETIME, contract, fetch, files, offline, pin, put, run, tmp_path,
)


def automatic(**kw):
    return dict(schema_version=sc.AUTO_SCHEMA, season=2027, activation_id=ID,
                active_streak=12, best_streak=14, source="mlb_bts_profile",
                source_date="2027-07-14", recorded_at="2027-07-15T16:00:00Z") | kw


def dated(day="2027-07-15", **kw):
    d = dt.date.fromisoformat(day)
    return dict(schema_version=cs.DATED_MANUAL_SCHEMA, effective_date=day,
                season=d.year, active_streak=12, best_streak=14, saver_state="used",
                source="manual_cli", recorded_at=NOW.isoformat(),
                expires_at=REAL_DATETIME.combine(d, dt.time.max, ET).isoformat()) | kw


SWITCHES = {"absent": None, "automatic": contract(),
            "revoked": contract(status="revoked", revoked_at=NOW.isoformat()),
            "other_season": contract(season=2026)}
for n, value in enumerate([[], contract(schema_version=None), contract(season=True),
                          contract(status="bad"), contract(activation_id="bad"),
                          contract(switched_at=None), contract(revoked_at="bad"),
                          contract(note=[])], 1):
    SWITCHES[f"corrupt{n}"] = value
MANUALS = {"absent": None, "legacy": dict(active_streak=5, best_streak=9,
            override_expires_at="2027-07-16T16:00:00Z"),
           "expired": dict(active_streak=5, override_expires_at="2027-07-14T16:00:00Z"),
           "undated": dict(active_streak=5), "today": dated(),
           "yesterday": dated("2027-07-14"), "tomorrow": dated("2027-07-16"),
           "invalid": dated(saver_state="bad"), "unreadable": b"\xff"}
AUTOS = {"absent": None, "covered": automatic(),
         "v1": automatic(schema_version="bts_contest_streak_auto_v1"),
         "missing_schema": {k: v for k, v in automatic().items() if k != "schema_version"},
         "null_schema": automatic(schema_version=None), "number_schema": automatic(schema_version=3),
         "unknown_schema": automatic(schema_version="unknown"),
         "other_season": automatic(season=2026), "other_id": automatic(activation_id="b" * 32),
         "refused": automatic(active_streak=58, best_streak=58), "corrupt": b"\xff",
         "undated": automatic(source_date=None, recorded_at=None),
         "timestamp_only": automatic(source_date=None)}


def install(picks, name, value):
    path = picks / "account_state" / name
    path.unlink(missing_ok=True)
    if value is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())


def reason(selected, sw, now=NOW):
    """Contract oracle, independent of production refusal formatting."""
    if sw.state == "corrupt":
        return "corrupt"
    if selected is not None and selected.path.name == "contest_streak.manual.json":
        return "legacy"
    if sw.state in {"absent", "revoked", "other_season"}:
        return sw.state
    if selected is None:
        return "no automatic observation"
    if selected.schema_version != sc.AUTO_SCHEMA:
        return "schema_version"
    if selected.season != now.astimezone(ET).year:
        return "season"
    if selected.activation_id != sw.snapshot["activation_id"]:
        return "activation_id"
    return None


def live(picks, now=NOW):
    return cs.load_decision_streak_state(picks, now=now, require_contest_state=True)


@pytest.mark.parametrize("switch", SWITCHES)
@pytest.mark.parametrize("manual", MANUALS)
@pytest.mark.parametrize("auto", AUTOS)
def test_selection_matrix(tmp_path, monkeypatch, switch, manual, auto):
    install(tmp_path, "season_contract.json", SWITCHES[switch])
    install(tmp_path, "contest_streak.manual.json", MANUALS[manual])
    install(tmp_path, "contest_streak.json", AUTOS[auto])
    (tmp_path / "streak.json").write_bytes(b"\xff")
    sw = sc.switch_state(tmp_path, NOW)
    reads = []
    original = sc.switch_state
    def switching(picks, now, **kw):
        reads.append((picks, now))
        return original(picks, now, **kw)
    monkeypatch.setattr(sc, "switch_state", switching)
    try:
        selected = cs.load_contest_streak_state(tmp_path, now=NOW, require_dated_auto=True)
    except cs.ContestStateError as exc:
        with pytest.raises(cs.ContestStateError) as actual:
            live(tmp_path)
        assert str(actual.value) == str(exc)
        assert [a.message for a in health.check(tmp_path, now=NOW, expected=True)] == [str(exc)]
        assert reads == []
        return
    bypass = selected is not None and cs.dated_observation_in_effect(selected, NOW)
    expected = None if bypass else reason(selected, sw)
    if expected:
        with pytest.raises(cs.ContestStateError) as exc:
            live(tmp_path)
        assert expected in str(exc.value).split("\n", 1)[0]
        alerts = health.check(tmp_path, now=NOW, expected=True)
        assert alerts[-1].level == "CRITICAL" and alerts[-1].source == "contest_state"
        assert alerts[-1].message == str(exc.value)
        assert sum(a.message == str(exc.value) for a in alerts) == 1
        if selected is None:
            assert len(alerts) == 2 and "expected but missing" in alerts[0].message
    else:
        assert live(tmp_path).streak == selected.streak
        health.check(tmp_path, now=NOW, expected=True)
    assert reads == ([] if bypass else [(tmp_path, NOW), (tmp_path, NOW)])


@pytest.mark.parametrize("switch", SWITCHES)
@pytest.mark.parametrize("kind", ["none", "legacy", "v1", "covered", "schema_season_id", "season_id", "id"])
def test_refusal_one_snapshot_no_observation_io(tmp_path, monkeypatch, switch, kind):
    sw = sc.switch_state(tmp_path, NOW)  # no file reads allowed after selecting
    install(tmp_path, "season_contract.json", SWITCHES[switch])
    sw = sc.switch_state(tmp_path, NOW)
    selected = None
    if kind != "none":
        payload = automatic()
        if kind in {"v1", "schema_season_id"}:
            payload.update(schema_version="not v2", season=2026, activation_id="b" * 32)
        elif kind == "season_id":
            payload.update(season=2026, activation_id="b" * 32)
        elif kind == "id":
            payload["activation_id"] = "b" * 32
        path = put(tmp_path, "contest_streak.manual.json" if kind == "legacy" else "contest_streak.json", payload)
        selected = cs._parse_state_file(path)
    spy = Mock(return_value=sw)
    monkeypatch.setattr(sc, "switch_state", spy)
    monkeypatch.setattr(Path, "open", Mock(side_effect=AssertionError("refusal reopened observation")))
    message = sc.live_automatic_refusal(tmp_path, NOW, selected)
    expected = reason(selected, sw)
    assert (message is None) == (expected is None)
    if expected:
        assert expected in message.split("\n", 1)[0]
    spy.assert_called_once_with(tmp_path, NOW)


@pytest.mark.parametrize("required", [False, True])
@pytest.mark.parametrize("today", [False, True])
def test_optional_and_dated_bypass_corrupt_switch(tmp_path, monkeypatch, required, today):
    put(tmp_path, "contest_streak.json", automatic())
    put(tmp_path, "season_contract.json", [])
    if today:
        put(tmp_path, "contest_streak.manual.json", dated())
    spy = Mock(side_effect=AssertionError("switch must not be read"))
    if required and not today:
        return  # the matrix asserts this required non-dated refusal and its read
    monkeypatch.setattr(sc, "switch_state", spy)
    assert cs.load_decision_streak_state(tmp_path, now=NOW, require_contest_state=required).streak == 12
    health.check(tmp_path, now=NOW, expected=required)
    spy.assert_not_called()


@pytest.mark.parametrize("schema", [None, 3, [], "other"])
def test_schema_diagnostics(tmp_path, schema):
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.json", automatic(schema_version=schema))
    with pytest.raises(cs.ContestStateError) as exc:
        live(tmp_path)
    message = str(exc.value)
    assert (repr(schema) if isinstance(schema, str) else "schema_version is missing, null or not a string") in message
    if not isinstance(schema, str):
        assert "status" in message and "exact value" in message


def test_exact_alert_order_and_post_loader_exception(tmp_path):
    put(tmp_path, "contest_streak.manual.json", MANUALS["expired"])
    put(tmp_path, "contest_streak.json", automatic())
    for day in ["2027-07-15", "2027-07-16"]:
        (tmp_path / f"{day}.json").write_text('{"result":"hit"}')
    alerts = health.check(tmp_path, now=NOW, expected=True)
    assert [(a.level, a.source) for a in alerts] == [
        ("WARN", "contest_state"), ("WARN", "contest_state"),
        ("WARN", "saver_state"), ("CRITICAL", "contest_state")]
    assert "coverage" in alerts[0].message and "legacy" in alerts[1].message
    put(tmp_path, "season_contract.json", contract())
    (tmp_path / "account_state/saver_state.json").write_text('{"n":' + '9' * 5000 + '}')
    with pytest.raises(ValueError):
        health.check(tmp_path, now=NOW, expected=True)
    assert "live decision now: unavailable" in run(tmp_path, "season-contract", "status").output


@pytest.mark.parametrize("mode", ["accepted", "refused", "unavailable"])
def test_status_real_decision_and_read_only(tmp_path, monkeypatch, mode):
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.json", automatic(activation_id="b" * 32 if mode == "refused" else ID))
    if mode == "unavailable":
        (tmp_path / "account_state/saver_state.json").write_text('{"n":' + '9' * 5000 + '}')
    before = files(tmp_path.parent)
    opener = Path.open
    writes = []
    def opening(path, mode="r", *a, **kw):
        if any(flag in mode for flag in "wax+"):
            writes.append((path, mode))
            raise AssertionError("status opened for writing")
        return opener(path, mode, *a, **kw)
    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(Path, "mkdir", Mock(side_effect=AssertionError("status mkdir")))
    result = run(tmp_path, "season-contract", "status", "--season", "2026")
    assert result.exit_code == 0
    assert f"live decision now: {mode}" in result.output
    assert result.output.index("saver:") < result.output.index("live decision now:") < result.output.index("ledger:")
    if mode == "refused":
        with pytest.raises(cs.ContestStateError) as exc:
            live(tmp_path)
        assert str(exc.value) in result.output
    elif mode == "accepted":
        state = live(tmp_path)
        for text in [state.source, f"streak={state.streak}", state.status, state.saver_flag_state]:
            assert text in result.output
    assert files(tmp_path.parent) == before
    assert writes == []


def commands(message):
    return [shlex.split(text) for text in re.findall(r"`(bts [^`]+)`", message)]


@pytest.mark.parametrize("case", ["corrupt", "legacy", "absent", "other_season", "revoked", "missing", "schema", "season", "id"])
@pytest.mark.parametrize("alternative", ["dated", "automatic"])
def test_message_complete_recovery_paths(tmp_path, monkeypatch, case, alternative):
    picks = tmp_path / "space ' quote & picks"
    picks.mkdir()
    if case != "absent":
        put(picks, "season_contract.json", [] if case == "corrupt" else contract(
            season=2026 if case == "other_season" else 2027,
            status="revoked" if case == "revoked" else "automatic", revoked_at=NOW.isoformat()))
    if case != "missing":
        put(picks, "contest_streak.json", automatic(
            schema_version="old" if case == "schema" else sc.AUTO_SCHEMA,
            season=2026 if case == "season" else 2027,
            activation_id="b" * 32 if case == "id" else ID))
    if case == "legacy":
        put(picks, "contest_streak.manual.json", MANUALS["legacy"])
    with pytest.raises(cs.ContestStateError) as exc:
        live(picks)
    message = str(exc.value)
    assert "MLB" in message and "TEMPLATE" in message
    for text in ["another blocker", "non-zero fetch exit", "not itself a verification"]:
        assert text in message
    extracted = commands(message)
    assert extracted
    for cmd in extracted:
        assert cmd[cmd.index("--picks-dir") + 1] == str(picks)
    if alternative == "dated" or case == "legacy":
        cmd = next(c for c in extracted if c[1] == "set-contest-streak")
        cmd = [{"N": "12", "B": "14", "S": "used"}.get(c, c) for c in cmd]
        result = CliRunner().invoke(cli.cli, cmd[1:])
        assert result.exit_code == 0, result.output
    else:
        if case == "corrupt":
            # Deliberate removal is an operator action; it reveals reason 3.
            (picks / "account_state/season_contract.json").unlink()
            with pytest.raises(cs.ContestStateError, match="absent"):
                live(picks)
        for cmd in extracted:
            if cmd[1] == "set-contest-streak":
                continue
            result = CliRunner().invoke(cli.cli, cmd[1:])
            assert result.exit_code == 0, result.output
    assert live(picks).streak in {7, 12}


@pytest.mark.parametrize("args", [[], ["--season", "2026", "--reason", "x"],
                                  ["--season", "2027"], ["--season", "2027", "--reason", "  "]])
def test_revoke_usage_no_application_reads(tmp_path, monkeypatch, args):
    spy = Mock(side_effect=AssertionError("usage read application state"))
    monkeypatch.setattr(sc, "switch_state", spy)
    monkeypatch.setattr(Path, "open", spy)
    result = run(tmp_path, "season-contract", "revoke", *args)
    assert result.exit_code == 2
    assert "No such command" not in result.output  # availability alone is not coverage
    spy.assert_not_called()


@pytest.mark.parametrize("prior", SWITCHES)
@pytest.mark.parametrize("present", [False, True])
def test_revoke_snapshot_archives_and_repeat(tmp_path, prior, present):
    install(tmp_path, "season_contract.json", SWITCHES[prior])
    untouched = {name: put(tmp_path, name, automatic()).read_bytes() for name in
                 ["contest_streak.json", "contest_ledger.jsonl", "contest_streak_fetch_status.json"]}
    if present:
        put(tmp_path, "contest_streak.manual.json", dated())
        put(tmp_path, "saver_state.json", dict(state="used", season=2027, history=["evidence"]))
    before = {name: (tmp_path / "account_state" / name).read_bytes() for name in
              ["contest_streak.manual.json", "saver_state.json"]} if present else {}
    result = run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "  restore  ")
    assert result.exit_code == 0, result.output
    value = sc.switch_state(tmp_path, NOW)
    assert value.state == "revoked"
    data = value.snapshot
    assert data["schema_version"] == sc.SCHEMA and data["note"] == "  restore  "
    assert data["revoked_at"] == "2027-07-15T16:00:00Z"
    carry = prior in {"automatic", "revoked"}
    assert (data["activation_id"] == ID) == carry
    assert data["switched_at"] == (contract()["switched_at"] if carry else None)
    for name, content in before.items():
        stem = name[:-5]
        archive = tmp_path / "account_state" / f"{stem}.revoked-20270715T160000000000Z.json"
        assert archive.read_bytes() == content
        assert str(archive) in result.output
    for name, content in untouched.items():
        assert (tmp_path / "account_state" / name).read_bytes() == content
    assert load_saver_state(tmp_path, season=2027).state == "uninitialized"
    with pytest.raises(cs.ContestStateError, match="revoked"):
        live(tmp_path)
    for text in ["KEPT", "NEW dated observation", "uninitialized", "does not initialise", "scheduler",
                 "already running", "not cancelled", "season-contract status", "TEMPLATE"]:
        assert text in result.output
    second = run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "again")
    assert second.exit_code == 0 and sc.switch_state(tmp_path, NOW).snapshot["note"] == "again"


@pytest.mark.parametrize("stage", ["switch", "manual", "saver"])
@pytest.mark.parametrize("failure", ["permission", "occupied", "dangling_target"])
def test_revoke_failure_prefixes_and_retry(tmp_path, monkeypatch, stage, failure):
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.json", automatic())
    manual = put(tmp_path, "contest_streak.manual.json", dated())
    saver = put(tmp_path, "saver_state.json", dict(state="used", season=2027))
    before = files(tmp_path)
    protected = {put(tmp_path, "contest_ledger.jsonl", {"evidence": "ledger"}): b'{"evidence": "ledger"}'}
    status = tmp_path.parent / "health_state/contest_streak_fetch_status.json"
    status.parent.mkdir()
    status.write_text("fetch evidence")
    protected[status] = status.read_bytes()
    writer, rename = util.atomic_write_text, os.rename
    events = []
    target = None
    if failure != "permission":
        if stage == "switch":
            (tmp_path / "account_state/season_contract.json").unlink()
            (tmp_path / "account_state/season_contract.json").mkdir()
        else:
            stem = "contest_streak.manual" if stage == "manual" else "saver_state"
            target = tmp_path / "account_state" / f"{stem}.revoked-20270715T160000000000Z.json"
            if failure == "dangling_target":
                target.symlink_to(target.parent / "missing")
            else:
                target.write_text("occupied")
    def write(path, text):
        events.append("switch")
        if stage == "switch" and failure == "permission":
            raise PermissionError("injected")
        return writer(path, text)
    def move(src, dst):
        label = "manual" if src == manual else "saver"
        events.append(label)
        if stage == label and failure == "permission":
            raise PermissionError("injected")
        return rename(src, dst)
    with monkeypatch.context() as patcher:
        patcher.setattr(util, "atomic_write_text", write)
        patcher.setattr(os, "rename", move)
        result = run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "restore")
    assert result.exit_code == 1, result.output
    assert ("NOT revoked" if stage == "switch" else "wrote the revoked switch") in result.output
    if stage == "manual":
        assert "manual file was not set aside" in result.output and "saver flag was not attempted" in result.output
    if stage == "saver":
        assert "saver flag was not set aside" in result.output and "renamed to" in result.output
    assert manual.exists() == (stage != "saver") and saver.exists()
    assert "saver" not in events if stage != "saver" else True
    assert (tmp_path / "account_state/contest_streak.json").read_bytes() == before["account_state/contest_streak.json"]
    for path, content in protected.items():
        assert path.read_bytes() == content
    if target:
        target.unlink()
    if stage == "switch" and failure != "permission":
        (tmp_path / "account_state/season_contract.json").rmdir()
    retry = run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "retry")
    assert retry.exit_code == 0, retry.output
    for path, content in protected.items():
        assert path.read_bytes() == content


def test_dangling_source_and_absent_source_occupied_target(tmp_path):
    manual = tmp_path / "account_state/contest_streak.manual.json"
    manual.parent.mkdir()
    manual.symlink_to(manual.parent / "missing")
    archive = manual.with_name("contest_streak.manual.revoked-20270715T160000000000Z.json")
    assert run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "x").exit_code == 0
    assert archive.is_symlink() and not manual.is_symlink()
    # The occupied target is irrelevant when the source is absent.
    assert run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "again").exit_code == 0


def test_year_boundary_and_dated_last_microsecond(tmp_path):
    end = REAL_DATETIME(2027, 12, 31, 23, 59, 59, 999999, tzinfo=ET)
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.json", automatic(source_date="2027-12-31"))
    assert live(tmp_path, end).streak == 12
    with pytest.raises(cs.ContestStateError, match="other_season"):
        live(tmp_path, end + dt.timedelta(microseconds=1))
    put(tmp_path, "contest_streak.manual.json", dated("2027-12-31"))
    put(tmp_path, "season_contract.json", [])
    assert live(tmp_path, end).streak == 12
    with pytest.raises(cs.ContestStateError, match="corrupt"):
        live(tmp_path, end + dt.timedelta(microseconds=1))


def test_revoke_entry_clock_crossing_year(tmp_path, monkeypatch):
    entry = REAL_DATETIME(2027, 12, 31, 23, 59, 59, 999999, tzinfo=ET)
    pin(monkeypatch, entry)
    original = sc.switch_state
    def crossing(*a, **kw):
        pin(monkeypatch, entry + dt.timedelta(microseconds=1))
        return original(*a, **kw)
    monkeypatch.setattr(sc, "switch_state", crossing)
    result = run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "x")
    assert result.exit_code == 0, result.output
    data = json.loads((tmp_path / "account_state/season_contract.json").read_text())
    assert data["season"] == 2027 and data["revoked_at"] == "2028-01-01T04:59:59.999999Z"


@pytest.mark.parametrize("failure", [None, "before", "after"])
def test_restore_reenable_fetch_sequence(tmp_path, monkeypatch, failure):
    put(tmp_path, "season_contract.json", contract())
    old_auto = put(tmp_path, "contest_streak.json", automatic())
    restored = old_auto.read_bytes()
    assert live(tmp_path).streak == 12
    old_auto.write_bytes(restored)  # restoring a matching pair is deliberately undetectable
    assert live(tmp_path).streak == 12
    assert run(tmp_path, "season-contract", "revoke", "--season", "2027", "--reason", "restore").exit_code == 0
    with pytest.raises(cs.ContestStateError, match="revoked"):
        live(tmp_path)
    assert run(tmp_path, "set-contest-streak", "--date", "2027-07-15", "--streak", "12",
               "--best-streak", "14", "--saver", "used").exit_code == 0
    assert live(tmp_path).saver_flag_state == "used"
    assert run(tmp_path, "season-contract", "enable", "--season", "2027").exit_code == 0
    tomorrow = NOW + dt.timedelta(days=1)
    pin(monkeypatch, tomorrow)
    with pytest.raises(cs.ContestStateError, match="activation_id"):
        live(tmp_path, tomorrow)
    writer = cli._atomic_write_json
    def writing(path, value):
        if (failure == "before" and path.name == "contest_streak.json" or
                failure == "after" and path.name == "contest_streak_fetch_status.json"):
            raise OSError("injected")
        return writer(path, value)
    monkeypatch.setattr(cli, "_atomic_write_json", writing)
    result = fetch(tmp_path)
    assert result.exit_code == (0 if failure is None else 1)
    if failure == "before":
        with pytest.raises(cs.ContestStateError, match="activation_id"):
            live(tmp_path, tomorrow)
    else:
        assert live(tmp_path, tomorrow).streak == 7
