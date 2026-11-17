"""CS4b-1: offline command contracts, persistence prefixes and legacy controls."""
from __future__ import annotations

import builtins
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock
from zoneinfo import ZoneInfo

import httpx
import pytest
from click.testing import CliRunner

import bts.cli as cli
import bts.contest_fetch as cf
import bts.contest_state as cs
import bts.leaderboard.auth as auth
import bts.saver_state as saver

ET = ZoneInfo("America/New_York")
NOW = dt.datetime(2027, 7, 15, 12, tzinfo=ET)
REAL_DATETIME = dt.datetime
ID = "a" * 32


def contract(**changes):
    return dict(schema_version="bts_season_contract_v1", season=2027,
                status="automatic", activation_id=ID,
                switched_at="2027-07-15T16:00:00Z", revoked_at=None, note=None) | changes


def put(picks, name, value):
    path = picks / "account_state" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def pin(monkeypatch, now=NOW):
    # pandas must initialize against the real datetime types.
    import pandas  # noqa: F401

    class Clock(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            return now.astimezone(tz) if tz else now.replace(tzinfo=None)

    monkeypatch.setattr(dt, "datetime", Clock)
    return Clock


@pytest.fixture
def tmp_path(tmp_path):
    # The status lives in picks.parent: give every test its own parent as well.
    picks = tmp_path / "picks"
    picks.mkdir()
    return picks


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    pin(monkeypatch)
    monkeypatch.setattr(httpx, "get", Mock(side_effect=AssertionError("uncaptured transport")))
    monkeypatch.setattr(auth, "load_session_cookies", lambda: {})
    monkeypatch.setattr(auth, "extract_uid", lambda _: "u")
    monkeypatch.setattr(auth, "fetch_login_session", lambda **_: SimpleNamespace(
        user_id=1, username="offline", xsid="x"))
    monkeypatch.setattr(cf, "fetch_profile", lambda *_: {
        "activeStreak": 7, "seasonBestStreak": 9,
        "predictions": [{"roundId": 1, "result": "hit"}]})
    monkeypatch.setattr(cli, "_fetch_rounds", lambda: {1: dt.date(2027, 7, 14)})
    monkeypatch.setattr("bts.dm.send_dm", Mock())


def run(picks, *args):
    return CliRunner().invoke(cli.cli, [*args, "--picks-dir", str(picks)])


def fetch(picks, dry=False):
    return run(picks, "fetch-contest-streak", "--dm-recipient", "offline.test",
               *(["--dry-run"] if dry else []))


def files(root):
    return {str(p.relative_to(root)): p.read_bytes()
            for p in root.rglob("*") if p.is_file()}


BAD_CHECKS = [
    (1, []), (2, contract(schema_version=None)), (3, contract(season=True)),
    (4, contract(status=[])), (5, contract(activation_id="A" * 32)),
    (6, contract(switched_at=None)), (7, contract(revoked_at="naive")),
    (8, contract(note=[])),
]


@pytest.mark.parametrize("number,value", BAD_CHECKS)
def test_switch_ordered_checks(tmp_path, number, value):
    from bts.season_contract import switch_state
    put(tmp_path, "season_contract.json", value)
    state = switch_state(tmp_path, NOW)
    assert state.state == "corrupt"
    assert state.check == number
    result = run(tmp_path, "season-contract", "status")
    assert result.exit_code == 1
    assert f"check {number}" in result.output
    assert "manual" in result.output and "ledger" in result.output


@pytest.mark.parametrize("first", range(2, 8))
def test_lower_invalid_check_wins_before_season_comparison(tmp_path, first):
    from bts.season_contract import switch_state
    value = dict(BAD_CHECKS[first - 1][1], season=2026) if first != 3 else contract(season=True)
    value["note"] = []
    put(tmp_path, "season_contract.json", value)
    assert switch_state(tmp_path, NOW).check == first


@pytest.mark.parametrize("field", ["switched_at", "revoked_at"])
@pytest.mark.parametrize("status", ["automatic", "revoked"])
@pytest.mark.parametrize("value", ["missing", "2027-07-15T12:00:00", None,
                                   "2027-07-15T12:00:00-04:00", "2027-07-15T16:00:00Z"])
def test_nullable_timestamps(tmp_path, field, status, value):
    from bts.season_contract import switch_state
    data = contract(status=status, revoked_at="2027-07-15T16:00:00Z")
    if value == "missing":
        del data[field]
    else:
        data[field] = value
    put(tmp_path, "season_contract.json", data)
    required = (status == "automatic" and field == "switched_at") or (
        status == "revoked" and field == "revoked_at")
    valid = value not in {"missing", "2027-07-15T12:00:00"} and (value is not None or not required)
    assert switch_state(tmp_path, NOW).state == (status if valid else "corrupt")


@pytest.mark.parametrize("value", [None, True, 32, [], "", "a" * 31, "a" * 33, "A" * 32, "g" * 32])
def test_identifier_format(tmp_path, value):
    from bts.season_contract import switch_state
    put(tmp_path, "season_contract.json", contract(activation_id=value))
    assert switch_state(tmp_path, NOW).check == 5


@pytest.mark.parametrize("kind", ["dangling", "directory", "utf8", "huge", "deep", "lstat"])
def test_unreadable_switch_is_corrupt(tmp_path, monkeypatch, kind):
    from bts.season_contract import switch_state
    path = tmp_path / "account_state/season_contract.json"
    path.parent.mkdir()
    if kind == "dangling":
        path.symlink_to(path.parent / "missing")
    elif kind == "directory":
        path.mkdir()
    elif kind == "utf8":
        path.write_bytes(b"\xff")
    elif kind == "huge":
        path.write_text("1" * 5000)
    elif kind == "deep":
        path.write_text("[" * 3000 + "]" * 3000)
    else:
        original = Path.lstat
        def lstat(p, *a, **kw):
            if p == path:
                raise PermissionError("injected")
            return original(p, *a, **kw)
        monkeypatch.setattr(Path, "lstat", lstat)
    assert switch_state(tmp_path, NOW).check == 1


@pytest.mark.parametrize("prior", ["absent", "corrupt", "other_season", "revoked", "automatic"])
def test_enable_snapshot_carry_and_noop(tmp_path, monkeypatch, prior):
    revoked = "2027-07-14T12:00:00Z"
    if prior != "absent":
        put(tmp_path, "season_contract.json", contract(
            season=2026 if prior == "other_season" else 2027,
            status="revoked" if prior == "revoked" else "automatic",
            revoked_at=revoked, note=[] if prior == "corrupt" else "old"))
    import bts.util as util
    write = Mock(wraps=util.atomic_write_text)
    monkeypatch.setattr(util, "atomic_write_text", write)
    before = files(tmp_path)
    r = run(tmp_path, "season-contract", "enable", "--season", "2027", "--note", "new")
    assert r.exit_code == 0, r.output
    data = json.loads((tmp_path / "account_state/season_contract.json").read_text())
    assert data["activation_id"] == ID if prior == "automatic" else data["activation_id"] != ID
    assert data["note"] == ("old" if prior == "automatic" else "new")
    assert data["revoked_at"] == (revoked if prior in {"revoked", "automatic"} else None)
    assert write.call_count == (0 if prior == "automatic" else 1)
    if prior == "automatic":
        assert files(tmp_path) == before
        assert "already switched" in r.output
    else:
        assert "dated observation" in r.output and "non-zero" in r.output
    identifier = data["activation_id"]
    write.reset_mock()
    assert run(tmp_path, "season-contract", "enable", "--season", "2027").exit_code == 0
    write.assert_not_called()
    assert json.loads((tmp_path / "account_state/season_contract.json").read_text())["activation_id"] == identifier
    assert list(tmp_path.rglob("*.json")) == [tmp_path / "account_state/season_contract.json"]


def test_wrong_enable_season_precedes_state_reads(tmp_path, monkeypatch):
    original = Path.open
    def opening(p, *args, **kw):
        if "account_state" in p.parts or "health_state" in p.parts:
            pytest.fail("application read before usage error")
        return original(p, *args, **kw)
    monkeypatch.setattr(Path, "open", opening)
    r = run(tmp_path, "season-contract", "enable", "--season", "2026")
    assert r.exit_code == 2 and "2027" in r.output
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("kind", ["directory", "readonly", "injected"])
def test_enable_write_failure_preserves_prior(tmp_path, monkeypatch, kind):
    path = put(tmp_path, "season_contract.json", contract(status="revoked", revoked_at="2027-07-14T16:00:00Z"))
    before = path.read_bytes()
    if kind == "directory":
        path.unlink()
        path.mkdir()
    elif kind == "readonly":
        path.parent.chmod(0o555)
    else:
        monkeypatch.setattr("bts.util.atomic_write_text", Mock(side_effect=OSError("injected")))
    try:
        r = run(tmp_path, "season-contract", "enable", "--season", "2027")
        assert r.exit_code == 1 and "NOT switched" in r.output
        assert path.is_dir() if kind == "directory" else path.read_bytes() == before
    finally:
        path.parent.chmod(0o755)


@pytest.mark.parametrize("state", ["absent", "other_season", "revoked", "automatic", "corrupt"])
@pytest.mark.parametrize("dry", [False, True])
def test_fetch_switch_paths_order_and_spies(tmp_path, monkeypatch, state, dry):
    picks = tmp_path / "picks"
    if state != "absent":
        put(picks, "season_contract.json", contract(
            season=2026 if state == "other_season" else 2027,
            status="revoked" if state == "revoked" else "automatic",
            revoked_at="2027-07-14T16:00:00Z", note=[] if state == "corrupt" else None))
    events = []
    writer, opener = cli._atomic_write_json, Path.open
    def write(p, value):
        events.append(p.name)
        return writer(p, value)
    def opening(p, mode="r", *a, **kw):
        if p.name == "contest_ledger.jsonl" and mode == "a":
            events.append("ledger")
        if p.name in {"saver_state.json", "saver_state.lock", "saver_transitions.jsonl"}:
            pytest.fail("saver I/O outside helper")
        if p.name == "contest_streak.json" and any(s in mode for s in "wax+"):
            assert state == "automatic" and not dry
        return opener(p, mode, *a, **kw)
    monkeypatch.setattr(cli, "_atomic_write_json", write)
    monkeypatch.setattr(Path, "open", opening)
    earn = Mock(side_effect=lambda *a, **kw: events.append("saver"))
    monkeypatch.setattr(saver, "maybe_auto_earn_saver", earn)
    before = files(tmp_path)
    r = fetch(picks, dry)
    assert r.exit_code == (2 if state == "corrupt" else 0), r.output
    if state == "corrupt":
        assert events == ["contest_streak_fetch_status.json"]
        import bts.dm
        assert bts.dm.send_dm.call_count == 1
    elif dry:
        assert files(tmp_path) == before and events == []
        assert ("automatic" if state == "automatic" else "not switched") in r.output
    else:
        switched = state == "automatic"
        assert events == (["contest_streak.json"] if switched else []) + [
            "contest_streak_fetch_status.json", "ledger"] + (["saver"] if switched else [])
        row = json.loads((picks / "account_state/contest_ledger.jsonl").read_text())
        assert row["season"] == 2027 and row["switched"] is switched
        assert row["activation_id"] == (ID if switched else None)
        if switched:
            auto = json.loads((picks / "account_state/contest_streak.json").read_text())
            assert (auto["schema_version"], auto["season"], auto["activation_id"]) == (
                "bts_contest_streak_auto_v2", 2027, ID)
        else:
            assert "not switched" in r.output and not earn.called


@pytest.mark.parametrize("bad", ["missing", None, {}, "x", [None], [] , "rounds", "best58"])
@pytest.mark.parametrize("dry", [False, True])
def test_response_validation_and_alerts(tmp_path, monkeypatch, bad, dry):
    prior = put(tmp_path, "contest_streak.json", {"active_streak": 4})
    before = prior.read_bytes()
    profile = {"activeStreak": 7, "seasonBestStreak": 58 if bad == "best58" else 9}
    if bad != "missing":
        profile["predictions"] = [{"roundId": 1, "result": "hit"}] if bad in ("rounds", "best58") else bad
    monkeypatch.setattr(cf, "fetch_profile", lambda *_: profile)
    if bad == "rounds":
        monkeypatch.setattr(cli, "_fetch_rounds", lambda: {})
    r = fetch(tmp_path, dry)
    valid_empty = bad == []
    assert r.exit_code == (0 if valid_empty else 2), r.output
    assert prior.read_bytes() == before
    if not valid_empty:
        import bts.dm
        assert bts.dm.send_dm.call_count == 1
        assert (tmp_path.parent / "health_state/contest_streak_fetch_status.json").exists()
        assert not (tmp_path / "account_state/contest_ledger.jsonl").exists()
    else:
        # A valid empty list (review round 1, Codex note): a dry run persists
        # nothing; a real not-switched run records status and one ledger line.
        import bts.dm
        assert bts.dm.send_dm.call_count == 0
        status = tmp_path.parent / "health_state/contest_streak_fetch_status.json"
        ledger = tmp_path / "account_state/contest_ledger.jsonl"
        assert status.exists() is (not dry)
        assert ledger.exists() is (not dry)
        if not dry:
            assert len(ledger.read_text().splitlines()) == 1


@pytest.mark.parametrize("profile,rounds_called,category", [
    (None, False, "actionable"), ({"predictions": None}, True, "transient")])
def test_shape_failure_precedence(tmp_path, monkeypatch, profile, rounds_called, category):
    monkeypatch.setattr(cf, "fetch_profile", lambda *_: profile)
    rounds = Mock(side_effect=httpx.HTTPStatusError("503", request=httpx.Request("GET", "https://offline"),
                                                  response=httpx.Response(503)))
    monkeypatch.setattr(cli, "_fetch_rounds", rounds)
    r = fetch(tmp_path)
    assert r.exit_code == 2
    assert rounds.called is rounds_called
    status = json.loads((tmp_path.parent / "health_state/contest_streak_fetch_status.json").read_text())
    assert category in status["last_alert_at_by_category"]


def test_ledger_row_counts_partition(tmp_path, monkeypatch):
    predictions = [{"result": r, **({"roundId": i} if i is not None else {})}
                   for r, i in [("hit", 1), ("void", 1), ("not_hit", 2), ("miss", None),
                                ("hit", 999), (None, 1), ("pending", 2), ("unknown", 1)]] + [{}]
    monkeypatch.setattr(cf, "fetch_profile", lambda *_: {
        "activeStreak": 7, "seasonBestStreak": 9, "predictions": predictions})
    monkeypatch.setattr(cli, "_fetch_rounds", lambda: {1: dt.date(2026, 7, 14), 2: dt.date(2027, 7, 14)})
    assert fetch(tmp_path).exit_code == 0
    row = json.loads((tmp_path / "account_state/contest_ledger.jsonl").read_text())
    assert row["settled_by_year"] == {"2026": 2, "2027": 1}
    assert (row["settled_unmapped"], row["unsettled"]) == (2, 4)
    assert row["predictions"] == predictions


@pytest.mark.parametrize("field,value", [("season", None), ("season", True), ("season", "2027"),
                                         ("activation_id", None), ("activation_id", "A" * 32)])
def test_auto_v2_fields_after_legacy_checks(tmp_path, field, value):
    data = dict(schema_version="bts_contest_streak_auto_v2", active_streak=7, best_streak=9,
                source="mlb_bts_profile", source_date="2027-07-14", season=2027, activation_id=ID)
    data[field] = value
    path = put(tmp_path, "contest_streak.json", data)
    with pytest.raises(cs.ContestStateError, match=field):
        cs._parse_state_file(path)
    path = put(tmp_path, "contest_streak.manual.json", data)
    parsed = cs._parse_state_file(path)
    assert parsed.season is None and parsed.activation_id is None
    assert cs.classify_best_streak(parsed, now=NOW, now_year=2027) == (9, "trusted")


def test_auto_v2_season_error_wins_and_valid_metadata(tmp_path):
    path = put(tmp_path, "contest_streak.json", dict(schema_version="bts_contest_streak_auto_v2",
               active_streak=7, best_streak=9, season=True, activation_id=None))
    with pytest.raises(cs.ContestStateError, match="season"):
        cs._parse_state_file(path)
    data = json.loads(path.read_text()) | {"season": 2026, "activation_id": ID,
            "source": "mlb_bts_profile", "source_date": "2027-07-14"}
    path.write_text(json.dumps(data))
    state = cs._parse_state_file(path)
    assert (state.season, state.activation_id) == (2026, ID)
    assert cs.classify_best_streak(state, now=NOW, now_year=2027) == (9, "trusted")


@pytest.mark.parametrize("broken", ["manual", "automatic", "saver", "ledger"])
def test_status_independent_sections_readonly_and_inspection_season(tmp_path, monkeypatch, broken):
    put(tmp_path, "season_contract.json", contract())
    put(tmp_path, "contest_streak.manual.json", {"active_streak": 2})
    put(tmp_path, "contest_streak.json", {"schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 7, "best_streak": 9, "season": 2027, "activation_id": ID})
    put(tmp_path, "saver_state.json", {"season": 2027, "state": "used"})
    ledger = tmp_path / "account_state/contest_ledger.jsonl"
    ledger.write_text("\n[]\n" + "\n".join(json.dumps({"active_streak": i}) for i in range(12)) + "\n{truncated")
    names = {"manual": "contest_streak.manual.json", "automatic": "contest_streak.json",
             "saver": "saver_state.json", "ledger": "contest_ledger.jsonl"}
    (tmp_path / "account_state" / names[broken]).write_bytes(b"\xff")
    before = files(tmp_path)
    opener, builtin_open = Path.open, builtins.open
    def guard(p, mode="r", *a, **kw):
        assert not any(c in mode for c in "wax+")
        return opener(p, mode, *a, **kw)
    def no_builtin_write(p, mode="r", *a, **kw):
        assert not any(c in mode for c in "wax+")
        return builtin_open(p, mode, *a, **kw)
    monkeypatch.setattr(Path, "open", guard)
    monkeypatch.setattr(builtins, "open", no_builtin_write)
    monkeypatch.setattr(Path, "mkdir", Mock(side_effect=AssertionError("status mkdir")))
    monkeypatch.setattr(cs, "load_contest_streak_state", Mock(side_effect=AssertionError("selecting loader")))
    r = run(tmp_path, "season-contract", "status", "--season", "2026")
    assert r.exit_code == 0, r.output
    for text in ["2027-07-15", "2027", "2026", "other_season", "manual", "automatic", "saver", "ledger", "unavailable"]:
        assert text in r.output
    if broken != "ledger":
        assert "bad lines: 3" in r.output
    assert files(tmp_path) == before


@pytest.mark.parametrize("phase", ["auth", "persistence"])
@pytest.mark.parametrize("source_year", [2027, 2028])
def test_entry_clock_owns_season_and_saver_gate(tmp_path, monkeypatch, phase, source_year):
    entry = REAL_DATETIME(2027, 12, 31, 23, 59, 59, tzinfo=ET)
    later = REAL_DATETIME(2028, 1, 1, 0, 0, 1, tzinfo=ET)
    current = [entry]
    class Clock(REAL_DATETIME):
        @classmethod
        def now(cls, tz=None):
            return current[0].astimezone(tz) if tz else current[0].replace(tzinfo=None)
    monkeypatch.setattr(dt, "datetime", Clock)
    if phase == "auth":
        def cookies():
            current[0] = later
            return {}
        monkeypatch.setattr(auth, "load_session_cookies", cookies)
    original = cli._atomic_write_json
    def write(*args):
        current[0] = later
        return original(*args)
    monkeypatch.setattr(cli, "_atomic_write_json", write)
    put(tmp_path, "season_contract.json", contract())
    monkeypatch.setattr(cli, "_fetch_rounds", lambda: {1: dt.date(source_year, 12 if source_year == 2027 else 1, 31 if source_year == 2027 else 1)})
    earn = Mock()
    monkeypatch.setattr(saver, "maybe_auto_earn_saver", earn)
    assert fetch(tmp_path).exit_code == 0
    auto = json.loads((tmp_path / "account_state/contest_streak.json").read_text())
    ledger = json.loads((tmp_path / "account_state/contest_ledger.jsonl").read_text())
    assert auto["season"] == ledger["season"] == 2027
    assert auto["recorded_at"] == ledger["recorded_at"] == "2028-01-01T04:59:59Z"
    assert earn.call_count == (1 if source_year == 2027 else 0)
    if earn.called:
        assert earn.call_args.kwargs["season"] == 2027


def test_full_activation_sequence(tmp_path):
    path = put(tmp_path, "contest_streak.json", {"schema_version": "bts_contest_streak_auto_v1", "active_streak": 4})
    old = path.read_bytes()
    for _ in range(3):
        assert fetch(tmp_path).exit_code == 0
        assert path.read_bytes() == old
    assert len((tmp_path / "account_state/contest_ledger.jsonl").read_text().splitlines()) == 3
    assert run(tmp_path, "season-contract", "status").exit_code == 0
    assert run(tmp_path, "season-contract", "enable", "--season", "2027").exit_code == 0
    switch = tmp_path / "account_state/season_contract.json"
    first = json.loads(switch.read_text())["activation_id"]
    assert fetch(tmp_path).exit_code == 0
    auto = path.read_bytes()
    assert json.loads(auto)["activation_id"] == first
    assert run(tmp_path, "season-contract", "enable", "--season", "2027").exit_code == 0
    assert json.loads(switch.read_text())["activation_id"] == first
    data = json.loads(switch.read_text()) | {"status": "revoked", "revoked_at": "2027-07-15T16:00:01Z"}
    switch.write_text(json.dumps(data))
    assert fetch(tmp_path).exit_code == 0 and path.read_bytes() == auto
    assert run(tmp_path, "season-contract", "enable", "--season", "2027").exit_code == 0
    second = json.loads(switch.read_text())["activation_id"]
    assert second != first
    r = run(tmp_path, "season-contract", "status")
    assert "identifier matches switch: False" in r.output
    assert fetch(tmp_path).exit_code == 0
    assert json.loads(path.read_text())["activation_id"] == second
