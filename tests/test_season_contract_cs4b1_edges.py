"""Failure prefixes, read isolation, status edge cases and legacy controls."""
import datetime as dt
import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from tests.test_season_contract_cs4b1 import (
    ID, NOW, ET, REAL_DATETIME, contract, fetch, files, offline, pin, put, run, tmp_path,
)
import bts.cli as cli
import bts.contest_fetch as cf
import bts.contest_state as cs
import bts.saver_state as saver
from bts.health import contest_state as health


@pytest.mark.parametrize("state", ["absent", "other_season", "revoked", "automatic", "corrupt"])
@pytest.mark.parametrize("stage", ["observation", "status", "ledger", "saver"])
def test_persistence_prefixes(tmp_path, monkeypatch, state, stage):
    if state != "absent":
        put(tmp_path, "season_contract.json", contract(
            status="revoked" if state == "revoked" else "automatic",
            season=2026 if state == "other_season" else 2027,
            revoked_at="2027-07-14T16:00:00Z", note=[] if state == "corrupt" else None))
    path = put(tmp_path, "contest_streak.json", {"active_streak": 2})
    before = path.read_bytes()
    original, opener = cli._atomic_write_json, Path.open
    events = []
    def write(p, value):
        name = "observation" if p.name == "contest_streak.json" else "status"
        events.append(name)
        if stage == name:
            raise OSError("injected " + name)
        return original(p, value)
    def opening(p, mode="r", *a, **kw):
        if p.name == "contest_ledger.jsonl" and mode == "a":
            events.append("ledger")
            if stage == "ledger":
                raise OSError("injected ledger")
        return opener(p, mode, *a, **kw)
    def earn(*a, **kw):
        events.append("saver")
        if stage == "saver":
            raise OSError("injected saver")
    monkeypatch.setattr(cli, "_atomic_write_json", write)
    monkeypatch.setattr(Path, "open", opening)
    monkeypatch.setattr(saver, "maybe_auto_earn_saver", earn)
    r = fetch(tmp_path)
    automatic = state == "automatic"
    relevant = state == "corrupt" or stage in {"status", "ledger"} or automatic
    assert r.exit_code == (2 if state == "corrupt" or stage == "status" else
                           1 if relevant else 0), r.output
    expected = (["observation"] if automatic else []) + ["status"]
    if state != "corrupt":
        expected += ["ledger"] + (["saver"] if automatic else [])
    if stage in expected:
        expected = expected[:expected.index(stage) + 1]
    assert events == expected
    installed = automatic and stage != "observation"
    assert (path.read_bytes() != before) is installed
    if installed:
        assert json.loads(path.read_text())["activation_id"] == ID
    import bts.dm
    assert bts.dm.send_dm.call_count == (1 if state == "corrupt" else 0)


@pytest.mark.parametrize("state", ["absent", "corrupt", "other_season", "revoked", "automatic"])
@pytest.mark.parametrize("dry", [False, True])
def test_prior_identity_guard_precedes_profile_and_switch(tmp_path, monkeypatch, state, dry):
    put(tmp_path, "season_contract.json", contract(note=[] if state == "corrupt" else None,
        status="revoked" if state == "revoked" else "automatic",
        revoked_at="2027-07-14T16:00:00Z", season=2026 if state == "other_season" else 2027))
    if state == "absent":
        (tmp_path / "account_state/season_contract.json").unlink()
    path = put(tmp_path, "contest_streak.json", {"active_streak": 2, "user_id": 999})
    before = path.read_bytes()
    profile = Mock(side_effect=AssertionError("identity must stop before profile"))
    monkeypatch.setattr(cf, "fetch_profile", profile)
    r = fetch(tmp_path, dry)
    assert r.exit_code == 2 and "prior contest_streak.json" in r.output
    assert not profile.called and path.read_bytes() == before
    import bts.dm
    assert bts.dm.send_dm.call_count == 1


@pytest.mark.parametrize("schema", ["missing", None, "unknown", "bts_contest_streak_auto_v1", "bts_contest_streak_auto_v2"])
@pytest.mark.parametrize("dating", ["none", "recorded", "source"])
def test_status_schema_dating_and_unavailable_comparison(tmp_path, schema, dating):
    data = {"active_streak": 7, "best_streak": 9, "season": 2027, "activation_id": ID}
    if schema != "missing":
        data["schema_version"] = schema
    if dating == "recorded":
        data["recorded_at"] = "2027-07-15T16:00:00Z"
    elif dating == "source":
        data["source_date"] = "2027-07-14"
    put(tmp_path, "contest_streak.json", data)
    for corrupt in [False, True]:
        if corrupt:
            put(tmp_path, "season_contract.json", [])
        r = run(tmp_path, "season-contract", "status")
        assert r.exit_code == (1 if corrupt else 0)
        assert "streak=7" in r.output
        assert ("missing" if schema == "missing" else "null" if schema is None else str(schema)) in r.output
        if schema == "bts_contest_streak_auto_v2":
            assert "identifier matches switch: unavailable" in r.output


@pytest.mark.parametrize("name", ["contest_streak.manual.json", "contest_streak.json", "saver_state.json", "contest_ledger.jsonl"])
@pytest.mark.parametrize("kind", ["dangling", "directory", "utf8", "deep", "huge"])
def test_status_bad_neighbor_does_not_hide_later_sections(tmp_path, name, kind):
    put(tmp_path, "season_contract.json", contract())
    path = tmp_path / "account_state" / name
    if kind == "dangling":
        path.symlink_to(path.parent / "missing")
    elif kind == "directory":
        path.mkdir()
    else:
        path.write_bytes(b"\xff" if kind == "utf8" else
                         ("[" * 3000 + "]" * 3000 if kind == "deep" else "1" * 5000).encode())
    r = run(tmp_path, "season-contract", "status")
    assert r.exit_code == 0, r.output
    for section in ["manual", "automatic", "saver", "ledger"]:
        assert section in r.output
    assert "unavailable" in r.output or "bad lines: 1" in r.output


@pytest.mark.parametrize("manual", [False, True])
@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("required", [False, True])
def test_decision_and_health_never_read_switch(tmp_path, monkeypatch, manual, corrupt, required):
    put(tmp_path, "season_contract.json", [] if corrupt else contract())
    auto = put(tmp_path, "contest_streak.json", {"schema_version": "bts_contest_streak_auto_v1",
        "active_streak": 7, "best_streak": 9, "source": "mlb_bts_profile", "source_date": "2027-07-14"})
    if manual:
        put(tmp_path, "contest_streak.manual.json", {"schema_version": "bts_contest_streak_manual_v3",
            "effective_date": "2027-07-15", "season": 2027, "active_streak": 3,
            "best_streak": 9, "saver_state": "used", "source": "manual_cli",
            "expires_at": "2027-07-16T03:59:59.999999Z", "recorded_at": "2027-07-15T16:00:00Z"})
    opener = Path.open
    def opening(p, *a, **kw):
        if not required or manual:
            assert p.name != "season_contract.json"
        return opener(p, *a, **kw)
    monkeypatch.setattr(Path, "open", opening)
    if required and not manual:
        with pytest.raises(cs.ContestStateError, match="corrupt" if corrupt else "schema_version") as exc:
            cs.load_decision_streak_state(tmp_path, now=NOW, require_contest_state=True)
        alerts = health.check(tmp_path, now=NOW, expected=True)
        assert alerts[-1].message == str(exc.value)
        return
    assert cs.load_decision_streak_state(tmp_path, now=NOW, require_contest_state=required).streak == (3 if manual else 7)
    health.check(tmp_path, now=NOW, expected=required)


def test_saver_audit_failure_nonfatal_and_helper_no_transition(tmp_path, monkeypatch):
    put(tmp_path, "season_contract.json", contract())
    original = saver.maybe_auto_earn_saver
    earn = Mock(wraps=original)
    monkeypatch.setattr(saver, "maybe_auto_earn_saver", earn)
    opener = __import__("builtins").open
    def opening(p, mode="r", *a, **kw):
        if Path(p).name == "saver_transitions.jsonl":
            raise OSError("audit injected")
        return opener(p, mode, *a, **kw)
    monkeypatch.setattr("builtins.open", opening)
    assert fetch(tmp_path).exit_code == 0
    assert earn.call_count == 1
    assert json.loads((tmp_path / "account_state/saver_state.json").read_text())["state"] == "not_earned"
    put(tmp_path, "saver_state.json", {"state": "used", "season": 2027})
    before = files(tmp_path)["account_state/saver_state.json"]
    assert fetch(tmp_path).exit_code == 0
    assert earn.call_count == 2
    assert (tmp_path / "account_state/saver_state.json").read_bytes() == before


def test_best_bound_and_builder_v1_control():
    for value in [0, 57]:
        obs = cf.build_observation({"activeStreak": 0, "seasonBestStreak": value}, None, 1, "offline", NOW)
        assert obs["schema_version"] == "bts_contest_streak_auto_v1"
        assert "season" not in obs and "activation_id" not in obs
    with pytest.raises(cf.ContestFetchError, match="400"):
        cf.validate_fetch({"activeStreak": 0, "seasonBestStreak": 401})


def test_enable_entry_clock_and_next_operation_other_season(tmp_path, monkeypatch):
    entry = REAL_DATETIME(2027, 12, 31, 23, 59, 59, tzinfo=ET)
    later = REAL_DATETIME(2028, 1, 1, 0, 0, 1, tzinfo=ET)
    pin(monkeypatch, entry)
    original = Path.lstat
    def lstat(p, *a, **kw):
        if p.name == "season_contract.json":
            pin(monkeypatch, later)
        return original(p, *a, **kw)
    monkeypatch.setattr(Path, "lstat", lstat)
    r = run(tmp_path, "season-contract", "enable", "--season", "2027")
    assert r.exit_code == 0, r.output
    data = json.loads((tmp_path / "account_state/season_contract.json").read_text())
    assert data["season"] == 2027 and data["switched_at"] == "2028-01-01T04:59:59Z"
    assert "other_season" in run(tmp_path, "season-contract", "status").output


def test_cold_status_creates_nothing(tmp_path, monkeypatch):
    target = tmp_path / "does-not-exist/picks"
    before = files(tmp_path)
    mkdir = Mock(side_effect=AssertionError("read-only status creates a directory"))
    monkeypatch.setattr(Path, "mkdir", mkdir)
    r = run(target, "season-contract", "status")
    assert r.exit_code == 0, r.output
    for label in ["switch: absent", "manual: absent", "automatic: absent", "saver: unavailable (absent)", "ledger: absent"]:
        assert label in r.output
    mkdir.assert_not_called()
    assert files(tmp_path) == before


@pytest.mark.parametrize("state", ["automatic", "revoked", "other_season"])
def test_status_complete_valid_snapshot_and_identifier_comparison(tmp_path, monkeypatch, state):
    put(tmp_path, "season_contract.json", contract(status="revoked" if state == "revoked" else "automatic",
        season=2026 if state == "other_season" else 2027, note="operator inspected",
        revoked_at="2027-07-14T16:00:00Z"))
    put(tmp_path, "contest_streak.json", {"schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 7, "best_streak": 9, "season": 2027, "activation_id": ID})
    original = Path.read_text
    reads = []
    def reading(p, *a, **kw):
        if p.name == "season_contract.json":
            reads.append(p)
        return original(p, *a, **kw)
    monkeypatch.setattr(Path, "read_text", reading)
    r = run(tmp_path, "season-contract", "status")
    assert r.exit_code == 0, r.output
    assert len(reads) == 1
    for value in [state, "operator inspected", ID, "2027-07-14T16:00:00Z", "2027-07-15T16:00:00Z",
                  "identifier matches switch: True"]:
        assert value in r.output


@pytest.mark.parametrize("dated", [False, True])
def test_status_shows_inactive_manual_and_current_saver_and_last_ten_rows(tmp_path, dated):
    if dated:
        put(tmp_path, "contest_streak.manual.json", {"schema_version": "bts_contest_streak_manual_v3",
            "effective_date": "2027-07-14", "season": 2027, "active_streak": 3, "best_streak": 9,
            "saver_state": "used", "source": "manual_cli", "expires_at": "2027-07-15T03:59:59.999999Z",
            "recorded_at": "2027-07-14T16:00:00Z"})
    else:
        put(tmp_path, "contest_streak.manual.json", {"active_streak": 3, "override_expires_at": "2027-07-14T16:00:00Z"})
    put(tmp_path, "saver_state.json", {"season": 2026, "state": "used"})
    (tmp_path / "account_state/contest_ledger.jsonl").write_text(
        "\n[]\n" + "\n".join(json.dumps({"active_streak": i, "recorded_at": f"entry-{i}"}) for i in range(12)) + "\n{")
    r = run(tmp_path, "season-contract", "status", "--season", "2026")
    assert r.exit_code == 0
    assert ("in effect=False" if dated else "legacy streak=3 expiry=") in r.output
    assert "another season" in r.output and "uninitialized" in r.output
    assert "bad lines: 3" in r.output
    rows = [line for line in r.output.splitlines() if line.startswith("  recorded_at=")]
    assert len(rows) == 10
    assert [line.split(";", 1)[0] for line in rows] == [f"  recorded_at=entry-{i}" for i in range(2, 12)]
    assert all("settled_by_year=—" in line for line in rows)


@pytest.mark.parametrize("stage", ["observation", "status"])
def test_enable_message_beside_fetch_failure_before_or_after_install(tmp_path, monkeypatch, stage):
    auto = put(tmp_path, "contest_streak.json", {"active_streak": 2})
    before = auto.read_bytes()
    enabled = run(tmp_path, "season-contract", "enable", "--season", "2027")
    assert enabled.exit_code == 0, enabled.output
    identifier = json.loads((tmp_path / "account_state/season_contract.json").read_text())["activation_id"]
    assert identifier in enabled.output and "dated observation" in enabled.output
    assert "non-zero fetch exit does not prove" in enabled.output
    writer = cli._atomic_write_json
    def write(path, value):
        if path.name == ("contest_streak.json" if stage == "observation" else "contest_streak_fetch_status.json"):
            raise OSError("injected")
        return writer(path, value)
    monkeypatch.setattr(cli, "_atomic_write_json", write)
    r = fetch(tmp_path)
    assert r.exit_code == (1 if stage == "observation" else 2)
    if stage == "observation":
        assert auto.read_bytes() == before
    else:
        assert json.loads(auto.read_text())["activation_id"] == identifier


def test_partial_ledger_append_preserves_completed_prefix(tmp_path, monkeypatch):
    put(tmp_path, "season_contract.json", contract())
    opener = Path.open
    class Partial:
        def __init__(self, stream):
            self.stream = stream
        def __enter__(self):
            return self
        def write(self, text):
            self.stream.write(text[:10])
            self.stream.flush()
            raise OSError("partial append")
        def __exit__(self, *args):
            self.stream.close()
    def opening(path, mode="r", *a, **kw):
        stream = opener(path, mode, *a, **kw)
        return Partial(stream) if path.name == "contest_ledger.jsonl" and mode == "a" else stream
    monkeypatch.setattr(Path, "open", opening)
    r = fetch(tmp_path)
    assert r.exit_code == 1
    assert json.loads((tmp_path / "account_state/contest_streak.json").read_text())["activation_id"] == ID
    assert (tmp_path.parent / "health_state/contest_streak_fetch_status.json").exists()
    assert len((tmp_path / "account_state/contest_ledger.jsonl").read_text()) == 10
    assert not (tmp_path / "account_state/saver_state.json").exists()
