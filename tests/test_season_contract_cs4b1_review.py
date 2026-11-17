"""Adversarial review regression tests for CS4b-1 (ready to adopt into tests/).

Fails against unpatched 28ae3aa9 on B1 and B2, fails against base 8421e3cd on all CS4b-1
assertions, passes against 28ae3aa9 once B1 and B2 in src/bts/season_contract.py are
fixed, and kills mutations M1, M2, and M6 that survived the original CS4b-1 test suite.
"""
from __future__ import annotations

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
NOW = dt.datetime(2027, 7, 15, 12, 0, 0, tzinfo=ET)
REAL_DATETIME = dt.datetime
ID_A = "a" * 32


def _contract(**changes) -> dict:
    base = {
        "schema_version": "bts_season_contract_v1",
        "season": 2027,
        "status": "automatic",
        "activation_id": ID_A,
        "switched_at": "2027-07-15T16:00:00Z",
        "revoked_at": None,
        "note": None,
    }
    base.update(changes)
    return base


def _pin(monkeypatch, target=NOW):
    import pandas  # noqa: F401

    class _Clock(REAL_DATETIME):
        current = target

        @classmethod
        def now(cls, tz=None):
            c = cls.current
            return c.astimezone(tz) if tz else c.replace(tzinfo=None)

    monkeypatch.setattr(dt, "datetime", _Clock)
    return _Clock


@pytest.fixture(autouse=True)
def _offline(monkeypatch):
    _pin(monkeypatch, NOW)
    monkeypatch.setattr(httpx, "get", Mock(side_effect=AssertionError("uncaptured httpx.get")))
    monkeypatch.setattr(auth, "load_session_cookies", lambda: {"oktaid": "u"})
    monkeypatch.setattr(auth, "extract_uid", lambda _: "u")
    monkeypatch.setattr(auth, "fetch_login_session", lambda **_: SimpleNamespace(user_id=1, username="offline", xsid="x"))
    monkeypatch.setattr(cf, "fetch_profile", lambda *_: {
        "activeStreak": 7,
        "seasonBestStreak": 9,
        "predictions": [{"roundId": 1, "result": "hit"}],
    })
    monkeypatch.setattr(cli, "_fetch_rounds", lambda: {1: dt.date(2027, 7, 14)})
    monkeypatch.setattr("bts.dm.send_dm", Mock())


def test_b1_status_ledger_line_repr_recursion_error_never_emits_double_ledger_header(tmp_path):
    """Row 1(f) / B1: find the current stack's json.loads recursion threshold by binary search,
    then probe the boundary depths right below it; status must emit exactly ONE 'ledger:' line
    (never both 'ledger: readable' and 'ledger: unreadable')."""
    lo, hi = 500, 15000
    while lo < hi:
        mid = (lo + hi + 1) // 2
        try:
            json.loads('{"active_streak": ' + "[" * mid + "0" + "]" * mid + "}")
            lo = mid
        except RecursionError:
            hi = mid - 1

    picks = tmp_path / "picks"
    acc = picks / "account_state"
    acc.mkdir(parents=True)
    good_row = json.dumps({"recorded_at": "2027-07-15T12:00:00Z", "active_streak": 7})
    runner = CliRunner()

    # Sweep the boundary window around the current stack's threshold (covers CliRunner + generator frame offset)
    for depth in range(max(1, lo - 40), lo + 5):
        deep_row = '{"active_streak": ' + "[" * depth + "0" + "]" * depth + "}"
        (acc / "contest_ledger.jsonl").write_text(f"{good_row}\n{deep_row}\n", encoding="utf-8")
        res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks)])
        assert res.exit_code == 0, res.output
        ledger_headers = [ln for ln in res.output.splitlines() if ln.startswith("ledger:")]
        assert len(ledger_headers) == 1, f"depth={depth} emitted duplicate headers: {ledger_headers}\n{res.output}"
        assert ledger_headers[0] in {
            "ledger: readable; rows: 2; bad lines: 0",
            "ledger: readable; rows: 1; bad lines: 1",
        }, ledger_headers


@pytest.mark.parametrize("bool_val", [True, False])
def test_b2_status_saver_boolean_season_classified_invalid_not_another_season(tmp_path, bool_val):
    """Row 1(e) / B2: a JSON boolean in saver_state.json's 'season' is invalid, never 'another season'."""
    picks = tmp_path / "picks"
    acc = picks / "account_state"
    acc.mkdir(parents=True)
    (acc / "saver_state.json").write_text(json.dumps({"state": "used", "season": bool_val}), encoding="utf-8")

    res = CliRunner().invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks)])
    assert res.exit_code == 0, res.output
    saver_lines = [ln for ln in res.output.splitlines() if ln.startswith("saver:")]
    assert len(saver_lines) == 1
    assert "unavailable (invalid)" in saver_lines[0], saver_lines[0]
    assert "another season" not in saver_lines[0], saver_lines[0]


def test_m1_wrong_enable_season_precedes_lstat_and_open_with_seeded_switch(tmp_path, monkeypatch):
    """Kills M1: seed season_contract.json on disk and spy both Path.lstat and Path.open."""
    picks = tmp_path / "picks"
    acc = picks / "account_state"
    acc.mkdir(parents=True)
    (acc / "season_contract.json").write_text(json.dumps(_contract()), encoding="utf-8")

    orig_lstat, orig_open = Path.lstat, Path.open

    def guard_lstat(p, *a, **kw):
        if "account_state" in p.parts or "health_state" in p.parts:
            pytest.fail(f"application lstat before usage error: {p}")
        return orig_lstat(p, *a, **kw)

    def guard_open(p, *a, **kw):
        if "account_state" in p.parts or "health_state" in p.parts:
            pytest.fail(f"application open before usage error: {p}")
        return orig_open(p, *a, **kw)

    monkeypatch.setattr(Path, "lstat", guard_lstat)
    monkeypatch.setattr(Path, "open", guard_open)
    res = CliRunner().invoke(cli.cli, ["season-contract", "enable", "--season", "2026", "--picks-dir", str(picks)])
    assert res.exit_code == 2 and "2027" in res.output


def test_m2_genuine_12000_deep_json_triggers_recursion_error_in_switch_and_status(tmp_path):
    """Kills M2: depth 12,000 JSON object genuinely raises RecursionError in Python 3.12 _json."""
    from bts.season_contract import switch_state

    picks = tmp_path / "picks"
    acc = picks / "account_state"
    acc.mkdir(parents=True)
    deep_obj = '{"a":' * 12000 + "0" + "}" * 12000
    for fname in ["season_contract.json", "contest_streak.manual.json", "contest_streak.json", "saver_state.json", "contest_ledger.jsonl"]:
        (acc / fname).write_text(deep_obj + "\n", encoding="utf-8")

    st = switch_state(picks, NOW)
    assert st.state == "corrupt" and st.check == 1 and "recursion" in st.error.lower()

    res = CliRunner().invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks)])
    assert res.exit_code == 1
    assert "ledger: readable; rows: 0; bad lines: 1" in res.output


def test_m6_v2_season_error_precedes_activation_id_without_season_in_test_name(tmp_path):
    """Kills M6: match exact field diagnostic so 'season' in tmp_path cannot mask swapped check order."""
    p = tmp_path / "p" / "account_state" / "contest_streak.json"
    p.parent.mkdir(parents=True)
    p.write_text(json.dumps({
        "schema_version": "bts_contest_streak_auto_v2",
        "active_streak": 5,
        "best_streak": 8,
        "season": True,
        "activation_id": "INVALID",
    }), encoding="utf-8")
    with pytest.raises(cs.ContestStateError, match="automatic contest season invalid"):
        cs._parse_state_file(p)


def test_between_profile_and_rounds_dec31_clock_advance_preserves_entry_year(tmp_path, monkeypatch):
    """Item 3: advancing wall clock across midnight ET between profile and rounds still stamps 2027."""
    picks = tmp_path / "picks"
    acc = picks / "account_state"
    acc.mkdir(parents=True)
    (acc / "season_contract.json").write_text(json.dumps(_contract(season=2027)), encoding="utf-8")

    clk = _pin(monkeypatch, REAL_DATETIME(2027, 12, 31, 23, 59, 59, 500000, tzinfo=ET))
    earn = Mock(wraps=saver.maybe_auto_earn_saver)
    monkeypatch.setattr(saver, "maybe_auto_earn_saver", earn)

    def profile_then_midnight(*_):
        clk.current = REAL_DATETIME(2028, 1, 1, 0, 0, 0, 500000, tzinfo=ET)
        return {"activeStreak": 4, "seasonBestStreak": 6, "predictions": [{"roundId": 1, "result": "hit"}]}

    monkeypatch.setattr(cf, "fetch_profile", profile_then_midnight)
    monkeypatch.setattr(cli, "_fetch_rounds", lambda: {1: dt.date(2027, 12, 31)})

    res = CliRunner().invoke(cli.cli, ["fetch-contest-streak", "--picks-dir", str(picks)])
    assert res.exit_code == 0, res.output
    auto_doc = json.loads((acc / "contest_streak.json").read_text(encoding="utf-8"))
    led_doc = json.loads((acc / "contest_ledger.jsonl").read_text(encoding="utf-8").strip())
    assert auto_doc["season"] == led_doc["season"] == 2027
    assert auto_doc["recorded_at"] == led_doc["recorded_at"] == "2028-01-01T04:59:59.500000Z"
    assert earn.call_count == 1 and earn.call_args.kwargs["season"] == 2027
