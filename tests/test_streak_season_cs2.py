"""CS2 contract: model state belongs to an explicit calendar season.

Real legacy bytes deliberately have no inferred season. No read may migrate them.
"""
import json
from contextlib import nullcontext
from datetime import datetime
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

import pytest
from click.testing import CliRunner

from bts import picks as p

ET = ZoneInfo("America/New_York")
LEGACY = b'{"streak": 13, "saver_available": false, "updated": "2026-09-13T05:01:02.123456+00:00", "applied_dates": ["2026-09-11", "2026-09-12"]}'


def plant(d, season=None, **changes):
    data = json.loads(LEGACY)
    if season is not None:
        data["season"] = season
    data.update(changes)
    (d / "streak.json").write_text(json.dumps(data))


def snapshot(d):
    return json.loads((d / "streak.json").read_text())


@pytest.mark.parametrize("tag", [None, 2026, True, "2027", 2027.0, 2027])
def test_row1_readers_require_an_exact_integer_season_without_writing(tmp_path, tag):
    plant(tmp_path, tag)
    before = (tmp_path / "streak.json").read_bytes()
    assert (p.load_streak(tmp_path), p.load_saver_available(tmp_path),
            p.load_applied_dates(tmp_path)) == (13, False, ["2026-09-11", "2026-09-12"])
    expected = (13, False, ["2026-09-11", "2026-09-12"]) if type(tag) is int and tag == 2027 else (0, True, [])
    assert (p.load_streak(tmp_path, season=2027), p.load_saver_available(tmp_path, season=2027),
            p.load_applied_dates(tmp_path, season=2027)) == expected
    assert (tmp_path / "streak.json").read_bytes() == before


@pytest.mark.parametrize("raw, error", [(b"{", json.JSONDecodeError), (b"\xff", UnicodeDecodeError),
                                       (b"[]", AttributeError)])
def test_row1_dateless_corruption_behavior_is_unchanged(tmp_path, raw, error):
    (tmp_path / "streak.json").write_bytes(raw)
    for loader in (p.load_streak, p.load_saver_available):
        with pytest.raises(error):
            loader(tmp_path)
    assert p.load_applied_dates(tmp_path) == []


@pytest.mark.parametrize("raw", [b"{", b"\xff", b"[]", b"null", b"7"])
def test_row1_nonobject_or_unparseable_file_cannot_match_season(tmp_path, raw):
    (tmp_path / "streak.json").write_bytes(raw)
    assert (p.load_streak(tmp_path, season=2027), p.load_saver_available(tmp_path, season=2027),
            p.load_applied_dates(tmp_path, season=2027)) == (0, True, [])
    assert (tmp_path / "streak.json").read_bytes() == raw


def test_row1_missing_file_and_missing_keys_keep_defaults(tmp_path):
    for raw in (None, b'{"season": 2027}'):
        if raw is not None:
            (tmp_path / "streak.json").write_bytes(raw)
        assert (p.load_streak(tmp_path, season=2027), p.load_saver_available(tmp_path, season=2027),
                p.load_applied_dates(tmp_path, season=2027)) == (0, True, [])


@pytest.mark.parametrize("tag", [None, 2026, True, "2027", 2027.0, 2027])
def test_row2_save_resets_only_preserved_state_on_mismatch(tmp_path, tag):
    plant(tmp_path, tag)
    p.save_streak(4, tmp_path, applied_date="2027-03-24", season=2027)
    data = snapshot(tmp_path)
    match = type(tag) is int and tag == 2027
    assert data["season"] == 2027 and type(data["season"]) is int
    assert data["streak"] == 4
    assert data["saver_available"] is (False if match else True)
    assert data["applied_dates"] == (["2026-09-11", "2026-09-12", "2027-03-24"] if match else ["2027-03-24"])
    p.save_streak(5, tmp_path, saver_available=False, season=2028)
    assert snapshot(tmp_path)["saver_available"] is False
    assert snapshot(tmp_path)["applied_dates"] == []


@pytest.mark.parametrize("tag", [2027, True, "2027", 2027.0, None])
@pytest.mark.parametrize("saver", [None, False])
def test_row2_dateless_save_preserves_tag_verbatim(tmp_path, tag, saver):
    plant(tmp_path)
    data = snapshot(tmp_path)
    data["season"] = tag
    (tmp_path / "streak.json").write_text(json.dumps(data))
    p.save_streak(9, tmp_path, saver_available=saver)
    saved = snapshot(tmp_path)
    assert "season" in saved and saved["season"] == tag and type(saved["season"]) is type(tag)
    assert saved["applied_dates"] == data["applied_dates"]
    assert saved["saver_available"] is False


@pytest.mark.parametrize("results, expected", [([True], 1), ([True, True], 2), ([False], 0)])
def test_row3_legacy_rollover_then_same_date_retry(tmp_path, results, expected):
    (tmp_path / "streak.json").write_bytes(LEGACY)
    assert p.update_streak(results, tmp_path, date="2027-03-24") == expected
    data = snapshot(tmp_path)
    assert (data["streak"], data["saver_available"], data["season"], data["applied_dates"]) == (
        expected, True, 2027, ["2027-03-24"])
    before = (tmp_path / "streak.json").read_bytes()
    assert p.update_streak(results, tmp_path, date="2027-03-24") == expected
    assert (tmp_path / "streak.json").read_bytes() == before


def test_row3_matching_season_retains_saver_forgiveness(tmp_path):
    plant(tmp_path, 2027, streak=12, saver_available=True, applied_dates=[])
    assert p.update_streak([False], tmp_path, date="2027-03-24") == 12
    assert snapshot(tmp_path)["saver_available"] is False
    assert snapshot(tmp_path)["season"] == 2027


def test_row3_tagless_three_starts_at_one_instead_of_carrying_over(tmp_path):
    (tmp_path / "streak.json").write_text('{"streak": 3, "saver_available": true}')
    assert p.update_streak([True], tmp_path, date="2027-04-02") == 1
    assert snapshot(tmp_path)["season"] == 2027


@pytest.mark.parametrize("day", ["not-a-date", "2027-02-29", "20270324", "2027-3-24", "2027-03-24T00:00:00"])
def test_row3_invalid_date_fails_before_io(tmp_path, day):
    with patch.object(p, "load_streak", side_effect=AssertionError("read before validation")), \
         patch.object(p, "save_streak", side_effect=AssertionError("write before validation")):
        with pytest.raises(ValueError):
            p.update_streak([True], tmp_path, date=day)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("tag", [2027, "2027", True, 2027.0])
def test_row4_dateless_update_preserves_tag_and_state(tmp_path, tag):
    plant(tmp_path, tag, streak=5)
    assert p.update_streak([True], tmp_path) == 6
    assert snapshot(tmp_path)["season"] == tag and type(snapshot(tmp_path)["season"]) is type(tag)
    assert snapshot(tmp_path)["saver_available"] is False
    expected = 7 if type(tag) is int else 1
    assert p.update_streak([True], tmp_path, date="2027-03-24") == expected
    assert snapshot(tmp_path)["season"] == 2027


def test_row4_tagless_dateless_update_keeps_legacy_state(tmp_path):
    (tmp_path / "streak.json").write_bytes(LEGACY)
    assert p.update_streak([True], tmp_path) == 14
    assert "season" not in snapshot(tmp_path)
    assert snapshot(tmp_path)["applied_dates"] == ["2026-09-11", "2026-09-12"]


def resolved_pick(d, day="2027-03-23", result="hit"):
    from tests.test_scoring_lock import _plant_scoreable_pick
    _plant_scoreable_pick(d, day)
    daily = p.load_pick(day, d)
    daily.result = result
    daily.slot_results = {"pick": result} if result else None
    p.save_pick(daily, d)


def test_row5_reconcile_stamps_replayed_season_and_discards_old_markers(tmp_path):
    plant(tmp_path, 2026, streak=20)
    resolved_pick(tmp_path)
    p.reconcile_results(tmp_path, clock=lambda: datetime(2027, 3, 25, 2, tzinfo=ET))
    data = snapshot(tmp_path)
    assert (data["streak"], data["saver_available"], data["season"], data["applied_dates"]) == (1, True, 2027, [])


def test_row5_tagless_same_day_marker_still_blocks_incomplete_replay(tmp_path):
    plant(tmp_path, streak=14, applied_dates=["2027-03-24"])
    resolved_pick(tmp_path, "2027-03-24", None)
    before = (tmp_path / "streak.json").read_bytes()
    assert p._replay_season_streak(tmp_path, 2027, "2027-03-24") is None
    p.reconcile_results(tmp_path, clock=lambda: datetime(2027, 3, 24, 23, tzinfo=ET))
    assert (tmp_path / "streak.json").read_bytes() == before


@pytest.mark.parametrize("tag", [None, 2026])
def test_row6_cli_void_display_reads_scored_season(tmp_path, monkeypatch, tag):
    from bts.cli import cli
    plant(tmp_path, tag)
    resolved_pick(tmp_path, "2027-03-24", None)
    monkeypatch.setattr(p, "resolve_daily_slot_results", lambda *a: {"pick": "void"})
    result = CliRunner().invoke(cli, ["check-results", "--date", "2027-03-24", "--picks-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert "All picks void. Streak unchanged: 0" in result.output
    assert snapshot(tmp_path)["streak"] == 13  # display is read-only


@pytest.mark.parametrize("mode", ["void", "final_peer", "midgame_peer"])
def test_row6_scheduler_display_reads_scored_season(tmp_path, monkeypatch, capsys, mode):
    from bts import scheduler as s
    plant(tmp_path, 2026)
    resolved_pick(tmp_path, "2027-03-24", None)
    monkeypatch.setattr(s, "_now_et", lambda: datetime(2027, 3, 24, 23, tzinfo=ET))
    monkeypatch.setattr(s, "poll_game_result", lambda *a: "live" if mode == "midgame_peer" else "final")
    monkeypatch.setattr(p, "get_game_statuses_detailed", lambda *a: {})
    def peer(*a):
        daily = p.load_pick("2027-03-24", tmp_path)
        daily.result = "hit"
        p.save_pick(daily, tmp_path)
        return [True]
    if mode == "midgame_peer":
        monkeypatch.setattr(s, "_check_hits_midgame", peer)
        # End polling after the early reply; no sleeping or network.
        monkeypatch.setattr(s, "_poll_interval_sleep", lambda *a, **k: None)
        polls = iter(["live", "final"])
        monkeypatch.setattr(s, "poll_game_result", lambda *a: next(polls))
        daily = p.load_pick("2027-03-24", tmp_path)
        daily.bluesky_uri = "at://test"
        p.save_pick(daily, tmp_path)
    def resolve(*a):
        if mode == "final_peer":
            peer()
        return {"pick": "void"}
    monkeypatch.setattr(p, "resolve_daily_slot_results", resolve)
    with patch("bts.posting.format_result_reply", return_value="reply") as reply, \
         patch("bts.posting.reply_to_bluesky"):
        assert s.run_result_polling(111, "2027-03-24", tmp_path) == "final"
    if mode == "midgame_peer":
        reply.assert_called_once_with("hit", 0)
    else:
        assert "Streak: 0" in capsys.readouterr().err


@pytest.mark.parametrize("corrections", [[], [{"date": "2027-03-23", "batter": "B", "old_result": "hit", "new_result": "miss"}]])
def test_row7_reconcile_summary_uses_replay_et_clock(tmp_path, monkeypatch, corrections):
    import bts.cli as c
    from bts.reconcile_receipt import ReconcileReceipt
    plant(tmp_path, 2026)
    # UTC has already crossed New Year, ET has not. Replay is blocked in this probe.
    fixed = datetime.fromisoformat("2028-01-01T01:00:00+00:00")
    real_receipt = ReconcileReceipt
    monkeypatch.setattr("bts.reconcile_receipt.ReconcileReceipt",
                        lambda **kw: real_receipt(clock=lambda: fixed, lookback_days=kw["lookback_days"]))
    monkeypatch.setattr(p, "reconcile_results", lambda *a, **k: corrections)
    with patch.object(p, "load_streak", wraps=p.load_streak) as load:
        result = CliRunner().invoke(c.cli, ["reconcile", "--picks-dir", str(tmp_path)])
    assert result.exit_code == 0, result.output
    load.assert_called_once_with(tmp_path, season=2027)
    assert ("Streak: 0" if not corrections else "Streak recalculated: 0") in result.output


def test_row8_shadow_selection_uses_own_date_season(tmp_path, monkeypatch):
    from bts import scheduler as s
    plant(tmp_path, 2026)
    seen = []
    monkeypatch.setattr(s, "predict_local_shadow", lambda *a, **k: object())
    monkeypatch.setattr(s, "heartbeat_watchdog", lambda *a, **k: nullcontext())
    monkeypatch.setattr(s, "_analytics_job_status", lambda *a: {})
    monkeypatch.setattr(s, "_update_analytics_job_status", lambda *a, **k: None)
    monkeypatch.setattr(p, "get_game_statuses_detailed", lambda *a: {})
    def select(*a, **kw):
        seen.append(kw["streak"])
        return SimpleNamespace(pick_result=None)
    monkeypatch.setattr(s, "select_pick", select)
    s._run_shadow_prediction({"orchestrator": {"picks_dir": str(tmp_path)}}, "2027-03-24", "Prod")
    assert seen == [0]


def test_row9_strategy_saver_fallback_uses_selection_date(tmp_path):
    from bts import strategy as s
    from tests.test_strategy import _predictions
    plant(tmp_path, 2026)
    preds = _predictions([{"p_game_hit": .85}])
    with patch.object(s, "get_game_statuses", return_value={778899: "P"}), \
         patch.object(s, "_load_mdp", return_value=None), \
         patch.object(s, "resolve_policy_decision", wraps=s.resolve_policy_decision) as decide:
        s.select_pick(preds, "2027-03-24", tmp_path, streak=13)
    assert decide.call_count == 1
    assert decide.call_args.args[2] is True


@pytest.mark.parametrize("tag, expected", [(None, 0), (2026, 0), (2027, 13), (True, 0), ("2027", 0), (2027.0, 0)])
@pytest.mark.parametrize("contest", [False, True])
def test_row10_model_diagnostics_reset_without_changing_contest_or_saver_authority(tmp_path, tag, expected, contest):
    from bts.contest_state import load_decision_streak_state
    plant(tmp_path, tag)
    if contest:
        state_dir = tmp_path / "account_state"
        state_dir.mkdir()
        (state_dir / "contest_streak.json").write_text(json.dumps({
            "schema_version": "bts_contest_streak_auto_v1", "active_streak": 7,
            "best_streak": 7, "source": "mlb_bts_profile", "source_date": "2027-03-24"}))
    before = (tmp_path / "streak.json").read_bytes()
    state = load_decision_streak_state(tmp_path, require_contest_state=False,
                                       now=datetime(2027, 3, 24, 23, tzinfo=ET))
    assert state.streak == (7 if contest else expected)
    assert state.model_streak == expected
    assert state.model_saver_available is (False if expected else True)
    assert state.model_state_error is None
    assert state.saver_available is False  # uninitialized saver_state, unaffected by CS2
    assert (tmp_path / "streak.json").read_bytes() == before


def test_row11_tagless_reads_do_not_migrate_then_first_dated_write_stamps(tmp_path):
    (tmp_path / "streak.json").write_bytes(LEGACY)
    before = (tmp_path / "streak.json").read_bytes()
    assert p.load_streak(tmp_path, season=2027) == 0
    assert p.load_saver_available(tmp_path, season=2027) is True
    assert p.load_applied_dates(tmp_path, season=2027) == []
    assert (tmp_path / "streak.json").read_bytes() == before
    p.update_streak([True], tmp_path, date="2027-03-24")
    assert snapshot(tmp_path)["season"] == 2027


def test_row12_tagged_file_still_validates_and_exports(tmp_path):
    from bts.health.streak_validation import check
    from bts.state.export import export_initial_state
    plant(tmp_path, 2027)
    assert check(tmp_path) == []
    exported = export_initial_state(tmp_path, tmp_path / "export.json")
    assert (exported["streak_at_cutoff"], exported["saver_available"]) == (13, False)
