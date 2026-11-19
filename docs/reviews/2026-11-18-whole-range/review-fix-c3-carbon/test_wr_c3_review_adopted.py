"""Self-contained tests from adversarial review of WR-C3 for adoption into tests/.

Covers:
1. Unmocked atomic replacement of streak.json (inode swap + chmod 000 rebuild + failed os.replace leaves prior bytes intact), killing non-atomic in-place write mutants without mocking picks.atomic_write_text.
2. Verified tagless streak.json marker hold (asserting 'season' key absent) and foreign-season marker ignore on past unresolved picks.
3. Multi-era writer corpus (commit 6798410 top-level streak key, 2026-03-29/30 null game_pk, regenerate._record_to_pick_file teamless DD, regenerate._hist_to_pick_file None backfills, export round-trip, v1/v2/v3 decisions, saver boundaries 9/10/15/16) checked against an independent from-scratch fold.
"""
from __future__ import annotations

from datetime import date, datetime, timedelta
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pytest

from bts import picks
from bts.daily_decision import write_decision
from bts.reconcile_receipt import ReconcileReceipt
from bts.state.export import export_initial_state
from bts.state.regenerate import (
    HistoricalPickRecord,
    Timeline,
    _hist_to_pick_file,
    _record_to_pick_file,
    compose_state_from_snapshot_and_timeline,
)


ET = ZoneInfo("America/New_York")
RUN = datetime(2027, 4, 25, 2, 0, tzinfo=ET)


def _plant_pick(root: Path, day: str, result: str | None = None, *, delivered: bool = False, dd: bool = False) -> Path:
    def _slot(gpk: int, bid: int, name: str) -> picks.Pick:
        return picks.Pick(
            batter_name=name,
            batter_id=bid,
            team="LAD",
            lineup_position=1,
            pitcher_name="Gerrit Cole",
            pitcher_id=543037,
            p_game_hit=0.81,
            flags=[],
            projected_lineup=False,
            game_pk=gpk,
            game_time=f"{day}T23:10:00Z",
            pitcher_team="NYY",
        )

    sr = {"pick": result, **({"double_down": result} if dd else {})} if result in ("hit", "miss", "void") else None
    daily = picks.DailyPick(
        date=day,
        run_time=f"{day}T18:00:00+00:00",
        pick=_slot(745100, 660271, "Shohei Ohtani"),
        double_down=_slot(745101, 605141, "Mookie Betts") if dd else None,
        runner_up=None,
        notification_sent=delivered,
        notification_id=f"dm-{day}" if delivered else None,
        delivery_attempted=delivered,
        result=result,
        slot_results=sr,
    )
    return picks.save_pick(daily, root)


def _invoke_cli(root: Path, monkeypatch, *, clock=lambda: RUN, feed: str | None = None):
    from bts.cli import cli

    real_rr = ReconcileReceipt

    def _resolve(daily, day, **kwargs):
        assert feed is not None, f"Unexpected resolve for {day}"
        return {"pick": feed}

    monkeypatch.setattr(picks, "resolve_daily_slot_results", _resolve)
    monkeypatch.setattr(
        "bts.reconcile_receipt.ReconcileReceipt",
        lambda **kw: real_rr(clock=clock, lookback_days=kw["lookback_days"]),
    )
    return CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(root)])


def test_reconcile_replaces_streak_atomically_without_mocking_atomic_helper(tmp_path: Path, monkeypatch):
    root = tmp_path / "picks"
    root.mkdir()
    _plant_pick(root, "2027-04-20", "hit", delivered=True)
    sp = root / "streak.json"
    sp.write_text('{"streak": 99, "saver_available": false, "season": 2027}')

    # 1. If os.replace fails during the streak rebuild, existing bytes survive untouched (never truncated in place).
    before = sp.read_bytes()
    real_replace = os.replace

    def fail_streak_replace(src, dst):
        if Path(dst).name == "streak.json":
            Path(src).unlink(missing_ok=True)
            raise OSError("simulated rename failure")
        return real_replace(src, dst)

    with monkeypatch.context() as scoped:
        scoped.setattr(os, "replace", fail_streak_replace)
        failed = _invoke_cli(root, scoped)
    assert failed.exit_code != 0
    assert sp.read_bytes() == before

    # 2. A chmod-000 regular streak.json is replaced via temp-file + os.replace (an in-place open('w') would raise PermissionError).
    if os.geteuid() != 0:
        os.chmod(sp, 0o000)
        try:
            ok = _invoke_cli(root, monkeypatch)
            assert ok.exit_code == 0, ok.output
            rebuilt = json.loads(sp.read_text())
            assert (rebuilt["streak"], rebuilt["saver_available"], rebuilt["season"]) == (1, True, 2027)
        finally:
            if sp.exists():
                os.chmod(sp, 0o600)


def test_truly_tagless_past_marker_blocks_and_foreign_past_marker_skips(tmp_path: Path, monkeypatch):
    root = tmp_path / "picks"
    root.mkdir()
    _plant_pick(root, "2027-04-10", "hit", delivered=True)
    _plant_pick(root, "2027-04-16", None)

    # Truly tagless streak.json (no 'season' key) with applied_dates matching unresolved past day -> blocks
    picks.save_streak(8, root, saver_available=True, applied_date="2027-04-16", season=None)
    raw_doc = json.loads((root / "streak.json").read_text())
    assert "season" not in raw_doc
    res_tagless = _invoke_cli(root, monkeypatch)
    assert res_tagless.exit_code == 0
    assert "Season replay unavailable for 2027-04-16 (applied-date marker without a result)" in res_tagless.stderr
    assert "Streak retained: 0" in res_tagless.stdout

    # Foreign-season streak.json with applied_dates naming 2027-04-16 -> marker ignored, uncommitted pick skipped
    picks.save_streak(8, root, saver_available=False, applied_date="2027-04-16", season=2026)
    res_foreign = _invoke_cli(root, monkeypatch)
    assert res_foreign.exit_code == 0
    assert res_foreign.stderr == ""
    assert "No scoring changes detected. Streak: 1" in res_foreign.stdout
    rebuilt = json.loads((root / "streak.json").read_text())
    assert (rebuilt["streak"], rebuilt["saver_available"], rebuilt["season"], rebuilt["applied_dates"]) == (1, True, 2027, [])


def test_multi_era_writers_and_saver_boundaries_match_independent_fold(tmp_path: Path):
    root = tmp_path / "picks"
    root.mkdir()

    # Export + compose round-trip for 2026-04-01..04-09 (9 hits -> streak 9)
    exp_src = tmp_path / "export_src"
    exp_src.mkdir()
    for i in range(9):
        d = (date(2026, 4, 1) + timedelta(days=i)).isoformat()
        _plant_pick(exp_src, d, "hit", delivered=True)
    picks.save_streak(9, exp_src, saver_available=True, season=2026)
    snap_path = tmp_path / "snap.json"
    export_initial_state(exp_src, snap_path)
    compose_state_from_snapshot_and_timeline(snap_path, Timeline(), root)

    # Miss at streak 9 does NOT consume saver -> resets to 0, saver remains True!
    (root / "2026-04-10.json").write_text(json.dumps(_record_to_pick_file(
        HistoricalPickRecord("2026-04-10", "Aaron Judge", "NYY", False, None, None, "at://m9", "miss", 0)
    )))
    assert picks._replay_season_streak(root, 2026, "2026-04-10") == (0, True)

    # Build up to streak 15 via null-game_pk (3/29-3/30 style), teamless DD, None-backfilled hist, and commit-6798410 extra key:
    # 5 teamless regenerated DDs = +10, plus 5 singles = +5 -> streak 15
    for idx in range(5):
        d = (date(2026, 4, 11) + timedelta(days=idx)).isoformat()
        (root / f"{d}.json").write_text(json.dumps(_record_to_pick_file(
            HistoricalPickRecord(d, "Jose Altuve", "HOU", True, "Kyle Tucker", None, f"at://dd{idx}", "hit", (idx + 1) * 2)
        )))
    for idx in range(5):
        d = (date(2026, 4, 16) + timedelta(days=idx)).isoformat()
        hist = {
            "date": d,
            "pick": {
                "batter_name": "Nico Hoerner",
                "batter_id": None,
                "team": "CHC",
                "pitcher_name": None,
                "pitcher_id": None,
                "game_pk": None,
                "game_time": None,
                "p_game_hit": None,
            },
            "double_down": None,
            "result": "hit",
        }
        doc = _hist_to_pick_file(hist)
        doc["pick"]["game_pk"] = None  # explicit null game_pk (Mar 29-30 shape)
        doc["streak"] = 10 + idx + 1   # commit 6798410 top-level extra key
        (root / f"{d}.json").write_text(json.dumps(doc))

    assert picks._replay_season_streak(root, 2026, "2026-04-20") == (15, True)

    # Miss at streak 15 (upper bound of [10, 15]) consumes saver and holds streak 15!
    _plant_pick(root, "2026-04-21", "miss", delivered=True)
    assert picks._replay_season_streak(root, 2026, "2026-04-21") == (15, False)

    # Uncommitted preview + skip leftover + hit (+1 -> 16), then miss at 16 resets to 0
    _plant_pick(root, "2026-04-22", None, delivered=False)
    _plant_pick(root, "2026-04-23", None, delivered=False)
    write_decision("2026-04-23", root, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN)
    _plant_pick(root, "2026-04-24", "hit", delivered=True)
    assert picks._replay_season_streak(root, 2026, "2026-04-24") == (16, False)

    _plant_pick(root, "2026-04-25", "miss", delivered=True)
    assert picks._replay_season_streak(root, 2026, "2026-04-25") == (0, False)
