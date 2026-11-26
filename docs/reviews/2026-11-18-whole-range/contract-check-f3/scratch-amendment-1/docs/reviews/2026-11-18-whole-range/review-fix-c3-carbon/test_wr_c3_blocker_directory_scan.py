"""RED regression tests for the _fold_season directory-discovery and symlink-parent blocker
(for adoption into tests/ once _fold_season uses os.scandir and checks day_dir lstat S_ISDIR)."""
from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
from zoneinfo import ZoneInfo

from click.testing import CliRunner
import pytest

from bts import picks
from bts.daily_decision import write_decision
from bts.reconcile_receipt import ReconcileReceipt


ET = ZoneInfo("America/New_York")
RUN = datetime(2027, 4, 20, 2, 0, tzinfo=ET)


def _plant_pick(root: Path, day: str, result: str | None = None, *, delivered: bool = True) -> Path:
    slot = picks.Pick(
        batter_name="Shohei Ohtani",
        batter_id=660271,
        team="LAD",
        lineup_position=1,
        pitcher_name="Gerrit Cole",
        pitcher_id=543037,
        p_game_hit=0.81,
        flags=[],
        projected_lineup=False,
        game_pk=745100,
        game_time=f"{day}T23:10:00Z",
        pitcher_team="NYY",
    )
    daily = picks.DailyPick(
        date=day,
        run_time=f"{day}T18:00:00+00:00",
        pick=slot,
        double_down=None,
        runner_up=None,
        notification_sent=delivered,
        notification_id=f"dm-{day}" if delivered else None,
        delivery_attempted=delivered,
        result=result,
        slot_results={"pick": result} if result in ("hit", "miss", "void") else None,
    )
    return picks.save_pick(daily, root)


def _invoke(root: Path, monkeypatch):
    from bts.cli import cli

    real_rr = ReconcileReceipt
    monkeypatch.setattr(picks, "resolve_daily_slot_results", lambda *a, **k: {"pick": "hit"})
    monkeypatch.setattr(
        "bts.reconcile_receipt.ReconcileReceipt",
        lambda **kw: real_rr(clock=lambda: RUN, lookback_days=kw["lookback_days"]),
    )
    return CliRunner().invoke(cli, ["reconcile", "--picks-dir", str(root)])


def test_unsearchable_decision_directory_without_root_pick_blocks_replay(tmp_path: Path, monkeypatch):
    if os.geteuid() == 0:
        pytest.skip("root bypasses chmod 000")
    root = tmp_path / "picks"
    root.mkdir()
    _plant_pick(root, "2027-04-01", "hit")
    write_decision("2027-04-16", root, action="single", source="mdp", delivery_status="delivered", scoreable=True, now=RUN)
    picks.save_streak(9, root, saver_available=False, season=2027)
    os.chmod(root / "2027-04-16", 0o000)
    try:
        res = _invoke(root, monkeypatch)
        assert res.exit_code == 0, res.output
        assert "Season replay unavailable for 2027-04-16 (unreadable or invalid decision record)" in res.stderr
        assert "Streak retained: 9" in res.stdout
        st = json.loads((root / "streak.json").read_text())
        assert (st["streak"], st["saver_available"]) == (9, False)
    finally:
        os.chmod(root / "2027-04-16", 0o700)


def test_non_directory_or_symlink_at_decision_parent_blocks_replay(tmp_path: Path, monkeypatch):
    write_decision("2027-04-16", tmp_path, action="skip", source="mdp", delivery_status="not_applicable", scoreable=False, now=RUN)

    # 1. Symlinked date directory cannot supply a skip decision over a delivered pending pick.
    p_sym = tmp_path / "sym_picks"
    p_sym.mkdir()
    _plant_pick(p_sym, "2027-04-01", "hit")
    _plant_pick(p_sym, "2027-04-16", None, delivered=True)
    (p_sym / "2027-04-16").symlink_to(tmp_path / "2027-04-16")
    picks.save_streak(9, p_sym, saver_available=False, season=2027)
    r_sym = _invoke(p_sym, monkeypatch)
    assert r_sym.exit_code == 0
    assert "Season replay unavailable for 2027-04-16 (unreadable or invalid decision record)" in r_sym.stderr
    assert json.loads((p_sym / "streak.json").read_text())["streak"] == 9

    # 2. FIFO at picks/<date> without a root pick file is a dated blocker, never silently ignored.
    p_fifo = tmp_path / "fifo_picks"
    p_fifo.mkdir()
    _plant_pick(p_fifo, "2027-04-01", "hit")
    os.mkfifo(p_fifo / "2027-04-16")
    picks.save_streak(9, p_fifo, saver_available=False, season=2027)
    r_fifo = _invoke(p_fifo, monkeypatch)
    assert r_fifo.exit_code == 0
    assert "Season replay unavailable for 2027-04-16 (unreadable or invalid decision record)" in r_fifo.stderr
    assert json.loads((p_fifo / "streak.json").read_text())["streak"] == 9


def test_unreadable_picks_dir_in_library_call_preserves_streak(tmp_path: Path):
    if os.geteuid() == 0:
        pytest.skip("root bypasses chmod 0300")
    root = tmp_path / "picks"
    root.mkdir()
    _plant_pick(root, "2027-04-01", "hit")
    picks.save_streak(10, root, saver_available=False, season=2027)
    os.chmod(root, 0o300)
    try:
        with pytest.raises(OSError):
            picks.reconcile_results(root, clock=lambda: RUN)
        st = json.loads((root / "streak.json").read_text())
        assert (st["streak"], st["saver_available"]) == (10, False)
    finally:
        os.chmod(root, 0o700)
