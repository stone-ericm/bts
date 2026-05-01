"""Tests for lineup_evolution audit log appended by save_pick.

Goal: every save_pick call should also append one line to
lineup_evolution_{date}.jsonl. Through the day, this file becomes the
trajectory of pick choices across lineup confirmations, supporting
gap #6 analysis (does morning projected-lineup pick underperform?).
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from bts.picks import DailyPick, Pick, append_lineup_evolution, save_pick


def _pick(name: str, p: float, projected: bool, batter_id: int = 100) -> Pick:
    return Pick(
        batter_name=name,
        batter_id=batter_id,
        team="NYM",
        lineup_position=2,
        pitcher_name="Some Pitcher",
        pitcher_id=999,
        p_game_hit=p,
        flags=["PROJECTED"] if projected else [],
        projected_lineup=projected,
        game_pk=12345,
        game_time="2026-05-01T19:10:00+00:00",
        pitcher_team="WSH",
    )


def _daily(date_str: str, primary: Pick, dd: Pick | None = None) -> DailyPick:
    return DailyPick(
        date=date_str,
        run_time="2026-05-01T14:00:00+00:00",
        pick=primary,
        double_down=dd,
        runner_up=None,
    )


class TestAppendLineupEvolution:
    def test_first_call_creates_file_with_one_line(self, tmp_path):
        daily = _daily("2026-05-01", _pick("Juan Soto", 0.83, projected=True))
        path = append_lineup_evolution(daily, tmp_path)
        assert path.exists()
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 1
        entry = json.loads(lines[0])
        assert entry["date"] == "2026-05-01"
        assert entry["primary"]["batter_name"] == "Juan Soto"
        assert entry["primary"]["projected_lineup"] is True
        assert entry["double_down"] is None

    def test_subsequent_calls_append(self, tmp_path):
        daily1 = _daily("2026-05-01", _pick("Juan Soto", 0.83, projected=True))
        daily2 = _daily("2026-05-01", _pick("Juan Soto", 0.85, projected=False))
        append_lineup_evolution(daily1, tmp_path)
        append_lineup_evolution(daily2, tmp_path)
        path = tmp_path / "lineup_evolution_2026-05-01.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["primary"]["projected_lineup"] is True
        assert json.loads(lines[1])["primary"]["projected_lineup"] is False

    def test_captures_double_down(self, tmp_path):
        primary = _pick("Juan Soto", 0.83, projected=False, batter_id=1)
        dd = _pick("Vlad Jr.", 0.80, projected=True, batter_id=2)
        daily = _daily("2026-05-01", primary, dd)
        append_lineup_evolution(daily, tmp_path)
        line = (tmp_path / "lineup_evolution_2026-05-01.jsonl").read_text().strip()
        entry = json.loads(line)
        assert entry["primary"]["batter_id"] == 1
        assert entry["double_down"]["batter_id"] == 2
        assert entry["double_down"]["projected_lineup"] is True

    def test_captured_at_is_iso_utc(self, tmp_path):
        daily = _daily("2026-05-01", _pick("Juan Soto", 0.83, projected=True))
        append_lineup_evolution(daily, tmp_path)
        line = (tmp_path / "lineup_evolution_2026-05-01.jsonl").read_text().strip()
        captured = json.loads(line)["captured_at"]
        assert captured.endswith("+00:00") or captured.endswith("Z")

    def test_separate_dates_separate_files(self, tmp_path):
        d1 = _daily("2026-05-01", _pick("Juan Soto", 0.83, projected=True))
        d2 = _daily("2026-05-02", _pick("Vlad Jr.", 0.79, projected=True))
        append_lineup_evolution(d1, tmp_path)
        append_lineup_evolution(d2, tmp_path)
        assert (tmp_path / "lineup_evolution_2026-05-01.jsonl").exists()
        assert (tmp_path / "lineup_evolution_2026-05-02.jsonl").exists()


class TestSavePickAlsoAppendsLog:
    def test_save_pick_writes_both_pick_json_and_evolution_log(self, tmp_path):
        daily = _daily("2026-05-01", _pick("Juan Soto", 0.83, projected=True))
        save_pick(daily, tmp_path)
        assert (tmp_path / "2026-05-01.json").exists()
        assert (tmp_path / "lineup_evolution_2026-05-01.jsonl").exists()

    def test_save_pick_evolution_log_failure_does_not_break_save(self, tmp_path, monkeypatch):
        """If the audit log append raises, save_pick must still succeed."""
        daily = _daily("2026-05-01", _pick("Juan Soto", 0.83, projected=True))

        def explode(*a, **kw):
            raise OSError("disk full or something")

        monkeypatch.setattr("bts.picks.append_lineup_evolution", explode)
        # Should not raise — exception is swallowed
        path = save_pick(daily, tmp_path)
        assert path.exists()
        # Audit log was NOT created (because the function raised), but pick file IS there
        assert not (tmp_path / "lineup_evolution_2026-05-01.jsonl").exists()

    def test_multiple_save_pick_calls_grow_evolution_log(self, tmp_path):
        d1 = _daily("2026-05-01", _pick("Juan Soto", 0.83, projected=True))
        d2 = _daily("2026-05-01", _pick("Juan Soto", 0.85, projected=False))
        d3 = _daily("2026-05-01", _pick("Juan Soto", 0.86, projected=False))
        save_pick(d1, tmp_path)
        save_pick(d2, tmp_path)
        save_pick(d3, tmp_path)
        path = tmp_path / "lineup_evolution_2026-05-01.jsonl"
        lines = path.read_text().strip().split("\n")
        assert len(lines) == 3
        # Latest line should reflect the latest pick state
        last = json.loads(lines[-1])
        assert last["primary"]["p_game_hit"] == 0.86
        assert last["primary"]["projected_lineup"] is False
