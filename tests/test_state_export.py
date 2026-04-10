"""Tests for bts state export."""
import json
from pathlib import Path

import pytest

from bts.state.export import export_initial_state, UnresolvedPickError


def _write_pick(picks_dir: Path, date: str, result: str | None, double_down: bool = False):
    pick = {
        "date": date,
        "run_time": f"{date}T12:00:00+00:00",
        "pick": {
            "batter_name": "Test Batter",
            "batter_id": 100,
            "team": "NYY",
            "pitcher_name": "Test Pitcher",
            "pitcher_id": 200,
            "game_pk": 12345,
            "game_time": f"{date}T19:05:00-04:00",
            "p_game_hit": 0.85,
            "p_hit_pa": 0.31,
            "projected_lineup": False,
        },
        "double_down": {
            "batter_name": "Other Batter",
            "batter_id": 101,
            "team": "BOS",
            "pitcher_name": "Other Pitcher",
            "pitcher_id": 201,
            "game_pk": 12346,
            "game_time": f"{date}T19:10:00-04:00",
            "p_game_hit": 0.80,
            "p_hit_pa": 0.28,
            "projected_lineup": False,
        } if double_down else None,
        "runner_up": None,
        "bluesky_posted": True,
        "bluesky_uri": f"at://did:test/app.bsky.feed.post/{date}",
        "result": result,
    }
    (picks_dir / f"{date}.json").write_text(json.dumps(pick))


def test_export_refuses_when_unresolved(tmp_path: Path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pick(picks_dir, "2026-04-01", "hit")
    _write_pick(picks_dir, "2026-04-02", None)  # Unresolved
    _write_pick(picks_dir, "2026-04-03", "hit")

    (picks_dir / "streak.json").write_text('{"streak": 2, "saver_available": true}')

    out_path = tmp_path / "initial-state.json"
    with pytest.raises(UnresolvedPickError, match="2026-04-02"):
        export_initial_state(picks_dir=picks_dir, output_path=out_path)


def test_export_succeeds_when_all_resolved(tmp_path: Path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pick(picks_dir, "2026-04-01", "hit")
    _write_pick(picks_dir, "2026-04-02", "miss")
    _write_pick(picks_dir, "2026-04-03", "hit")
    (picks_dir / "streak.json").write_text('{"streak": 1, "saver_available": true}')

    out_path = tmp_path / "initial-state.json"
    export_initial_state(picks_dir=picks_dir, output_path=out_path)

    exported = json.loads(out_path.read_text())
    assert exported["version"] == 1
    assert exported["cutoff_date"] == "2026-04-03"
    assert exported["streak_at_cutoff"] == 1
    assert exported["saver_available"] is True
    assert len(exported["historical_picks"]) == 3
    assert {p["date"] for p in exported["historical_picks"]} == {"2026-04-01", "2026-04-02", "2026-04-03"}


def test_export_excludes_non_date_files(tmp_path: Path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pick(picks_dir, "2026-04-01", "hit")
    (picks_dir / "streak.json").write_text('{"streak": 1, "saver_available": false}')
    (picks_dir / "notes.txt").write_text("not a pick")
    (picks_dir / "orchestrator.log").write_text("log data")

    out_path = tmp_path / "initial-state.json"
    export_initial_state(picks_dir=picks_dir, output_path=out_path)

    exported = json.loads(out_path.read_text())
    # Only the one pick, not notes.txt or orchestrator.log
    assert len(exported["historical_picks"]) == 1


def test_export_includes_bluesky_uri(tmp_path: Path):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pick(picks_dir, "2026-04-01", "hit")
    (picks_dir / "streak.json").write_text('{"streak": 1, "saver_available": false}')

    out_path = tmp_path / "initial-state.json"
    export_initial_state(picks_dir=picks_dir, output_path=out_path)

    exported = json.loads(out_path.read_text())
    assert exported["historical_picks"][0]["bluesky_uri"] == "at://did:test/app.bsky.feed.post/2026-04-01"
