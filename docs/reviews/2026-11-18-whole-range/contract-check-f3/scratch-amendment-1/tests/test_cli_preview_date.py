"""`bts preview` default-date timezone (GPT-5.6 audit F15; same class as O2).

Preview defaults to "tomorrow". If tomorrow is derived from UTC, a manual run
between ~8pm and midnight ET computes the day AFTER tomorrow (UTC has already
rolled over), so recovery/evening previews write the wrong slate. The default
must be tomorrow in US Eastern — the contest's timezone.

No mocks beyond the clock: resolved pick files are planted on disk for BOTH
candidate dates (ET-tomorrow and UTC-tomorrow) and the early-exit echo reveals
which date preview actually targeted.
"""
import json
from datetime import datetime as _dt
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner

from bts.cli import cli


def _plant_resolved_pick(picks_dir, date):
    """Minimal on-disk pick file that load_pick parses and preview treats as
    already-resolved (result set -> early exit before the heavy pipeline)."""
    payload = {
        "date": date,
        "run_time": f"{date}T07:00:00+00:00",
        "pick": {
            "batter_name": "Test Batter", "batter_id": 1, "team": "SF",
            "lineup_position": 1, "pitcher_name": "Test Pitcher",
            "pitcher_id": 2, "p_game_hit": 0.8, "flags": [],
            "projected_lineup": False, "game_pk": 111, "game_time": None,
            "pitcher_team": "LAD",
        },
        "double_down": None,
        "runner_up": None,
        "bluesky_posted": False,
        "bluesky_uri": None,
        "result": "hit",
    }
    (picks_dir / f"{date}.json").write_text(json.dumps(payload))


def _preview_default_target(fixed_now, tmp_path, et_tomorrow, utc_tomorrow):
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    for d in {et_tomorrow, utc_tomorrow}:
        _plant_resolved_pick(picks_dir, d)

    # Tripwire: if the computed default matches NO planted fixture, preview
    # falls through toward the real pipeline — fail loudly instead of running
    # a live data refresh (same patch seam the existing integration tests use).
    with patch("bts.cli.datetime") as mock_dt, \
            patch("bts.model.predict.run_pipeline",
                  side_effect=RuntimeError("tripwire: preview reached the "
                                           "pipeline — wrong default date")):
        mock_dt.now.return_value = fixed_now
        result = CliRunner().invoke(cli, [
            "preview",
            "--picks-dir", str(picks_dir),
            "--models-dir", str(tmp_path / "models"),
            "--data-dir", str(tmp_path / "processed"),
        ])
    assert result.exit_code == 0, result.output
    assert "already resolved" in result.output, result.output
    return result.output


def test_preview_summer_evening_uses_eastern_tomorrow(tmp_path):
    # 2026-07-09 23:00 ET == 2026-07-10 03:00 UTC: utcnow()+1d would say 07-11.
    fixed = _dt(2026, 7, 9, 23, 0, tzinfo=ZoneInfo("America/New_York"))
    out = _preview_default_target(fixed, tmp_path, "2026-07-10", "2026-07-11")
    assert "2026-07-10" in out, (
        f"preview targeted the UTC tomorrow, not ET tomorrow: {out!r}"
    )


def test_preview_winter_evening_uses_eastern_tomorrow(tmp_path):
    # 2026-01-15 20:00 EST == 2026-01-16 01:00 UTC (5h offset in winter).
    fixed = _dt(2026, 1, 15, 20, 0, tzinfo=ZoneInfo("America/New_York"))
    out = _preview_default_target(fixed, tmp_path, "2026-01-16", "2026-01-17")
    assert "2026-01-16" in out


def test_preview_overnight_cron_hour_unchanged(tmp_path):
    # 03:00 ET: UTC and ET agree on the calendar date — the 3am cron behavior
    # (preview tomorrow's slate) must not shift under the ET fix.
    fixed = _dt(2026, 7, 9, 3, 0, tzinfo=ZoneInfo("America/New_York"))
    out = _preview_default_target(fixed, tmp_path, "2026-07-10", "2026-07-10")
    assert "2026-07-10" in out
