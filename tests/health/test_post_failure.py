"""Tests for Tier-1 Bluesky post failure check."""

import json
from datetime import date

from bts.health.post_failure import check, SOURCE


def _write_pick(picks_dir, date_iso, *, posted, uri="at://x/y/z", with_pick=True):
    data = {"date": date_iso}
    if with_pick:
        data["pick"] = {"batter_name": "X", "p_game_hit": 0.75}
    if posted is not None:
        data["bluesky_posted"] = posted
    if uri is not None:
        data["bluesky_uri"] = uri
    (picks_dir / f"{date_iso}.json").write_text(json.dumps(data))


class TestPostFailure:
    def test_no_alert_when_posted_with_uri(self, tmp_path):
        _write_pick(tmp_path, "2026-04-27", posted=True, uri="at://abc/def")
        alerts = check(tmp_path, today=date(2026, 4, 27))
        assert alerts == []

    def test_critical_when_posted_false(self, tmp_path):
        _write_pick(tmp_path, "2026-04-27", posted=False, uri=None)
        alerts = check(tmp_path, today=date(2026, 4, 27))
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"
        assert alerts[0].source == SOURCE

    def test_critical_when_posted_true_but_uri_missing(self, tmp_path):
        # Edge case: bluesky_posted=true but no URI → still suspect
        _write_pick(tmp_path, "2026-04-27", posted=True, uri=None)
        alerts = check(tmp_path, today=date(2026, 4, 27))
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"

    def test_no_alert_when_no_pick(self, tmp_path):
        # Day with no pick (all games skipped) → no alert
        _write_pick(tmp_path, "2026-04-27", posted=None, with_pick=False)
        alerts = check(tmp_path, today=date(2026, 4, 27))
        assert alerts == []

    def test_no_alert_when_pick_file_missing(self, tmp_path):
        # No file at all (pre-pick or rest day) → no alert
        alerts = check(tmp_path, today=date(2026, 4, 27))
        assert alerts == []

    def test_handles_corrupt_pick_file(self, tmp_path):
        (tmp_path / "2026-04-27.json").write_text("not json{{{")
        # Should not crash; logs warning and returns []
        alerts = check(tmp_path, today=date(2026, 4, 27))
        assert alerts == []
