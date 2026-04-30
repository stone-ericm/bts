"""Tests for Tier-1 Bluesky post failure check."""

import json
from datetime import date, datetime
from zoneinfo import ZoneInfo

from bts.health.post_failure import check, SOURCE

ET = ZoneInfo("America/New_York")


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


class TestPostFailureTimeGuard:
    """Time-of-day guard: don't alert until after the post window has actually closed.

    Production pattern: pick file is generated at 3 AM ET with projected lineup;
    Bluesky post happens 45min before each game's first pitch (lineup confirm) or
    via 1 AM safety-net cron the next day. Alerts fired before ~22:00 ET on day N
    will fire on legitimately-not-yet-posted picks where games haven't started.
    Cutoff: 22:00 ET (well after the latest typical first-pitch).
    """

    def test_no_alert_before_22et_when_post_failed(self, tmp_path):
        # 12:37 AM ET on 2026-04-30 — first game is 12:15 PM ET, post hasn't fired yet
        _write_pick(tmp_path, "2026-04-30", posted=False, uri=None)
        now = datetime(2026, 4, 30, 0, 37, tzinfo=ET)
        alerts = check(tmp_path, today=date(2026, 4, 30), now=now)
        assert alerts == []

    def test_no_alert_at_2159et_when_post_failed(self, tmp_path):
        # Just before cutoff
        _write_pick(tmp_path, "2026-04-30", posted=False, uri=None)
        now = datetime(2026, 4, 30, 21, 59, tzinfo=ET)
        alerts = check(tmp_path, today=date(2026, 4, 30), now=now)
        assert alerts == []

    def test_critical_at_22et_when_post_failed(self, tmp_path):
        # At cutoff — post window has closed, real failure
        _write_pick(tmp_path, "2026-04-30", posted=False, uri=None)
        now = datetime(2026, 4, 30, 22, 0, tzinfo=ET)
        alerts = check(tmp_path, today=date(2026, 4, 30), now=now)
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"

    def test_no_alert_before_22et_when_posted(self, tmp_path):
        # Pre-cutoff with successful post — no alert regardless
        _write_pick(tmp_path, "2026-04-30", posted=True, uri="at://abc/def")
        now = datetime(2026, 4, 30, 1, 0, tzinfo=ET)
        alerts = check(tmp_path, today=date(2026, 4, 30), now=now)
        assert alerts == []

    def test_now_defaults_to_actual_now(self, tmp_path):
        # When now is None, falls through to datetime.now(ET).
        result = check(tmp_path, today=date(2026, 4, 30), now=None)
        assert isinstance(result, list)
