"""Tests for Tier-1 NRestarts spike check."""

import json
from datetime import date

from bts.health.restart_spike import check, SOURCE


class TestRestartSpike:
    def test_first_run_records_baseline_no_alert(self, tmp_path):
        # No prior checkpoint exists → record baseline, no alert
        alerts = check(tmp_path, current_nrestarts=52)
        assert alerts == []
        cp = tmp_path / ".nrestarts_checkpoint"
        assert cp.exists()
        data = json.loads(cp.read_text())
        assert data["nrestarts"] == 52

    def test_no_alert_when_no_change(self, tmp_path):
        check(tmp_path, current_nrestarts=52)  # baseline
        alerts = check(tmp_path, current_nrestarts=52)
        assert alerts == []

    def test_no_alert_below_threshold(self, tmp_path):
        check(tmp_path, current_nrestarts=52)  # baseline
        alerts = check(tmp_path, current_nrestarts=54)  # +2 < 3
        assert alerts == []

    def test_critical_at_threshold(self, tmp_path):
        check(tmp_path, current_nrestarts=52)  # baseline
        alerts = check(tmp_path, current_nrestarts=55)  # +3 == threshold
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"
        assert alerts[0].source == SOURCE
        assert "+3" in alerts[0].message
        assert "Scheduler restart loop suspected" in alerts[0].message
        assert "journal/OOM/watchdog/crash" in alerts[0].message
        assert "Heartbeat-gap regression suspected" not in alerts[0].message

    def test_critical_above_threshold(self, tmp_path):
        check(tmp_path, current_nrestarts=52)  # baseline
        alerts = check(tmp_path, current_nrestarts=70)  # +18
        assert len(alerts) == 1
        assert "+18" in alerts[0].message

    def test_custom_threshold(self, tmp_path):
        check(tmp_path, current_nrestarts=52)  # baseline
        alerts = check(tmp_path, current_nrestarts=53, spike_threshold=1)
        assert len(alerts) == 1
        assert alerts[0].level == "CRITICAL"

    def test_checkpoint_anchored_within_a_day(self, tmp_path):
        # 2026-07-12 incident: the checkpoint advanced on EVERY run, so a
        # Restart=always loop re-running EOD every ~48s moved the baseline
        # +1 at a time — a 47-restart storm never tripped the +3 threshold.
        # Same-day runs must keep the first observation as the anchor.
        d = date(2026, 7, 12)
        check(tmp_path, current_nrestarts=52, today=d)
        check(tmp_path, current_nrestarts=53, today=d)
        cp = json.loads((tmp_path / ".nrestarts_checkpoint").read_text())
        assert cp["nrestarts"] == 52, "same-day runs must not advance the anchor"

    def test_same_day_creep_accumulates_to_critical(self, tmp_path):
        # +1 per run, three runs: the deltas alias to +1 each under the old
        # behavior; day-anchored they accumulate to +3 → CRITICAL.
        d = date(2026, 7, 12)
        assert check(tmp_path, current_nrestarts=52, today=d) == []
        assert check(tmp_path, current_nrestarts=53, today=d) == []
        assert check(tmp_path, current_nrestarts=54, today=d) == []
        alerts = check(tmp_path, current_nrestarts=55, today=d)
        assert len(alerts) == 1 and alerts[0].level == "CRITICAL"
        assert "+3" in alerts[0].message

    def test_new_day_advances_anchor(self, tmp_path):
        d1, d2 = date(2026, 7, 12), date(2026, 7, 13)
        check(tmp_path, current_nrestarts=52, today=d1)
        assert check(tmp_path, current_nrestarts=54, today=d2) == []  # +2 vs 52
        cp = json.loads((tmp_path / ".nrestarts_checkpoint").read_text())
        assert cp["nrestarts"] == 54 and cp["day"] == "2026-07-13"
        # Later same-new-day run diffs against the new anchor
        alerts = check(tmp_path, current_nrestarts=57, today=d2)  # +3 vs 54
        assert len(alerts) == 1

    def test_multiday_gap_budgets_planned_restarts(self, tmp_path):
        # Round-2 review #4: the daily lifecycle exits once per day by design
        # (idle → return → Restart=always), and no-games days never run this
        # check. A 4-day break accumulates +4 PLANNED restarts against a
        # frozen anchor — that must not read as a spike.
        check(tmp_path, current_nrestarts=50, today=date(2026, 7, 12))
        alerts = check(tmp_path, current_nrestarts=54, today=date(2026, 7, 16))
        assert alerts == [], "planned one-exit-per-day restarts must be budgeted"

    def test_multiday_gap_still_catches_real_loops(self, tmp_path):
        check(tmp_path, current_nrestarts=50, today=date(2026, 7, 12))
        alerts = check(tmp_path, current_nrestarts=97, today=date(2026, 7, 16))
        assert len(alerts) == 1 and alerts[0].level == "CRITICAL"

    def test_legacy_checkpoint_without_day_still_compares(self, tmp_path):
        (tmp_path / ".nrestarts_checkpoint").write_text(
            json.dumps({"nrestarts": 52, "checkpointed_at": "2026-07-11T23:00:00+00:00"}))
        alerts = check(tmp_path, current_nrestarts=55, today=date(2026, 7, 12))
        assert len(alerts) == 1 and "+3" in alerts[0].message

    def test_corrupt_checkpoint_treated_as_fresh(self, tmp_path):
        (tmp_path / ".nrestarts_checkpoint").write_text("not json{{{")
        alerts = check(tmp_path, current_nrestarts=999)  # huge value, but no prior to compare
        assert alerts == []  # treated as fresh baseline
        # And the checkpoint is now valid JSON
        data = json.loads((tmp_path / ".nrestarts_checkpoint").read_text())
        assert data["nrestarts"] == 999
