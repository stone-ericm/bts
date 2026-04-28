"""Tests for Tier-1 NRestarts spike check."""

import json

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

    def test_checkpoint_advances_after_run(self, tmp_path):
        # Each run updates the checkpoint to the current value
        check(tmp_path, current_nrestarts=52)
        check(tmp_path, current_nrestarts=53)
        cp = json.loads((tmp_path / ".nrestarts_checkpoint").read_text())
        assert cp["nrestarts"] == 53

    def test_corrupt_checkpoint_treated_as_fresh(self, tmp_path):
        (tmp_path / ".nrestarts_checkpoint").write_text("not json{{{")
        alerts = check(tmp_path, current_nrestarts=999)  # huge value, but no prior to compare
        assert alerts == []  # treated as fresh baseline
        # And the checkpoint is now valid JSON
        data = json.loads((tmp_path / ".nrestarts_checkpoint").read_text())
        assert data["nrestarts"] == 999
