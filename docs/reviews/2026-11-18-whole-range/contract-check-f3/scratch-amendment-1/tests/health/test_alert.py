"""Tests for the shared Alert type + DM dispatcher."""

import json
from unittest.mock import patch

from bts.health.alert import (
    Alert,
    dispatch_dm_for_critical,
    dispatch_dm_for_health_alerts,
    format_health_dm_body,
    log_alerts,
)


class TestAlert:
    def test_frozen(self):
        a = Alert(level="INFO", source="test", message="m")
        try:
            a.level = "WARN"  # type: ignore
            assert False, "expected immutability"
        except Exception:
            pass

    def test_fields(self):
        a = Alert(level="CRITICAL", source="cal", message="drift big")
        assert a.level == "CRITICAL"
        assert a.source == "cal"
        assert a.message == "drift big"


class TestDispatchDmForCritical:
    def test_no_critical_no_dm(self):
        alerts = [Alert("WARN", "x", "m"), Alert("INFO", "y", "m")]
        with patch("bts.health.alert.send_dm") as mock_dm:
            sent = dispatch_dm_for_critical(alerts, "x.bsky.social")
            mock_dm.assert_not_called()
            assert sent is False

    def test_no_recipient_no_dm(self):
        alerts = [Alert("CRITICAL", "x", "m")]
        with patch("bts.health.alert.send_dm") as mock_dm:
            sent = dispatch_dm_for_critical(alerts, None)
            mock_dm.assert_not_called()
            assert sent is False

    def test_critical_sends_single_dm(self):
        alerts = [
            Alert("CRITICAL", "calibration_drift", "drift -0.07"),
            Alert("CRITICAL", "blend_training", "missing pkl"),
            Alert("WARN", "x", "ignore"),
        ]
        with patch("bts.health.alert.send_dm") as mock_dm:
            mock_dm.return_value = "msg-id"
            sent = dispatch_dm_for_critical(alerts, "x.bsky.social")
            mock_dm.assert_called_once()
            args = mock_dm.call_args.args
            assert args[0] == "x.bsky.social"
            # Body should include both CRITICAL alerts but not the WARN
            assert "calibration_drift" in args[1]
            assert "blend_training" in args[1]
            assert "ignore" not in args[1]
            assert sent is True

    def test_send_dm_failure_swallowed(self):
        alerts = [Alert("CRITICAL", "x", "m")]
        with patch("bts.health.alert.send_dm", side_effect=RuntimeError("boom")):
            sent = dispatch_dm_for_critical(alerts, "x.bsky.social")
            # Returned True (DM was attempted), but no exception propagated
            assert sent is True


class TestDispatchDmForHealthAlerts:
    def test_no_alerts_no_dm(self):
        with patch("bts.health.alert.send_dm") as mock_dm:
            sent = dispatch_dm_for_health_alerts([], "x.bsky.social")
            mock_dm.assert_not_called()
            assert sent is False

    def test_attention_only_sends_warn_digest(self):
        warn = Alert("WARN", "disk_fill", "disk 91%")
        with patch("bts.health.alert.send_dm") as mock_dm:
            sent = dispatch_dm_for_health_alerts(
                [],
                "x.bsky.social",
                warn_attention=[warn],
            )
            mock_dm.assert_called_once()
            body = mock_dm.call_args.args[1]
            assert body.startswith("BTS health WARN attention:")
            assert "disk_fill" in body
            assert sent is True

    def test_critical_and_attention_share_one_dm(self):
        critical = Alert("CRITICAL", "restart_spike", "NRestarts +7")
        warn = Alert("WARN", "same_team_corr", "3rd day")
        with patch("bts.health.alert.send_dm") as mock_dm:
            sent = dispatch_dm_for_health_alerts(
                [critical],
                "x.bsky.social",
                warn_attention=[warn],
            )
            mock_dm.assert_called_once()
            body = mock_dm.call_args.args[1]
            assert "BTS health CRITICAL alert(s):" in body
            assert "WARN attention:" in body
            assert "restart_spike" in body
            assert "same_team_corr" in body
            assert sent is True

    def test_success_writes_delivery_status(self, tmp_path):
        status_path = tmp_path / "health_dm_delivery_status.json"
        critical = Alert("CRITICAL", "restart_spike", "NRestarts +7")
        with patch("bts.health.alert.send_dm") as mock_dm:
            sent = dispatch_dm_for_health_alerts(
                [critical],
                "x.bsky.social",
                status_path=status_path,
            )

        assert sent is True
        mock_dm.assert_called_once()
        data = json.loads(status_path.read_text())
        assert data["status"] == "sent"
        assert data["recipient_configured"] is True
        assert data["critical_count"] == 1
        assert data["warn_attention_count"] == 0
        assert data["body_first_line"] == "BTS health CRITICAL alert(s):"
        assert data["body_sha256"]

    def test_failure_writes_delivery_status(self, tmp_path):
        status_path = tmp_path / "health_dm_delivery_status.json"
        critical = Alert("CRITICAL", "restart_spike", "NRestarts +7")
        with patch("bts.health.alert.send_dm", side_effect=RuntimeError("boom")):
            sent = dispatch_dm_for_health_alerts(
                [critical],
                "x.bsky.social",
                status_path=status_path,
            )

        assert sent is True
        data = json.loads(status_path.read_text())
        assert data["status"] == "failed"
        assert data["critical_count"] == 1
        assert data["error"] == "boom"

    def test_missing_recipient_writes_status_only_when_body_exists(self, tmp_path):
        status_path = tmp_path / "health_dm_delivery_status.json"
        warn = Alert("WARN", "disk_fill", "disk 91%")
        with patch("bts.health.alert.send_dm") as mock_dm:
            sent = dispatch_dm_for_health_alerts(
                [],
                None,
                warn_attention=[warn],
                status_path=status_path,
            )

        assert sent is False
        mock_dm.assert_not_called()
        data = json.loads(status_path.read_text())
        assert data["status"] == "skipped_no_recipient"
        assert data["recipient_configured"] is False
        assert data["warn_attention_count"] == 1

    def test_no_body_does_not_clear_prior_status(self, tmp_path):
        status_path = tmp_path / "health_dm_delivery_status.json"
        status_path.write_text(json.dumps({"status": "failed", "error": "old"}))

        with patch("bts.health.alert.send_dm") as mock_dm:
            sent = dispatch_dm_for_health_alerts(
                [],
                "x.bsky.social",
                status_path=status_path,
            )

        assert sent is False
        mock_dm.assert_not_called()
        assert json.loads(status_path.read_text()) == {"status": "failed", "error": "old"}

    def test_format_empty_returns_none(self):
        assert format_health_dm_body([], []) is None


class TestHealthDmDedup:
    """Per-day source-set dedup (2026-07-12 incident).

    The dispatcher is designed to run once per day, but a Restart=always
    thrash re-walked EOD every ~48s and re-sent the same CRITICAL 47 times.
    Dedup key is the (level, source) set — NOT the body hash — because a
    growing metric (e.g. restart_spike's delta) changes the body every run
    while remaining the same problem. A NEW source same-day must still send
    (escalation stays visible); a day rollover must send (daily reminder);
    a failed send must never be suppressed (H6 resurface flow).
    """

    from datetime import date as _date
    D1 = _date(2026, 7, 12)
    D2 = _date(2026, 7, 13)

    def _crit(self, source="predicted_vs_realized"):
        return Alert("CRITICAL", source, "drift +0.17")

    def test_same_day_same_sources_suppressed(self, tmp_path):
        status_path = tmp_path / "s.json"
        with patch("bts.health.alert.send_dm") as mock_dm:
            first = dispatch_dm_for_health_alerts(
                [self._crit()], "x.bsky.social",
                status_path=status_path, now_et_date=self.D1)
            second = dispatch_dm_for_health_alerts(
                [self._crit()], "x.bsky.social",
                status_path=status_path, now_et_date=self.D1)
        assert first is True and second is False
        mock_dm.assert_called_once()
        assert json.loads(status_path.read_text())["status"] == "sent"

    def test_new_source_same_day_sends_and_unions(self, tmp_path):
        status_path = tmp_path / "s.json"
        with patch("bts.health.alert.send_dm") as mock_dm:
            dispatch_dm_for_health_alerts(
                [self._crit("a")], "x.bsky.social",
                status_path=status_path, now_et_date=self.D1)
            # New source b escalates → must send
            sent2 = dispatch_dm_for_health_alerts(
                [self._crit("a"), self._crit("b")], "x.bsky.social",
                status_path=status_path, now_et_date=self.D1)
            # b alone is a subset of the sent union {a, b} → suppressed
            sent3 = dispatch_dm_for_health_alerts(
                [self._crit("b")], "x.bsky.social",
                status_path=status_path, now_et_date=self.D1)
        assert sent2 is True and sent3 is False
        assert mock_dm.call_count == 2

    def test_next_day_sends_again(self, tmp_path):
        status_path = tmp_path / "s.json"
        with patch("bts.health.alert.send_dm") as mock_dm:
            dispatch_dm_for_health_alerts(
                [self._crit()], "x.bsky.social",
                status_path=status_path, now_et_date=self.D1)
            sent = dispatch_dm_for_health_alerts(
                [self._crit()], "x.bsky.social",
                status_path=status_path, now_et_date=self.D2)
        assert sent is True
        assert mock_dm.call_count == 2

    def test_failed_send_not_suppressed_then_resurfaces(self, tmp_path):
        status_path = tmp_path / "s.json"
        with patch("bts.health.alert.send_dm", side_effect=RuntimeError("boom")):
            dispatch_dm_for_health_alerts(
                [self._crit()], "x.bsky.social",
                status_path=status_path, now_et_date=self.D1)
        with patch("bts.health.alert.send_dm") as mock_dm:
            sent = dispatch_dm_for_health_alerts(
                [self._crit()], "x.bsky.social",
                status_path=status_path, now_et_date=self.D1)
        assert sent is True
        body = mock_dm.call_args.args[1]
        assert "Previous health alert DM FAILED" in body

    def test_warn_attention_digest_deduped(self, tmp_path):
        status_path = tmp_path / "s.json"
        warn = Alert("WARN", "disk_fill", "disk 91%")
        with patch("bts.health.alert.send_dm") as mock_dm:
            dispatch_dm_for_health_alerts(
                [], "x.bsky.social", warn_attention=[warn],
                status_path=status_path, now_et_date=self.D1)
            sent = dispatch_dm_for_health_alerts(
                [], "x.bsky.social", warn_attention=[warn],
                status_path=status_path, now_et_date=self.D1)
        assert sent is False
        mock_dm.assert_called_once()

    def test_legacy_status_file_without_dedup_fields_sends(self, tmp_path):
        status_path = tmp_path / "s.json"
        status_path.write_text(json.dumps({
            "schema_version": "bts_health_dm_delivery_status_v1",
            "status": "sent", "critical_count": 1,
            "body_sha256": "abc", "body_first_line": "BTS health CRITICAL alert(s):",
        }))
        with patch("bts.health.alert.send_dm") as mock_dm:
            sent = dispatch_dm_for_health_alerts(
                [self._crit()], "x.bsky.social",
                status_path=status_path, now_et_date=self.D1)
        assert sent is True
        mock_dm.assert_called_once()

    def test_distinct_incident_keys_share_source_not_suppressed(self, tmp_path):
        # Round-2 review #6: two DIFFERENT checks crashing both surface as
        # source="health_runner"; the second crash must still reach the
        # operator. incident_key (when set) is the dedup identity.
        status_path = tmp_path / "s.json"
        a = Alert("CRITICAL", "health_runner", "check 'a' crashed",
                  incident_key="health_runner:a")
        b = Alert("CRITICAL", "health_runner", "check 'b' crashed",
                  incident_key="health_runner:b")
        with patch("bts.health.alert.send_dm") as mock_dm:
            dispatch_dm_for_health_alerts(
                [a], "x.bsky.social", status_path=status_path, now_et_date=self.D1)
            sent_b = dispatch_dm_for_health_alerts(
                [b], "x.bsky.social", status_path=status_path, now_et_date=self.D1)
            sent_a_again = dispatch_dm_for_health_alerts(
                [a], "x.bsky.social", status_path=status_path, now_et_date=self.D1)
        assert sent_b is True, "a different crashed check must not be suppressed"
        assert sent_a_again is False, "the same crash dedups"
        assert mock_dm.call_count == 2

    def test_no_status_path_never_suppresses(self):
        with patch("bts.health.alert.send_dm") as mock_dm:
            first = dispatch_dm_for_health_alerts(
                [self._crit()], "x.bsky.social", now_et_date=self.D1)
            second = dispatch_dm_for_health_alerts(
                [self._crit()], "x.bsky.social", now_et_date=self.D1)
        assert first is True and second is True
        assert mock_dm.call_count == 2


class TestLogAlerts:
    def test_log_at_appropriate_levels(self, caplog):
        import logging
        with caplog.at_level(logging.INFO, logger="bts.health.alert"):
            log_alerts([
                Alert("INFO", "s1", "info-msg"),
                Alert("WARN", "s2", "warn-msg"),
                Alert("CRITICAL", "s3", "critical-msg"),
            ])
        levels = [r.levelname for r in caplog.records if "info-msg" in r.message
                  or "warn-msg" in r.message or "critical-msg" in r.message]
        assert "INFO" in levels
        assert "WARNING" in levels
        assert "ERROR" in levels  # CRITICAL alerts log at ERROR level
