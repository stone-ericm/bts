"""Tests for the shared Alert type + DM dispatcher."""

from unittest.mock import patch

from bts.health.alert import Alert, dispatch_dm_for_critical, log_alerts


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
