"""H6: a failed health DM resurfaces on the next dispatch."""
import json
from unittest.mock import patch

from bts.health.alert import dispatch_dm_for_health_alerts, Alert


def _status(tmp_path, status, critical_count=2):
    p = tmp_path / "status.json"
    p.write_text(json.dumps({"status": status, "critical_count": critical_count}))
    return p


@patch("bts.health.alert.send_dm")
def test_prior_failed_prepends_resurface_note(mock_dm, tmp_path):
    sp = _status(tmp_path, "failed", critical_count=2)
    dispatch_dm_for_health_alerts([Alert("CRITICAL", "x", "new issue")], "me", status_path=sp)
    body = mock_dm.call_args[0][1]
    assert "Previous health alert DM FAILED" in body
    assert "2 CRITICAL" in body
    assert "new issue" in body  # today's alert still included below the note


@patch("bts.health.alert.send_dm")
def test_prior_sent_no_resurface(mock_dm, tmp_path):
    sp = _status(tmp_path, "sent")
    dispatch_dm_for_health_alerts([Alert("CRITICAL", "x", "new issue")], "me", status_path=sp)
    assert "Previous health alert DM FAILED" not in mock_dm.call_args[0][1]


@patch("bts.health.alert.send_dm")
def test_no_status_file_no_resurface(mock_dm, tmp_path):
    dispatch_dm_for_health_alerts([Alert("CRITICAL", "x", "new issue")], "me",
                                  status_path=tmp_path / "missing.json")
    assert "Previous health alert DM FAILED" not in mock_dm.call_args[0][1]


@patch("bts.health.alert.send_dm")
def test_prior_failed_resurfaces_even_when_today_clean(mock_dm, tmp_path):
    sp = _status(tmp_path, "failed", critical_count=1)
    sent = dispatch_dm_for_health_alerts([], "me", status_path=sp)  # no alerts today
    assert sent is True
    assert "Previous health alert DM FAILED" in mock_dm.call_args[0][1]
