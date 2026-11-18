"""Show what the real dispatcher remembers across warning/critical sequences."""

from datetime import date
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from probe_main import reject_network

from bts.health import alert


def sequence(status_path):
    d = date(2027, 7, 15)
    d_next = date(2027, 7, 16)
    critical = alert.Alert('CRITICAL', 'contest_state', 'existing refusal')
    warn = alert.Alert('WARN', 'contest_state', 'age test', incident_key='contest_state:stale_reading')

    def call(recipient='offline.test', *, alerts=(), warnings=(), day=d):
        sent = alert.dispatch_dm_for_health_alerts(
            list(alerts), recipient, warn_attention=list(warnings),
            status_path=status_path, now_et_date=day)
        status = json.loads(status_path.read_text())
        print(day, sent, status['status'], status.get('sent_sources'), status.get('body_first_line'))
        return sent, status

    with patch.object(alert, 'send_dm', Mock()) as send:
        assert call(alerts=[critical])[0]
        assert call(warnings=[warn])[1]['sent_sources'] == [
            'CRITICAL:contest_state', 'WARN_ATTENTION:contest_state:stale_reading']
        assert send.call_args.args[1] == 'BTS health WARN attention:\n- [contest_state] age test'
        assert not call(warnings=[warn])[0] and not call(alerts=[critical])[0]
        other_warn = alert.Alert('WARN', 'another', 'an unrelated warning')
        send.side_effect = OSError('transport failed')
        failed = call(warnings=[other_warn])[1]
        assert failed['status'] == 'failed' and 'sent_sources' not in failed
        assert alert._resurface_note(status_path) is None
        send.side_effect = None
        assert call(alerts=[critical])[0] and call(warnings=[warn])[0]
        send.side_effect = OSError('next warning transport failed')
        call(warnings=[warn], day=d_next)
        send.side_effect = None
        n = send.call_count
        assert not call(day=d_next)[0] and send.call_count == n
        assert call(warnings=[warn], day=d_next)[0]
        assert send.call_args.args[1].startswith('BTS health WARN attention:')
        # A new CS4C warning failure replaces a prior same-day successful critical stamp.
        assert call(alerts=[critical], day=d_next)[0]
        day_after = date(2027, 7, 17)
        assert call(alerts=[critical], day=day_after)[0]
        send.side_effect = OSError('CS4C unsuccessful')
        assert call(warnings=[warn], day=day_after)[1]['status'] == 'failed'
        send.side_effect = None
        assert call(alerts=[critical], day=day_after)[0]
        assert 'Previous health alert' not in send.call_args.args[1]
        # The reverse direction also erases the successful warning's dedup stamp.
        assert call(warnings=[warn], day=day_after)[0]
        send.side_effect = OSError('critical unsuccessful')
        assert call(alerts=[alert.Alert('CRITICAL', 'different', 'another critical')],
                    day=day_after)[1]['status'] == 'failed'
        send.side_effect = None
        assert call(warnings=[warn], day=day_after)[0]
        assert send.call_args.args[1].startswith('⚠️ Previous health alert DM FAILED')


if __name__ == '__main__':
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as tmp:
        sequence(Path(tmp) / 'health_dm_delivery_status.json')
