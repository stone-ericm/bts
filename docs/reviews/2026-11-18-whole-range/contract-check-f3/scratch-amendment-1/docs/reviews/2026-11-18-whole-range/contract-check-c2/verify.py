"""Run the C2 evidence probes, plus transport/receipt interruption checks."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

import probe
from bts import picks, scheduler
from bts.daily_decision import load_decision
from bts.health import alert
import test_slice1_probes_v3 as carbon


def interrupt_receipt(directory):
    day = '2027-03-25'
    pick_dir = directory / 'picks'
    config = probe.config_at(pick_dir)
    daily = probe.daily_for(day)
    daily.delivery_attempted = True
    sent = []

    def accept(recipient, body):
        assert scheduler.load_state(day, pick_dir).pick_locked
        assert load_decision(day, pick_dir)['delivery_status'] == 'locked_unconfirmed'
        sent.append(body)
        if len(sent) == 1:
            raise KeyboardInterrupt('after remote acceptance')
        return 'ack'

    games = [{'gamePk': 101, 'gameDate': daily.pick.game_time,
              'teams': {'away': {'team': {'name': 'Away'}}, 'home': {'team': {'name': 'Home'}}}}]
    clock = probe.opening.Clock(probe.opening.at(day, '18:21'))
    with probe.opening.sandbox(clock, directory), patch.object(alert, 'send_dm', side_effect=accept), \
            patch.object(scheduler, 'fetch_schedule', side_effect=lambda date: games if date == day else []), \
            patch.object(scheduler, 'count_new_confirmations', return_value=0), \
            patch.object(scheduler, '_watchdog_ping_sleep', clock.sleep), \
            patch.object(scheduler, '_idle_until_next_wakeup'), patch.object(scheduler, 'run_result_polling'), \
            patch.object(scheduler, 'notify_watchdog'), patch.object(scheduler, 'notify_ready'):
        try:
            probe.direct(daily, config)
        except KeyboardInterrupt:
            pass
        else:
            raise AssertionError('interrupt lost')
        status = directory / 'health_state/health_dm_delivery_status.json'
        assert not status.exists()
        scheduler.run_day(day, config)
        assert len(sent) == 1 and not status.exists()  # Base has no restart dispatch.
        assert picks.load_pick(day, pick_dir).delivery_attempted
        probe.direct(daily, config)
        assert len(sent) == 2 and status.is_file()  # A direct encounter can retry.
    print('U5 interrupt after remote acceptance', 'durable lock and decision; real restart is silent; direct retry sends')


def main():
    with TemporaryDirectory(dir=probe.HERE) as tmp, redirect_stderr(StringIO()):
        root = Path(tmp)
        real_run = subprocess.run
        with patch.object(subprocess, 'run', side_effect=lambda *a, **k: real_run(*a, timeout=10, **k)):
            carbon.test_probe_2_midsend_restart_silently_locks_without_critical_dm(root)
        evidence = carbon.EVIDENCE['probe_2_midsend_restart']
        print('Carbon 2', 'DMs', evidence['dms_sent_all_day'], 'decision',
              evidence['decision_json']['delivery_status'], 'model streak', evidence['streak_json']['streak'])
        probe.opening.REPO = probe.ROOT
        for phase in ['before_remote_accept', 'after_remote_accept', 'corrected_after_marker', 'after_checkpoint']:
            directory = root / phase
            directory.mkdir()
            replay = probe.rehearsal(directory, phase)
            if phase == 'corrected_after_marker':
                corrected = replay
        for name, run in [('render', lambda p: probe.render_cases(p, corrected)),
                          ('receipts', probe.receipt_cases), ('run_day', probe.run_day_cases),
                          ('interrupt', interrupt_receipt)]:
            directory = root / name
            directory.mkdir()
            run(directory)


if __name__ == '__main__':
    signal.alarm(45)
    main()
    signal.alarm(0)
