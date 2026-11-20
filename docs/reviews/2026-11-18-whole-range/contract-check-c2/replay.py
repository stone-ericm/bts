"""Run the offline C2 contract probes against the current checkout."""

from contextlib import redirect_stderr
from io import StringIO
from pathlib import Path
import signal
import subprocess
from tempfile import TemporaryDirectory
from unittest.mock import patch

import probe
import test_slice1_probes_v3 as carbon
from verify import interrupt_receipt


def corrected_rehearsal(directory, control_directory):
    opening = probe.opening
    day = opening.DAYS[1]
    season = opening.Season(directory)
    seed = {'schema_version': 'bts_contest_streak_auto_v1', 'season': 2026,
            'active_streak': 17, 'best_streak': 26, 'saver_available': False,
            'source_date': opening.OLD_DAY, 'recorded_at': '2026-10-21T14:00:00Z',
            'source': 'manual_cli'}
    opening.write_json(season.account_dir / 'contest_streak.json', seed)
    with season.run_offline():
        season.restore()
        result = probe.edges.restart_at_send(season, 'corrected_after_marker')
    messages = [m['body'] for m in result['messages']]
    assert not probe.unknown(messages) and not probe.edges.picks_sent(season)
    assert any('Delivery of this pick was NOT confirmed' in body for body in messages)
    snapshot = result['snapshot']
    assert snapshot['decision']['delivery_status'] == 'locked_unconfirmed'
    assert snapshot['decision']['scoreable'] and result['next_morning']['state'] == 'not_entered'
    pick = snapshot['pick']
    assert pick['chosen_on']['streak'] == 1 and pick['double_down'] is not None

    control = opening.Season(control_directory)
    opening.write_json(control.account_dir / 'contest_streak.json', seed)
    with control.run_offline():
        control.restore()
        control.clock.move(opening.at(day, '09:30'))
        control.dated(day, 46, saver='used')
        control.clock.move(opening.at(day, '10:00'))
        control.start(day)
    corrected = control.snapshot(day)['pick']
    assert corrected['chosen_on']['streak'] == 46 and corrected['double_down'] is None
    assert len(probe.edges.picks_sent(control)) == 1
    print('corrected restart', 'pick sends 0, unknown DMs 0, scoreable locked_unconfirmed, next morning not_entered')
    print('correction control', 'attempted double at 1, current-state single at 46',
          'primary', pick['pick']['batter_name'], corrected['pick']['batter_name'])
    return result


def main(root):
    real_run = subprocess.run
    with patch.object(subprocess, 'run', side_effect=lambda *a, **k: real_run(*a, timeout=10, **k)):
        carbon.test_probe_2_midsend_restart_silently_locks_without_critical_dm(root)
    evidence = carbon.EVIDENCE['probe_2_midsend_restart']
    print('Carbon 2', 'DMs', evidence['dms_sent_all_day'], 'decision',
          evidence['decision_json']['delivery_status'], 'model streak', evidence['streak_json']['streak'])
    probe.opening.REPO = probe.ROOT
    for phase in ['before_remote_accept', 'after_remote_accept', 'after_checkpoint']:
        directory = root / phase
        directory.mkdir()
        probe.rehearsal(directory, phase)
    corrected = corrected_rehearsal(root / 'corrected', root / 'control')
    for name, run in [('render', lambda p: probe.render_cases(p, corrected)),
                      ('receipts', probe.receipt_cases), ('run_day', probe.run_day_cases),
                      ('interrupt', interrupt_receipt)]:
        directory = root / name
        directory.mkdir()
        run(directory)


if __name__ == '__main__':
    signal.alarm(45)
    with TemporaryDirectory(dir=probe.HERE) as tmp, redirect_stderr(StringIO()):
        main(Path(tmp))
    signal.alarm(0)
