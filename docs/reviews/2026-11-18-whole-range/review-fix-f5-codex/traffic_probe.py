"""Check every added launch with local states, captured transports, and real readers."""
from collections import Counter
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from subprocess import CompletedProcess
from tempfile import TemporaryDirectory

from probe_support import Replay, clock_slots, load_line
from bts import entry_receipt
from bts.data import backup
from bts.health import pick_entry
from tests.test_wr_f1_wrong_game import DAY, RIGHT, row


def check(root, label, version='f5'):
    site = root / f'{version}-{label}'
    world = Replay(site / 'data', version)
    old_day = (date.fromisoformat(DAY) - timedelta(days=1)).isoformat()
    if label == 'yesterday':
        yesterday = Replay(site / 'data', version, day=old_day)
        yesterday.commit_at('06:10')
        yesterday.set_marker({'date': old_day, 'status': 'alerted'})
    elif label == 'preview':
        world.commit_at('06:10', at='03:00', delivery=None)
    elif label == 'evening':
        world.commit_at('19:10')
    elif label == 'base-window':
        world.commit_at('10:50')
    elif label != 'absent':
        world.commit_at('06:10', at='04:50')
        if label == 'already-confirmed':
            world.set_marker({'date': DAY, 'status': 'confirmed'})
        elif label == 'yesterday-marker':
            world.set_marker({'date': old_day, 'status': 'confirmed'})
        elif label == 'alerted':
            world.set_marker({'date': DAY, 'status': 'alerted', 'escalations': ['initial']})
    world.capture([RIGHT], f'{DAY}T085000Z')
    slots = list(clock_slots(load_line(version)))
    outcomes, calls, sent, changes = Counter(), Counter(), 0, 0
    for slot in slots:
        if label not in ('absent', 'base-window', 'yesterday-marker') and slot.hour >= 10:
            break
        hm = slot.strftime('%H:%M')
        matched = label == 'yesterday-marker' or (label == 'base-window' and hm == '10:00')
        received = [row()] if matched else [row(player=2)] if label == 'mismatch' else []
        auth = {'identity': 'wrong-user', 'auth-failure': 'login'}.get(label, 'ok')
        got = world.invoke(hm, pending=received, auth=auth)
        rec = got['receipt']
        assert rec and got['exception'] in (None, 'SystemExit'), got
        outcomes[rec['outcome']] += 1
        calls.update(got['calls'])
        sent += len(got['sent'])
        changes += int(got['marker_changed'])
        if rec['outcome'] in ('no_pick_file', 'not_committed', 'outside_window', 'base_slot_available',
                              'already_confirmed'):
            assert not (got['calls'] or got['attempts'] or got['marker_changed'])
            assert all(rec[key] is None for key in ('account', 'observation', 'verifier', 'marker_status'))
    count = sum(outcomes.values())
    directory = entry_receipt.receipts_root(world.picks) / DAY
    assert len(list(directory.glob('*.json'))) == len(list(directory.glob('*.json.sealed'))) == count
    assert count == ((56 if version == 'base' else 96) if label in ('absent', 'base-window', 'yesterday-marker') else 40)
    if label in ('absent', 'yesterday', 'preview', 'evening', 'already-confirmed'):
        assert not (calls or sent or changes)
    if label == 'base-window':
        assert outcomes == {'outside_window': 92, 'base_slot_available': 1, 'observed': 1, 'already_confirmed': 2}
        assert world.morning()[1]['state'] == 'entered'
    if label == 'yesterday-marker':
        assert outcomes['observed'] == 1 and outcomes['already_confirmed'] == 4
        assert world.morning()[1]['state'] == 'entered'
    if label in ('missing', 'mismatch', 'alerted'):
        assert calls['login'] == calls['pending'] == 5 and bool(sent) and changes == 5
        if label == 'missing':
            assert world.morning()[1]['state'] == 'not_entered'
            assert pick_entry.check(world.picks, date.fromisoformat(DAY), datetime(2030, 1, 1, tzinfo=timezone.utc))
    if label in ('identity', 'auth-failure'):
        assert calls == {'cookies': 5, 'login': 5} and not (sent or changes)
    print(version, label, count, dict(outcomes), dict(calls), f'{sent} DMs / {changes} marker writes')
    return site


def verify_backup(site):
    calls = []

    def capture(command, **kwargs):
        calls.append(command)
        assert kwargs['timeout'] > 0
        return CompletedProcess(command, 0, '{"message_type":"summary","snapshot_id":"test"}\n', '')

    result = backup.run_backup('ops', site,
                               env={'RESTIC_PASSWORD': 'fixture', 'BTS_RESTIC_REPOSITORY': '/fake-repo',
                                    'BTS_RESTIC_BIN': '/no-execution/restic'}, runner=capture)
    assert result['ok'] and str(site / 'data/health_state') in calls[0]
    assert calls[0][calls[0].index('--exclude') + 1] == '*.lock'
    assert calls[1][1] == 'forget'
    assert len(list((site / 'data/health_state/pick_entry_receipts' / DAY).glob('*.json.sealed'))) == 96
    print('ops backup includes all JSONs and seals; retention command leaves local files intact')


if __name__ == '__main__':
    with TemporaryDirectory() as scratch:
        root = Path(scratch)
        check(root, 'absent', 'base')
        for label in ('absent', 'yesterday', 'preview', 'evening', 'base-window', 'already-confirmed',
                      'yesterday-marker', 'missing', 'mismatch', 'alerted', 'identity', 'auth-failure'):
            site = check(root, label)
            if label == 'absent':
                verify_backup(site)
