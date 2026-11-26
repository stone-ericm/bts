import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

from bts.picks import save_pick
from review_support import DAY, ROOT, RIGHT, WRONG, World, assert_source, health, row


def correction_loses_all_clear(root):
    for wrong in (False, True):
        for manual in (False, True):
            w = World(root / f'clear-{wrong}-{manual}')
            daily = w.commit_at('06:10', at='04:55')
            if wrong:
                w.capture([WRONG], f'{DAY}T085000Z')
            start = w.invoke('05:00', pending=[row()] if wrong else [], manual=manual)
            assert len(start['sent']) == 1 and start['sent'][0].startswith('⚠')
            w.edit_pitch(daily, '06:20')
            w.capture([{**RIGHT, 'id': 78}], f'{DAY}T091000Z')
            fixed = w.invoke('05:15', pending=[row(78)], manual=manual)
            assert fixed['receipt']['qualification']['all_confirmed'] is True
            assert (fixed['marker'] if manual else w.early_state())['status'] == 'confirmed'
            repeat = w.invoke('05:30', pending=[row(78)], manual=manual)
            assert repeat['receipt']['outcome'] == 'already_confirmed' and not repeat['calls']
            assert bool(fixed['sent']) is manual, fixed
            print('actual missing all-clear', 'wrong-game' if wrong else 'missing-entry',
                  'manual' if manual else 'installed', 'DMs', start['sent'], fixed['sent'], repeat['sent'])


def equivalent_pitch_spends_extra_traffic(root):
    for reformat in (False, True):
        w = World(root / f'format-{reformat}')
        daily = w.commit_at('06:10', at='04:55')
        w.capture([RIGHT], f'{DAY}T085000Z')
        assert w.cron('05:00', pending=[row()])['receipt']['qualification']['all_confirmed']
        if reformat:
            daily.pick.game_time = f'{DAY}T10:10:00Z'
            save_pick(daily, w.picks)
        repeat = w.cron('05:15', pending=[row()])
        assert bool(repeat['calls']) is reformat
        print('same 06:10 ET instant, UTC spelling' if reformat else 'same spelling',
              repeat['receipt']['outcome'], repeat['calls'])


def invalid_marker_skips(root):
    for field, invalid in (('observed_at', 'not a datetime'), ('receipt', ['not', 'an', 'id'])):
        w = World(root / f'invalid-{field}')
        w.commit_at('06:10', at='04:55')
        first = w.cron('05:00', pending=[row()])
        assert first['receipt']['outcome'] == 'observed'
        marker = w.early_state()
        marker[field] = invalid
        w.early.write_text(json.dumps(marker))
        next_run = w.cron('05:15')
        assert next_run['receipt']['outcome'] == 'already_confirmed' and not next_run['calls'] and not next_run['sent']
        print('corrupt early confirmed field', field, next_run['receipt']['outcome'], next_run['receipt']['references'])


def restart_after_unconfirmed_send(root):
    w = World(root / 'restart')
    w.commit_at('06:10', at='04:55')
    first = w.cron('05:00', dm='failed')
    assert first['receipt']['marker_status'] == 'dm_failed' and len(first['attempts']) == 1
    original = w.early.read_bytes()
    interrupted = w.cron('05:15', dm='interrupt')
    assert interrupted['receipt']['detail'] == 'raised KeyboardInterrupt' and len(interrupted['attempts']) == 1
    assert w.early.read_bytes() == original and not w.marker.exists()
    w = World(w.root)
    recovered = w.cron('05:30')
    assert len(recovered['sent']) == 1 and w.early_state()['status'] == 'alerted'
    assert [level for level, _ in health(w)] == ['WARN']
    blocked = World(root / 'write-failure')
    blocked.commit_at('06:10', at='04:55')
    blocked.early.mkdir(parents=True)
    attempt = blocked.cron('05:00')
    assert (attempt['exit'], len(attempt['sent'])) == (1, 1)
    assert 'early marker unavailable' in attempt['stderr'] and attempt['receipt']['detail'] in attempt['stderr']
    assert not blocked.marker.exists()
    print('restart, unconfirmed send and invalid destination', interrupted['receipt']['detail'], attempt['receipt']['detail'])


if __name__ == '__main__':
    assert_source(ROOT / 'src')
    with TemporaryDirectory() as dirname:
        root = Path(dirname)
        correction_loses_all_clear(root)
        equivalent_pitch_spends_extra_traffic(root)
        invalid_marker_skips(root)
        restart_after_unconfirmed_send(root)
