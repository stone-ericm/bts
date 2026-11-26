"""Drive successive watchdog runs and consume the escalation ledger."""
from itertools import product
import sys

from support import ALL_CLEAR, DAY, HERE, MATCH, PITCH, RIGHT, WRONG, World, dump, print_source, row, world

FIRST = '⚠️ BTS entry game needs verification.'
REMINDER = '⏰ BTS entry game verification is still needed.'


def dirty(result):
    assert result['marker']['status'] != 'confirmed'
    assert result['marker']['game_unresolved'] is True
    assert ALL_CLEAR not in result['attempted_dm']


def orders():
    bridges = {'crosswalk': (MATCH, {'crosswalk': {}}), 'removed': ((), {}), 'mismatch': ([row(player=3)], {})}
    choices = ('success', 'failure', 'none')
    checked = 0
    for initial, first_dm, bridge, bridge_dm in product((False, True), choices, bridges, choices):
        with world('orders') as w:
            w.commit(pitch=PITCH)
            w.capture([WRONG])
            if initial:
                assert w.check('18:25')['marker']['escalations'] == ['initial']
            first = w.check('18:35', MATCH, dm=first_dm)
            dirty(first)
            assert ('wrong_game' in first['marker']['escalations']) is (first_dm != 'failure')
            assert ('initial' in first['marker']['escalations']) is initial
            rows, options = bridges[bridge]
            middle = w.check('18:45', rows, dm=bridge_dm, **options)
            dirty(middle)
            if not initial and bridge != 'crosswalk' and bridge_dm != 'none':
                assert len(middle['attempted_dm']) == 1 and 'Fix it now!' in middle['attempted_dm'][0]
            w.capture([RIGHT], f'{DAY}T230000Z')
            unresolved = w.check('19:00', MATCH)
            dirty(unresolved)
            assert bool(unresolved['dm']) is (first_dm == 'failure')
            assert all(dm.startswith(FIRST) for dm in unresolved['dm'])
            for time, retry in (('19:05', '19:06'), ('19:20', '19:21')):
                failed = w.check(time, MATCH, dm='failure')
                dirty(failed)
                assert len(failed['attempted_dm']) == 1 and failed['attempted_dm'][0].startswith(REMINDER)
                assert w.check(retry, MATCH)['dm'] == failed['attempted_dm']
                assert w.check(retry, MATCH)['attempted_dm'] == []
            assert 'outside window' in w.check('19:35', MATCH)['stdout']
            assert w.read_marker()['game_unresolved'] is True
            checked += 1
    print(f'{checked} orders across first warning, intermediate state and three DM outcomes')


def proof():
    for dm in ('success', 'failure', 'none'):
        with world('full-proof') as w:
            w.commit(pitch=PITCH)
            w.capture([WRONG, {**RIGHT, 'id': 88}])
            assert w.check('18:35', MATCH)['marker']['game_unresolved'] is True
            fixed = w.check('19:20', [row(88)], dm=dm)
            assert fixed['marker']['status'] == 'confirmed' and 'game_unresolved' not in fixed['marker']
            assert fixed['attempted_dm'] == ([] if dm == 'none' else [ALL_CLEAR])
            assert fixed['dm'] == ([ALL_CLEAR] if dm == 'success' else [])
            w.capture([{**WRONG, 'id': 88}])
            later = w.check('19:21', [row(88)])
            assert later['fetches'] == [] and later['attempted_dm'] == []
            dump('full proof then capture change ' + dm, {k: fixed[k] for k in ('marker', 'dm', 'attempted_dm')})
    with world('cutoff') as w:
        w.commit(pitch=PITCH)
        w.capture([WRONG])
        cutoff = w.check('19:34', MATCH, elapsed=61)
        dirty(cutoff)
        assert not cutoff['attempted_dm'] and cutoff['marker']['escalations'] == []
        assert w.check('19:35', MATCH)['fetches'] == []
        w.commit(pitch=f'{DAY}T20:00:00-04:00')
        w.capture([RIGHT], f'{DAY}T235000Z')
        again = w.check('19:40', MATCH)
        dirty(again)
        assert again['dm'][0].startswith(FIRST)


def legacy(side):
    storage = HERE / 'legacy-states'
    if side == 'round1':
        for key, send in (('reason', 'failure'), ('tier', 'success')):
            w = World(storage / key)
            w.commit(pitch=PITCH)
            w.capture([WRONG])
            prior = w.check('18:30', MATCH, dm=send)['marker']
            if key == 'tier':
                prior = w.check('18:40', MATCH, crosswalk={})['marker']
            assert 'game_unresolved' not in prior
            dump(key, prior)
    elif side == 'base':
        w = World(storage / 'base')
        w.commit(pitch=PITCH)
        prior = w.check('18:30')['marker']
        assert 'game_unresolved' not in prior
        dump('base prior', prior)
    else:
        for key in ('reason', 'tier', 'base'):
            w = World(storage / key)
            start = w.read_marker()
            assert start and 'game_unresolved' not in start
            gap = w.check('18:45', MATCH, crosswalk={})
            w.capture([RIGHT], f'{DAY}T230000Z')
            final = w.check('19:00', MATCH)
            if key in ('reason', 'tier'):
                dirty(gap)
                dirty(final)
            else:
                assert not gap['marker'].get('game_unresolved')
                assert final['marker']['status'] == 'confirmed' and final['dm'] == [ALL_CLEAR]
            dump(key + ' follow-up', {k: final[k] for k in ('exit', 'marker', 'dm')})


def unreadable():
    with world('unreadable-after-observation') as w:
        w.commit(pitch=PITCH)
        w.capture([WRONG])
        dirty(w.check('18:30', MATCH))
        w.marker.write_text('{')
        w.capture([RIGHT], f'{DAY}T230000Z')
        result = w.check('19:00', MATCH)
        assert result['marker']['status'] == 'confirmed' and not result['attempted_dm']
        dump('unreadable prior marker loses state per R1', result['marker'])


if __name__ == '__main__':
    print_source()
    if len(sys.argv) > 1:
        legacy(sys.argv[1])
    else:
        orders()
        proof()
        unreadable()
