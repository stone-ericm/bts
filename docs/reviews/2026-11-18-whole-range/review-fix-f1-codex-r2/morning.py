"""Check the unchanged morning and health entry points, including rollback."""
import runpy
import sys

from support import ALL_CLEAR, HERE, MATCH, PITCH, WRONG, current_receipt, dump, print_source, World


def rollback(side):
    root = HERE / 'rollback-state'
    for recipient in ('success', 'failure', 'none'):
        w = World(root / recipient)
        if side == 'candidate':
            w.commit(pitch=PITCH)
            w.capture([WRONG])
            first = w.check('18:30', MATCH)
            assert first['marker']['game_unresolved'] is True and first['marker']['status'] == 'alerted'
            assert w.health()[0]['level'] == 'WARN'
        else:
            assert w.read_marker()['game_unresolved'] is True
            result = w.check('18:45', MATCH, dm=recipient)
            assert result['marker']['status'] == 'confirmed' and 'game_unresolved' not in result['marker']
            assert result['attempted_dm'] == ([] if recipient == 'none' else [ALL_CLEAR])
            assert result['dm'] == ([ALL_CLEAR] if recipient == 'success' else [])
            assert current_receipt(w, result)['qualification']['slots'][0]['state'] == 'wrong_game'
            assert w.check('19:00', MATCH)['fetches'] == []
            dump('rollback ' + recipient, {k: result[k] for k in ('exit', 'marker', 'dm', 'attempted_dm')})


if __name__ == '__main__':
    from support import ARCHIVED_REVIEW

    side, family = sys.argv[1:]
    if family == 'rollback':
        print_source()
        rollback(side)
    else:
        runpy.run_path(str(ARCHIVED_REVIEW / 'outcome.py'), run_name='__main__')
