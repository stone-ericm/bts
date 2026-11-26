"""Check source combinations, duplicates and exact saved game IDs through the CLI."""
import sys
from itertools import product

from support import DOUBLE, MATCH, PITCH, RIGHT, RIGHT_DD, WRONG, WRONG_DD, current_receipt, print_source, world


def check_sources(candidate):
    for sources, (wrong_primary, wrong_second) in product(product(('pending', 'profile', 'both'), repeat=2),
                                                          product((False, True), repeat=2)):
        pending, settled = [], []
        for entry, source in zip(DOUBLE, sources, strict=True):
            if source in ('pending', 'both'):
                pending.append(entry)
            if source in ('profile', 'both'):
                settled.append({**entry, 'result': 'hit'})
        with world('sources') as w:
            w.commit(pitch=PITCH, dd=(1020, 201, PITCH))
            w.capture([WRONG if wrong_primary else RIGHT, WRONG_DD if wrong_second else RIGHT_DD])
            result = w.check('18:30', pending, profile={'predictions': [{'roundId': 1, 'roundPredictions': settled}]})
            states = [s['state'] for s in current_receipt(w, result)['qualification']['slots']]
            assert states == ['wrong_game' if wrong else 'confirmed' for wrong in (wrong_primary, wrong_second)]
            if candidate and (wrong_primary or wrong_second):
                assert result['exit'] == 1
                message, = result['dm']
                assert ('pick Batter 1010:' in message) is wrong_primary
                assert ('double-down Batter 1020:' in message) is wrong_second
            else:
                assert result['exit'] == 0 and result['marker']['status'] == 'confirmed' and not result['dm']
    print('36 single-source and mixed-source double checks passed')


def check_ids(candidate):
    for game in (None, '101', 101.0, True, 101):
        with world('saved-game-id') as w:
            w.commit(game=game, pitch=PITCH)
            w.capture([WRONG])
            result = w.check('18:30', MATCH)
            assert result['exit'] == int(candidate and type(game) is int)
            print(f'saved game {game!r} ({type(game).__name__}): {result["marker"]["reason"]}')
    for other_game in (101, 202):
        with world('same-batter') as w:
            w.commit(pitch=PITCH, dd=(1010, other_game, PITCH))
            w.capture([WRONG])
            assert w.check('18:30', MATCH)['marker']['reason'] == 'match'
    print('shared batter confirms with and without an identical saved game, absent prior observation')


if __name__ == '__main__':
    print_source()
    candidate = sys.argv[1] == 'candidate'
    check_sources(candidate)
    check_ids(candidate)
