"""Compare the original watchdog and F1 from file-backed observations."""
from collections import Counter
from datetime import datetime
from itertools import combinations, product
import json
import sys
from types import SimpleNamespace

from bts.daily_decision import write_decision
from support import ALL_CLEAR, Arena, DAY, HERE, PITCH, normalized, print_source, read_marker_bytes, row

STATES = ('confirmed', 'wrong_game', 'unit_unverified', 'wrong_round', 'ambiguous', 'missing', 'player_unverified')
DELIVERIES = (None, 'legacy', 'delivered', 'private_locked', 'locked_unconfirmed', 'not_applicable')
TIMES = (('18:24', 0), ('18:25', 0), ('18:45', 0), ('19:04', 0), ('19:05', 0),
         ('19:19', 0), ('19:20', 0), ('19:34', 61), ('19:35', 0), ('19:36', 0))
DM = ('success', 'failure', 'none')
TIERS = ('initial', 't30', 't15', 'wrong_game')
ALL_TIERS = tuple(t for n in range(5) for t in combinations(TIERS, n))
PRIORS = (
    'none', 'unreadable', 'invalid_utf8', ('alerted', 'no_pick', None), ('dm_failed', 'no_pick', None),
    ('present_unverified', 'present_unverified', ()), ('confirmed', 'match', ()),
    ('alerted', 'wrong_game', ('wrong_game',), '2027-03-23'), ('alerted', 'wrong_game', ()),
    ('alerted', 'wrong_game', ('wrong_game',)), ('dm_failed', 'wrong_game', ()),
    ('present_unverified', 'present_unverified', (), DAY, True),
)


def cases(group):
    if group == 'qualification':
        singles_doubles = [(state,) for state in STATES] + list(product(STATES, repeat=2))
        singles_doubles += [('same_batter_same_game',), ('same_batter_other_game',)]
        for states, delivery, prior in product(singles_doubles, DELIVERIES, PRIORS):
            yield states, delivery, prior, ('18:30', 0), 'success', 'derived'
    elif group == 'tiers':
        for i, (tiers, status, at, dm, outcome) in enumerate(product(
                ALL_TIERS, ('alerted', 'dm_failed', 'present_unverified'), TIMES, DM,
                ('no_pick', 'mismatch', 'gap', 'derived'))):
            reason = 'wrong_game' if i % 11 == 0 else 'present_unverified' if status == 'present_unverified' else 'no_pick'
            history = (status, reason, tiers, DAY, i % 13 == 0)
            states = (('wrong_game', 'confirmed'), ('confirmed',), ('unit_unverified', 'confirmed'))[i % 3]
            yield states, DELIVERIES[i % len(DELIVERIES)], history, at, dm, outcome
    elif group == 'fetch':
        for prior, delivery, reason in product(PRIORS, DELIVERIES, (
                'profile', 'pending', 'rounds', 'crosswalk', 'identity', 'missing_file')):
            yield ('wrong_game',), delivery, prior, ('18:30', 0), 'success', reason
    else:
        raise ValueError(group)


def input_files(w, states, delivery, history, outcome):
    shared = len(states) == 1 and states[0].startswith('same_batter')
    if outcome != 'missing_file':
        dd = None
        if shared:
            dd = (1010, 101 if states[0].endswith('same_game') else 201, PITCH)
        elif len(states) == 2:
            dd = (1020, 201, PITCH)
        if delivery == 'legacy':
            daily = w.commit(pitch=PITCH, dd=dd, delivery=None)
            daily.notification_sent = True
            from bts.picks import save_pick
            save_pick(daily, w.picks)
        else:
            w.commit(pitch=PITCH, dd=dd, delivery=delivery)
        if delivery == 'not_applicable':
            assert write_decision(DAY, w.picks, action='skip', source='mdp', delivery_status=delivery,
                                  scoreable=False, now=datetime.fromisoformat(f'{DAY}T20:00:00+00:00'))
    if isinstance(history, tuple):
        status, reason, tiers, *extra = history
        marker = dict(status=status, reason=reason, date=extra[0] if extra else DAY)
        if tiers is not None:
            marker['escalations'] = list(tiers)
        if len(extra) == 2:
            marker['game_unresolved'] = extra[1]
        w.set_marker(marker)
    elif history != 'none':
        w.marker.parent.mkdir(exist_ok=True)
        w.marker.write_bytes(b'\xff' if history == 'invalid_utf8' else b'{')
    rows, bindings = [], []
    for i, state in enumerate(('wrong_game',) if shared else states):
        unit, batter, game = ((77, 1, 101), (78, 2, 201))[i]
        if state == 'missing':
            continue
        rows.append(row(unit, str(batter) if state == 'player_unverified' else batter, i + 1))
        if state == 'ambiguous':
            rows.append(row(unit + 100, batter, i + 1))
        bounds = None if state == 'unit_unverified' else game + (state == 'wrong_game')
        bindings.append(dict(id=unit, roundId=2 if state == 'wrong_round' else 1, feedId=bounds))
    w.capture(bindings)
    return rows


def observe(w, case):
    states, delivery, history, (at, elapsed), dm, outcome = case
    rows = input_files(w, states, delivery, history, outcome)
    kwargs = {}
    if outcome == 'no_pick':
        rows = []
    elif outcome == 'mismatch':
        rows = [row(player=3)]
    elif outcome == 'gap':
        kwargs['crosswalk'] = {}
    elif outcome in ('profile', 'pending', 'rounds', 'crosswalk'):
        kwargs['fetch_failure'] = outcome
    elif outcome == 'identity':
        kwargs['more_patches'] = [('bts.leaderboard.auth.fetch_login_session',
            lambda *a, **k: SimpleNamespace(user_id='fixture', username='other', xsid='fixture'))]
    result = w.check(at, rows, dm=dm, elapsed=elapsed, **kwargs)
    result['marker_bytes'] = read_marker_bytes(w)
    return normalized(result, w.root)


def active_prior(history):
    if isinstance(history, str):
        return False
    _, reason, tiers, *extra = history
    if extra and extra[0] != DAY:
        return False
    return reason == 'wrong_game' or 'wrong_game' in (tiers or ()) or (len(extra) == 2 and extra[1] is True)


def compare(case, base, tested):
    states, _, history, _, _, _ = case
    before, after = ((o['receipts'] or [{}])[0] for o in (base, tested))
    verdict = (before.get('verifier') or {}).get('reason')
    slots = (before.get('qualification') or {}).get('slots', [])
    distinct = not states[0].startswith('same_batter')
    current_wrong = distinct and any(s['state'] == 'wrong_game' and type(s['game_pk']) is int for s in slots)
    all_proved = distinct and slots and all(s['state'] == 'confirmed' for s in slots)
    seen_wrong = active_prior(history)
    if verdict == 'match' and (current_wrong or (seen_wrong and not all_proved)):
        assert tested['exit'] == 1 and tested['marker']['reason'] == 'wrong_game', tested
        assert tested['marker']['game_unresolved'] is True and ALL_CLEAR not in tested['attempted_dm'], tested
        omitted = {'exit_code', 'marker_status'}
        assert {k: v for k, v in before.items() if k not in omitted} == {k: v for k, v in after.items() if k not in omitted}
        return 'A1 current' if current_wrong else 'A1 prior'
    if verdict == 'match' and seen_wrong:
        assert tested['exit'] == 0 and tested['marker']['status'] == 'confirmed', tested
        assert 'game_unresolved' not in tested['marker'] and before == after
        return 'full proof after observation'
    if seen_wrong and verdict in ('no_pick', 'mismatch', 'present_unverified'):
        assert tested['marker']['game_unresolved'] is True and before.get('verifier') == after.get('verifier')
        return 'R1 nonterminal history'
    assert base == tested, (base, tested)
    return 'identical'


def main():
    side, group, *sharding = sys.argv[1:]
    shard, count = map(int, sharding or ('0', '1'))
    print_source()
    arena = Arena(f'{side}-{group}-{shard}')
    filename = HERE / f'grid-{group}-{shard}.jsonl'
    observations = [(i, case) for i, case in enumerate(cases(group)) if i % count == shard]
    verdicts, slots, comparisons = Counter(), Counter(), Counter()
    with filename.open('w' if side == 'base' else 'r') as data:
        for i, case in observations:
            w = arena.world()
            observed = observe(w, case)
            arena.clear(w)
            if side == 'base':
                data.write(json.dumps([i, observed]) + '\n')
                continue
            index, base = json.loads(data.readline())
            assert index == i
            rec = (base['receipts'] or [{}])[0]
            verdicts.update([(rec.get('verifier') or {}).get('reason') or 'no verifier'])
            slots.update(s['state'] for s in (rec.get('qualification') or {}).get('slots', []))
            try:
                comparisons.update([compare(case, base, observed)])
            except AssertionError:
                print('Unexpected comparison', i, case, flush=True)
                raise
    print(side, group, shard, len(observations), dict(comparisons), dict(verdicts), dict(slots), flush=True)


if __name__ == '__main__':
    main()
