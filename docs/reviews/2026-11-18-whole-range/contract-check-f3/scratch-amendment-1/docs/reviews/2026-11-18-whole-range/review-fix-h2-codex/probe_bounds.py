import argparse
import json
from dataclasses import asdict, replace
from datetime import time, timedelta
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import bts
from bts import contest_fetch as cf, contest_state as cs, picks as pk
from support import ABSENT, DAY, at, changed, counts, invoke, walk


GRID = (-1, 0, 1, 56, 57, 58, 100, 399, 400, 401, 10**100,
        False, True, 57.0, 58.0, '57', '58', None, ABSENT, [], {})
PAIRS = list(product(GRID, repeat=2))


def dated(active, best):
    body = {'schema_version': cs.DATED_MANUAL_SCHEMA, 'effective_date': str(DAY), 'season': DAY.year,
            'expires_at': at().combine(DAY, time.max, walk.ET).isoformat(), 'recorded_at': at().isoformat(),
            'saver_state': 'used', 'source': 'manual_cli', 'reason': 'review'}
    return changed(changed(body, 'active_streak', active), 'best_streak', best)


def auto(active, best):
    body = {'schema_version': 'bts_contest_streak_auto_v2', 'source': 'mlb_bts_profile',
            'source_date': str(DAY), 'recorded_at': at().isoformat(), 'season': DAY.year,
            'activation_id': 'e' * 32, 'saver_available': None}
    return changed(changed(body, 'active_streak', active), 'best_streak', best)


def outcome(operation, root):
    try:
        value = operation()
    except Exception as exc:
        return ['error', type(exc).__name__, str(exc).replace(str(root), '<root>')]
    if isinstance(value, cs.ContestStreakState):
        value = asdict(value)
        value['path'] = str(value['path'].relative_to(root))
    return ['return', json.loads(json.dumps(value, default=str))]


def chosen(kind, streak, active, best):
    body = {'schema': 'bts_pick_chosen_on_v1', 'required': True, 'kind': kind,
            'saver_available': False, 'dated': None}
    if kind == 'dated':
        body['dated'] = changed(changed({'effective_date': str(DAY), 'saver_state': 'used'},
                                       'active_streak', active), 'best_streak', best)
    return changed(body, 'streak', streak)


def high(value):
    return type(value) is int and 58 <= value <= 400


def nonnegative(value):
    return type(value) is int and value >= 0


def cli_case(directory, active, best, extra=(), dated_mode=True):
    args = ['set-contest-streak', '--picks-dir', directory]
    if dated_mode:
        args += ['--date', DAY, '--saver', 'used']
    for name, value in (('--streak', active), ('--best-streak', best)):
        if value is not ABSENT:
            args.extend((name, value))
    args.extend(extra)
    expected_valid = (nonnegative(active) and nonnegative(best) and active <= best
                      and (not dated_mode or best <= LIMIT) and not extra)
    if expected_valid:
        result = invoke(*args, standalone=False)
    else:
        with patch.object(Path, 'read_text', side_effect=AssertionError('read before validation')):
            result = invoke(*args, standalone=False)
    files = {str(path.relative_to(directory)): path.read_text() for path in directory.rglob('*') if path.is_file()}
    result = [part.replace(str(directory), '<root>') if isinstance(part, str) else part for part in result]
    assert (result[0] == 0) == expected_valid, (args, result)
    if not expected_valid:
        assert result[2] in {'BadParameter', 'MissingParameter'} and not files, (args, result, files)
    return [result, files]


LIMIT = getattr(cs, 'MAX_SEASON_STREAK', 57)


def run():
    print('bts', bts.__file__, 'cap', LIMIT)
    records = {}
    with TemporaryDirectory() as temp, walk.offline(at()):
        root = Path(temp)
        manual = root / 'contest_streak.manual.json'
        automatic = root / 'contest_streak.json'

        def reader(path, body, history):
            path.write_text(json.dumps(body))
            return outcome(lambda: cs._read_state_file(path, require_dated=True, historical=history), root)

        for index, (active, best) in enumerate(PAIRS):
            valid_active = nonnegative(active) and active <= LIMIT
            valid_best = nonnegative(best) and valid_active and active <= best <= LIMIT
            auto_ok = valid_active and (type(best) is not int or active <= best <= LIMIT)
            for history in (False, True):
                result = reader(manual, dated(active, best), history)
                assert (result[0] == 'return') == valid_best, (active, best, result)
                if not valid_best:
                    check = 6 if valid_active else 5
                    field = 'best_streak' if valid_active else 'active_streak'
                    assert (f'for {field}.' if history else f'check {check}, {field}:') in result[2], result
                records[f'S1/{index}/{history}'] = result
                result = reader(automatic, auto(active, best), history)
                assert (result[0] == 'return') == auto_ok, (active, best, result)
                if auto_ok:
                    assert result[1]['best_streak'] == (best if type(best) is int else None)
                records[f'S2/{index}/{history}'] = result
            parsed_best = best if type(best) is int else None
            sample = cs._read_state_file(automatic, require_dated=True, historical=False) if auto_ok else None
            if sample is None:
                automatic.write_text(json.dumps(auto(0, 0)))
                sample = cs._parse_state_file(automatic)
            sample = replace(sample, streak=active if type(active) is int else 0, best_streak=parsed_best)
            result = outcome(lambda: cs.classify_best_streak(sample, now=at(), now_year=DAY.year), root)
            expected = 'missing' if parsed_best is None else ('trusted' if max(0, sample.streak) <= parsed_best <= LIMIT else 'untrusted')
            assert result == ['return', [parsed_best, expected]], (sample, result)
            records[f'S3/{index}'] = result
            payload = changed(changed(counts(0, 0), 'activeStreak', active), 'seasonBestStreak', best)
            result = outcome(lambda: cf.validate_fetch(payload), root)
            assert (result[0] == 'return') == valid_best, (active, best, result)
            if not valid_best:
                if not nonnegative(active) or not nonnegative(best):
                    field, value = ('activeStreak', active) if not nonnegative(active) else ('seasonBestStreak', best)
                    expected = f'{field} must be ' + ('non-negative' if type(value) is int else 'an integer')
                else:
                    expected = 'seasonBestStreak must be ' + ('>= activeStreak' if best < active else f'<= {LIMIT}')
                assert result == ['error', 'ContestFetchError', expected]
            records[f'S4/{index}'] = result
            cli_active, cli_best = (int(value) if type(value) is str and value.isdecimal() else value for value in (active, best))
            records[f'S5/{index}'] = cli_case(root / f'cli-{index}', cli_active, cli_best)
            for kind in ('dated', 'automatic', 'legacy_manual', 'model'):
                record = chosen(kind, active, active, best)
                valid = (valid_best if kind == 'dated' else valid_active if kind == 'automatic' else nonnegative(active))
                normalized = pk._valid_chosen_on(record)
                assert normalized == (record if valid else None), (record, normalized)
                records[f'S6/{index}/{kind}'] = normalized

        good = dated(12, 57)
        controls = [(2, 'effective_date', DAY.strftime('%Y%m%d')), (3, 'season', True),
                    (4, 'expires_at', at().isoformat()), (7, 'saver_state', []),
                    (8, 'recorded_at', None), (9, 'source', []), (10, 'reason', None)]
        for offset, (number, field, value) in enumerate(controls):
            for later_number, later_field, later_value in controls[offset:]:
                invalid = changed(changed(good.copy(), later_field, later_value), field, value)
                for history in (False, True):
                    result = reader(manual, invalid, history)
                    assert (f'for {field}.' if history else f'check {number}, {field}:') in result[2], result
                    records[f'order-{number}-{later_number}-{history}'] = result
        invalid = outcome(lambda: cs.validate_dated_observation([], manual), root)
        assert 'check 1, file:' in invalid[2]
        records['order-1'] = invalid
        for i, opts in enumerate((('--date', DAY + timedelta(days=1)), ('--ttl-hours', 1), ('--saver', 'bad'))):
            records[f'cli-control-{i}'] = cli_case(root / f'cli-control-{i}', 12, 57, opts)
        records['cli-legacy'] = cli_case(root / 'legacy', 401, 401, dated_mode=False)
    return records


def compare(base, head):
    allowed = {'S1': 0, 'S2': 0, 'S3': 0, 'S4': 0, 'S5': 0, 'S6': 0}
    assert base.keys() == head.keys()

    def remove_message_bound(result):
        text = json.dumps(result)
        for phrase in ('from 0 to', 'from active_streak to', 'must be <=', 'must be at most'):
            text = text.replace(f'{phrase} 57', f'{phrase} LIMIT').replace(f'{phrase} 400', f'{phrase} LIMIT')
        return text

    for key in base:
        if remove_message_bound(base[key]) == remove_message_bound(head[key]):
            continue
        site, index, *_ = key.split('/')
        active, best = PAIRS[int(index)]
        assert high(active) or high(best) or site == 'S5' and '58' in (active, best), (key, base[key], head[key])
        allowed[site] += 1
        if site == 'S5' and head[key][0][0] == 0:
            expected_files = base[f'S5/{PAIRS.index((0, 0))}'][1]
            assert head[key][1].keys() == expected_files.keys()
        if site in {'S1', 'S2'} and head[key][0] == 'return':
            low_active, low_best = min(active, 57), min(best, 57) if type(best) is int else best
            reference_index = next(i for i, pair in enumerate(PAIRS)
                                   if pair[0] is low_active and pair[1] is low_best
                                   or all(type(a) is type(b) and a == b for a, b in zip(pair, (low_active, low_best))))
            counterpart = base[f'{site}/{reference_index}/{key.rsplit("/", 1)[1]}']
            assert counterpart[0] == 'return'
            restored = dict(counterpart[1], streak=active, best_streak=head[key][1]['best_streak'])
            assert restored == head[key][1], key
    assert all(allowed.values()), allowed
    print(f'{len(base)} results compared, differences by site', allowed)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out', type=Path)
    parser.add_argument('--base', type=Path)
    args = parser.parse_args()
    records = run()
    args.out.write_text(json.dumps(records, indent=1))
    if args.base:
        compare(json.loads(args.base.read_text()), records)
