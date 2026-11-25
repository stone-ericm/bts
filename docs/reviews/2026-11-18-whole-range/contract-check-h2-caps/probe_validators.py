import json
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from click.testing import CliRunner
from bts import cli, contest_fetch as cf, contest_state as cs, picks as pk
from scripts.audit.season_walk import walk as w

DAY = date(2027, 7, 15)
NOW = w.REAL_DATETIME(2027, 7, 15, 12, tzinfo=w.ET)
MISSING = object()
BAD_TYPES = (True, '12', 12.0, None, MISSING, [], {})
IDENTIFIER = 'a' * 32


def changed(body, field, value):
    result = dict(body)
    if value is MISSING:
        result.pop(field, None)
    else:
        result[field] = value
    return result


def description(value):
    return '<missing>' if value is MISSING else repr(value)


def call(fn):
    try:
        return fn()
    except Exception as exc:
        return f'{type(exc).__name__}: {exc}'


def dated(active=12, best=57):
    return dict(schema_version=cs.DATED_MANUAL_SCHEMA, effective_date=str(DAY), season=DAY.year,
                active_streak=active, best_streak=best, reason='synthetic', source='manual_cli',
                saver_state='used', expires_at=f'{DAY}T23:59:59.999999-04:00', recorded_at=NOW.isoformat())


def auto(active=12, best=57):
    return dict(schema_version='bts_contest_streak_auto_v2', source='mlb_bts_profile',
                active_streak=active, best_streak=best, source_date=str(DAY), recorded_at=NOW.isoformat(),
                season=DAY.year, activation_id=IDENTIFIER, saver_available=None)


def invoke(args, *, standalone_mode=True):
    result = CliRunner().invoke(cli.cli, [str(a) for a in args], standalone_mode=standalone_mode)
    return result.exit_code, type(result.exception).__name__, result.output.strip() or str(result.exception)


def written(directory):
    return sorted(str(p.relative_to(directory)) for p in directory.rglob('*') if p.is_file())


def run():
    print(cs.__file__, 'cap', getattr(cs, 'MAX_SEASON_STREAK', cs.TARGET_STREAK))
    with TemporaryDirectory() as root, w.offline(NOW):
        directory = Path(root)
        manual = directory / 'contest_streak.manual.json'
        for number, field, values in [
            (5, 'active_streak', (-1, *BAD_TYPES, 58, 400, 401, 570)),
            (6, 'best_streak', (-1, 11, *BAD_TYPES, 58, 400, 401, 1570)),
        ]:
            for value in values:
                body = changed(dated(best=400 if value == 400 else 57), field, value)
                print('K1-3', number, field, description(value), call(lambda: cs.validate_dated_observation(body, manual)))
        orders = [(1, None, None), (2, 'effective_date', '20270715'), (3, 'season', True),
                  (4, 'expires_at', NOW.isoformat()), (7, 'saver_state', 'bad'), (8, 'recorded_at', None),
                  (9, 'source', None), (10, 'reason', None)]
        for number, field, value in orders:
            body = [] if field is None else changed(dated(), field, value)
            print('K9 dated', number, call(lambda: cs.validate_dated_observation(body, manual)))

        path = directory / 'contest_streak.json'
        for field, values in [('active_streak', (-1, *BAD_TYPES, 58, 400, 401, 570)),
                              ('best_streak', (-1, 11, *BAD_TYPES, 58, 400, 401, 1570)),
                              ('saver_available', (1, 'true')), ('season', (True, MISSING)),
                              ('activation_id', (None, 'A' * 32))]:
            for value in values:
                body = changed(auto(best=None) if field == 'active_streak' else auto(), field, value)
                path.write_text(json.dumps(body))
                for historical in [False, True]:
                    def read():
                        state = cs._read_state_file(path, require_dated=True, historical=historical)
                        return ('accepted', state.streak, state.best_streak)
                    print('K4', historical, field, description(value), call(read))
        path.write_text(json.dumps(auto(400, 399)))
        print('K4 high below active', call(lambda: cs._parse_state_file(path)))

        for key, values in [('activeStreak', (-1, *BAD_TYPES, 58, 401)),
                            ('seasonBestStreak', (-1, 11, *BAD_TYPES, 58, 400, 401, 570, 1570))]:
            for index, value in enumerate(values):
                body = changed(dict(activeStreak=12, seasonBestStreak=57), key, value)
                run_dir = directory / f'fetch-{key}-{index}'
                with w._fetch_stubs(12, 57, DAY), patch.object(cf, 'fetch_profile', lambda *a: {
                    **body, 'predictions': [{'roundId': 1, 'result': 'hit'}],
                }), patch('bts.dm.send_dm', Mock(return_value='id')) as dm:
                    result = invoke(['fetch-contest-streak', '--picks-dir', run_dir / 'picks', '--dm-recipient', 'local'])
                print('K5', key, description(value), call(lambda: cf.validate_fetch(body)), result,
                      written(run_dir), [args.args[1] for args in dm.call_args_list])

        dated_arguments = ['--date', str(DAY), '--streak', '12', '--best-streak', '57', '--saver', 'used']
        cases = {
            'below': ['--best-streak', 11], 'negative': ['--streak', -1], 'bound': ['--best-streak', 401],
            'active above best and cap': ['--streak', 401], 'date': ['--date', DAY - timedelta(days=1)],
            'text': ['--best-streak', '12.0'], 'bad saver': ['--saver', 'bad'],
            'legacy option': ['--ttl-hours', 1], 'new cap': ['--streak', 400, '--best-streak', 400],
            'old cap': ['--streak', 58, '--best-streak', 58],
        }
        for index, (label, extra) in enumerate([*cases.items(), ('missing best', []), ('missing saver', [])]):
            opts = (['--date', DAY, '--streak', 12, '--saver', 'used'] if label == 'missing best' else
                    ['--date', DAY, '--streak', 12, '--best-streak', 57] if label == 'missing saver' else dated_arguments)
            run_dir = directory / f'cli-{index}'
            args = ['set-contest-streak', '--picks-dir', run_dir, *opts, *extra]
            normal = invoke(args)
            files = written(run_dir)
            unwrapped = invoke(args, standalone_mode=False) if normal[0] else None
            print('K6', label, normal, unwrapped, files)
        legacy_dir = directory / 'legacy-cli'
        print('legacy 401', invoke(['set-contest-streak', '--picks-dir', legacy_dir, '--streak', 401,
                                     '--best-streak', 401]), written(legacy_dir))

        chosen = dict(schema='bts_pick_chosen_on_v1', required=True, kind='dated', streak=12,
                      saver_available=False, dated=dict(effective_date=str(DAY), active_streak=12,
                                                       best_streak=57, saver_state='used'))
        for field, values in [('streak', (-1, True, '12', 12.0, None, MISSING, 58, 401)),
                              ('kind', ('unknown', [], None, MISSING, 'automatic')),
                              ('required', (1, None, MISSING)), ('saver_available', (1, True, None, MISSING)),
                              ('schema', ('invalid', MISSING)), ('dated', (None, [], MISSING))]:
            for value in values:
                print('K7 outer', field, description(value), pk._valid_chosen_on(changed(chosen, field, value)))
        for field, values in [('active_streak', (-1, 11, *BAD_TYPES, 58, 401)),
                              ('best_streak', (-1, 11, *BAD_TYPES, 58, 400, 401)),
                              ('effective_date', ('20270715', '2027-02-30', *BAD_TYPES)),
                              ('saver_state', ('bad', *BAD_TYPES))]:
            for value in values:
                print('K7 identity', field, description(value),
                      pk._valid_chosen_on(dict(chosen, dated=changed(chosen['dated'], field, value))))
        for streak in (58, 100, 400, 401):
            for kind in ('automatic', 'dated', 'legacy_manual', 'model'):
                inner = dict(chosen['dated'], active_streak=streak, best_streak=streak) if kind == 'dated' else None
                print('K7 matching count', streak, kind,
                      bool(pk._valid_chosen_on(dict(chosen, streak=streak, kind=kind, dated=inner))))

        sample = cs.ContestStreakState(0, None, 'mlb_bts_profile', DAY, NOW.isoformat(), path,
                                       best_streak=57, schema_version='bts_contest_streak_auto_v2')
        for label, value in [('best', v) for v in (-1, 58, 100, 400, 401, 570, None, True, 12.5, '12', 'x')]:
            print('K8', label, value, call(lambda: cs.classify_best_streak(replace(sample, best_streak=value), now=NOW, now_year=DAY.year)))
        for updates in [dict(streak=58), dict(source_date=None), dict(source_date=DAY.replace(year=2026)),
                        dict(source_date=DAY + timedelta(days=1)), dict(source='unknown'),
                        dict(schema_version=None), dict(schema_version='bts_contest_streak_manual_v2'),
                        dict(schema_version='bts_contest_streak_manual_v2', override_expires_at=NOW),
                        dict(schema_version='bts_contest_streak_manual_v2', override_expires_at=NOW + timedelta(seconds=1))]:
            print('K8 provenance', updates, cs.classify_best_streak(replace(sample, **updates), now=NOW, now_year=DAY.year))


if __name__ == '__main__':
    run()
