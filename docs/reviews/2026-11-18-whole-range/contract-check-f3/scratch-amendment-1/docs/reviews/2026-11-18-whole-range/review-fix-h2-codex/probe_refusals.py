import json
from contextlib import ExitStack
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from bts import contest_state as cs, picks, scheduler
from probe_bounds import GRID, auto, chosen
from support import ABSENT, DAY, at, changed, counts, fetch, fixture, fresh, install, live, walk


CAP = getattr(cs, 'MAX_SEASON_STREAK', 57)


def paths(directory):
    state = directory / 'account_state'
    return [state / name for name in ('contest_streak.json', 'contest_ledger.jsonl', 'saver_state.json')]


def fetch_matrix(root):
    cases = []
    for field in ('activeStreak', 'seasonBestStreak'):
        for value in (v for v in GRID if type(v) is not int or v < 0):
            message = 'non-negative' if type(value) is int else 'an integer'
            cases.append((changed(counts(12, 57), field, value), f'{field} must be {message}'))
    for active, best, msg in ((12, 11, '>= activeStreak'), (401, 400, '>= activeStreak'),
                              (401, 401, f'<= {CAP}'), (57, 401, f'<= {CAP}'), (57, 10**100, f'<= {CAP}')):
        cases.append((counts(active, best), f'seasonBestStreak must be {msg}'))
    total = 0
    with walk.offline(at()), patch('bts.dm.send_dm', return_value='id') as dm:
        for dry in (False, True):
            for recipient in (False, True):
                for inside in (False, True):
                    directory = root / f'{dry}-{recipient}-{inside}' / 'picks'
                    install(directory, 'automatic', 57, 57)
                    status = directory.parent / 'health_state/contest_streak_fetch_status.json'
                    prior = tuple(path.read_bytes() for path in paths(directory))
                    for payload, message in cases:
                        walk._set_clock(at(hour=3))
                        before = fetch(directory, counts(401, 401), dry=dry, recipient=True)
                        assert before[0] == 2 and f'<= {CAP}' in before[1]
                        # Set a known last send for every trial, independent of previous trial order.
                        stamped = json.loads(status.read_text())
                        stamped['last_alert_at_by_category'] = {'actionable': at(hour=3).isoformat()}
                        stamped['last_alert_at'] = at(hour=3).isoformat()
                        status.write_text(json.dumps(stamped))
                        walk._set_clock(at(hour=8, minute=59) if inside else at(hour=9))
                        dm.reset_mock()
                        result = fetch(directory, payload, dry=dry, recipient=recipient)
                        assert result[0] == 2 and message in result[1], (payload, result)
                        assert result[2] == 'SystemExit'
                        assert tuple(path.read_bytes() for path in paths(directory)) == prior
                        logged = json.loads(status.read_text())
                        assert logged['last_error_category'] == 'actionable' and logged['last_error'].endswith(message)
                        assert dm.call_count == int(recipient and not inside)
                        if dm.called:
                            assert dm.call_args.args[0] == 'captured' and message in dm.call_args.args[1]
                        total += 1
    print(f'K5 {total} refusals through real fetch, dry-run/recipient/cooldown, cap {CAP}')


def replacement(directory):
    with walk.offline(at()), ExitStack() as stack:
        cfg, games, _, dm = fixture(stack, directory)
        install(directory, 'automatic', 57, 57)
        obs, ledger, saver = paths(directory)
        status = directory.parent / 'health_state/contest_streak_fetch_status.json'
        prior = [p.read_bytes() for p in (obs, ledger, saver, status)]
        result = fetch(directory, counts(58, 58), dry=True)
        assert result[0] == (0 if CAP == 400 else 2) and ('would write' if CAP == 400 else '<= 57') in result[1]
        assert [p.read_bytes() for p in (obs, ledger, saver)] == prior[:3]
        if CAP == 400:
            assert status.read_bytes() == prior[3] and not dm.called
        walk._set_clock(at(hour=12, minute=5))
        result = fetch(directory, counts(58, 58))
        if CAP == 57:
            assert result[0] == 2 and '<= 57' in result[1] and live(directory).streak == 57
            print('K10 base refused 58 and kept 57; dry-run failure wrote status and sent DM')
            return
        assert result[0] == 0, result
        rows = ledger.read_bytes().splitlines()
        assert rows[:-1] == prior[1].splitlines() and json.loads(rows[-1])['active_streak'] == 58
        assert (live(directory).streak, live(directory).best_streak, live(directory).best_status) == (58, 58, 'trusted')
        selection = fresh(DAY, cfg, games)
        assert (selection.streak, selection.decision.objective) == (58, 'longest_streak')
        good = [p.read_bytes() for p in (obs, ledger, saver)]
        dm.reset_mock()
        assert fetch(directory, counts(401, 401))[0] == 2 and dm.call_count == 1
        assert fetch(directory, counts(401, 401))[0] == 2 and dm.call_count == 1
        assert [p.read_bytes() for p in (obs, ledger, saver)] == good
        snap = live(directory)
        accepted = at(hour=12, minute=5)
        edge = accepted + timedelta(hours=14)
        assert not cs.automatic_reading_record(snap, edge)['stale']
        assert cs.automatic_reading_record(snap, edge + timedelta(minutes=1))['reason'] == 'old'
        print('K10 installed 58, ledger appended only on success, selected 58, 401 left files intact, age independent')


def pick_loaders(directory):
    with walk.offline(at()), ExitStack() as stack:
        cfg, games, _, _ = fixture(stack, directory)
        install(directory, 'dated', 57, 57)
        fresh(DAY, cfg, games)
        regular = directory / f'{DAY}.json'
        shadow = directory / f'{DAY}.shadow.json'
        sample = json.loads(regular.read_text())
        current = live(directory)
        valid = chosen('dated', 57, 57, 57)
        vectors = [(valid, True), (dict(valid, required=False), False),
                   (dict(valid, kind='unknown'), False), (dict(valid, streak=True), False),
                   (dict(valid, streak=-1), False), (chosen('automatic', 401, 0, 0), False),
                   (chosen('dated', 401, 401, 401), False), (chosen('dated', 57, 57, 401), False),
                   (chosen('dated', 57, 56, 57), False), (chosen('dated', 57, 57, 56), False),
                   (dict(valid, saver_available=True), False), (dict(valid, extra=1), False),
                   (dict(valid, dated=dict(valid['dated'], effective_date='20270926')), False),
                   (dict(valid, dated=dict(valid['dated'], saver_state=[])), False)]
        if CAP == 400:
            vectors.extend([(chosen('dated', 57, 57, 400), False), (chosen('automatic', 400, 0, 0), False)])
        for chosen_on, deliverable in vectors:
            raw = json.dumps(dict(sample, chosen_on=chosen_on))
            regular.write_text(raw)
            shadow.write_text(raw)
            expected = picks._valid_chosen_on(chosen_on)
            for loaded in (picks.load_pick(str(DAY), directory), picks.load_pick_bytes(str(DAY), directory)[0],
                           picks.load_shadow_pick(str(DAY), directory), picks.parse_daily_pick(raw)):
                assert loaded.chosen_on == expected
                refusal = picks.saved_pick_refusal(loaded, current)
                assert (refusal is None) == deliverable
                if expected is None:
                    assert 'there is no valid record' in refusal
        print('K7 four pick loaders and required matcher', len(vectors), 'cases')


def provenance(directory):
    with walk.offline(at()), patch('bts.dm.send_dm', Mock()):
        install(directory, 'automatic', 12, CAP)
        path = paths(directory)[0]
        state = cs._parse_state_file(path, require_dated=True)
        for changes, status in [({}, 'trusted'), ({'best_streak': None}, 'missing'),
                                ({'best_streak': -1}, 'untrusted'), ({'best_streak': 11}, 'untrusted'),
                                ({'best_streak': 401}, 'untrusted'), ({'source_date': None}, 'untrusted'),
                                ({'source_date': DAY.replace(year=2026)}, 'untrusted'),
                                ({'source_date': DAY + timedelta(days=1)}, 'untrusted'),
                                ({'schema_version': 'unknown'}, 'untrusted'), ({'source': 'manual'}, 'untrusted'),
                                ({'schema_version': 'bts_contest_streak_manual_v2', 'override_expires_at': at()}, 'untrusted')]:
            sample = replace(state, **changes)
            assert cs.classify_best_streak(sample, now=at(), now_year=DAY.year) == (sample.best_streak, status)
        assert fetch(directory, counts(12, CAP), coverage=False)[0] == 0
        assert (live(directory).best_streak, live(directory).best_status) == (CAP, 'untrusted')
        print('K8 parsed trust controls; switched high best with no coverage installs but stays untrusted')


if __name__ == '__main__':
    with TemporaryDirectory() as temp:
        root = Path(temp)
        fetch_matrix(root)
        replacement(root / 'replace' / 'picks')
        pick_loaders(root / 'pick' / 'picks')
        provenance(root / 'provenance' / 'picks')
