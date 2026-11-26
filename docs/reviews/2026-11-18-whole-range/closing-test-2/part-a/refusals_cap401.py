# VARIANT for the closing test re-run (FINAL2): the two "too high" cases moved from 58 to 401, the cap H2 set
# (MAX_SEASON_STREAK = 400); everything else identical to refusals.py.
"""Exercise persistent refusal branches and their operator-visible signals."""

import io
import json
import os
from urllib.parse import urlsplit
from unittest.mock import patch

from bts import contest_state, scheduler
from bts.health import alert, contest_state as health_state

from clocks_and_scoring import fetch_fixture
from offline import game, instant, offline


DAY = '2027-03-24'


def replacements(path, **changes):
    path.write_text(json.dumps(json.loads(path.read_text()) | changes))


def live_failures():
    print('\n=== persistent live refusal matrix ===')
    for case, expected in (
        ('absent', 'switch is absent'), ('other-season', 'switch is other_season'),
        ('revoked', 'switch is revoked'), ('corrupt', 'corrupt season switch'),
        ('legacy-manual', 'selected legacy manual override'), ('missing-auto', 'no automatic observation'),
        ('schema', 'schema_version'), ('wrong-season', 'automatic observation season'),
        ('wrong-id', 'automatic observation activation_id'), ('bad-id', 'automatic contest activation_id invalid'),
        ('undated-auto', 'automatic contest observation is undated'), ('too-high', 'automatic contest streak invalid'),
        ('bad-dated', 'check 2'), ('malformed', 'contest streak state malformed'),
    ):
        with offline(instant(DAY, 10)) as (w, _):
            w.automatic(7)
            switch = w.account / 'season_contract.json'
            auto = w.account / 'contest_streak.json'
            manual = w.account / 'contest_streak.manual.json'
            if case == 'absent':
                switch.unlink()
            elif case == 'other-season':
                replacements(switch, season=2026)
            elif case == 'revoked':
                w.call('season-contract', 'revoke', '--season', '2027', '--reason', 'matrix',
                       '--picks-dir', str(w.picks))
            elif case == 'corrupt':
                switch.write_text('null')
            elif case == 'legacy-manual':
                w.call('set-contest-streak', '--streak', '7', '--picks-dir', str(w.picks))
            elif case == 'missing-auto':
                auto.unlink()
            elif case == 'schema':
                replacements(auto, schema_version='bts_contest_streak_auto_v1')
            elif case == 'wrong-season':
                replacements(auto, season=2026)
            elif case == 'wrong-id':
                replacements(auto, activation_id='f' * 32)
            elif case == 'bad-id':
                replacements(auto, activation_id='x')
            elif case == 'undated-auto':
                replacements(auto, source_date=None, recorded_at=None)
            elif case == 'too-high':
                replacements(auto, active_streak=401, best_streak=401)
            elif case == 'bad-dated':
                w.dated(7)
                replacements(manual, effective_date='bad')
            else:
                auto.write_text('{')
            for repeat in range(2):
                try:
                    contest_state.load_decision_streak_state(w.picks, require_contest_state=True)
                except contest_state.ContestStateError as exc:
                    assert expected in str(exc), (case, str(exc))
                    signal = str(exc).splitlines()[0]
                    checks = health_state.check(w.picks, expected=True, now=w.now)
                    assert any(a.level == 'CRITICAL' and a.message == str(exc) for a in checks)
                else:
                    raise AssertionError(f'Accepted {case}')
            print(case, signal)
            w.dated(8)
            live = contest_state.load_decision_streak_state(w.picks, require_contest_state=True)
            assert live.observation_kind == 'dated' and live.streak == 8


def configuration_and_writes():
    print('\n=== config, write failures and corrupt health-DM status ===')
    with offline(instant(DAY, 10)) as (w, _):
        w.config['scheduler']['contest_state_required'] = False
        w.config['health_checks']['contest_state_expected'] = False
        w.write_config()
        w.call('schedule', '--config', str(w.config_path), code=2)
        w.call('orchestrate', '--date', DAY, '--config', str(w.config_path), code=2)
        w.call('entry-intent', '--config', str(w.config_path), code=1)
        assert not (w.root / 'heartbeat').exists() and not w.events
        w.config['scheduler']['contest_state_required'] = True
        w.write_config()
        w.account.mkdir()
        (w.account / 'contest_manual_requests.jsonl').mkdir()
        for _ in range(2):
            w.dated(8, code=1)
            assert not (w.account / 'contest_streak.manual.json').exists()
        # A human must remove the directory at the ledger path.
        (w.account / 'contest_manual_requests.jsonl').rmdir()
        w.dated(8)

        for name, contents in [('array', '[]'), ('null', 'null'), ('bad-utf8', b'\xff'), ('fifo', None)]:
            status = w.root / f'alert-{name}'
            if contents is None:
                os.mkfifo(status)
            elif isinstance(contents, bytes):
                status.write_bytes(contents)
            else:
                status.write_text(contents)
            before = len(w.sends('health-dm'))
            invoked = alert.dispatch_dm_for_health_alerts(
                [alert.Alert('CRITICAL', 'contest_state', 'probe')], 'offline.invalid', status_path=status,
            )
            assert invoked and len(w.sends('health-dm')) == before + 1
            if contents is not None:
                assert json.loads(status.read_text())['status'] == 'sent'


def fetch_failures():
    print('\n=== captured fetch shape, switch and status failures ===')
    for case, streak, rounds, rows, why in (
        ('rows-not-list', 7, True, {}, 'predictions must be a list'),
        ('rows-not-objects', 7, True, [0], 'predictions must contain only objects'),
        ('rounds-empty', 7, False, [], 'rounds mapping must be non-empty'),
        ('streak-too-high', 401, True, [], '<= 400'),
        ('switch', 7, True, [], 'season contract corrupt'),
    ):
        with offline(instant(DAY, 10)) as (w, mp):
            w.automatic(7)
            if case == 'switch':
                (w.account / 'season_contract.json').write_text('null')
            auto = w.account / 'contest_streak.json'
            previous = auto.read_bytes()
            fetch_fixture(w, mp, rounds={9: w.now.date()} if rounds else {}, streak=streak, rows=rows)
            for count in range(2):
                result = w.call('fetch-contest-streak', '--picks-dir', str(w.picks),
                                '--expected-username', 'offline', '--dm-recipient', 'offline.invalid', code=2)
                assert why in result.stderr and auto.read_bytes() == previous
                assert len(w.sends('dm')) == 1  # failure DM cooldown
                status = json.loads((w.health / 'contest_streak_fetch_status.json').read_text())
                assert why in status['last_error']

    with offline(instant(DAY, 10)) as (w, mp):
        w.automatic(7)
        w.health.mkdir(exist_ok=True)
        status = w.health / 'contest_streak_fetch_status.json'
        status.write_text('[]')
        fetch_fixture(w, mp, rounds={9: w.now.date()}, streak=8)
        for count in range(2):
            result = w.call('fetch-contest-streak', '--picks-dir', str(w.picks),
                            '--dm-recipient', 'offline.invalid', code=1, catch=True)
            assert isinstance(result.exception, AttributeError)
            assert json.loads((w.account / 'contest_streak.json').read_text())['active_streak'] == 8
            assert status.read_text() == '[]' and not w.sends('dm')
            assert not (w.account / 'contest_ledger.jsonl').exists()
        print('Fetch status array repeatedly leaves the new auto file installed, ledger and saver step skipped')


def malformed_schedule():
    print('\n=== scheduler and day-outcome share a silent schedule failure ===')
    for malformed in (False, True):
        with offline(instant(DAY, 10)) as (w, mp):
            def response(url, **kwargs):
                assert urlsplit(url).path == '/api/v1/schedule'
                body = {'message': 'synthetic upstream error'} if malformed else {'dates': []}
                return io.BytesIO(json.dumps(body).encode())

            mp.setattr(scheduler, 'retry_urlopen', response)
            scheduler.run_day(DAY, w.config)
            outcome = w.call('day-outcome', '--date', DAY, '--config', str(w.config_path),
                             '--picks-dir', str(w.picks), '--dm-recipient', 'offline.invalid')
            assert json.loads(outcome.stdout)['state'] == 'no_games' and not w.sends('dm')
            assert not (w.picks / DAY / 'scheduler_state.json').exists()
            assert not w.sends('health-dm')


def morning_dispatch_across_midnight():
    print('\n=== morning dedup when the dispatcher samples its clock after midnight ===')
    with offline(instant(DAY, 23, 59, 59)) as (w, _):
        w.schedule[DAY] = [game(101, instant('2027-03-25', 0, 55))]
        dispatch = alert.dispatch_dm_for_health_alerts

        def delay(*args, **kwargs):
            if 'morning' in str(kwargs.get('status_path')):
                w.advance(2)
            return dispatch(*args, **kwargs)

        with patch.object(alert, 'dispatch_dm_for_health_alerts', delay):
            scheduler.run_day(DAY, w.config)
        status = json.loads((w.health / 'contest_morning_check_dm_delivery_status.json').read_text())
        sent = [t for kind, t, body in w.events if kind == 'health-dm' and 'Morning check' in body]
        assert sent == [instant('2027-03-25', 0, 0, 1)]
        assert status['sent_day'] == DAY
        print('Morning alert pinned to', status['sent_day'], 'and emitted at', sent[0].isoformat())


if __name__ == '__main__':
    import sys
    import traceback

    failed = False
    for check in (live_failures, configuration_and_writes, fetch_failures, malformed_schedule,
                  morning_dispatch_across_midnight):
        if len(sys.argv) == 1 or check.__name__ in sys.argv[1:]:
            try:
                check()
            except Exception:
                traceback.print_exc()
                failed = True
    raise SystemExit(failed)
