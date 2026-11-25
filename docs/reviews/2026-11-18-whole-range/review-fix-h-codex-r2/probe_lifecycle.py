from contextlib import ExitStack, contextmanager, nullcontext
from datetime import date, datetime, time, timedelta
from itertools import product
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from bts import contest_state, daily_decision, orchestrator, picks, scheduler, strategy
from bts.health import tail_policy as tail_health
from scripts.audit.season_walk import walk


def clock(day, hour=12, minute=0):
    return datetime.combine(day, time(hour, minute), walk.ET)


@contextmanager
def slate(folder, day, kind='dated', streak=57):
    folder.mkdir(exist_ok=True)
    frame = walk._predictions(day, .84).iloc[:2].copy()
    statuses = dict.fromkeys(map(int, frame.game_pk), {'abstract': 'P', 'detailed': 'Scheduled'})
    cfg = walk._config(folder) | {'bluesky': {'dm_recipient': 'review'}}
    cfg['health_checks']['enabled'] = False
    cfg['orchestrator']['heartbeat_path'] = str(folder / '.heartbeat')
    with walk.offline(clock(day)), ExitStack() as stack:
        for module, attr, replacement in (
            (orchestrator, 'run_cascade', Mock(return_value=(frame, 'offline'))),
            (picks, 'get_game_statuses_detailed', Mock(return_value=statuses)),
            (strategy, 'get_game_statuses_detailed', Mock(return_value=statuses)),
            (scheduler, 'heartbeat_watchdog', lambda *args, **kwargs: nullcontext()),
            (scheduler, 'count_new_confirmations', Mock(return_value=6)),
            (scheduler, '_lock_decision_from_predictions', Mock(return_value=scheduler.LockDecision(True, None, True))),
            (scheduler, '_void_committed_games', Mock(return_value={})),
            (scheduler, '_trigger_live_forward_capture_on_lock', Mock()),
            (scheduler, '_alert_delivery_outcome_unknown', Mock()),
            (scheduler, '_alert_contest_state_failure', Mock()),
            (scheduler, 'notify_ready', Mock()),
            (scheduler, 'notify_watchdog', Mock()),
        ):
            stack.enter_context(patch.object(module, attr, replacement))
        assert not walk.install_state(folder, day, walk.State(kind, streak, 57, 'used', .84))
        yield cfg, frame, statuses


def load_pick_after_check(folder, day, cfg, statuses):
    result = scheduler.run_single_check(str(day), list(statuses), set(), cfg, .01)
    saved = picks.load_pick(str(day), folder)
    observation = contest_state.load_decision_streak_state(folder, require_contest_state=True)
    assert picks._valid_chosen_on(saved.chosen_on) == picks.chosen_on_record(observation, required=True)
    assert saved.policy_decision == result['selection'].pick_result.daily.policy_decision
    return saved, result['selection']


def deliver(saved, cfg, folder, selection=None):
    state = scheduler.SchedulerState(saved.date, None, [], [], [], False, None, None, None)
    return scheduler._deliver_and_lock_pick(saved, cfg, folder, state, saved.date,
                                           'review delivery', selection=selection)


def check_record(folder, saved, status, objective='longest_streak'):
    record = daily_decision.load_decision(saved.date, folder)
    assert (record['action'], daily_decision.decision_objective(record), record['delivery_status']) == (
        'double' if saved.double_down else 'single', objective, status)
    assert record['scoreable'] and record['primary']['batter_id'] == saved.pick.batter_id
    assert not daily_decision.is_reach57_mdp_skip(record)
    for field, stored in [('source', 'source'), ('objective', 'objective'), ('best_supplied', 'best_streak'),
                          ('best_status', 'best_status'), ('effective_best', 'effective_best'),
                          ('tail_sha256', 'tail_policy_sha256'), ('degraded_reason', 'degraded_reason'),
                          ('guard', 'policy_guard')]:
        assert record[stored] == saved.policy_decision[field], (field, record)
    if saved.double_down:
        assert record['double_down']['batter_id'] == saved.double_down.batter_id
    return record


def check_fresh(root):
    dates = (date(2027, 7, 15), *(date(2027, 9, x) for x in (24, 25, 26)))
    for day, kind in product(dates, ('dated', 'automatic')):
        folder = root / f'{day}-{kind}'
        with slate(folder, day, kind) as (cfg, _, statuses):
            _, direct, _ = orchestrator.run_and_pick(cfg, str(day))
            saved, selection = load_pick_after_check(folder, day, cfg, statuses)
            assert direct.action == selection.action == ('single' if day.month == 7 else 'double')
            assert saved.policy_decision == direct.pick_result.daily.policy_decision
            assert saved.chosen_on == direct.pick_result.daily.chosen_on
            with patch('bts.dm.send_dm', return_value='capture') as transport:
                assert deliver(saved, cfg, folder, selection)
            transport.assert_called_once()
            msg = transport.call_args.args[1]
            assert ('Batter 0' in msg, 'Batter 1' in msg) == (True, day.month == 9)
            record = check_record(folder, saved, 'delivered')
            assert record['streak'] == 57 and record['tail_policy_sha256'] == strategy._load_mdp()['tail'].sha256
            print('fresh', day, kind, record['action'], record['objective'])


def restart(root):
    day = date(2027, 9, 26)
    for scenario in ['retryable-failure', 'unconfirmed', 'empty-refresh']:
        folder = root / scenario
        with slate(folder, day, 'automatic') as (cfg, frame, statuses):
            saved, _ = load_pick_after_check(folder, day, cfg, statuses)
            assert saved.double_down and daily_decision.load_decision(str(day), folder) is None
            reloaded = picks.load_pick(str(day), folder)
            assert reloaded is not saved and reloaded.policy_decision == saved.policy_decision
            with patch.object(orchestrator, 'run_cascade', return_value=(frame.iloc[:0], 'empty')):
                empty = scheduler._refresh_pick_at_fallback_decision(cfg, str(day), reloaded, .01)
            assert empty.daily is reloaded and empty.selection is None
            expected_delivery = 'delivered'
            if scenario == 'retryable-failure':
                with patch('bts.dm.send_dm', side_effect=OSError('transport rejected')) as transport:
                    assert not deliver(reloaded, cfg, folder)
                transport.assert_called_once()
                reloaded = picks.load_pick(str(day), folder)
                assert not reloaded.delivery_attempted and daily_decision.load_decision(str(day), folder) is None
            if scenario == 'unconfirmed':
                with patch('bts.dm.send_dm', side_effect=KeyboardInterrupt):
                    try:
                        deliver(reloaded, cfg, folder)
                    except KeyboardInterrupt:
                        pass
                    else:
                        raise AssertionError('transport interrupt swallowed')
                reloaded = picks.load_pick(str(day), folder)
                assert reloaded.delivery_attempted and daily_decision.load_decision(str(day), folder) is None
                schedule = [{'gamePk': int(row.game_pk), 'gameDate': row.game_time,
                             'status': {'abstractGameCode': 'P', 'detailedState': 'Scheduled'},
                             'teams': {'away': {'team': {'name': row.team}},
                                       'home': {'team': {'name': row.pitcher_team}}}}
                            for row in frame.itertuples()]
                walk._set_clock(clock(day, 18, 40))
                with (patch.object(scheduler, 'fetch_schedule', side_effect=[schedule, []]),
                      patch.object(scheduler, '_watchdog_ping_sleep'),
                      patch.object(scheduler, '_idle_until_next_wakeup'),
                      patch.object(scheduler, 'run_result_polling', return_value='final'),
                      patch.object(scheduler, 'retry_urlopen', side_effect=OSError('offline')),
                      patch('bts.dm.send_dm', side_effect=AssertionError('second send')) as transport):
                    scheduler.run_day(str(day), cfg)
                transport.assert_not_called()
                expected_delivery = 'locked_unconfirmed'
            else:
                with patch('bts.dm.send_dm', return_value='capture-after-restart') as transport:
                    assert deliver(reloaded, cfg, folder, empty.selection)
                transport.assert_called_once()
            record = check_record(folder, reloaded, expected_delivery)
            assert record['tail_policy_sha256'] and record['streak'] is None
            print('restart', scenario, record['delivery_status'], record['objective'])


def check_calendar_and_reuse(root):
    finale = date(2027, 9, 26)
    for offset, expected in ((0, 'double'), (1, 'skip')):
        day = finale + timedelta(days=offset)
        with slate(root / str(day), day) as (cfg, _, _):
            _, sel, _ = orchestrator.run_and_pick(cfg, str(day))
            assert (sel.action, sel.decision.days_effective) == (expected, 1 - offset)
            print('finale sequence', day, sel.action, sel.decision.days_effective)
    for before, after, objective in [(56, 57, 'reach57'), (57, 0, 'longest_streak')]:
        folder = root / f'reuse-{before}'
        with slate(folder, finale, 'automatic', before) as (cfg, frame, statuses):
            saved, _ = load_pick_after_check(folder, finale, cfg, statuses)
            walk._set_clock(clock(finale, 13))
            with walk._fetch_stubs(after, 57, finale - timedelta(days=1)):
                code, output = walk._invoke('fetch-contest-streak', '--picks-dir', folder,
                                            '--expected-username', walk.ACCOUNT)
            assert code == 0, output
            with patch.object(orchestrator, 'run_cascade', return_value=(frame.iloc[:0], 'empty')):
                refresh = scheduler._refresh_pick_at_fallback_decision(cfg, str(finale), saved, .01)
            assert refresh.daily is saved and refresh.selection is None
            with patch('bts.dm.send_dm', return_value='capture') as transport:
                assert deliver(saved, cfg, folder, refresh.selection)
            assert f'streak: {after}' in transport.call_args.args[1].lower()
            check_record(folder, saved, 'delivered', objective=objective)
            print('R33 reuse', before, after, objective)
    with slate(root / 'refused-above-cap', finale, 'automatic'):
        with walk._fetch_stubs(58, 58, finale - timedelta(days=1)):
            code, output = walk._invoke('fetch-contest-streak', '--picks-dir', root / 'refused-above-cap',
                                        '--expected-username', walk.ACCOUNT)
        assert code and '57' in output
        reading = contest_state.load_decision_streak_state(root / 'refused-above-cap', require_contest_state=True)
        assert reading.streak == reading.best_streak == 57
        print('above-cap fetch refused 58 and retained automatic 57')
    december, january = date(2027, 12, 31), date(2028, 1, 1)
    folder = root / 'year-switch'
    with slate(folder, december, 'automatic') as (cfg, _, _):
        walk._set_clock(clock(december, 23, 59))
        _, end, _ = orchestrator.run_and_pick(cfg, str(december))
        assert end.action == 'skip' and strategy._days_remaining(str(december)) == 0
        walk._set_clock(clock(january, 0, 1))
        try:
            contest_state.load_decision_streak_state(folder, require_contest_state=True)
        except contest_state.ContestStateError as exc:
            print('midnight ET rejects old switch', exc)
        else:
            raise AssertionError('old switch accepted across ET midnight')
    with slate(folder, january, 'automatic') as (cfg, _, _):
        walk._set_clock(clock(january, 0, 1))
        _, start, _ = orchestrator.run_and_pick(cfg, str(january))
        assert (start.action, start.decision.days_effective, strategy._days_remaining(str(january))) == ('single', 180, 271)
        assert any('2028' in alert.message for alert in tail_health.season_end_alerts(today=january))
        print('new-season accepted reading', start.action, 'raw 271 effective 180')


if __name__ == '__main__':
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        check_fresh(root)
        restart(root)
        check_calendar_and_reuse(root)
