from contextlib import ExitStack, nullcontext
from datetime import date, timedelta
from unittest.mock import Mock, patch

from click.testing import CliRunner

from bts import cli, contest_fetch, contest_state, orchestrator, picks, scheduler
from scripts.audit.season_walk import walk


DAY = date(2027, 9, 26)
ABSENT = object()


def at(day=DAY, hour=12, minute=0):
    return walk.REAL_DATETIME(day.year, day.month, day.day, hour, minute, tzinfo=walk.ET)


def invoke(*args, standalone=True):
    result = CliRunner().invoke(cli.cli, [str(arg) for arg in args], standalone_mode=standalone)
    error = result.exception
    return result.exit_code, result.output.strip(), type(error).__name__, str(error)


def fetch(directory, payload, *, day=DAY, dry=False, recipient=True, coverage=True):
    opts = ['--dry-run'] if dry else []
    if recipient:
        opts += ['--dm-recipient', 'captured']
    profile = dict(payload, predictions=[{'roundId': 1, 'result': 'hit'}] if coverage else [])
    with walk._fetch_stubs(0, 0, day - timedelta(days=1)), patch.object(
        contest_fetch, 'fetch_profile', return_value=profile
    ):
        return invoke('fetch-contest-streak', '--picks-dir', directory,
                      '--expected-username', walk.ACCOUNT, *opts)


def counts(active, best):
    return {'activeStreak': active, 'seasonBestStreak': best}


def install(directory, kind, active, best, *, day=DAY, saver='used'):
    assert walk.install_state(directory, day, walk.State(kind, active, best, saver, .96)) == []


def live(directory):
    return contest_state.load_decision_streak_state(directory, require_contest_state=True)


def fixture(stack: ExitStack, directory, day=DAY, p=.96, dm=None):
    frame = walk._predictions(day, p).head(2)
    games = list(frame['game_pk'])
    statuses = {game: {'abstract': 'P', 'detailed': 'Scheduled'} for game in games}
    config = walk._config(directory)
    config['orchestrator']['heartbeat_path'] = str(directory / '.heartbeat')
    config['bluesky'] = {'dm_recipient': 'captured'}
    dm = Mock(return_value='captured-id') if dm is None else dm
    stack.enter_context(patch.object(orchestrator, 'run_cascade', return_value=(frame, 'synthetic')))
    stack.enter_context(patch.object(picks, 'get_game_statuses_detailed', return_value=statuses))
    stack.enter_context(patch.object(scheduler, 'heartbeat_watchdog', side_effect=lambda *a, **k: nullcontext()))
    stack.enter_context(patch.object(scheduler, 'count_new_confirmations', return_value=6))
    stack.enter_context(patch.object(scheduler, '_trigger_live_forward_capture_on_lock'))
    stack.enter_context(patch.object(scheduler, 'fetch_schedule', return_value=[{'status': {'detailedState': 'Scheduled'}}]))
    stack.enter_context(patch('bts.dm.send_dm', dm))
    return config, games, frame, dm


def fresh(day, config, games):
    return scheduler.run_single_check(str(day), games, set(), config, .01)['selection']


def scheduler_state(day=DAY):
    return scheduler.SchedulerState(str(day), None, [], [], [], False, None, None, None)


def deliver(directory, config, daily, selection, day=DAY):
    return scheduler._deliver_and_lock_pick(daily, config, directory, scheduler_state(day),
                                            str(day), 'review probe', selection=selection)


def changed(body, field, value):
    if value is ABSENT:
        body.pop(field, None)
    else:
        body[field] = value
    return body
