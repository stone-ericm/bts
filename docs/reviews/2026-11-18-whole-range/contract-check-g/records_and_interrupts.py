from contextlib import nullcontext
from datetime import date
import hashlib
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
from tempfile import TemporaryDirectory
from types import ModuleType
from unittest.mock import patch

from bts import daily_decision, day_outcome, dm, picks, scheduler, skip_policy_shadow, strategy
from bts.health import tail_policy as tail_health
from bts.simulate import mdp, tail_policy
from scripts.audit.boundary_shadow_census import CensusHalt, attribution
from scripts.audit.season_ledger.sources import day_records
from scripts.audit.season_walk import walk


DAY = '2027-03-24'
NOW = walk.REAL_DATETIME(2027, 3, 24, 16, tzinfo=walk.ET)
GUARD = dict(rule='monotone_skip_v1', bin=4, from_bin=3, table_action='skip')


class Interrupt(BaseException):
    pass


def load_old_picks():
    source = subprocess.run(['git', 'show', '619abda7:src/bts/picks.py'], check=True,
                            capture_output=True, timeout=2).stdout
    module = ModuleType('base_619abda7_picks')
    sys.modules[module.__name__] = module
    exec(compile(source, '619abda7:src/bts/picks.py', 'exec'), vars(module))
    return module


def state(**fields):
    values = dict(date=DAY, schedule_fetched_at=NOW.isoformat(), games=[], confirmed_game_pks=[],
                  runs_completed=[], pick_locked=False, pick_locked_at=None,
                  result_status=None, next_wakeup=None)
    return scheduler.SchedulerState(**(values | fields))


def fresh_pick(root):
    # Use a donor-bin play from the real base, then use its pick to test readers.
    predictions = walk._predictions(date.fromisoformat(DAY), 0.83)
    statuses = {int(pk): {'abstract': 'P', 'detailed': 'Scheduled'} for pk in predictions.game_pk}
    selection = strategy.select_pick(predictions, DAY, root, streak=50, saver_available=False,
                                     game_statuses_detailed=statuses, require_detailed_statuses=True)
    assert selection.action == 'single'
    return selection.pick_result.daily, selection


def readers(root):
    daily, selection = fresh_pick(root)
    assert 'guard' not in daily.policy_decision
    daily.pick.p_game_hit = 0.96
    daily.policy_decision['guard'] = GUARD
    path = picks.save_pick(daily, root)
    old = load_old_picks()
    assert old.load_pick(DAY, root).policy_decision['guard'] == GUARD
    assert picks.load_pick(DAY, root).policy_decision['guard'] == GUARD
    picks.save_shadow_pick(daily, root)
    assert picks.load_shadow_pick(DAY, root).policy_decision['guard'] == GUARD

    scheduler._commit_decision_for_pick(root, DAY, daily, selection=None,
                                        delivery_status='delivered', state=state())
    decision = daily_decision.load_decision(DAY, root)
    assert decision.get('policy_guard') is None and decision['source'] == 'mdp'
    decision['policy_guard'] = GUARD
    raw = json.dumps(decision).encode()
    file = daily_decision.decision_path(DAY, root)
    file.write_bytes(raw)
    decoded, seen = daily_decision.load_decision_bytes(DAY, root)
    assert decoded == decision and seen == raw
    assert daily_decision.scoreable_commit_from(decoded, daily)
    assert not daily_decision.is_reach57_mdp_skip(decoded)
    assert skip_policy_shadow.record_skip_from_decision(DAY, root, now=NOW) is None
    ledger = day_records.parse_decision(f'picks/{DAY}/decision.json', raw)
    assert ledger.quarantined == []
    assert json.loads(ledger.rows[0]['record_raw_json'])['policy_guard'] == GUARD
    assert 'policy_guard' not in ledger.rows[0]
    bound = hashlib.sha256(raw).hexdigest()
    receipt = {'outcome': 'observed', 'et_date': DAY, 'season': 2027,
               'selection': {'commit': {'decision_sha256': bound}},
               'account': {'username': 'probe'}}
    assert day_outcome._binds(receipt, date=DAY, decision_sha256=bound, expected_username='probe')
    prior = dict(decision)
    del prior['policy_guard']
    assert not day_outcome._binds(receipt, date=DAY,
                                  decision_sha256=hashlib.sha256(json.dumps(prior).encode()).hexdigest(),
                                  expected_username='probe')
    try:
        attribution('skip', decoded['action'])
    except CensusHalt as exc:
        print('census mismatch:', exc)
    else:
        raise AssertionError('census should halt on raw skip vs play')
    body = json.loads(path.read_text())
    body['policy_decision']['guard']['bin'] = strategy.np.int64(4)
    try:
        json.dumps(body)
    except TypeError as exc:
        print('non-native guard integer:', exc)
    else:
        raise AssertionError('NumPy int unexpectedly encodes')
    print('base/rollback readers, shadow filter, ledger decoding and receipt byte binding verified')


def artifacts(root):
    base = root / 'base.npz'
    tail = Path('data/models/mdp_tail_policy.npz').resolve()
    base.write_text('not a zip')
    with patch.object(mdp, 'DEFAULT_POLICY_PATH', base), patch.object(strategy, '_mdp_cache', None):
        artifact = strategy._load_mdp()
        assert artifact['tail'] is None and 'unverifiable' in artifact['tail_error']
        for day, want in (('2027-09-20', ('emax_season_best', 'single')),
                          ('2027-09-26', ('emax_season_best', 'skip')),
                          (DAY, ('reach57', 'single'))):
            ctx = strategy.DecisionContext(0.96, None, False, day, True, artifact, 18, 'trusted')
            decision = strategy.resolve_policy_decision(ctx, 0, False)
            assert (decision.objective, decision.action) == want
    directory = root / 'directory'
    directory.mkdir()
    with patch.object(mdp, 'DEFAULT_POLICY_PATH', directory), patch.object(strategy, '_mdp_cache', None):
        assert 'IsADirectoryError' in strategy._load_mdp()['base_error']

    fifo = root / 'fifo'
    os.mkfifo(fifo)

    def read_deadline(*_):
        raise Interrupt('FIFO reader deadline')

    def expect_block(label, read):
        signal.setitimer(signal.ITIMER_REAL, 0.15)
        try:
            read()
        except Interrupt:
            print('waits at unwritten FIFO:', label)
        else:
            raise AssertionError(f'{label} returned instead of blocking')
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)

    original = signal.signal(signal.SIGALRM, read_deadline)
    try:
        with patch.object(mdp, 'DEFAULT_POLICY_PATH', fifo), patch.object(strategy, '_mdp_cache', None):
            expect_block('strategy base loader', strategy._load_mdp)
        expect_block('tail loader', lambda: tail_policy.load_tail_policy(fifo))
        expect_block('tail health base hash', lambda: tail_health.check(base_path=fifo, tail_path=tail))
    finally:
        signal.signal(signal.SIGALRM, original)


def delivery(root):
    config = {'orchestrator': {'picks_dir': str(root)}, 'bluesky': {'dm_recipient': 'probe'},
              'scheduler': {'pick_delivery': 'dm'}}
    daily, _ = fresh_pick(root)
    file = daily_decision.decision_path(DAY, root)
    path = picks.save_pick(daily, root)
    before = path.read_bytes()
    with patch('bts.util.os.replace', side_effect=Interrupt):
        try:
            picks.save_pick(daily, root)
        except Interrupt:
            pass
        else:
            raise AssertionError('expected interruption before atomic replace')
    assert path.read_bytes() == before and not file.exists()
    with patch.object(scheduler, '_now_et', return_value=NOW), \
            patch.object(scheduler, '_void_committed_games', return_value={}), \
            patch.object(scheduler, '_trigger_live_forward_capture_on_lock'), \
            patch.object(scheduler, '_alert_delivery_outcome_unknown'):
        with patch.object(dm, 'send_dm', side_effect=RuntimeError('probe failed send')):
            assert not scheduler._deliver_and_lock_pick(daily, config, root, state(), DAY, 'probe')
        daily = picks.load_pick(DAY, root)
        assert not daily.delivery_attempted and not file.exists()
        with patch.object(dm, 'send_dm', side_effect=Interrupt):
            try:
                scheduler._deliver_and_lock_pick(daily, config, root, state(), DAY, 'probe')
            except Interrupt:
                pass
            else:
                raise AssertionError('expected interruption after attempt marker')
        daily = picks.load_pick(DAY, root)
        assert daily.delivery_attempted and not file.exists()
        with patch.object(dm, 'send_dm') as transport:
            assert not scheduler._deliver_and_lock_pick(daily, config, root, state(), DAY, 'probe')
            transport.assert_not_called()
        assert daily_decision.load_decision(DAY, root)['delivery_status'] == 'locked_unconfirmed'
        file.unlink()
        scheduler._write_classification_decision(root, DAY, action='single', delivered=False,
                                                 attempted=True, primary=scheduler._row_from_daily(daily.pick),
                                                 double_down=None, state=state(), daily=daily)
        before = file.read_bytes()
        assert daily_decision.load_decision(DAY, root)['delivery_status'] == 'locked_unconfirmed'
        scheduler._write_endofday_skip(root, DAY, state(final_skip_candidate={'source': 'mdp'}))
        assert file.read_bytes() == before

    # Use a real cached fallback refresh with no new predictions and reload the pick it returns.
    file.unlink()
    daily.delivery_attempted = False
    picks.save_pick(daily, root)
    config['scheduler']['pick_delivery'] = 'private'
    with patch.object(scheduler, 'run_and_pick', return_value=(None, None, None)), \
            patch.object(scheduler, 'heartbeat_watchdog', return_value=nullcontext()), \
            patch.object(scheduler, '_trigger_live_forward_capture_on_lock'), \
            patch.object(scheduler, '_now_et', return_value=NOW), \
            patch.object(scheduler, '_void_committed_games', return_value={}):
        refresh = scheduler._refresh_pick_at_fallback_decision(config, DAY, picks.load_pick(DAY, root), 0.03)
        assert refresh.selection is None and refresh.daily.policy_decision == daily.policy_decision
        assert scheduler._deliver_and_lock_pick(refresh.daily, config, root, state(), DAY, 'cached probe')
    assert daily_decision.load_decision(DAY, root)['delivery_status'] == 'private_locked'
    print('atomic pick interruption, caught send, unconfirmed retry, classification, skip overwrite and cached delivery verified')


if __name__ == '__main__':
    with TemporaryDirectory(dir='.codex-review/wr-check-g') as name:
        folder = Path(name)
        with walk.offline(NOW):
            readers(folder / 'readers')
            (folder / 'artifacts').mkdir()
            artifacts(folder / 'artifacts')
            delivery(folder / 'delivery')
