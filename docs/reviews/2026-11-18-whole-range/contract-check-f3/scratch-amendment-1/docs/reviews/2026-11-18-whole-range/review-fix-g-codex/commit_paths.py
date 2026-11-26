"""Reproduce commit transitions using the real scheduler and captured transports."""
from contextlib import ExitStack, nullcontext
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch
import json
import sys

import bts
from bts import daily_decision, dm, orchestrator, picks, posting, scheduler, strategy
from scripts.audit.season_walk import walk

DAY = '2027-03-24'
NOW = walk.REAL_DATETIME(2027, 3, 24, 18, 32, tzinfo=walk.ET)
GUARD = {'rule': 'monotone_skip_v1', 'bin': 4, 'from_bin': 3, 'table_action': 'skip'}


def state(**kwargs):
    return scheduler.SchedulerState(date=DAY, schedule_fetched_at=NOW.isoformat(), games=[],
                                    confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                                    pick_locked_at=None, result_status=None, next_wakeup=None, **kwargs)


def data(p=.83, start_id=100, projected=False):
    frame = walk._predictions(date.fromisoformat(DAY), p)
    frame['batter_id'] += start_id - 100
    frame['batter_name'] = [f'Fixture batter {id_}' for id_ in frame['batter_id']]
    if projected:
        frame['flags'] = 'PROJECTED'
    return frame


def setup(stack, root, frame, mode='dm'):
    stack.enter_context(walk.offline(NOW))
    assert not walk.install_state(root, date.fromisoformat(DAY),
                                  walk.State('dated', 50, 50, 'used', .83))
    walk._set_clock(NOW)
    cfg = walk._config(root)
    cfg['orchestrator']['heartbeat_path'] = str(root / 'heartbeat')
    cfg['scheduler'].update(pick_delivery=mode, live_forward_capture_on_lock=False)
    cfg['bluesky'] = {'dm_recipient': 'captured-transport'}
    cfg['health_checks']['enabled'] = False
    statuses = {pk: {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame.game_pk}
    stack.enter_context(patch.object(picks, 'get_game_statuses_detailed', lambda *_: statuses))
    stack.enter_context(patch.object(orchestrator, 'run_cascade', lambda *_: (frame, 'probe')))
    stack.enter_context(patch.object(scheduler, '_now_et', lambda: NOW))
    stack.enter_context(patch.object(scheduler, 'heartbeat_watchdog', lambda *a, **k: nullcontext()))
    stack.enter_context(patch.object(scheduler, '_delivery_game_statuses', lambda *_: statuses))
    for name in ('_trigger_live_forward_capture_on_lock', '_alert_delivery_outcome_unknown',
                 '_idle_until_next_wakeup', '_watchdog_ping_sleep', '_morning_state_check'):
        stack.enter_context(patch.object(scheduler, name, Mock()))
    stack.enter_context(patch.object(scheduler, 'run_result_polling', Mock(return_value='hit')))
    stack.enter_context(patch.object(scheduler, 'count_new_confirmations', lambda *_: 0))
    games = [{'gamePk': pk, 'gameDate': gt, 'doubleHeader': 'N', 'status': {'detailedState': 'Scheduled'}}
             for pk, gt in zip(frame.game_pk, frame.game_time)]
    stack.enter_context(patch.object(scheduler, 'fetch_schedule', lambda *_: games))
    return cfg, statuses


def choose(cfg):
    _, sel, _ = orchestrator.run_and_pick(cfg, DAY)
    assert sel.action == 'single' and not sel.pick_result.locked, sel
    return sel.pick_result.daily, sel


def record(root):
    path = daily_decision.decision_path(DAY, root)
    raw = path.read_bytes()
    rec = daily_decision.parse_decision(raw.decode())
    assert rec and rec['scoreable']
    return rec, raw


def deliver(cfg, root, daily, sel=None):
    return scheduler._deliver_and_lock_pick(daily, cfg, root, state(), DAY, 'review', selection=sel)


def summary(rec):
    return (rec['action'], (rec['primary'] or {}).get('batter_id'), rec['delivery_status'],
            rec.get('policy_guard'))


def lifecycle(side):
    for scenario in ('confirmed_after_recovery', 'private_to_dm', 'refused_then_repicked',
                     'prior_process_missing_pick', 'earlier_same_date_run'):
        with TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            frame = data()
            cfg, _ = setup(stack, root, frame)
            a, selection_a = choose(cfg)
            send = stack.enter_context(patch.object(dm, 'send_dm', Mock(return_value='dm-ok')))
            if scenario == 'confirmed_after_recovery':
                def finish_send(*_):
                    pending = picks.load_pick(DAY, root)
                    assert pending.delivery_attempted and not picks.pick_was_delivered(pending)
                    assert deliver(cfg, root, pending) is False
                    assert record(root)[0]['delivery_status'] == 'locked_unconfirmed'
                    before.append(record(root)[1])
                    return 'confirmed-dm'
                before = []
                send.side_effect = finish_send
                assert deliver(cfg, root, a, selection_a)
                old = before[0]
                expected = ('single', 100, 'delivered')
            else:
                cfg['scheduler']['pick_delivery'] = 'private'
                assert deliver(cfg, root, a, selection_a)
                old = record(root)[1]
                cfg['scheduler']['pick_delivery'] = 'dm'
                if scenario == 'private_to_dm':
                    assert deliver(cfg, root, picks.load_pick(DAY, root))
                    expected = ('single', 100, 'delivered')
                else:
                    if scenario == 'refused_then_repicked':
                        stale = picks.load_pick(DAY, root)
                        stale.pick.game_time = (NOW - timedelta(minutes=10)).isoformat()
                        picks.save_pick(stale, root)
                        from bts.health import alert
                        with patch.object(alert, 'dispatch_dm_for_health_alerts'):
                            assert deliver(cfg, root, stale) is False
                        assert list((root / DAY).glob('refused_delivery_*.json'))
                    else:
                        pick_path = root / f'{DAY}.json'
                        pick_path.rename(root / 'archived-by-earlier-run.json')
                    assert picks.load_pick(DAY, root) is None
                    frame['batter_id'] += 100
                    frame['batter_name'] = ['Fixture batter 200', 'Fixture batter 201', 'Fixture batter 202']
                    if scenario == 'earlier_same_date_run':
                        (root / DAY / 'scheduler_state.json').unlink()
                    scheduler.run_day(DAY, cfg)
                    expected = ('single', 200, 'delivered')
            after, raw = record(root)
            loaded = picks.load_pick(DAY, root)
            assert loaded.pick.batter_id == expected[1] and picks.pick_was_delivered(loaded)
            send.assert_called_once()
            assert f'Fixture batter {expected[1]}' in send.call_args.args[1]
            if side == 'base':
                assert summary(after)[:3] == expected and raw != old
            else:
                assert raw == old and summary(after)[:3] != expected
            print(f'{side} {scenario}: committed={expected}, recorded={summary(after)}, same_bytes={raw == old}')


def fresh_and_cached():
    for scenario in ('dm', 'public', 'private', 'cached_failed', 'cached_warmup'):
        with TemporaryDirectory() as d, ExitStack() as stack:
            root = Path(d)
            cfg, statuses = setup(stack, root, data(.96), mode='dm' if scenario.startswith('cached') else scenario)
            daily, sel = choose(cfg)
            assert sel.decision.guard == GUARD and daily.policy_decision['guard'] == GUARD
            saved = picks.save_pick(daily, root)
            assert json.loads(saved.read_bytes())['policy_decision']['guard'] == GUARD
            sent = Mock(return_value='fixture-transport-id')
            stack.enter_context(patch.object(dm, 'send_dm', sent))
            stack.enter_context(patch.object(posting, 'post_to_bluesky', sent))
            if scenario == 'cached_failed':
                stack.enter_context(patch.object(orchestrator, 'run_cascade', Mock(side_effect=RuntimeError('fixture cascade failure'))))
            elif scenario == 'cached_warmup':
                for pk in statuses:
                    statuses[pk] = {'abstract': 'P', 'detailed': 'Warmup'}
            if scenario.startswith('cached'):
                result = scheduler._refresh_pick_at_fallback_decision(cfg, DAY, picks.load_pick(DAY, root), .03)
                daily, sel = result.daily, result.selection
                if scenario == 'cached_warmup':
                    assert sel is not None and sel.decision is None and sel.primary_candidate is None
                else:
                    assert sel is None
            assert deliver(cfg, root, daily, sel)
            rec, _ = record(root)
            assert json.loads(json.dumps(rec))['policy_guard'] == GUARD
            expected_status = 'private_locked' if scenario == 'private' else 'delivered'
            expected_id = None if scenario == 'cached_warmup' else 100
            assert summary(rec) == ('single', expected_id, expected_status, GUARD)
            assert sent.call_count == int(scenario != 'private')
            print(scenario, 'saved/current ids', daily.pick.batter_id, expected_id, 'record:', summary(rec))


def corrupt_record(side):
    with TemporaryDirectory() as d, ExitStack() as stack:
        root = Path(d)
        cfg, _ = setup(stack, root, data())
        daily, sel = choose(cfg)
        path = daily_decision.decision_path(DAY, root)
        path.parent.mkdir(exist_ok=True)
        path.write_bytes(b'\xff')
        stack.enter_context(patch.object(dm, 'send_dm', Mock(return_value='confirmed-dm')))
        current = state()
        assert scheduler._deliver_and_lock_pick(daily, cfg, root, current, DAY, 'corrupt', selection=sel)
        assert picks.pick_was_delivered(picks.load_pick(DAY, root))
        if side == 'base':
            assert current.committed_pick_written and record(root)[0]['delivery_status'] == 'delivered'
        else:
            assert not current.committed_pick_written and path.read_bytes() == b'\xff'
            try:
                daily_decision.is_scoreable_commit(DAY, root, daily)
            except UnicodeDecodeError:
                pass
            else:
                raise AssertionError('expected unreadable decision')
        print(side, 'non-UTF-8 existing record: delivered, committed_pick_written=', current.committed_pick_written)


if __name__ == '__main__':
    side = sys.argv[1]
    print(side, bts.__file__)
    lifecycle(side)
    corrupt_record(side)
    if side == 'head':
        fresh_and_cached()
