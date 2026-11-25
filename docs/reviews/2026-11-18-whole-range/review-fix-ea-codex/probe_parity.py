from collections import Counter
from collections.abc import Mapping
from contextlib import ExitStack, nullcontext, redirect_stderr
from dataclasses import asdict
from datetime import datetime, timedelta
from hashlib import sha256
from io import StringIO
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

import bts
from bts import contest_state, orchestrator, picks, scheduler, strategy
from bts.health import alert
from scripts.audit.season_walk import walk
from tests.test_pick_chosen_on_cs5a import daily_pick
from tests.test_scheduler_tail_objective import _sel, _tail_skip_decision


class BrokenClass:
    @property
    def __class__(self):
        raise RuntimeError('class lookup')


class BrokenMapping(Mapping):
    def __getitem__(self, key):
        if key == 'p_game_hit':
            return .75
        raise RuntimeError(key)

    def __len__(self):
        return 2

    def __iter__(self):
        return iter(('p_game_hit', 'batter_name'))


class BrokenFrame:
    @property
    def columns(self):
        raise RuntimeError('columns')


class BrokenIndex:
    columns = ('p_game_hit', 'batter_name')

    def __getitem__(self, key):
        raise RuntimeError(key)


mode = sys.argv[1]
work = Path(__file__).resolve().parent
repo = work.parent.parent
src = work / 'base/src' if mode == 'base' else repo / 'src'
assert Path(bts.__file__).is_relative_to(src), bts.__file__
print('Package:', bts.__file__)
DAY = '2027-07-15'
now = datetime.fromisoformat(DAY + 'T18:35:00-04:00')


def state(**values):
    return scheduler.SchedulerState(DAY, 'pinned', [], [], [], False, None, None, None, **values)


def frame(*ps, name='Top'):
    return pd.DataFrame([dict(batter_name=name, team='T', p_game_hit=p) for p in ps])


def stripped(summary):
    return None if summary is None else {k: v for k, v in summary.items()
                                        if k not in {'best_batter', 'best_team', 'best_p', 'basis', 'candidate_verified'}}


def summaries():
    candidate = dict(batter_name='Chosen', team='Y', p_game_hit=.71)
    candidates = [None, [], 2, {}, {'p_game_hit': True}, {'p_game_hit': '0.8'},
                  {'p_game_hit': float('nan')}, {'p_game_hit': float('inf')},
                  {'p_game_hit': 10**400}, BrokenClass(), BrokenMapping(), candidate]
    frames = [None, [], 'bad', BrokenFrame(), BrokenIndex(), pd.DataFrame(),
              pd.DataFrame(columns=['batter_name', 'p_game_hit']),
              frame(.8).drop(columns='batter_name'), frame(.8).drop(columns='p_game_hit'),
              frame(None), frame(pd.NA), frame('bad'), frame(float('nan')),
              frame(float('inf'), float('-inf')), frame({'p': .8}), frame([.8]),
              frame(.8, name=pd.NA), frame(.8, name=['two', 'values']),
              frame(.7, .8).set_axis([0, 0]), pd.concat([frame(.8), frame(.7)], axis=1),
              frame(.8, name=None), frame('0.8'), frame(.7, float('inf'), .81),
              frame(.8), frame(1e308, -.1)]
    values = []
    for predictions in frames:
        for decision in (None, _tail_skip_decision()):
            for contender in candidates:
                args = (predictions, 10, .79, decision)
                found = (scheduler.build_skip_summary(*args) if mode == 'base'
                         else scheduler.build_skip_summary(*args, candidate=contender))
                if predictions is frames[-2] and decision is not None and mode == 'head':
                    if contender is candidate:
                        assert found['candidate_verified'] and found['best_batter'] == 'Chosen'
                    else:
                        assert not found['candidate_verified'] and found['best_batter'] == 'Top'
                values.append(stripped(found))
    print('Prediction/candidate combinations:', len(values))
    return values


def checks(path):
    cfg = walk._config(path)
    cfg['orchestrator']['heartbeat_path'] = str(path / 'heartbeat')
    assert walk.install_state(path, now.date(), walk.State('dated', 10, 10, 'used', .75)) == []
    walk._set_clock(now)
    results = {}
    for case in ('eligible', 'removed', 'none_eligible', 'unreadable', 'no_selection', 'no_predictions', 'tail', 'pick'):
        pred = walk._predictions(now.date(), .99 if case == 'pick' else .76)
        if case == 'removed':
            pred.loc[0, 'p_game_hit'] = .999
        def statuses(_):
            if case == 'unreadable':
                raise OSError('status feed unavailable')
            return {pk: ({'abstract': 'F', 'detailed': 'Final'} if case == 'none_eligible'
                         or (pk == 70000 and case == 'removed') else
                         {'abstract': 'P', 'detailed': 'Scheduled'}) for pk in (70000, 70001, 70002)}
        with ExitStack() as stack:
            stack.enter_context(patch.object(scheduler, 'count_new_confirmations', return_value=3))
            stack.enter_context(patch.object(scheduler, 'heartbeat_watchdog', side_effect=lambda *a, **kw: nullcontext()))
            stack.enter_context(patch.object(orchestrator, 'run_cascade', return_value=(pred, 'synthetic')))
            stack.enter_context(patch.object(picks, 'get_game_statuses_detailed', side_effect=statuses))
            if case == 'tail':
                stack.enter_context(patch.object(orchestrator, 'run_and_pick', return_value=(pred, _sel(_tail_skip_decision()), 'tail')))
            if case in ('no_selection', 'no_predictions'):
                stack.enter_context(patch.object(orchestrator, 'run_and_pick',
                    return_value=(pred if case == 'no_selection' else None, None, 'missing')))
            with redirect_stderr(StringIO()):
                result = scheduler.run_single_check(DAY, [70000, 70001, 70002], set(), cfg, .03)
        if 'skip_summary' in result:
            result['skip_summary'] = stripped(result['skip_summary'])
        for key in ('selection', 'pick_result'):
            if result[key] is not None:
                result[key] = asdict(result[key])
        results[case] = result
    print('Compared complete check fields in:', ', '.join(results))
    return results


def notice_and_gate(path):
    cfg = walk._config(path) | {'bluesky': {'dm_recipient': 'capture.invalid'}}
    summary = dict(best_batter='A', best_team='T', best_p=.75, streak=10)
    outcomes = {}
    for failure in (None, OSError, KeyboardInterrupt, SystemExit):
        current = state(skip_summary=summary)
        scheduler.save_state(current, path)
        sends = []
        def send(recipient, body):
            sends.append(recipient)
            if len(sends) == 1 and failure:
                raise failure('send')
        def notify(current):
            try:
                return scheduler.maybe_notify_skip(current, summary, cfg, now_iso=now.isoformat(), send=send)
            except BaseException as exc:
                return type(exc).__name__
        first = notify(current)
        second = notify(current)
        scheduler.save_state(current, path)
        restarted = scheduler.carry_forward_skip_state(state(), scheduler.load_state(DAY, path))
        outcomes[str(failure)] = [first, second, notify(restarted), sends, restarted.skip_notified_at]
    daily = daily_pick()
    picks.save_pick(daily, path)
    walk._set_clock(now + timedelta(minutes=35))
    with patch.object(scheduler, '_alert_missed_pick') as missed:
        for flags in ({}, {'skip_summary': summary}, {'final_skip_candidate': {'reason': 'stop'}}):
            scheduler._maybe_alert_missed_pick(cfg, DAY, path, 5, None, state(**flags))
            outcomes[str(flags.keys())] = missed.call_count
            missed.reset_mock()
    walk._set_clock(now)
    return outcomes


def side_effects(root):
    info = {}
    for variant in ('sequence', 'keyboard', 'exit', 'error', 'late', 'void'):
        path = root / variant / 'picks'
        path.mkdir(parents=True)
        cfg = walk._config(path) | {'bluesky': {'dm_recipient': 'capture.invalid'}}
        synthetic = contest_state.DecisionStreakState(
            12, True, True, 'contest', 'fresh', None, None, observation_kind='automatic',
            contest_recorded_at=(now - timedelta(hours=20)).isoformat(),
        )
        record = contest_state.automatic_reading_record(synthetic, now)
        real_dispatch = alert.dispatch_dm_for_health_alerts
        calls, sent, raised = [], [], []
        fault = {'keyboard': KeyboardInterrupt, 'exit': SystemExit, 'error': OSError}.get(variant)
        def send(to, body):
            sent.append(to)
            if fault:
                raise fault('capture fault')
            return 'captured-id'
        def dispatch(alerts, to, **kwargs):
            calls.append([[(a.level, a.source, a.incident_key) for a in alerts + kwargs.get('warn_attention', [])],
                         to, kwargs['status_path'].name, str(kwargs.get('now_et_date'))])
            return real_dispatch(alerts, to, **kwargs)
        with patch.object(alert, 'dispatch_dm_for_health_alerts', side_effect=dispatch), \
                patch.object(alert, 'send_dm', side_effect=send), \
                patch.object(contest_state, 'load_decision_streak_state', return_value=synthetic):
            if variant in ('late', 'void'):
                daily = daily_pick()
                picks.save_pick(daily, path)
                if variant == 'late':
                    scheduler._refuse_late_delivery(daily, cfg, path, state(), DAY, variant, now - timedelta(minutes=1), now)
                else:
                    scheduler._refuse_void_game_delivery(daily, cfg, path, state(), DAY, variant, {101: 'Postponed'})
                assert picks.load_pick(DAY, path) is None
            else:
                def morning():
                    scheduler._morning_state_check(cfg, path, DAY, now, now + timedelta(hours=1))
                def delivery():
                    scheduler._warn_stale_contest_reading(record, cfg, path)
                for check in (morning, delivery, morning, delivery):
                    try:
                        check()
                        raised.append(None)
                    except BaseException as exc:
                        raised.append(type(exc).__name__)
        files = {str(file.relative_to(path.parent)): json.loads(file.read_text())
                 for file in path.parent.rglob('*.json')}
        info[variant] = [calls, sent, raised, files]
    return info


def bars():
    counts = Counter()
    values = []
    end = walk.LAST_GAME_DAY + timedelta(days=1)
    artifact = strategy._load_mdp()
    for days in range(182):
        day = (end - timedelta(days=days)).isoformat()
        for streak in range(57):
            regime, _ = strategy._regime(artifact, streak, day)
            states = [(streak, 'trusted')]
            if regime == strategy.OBJECTIVE_TAIL:
                states.extend([(56, 'trusted'), (56, 'untrusted')])
            for saver in (False, True):
                for best, trust in states:
                    counts[regime] += 1
                    values.append(strategy.effective_pick_bar(streak, day, saver, artifact, best, trust))
    print('Policy lookups:', dict(counts))
    return dict(counts), sha256(json.dumps(values).encode()).hexdigest()


with TemporaryDirectory(dir=work) as temp, walk.offline(now), redirect_stderr(StringIO()) as log:
    temp = Path(temp)
    evidence = dict(summaries=summaries(), checks=checks(temp / 'checks'),
                    notice_and_gate=notice_and_gate(temp / 'notice'), side_effects=side_effects(temp), bars=bars())
(work / f'parity-{mode}.json').write_text(json.dumps(evidence, default=str, sort_keys=True, indent=1) + '\n')
(work / f'parity-{mode}.stderr').write_text(log.getvalue())
