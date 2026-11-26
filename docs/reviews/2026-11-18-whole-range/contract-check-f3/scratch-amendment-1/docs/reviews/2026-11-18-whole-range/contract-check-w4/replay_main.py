"""Replay the W4 defect and compare real scheduler actions with W4 inputs."""

from datetime import timedelta
from pathlib import Path
import sys

from bts import contest_state, daily_decision, orchestrator, picks, scheduler


SCRATCH = Path(__file__).resolve().parent
REPO = SCRATCH.parents[1]
sys.path.insert(0, str(REPO / 'docs/reviews/2026-11-18-whole-range/slice-3-codex'))
import offline

offline.ROOT = REPO
offline.HERE = SCRATCH
DAY = '2027-03-24'


def seed(w):
    w.dated(30)
    _, selection, _ = orchestrator.run_and_pick(w.config, DAY)
    daily = selection.pick_result.daily
    picks.save_pick(daily, w.picks)
    live = contest_state.load_decision_streak_state(w.picks, require_contest_state=True)
    assert picks.saved_pick_refusal(daily, live) is None
    w.events.clear()
    return daily


def predictions_at(w):
    return [when.strftime('%H:%M') for kind, when, _ in w.events if kind == 'predict']


def sends_at(w, kind):
    return [when.strftime('%H:%M') for event, when, _ in w.events if event == kind]


def reproduce():
    for case in ('two-game', 'one-game', 'restart'):
        with offline.offline(offline.instant(DAY, 18, 20)) as (w, _):
            if case == 'one-game':
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.predictions = lambda: w.frame((0.99,))
            daily = seed(w)
            assert picks.submission_cutoff_et(daily) == offline.instant(DAY, 19, 5)
            if case == 'restart':
                w.config['scheduler']['lineup_check_offset_min'] = 20
                old_state = scheduler.SchedulerState(
                    date=DAY, schedule_fetched_at=w.now.isoformat(), games=[],
                    confirmed_game_pks=[], runs_completed=[{'time': w.now.isoformat()}],
                    pick_locked=False, pick_locked_at=None, result_status=None, next_wakeup=None,
                )
                scheduler.save_state(old_state, w.picks)
                remaining = scheduler.plan_run_times(w.schedule[DAY], offset_min=20, now=w.now)
                assert [run['time_et'].strftime('%H:%M') for run in remaining] == ['18:50', '19:50']
            w.predictions = lambda: None
            scheduler.run_day(DAY, w.config)
            archived = list((w.picks / DAY).glob('stale_pick_*.json'))
            decision = daily_decision.load_decision(DAY, w.picks)
            assert not w.sends('health-dm')
            if case == 'one-game':
                assert predictions_at(w) == ['18:25', '18:35']
                assert sends_at(w, 'dm') == ['18:35'] and decision['scoreable'] and not archived
            else:
                assert predictions_at(w) == (['18:25', '19:25'] if case == 'two-game'
                                             else ['18:50', '19:50'])
                assert not w.sends('dm') and decision is None and len(archived) == 1
                assert picks.load_pick(DAY, w.picks) is None
            print(case, predictions_at(w), 'pick DMs', sends_at(w, 'dm'),
                  'health DMs', sends_at(w, 'health-dm'), 'archives', len(archived),
                  'commit', bool(decision))


def early_delivery():
    for case in ('morning-preview', 'better-later'):
        for insert_refresh in (False, True):
            with offline.offline(offline.instant(DAY, 3 if case == 'morning-preview' else 18,
                                                 0 if case == 'morning-preview' else 20)) as (w, _):
                if case == 'morning-preview':
                    w.schedule[DAY].append(offline.game(103, offline.instant(DAY, 21, 10)))
                    w.config['scheduler']['cascade_budget_min'] = 40
                    w.dated(30)
                    w.predictions = lambda: w.frame((0.99, 0.81, 0.60), (True, True, True))
                    w.call('preview', '--date', DAY, '--picks-dir', str(w.picks),
                           '--models-dir', str(w.models))
                    daily = picks.load_pick(DAY, w.picks)
                    live = contest_state.load_decision_streak_state(w.picks, require_contest_state=True)
                    assert picks.saved_pick_refusal(daily, live)
                    w.events.clear()
                    next_check = offline.instant(DAY, 18, 25)
                    deadline = offline.instant(DAY, 18, 15)
                else:
                    w.schedule[DAY][1] = offline.game(102, offline.instant(DAY, 19, 30))
                    daily = seed(w)
                    next_check = offline.instant(DAY, 18, 45)
                    deadline = offline.instant(DAY, 18, 35)
                assert deadline < next_check < picks.submission_cutoff_et(daily)
                if case == 'morning-preview':
                    # 101 is selectable but projected at D. After the scheduled
                    # check we expose a stronger, confirmed 102.
                    w.predictions = lambda: (w.frame((0.99, 0.81, 0.60), (True, False, False))
                                             if w.now < next_check else w.frame((0.60, 0.99, 0.60),
                                                                                 (True, False, True)))
                else:
                    w.predictions = lambda: (None if w.now < next_check else
                                             w.frame((0.60, 0.99), (True, False)))
                if insert_refresh:
                    # Exercise the real refresh/planner/chokepoint at the proposed
                    # W4 time. This is an action replay, not a W4 implementation.
                    w.now = deadline
                    runs = scheduler.plan_run_times(w.schedule[DAY], now=w.now)
                    remaining = [run for run in runs if run['time_et'] >= next_check]
                    state = scheduler.SchedulerState(
                        date=DAY, schedule_fetched_at=w.now.isoformat(), games=[],
                        confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                        pick_locked_at=None, result_status=None, next_wakeup=None,
                    )
                    confirmed_sides = set()
                    sync = sorted({pk for run in remaining for pk in run['game_pks']})
                    scheduler.count_new_confirmations(sync, confirmed_sides)
                    refresh = scheduler._refresh_pick_at_fallback_decision(w.config, DAY, daily, 0.03)
                    scheduler.count_new_confirmations(sync, confirmed_sides)
                    plan = scheduler.plan_fallback_action(
                        now=w.now, cutoff=picks.submission_cutoff_et(refresh.daily),
                        should_post=refresh.should_post, should_post_ungated=refresh.should_post_ungated,
                        block_reason=refresh.block_reason, contender_game_pk=refresh.contender_game_pk,
                        contender_game_pks=refresh.contender_game_pks, remaining_runs=remaining,
                        confirmed_sides=confirmed_sides,
                        budget_min=w.config['scheduler'].get('cascade_budget_min', 12),
                        operator_reserve_min=10,
                    )
                    assert plan.action == 'deliver', plan
                    assert scheduler._deliver_and_lock_pick(
                        refresh.daily, w.config, w.picks, state, DAY, 'fallback',
                        selection=refresh.selection,
                    )
                else:
                    scheduler.run_day(DAY, w.config)
                decision = daily_decision.load_decision(DAY, w.picks)
                target = 101 if insert_refresh else 102
                assert decision['primary']['game_pk'] == target, decision
                assert sends_at(w, 'dm') == [(deadline if insert_refresh else next_check).strftime('%H:%M')]
                lead = scheduler._fallback_min_with_floor(
                    35, w.config['scheduler'].get('cascade_budget_min', 12), 10,
                )
                assert picks.earliest_pick_game_et(daily) - timedelta(minutes=lead) == deadline
                print(case, 'W4 action replay' if insert_refresh else 'main', 'game', target,
                      'pick DMs', sends_at(w, 'dm'), 'predictions', predictions_at(w))


if __name__ == '__main__':
    {'reproduce': reproduce, 'early-delivery': early_delivery}[sys.argv[1]]()
