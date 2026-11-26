"""Pin the existing archive, delivery, state and dispatcher action boundaries."""

from dataclasses import replace
from pathlib import Path
import sys

from bts import dm
from bts.health import alert

from replay_main import DAY, daily_decision, offline, picks, scheduler, seed


def new_state(w):
    return scheduler.SchedulerState(
        date=DAY, schedule_fetched_at=w.now.isoformat(), games=[],
        confirmed_game_pks=[], runs_completed=[], pick_locked=False,
        pick_locked_at=None, result_status=None, next_wakeup=None,
    )


def marker():
    for case in ('slow-marker', 'unconfirmed-id'):
        with offline.offline(offline.instant(DAY, 18, 20)) as (w, mp):
            daily = seed(w)
            w.now = offline.instant(DAY, 19, 4, 59)
            state = new_state(w)
            if case == 'slow-marker':
                save = picks.save_pick

                def slow_save(value, directory):
                    if value.delivery_attempted and not picks.pick_was_delivered(value):
                        w.advance(2)
                    return save(value, directory)

                mp.setattr(picks, 'save_pick', slow_save)
            else:
                mp.setattr(dm, 'send_dm', lambda *args: None)
            assert scheduler._deliver_and_lock_pick(daily, w.config, w.picks, state, DAY, 'fallback')
            committed = daily_decision.load_decision(DAY, w.picks)
            assert committed['scoreable'] and committed['delivery_status'] == 'delivered'
            if case == 'slow-marker':
                sent, = [when for event, when, _ in w.events if event == 'dm']
                assert sent == offline.instant(DAY, 19, 5, 1) > picks.submission_cutoff_et(daily)
                print('slow marker: transport started', sent.isoformat(), 'cutoff 19:05:00')
            else:
                assert state.pick_locked and not picks.pick_was_delivered(picks.load_pick(DAY, w.picks))
                print('missing DM ID: delivered scoreable record, pick is not durably delivered')


def archive():
    with offline.offline(offline.instant(DAY, 18, 20)) as (w, mp):
        daily = seed(w)
        w.now = offline.instant(DAY, 18, 42, 59)
        cutoff = picks.submission_cutoff_et(daily)

        def plan():
            return scheduler.plan_fallback_action(
                now=w.now, cutoff=cutoff, should_post=False, should_post_ungated=False,
                block_reason='gap', contender_game_pk=102,
                remaining_runs=[{'time_et': offline.instant(DAY, 18, 40), 'game_pks': [102]}],
                confirmed_sides=set(), budget_min=12, operator_reserve_min=10,
            )

        initial = plan()
        assert initial.action == 'defer'
        write = Path.write_text
        replacement = replace(daily, delivery_attempted=True)

        def slow_archive(path, *args, **kwargs):
            if path.name.startswith('deferred_fallback_'):
                w.advance(2)
                picks.save_pick(replacement, w.picks)
            return write(path, *args, **kwargs)

        mp.setattr(Path, 'write_text', slow_archive)
        saved = scheduler._defer_pick_at_fallback(w.picks, DAY, daily, initial.reason)
        assert plan().action == 'deliver'
        assert saved.exists() and picks.load_pick(DAY, w.picks) is None
        print('archive: write crossed 18:43; original defer completed and removed newer attempt marker')


def receipt():
    with offline.offline(offline.instant(DAY, 19, 0)) as (w, mp):
        state = new_state(w)
        entry = {'started': w.now.isoformat(), 'identity': [DAY, 601, 101, None, None, '19:10'],
                 'trigger': 'saved_candidate', 'alert_at': w.now.isoformat(), 'archive': None,
                 'duration_sec': None, 'reused_check_cascade': False, 'action': 'contest_state'}
        state.fallback_refreshes = [entry]
        scheduler.save_state(state, w.picks)
        after_rollback = scheduler.load_state(DAY, w.picks)
        assert after_rollback.fallback_refreshes == [entry]
        started_again = scheduler.carry_forward_skip_state(new_state(w), after_rollback)
        assert started_again.fallback_refreshes == [entry]
        assert scheduler._measured_cascade_durations(started_again) == []
        print('old main round-trips nested entry, carries it forward, skips duration=None')

        calls = []

        def send(_recipient, body):
            if 'NO PICK DELIVERED' in body:
                calls.append('missed_pick')
                if calls.count('missed_pick') == 1:
                    raise OSError('synthetic failed alert transport')
            else:
                calls.append('contest_state')
            return 'captured-health-dm'

        mp.setattr(alert, 'send_dm', send)
        status = w.health / 'health_dm_delivery_status.json'
        alert_entry = [alert.Alert('CRITICAL', 'contest_state', 'synthetic contest alert')]
        alert.dispatch_dm_for_health_alerts(alert_entry, 'offline.invalid', status_path=status)
        scheduler._alert_missed_pick(w.config, None, 10)
        alert.dispatch_dm_for_health_alerts(alert_entry, 'offline.invalid', status_path=status)
        scheduler._alert_missed_pick(w.config, None, 10)
        scheduler._alert_missed_pick(w.config, None, 10)
        assert calls == ['contest_state', 'missed_pick', 'contest_state', 'missed_pick'], calls
        w.advance(5 * 3600)
        scheduler._alert_missed_pick(w.config, None, -290)
        assert calls.count('missed_pick') == 3
        print('shared dispatcher: failed missed_pick repeats successful contest_state; same slate repeats at midnight')


if __name__ == '__main__':
    {'marker': marker, 'archive': archive, 'receipt': receipt}[sys.argv[1]]()
