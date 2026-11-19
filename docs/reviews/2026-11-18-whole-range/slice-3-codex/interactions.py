"""Exercise scheduler/CLI interactions with real selection, persistence and alert logic."""

import datetime as calendar
import json
from unittest.mock import patch

from bts import contest_state, daily_decision, picks, scheduler
from bts.health import contest_state as health_state

from offline import instant, offline


DAY = '2027-03-24'


def cli_pick(w, command='run'):
    if command == 'orchestrate':
        return w.call(command, '--date', w.day, '--config', str(w.config_path))
    return w.call(command, '--date', w.day, '--picks-dir', str(w.picks), '--models-dir', str(w.models))


def cached_fallbacks():
    print('\n=== in-loop and end-of-checks fallback ===')
    for in_loop in (False, True):
        with offline(instant(DAY, 18, 20)) as (w, mp):
            if not in_loop:
                w.schedule[DAY] = w.schedule[DAY][:1]
                w.dated(0)
                w.predictions = lambda: w.frame((0.86,))
                cli_pick(w, 'preview')
                w.dated(30)
                w.predictions = lambda: None
            else:
                w.dated(0)
                # Only the early game is considered; late contenders cannot defer it.
                w.schedule[DAY] = w.schedule[DAY][:1]
                original_save = picks.save_pick
                corrected = False

                def save_and_correct(daily, directory):
                    nonlocal corrected
                    path = original_save(daily, directory)
                    if not corrected:
                        corrected = True
                        w.dated(30)
                    return path

                mp.setattr(picks, 'save_pick', save_and_correct)
                w.predictions = lambda: (None if corrected else w.frame((0.99,), (True,)))
            scheduler.run_day(DAY, w.config)
            assert not w.sends('dm') and not w.sends('post')
            cached = picks.load_pick(DAY, w.picks)
            assert cached and not cached.delivery_attempted
            assert daily_decision.load_decision(DAY, w.picks) is None
            state = scheduler.load_state(DAY, w.picks)
            assert not state.pick_locked
            assert any('Saved pick' in body for body in w.sends('health-dm'))
            print('fallback', 'in-loop' if in_loop else 'end-of-checks', state.fallback_refreshes,
                  'predictions:', [t.isoformat() for kind, t, _ in w.events if kind == 'predict'])
            assert bool(state.fallback_refreshes) is in_loop


def deadline_starved_by_later_check():
    print('\n=== no predictions before a still-deliverable saved pick deadline ===')
    from bts import orchestrator

    for two_games in (False, True):
        with offline(instant(DAY, 18, 20)) as (w, _):
            if not two_games:
                w.schedule[DAY] = w.schedule[DAY][:1]
            w.dated(30)
            _, selection, _ = orchestrator.run_and_pick(w.config, DAY)
            daily = selection.pick_result.daily
            picks.save_pick(daily, w.picks)
            live = contest_state.load_decision_streak_state(w.picks, require_contest_state=True)
            assert picks.saved_pick_refusal(daily, live) is None
            w.predictions = lambda: None
            scheduler.run_day(DAY, w.config)
            assert bool(w.sends('dm')) is (not two_games)
            if two_games:
                assert list((w.picks / DAY).glob('stale_pick_*.json'))
                assert not w.sends('health-dm') and daily_decision.load_decision(DAY, w.picks) is None
            print('two games', two_games, 'pick DMs', len(w.sends('dm')),
                  'predictions:', [t.isoformat() for kind, t, _ in w.events if kind == 'predict'])


def fresh_correction_race():
    print('\n=== fresh selection after a dated correction ===')
    for driver in ('scheduler', 'run', 'orchestrate'):
        with offline(instant(DAY, 18, 25)) as (w, mp):
            w.dated(0)
            w.predictions = lambda: w.frame((0.99, 0.98))
            original_save = picks.save_pick
            corrected = False

            def save_and_correct(daily, directory):
                nonlocal corrected
                path = original_save(daily, directory)
                if not corrected:
                    corrected = True
                    w.dated(30)
                return path

            mp.setattr(picks, 'save_pick', save_and_correct)
            if driver == 'scheduler':
                scheduler.run_day(DAY, w.config)
                sent = w.sends('dm')
            else:
                cli_pick(w, driver)
                sent = w.sends('post')
            cached = picks.load_pick(DAY, w.picks)
            assert cached.chosen_on['dated']['active_streak'] == 0
            assert cached.double_down is not None and len(sent) == 1 and 'Streak: 30' in sent[0]
            assert cached.automatic_reading is None
        with offline(instant(DAY, 18, 25)) as (control, _):
            control.dated(30)
            control.predictions = lambda: control.frame((0.99, 0.98))
            if driver == 'scheduler':
                scheduler.run_day(DAY, control.config)
            else:
                cli_pick(control, driver)
            assert picks.load_pick(DAY, control.picks).double_down is None
            print(driver, 'corrected-before-selection control: single')
    with offline(instant(DAY, 18, 25)) as (w, mp):
        w.automatic(0, age_hours=20)
        w.predictions = lambda: w.frame((0.99, 0.98))
        original_save = picks.save_pick

        def save_then_replace(daily, directory):
            path = original_save(daily, directory)
            if not (w.account / 'contest_streak.manual.json').exists():
                w.dated(30)
            return path

        mp.setattr(picks, 'save_pick', save_then_replace)
        scheduler.run_day(DAY, w.config)
        cached = picks.load_pick(DAY, w.picks)
        assert cached.chosen_on['kind'] == 'automatic' and cached.double_down
        assert cached.automatic_reading is None and not w.sends('health-dm')
        assert len(w.sends('dm')) == 1 and 'Streak: 30' in w.sends('dm')[0]
        print('Fresh choice from an old automatic read went out after dated replacement with a null stale stamp')


def stale_stamp_and_restart():
    print('\n=== stale delivery, dated replacement, CLI exemption and restart ===')
    for driver in ('automatic-scheduler', 'dated-scheduler', 'run'):
        with offline(instant(DAY, 10)) as (w, _):
            w.automatic(30, age_hours=20)
            if driver == 'dated-scheduler':
                w.dated(30)
            if driver == 'run':
                w.now = instant(DAY, 18, 25)
                cli_pick(w)
            elif driver == 'dated-scheduler':
                scheduler.run_day(DAY, w.config)
            else:
                # Simulate service stop after commit, before waiting to poll results.
                with patch.object(scheduler, '_compute_result_poll_start', side_effect=KeyboardInterrupt):
                    try:
                        scheduler.run_day(DAY, w.config)
                    except KeyboardInterrupt:
                        pass
                    else:
                        raise AssertionError('Delivery did not reach polling setup')
            cached = picks.load_pick(DAY, w.picks)
            stamp = cached.automatic_reading
            assert (stamp is not None) is (driver == 'automatic-scheduler')
            warns = [body for body in w.sends('health-dm') if 'stale' in body]
            assert len(warns) == (1 if stamp else 0)
            if stamp:
                assert stamp['stale'] and stamp['age_hours'] > 20
            print(driver, 'stamp', stamp, 'warning count', len(warns))
            if driver != 'automatic-scheduler':
                continue
            decision_before = daily_decision.decision_path(DAY, w.picks).read_bytes()
            stamp_before = stamp.copy()
            # Force the durable pick/decision guard by simulating missing scheduler state.
            # Keep the restart before the first planned check so morning is eligible.
            w.now += calendar.timedelta(minutes=1)
            (w.picks / DAY / 'scheduler_state.json').unlink()
            w.call('season-contract', 'revoke', '--season', '2027', '--reason', 'offline restart',
                   '--picks-dir', str(w.picks))
            original_reader = contest_state.load_decision_streak_state
            read_times = []

            def track(*args, **kwargs):
                read_times.append(w.now)
                return original_reader(*args, **kwargs)

            with patch.object(contest_state, 'load_decision_streak_state', track):
                scheduler.run_day(DAY, w.config)
            assert not read_times and len(w.sends('dm')) == 1 and len(w.sends('health-dm')) == 1
            assert daily_decision.decision_path(DAY, w.picks).read_bytes() == decision_before
            assert picks.load_pick(DAY, w.picks).automatic_reading == stamp_before


def live_refusal_health_outcome():
    print('\n=== live refusal, health and day-outcome ===')
    with offline(instant(DAY, 10)) as (w, _):
        w.automatic(30)
        w.call('season-contract', 'revoke', '--season', '2027', '--reason', 'offline refusal',
               '--picks-dir', str(w.picks))
        scheduler.run_day(DAY, w.config)
        assert not (w.picks / f'{DAY}.json').exists()
        try:
            contest_state.load_decision_streak_state(w.picks, require_contest_state=True, now=w.now)
        except contest_state.ContestStateError as exc:
            reason = str(exc)
        else:
            raise AssertionError('Revoked switch accepted')
        health = health_state.check(w.picks, expected=True, now=w.now)
        assert health[-1].level == 'CRITICAL' and health[-1].message == reason
        assert len(w.sends('health-dm')) == 2 and not any('NO PICK DELIVERED' in m for m in w.sends('health-dm'))
        assert not w.sends('dm') and not w.sends('post')
        result = w.call('day-outcome', '--date', DAY, '--config', str(w.config_path), '--picks-dir',
                        str(w.picks), '--expected-username', 'offline', '--dm-recipient', 'offline.invalid', code=1)
        assert json.loads(result.stdout.splitlines()[0])['state'] == 'no_decision'
        assert 'NO DECISION' in w.sends('dm')[-1]


def overrun_double_cascade():
    print('\n=== regular check overrun and real fallback input ===')
    with offline(instant(DAY, 10)) as (w, _):
        w.schedule[DAY] = w.schedule[DAY][:1]
        w.dated(0)
        calls = []

        def forecast():
            calls.append(w.now)
            if w.now < instant(DAY, 18, 25):
                return None
            if w.now == instant(DAY, 18, 25):
                w.advance(35 * 60)
                return w.frame((0.99,), (True,))
            w.advance(6 * 60)
            return None

        w.predictions = forecast
        scheduler.run_day(DAY, w.config)
        state = scheduler.load_state(DAY, w.picks)
        assert state.fallback_refreshes[0]['reused_check_cascade'] is False
        assert calls == [instant(DAY, h, m) for h, m in ((16, 10), (17, 10), (18, 25), (19, 0))]
        assert not w.sends('dm')
        assert list((w.picks / DAY).glob('refused_delivery*.json'))
        print('cascade starts:', [t.isoformat() for t in calls])


if __name__ == '__main__':
    import sys

    checks = (cached_fallbacks, deadline_starved_by_later_check, fresh_correction_race, stale_stamp_and_restart,
              live_refusal_health_outcome, overrun_double_cascade)
    failures = 0
    for check in checks:
        if len(sys.argv) > 1 and check.__name__ not in sys.argv[1:]:
            continue
        try:
            check()
        except Exception:
            import traceback

            traceback.print_exc()
            failures += 1
    raise SystemExit(bool(failures))
