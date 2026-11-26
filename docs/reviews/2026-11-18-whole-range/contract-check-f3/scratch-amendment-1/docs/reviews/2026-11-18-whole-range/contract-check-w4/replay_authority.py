"""Compare authority recovery and game-status behavior on existing boundaries."""

from dataclasses import replace

from replay_boundaries import new_state
from replay_main import DAY, daily_decision, offline, picks, scheduler, seed


def authority():
    with offline.offline(offline.instant(DAY, 18, 20)) as (w, _):
        daily = seed(w)
        w.now = offline.instant(DAY, 18, 35)
        state = new_state(w)
        committed = {'game_pk': 102, 'batter_id': 602, 'batter_name': 'Private Batter'}
        daily_decision.write_decision(DAY, w.picks, action='single', source='mdp',
                                      primary=committed, delivery_status='private_locked', scoreable=True)
        attempted = replace(daily, delivery_attempted=True)
        picks.save_pick(attempted, w.picks)
        path = daily_decision.decision_path(DAY, w.picks)
        before = path.read_bytes()
        scheduler._write_classification_decision(w.picks, DAY, action='single', delivered=False,
                                                 primary=scheduler._row_from_daily(attempted.pick),
                                                 double_down=None, state=state, attempted=True, daily=attempted)
        assert path.read_bytes() == before
        assert not scheduler._deliver_and_lock_pick(attempted, w.config, w.picks, state, DAY, 'fallback')
        after = daily_decision.load_decision(DAY, w.picks)
        assert not w.sends('dm') and after['primary']['game_pk'] == 101
        assert after['delivery_status'] == 'locked_unconfirmed'
        print('wake precedence: delivered/attempted branch can replace an authoritative private commit')


def status():
    for w4_action in (False, True):
        with offline.offline(offline.instant(DAY, 18, 20)) as (w, mp):
            daily = seed(w)
            w.predictions = lambda: None
            w.schedule[DAY][0]['status'].update(
                abstractGameCode='F', detailedState='Postponed', statusCode='D',
            )
            if w4_action:
                w.now = offline.instant(DAY, 18, 35)

                def unavailable(*args):
                    raise OSError('synthetic intermittent schedule outage')

                mp.setattr(scheduler, '_delivery_game_statuses', unavailable)
                refresh = scheduler._refresh_pick_at_fallback_decision(w.config, DAY, daily, 0.03)
                assert refresh.should_post is None
                assert scheduler._deliver_and_lock_pick(
                    refresh.daily, w.config, w.picks, new_state(w), DAY, 'fallback',
                )
            else:
                scheduler.run_day(DAY, w.config)
            assert bool(w.sends('dm')) is w4_action
            print('postponed:', 'early fallback with status outage sends' if w4_action else 'main stays silent')


if __name__ == '__main__':
    authority()
    status()
