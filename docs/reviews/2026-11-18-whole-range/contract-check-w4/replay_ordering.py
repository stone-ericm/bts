"""Use a ranked prediction frame to compare the proposed early action with main."""

from datetime import timedelta

from replay_main import DAY, daily_decision, offline, picks, predictions_at, scheduler, seed


def ordering(case, early):
    morning = case == 'morning-preview'
    start = offline.instant(DAY, 3, 0) if morning else offline.instant(DAY, 18, 20)
    with offline.offline(start) as (w, _):
        if morning:
            w.schedule[DAY].append(offline.game(103, offline.instant(DAY, 21, 10)))
            w.config['scheduler']['cascade_budget_min'] = 40
            w.dated(30)
            w.predictions = lambda: w.frame((0.99, 0.81, 0.60), (True, True, True))
            w.call('preview', '--date', DAY, '--picks-dir', str(w.picks),
                   '--models-dir', str(w.models))
            daily = picks.load_pick(DAY, w.picks)
            assert not daily.chosen_on or not daily.chosen_on['required']
            w.events.clear()
            next_check = offline.instant(DAY, 18, 25)
            at_deadline = lambda: w.frame((0.90, 0.81, 0.60), (False, False, False))
            at_check = lambda: w.frame((0.60, 0.99, 0.60), (True, False, True))
        else:
            w.schedule[DAY][1] = offline.game(102, offline.instant(DAY, 19, 30))
            daily = seed(w)
            next_check = offline.instant(DAY, 18, 45)
            at_deadline = lambda: None
            at_check = lambda: w.frame((0.60, 0.99), (True, False))

        def predict():
            frame = at_deadline() if w.now < next_check else at_check()
            return None if frame is None else frame.sort_values('p_game_hit', ascending=False)

        w.predictions = predict
        budget = w.config['scheduler'].get('cascade_budget_min', 12)
        lead = scheduler._fallback_min_with_floor(35, budget, 10)
        deadline = picks.earliest_pick_game_et(daily) - timedelta(minutes=lead)
        assert deadline < next_check < picks.submission_cutoff_et(daily)
        if early:
            # This replays the required W4 action using the real helpers.
            w.now = deadline
            runs = scheduler.plan_run_times(w.schedule[DAY], now=w.now)
            remaining = [run for run in runs if run['time_et'] >= next_check]
            state = scheduler.SchedulerState(
                date=DAY, schedule_fetched_at=w.now.isoformat(), games=[],
                confirmed_game_pks=[], runs_completed=[], pick_locked=False,
                pick_locked_at=None, result_status=None, next_wakeup=None,
            )
            sides = set()
            sync = sorted({pk for run in remaining for pk in run['game_pks']})
            scheduler.count_new_confirmations(sync, sides)
            refresh = scheduler._refresh_pick_at_fallback_decision(w.config, DAY, daily, 0.03)
            scheduler.count_new_confirmations(sync, sides)
            plan = scheduler.plan_fallback_action(
                now=w.now, cutoff=picks.submission_cutoff_et(refresh.daily),
                should_post=refresh.should_post, should_post_ungated=refresh.should_post_ungated,
                block_reason=refresh.block_reason, contender_game_pk=refresh.contender_game_pk,
                contender_game_pks=refresh.contender_game_pks, remaining_runs=remaining,
                confirmed_sides=sides, budget_min=budget, operator_reserve_min=10,
            )
            assert plan.action == 'deliver', plan
            assert scheduler._deliver_and_lock_pick(
                refresh.daily, w.config, w.picks, state, DAY, 'fallback', selection=refresh.selection,
            )
        else:
            scheduler.run_day(DAY, w.config)
        decision = daily_decision.load_decision(DAY, w.picks)
        target = 101 if early else 102
        assert decision['primary']['game_pk'] == target, decision
        sent = [when for event, when, body in w.events
                if event == 'dm' and body.startswith("Today's pick:")]
        assert sent == [deadline if early else next_check], sent
        print(case, 'W4 action replay' if early else 'main', 'game', target,
              'pick DM', sent[0].strftime('%H:%M'), 'predictions', predictions_at(w))


if __name__ == '__main__':
    for scenario in ('morning-preview', 'better-later'):
        ordering(scenario, early=False)
        ordering(scenario, early=True)
