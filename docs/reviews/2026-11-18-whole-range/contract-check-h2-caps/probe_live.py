import json
from contextlib import ExitStack, nullcontext
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from bts import cli, contest_fetch as cf, contest_state as cs, daily_decision, orchestrator, picks, scheduler, strategy, web
from bts.day_outcome import gather
from bts.health import contest_state as health
from bts.leaderboard.scraper import _tab_streak_fields
from bts.saver_state import load_saver_state
from scripts.audit.season_walk import walk as w
from probe_validators import call, dated, invoke

CAP = getattr(cs, 'MAX_SEASON_STREAK', cs.TARGET_STREAK)
DAY = date(2027, 9, 26)


def at(day, hour=12, minute=0):
    return w.REAL_DATETIME(day.year, day.month, day.day, hour, minute, tzinfo=w.ET)


def cfg_for(directory):
    cfg = w._config(directory)
    cfg['orchestrator']['heartbeat_path'] = str(directory / '.heartbeat')
    cfg['bluesky'] = {'dm_recipient': 'synthetic'}
    return cfg


def fetch(directory, day, active, best, *, recipient=True, coverage=True):
    opts = ['--dm-recipient', 'synthetic'] if recipient else []
    with w._fetch_stubs(active, best, day - timedelta(days=1) if coverage else day), ExitStack() as st:
        if not coverage:
            st.enter_context(patch.object(cf, 'fetch_profile', lambda *a: dict(activeStreak=active, seasonBestStreak=best, predictions=[])))
        return invoke(['fetch-contest-streak', '--picks-dir', directory, '--expected-username', w.ACCOUNT, *opts])


def model_and_switch(directory, day, prior=57):
    assert not w.install_state(directory, day, w.State('automatic', prior, prior, 'used', .96))


def state(directory):
    return cs.load_decision_streak_state(directory, require_contest_state=True)


def sched_state(day):
    return scheduler.SchedulerState(str(day), None, [], [], [], False, None, None, None)


def inputs(stack, day, top=.96, dm=None):
    frame = w._predictions(day, top).head(2)
    statuses = {int(pk): {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame['game_pk']}
    stack.enter_context(patch.object(orchestrator, 'run_cascade', lambda *a: (frame, 'synthetic')))
    stack.enter_context(patch.object(picks, 'get_game_statuses_detailed', lambda *a: statuses))
    stack.enter_context(patch.object(scheduler, 'heartbeat_watchdog', lambda *a, **k: nullcontext()))
    stack.enter_context(patch.object(scheduler, 'count_new_confirmations', lambda *a: 6))
    stack.enter_context(patch.object(scheduler, '_trigger_live_forward_capture_on_lock', Mock()))
    stack.enter_context(patch.object(scheduler, 'fetch_schedule', lambda *a: [{'status': {'detailedState': 'Scheduled'}}]))
    stack.enter_context(patch('bts.dm.send_dm', dm if dm is not None else Mock(return_value='local-id')))
    return frame, list(statuses)


def select(day, directory, cfg, games):
    result = scheduler.run_single_check(str(day), games, set(), cfg, .01)
    selection = result['selection']
    print('selection', selection.action, selection.streak, selection.decision.objective,
          selection.decision.best_supplied, selection.decision.best_status, selection.decision.effective_best)
    return selection


def entry_and_grading(directory, day, cfg, model_start, saved):
    w._set_clock(at(day, 18, 30))
    slots = list(picks.iter_daily_pick_slots(saved))
    pending = [{'roundId': 1, 'playerId': slot.batter_id + 1000} for _, slot in slots]
    crosswalk = {slot.batter_id + 1000: slot.batter_id for _, slot in slots}
    with w._fetch_stubs(58, 100, day), patch.object(cf, 'fetch_pending_predictions', lambda *a: pending), \
            patch.object(cli, '_fetch_bts_to_mlb', lambda: crosswalk):
        result = invoke(['check-pick-entered', '--picks-dir', directory, '--expected-username', w.ACCOUNT])
    print('entry', result)
    assert result[0] == 0 and 'pick entered (match)' in result[2]
    result = gather(str(day), directory, cfg, expected_username=w.ACCOUNT)
    print('day outcome', result.as_dict())
    assert result.ok
    w._set_clock(at(day + timedelta(days=1), 1))
    with patch.object(picks, 'check_hit', return_value='hit'):
        first = invoke(['check-results', '--date', day, '--picks-dir', directory])
        second = invoke(['check-results', '--date', day, '--picks-dir', directory])
    model = picks.load_streak(directory, season=day.year)
    print('grade', first, second, 'model', model, 'slots', picks.load_pick(str(day), directory).slot_results)
    assert first[0] == second[0] == 0 and model == model_start + len(slots)


def dash(directory, expected):
    with patch.object(web, 'PICKS_DIR', directory), patch.object(web, 'fetch_bluesky_posts', return_value=[]), \
            patch.object(web, '_build_live_game_data', return_value={}):
        html = web.render_page()
    print('dashboard', f'class="streak-number">{expected}<' in html, 'leaderboard',
          _tab_streak_fields('active_streak', dict(streak=expected, activeStreak=expected)),
          _tab_streak_fields('all_season', dict(streak=expected, activeStreak=12)))
    assert f'class="streak-number">{expected}<' in html


def full_day(kind, count, day, *, restart=False):
    with TemporaryDirectory() as tmp, w.offline(at(day)), ExitStack() as st:
        directory = Path(tmp) / 'picks'
        directory.mkdir()
        dm = Mock(return_value='local-id')
        _, games = inputs(st, day, dm=dm)
        cfg = cfg_for(directory)
        problems = w.install_state(directory, day, w.State(kind, count, count, 'used', .96))
        print('day', kind, count, day, 'restart', restart, 'setup', problems)
        if problems:
            if kind == 'dated':
                manual = directory / 'account_state/contest_streak.manual.json'
                manual.parent.mkdir(exist_ok=True)
                sample = dict(dated(count, count), effective_date=str(day), expires_at=f'{day}T23:59:59.999999-04:00')
                manual.write_text(json.dumps(sample))
            print('live failure', call(lambda: state(directory)))
            return
        picks.save_streak(count, directory, saver_available=True, season=day.year)
        captured = state(directory)
        assert (captured.streak, captured.best_streak, captured.best_status) == (count, count, 'trusted')
        print('live', captured.streak, captured.best_status, captured.saver_available, 'health', health.check(directory, now=at(day), expected=True),
              'historical', health.check(directory, now=at(day), expected=True, historical_day=day))
        scheduler._morning_state_check(cfg, directory, str(day), at(day), at(day, 13))
        status = invoke(['season-contract', 'status', '--picks-dir', directory])
        print('status', status)
        work = select(day, directory, cfg, games)
        assert work.decision.objective == 'longest_streak'
        saved = picks.load_pick(str(day), directory)
        assert saved and saved.chosen_on and picks.saved_pick_refusal(saved, captured) is None
        if restart:
            with patch.object(orchestrator, 'run_cascade', return_value=(w._predictions(day, .96).iloc[:0], 'synthetic')):
                refresh = scheduler._refresh_pick_at_fallback_decision(cfg, str(day), saved, .01)
            assert refresh.daily is saved and not scheduler._refresh_is_genuine_skip(refresh)
            work = refresh.selection
        dm.reset_mock()
        ok = scheduler._deliver_and_lock_pick(saved, cfg, directory, sched_state(day), str(day),
                                              'restarted fallback' if restart else 'lineup lock', selection=work)
        record = daily_decision.load_decision(str(day), directory)
        print('send', ok, dm.call_args_list, 'chosen_on', saved.chosen_on, 'decision', record)
        assert ok and dm.call_count == 1 and record['objective'] == 'longest_streak' and record['scoreable']
        dash(directory, count)
        auto = directory / 'account_state/contest_streak.json'
        manual = directory / 'account_state/contest_streak.manual.json'
        obs = (auto if kind == 'automatic' else manual).read_bytes()
        entry_and_grading(directory, day, cfg, count, saved)
        assert (auto if kind == 'automatic' else manual).read_bytes() == obs


def stale_and_bad_fetch():
    with TemporaryDirectory() as tmp, w.offline(at(DAY)), ExitStack() as st:
        directory = Path(tmp) / 'picks'
        dm = Mock(return_value='local-id')
        _, games = inputs(st, DAY, dm=dm)
        cfg = cfg_for(directory)
        model_and_switch(directory, DAY)
        body = directory / 'account_state/contest_streak.json'
        ledger = directory / 'account_state/contest_ledger.jsonl'
        status = Path(tmp) / 'health_state/contest_streak_fetch_status.json'
        before = body.read_bytes(), ledger.read_bytes()
        for value, recipient in [(58, True), (401, True), (401, True), (401, False)]:
            dm.reset_mock()
            result = fetch(directory, DAY, value, value, recipient=recipient)
            print('K10', value, recipient, result, 'observation/ledger changed',
                  (before[0] != body.read_bytes(), before[1] != ledger.read_bytes()),
                  'status', json.loads(status.read_text()), 'DMs', dm.call_args_list)
            before = body.read_bytes(), ledger.read_bytes()
        current = state(directory)
        print('K10 live', current.streak, current.best_status, cs.automatic_reading_record(current, at(DAY)))
        dm.reset_mock()
        scheduler._morning_state_check(cfg, directory, str(DAY), at(DAY), at(DAY, 13))
        work = select(DAY, directory, cfg, games)
        saved = picks.load_pick(str(DAY), directory)
        scheduler._deliver_and_lock_pick(saved, cfg, directory, sched_state(DAY), str(DAY), 'K10', selection=work)
        print('K10 delivered', dm.call_args_list, 'decision', daily_decision.load_decision(str(DAY), directory))
        for clock in (at(DAY, 23), at(DAY, 23, 1)):
            print('age boundary', cs.automatic_reading_record(current, clock))


def wrong_case(kind, active, best, day, *, top=.96):
    with TemporaryDirectory() as tmp, w.offline(at(day)), ExitStack() as st:
        directory = Path(tmp) / 'picks'
        dm = Mock(return_value='local-id')
        _, games = inputs(st, day, top, dm=dm)
        cfg = cfg_for(directory)
        if kind == 'automatic':
            model_and_switch(directory, day, prior=5)
            invoke(['saver-state', '--init', 'not_earned', '--force', '--season', day.year, '--picks-dir', directory])
            result = fetch(directory, day, active, best)
        else:
            result = w.install_state(directory, day, w.State(kind, active, best, 'used', top))
        print('typo', kind, active, best, day, result, 'initial DMs', dm.call_args_list)
        if kind == 'dated' and result:
            return
        current = state(directory)
        print('typo health', health.check(directory, now=at(day), expected=True))
        scheduler._morning_state_check(cfg, directory, str(day), at(day), at(day, 13))
        work = select(day, directory, cfg, games)
        if work.action != 'skip':
            scheduler._deliver_and_lock_pick(picks.load_pick(str(day), directory), cfg, directory,
                                             sched_state(day), str(day), 'typo', selection=work)
        else:
            scheduler._write_classification_decision(directory, str(day), action='skip', delivered=False,
                                                     primary=None, double_down=None, state=sched_state(day),
                                                     selection=work)
        print('typo DMs', dm.call_args_list, 'decision', daily_decision.load_decision(str(day), directory),
              'saver', current.saver_flag_state, 'dashboard saver warning',
              web.saver_dashboard_context(directory, now=at(day)).warning)
        if kind == 'automatic':
            print('correct fetch', fetch(directory, day, 5, 5), 'saver remains', load_saver_state(directory, season=day.year))


def end_of_day_correction():
    with TemporaryDirectory() as tmp, w.offline(at(DAY)), ExitStack() as st:
        directory = Path(tmp) / 'picks'
        dm = Mock(return_value='local-id')
        _, games = inputs(st, DAY, dm=dm)
        cfg = cfg_for(directory)
        assert not w.install_state(directory, DAY, w.State('dated', 12, 57, 'used', .96))
        # Choose a real pick while reach-57 is still reachable on an earlier day.
        early = date(2027, 7, 15)
        w._set_clock(at(early))
        games = inputs(st, early, dm=dm)[1]
        assert not w.install_state(directory, early, w.State('dated', 12, 57, 'used', .96))
        work = select(early, directory, cfg, games)
        assert work.action != 'skip'
        saved = picks.load_pick(str(early), directory)
        w._set_clock(at(early, 18, 45))
        command = ['set-contest-streak', '--picks-dir', directory, '--date', early, '--streak', 12,
                   '--best-streak', 58, '--saver', 'used']
        print('late best-only correction', invoke(command))
        with patch.object(orchestrator, 'run_cascade', return_value=(w._predictions(early, .96).iloc[:0], 'synthetic')):
            refresh = scheduler._refresh_pick_at_fallback_decision(cfg, str(early), saved, .01)
        print('late fallback retains', refresh.daily is saved, 'refusal', picks.saved_pick_refusal(saved, state(directory)))
        ok = scheduler._deliver_and_lock_pick(refresh.daily, cfg, directory, sched_state(early), str(early),
                                              'final fallback', selection=refresh.selection)
        print('late best-only send', ok, dm.call_args_list, daily_decision.load_decision(str(early), directory))
        assert ok == (CAP == 57)


def main():
    print(cs.__file__, 'cap', CAP)
    for kind in ('dated', 'automatic'):
        full_day(kind, 58, date(2027, 7, 15), restart=False)
        full_day(kind, 100, DAY, restart=True)
    stale_and_bad_fetch()
    wrong_case('dated', 75, 75, DAY)
    wrong_case('dated', 57, 157, DAY)
    wrong_case('dated', 12, 157, DAY)
    wrong_case('automatic', 75, 157, DAY)
    end_of_day_correction()


if __name__ == '__main__':
    main()
