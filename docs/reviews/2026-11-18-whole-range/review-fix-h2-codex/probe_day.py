import argparse
import json
from contextlib import ExitStack
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

from bts import cli, contest_fetch, daily_decision, orchestrator, picks, scheduler, web
from bts.day_outcome import gather
from bts.health import contest_state as health
from bts.saver_state import load_saver_state
from support import DAY, at, counts, deliver, fetch, fixture, fresh, install, invoke, live, scheduler_state, walk


def observe(directory, count, kind, day):
    with patch.object(scheduler, '_print_morning_check') as journal:
        scheduler._morning_state_check(walk._config(directory), directory, str(day), at(day), at(day, 13))
    line = journal.call_args.args[0]
    assert f'{kind} contest state accepted (streak {count})' in line, line
    command = invoke('season-contract', 'status', '--picks-dir', directory)
    assert command[0] == 0 and f'streak={count}' in command[1], command
    for historic in (None, day):
        alerts = health.check(directory, now=at(day), expected=True, historical_day=historic)
        assert all(alert.level != 'CRITICAL' for alert in alerts), alerts
    return line


def dashboard(directory, count):
    with patch.object(web, 'PICKS_DIR', directory), patch.object(web, 'fetch_bluesky_posts', return_value=[]), \
            patch.object(web, '_build_live_game_data', return_value=([], None)):
        html = web.render_page()
    assert f'class="streak-number">{count}<' in html


def entry_and_grade(directory, config, day, count, daily):
    slots = list(picks.iter_daily_pick_slots(daily))
    walk._set_clock(at(day, 18, 30))
    pending = [{'roundId': 1, 'playerId': slot.batter_id + 1000} for _, slot in slots]
    crosswalk = {slot.batter_id + 1000: slot.batter_id for _, slot in slots}
    with walk._fetch_stubs(count, count, day), \
            patch.object(contest_fetch, 'fetch_pending_predictions', return_value=pending), \
            patch.object(cli, '_fetch_bts_to_mlb', return_value=crosswalk):
        checked = invoke('check-pick-entered', '--picks-dir', directory, '--expected-username', walk.ACCOUNT)
    assert checked[0] == 0 and 'pick entered (match)' in checked[1], checked
    classified = gather(str(day), directory, config, expected_username=walk.ACCOUNT)
    assert classified.ok and classified.state in {'entered', 'entered_game_unverified'}, classified
    boxscore = {f'ID{slot.batter_id}': {'stats': {'batting': {'hits': 1, 'atBats': 4}}} for _, slot in slots}
    fake_game = {'gameData': {'status': {'abstractGameCode': 'F'}},
                 'liveData': {'boxscore': {'teams': {'home': {'players': boxscore}, 'away': {'players': {}}}}}}
    walk._set_clock(at(day + timedelta(days=1), 1))
    with patch.object(picks, '_fetch_json', return_value=fake_game) as feed:
        first = invoke('check-results', '--date', day, '--picks-dir', directory)
        assert first[0] == 0 and feed.call_count >= len(slots), first
        second = invoke('check-results', '--date', day, '--picks-dir', directory)
    assert second[0] == 0, second
    stored = picks.load_streak(directory, season=day.year)
    assert stored == count + len(slots), (first, second, stored)
    assert set(picks.load_pick(str(day), directory).slot_results.values()) == {'hit'}
    return classified.state, stored


def full_day(kind, count, day, restart):
    with TemporaryDirectory() as tmp, walk.offline(at(day)), ExitStack() as stack:
        directory = Path(tmp) / 'picks'
        cfg, games, frame, dm = fixture(stack, directory, day)
        install(directory, kind, count, count, day=day)
        picks.save_streak(count, directory, saver_available=True, season=day.year)
        current = live(directory)
        assert (current.streak, current.best_streak, current.best_status) == (count, count, 'trusted')
        observe(directory, count, kind, day)
        selected = fresh(day, cfg, games)
        assert selected.streak == count and selected.decision.objective == 'longest_streak'
        assert selected.action == ('double' if day == DAY else 'single')
        daily, raw = picks.load_pick_bytes(str(day), directory)
        assert daily.chosen_on['streak'] == count and daily.chosen_on['kind'] == kind
        assert raw is not None and picks.saved_pick_refusal(daily, current) is None
        if restart:
            with patch.object(orchestrator, 'run_cascade', return_value=(frame.iloc[:0], 'empty after restart')):
                refreshed = scheduler._refresh_pick_at_fallback_decision(cfg, str(day), daily, .01)
            assert refreshed.daily is daily and not scheduler._refresh_is_genuine_skip(refreshed)
            selected = refreshed.selection
        dm.reset_mock()
        assert deliver(directory, cfg, daily, selected, day)
        dm.assert_called_once()
        recorded = daily_decision.load_decision(str(day), directory)
        assert recorded['objective'] == 'longest_streak' and recorded['best_streak'] == count and recorded['scoreable']
        assert recorded['streak'] == count or (restart and recorded['streak'] is None), recorded
        dashboard(directory, count)
        name = 'contest_streak.json' if kind == 'automatic' else 'contest_streak.manual.json'
        state = directory / 'account_state' / name
        before = state.read_bytes()
        outcome, graded = entry_and_grade(directory, cfg, day, count, daily)
        assert state.read_bytes() == before
        print(kind, count, day, 'restart', restart, 'action', recorded['action'], 'commit streak', recorded['streak'],
              'day outcome', outcome, 'graded model streak', graded)


def typo(kind, active, best, actual, *, coverage=True):
    with TemporaryDirectory() as tmp, walk.offline(at()), ExitStack() as stack:
        directory = Path(tmp) / 'picks'
        cfg, games, _, dm = fixture(stack, directory)
        install(directory, kind, *actual, saver='not_earned' if kind == 'automatic' else 'used')
        prior = fresh(DAY, cfg, games)
        if kind == 'automatic':
            change = fetch(directory, counts(active, best), coverage=coverage)
        else:
            change = invoke('set-contest-streak', '--picks-dir', directory, '--date', DAY,
                            '--streak', active, '--best-streak', best, '--saver', 'used')
        assert change[0] == 0, change
        assert not dm.called, dm.call_args_list
        state = live(directory)
        assert (state.streak, state.best_streak) == (active, best)
        morning = observe(directory, active, kind, DAY)
        current = fresh(DAY, cfg, games)
        if current.action == 'skip':
            status = scheduler_state()
            scheduler._capture_fallback_skip(status, SimpleNamespace(selection=current))
            scheduler._write_endofday_skip(directory, str(DAY), status)
            assert not dm.called
        else:
            assert deliver(directory, cfg, picks.load_pick(str(DAY), directory), current)
            dm.assert_called_once()
        decision = daily_decision.load_decision(str(DAY), directory)
        assert decision['action'] == current.action and decision['best_streak'] == best
        assert decision['streak'] == active
        if current.action == 'skip':
            assert decision['effective_best'] == 57 and 'season-best 57' in current.decision.reason
        flag = state.saver_flag_state
        if kind == 'automatic':
            assert flag == ('active' if coverage else 'not_earned'), flag
            assert fetch(directory, counts(*actual))[0] == 0
            assert load_saver_state(directory, season=DAY.year).state == flag
        print('typo', kind, (active, best), 'actual', actual, 'coverage', coverage, 'action', prior.action,
              '->', current.action, 'trust', state.best_status, 'saver', flag, 'DM count', dm.call_count,
              'journal', morning, 'record', json.dumps({k: decision[k] for k in ('streak', 'best_streak', 'action', 'effective_best')}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('family', choices=('day', 'typo'))
    if parser.parse_args().family == 'day':
        for mode in ('dated', 'automatic'):
            full_day(mode, 58, DAY, False)
            full_day(mode, 100, date(2027, 7, 15), True)
            full_day(mode, 400, DAY, True)
    else:
        typo('dated', 75, 75, (57, 57))
        typo('dated', 12, 157, (12, 12))
        typo('automatic', 5, 157, (5, 5))
        typo('automatic', 5, 400, (5, 5), coverage=False)
        typo('dated', 75, 75, (56, 56))
