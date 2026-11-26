from datetime import date, datetime, time, timedelta
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

CHECK = Path(__file__).resolve().parent
mode = sys.argv[1]
sys.path.insert(0, str(CHECK / 'scratch'))
sys.path.insert(0, str(CHECK / ('base' if mode == 'base' else 'scratch') / 'src'))
from bts import daily_decision as decision, entry_receipt, orchestrator, picks, scheduler
from bts.contest_state import load_decision_streak_state
from bts.day_outcome import classify_day
from bts.health import contest_state as state_health
from scripts.audit.season_walk import walk


def fresh_state(day):
    return scheduler.SchedulerState(date=str(day), schedule_fetched_at=None, games=[], confirmed_game_pks=[],
                                    runs_completed=[], pick_locked=False, pick_locked_at=None,
                                    result_status=None, next_wakeup=None)


def choose(root, day, kind, builder=None):
    status = walk.install_state(root, day, walk.State(kind, 57, 57, 'used', .96))
    print('setup', status)
    cfg = walk._config(root) | {'bluesky': {'dm_recipient': 'test'}}
    frame = walk._predictions(day, .96)
    if builder is not None:
        frame = builder(frame)
    statuses = {int(pk): {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame['game_pk']}
    with patch.object(picks, 'get_game_statuses_detailed', return_value=statuses), \
            patch.object(orchestrator, 'run_cascade', return_value=(frame, 'probe')):
        _, selection, _ = orchestrator.run_and_pick(cfg, str(day))
    if selection.pick_result:
        daily = selection.pick_result.daily
    else:
        daily = picks.DailyPick(str(day), now.isoformat(), picks.pick_from_row(frame.iloc[0]), None, None)
        daily.chosen_on = picks.chosen_on_record(load_decision_streak_state(root, require_contest_state=True),
                                                required=True)
        daily.policy_decision = {'action': 'single', 'source': 'mdp', 'objective': 'longest_streak',
                                 'streak': 57, 'best_supplied': 57, 'best_status': 'trusted'}
    picks.save_pick(daily, root)
    return cfg, picks.load_pick(str(day), root), selection


day = date(2027, 9, 26)
now = datetime.combine(day, time(12), walk.ET)
for operation in ('cached', 'restart', 'failed_send', 'unknown_send', 'crossed', 'clamped', 'over_cap'):
    with TemporaryDirectory() as temp, walk.offline(now):
        root = Path(temp)
        builder = (lambda frame: frame.assign(game_pk=int(frame.iloc[0]['game_pk']))) if operation == 'clamped' else None
        cfg, daily, selection = choose(root, day, 'automatic', builder)
        path = decision.decision_path(str(day), root)
        state = fresh_state(day)
        print('probe', operation, 'pick', daily.policy_decision['action'], 'record', path.exists())
        if operation == 'restart':
            scheduler._write_classification_decision(root, str(day), action='double' if daily.double_down else 'single',
                                                      delivered=False, attempted=True,
                                                      primary=scheduler._row_from_daily(daily.pick),
                                                      double_down=scheduler._row_from_daily(daily.double_down),
                                                      state=state, daily=daily)
        else:
            if operation in ('crossed', 'over_cap'):
                later_streak, later_best = (0, 57) if operation == 'crossed' else (59, 59)
                with walk._fetch_stubs(later_streak, later_best, day - timedelta(days=1)):
                    exit_code, msg = walk._invoke('fetch-contest-streak', '--picks-dir', root,
                                                  '--expected-username', walk.ACCOUNT)
                live = load_decision_streak_state(root, require_contest_state=True)
                print('new fetch', exit_code, msg.strip().splitlines()[0][:200], 'accepted live',
                      live.streak, live.best_streak, 'refusal', picks.saved_pick_refusal(daily, live),
                      'alerts', [(a.level, a.message[:100]) for a in state_health.check(root, expected=True)])
            if operation == 'unknown_send':
                daily.delivery_attempted = True
                picks.save_pick(daily, root)
            opts = {'side_effect': OSError('synthetic 503')} if operation == 'failed_send' else {'return_value': 'local-id'}
            with patch('bts.dm.send_dm', **opts) as send, \
                    patch.object(scheduler, '_trigger_live_forward_capture_on_lock'), \
                    patch.object(scheduler, '_alert_delivery_outcome_unknown'), \
                    patch.object(scheduler, '_void_committed_games', return_value={}):
                sent = scheduler._deliver_and_lock_pick(daily, cfg, root, state, str(day), 'probe',
                                                        selection=selection if operation == 'clamped' and mode == 'patched' else None)
                print('send result', sent, 'calls', send.call_count,
                      send.call_args.args[1].replace('\n', ' ')[:210] if send.call_count else None)
        record, raw = decision.load_decision_bytes(str(day), root)
        print('record', record)
        if record:
            receipt = entry_receipt.EntryReceipt(et_date=str(day), started_at=now, clock=lambda: now,
                                                 expected_username='test', picks_dir=root)
            receipt.selection(daily, (root / f'{day}.json').read_bytes(), record, raw)
            built = receipt.record()
            print('day/receipt', classify_day(date=str(day), games=3, decision=record, decision_sha256=None,
                                             receipts=[], intent='enter', expected_username='test').state,
                  built['selection']['commit'], len(built['selection']['slots']))
        picks.save_streak(57, root, saver_available=True, season=2027)
        print('model after single hit', picks.update_streak([True], root, date=str(day)))

with TemporaryDirectory() as temp, walk.offline(now):
    root = Path(temp)
    _, daily, _ = choose(root, day, 'dated')
    print('dated CLI next value', walk._invoke('set-contest-streak', '--picks-dir', root, '--date', str(day),
                                               '--streak', 58, '--best-streak', 58, '--saver', 'used')[0])
    print('legacy CLI next value', walk._invoke('set-contest-streak', '--picks-dir', root, '--streak', 58,
                                                '--best-streak', 58)[0])
