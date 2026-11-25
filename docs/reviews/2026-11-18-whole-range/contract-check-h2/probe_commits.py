from contextlib import ExitStack, nullcontext
from datetime import date, datetime, time, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
import os
import sys


check = Path(__file__).resolve().parent
mode, family = sys.argv[1:]
scratch = check / 'scratch'
os.chdir(scratch)
sys.path[:0] = [str((check / 'base' if mode == 'base' else scratch) / 'src'), str(scratch)]
import bts
from bts import daily_decision, orchestrator, picks, scheduler, strategy
from bts.contest_state import load_decision_streak_state
from bts.simulate import mdp, tail_policy
from scripts.audit.season_walk import walk


print('bts', bts.__file__, flush=True)


def frame_at(day, p):
    frame = walk._predictions(day, p)
    statuses = {int(pk): {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame['game_pk']}
    return frame, statuses


def state(day):
    return scheduler.SchedulerState(str(day), None, [], [], [], False, None, None, None)


if family == 'checks':
    cases = ('2027-07-15', '2027-09-24', '2027-09-25', '2027-09-26', '2028-01-01')
    for kind in ('dated', 'automatic'):
        for text in cases:
            day = date.fromisoformat(text)
            with TemporaryDirectory() as folder, walk.offline(datetime.combine(day, time(12), walk.ET)):
                root = Path(folder)
                cfg = walk._config(root)
                cfg['orchestrator']['heartbeat_path'] = str(root / '.heartbeat')
                frame, statuses = frame_at(day, .96)
                setup = walk.install_state(root, day, walk.State(kind, 57, 57, 'used', .96))
                if setup:
                    print(day, kind, 'setup', setup)
                with patch.object(scheduler, 'count_new_confirmations', return_value=6), \
                        patch.object(scheduler, 'heartbeat_watchdog', return_value=nullcontext()), \
                        patch.object(orchestrator, 'run_cascade', return_value=(frame, 'probe')), \
                        patch.object(picks, 'get_game_statuses_detailed', return_value=statuses), \
                        patch.object(scheduler, '_lock_decision_from_predictions', return_value=scheduler.LockDecision(
                            True, None, True)):
                    result = scheduler.run_single_check(text, list(statuses), set(), cfg, .01)
                    sel = result['selection']
                    saved = picks.load_pick(text, root)
                    if saved is not None:
                        scheduler._commit_decision_for_pick(root, text, saved, selection=sel,
                                                            delivery_status='delivered', state=state(day))
                    rec = daily_decision.load_decision(text, root)
                    print('H5', text, kind, sel.action if sel else None,
                          bool(saved and picks._valid_chosen_on(saved.chosen_on)),
                          saved.policy_decision['objective'] if saved else None,
                          ((rec['objective'], rec['tail_policy_sha256'], rec['policy_guard'],
                            bool(saved.double_down), bool(rec['double_down'])) if rec else None))
    sys.exit(0)


day = date(2027, 7, 15)
now = datetime.combine(day, time(12), walk.ET)
with TemporaryDirectory() as folder, walk.offline(now):
    root = Path(folder)
    cfg = walk._config(root) | {'bluesky': {'dm_recipient': 'synthetic'}}
    cfg['orchestrator']['heartbeat_path'] = str(root / '.heartbeat')
    assert not walk.install_state(root, day, walk.State('automatic', 57, 57, 'used', .96))
    frame, statuses = frame_at(day, .96)
    with patch.object(orchestrator, 'run_cascade', return_value=(frame, 'probe')), \
            patch.object(picks, 'get_game_statuses_detailed', return_value=statuses):
        _, selection, _ = orchestrator.run_and_pick(cfg, str(day))
    daily = selection.pick_result.daily
    picks.save_pick(daily, root)
    low = frame.assign(p_game_hit=[.70, .69], batter_id=[900, 901], batter_name=['Fresh A', 'Fresh B'])
    for operation in ('empty', 'known-failure', 'crossed-heuristic'):
        with ExitStack() as st:
            daily_decision.decision_path(str(day), root).unlink(missing_ok=True)
            picks.save_pick(daily, root)
            if operation == 'crossed-heuristic':
                with walk._fetch_stubs(0, 57, day - timedelta(days=1)):
                    code, message = walk._invoke('fetch-contest-streak', '--picks-dir', root,
                                                 '--expected-username', walk.ACCOUNT)
                assert code == 0, message
                st.enter_context(patch.object(mdp, 'DEFAULT_POLICY_PATH', root / 'gone-base.npz'))
                st.enter_context(patch.object(tail_policy, 'DEFAULT_TAIL_POLICY_PATH', root / 'gone-tail.npz'))
                st.enter_context(patch.object(strategy, '_mdp_cache', None))
                updated = low
            else:
                updated = frame.assign(p_game_hit=float('nan'))
            st.enter_context(patch.object(scheduler, 'heartbeat_watchdog', return_value=nullcontext()))
            st.enter_context(patch.object(picks, 'get_game_statuses_detailed', return_value=statuses))
            st.enter_context(patch.object(orchestrator, 'run_cascade', return_value=(updated, 'probe')))
            st.enter_context(patch.object(scheduler, '_trigger_live_forward_capture_on_lock'))
            st.enter_context(patch.object(scheduler, '_void_committed_games', return_value={}))
            send_kwargs = ({'side_effect': OSError('synthetic reject')} if operation == 'known-failure'
                           else {'return_value': 'local-id'})
            send = st.enter_context(patch('bts.dm.send_dm', **send_kwargs))
            refresh = scheduler._refresh_pick_at_fallback_decision(cfg, str(day), daily, .01)
            assert refresh.daily is daily and not scheduler._refresh_is_genuine_skip(refresh)
            outcome = scheduler._deliver_and_lock_pick(daily, cfg, root, state(day), str(day), 'final fallback',
                                                        selection=refresh.selection)
            record = daily_decision.load_decision(str(day), root)
            saved = picks.load_pick(str(day), root)
            print(operation, 'send', outcome, send.call_count, 'choice', saved.policy_decision['objective'],
                  saved.policy_decision['tail_sha256'], 'attempt marker', saved.delivery_attempted,
                  'live streak', load_decision_streak_state(root, require_contest_state=True).streak,
                  'record', record)
