from contextlib import nullcontext
from datetime import date, datetime, time
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

CHECK = Path(__file__).resolve().parent
sys.path.insert(0, str(CHECK / 'scratch'))
sys.path.insert(0, str(CHECK / 'scratch/src'))
from bts import daily_decision, orchestrator, picks, scheduler, strategy
from scripts.audit.season_walk import walk

day = date(2027, 7, 15)
now = datetime.combine(day, time(12), walk.ET)
with TemporaryDirectory() as temp, walk.offline(now):
    root = Path(temp)
    cfg = walk._config(root) | {'bluesky': {'dm_recipient': 'synthetic'}}
    cfg['orchestrator']['heartbeat_path'] = str(root / '.heartbeat')
    walk.install_state(root, day, walk.State('dated', 57, 57, 'used', .96))
    frame = walk._predictions(day, .96)
    statuses = {int(pk): {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame['game_pk']}
    with patch.object(orchestrator, 'run_cascade', return_value=(frame, 'probe')), \
            patch.object(picks, 'get_game_statuses_detailed', return_value=statuses):
        _, sel, _ = orchestrator.run_and_pick(cfg, str(day))
    daily = sel.pick_result.daily
    picks.save_pick(daily, root)
    with patch.object(orchestrator, 'run_cascade', return_value=(frame.assign(p_game_hit=float('nan')), 'probe')), \
            patch.object(picks, 'get_game_statuses_detailed', return_value=statuses), \
            patch.object(scheduler, 'heartbeat_watchdog', return_value=nullcontext()):
        refresh = scheduler._refresh_pick_at_fallback_decision(cfg, str(day), daily, 0.01)
    state = scheduler.SchedulerState(date=str(day), schedule_fetched_at=None, games=[], confirmed_game_pks=[],
                                     runs_completed=[], pick_locked=False, pick_locked_at=None,
                                     result_status=None, next_wakeup=None)
    with patch('bts.dm.send_dm', return_value='synthetic'), \
            patch.object(scheduler, '_trigger_live_forward_capture_on_lock'), \
            patch.object(scheduler, '_void_committed_games', return_value={}):
        delivered = scheduler._deliver_and_lock_pick(refresh.daily, cfg, root, state, str(day), 'fallback',
                                                         selection=refresh.selection)
    print('cached fallback', refresh.daily is daily, refresh.selection.no_pick_reason, delivered,
          daily_decision.load_decision(str(day), root))

for exc in (KeyboardInterrupt, SystemExit, ValueError):
    with patch.object(strategy, 'lookup_tail_action', side_effect=exc):
        try:
            print('existing tail rule', exc, strategy._tail_action(strategy._load_mdp(), .96, 5, 10, False, 5))
        except BaseException as err:
            print('existing tail rule propagated', type(err).__name__)
