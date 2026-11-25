from contextlib import nullcontext
from datetime import date, datetime, time
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from unittest.mock import patch

CHECK = Path(__file__).resolve().parent
mode = sys.argv[1]
sys.path.insert(0, str(CHECK / 'scratch'))
sys.path.insert(0, str(CHECK / ('base' if mode == 'base' else 'scratch') / 'src'))
from bts import daily_decision as decision, orchestrator, picks, scheduler, web
from bts.health import mdp_policy_alignment
from bts.simulate.mdp import DEFAULT_POLICY_PATH
from scripts.audit import boundary_shadow_census as census
from scripts.audit.season_walk import walk


day = date(2027, 7, 15)
now = datetime.combine(day, time(12), walk.ET)
for no_eligible in (False, True):
    with TemporaryDirectory() as temp, walk.offline(now):
        root = Path(temp)
        cfg = walk._config(root)
        cfg['orchestrator']['heartbeat_path'] = str(root / '.heartbeat')
        frame = walk._predictions(day, .96)
        statuses = {int(pk): {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame['game_pk']}
        all_games = list(statuses)
        walk.install_state(root, day, walk.State('dated', 57, 57, 'used', .96))
        with patch.object(scheduler, 'count_new_confirmations', return_value=6), \
                patch.object(scheduler, 'heartbeat_watchdog', return_value=nullcontext()), \
                patch.object(picks, 'get_game_statuses_detailed', return_value=statuses), \
                patch.object(orchestrator, 'run_cascade', return_value=(frame, 'probe')), \
                patch.object(scheduler, '_lock_decision_from_predictions', return_value=scheduler.LockDecision(
                    True, None, True)), patch.object(scheduler, 'predict_local_shadow', return_value=frame):
            result = scheduler.run_single_check(str(day), all_games, set(), cfg, 0.01,
                                                unavailable_game_pks=set(all_games) if no_eligible else set())
            if not no_eligible:
                picks.save_streak(57, root, saver_available=False, season=2027)
                scheduler._run_shadow_prediction(cfg, str(day), 'Synthetic')
                shadow = picks.load_shadow_pick(str(day), root)
                print('shadow model 57:', shadow.policy_decision['objective'] if shadow else None)
        selection = result['selection']
        summary = result.get('skip_summary')
        print('scheduler', no_eligible, selection.action, selection.no_pick_reason, summary, web.render_skip_banner(summary))
        if result['pick_result']:
            daily = picks.load_pick(str(day), root)
            print('precommit alignment includes:', mdp_policy_alignment.compute_metrics(
                root, DEFAULT_POLICY_PATH, today=day).primary.n)
            state = scheduler.SchedulerState(date=str(day), schedule_fetched_at=None, games=[], confirmed_game_pks=[],
                                             runs_completed=[], pick_locked=False, pick_locked_at=None,
                                             result_status=None, next_wakeup=None)
            scheduler._commit_decision_for_pick(root, str(day), daily, selection=selection,
                                                delivery_status='delivered', state=state)
            record = decision.load_decision(str(day), root)
            print('saved action/runner and record:', daily.policy_decision['action'], daily.double_down, record)
            ledger = root / 'ledger.jsonl'
            ledger.touch()
            kwargs = dict(picks_dir=root, ledger_path=ledger, policy_path=DEFAULT_POLICY_PATH, saver_active=False,
                          start=str(day), end=str(day), as_of=str(day), output=root / 'census.json')
            c = census.run_census(**kwargs)
            print('census', c['rows'], 'sample sizes', {k: v['n'] for k, v in c['boundaries'].items()})
            daily.policy_npz_sha256 = '1' * 64
            picks.save_pick(daily, root)
            try:
                census.run_census(**kwargs)
            except census.CensusHalt as exc:
                print('census invalid provenance before objective filter:', exc)
