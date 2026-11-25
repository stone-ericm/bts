from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

CHECK = Path(__file__).resolve().parent
mode = sys.argv[1]
sys.path.insert(0, str(CHECK / 'scratch'))
sys.path.insert(0, str(CHECK / ('base' if mode == 'base' else 'scratch') / 'src'))
from bts import daily_decision as decision, orchestrator, picks, scheduler, strategy
from bts.contest_fetch import validate_fetch
from bts.contest_state import load_decision_streak_state
from bts.health import contest_state as state_health, mdp_policy_alignment
from bts.simulate import mdp as mdp_module
from bts.skip_policy_shadow import record_skip_from_decision
from scripts.audit import bin_collapse_read, boundary_shadow_census, p05_tail_policy_audit
from scripts.audit.season_ledger.sources import day_records, pick_files
from scripts.audit.season_walk import walk


def entrypoints(kind, day):
    cfg = walk._config(root)
    frame = walk._predictions(day, .96)
    statuses = {int(pk): {'abstract': 'P', 'detailed': 'Scheduled'} for pk in frame['game_pk']}
    now = datetime.combine(day, time(12), walk.ET)
    with walk.offline(now), patch.object(picks, 'get_game_statuses_detailed', return_value=statuses), \
            patch.object(orchestrator, 'run_cascade', return_value=(frame, 'probe')), \
            patch('bts.model.predict.run_pipeline', return_value=frame), \
            patch('bts.posting.should_post_now', return_value=False):
        problems = walk.install_state(root, day, walk.State(kind, 57, 57, 'used', .96))
        s = load_decision_streak_state(root, require_contest_state=True)
        print('state', kind, day, problems, s.streak, s.best_streak, s.best_status, s.allow_double,
              'health', [(a.level, a.source) for a in state_health.check(root, expected=True)])
        _, sel, _ = orchestrator.run_and_pick(cfg, day.isoformat())
        print('orchestrator.run_and_pick', sel.action, sel.decision.objective,
              'files written', (root / f'{day}.json').exists(), decision.decision_path(str(day), root).exists())
        if sel.pick_result:
            d = sel.pick_result.daily
            picks.save_pick(d, root)
            recovered = picks.load_pick(str(day), root)
            print('chosen_on', recovered.chosen_on, 'refusal', picks.saved_pick_refusal(recovered, s))
            (root / f'{day}.json').unlink()
        for command in ('preview', 'run'):
            status, out = walk._invoke(command, '--date', day.isoformat(), '--picks-dir', root,
                                       '--models-dir', root / 'models')
            print(command, status, out.strip().splitlines()[-3:])
            (root / f'{day}.json').unlink(missing_ok=True)
        cfg['bluesky'] = {'dm_recipient': 'synthetic'}
        print('orchestrate result', orchestrator.orchestrate(Path('unused.toml'), str(day), config=cfg))
        (root / f'{day}.json').unlink(missing_ok=True)


loaded = strategy._load_mdp()
totals = {}
for d in range(1, 181):
    slate = (date(2027, 9, 27) - timedelta(days=d)).isoformat()
    for saver in (False, True):
        for p in (.01, .50, .99):
            for allow, partner in ((True, True), (False, True), (True, False)):
                ctx = strategy.DecisionContext(p, .80 if partner else None, partner, slate, allow, loaded, 57, 'trusted')
                result = strategy.resolve_policy_decision(ctx, 57, saver)
                key = ('late' if d <= 3 else 'early', result.action, result.policy_action,
                       result.objective, result.source, bool(result.tail_sha256))
                totals[key] = totals.get(key, 0) + 1
print('H1-H3 outcomes:', totals)
for day in ('2027-07-15', '2027-09-26', '2027-09-27', '2027-12-31', '2028-01-01', '2028-09-27', '2028-09-28'):
    ctx = strategy.DecisionContext(.99, .95, True, day, True, loaded, 57, 'trusted')
    print('year/bar', day, strategy._days_remaining(day), strategy.resolve_policy_decision(ctx, 57, False).action,
          strategy.effective_pick_bar(57, day, False), strategy.effective_pick_bar(57, day, False, mdp=None))

with TemporaryDirectory() as temp:
    root = Path(temp)
    entrypoints('dated', date(2027, 7, 15))
with TemporaryDirectory() as temp:
    root = Path(temp)
    entrypoints('automatic', date(2027, 9, 26))

with TemporaryDirectory() as temp:
    root = Path(temp)
    date_str = '2027-09-26'
    day = date.fromisoformat(date_str)
    now = datetime.combine(day, time(12), walk.ET)
    frame = walk._predictions(day, .96)
    pick = picks.pick_from_row(frame.iloc[0])
    primary = {'batter_id': pick.batter_id, 'game_pk': pick.game_pk, 'p_game_hit': .96}
    kwargs = dict(action='single', source='mdp', objective='longest_streak', primary=primary,
                  delivery_status='delivered', scoreable=True, streak=57, now=now)
    print('H7 writer accepted:', decision.write_decision(date_str, root, **kwargs) is not None)
    path = decision.decision_path(date_str, root)
    if not path.exists():
        path.parent.mkdir()
        path.write_text(json.dumps(kwargs | {'date': date_str, 'schema_version': decision.DECISION_SCHEMA,
                                            'now': now.isoformat()}))
    parsed = decision.parse_decision(path.read_text())
    print('rollback parse/objective/scoreable', bool(parsed), decision.decision_objective(parsed),
          decision.scoreable_commit_from(parsed, None))
    raw_pick = picks.DailyPick(date_str, now.isoformat(), pick, None, None, policy_decision={
        'source': 'mdp', 'objective': 'longest_streak', 'streak': 57, 'best_supplied': 57,
        'best_status': 'trusted', 'action': 'single', 'effective_best': None,
        'tail_sha256': 'abc', 'degraded_reason': None,
    })
    picks.save_pick(raw_pick, root)
    loaded_pick = picks.load_pick(date_str, root)
    print('pick round-trip', loaded_pick.policy_decision == raw_pick.policy_decision)
    print('alignment count', mdp_policy_alignment.compute_metrics(root, mdp_module.DEFAULT_POLICY_PATH, today=day).primary.n)
    row = day_records.parse_decision(f'{date_str}/decision.json', path.read_bytes()).observations[0]
    pk = pick_files.parse_pick_file(f'{date_str}.json', (root / f'{date_str}.json').read_bytes()).observations[0]
    print('ledger', row['objective_raw'], row['objective'], pk['pick_policy_objective'])
    print('bin collapse', bin_collapse_read.epoch(date_str, row['objective']), bin_collapse_read.epoch(date_str, 'longest_streak'))
    dec = strategy.PolicyDecision('single', 'single', 'mdp', 'longest_streak', 57, 1, 57, 'trusted', None, 'abc', None, None, None)
    sel = SimpleNamespace(decision=dec, source='mdp')
    scheduler._log_policy_decision(sel)
    print('scheduler selection/saved meta', scheduler._selection_decision_meta(sel), scheduler._selection_decision_meta(None, loaded_pick))
    print('summary', scheduler.build_skip_summary(frame, 57, decision=dec))
    print('p05', p05_tail_policy_audit.expected_objective(57, 1), p05_tail_policy_audit.parse_policy_line(
        f'{date_str}T12:00:00-04:00 x Policy: objective=longest_streak action=single streak=57 best=57 '
        '(trusted) effective_best=None days=1 tail=abc'))
    try:
        p05_tail_policy_audit.stop_rule_skip(57, 1, None)
    except TypeError as exc:
        print('p05 stop check error', exc)
    skip = parsed | {'action': 'skip', 'scoreable': False}
    path.write_text(json.dumps(skip))
    print('skip readers', decision.is_reach57_mdp_skip(skip), record_skip_from_decision(date_str, root, now=now),
          boundary_shadow_census._boundary_sample_decisions([(path, skip)]))
    with walk.offline(now):
        for classification in (False, True):
            path.unlink(missing_ok=True)
            state = scheduler.SchedulerState(date=date_str)
            if classification:
                scheduler._write_classification_decision(root, date_str, action='single', delivered=False,
                                                          attempted=True, primary=primary, double_down=None,
                                                          state=state, daily=loaded_pick)
            else:
                scheduler._commit_decision_for_pick(root, date_str, loaded_pick, selection=None,
                                                    delivery_status='delivered', state=state)
            print('H8 commit', classification, decision.load_decision(date_str, root))

for active, best in ((57, 57), (58, 58), (57, 58), (58, 57)):
    try:
        validate_fetch({'activeStreak': active, 'seasonBestStreak': best})
        print('fetch accepted', active, best)
    except Exception as exc:
        print('fetch rejected', active, best, exc)
