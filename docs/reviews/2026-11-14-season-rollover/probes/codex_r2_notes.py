import sys,io,json,tempfile
from pathlib import Path
from datetime import datetime,date
from zoneinfo import ZoneInfo
from contextlib import ExitStack
from unittest.mock import patch
sys.path[:0]=['/Users/eric/projects/bts-pp-c1/src']
from bts import picks,contest_state,strategy
from bts.health import streak_validation,contest_state as contest_health,pitcher_sparsity,projected_lineup,realized_calibration,slate_auc,result_resolution
from bts.cli import cli
from click.testing import CliRunner
ROOT=Path(tempfile.mkdtemp(prefix='g2-r2-notes-'));now=datetime(2027,3,24,8,30,tzinfo=ZoneInfo('America/New_York'));policy=strategy._load_mdp()
print('ROOT',ROOT,flush=True)
for kind in ['empty','old_model','old_contest','fresh_zero']:
    p=ROOT/kind;p.mkdir()
    if kind.startswith('old'):(p/'streak.json').write_text(json.dumps({'streak':18,'saver_available':False,'updated':'2026-09-26T23:00:00Z','applied_dates':['2026-09-26']}))
    if kind=='old_contest':
        (p/'account_state').mkdir();(p/'account_state/contest_streak.json').write_text(json.dumps({'active_streak':18,'source_date':'2026-09-26','recorded_at':'2026-09-27T08:00:00Z','source':'auto','best_streak':18}))
    if kind=='fresh_zero':
        from bts.contest_fetch import build_observation,derive_source_date
        profile={'activeStreak':0,'seasonBestStreak':0,'predictions':[]}
        observed=build_observation(profile,derive_source_date([],{}),1,'synthetic',now)
        (p/'account_state').mkdir();(p/'account_state/contest_streak.json').write_text(json.dumps(observed))
    ds=contest_state.load_decision_streak_state(p,now=now,require_contest_state=(kind in ['old_contest','fresh_zero']))
    print('STATE',kind,json.dumps(vars(ds),default=str),flush=True)
    for conf in [.75,.79,.85]:
        ctx=strategy.DecisionContext(primary_p=conf,second_p=None,has_diff_game=False,date='2027-03-24',allow_double=True,mdp=policy)
        print('ACTION',kind,conf,strategy.resolve_policy_decision(ctx,ds.streak,ds.saver_available).action,flush=True)
    for name,alerts in [('streak_validation',streak_validation.check(p)),('contest_state',contest_health.check(p,expected=True,now=now)),('pitcher_sparsity',pitcher_sparsity.check(p,today=now.date())),('projected_lineup',projected_lineup.check(p,today=now.date())),('realized_calibration',realized_calibration.check(p,today=now.date(),data_dir=ROOT/'no_parquet')),('slate_auc',slate_auc.check(p,ROOT/'no_parquet',today=now.date())),('result_resolution',result_resolution.check(p,today=now.date()))]:
        print('HEALTH',kind,name,json.dumps([{'level':a.level,'message':a.message} for a in alerts]),flush=True)
    before=picks.load_applied_dates(p)
    print('APPLY_2027',kind,picks.update_streak([True],p,date='2027-03-24'),'markers_before',before,'markers_after',picks.load_applied_dates(p),flush=True)
    print('REPLAY_EMPTY_2027',kind,picks._replay_season_streak(p,2027,'2027-03-24'),flush=True)
    result=CliRunner().invoke(cli,['check-results','--date','2027-03-23','--picks-dir',str(p)])
    print('EMPTY_GRADER_CLI',kind,'exit',result.exit_code,'output',repr(result.output),flush=True)
