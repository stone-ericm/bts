"""Independent JSON-domain and actual delivery/catch probes across three source trees."""
import contextlib
import dataclasses
import datetime as dt
import itertools
import json
import logging
from pathlib import Path
import socket
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo
import pandas as pd
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2];package=sys.argv[1]
sys.path.insert(0,str(ROOT/package/'src' if package!='branch' else REPO/'src'))
import bts.contest_state as cs
import bts.saver_state as ss
import bts.scheduler as sched
import bts.orchestrator as orch
import bts.health.alert as ha
import bts.dm as dm
import bts.posting as posting
import bts.picks as picks
import bts.progress as progress
import bts.web as web
from bts.strategy import _load_mdp,DecisionContext,resolve_policy_decision
socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(AssertionError('external network prohibited'))
logging.disable(logging.CRITICAL)
ET=ZoneInfo('America/New_York');NOW=dt.datetime(2027,7,15,12,tzinfo=ET)
class Clock(dt.datetime):
    @classmethod
    def now(cls,tz=None):return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)
raws=['MISSING','0','-0','1','9','10','57','58',str(10**30),'-1','-40','1.0','7.5','1e2','true','false','null','"7"','""','"active"','[]','[7]','{}','{"n":7}','-0.0','1e309','-1e309','NaN','Infinity','-Infinity']
work=ROOT/'evidence'/('domain-v3-'+package);work.mkdir(exist_ok=True);rows=[]
def folder(name):
    p=work/name/'picks';(p/'account_state').mkdir(parents=True,exist_ok=True);return p
def setup(p,raw,flag='not_earned',obs=False,tag='current'):
    f=p/'account_state/saver_state.json'
    if f.exists():f.unlink()
    if flag!='uninitialized':f.write_text(json.dumps({'state':flag,'season':2027}))
    prefix='' if tag=='tagless' else '"season":'+str(2027 if tag=='current' else 2026)+','
    body=prefix+'"streak":'+raw if raw!='MISSING' else prefix+'"saver_available":false'
    (p/'streak.json').write_text('{'+body+'}')
    c=p/'account_state/contest_streak.json'
    if c.exists():c.unlink()
    if obs:c.write_text('{"active_streak":7,"best_streak":7,"source_date":"2027-07-14"}')
def state_result(p,required):
    try:
        s=cs.load_decision_streak_state(p,now=NOW,require_contest_state=required)
        return {'state':dataclasses.asdict(s),'model_type':type(s.model_streak).__name__,
                'model_repr':repr(s.model_streak),'subtitle':web._streak_subtitle(s)}
    except Exception as e:return {'error':type(e).__name__,'message':str(e)}
mdp=_load_mdp()
for raw,flag,obs,required,tag in itertools.product(raws,['uninitialized','not_earned','active','used'],[False,True],[False,True],['current','prior','tagless']):
    p=folder('values');setup(p,raw,flag,obs,tag);r=state_result(p,required)
    value=0 if raw=='MISSING' or tag!='current' else json.loads(raw)
    valid=type(value) is int and value>=0
    if package=='branch':
        if not obs and (required or not valid):
            assert r['error']=='ContestStateError',(raw,flag,obs,required,tag,r)
            if not required:assert 'model state is corrupt; cannot use model fallback:' in r['message']
        else:
            assert 'error' not in r,(raw,flag,obs,required,tag,r)
            n=7 if obs else value
            assert r['state']['streak']==n
            assert r['state']['saver_available']==(flag=='active' or flag=='not_earned' and n<10)
            if obs:assert r['model_repr']==repr(value)
    if 'state' in r and type(r['state']['streak']) is int and r['state']['streak']>=0:
        out=resolve_policy_decision(DecisionContext(.82,.82,True,'2027-07-15',True,mdp),r['state']['streak'],r['state']['saver_available'])
        r['action']=out.action
    rows.append({'family':'values','key':[raw,flag,obs,required,tag],'result':r})
print(package,'domain matrix',len(rows),flush=True)

def daily():
    pick=picks.Pick('Review Batter',1,'NYY',1,'Review Pitcher',2,.82,[],False,3,'2027-07-15T23:00:00Z')
    return picks.DailyPick('2027-07-15','2027-07-15T15:00:00Z',pick,None,None)
def state():return sched.SchedulerState(date='2027-07-15',schedule_fetched_at='',games=[],confirmed_game_pks=[],
    runs_completed=[],pick_locked=False,pick_locked_at=None,result_status=None,next_wakeup=None)
status={3:{'abstract':'P','detailed':'Scheduled'}}
predictions=pd.DataFrame([{'batter_name':'Review Batter','batter_id':1,'team':'NYY','lineup':1,'pitcher_name':'Review Pitcher',
                         'pitcher_id':2,'p_game_hit':.82,'flags':[],'projected_lineup':False,'game_pk':3,'game_time':'2027-07-15T23:00:00Z'}])
cases=['null','"7"','[7]','{}','true','false','7.5','1.0','-1','7']
for i,(raw,entry,mode,flag,obs) in enumerate(itertools.product(cases,['deliver','single-check','orchestrate'],
                   ['dm','public','private'],['not_earned','active'],[False,True])):
    p=folder('delivery-'+str(i));setup(p,raw,flag,obs);dp=daily();picks.save_pick(dp,p);st=state();sent=[]
    config={'orchestrator':{'picks_dir':str(p),'heartbeat_path':str(p/'heartbeat')},'tiers':[],
        'scheduler':{'pick_delivery':mode,'contest_state_required':False},'bluesky':{'dm_recipient':'captured.test'}}
    with patch.object(cs,'datetime',Clock),patch.object(picks,'datetime',Clock),patch.object(ha,'datetime',Clock), \
         patch.object(sched,'_now_et',return_value=NOW),patch.object(sched,'_void_committed_games',return_value={}), \
         patch.object(sched,'_trigger_live_forward_capture_on_lock',return_value=None), \
         patch.object(picks,'get_game_statuses_detailed',return_value=status), \
         patch.object(sched,'count_new_confirmations',return_value=0), \
         patch.object(sched,'heartbeat_watchdog',return_value=contextlib.nullcontext()), \
         patch.object(orch,'run_cascade',return_value=(predictions,'captured')), \
         patch.object(orch,'load_config',return_value=config),patch.object(progress,'mark',return_value=None), \
         patch.object(dm,'send_dm',side_effect=lambda who,body:sent.append(['pick',body]) or 'captured-id'), \
         patch.object(ha,'send_dm',side_effect=lambda who,body:sent.append(['health',body]) or 'captured-id'), \
         patch.object(posting,'post_to_bluesky',side_effect=lambda body:sent.append(['public',body]) or 'captured-uri'), \
         patch.object(posting,'should_post_now',return_value=False):
        try:
            if entry=='deliver':out=sched._deliver_and_lock_pick(dp,config,p,st,'2027-07-15','review')
            elif entry=='single-check':
                out=sched.run_single_check('2027-07-15',[3],set(),config,0.1)
                out={k:v for k,v in out.items() if k not in ['pick_result','selection']}
            else:out=orch.orchestrate(p/'unused-config','2027-07-15')
            r={'return':out,'sent':sent,'locked':st.pick_locked,'notification':dp.notification_sent}
        except Exception as e:r={'error':type(e).__name__,'message':str(e),'sent':sent}
    invalid=type(json.loads(raw)) is not int or json.loads(raw)<0
    if package=='branch' and invalid and not obs and not (entry=='deliver' and mode=='private'):
        assert 'error' not in r,(raw,entry,mode,flag,obs,r)
        assert len(r['sent'])==1,(raw,entry,mode,flag,obs,r)
        assert 'model state is corrupt' in r['sent'][0][1]
        assert not st.pick_locked and not dp.notification_sent
    rows.append({'family':'delivery','key':[raw,entry,mode,flag,obs],'result':r})
print(package,'delivery probes',sum(r['family']=='delivery' for r in rows),flush=True)
out=ROOT/(package+'-domain-delivery-v3.json');out.write_text(json.dumps(rows,default=str,separators=(',',':')))
