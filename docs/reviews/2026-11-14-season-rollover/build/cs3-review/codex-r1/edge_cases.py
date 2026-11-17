"""Real-file edge and concurrency probes supplementing the normal contract matrix."""
import dataclasses
import datetime as dt
import importlib.util
import json
import logging
from pathlib import Path
import socket
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo
import pandas as pd

ROOT=Path(__file__).resolve().parent
REPO=ROOT.parents[1]
package=sys.argv[1]
sys.path.insert(0,str(ROOT/'base/src' if package=='base' else REPO/'src'))
import bts.contest_state as cs
import bts.saver_state as ss
import bts.web as web
import bts.strategy as strategy
import bts.cli as cm
from bts.health.contest_state import check
from bts.health.attention import apply_warn_attention_policy
import bts.health.alert as alert
from bts.picks import DailyPick,Pick,save_pick
from click.testing import CliRunner
assert ('base/src' in cs.__file__)==(package=='base')
socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(AssertionError('network forbidden'))
logging.disable(logging.CRITICAL)
ET=ZoneInfo('America/New_York'); NOW=dt.datetime(2027,7,15,12,tzinfo=ET)
rows=[]; work=ROOT/'evidence'/('edges-'+package); work.mkdir(exist_ok=True)

def folder(name):
    p=work/name; (p/'account_state').mkdir(parents=True,exist_ok=True); return p
def flag(p,state='not_earned'):
    path=p/'account_state/saver_state.json'; path.write_text(json.dumps({'state':state,'season':2027})); return path
def contest(p,n=7):
    path=p/'account_state/contest_streak.json'
    path.write_text(json.dumps({'active_streak':n,'best_streak':n,'source_date':'2027-07-14'})); return path
def call(fn):
    try:
        r=fn();return dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r
    except Exception as e:return {'error':type(e).__name__,'message':str(e)}
def cli(p,*args):
    r=CliRunner().invoke(cm.cli,['saver-state','--picks-dir',str(p),'--season','2027',*args])
    return {'exit':r.exit_code,'output':r.output,'exception':repr(r.exception)}
def audits(p):
    path=p/'account_state/saver_transitions.jsonl'
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

# Non-integers really do reach the caller via JSON; a posted pick previously remained reusable.
for i,value in enumerate([None,'7',[],{},True,-1,7.5,10,0]):
    p=folder('model-'+str(i));flag(p)
    (p/'streak.json').write_text(json.dumps({'season':2027,'streak':value}))
    pick=Pick('Fixture Batter',1,'NYY',1,'Fixture Pitcher',2,.8,[],False,3,'2027-07-15T23:00:00Z')
    save_pick(DailyPick('2027-07-15','2027-07-15T12:00:00Z',pick,None,None,bluesky_posted=True),p)
    def real_path():
        s=cs.load_decision_streak_state(p,now=NOW)
        out=strategy.select_pick(pd.DataFrame([{'game_pk':3,'p_game_hit':.8}]),'2027-07-15',p,
            streak=s.streak,saver_available=s.saver_available,game_statuses_detailed={3:{'abstractGameState':'Preview','detailedState':'Scheduled'}})
        return {'state':dataclasses.asdict(s),'returned_posted_pick':out.pick_result is not None,
                'locked':out.pick_result.locked if out.pick_result else None}
    rows.append({'family':'model-existing-pick','key':value,'result':call(real_path)})

# Real permission failures, as opposed to malformed bytes called "unreadable".
for st in ['missing','permissions','invalid_utf8','non_object','invalid_state','stale','not_earned','active','used']:
    for n in [0,5,9,10,15,16,57]:
        p=folder('health-'+st+'-'+str(n)); f=flag(p,st if st in ['not_earned','active','used'] else 'active');contest(p,n)
        if st=='missing':f.unlink()
        elif st=='permissions':f.chmod(0)
        elif st=='invalid_utf8':f.write_bytes(b'\xff')
        elif st=='non_object':f.write_text('[]')
        elif st=='invalid_state':f.write_text('{"state":{},"season":2027}')
        elif st=='stale':f.write_text('{"state":"active","season":2026}')
        try:
            for expected in [False,True]:
                aa=check(p,expected=expected,now=NOW)
                rows.append({'family':'health-files','key':[st,n,expected],'alerts':[dataclasses.asdict(a) for a in aa]})
        finally:
            if st=='permissions': f.chmod(0o600)

# Exercise the new retry across a real promotion, not mocked transition results.
for initial,concurrent in [('not_earned','active'),('active','used')]:
    p=folder('race-'+initial);flag(p,initial);real=ss.transition_saver_state;seen=[]
    def racing(*args,**kw):
        if not seen:
            real(*args,expected_prior=initial,new_state=concurrent,season=2027,source='concurrent')
        seen.append(kw['expected_prior'])
        return real(*args,**kw)
    with patch.object(ss,'transition_saver_state',side_effect=racing):result=cli(p,'--use')
    rows.append({'family':'race','key':[initial,concurrent],'result':result,'candidates':seen,'audits':audits(p),
                 'state':ss.load_saver_state(p,season=2027).state})

# Verify POST's independence even if the contest loader itself raises arbitrary exceptions.
p=folder('post-no-read');flag(p,'active')
with patch.object(cs,'load_contest_streak_state',side_effect=AssertionError('must not read')):
    rows.append({'family':'post-no-read','result':call(lambda:web.saver_transition_response(p,expected_prior='active',
                new_state='used',same_origin=True,now=NOW))})

# Existing storage-layer failures are separate from the contest-file promise in row 7.
for bad,body in [('deep',b'['*2000+b'0'+b']'*2000),('large_int',b'{"season":'+b'9'*5000+b'}')]:
    p=folder('bad-saver-'+bad);flag(p,'active').write_bytes(body);contest(p)
    rows.append({'family':'existing-saver-loader','key':bad,'load':call(lambda:ss.load_saver_state(p,season=2027)),
                 'post':call(lambda:web.saver_transition_response(p,expected_prior='active',new_state='used',same_origin=True,now=NOW))})

# Direct labels and serialized decision fields, including unknown-streak fallback.
for st,n in [(s,n) for s in ['not_earned','active','used','uninitialized'] for n in [0,9,10,16,None]]:
    context=web.SaverDashboardContext(st,None,st,False,False)
    if package=='branch':page=web.render_saver_section(context,decision_streak=n)
    else:page=web.render_saver_section(context)
    rows.append({'family':'labels','key':[st,n],'html':page})

# Preserve dedup compatibility: old contest_state entries cannot suppress the new warning.
p=folder('dedup-seeded');flag(p,'not_earned');contest(p,12)
(p/'attention.json').write_text(json.dumps({'schema_version':'bts_warn_attention_state_v1','sources':{
    'contest_state':{'last_seen':'2027-07-14','streak':29,'last_message':'old saver warning'}}}))
(p/'dm.json').write_text(json.dumps({'status':'sent','sent_day':'2027-07-15','sent_sources':['WARN_ATTENTION:contest_state']}))
aa=check(p,expected=True,now=NOW)
_,attention=apply_warn_attention_policy(aa,state_path=p/'attention.json',today=NOW.date())
sent=[]
with patch.object(alert,'send_dm',side_effect=lambda who,body:sent.append(body)):
    attempted=alert.dispatch_dm_for_health_alerts(aa,'captured.test',attention,status_path=p/'dm.json',now_et_date=NOW.date())
rows.append({'family':'dedup-seeded','attempted':attempted,'sent':sent,'store':json.loads((p/'attention.json').read_text())})

# Explicit known contract limitation: an unreported burn followed by a streak reset.
p=folder('consumed-reset');flag(p,'not_earned');contest(p,0)
rows.append({'family':'consumed-reset','state':call(lambda:cs.load_decision_streak_state(p,now=NOW)),
             'health':[dataclasses.asdict(a) for a in check(p,expected=True,now=NOW)]})

out=ROOT/(package+'-edges.json');out.write_text(json.dumps(rows,default=str,indent=2))
print(package,len(rows),'edge outcomes',out)
for row in rows:
    if row['family'] in ['race','model-existing-pick','existing-saver-loader','consumed-reset']:print(row)
