"""Every streak 0..57, file state, expected value and observation case."""
import dataclasses
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import shlex
import sys
from click.testing import CliRunner
ROOT=Path(__file__).resolve().parent; REPO=ROOT.parents[1]
package=sys.argv[1]; sys.path.insert(0,str(ROOT/'base/src' if package=='base' else REPO/'src'))
from bts.health.contest_state import check
from bts.contest_state import load_decision_streak_state,DecisionStreakState
from bts.saver_state import load_saver_state
from bts.daily_decision import write_decision
from bts.cli import cli
NOW=datetime(2027,7,15,16,tzinfo=timezone.utc)
states=['missing','permissions','non_utf8','stale_not_earned','stale_active','stale_used','not_earned','active','used']
work=ROOT/'evidence'/('health-exhaustive-'+package);work.mkdir(exist_ok=True)
p=work/'data/picks';account=p/'account_state';account.mkdir(parents=True,exist_ok=True)
flag=account/'saver_state.json';observation=account/'contest_streak.json'
rows=[];pasted=0;errors=[]
for st,n,expected,obs in itertools.product(states,range(58),[False,True],['none','undated','prior','current']):
    if flag.exists():flag.chmod(0o600);flag.unlink()
    state='uninitialized'
    if st!='missing':
        body={'state':'active' if st in ['permissions','non_utf8'] else st.removeprefix('stale_'),
              'season':2026 if st.startswith('stale_') else 2027}
        flag.write_text(json.dumps(body))
        if st=='permissions':flag.chmod(0)
        elif st=='non_utf8':flag.write_bytes(b'\xff')
        if st in ['not_earned','active','used']:state=st
    source={'undated':None,'prior':'2026-07-14','current':'2027-07-14'}.get(obs)
    if obs=='none':
        if observation.exists():observation.unlink()
    else:observation.write_text(json.dumps({'active_streak':n,'best_streak':n,'source_date':source}))
    try:
        alerts=check(p,expected=expected,now=NOW)
        warn=obs!='none' and ((state=='uninitialized' and (expected or 10<=n<=15)) or
                                         (state=='not_earned' and 10<=n<=15))
        saver=[a for a in alerts if 'Streak Saver' in a.message]
        if package=='branch':
            assert len(saver)==int(warn),(st,n,expected,obs,alerts)
            for a in saver:assert a.source=='saver_state' and a.level=='WARN'
        row={'key':[st,n,expected,obs],'alerts':[dataclasses.asdict(a) for a in alerts]}
        # Paste each emitted message's exact command. Only add the authorized scratch path.
        for a in saver:
            command=a.message.split('`')[1]
            result=CliRunner().invoke(cli,[*shlex.split(command)[1:],'--picks-dir',str(p)])
            row['paste']={'command':command,'exit':result.exit_code,'output':result.output,
                          'state':load_saver_state(p,season=2027).state}
            pasted+=1
            if package=='branch':assert result.exit_code==0 and row['paste']['state']=='used'
        rows.append(row)
    finally:
        if st=='permissions' and flag.exists():flag.chmod(0o600)
# Schema stays untouched; mapped boolean persists; raw flag never enters decision.json.
flag.write_text('{"state":"not_earned","season":2027}')
observation.write_text('{"active_streak":7,"best_streak":7,"source_date":"2027-07-14"}')
state=load_decision_streak_state(p,now=NOW)
rec=write_decision('2027-07-15',p,action='single',source='mdp',streak=state.streak,
    saver_available=state.saver_available,delivery_status='pending',scoreable=False,now=NOW)
assert rec is not None and 'saver_flag_state' not in rec
assert rec['saver_available']==(package=='branch')
if package=='branch':
    default=DecisionStreakState(0,False,True,'model','model_only',0,True)
    assert default.saver_flag_state is None
out=ROOT/(package+'-health-exhaustive.json');out.write_text(json.dumps(rows,default=str,separators=(',',':')))
print(package,'health cases',len(rows),'commands pasted',pasted,'schema fields',sorted(rec))
