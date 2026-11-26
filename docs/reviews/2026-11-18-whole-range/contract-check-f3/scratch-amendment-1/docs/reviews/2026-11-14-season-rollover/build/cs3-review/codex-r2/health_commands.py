"""All health states, all streaks, and every conditional command independently pasted."""
import dataclasses
from datetime import datetime,timezone
import itertools
import json
from pathlib import Path
import shlex
import sys
from click.testing import CliRunner
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2]
package=sys.argv[1];sys.path.insert(0,str(ROOT/package/'src' if package!='branch' else REPO/'src'))
from bts.health.contest_state import check
from bts.saver_state import load_saver_state
from bts.cli import cli
NOW=datetime(2027,7,15,16,tzinfo=timezone.utc)
states=['missing','permissions','non_utf8','non_object','stale_not_earned','stale_active','stale_used','not_earned','active','used']
work=ROOT/'evidence'/('health-'+package);work.mkdir(exist_ok=True)
rows=[];pasted=0
def setup(p,st,n,obs):
    a=p/'account_state';a.mkdir(parents=True,exist_ok=True);f=a/'saver_state.json'
    if f.exists():f.chmod(0o600);f.unlink()
    if st!='missing':
        body={'state':'active' if st in ['permissions','non_utf8','non_object'] else st.removeprefix('stale_'),
              'season':2026 if st.startswith('stale_') else 2027}
        f.write_text(json.dumps(body))
        if st=='permissions':f.chmod(0)
        elif st=='non_utf8':f.write_bytes(b'\xff')
        elif st=='non_object':f.write_text('[]')
    c=a/'contest_streak.json'
    if obs=='none':
        if c.exists():c.unlink()
    else:c.write_text(json.dumps({'active_streak':n,'best_streak':n,'source_date':{'undated':None,'prior':'2026-07-14','current':'2027-07-14'}[obs]}))
    return f
for st,n,expected,obs in itertools.product(states,range(58),[False,True],['none','undated','prior','current']):
    p=work/'read';f=setup(p,st,n,obs)
    try:alerts=check(p,expected=expected,now=NOW)
    finally:
        if st=='permissions' and f.exists():f.chmod(0o600)
    saver=[a for a in alerts if 'Streak Saver' in a.message]
    state=st if st in ['not_earned','active','used'] else 'uninitialized'
    warn=obs!='none' and ((state=='uninitialized' and (expected or 10<=n<=15)) or (state=='not_earned' and 10<=n<=15))
    if package!='base':assert len(saver)==int(warn)
    row={'key':[st,n,expected,obs],'alerts':[dataclasses.asdict(a) for a in alerts],'pastes':[]}
    for a in saver:
        if package!='base':
            assert a.source=='saver_state' and a.level=='WARN'
            assert 'season 2027' in a.message and str(n) in a.message and 'counting the saver as spent' in a.message
        commands=a.message.split('`')[1::2]
        if package=='branch':assert len(commands)==(3 if state=='uninitialized' else 2)
        for i,command in enumerate(commands):
            cp=work/('paste-'+str(pasted));ff=setup(cp,st,n,obs)
            result=CliRunner().invoke(cli,[*shlex.split(command)[1:],'--picks-dir',str(cp)])
            actual=load_saver_state(cp,season=2027).state
            row['pastes'].append({'command':command,'exit':result.exit_code,'state':actual,'output':result.output})
            if package=='branch':
                target=['not_earned','active','used'][i] if state=='uninitialized' else ['active','used'][i]
                assert result.exit_code==0 and actual==target,(row,target)
            if st=='permissions' and ff.exists():ff.chmod(0o600)
            pasted+=1
    rows.append(row)
out=ROOT/(package+'-health.json');out.write_text(json.dumps(rows,default=str,separators=(',',':')))
print(package,'health rows',len(rows),'commands',pasted,flush=True)
