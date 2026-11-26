"""Independent real-file base/branch matrix, ordered decoding and policy checks."""
import dataclasses
import datetime as dt
import io
import itertools as it
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ET = ZoneInfo('America/New_York')
NOW = dt.datetime(2027, 7, 15, 12, tzinfo=ET)
SCHEMA = 'bts_contest_streak_manual_v3'

def obs(day='2027-07-15', **kw):
    d = dt.date.fromisoformat(day)
    return dict(schema_version=SCHEMA, effective_date=day, season=d.year,
                expires_at=dt.datetime.combine(d, dt.time.max, ET).isoformat(),
                active_streak=12, best_streak=14, saver_state='not_earned',
                source='manual_cli', recorded_at='2027-07-15T16:00:00Z', **kw)

def changed(**kw):
    b=obs(); b.update(kw); return b

def put(p, value):
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        if p.is_dir(): p.rmdir()
        else:
            p.chmod(0o600); p.unlink()
    if value is None: return
    if value == 'DIRECTORY': p.mkdir(); return
    if value == 'UNREADABLE':
        p.write_text(json.dumps(obs())); p.chmod(0); return
    if isinstance(value, bytes): p.write_bytes(value)
    else: p.write_text(json.dumps(value))

def clean(v):
    if dataclasses.is_dataclass(v):
        return {k:clean(x) for k,x in dataclasses.asdict(v).items()
                if k not in ('effective_date','saver_state') or x is not None}
    if isinstance(v, dict): return {k:clean(x) for k,x in v.items()}
    if isinstance(v, (list,tuple)): return [clean(x) for x in v]
    if isinstance(v, (Path,dt.date,dt.datetime)): return str(v)
    return v

def call(fn):
    try: return {'ok':clean(fn())}
    except Exception as e: return {'error':type(e).__name__, 'message':str(e)}

def fixture_cases():
    legacy=dict(active_streak=12,best_streak=14,source='manual_cli',source_date='2027-07-15')
    manuals={'absent':None,
        'legacy_live':dict(legacy,override_expires_at='2099-01-01T00:00:00Z'),
        'legacy_expired':dict(legacy,override_expires_at='2027-07-14T00:00:00Z'),
        'legacy_no_expiry':legacy}
    for name,schema in [('missing','MISSING'),('null',None),('unknown','unknown'),
                        ('v1','bts_contest_streak_manual_v1'),('v2','bts_contest_streak_manual_v2')]:
        b=changed(reason=None,season=False,saver_state={},expires_at=[],override_expires_at='2099-01-01T00:00:00Z')
        if schema=='MISSING': b.pop('schema_version')
        else: b['schema_version']=schema
        manuals['legacy_'+name]=b
    for day,label in [('2027-07-14','yesterday'),('2027-07-15','today'),('2027-07-16','tomorrow')]:
        for ignored in (False,True):
            b=obs(day)
            if ignored: b['override_expires_at']='2099-01-01T00:00:00Z'
            manuals['dated_'+label+('_ignored' if ignored else '')]=b
    bad={2:('effective_date','20270715'),3:('season',True),4:('expires_at','2027-07-15T23:59:59'),
         5:('active_streak',58),6:('best_streak',11),7:('saver_state','uninitialized'),
         8:('recorded_at','2027-07-15'),9:('source',None),10:('reason',None)}
    for n,(k,v) in bad.items(): manuals['invalid_'+str(n)]=changed(**{k:v})
    manuals.update(malformed=b'{', utf8=b'\xff', nonobject=[], directory='DIRECTORY', unreadable='UNREADABLE',
                   huge=b'{"schema_version":"bts_contest_streak_manual_v3","active_streak":'+b'9'*5000+b'}')
    auto={'absent':None,'valid':dict(active_streak=3,best_streak=4,source_date='2027-07-15',
          schema_version='bts_contest_streak_auto_v1',source='mlb_bts_profile'),
          'undated':dict(active_streak=3), 'refused':dict(active_streak=58), 'corrupt':b'{',
          'v3schema':obs()}
    flags={'absent':None,'not_earned':dict(state='not_earned',season=2027),
           'active':dict(state='active',season=2027),'used':dict(state='used',season=2027),
           'other_season':dict(state='active',season=2026),'corrupt':b'{',
           'huge':b'{"season":'+b'9'*5000+b',"state":"active"}'}
    return manuals,auto,flags

def matrix_worker(package):
    sys.path.insert(0,str(package))
    import bts.contest_state as cs
    from bts.health.contest_state import check
    picks=HERE/'matrix_files'; state=picks/'account_state'
    put(picks/'streak.json',dict(streak=2,saver_available=False))
    manuals,autos,flags=fixture_cases(); results={}
    for (mn,m),(an,a),(fn,f) in it.product(manuals.items(),autos.items(),flags.items()):
        put(state/'contest_streak.manual.json',m); put(state/'contest_streak.json',a); put(state/'saver_state.json',f)
        reader=call(lambda:cs.load_contest_streak_state(picks,now=NOW))
        for required,expected in it.product((False,True),repeat=2):
            key='|'.join(map(str,(mn,an,fn,int(required),int(expected))))
            results[key]=dict(reader=reader,
                decision=call(lambda:cs.load_decision_streak_state(picks,now=NOW,require_contest_state=required)),
                health=call(lambda:check(picks,now=NOW,expected=expected)))
    out=HERE/('matrix_'+('base' if 'base_src' in str(package) else 'branch')+'.json')
    out.write_text(json.dumps(results,indent=1)); print(out.name,len(results),flush=True)

def validation_and_policy():
    sys.path.insert(0,str(ROOT/'src'))
    import bts.contest_state as cs
    from bts.health.contest_state import check
    from bts import strategy
    p=HERE/'edge_files'; path=p/'account_state/contest_streak.manual.json'
    failures=[]; count=0
    fields={2:'effective_date',3:'season',4:'expires_at',5:'active_streak',6:'best_streak',
            7:'saver_state',8:'recorded_at',9:'source',10:'reason'}
    values={2:[None,True,1,{},'2027-7-15','20270715','2027-W28-4','2027-07-15T00:00:00','bad'],
            3:[None,True,2027.0,2026,'2027',{},[]],
            4:[None,True,{},'bad','2027-07-15T23:59:59.999999','2027-07-16T03:59:59Z',
               '2027-07-16T03:59:59.999998Z','2027-07-16T04:00:00Z'],
            5:[None,True,1.0,-1,58,'57',{},float('nan')],
            6:[None,True,14.0,11,58,'57',{},float('nan')],
            7:[None,True,{},[],'uninitialized',1],
            8:[None,True,{},'bad','2027-07-15','2027-07-15T00:00:00'],
            9:[None,True,{},[]],10:[None,True,{},[]]}
    invalids=[]
    for n,k in fields.items():
        if n!=10:
            b=obs(); del b[k]; invalids.append((n,'missing',b))
        for v in values[n]: invalids.append((n,repr(v),changed(**{k:v})))
    def require_error(body,n,day=NOW):
        nonlocal count
        put(path,body); got=call(lambda:cs.load_contest_streak_state(p,now=day)); count+=1
        if got.get('error')!='ContestStateError' or f'check {n},' not in got.get('message',''):
            failures.append((n,body,got))
    for n,_,b in invalids:
        for day in (NOW,NOW+dt.timedelta(days=1)): require_error(b,n,day)
    for (n,_,b),(m,_,c) in it.combinations(invalids,2):
        if n==m: continue
        combo=obs()
        for num,body in ((n,b),(m,c)):
            k=fields[num]
            if k in body: combo[k]=body[k]
            else: combo.pop(k,None)
        require_error(combo,min(n,m))
    good=[changed(active_streak=-0,best_streak=57),changed(active_streak=57,best_streak=57),
          changed(source='',reason=''),changed(recorded_at='2027-07-15T12:00:00-04:00'),
          changed(recorded_at='2027-07-15T16:00:00.123456Z'),
          changed(expires_at='2027-07-16T03:59:59.999999Z'),
          changed(expires_at='2027-07-16T09:29:59.999999+05:30'),
          changed(expires_at='2027-07-16T03:59:59.999999+00:00',override_expires_at=[],source_date={},
                  streak='bad',saver_available={'x':1},username=float('nan'))]
    for b in good:
        put(path,b); assert cs.load_contest_streak_state(p,now=NOW).effective_date==NOW.date(); count+=1
    raw=json.dumps(obs())[:-1]+',"active_streak":9}'
    path.write_text(raw); assert cs.load_contest_streak_state(p,now=NOW).streak==9
    print('VALIDATION',count,'cases failures',len(failures),flush=True)
    (HERE/'validation_failures.json').write_text(json.dumps(failures,default=str,indent=2))
    assert not failures
    mdp=strategy._load_mdp()
    assert mdp and mdp['policy_table'] is not None and mdp['tail'] is not None
    _,autos,flags=fixture_cases(); decision_count=policy_count=health_count=0; counts={}
    # Full state/flag/auto/require/model product; each policy probability is evaluated using the loaded state.
    for streak,saver,(fn,f),(an,a),required,model in it.product(range(58),('not_earned','active','used'),
            flags.items(),autos.items(),(False,True),(dict(streak=55,saver_available=True),b'{')):
        put(path,changed(active_streak=streak,best_streak=streak,saver_state=saver))
        put(p/'account_state/saver_state.json',f); put(p/'account_state/contest_streak.json',a)
        put(p/'streak.json',model)
        d=cs.load_decision_streak_state(p,now=NOW,require_contest_state=required)
        wanted=saver=='active' or saver=='not_earned' and streak<10
        assert (d.streak,d.saver_available,d.saver_flag_state,d.status,d.best_status)==(streak,wanted,saver,'fresh','trusted')
        decision_count+=1
        for probability in range(70,91):
            ctx=strategy.DecisionContext(probability/100,(probability-1)/100,True,'2027-07-15',True,mdp,streak,'trusted')
            actual=strategy.resolve_policy_decision(ctx,d.streak,d.saver_available)
            expected=strategy.resolve_policy_decision(ctx,streak,wanted)
            assert actual==expected
            counts[actual.action]=counts.get(actual.action,0)+1; policy_count+=1
    # Tail and early reach-57 dates exercise the checked-in pair, including best=57 terminal stops.
    for day,streak,saver,probability,best in it.product(('2027-03-24','2027-09-27'),range(58),
                  ('not_earned','active','used'),range(70,91),(0,57)):
        now=dt.datetime.fromisoformat(day+'T12:00:00-04:00')
        b=obs(day); b.update(active_streak=streak,best_streak=max(streak,best),saver_state=saver)
        put(path,b); d=cs.load_decision_streak_state(p,now=now)
        ctx=strategy.DecisionContext(probability/100,(probability-1)/100,True,day,True,mdp,b['best_streak'],'trusted')
        assert strategy.resolve_policy_decision(ctx,d.streak,d.saver_available)==strategy.resolve_policy_decision(ctx,streak,saver=='active' or saver=='not_earned' and streak<10)
        policy_count+=1
    examples=[]
    for streak,saver,expected,(an,a) in it.product(range(58),('not_earned','active','used'),(False,True),autos.items()):
        put(path,changed(active_streak=streak,best_streak=streak,saver_state=saver))
        put(p/'account_state/contest_streak.json',a)
        with patch('bts.saver_state.load_saver_state',side_effect=AssertionError('flag read')):
            alerts=check(p,now=NOW,expected=expected)
        wanted=int(saver=='not_earned' and 10<=streak<=15)
        assert len(alerts)==wanted,alerts
        if alerts:
            assert (alerts[0].level,alerts[0].source)==('WARN','saver_state')
            if not examples: examples=clean(alerts)
        health_count+=1
    summary=dict(decisions=decision_count,policy_calls=policy_count,actions=counts,health=health_count,
                 artifact_sha=mdp['base_sha256'],tail_sha=mdp['tail'].sha256,warning=examples)
    (HERE/'policy_summary.json').write_text(json.dumps(summary,indent=2)); print('POLICY/HEALTH',summary,flush=True)

def compare():
    base=json.loads((HERE/'matrix_base.json').read_text()); branch=json.loads((HERE/'matrix_branch.json').read_text())
    differences=[]; problems=[]; groups={}
    for key,b in branch.items():
        mn,an,fn,req,exp=key.split('|'); a=base[key]
        diff=a!=b; group='dated' if mn.startswith('dated_') else 'invalid' if mn.startswith('invalid_') else 'preserved'
        groups.setdefault(group,[0,0]); groups[group][0]+=1; groups[group][1]+=diff
        if diff: differences.append(dict(key=key,components=[k for k in a if a[k]!=b[k]]))
        if group=='preserved' and diff: problems.append(dict(key=key,base=a,branch=b))
        if group=='dated' and not mn.startswith('dated_today'):
            absent='|'.join(('absent',an,fn,req,exp))
            if b!=base[absent]: problems.append(dict(key=key,expected=base[absent],branch=b))
        if group=='invalid':
            n=int(mn.split('_')[1])
            for component in ('reader','decision'):
                if f'check {n},' not in b[component].get('message',''): problems.append(dict(key=key,branch=b))
    summary=dict(total=len(base),groups=groups,differences=differences,unexpected=problems)
    (HERE/'matrix_summary.json').write_text(json.dumps(summary,indent=1))
    print('MATRIX',len(base),'groups [total,different]',groups,'unexpected',len(problems),flush=True)
    assert not problems

if __name__=='__main__':
    if len(sys.argv)>1:
        matrix_worker(Path(sys.argv[1])); sys.exit()
    base=HERE/'base_src'; base.mkdir(exist_ok=True)
    archive=subprocess.check_output(['git','archive','22e3ebf9','src'],cwd=ROOT)
    with tarfile.open(fileobj=io.BytesIO(archive)) as t: t.extractall(base,filter='data')
    for package in (base/'src',ROOT/'src'):
        subprocess.run([sys.executable,__file__,str(package)],cwd=ROOT,check=True)
    compare(); validation_and_policy()
