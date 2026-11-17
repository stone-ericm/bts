"""CLI compatibility, write failures, interleavings, clocks and operator sequences."""
import builtins
from contextlib import contextmanager, ExitStack
import datetime as dt
import itertools as it
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

from matrix_probe import HERE,ROOT,NOW,ET,obs,changed,put,clean

def deny_network(event,args):
    if event in {'socket.connect','socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr'}:
        raise AssertionError('network forbidden: '+event)
sys.addaudithook(deny_network)
REAL_DT=dt.datetime

@contextmanager
def clock(*times):
    import bts.contest_state as cs
    import bts.health.contest_state as health
    import bts.saver_state as flags
    class Clock(REAL_DT):
        calls=0
        @classmethod
        def now(cls,tz=None):
            v=times[min(cls.calls,len(times)-1)]; cls.calls+=1
            return v.astimezone(tz) if tz else v.replace(tzinfo=None)
    with ExitStack() as stack:
        for module in (dt,cs,health,flags): stack.enter_context(patch.object(module,'datetime',Clock))
        yield Clock

def snapshot(p):
    return {str(f.relative_to(p)):f.read_bytes().hex() for f in p.rglob('*') if f.is_file()}

def invoke(p,*extra,day='2027-07-15',streak=12,best=14,saver='not_earned'):
    from click.testing import CliRunner
    from bts.cli import cli
    return CliRunner().invoke(cli,['set-contest-streak','--picks-dir',str(p),'--date',day,
        '--streak',str(streak),'--best-streak',str(best),'--saver',saver,*extra])

def legacy_worker(package,label):
    sys.path.insert(0,str(package))
    from click.testing import CliRunner
    from bts.cli import cli
    p=HERE/'legacy_files'; p.mkdir(exist_ok=True)
    options=[[],['--best-streak','0'],['--best-streak','12'],['--best-streak','58'],
       ['--source-date','2026-09-30'],['--source-date','20270715'],['--source',''],['--source','custom'],
       ['--username',''],['--username','name'],['--ttl-hours','0'],['--ttl-hours','-1'],['--ttl-hours','2.5'],
       ['--override-expires-at','2027-07-16T00:00:00'],['--override-expires-at','2027-07-16T01:00:00+03:00'],
       ['--reason',''],['--reason','a\nb'],['--saver-available'],['--saver-unavailable'],
       ['--best-streak','57','--source','custom','--username','name','--ttl-hours','24','--reason','combined','--saver-available']]
    results={}
    for streak,extra in it.product((0,7,10,57,58),options):
        for f in p.rglob('*'):
            if f.is_file(): f.unlink()
        with clock(NOW):
            r=CliRunner().invoke(cli,['set-contest-streak','--picks-dir',str(p),'--streak',str(streak),*extra])
        results[str((streak,extra))]=dict(code=r.exit_code,output=r.output,files=snapshot(p))
    (HERE/f'legacy_{label}.json').write_text(json.dumps(results,indent=1)); print('LEGACY',label,len(results),flush=True)

def new_commands():
    import bts.contest_state as cs
    import bts.health.contest_state as health
    import bts.util as util
    from bts.cli import cli
    from click.testing import CliRunner
    p=HERE/'command_files'; p.mkdir(exist_ok=True)
    path=p/'account_state/contest_streak.manual.json'; log=path.parent/'contest_manual_requests.jsonl'
    events=[]; orig_po=Path.open; orig_bo=builtins.open; orig_replace=os.replace; orig_mkdir=Path.mkdir
    def relevant(v):
        return isinstance(v,(str,Path)) and str(v).startswith(str(p))
    def po(path,mode='r',*args,**kw):
        if relevant(path): events.append(('path.open',str(path),mode))
        return orig_po(path,mode,*args,**kw)
    def bo(path,mode='r',*args,**kw):
        if relevant(path): events.append(('open',str(path),mode))
        return orig_bo(path,mode,*args,**kw)
    def replace(a,b):
        if relevant(b): events.append(('replace',str(b)))
        return orig_replace(a,b)
    def mkdir(path,*a,**kw):
        if relevant(path): events.append(('mkdir',str(path)))
        return orig_mkdir(path,*a,**kw)
    def run(args): return CliRunner().invoke(cli,['set-contest-streak','--picks-dir',str(p),*args])
    valid=['--date','2027-07-15','--streak','12','--best-streak','14','--saver','not_earned']
    rejects=[(['--streak','1','--saver','active'],'--saver'),
             (['--date','2027-07-15','--streak','1','--saver','active'],'--best-streak'),
             (['--date','2027-07-15','--streak','1','--best-streak','1'],'--saver')]
    for extra,hint in [(['--source-date','2027-07-15'],'--source-date'),(['--source','manual_cli'],'--source'),
        (['--username',''],'--username'),(['--ttl-hours','24'],'--ttl-hours'),
        (['--override-expires-at','2027-07-16T00:00:00Z'],'--override-expires-at'),
        (['--saver-available'],'--saver'),(['--saver-unavailable'],'--saver'),
        (['--date','2027-07-14'],'--date'),(['--date','2027-07-16'],'--date'),
        (['--streak','-1'],'--streak'),(['--best-streak','11'],'--best-streak'),
        (['--best-streak','58'],'--best-streak')]: rejects.append((valid+extra,hint))
    for date,best,saver,reason in it.product((False,True),repeat=4):
        args=['--streak','12']
        if date: args+=['--date','2027-07-15']
        if best: args+=['--best-streak','14']
        if saver: args+=['--saver','active']
        if reason: args+=['--reason','']
        if saver and not date: rejects.append((args,'--saver'))
        elif date and not best: rejects.append((args,'--best-streak'))
        elif date and not saver: rejects.append((args,'--saver'))
    rejection_results=[]
    for present in (False,True):
        put(path,obs() if present else None)
        for args,hint in rejects:
            before=snapshot(p); events.clear()
            with clock(NOW),patch.object(Path,'open',po),patch.object(builtins,'open',bo),patch.object(os,'replace',replace),patch.object(Path,'mkdir',mkdir): r=run(args)
            assert r.exit_code==2 and hint in r.output,(args,r.output)
            assert events==[],events
            assert snapshot(p)==before
            rejection_results.append(dict(args=args,exit=r.exit_code,output=r.output,events=events.copy()))
    put(path,None)
    with clock(NOW),patch.object(Path,'open',po),patch.object(builtins,'open',bo),patch.object(os,'replace',replace):
        events.clear(); r=invoke(p,'--reason','checked')
        assert r.exit_code==0,r.output
        trace=events.copy(); before=snapshot(p); events.clear(); r2=invoke(p,'--reason','checked')
        assert r2.exit_code==0 and 'unchanged' in r2.output and snapshot(p)==before
        assert all(e[0]=='path.open' and e[2].startswith('r') for e in events),events
    assert [e[0] for e in trace if e[0] in ('open','replace')]==['open','replace'],trace
    failures=[]
    for kind in ('open','write','flush','close','replace','readonly','directory'):
        put(path,changed(saver_state='active')); put(log,None)
        old=path.read_bytes()
        if kind=='readonly': log.write_text(''); path.parent.chmod(0o555)
        if kind=='directory': put(path,'DIRECTORY')
        atomic_calls=[]
        class Failing:
            def __init__(self): self.file=orig_bo(log,'a',encoding='utf-8')
            def __enter__(self): return self
            def write(self,s):
                self.file.write(s[:9] if kind=='write' else s)
                if kind=='write': raise OSError('write probe')
            def flush(self):
                self.file.flush()
                if kind=='flush': raise OSError('flush probe')
            def __exit__(self,*a):
                self.file.close()
                if kind=='close': raise OSError('close probe')
        def opened(f,mode='r',*a,**kw):
            if Path(f)==log and mode=='a':
                if kind=='open': raise OSError('open probe')
                return Failing()
            return orig_bo(f,mode,*a,**kw)
        def replaced(a,b):
            atomic_calls.append(str(b))
            if kind=='replace': raise OSError('replace probe')
            orig_replace(a,b)
        try:
            with clock(NOW),patch.object(builtins,'open',opened),patch.object(os,'replace',replaced): r=invoke(p,saver='used')
            assert r.exit_code==1,(kind,r.output)
            if kind in ('open','write','flush','close'):
                assert 'not written' in r.output and 'partial' in r.output and not atomic_calls
            else: assert 'NOT saved' in r.output and 'appended' in r.output
            if kind!='directory': assert path.read_bytes()==old
            failures.append(dict(kind=kind,output=r.output,replace_calls=atomic_calls))
        finally: path.parent.chmod(0o755)
        if kind=='directory': path.rmdir()
        with clock(NOW):
            retry=invoke(p,saver='used'); assert retry.exit_code==0,retry.output
            before=snapshot(p); retry2=invoke(p,saver='used'); assert 'unchanged' in retry2.output and snapshot(p)==before
    # Valid snapshots of any provenance are no-ops; all other file kinds are corrections.
    from matrix_probe import fixture_cases
    replacements=[]
    for name,body in fixture_cases()[0].items():
        if name=='directory': continue
        put(path,body)
        with clock(NOW): r=invoke(p)
        assert r.exit_code==0,(name,r.output)
        assert cs.load_decision_streak_state(p,now=NOW).streak==12
        replacements.append((name,'unchanged' in r.output))
    races=[]
    for point in ('before-replace','after-replace'):
        put(path,changed(saver_state='active')); put(log,None); entered=False
        def replacement(a,b):
            nonlocal entered
            def b_call():
                nonlocal entered
                entered=True; result=invoke(p,saver='used'); assert result.exit_code==0,result.output
            if not entered and point=='before-replace': b_call()
            orig_replace(a,b)
            if not entered and point=='after-replace': b_call()
        with clock(NOW),patch.object(os,'replace',replacement): r=invoke(p,saver='not_earned')
        final=json.loads(path.read_text())['saver_state']
        assert final==('not_earned' if point=='before-replace' else 'used')
        assert r.output.splitlines()[-1]=='contest_streak: 12 (best 14, saver not_earned) for 2027-07-15, in effect today'
        races.append(dict(point=point,final=final,log=[json.loads(s)['saver_state'] for s in log.read_text().splitlines()]))
    clock_results=[]
    for day in ('2027-07-15','2027-03-14','2027-11-07','2027-12-31','2028-01-01'):
        date=dt.date.fromisoformat(day); start=REAL_DT.combine(date,dt.time.min,ET); end=REAL_DT.combine(date,dt.time.max,ET)
        put(path,obs(day))
        for now,effect in ((start,True),(end,True),(start-dt.timedelta(microseconds=1),False),(end+dt.timedelta(microseconds=1),False),(start,True)):
            state=cs.load_contest_streak_state(p,now=now); decision=cs.load_decision_streak_state(p,now=now)
            alerts=health.check(p,now=now,expected=True)
            assert (state is not None and state.effective_date is not None)==effect
            if effect:
                assert decision.status=='fresh' and decision.best_status=='trusted'
                assert len(alerts)==1 and alerts[0].source=='saver_state',alerts
            clock_results.append(dict(now=str(now),effect=effect,decision=clean(decision),alerts=clean(alerts)))
    end=REAL_DT(2027,12,31,23,59,59,900000,tzinfo=ET); nextday=end+dt.timedelta(seconds=2)
    counts={}
    for op in ('reader','decision','health','command'):
        put(path,obs('2027-12-31'))
        with clock(end,nextday) as c:
            if op=='reader': assert cs.load_contest_streak_state(p).effective_date==end.date()
            elif op=='decision': assert cs.load_decision_streak_state(p).best_status=='trusted'
            elif op=='health': assert health.check(p,expected=True)[0].source=='saver_state'
            else:
                def after_midnight(a,b):
                    # The independently advanced wall time is available for a second read.
                    orig_replace(a,b)
                with patch.object(os,'replace',after_midnight): r=invoke(p,day='2027-12-31',saver='used')
                assert r.exit_code==0 and 'for 2027-12-31, in effect today' in r.output
                assert json.loads(path.read_text())['recorded_at']=='2028-01-01T04:59:59.900000Z'
            counts[op]=c.calls; assert c.calls==1
    (HERE/'command_summary.json').write_text(json.dumps(dict(rejections=rejection_results,trace=trace,
        failures=failures,replacements=replacements,races=races,clocks=clock_results,clock_reads=counts),indent=2))
    print('COMMAND',len(rejection_results),'rejections;',len(failures),'failure/retries;',len(replacements),'replacements;',counts,flush=True)

if __name__=='__main__':
    if len(sys.argv)>1: legacy_worker(Path(sys.argv[1]),sys.argv[2]); sys.exit()
    for package,label in ((HERE/'base_src/src','base'),(ROOT/'src','branch')):
        subprocess.run([sys.executable,__file__,str(package),label],check=True,cwd=ROOT)
    assert json.loads((HERE/'legacy_base.json').read_text())==json.loads((HERE/'legacy_branch.json').read_text())
    sys.path.insert(0,str(ROOT/'src')); new_commands()

