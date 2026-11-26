"""Attack exact-str totality and the new rejected-date error construction.

No pytest; only scratch files are written. Parser faults are explicitly synthetic,
not evidence of a date-controlled exception in the helper.
"""
import datetime as dt
import json
from pathlib import Path
import signal
import socket
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

H=Path(__file__).resolve().parent;R=H.parents[1]
sys.path.insert(0,str(R/'src'));sys.path.insert(0,str(R))
import bts.picks as p
socket.socket.connect=lambda *a,**kw: (_ for _ in ()).throw(RuntimeError('no network'))
socket.socket.connect_ex=socket.socket.connect
events=[]
out=[]
def call(fn):
    try:
        value=fn()
        return {'return':value,'return_type':type(value).__name__}
    except BaseException as error:
        return {'error':type(error).__name__,'message':str(error)}

class NameTrap(type):
    def __getattribute__(cls,name):
        if name=='__name__':
            events.append('metaclass.__getattribute__(__name__)')
            raise RuntimeError('type-name hook ran')
        return type.__getattribute__(cls,name)
class BadDate(metaclass=NameTrap):pass
class BadName(str):
    def __format__(self,spec):
        events.append('type-name string.__format__')
        raise SystemExit('type-name format hook ran')
class NameStringTrap(type):
    def __getattribute__(cls,name):
        if name=='__name__':return BadName('Date')
        return type.__getattribute__(cls,name)
class BadNameDate(metaclass=NameStringTrap):pass
class NameSideEffect(type):
    def __getattribute__(cls,name):
        if name=='__name__':
            events.append('metaclass writes sentinel')
            (H/'r2-metaclass-IO-sentinel.txt').write_text('called during rejected date formatting')
            raise RuntimeError('type-name hook performed IO')
        return type.__getattribute__(cls,name)
class SideEffectDate(metaclass=NameSideEffect):pass
class NameSleeps(type):
    def __getattribute__(cls,name):
        if name=='__name__':
            import time
            time.sleep(60)
        return type.__getattribute__(cls,name)
class SleepingDate(metaclass=NameSleeps):pass

if len(sys.argv)>1 and sys.argv[1]=='sleep-child':
    print(call(lambda:p.season_of_date(SleepingDate())),flush=True)
    p.update_streak([True],H/'r2-metaclass-sleep',date=SleepingDate())
    raise SystemExit()
if len(sys.argv)>1 and sys.argv[1]=='signal-child':
    def interrupted(*args):raise KeyboardInterrupt('external signal')
    signal.signal(signal.SIGALRM,interrupted)
    signal.setitimer(signal.ITIMER_REAL,.01)
    try:
        for _ in range(100000):p.season_of_date('junk'*10000)
    except KeyboardInterrupt as exc:print(type(exc).__name__,str(exc),flush=True)
    finally:signal.setitimer(signal.ITIMER_REAL,0)
    raise SystemExit()

for label,value in [('name-lookup',BadDate()),('name-format',BadNameDate()),('name-IO',SideEffectDate())]:
    d=H/f'r2-metaclass-{label}';d.mkdir(exist_ok=True)
    (d/'streak.json').write_text('{"season":2027,"streak":5}')
    before=(d/'streak.json').read_bytes()
    events.clear()
    helper=call(lambda:p.season_of_date(value))
    assert helper=={'return':None,'return_type':'NoneType'} and events==[]
    reads=[]
    def forbidden(*a,**k):
        reads.append('state IO');raise AssertionError('state IO happened')
    with patch.object(p,'load_streak',side_effect=forbidden),patch.object(p,'load_saver_available',side_effect=forbidden), \
         patch.object(p,'load_applied_dates',side_effect=forbidden),patch.object(p,'save_streak',side_effect=forbidden):
        update=call(lambda:p.update_streak([True],d,date=value))
    after=(d/'streak.json').read_bytes()
    out.append(dict(case=label,helper=helper,update=update,events=list(events),state_reads=reads,
                    before=before.hex(),after=after.hex()))
    assert before==after and reads==[]

try:
    r=subprocess.run([sys.executable,__file__,'sleep-child'],capture_output=True,text=True,timeout=2)
    out.append(dict(case='name-stall',exit=r.returncode,stdout=r.stdout,stderr=r.stderr))
except subprocess.TimeoutExpired as exc:
    out.append(dict(case='name-stall',outcome='own child killed after 2 seconds',stdout=str(exc.stdout)))
r=subprocess.run([sys.executable,__file__,'signal-child'],capture_output=True,text=True,timeout=5)
out.append(dict(case='external-signal',exit=r.returncode,stdout=r.stdout,stderr=r.stderr))

for error_type in (MemoryError,RecursionError,ValueError):
    parser_calls=[]
    def fail_iso(value):parser_calls.append('iso');raise error_type('synthetic parser fault')
    def fail_strptime(value,format):parser_calls.append('strptime');raise error_type('synthetic parser fault')
    with patch.object(p,'date_type',SimpleNamespace(fromisoformat=fail_iso)), \
         patch.object(p,'datetime',SimpleNamespace(strptime=fail_strptime)):
        value=call(lambda:p.season_of_date('2027-03-24'))
    assert value=={'return':None,'return_type':'NoneType'} and parser_calls==['iso','strptime']
    out.append(dict(case='synthetic-'+error_type.__name__,outcome=value,calls=parser_calls))

# Independently confirm interpreter behavior behind the accepted Unicode rule.
for text in ('٢٠٢٧-03-24','2027-03-2٤','２０２７-03-24','2027-03-٢٤','٢٠٢٧-٠٣-٢٤'):
    out.append(dict(case='unicode',text=text,iso=call(lambda:dt.date.fromisoformat(text).year),
                    strptime=call(lambda:dt.datetime.strptime(text,'%Y-%m-%d').year),
                    helper=call(lambda:p.season_of_date(text))))
(H/'r2-attacks.json').write_text(json.dumps(out,indent=2))
for row in out:print(json.dumps({k:v for k,v in row.items() if k not in ('before','after')},ensure_ascii=True))
