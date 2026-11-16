"""Verify a local function copy can retain type names without executing rejected-object hooks.

This is a suggested-fix experiment. The original production calls and failures are
recorded separately by r2_attack.py. No source tree is edited.
"""
import inspect
import json
from pathlib import Path
import sys
H=Path(__file__).resolve().parent;R=H.parents[1]
sys.path.insert(0,str(R/'src'))
import bts.picks as p
events=[]
class EvilName(str):
    def __format__(self,spec):events.append('name format');raise SystemExit('name format')
class EvilMeta(type):
    def __getattribute__(cls,name):
        if name=='__name__':events.append('name lookup');raise RuntimeError('name lookup')
        return type.__getattribute__(cls,name)
class BadDate(metaclass=EvilMeta):pass
class NameSubclassDate:pass
NameSubclassDate.__name__=EvilName('NameSubclassDate')

original=inspect.getsource(p.update_streak)
fixed=original.replace('f"<{type(date).__name__}>"',
    '"<" + str.__str__(type.__dict__["__name__"].__get__(type(date))) + ">"')
assert fixed!=original
namespace=dict(p.__dict__)
exec(compile(fixed,'<untracked proposed error-message fix>','exec'),namespace)
candidate=namespace['update_streak']
d=H/'r2-fix-candidate';d.mkdir(exist_ok=True)
(d/'streak.json').write_text('{"season":2027,"streak":5}')
before=(d/'streak.json').read_bytes()
results=[]
for value in (BadDate(),NameSubclassDate()):
    events.clear()
    try:candidate([True],d,date=value)
    except ValueError as exc:results.append(dict(error=type(exc).__name__,message=str(exc),events=list(events)))
    else:raise AssertionError('did not reject invalid date')
    assert not events and before==(d/'streak.json').read_bytes()
(H/'r2-minimal-fix.json').write_text(json.dumps(results,indent=2))
print(json.dumps(results))
