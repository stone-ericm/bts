"""Freeze the wall clock before test import; keep per-node outcomes and spy inventory."""
import ast
import collections
import datetime as dt
import json
from pathlib import Path
import sys

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
package=Path(sys.argv[1]).resolve()
year=int(sys.argv[2]); label=sys.argv[3]
sys.path.insert(0,str(package))
# pandas captures CPython's datetime types; load it before substituting the wall clock.
import pandas
import pytest
real_datetime=dt.datetime; real_date=dt.date
class FrozenDate(real_date):
    @classmethod
    def today(cls): return cls(year,7,15)
class FrozenDatetime(real_datetime):
    @classmethod
    def now(cls,tz=None):
        value=real_datetime(year,7,15,16,tzinfo=dt.timezone.utc)
        return value.astimezone(tz) if tz else value.replace(tzinfo=None)
    @classmethod
    def utcnow(cls): return cls(year,7,15,16)
    @classmethod
    def today(cls): return cls.now()
dt.datetime=FrozenDatetime; dt.date=FrozenDate
outcomes={}
class Recorder:
    def pytest_runtest_logreport(self,report):
        if report.when=='call' or report.failed:
            outcomes[report.nodeid]=dict(outcome=report.outcome,phase=report.when)

import bts.contest_state as cs
assert str(Path(cs.__file__).resolve()).startswith(str(package)),cs.__file__
code=pytest.main(['tests/test_contest_dated_cs4a.py','-q','-p','no:cacheprovider','--tb=no',
                  '--basetemp',str(HERE/f'pytest_tmp_{label}')],plugins=[Recorder()])
tree=ast.parse((ROOT/'tests/test_contest_dated_cs4a.py').read_text())
inventory={}
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name.startswith('test_'):
        cases={k:v for k,v in outcomes.items() if k.split('::')[-1].split('[')[0]==node.name}
        inventory[node.name]=dict(assertions=sum(isinstance(x,ast.Assert) for x in ast.walk(node)),
            raises=sum(isinstance(x,ast.Call) and isinstance(x.func,ast.Attribute) and x.func.attr=='raises' for x in ast.walk(node)),
            outcomes=dict(collections.Counter(v['outcome'] for v in cases.values())),
            passes=[k for k,v in cases.items() if v['outcome']=='passed'])
(HERE/f'tests_{label}.json').write_text(json.dumps(dict(package=str(package),year=year,exit=code,
    totals=dict(collections.Counter(v['outcome'] for v in outcomes.values())),inventory=inventory),indent=2))
sys.exit(code)
