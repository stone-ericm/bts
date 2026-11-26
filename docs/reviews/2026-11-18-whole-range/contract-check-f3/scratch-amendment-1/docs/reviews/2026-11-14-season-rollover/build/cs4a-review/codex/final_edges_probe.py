"""Advance external wall time during replacement; check check-1 priority against every property."""
import datetime as dt
import json
import os
from pathlib import Path
import sys
from unittest.mock import patch
from matrix_probe import HERE,ROOT,NOW,ET,obs,changed,put
sys.path.insert(0,str(ROOT/'src'))
import pandas
from click.testing import CliRunner
import click
from bts.cli import cli
import bts.contest_state as cs
from bts.health.contest_state import check
p=HERE/'midnight_files'; path=p/'account_state/contest_streak.manual.json'
real_datetime=dt.datetime
start=real_datetime(2027,12,31,23,59,59,900000,tzinfo=ET)
finish=real_datetime(2028,1,1,0,0,0,100000,tzinfo=ET)
wall=start; prints=[]; real_replace=os.replace; real_echo=click.echo
class Clock(real_datetime):
    calls=0
    @classmethod
    def now(cls,tz=None):
        cls.calls+=1
        return wall.astimezone(tz) if tz else wall.replace(tzinfo=None)
def replace(a,b):
    global wall
    real_replace(a,b); wall=finish
def echo(*a,**kw):
    prints.append(str(wall)); return real_echo(*a,**kw)
with patch.object(dt,'datetime',Clock),patch.object(cs,'datetime',Clock),patch.object(os,'replace',replace),patch.object(click,'echo',echo):
    r=CliRunner().invoke(cli,['set-contest-streak','--picks-dir',str(p),'--date','2027-12-31',
        '--streak','12','--best-streak','14','--saver','used'])
assert r.exit_code==0,r.output
assert Clock.calls==1 and prints==[str(finish),str(finish)]
assert r.output.splitlines()[-1]=='contest_streak: 12 (best 14, saver used) for 2027-12-31, in effect today'
body=json.loads(path.read_text())
assert body['recorded_at']=='2028-01-01T04:59:59.900000Z' and body['season']==2027
assert cs.load_decision_streak_state(p,now=start).saver_flag_state=='used'
assert cs.load_decision_streak_state(p,now=finish).source=='model'
midnight=dict(start=str(start),finish=str(finish),print_wall_times=prints,clock_reads=Clock.calls,
              output=r.output,observation=body)
p=HERE/'check1_files'; path=p/'account_state/contest_streak.manual.json'
auto=p/'account_state/contest_streak.json'; put(auto,dict(active_streak=3,source_date='2027-07-15'))
original=Path.open
def opened(path,*a,**kw):
    if path==auto: raise AssertionError('automatic read before rejecting invalid manual')
    return original(path,*a,**kw)
count=0
for field,bad in dict(effective_date=None,season=False,expires_at='naive',active_streak=True,
                     best_streak=0,saver_state=[],recorded_at='naive',source=None,reason=None).items():
    data=changed(**{field:bad}); raw=json.dumps(data).encode()
    for kind in ('permission','utf8','json','nonobject'):
        put(path,raw if kind=='permission' else raw+b'\xff' if kind=='utf8' else raw[:-1] if kind=='json' else [data])
        if kind=='permission': path.chmod(0)
        try:
            with patch.object(Path,'open',opened):
                try: cs.load_contest_streak_state(p,now=NOW)
                except cs.ContestStateError as e:
                    assert str(path) in str(e) and 'malformed' in str(e),str(e)
                else: raise AssertionError('invalid file was accepted')
                alerts=check(p,now=NOW,expected=True)
                assert len(alerts)==1 and alerts[0].level=='CRITICAL'
        finally: path.chmod(0o600)
        count+=1
(HERE/'final_edges_summary.json').write_text(json.dumps(dict(midnight=midnight,check1_pairs=count),indent=2))
print('MIDNIGHT actual stepped replacement/printing passed;',count,'check-1/property pairs passed.')
