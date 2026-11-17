"""A conditional-command contract edge: last year's high streak, this year's unearned saver."""
import datetime as dt
import json
from pathlib import Path
import re
import shlex
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo
from click.testing import CliRunner
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2];sys.path.insert(0,str(REPO/'src'))
from bts.cli import cli
from bts.health.contest_state import check
from bts.saver_state import load_saver_state
from bts.contest_state import load_decision_streak_state
NOW=dt.datetime(2027,3,24,12,tzinfo=ZoneInfo('America/New_York'))
class Clock(dt.datetime):
    @classmethod
    def now(cls,tz=None):return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)
p=ROOT/'evidence/stale-warning/data/picks';(p/'account_state').mkdir(parents=True,exist_ok=True)
(p/'account_state/contest_streak.json').write_text('{"active_streak":12,"best_streak":12,"source_date":"2026-09-28"}')
(p/'account_state/saver_state.json').write_text('{"state":"used","season":2026}')
first=check(p,expected=True,now=NOW);assert len(first)==1
commands=re.findall(r'`(bts saver-state[^`]+)`',first[0].message)
assert len(commands)==3
result=CliRunner().invoke(cli,[*shlex.split(commands[0])[1:],'--picks-dir',str(p)])
assert result.exit_code==0 and load_saver_state(p,season=2027).state=='not_earned'
second=check(p,expected=True,now=NOW);assert len(second)==1
commands2=re.findall(r'`(bts saver-state[^`]+)`',second[0].message)
assert len(commands2)==2 and all('not_earned' not in c for c in commands2)
with patch.object(dt,'datetime',Clock):
    repair=CliRunner().invoke(cli,['set-contest-streak','--streak','0','--best-streak','0',
                                 '--source-date','2027-03-24','--picks-dir',str(p)])
assert repair.exit_code==0 and check(p,expected=True,now=NOW)==[]
state=load_decision_streak_state(p,now=NOW);assert state.streak==0 and state.saver_available is True
out={'truth':'2027 saver not yet earned; current streak 0','first':first[0].message,'pasted':commands[0],
     'after_init':second[0].message,'remaining_choices':commands2,'correct_contest_repair':repair.output,
     'after_repair':{'flag':state.saver_flag_state,'streak':state.streak,'saver':state.saver_available}}
(ROOT/'stale-warning.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
