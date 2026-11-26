"""Additional offline fetch/clock observations; no future implementation."""
import sys
sys.dont_write_bytecode=True
def deny(event,args):
    if event in {'socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr','socket.connect','socket.connect_ex','socket.sendto'}:
        raise RuntimeError('network forbidden: '+event)
sys.addaudithook(deny)
import contextlib
import datetime as dt
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from zoneinfo import ZoneInfo
from click.testing import CliRunner
import bts.cli as cli
import bts.contest_fetch as cf
import bts.leaderboard.auth as auth
import bts.saver_state as saver
ROOT=Path(__file__).resolve().parent
ET=ZoneInfo('America/New_York')
times=[dt.datetime(2027,12,31,23,59,59,tzinfo=ET),dt.datetime(2028,1,1,0,0,1,tzinfo=ET)]
class Tick(dt.datetime):
    calls=[]
    @classmethod
    def now(cls,tz=None):
        x=times[min(len(cls.calls),len(times)-1)]
        cls.calls.append(x.isoformat())
        return x.astimezone(tz) if tz else x.replace(tzinfo=None)
out={}
for name,profile,rounds in [
    ('missing_predictions',dict(activeStreak=7,seasonBestStreak=9),{1:dt.date(2027,12,31)}),
    ('empty_rounds',dict(activeStreak=7,seasonBestStreak=9,predictions=[dict(roundId=1,result='hit')]),{}),
    ('year_tick',dict(activeStreak=7,seasonBestStreak=9,predictions=[dict(roundId=1,result='hit')]),{1:dt.date(2027,12,31)})]:
    p=ROOT/'edges-cases'/name/'picks'
    p.mkdir(parents=True,exist_ok=True)
    Tick.calls=[];earn=[]
    with contextlib.ExitStack() as s:
        for q in [patch.object(dt,'datetime',Tick),patch.object(auth,'load_session_cookies',lambda:{}),
            patch.object(auth,'extract_uid',lambda c:'u'),patch.object(auth,'fetch_login_session',lambda **kw:NS(user_id=1,username='offline',xsid='x')),
            patch.object(cf,'fetch_profile',lambda *a:profile),patch.object(cli,'_fetch_rounds',lambda:rounds),
            patch.object(saver,'maybe_auto_earn_saver',lambda *a,**kw:earn.append(kw))]:s.enter_context(q)
        r=CliRunner().invoke(cli.cli,['fetch-contest-streak','--picks-dir',str(p)])
    out[name]=dict(exit=r.exit_code,output=r.output,clock_calls=Tick.calls,saver_calls=earn,
        observation=json.loads((p/'account_state/contest_streak.json').read_text()))
(ROOT/'edges-results.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
