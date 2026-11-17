"""Verify inherited notes and identify fixture greens with real test functions."""
import datetime as dt
import importlib.util
import json
from pathlib import Path
import runpy
import socket
import subprocess
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo
from click.testing import CliRunner
import pytest
ROOT=Path(__file__).resolve().parent;REPO=ROOT.parents[2];package=sys.argv[1]
sys.path.insert(0,str(ROOT/package/'src' if package!='branch' else REPO/'src'));sys.path.insert(1,str(REPO))
import bts.saver_state as ss
import bts.contest_state as cs
import bts.cli as cm
import bts.web as web
socket.socket.connect=lambda *a,**k:(_ for _ in ()).throw(AssertionError('network forbidden'))
ET=ZoneInfo('America/New_York');NOW=dt.datetime(2027,7,15,12,tzinfo=ET)
def clock(at):
    class Clock(dt.datetime):
        @classmethod
        def now(cls,tz=None):return at.astimezone(tz) if tz else at.replace(tzinfo=None)
    return Clock
work=ROOT/'evidence'/('notes-'+package);work.mkdir(exist_ok=True);rows=[]
def folder(name):
    p=work/name;(p/'account_state').mkdir(parents=True,exist_ok=True);return p
def cli(p,*args):
    r=CliRunner().invoke(cm.cli,['saver-state','--picks-dir',str(p),*args])
    path=p/'account_state/saver_state.json'
    return {'exit':r.exit_code,'output':r.output,'file':json.loads(path.read_text()) if path.exists() else None}
p=folder('season-zero')
with patch.object(cm,'datetime',clock(NOW)):r=cli(p,'--season','0','--init','active')
rows.append({'family':'season-zero','result':r})
p=folder('flags-together');rows.append({'family':'flags-together','result':cli(p,'--season','2027','--init','active','--use')})
p=folder('init-race');real=ss.transition_saver_state;calls=[]
def race(*a,**kw):
    if not calls:real(*a,expected_prior='uninitialized',new_state='used',season=2027,source='concurrent')
    calls.append(kw);return real(*a,**kw)
with patch.object(ss,'transition_saver_state',side_effect=race):r=cli(p,'--season','2027','--init','active')
rows.append({'family':'init-race','result':r})

p=folder('midnight')
(p/'streak.json').write_text('{"season":2026,"streak":12}')
(p/'account_state/saver_state.json').write_text('{"season":2027,"state":"not_earned"}')
before=dt.datetime(2026,12,31,23,59,59,tzinfo=ET);after=dt.datetime(2027,1,1,0,0,1,tzinfo=ET);calls=[]
class SteppedClock(dt.datetime):
    @classmethod
    def now(cls,tz=None):
        at=before if not calls else after;calls.append(at)
        return at.astimezone(tz) if tz else at.replace(tzinfo=None)
with patch.object(web,'PICKS_DIR',p),patch.object(web,'fetch_bluesky_posts',return_value=[]), \
     patch.object(cs,'datetime',SteppedClock),patch.object(web,'datetime',SteppedClock):
    page=web.render_page()
rows.append({'family':'midnight-render','clocks':calls,'section_present':'Streak Saver:' in page,
             'reserved':'reserved (not yet earned)' in page,'old_label':'Streak Saver: not yet earned' in page,
             'new_year_streak':cs.load_decision_streak_state(p,now=after).streak})

if package=='branch':
    old=ROOT/'fallback_round1.py'
    old.write_text(subprocess.run(['git','show','0768a6bd:tests/test_decision_saver_fallback.py'],check=True,capture_output=True,text=True).stdout)
    for label,path in [('round1',old),('current',REPO/'tests/test_decision_saver_fallback.py')]:
        mod=runpy.run_path(str(path))
        for name,fn in list(mod.items()):
            if not name.startswith('test_'):continue
            p=folder(label+'-'+name);fn(p)
            diag=cs._load_model_diagnostics(p,2026)
            rows.append({'family':'fixture-tag','key':[label,name],'diagnostics':diag,'passed':True})
    fetch=runpy.run_path(str(REPO/'tests/test_contest_fetch.py'))
    manual=runpy.run_path(str(REPO/'tests/test_cli_integration.py'))
    for year in [2026,2027,2028]:
        for name in ['test_fetch_cli_persists_current_activestreak_despite_lag','test_fetch_cli_persists_per_round_ledger']:
            p=folder('drift-'+str(year)+'-'+name);calls=[];real=ss.maybe_auto_earn_saver
            with patch.object(dt,'datetime',clock(NOW.replace(year=year))),patch.object(ss,'maybe_auto_earn_saver',side_effect=lambda *a,**k:calls.append(k) or real(*a,**k)):
                mp=pytest.MonkeyPatch()
                try:fetch[name](p,mp)
                finally:mp.undo()
            rows.append({'family':'fixture-year-drift','key':[year,name],'hook_calls':calls,'passed':True})
        p=folder('drift-'+str(year)+'-manual');calls=[];real=ss.maybe_auto_earn_saver
        with patch.object(dt,'datetime',clock(NOW.replace(year=year))),patch.object(ss,'maybe_auto_earn_saver',side_effect=lambda *a,**k:calls.append(k) or real(*a,**k)):
            manual['TestSetContestStreak']().test_set_contest_streak_writes_manual_state(p)
        rows.append({'family':'fixture-year-drift','key':[year,'test_set_contest_streak_writes_manual_state'],'hook_calls':calls,'passed':True})

out=ROOT/(package+'-notes.json');out.write_text(json.dumps(rows,default=str,indent=2))
for row in rows:print(package,row)
