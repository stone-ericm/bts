"""One-directory chronological sequence with genuinely concurrent Opening Night scorers."""
import contextlib
import datetime as dt
import json
from pathlib import Path
import socket
import subprocess
import sys
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo
H=Path(__file__).resolve().parent;R=H.parents[1];mode=sys.argv[1]
sys.path.insert(0,str(H/'base/src' if mode=='base' else R/'src'))
import bts.picks as p
import bts.cli as c
import bts.scheduler as s
from bts.contest_state import load_decision_streak_state
from click.testing import CliRunner
ET=ZoneInfo('America/New_York');NOW=dt.datetime(2027,3,24,23,tzinfo=ET)
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('no network'))
socket.socket.connect_ex=socket.socket.connect
d=H/f'boundary-race-{mode}';d.mkdir(exist_ok=True)
def snap(label,output):
    readings={}
    for year in (None,2026,2027):
        kw={'season':year} if mode!='base' else {}
        readings[str(year)]=[p.load_streak(d,**kw),p.load_saver_available(d,**kw),p.load_applied_dates(d,**kw)]
    readings['decision']=vars(load_decision_streak_state(d,now=NOW))
    return dict(step=label,output=output,files={f.name:f.read_bytes().hex() for f in d.glob('*.json')},readers=readings)
def plant(day,double=False):
    pk=p.Pick(batter_name='Test',batter_id=1,team='NYY',lineup_position=1,pitcher_name='P',pitcher_id=2,
        p_game_hit=.85,flags=[],projected_lineup=False,game_pk=111,game_time=day+'T23:00:00Z')
    p.save_pick(p.DailyPick(date=day,run_time=day+'T18:00:00Z',pick=pk,double_down=pk if double else None,
        runner_up=None,bluesky_posted=True),d)
def cli(day):
    r=CliRunner().invoke(c.cli,['check-results','--date',day,'--picks-dir',str(d)],catch_exceptions=False)
    return {'exit':r.exit_code,'output':r.output}
def resolve(daily,*a,**k):return {'pick':'hit',**({'double_down':'hit'} if daily.double_down else {})}
if len(sys.argv)>2:
    role=sys.argv[2];barrier=d/'barrier'
    def ready(*a,**kw):
        (barrier/role).write_text('ready')
        deadline=time.monotonic()+5
        while len(list(barrier.iterdir()))<2 and time.monotonic()<deadline:time.sleep(.005)
        assert len(list(barrier.iterdir()))==2
        return {'pick':'hit'}
    with patch.object(c,'_now_et',side_effect=lambda:NOW),patch.object(s,'_now_et',side_effect=lambda:NOW), \
         patch.object(p,'resolve_daily_slot_results',side_effect=ready),patch.object(p,'get_game_statuses_detailed',return_value={}), \
         patch.object(s,'poll_game_result',return_value='final'):
        print(json.dumps(cli('2027-03-24') if role=='cli' else {'poll':s.run_result_polling(111,'2027-03-24',d)}))
    raise SystemExit()

steps=[]
class Crash(BaseException):pass
with patch.object(c,'_now_et',side_effect=lambda:NOW),patch.object(p,'resolve_daily_slot_results',side_effect=resolve), \
     patch.object(p,'get_game_statuses_detailed',return_value={}):
    (d/'streak.json').write_text('{"season":2026,"streak":12,"saver_available":true}')
    NOW=dt.datetime(2026,12,31,23,tzinfo=ET);plant('2026-12-31')
    steps.append(snap('last-2026',cli('2026-12-31')))
    NOW=dt.datetime(2027,2,1,12,tzinfo=ET);steps.append(snap('winter','read only'))
    NOW=dt.datetime(2027,3,24,23,tzinfo=ET);plant('2027-03-24');(d/'barrier').mkdir(exist_ok=True)
    for marker in (d/'barrier').iterdir():marker.unlink()
    jobs=[subprocess.Popen([sys.executable,__file__,mode,role],stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True) for role in ('cli','poll')]
    outputs=[]
    for job in jobs:
        try:stdout,stderr=job.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            job.kill();stdout,stderr=job.communicate()
        assert job.returncode==0
        outputs.append(dict(pid=job.pid,stdout=stdout,stderr=stderr))
    steps.append(snap('opening-race',outputs))
    assert p.load_streak(d)==(1 if mode!='base' else 14)
    NOW=dt.datetime(2027,3,25,23,tzinfo=ET);plant('2027-03-25',double=True)
    with patch.object(p,'save_pick',side_effect=Crash('after streak write')):
        try:cli('2027-03-25')
        except Crash:pass
    steps.append(snap('day-two-crash','crashed'))
    with patch.object(p,'resolve_daily_slot_results',return_value=None):
        p.reconcile_results(d,clock=lambda:NOW)
    steps.append(snap('incomplete-today-replay','no rollback'))
    steps.append(snap('restart',cli('2027-03-25')))
    NOW=dt.datetime(2027,3,26,2,tzinfo=ET)
    p.reconcile_results(d,clock=lambda:NOW)
    steps.append(snap('02-reconcile','replayed'))
    assert p.load_streak(d)==3
(H/f'boundary-race-{mode}.json').write_text(json.dumps(steps,default=str,indent=2))
print(mode,'boundary race and crash/restart/replay assertions passed')
