import sys,os,io,json,copy,tempfile,shutil
from pathlib import Path
from datetime import datetime,timedelta,date
from zoneinfo import ZoneInfo
from contextlib import ExitStack
from unittest.mock import patch
from urllib.parse import urlparse,parse_qs
sys.path[:0]=['/Users/eric/projects/bts-pp-c1','/Users/eric/projects/bts-pp-c1/src']
import numpy as np
import pandas as pd
from tests.model.test_predict_feature_coverage import _synthetic_pa_frame
from tests.conftest import sample_game_feed
import bts.model.predict as m
import bts.data.pull as pull
import bts.picks as picks
import bts.strategy as strategy
import bts.scheduler as sch
import bts.contest_state as contest
import bts.daily_decision as decisions
import bts.orchestrator as orch
from bts.health import streak_validation,projected_lineup,pitcher_sparsity,realized_calibration,result_resolution,blend_training,tail_policy
from click.testing import CliRunner
from bts.cli import cli
REPO=Path('/Users/eric/projects/bts-pp-c1');ROOT=Path(tempfile.mkdtemp(prefix='g2-r2-cold-'));ET=ZoneInfo('America/New_York')
(ROOT/'data/processed').mkdir(parents=True);(ROOT/'data/models').mkdir()
for fn in ['mdp_policy.npz','mdp_tail_policy.npz']:shutil.copyfile(REPO/'data/models'/fn,ROOT/'data/models'/fn)
frame=_synthetic_pa_frame();frame['date']=pd.to_datetime(frame['date'])+pd.DateOffset(years=1,months=5);frame['season']=2026
# Both classes, with a synthetic population hit rate of 40%; not a quality estimate.
frame['is_hit']=((frame.batter_id+frame.pitcher_id+frame.date.dt.day)%5<2).astype(int)
frame['event_type']=np.where(frame.is_hit,'single','strikeout')
from bts.data.build import parse_game_feed
seed=parse_game_feed(sample_game_feed.__wrapped__())[0]
for col,value in seed.items():
    if col not in frame: frame[col]=[copy.deepcopy(value) for _ in range(len(frame))]
from bts.data.schema import PA_COLUMNS
frame=frame[PA_COLUMNS]
frame.to_parquet(ROOT/'data/processed/pa_2026.parquet',index=False)
state={'day':'2027-03-24','posted':False};calls=[];dms=[]
def now():return datetime.fromisoformat(state['day']+'T18:10:00').replace(tzinfo=ET)
class Clock(datetime):
    @classmethod
    def now(cls,tz=None):return now().astimezone(tz) if tz else now().replace(tzinfo=None)
def schedule(day):
    if day not in ['2027-03-24','2027-03-25']:return []
    final=day<state['day'];pk=270324 if day.endswith('24') else 270325
    return [{'gamePk':pk,'gameType':'R','officialDate':day,'gameDate':day+'T23:10:00Z','status':{'abstractGameCode':'F' if final else 'P','detailedState':'Final' if final else 'Scheduled'},'teams':{'away':{'team':{'id':134,'name':'AAA'},'probablePitcher':{'id':200,'fullName':'Away Pitcher'}},'home':{'team':{'id':121,'name':'BBB'},'probablePitcher':{'id':201,'fullName':'Home Pitcher'}}}}]
def feed(pk):
    day='2027-03-24' if pk==270324 else '2027-03-25';final=day<state['day'];posted=final or state['posted']
    f=copy.deepcopy(sample_game_feed.__wrapped__());gd=f['gameData'];gd['game'].update(pk=pk,season='2027');gd['datetime']['officialDate']=day;gd['status']=schedule(day)[0]['status']
    gd['teams']['away']['abbreviation']='AAA';gd['teams']['home']['abbreviation']='BBB'
    if not final:f['liveData']['plays']['allPlays']=[]
    else:
        for i,p in enumerate(f['liveData']['plays']['allPlays']):
            p['matchup']['batter']['id']=100+i;p['matchup']['pitcher']['id']=201-i
    for side,bid in [('away',100),('home',101)]:
        players={}
        if posted:
            for pos,b in [(1,bid),(2,9999+bid)]:
                players['ID'+str(b)]={'person':{'id':b,'fullName':'Batter '+str(b)},'battingOrder':str(pos*100),'stats':{'batting':{'hits':1,'atBats':3}}}
        f['liveData']['boxscore']['teams'][side]={'players':players,'batters':[x['person']['id'] for x in players.values()]}
    return f

def net(url,*a,**k):
    url=getattr(url,'full_url',url);calls.append(url)
    if '/schedule?' in url:
        q=parse_qs(urlparse(url).query)
        games=schedule(q['date'][0]) if 'date' in q else (schedule('2027-03-24') if state['day']>'2027-03-24' else [])
        return io.BytesIO(json.dumps({'dates':[{'date':g['officialDate'],'games':[g]} for g in games]}).encode())
    if '/feed/live' in url:return io.BytesIO(json.dumps(feed(int(url.split('/game/')[1].split('/')[0]))).encode())
    raise AssertionError(url)

def summary(label,pred):
    rec={'label':label,'date':state['day'],'posted':state['posted'],'rows':len(pred),'raw_2027':len(list(Path('data/raw/2027').glob('*.json'))),'pa_2027':Path('data/processed/pa_2027.parquet').exists(),'marker':sorted(p.name for p in Path('data/processed').glob('.refreshed_*')),'blend':sorted(p.name for p in Path('data/models').glob('blend_*'))}
    if len(pred):rec['players']=pred[['batter_id','p_game_hit','days_rest','flags']].to_dict('records')
    print('PIPELINE',json.dumps(rec,default=str),flush=True)

def delivery(label):
    config={'orchestrator':{'picks_dir':'data/picks','models_dir':'data/models','heartbeat_path':'data/.hb'},'tiers':[{'name':'local','type':'local'}],'scheduler':{'pick_delivery':'dm','live_forward_capture_on_lock':False},'bluesky':{'dm_recipient':'fake'}}
    predictions,sel,tier=orch.run_and_pick(config,state['day']); summary(label,predictions)
    info={'label':label,'tier':tier,'selection':None if sel is None else {'action':sel.action,'source':sel.source,'no_pick_reason':sel.no_pick_reason,'decision':vars(sel.decision) if sel.decision else None}}
    if sel and sel.pick_result:
        daily=sel.pick_result.daily;picks.save_pick(daily,Path('data/picks'))
        scheduler_state=sch.SchedulerState(date=state['day'],schedule_fetched_at=now().isoformat(),games=[],confirmed_game_pks=[],runs_completed=[],pick_locked=False,pick_locked_at=None,result_status=None,next_wakeup=None)
        info['delivered']=sch._deliver_and_lock_pick(daily,config,Path('data/picks'),scheduler_state,state['day'],'lineup',selection=sel)
        info['decision_file']=decisions.load_decision(state['day'],Path('data/picks'))
    print('DELIVERY',json.dumps(info,default=str),flush=True)

cwd=os.getcwd();os.chdir(ROOT)
try:
    with ExitStack() as st:
        for mod in [sch,picks,strategy,contest,decisions]:st.enter_context(patch.object(mod,'datetime',Clock))
        for mod,attr in [(m,'urlopen'),(pull,'retry_urlopen'),(pull,'urlopen'),(picks,'retry_urlopen')]:st.enter_context(patch.object(mod,attr,net))
        st.enter_context(patch('bts.dm.send_dm',lambda *a:dms.append(a) or 'dm-fake'))
        st.enter_context(patch('bts.health.alert.send_dm',lambda *a:dms.append(a) or 'dm-fake'))
        print('ROOT',ROOT,flush=True)
        cache=Path('data/models/blend_2027-03-24.pkl')
        pred=m.run_pipeline(state['day'],save_blend_path=cache);summary('opening_unposted_real_training',pred)
        state['posted']=True;delivery('opening_confirmed_cached_blend')
        # The following day has only Opening Night's one regular-season feed.
        state['day']='2027-03-25';state['posted']=False
        cache=Path('data/models/blend_2027-03-25.pkl')
        pred=m.run_pipeline(state['day'],save_blend_path=cache);summary('next_day_prior_lineup_real_training',pred)
        state['posted']=True;delivery('next_day_confirmed_cached_blend')
        model_state=contest.load_decision_streak_state(Path('data/picks'),now=now());print('EMPTY_CONTEST_STATE',vars(model_state),flush=True)
        print('DMS',len(dms),flush=True)
        # Policy: real tracked artifacts; full-season days are clamped by the actual lookup.
        policy=strategy._load_mdp()
        for day in ['2027-03-24','2027-03-25']:
            ctx=strategy.DecisionContext(primary_p=.85,second_p=None,has_diff_game=False,date=day,allow_double=True,mdp=policy)
            print('POLICY',day,'raw_days',strategy._days_remaining(day),'decision',vars(strategy.resolve_policy_decision(ctx,0,False)),flush=True)
        print('CACHE_RELOAD',sorted(m.load_blend(Path('data/models/blend_2027-03-24.pkl'))),flush=True)
        # Direct historical feature and debut feature checks, no new-season requirement.
        featured=m.compute_all_features(frame);lookup=m._build_feature_lookups(featured)
        print('HISTORICAL_FEATURES',{'rows':len(featured),'last_date_100':str(lookup['last_date'].get(100)),'rate_100':lookup['batter']['batter_hr_30g'].get(100),'debut_lookup_missing':10099 not in lookup['last_date']},flush=True)
finally:os.chdir(cwd)
