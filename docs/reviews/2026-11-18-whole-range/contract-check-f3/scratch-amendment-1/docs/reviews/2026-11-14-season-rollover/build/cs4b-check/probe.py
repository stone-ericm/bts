"""Offline contract audit; future behavior is labeled as deductions in the report."""
import sys
sys.dont_write_bytecode = True
def deny_network(event, args):
    if event in {'socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr',
                 'socket.connect', 'socket.connect_ex', 'socket.sendto'}:
        raise RuntimeError('audit: network forbidden: ' + event)
sys.addaudithook(deny_network)

import contextlib
import datetime as dt
import io
import json
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[1] / 'src'))
sys.path.insert(0, str(ROOT.parents[1]))
from click.testing import CliRunner
import bts.cli as cli
import bts.contest_state as cs
import bts.contest_fetch as cf
import bts.saver_state as saver
import bts.leaderboard.auth as auth
from bts.health.contest_state import check
from bts.health import alert

ET = ZoneInfo('America/New_York')
NOW = dt.datetime(2027, 7, 15, 12, tzinfo=ET)
OUT = {}

def directory(name):
    p = ROOT / 'cases-final' / name
    p.mkdir(parents=True, exist_ok=True)
    return p

def write(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(obj if isinstance(obj, bytes) else json.dumps(obj).encode())

def auto(**kw):
    return dict(schema_version='bts_contest_streak_auto_v1', active_streak=7,
                best_streak=9, source='mlb_bts_profile', source_date='2027-07-14',
                recorded_at='2027-07-15T14:00:00Z', user_id=50311,
                username='stonehengee', saver_available=None, **kw)

def dated(day='2027-07-15', **kw):
    d = dt.date.fromisoformat(day)
    v = dict(schema_version=cs.DATED_MANUAL_SCHEMA, active_streak=12, best_streak=14,
             source='manual_cli', effective_date=day, season=d.year,
             expires_at=dt.datetime.combine(d, dt.time.max, ET).isoformat(),
             recorded_at=NOW.isoformat(), saver_state='used')
    v.update(kw)
    return v

def result(fn):
    try:
        x = fn()
        if isinstance(x, list):
            return [dict(level=a.level, source=a.source, message=a.message) for a in x]
        if x is None:
            return None
        return {k: str(getattr(x, k, None)) for k in
                ('streak', 'source', 'schema_version', 'best_status', 'saver_available', 'season')}
    except Exception as e:
        return dict(error=type(e).__name__, message=str(e))

def matrix():
    manuals = dict(absent=None, unexpired=dict(active_streak=4, best_streak=6,
                   schema_version='bts_contest_streak_manual_v2', source_date='2027-07-15',
                   override_expires_at='2027-07-16T23:00:00Z'),
                   expired=dict(active_streak=4, override_expires_at='2027-07-14T01:00:00Z'),
                   no_expiry=dict(active_streak=4), today=dated(), yesterday=dated('2027-07-14'),
                   tomorrow=dated('2027-07-16'), invalid_dated=dated(saver_state='bogus'),
                   corrupt=b'\xff')
    automatic = dict(absent=None, v1=auto(), v2=dict(auto(), schema_version='bts_contest_streak_auto_v2', season=2027),
                     v2_old=dict(auto(), schema_version='bts_contest_streak_auto_v2', season=2026),
                     v2_bad_season=dict(auto(), schema_version='bts_contest_streak_auto_v2', season=True),
                     unknown=dict(auto(), schema_version='garbage'),
                     missing_schema={k:v for k,v in auto().items() if k != 'schema_version'},
                     undated=dict(active_streak=7),
                     timestamp_only=dict(active_streak=7, recorded_at='2027-07-15T14:00:00Z'),
                     refused=dict(active_streak=58), corrupt=b'\xff')
    rows=[]
    for m, mb in manuals.items():
        for a, ab in automatic.items():
            p=directory('matrix/'+m+'-'+a)
            if mb is not None: write(p/'account_state/contest_streak.manual.json',mb)
            if ab is not None: write(p/'account_state/contest_streak.json',ab)
            write(p/'account_state/season_contract.json',b'not json')
            reads=[]
            parse=cs._parse_state_file
            def parsed(path, **kw):
                reads.append([path.name, kw])
                return parse(path, **kw)
            with patch.object(cs,'_parse_state_file',parsed):
                selected=result(lambda:cs.load_contest_streak_state(p,now=NOW))
            rows.append(dict(manual=m,auto=a,selected=selected,selected_reads=reads,
                             live=result(lambda:cs.load_decision_streak_state(p,now=NOW,require_contest_state=True)),
                             optional=result(lambda:cs.load_decision_streak_state(p,now=NOW)),
                             expected=result(lambda:check(p,expected=True,now=NOW)),
                             research=result(lambda:check(p,expected=False,now=NOW))))
    OUT['matrix']=rows

class Frozen(dt.datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

def fetch(name, *, dry=False, prior=None, best=9, fail_at=None):
    p=directory('fetch/'+name)/'picks'
    p.mkdir(parents=True,exist_ok=True)
    if prior is not None: write(p/'account_state/contest_streak.json',prior)
    events=[]; dms=[]
    orig_atomic=cli._atomic_write_json
    orig_open=Path.open
    orig_read=Path.read_text
    orig_earn=saver.maybe_auto_earn_saver
    def atomic(path,obj):
        events.append('write:'+path.name)
        if fail_at == path.name: raise OSError('injected '+fail_at)
        return orig_atomic(path,obj)
    def opened(path,mode='r',*a,**kw):
        if path.name=='contest_ledger.jsonl' and mode=='a':
            events.append('ledger:append')
            if fail_at=='ledger': raise OSError('injected ledger')
        return orig_open(path,mode,*a,**kw)
    def read(path,*a,**kw):
        if path.name=='contest_streak.json': events.append('read:prior')
        return orig_read(path,*a,**kw)
    def earn(*a,**kw):
        events.append('saver')
        if fail_at=='saver': raise OSError('injected saver')
        return orig_earn(*a,**kw)
    def profile(*args):
        events.append('profile')
        return dict(activeStreak=7,seasonBestStreak=best,predictions=[dict(roundId=1,result='hit')])
    with contextlib.ExitStack() as stack:
        patches=[patch.object(dt,'datetime',Frozen),patch.object(auth,'load_session_cookies',lambda:{}),
                 patch.object(auth,'extract_uid',lambda c:'u'),
                 patch.object(auth,'fetch_login_session',lambda **kw:NS(user_id=50311,username='stonehengee',xsid='x')),
                 patch.object(cf,'fetch_profile',profile),patch.object(cli,'_fetch_rounds',lambda:{1:dt.date(2027,7,14)}),
                 patch.object(cli,'_atomic_write_json',atomic),patch.object(Path,'open',opened),
                 patch.object(Path,'read_text',read),patch.object(saver,'maybe_auto_earn_saver',earn),
                 patch('bts.dm.send_dm',lambda *a,**kw:dms.append(a))]
        for q in patches:stack.enter_context(q)
        r=CliRunner().invoke(cli.cli,['fetch-contest-streak','--picks-dir',str(p),'--dm-recipient','offline.test']+(['--dry-run'] if dry else []))
    return dict(exit=r.exit_code,output=r.output,exception=repr(r.exception),events=events,dms=dms,
                files={str(x.relative_to(p.parent)):x.read_bytes().decode('utf-8',errors='replace') for x in p.parent.rglob('*') if x.is_file()})

def consumers():
    from bts.contest_ledger import parse_latest_ledger
    from bts.data.backup import verify_restored_ops_tree
    from scripts.audit.boundary_shadow_census import load_ledger
    from scripts.audit.season_ledger.sources.contest_ledger import parse_contest_ledger
    import bts.web as web
    p=directory('consumers')
    ledger=p/'account_state/contest_ledger.jsonl'
    row=dict(recorded_at='2027-07-15T14:00:00Z',active_streak=7,best_streak=9,source_date='2027-07-14',
             predictions=[dict(roundId=1,result='hit',streak=7,roundPredictions=[])])
    extra=dict(season=2027,switched=False,settled_by_year={'2026':3,'2027':1},settled_unmapped=1,unsettled=2)
    vals=[]
    for additions in ({},extra):
        body=json.dumps(dict(row,**additions))+'\n'
        ledger.parent.mkdir(parents=True,exist_ok=True)
        ledger.write_text(body)
        write(p/'account_state/contest_streak.json',auto())
        write(p/'account_state/saver_state.json',dict(state='active',season=2027))
        write(p/'streak.json',dict(streak=7,season=2027))
        write(p/'2027-07-14/decision.json',dict(action='single',date='2027-07-14'))
        parsed=parse_contest_ledger('picks/account_state/contest_ledger.jsonl',body.encode())
        vals.append(dict(latest=repr(parse_latest_ledger(ledger)),backup=verify_restored_ops_tree(p),
                         census=repr(load_ledger(ledger)),script_rows=len(parsed.rows),
                         script_quarantined=parsed.quarantined,
                         script_raw=json.loads(parsed.rows[0]['record_raw_json']),
                         dashboard=repr(web.saver_dashboard_context(p,now=NOW))))
    OUT['ledger_consumers']=vals

def dispatch():
    sent=[];status=directory('dispatch')/'status.json'
    attempts=[]
    with patch.object(alert,'send_dm',lambda *a:sent.append(a)):
        for day,msg in [(15,'supply observation'),(15,'now fetch'),(16,'supply observation')]:
            attempts.append(alert.dispatch_dm_for_health_alerts([alert.Alert('CRITICAL','contest_state',msg)],'offline.test',
                           status_path=status,now_et_date=dt.date(2027,7,day)))
    OUT['dispatcher']=dict(attempts=attempts,sent=sent,status=json.loads(status.read_text()))

def callers():
    import bts.scheduler as sch
    import bts.orchestrator as orch
    import bts.picks as picks
    from bts.strategy import PickResult
    from contextlib import nullcontext
    p=directory('callers')
    config=dict(orchestrator=dict(picks_dir=str(p),heartbeat_path=str(p/'.hb')),
                scheduler=dict(pick_delivery='dm',contest_state_required=True),
                bluesky=dict(dm_recipient='offline.test'),tiers=[])
    OUT['config_gates']={str(c):[sch._contest_state_required(c),orch._contest_state_required(c)] for c in
            ({},{'scheduler':{'pick_delivery':'dm'}},{'scheduler':{'pick_delivery':'private','contest_state_required':True}},
             {'health_checks':{'contest_state_expected':True}})}
    day='2027-07-15'
    def daily(**kw):
        d=picks.DailyPick(date=day,run_time=NOW.isoformat(),
                pick=picks.Pick(batter_name='Offline',batter_id=1,team='NYM',lineup_position=1,
                    pitcher_name='P',pitcher_id=2,p_game_hit=.8,flags=[],projected_lineup=False,
                    game_pk=999,game_time='2027-07-15T23:05:00Z'),double_down=None,runner_up=None)
        for k,v in kw.items():setattr(d,k,v)
        return d
    def state():
        return sch.SchedulerState(date=day,schedule_fetched_at='t',games=[],confirmed_game_pks=[],runs_completed=[],
             pick_locked=False,pick_locked_at=None,result_status=None,next_wakeup=None)
    err=cs.ContestStateError('injected proposed season refusal')
    alerts=[]; sent=[]; reads=[]
    def refuse(*a,**kw):reads.append(kw);raise err
    with contextlib.ExitStack() as stack:
        for q in [patch.object(cs,'load_decision_streak_state',refuse),patch.object(sch,'_now_et',lambda:NOW),
                  patch.object(sch,'_void_committed_games',lambda *a:{}),
                  patch.object(sch,'_trigger_live_forward_capture_on_lock',lambda *a:None),
                  patch.object(sch,'_alert_contest_state_failure',lambda *a:alerts.append(str(a[-1]))),
                  patch('bts.dm.send_dm',lambda *a:sent.append(a)),patch.object(alert,'send_dm',lambda *a:sent.append(a))]:
            stack.enter_context(q)
        delivery=[]
        for desc,kw,mode in [('undelivered',{},'dm'),('delivered',{'notification_sent':True,'notification_channel':'bluesky_dm','notification_id':'offline-id','delivered_at':NOW.isoformat()},'dm'),
                              ('crash_guard',{'delivery_attempted':True},'dm'),('private',{},'private')]:
            st=state();config['scheduler']['pick_delivery']=mode
            n=len(reads)
            ok=sch._deliver_and_lock_pick(daily(**kw),config,p,st,day,desc)
            delivery.append(dict(path=desc,ok=ok,locked=st.pick_locked,state_reads=len(reads)-n))
        OUT['deliver_refusal']=delivery
        OUT['scheduler_alerts']=alerts
        OUT['transport_calls']=sent
        with patch.object(sch,'count_new_confirmations',lambda *a:0),patch.object(picks,'load_pick',lambda *a:None),\
             patch.object(sch,'heartbeat_watchdog',lambda *a,**kw:nullcontext()),\
             patch.object(orch,'run_and_pick',side_effect=err):
            OUT['run_single_refusal']=repr(sch.run_single_check(day,[],set(),config,.03))
        with patch.object(sch,'heartbeat_watchdog',lambda *a,**kw:nullcontext()),patch.object(sch,'run_and_pick',side_effect=err):
            OUT['fallback_refusal']=result(lambda:sch._refresh_pick_at_fallback_decision(config,day,daily(),.03))
        import pandas as pd
        frame=pd.DataFrame([dict(batter_name='Offline',p_game_hit=.8,team='NYM')])
        with patch.object(orch,'load_config',lambda *a:config),patch.object(orch,'run_and_pick',side_effect=err):
            OUT['orchestrate_initial_refusal']=orch.orchestrate(p/'unused.toml',day)
        sel=NS(pick_result=PickResult(daily=daily(),locked=False))
        with patch.object(orch,'load_config',lambda *a:config),patch.object(orch,'run_and_pick',lambda *a:(frame,sel,'offline')),\
             patch.object(picks,'attach_provenance',lambda *a,**kw:None):
            OUT['orchestrate_second_refusal']=result(lambda:orch.orchestrate(p/'unused.toml',day))
        # Scoring is independent of the decision loader; injected refusal must not run.
        d=daily(notification_sent=True,notification_channel='bluesky_dm',notification_id='offline-id',delivered_at=NOW.isoformat())
        picks.save_pick(d,p)
        # A preceding private delivery wrote decision.json. Remove its scoreable authority by replacing it explicitly.
        write(p/day/'decision.json',dict(action='single',date=day,scoreable=True))
        with patch.object(cli,'_now_et',lambda:NOW),patch.object(picks,'resolve_daily_slot_results',lambda *a:{'primary':'hit'}):
            r=CliRunner().invoke(cli.cli,['check-results','--date',day,'--picks-dir',str(p)])
            OUT['grading']=dict(exit=r.exit_code,output=r.output,exception=repr(r.exception),
                                streak=json.loads((p/'streak.json').read_text()) if (p/'streak.json').exists() else None)

def clocks_and_primitives():
    p=directory('midnight')
    write(p/'account_state/contest_streak.manual.json',dated('2027-12-31'))
    values=[]
    for now in [dt.datetime(2027,12,31,23,59,59,999999,tzinfo=ET),dt.datetime(2028,1,1,tzinfo=ET),dt.datetime(2027,12,31,23,59,59,999999,tzinfo=ET)]:
        values.append(dict(now=now.isoformat(),live=result(lambda:cs.load_decision_streak_state(p,now=now,require_contest_state=True))))
    OUT['midnight']=values
    a=p/'from';b=p/'target';write(a,{'old':1});write(b,{'collision':1})
    a.rename(b)
    OUT['rename_overwrites_existing']=json.loads(b.read_text())

def simplified_sequences():
    p=directory('reenable')
    old=dict(auto(),schema_version='bts_contest_streak_auto_v2',season=2027)
    old['recorded_at']='2027-06-01T14:00:00Z'
    write(p/'account_state/contest_streak.json',old)
    write(p/'account_state/saver_state.json',dict(state='used',season=2027))
    switch=dict(schema_version='bts_season_contract_v1',season=2027,status='automatic',
                switched_at=NOW.isoformat(),revoked_at='2027-07-15T15:00:00Z',note=None)
    write(p/'account_state/season_contract.json',switch)
    parsed=cs._parse_state_file(p/'account_state/contest_streak.json',require_dated=True)
    # Exactly the three stated row-9 acceptance predicates, not a CS4b implementation.
    accepts=(switch['status']=='automatic' and switch['season']==NOW.year and
             parsed.schema_version=='bts_contest_streak_auto_v2' and old['season']==NOW.year)
    OUT['reenable_old_v2']=dict(row9_accepts=accepts,observation_recorded_at=old['recorded_at'],
                              switched_at=switch['switched_at'],real_loader=result(lambda:cs.load_decision_streak_state(p,now=NOW,require_contest_state=True)))
    p=directory('late_fetch')
    write(p/'account_state/saver_state.json',dict(state='used',season=2027))
    # Fetch has captured automatic; revoke now quarantines the spent flag; old fetch resumes.
    (p/'account_state/saver_state.json').rename(p/'account_state/saver_state.revoked-stamp.json')
    saver.maybe_auto_earn_saver(p,best_streak=7,season=2027)
    OUT['late_fetch_after_revoke']=dict(flag=saver.load_saver_state(p,season=2027).state,
        policy_saver_at_7=saver.policy_saver_available(saver.load_saver_state(p,season=2027),7),
        explanation='real helper after quarantine; fetch/revoke scheduling and future gate are deductions')
    # File refresh is stopped, while ledger/status successes continue: no age/mtime check exists.
    p=directory('frozen_auto')
    write(p/'account_state/contest_streak.json',dict(auto(),source_date='2026-09-01'))
    for day in (13,14,15):write(p/f'2027-07-{day}.json',dict(result='hit'))
    OUT['frozen_auto_health']=result(lambda:check(p,expected=True,now=NOW))
    # Before a first switched fetch the account_state directory need not exist.
    p=directory('cold_ledger')
    try:
        with (p/'account_state/contest_ledger.jsonl').open('a') as f:f.write('{}\n')
    except OSError as e:OUT['cold_ledger_append']=dict(error=type(e).__name__,message=str(e))

def main():
    OUT['commands']={name:dict(exit=(r:=CliRunner().invoke(cli.cli,['season-contract',name])).exit_code,output=r.output) for name in ['status','enable','revoke']}
    matrix()
    OUT['fetch']={name:fetch(name,**kw) for name,kw in [
        ('success',{}),('dry_success',dict(dry=True)),('dry_identity',dict(dry=True,prior=dict(auto(),username='other'))),
        ('best58',dict(best=58)),('prior_nonobject',dict(prior=[])),('prior_nonutf8',dict(prior=b'\xff')),
        ('observation_fail',dict(fail_at='contest_streak.json')),('status_fail',dict(fail_at='contest_streak_fetch_status.json')),
        ('ledger_fail',dict(fail_at='ledger')),('saver_fail',dict(fail_at='saver'))]}
    consumers();dispatch();callers();clocks_and_primitives();simplified_sequences()
    (ROOT/'results.json').write_text(json.dumps(OUT,indent=2,default=str)+'\n')
    print(json.dumps(dict(matrix_cells=len(OUT['matrix']),matrix_calls=len(OUT['matrix'])*5,
                         fetch_cases=len(OUT['fetch']),result_file=str(ROOT/'results.json'))))

if __name__=='__main__':
    log=io.StringIO()
    with contextlib.redirect_stderr(log):main()
    (ROOT/'probe.log').write_text(log.getvalue())
