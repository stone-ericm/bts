"""Real cascade, policy, scheduler, DM serialization and operator sequences; no sockets."""
import datetime as dt
from contextlib import ExitStack
import io
import json
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from matrix_probe import HERE,ROOT,NOW,ET,obs,changed,put,clean
sys.path.insert(0,str(ROOT/'src'))
from command_probe import clock,invoke
# command_probe installs a permanent audit hook rejecting DNS and connect.
import bts.contest_state as cs
import bts.health.contest_state as health
import bts.health.attention as attention
import bts.health.alert as alertmod
import bts.orchestrator as orch
import bts.scheduler as sch
import bts.strategy as strategy
import bts.picks as picksmod
import bts.dm as dm
import bts.cli as climod
from bts.cli import cli
from bts.daily_decision import load_decision
from click.testing import CliRunner

def sequences():
    p=HERE/'operator_files'; state=p/'account_state'; rows=[]
    put(state/'contest_streak.json',dict(active_streak=22,best_streak=30,source_date='2026-09-30'))
    put(p/'streak.json',dict(streak=44,season=2026,saver_available=True))
    put(state/'saver_state.json',dict(state='used',season=2026))
    def record(label,now):
        d=cs.load_decision_streak_state(p,now=now,require_contest_state=True)
        rows.append(dict(step=label,decision=clean(d),alerts=clean(health.check(p,now=now,expected=True))))
        return d
    def command(now,**kw):
        with clock(now): r=invoke(p,day=now.astimezone(ET).date().isoformat(),**kw)
        assert r.exit_code==0,r.output
    opening=dt.datetime(2027,3,24,20,tzinfo=ET)
    command(opening,streak=0,best=0)
    assert record('opening night',opening).saver_available
    assert json.loads((state/'saver_state.json').read_text())==dict(state='used',season=2026)
    nextmorning=opening+dt.timedelta(hours=12)
    d=record('next morning without observation',nextmorning)
    assert (d.streak,d.saver_flag_state,d.saver_available)==(22,'uninitialized',False)
    command(nextmorning,streak=1,best=1); record('first morning observation',nextmorning)
    command(nextmorning,streak=2,best=2); assert record('first correction',nextmorning).streak==2
    command(nextmorning,streak=3,best=3); assert record('second correction',nextmorning).streak==3
    with clock(nextmorning):
        r=CliRunner().invoke(cli,['set-contest-streak','--picks-dir',str(p),'--streak','4','--best-streak','4'])
    assert r.exit_code==0,r.output
    assert record('legacy replaces dated observation',nextmorning).streak==4
    assert json.loads((state/'contest_streak.manual.json').read_text())['schema_version'].endswith('_v2')
    command(nextmorning,streak=10,best=10)
    d=record('10 not earned warns',nextmorning); assert not d.saver_available
    command(nextmorning,streak=10,best=10,saver='active')
    assert record('10 corrected to active',nextmorning).saver_available
    missed=nextmorning+dt.timedelta(days=1)
    command(missed,streak=10,best=10,saver='used')
    d=record('forgiven miss entered next day as used',missed)
    assert d.streak==10 and not d.saver_available
    # Run the real fetch writer and auto-earn with fake authentication/profile/round transports.
    put(state/'saver_state.json',dict(state='not_earned',season=2027))
    import bts.contest_fetch as cf
    import bts.leaderboard.auth as auth
    with clock(missed),ExitStack() as stack:
        stack.enter_context(patch.object(auth,'load_session_cookies',return_value={'oktaid':'fixture'}))
        stack.enter_context(patch.object(auth,'extract_uid',return_value='fixture'))
        stack.enter_context(patch.object(auth,'fetch_login_session',return_value=SimpleNamespace(xsid='x',user_id=123,username='fixture')))
        stack.enter_context(patch.object(cf,'fetch_profile',return_value=dict(activeStreak=11,seasonBestStreak=11,predictions=[dict(roundId=1,result='hit')])) )
        stack.enter_context(patch.object(climod,'_fetch_rounds',return_value={1:missed.date()}))
        r=CliRunner().invoke(cli,['fetch-contest-streak','--picks-dir',str(p),'--expected-username','fixture'])
    assert r.exit_code==0,r.output
    assert json.loads((state/'saver_state.json').read_text())['state']=='active'
    d=record('fetch auto earns flag while observation says used',missed)
    assert d.streak==10 and d.saver_flag_state=='used' and not d.saver_available
    (HERE/'operator_sequences.json').write_text(json.dumps(rows,indent=2))
    print('OPERATOR',[(r['step'],r['decision']['streak'],r['decision']['saver_flag_state']) for r in rows],flush=True)

def delivery():
    result=[]; outgoing=[]
    # Real send_dm, including its auth/resolve/conversation/send request serialization.
    def transport(req,**kw):
        url=req.full_url
        if 'createSession' in url: body={'accessJwt':'fixture-jwt'}
        elif 'resolveHandle' in url: body={'did':'did:plc:fixture'}
        elif 'getConvoForMembers' in url: body={'convo':{'id':'fixture-convo'}}
        elif 'sendMessage' in url:
            payload=json.loads(req.data); outgoing.append(payload['message']['text']); body={'id':'fixture-message'}
        else: raise AssertionError('unexpected URL '+url)
        return io.BytesIO(json.dumps(body).encode())
    with patch.object(dm,'get_dm_password',return_value='fixture-password'),patch.object(dm,'retry_urlopen',side_effect=transport):
        for offset in (0,1):
            day=(NOW+dt.timedelta(days=offset)).date().isoformat(); now=NOW+dt.timedelta(days=offset)
            p=HERE/f'delivery_{offset}'; state=p/'account_state'
            put(state/'contest_streak.manual.json',changed(active_streak=7,best_streak=9,saver_state='not_earned'))
            put(state/'contest_streak.json',dict(active_streak=2,best_streak=4,source_date='2027-07-15',schema_version='bts_contest_streak_auto_v1',source='mlb_bts_profile'))
            put(state/'saver_state.json',dict(state='used',season=2027)); put(p/'streak.json',b'{')
            slate=[dict(batter_name='Fixture A',batter_id=1,team='NYM',lineup=1,pitcher_name='P A',pitcher_id=3,
                        game_pk=100,game_time=day+'T23:10:00Z',p_hit_pa=.4,p_game_hit=.90,flags=''),
                   dict(batter_name='Fixture B',batter_id=2,team='NYY',lineup=2,pitcher_name='P B',pitcher_id=4,
                        game_pk=200,game_time=day+'T23:10:00Z',p_hit_pa=.4,p_game_hit=.89,flags='')]
            cfg={'orchestrator':{'picks_dir':str(p),'heartbeat_path':str(p/'.heartbeat')},
                 'tiers':[dict(name='captured',ssh_host='fixture',bts_dir='/fixture',timeout_min=1)],
                 'bluesky':{'dm_recipient':'fixture.test'},
                 'scheduler':{'pick_delivery':'dm','contest_state_required':True,'live_forward_capture_on_lock':False}}
            statuses={100:dict(abstract='P',detailed='Pre-Game'),200:dict(abstract='P',detailed='Pre-Game')}
            reads=[]; orig=Path.open
            def opened(path,mode='r',*a,**kw):
                if 'r' in mode and str(path).startswith(str(p)): reads.append(str(path.relative_to(p)))
                return orig(path,mode,*a,**kw)
            def subprocess_transport(args,**kw):
                assert args[0]=='ssh',args
                return subprocess.CompletedProcess(args,0,json.dumps(slate),'')
            with clock(now) as c,patch.object(strategy,'datetime',c),patch.object(sch,'_now_et',return_value=now),\
                 patch.object(orch.subprocess,'run',side_effect=subprocess_transport),\
                 patch.object(picksmod,'get_game_statuses_detailed',return_value=statuses),patch.object(Path,'open',opened):
                _,sel,tier=orch.run_and_pick(cfg,day)
                assert sel is not None and sel.pick_result is not None
                daily=sel.pick_result.daily
                scheduler_state=sch.SchedulerState(day,'fixture',[],[],[],False,None,None,None)
                assert sch._deliver_and_lock_pick(daily,cfg,p,scheduler_state,day,'review',selection=sel)
            record=load_decision(day,p); pick=picksmod.load_pick(day,p)
            wanted=(7,True) if offset==0 else (2,False)
            assert (record['streak'],record['saver_available'])==wanted,record
            assert record['action']==sel.action and record['state_source']=='contest'
            assert record['contest_source_date']=='2027-07-15'
            assert pick.notification_sent and record['delivery_status']=='delivered'
            if offset==0: assert 'account_state/saver_state.json' not in reads,reads
            else: assert 'account_state/saver_state.json' in reads,reads
            result.append(dict(day=day,selection=clean(sel),decision=record,reads=reads,dm=outgoing[-1]))
        # Warning corrections copied from the real health message, including quoted directory names.
        p=HERE/'health picks with spaces'; put(p/'account_state/contest_streak.manual.json',obs())
        messages=health.check(p,now=NOW,expected=True)
        warning=next(a for a in messages if a.source=='saver_state').message
        commands=warning.split('`')[1::2]; correction_results=[]
        for i,command in enumerate(commands):
            if i==2: command=command.replace('--streak N','--streak 9').replace('--best-streak B','--best-streak 9')
            with clock(NOW): r=CliRunner().invoke(cli,shlex.split(command)[1:])
            assert r.exit_code==0,r.output
            d=cs.load_decision_streak_state(p,now=NOW)
            assert not any(a.source=='saver_state' for a in health.check(p,now=NOW,expected=True))
            assert d.saver_flag_state==('active','used','not_earned')[i]
            correction_results.append(dict(command=command,decision=clean(d)))
        # Actual health attention and dispatch chain, today and next day without an observation.
        put(p/'account_state/contest_streak.manual.json',obs()); put(p/'account_state/contest_streak.json',dict(active_streak=12,source_date='2027-07-15'))
        put(p/'account_state/saver_state.json',dict(state='not_earned',season=2027))
        health_dms=[]
        for offset in (0,1):
            now=NOW+dt.timedelta(days=offset)
            with clock(now):
                alerts=health.check(p,now=now,expected=True)
                critical,attention_warn=attention.apply_warn_attention_policy(alerts,state_path=p/'attention.json',today=now.date())
                sent=alertmod.dispatch_dm_for_health_alerts(alerts+critical,'fixture.test',warn_attention=attention_warn,
                    status_path=p/'health_dm_status.json',now_et_date=now.date())
            assert sent
            health_dms.append(dict(day=str(now.date()),body=outgoing[-1],characters=len(outgoing[-1])))
        assert 'dated observation' in health_dms[0]['body'] and 'bts set-contest-streak --date' in health_dms[0]['body']
        assert 'flag not_earned' in health_dms[1]['body'] and 'bts saver-state' in health_dms[1]['body']
    (HERE/'delivery_summary.json').write_text(json.dumps(dict(deliveries=result,corrections=correction_results,health_dms=health_dms),default=str,indent=2))
    print('DELIVERY',[(r['day'],r['decision']['streak'],r['decision']['saver_available'],r['decision']['action']) for r in result],flush=True)
    print('HEALTH DMS',health_dms,flush=True)

if __name__=='__main__': sequences(); delivery()
