"""Independent CS3 review. Run workers with explicit source tree; no network."""
import argparse
import dataclasses
import datetime as dt
import itertools
import json
import logging
import math
import os
from pathlib import Path
import socket
import sys
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
args = argparse.ArgumentParser()
args.add_argument('package', choices=['base', 'branch'])
args.add_argument('family', choices=['matrix', 'policy', 'operators'])
opts = args.parse_args()
sys.path.insert(0, str(ROOT / 'base/src' if opts.package == 'base' else REPO / 'src'))
import bts.contest_state as cs
import bts.saver_state as ss
import bts.web as web
from bts.health.contest_state import check
from click.testing import CliRunner
import bts.cli as cm

assert str(Path(cs.__file__).resolve()).startswith(str(ROOT / 'base' if opts.package == 'base' else REPO / 'src'))
logging.disable(logging.CRITICAL)
socket.socket.connect = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network forbidden'))
ET = ZoneInfo('America/New_York')
NOW = dt.datetime(2027, 7, 15, 12, tzinfo=ET)
STATES = ['missing', 'unreadable', 'stale_not_earned', 'stale_active', 'stale_used', 'not_earned', 'active', 'used']
WORK = ROOT / 'evidence' / (opts.package + '-' + opts.family + '-v2')
WORK.mkdir(parents=True, exist_ok=True)
rows = []

def save_result():
    out = ROOT / (opts.package + '-' + opts.family + '.json')
    out.write_text(json.dumps(rows, default=str, separators=(',', ':')))
    print(opts.package, opts.family, len(rows), out, flush=True)

def folder(name):
    p = WORK / str(name)
    (p / 'account_state').mkdir(parents=True, exist_ok=True)
    return p

def flag(p, state, year=2027):
    path = p / 'account_state/saver_state.json'
    if path.exists():
        path.chmod(0o600)
        path.unlink()
    if state == 'unreadable':
        path.write_bytes(b'\xff')
    elif state != 'missing':
        if state.startswith('stale_'):
            state, year = state[6:], year - 1
        path.write_text(json.dumps({'state': state, 'season': year}))
    return path

def contest(p, n=7, source='2027-07-14', best=None):
    path = p / 'account_state/contest_streak.json'
    path.write_text(json.dumps({'active_streak': n, 'best_streak': n if best is None else best,
        'source_date': source, 'schema_version': 'bts_contest_streak_auto_v1', 'source': 'mlb_bts_profile'}))
    return path

def capture(fn):
    try:
        r = fn()
        return dataclasses.asdict(r) if dataclasses.is_dataclass(r) else r
    except Exception as e:
        return {'error': type(e).__name__, 'message': str(e)}

def cli(p, *argv):
    result = CliRunner().invoke(cm.cli, ['saver-state', '--picks-dir', str(p), '--season', '2027', *argv])
    return {'exit': result.exit_code, 'output': result.output,
            'exception': type(result.exception).__name__ if result.exception else None,
            'state': ss.load_saver_state(p, season=2027).state}

def audits(p):
    path = p / 'account_state/saver_transitions.jsonl'
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []

def clock(at):
    class Clock(dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return at.astimezone(tz) if tz else at.replace(tzinfo=None)
    return Clock

def observe(p, cmd, source, n, best, at=NOW, dry=False):
    import bts.contest_fetch as cf
    import bts.leaderboard.auth as auth
    argv = [cmd, '--picks-dir', str(p)]
    profile = {'activeStreak': n, 'seasonBestStreak': best,
               'predictions': [] if source is None else [{'roundId': 1, 'result': 'hit', 'streak': n}]}
    session = SimpleNamespace(xsid='captured', user_id=50311, username='review-fixture')
    with patch.object(dt, 'datetime', clock(at)), \
         patch.object(auth, 'load_session_cookies', return_value={'oktaid':'captured'}), \
         patch.object(auth, 'extract_uid', return_value='captured'), \
         patch.object(auth, 'fetch_login_session', return_value=session), \
         patch.object(cf, 'fetch_profile', return_value=profile), \
         patch.object(cm, '_fetch_rounds', return_value={} if source is None else {1: source}):
        if cmd == 'set-contest-streak':
            argv += ['--streak', str(n), '--best-streak', str(best)]
            if source is not None:
                argv += ['--source-date', source.isoformat()]
        else:
            argv += ['--expected-username', 'review-fixture']
            if dry:
                argv += ['--dry-run']
        r = CliRunner().invoke(cm.cli, argv)
    return {'exit': r.exit_code, 'output': r.output, 'exception': repr(r.exception) if r.exception else None,
            'flag': dataclasses.asdict(ss.load_saver_state(p, season=at.year))}

def matrix():
    for year, n, st, source, branch in itertools.product([2027,2028], [0,5,9,10,15,16,57], STATES,
            ['none','undated','prior','current'], ['contest','model']):
        p = folder('read')
        f = flag(p, st, year)
        source_date = None if source in ['none','undated'] else f'{year-(source=="prior")}-07-14'
        c = contest(p, n, source_date)
        if source == 'none' or branch == 'model':
            c.unlink()
        (p / 'streak.json').write_text(json.dumps({'streak':n, 'season':year, 'saver_available':False}))
        now = NOW.replace(year=year)
        d = capture(lambda: cs.load_decision_streak_state(p, now=now))
        if isinstance(d,dict):
            d.pop('message',None)
        h = {str(expected):[dataclasses.asdict(a) for a in check(p, expected=expected, now=now)]
             for expected in [False,True]}
        ctx = capture(lambda: web.saver_dashboard_context(p, now=now))
        rows.append({'family':'read','key':[year,n,st,source,branch], 'decision':d, 'health':h, 'dashboard':ctx})
    print('read matrix done',len(rows),flush=True)
    for n, st in itertools.product([0,9,10,15,16,57], ['missing','not_earned','active','used']):
        if hasattr(ss,'policy_saver_available'):
            policy = ss.policy_saver_available(ss.SaverState('uninitialized' if st=='missing' else st,2027),n)
        else:
            policy = ss.SaverState(st,2027).is_available
        rows.append({'family':'mapping','key':[n,st],'value':policy})
    raw_values = [None, '7','12',[],{},True,False,-1,7.5,10.5,0,7,12,57,58]
    for value, st in itertools.product(raw_values,['missing','not_earned','active','used']):
        p = folder('model-types')
        flag(p,st)
        (p / 'streak.json').write_text(json.dumps({'season':2027,'streak':value,'saver_available':False}))
        rows.append({'family':'model-types','key':[value,st],
                     'result':capture(lambda:cs.load_decision_streak_state(p,now=NOW))})
    for i,(st,use,undo,show,init,force) in enumerate(itertools.product(STATES,[False,True],[False,True],
            [False,True],[None,'not_earned','active','used'],[False,True])):
        p = folder('cli-'+str(i)); f=flag(p,st)
        before=f.read_bytes().hex() if f.exists() else None
        argv = (['--use'] if use else [])+(['--undo'] if undo else [])+(['--show'] if show else [])
        argv += ['--init',init] if init else []
        argv += ['--force'] if force else []
        r=cli(p,*argv)
        rows.append({'family':'cli','key':[st,use,undo,show,init,force], 'result':r,
                     'before':before,'after':f.read_bytes().hex() if f.exists() else None,'audits':audits(p)})
    print('cli matrix done',flush=True)
    for i,(cmd,source,best,st) in enumerate(itertools.product(['fetch-contest-streak','set-contest-streak'],
            [None,dt.date(2026,7,14),dt.date(2027,7,14),dt.date(2028,7,14)],[0,7,10,12],STATES)):
        p=folder('observe-'+str(i)); f=flag(p,st)
        before=f.read_bytes().hex() if f.exists() else None
        r=observe(p,cmd,source,0,best)
        rows.append({'family':'observe','key':[cmd,source,best,st],'result':r,
                    'before':before,'after':f.read_bytes().hex() if f.exists() else None,'audits':audits(p)})
    print('observation matrix done',flush=True)
    bads=['missing','unreadable','non_utf8','non_object','refused','directory','malformed']
    for i,(bad,target,st,transition) in enumerate(itertools.product(bads,['auto','manual'],
            ['missing','stale_active','not_earned','active','used'],[('active','used'),('used','active')])):
        p=folder('web-'+str(i)); f=flag(p,st)
        path=contest(p,12,'2026-07-14',16)
        if target=='manual':
            path.rename(path.with_name('contest_streak.manual.json'))
            path=path.with_name('contest_streak.manual.json')
        if bad=='missing': path.unlink()
        elif bad=='unreadable': path.chmod(0)
        elif bad=='directory': path.unlink(); path.mkdir()
        else:
            path.write_bytes({'non_utf8':b'\xff','non_object':b'[]','refused':b'{"active_streak":58,"best_streak":58}',
                             'malformed':b'{'}[bad])
        before=f.read_bytes().hex() if f.exists() else None
        try:
            ctx=capture(lambda:web.saver_dashboard_context(p,now=NOW))
            with patch.object(web,'PICKS_DIR',p),patch.object(web,'datetime',clock(NOW)), \
                 patch.object(web,'fetch_bluesky_posts',return_value=[]):
                page=capture(lambda:'Streak Saver:' in web.render_page())
            post=capture(lambda:web.saver_transition_response(p,expected_prior=transition[0],new_state=transition[1],
                                  same_origin=True,now=NOW,peer_ip='127.0.0.2'))
            rows.append({'family':'web','key':[bad,target,st,transition], 'context':ctx,'page':page,'post':post,
                         'before':before,'after':f.read_bytes().hex() if f.exists() else None,'audits':audits(p)})
        finally:
            if bad=='unreadable': path.chmod(0o600)
    print('web files matrix done',flush=True)

def policy():
    import bts.strategy as strategy
    mdp=strategy._load_mdp()
    assert mdp['tail'] is not None and mdp['tail_error'] is None
    probs=sorted(set([round(.70+.01*i,2) for i in range(21)]+[v for b in mdp['boundaries'] for v in
                     [math.nextafter(b,-math.inf),b,math.nextafter(b,math.inf)]]))
    rows.append({'family':'artifacts','probabilities':probs, 'base_sha':mdp['base_sha256'],
                 'tail_sha':mdp['tail'].sha256,'source':cs.__file__})
    p=folder('policy-files')
    count=0
    for n in range(58):
        for st in ['missing','stale_not_earned','stale_active','stale_used','not_earned','active','used']:
            flag(p,st)
            contest(p,n,'2026-09-28')
            bits=[]
            for day in range(365):
                date=dt.date(2027,1,1)+dt.timedelta(days=day)
                now=dt.datetime.combine(date,dt.time(12),ET)
                state=cs.load_decision_streak_state(p,now=now)
                bits.append('1' if state.saver_available else '0')
            rows.append({'family':'bits','key':[n,st],'bits':''.join(bits)})
        for day in range(365):
            date=(dt.date(2027,1,1)+dt.timedelta(days=day)).isoformat()
            actions=[]
            for bit in [False,True]:
                group=[]
                for prob in probs:
                    ctx=strategy.DecisionContext(prob,prob,True,date,True,mdp,best_streak=n,best_status='trusted')
                    out=strategy.resolve_policy_decision(ctx,n,bit)
                    assert out.degraded_reason is None
                    group.append({'skip':'K','single':'S','double':'D'}[out.action])
                    count+=1
                actions.append(''.join(group))
            rows.append({'family':'actions','key':[date,n],'values':actions})
        if n%10==0: print('policy streak',n,'resolutions',count,flush=True)
    rows.append({'family':'count','value':count})

def operators():
    from bts.health.attention import apply_warn_attention_policy
    import bts.health.alert as alert
    # Fresh season first seven days: fixed prior observation, last-season flag, missing, init.
    for st in ['stale_not_earned','stale_active','stale_used','missing','initialized']:
        p=folder('week-'+st)
        flag(p,'missing' if st=='initialized' else st)
        contest(p,0,'2026-09-28')
        if st=='initialized': cli(p,'--init','not_earned')
        for day in range(7):
            now=dt.datetime(2027,3,24,12,tzinfo=ET)+dt.timedelta(days=day)
            rows.append({'family':'first-week','key':[st,now.date()],
                         'decision':capture(lambda:cs.load_decision_streak_state(p,now=now)),
                         'health':[dataclasses.asdict(a) for a in check(p,expected=True,now=now)]})
    def snapshot(p,label,at=NOW):
        rows.append({'family':'sequence','key':label,'decision':capture(lambda:cs.load_decision_streak_state(p,now=at)),
            'context':capture(lambda:web.saver_dashboard_context(p,now=at)),
            'health':[dataclasses.asdict(a) for a in check(p,expected=True,now=at)],'audits':audits(p)})
        # Try both UI transitions on copies, to retain the main operator's sequence.
        for prior,new in [('active','used'),('used','active')]:
            import shutil
            cp=folder('button-'+str(len(rows)))
            shutil.copytree(p,cp,dirs_exist_ok=True)
            rows.append({'family':'button','key':[label,prior,new], 'result':capture(lambda:web.saver_transition_response(
                cp,expected_prior=prior,new_state=new,same_origin=True,now=at)), 'audits':audits(cp)})
    p=folder('lifecycle'); flag(p,'stale_used'); contest(p,0,'2026-09-28'); snapshot(p,'last-year-used')
    for source,n,best,label in [(None,0,0,'first-undated'),(dt.date(2027,3,24),0,0,'first-settled'),
            (dt.date(2027,3,25),8,8,'climb-8'),(dt.date(2027,3,26),9,9,'climb-9'),
            (dt.date(2027,3,27),10,10,'climb-10'),(dt.date(2027,3,28),11,11,'climb-11')]:
        rows.append({'family':'fetch-sequence','key':label,'result':observe(p,'fetch-contest-streak',source,n,best)})
        snapshot(p,label)
    contest(p,11,'2027-03-29',11)
    ledger={'predictions':[{'roundId':1,'result':'hit','streak':11},{'roundId':2,'result':'not_hit','streak':11}]}
    (p/'account_state/contest_ledger.jsonl').write_text(json.dumps(ledger)+'\n'+json.dumps(ledger)+'\n')
    snapshot(p,'forgiven-miss-before-use')
    rows.append({'family':'use-sequence','key':'forgiven-miss','result':cli(p,'--use')})
    snapshot(p,'after-use')
    observe(p,'fetch-contest-streak',dt.date(2027,3,29),11,11); snapshot(p,'next-fetch-after-use')
    p=folder('between-fetch'); flag(p,'not_earned'); contest(p,9,'2027-03-25'); snapshot(p,'before-earned-consumed')
    contest(p,11,'2027-03-27'); snapshot(p,'earned-consumed-unreported')
    rows.append({'family':'use-sequence','key':'between-fetch','result':cli(p,'--use')})
    observe(p,'fetch-contest-streak',dt.date(2027,3,27),11,11); snapshot(p,'between-fetch-next-fetch')
    p=folder('mistake'); flag(p,'not_earned'); contest(p,5,'2027-07-14')
    rows.append({'family':'mistake','key':'use','result':cli(p,'--use')})
    rows.append({'family':'mistake','key':'undo','result':cli(p,'--undo')})
    for n in [5,10,16]:
        contest(p,n,'2027-07-14'); snapshot(p,'mistake-undo-'+str(n))
    # Exact commands copied from health, default relative data/picks path, captured CLI.
    import shlex
    for st,n in itertools.product(['missing','unreadable','stale_active','not_earned'],[0,10,15,16]):
        p=folder('paste-'+str(len(rows))) / 'data/picks'; (p/'account_state').mkdir(parents=True, exist_ok=True)
        flag(p,st); contest(p,n)
        alerts=check(p,expected=True,now=NOW)
        for a in alerts:
            if 'Streak Saver' not in a.message: continue
            command=a.message.split('`')[1]
            with patch.object(cm,'datetime',clock(NOW)):
                cwd=Path.cwd()
                try:
                    os.chdir(p.parents[1])
                    result=CliRunner().invoke(cm.cli,shlex.split(command)[1:])
                finally: os.chdir(cwd)
            rows.append({'family':'pasted-command','key':[st,n],'command':command,
                         'exit':result.exit_code,'output':result.output,'state':ss.load_saver_state(p,season=2027).state})
    # Real attention/dedup/DM dispatcher; capture outbound transport, never send.
    for scenario in ['missing','used','not_earned','undated','no-observation','coverage']:
        p=folder('attention-'+scenario); flag(p,scenario if scenario in ['used','not_earned'] else 'missing')
        if scenario!='no-observation': contest(p,12 if scenario in ['not_earned','coverage'] else 0,
                                               None if scenario=='undated' else '2026-09-28')
        if scenario=='coverage':
            flag(p,'used')
            for day in ['2027-03-24','2027-03-25']: (p/(day+'.json')).write_text('{"result":"hit"}')
        dm=[]
        for day in range(30):
            now=dt.datetime(2027,3,24,12,tzinfo=ET)+dt.timedelta(days=day)
            aa=check(p,expected=True,now=now)
            critical,attention=apply_warn_attention_policy(aa,state_path=p/'attention.json',today=now.date())
            attempts=[]
            with patch.object(alert,'send_dm',side_effect=lambda recipient,body:dm.append(body)):
                for repeat in range(2):
                    attempts.append(alert.dispatch_dm_for_health_alerts(aa+critical,'captured.test',attention,
                        status_path=p/'dm.json',now_et_date=now.date()))
            if day in [0,1,29]:
                rows.append({'family':'dm','key':[scenario,day+1], 'attempts':attempts,
                             'alerts':[dataclasses.asdict(a) for a in aa], 'attention':[dataclasses.asdict(a) for a in attention],
                             'last_body':dm[-1] if dm else None,'store':json.loads((p/'attention.json').read_text())})
    # POST's clock is frozen to the request; the underlying writer's timestamp may cross midnight.
    for i,(at,year) in enumerate([(dt.datetime(2026,12,31,23,59,tzinfo=ET),2026),
                (dt.datetime(2027,1,1,0,1,tzinfo=ET),2027),
                (dt.datetime(2027,1,1,0,1,tzinfo=ET),2026)]):
        p=folder('midnight-'+str(i)); f=flag(p,'active',year); contest(p,12,'2026-12-31')
        ctx=capture(lambda:web.saver_dashboard_context(p,now=at))
        with patch.object(ss,'datetime',clock(dt.datetime(2027,1,1,0,1,tzinfo=ET))):
            r=capture(lambda:web.saver_transition_response(p,expected_prior='active',new_state='used',same_origin=True,now=at))
        rows.append({'family':'midnight','key':[at,year],'display':ctx,'post':r,
                     'file':json.loads(f.read_text()),'audit':audits(p)})

globals()[opts.family]()
save_result()
