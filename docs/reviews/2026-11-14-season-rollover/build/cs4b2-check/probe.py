"""CS4b-2 contract audit. Hypothetical results are labelled spec, never implementation."""
import sys
sys.dont_write_bytecode = True
DENIED = []
def deny_network(event, args):
    if event in {'socket.getaddrinfo', 'socket.gethostbyname', 'socket.gethostbyaddr',
                 'socket.connect', 'socket.connect_ex', 'socket.sendto'}:
        DENIED.append(event)
        raise RuntimeError('audit: network forbidden: ' + event)
sys.addaudithook(deny_network)

import contextlib
import datetime as dt
import importlib.util
import io
import json
import shlex
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
from zoneinfo import ZoneInfo
import pandas as pd
from click.testing import CliRunner
import bts.cli as cli
import bts.contest_state as cs
import bts.season_contract as sc
import bts.health.contest_state as health
import bts.contest_fetch as cf
import bts.leaderboard.auth as auth
import bts.saver_state as saver
import bts.scheduler as sch
import bts.orchestrator as orch
import bts.picks as picks

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
ET = ZoneInfo('America/New_York')
NOW = dt.datetime(2027, 7, 15, 12, tzinfo=ET)
REAL_DT = dt.datetime
ID = 'a' * 32
OUT = {'note': 'main execution plus explicitly labelled contract deductions'}

def directory(name):
    path = ROOT / 'cases-verified' / name
    path.mkdir(parents=True, exist_ok=True)
    return path

def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(value if isinstance(value, bytes) else json.dumps(value).encode())

def auto(**kw):
    return dict(schema_version=sc.AUTO_SCHEMA, season=2027, activation_id=ID,
                active_streak=7, best_streak=9, source='mlb_bts_profile',
                source_date='2027-07-14', recorded_at='2027-07-15T14:00:00Z',
                user_id=50311, username='stonehengee') | kw

def dated(day='2027-07-15', **kw):
    d = dt.date.fromisoformat(day)
    return dict(schema_version=cs.DATED_MANUAL_SCHEMA, season=d.year,
                effective_date=day, active_streak=12, best_streak=14,
                saver_state='used', source='manual_cli', recorded_at=NOW.isoformat(),
                expires_at=REAL_DT.combine(d, dt.time.max, ET).isoformat()) | kw

def switch(**kw):
    return dict(schema_version=sc.SCHEMA, season=2027, status='automatic',
                activation_id=ID, switched_at=NOW.isoformat(), revoked_at=None, note=None) | kw

def attempt(fn):
    try:
        value = fn()
        if hasattr(value, '__dataclass_fields__'):
            return asdict(value)
        return value
    except Exception as exc:
        return {'error': type(exc).__name__, 'message': str(exc)}

def spec_reason(picks, now, selected, sw=None):
    """Literal rows 1/3, not a proposed implementation or new live decision."""
    if selected is not None and cs.dated_observation_in_effect(selected, now):
        return None
    sw = sc.switch_state(picks, now) if sw is None else sw
    if sw.state == 'corrupt':
        return '1:check-' + str(sw.check)
    if selected is not None and selected.path.name == 'contest_streak.manual.json':
        return '2:legacy'
    if sw.state in ('absent', 'other_season', 'revoked'):
        return '3:' + sw.state
    if selected is None:
        return '4:missing'
    if selected.schema_version != sc.AUTO_SCHEMA:
        return '5:schema=' + repr(selected.schema_version)
    if selected.season != now.astimezone(ET).year:
        return '5:season'
    if selected.activation_id != sw.snapshot['activation_id']:
        return '5:identifier'
    return None

def matrix():
    manuals = {'absent': None, 'unexpired': dict(active_streak=4, source_date='2027-07-15',
                override_expires_at='2027-07-17T00:00:00Z'),
               'expired': dict(active_streak=4, override_expires_at='2027-07-14T00:00:00Z'),
               'no_expiry': dict(active_streak=4), 'today': dated(),
               'yesterday': dated('2027-07-14'), 'tomorrow': dated('2027-07-16'),
               'invalid': dated(saver_state='bogus'), 'unreadable': b'\xff'}
    autos = {'absent': None, 'v1': auto(schema_version='bts_contest_streak_auto_v1'),
             'missing': {k:v for k,v in auto().items() if k != 'schema_version'},
             'null': auto(schema_version=None), 'number': auto(schema_version=17),
             'unknown': auto(schema_version='unknown'), 'covered': auto(),
             'other_id': auto(activation_id='b'*32), 'other_season': auto(season=2026),
             'bad_id': auto(activation_id='bad'), 'bad_season': auto(season=True),
             'refused': auto(active_streak=58, best_streak=58), 'corrupt': b'\xff',
             'undated': auto(source_date=None, recorded_at=None),
             'timestamp_only': auto(source_date=None),
             'future': auto(source_date='2027-07-20')}
    switches = {'absent': None, 'other_season': switch(season=2026),
                'revoked': switch(status='revoked', revoked_at=NOW.isoformat()),
                'automatic': switch()}
    bad = [[], switch(schema_version=None), switch(season=True), switch(status='bad'),
           switch(activation_id='bad'), switch(switched_at=None),
           switch(revoked_at='bad'), switch(note=[])]
    switches.update({f'corrupt{i+1}': val for i, val in enumerate(bad)})
    rows = []
    p = directory('matrix')
    def install(name, value):
        path = p / 'account_state' / name
        path.unlink(missing_ok=True)
        if value is not None:
            write(path, value)
    for mn, mv in manuals.items():
        install('contest_streak.manual.json', mv)
        for an, av in autos.items():
            install('contest_streak.json', av)
            for sn, sv in switches.items():
                install('season_contract.json', sv)
                sw = sc.switch_state(p, NOW)
                selected_live = attempt(lambda: cs.load_contest_streak_state(p, now=NOW, require_dated_auto=True))
                selected_health = attempt(lambda: cs.load_contest_streak_state(p, now=NOW))
                live_selected = (cs.load_contest_streak_state(p, now=NOW, require_dated_auto=True)
                                 if not isinstance(selected_live, dict) or 'error' not in selected_live else None)
                h_selected = (cs.load_contest_streak_state(p, now=NOW)
                              if not isinstance(selected_health, dict) or 'error' not in selected_health else None)
                live_spec = (selected_live if isinstance(selected_live, dict) and 'error' in selected_live
                             else spec_reason(p, NOW, live_selected, sw))
                health_spec = (selected_health if isinstance(selected_health, dict) and 'error' in selected_health
                               else spec_reason(p, NOW, h_selected, sw))
                for model in ('valid', 'corrupt'):
                    write(p/'streak.json', dict(season=2027, streak=99) if model == 'valid' else b'\xff')
                    row = dict(manual=mn, auto=an, switch=sn, model=model,
                               spec_live=live_spec, spec_health=health_spec,
                               selected_path=str(live_selected.path.name) if live_selected else None)
                    for required in (False, True):
                        row['main_live' if required else 'main_optional'] = attempt(
                            lambda: cs.load_decision_streak_state(p, now=NOW, require_contest_state=required))
                    for expected in (False, True):
                        row['main_expected' if expected else 'main_unexpected'] = attempt(
                            lambda: [asdict(a) for a in health.check(p, now=NOW, expected=expected)])
                    rows.append(row)
    OUT['matrix'] = rows
    p = directory('metadata')
    samples = []
    for name, value in [('missing', autos['missing']), ('null', autos['null']), ('number', autos['number'])]:
        write(p/'account_state/contest_streak.json', value)
        samples.append((name, asdict(cs.load_contest_streak_state(p, now=NOW))))
    OUT['schema_information_loss'] = samples

class Clock(REAL_DT):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)

def cli_call(picks, *args):
    r = CliRunner().invoke(cli.cli, [*args, '--picks-dir', str(picks)])
    return dict(exit=r.exit_code, output=r.output, exception=repr(r.exception))

def spec_live(picks, now=NOW):
    state = attempt(lambda: cs.load_contest_streak_state(picks, now=now, require_dated_auto=True))
    if isinstance(state, dict) and 'error' in state:
        return state
    selected = cs.load_contest_streak_state(picks, now=now, require_dated_auto=True)
    return spec_reason(picks, now, selected) or 'accepted'

def messages_and_sequences():
    events, dms = [], []
    def profile(*args):
        events.append('profile')
        return dict(activeStreak=7, seasonBestStreak=9, predictions=[dict(roundId=1,result='hit')])
    def rounds():
        events.append('rounds')
        return {1: dt.date(2027, 7, 14)}
    with contextlib.ExitStack() as stack:
        for p in [patch.object(dt, 'datetime', Clock), patch.object(auth, 'load_session_cookies', lambda:{}),
                  patch.object(auth, 'extract_uid', lambda _:'u'),
                  patch.object(auth, 'fetch_login_session', lambda **_:NS(user_id=50311, username='stonehengee', xsid='x')),
                  patch.object(cf, 'fetch_profile', profile), patch.object(cli, '_fetch_rounds', rounds),
                  patch('bts.dm.send_dm', lambda *a, **kw:dms.append(a))]:
            stack.enter_context(p)
        results = {}
        for reason in ('1', '2', '3', '4', '5'):
            p = directory('messages/reason'+reason+" path 'quoted' & safe") / 'picks'
            if reason != '4': write(p/'account_state/contest_streak.json', auto(schema_version='bts_contest_streak_auto_v1'))
            if reason == '1': write(p/'account_state/season_contract.json', b'corrupt')
            if reason in ('2', '4', '5'): write(p/'account_state/season_contract.json', switch())
            if reason == '2': write(p/'account_state/contest_streak.manual.json', dict(active_streak=4, override_expires_at='2027-07-17T00:00:00Z'))
            chain = [('initial', spec_live(p)), ('status', cli_call(p, 'season-contract', 'status'))]
            if reason in ('1','3'):
                chain += [('enable', cli_call(p, 'season-contract', 'enable', '--season', '2027')),
                          ('after_enable', spec_live(p))]
            if reason != '2':
                chain += [('fetch', cli_call(p, 'fetch-contest-streak', '--dm-recipient', 'offline.test')),
                          ('after_fetch', spec_live(p))]
            chain += [('dated', cli_call(p, 'set-contest-streak', '--date', '2027-07-15',
                                        '--streak', '7', '--best-streak', '9', '--saver', 'not_earned')),
                      ('after_dated', spec_live(p))]
            results[reason] = chain
            # Demonstrate that shlex.quote produces round-trippable commands.
            command = 'bts season-contract status --picks-dir ' + shlex.quote(str(p))
            assert shlex.split(command)[-1] == str(p)
        OUT['message_paths'] = results
        # Opening Night: status has NO fresh endpoint evidence until an unswitched fetch.
        p = directory('opening')/'picks'
        write(p/'account_state/season_contract.json', switch(season=2026))
        write(p/'account_state/contest_streak.json', auto(season=2026, source_date='2026-09-25'))
        OUT['opening'] = [('before', spec_live(p)), ('status', cli_call(p,'season-contract','status')),
                          ('unswitched_fetch',cli_call(p,'fetch-contest-streak')),
                          ('still_refused',spec_live(p)), ('status_with_reading',cli_call(p,'season-contract','status'))]
        # Revoke is unavailable. Write its specified valid tombstone explicitly, labelled.
        p = directory('restore')/'picks'
        write(p/'account_state/season_contract.json', switch())
        write(p/'account_state/contest_streak.json', auto())
        seq = [('old_pair_restored_without_revoke', spec_live(p))]
        write(p/'account_state/season_contract.json', switch(status='revoked', revoked_at=NOW.isoformat()))
        seq += [('specified_tombstone', spec_live(p)), ('enable',cli_call(p,'season-contract','enable','--season','2027')),
                ('old_observation_after_enable',spec_live(p)), ('next_day_without_fetch',spec_live(p,NOW+dt.timedelta(days=1)))]
        atomic = cli._atomic_write_json
        def fail_observation(path, body):
            if path.name == 'contest_streak.json': raise OSError('before replacement')
            return atomic(path, body)
        with patch.object(cli,'_atomic_write_json',fail_observation):
            seq += [('failed_before_replace',cli_call(p,'fetch-contest-streak')),('after',spec_live(p))]
        def fail_status(path, body):
            if path.name == 'contest_streak_fetch_status.json': raise OSError('after replacement')
            return atomic(path, body)
        with patch.object(cli,'_atomic_write_json',fail_status):
            seq += [('failed_after_replace',cli_call(p,'fetch-contest-streak')),('after',spec_live(p))]
        OUT['restore_and_failed_fetch'] = seq
        OUT['fetch_transports'] = events
        OUT['captured_fetch_dms'] = dms

def current_callers():
    # Re-execute the former review's actual caller probes against today's main.
    spec = importlib.util.spec_from_file_location('prior_audit', REPO/'docs/reviews/2026-11-14-season-rollover/build/cs4b-check/probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.ROOT = ROOT/'prior-replayed-verified'
    module.ROOT.mkdir(exist_ok=True)
    module.callers()
    OUT['prior_callers_reexecuted'] = module.OUT
    config = dict(orchestrator=dict(picks_dir=str(directory('callers-extra')), heartbeat_path=str(ROOT/'hb')),
                  scheduler=dict(contest_state_required=True, pick_delivery='public'),
                  health_checks=dict(contest_state_expected=True), bluesky=dict(dm_recipient='offline.test'), tiers=[])
    day = '2027-07-15'
    from bts.strategy import PickResult
    daily = picks.DailyPick(date=day, run_time=NOW.isoformat(), pick=picks.Pick(
        batter_name='Offline', batter_id=1, team='NYM', lineup_position=1,
        pitcher_name='P',pitcher_id=2,p_game_hit=.8,flags=[],projected_lineup=False,
        game_pk=999,game_time='2027-07-15T23:05:00Z'),double_down=None,runner_up=None)
    frame = pd.DataFrame([dict(batter_name='Offline', team='NYM', pitcher_name='P',p_game_hit=.8,
                               p_hit_pa=.3,lineup=1,game_time='2027-07-15T23:05:00Z')])
    reads, posts, dms = [], [], []
    original = cs.load_decision_streak_state
    def injected(picks_dir, **kw):
        reads.append(kw)
        if kw.get('require_contest_state'):
            raise cs.ContestStateError('injected NEW automatic refusal')
        return original(picks_dir, now=NOW, **kw)
    with contextlib.ExitStack() as stack:
        for p in [patch.object(cs,'load_decision_streak_state',injected),
                  patch.object(sch,'_now_et',lambda:NOW), patch.object(cli,'_now_et',lambda:NOW),
                  patch('bts.dm.send_dm',lambda *a,**kw:dms.append(a)),
                  patch('bts.posting.post_to_bluesky',lambda *a:posts.append(a) or 'at://offline'),
                  patch('bts.posting.should_post_now',lambda *a:True),
                  patch.object(orch,'load_config',lambda *a:config),
                  patch.object(picks,'attach_provenance',lambda *a,**kw:None)]:
            stack.enter_context(p)
        with patch.object(orch,'run_cascade',lambda *a:(frame,'offline')), patch('bts.slate.save_slate',lambda *a:None):
            OUT['run_and_pick_injected'] = attempt(lambda:orch.run_and_pick(config,day))
        cases = {}
        for name, result in [('skip',None),('locked',PickResult(daily=daily,locked=True)),('new',PickResult(daily=daily,locked=False))]:
            with patch.object(orch,'run_and_pick',lambda *a:(frame,NS(pick_result=result),'offline')):
                cases[name] = attempt(lambda:orch.orchestrate(ROOT/'unused.toml',day))
        OUT['later_orchestrator_refusals'] = cases
        p = directory('callers-extra')
        st = sch.SchedulerState(date=day,schedule_fetched_at='t',games=[],confirmed_game_pks=[],runs_completed=[],
             pick_locked=False,pick_locked_at=None,result_status=None,next_wakeup=None)
        st.final_skip_candidate = dict(streak=7,saver_available=True,state_source='contest',state_status='fresh')
        sch._write_endofday_skip(p,day,st)
        OUT['eod_skip_under_injection'] = json.loads((p/day/'decision.json').read_text())
        with patch.object(sch,'fetch_schedule',lambda *a:[dict(gamePk=999,status={})]):
            r = CliRunner().invoke(cli.cli,['day-outcome','--date',day,'--config',str(ROOT/'unused.toml'),'--picks-dir',str(p)])
        OUT['day_outcome_under_injection'] = dict(exit=r.exit_code,output=r.output,exception=repr(r.exception))
        import bts.web as web
        with patch.object(web,'PICKS_DIR',p):
            OUT['dashboard_under_injection'] = attempt(web.load_dashboard_decision_state)
        # Real CLI non-dry run can post while every required read refuses.
        import bts.model.predict as model
        import bts.strategy as strategy
        with patch.object(model,'run_pipeline',lambda *a,**kw:frame),\
             patch.object(picks,'get_game_statuses_detailed',lambda *a:{}),\
             patch.object(strategy,'select_pick',lambda *a,**kw:NS(pick_result=PickResult(daily=daily,locked=False))):
            r = CliRunner().invoke(cli.cli,['run','--date',day,'--picks-dir',str(p),
                                          '--models-dir',str(ROOT/'no-model-cache')])
            OUT['bts_run_under_required_only_injection'] = dict(exit=r.exit_code,output=r.output,exception=repr(r.exception))
        OUT['extra_caller_reads'] = reads
        OUT['extra_caller_posts'] = posts
        OUT['extra_caller_dms'] = dms

def clocks_files_and_status():
    p = directory('clocks')
    write(p/'account_state/season_contract.json',switch())
    write(p/'account_state/contest_streak.json',auto())
    instants = [REAL_DT(2027,12,31,23,59,59,999999,tzinfo=ET),REAL_DT(2028,1,1,tzinfo=ET)]
    OUT['year_boundary_automatic'] = [(str(n),spec_live(p,n)) for n in instants]
    write(p/'account_state/contest_streak.manual.json',dated('2027-12-31'))
    OUT['year_boundary_dated'] = [(str(n),spec_live(p,n)) for n in instants]
    write(p/'account_state/season_contract.json',switch(season=2026,activation_id='bad'))
    OUT['malformed_old_switch'] = spec_live(p)
    # Native rename is destructive on collision; dangling targets are present only by lstat.
    p = directory('filesystem')
    write(p/'source',dict(source=True)); write(p/'target',dict(target=True))
    (p/'source').rename(p/'target')
    OUT['path_rename_collision'] = json.loads((p/'target').read_text())
    (p/'dangling').symlink_to(p/'does-not-exist')
    OUT['dangling_target'] = dict(exists=(p/'dangling').exists(),lstat=attempt(lambda:str((p/'dangling').lstat())))
    write(p/'second',dict(second=True)); (p/'second').rename(p/'dangling')
    OUT['path_rename_dangling_collision'] = dict(is_symlink=(p/'dangling').is_symlink(),body=(p/'dangling').read_text())
    p = directory('status-readonly')/'picks'
    write(p/'account_state/contest_streak.json',auto())
    write(p/'account_state/season_contract.json',switch())
    def tree(root):
        return {str(x.relative_to(root)):x.read_bytes().hex() if x.is_file() else 'directory' for x in root.rglob('*')}
    before = tree(p.parent)
    events = []
    def audit(event,args):
        if event == 'open' and len(args)>2 and args[2] & (0x1 | 0x2 | 0x200 | 0x400):
            events.append((event,str(args[0]),args[2]))
        if event in {'os.mkdir','os.rename','os.remove','os.link','os.symlink'}:
            events.append((event,str(args)))
    # Restrict captured events to this lexical operation; audit hook cannot be removed.
    active = [True]
    sys.addaudithook(lambda event,args:audit(event,args) if active[0] else None)
    with patch('bts.dm.send_dm',side_effect=AssertionError('unexpected DM')):
        OUT['real_live_readonly'] = attempt(lambda:cs.load_decision_streak_state(p,now=NOW,require_contest_state=True))
        with patch.object(dt,'datetime',Clock):
            OUT['main_status'] = cli_call(p,'season-contract','status','--season','2026')
    active[0] = False
    OUT['readonly'] = dict(events=events,tree_equal=before==tree(p.parent))
    OUT['new_commands_availability'] = cli_call(p,'season-contract','revoke','--season','2027','--reason','audit')

def main():
    with contextlib.redirect_stderr(io.StringIO()) as log:
        matrix()
        messages_and_sequences()
        current_callers()
        clocks_files_and_status()
    (ROOT/'probe.log').write_text(log.getvalue())
    OUT['denied_network_events'] = DENIED
    (ROOT/'results.json').write_text(json.dumps(OUT,indent=2,default=str)+'\n')
    print(json.dumps(dict(matrix_cells=len(OUT['matrix']),denied_network_events=DENIED,result='results.json')))

if __name__ == '__main__':
    main()
