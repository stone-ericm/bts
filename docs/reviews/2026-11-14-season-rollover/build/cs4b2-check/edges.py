"""Additional offline execution of production fallback catches and rename primitives."""
import sys
sys.dont_write_bytecode = True
def deny(event, args):
    if event in {'socket.getaddrinfo','socket.gethostbyname','socket.gethostbyaddr',
                 'socket.getnameinfo','socket.connect','socket.connect_ex','socket.sendto'}:
        raise RuntimeError('audit: network forbidden: '+event)
sys.addaudithook(deny)

import contextlib
import ctypes
import errno
import io
import json
import os
from pathlib import Path
from unittest.mock import patch
import probe as p

OUT = {}
def scheduler_fallbacks():
    for which in ('in_loop', 'final'):
        root = p.directory('fallback-'+which)
        now = p.NOW.replace(hour=18, minute=40)
        day = now.date().isoformat()
        daily = p.picks.DailyPick(date=day,run_time=now.isoformat(),pick=p.picks.Pick(
            batter_name='Offline',batter_id=1,team='NYM',lineup_position=1,pitcher_name='P',
            pitcher_id=2,p_game_hit=.8,flags=[],projected_lineup=False,game_pk=999,
            game_time='2027-07-15T23:05:00Z'),double_down=None,runner_up=None)
        p.picks.save_pick(daily,root)
        config = dict(orchestrator=dict(picks_dir=str(root),heartbeat_path=str(root/'hb')),
                      scheduler=dict(pick_delivery='dm',contest_state_required=True),
                      health_checks=dict(enabled=False),bluesky=dict(dm_recipient='offline.test'),tiers=[])
        result = dict(skipped=False,new_lineups=0,should_post=False,pick_result=None,
                      pick_name=None,pick_p=None,selection=None)
        if which == 'in_loop':
            from bts.strategy import PickResult
            result['pick_result'] = PickResult(daily=daily,locked=False)
        runs = [dict(time_et=now,game_pks=[999])] if which == 'in_loop' else []
        alerts, sends, calls = [], [], []
        def refuse(*args,**kw):
            calls.append(kw)
            raise p.cs.ContestStateError('injected NEW refusal')
        with contextlib.ExitStack() as stack:
            for q in [patch.object(p.sch,'_now_et',lambda:now),
                      patch.object(p.sch,'fetch_schedule',lambda *a:[dict(gamePk=999,gameDate=daily.pick.game_time)]),
                      patch.object(p.sch,'detect_doubleheader_game2s',lambda *a:[]),
                      patch.object(p.sch,'plan_run_times',lambda *a,**kw:runs),
                      patch.object(p.sch,'game_time_et',lambda *a:now.replace(hour=19,minute=5)),
                      patch.object(p.sch,'run_single_check',lambda **kw:result),
                      patch.object(p.sch,'run_and_pick',refuse),
                      patch.object(p.sch,'heartbeat_watchdog',lambda *a,**kw:contextlib.nullcontext()),
                      patch.object(p.sch,'_alert_contest_state_failure',lambda *a, slate_date=None:alerts.append(str(a[-1]))),
                      patch.object(p.sch,'_next_day_wakeup',lambda *a:now+p.dt.timedelta(days=1)),
                      patch.object(p.sch,'_idle_until_next_wakeup',lambda *a:None),
                      patch.object(p.sch,'_watchdog_ping_sleep',lambda *a:None),
                      patch.object(p.sch,'notify_ready',lambda:None),
                      patch.object(p.sch,'notify_watchdog',lambda:None),
                      patch('bts.dm.send_dm',lambda *a,**kw:sends.append(a))]:
                stack.enter_context(q)
            outcome = p.attempt(lambda:p.sch.run_day(day,config))
        OUT['fallback_'+which] = dict(outcome=outcome,alerts=alerts,calls=calls,sends=sends,
            state=json.loads((root/day/'scheduler_state.json').read_text()))

def grading_and_outcome():
    root = p.directory('polling')
    day = '2027-07-15'
    daily = p.picks.DailyPick(date=day,run_time=p.NOW.isoformat(),pick=p.picks.Pick(
        batter_name='Offline',batter_id=1,team='NYM',lineup_position=1,pitcher_name='P',
        pitcher_id=2,p_game_hit=.8,flags=[],projected_lineup=False,game_pk=999,
        game_time='2027-07-15T23:05:00Z'),double_down=None,runner_up=None,
        notification_sent=True,notification_channel='bluesky_dm',notification_id='offline',delivered_at=p.NOW.isoformat())
    p.picks.save_pick(daily,root)
    with patch.object(p.cs,'load_decision_streak_state',side_effect=p.cs.ContestStateError('injected NEW refusal')),\
         patch.object(p.sch,'_now_et',lambda:p.NOW),\
         patch.object(p.sch,'poll_game_result',lambda *a:'final'),\
         patch.object(p.picks,'get_game_statuses_detailed',lambda *a:{}),\
         patch.object(p.picks,'resolve_daily_slot_results',lambda *a:{'primary':'hit'}),\
         patch('bts.dm.send_dm',lambda *a,**kw:'captured'):
        OUT['real_result_polling'] = p.attempt(lambda:p.sch.run_result_polling(999,day,root))
        OUT['polling_streak'] = json.loads((root/'streak.json').read_text())
        # A real declared research skip is a clean day, despite a refused live loader.
        config = dict(scheduler=dict(entry_intent='research',pick_delivery='private'))
        skiproot = p.directory('day-outcome-clean')
        from bts.daily_decision import write_decision
        write_decision(day,skiproot,action='skip',source='mdp',streak=7,saver_available=False,
                       state_source='contest',state_status='fresh',delivery_status='not_applicable',scoreable=False)
        from click.testing import CliRunner
        with patch.object(p.orch,'load_config',lambda *a:config),\
             patch.object(p.sch,'fetch_schedule',lambda *a:[dict(gamePk=999,status={})]):
            r = CliRunner().invoke(p.cli.cli,['day-outcome','--date',day,'--config',str(p.ROOT/'unused'),
                                             '--picks-dir',str(skiproot)])
            OUT['clean_day_outcome'] = dict(exit=r.exit_code,output=r.output,exception=repr(r.exception))

def filesystem_and_failures():
    root = p.directory('no-replace')
    native = ctypes.CDLL(None,use_errno=True).renamex_np
    native.argtypes = [ctypes.c_char_p,ctypes.c_char_p,ctypes.c_uint]
    native.restype = ctypes.c_int
    results = {}
    for name in ('regular','dangling','directory','absent'):
        source, target = root/('source-'+name), root/('target-'+name)
        source.write_text('source')
        if name == 'regular': target.write_text('target')
        elif name == 'dangling': target.symlink_to(root/'missing')
        elif name == 'directory': target.mkdir()
        rc = native(os.fsencode(source),os.fsencode(target),0x4)
        results[name] = dict(rc=rc,errno=ctypes.get_errno() if rc else 0,
                             source_exists=source.exists(),target_symlink=target.is_symlink())
    OUT['renamex_np_RENAME_EXCL'] = results
    from bts.util import atomic_write_text
    root = p.directory('write-failure')
    path = root/'account_state/season_contract.json'
    p.write(root/'account_state/contest_streak.manual.json',p.dated())
    p.write(root/'account_state/saver_state.json',dict(state='used',season=2027))
    path.mkdir()
    OUT['switch_path_directory'] = p.attempt(lambda:atomic_write_text(path,json.dumps(p.switch())))
    OUT['switch_failure_sources_untouched'] = [(x.name,x.exists()) for x in
        (root/'account_state/contest_streak.manual.json',root/'account_state/saver_state.json')]
    # Step (a) as literally enumerated in row 7 omits the mandatory schema key.
    literal = dict(season=2027,status='revoked',activation_id=p.ID,switched_at=p.NOW.isoformat(),
                   revoked_at=p.NOW.isoformat(),note='audit')
    OUT['literal_revoke_payload'] = p.attempt(lambda:p.sc.validate_contract(literal))
    # Health's existing post-loader saver failure can suppress the proposed append.
    root = p.directory('health-post-load-failure')
    p.write(root/'account_state/contest_streak.json',p.auto())
    (root/'account_state/saver_state.json').write_text('{"n":'+('9'*5000)+'}')
    OUT['health_saver_error'] = p.attempt(lambda:p.health.check(root,now=p.NOW,expected=True))

def main():
    with contextlib.redirect_stderr(io.StringIO()) as log:
        scheduler_fallbacks()
        grading_and_outcome()
        filesystem_and_failures()
    (p.ROOT/'edges.log').write_text(log.getvalue())
    (p.ROOT/'edges-results.json').write_text(json.dumps(OUT,indent=2,default=str)+'\n')
    print(json.dumps(OUT,indent=2,default=str))
if __name__ == '__main__': main()
