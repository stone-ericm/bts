from pathlib import Path
source=Path('/tmp/g2_r2_cold.py').read_text().split('cwd=os.getcwd();os.chdir(ROOT)')[0]
exec(compile(source,'/tmp/g2_r2_cold.py','exec'))
previous=Path(next(l.split(' ',1)[1] for l in Path('/tmp/g2-r2-cold.out').read_text().splitlines() if l.startswith('ROOT ')))
for fn in ['blend_2027-03-24.pkl','blend_2027-03-25.pkl']:shutil.copyfile(previous/'data/models'/fn,ROOT/'data/models'/fn)
cwd=os.getcwd();os.chdir(ROOT)
try:
    with ExitStack() as st:
        for mod in [sch,picks,strategy,contest,decisions]:st.enter_context(patch.object(mod,'datetime',Clock))
        for mod,attr in [(m,'urlopen'),(pull,'retry_urlopen'),(pull,'urlopen'),(picks,'retry_urlopen'),(sch,'retry_urlopen')]:st.enter_context(patch.object(mod,attr,net))
        st.enter_context(patch('bts.dm.send_dm',lambda *a:dms.append(a) or 'dm-fake'))
        st.enter_context(patch('bts.health.alert.send_dm',lambda *a:dms.append(a) or 'dm-fake'))
        print('ROOT',ROOT,flush=True)
        preview=CliRunner().invoke(cli,['preview','--date','2027-03-24'])
        print('REAL_PREVIEW_UNPOSTED',preview.exit_code,repr(preview.output),flush=True)
        for day in ['2027-03-24','2027-03-25']:
            state['day']=day;state['posted']=True
            config={'orchestrator':{'picks_dir':'data/picks','models_dir':'data/models','heartbeat_path':'data/.hb'},'tiers':[{'name':'local','type':'local'}],'scheduler':{'pick_delivery':'dm','live_forward_capture_on_lock':False},'bluesky':{'dm_recipient':'fake'}}
            check=sch.run_single_check(day,[schedule(day)[0]['gamePk']],set(),config,.03)
            print('REAL_GATE',day,'new_lineups',check['new_lineups'],'should_post',check['should_post'],'action',check['selection'].action if check['selection'] else None,flush=True)
            assert check['should_post']
            ss=sch.SchedulerState(date=day,schedule_fetched_at=now().isoformat(),games=[],confirmed_game_pks=[],runs_completed=[],pick_locked=False,pick_locked_at=None,result_status=None,next_wakeup=None)
            delivered=sch._deliver_and_lock_pick(check['pick_result'].daily,config,Path('data/picks'),ss,day,'lineup',selection=check['selection'])
            print('REAL_GATE_DELIVERY',day,delivered,json.dumps(decisions.load_decision(day,Path('data/picks')),default=str),flush=True)
            assert delivered
        print('DMS',len(dms),flush=True)
finally:os.chdir(cwd)
