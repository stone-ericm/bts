"""Compare the remaining newly strict display reads with the base's paths."""
import contextlib
import io
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
from click.testing import CliRunner
from bts import picks as p, scheduler as s
from bts.cli import cli

ROOT=Path(__file__).resolve().parent
label=sys.argv[1]
out=[]
for which in ('cli-void','poll-void','poll-final-peer','poll-early-peer'):
    d=Path(tempfile.mkdtemp(prefix=label+'-date-',dir=ROOT/'runs'))
    day='2027-3-24'
    pk=p.Pick(batter_name='A',batter_id=1,team='NYY',lineup_position=1,pitcher_name='P',pitcher_id=2,
        p_game_hit=.85,flags=[],projected_lineup=False,game_pk=111,game_time='2027-03-24T23:10:00Z')
    p.save_pick(p.DailyPick(date=day,run_time='2027-03-24T18:00:00Z',pick=pk,double_down=None,
        runner_up=None,notification_sent=True,notification_channel='bluesky_dm',notification_id='probe'),d)
    (d/'streak.json').write_text('{"season":2027,"streak":3,"saver_available":true}')
    original=(d/'streak.json').read_bytes()
    def peer():
        daily=p.load_pick(day,d)
        daily.result='hit'
        p.save_pick(daily,d)
    def resolve(*a):
        if which=='poll-final-peer': peer()
        return {'pick':'void'}
    def midgame(*a):
        peer()
        return [True]
    polls=iter(['live','final'])
    with patch('bts.cli._now_et',return_value=datetime(2027,3,24,23,tzinfo=ZoneInfo('America/New_York'))), \
         patch.object(s,'_now_et',return_value=datetime(2027,3,24,23,tzinfo=ZoneInfo('America/New_York'))), \
         patch.object(s,'poll_game_result',side_effect=(lambda *a: next(polls)) if which=='poll-early-peer' else None,return_value='final'), \
         patch.object(s,'_check_hits_midgame',side_effect=midgame), \
         patch.object(s,'_poll_interval_sleep',return_value=None), \
         patch.object(p,'get_game_statuses_detailed',return_value={}), \
         patch.object(p,'resolve_daily_slot_results',side_effect=resolve), \
         contextlib.redirect_stderr(io.StringIO()) as stderr:
        if which.startswith('cli'):
            result=CliRunner().invoke(cli,['check-results','--date',day,'--picks-dir',str(d)])
            outcome={'exit':result.exit_code,'error':type(result.exception).__name__ if result.exception else None,'output':result.output}
        else:
            try: outcome={'value':s.run_result_polling(111,day,d)}
            except Exception as exc: outcome={'error':type(exc).__name__,'detail':str(exc)}
    out.append({'which':which,'date':day,'source':p.__file__,'outcome':outcome,'stderr':stderr.getvalue(),
        'pick_result':p.load_pick(day,d).result,'streak_unchanged':(d/'streak.json').read_bytes()==original})
(ROOT/(label+'-date-reads.json')).write_text(json.dumps(out,indent=2)+'\n')
print(label,[(e['which'],e['outcome'],e['pick_result']) for e in out])
