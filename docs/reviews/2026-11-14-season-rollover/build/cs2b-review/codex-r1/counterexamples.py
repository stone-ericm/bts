"""Small standalone Row L regressions, run against both production trees.

Expected contract: unrecognized non-string reads base state and returns final;
no subclass callback may introduce a new read failure or hang.
"""
from datetime import datetime
import json
from pathlib import Path
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo

H=Path(__file__).resolve().parent
R=H.parents[1]
mode=sys.argv[1]
sys.path.insert(0,str(H/'base/src' if mode=='base' else R/'src'))
import bts.picks as p
import bts.scheduler as s

class NotString:
    @property
    def __class__(self):
        raise RuntimeError('non-string class hook')
    def __str__(self):
        return 'junk'

class StringHook(str):
    def __len__(self):
        raise SystemExit('string length hook')

out=[]
for name,day in [('not-string',NotString()),('string-hook',StringHook('2027-3-24'))]:
    d=H/f'counterexample-{mode}-{name}'
    d.mkdir(exist_ok=True)
    (d/'streak.json').write_text('{"season":2026,"streak":13,"saver_available":false}')
    pick=p.Pick(batter_name='Test',batter_id=1,team='NYY',lineup_position=1,pitcher_name='P',pitcher_id=2,
        p_game_hit=.85,flags=[],projected_lineup=False,game_pk=111,game_time='2027-03-24T23:00:00Z')
    p.save_pick(p.DailyPick(date=str(day),run_time='2027-03-24T18:00:00Z',pick=pick,
        double_down=None,runner_up=None,notification_sent=True,bluesky_posted=True),d)
    before=(d/'streak.json').read_bytes()
    pb=(d/f'{day}.json').read_bytes()
    with patch.object(s,'_now_et',return_value=datetime(2027,3,24,23,tzinfo=ZoneInfo('America/New_York'))), \
         patch.object(s,'poll_game_result',return_value='final'), \
         patch.object(p,'get_game_statuses_detailed',return_value={}), \
         patch.object(p,'resolve_daily_slot_results',return_value={'pick':'void'}):
        try:r={'return':s.run_result_polling(111,day,d)}
        except BaseException as exc:r={'error':type(exc).__name__,'message':str(exc)}
    out.append(dict(case=name,outcome=r,streak_before=before.hex(),streak_after=(d/'streak.json').read_bytes().hex(),
        pick_before=pb.hex(),pick_after=(d/f'{day}.json').read_bytes().hex(),pick_result=p.load_pick(day,d).result))
(H/f'counterexamples-{mode}.json').write_text(json.dumps(out,indent=2))
for x in out:print(mode,x['case'],x['outcome'],'pick',x['pick_result'],'streak_same',x['streak_before']==x['streak_after'])
