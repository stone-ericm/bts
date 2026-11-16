"""Corrupt streak bytes through scorer exits that do not consume the streak."""
import datetime as dt
import json
from pathlib import Path
import socket
import sys
from unittest.mock import patch
from zoneinfo import ZoneInfo
H=Path(__file__).resolve().parent;R=H.parents[1];mode=sys.argv[1]
sys.path.insert(0,str(H/'base/src' if mode=='base' else R/'src'))
import bts.picks as p
import bts.cli as c
import bts.scheduler as s
from click.testing import CliRunner
ET=ZoneInfo('America/New_York')
socket.socket.connect=lambda *a,**k: (_ for _ in ()).throw(RuntimeError('no network'))
socket.socket.connect_ex=socket.socket.connect
out=(H/f'exit-paths-{mode}.jsonl').open('w')
root=H/f'exit-fixtures-{mode}';root.mkdir(exist_ok=True)
corrupt={'brace':b'{','utf8':b'\xff','list':b'[]','null':b'null','number':b'42',
         'deep':b'['*200000+b']'*200000,'unreadable':b'{"streak":13}'}
for label,payload in corrupt.items():
    for path in ('cli-missing','cli-already','cli-pending','cli-error','cli-undelivered','cli-stale','cli-vanish',
                 'poll-missing','poll-already','poll-cap','poll-pending','poll-suspended','poll-vanish'):
        d=root/f'{label}-{path}';d.mkdir(exist_ok=True)
        (d/'streak.json').write_bytes(payload)
        pick=p.Pick(batter_name='Test',batter_id=1,team='NYY',lineup_position=1,pitcher_name='P',pitcher_id=2,
            p_game_hit=.85,flags=[],projected_lineup=False,game_pk=111,game_time='2027-03-24T23:00:00Z')
        if not path.endswith('missing'):
            p.save_pick(p.DailyPick(date='2027-03-24',run_time='2027-03-24T18:00:00Z',pick=pick,
                double_down=None,runner_up=None,result='hit' if path.endswith('already') else None,
                bluesky_posted=not path.endswith('undelivered')),d)
        before={f.name:f.read_bytes().hex() for f in d.glob('*.json')}
        now=dt.datetime(2027,3,25,5 if path.endswith('cap') else 2,tzinfo=ET)
        if path.endswith('stale'):now=dt.datetime(2027,4,1,2,tzinfo=ET)
        def resolve(*a,**kw):
            if path.endswith('error'):raise RuntimeError('feed unavailable')
            if path.endswith('vanish'):(d/'2027-03-24.json').unlink()
            return None if path.endswith('pending') else {'pick':'hit'}
        if label=='unreadable':(d/'streak.json').chmod(0)
        try:
            with patch.object(c,'_now_et',return_value=now),patch.object(s,'_now_et',return_value=now), \
                 patch.object(p,'resolve_daily_slot_results',side_effect=resolve), \
                 patch.object(p,'get_game_statuses_detailed',return_value={}), \
                 patch.object(s,'poll_game_result',return_value='suspended' if path.endswith('suspended') else 'final'), \
                 patch.object(s,'_check_hits_midgame',return_value=[]):
                if path.startswith('cli'):
                    r=CliRunner().invoke(c.cli,['check-results','--date','2027-03-24','--picks-dir',str(d)])
                    observed={'exit':r.exit_code,'output':r.output,'error':type(r.exception).__name__ if r.exception else None}
                else:
                    try:observed={'return':s.run_result_polling(111,'2027-03-24',d)}
                    except BaseException as e:observed={'error':type(e).__name__,'message':str(e)}
        finally:
            (d/'streak.json').chmod(0o600)
        after={f.name:f.read_bytes().hex() for f in d.glob('*.json')}
        assert after['streak.json']==before['streak.json']
        out.write(json.dumps(dict(case=label,path=path,before=before,after=after,outcome=observed))+'\n')
out.close()
print(mode,'exit records',len(corrupt)*13)
