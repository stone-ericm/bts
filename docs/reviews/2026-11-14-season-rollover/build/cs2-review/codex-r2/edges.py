"""Actual reconcile/receipt New Year crossing and post-read corruption window."""
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo
from click.testing import CliRunner
from bts import picks as p
from bts.cli import cli
from bts.reconcile_receipt import ReconcileReceipt

ROOT = Path(__file__).resolve().parent
LABEL = sys.argv[1]
ET = ZoneInfo("America/New_York")
d = Path(tempfile.mkdtemp(prefix=LABEL+"-midnight-",dir=ROOT / "runs")) / "picks"
d.mkdir()
pk = p.Pick(batter_name="A",batter_id=1,team="NYY",lineup_position=1,pitcher_name="P",pitcher_id=2,
    p_game_hit=.85,flags=[],projected_lineup=False,game_pk=111,game_time="2026-12-30T23:10:00Z")
p.save_pick(p.DailyPick(date="2026-12-30",run_time="2026-12-30T18:00:00Z",pick=pk,double_down=None,
    runner_up=None,result="hit",slot_results={"pick":"hit"}),d)
(d / "streak.json").write_text('{"season":2026,"streak":1,"saver_available":true}')
clock = {"now":datetime(2026,12,31,23,59,59,tzinfo=ET)}
saved = []
def factory(**kwargs):
    rec = ReconcileReceipt(clock=lambda: clock["now"],lookback_days=kwargs["lookback_days"])
    publish = rec.publish
    def crossing_publish(picks_dir):
        saved.append((d / "streak.json").read_bytes().hex())
        publish(picks_dir)
        clock["now"] = datetime(2027,1,1,0,0,1,tzinfo=ET)
    rec.publish = crossing_publish
    return rec
with patch("bts.reconcile_receipt.ReconcileReceipt",side_effect=factory):
    result = CliRunner().invoke(cli,["reconcile","--picks-dir",str(d)])
out = {"crossing":{"exit":result.exit_code,"output":result.output,"bytes_before_summary":saved,
    "bytes_after":(d / "streak.json").read_bytes().hex(),"file":json.loads((d / "streak.json").read_text())}}
assert saved == [out['crossing']['bytes_after']], "summary mutated the streak"

# An explicit saver is also used by the ordinary 10..15 miss branch. If another
# actor corrupts the file AFTER successful reads, it may replace it, as on base.
for results in ([True],[False]):
    f = d / "streak.json"
    f.write_text('{"season":2027,"streak":12,"saver_available":true,"applied_dates":["2027-06-09"]}')
    real_save = p.save_streak
    def corrupt_then_save(*args,**kwargs):
        f.write_text("{")
        return real_save(*args,**kwargs)
    with patch.object(p,"save_streak",side_effect=corrupt_then_save):
        try:
            value = p.update_streak(results,d,date="2027-06-10")
            update = {"value":value}
        except Exception as exc:
            update = {"error":type(exc).__name__}
    out[str(results)] = {"outcome":update,"bytes_after":f.read_bytes().hex()}
(ROOT / (LABEL+"-edges.json")).write_text(json.dumps(out,indent=2)+'\n')
print(LABEL,out['crossing']['output'].strip(),str({k:v['outcome'] for k,v in out.items() if k!='crossing'}))
