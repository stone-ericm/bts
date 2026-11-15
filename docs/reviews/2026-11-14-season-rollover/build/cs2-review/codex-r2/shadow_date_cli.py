"""Production shadow one-shot CLI: unpadded date works on base, fails after CS2.

Real argument parser, TOML loader, pick/state readers, analytics status writer,
strategy, provenance attachment and shadow persistence. Only model prediction,
MLB detailed statuses and watchdog are stubbed. No network or posting.
"""
import contextlib
import dataclasses
import io
import json
import sys
import tempfile
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from bts import picks as p, scheduler as s, strategy as st
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parents[2]))
from scripts import shadow_predict_once as script

label = sys.argv[1]
out = []
for day in ("2027-03-24", "2027-3-24"):
    d = Path(tempfile.mkdtemp(prefix=label+"-shadow-cli-",dir=ROOT / "runs"))
    pk = p.Pick(batter_name="A",batter_id=1,team="NYY",lineup_position=1,pitcher_name="P",pitcher_id=2,
        p_game_hit=.85,flags=[],projected_lineup=False,game_pk=111,game_time="2027-03-24T23:10:00Z")
    daily = p.DailyPick(date=day,run_time="2027-03-24T18:00:00Z",pick=pk,double_down=None,runner_up=None,
        notification_sent=True,notification_channel="bluesky_dm",notification_id="probe")
    p.save_pick(daily,d)
    (d / "streak.json").write_text('{"season":2027,"streak":3,"saver_available":true}')
    s.save_state(s.SchedulerState(date=day,schedule_fetched_at="2027-03-24T18:00:00Z",games=[],
        confirmed_game_pks=[],runs_completed=[],pick_locked=True,pick_locked_at="2027-03-24T18:00:00Z",
        result_status=None,next_wakeup=None),d)
    cfg = d / "config.toml"
    cfg.write_text('[orchestrator]\npicks_dir = '+json.dumps(str(d))+'\nmodels_dir = '+json.dumps(str(d))+'\n')
    preds = pd.DataFrame([dict(batter_id=1,batter_name="A",team="NYY",lineup=1,pitcher_name="P",pitcher_id=2,
        game_pk=111,game_time="2027-03-24T23:10:00Z",p_hit_pa=.3,flags="",p_game_hit=.85)])
    compat = (patch.object(s,"date_type",SimpleNamespace(fromisoformat=lambda value: datetime.strptime(value,"%Y-%m-%d").date()))
              if "--fix" in sys.argv else contextlib.nullcontext())
    with compat, patch.object(s,"predict_local_shadow",return_value=preds), \
         patch.object(s,"heartbeat_watchdog",side_effect=lambda *a,**kw: contextlib.nullcontext()), \
         patch.object(p,"get_game_statuses_detailed",return_value={111:{"abstract":"P","detailed":"Preview"}}), \
         patch.object(st,"_load_mdp",return_value=None), \
         contextlib.redirect_stderr(io.StringIO()) as stderr:
        rc = script.main(["--date",day,"--config",str(cfg),"--unit",""])
    state = dataclasses.asdict(s.load_state(day,d))
    out.append({"day":day,"source":p.__file__,"fixture":str(d),"exit":rc,"stderr":stderr.getvalue(),
        "shadow_written":(d / (day+".shadow.json")).exists(),"state":state})
(ROOT / (label+"-shadow-cli.json")).write_text(json.dumps(out,indent=2)+'\n')
print(label,[(e['day'],e['exit'],e['shadow_written'],e['state']['analytics_jobs']['shadow']['status']) for e in out])
