"""Offline differential through real shadow command and polling, in isolated packages."""
import contextlib
import dataclasses
import hashlib
import io
import json
import socket
import sys
import traceback
from datetime import date, datetime
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[1]
LABEL = sys.argv[1]
SOURCE = REPO if LABEL == "head" else ROOT / "sources" / LABEL
sys.path[:0] = [str(SOURCE / "src"), str(SOURCE)]
from bts import cli as c, picks as p, scheduler as s, strategy as st
from click.testing import CliRunner
from scripts import shadow_predict_once as command
import pandas as pd

CASES = [
    ("padded", "2027-03-24"), ("month", "2027-3-24"),
    ("day", "2027-03-4"), ("both", "2027-3-4"),
    ("leading", " 2027-03-24"), ("trailing", "2027-03-24 "),
    ("surrounding", " 2027-03-24 "), ("newline", "2027-03-24\n"),
    ("time_T", "2027-03-24T00:00:00"), ("time_space", "2027-03-24 00:00:00"),
    ("slash", "2027/03/24"), ("dot", "2027.03.24"),
    ("underscore", "2027_03_24"), ("compact", "20270324"),
    ("week", "2027-W12-3"), ("invalid_day", "2027-02-30"),
    ("junk", "nonsense"), ("empty", ""),
    ("space_day", "2027-03- 4"), ("unicode_month", "2027-\u0663-24"),
    ("date_object", date(2027, 3, 24)), ("none", None),
]
PREDS = pd.DataFrame([dict(batter_id=1,batter_name="A",team="NYY",lineup=1,
    pitcher_name="P",pitcher_id=2,game_pk=111,game_time="2027-03-24T23:10:00Z",
    p_hit_pa=.3,flags="",p_game_hit=.85)])
ET = ZoneInfo("America/New_York")
NOW = datetime(2027, 3, 24, 23, tzinfo=ET)

def hashes(folder):
    return {str(f.relative_to(folder)):hashlib.sha256(f.read_bytes()).hexdigest()
            for f in folder.rglob("*") if f.is_file()}

def fixture(folder, day):
    folder.mkdir(parents=True, exist_ok=True)
    stored = day.isoformat() if isinstance(day, date) else day
    # The paths use f-strings at the base too. Seed real files without teaching
    # production parsers to accept the probe's input.
    (folder / f"{day}.json").parent.mkdir(parents=True, exist_ok=True)
    pk = p.Pick(batter_name="A",batter_id=1,team="NYY",lineup_position=1,
        pitcher_name="P",pitcher_id=2,p_game_hit=.85,flags=[],projected_lineup=False,
        game_pk=111,game_time="2027-03-24T23:10:00Z")
    daily = p.DailyPick(date=stored,run_time="2027-03-24T18:00:00Z",pick=pk,
        double_down=None,runner_up=None,notification_sent=True,
        notification_channel="bluesky_dm",notification_id="probe")
    p.save_pick(daily, folder)
    (folder / "streak.json").write_text('{"season":2027,"streak":3,"saver_available":true}')
    return daily

def denied(*args, **kwargs):
    raise AssertionError("network access forbidden in confirmation probe")

out = []
with patch.object(socket, "create_connection", side_effect=denied), \
     patch.object(socket.socket, "connect", side_effect=denied):
    for name, day in CASES:
        try:
            parse = {"year":datetime.strptime(day, "%Y-%m-%d").year}
        except Exception as exc:
            parse = {"error":type(exc).__name__, "detail":str(exc)}
        for path in ("shadow", "void", "final_peer", "early_peer", "cli_void", "strategy", "update"):
            folder = ROOT / "runs" / "attempt2" / LABEL / path / name
            fixture(folder, day)
            events = []
            def peer():
                daily = p.load_pick(day, folder)
                daily.result = "hit"
                p.save_pick(daily, folder)
                events.append("peer_saved_hit")
            def resolve(*args):
                events.append("resolve")
                if path == "final_peer":
                    peer()
                return {"pick":"void"}
            def midgame(*args):
                peer()
                return [True]
            polls = iter(["live", "final"])
            if path == "early_peer":
                daily = p.load_pick(day, folder)
                daily.bluesky_uri = "at://probe"
                p.save_pick(daily, folder)
            if path == "shadow":
                state = s.SchedulerState(date=day, schedule_fetched_at="2027-03-24T18:00:00Z",
                    games=[],confirmed_game_pks=[],runs_completed=[],pick_locked=True,
                    pick_locked_at="2027-03-24T18:00:00Z",result_status=None,next_wakeup=None)
                # Scheduler state is JSON; a date object isn't a CLI input. The
                # direct run_once object probe still uses the same filename.
                if not isinstance(day, str):
                    state.date = str(day)
                s.save_state(state, folder)
                (folder / "config.toml").write_text('[orchestrator]\npicks_dir = '+json.dumps(str(folder))+
                    '\nmodels_dir = '+json.dumps(str(folder))+ '\n')
            before = hashes(folder)
            stderr = io.StringIO()
            stdout = io.StringIO()
            try:
                with contextlib.ExitStack() as stack:
                    stack.enter_context(contextlib.redirect_stderr(stderr))
                    stack.enter_context(contextlib.redirect_stdout(stdout))
                    stack.enter_context(patch.object(s, "_now_et", return_value=NOW))
                    stack.enter_context(patch.object(c, "_now_et", return_value=NOW))
                    stack.enter_context(patch.object(p, "get_game_statuses_detailed", return_value={111:{"abstract":"P","detailed":"Preview"}}))
                    stack.enter_context(patch.object(st, "get_game_statuses", return_value={111:"P"}))
                    stack.enter_context(patch.object(st, "_load_mdp", return_value=None))
                    load = stack.enter_context(patch.object(p, "load_streak", wraps=p.load_streak))
                    saver = stack.enter_context(patch.object(st, "load_saver_available", wraps=st.load_saver_available))
                    select = stack.enter_context(patch.object(s, "select_pick", wraps=s.select_pick))
                    save = stack.enter_context(patch.object(p, "save_pick", wraps=p.save_pick))
                    stack.enter_context(patch.object(s, "predict_local_shadow", return_value=PREDS))
                    stack.enter_context(patch.object(s, "heartbeat_watchdog", side_effect=lambda *a,**k:contextlib.nullcontext()))
                    stack.enter_context(patch.object(s, "poll_game_result", side_effect=(lambda *a:next(polls)) if path == "early_peer" else lambda *a:"final"))
                    stack.enter_context(patch.object(s, "_check_hits_midgame", side_effect=midgame))
                    stack.enter_context(patch.object(s, "_poll_interval_sleep", return_value=None))
                    stack.enter_context(patch.object(p, "resolve_daily_slot_results", side_effect=resolve))
                    reply = stack.enter_context(patch("bts.posting.format_result_reply", return_value="reply"))
                    stack.enter_context(patch("bts.posting.reply_to_bluesky", return_value="at://reply"))
                    if path == "shadow":
                        if isinstance(day, str):
                            value = command.main(["--date",day,"--config",str(folder / "config.toml"),"--unit",""])
                        else:
                            value = command.run_once(folder / "config.toml", day, None)
                    elif path == "cli_void":
                        if not isinstance(day, str):
                            raise TypeError("non-string not representable as CLI argv")
                        cli_result = CliRunner().invoke(c.cli,["check-results","--date",day,"--picks-dir",str(folder)])
                        value = {"exit":cli_result.exit_code,"output":cli_result.output,
                            "error":type(cli_result.exception).__name__ if cli_result.exception else None,
                            "detail":str(cli_result.exception) if cli_result.exception else None}
                    elif path == "strategy":
                        result = st.select_pick(PREDS,day,folder,streak=3,for_shadow=True,
                            game_statuses_detailed={111:{"abstract":"P","detailed":"Preview"}},require_detailed_statuses=True)
                        value = {"pick":result.pick_result is not None,"saver":result.saver_available}
                    elif path == "update":
                        value = p.update_streak([True],folder,date=day)
                    else:
                        value = s.run_result_polling(111,day,folder)
                outcome = {"value":value}
            except Exception as exc:
                outcome = {"error":type(exc).__name__,"detail":str(exc),"traceback":traceback.format_exc()}
            after = hashes(folder)
            row = {"label":LABEL,"source":s.__file__,"path":path,"case":name,
                "input":repr(day),"parser":parse,"outcome":outcome,"events":events,
                "streak_calls":[str(call) for call in load.call_args_list],
                "saver_calls":[str(call) for call in saver.call_args_list],
                "select_calls":select.call_count,"save_calls":save.call_count,
                "reply_calls":[str(call) for call in reply.call_args_list],
                "stderr":stderr.getvalue(),"stdout":stdout.getvalue(),
                "new_files":sorted(after.keys()-before.keys()),
                "changed_files":sorted(f for f in before.keys() & after.keys() if before[f]!=after[f]),
                "streak_unchanged":before.get("streak.json")==after.get("streak.json"),
                "pick_result":p.load_pick(day,folder).result}
            if path == "shadow":
                state = s.load_state(str(day),folder)
                row["analytics"] = dataclasses.asdict(state).get("analytics_jobs") if state else None
                row["shadow_written"] = (folder / f"{day}.shadow.json").exists()
            out.append(row)
(ROOT / f"{LABEL}-probe.json").write_text(json.dumps(out,indent=2)+"\n")
print(LABEL, len(out), "events", "source",s.__file__)
for path in ("shadow","void","final_peer","early_peer","cli_void","strategy","update"):
    print(path, [(e["case"],e["outcome"].get("error",e["outcome"].get("value")),e.get("shadow_written"),e["pick_result"]) for e in out if e["path"]==path])
