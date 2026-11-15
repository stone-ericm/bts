"""Real-file/scorer attacks for the final CS2 review; offline, one src per process."""
import contextlib
import dataclasses
import hashlib
import inspect
import io
import json
import sys
import tempfile
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

from click.testing import CliRunner
from bts import picks as p, scheduler as s, strategy as st
from bts.cli import cli

ROOT = Path(__file__).resolve().parent
LABEL = sys.argv[1]
RUN = Path(tempfile.mkdtemp(prefix=LABEL + "-attack-", dir=ROOT / "runs"))
OUT = ROOT / (LABEL + "-attack.jsonl")
OUT.write_text("")
ET = ZoneInfo("America/New_York")
SCOPED = "season" in inspect.signature(p.load_streak).parameters
assert SCOPED == (LABEL != "base"), p.__file__
DEEP = b"[" * 200_000 + b"]" * 200_000
RAWS = [("brace", b"{"), ("utf8", b"\xff"), ("list", b"[]"), ("null", b"null"),
        ("number", b"7"), ("deep", DEEP)]


def call(fn, *args, **kwargs):
    try:
        value = fn(*args, **kwargs)
        return {"value": dataclasses.asdict(value) if dataclasses.is_dataclass(value) else value}
    except BaseException as exc:
        return {"error": type(exc).__name__, "detail": str(exc)}


def record(name, **data):
    with OUT.open("a") as f:
        f.write(json.dumps({"name": name, **data}, default=str) + "\n")


def new(name):
    d = RUN / name
    d.mkdir(parents=True)
    return d


def pick(d, day="2027-06-10", result=None):
    pk = p.Pick(batter_name="A", batter_id=1, team="NYY", lineup_position=1,
        pitcher_name="P", pitcher_id=2, p_game_hit=.85, flags=[], projected_lineup=False,
        game_pk=111, game_time="2027-06-10T23:10:00Z")
    daily = p.DailyPick(date=day, run_time="2027-06-10T18:00:00Z", pick=pk,
        double_down=None, runner_up=None, result=result,
        slot_results={"pick": result} if result else None,
        notification_sent=True, notification_channel="bluesky_dm", notification_id="probe")
    p.save_pick(daily, d)


def bytes_state(d):
    return {f.name: f.read_bytes() for f in d.glob("*.json")}


def digest_state(data):
    return {k: {"sha256": hashlib.sha256(v).hexdigest(), "length": len(v),
                "bytes": v.hex() if len(v) < 2000 else "saved in fixture"} for k, v in data.items()}


def cli_score(d, kind):
    with patch("bts.cli._now_et", return_value=datetime(2027, 6, 10, 23, tzinfo=ET)), \
         patch.object(p, "resolve_daily_slot_results", return_value={"pick": kind}):
        r = CliRunner().invoke(cli, ["check-results", "--date", "2027-06-10", "--picks-dir", str(d),
            "--shadow-status-output", str(d / "shadow-status.json")])
    return {"exit": r.exit_code, "error": type(r.exception).__name__ if r.exception else None, "output": r.output}


def poll_score(d, kind, early=False):
    stream = io.StringIO()
    polls = iter(["live", "final"])
    with patch.object(s, "_now_et", return_value=datetime(2027, 6, 10, 23, tzinfo=ET)), \
         patch.object(s, "poll_game_result", side_effect=(lambda *a: next(polls)) if early else None, return_value="final"), \
         patch.object(s, "_check_hits_midgame", return_value=[True]), \
         patch.object(s, "_poll_interval_sleep", return_value=None), \
         patch.object(p, "get_game_statuses_detailed", return_value={}), \
         patch.object(p, "resolve_daily_slot_results", return_value={"pick": kind}), \
         contextlib.redirect_stderr(stream):
        result = call(s.run_result_polling, 111, "2027-06-10", d)
    return {**result, "output": stream.getvalue()}


for raw_name, raw in RAWS:
    d = new("readers/" + raw_name)
    f = d / "streak.json"
    f.write_bytes(raw)
    record("reader-" + raw_name, legacy=[call(fn, d) for fn in [p.load_streak, p.load_saver_available, p.load_applied_dates]],
           seasonal=[call(fn, d, season=2027) for fn in [p.load_streak, p.load_saver_available, p.load_applied_dates]],
           unchanged=f.read_bytes() == raw)
    for saver in (None, False, True):
        f.write_bytes(raw)
        result = call(p.save_streak, 4, d, saver_available=saver, **({"season": 2027} if SCOPED else {}))
        record(f"save-{raw_name}-{saver}", result=result, unchanged=f.read_bytes() == raw,
               after=digest_state(bytes_state(d)))
    for path in ("cli-hit", "cli-miss", "cli-void", "poll-hit", "poll-miss", "poll-void", "early-hit",
                 "reconcile-complete", "reconcile-incomplete"):
        d = new("scorers/" + raw_name + "/" + path)
        (d / "streak.json").write_bytes(raw)
        pick(d, result="hit" if path.startswith("reconcile") else None)
        if path == "reconcile-incomplete":
            (d / "2027-04-01.json").write_text("{")
        before = bytes_state(d)
        if path.startswith("cli-"):
            result = cli_score(d, path.split("-")[1])
        elif path.startswith(("poll-", "early-")):
            result = poll_score(d, path.split("-")[1], early=path.startswith("early"))
        else:
            with patch.object(p, "resolve_daily_slot_results", side_effect=lambda daily, *a, **kw: daily.slot_results):
                result = call(p.reconcile_results, d, clock=lambda: datetime(2027, 6, 11, 2, tzinfo=ET))
        after = bytes_state(d)
        same_files = all(after.get(k) == v for k,v in before.items())
        record(path + "-" + raw_name, fixture=str(d), result=result,
               before=digest_state(before), after=digest_state(after), unchanged=same_files,
               pick_result=p.load_pick("2027-06-10", d).result)
        if LABEL in ("base", "current"):
            if path.startswith("reconcile"):
                if path.endswith("complete") and not path.endswith("incomplete"):
                    assert result == {"value": []}, result
                    data = json.loads((d / "streak.json").read_text())
                    assert (data["streak"], data["saver_available"], data["applied_dates"]) == (1, True, [])
                else:
                    assert same_files and result == {"value": []}, result
            else:
                assert same_files and p.load_pick("2027-06-10", d).result is None, path
                assert result.get("error") in ("JSONDecodeError", "UnicodeDecodeError", "AttributeError", "RecursionError"), result
                if path.startswith("cli"):
                    assert result["exit"] == 1

# Guard before existence checks/mkdir/reads, including a bool that aliases year 1.
for i, season in enumerate((True, False, 2027.0, "2027", [], {}, 1+0j)):
    for exists in (False, True):
        d = new(f"season-args/{i}-{exists}")
        if exists:
            (d / "streak.json").write_text('{"season":1,"streak":12,"saver_available":false}')
        before = bytes_state(d)
        results = [call(fn, d, season=season) for fn in (p.load_streak,p.load_saver_available,p.load_applied_dates)]
        results += [call(p.save_streak, 7, d, season=season), call(p.save_streak, 7, d, saver_available=False, season=season)]
        record(f"seasonarg-{i}-{exists}", results=results, unchanged=before == bytes_state(d))
        if LABEL == "current":
            assert all(r.get("error") == "TypeError" for r in results)
            assert before == bytes_state(d)

# All production date-derived seasons are real Python int; private helper isn't a public boundary.
for i, bad in enumerate((date(2027,3,24), datetime(2027,3,24), b"2027-03-24", 20270324,
                         "2027-3-24", "20270324", "2027-03-24T00:00:00")):
    d = new(f"bad-date/{i}")
    (d / "streak.json").write_bytes(b"{")
    with patch.object(p, "load_streak", side_effect=AssertionError("I/O before validation")) as reader:
        result = call(p.update_streak, [True], d, date=bad)
    record(f"datearg-{i}", value=str(bad), result=result, reads=reader.call_count)
    if LABEL == "current":
        assert result.get("error") == "ValueError" and reader.call_count == 0

# Date syntax accepted by actual shadow entry point: real select/persist, only prediction feeds stubbed.
import pandas as pd
for day in ("2027-03-24", "2027-3-24", "2027-03-4", "20270324", "2027-W12-3"):
    d = new("syntax/" + day)
    pick(d, day=day)
    (d / "streak.json").write_text('{"season":2027,"streak":3,"saver_available":true}')
    predictions = pd.DataFrame([dict(batter_id=1,batter_name="A",team="NYY",lineup=1,
        pitcher_name="P",pitcher_id=2,game_pk=111,game_time="2027-03-24T23:10:00Z",p_hit_pa=.3,
        flags="",p_game_hit=.85)])
    statuses = {111: {"abstract":"P","detailed":"Preview"}}
    states = []
    with patch.object(s,"predict_local_shadow",return_value=predictions), \
         patch.object(s,"heartbeat_watchdog",side_effect=lambda *a,**kw: contextlib.nullcontext()), \
         patch.object(s,"_analytics_job_status",return_value={}), \
         patch.object(s,"_update_analytics_job_status",side_effect=lambda *a,**kw: states.append((a,kw))), \
         patch.object(p,"get_game_statuses_detailed",return_value=statuses), \
         patch.object(st,"_load_mdp",return_value=None), \
         patch.object(p,"attach_provenance",return_value=None), \
         patch("bts.orchestrator.shadow_cache_path",return_value=d / "model.pkl"), \
         contextlib.redirect_stderr(io.StringIO()) as stderr:
        result = call(s._run_shadow_prediction, {"orchestrator":{"picks_dir":str(d),"models_dir":str(d)}}, day, "A")
    record("shadow-syntax-" + day, result=result, states=states, stderr=stderr.getvalue(),
           shadow_exists=(d / (day + ".shadow.json")).exists())

print(json.dumps({"label":LABEL,"module":p.__file__,"events":len(OUT.read_text().splitlines()),"output":str(OUT)}))
