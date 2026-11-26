"""Independent base/current comparison of real entry points and file bytes.

Usage: python probe.py base|current. Results and real fixture directories retained.
No production function under review is replaced; only prediction/MLB/posting and clocks.
"""
import contextlib
import datetime as dt
import hashlib
import io
import json
import os
from pathlib import Path
import socket
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MODE = sys.argv[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE / "base/src" if MODE == "base" else ROOT / "src"))
import bts.picks as p
import bts.cli as c
import bts.scheduler as s
import bts.strategy as st
from bts.contest_state import load_decision_streak_state
from click.testing import CliRunner
from tests.test_strategy import _predictions

ET = ZoneInfo("America/New_York")
NOW = dt.datetime(2027, 3, 24, 23, tzinfo=ET)
REAL_DT = dt.datetime
class Clock(REAL_DT):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)
p.datetime = Clock
st.datetime = Clock
def no_network(*a, **kw):
    raise RuntimeError("review forbids network")
socket.socket.connect = no_network
socket.socket.connect_ex = no_network

class DateStr(str):
    pass
class ExitStr(str):
    def __len__(self):
        raise SystemExit("subclass len called")
class ClassTrap:
    @property
    def __class__(self):
        raise RuntimeError("not a str")

CASES = [
    ("padded", "2027-03-24"), ("unpadded", "2027-3-24"),
    ("unpadded-day", "2027-03-4"), ("unpadded-both", "2027-3-4"),
    ("compact", "20270324"), ("week", "2027-W12-3"),
    ("week-compact", "2027W123"), ("week-year-boundary", "2026-W53-7"),
    ("week-no-day", "2027-W12"), ("ordinal", "2027-083"),
    ("leading-space", " 2027-03-24"), ("trailing-space", "2027-03-24 "),
    ("tab", "\t2027-03-24"), ("newline", "2027-03-24\n"),
    ("time", "2027-03-24T10:00"), ("zone", "2027-03-24T10:00:00-04:00"),
    ("zulu", "2027-03-24Z"), ("slash", "2027/03/24"), ("dot", "2027.03.24"),
    ("impossible", "2027-02-30"), ("bad-month", "2027-13-01"),
    ("year-zero", "0000-03-24"), ("year-10000", "10000-03-24"),
    ("year-one", "0001-03-24"), ("year-9999", "9999-03-24"),
    ("arabic-year", "٢٠٢٧-03-24"), ("fullwidth-year", "２０２７-03-24"),
    ("arabic-day", "2027-03-٢٤"), ("arabic-all", "٢٠٢٧-٠٣-٢٤"),
    ("long", "x" * 10000), ("long-tail", "2027-03-24" + " " * 10000),
    ("empty", ""), ("none", None), ("date", dt.date(2027, 3, 24)),
    ("datetime", dt.datetime(2027, 3, 24, 1, tzinfo=ET)),
    ("bytes", b"2027-03-24"), ("int", 20270324), ("bool", True),
    ("str-subclass", DateStr("2027-03-24")),
    ("str-subclass-unpadded", DateStr("2027-3-24")),
    ("str-subclass-exit", ExitStr("2027-3-24")),
    ("class-trap", ClassTrap()), ("junk", "junk"),
]
OUT = (HERE / f"probe-{MODE}.jsonl").open("w")
FIXTURES = HERE / f"fixtures-{MODE}"
FIXTURES.mkdir(exist_ok=True)
serial = 0
def emit(data):
    OUT.write(json.dumps(data, default=str, ensure_ascii=True) + "\n")
    OUT.flush()
def outcome(fn):
    try:
        value = fn()
        if isinstance(value, (str, int, float, bool, list, dict, type(None))):
            return {"return": value}
        return {"return_type": type(value).__name__}
    except BaseException as e:
        return {"error": type(e).__name__, "message": str(e)[:250]}
def new_dir(group):
    global serial
    serial += 1
    d = FIXTURES / f"{serial:04d}-{group}"
    d.mkdir(exist_ok=True)
    return d
def seed(d, raw=None, **kw):
    data = dict(streak=13, saver_available=False, season=2026, applied_dates=["2026-09-27"])
    data.update(kw)
    (d / "streak.json").write_bytes(raw if raw is not None else json.dumps(data).encode())
def raw(d):
    path = d / "streak.json"
    return path.read_bytes().hex() if path.exists() else None
def pick_bytes(d):
    return {str(f.relative_to(d)): f.read_bytes().hex() for f in d.rglob("*.json")
            if f.name != "streak.json" and "decisions" not in f.parts}
def plant(d, day, result=None, double=False, marker=None):
    # File lookup uses f'{day}.json' in production, including for typed arguments.
    key = str(day)
    if len(key) > 240:
        return
    (d / f"{key}.json").parent.mkdir(parents=True, exist_ok=True)
    pick = p.Pick(batter_name="Probe", batter_id=1, team="NYY", lineup_position=1,
                  pitcher_name="P", pitcher_id=2, p_game_hit=.85, flags=[],
                  projected_lineup=False, game_pk=111, game_time="2027-03-24T23:10:00Z")
    daily = p.DailyPick(date=key, run_time="2027-03-24T18:00:00Z", pick=pick,
                        double_down=pick if double else None, runner_up=None,
                        result=result, slot_results={"pick": result} if result else None,
                        notification_sent=True, bluesky_uri="at://test", bluesky_posted=True)
    p.save_pick(daily, d)

def readers(d, year=2027):
    data = {}
    for name in ("load_streak", "load_saver_available", "load_applied_dates"):
        fn = getattr(p, name)
        data[name] = outcome(lambda: fn(d, **({"season": year} if MODE != "base" else {})))
    data["decision"] = outcome(lambda: vars(load_decision_streak_state(d, now=NOW, require_contest_state=False)))
    return data

def drive(path, d, day, result="hit", crash=False):
    """Drive real production; collect values returned by the real readers."""
    seen, replies, decisions = [], [], []
    real_load, real_saver = p.load_streak, st.load_saver_available
    def load(*a, **kw):
        value = real_load(*a, **kw)
        seen.append(["streak", kw, value])
        return value
    def saver(*a, **kw):
        value = real_saver(*a, **kw)
        seen.append(["saver", kw, value])
        return value
    class Crash(BaseException):
        pass
    def peer(*a):
        daily = p.load_pick(day, d)
        daily.result = "hit"
        p.save_pick(daily, d)
        return [True]
    def resolve(*a, **kw):
        if path == "poll-final-peer":
            peer()
        return {"pick": result, **({"double_down": result} if path.endswith("double") else {})}
    with contextlib.ExitStack() as stack:
        for target, name, value in [(c,"_now_et",lambda: NOW), (s,"_now_et",lambda: NOW),
                (p,"get_game_statuses_detailed",lambda *a: {}),
                (p,"resolve_daily_slot_results",resolve), (p,"load_streak",load),
                (st,"load_saver_available",saver), (st,"get_game_statuses",lambda *a: {111:"P",778899:"P"}),
                (st,"_load_mdp",lambda: None)]:
            stack.enter_context(patch.object(target,name,value))
        stack.enter_context(patch("bts.posting.format_result_reply", side_effect=lambda r,n: (replies.append([r,n]),"reply")[1]))
        stack.enter_context(patch("bts.posting.reply_to_bluesky", return_value="at://reply"))
        if crash:
            stack.enter_context(patch.object(p,"save_pick",side_effect=Crash("after streak write")))
        if path == "shadow":
            stack.enter_context(patch.object(s,"predict_local_shadow",return_value=object()))
            stack.enter_context(patch.object(s,"heartbeat_watchdog",side_effect=lambda *a,**kw: contextlib.nullcontext()))
            stack.enter_context(patch.object(s,"_analytics_job_status",return_value={}))
            stack.enter_context(patch.object(s,"_update_analytics_job_status",return_value=None))
            def choose(*a,**kw):
                seen.append(["shadow-streak",{},kw["streak"]])
                return SimpleNamespace(pick_result=None)
            stack.enter_context(patch.object(s,"select_pick",side_effect=choose))
            fn = lambda: s._run_shadow_prediction({"orchestrator":{"picks_dir":str(d)}},day,"Prod")
        elif path.startswith("poll"):
            mid = "early" in path
            states = iter(["live","final"] if mid else ["final"])
            stack.enter_context(patch.object(s,"poll_game_result",side_effect=lambda *a: next(states)))
            stack.enter_context(patch.object(s,"_check_hits_midgame",side_effect=peer if path=="poll-early-peer" else lambda *a:[True]))
            stack.enter_context(patch.object(s,"_poll_interval_sleep",return_value=None))
            fn = lambda: s.run_result_polling(111,day,d)
        elif path in ("cli", "cli-callback") or path.startswith("cli-"):
            if path == "cli-callback":
                fn = lambda: c.check_results.callback(date=day,picks_dir=str(d),shadow_status_output=None,
                        wait_deadline_et=None,wait_interval_min=1,allow_stale_scoring=True)
            else:
                def fn():
                    r = CliRunner().invoke(c.cli,["check-results","--date",day,"--allow-stale-scoring","--picks-dir",str(d)],catch_exceptions=False)
                    return {"exit":r.exit_code,"output":r.output}
        elif path == "select":
            real_decide = st.resolve_policy_decision
            def decide(ctx, streak, saver):
                decisions.append([streak,saver])
                return real_decide(ctx,streak,saver)
            stack.enter_context(patch.object(st,"resolve_policy_decision",side_effect=decide))
            fn = lambda: st.select_pick(_predictions([{"p_game_hit":.85}]),day,d,streak=13,for_shadow=True)
        elif path == "update":
            fn = lambda: p.update_streak([True],d,date=day)
        else:
            raise AssertionError(path)
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            ret = outcome(fn)
    return {"outcome":ret,"reads":seen,"replies":replies,"decisions":decisions,
            "stdout":stdout.getvalue(),"stderr":stderr.getvalue()[:3000]}

for label, day in CASES:
    if MODE != "base":
        started=time.monotonic()
        emit({"family":"helper","case":label,"outcome":outcome(lambda:p.season_of_date(day)),"elapsed":time.monotonic()-started})
    for path in ("shadow","poll-void","poll-final-peer","poll-early-peer","cli","cli-callback","select","update"):
        d=new_dir(f"date-{label}-{path}")
        seed(d)
        # Selection must not reuse a locked pick; use for_shadow but real saver branch.
        if path not in ("shadow","select","update"):
            plant(d,day)
        before, pb = raw(d), pick_bytes(d)
        result = drive(path,d,day,"void")
        emit(dict(family="date",case=label,path=path,before=before,after=raw(d),
                  pick_before=pb,pick_after=pick_bytes(d),**result))

CORRUPT = {"brace":b"{","utf8":b"\xff","list":b"[]","null":b"null","number":b"42",
           "deep":b"["*2000+b"]"*2000,"unreadable":b'{"season":2027,"streak":13}'}
for label, payload in CORRUPT.items():
    for path in ("cli-hit","cli-miss","cli-void","cli-double","poll-hit","poll-miss","poll-void",
                 "poll-double","poll-early-hit","poll-final-peer","poll-early-peer","shadow","select"):
        d=new_dir(f"corrupt-{label}-{path}")
        seed(d,raw=payload)
        plant(d,"2027-03-24",double=path.endswith("double"))
        before,pb=raw(d),pick_bytes(d)
        if label=="unreadable":
            (d/"streak.json").chmod(0)
        try:
            result=drive(path,d,"2027-03-24",result="miss" if "miss" in path else "void" if "void" in path else "hit")
        finally:
            (d/"streak.json").chmod(0o600)
        emit(dict(family="corrupt",case=label,path=path,before=before,after=raw(d),pick_before=pb,pick_after=pick_bytes(d),**result))
    for complete in (True,False):
        d=new_dir(f"corrupt-{label}-reconcile-{complete}")
        seed(d,raw=payload)
        plant(d,"2027-03-23","hit" if complete else None)
        before,pb=raw(d),pick_bytes(d)
        if label=="unreadable":
            (d/"streak.json").chmod(0)
        try:
            with patch.object(p,"resolve_daily_slot_results",return_value={"pick":"hit"}):
                result=outcome(lambda:p.reconcile_results(d,clock=lambda:dt.datetime(2027,3,24,2,tzinfo=ET)))
        finally:
            (d/"streak.json").chmod(0o600)
        emit(dict(family="corrupt",case=label,path=f"reconcile-{complete}",before=before,after=raw(d),pick_before=pb,pick_after=pick_bytes(d),outcome=result))

# Literal spelling of the applied marker, dateless writer, saver boundaries.
for spelling in ("2027-3-24","20270324","2027-W12-3"):
    d=new_dir("markers")
    seed(d,season=2027,streak=5,saver_available=True,applied_dates=[])
    steps=[]
    for day in ("2027-03-24",spelling,spelling):
        result=outcome(lambda:p.update_streak([True],d,date=day))
        steps.append(dict(day=day,outcome=result,bytes=raw(d),readers=readers(d)))
    emit(dict(family="markers",case=spelling,steps=steps))
for current in range(9,17):
    d=new_dir("saver")
    seed(d,season=2027,streak=current,saver_available=True,applied_dates=[])
    steps=[]
    for day,results in [("2027-03-24",[False]),("2027-03-24",[False]),(None,[True]),("2027-03-25",[False])]:
        result=outcome(lambda:p.update_streak(results,d,date=day))
        steps.append(dict(day=day,outcome=result,bytes=raw(d),readers=readers(d)))
    emit(dict(family="saver",case=current,steps=steps))

# Actual sequence on one directory for each opening outcome, including a crash/restart.
for opening in ("hit","miss","void","double"):
    d=new_dir(f"sequence-{opening}")
    seed(d,season=2026,streak=12,saver_available=True,applied_dates=[])
    steps=[]
    def step(label,result):
        steps.append(dict(label=label,outcome=result,bytes=raw(d),picks=pick_bytes(d),readers=readers(d)))
    NOW=dt.datetime(2026,12,31,23,tzinfo=ET)
    plant(d,"2026-12-31")
    step("last-2026",drive("cli",d,"2026-12-31","hit"))
    NOW=dt.datetime(2027,2,1,12,tzinfo=ET)
    step("idle-winter",drive("shadow",d,"2027-03-24"))
    NOW=dt.datetime(2027,3,24,23,tzinfo=ET)
    plant(d,"2027-03-24",double=opening=="double")
    step("opening",drive("poll-double" if opening=="double" else "poll-hit",d,"2027-03-24",opening if opening!="double" else "hit"))
    step("second-scorer",drive("cli",d,"2027-03-24","hit"))
    NOW=dt.datetime(2027,3,25,23,tzinfo=ET)
    plant(d,"2027-03-25",double=True)
    step("day-two-crash",drive("cli-double",d,"2027-03-25","hit",crash=True))
    with patch.object(p,"resolve_daily_slot_results",return_value=None):
        step("incomplete-today-replay",outcome(lambda:p.reconcile_results(d,clock=lambda:NOW)))
    step("restart",drive("cli-double",d,"2027-03-25","hit"))
    NOW=dt.datetime(2027,3,26,2,tzinfo=ET)
    with patch.object(p,"resolve_daily_slot_results",return_value={"pick":"hit","double_down":"hit"}):
        step("02-reconcile",outcome(lambda:p.reconcile_results(d,clock=lambda:NOW)))
    emit(dict(family="sequence",case=opening,steps=steps))
OUT.close()
print(MODE,"records",sum(1 for _ in (HERE/f"probe-{MODE}.jsonl").open()),"fixtures",serial)
