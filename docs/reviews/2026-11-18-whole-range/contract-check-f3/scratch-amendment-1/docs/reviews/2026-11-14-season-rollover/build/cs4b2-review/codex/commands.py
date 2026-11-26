"""Independent real CLI, recovery-message, filesystem and operator-sequence probes."""
import contextlib
import datetime as dt
import hashlib
import itertools
import json
import logging
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import review_support as audit
from matrix import OUT, ROOT, NOW, ET, ID, AUTO, SWITCHES, MANUALS, AUTOS, contract, automatic, dated, install

CURRENT = NOW
audit.freeze(NOW)
class Clock(audit.REAL_DATETIME):
    @classmethod
    def now(cls, tz=None):
        return CURRENT.astimezone(tz) if tz else CURRENT.replace(tzinfo=None)
dt.datetime = Clock
label = sys.argv[1] if len(sys.argv)>1 else "branch"
sys.path.insert(0, str(OUT / "base/src" if label == "base" else ROOT / "src"))
from click.testing import CliRunner
import bts.cli as cli
import bts.contest_state as cs
import bts.contest_fetch as cf
import bts.season_contract as sc
import bts.saver_state as saver
import bts.util as util
import bts.leaderboard.auth as auth
from bts.health.contest_state import check
logging.disable(logging.CRITICAL)
RESULTS = {"source":cs.__file__, "commands":[], "status":[], "messages":[], "revoke":[], "sequences":[]}
BASE = OUT / ("commands-" + label + "-files")
BASE.mkdir(exist_ok=True)
TRANSPORTS = []


def directory(name):
    path = BASE / name / 'space \' " quote & picks'
    (path / "account_state").mkdir(parents=True, exist_ok=True)
    return path


def put(picks, name, value):
    install(picks / "account_state" / name, value)


def files(root):
    result = {}
    for path in root.rglob("*"):
        key = str(path.relative_to(root))
        if path.is_symlink():
            result[key] = ["link", os.readlink(path)]
        elif path.is_file():
            try:
                result[key] = ["file", hashlib.sha256(path.read_bytes()).hexdigest()]
            except OSError as exc:
                result[key] = ["unreadable",type(exc).__name__,exc.errno]
        elif path.is_dir():
            result[key] = ["dir"]
        else:
            result[key] = ["special"]
    return result


def command(picks, *args):
    audit.begin("command")
    value = CliRunner().invoke(cli.cli, [*args,"--picks-dir",str(picks)])
    events = audit.end()
    RESULTS["commands"].append(dict(picks=str(picks), args=args, exit=value.exit_code,
                                   output=value.output, exception=repr(value.exception), audit=events))
    return value


def observe(picks, tag):
    try:
        state = cs.load_decision_streak_state(picks, now=CURRENT, require_contest_state=True)
        live = dict(accepted=True, streak=state.streak, status=state.status,
                    saver=state.saver_flag_state, available=state.saver_available, best=state.best_status)
    except Exception as exc:
        live = dict(accepted=False, error=type(exc).__name__, message=str(exc))
    try:
        alerts = [dict(level=a.level,source=a.source,message=a.message) for a in check(picks,now=CURRENT,expected=True)]
    except Exception as exc:
        alerts = [dict(error=type(exc).__name__, message=str(exc))]
    RESULTS["sequences"].append(dict(tag=tag,now=CURRENT.isoformat(),live=live,health=alerts,
                                     flag=saver.load_saver_state(picks,season=CURRENT.astimezone(ET).year).state))
    return live


def transports():
    stack = contextlib.ExitStack()
    def capture(kind, result):
        def fn(*a, **kw):
            TRANSPORTS.append([kind,str(a),str(kw)])
            return result() if callable(result) else result
        return fn
    for target, value in [
        ("bts.leaderboard.auth.load_session_cookies",capture("cookies",{})),
        ("bts.leaderboard.auth.extract_uid",capture("uid","u")),
        ("bts.leaderboard.auth.fetch_login_session",capture("login",SimpleNamespace(user_id=1,username="offline",xsid="x"))),
        ("bts.contest_fetch.fetch_profile",capture("profile",dict(activeStreak=7,seasonBestStreak=9,predictions=[dict(roundId=1,result="hit")]))),
        ("bts.cli._fetch_rounds",capture("rounds",lambda:{1:CURRENT.astimezone(ET).date()-dt.timedelta(days=1)})),
        ("bts.dm.send_dm",capture("dm","captured")),
        ("bts.posting.post_to_bluesky",capture("post","at://captured")),
        ("bts.scheduler.fetch_schedule",capture("schedule",[])),
    ]:
        stack.enter_context(patch(target,value))
    return stack


def statuses():
    picks = directory("status-matrix")
    for sw, manual, auto in itertools.product(SWITCHES,MANUALS,AUTOS):
        put(picks,"season_contract.json",SWITCHES[sw])
        put(picks,"contest_streak.manual.json",MANUALS[manual])
        put(picks,"contest_streak.json",AUTOS[auto])
        for season in [None,2026]:
            before = files(picks.parent)
            result = command(picks,"season-contract","status",*(["--season",str(season)] if season else []))
            events = RESULTS["commands"][-1].pop("audit")
            assert files(picks.parent) == before, (sw,manual,auto,"status changed tree")
            writes = [(e,a) for e,a in events if e in {"os.mkdir","os.rename","os.remove"} or
                      (e=="open" and a[2].isdigit() and int(a[2])&(os.O_CREAT|os.O_WRONLY|os.O_RDWR|os.O_TRUNC))]
            assert not writes, writes
            expected_exit = 1 if sw.startswith("corrupt") else 0
            assert result.exit_code == expected_exit, (sw,manual,auto,result.output,result.exception)
            text = result.output
            if label == "branch":
                try:
                    live = cs.load_decision_streak_state(picks,now=CURRENT,require_contest_state=True)
                    expected = f"live decision now: accepted; source={live.source}; streak={live.streak}; status={live.status}; saver={live.saver_flag_state}"
                except cs.ContestStateError as exc:
                    expected = "live decision now: refused; " + str(exc)
                except Exception as exc:
                    expected = "live decision now: unavailable; " + str(exc)
                assert expected in text, (sw,manual,auto,season,expected,text)
                assert text.index("saver:") < text.index("live decision now:") < text.index("ledger:")
                text = text[:text.index("live decision now:")] + text[text.index("ledger:"):]
            RESULTS["status"].append([sw,manual,auto,season,result.exit_code,text.replace(str(picks),"<P>")])
    # Exceptional saver and a completely absent picks directory are independently covered.
    for scenario in ["unavailable","creation-free"]:
        p = directory(scenario) if scenario == "unavailable" else BASE / scenario / "missing"
        if scenario == "unavailable":
            put(p,"season_contract.json",contract())
            put(p,"contest_streak.json",automatic())
            put(p,"saver_state.json",b'{"n":'+b'9'*5000+b'}')
        before = files(BASE)
        r = command(p,"season-contract","status","--season","2026")
        assert files(BASE)==before
        assert r.exit_code==0
        if label == "branch":
            assert "live decision now: " + ("unavailable" if scenario == "unavailable" else "refused") in r.output


def messages():
    cases = {f"corrupt{n}":(SWITCHES[f"corrupt{n}"],automatic(),None) for n in range(1,9)}
    cases |= {"switch_"+sw:(SWITCHES[sw],automatic(),None) for sw in ["absent","other_season","revoked"]}
    cases |= {"legacy":(contract(),automatic(),MANUALS["legacy"]),
              "manual-auto-schema":(contract(),automatic(),MANUALS["claimed_auto_v2"]),
              "expired-fallback":(contract(),None,MANUALS["expired"]),
              "missing":(contract(),None,None)}
    cases |= {auto:(contract(),AUTOS[auto],None) for auto in
              ["v1","missing_schema","null_schema","number_schema","list_schema","object_schema","unknown_schema","other_season","other_id","both_mismatch"]}
    for name,(sw,auto,manual) in cases.items():
        for alternative in ["dated","automatic"]:
            picks = directory("message-"+name+"-"+alternative)
            put(picks,"season_contract.json",sw)
            put(picks,"contest_streak.json",auto)
            put(picks,"contest_streak.manual.json",manual)
            refusal = observe(picks,"message-before")
            assert not refusal["accepted"]
            message = refusal["message"]
            commands = [shlex.split(x) for x in re.findall(r"`(bts [^`]+)`",message)]
            for cmd in commands:
                assert cmd[cmd.index("--picks-dir")+1] == str(picks), cmd
            assert all(x in message for x in ["TEMPLATE","MLB","another blocker","non-zero fetch exit","not itself a verification"])
            if alternative == "dated" or manual is not None:
                cmd = next(c for c in commands if c[1]=="set-contest-streak")
                cmd = [{"N":"12","B":"14","S":"used"}.get(c,c) for c in cmd]
                result = CliRunner().invoke(cli.cli,cmd[1:])
                assert result.exit_code==0,(cmd,result.output,result.exception)
            else:
                if name.startswith("corrupt"):
                    # Explicit operator removal reveals the next independent blocker.
                    (picks/"account_state/season_contract.json").unlink()
                    assert "switch is absent" in observe(picks,"corrupt-removed")["message"]
                for cmd in commands:
                    if cmd[1] == "set-contest-streak":
                        continue
                    result = CliRunner().invoke(cli.cli,cmd[1:])
                    assert result.exit_code==0,(cmd,result.output,result.exception)
                    observe(picks,"message-step-"+" ".join(cmd[1:3]))
            assert observe(picks,"message-recovered")["accepted"], name
            RESULTS["messages"].append(dict(case=name,alternative=alternative,commands=commands,message=message))


def revoke():
    global CURRENT
    for prior in SWITCHES:
        for present in [False,True]:
            picks = directory("revoke-"+prior+str(present))
            put(picks,"season_contract.json",SWITCHES[prior])
            put(picks,"contest_streak.json",automatic())
            put(picks,"contest_ledger.jsonl",dict(evidence="untouched"))
            status = picks.parent/"health_state/contest_streak_fetch_status.json"
            status.parent.mkdir(exist_ok=True)
            status.write_text("fetch evidence")
            protected = {p:p.read_bytes() for p in [picks/"account_state/contest_streak.json",picks/"account_state/contest_ledger.jsonl",status]}
            if present:
                put(picks,"contest_streak.manual.json",dated())
                put(picks,"saver_state.json",dict(state="used",season=2027,history=["preserve"]))
            r = command(picks,"season-contract","revoke","--season","2027","--reason","  restored  ")
            assert r.exit_code==0,(prior,r.output,r.exception)
            events = RESULTS["commands"][-1]["audit"]
            moves = [a for e,a in events if e=="os.rename"]
            assert "season_contract.json" in moves[0][1]
            assert len(moves)==(3 if present else 1)
            if present:
                assert "contest_streak.manual" in moves[1][0] and "saver_state" in moves[2][0]
            data = sc.switch_state(picks,CURRENT).snapshot
            assert data["note"]=="  restored  " and data["schema_version"]==sc.SCHEMA
            assert (data["activation_id"]==ID)==(prior in ["automatic","revoked"])
            assert data["switched_at"]==(contract()["switched_at"] if prior in ["automatic","revoked"] else None)
            assert all(p.read_bytes()==v for p,v in protected.items())
            assert saver.load_saver_state(picks,season=2027).state=="uninitialized"
            assert "switch is revoked" in observe(picks,"revoke-success")["message"]
            status_result = command(picks,"season-contract","status")
            assert "live decision now: refused" in status_result.output
            CURRENT += dt.timedelta(microseconds=1)
            again = command(picks,"season-contract","revoke","--season","2027","--reason","again")
            assert again.exit_code==0
            assert sc.switch_state(picks,CURRENT).snapshot["activation_id"]==data["activation_id"]
            RESULTS["revoke"].append(dict(prior=prior,present=present,snapshot=data,tree=files(picks.parent)))
    CURRENT = NOW
    for step in ["switch","manual","saver"]:
        failures = ["permission","directory"] if step=="switch" else ["permission","file","directory","symlink","dangling","fifo","source_lstat","target_lstat"]
        for failure in failures:
            picks=directory("revoke-fail-"+step+"-"+failure)
            put(picks,"season_contract.json",contract())
            put(picks,"contest_streak.manual.json",dated())
            put(picks,"saver_state.json",dict(state="used",season=2027))
            put(picks,"contest_streak.json",automatic())
            put(picks,"contest_ledger.jsonl",dict(protected=True))
            protected={name:(picks/"account_state"/name).read_bytes() for name in ["contest_streak.json","contest_ledger.jsonl"]}
            prefix="contest_streak.manual" if step=="manual" else "saver_state"
            target=picks/"account_state"/(prefix+".revoked-20270715T160000000000Z.json")
            source=picks/"account_state"/(prefix+".json")
            switch=picks/"account_state/season_contract.json"
            with contextlib.ExitStack() as stack:
                if failure=="permission":
                    if step=="switch":
                        stack.enter_context(patch.object(util,"atomic_write_text",side_effect=PermissionError("readonly directory")))
                    else:
                        rename=os.rename
                        def blocked(a,b,*,_source=source):
                            if a==_source: raise PermissionError("readonly directory")
                            return rename(a,b)
                        stack.enter_context(patch.object(os,"rename",blocked))
                elif step=="switch" and failure=="directory":
                    switch.unlink(); switch.mkdir()
                elif failure=="file": target.write_text("keep")
                elif failure=="directory": target.mkdir()
                elif failure=="symlink": target.symlink_to(source)
                elif failure=="dangling": target.symlink_to("missing")
                elif failure=="fifo": os.mkfifo(target)
                elif failure.endswith("lstat"):
                    original=Path.lstat
                    bad=source if failure=="source_lstat" else target
                    def stat(path,*a,**kw):
                        if path==bad: raise PermissionError("injected lstat")
                        return original(path,*a,**kw)
                    stack.enter_context(patch.object(Path,"lstat",stat))
                before=files(picks.parent)
                r=command(picks,"season-contract","revoke","--season","2027","--reason","restore")
            assert r.exit_code==1,(step,failure,r.output,r.exception)
            if step=="switch":
                assert "NOT revoked" in r.output and files(picks.parent)==before
            elif step=="manual":
                assert "saver flag was not attempted" in r.output
                assert source.exists() and (picks/"account_state/saver_state.json").exists()
            else:
                assert "manual file outcome" in r.output and "saver flag was not set aside" in r.output
                assert not (picks/"account_state/contest_streak.manual.json").exists() and source.exists()
            for name,content in protected.items(): assert (picks/"account_state"/name).read_bytes()==content
            RESULTS["revoke"].append(dict(step=step,failure=failure,exit=r.exit_code,output=r.output,tree=files(picks.parent)))
    # Sources are moved as directory entries, including a dangling symlink.
    for kind in ["dangling","directory"]:
        picks=directory("source-"+kind)
        source=picks/"account_state/contest_streak.manual.json"
        source.symlink_to("missing") if kind=="dangling" else source.mkdir()
        r=command(picks,"season-contract","revoke","--season","2027","--reason","x")
        assert r.exit_code==0
        target=source.with_name("contest_streak.manual.revoked-20270715T160000000000Z.json")
        assert target.is_symlink() if kind=="dangling" else target.is_dir()
    # Usage checks must not read application state.
    for args in [[],["--season","2026","--reason","x"],["--season","2027"],["--season","2027","--reason"," \t "]]:
        picks=directory("usage-"+str(len(RESULTS["commands"])))
        r=command(picks,"season-contract","revoke",*args)
        assert r.exit_code==2 and "No such command" not in r.output
        assert not [a for e,a in RESULTS["commands"][-1]["audit"] if e=="open" and str(picks) in a[0]]
    # Crossing ET midnight during the write cannot change the operation's year/stamp.
    picks=directory("revoke-midnight")
    CURRENT=NOW.replace(month=12,day=31,hour=23,minute=59,second=59,microsecond=999999)
    entry=CURRENT
    put(picks,"season_contract.json",contract())
    put(picks,"contest_streak.manual.json",dated("2027-12-31"))
    put(picks,"saver_state.json",dict(state="used",season=2027))
    writer=util.atomic_write_text
    def cross(path,text):
        global CURRENT
        CURRENT=entry+dt.timedelta(microseconds=1)
        return writer(path,text)
    with patch.object(util,"atomic_write_text",cross):
        r=command(picks,"season-contract","revoke","--season","2027","--reason","midnight")
    assert r.exit_code==0 and "--date 2027-12-31" in r.output
    data=json.loads((picks/"account_state/season_contract.json").read_text())
    assert data["season"]==2027 and data["revoked_at"]==entry.astimezone(dt.timezone.utc).isoformat().replace("+00:00","Z")
    assert (picks/"account_state/contest_streak.manual.revoked-20280101T045959999999Z.json").exists()
    CURRENT=NOW


def sequences():
    global CURRENT
    for legacy in [False,True]:
        picks=directory("season-sequence-"+str(legacy))
        put(picks,"contest_streak.json",automatic(schema_version="bts_contest_streak_auto_v1",source_date="2026-09-30"))
        put(picks,"saver_state.json",dict(state="active",season=2026))
        if legacy: put(picks,"contest_streak.manual.json",MANUALS["legacy"])
        assert not observe(picks,"start-day1")["accepted"]
        assert command(picks,"set-contest-streak","--date","2027-07-15","--streak","0","--best-streak","0","--saver","not_earned").exit_code==0
        assert observe(picks,"dated-day1")["accepted"]
        CURRENT=NOW+dt.timedelta(days=1)
        assert not observe(picks,"day2-before-observation")["accepted"]
        assert command(picks,"fetch-contest-streak").exit_code==0
        assert not observe(picks,"evidence-only")["accepted"]
        command(picks,"season-contract","status")
        assert command(picks,"season-contract","enable","--season","2027").exit_code==0
        assert not observe(picks,"enabled-before-fetch")["accepted"]
        assert command(picks,"fetch-contest-streak").exit_code==0
        assert observe(picks,"switched-fetch")["accepted"]
        saved_switch=(picks/"account_state/season_contract.json").read_bytes()
        saved_auto=(picks/"account_state/contest_streak.json").read_bytes()
        (picks/"account_state/season_contract.json").write_bytes(saved_switch)
        (picks/"account_state/contest_streak.json").write_bytes(saved_auto)
        assert observe(picks,"restore-without-revoke-limit")["accepted"]
        assert command(picks,"season-contract","revoke","--season","2027","--reason","restore").exit_code==0
        assert not observe(picks,"restore-revoked")["accepted"]
        assert command(picks,"set-contest-streak","--date","2027-07-16","--streak","7","--best-streak","9","--saver","used").exit_code==0
        assert observe(picks,"restore-dated")["accepted"]
        assert command(picks,"season-contract","enable","--season","2027").exit_code==0
        assert observe(picks,"restore-enabled-still-dated")["accepted"]
        CURRENT += dt.timedelta(days=1)
        assert "activation_id" in observe(picks,"restore-next-day-old-id")["message"]
        before=(picks/"account_state/contest_streak.json").read_bytes()
        atomic=cli._atomic_write_json
        def fail_before(path,data):
            if path.name=="contest_streak.json": raise PermissionError("before observation")
            return atomic(path,data)
        with patch.object(cli,"_atomic_write_json",fail_before):
            assert command(picks,"fetch-contest-streak").exit_code==1
        assert (picks/"account_state/contest_streak.json").read_bytes()==before
        assert not observe(picks,"failed-before-replace")["accepted"]
        def fail_after(path,data):
            if path.name=="contest_streak_fetch_status.json": raise PermissionError("after observation")
            return atomic(path,data)
        with patch.object(cli,"_atomic_write_json",fail_after):
            assert command(picks,"fetch-contest-streak").exit_code==1
        assert observe(picks,"failed-after-replace")["accepted"]
        assert command(picks,"fetch-contest-streak").exit_code==0
        assert observe(picks,"restore-fetched")["accepted"]
        # Restoring an old observation cannot match a fresh activation.
        (picks/"account_state/contest_streak.json").write_bytes(saved_auto)
        assert not observe(picks,"restored-old-auto-only")["accepted"]
        assert command(picks,"fetch-contest-streak").exit_code==0
        CURRENT=NOW.replace(month=12,day=31,hour=23,minute=59,second=59,microsecond=999999)
        assert observe(picks,"last-microsecond-2027")["accepted"]
        CURRENT += dt.timedelta(microseconds=1)
        assert "other_season" in observe(picks,"first-microsecond-2028")["message"]
        assert command(picks,"season-contract","enable","--season","2028").exit_code==0
        assert not observe(picks,"january-enabled")["accepted"]
        assert command(picks,"fetch-contest-streak").exit_code==0
        assert observe(picks,"january-fetched")["accepted"]
        CURRENT=NOW


with transports():
    statuses()
    if label == "branch":
        messages()
        revoke()
        sequences()
RESULTS["transports"]=TRANSPORTS
(OUT/("commands-"+label+".json")).write_text(json.dumps(RESULTS,indent=2,default=str))
print(label,{k:len(v) for k,v in RESULTS.items() if isinstance(v,list)})
