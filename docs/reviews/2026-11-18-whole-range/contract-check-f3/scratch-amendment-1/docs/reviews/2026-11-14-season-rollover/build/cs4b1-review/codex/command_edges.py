"""Offline filesystem, clock, persistence, status, reader and operator probes."""
import collections
import contextlib
import dataclasses
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from unittest.mock import patch

import review_support as support
from review_support import *

RESULTS = []
FAILURES = []


def check(name, fn):
    try:
        detail = fn()
        RESULTS.append({"name":name,"ok":True,"detail":detail})
    except Exception as exc:
        import traceback
        FAILURES.append({"name":name,"exception":repr(exc),"traceback":traceback.format_exc()})


def fresh(name):
    root = ROOT / "edge-cases" / name
    if root.exists(): shutil.rmtree(root)
    root.mkdir(parents=True)
    return root,root/"picks"


def switch_inputs():
    out = [("absent",None,"absent",None), ("valid",contract(),"automatic",None),
           ("revoked",contract(status="revoked",revoked_at="2027-01-01T00:00:00Z"),"revoked",None),
           ("other",contract(season=2026),"other_season",None)]
    checks={"schema_version":2,"season":3,"status":4,"activation_id":5,"switched_at":6,"revoked_at":7,"note":8}
    for field,n in checks.items():
        if field != "note":
            out.append((field+"-missing",{k:v for k,v in contract().items() if k != field},"corrupt",n))
        for i,v in enumerate([None,True,False,0,1.5,[],{},"", "bad"]):
            valid=(field=="note" and (v is None or isinstance(v,str))) or (field=="season" and type(v) is int) or (field=="revoked_at" and v is None)
            value=contract(**{field:v})
            out.append((field+"-"+str(i),value,"other_season" if valid and field=="season" else "automatic" if valid else "corrupt",None if valid else n))
    for field,n in [("switched_at",6),("revoked_at",7)]:
        for status in ["automatic","revoked"]:
            for i,v in enumerate([None,"2027-07-15T12:00:00", "2027-07-15T12:00:00-04:00", "2027-07-15T16:00:00Z",
                                  "20270715T160000+00:00", "2027-07-15X12:00:00+04:30", "2027-07-15T12:00:00+25:00"]):
                valid=(i in {2,3,4,5}) or (i==0 and status != ("automatic" if n==6 else "revoked"))
                value=contract(status=status,revoked_at="2027-07-15T16:00:00Z",**({field:v} if field!="revoked_at" else {}))
                value[field]=v
                out.append((field+status+str(i),value,status if valid else "corrupt",None if valid else n))
    for i,ident in enumerate(["a"*31,"a"*33,"A"*32,"g"*32,"a"*31+"\n","０"*32,"0"*32]):
        out.append(("ident"+str(i),contract(activation_id=ident),"automatic" if i==6 else "corrupt",None if i==6 else 5))
    out += [("nonobject",[],"corrupt",1),("utf8",b"\xff","corrupt",1),
            ("huge",b"1"*5000,"corrupt",1),("deep",b"["*3000+b"]"*3000,"corrupt",1),
            ("nan",b'NaN',"corrupt",1),("nan_note",contract(note=float("nan")),"corrupt",8),
            ("extra_nan",contract(extra=float("nan")),"automatic",None),
            ("duplicate_valid_last",b'{"schema_version":null,'+json.dumps(contract())[1:].encode(),"automatic",None),
            ("duplicate_bad_last",json.dumps(contract()).encode()[:-1]+b',"schema_version":null}',"corrupt",2)]
    for kind in ["directory","dangling","loop","unreadable","lstat"]:
        out.append((kind,kind,"corrupt",1))
    return out


def seed_kind(picks,name,value):
    path=picks/"account_state"/name
    path.parent.mkdir(parents=True,exist_ok=True)
    if value == "directory": path.mkdir()
    elif value == "dangling": path.symlink_to(path.parent/"missing")
    elif value == "loop": path.symlink_to(path.name)
    elif value == "unreadable": path.write_text("{}"); path.chmod(0)
    elif value == "lstat": path.write_text("{}")
    else: put(picks,name,value)
    return path


def switches(h):
    from bts.season_contract import switch_state
    seen=set()
    for name,value,want,n in switch_inputs():
        def probe():
            root,picks=fresh("switch-"+name)
            path=None if value is None else seed_kind(picks,"season_contract.json",value)
            original=Path.lstat
            def lstat(p,*a,**kw):
                if p==path: raise PermissionError("audit lstat denied")
                return original(p,*a,**kw)
            with patch.object(Path,"lstat",lstat) if name=="lstat" else contextlib.nullcontext():
                state=switch_state(picks,NOW)
                assert (state.state,state.check)==(want,n),(state,want,n)
                status=h.invoke(root,picks,["season-contract","status"])
                assert status["exit"]==(1 if want=="corrupt" else 0),status
                assert status["before"]==status["after"]
                assert all(x in status["output"] for x in ["manual:","automatic:","saver:","ledger:"])
            before=snapshot(root)
            enabled=h.invoke(root,picks,["season-contract","enable","--season","2027","--note","new"])
            if name=="directory":
                assert enabled["exit"]==1 and "NOT switched" in enabled["output"]
            elif want=="automatic":
                assert enabled["exit"]==0 and before==enabled["after"] and "unchanged" in enabled["output"]
            else:
                assert enabled["exit"]==0,enabled
                installed=json.loads((picks/"account_state/season_contract.json").read_text())
                assert installed["season"]==2027 and installed["note"]=="new"
                assert installed["activation_id"] not in seen
                seen.add(installed["activation_id"])
                assert installed["revoked_at"]==(value["revoked_at"] if want=="revoked" else None)
                assert installed["switched_at"]=="2027-07-15T16:00:00Z"
                assert all("season_contract.json" in str(e) or ".tmp" in str(e) for e in enabled["trace"] if e[0]=="open"), enabled["trace"]
                again=h.invoke(root,picks,["season-contract","enable","--season","2027","--note","ignore"])
                assert again["exit"]==0 and again["before"]==again["after"]
            if path and name=="unreadable" and path.exists(): path.chmod(0o600)
            return {"state":dataclasses.asdict(state),"status":status,"enable":enabled}
        check("switch:"+name,probe)


def statuses(h):
    names=["season_contract.json","contest_streak.manual.json","contest_streak.json","saver_state.json","contest_ledger.jsonl"]
    good=[contract(),{"active_streak":3},auto(),{"season":2027,"state":"used"},b'{}\n']
    for combo in itertools.product(range(3),repeat=5):
        def probe():
            root,picks=fresh("status-"+"".join(map(str,combo)))
            for name,v,kind in zip(names,good,combo):
                if kind: put(picks,name,v if kind==1 else b"\xff")
            for season in [2027,2026]:
                before=snapshot(root)
                def deny(*_): raise AssertionError("status attempted mutation")
                # Actual audit hooks cover all open variants, mkdir and rename;
                # explicit mkdir guard also detects exist_ok no-op attempts.
                with patch.object(Path,"mkdir",deny):
                    r=h.invoke(root,picks,["season-contract","status","--season",str(season)])
                assert r["exit"]==(1 if combo[0]==2 else 0),r
                assert before==r["after"]
                assert not any(e[0]!="open" or int(e[-1]) & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND) for e in r["trace"]),r["trace"]
                assert all(x in r["output"] for x in ["ET date:","switch:","manual:","automatic:","saver:","ledger:"])
            return r
        check("status-combination:"+str(combo),probe)
    for filename in names[1:]:
        for kind in ["directory","dangling","loop","unreadable","lstat",b"["*3000+b"]"*3000,b"1"*5000,[],b'NaN']:
            def probe():
                root,picks=fresh("status-edge")
                put(picks,"season_contract.json",contract(status="revoked",revoked_at="2027-07-15T00:00:00Z"))
                path=seed_kind(picks,filename,kind)
                orig=Path.lstat
                def denied(p,*a,**kw):
                    if p==path: raise PermissionError("injected")
                    return orig(p,*a,**kw)
                with patch.object(Path,"lstat",denied) if kind=="lstat" else contextlib.nullcontext():
                    r=h.invoke(root,picks,["season-contract","status"])
                assert r["exit"]==0,r
                assert r["before"]==r["after"]
                assert all(x in r["output"] for x in ["manual:","automatic:","saver:","ledger:"])
                if kind=="unreadable": path.chmod(0o600)
                return r
            check("status-edge:"+filename+":"+str(kind)[:25],probe)
    def ledger_and_comparison():
        root,picks=fresh("ledger-lines")
        put(picks,"season_contract.json",contract(status="revoked",revoked_at="2027-07-15T00:00:00Z"))
        put(picks,"contest_streak.json",auto())
        rows=[{"active_streak":i} for i in range(12)]
        content="\n[]\nnot-json\n"+"1"*5000+"\n"+"["*3000+"]"*3000+"\n"+"\n".join(map(json.dumps,rows))
        put(picks,"contest_ledger.jsonl",content.encode())
        r=h.invoke(root,picks,["season-contract","status"])
        assert r["exit"]==0 and "bad lines: 5" in r["output"],r
        assert "identifier matches switch: True (comparison only)" in r["output"]
        displayed=[x for x in r["output"].splitlines() if x.startswith("  recorded_at=")]
        assert len(displayed)==10
        assert [int(x.split("active_streak=")[1].split(";")[0]) for x in displayed]==list(range(2,12))
        return r
    check("status-ledger-and-revoked-comparison",ledger_and_comparison)


def persistence(h):
    states=["absent","other","revoked","corrupt","automatic"]
    for state,stage in itertools.product(states,["observation","status","ledger","ledger_partial","saver_lock","saver_write","saver_audit"]):
        def probe():
            root,picks=fresh("fail-"+state+stage)
            if state!="absent": put(picks,"season_contract.json",contract(season=2026 if state=="other" else 2027,
                status="revoked" if state=="revoked" else "automatic",revoked_at="2027-07-14T00:00:00Z",note=[] if state=="corrupt" else None))
            path=put(picks,"contest_streak.json",auto(schema_version="bts_contest_streak_auto_v1",active_streak=2))
            before=path.read_bytes()
            orig_write=h.cli._atomic_write_json
            def writer(path,value):
                if stage==("observation" if path.name=="contest_streak.json" else "status"):
                    raise OSError("write "+stage)
                return orig_write(path,value)
            orig_open=__import__("builtins").open
            orig_path_open=Path.open
            class Partial:
                def __init__(self,fh): self.fh=fh
                def __enter__(self): return self
                def __exit__(self,*_): self.fh.close()
                def write(self,s):
                    self.fh.write(s[:7]);self.fh.flush();raise OSError("partial")
            def pathopen(path,mode="r",*a,**kw):
                if path.name=="contest_ledger.jsonl" and mode=="a":
                    if stage=="ledger": raise OSError("ledger")
                    if stage=="ledger_partial": return Partial(orig_path_open(path,mode,*a,**kw))
                return orig_path_open(path,mode,*a,**kw)
            def builtin(path,mode="r",*a,**kw):
                name=Path(path).name if isinstance(path,(str,Path)) else ""
                if (stage=="saver_lock" and name=="saver_state.lock") or (stage=="saver_audit" and name=="saver_transitions.jsonl"):
                    raise OSError(stage)
                return orig_open(path,mode,*a,**kw)
            orig_save_write=h.saver.atomic_write_text
            def savewrite(*a,**kw):
                if stage=="saver_write": raise OSError(stage)
                return orig_save_write(*a,**kw)
            with contextlib.ExitStack() as st:
                st.enter_context(patch.object(h.cli,"_atomic_write_json",writer))
                st.enter_context(patch.object(Path,"open",pathopen))
                st.enter_context(patch("builtins.open",builtin))
                st.enter_context(patch.object(h.saver,"atomic_write_text",savewrite))
                r=h.fetch(root,picks)
            reached=["observation","status","ledger","ledger_partial","saver_lock","saver_write"] if state=="automatic" else ["status"] if state=="corrupt" else ["status","ledger","ledger_partial"]
            want=1 if stage in reached else 2 if state=="corrupt" else 0
            assert r["exit"]==want,r
            assert (path.read_bytes()!=before)==(state=="automatic" and stage!="observation")
            ledger=picks/"account_state/contest_ledger.jsonl"
            assert ledger.exists()==(state!="corrupt" and (state!="automatic" or stage!="observation") and stage not in {"status","ledger"})
            if stage=="ledger_partial" and ledger.exists(): assert len(ledger.read_bytes())==7
            assert len(r["dms"])==(1 if state=="corrupt" else 0)
            if state!="automatic": assert not r["earn"]
            return r
        check("persistence:"+state+":"+stage,probe)
    for point in ["mkdir","mkstemp","fsync","replace","output"]:
        def probe():
            root,picks=fresh("enable-fail"+point)
            path=put(picks,"season_contract.json",contract(status="revoked",revoked_at="2027-07-14T00:00:00Z"))
            old=path.read_bytes()
            import tempfile
            target={"mkdir":(Path,"mkdir"),"mkstemp":(tempfile,"mkstemp"),"fsync":(os,"fsync"),"replace":(os,"replace"),"output":(h.cli.click,"echo")}[point]
            with patch.object(*target,side_effect=OSError("injected")):
                r=h.invoke(root,picks,["season-contract","enable","--season","2027"])
            assert r["exit"]==1,r
            assert (path.read_bytes()==old)==(point!="output")
            again=h.invoke(root,picks,["season-contract","enable","--season","2027"])
            assert again["exit"]==0
            if point=="output": assert "already switched" in again["output"]
            return r
        check("enable-failure:"+point,probe)
    def wrong_year():
        root,picks=fresh("wrong-year")
        put(picks,"season_contract.json",b"\xff")
        r=h.invoke(root,picks,["season-contract","enable","--season","2026"])
        assert r["exit"]==2 and not r["trace"] and r["before"]==r["after"],r
        return r
    check("wrong-year-before-state-io",wrong_year)


def clocks(h):
    entry=REAL_DT(2027,12,31,23,59,59,tzinfo=ET)
    later=REAL_DT(2028,1,1,0,0,1,tzinfo=ET)
    for state,phase,source_year in itertools.product(["automatic","absent","revoked","corrupt"],
            ["auth","profile","rounds","write:contest_streak.json","write:contest_streak_fetch_status.json","ledger","saver_write"],[2027,2028]):
        def probe():
            root,picks=fresh("clock")
            if state!="absent": put(picks,"season_contract.json",contract(status="revoked" if state=="revoked" else "automatic",revoked_at="2027-07-14T00:00:00Z",note=[] if state=="corrupt" else None))
            h.current,h.phase,h.cross=entry,later,phase
            h.rounds={1:REAL_DATE(source_year,12,31) if source_year==2027 else REAL_DATE(source_year,1,1)}
            orig_open=Path.open
            orig_save=h.saver.atomic_write_text
            def op(path,mode="r",*a,**kw):
                if path.name=="contest_ledger.jsonl" and mode=="a": h.event("ledger")
                return orig_open(path,mode,*a,**kw)
            def save(*a,**kw): h.event("saver_write");return orig_save(*a,**kw)
            with patch.object(Path,"open",op),patch.object(h.saver,"atomic_write_text",save): r=h.fetch(root,picks)
            assert r["exit"]==(2 if state=="corrupt" else 0),r
            if state!="corrupt":
                ledger=json.loads((picks/"account_state/contest_ledger.jsonl").read_text())
                assert ledger["season"]==2027 and ledger["recorded_at"]=="2028-01-01T04:59:59Z"
                if state=="automatic":
                    obs=json.loads((picks/"account_state/contest_streak.json").read_text())
                    assert obs["season"]==2027 and obs["recorded_at"]==ledger["recorded_at"]
                assert len(r["earn"])==(1 if state=="automatic" and source_year==2027 else 0)
                if r["earn"]: assert r["earn"][0]["season"]==2027
            h.current,h.cross=NOW,None
            return r
        check("clock:"+str((state,phase,source_year)),probe)
    for phase in ["read","write"]:
        def probe():
            root,picks=fresh("enable-clock")
            put(picks,"season_contract.json",contract(status="revoked",revoked_at="2027-07-14T00:00:00Z"))
            h.current=entry
            orig=Path.read_text
            import bts.util as util
            old=util.atomic_write_text
            def read(*a,**kw):
                if phase=="read": h.current=later
                return orig(*a,**kw)
            def write(*a,**kw):
                if phase=="write": h.current=later
                return old(*a,**kw)
            with patch.object(Path,"read_text",read),patch.object(util,"atomic_write_text",write): r=h.invoke(root,picks,["season-contract","enable","--season","2027"])
            value=json.loads((picks/"account_state/season_contract.json").read_text())
            assert r["exit"]==0 and value["season"]==2027 and value["switched_at"]=="2028-01-01T04:59:59Z",r
            h.current=NOW
            return r
        check("enable-clock:"+phase,probe)
    h.current,h.cross=NOW,None
    h.rounds={1:REAL_DATE(2027,7,14)}


def sequence(h):
    root,picks=fresh("operator-sequence")
    path=put(picks,"contest_streak.json",auto(schema_version="bts_contest_streak_auto_v1",active_streak=12,best_streak=18,source_date="2026-09-27"))
    put(picks,"saver_state.json",{"season":2026,"state":"used"})
    steps=[]
    def record(label,r=None):
        decision=h.cs.load_decision_streak_state(picks,now=h.current,require_contest_state=True)
        alerts=h.health.check(picks,now=h.current,expected=True)
        steps.append(dict(step=label,now=str(h.current),command=r,decision=dataclasses.asdict(decision),health=[dataclasses.asdict(a) for a in alerts],files=snapshot(root)))
    record("initial-last-season-auto")
    frozen=path.read_bytes()
    for day in [15,16,17]:
        h.current=REAL_DT(2027,7,day,12,tzinfo=ET)
        r=h.invoke(root,picks,["set-contest-streak","--date",f"2027-07-{day}","--streak","3","--best-streak","8","--saver","used"])
        assert r["exit"]==0,r
        r=h.fetch(root,picks)
        assert r["exit"]==0 and path.read_bytes()==frozen
        record("daily-observation-and-unswitched-fetch",r)
        assert steps[-1]["decision"]["streak"]==3 and steps[-1]["decision"]["saver_flag_state"]=="used"
    record("status-before-enable",h.invoke(root,picks,["season-contract","status"]))
    r=h.invoke(root,picks,["season-contract","enable","--season","2027"])
    assert r["exit"]==0
    record("enable",r)
    r=h.fetch(root,picks);assert r["exit"]==0;record("first-automatic-fetch",r)
    orig=h.cli._atomic_write_json
    for stage in ["contest_streak.json","contest_streak_fetch_status.json"]:
        h.profile={"activeStreak":8,"seasonBestStreak":10,"predictions":[{"roundId":1,"result":"hit"}]}
        def fail(p,v):
            if p.name==stage:raise OSError("sequence failure")
            return orig(p,v)
        with patch.object(h.cli,"_atomic_write_json",fail): r=h.fetch(root,picks)
        assert r["exit"]==1
        record("failure-at-"+stage,r)
    switch=picks/"account_state/season_contract.json"
    value=json.loads(switch.read_text());first=value["activation_id"]
    value.update(status="revoked",revoked_at="2027-07-17T16:00:00Z");switch.write_text(json.dumps(value))
    prior=path.read_bytes();r=h.fetch(root,picks);assert r["exit"]==0 and path.read_bytes()==prior
    record("hand-revoke-and-fetch",r)
    r=h.invoke(root,picks,["season-contract","enable","--season","2027"]);assert r["exit"]==0
    assert json.loads(switch.read_text())["activation_id"]!=first
    record("reenable",r)
    r=h.invoke(root,picks,["season-contract","status"]);assert "identifier matches switch: False" in r["output"]
    record("status-before-new-fetch",r)
    h.current=REAL_DT(2027,7,18,12,tzinfo=ET)
    record("next-day-before-matching-fetch")
    assert steps[-1]["decision"]["streak"]==8
    r=h.fetch(root,picks);assert r["exit"]==0;record("matching-fetch",r)
    h.current=REAL_DT(2028,1,1,0,0,1,tzinfo=ET)
    prior=path.read_bytes();r=h.fetch(root,picks)
    assert r["exit"]==0 and path.read_bytes()==prior
    record("first-january-1-fetch",r)
    assert steps[-1]["decision"]["streak"]==8 and steps[-1]["decision"]["saver_flag_state"]=="uninitialized"
    h.current=NOW
    write_json(ROOT/"operator-sequence.json",steps)
    return len(steps)


def reader_worker(label,source):
    h=Harness(source)
    import dataclasses
    def capture(fn):
        try:
            result=fn()
            def normalize(value):
                if dataclasses.is_dataclass(value):
                    value=dataclasses.asdict(value)
                    for k in ["season","activation_id"]:
                        if value.get(k) is None:value.pop(k,None)
                if isinstance(value,dict):return {k:normalize(v) for k,v in value.items()}
                if isinstance(value,(tuple,list)):return [normalize(v) for v in value]
                if isinstance(value,(Path,REAL_DATE,REAL_DT)):return str(value).replace(str(root),"<case>")
                return value
            return ["ok",normalize(result)]
        except Exception as exc:return ["error",type(exc).__name__,str(exc).replace(str(root),"<case>")]
    schemas=["missing",None,"", "unknown",[],{},12,True,"bts_contest_streak_auto_v1","bts_contest_streak_auto_v2","bts_contest_streak_manual_v3"]
    variants=[{}, {"season":None,"activation_id":None},{"season":True},{"activation_id":"A"*32},
              {"active_streak":True},{"active_streak":-1},{"active_streak":58},{"best_streak":58},
              {"best_streak":3},{"best_streak":True},{"saver_available":"bad"},{"source_date":None,"recorded_at":None},
              {"source_date":"2026-09-27"},{"source_date":"2028-01-01"},{"source":"other"},
              {"override_expires_at":"2027-08-01T00:00:00Z"},{"override_expires_at":"2027-01-01T00:00:00Z"}]
    records=[]
    for schema,variant,filename,manual_mode,flag in itertools.product(schemas,variants,["contest_streak.json","contest_streak.manual.json"],["absent","override","expired","dated_today","dated_yesterday"],[False,True]):
        root,picks=fresh("reader-"+label)
        data=auto(**variant)
        if schema=="missing":data.pop("schema_version")
        else:data["schema_version"]=schema
        path=put(picks,filename,data)
        if filename=="contest_streak.json" and manual_mode!="absent":
            if manual_mode in {"override","expired"}:manual={"active_streak":2,"override_expires_at":"2027-08-01T00:00:00Z" if manual_mode=="override" else "2027-01-01T00:00:00Z"}
            else:
                day=15 if manual_mode=="dated_today" else 14
                manual={"schema_version":"bts_contest_streak_manual_v3","season":2027,"effective_date":f"2027-07-{day}","active_streak":3,"best_streak":9,"saver_state":"used","source":"manual_cli","recorded_at":"2027-07-15T16:00:00Z","expires_at":f"2027-07-{day+1}T03:59:59.999999Z"}
            put(picks,"contest_streak.manual.json",manual)
        put(picks,"season_contract.json",b"\xff")
        def switch_guard(event,args):
            if event=="open" and str(args[0]).endswith("season_contract.json"):
                raise AssertionError("existing reader opened switch")
        # A single audit hook per worker with ACTIVE is enough for tracing; guard
        # the actual file-open function for the reader's switch prohibition.
        orig=Path.open
        def op(p,*a,**kw):
            assert p.name!="season_contract.json"
            return orig(p,*a,**kw)
        with patch.object(Path,"open",op):
            values=[capture(lambda:h.cs._parse_state_file(path,require_dated=flag)),
                    capture(lambda:h.cs.load_contest_streak_state(picks,now=NOW,require_dated_auto=flag)),
                    capture(lambda:h.cs.load_decision_streak_state(picks,now=NOW,require_contest_state=flag)),
                    capture(lambda:h.health.check(picks,now=NOW,expected=flag))]
        records.append({"case":[schema,variant,filename,manual_mode,flag],"values":values})
    write_json(ROOT/("reader-"+label+".json"),records)
    print("readers",label,len(records),flush=True)


def main():
    if len(sys.argv)>1:
        reader_worker(sys.argv[1],Path(sys.argv[2]));return
    h=Harness(REPO/"src")
    switches(h);statuses(h);persistence(h);clocks(h)
    check("operator-sequence",lambda:sequence(h))
    write_json(ROOT/"command-edge-results.json",RESULTS)
    write_json(ROOT/"command-edge-failures.json",FAILURES)
    print("edge checks",len(RESULTS),"failures",len(FAILURES),flush=True)
    for label,source in [("base",ROOT/"base/src"),("branch",REPO/"src")]:
        subprocess.run([sys.executable,__file__,label,str(source)],check=True)
    a=json.loads((ROOT/"reader-base.json").read_text());b=json.loads((ROOT/"reader-branch.json").read_text())
    diff=collections.Counter();unexpected=[]
    for av,bv in zip(a,b,strict=True):
        assert av["case"]==bv["case"]
        if av["values"]==bv["values"]:diff["same"]+=1;continue
        schema,variant,filename,_,_=av["case"]
        if schema=="bts_contest_streak_auto_v2":diff["row7_v2"]+=1
        else:unexpected.append({"before":av,"after":bv})
    write_json(ROOT/"reader-comparison.json",dict(cells=len(a),groups=diff,unexpected=unexpected))
    print("reader differences",dict(diff),"unexpected",len(unexpected),flush=True)


if __name__=="__main__":main()
