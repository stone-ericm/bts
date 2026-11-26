"""Compare CLI preservation and exercise further repairs, restored legacy files,
real permission failures, archive collisions and protected fetch evidence."""
import contextlib
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shlex
import sys
from unittest.mock import patch
import review_support as audit

OUT=Path(__file__).resolve().parent
loader=(OUT/"commands_v2.py").read_text().split("\ntry:\n    with namespace")[0]
sys.argv=[__file__,"branch"]
exec(compile(loader,str(OUT/"commands_v2.py"),"exec"))
n=namespace
n["BASE"]=OUT/"remaining-files"
n["BASE"].mkdir(exist_ok=True)
directory,put,command,observe=[n[x] for x in ["directory","put","command","observe"]]
NOW,ID=n["NOW"],n["ID"]
contract,automatic,dated=[n[x] for x in ["contract","automatic","dated"]]
cs,sc,saver,cli=[n[x] for x in ["cs","sc","saver","cli"]]
results=n["RESULTS"]

base=json.loads((OUT/"commands-base.json").read_text())
branch=json.loads((OUT/"commands-branch-final.json").read_text())
assert len(base["status"])==len(branch["status"])==8424
for b,c in zip(base["status"],branch["status"]):
    assert b[:5]==c[:5]
    # Only synthetic filesystem roots differ; the live section was already removed.
    normalize=lambda text:text.replace("commands-base-files","<ROOT>").replace("commands-branch-v2-files","<ROOT>")
    assert normalize(b[5])==normalize(c[5]),(b[:5],b[5],c[5])
results["status_comparison"]="8424 identical after removing only the added live section"

with n["transports"]():
    for repair in ["remove","revoked","enable","matching-automatic"]:
        p=directory("corrupt-repair-"+repair)
        put(p,"season_contract.json",[])
        put(p,"contest_streak.json",automatic())
        assert "corrupt" in observe(p,"repair-before")["message"]
        if repair=="remove": (p/"account_state/season_contract.json").unlink()
        elif repair=="revoked": put(p,"season_contract.json",contract(status="revoked",revoked_at=NOW.isoformat()))
        elif repair=="enable": assert command(p,"season-contract","enable","--season","2027").exit_code==0
        else: put(p,"season_contract.json",contract())
        step=observe(p,"repair-exposes-blocker")
        if repair=="matching-automatic": assert step["accepted"]
        else: assert not step["accepted"]
        assert command(p,"fetch-contest-streak").exit_code==0
        command(p,"season-contract","status")
        assert command(p,"season-contract","enable","--season","2027").exit_code==0
        assert command(p,"fetch-contest-streak").exit_code==0
        assert observe(p,"repair-complete")["accepted"]

    for kind in ["unexpired","expired","no-expiry","auto-schema"]:
        p=directory("persistent-legacy-"+kind)
        manual=dict(active_streak=5,best_streak=9,source_date="2027-07-14")
        if kind=="unexpired": manual["override_expires_at"]="2027-07-17T16:00:00Z"
        if kind=="expired": manual["override_expires_at"]="2027-07-14T16:00:00Z"
        if kind=="auto-schema": manual=automatic(override_expires_at="2027-07-17T16:00:00Z")
        put(p,"contest_streak.manual.json",manual)
        for name,args in [("evidence",["fetch-contest-streak"]),("enable",["season-contract","enable","--season","2027"]),
                          ("fetch",["fetch-contest-streak"])]:
            assert command(p,*args).exit_code==0
            state=observe(p,"persistent-"+kind+"-"+name)
            wins=kind in ["unexpired","auto-schema"] or name!="fetch"
            assert not state["accepted"] if wins else state["accepted"]
        assert command(p,"season-contract","revoke","--season","2027","--reason","restore").exit_code==0
        assert "revoked" in observe(p,"legacy-after-revoke")["message"]
        put(p,"contest_streak.manual.json",manual)
        state=observe(p,"restored-legacy-after-revoke")
        assert "legacy" in state["message"] if kind in ["unexpired","auto-schema"] else "revoked" in state["message"]
        assert command(p,"season-contract","enable","--season","2027").exit_code==0
        state=observe(p,"restored-legacy-after-enable")
        assert "legacy" in state["message"] if kind in ["unexpired","auto-schema"] else "activation_id" in state["message"]
        assert command(p,"fetch-contest-streak").exit_code==0
        state=observe(p,"restored-legacy-after-fetch")
        assert not state["accepted"] if kind in ["unexpired","auto-schema"] else state["accepted"]
        assert command(p,"set-contest-streak","--date","2027-07-15","--streak","5","--best-streak","9","--saver","used").exit_code==0
        assert observe(p,"legacy-replaced-by-dated")["accepted"]

    # Actual read-only folder at write (a); monkeypatched write failures elsewhere
    # are separately exercised by the primary revoke and edge tests.
    p=directory("readonly")
    put(p,"season_contract.json",contract())
    put(p,"contest_streak.manual.json",dated())
    folder=p/"account_state"
    folder.chmod(0o500)
    try:
        r=command(p,"season-contract","revoke","--season","2027","--reason","readonly")
        assert r.exit_code==1 and "NOT revoked" in r.output,(r.output,r.exception)
    finally: folder.chmod(0o700)
    assert (folder/"contest_streak.manual.json").exists()

    for step in ["switch","manual","saver","success"]:
        p=directory("protected-"+step)
        for name,value in [("season_contract.json",contract()),("contest_streak.manual.json",dated()),
                           ("saver_state.json",dict(state="used",season=2027)),("contest_streak.json",automatic()),
                           ("contest_ledger.jsonl",dict(evidence="ledger"))]: put(p,name,value)
        status=p.parent/"health_state/contest_streak_fetch_status.json"
        status.parent.mkdir();status.write_bytes(b'fetch status \xff evidence')
        protected=[p/"account_state/contest_streak.json",p/"account_state/contest_ledger.jsonl",status]
        before=[x.read_bytes() for x in protected]
        if step=="switch":
            path=p/"account_state/season_contract.json";path.unlink();path.mkdir()
        elif step in ["manual","saver"]:
            name="contest_streak.manual" if step=="manual" else "saver_state"
            (p/"account_state"/(name+".revoked-20270715T160000000000Z.json")).symlink_to("missing")
        r=command(p,"season-contract","revoke","--season","2027","--reason","protected")
        assert r.exit_code==(0 if step=="success" else 1)
        assert [x.read_bytes() for x in protected]==before

    # Occupied targets are not consulted when the corresponding source is absent.
    p=directory("absent-sources-occupied-targets")
    for name in ["contest_streak.manual","saver_state"]:
        (p/"account_state"/(name+".revoked-20270715T160000000000Z.json")).symlink_to("missing")
    r=command(p,"season-contract","revoke","--season","2027","--reason","x")
    assert r.exit_code==0 and r.output.count(": absent")==2

    # Execute both filled revoke templates: the dated saver is authoritative for
    # today and does not initialize the flag; initializing the flag is separate.
    p=directory("revoke-templates")
    r=command(p,"season-contract","revoke","--season","2027","--reason","x")
    import re
    commands=re.findall(r"`(bts [^`]+)`",r.output)
    dated_cmd=next(c for c in commands if c.startswith("bts set-contest"))
    args=[{"N":"7","B":"9","S":"used"}.get(x,x) for x in shlex.split(dated_cmd)]
    result=n["CliRunner"]().invoke(cli.cli,args[1:])
    assert result.exit_code==0
    assert saver.load_saver_state(p,season=2027).state=="uninitialized"
    assert observe(p,"revoke-template-dated")["saver"]=="used"
    saver_cmd=next(c for c in commands if c.startswith("bts saver-state"))
    result=n["CliRunner"]().invoke(cli.cli,shlex.split(saver_cmd.replace("<state read from MLB>","active"))[1:])
    assert result.exit_code==0
    assert saver.load_saver_state(p,season=2027).state=="active"
    assert observe(p,"revoke-template-flag-independent")["saver"]=="used"

results["transports"]=n["TRANSPORTS"]
(OUT/"remaining.json").write_text(json.dumps(results,indent=2,default=str))
print(results["status_comparison"])
print({k:len(v) for k,v in results.items() if isinstance(v,list)})
