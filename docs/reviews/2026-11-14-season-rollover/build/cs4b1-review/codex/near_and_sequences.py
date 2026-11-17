"""Small supplemental real-command differential and a sequence with settled picks."""
import collections
import dataclasses
import json
from pathlib import Path
import shutil
import subprocess
import sys

from review_support import *


def main():
    if len(sys.argv)==1:
        for label,source in [("base",ROOT/"base/src"),("branch",REPO/"src")]:
            subprocess.run([sys.executable,__file__,label,str(source)],check=True)
        return
    label,source=sys.argv[1],Path(sys.argv[2])
    h=Harness(source)
    root=ROOT/("near-cases-"+label)
    if root.exists():shutil.rmtree(root)
    root.mkdir()
    picks=root/"picks"
    records=[]
    def command(name,args):
        r=h.invoke(root,picks,args);records.append({"name":name,"result":r});return r
    command("row1-cold-status",["season-contract","status"])
    command("row2-enable",["season-contract","enable","--season","2027"])
    command("row2-enable-again",["season-contract","enable","--season","2027"])
    good={"activeStreak":7,"seasonBestStreak":9,"predictions":[{"roundId":1,"result":"hit"}]}
    import httpx
    errors={n:httpx.HTTPStatusError(str(n),request=httpx.Request("GET","https://offline.invalid/rounds"),response=httpx.Response(n)) for n in [503,429]}
    profiles={"missing":{k:v for k,v in good.items() if k!="predictions"},"null":good|{"predictions":None},
              "nonlist":good|{"predictions":{}},"nonobject":good|{"predictions":[None]},
              "empty":good|{"predictions":[]},"profile_null":None,
              "active_bool_best58":good|{"activeStreak":True,"seasonBestStreak":58},
              "active_above_best58":good|{"activeStreak":59,"seasonBestStreak":58}}
    for name,profile in profiles.items():
        for round_name,rounds in {"empty":{},"503":errors[503],"429":errors[429],"valid":{1:REAL_DATE(2027,7,14)}}.items():
            # Avoid accumulated cooldown masking the transport comparison.
            status=root/"health_state/contest_streak_fetch_status.json"
            if status.exists():status.unlink()
            h.profile,h.rounds=profile,rounds
            r=h.fetch(root,picks)
            records.append({"name":"response-pair:"+name+":"+round_name,"result":r})
            if label=="branch":
                assert r["exit"]==(0 if name=="empty" and round_name=="valid" else 2),r
                if name=="profile_null":assert "rounds" not in [e[0] for e in r["events"]]
                elif round_name in {"503","429"}:assert round_name in r["output"]
    # Known near-row preservation cases: unknown schema, recorded-time-only auto,
    # unexpired legacy overriding an invalid auto, and stamped mismatched IDs.
    readers=[]
    for data,manual in [(auto(schema_version="unknown",season=None,activation_id=[]),None),
                        (auto(schema_version=None,source_date=None),None),
                        (auto(season=None),{"active_streak":2,"override_expires_at":"2027-08-01T00:00:00Z"}),
                        (auto(season=2026,activation_id="f"*32),None)]:
        put(picks,"contest_streak.json",data)
        mp=picks/"account_state/contest_streak.manual.json"
        if mp.exists():mp.unlink()
        if manual is not None:put(picks,mp.name,manual)
        put(picks,"season_contract.json",contract(status="revoked",revoked_at="2027-07-15T00:00:00Z"))
        for setting in [False,True]:
            readers.append(dict(data=data,manual=manual,setting=setting,
                decision=dataclasses.asdict(h.cs.load_decision_streak_state(picks,now=NOW,require_contest_state=setting)),
                health=[dataclasses.asdict(a) for a in h.health.check(picks,now=NOW,expected=setting)]))
    write_json(ROOT/("near-"+label+".json"),dict(commands=records,readers=readers))
    if label=="branch":
        # Reuse the independently authored long sequence, adding real settled
        # local picks before each dated observation to expose coverage warnings.
        import command_edges_v2 as edges
        orig=edges.fresh
        def seeded(name):
            sr,sp=orig(name)
            if name=="operator-sequence":
                sp.mkdir()
                for d in [15,16,17]:(sp/f"2027-07-{d}.json").write_text('{"result":"hit"}')
            return sr,sp
        edges.fresh=seeded
        h.profile=good;h.rounds={1:REAL_DATE(2027,7,14)}
        edges.sequence(h)
        # Preserve both sequence variants rather than replace the first result.
        (ROOT/"operator-sequence-settled-picks.json").write_bytes((ROOT/"operator-sequence.json").read_bytes())
    print("near and response pairs",label,"PASS",flush=True)


if __name__=="__main__":main()
