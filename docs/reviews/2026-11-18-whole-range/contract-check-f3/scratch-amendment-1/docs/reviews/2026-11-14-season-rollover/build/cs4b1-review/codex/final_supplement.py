"""Realistic daily pick chronology, optional note and legacy-before-v2 checks."""
import json
from pathlib import Path
import uuid

from review_support_v2 import *
import command_edges_v2 as edges


def main():
    h=Harness(REPO/"src")
    original=h.invoke
    def invoke(root,picks,args):
        if args[0]=="set-contest-streak" and "--date" in args:
            day=args[args.index("--date")+1]
            picks.mkdir(parents=True,exist_ok=True)
            (picks/(day+".json")).write_text('{"result":"hit"}')
        return original(root,picks,args)
    h.invoke=invoke
    edges.sequence(h)
    sequence=json.loads((ROOT/"operator-sequence.json").read_text())
    (ROOT/"operator-sequence-realistic.json").write_text(json.dumps(sequence,indent=2)+"\n")
    assert all(not s["health"] for s in sequence if s["step"]=="daily-observation-and-unswitched-fetch")
    next_day=next(s for s in sequence if s["step"]=="next-day-before-matching-fetch")
    assert any(a["source"]=="contest_state" and a["level"]=="WARN" for a in next_day["health"])
    root,picks=edges.fresh("supplement")
    data=contract();data.pop("note")
    put(picks,"season_contract.json",data)
    from bts.season_contract import switch_state
    assert switch_state(picks,NOW).state=="automatic"
    r=h.invoke(root,picks,["season-contract","enable","--season","2027"])
    assert r["exit"]==0 and r["before"]==r["after"]
    ids=[]
    for _ in range(3):
        put(picks,"season_contract.json",contract(status="revoked",revoked_at="2027-07-15T16:00:00Z"))
        r=h.invoke(root,picks,["season-contract","enable","--season","2027"])
        ident=json.loads((picks/"account_state/season_contract.json").read_text())["activation_id"]
        assert r["exit"]==0 and uuid.UUID(hex=ident).version==4 and ident not in ids
        ids.append(ident)
    checks=[]
    for override,dated,expected in [({"active_streak":True},False,"streak invalid"),
                                  ({"saver_available":"bad"},False,"saver_available"),
                                  ({"best_streak":58},False,"automatic contest streak"),
                                  ({"source_date":None,"recorded_at":None},True,"undated"),
                                  ({},False,"season")]:
        path=put(picks,"contest_streak.json",auto(season=None,activation_id=None,**override))
        try:h.cs._parse_state_file(path,require_dated=dated)
        except h.cs.ContestStateError as exc:
            assert expected in str(exc),(expected,str(exc))
            checks.append(str(exc))
        else:raise AssertionError("invalid observation parsed")
    write_json(ROOT/"final-supplement.json",dict(uuid4_ids=ids,parser_order=checks,optional_note="valid and enable no-op",sequence="operator-sequence-realistic.json"))
    print("realistic operator sequence, UUID4, optional note, old-check precedence: PASS")


if __name__=="__main__":main()
