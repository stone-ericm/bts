"""Audit the completed matrix's actual file-access ordering and preserved failures."""
import collections
import gzip
import json
import os
from review_support import ROOT,write_json


def main():
    problems=[]
    counts=collections.Counter()
    with gzip.open(ROOT/"matrix-branch.jsonl.gz","rt") as f:
        for line in f:
            r=json.loads(line)
            sn,an,rn,dry=r["case"]
            trace=r["trace"]
            opens=[e for e in trace if e[0]=="open"]
            writes=[e for e in opens if int(e[-1]) & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)]
            replacements=[(i,e) for i,e in enumerate(trace) if e[0]=="os.rename"]
            def index_name(name):
                return [i for i,e in enumerate(trace) if e[0]=="open" and name in e[1]]
            try:
                if sn!="automatic":
                    assert not any("contest_streak.json" in e[1] for e in writes)
                    assert not any("contest_streak.json" in e[2] for _,e in replacements)
                    assert not any("saver_" in str(e) for e in trace)
                if r["exit"]==0 and dry:
                    assert not writes and not replacements and r["before"]==r["after"]
                if r["exit"]==0 and not dry:
                    ledger=index_name("contest_ledger.jsonl")
                    assert len(ledger)==1
                    status=[i for i,e in replacements if "contest_streak_fetch_status.json" in e[2]]
                    assert len(status)==1 and status[0]<ledger[0]
                    switches=index_name("season_contract.json")
                    if switches:assert switches[0]<status[0]
                    if sn=="automatic":
                        observation=[i for i,e in replacements if e[2].endswith("/contest_streak.json")]
                        assert len(observation)==1 and observation[0]<status[0]
                        saver=[i for i,e in enumerate(trace) if "saver_" in str(e)]
                        if saver:assert ledger[0]<min(saver)
                    counts["automatic_success" if sn=="automatic" else "ledger_only_success"]+=1
                if an in {"utf8","nonobject","other_account"}:
                    assert not index_name("season_contract.json")
                counts["checked"]+=1
            except AssertionError:
                problems.append(r["case"])
    write_json(ROOT/"file-order-audit.json",dict(counts=counts,problems=problems))
    assert not problems,problems[:5]
    print("actual file access order:",dict(counts),"PASS")


if __name__=="__main__":main()
