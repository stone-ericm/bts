"""Exercise the real ledger readers and compiler with old versus augmented lines."""
import ast
import dataclasses
import importlib.util
import itertools
import json
from pathlib import Path
import shutil
import sys

from review_support import *


def load(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


def main():
    h=Harness(REPO/"src")
    sys.path.insert(0,str(REPO))
    from bts.leaderboard.scraper import parse_rounds_lookup
    from bts.contest_ledger import parse_latest_ledger, likely_save
    from bts.web import saver_dashboard_context
    from bts.data.backup import verify_restored_ops_tree
    from scripts.audit.season_ledger.sources.contest_ledger import parse_contest_ledger
    from scripts.audit.season_ledger.contest import slot_history,line_round_streaks,entered_rounds
    from scripts.audit.season_ledger.reconcile import required_locators,account,check_sources
    from tests.scripts.season_ledger.test_compile import _season_files,_compile
    import pyarrow.parquet as pq
    census=load("review_boundary_census",REPO/"scripts/audit/boundary_shadow_census.py")
    root=ROOT/"ledger-cases"
    if root.exists():shutil.rmtree(root)
    root.mkdir()
    picks=root/"picks"
    counts=[]
    predictions=[{"roundId":1,"result":"hit"},{"roundId":1,"result":"hit"},
                 {"roundId":2,"result":"not_hit"},{"result":"void"},{"roundId":999,"result":"miss"},
                 {"roundId":1,"result":None},{"roundId":2,"result":"pending"},{"roundId":2,"result":"unknown"},{}]
    for duplicate_order in [False,True]:
        raw=[{"id":1,"date":"2026-09-27T08:00:00-04:00"},{"id":1,"date":"2027-07-14T08:00:00-04:00"}]
        if duplicate_order:raw.reverse()
        raw.append({"id":2,"date":"2027-07-13T08:00:00-04:00"})
        h.rounds=parse_rounds_lookup({"rounds":raw})
        h.profile={"activeStreak":7,"seasonBestStreak":9,"predictions":predictions}
        r=h.fetch(root,picks)
        assert r["exit"]==0,r
        row=json.loads((picks/"account_state/contest_ledger.jsonl").read_text().splitlines()[-1])
        assert row["settled_by_year"]==({"2026":2,"2027":1} if duplicate_order else {"2027":3})
        assert row["settled_unmapped"]==2 and row["unsettled"]==4
        assert sum(row["settled_by_year"].values())+row["settled_unmapped"]+row["unsettled"]==len(predictions)
        counts.append({"raw_rounds":raw,"row":row,"fetch":r})
    write_json(ROOT/"ledger-counts.json",counts)
    old={"recorded_at":"2027-07-15T16:00:00Z","active_streak":11,"best_streak":15,"source_date":"2027-07-14",
         "predictions":[{"roundId":1,"result":"hit","streak":11,"streakIncrease":1,"roundPredictions":[{"unitId":10,"playerId":20,"number":1,"result":"hit"}]},
                        {"roundId":2,"result":"not_hit","streak":11,"streakIncrease":0,"roundPredictions":[{"unitId":11,"playerId":21,"number":1,"result":"not_hit"}]}]}
    extra={"season":2027,"switched":False,"activation_id":None,"settled_by_year":{"2027":2},"settled_unmapped":0,"unsettled":0}
    results=[]
    for augmented in [False,True]:
        row=old|extra if augmented else old
        data=((json.dumps(row)+"\n")*2).encode()
        ledger=put(picks,"contest_ledger.jsonl",data)
        put(picks,"saver_state.json",{"season":2027,"state":"active"})
        put(picks,"contest_streak.json",auto())
        parsed=parse_contest_ledger("picks/account_state/contest_ledger.jsonl",data)
        # Execute main's exact AST statements for the P05 ledger reader, avoiding
        # unrelated policy-artifact loading while retaining the production code.
        tree=ast.parse((REPO/"scripts/audit/p05_tail_policy_audit.py").read_text())
        main_node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=="main")
        statements=[]
        for n in main_node.body:
            if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="ledger_last" for t in n.targets):statements.append(n)
            elif isinstance(n,ast.For) and "contest_ledger.jsonl" in ast.unparse(n):statements.append(n)
            elif isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=="contest" for t in n.targets):statements.append(n)
        assert len(statements)==3
        env={"picks":picks,"json":json}
        exec(compile(ast.Module(statements,type_ignores=[]),"production-p05-ledger-reader","exec"),env)
        routed={"picks/account_state/contest_ledger.jsonl":"contest_ledger"}
        accounting=account({"picks/account_state/contest_ledger.jsonl":data},routed,{"picks/account_state/contest_ledger.jsonl":parsed})
        check_sources({"picks/account_state/contest_ledger.jsonl":data},routed,accounting)
        results.append(dict(augmented=augmented,parse_latest=[dataclasses.asdict(r) for r in parse_latest_ledger(ledger)],
            likely_save=likely_save(parse_latest_ledger(ledger)),dashboard=dataclasses.asdict(saver_dashboard_context(picks,now=NOW)),
            backup=verify_restored_ops_tree(root),census=[dataclasses.asdict(r) for r in census.load_ledger(ledger)],
            p05=env["contest"],typed=dataclasses.asdict(parsed),locators=sorted(required_locators("contest_ledger",data)),
            accounting=accounting,slot_history=slot_history(parsed.rows),streaks=line_round_streaks(parsed.rows),entered=entered_rounds(parsed.rows)))
    a,b=results
    for key in ["parse_latest","likely_save","dashboard","backup","census","p05","locators","streaks","entered"]:
        assert a[key]==b[key],key
    def strip(value):
        if isinstance(value,dict):
            return {k:strip(v) for k,v in value.items() if k not in {"obs_id","content_sha256","record_raw_json","last_obs_id","fields_json"}}
        if isinstance(value,list):return [strip(v) for v in value]
        return value
    assert strip(a["typed"])==strip(b["typed"])
    assert strip(a["slot_history"])==strip(b["slot_history"])
    assert strip(a["accounting"])==strip(b["accounting"])
    write_json(ROOT/"ledger-readers.json",results)
    compiled=[]
    for augmented in [False,True]:
        files=_season_files()
        key="picks/account_state/contest_ledger.jsonl"
        if augmented:
            files[key]="".join(json.dumps(json.loads(line)|extra)+"\n" for line in files[key].decode().splitlines()).encode()
        _,out=_compile(root,"augmented" if augmented else "original",files=files)
        tables={str(p.name):pq.read_table(p).to_pylist() for p in out.glob("*.parquet")}
        compiled.append(tables)
    # Content-derived IDs and raw evidence must change; analytic values must not.
    def analytic(value):
        if isinstance(value,dict):
            out={}
            for k,v in value.items():
                if k.endswith("obs_id") or k in {"content_sha256","record_raw_json","fields_json"}:continue
                out[k]=analytic(v)
            return out
        if isinstance(value,list):return [analytic(v) for v in value]
        return value
    assert analytic(compiled[0])==analytic(compiled[1])
    write_json(ROOT/"ledger-compiled-comparison.json",dict(tables={k:len(v) for k,v in compiled[0].items()},analytic_equal=True,
        exceptions=["content SHA","raw line JSON","observation IDs","references to observation IDs"]))
    print("ledger partition, all readers, reconciliation and full compiler: PASS",flush=True)


if __name__=="__main__":main()
