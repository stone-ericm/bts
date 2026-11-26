"""Source preservation, fixture scope, and the real health-runner exception path."""
import ast
import datetime as dt
import json
import logging
from pathlib import Path
import subprocess
import sys
import review_support as audit
from matrix import OUT,ROOT,NOW,automatic,contract,install

sys.path.insert(0,str(ROOT/"src"))
import bts.contest_state as cs
from bts.health.contest_state import check
from bts.health.runner import _safe_run
from tests.conftest import covered_automatic
logging.disable(logging.CRITICAL)
results={"unchanged_functions":{},"fixture":{},"health_runner":{}}
def funcs(text):
    return {n.name:ast.dump(n) for n in ast.parse(text).body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
for filename,allowed in [("src/bts/cli.py",{"season_contract","season_contract_revoke"}),
                         ("src/bts/contest_state.py",{"load_decision_streak_state"}),
                         ("src/bts/season_contract.py",{"dated_template","live_automatic_refusal","inspection_lines"}),
                         ("src/bts/health/contest_state.py",{"check"})]:
    old=funcs(subprocess.check_output(["git","show","0bbfc8d4:"+filename],text=True))
    new=funcs((ROOT/filename).read_text())
    changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    assert changed<=allowed,(filename,changed)
    results["unchanged_functions"][filename]=len(old.keys()-changed)
for filename in ["src/bts/scheduler.py","src/bts/orchestrator.py","src/bts/saver_state.py","src/bts/contest_fetch.py", "src/bts/daily_decision.py","src/bts/web.py"]:
    assert subprocess.check_output(["git","show","0bbfc8d4:"+filename])==(ROOT/filename).read_bytes()
    results["unchanged_functions"][filename]="byte-identical"
p=OUT/"final-check-files/picks"
(p/"account_state").mkdir(parents=True,exist_ok=True)
factory=covered_automatic.__wrapped__()
raw=automatic(schema_version=None)
path=factory(p,NOW,raw)
assert cs.load_decision_streak_state(p,now=NOW,require_contest_state=True).streak==12
results["fixture"]["schema_none_becomes_covered"]="yes; explicit fixture opt-in, not autouse"
for payload,field in [(automatic(active_streak=58,best_streak=58),"automatic contest streak invalid"),
                      (automatic(source_date=None,recorded_at=None),"undated")]:
    factory(p,NOW,payload)
    try: cs.load_decision_streak_state(p,now=NOW,require_contest_state=True)
    except cs.ContestStateError as exc: assert field in str(exc)
    else: raise AssertionError("factory erased corruption")
results["fixture"]["value_and_date_corruptions"]="still rejected by the real loader"
install(p/"account_state/contest_streak.json",automatic())
install(p/"account_state/season_contract.json",contract(status="revoked",revoked_at=NOW.isoformat()))
install(p/"account_state/saver_state.json",b'{"n":'+b'9'*5000+b'}')
try: cs.load_decision_streak_state(p,now=NOW,require_contest_state=True)
except cs.ContestStateError as exc: assert "revoked" in str(exc)
else: raise AssertionError("live accepted revoked")
alerts=_safe_run("contest_state",lambda:check(p,now=NOW,expected=True))
assert len(alerts)==1 and alerts[0].source=="health_runner" and alerts[0].level=="CRITICAL"
assert "ValueError" in alerts[0].message
results["health_runner"]=dict(source=alerts[0].source,level=alerts[0].level,message=alerts[0].message)
(OUT/"final-checks.json").write_text(json.dumps(results,indent=2))
print(json.dumps(results,indent=2))
