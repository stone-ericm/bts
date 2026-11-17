"""Feed all six proposed ledger additions to each executable direct consumer."""
import sys
sys.dont_write_bytecode = True
def deny(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
                 "socket.getnameinfo", "socket.connect", "socket.connect_ex", "socket.sendto"}:
        raise RuntimeError("network forbidden: " + event)
sys.addaudithook(deny)
import ast
import datetime as dt
import json
from pathlib import Path
from zoneinfo import ZoneInfo
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
from bts.contest_ledger import parse_latest_ledger
from bts.data.backup import verify_restored_ops_tree
from bts.web import saver_dashboard_context
from scripts.audit.boundary_shadow_census import load_ledger
from scripts.audit.season_ledger.sources.contest_ledger import parse_contest_ledger
source = ast.parse((REPO / "scripts/audit/p05_tail_policy_audit.py").read_text())
main = next(n for n in source.body if isinstance(n, ast.FunctionDef) and n.name == "main")
nodes = [n for n in main.body if
         (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in {"ledger_last", "contest"} for t in n.targets))
         or (isinstance(n, ast.For) and isinstance(n.target, ast.Name) and n.target.id == "line")][:3]
code = compile(ast.Module(body=nodes, type_ignores=[]), "real-p05-ledger-consumer", "exec")
picks = HERE / "consumer-cases" / "picks"
(picks / "account_state").mkdir(parents=True, exist_ok=True)
(picks / "2027-07-14").mkdir(exist_ok=True)
def write(path, obj):
    path.write_text(json.dumps(obj))
write(picks / "streak.json", dict(streak=7, season=2027))
write(picks / "account_state/saver_state.json", dict(state="active", season=2027))
write(picks / "account_state/contest_streak.json", dict(active_streak=7, best_streak=9, source_date="2027-07-14"))
write(picks / "2027-07-14/decision.json", dict(action="single", date="2027-07-14"))
row = dict(recorded_at="2027-07-15T14:00:00Z", active_streak=7, best_streak=9, source_date="2027-07-14",
           predictions=[dict(roundId=1, result="hit", streak=7, roundPredictions=[])])
out = []
for extra in [{}, dict(season=2027, switched=False, activation_id=None, settled_by_year={"2027":1}, settled_unmapped=0, unsettled=0),
              dict(season=2027, switched=True, activation_id="a"*32, settled_by_year={"2027":1}, settled_unmapped=0, unsettled=0)]:
    body = json.dumps(dict(row, **extra)) + "\n"
    path = picks / "account_state/contest_ledger.jsonl"
    path.write_text(body + body)
    parsed = parse_contest_ledger("picks/account_state/contest_ledger.jsonl", path.read_bytes())
    env = dict(picks=picks, json=json)
    exec(code, env)
    out.append(dict(extra=extra, latest=repr(parse_latest_ledger(path)), backup=verify_restored_ops_tree(picks),
                    census=repr(load_ledger(path)), dashboard=repr(saver_dashboard_context(picks, now=dt.datetime(2027,7,15,12,tzinfo=ZoneInfo("America/New_York")))),
                    p05=env["contest"], projection=parsed.rows, quarantine=parsed.quarantined))
checks = {key: all(v[key] == out[0][key] for v in out[1:]) for key in ["latest", "backup", "census", "dashboard", "p05", "quarantine"]}
assert all(checks.values()), checks
strip = lambda rows: [{k:v for k,v in r.items() if k not in {"content_sha256", "obs_id", "record_raw_json"}} for r in rows]
assert all(strip(v["projection"]) == strip(out[0]["projection"]) for v in out[1:])
(HERE / "consumer-results.json").write_text(json.dumps(dict(checks=checks, cases=out), indent=2) + "\n")
print(json.dumps(dict(checks=checks, lossless_difference="raw JSON, content SHA and observation IDs only")))
