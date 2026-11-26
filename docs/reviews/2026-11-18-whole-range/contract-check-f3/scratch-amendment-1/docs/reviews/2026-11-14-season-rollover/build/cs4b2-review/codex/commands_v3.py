"""Finish filesystem/sequence checks; keep the completed v2 status/message evidence.

For a directory deliberately placed at the switch path, the expected tree must
include that deliberate fixture change before comparing the failed write.
"""
import json
from pathlib import Path
import review_support  # noqa: F401

OUT=Path(__file__).resolve().parent
loader=(OUT/"commands_v2.py").read_text().split("\ntry:\n    with namespace")[0]
exec(compile(loader,str(OUT/"commands_v2.py"),"exec"))
source=source.replace('assert "NOT revoked" in r.output and files(picks.parent)==before',
    'assert "NOT revoked" in r.output\n                expected_tree=before.copy()\n                if failure=="directory": expected_tree[str(switch.relative_to(picks.parent))]=["dir"]\n                assert files(picks.parent)==expected_tree')
namespace=dict(__file__=str(OUT/"commands.py"),__name__="command_review")
exec(compile(source,str(OUT/"commands.py"),"exec"),namespace)
namespace["BASE"]=OUT/"commands-branch-v3-files"
namespace["BASE"].mkdir(exist_ok=True)
results=namespace["RESULTS"]
previous=json.loads((OUT/"commands-branch.json").read_text())
results["status"]=previous["status"]
results["messages"]=previous["messages"]
results["commands"]=previous["commands"][:8426]
results["sequences"]=[x for x in previous["sequences"] if x["tag"].startswith("message") or x["tag"]=="corrupt-removed"]
try:
    with namespace["transports"]():
        for family in ["revoke","sequences"]:
            namespace[family]()
            print("completed",family,flush=True)
finally:
    results["transports"]=previous["transports"]+namespace["TRANSPORTS"]
    (OUT/"commands-branch-final.json").write_text(json.dumps(results,indent=2,default=str))
print({k:len(v) for k,v in results.items() if isinstance(v,list)})
