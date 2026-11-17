"""Re-run commands.py with tree capture outside the injected lstat failure.

The original script and raw matrix data are kept intact for auditability.
"""
import json
from pathlib import Path
import review_support  # noqa: F401

OUT=Path(__file__).resolve().parent
source=(OUT/"commands.py").read_text()
source=source[:source.rindex("\nwith transports():")]
old='            with contextlib.ExitStack() as stack:\n                if failure=="permission":'
new='            before=files(picks.parent)\n            with contextlib.ExitStack() as stack:\n                if failure=="permission":'
assert source.count(old)==1
source=source.replace(old,new)
old='                before=files(picks.parent)\n                r=command(picks,"season-contract","revoke"'
new='                r=command(picks,"season-contract","revoke"'
assert source.count(old)==1
source=source.replace(old,new)
namespace=dict(__file__=str(OUT/"commands.py"),__name__="command_review")
exec(compile(source,str(OUT/"commands.py"),"exec"),namespace)
namespace["BASE"]=OUT/"commands-branch-v2-files"
namespace["BASE"].mkdir(exist_ok=True)
try:
    with namespace["transports"]():
        for family in ["statuses","messages","revoke","sequences"]:
            namespace[family]()
            print("completed",family,flush=True)
finally:
    results=namespace["RESULTS"]
    results["transports"]=namespace["TRANSPORTS"]
    (OUT/"commands-branch.json").write_text(json.dumps(results,indent=2,default=str))
print({k:len(v) for k,v in results.items() if isinstance(v,list)})
