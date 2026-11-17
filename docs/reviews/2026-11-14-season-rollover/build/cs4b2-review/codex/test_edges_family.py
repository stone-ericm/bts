"""Remaining evidence: named base GREEN controls, final health append mutation,
and AST inventory of migrated tests. Run through yield_run.sh.
"""
import ast
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import xml.etree.ElementTree as ET
import review_support  # noqa: F401

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
os.chdir(ROOT)
env=dict(os.environ,REVIEW_YEAR="2027",PYTHONPATH=str(OUT)+os.pathsep+str(ROOT),PYTHONDONTWRITEBYTECODE="1",TZ="America/New_York")
new=["tests/test_season_contract_cs4b2.py","tests/test_season_contract_cs4b2_edges.py"]
def pytest(label,src,extra):
    command=[sys.executable,"-m","pytest","-p","no:cacheprovider","-p","review_support",*new,"-q","--tb=short",
             "--basetemp",str(OUT/"test-work"/label),*extra]
    with (OUT/(label+".log")).open("w") as log:
        result=subprocess.run(command,env=env|dict(REVIEW_SRC=str(src)),stdout=log,stderr=subprocess.STDOUT)
    print(label,result.returncode,(OUT/(label+".log")).read_text().splitlines()[-1],flush=True)
    return result.returncode

xml=OUT/"base-new-names.xml"
assert pytest("base-new-names",OUT/"base/src",["--junitxml",str(xml)])==1
root=ET.parse(xml).getroot()
greens=[dict(file=x.get("classname"),name=x.get("name")) for x in root.iter("testcase") if x.find("failure") is None and x.find("error") is None and x.find("skipped") is None]
(OUT/"base-green-controls.json").write_text(json.dumps(dict(count=len(greens),groups=Counter(x["name"].split("[")[0] for x in greens),cases=greens),indent=2))
source=OUT/"mutants/no-health-final/src"
shutil.copytree(ROOT/"src/bts",source/"bts",dirs_exist_ok=True,ignore=shutil.ignore_patterns("__pycache__"))
path=source/"bts/health/contest_state.py"
text=path.read_text()
old="    append_refusal()\n    return alerts\n"
assert text.count(old)==1
path.write_text(text.replace(old,"    return alerts\n"))
assert pytest("mutant-no-health-final",source,["-k","selection_matrix"])==1

changed=subprocess.check_output(["git","diff","--name-only","0bbfc8d4","HEAD"],text=True).splitlines()
inventory=[]
def functions(text):
    return {n.name:n for n in ast.walk(ast.parse(text)) if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef))}
for file in changed:
    if not file.startswith("tests/") or file in new: continue
    before=subprocess.check_output(["git","show","0bbfc8d4:"+file],text=True)
    after=(ROOT/file).read_text()
    old_functions,new_functions=functions(before),functions(after)
    entries=[]
    for name,node in new_functions.items():
        old_node=old_functions.get(name)
        if old_node is None or ast.dump(old_node)!=ast.dump(node):
            asserts=lambda n:[ast.unparse(x.test) for x in ast.walk(n) if isinstance(x,ast.Assert)] if n else []
            old_asserts,new_asserts=asserts(old_node),asserts(node)
            entries.append(dict(name=name,base_asserts=old_asserts,branch_asserts=new_asserts,
                                removed_asserts=[a for a in old_asserts if a not in new_asserts],
                                added_asserts=[a for a in new_asserts if a not in old_asserts]))
    inventory.append(dict(file=file,functions=entries))
(OUT/"migration-inventory.json").write_text(json.dumps(inventory,indent=2))
print("GREEN",len(greens),"migrated functions",sum(len(x["functions"]) for x in inventory),flush=True)
