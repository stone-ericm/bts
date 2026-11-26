"""Verify the raw matrix records (including Alert.incident_key and absent-file lstat)."""
import itertools
import json
from collections import Counter
from pathlib import Path
import review_support  # noqa: F401

OUT=Path(__file__).resolve().parent
base=json.loads((OUT/"matrix-base.json").read_text())
branch=json.loads((OUT/"matrix-branch.json").read_text())
summary, errors, disagreements=Counter(),[],Counter()
assert base["axes"]==branch["axes"]
AUTO="bts_contest_streak_auto_v2"
for names,b,c in zip(itertools.product(*base["axes"]),base["rows"],branch["rows"]):
    bv,cv=[[data["values"][entry[0]] for entry in row] for data,row in [(base,b),(branch,c)]]
    def require(ok,kind):
        if not ok: errors.append(dict(names=names,kind=kind,base=bv,branch=cv,reads=[b,c]))
    for i in [0,1,2,4]:
        require(bv[i]==cv[i],f"preserved_{i}")
        require(c[i][1]==0,f"no_switch_open_{i}")
    selected=cv[1][1]
    loaded=cv[1][0]=="ok"
    bypass=loaded and selected is not None and selected["effective_date"]=="2027-07-15"
    sw=names[0]
    reason=None
    if loaded and not bypass:
        if sw.startswith("corrupt"): reason="corrupt season switch, check "+sw[7:]
        elif selected is not None and selected["path"].endswith("contest_streak.manual.json"): reason="legacy manual override"
        elif sw in ["absent","other_season","revoked"]: reason="switch is "+sw
        elif selected is None: reason="no automatic observation"
        elif selected["schema_version"]!=AUTO: reason="schema_version"
        elif selected["season"]!=2027: reason="automatic observation season"
        elif selected["activation_id"]!="a"*32: reason="automatic observation activation_id"
    # An absent snapshot is determined by lstat and cannot generate an open event.
    opens=int(loaded and not bypass and sw!="absent")
    require(c[3][1]==opens,"decision_switch_opens")
    if not loaded:
        require(cv[3]==cv[1],"loader_error_priority")
        alert=dict(level="CRITICAL",source="contest_state",message=cv[1][1],incident_key=None)
        require(cv[5]==["ok",[alert]],"health_loader_error")
        require(c[5][1]==0,"health_loader_no_switch")
    elif reason:
        require(cv[3][0]=="ContestStateError" and reason in cv[3][1].split("\n")[0],"refusal_reason")
        if bv[5][0]!="ok":
            require(cv[5]==bv[5],"health_existing_exception")
            disagreements["preexisting_postloader_exception"]+=1
        else:
            alert=dict(level="CRITICAL",source="contest_state",message=cv[3][1],incident_key=None)
            require(cv[5]==["ok",bv[5][1]+[alert]],"health_exact_append_last")
            require(c[5][1]==opens,"health_switch_once")
    else:
        require(cv[3]==bv[3],"accepted_exact_decision")
        require(cv[5]==bv[5],"accepted_exact_health")
        require(c[5][1]==(opens if bv[5][0]=="ok" else 0),"accepted_health_switch")
    summary["decision_changed" if bv[3]!=cv[3] else "decision_identical"]+=1
    summary["health_changed" if bv[5]!=cv[5] else "health_identical"]+=1
    summary["live_accepted" if cv[3][0]=="ok" else "live_raises"]+=1
    if cv[3][0]=="ok":
        require(bypass or reason is None,"uncovered_accepted")
        if cv[5][0]=="ok" and any(x["level"]=="CRITICAL" for x in cv[5][1]):
            disagreements["accepted_existing_health_critical"]+=1
    summary["optional_identical"]+=int(bv[2]==cv[2])
    summary["health_optional_identical"]+=int(bv[4]==cv[4])
report=dict(cells=len(base["rows"]),axes=base["axes"],sources=[base["source"],branch["source"]],summary=summary,
            disagreements=disagreements,errors=len(errors),error_kinds=Counter(e["kind"] for e in errors),examples=errors[:3])
(OUT/"matrix-verified.json").write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k not in ["axes","examples"]},indent=2))
assert not errors,(len(errors),"see matrix-verified.json")
