"""Print what the lead's reviewer decision needs for one replay or defence run (read-only)."""
import json, sys
from pathlib import Path

d = Path(sys.argv[1])
a = json.loads((d / "acceptance.json").read_text())
print(f"== {a.get('label')}: {a.get('verdict')}  reasons={a.get('reasons')[:3]}")
if (d / "red.events.jsonl").exists():          # replay
    print(f"   fix {str(a.get('fix'))[:7]} parent {str(a.get('parent'))[:7]} kind {a.get('label_kind')} audit_problems={a.get('audit_problems')}")
    print(f"   harness changes: {[c['path'] for c in a.get('harness_changes', [])]}")
    for stage in ("green", "red"):
        for line in (d / f"{stage}.events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if e["kind"] == "report" and e["when"] == "call":
                msg = (e.get("message") or "").replace("\n", " | ")[:220]
                print(f"   {stage} {e['outcome']:7} {e['nodeid'].split('::')[-1][:70]} assertion={e.get('exc_is_assertion')} {msg}")
    print(f"   symptoms: {a.get('symptoms')}  classes: {a.get('classes')}")
else:                                           # defence
    st = a.get("stages", {})
    print(f"   stages: " + "; ".join(f"{k} rc={v.get('returncode')} gate={v.get('gate')}" + (f" kills={len(v.get('kills', []))}" if 'kills' in v else "")
                                    for k, v in st.items()))
    print(f"   touched {a.get('touched')} branch_line {a.get('branch_line')}")
    ev = {}
    if (d / "mutant.events.jsonl").exists():
        for line in (d / "mutant.events.jsonl").read_text().splitlines():
            e = json.loads(line)
            if "seq" in e:
                ev[e["seq"]] = e
    for node, c in (a.get("certificates") or {}).items():
        print(f"   cert {node.split('::')[-1][:70]}: ok={c.get('ok')} linked={c.get('linked')} reasons={(c.get('reasons') or [])[:2]}")
        for role, seq in (c.get("linked") or {}).items():
            e = ev.get(seq, {})
            what = e.get("identity") or {k: e.get(k) for k in ("value", "category", "line", "qualname", "file") if k in e}
            print(f"      {role} seq {seq}: {e.get('kind')} {e.get('qualname') or e.get('name') or ''} {json.dumps(what)[:300]}")
    patch = (d / "mutant.patch").read_text() if (d / "mutant.patch").exists() else ""
    print("   patch: " + " | ".join(l for l in patch.splitlines() if l.startswith(("+", "-")) and not l.startswith(("+++", "---")))[:600])
