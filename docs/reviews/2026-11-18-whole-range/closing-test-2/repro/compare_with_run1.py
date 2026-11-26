import json,sys,difflib,re
def norm(s):
    s=re.sub(r"[0-9a-f]{40}","SHA",s); s=re.sub(r"[0-9a-f]{32}","RUNID",s)
    s=re.sub(r'"captured_at": "[^"]*"','"captured_at": T',s); s=re.sub(r"wr-closing-harness|wr2-harness","X",s)
    s=re.sub(r"/var/folders/[^\"' ]*|/private/tmp/[^\"' ]*|/tmp/[^\"' ]*","TMP",s)
    return s
def load(p):
    out={}
    for line in open(p):
        line=line.strip()
        if not line.startswith("{"): continue
        d=json.loads(line); out[d["case"]]=d
    return out
a,b=load(sys.argv[1]),load(sys.argv[2])
print("cases equal:",sorted(a)==sorted(b), len(a))
for c in a:
    da,db=a[c],b[c]
    for k in sorted(set(da)|set(db)):
        if k=="module": continue
        sa=norm(json.dumps(da.get(k),indent=0,sort_keys=True)).splitlines(); sb=norm(json.dumps(db.get(k),indent=0,sort_keys=True)).splitlines()
        d=[l for l in difflib.unified_diff(sa,sb,lineterm="",n=0) if not l.startswith(("---","+++","@@"))]
        if d:
            print(f"## {c}.{k}: {len(d)} lines"); print("\n".join(x[:260] for x in d[:int(sys.argv[3]) if len(sys.argv)>3 else 14]))
