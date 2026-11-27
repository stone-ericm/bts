import re,sys,difflib,pathlib
def norm(s):
    s=re.sub(r"wr[23]-harness|wr-closing-harness","X",s)
    s=re.sub(r"/var/folders/\S*|/private/tmp/\S*|/tmp/\S*","TMP",s)
    s=re.sub(r"\b[0-9a-f]{40}\b","SHA",s); s=re.sub(r"\b[0-9a-f]{32}\b","RUNID",s)
    s=re.sub(r"\d+\.\d+s\b","Ns",s); s=re.sub(r"in \d+\.\d+s","in Ns",s)
    s=re.sub(r"20\d\d-\d\d-\d\d[ T]\d\d:\d\d:\d\d(\.\d+)?(Z|[+-]\d\d:\d\d)?","TS",s)
    s=re.sub(r"0x[0-9a-f]+","0xADDR",s)
    s=re.sub(r"(cron|mutants|tmp|pytest|probe|wrk1-caller|season-walk)-[A-Za-z0-9_]{6,}","\\1-RAND",s)
    s=re.sub(r"\.py:\d+",".py:N",s)
    s=re.sub(r"pytest-of-eric/pytest-\d+","PYT",s)
    s=re.sub(r"\b\d+ (passed|failed|skipped)"," N \\1",s)
    s=re.sub(r"[0-9a-f]{64}","SHA256",s)
    return s.splitlines()
a,b=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2])
for f in sorted(p.name for p in a.glob("*.log")):
    if not (b/f).exists(): print("MISSING in FINAL3:",f); continue
    d=[l for l in difflib.unified_diff(norm((a/f).read_text(errors="replace")),norm((b/f).read_text(errors="replace")),lineterm="",n=0) if not l.startswith(("---","+++","@@"))]
    print(f"{f}: {len(d)} differing lines")
