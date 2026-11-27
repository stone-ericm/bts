import sys,difflib,pathlib,importlib.util
spec=importlib.util.spec_from_file_location("n","/Users/eric/bts-runs/fresh15/closing3/normdiff.py")
src=open("/Users/eric/bts-runs/fresh15/closing3/normdiff.py").read().split("a,b=")[0]
ns={}; exec(src,ns)
a,b=pathlib.Path(sys.argv[1]),pathlib.Path(sys.argv[2]); n=int(sys.argv[3]) if len(sys.argv)>3 else 40
for l in [l for l in difflib.unified_diff(ns["norm"](a.read_text(errors="replace")),ns["norm"](b.read_text(errors="replace")),lineterm="",n=0) if not l.startswith(("---","+++"))][:n]: print(l[:300])
