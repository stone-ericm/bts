import json, os, subprocess, sys, tempfile, importlib.util
from pathlib import Path
sys.path.insert(0, "src")
from bts.data.build import build_season
src = subprocess.run(["git", "show", "e3edbf58:src/bts/data/build.py"], capture_output=True, text=True, check=True).stdout
pp = Path(tempfile.mkdtemp()) / "parent_build.py"; pp.write_text(src)
spec = importlib.util.spec_from_file_location("parent_build", pp); parent = importlib.util.module_from_spec(spec); spec.loader.exec_module(parent)
def run(fn, raw, out):
    try:
        r = fn(raw, out, 2027); return "returned " + ("None" if r is None else f"{len(r)} rows")
    except Exception as e:
        return f"raised {type(e).__name__}: {str(e)[:60]}"
for label in ("unreadable directory (chmod 000)", "a regular file where the directory should be"):
    root = Path(tempfile.mkdtemp()); raw = root / "raw"; raw.mkdir(); sd = raw / "2027"
    if label.startswith("unreadable"):
        sd.mkdir(); (sd / "900001.json").write_text(json.dumps({"gameData": {"game": {"type": "R"}}})); os.chmod(sd, 0)
    else:
        sd.write_text("not a directory")
    print(label)
    print("  parent e3edbf58:", run(parent.build_season, raw, root / "p.parquet"))
    print("  main          :", run(build_season, raw, root / "m.parquet"))
    if sd.is_dir(): os.chmod(sd, 0o755)
