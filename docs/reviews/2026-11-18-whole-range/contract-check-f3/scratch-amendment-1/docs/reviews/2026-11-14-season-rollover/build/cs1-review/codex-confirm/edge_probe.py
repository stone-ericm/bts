"""Bounded subprocess probes for reads that cannot safely run inline."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading

ROOT = Path(__file__).resolve().parent
CHILD = r'''
import dataclasses,json,sys
from pathlib import Path
from datetime import date,datetime,timezone
from bts import contest_state as c
p=Path(sys.argv[1]); site=sys.argv[2]; action=sys.argv[3]
paths={"model":p/"streak.json", "manual":p/"account_state/contest_streak.manual.json", "auto":p/"account_state/contest_streak.json", "pick":p/"2026-06-18.json"}
print("ENTER",c.__file__,site,action,flush=True)
try:
 if action=="read":
  out=paths[site].read_text(encoding="utf-8")
  result={"bytes":len(out)}
 elif site=="pick":
  result=c._has_unconfirmed_miss(p,date(2026,6,16)) if action=="miss" else c._resolved_pick_dates(p)
 else:
  out=c.load_decision_streak_state(p,require_contest_state=True,now=datetime(2026,6,18,16,tzinfo=timezone.utc))
  result=dataclasses.asdict(out)
 print(json.dumps({"result":result},default=str),flush=True)
except Exception as exc:
 print(json.dumps({"exception":type(exc).__name__,"message":str(exc),"cause":type(exc.__cause__).__name__ if exc.__cause__ else None}),flush=True)
'''


def child(p, site, action="production", old=False, timeout=4):
    env = os.environ.copy()
    if old:
        env["PYTHONPATH"] = str(ROOT / "scratch-acea53f8")
    try:
        r = subprocess.run([sys.executable, "-c", CHILD, str(p), site, action], env=env, text=True, capture_output=True, timeout=timeout)
        assert r.returncode == 0, (r.returncode, r.stdout, r.stderr)
        lines = r.stdout.splitlines()
        return {"loader": lines[0], "outcome": json.loads(lines[-1]), "stderr": r.stderr}
    except subprocess.TimeoutExpired as exc:
        # subprocess.run kills and reaps only its own child on timeout.
        return {"timeout_seconds": timeout, "stdout": (exc.stdout or b"").decode(), "child_killed_and_reaped": True}


def setup(p, site):
    (p / "account_state").mkdir()
    if site in ("model", "pick"):
        (p / "account_state/contest_streak.manual.json").write_text('{"active_streak":7,"best_streak":20,"source_date":"2026-06-18","override_expires_at":"2026-06-18T23:59:59Z"}')
    return p / {"model": "streak.json", "manual": "account_state/contest_streak.manual.json", "auto": "account_state/contest_streak.json", "pick": "2026-06-18.json"}[site]


def sparse_memory():
    results = []
    for site in ("model", "manual", "auto", "pick"):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="memory-") as tmp:
            p = Path(tmp)
            target = setup(p, site)
            with target.open("wb") as f:
                f.write(b'"')
                # A sparse 1 PiB file exceeds the user address space. readall's
                # initial allocation fails before reading/committing that size.
                f.truncate(1 << 50)
            st = target.stat()
            assert st.st_blocks * 512 < 1024 * 1024
            read = child(p, site, action="read")
            assert read["outcome"]["exception"] == "MemoryError", read
            current = child(p, site)
            old = child(p, site, old=True)
            assert current["outcome"]["exception"] == old["outcome"]["exception"] == "MemoryError", (current, old)
            extra = {}
            if site == "pick":
                extra = {"miss_head": child(p, site, action="miss"), "miss_before": child(p, site, action="miss", old=True)}
                assert all(x["outcome"]["exception"] == "MemoryError" for x in extra.values())
            results.append({"site": site, "logical_bytes": st.st_size, "allocated_bytes": st.st_blocks * 512, "read": read, "head": current, "before": old, "sparse_removed_after_probe": True, **extra})
    return results


def fifo_no_writer():
    results = []
    for site in ("model", "manual", "auto", "pick"):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="fifo-") as tmp:
            p = Path(tmp)
            target = setup(p, site)
            os.mkfifo(target)
            current = child(p, site, timeout=1)
            old = child(p, site, old=True, timeout=1)
            assert "timeout_seconds" in current and "timeout_seconds" in old
            results.append({"site": site, "head": current, "before": old})
    return results


def fifo_writer():
    results = []
    for site in ("model", "manual", "auto", "pick"):
        with tempfile.TemporaryDirectory(dir=ROOT, prefix="fifo-fed-") as tmp:
            p = Path(tmp)
            target = setup(p, site)
            os.mkfifo(target)
            for old in (False, True):
                for action in (("resolved", "miss") if site == "pick" else ("production",)):
                    def writer():
                        with target.open("wb") as f:
                            f.write(b"9" * 4301)
                    thread = threading.Thread(target=writer, daemon=True)
                    thread.start()
                    out = child(p, site, action=action, old=old)
                    thread.join(timeout=2)
                    assert not thread.is_alive()
                    assert "timeout_seconds" not in out, out
                    if old:
                        assert out["outcome"]["exception"] == "ValueError"
                    elif site == "pick":
                        assert out["outcome"]["result"] == ([] if action == "resolved" else False)
                    elif site == "model":
                        assert out["outcome"]["result"]["streak"] == 7
                        assert out["outcome"]["result"]["model_streak"] is None
                    else:
                        assert out["outcome"]["exception"] == "ContestStateError"
                        assert out["outcome"]["cause"] == "ValueError"
                    results.append({"site": site, "version": "before" if old else "head", "action": action, **out})
    return results


if __name__ == "__main__":
    result = {}
    for name, fn in (("sparse_memory", sparse_memory), ("fifo_no_writer", fifo_no_writer), ("fifo_writer", fifo_writer)):
        result[name] = fn()
        (ROOT / "edge-results.json").write_text(json.dumps(result, indent=2))
        print(name, len(result[name]), "PASS", flush=True)
