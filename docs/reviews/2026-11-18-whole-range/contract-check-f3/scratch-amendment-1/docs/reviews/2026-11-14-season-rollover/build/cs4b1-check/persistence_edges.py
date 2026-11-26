"""Probe the existing saver helper's best-effort audit failure through the real CLI."""
import sys
sys.dont_write_bytecode = True
def deny(event, args):
    if event in {"socket.getaddrinfo", "socket.gethostbyname", "socket.gethostbyaddr",
                 "socket.getnameinfo", "socket.connect", "socket.connect_ex", "socket.sendto"}:
        raise RuntimeError("network forbidden: " + event)
sys.addaudithook(deny)
import builtins
import importlib.util
import json
from pathlib import Path
from unittest.mock import patch
HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO)]
spec = importlib.util.spec_from_file_location("earlier_persistence", REPO / "docs/reviews/2026-11-14-season-rollover/build/cs4b-check/probe.py")
old = importlib.util.module_from_spec(spec)
spec.loader.exec_module(old)
old.ROOT = HERE / "persistence"
opened = builtins.open
def reject_audit(path, mode="r", *a, **kw):
    if Path(path).name == "saver_transitions.jsonl" and mode == "a":
        raise OSError("injected saver audit append failure")
    return opened(path, mode, *a, **kw)
with patch.object(builtins, "open", reject_audit):
    result = old.fetch("audit_fail")
(HERE / "persistence-results.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(dict(exit=result["exit"], events=result["events"], output=result["output"], files=list(result["files"]))))
