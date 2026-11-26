"""Reproduce row 5's no-op comparison race using the named real primitives."""
import sys


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
                 "socket.gethostbyaddr", "socket.getnameinfo"}:
        raise RuntimeError("network forbidden")


sys.addaudithook(deny_network)
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
from bts.util import atomic_write_text

p = HERE / "snapshot-data/account_state/contest_streak.manual.json"
p.parent.mkdir(parents=True, exist_ok=False)
a = dict(schema_version="bts_contest_streak_manual_v3", effective_date="2027-07-15",
         season=2027, expires_at="2027-07-16T03:59:59.999999Z", active_streak=7,
         best_streak=9, saver_state="active", source="manual_cli",
         recorded_at="2027-07-15T16:00:00Z")
b = dict(a, active_streak=8)
atomic_write_text(p, json.dumps(a))
read_snapshot = json.loads(p.read_text())
atomic_write_text(p, json.dumps(b))  # Writer B between A's raw read and its comparison.
identity = lambda x: tuple(x.get(k) for k in (
    "effective_date", "active_streak", "best_streak", "saver_state", "reason"))
out = dict(comparison_says_unchanged=identity(read_snapshot) == identity(a),
           disk_held_requested_observation_at_comparison=identity(json.loads(p.read_text())) == identity(a),
           compared_streak=read_snapshot["active_streak"], actual_streak=json.loads(p.read_text())["active_streak"])
(HERE / "snapshot-results.json").write_text(json.dumps(out, indent=2) + "\n")
print(json.dumps(out, indent=2))
