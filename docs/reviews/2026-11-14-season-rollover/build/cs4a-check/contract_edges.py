"""Additional parser and failure-boundary probes; no CS4a implementation."""
import sys


def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("network forbidden in contract probe")


sys.addaudithook(deny_network)
import json
import logging
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))
import bts.contest_state as cs
import bts.saver_state as ss
from bts.util import atomic_write_text

logging.disable(logging.CRITICAL)
NOW = datetime(2027, 7, 15, 12, tzinfo=ZoneInfo("America/New_York"))
BASE = dict(schema_version="bts_contest_streak_manual_v3", effective_date="2027-07-15",
            season=2027, expires_at="2027-07-15T23:59:59.999999-04:00", active_streak=7,
            best_streak=9, saver_state="active", recorded_at=NOW.isoformat(), source="manual_cli")
OUT = {}


def folder(name):
    p = HERE / "edge-data" / name
    p.mkdir(parents=True, exist_ok=False)
    (p / "account_state").mkdir()
    return p


def run(fn):
    try:
        v = fn()
        return {"ok": True, "value": asdict(v) if hasattr(v, "__dataclass_fields__") else v}
    except Exception as e:
        return {"ok": False, "error": type(e).__name__, "message": str(e)[:300]}


cases = {"valid": BASE}
for k in BASE:
    missing = dict(BASE)
    del missing[k]
    cases[f"missing_{k}"] = missing
    cases[f"null_{k}"] = dict(BASE, **{k: None})
for name, changes in {
    "schema_typo": {"schema_version": "bts_contest_streak_manual_v33"},
    "unpadded_date": {"effective_date": "2027-7-15"},
    "bad_date": {"effective_date": "2027-02-30"},
    "season_bool": {"season": True}, "season_mismatch": {"season": 2026},
    "expires_early": {"expires_at": "2027-07-15T22:59:59.999999-04:00"},
    "expires_naive": {"expires_at": "2027-07-15T23:59:59.999999"},
    "expires_utc_valid": {"expires_at": "2027-07-16T03:59:59.999999Z"},
    "active_bool": {"active_streak": True}, "active_negative": {"active_streak": -1},
    "active_58": {"active_streak": 58, "best_streak": 58},
    "best_bool": {"best_streak": True}, "best_below_active": {"best_streak": 6},
    "best_58": {"best_streak": 58}, "saver_list": {"saver_state": []},
    "saver_unknown": {"saver_state": "uninitialized"},
    "recorded_naive": {"recorded_at": "2027-07-15T12:00:00"},
    "recorded_bad": {"recorded_at": "junk"}, "source_int": {"source": 3},
    "empty_source_valid": {"source": ""}, "reason_null": {"reason": None},
    "reason_empty_valid": {"reason": ""}, "reason_int": {"reason": 3},
    "extra_legacy_valid": {"source_date": "garbage", "saver_available": [], "override_expires_at": "bad"},
    "legacy_manual_above57": {"schema_version": "bts_contest_streak_manual_v2", "active_streak": 58, "best_streak": 58},
}.items():
    cases[name] = dict(BASE, **changes)
rows = []
for name, body in cases.items():
    p = folder(f"parse-{name}")
    f = p / "account_state/contest_streak.manual.json"
    f.write_text(json.dumps(body))
    rows.append(dict(case=name, parse=run(lambda: cs._parse_state_file(f))))
OUT["main_property_cases"] = rows

p = folder("b-fails-after-a")
manual = p / "account_state/contest_streak.manual.json"
old = dict(active_streak=12, best_streak=12, source_date="2027-07-15", source="old",
           schema_version="bts_contest_streak_manual_v2", override_expires_at=(NOW + timedelta(days=1)).isoformat())
manual.write_text(json.dumps(old))
ss.transition_saver_state(p, expected_prior="uninitialized", new_state="active", season=2027, source="old")
ss.transition_saver_state(p, expected_prior="active", new_state="used", season=2027, source="manual_observation", force=True)
with patch("bts.util.os.replace", side_effect=OSError("injected manual rename failure")):
    failed = run(lambda: atomic_write_text(manual, json.dumps(BASE)))
OUT["row3_primitives_b_failure_old_file"] = dict(failure=failed, old_file_intact=json.loads(manual.read_text()) == old,
                                               decision=run(lambda: cs.load_decision_streak_state(p, now=NOW)))

p = folder("c-fails-after-b")
ss.transition_saver_state(p, expected_prior="uninitialized", new_state="active", season=2027, source="manual_observation", force=True)
manual = p / "account_state/contest_streak.manual.json"
atomic_write_text(manual, json.dumps(BASE))
history = p / "account_state/contest_manual_history.jsonl"
history.mkdir()


def append():
    with history.open("a") as f:
        f.write(json.dumps(dict(BASE, written_at=NOW.isoformat())) + "\n")


OUT["row3_primitives_c_failure"] = dict(append=run(append), manual_intact=json.loads(manual.read_text()) == BASE,
                                        flag=run(lambda: ss.load_saver_state(p, season=2027)))

p = folder("flag-changed-after-observation")
f = p / "account_state/contest_streak.manual.json"
f.write_text(json.dumps(dict(BASE, source_date="2027-07-15", override_expires_at=(NOW + timedelta(hours=1)).isoformat())))
ss.transition_saver_state(p, expected_prior="uninitialized", new_state="active", season=2027, source="manual_observation", force=True)
ss.transition_saver_state(p, expected_prior="active", new_state="used", season=2027, source="cli")
OUT["real_flag_authority"] = run(lambda: cs.load_decision_streak_state(p, now=NOW))

def default(v):
    if isinstance(v, (Path, date, datetime)):
        return str(v)
    raise TypeError(type(v).__name__)


(HERE / "edge-results.json").write_text(json.dumps(OUT, indent=2, default=default) + "\n")
print(json.dumps({"property_cases": len(rows), "accepted_by_main": [x["case"] for x in rows if x["parse"]["ok"]],
                  "partial_b": OUT["row3_primitives_b_failure_old_file"], "partial_c": OUT["row3_primitives_c_failure"],
                  "flag_authority": OUT["real_flag_authority"]}, indent=2, default=default))
