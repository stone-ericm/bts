"""Read-only source review probes. All generated files stay beside this script."""
import copy

def deny_network(event, args):
    if event in {"socket.connect", "socket.getaddrinfo"}:
        raise RuntimeError("network forbidden in contract probe")

import sys
sys.addaudithook(deny_network)
import importlib
import json
import logging
import sys
from collections import Counter
from dataclasses import asdict, replace
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[2]
SCRATCH = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
import click
from click.testing import CliRunner
import bts.contest_state as cs
import bts.saver_state as ss
from bts.health.contest_state import check
from bts.util import atomic_write_text

logging.disable(logging.CRITICAL)
ET = ZoneInfo("America/New_York")
NOW = datetime(2027, 7, 15, 12, tzinfo=ET)
D = NOW.date()
V3 = "bts_contest_streak_manual_v3"
OUT = {}


def folder(name):
    p = SCRATCH / "probe-data-v2" / name
    p.mkdir(parents=True, exist_ok=False)
    return p


def write(p, name, body):
    f = p / "account_state" / name
    f.parent.mkdir(parents=True, exist_ok=True)
    if body == "DIRECTORY":
        f.mkdir()
    elif isinstance(body, bytes):
        f.write_bytes(body)
    else:
        f.write_text(json.dumps(body))
    return f


def observation(d=D, **kw):
    obj = dict(schema_version=V3, effective_date=d.isoformat(), season=d.year,
               expires_at=datetime.combine(d, time.max, ET).isoformat(),
               active_streak=7, best_streak=9, saver_state="active",
               recorded_at=NOW.isoformat(), source="manual_cli")
    obj.update(kw)
    return obj


def automatic(d=D):
    return dict(schema_version="bts_contest_streak_auto_v1", active_streak=12,
                best_streak=12, source_date=d.isoformat(), source="mlb_bts_profile",
                recorded_at=NOW.isoformat())


def result(fn):
    try:
        r = fn()
        if hasattr(r, "__dataclass_fields__"):
            r = asdict(r)
        elif isinstance(r, list):
            r = [asdict(x) for x in r]
        return dict(ok=True, value=r)
    except Exception as exc:
        return dict(ok=False, error=type(exc).__name__, message=str(exc)[:350])


legacy = dict(active_streak=7, best_streak=9, source_date=D.isoformat(), source="manual_cli")
variants = {"absent": None, "schema_less": legacy, "v1": dict(legacy, schema_version="bts_contest_streak_manual_v1")}
for version in ("v1", "v2"):
    for expiry, offset in (("missing", None), ("past", -1), ("future", 1)):
        obj = dict(legacy, schema_version=f"bts_contest_streak_manual_{version}")
        if offset is not None:
            obj["override_expires_at"] = (NOW + timedelta(days=offset)).isoformat()
        variants[f"{version}_{expiry}"] = obj
for offset, name in ((-1, "yesterday"), (0, "today"), (1, "tomorrow")):
    variants[f"v3_{name}"] = observation(D + timedelta(days=offset))
    variants[f"v3_{name}_mixed"] = observation(D + timedelta(days=offset), override_expires_at=(NOW + timedelta(days=4)).isoformat())
variants.update({
    "unknown_schema_unexpired": dict(legacy, schema_version="bts_contest_streak_manual_typo", override_expires_at=(NOW + timedelta(days=1)).isoformat()),
    "missing_schema_unexpired": dict(legacy, override_expires_at=(NOW + timedelta(days=1)).isoformat()),
    "v3_extra_legacy_saver": observation(saver_available={"junk": 1}),
    "v3_bad_reason": observation(reason=None),
    "v3_bad_season": observation(season=True),
    "v3_naive_recorded": observation(recorded_at="2027-07-15T12:00:00"),
    "v3_empty_source": observation(source=""),
    "v3_bad_streak": observation(active_streak=True),
    "v3_58": observation(active_streak=58, best_streak=58),
    "malformed": b"{", "non_utf8": b"\xff", "non_object": [],
    "directory": "DIRECTORY", "unreadable": legacy,
    "long_integer": b'{"active_streak":' + b"1" * 5000 + b"}",
})
for key in ("source", "best_streak", "effective_date", "schema_version"):
    obj = observation()
    del obj[key]
    variants[f"v3_missing_{key}"] = obj
autos = {"absent": None, "valid": automatic(), "prior_season": automatic(date(2026, 9, 27)),
         "refused": dict(automatic(), active_streak=58, best_streak=58), "corrupt": b"{",
         "undated": dict(active_streak=12, best_streak=12)}
matrix = []
for mname, mbody in variants.items():
    for aname, abody in autos.items():
        p = folder(f"matrix-{mname}-{aname}")
        mp = None if mbody is None else write(p, "contest_streak.manual.json", mbody)
        if mname == "unreadable":
            mp.chmod(0)
        if abody is not None:
            write(p, "contest_streak.json", abody)
        write(p, "saver_state.json", dict(state="active", season=2027))
        row = dict(manual=mname, automatic=aname)
        row["reader"] = result(lambda: cs.load_contest_streak_state(p, now=NOW))
        for required in (False, True):
            row[f"decision_{required}"] = result(lambda: cs.load_decision_streak_state(p, now=NOW, require_contest_state=required))
            row[f"health_{required}"] = result(lambda: check(p, now=NOW, expected=required))
        matrix.append(row)
        if mname == "unreadable":
            mp.chmod(0o600)
OUT["main_matrix"] = matrix

cli_mod = importlib.import_module("bts.cli")


class Frozen(datetime):
    @classmethod
    def now(cls, tz=None):
        return NOW.astimezone(tz) if tz else NOW.replace(tzinfo=None)


commands = []
for name, args in [
    ("legacy", ["--streak", "7", "--best-streak", "9", "--saver-unavailable", "--username", "probe", "--reason", "fixed"]),
    ("legacy_58", ["--streak", "58", "--best-streak", "58"]),
    ("date", ["--streak", "7", "--date", "2027-07-15"]),
    ("saver", ["--streak", "7", "--saver", "active"]),
]:
    p = folder(f"command-{name}")
    with patch("datetime.datetime", Frozen), patch.object(ss, "datetime", Frozen):
        r = CliRunner().invoke(cli_mod.cli, ["set-contest-streak", "--picks-dir", str(p), *args])
        manual = p / "account_state/contest_streak.manual.json"
        before = manual.read_bytes() if manual.exists() else None
        r2 = CliRunner().invoke(cli_mod.cli, ["set-contest-streak", "--picks-dir", str(p), *args])
    commands.append(dict(name=name, exit=r.exit_code, output=r.output, exception=repr(r.exception),
                         repeat_exit=r2.exit_code, repeat_identical=before == (manual.read_bytes() if manual.exists() else None),
                         body=json.loads(before) if before else None,
                         flag=result(lambda: ss.load_saver_state(p, season=2027))))
OUT["real_cli"] = commands
OUT["click_dates"] = {s: result(lambda s=s: click.DateTime(formats=["%Y-%m-%d"]).convert(s, None, None))
                      for s in ["2027-7-5", "2027-07-15", "20270715", "0001-01-01", "9999-12-31"]}

p = folder("boundary")
write(p, "contest_streak.manual.json", observation())
parsed = cs._parse_state_file(p / "account_state/contest_streak.manual.json")
boundary = []
for d in [D, date(2027, 3, 14), date(2027, 11, 7), date(2027, 12, 31)]:
    expires = datetime.combine(d, time.max, ET)
    mapped = replace(parsed, source_date=d, override_expires_at=expires)
    # Map the row-7 fields on the EXISTING classifier; this is not a v3 implementation.
    for n in [datetime.combine(d, time.min, ET) - timedelta(microseconds=1),
              datetime.combine(d, time.min, ET), expires, expires + timedelta(microseconds=1)]:
        boundary.append(dict(date=str(d), now=n.isoformat(),
                             required_in_effect=n.astimezone(ET).date() == d,
                             existing_classifier=cs.classify_best_streak(mapped, now=n, now_year=n.astimezone(ET).year)))
OUT["existing_classifier_with_row7_mapping"] = boundary

p = folder("race")
ss.transition_saver_state(p, expected_prior="uninitialized", new_state="active", season=2027, source="probe")
expected = ss.load_saver_state(p, season=2027).state
ss.transition_saver_state(p, expected_prior="active", new_state="used", season=2027, source="intervening_use")
accepted = ss.transition_saver_state(p, expected_prior=expected, new_state="not_earned", season=2027, source="manual_observation", force=True)
OUT["real_forced_transition_race"] = dict(expected=expected, result=accepted, flag=ss.load_saver_state(p, season=2027).state,
                                          audit=[json.loads(s) for s in (p / "account_state/saver_transitions.jsonl").read_text().splitlines()])

p = folder("partial_failure")
write(p, "contest_streak.json", automatic())
mp = write(p, "contest_streak.manual.json", "DIRECTORY")
ss.transition_saver_state(p, expected_prior="uninitialized", new_state="used", season=2027, source="manual_observation", force=True)
OUT["row3_primitives_manual_write_failure"] = dict(write=result(lambda: atomic_write_text(mp, json.dumps(observation(saver_state="used")))),
                                                   flag=ss.load_saver_state(p, season=2027).state,
                                                   decision=result(lambda: cs.load_decision_streak_state(p, now=NOW)))

p = folder("single_flag_two_seasons")
ss.transition_saver_state(p, expected_prior="uninitialized", new_state="active", season=2027, source="today")
ss.transition_saver_state(p, expected_prior="uninitialized", new_state="not_earned", season=2028, source="future_observation", force=True)
OUT["future_season_single_flag"] = dict(today=asdict(ss.load_saver_state(p, season=2027)), future=asdict(ss.load_saver_state(p, season=2028)))

p = folder("unreadable_saver")
write(p, "saver_state.json", b"\xff")
OUT["invalid_flag_transition"] = dict(before=asdict(ss.load_saver_state(p, season=2027)),
                                      transition=result(lambda: ss.transition_saver_state(p, expected_prior="uninitialized", new_state="active", season=2027, source="manual_observation", force=True)),
                                      after=asdict(ss.load_saver_state(p, season=2027)))
p = folder("long_integer_saver")
write(p, "saver_state.json", b'{"season":2027,"state":"used","unknown":' + b"1" * 5000 + b"}")
OUT["flag_decoder_limit"] = result(lambda: ss.load_saver_state(p, season=2027))

p = folder("split_clock")
write(p, "contest_streak.json", automatic(date(2028, 1, 1)))
write(p, "saver_state.json", dict(state="active", season=2028))
clocks = iter([datetime(2027, 12, 31, 23, 59, 59, 999999, tzinfo=ET), datetime(2028, 1, 1, tzinfo=ET)])
seen = []


class StepClock(datetime):
    @classmethod
    def now(cls, tz=None):
        n = next(clocks)
        seen.append(n.isoformat())
        return n.astimezone(tz) if tz else n.replace(tzinfo=None)


with patch.object(cs, "datetime", StepClock):
    OUT["real_decision_two_clocks"] = result(lambda: cs.load_decision_streak_state(p, require_contest_state=True))
OUT["real_decision_two_clocks"]["clock_reads"] = seen

from bts.picks import DailyPick, Pick, save_pick, classify_pick_lock_state
from bts.strategy import select_pick
import pandas as pd
p = folder("locked_pick")
pick = Pick("Before correction", 1, "T", 1, "P", 2, .90, [], False, 123, "2027-07-15T20:00:00Z")
daily = DailyPick(D.isoformat(), NOW.isoformat(), pick, None, None, notification_sent=True, notification_id="probe-delivered")
save_pick(daily, p)
before = (p / f"{D}.json").read_bytes()
sel = select_pick(pd.DataFrame([dict(game_pk=123)]), D.isoformat(), p,
                  streak=0, saver_available=False, best_streak=0, best_status="trusted",
                  game_statuses_detailed={123: {"abstract": "P", "detailed": "Scheduled"}})
OUT["real_locked_pick_after_corrected_inputs"] = dict(lock=asdict(classify_pick_lock_state(daily, str(D))),
                                                      selected=sel.pick_result.daily.pick.batter_name,
                                                      locked=sel.pick_result.locked,
                                                      bytes_unchanged=before == (p / f"{D}.json").read_bytes())

def default(obj):
    if isinstance(obj, (Path, date, datetime)):
        return str(obj)
    raise TypeError(type(obj).__name__)


(SCRATCH / "probe-results.json").write_text(json.dumps(OUT, indent=2, default=default) + "\n")
summary = dict(manual_cases=len(variants), automatic_cases=len(autos), matrix_cells=len(matrix),
               real_reader_health_decision_calls=len(matrix) * 5,
               cli=[dict(name=r["name"], exit=r["exit"], repeat_identical=r["repeat_identical"]) for r in commands],
               classifier_last_instant=[r for r in boundary if r["required_in_effect"] and r["existing_classifier"][1] != "trusted"],
               race=OUT["real_forced_transition_race"],
               split_clock=OUT["real_decision_two_clocks"],
               locked_pick=OUT["real_locked_pick_after_corrected_inputs"])
print(json.dumps(summary, indent=2, default=default))
