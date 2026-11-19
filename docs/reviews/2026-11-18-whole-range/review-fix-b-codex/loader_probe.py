"""Compare exact live-loader and health results against archived base source."""
import hashlib
import itertools
import json
import os
import pickle
import signal
import stat
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

REVIEW = Path(__file__).resolve().parent
revision = sys.argv[1]
ROOT = REVIEW.parents[1] if revision == "head" else REVIEW / "base"
sys.path.insert(0, str(ROOT / "src"))

from bts import contest_state as cs, saver_state as saver, season_contract as sc, web
from bts.health import contest_state as health

ET = ZoneInfo("America/New_York")
PICKS = REVIEW / "loader-fixture"
ACCOUNT = PICKS / "account_state"
ACCOUNT.mkdir(parents=True, exist_ok=True)
MANUAL, AUTO, SWITCH, SAVER, MODEL = [
    PICKS / name for name in (
        "account_state/contest_streak.manual.json", "account_state/contest_streak.json",
        "account_state/season_contract.json", "account_state/saver_state.json", "streak.json",
    )
]
MIDNIGHT = (datetime(2027, 7, 15, 23, 59, 59, tzinfo=ET),
            datetime(2027, 12, 31, 23, 59, 59, tzinfo=ET))
BAD = (b"", b"{", b"\xff", b"[]", b"null", b"9" * 5000, b"[" * 3000 + b"]" * 3000)
DATES = [-2, -1, 0, 1]
_managed = set()


def set_path(path, value):
    if path.is_dir():
        path.rmdir()
    elif path.exists():
        path.unlink()
    if value == "directory":
        path.mkdir()
    elif value == "fifo":
        os.mkfifo(path)
    elif isinstance(value, bytes):
        path.write_bytes(value)
    elif value is not None:
        path.write_text(json.dumps(value))
    _managed.add(path)


def inputs(now):
    day = now.date()
    year = day.year
    identifier = "a" * 32
    recorded = now.isoformat()
    source = (day - timedelta(days=2)).isoformat()
    v1 = dict(schema_version="bts_contest_streak_auto_v1", active_streak=12, best_streak=13,
              source="mlb_bts_profile", source_date=source, recorded_at=recorded)
    v2 = v1 | dict(schema_version="bts_contest_streak_auto_v2", season=year,
                   activation_id=identifier)
    switch = dict(schema_version="bts_season_contract_v1", season=year, status="automatic",
                  switched_at=recorded, revoked_at=None, activation_id=identifier, note=None)
    manual = []
    for shift in (-1, 0, 1):
        effective = day + timedelta(days=shift)
        manual.append(dict(schema_version="bts_contest_streak_manual_v3",
                           effective_date=effective.isoformat(), season=effective.year,
                           expires_at=datetime.combine(effective, time.max, ET).isoformat(),
                           active_streak=11, best_streak=12, saver_state="not_earned",
                           recorded_at=recorded, source="manual"))
    legacy = dict(active_streak=4, source_date=source, best_streak=9)
    manual.extend(legacy | dict(override_expires_at=(now + timedelta(hours=h)).isoformat())
                  for h in (-1, 1))
    flags = [dict(season=year, state=s) for s in ("not_earned", "active", "used")]
    flags.append(dict(season=year-1, state="active"))
    switches = [switch, switch | dict(status="revoked", revoked_at=recorded),
                switch | dict(season=year-1), switch | dict(note=[])]
    models = [dict(season=year, streak=6, saver_available=True),
              dict(season=year-1, streak=6, saver_available=True),
              dict(season=year, streak=[], saver_available=True)]
    return v1, v2, manual, switches, flags, models


def capture(callback):
    def error(exc):
        return (type(exc).__module__, type(exc).__name__, str(exc), exc.args,
                error(exc.__cause__) if exc.__cause__ is not None else None)
    try:
        return ("return", callback())
    except Exception as exc:
        return ("raise", error(exc))


def compare(now):
    results = {}
    changed = []
    for required in (False, True):
        results[f"decision-{required}"] = capture(
            lambda: cs.load_decision_streak_state(PICKS, now=now, require_contest_state=required))
        results[f"contest-{required}"] = capture(
            lambda: cs.load_contest_streak_state(PICKS, now=now, require_dated_auto=required))
        tag, value = capture(lambda: health.check(PICKS, now=now, expected=required))
        if tag == "return":
            alerts = []
            for a in value:
                prefix = "If MLB shows the saver not earned yet"
                if a.source == "saver_state" and prefix in a.message:
                    changed.append(a.message)
                    message = a.message.split("with ", 1)[0] + "with <advice>"
                else:
                    message = a.message
                alerts.append((a.level, a.source, a.incident_key, message))
            value = alerts
        results[f"health-{required}"] = (tag, value)
    results["switch"] = capture(lambda: sc.switch_state(PICKS, now))
    results["saver"] = capture(lambda: saver.load_saver_state(PICKS, season=now.year))
    results["web-saver"] = capture(lambda: web.saver_dashboard_context(PICKS, now=now))
    results["latest"] = capture(lambda: cs.latest_resolved_pick_date(PICKS))
    results["gap"] = capture(lambda: cs.resolved_pick_settlement_gap(PICKS, now.date()-timedelta(days=2)))
    return results, changed


class TimedOut(Exception):
    pass


def timed_out(*_):
    raise TimedOut("blocked in default reader")


def main():
    rows = []
    advice = set()
    for instant in MIDNIGHT:
        for now in (instant, instant + timedelta(seconds=2)):
            v1, v2, manual, switches, flags, models = inputs(now)
            values = [v2, None, switches[0], flags[0], models[0]]
            paths = [AUTO, MANUAL, SWITCH, SAVER, MODEL]
            variants = [
                [None, v1, v2, v2 | {"activation_id": "b" * 32}, v2 | {"source_date": None}],
                [None, *manual], [None, *switches], [None, *flags], [None, *models],
            ]
            pick_paths = [PICKS / ((now.date() + timedelta(days=d)).isoformat() + ".json") for d in DATES]
            cases = [("empty", [None] * 5, [None] * 4)]
            for i, items in enumerate(variants):
                for j, value in enumerate([*items, *BAD, "directory"]):
                    selection = values[:]
                    selection[i] = value
                    cases.append((f"field-{i}-{j}", selection, [None] * 4))
            for i, (auto, observed, switch, flag) in enumerate(itertools.product(
                variants[0][:3], variants[1], variants[2], variants[3],
            )):
                picks = ([{"result": "hit"}, {"result": "miss"}, {"result": None}, b"[]"]
                         if i % 2 else [None, {"result": "hit"}, b"\xff", {"result": None}])
                cases.append((f"cross-{i}", [auto, observed, switch, flag, models[0]], picks))
            for case, state, picks in cases:
                for path, val in zip(paths + pick_paths, state + picks):
                    set_path(path, val)
                output, warnings = compare(now)
                advice.update(warnings)
                rows.append((now.isoformat(), case, output))
            for special in paths + [pick_paths[1]]:
                for path, val in zip(paths + pick_paths, values + [None] * 4):
                    set_path(path, val)
                original = Path.open

                def refused(path, *args, **kwargs):
                    if path == special:
                        raise PermissionError(13, "Permission denied", str(path))
                    return original(path, *args, **kwargs)

                with patch.object(Path, "open", refused):
                    output, warnings = compare(now)
                advice.update(warnings)
                rows.append((now.isoformat(), f"permission-{special.name}", output))
            delayed = [
                (MANUAL, lambda: cs.load_contest_streak_state(PICKS, now=now)),
                (AUTO, lambda: cs.load_decision_streak_state(PICKS, now=now, require_contest_state=True)),
                (SWITCH, lambda: sc.switch_state(PICKS, now)),
                (SAVER, lambda: saver.load_saver_state(PICKS, season=now.year)),
                (MODEL, lambda: cs.load_decision_streak_state(PICKS, now=now)),
                (pick_paths[1], lambda: cs.latest_resolved_pick_date(PICKS)),
            ]
            for special, fn in delayed:
                for path, val in zip(paths, values):
                    set_path(path, val)
                set_path(special, "fifo")
                old = signal.signal(signal.SIGALRM, timed_out)
                signal.setitimer(signal.ITIMER_REAL, 0.05)
                try:
                    got = capture(fn)
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                    signal.signal(signal.SIGALRM, old)
                    set_path(special, None)
                assert got[1][1] == "TimedOut", (special, got)
                rows.append((now.isoformat(), f"fifo-{special.name}", got))
            for path in pick_paths:
                set_path(path, None)
    for path in _managed:
        set_path(path, None)
    data = pickle.dumps(rows, protocol=5)
    (REVIEW / f"loader-{revision}.pkl").write_bytes(data)
    (REVIEW / f"loader-{revision}-advice.txt").write_text("\n".join(sorted(advice)) + "\n")
    print(f"{revision}: {len(rows)} states, {len(advice)} distinct changed warnings, {hashlib.sha256(data).hexdigest()}")


if revision == "head":
    now = MIDNIGHT[0]
    _, auto, _, switches, flags, _ = inputs(now)
    for path, val in ((AUTO, auto), (SWITCH, switches[0]), (SAVER, flags[0])):
        set_path(path, val)
    calls = []

    def custom_reader(path):
        calls.append(path.name)
        return json.dumps(flags[1]) if path == SAVER else path.read_text()

    default = cs.load_decision_streak_state(PICKS, now=now, require_contest_state=True)
    custom = cs.load_decision_streak_state(PICKS, now=now, require_contest_state=True, read_text=custom_reader)
    assert calls.count(SAVER.name) == 1 and (default.saver_flag_state, custom.saver_flag_state) == ("not_earned", "active")
    set_path(AUTO, None)
    calls.clear()
    model = cs.load_decision_streak_state(PICKS, now=now, read_text=custom_reader)
    assert model.saver_flag_state == "not_earned" and SAVER.name not in calls
    print("Custom reader reaches the automatic saver branch; model-only branch does not forward it")

main()
