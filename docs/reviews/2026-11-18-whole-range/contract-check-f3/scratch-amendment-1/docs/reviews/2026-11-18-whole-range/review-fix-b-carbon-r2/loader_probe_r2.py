"""Item 1 probe: exact live-loader & health parity against base 9e0ec28a + AST call-site audit."""
import ast
import hashlib
import inspect
import itertools
import json
import os
import pickle
import signal
import sys
from datetime import datetime, time, timedelta
from pathlib import Path
from unittest.mock import patch
from zoneinfo import ZoneInfo

REVIEW = Path(__file__).resolve().parent
revision = sys.argv[1]
assert revision in ("base", "head", "compare"), revision

if revision == "compare":
    base_bytes = (REVIEW / "loader-base.pkl").read_bytes()
    head_bytes = (REVIEW / "loader-head.pkl").read_bytes()
    base_rows = pickle.loads(base_bytes)
    head_rows = pickle.loads(head_bytes)
    assert len(base_rows) == len(head_rows), (len(base_rows), len(head_rows))
    mismatches = [
        (i, b[0], b[1]) for i, (b, h) in enumerate(zip(base_rows, head_rows)) if b != h
    ]
    assert not mismatches and base_bytes == head_bytes, mismatches[:5]

    # AST call-site audit across HEAD src/
    src_root = REVIEW.parents[1] / "src"
    targets = {
        "load_decision_streak_state",
        "load_contest_streak_state",
        "load_saver_state",
        "live_automatic_refusal",
        "switch_state",
        "read_regular_file_text",
    }
    call_sites = []
    for py in sorted(src_root.rglob("*.py")):
        tree = ast.parse(py.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = None
                if isinstance(node.func, ast.Name) and node.func.id in targets:
                    name = node.func.id
                elif isinstance(node.func, ast.Attribute) and node.func.attr in targets:
                    name = node.func.attr
                if name:
                    kws = [kw.arg for kw in node.keywords]
                    call_sites.append({
                        "file": py.relative_to(src_root.parent).as_posix(),
                        "line": node.lineno,
                        "func": name,
                        "pos_args": len(node.args),
                        "keywords": kws,
                    })
    non_default = [
        c for c in call_sites
        if any(k in ("omit_selected_dated", "switch", "read_text") for k in c["keywords"])
    ]
    summary = {
        "states_compared": len(head_rows),
        "byte_identical_pickles": base_bytes == head_bytes,
        "sha256": hashlib.sha256(head_bytes).hexdigest(),
        "total_target_calls_in_src": len(call_sites),
        "non_default_status_option_calls": non_default,
        "read_regular_file_text_calls": [c for c in call_sites if c["func"] == "read_regular_file_text"],
        "all_calls": call_sites,
    }
    (REVIEW / "loader-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "all_calls"}, indent=2))
    raise SystemExit(0)

ROOT = REVIEW.parents[1] if revision == "head" else REVIEW / "base"
sys.path.insert(0, str(ROOT / "src"))

from bts import contest_state as cs, saver_state as saver, season_contract as sc, util, web
from bts.health import contest_state as health

expected_src = (ROOT / "src").resolve()
for mod in (cs, saver, sc, util, web, health):
    assert Path(mod.__file__).resolve().is_relative_to(expected_src), (revision, mod.__file__)

ET = ZoneInfo("America/New_York")
PICKS = REVIEW / f"loader-fixture-{revision}"
ACCOUNT = PICKS / "account_state"
ACCOUNT.mkdir(parents=True, exist_ok=True)
MANUAL, AUTO, SWITCH, SAVER, MODEL = [
    PICKS / name for name in (
        "account_state/contest_streak.manual.json",
        "account_state/contest_streak.json",
        "account_state/season_contract.json",
        "account_state/saver_state.json",
        "streak.json",
    )
]
MIDNIGHT = (
    datetime(2027, 7, 15, 23, 59, 59, tzinfo=ET),
    datetime(2027, 12, 31, 23, 59, 59, tzinfo=ET),
)
BAD = (b"", b"{", b"\xff", b"[]", b"null", b"9" * 5000, b"[" * 3000 + b"]" * 3000)
DATES = [-2, -1, 0, 1]
_managed = set()


def set_path(path, value):
    if path.is_symlink() or path.is_file() or (path.exists() and not path.is_dir()):
        path.unlink()
    elif path.is_dir():
        path.rmdir()
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
    flags.append(dict(season=year - 1, state="active"))
    switches = [switch, switch | dict(status="revoked", revoked_at=recorded),
                switch | dict(season=year - 1), switch | dict(note=[])]
    models = [dict(season=year, streak=6, saver_available=True),
              dict(season=year - 1, streak=6, saver_available=True),
              dict(season=year, streak=[], saver_available=True)]
    return v1, v2, manual, switches, flags, models


def normalize_str(text):
    return text.replace(str(PICKS), "<PICKS>")


def capture(callback):
    def error(exc):
        return (
            type(exc).__module__,
            type(exc).__name__,
            normalize_str(str(exc)),
            tuple(normalize_str(a) if isinstance(a, str) else a for a in exc.args),
            error(exc.__cause__) if exc.__cause__ is not None else None,
        )
    try:
        val = callback()
        if hasattr(val, "__dataclass_fields__"):
            from dataclasses import asdict
            d = asdict(val)
            for k, v in list(d.items()):
                if isinstance(v, Path):
                    d[k] = normalize_str(str(v))
                elif isinstance(v, str):
                    d[k] = normalize_str(v)
            return ("return", type(val).__name__, d)
        return ("return", val)
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
        tag, *rest = capture(lambda: health.check(PICKS, now=now, expected=required))
        if tag == "return":
            value = rest[0]
            alerts = []
            for a in value:
                prefix = "If MLB shows the saver not earned yet"
                msg = normalize_str(a.message)
                if a.source == "saver_state" and prefix in msg:
                    changed.append(msg)
                    msg = msg.split("with ", 1)[0] + "with <advice>"
                alerts.append((a.level, a.source, a.incident_key, msg))
            results[f"health-{required}"] = (tag, alerts)
        else:
            results[f"health-{required}"] = (tag, *rest)
    results["switch"] = capture(lambda: sc.switch_state(PICKS, now))
    results["saver"] = capture(lambda: saver.load_saver_state(PICKS, season=now.year))
    results["web-saver"] = capture(lambda: {
        k: (normalize_str(v) if isinstance(v, str) else v)
        for k, v in web.saver_dashboard_context(PICKS, now=now).items()
    })
    results["latest"] = capture(lambda: cs.latest_resolved_pick_date(PICKS))
    results["gap"] = capture(lambda: cs.resolved_pick_settlement_gap(PICKS, now.date() - timedelta(days=2)))
    return results, changed


class TimedOut(Exception):
    pass


def timed_out(*_):
    raise TimedOut("blocked in default reader")


def main():
    if revision == "head":
        assert not hasattr(util, "read_utf8"), "util.read_utf8 should be removed"
        for fn in (
            cs._parse_state_file, cs.load_contest_streak_state, cs._resolved_pick_dates,
            cs.latest_resolved_pick_date, cs.resolved_pick_settlement_gap,
            cs._has_unconfirmed_miss, cs.contest_state_is_fresh, cs._load_model_diagnostics,
            cs.load_decision_streak_state, saver.load_saver_state, sc.switch_state,
        ):
            assert "read_text" not in inspect.signature(fn).parameters, fn.__name__

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
    if ACCOUNT.exists():
        ACCOUNT.rmdir()
    if PICKS.exists():
        PICKS.rmdir()
    data = pickle.dumps(rows, protocol=5)
    (REVIEW / f"loader-{revision}.pkl").write_bytes(data)
    (REVIEW / f"loader-{revision}-advice.txt").write_text("\n".join(sorted(advice)) + "\n")
    print(f"{revision}: {len(rows)} states, {len(advice)} distinct changed warnings, {hashlib.sha256(data).hexdigest()}")


main()
