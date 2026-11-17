"""Round-two status review: one immutable offline wrapper family, two source trees."""
from __future__ import annotations

import contextlib
import datetime as dt
import hashlib
import io
import itertools
import json
import os
from pathlib import Path
import re
import resource
import shutil
import subprocess
import sys
import tarfile
import time
from unittest.mock import patch
from zoneinfo import ZoneInfo

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
REAL_DT = dt.datetime
ACTIVE = None
TRACE = []


def audit(event, args):
    if event.startswith("socket.") and event not in {"socket.__new__"}:
        raise AssertionError("network forbidden: " + event)
    if ACTIVE is not None:
        if event == "open":
            path, mode, flags = args
            if isinstance(path, (str, bytes)) and str(ACTIVE) in str(path):
                TRACE.append([event, str(path), mode, flags])
                if (isinstance(mode, str) and any(c in mode for c in "wax+")) or flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND):
                    raise AssertionError("status wrote: " + str(path))
        if event in {"os.mkdir", "os.rename", "os.remove", "os.rmdir", "os.chmod", "os.symlink", "os.link", "os.truncate"}:
            raise AssertionError("status mutation: " + event)


sys.addaudithook(audit)


def doc(**kw):
    return dict(schema_version="bts_season_contract_v1", season=2027, status="automatic",
                activation_id="a" * 32, switched_at="2027-07-15T16:00:00Z", revoked_at=None, note=None) | kw


AUTO = dict(schema_version="bts_contest_streak_auto_v2", active_streak=7, best_streak=9,
            season=2027, activation_id="a" * 32, source="mlb_bts_profile",
            source_date="2027-07-14", recorded_at="2027-07-15T16:00:00Z")
NAMES = ["season_contract.json", "contest_streak.manual.json", "contest_streak.json", "saver_state.json", "contest_ledger.jsonl"]
GOOD = [doc(), dict(active_streak=7, best_streak=9), AUTO, dict(state="active", season=2027), dict(recorded_at="row-0", season=2027)]
KEYS = ["recorded_at", "active_streak", "best_streak", "source_date", "season", "switched", "activation_id", "settled_by_year", "settled_unmapped", "unsettled"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def tree(root):
    return {str(p.relative_to(root)): ("link", os.readlink(p)) if p.is_symlink() else
            ("dir", p.stat().st_mode) if p.is_dir() else
            ("file", p.stat().st_mode, hashlib.sha256(p.read_bytes()).hexdigest())
            for p in root.rglob("*")}


def install(path, kind, good):
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.is_dir():
        path.rmdir()
    if kind == "absent":
        return
    if kind == "directory":
        path.mkdir()
    elif kind == "dangling":
        path.symlink_to("missing-target")
    else:
        value = json.dumps(good).encode() if kind == "good" else {
            "broken": b'{"bad":', "utf8": b'\xff\xfe',
        }[kind]
        path.write_bytes(value)


def worker(label, source, family):
    global ACTIVE
    sys.path.insert(0, str(source))
    import pandas
    import bts.cli as cli
    from click.testing import CliRunner
    assert Path(cli.__file__).is_relative_to(source), cli.__file__
    fixed = REAL_DT(2027, 7, 15, 12, tzinfo=ZoneInfo("America/New_York"))

    class Clock(REAL_DT):
        @classmethod
        def now(cls, tz=None):
            return fixed.astimezone(tz) if tz else fixed.replace(tzinfo=None)

    dt.datetime = Clock
    runner = CliRunner()
    root = HERE / ("status-work-" + label + "-" + family)
    root.mkdir()
    picks = root / "picks"
    acc = picks / "account_state"
    acc.mkdir(parents=True)
    records = []

    def invoke(key, season=None, inspect=True):
        global ACTIVE
        before = tree(root) if inspect else None
        TRACE.clear()
        ACTIVE = root
        try:
            res = runner.invoke(cli.cli, ["season-contract", "status", "--picks-dir", str(picks),
                                         *([] if season is None else ["--season", str(season)])])
        finally:
            ACTIVE = None
        if inspect:
            assert tree(root) == before, key
        output = res.output.replace(str(root), "<case>")
        record = dict(key=key, exit=res.exit_code, output=output,
                      exception=None if res.exception is None else type(res.exception).__name__)
        records.append(record)
        if label == "head":
            assert res.exit_code in (0, 1), record
            assert len([x for x in output.splitlines() if x.startswith("ledger:")]) == 1, record
            for section in ["manual:", "automatic:", "saver:", "ledger:"]:
                assert section in output, record
        return record

    if family == "matrix":
        kinds = ["absent", "good", "broken", "utf8", "directory", "dangling"]
        for ix, combo in enumerate(itertools.product(kinds, repeat=5)):
            for name, kind, good in zip(NAMES, combo, GOOD):
                install(acc / name, kind, good)
            for season in [None, 2028]:
                r = invoke([*combo, season], season)
                if label == "head":
                    assert r["exit"] == (0 if combo[0] in {"absent", "good"} else 1), r
            if ix % 1500 == 0:
                print(label, family, ix, flush=True)
        for name in NAMES:
            install(acc / name, "absent", {})
        acc.rmdir()
        picks.rmdir()
        invoke("cold-no-directory")
        acc.mkdir(parents=True)
        for fault in ["lstat", "open"]:
            for name in NAMES:
                path = acc / name
                path.write_text(json.dumps(GOOD[NAMES.index(name)]))
                original = getattr(Path, fault)

                def denied(p, *a, **kw):
                    if p == path:
                        raise PermissionError("review injected denial")
                    return original(p, *a, **kw)

                with patch.object(Path, fault, denied):
                    invoke(["denied", fault, name], inspect=False)
                path.unlink()
    elif family == "edges":
        # Every JSON type, each persisted saver state, current/other/int-limit seasons.
        seasons = [None, False, True, 0, 1, 2027, 2028, 2027.0, 1e300, "2027", "x" * 200000,
                   [], [2027], {}, {"year": 2027}, "9" * 4300]
        for state, season in itertools.product(["active", "used", "not_earned", "invalid"], seasons):
            (acc / NAMES[3]).write_text(json.dumps(dict(state=state, season=season)))
            r = invoke(["saver", state, type(season).__name__, str(season)[:25]])
            if label == "head":
                line = next(x for x in r["output"].splitlines() if x.startswith("saver:"))
                if state == "invalid" or type(season) is not int:
                    assert "invalid" in line, r
                elif season == 2027:
                    assert "saver: " + state + ";" in line, r
                else:
                    assert "another season" in line, r
        (acc / NAMES[3]).unlink()

        def deep(depth, shape="array", key="active_streak"):
            if shape == "array":
                value = "[" * depth + "0" + "]" * depth
            elif shape == "object":
                value = '{"k":' * depth + "0" + "}" * depth
            else:
                value = '[{"k":' * depth + "0" + "}]" * depth
            return '{"recorded_at":"deep",' + json.dumps(key) + ':' + value + '}'

        def boundary(render):
            lo, hi = 1, 50000
            while lo < hi:
                mid = (lo + hi + 1) // 2
                try:
                    row = json.loads(deep(mid))
                    if render:
                        "  " + "; ".join(f"{key}={row.get(key, '—')}" for key in KEYS)
                    lo = mid
                except (ValueError, RecursionError):
                    hi = mid - 1
            return lo

        bounds = {"decoder": boundary(False), "renderer": boundary(True)}
        depths = sorted({1, 250, 1000, 3000, 12000, 50000,
                         *(d for b in bounds.values() for d in range(max(1, b - 35), b + 8))})
        good = [json.dumps(dict(recorded_at=f"row-{i}", active_streak=i)) for i in range(25)]
        for shape, key, depth, pos in itertools.product(["array", "object", "mixed"],
                ["active_streak", "settled_by_year"], depths, [0, 12, 25]):
            rows = good.copy()
            rows.insert(pos, deep(depth, shape, key))
            (acc / NAMES[4]).write_text("\n".join(rows) + "\n")
            r = invoke(["depth", shape, key, depth, pos], inspect=False)
            if label == "head":
                header = next(x for x in r["output"].splitlines() if x.startswith("ledger:"))
                m = re.fullmatch(r"ledger: readable; rows: (\d+); bad lines: (\d+)", header)
                assert m and sum(map(int, m.groups())) == len(rows), r
                shown = re.findall(r"^  recorded_at=(.*?); active_streak=", r["output"], re.M)
                expected = [f"row-{i}" for i in range(25)]
                if int(m[2]) == 0:
                    expected.insert(pos, "deep")
                assert shown == expected[-10:], r
        for digits in [4300, 4301, 10000]:
            for pos in [0, 12, 25]:
                rows = good.copy()
                rows.insert(pos, '{"active_streak":' + '9' * digits + '}')
                (acc / NAMES[4]).write_text("\n".join(rows) + "\n")
                invoke(["huge-int", digits, pos])
        odd = [dict(recorded_at="odd", active_streak="z" * 1000000),
               dict(recorded_at="odd", settled_by_year={"1": [{"deep": [False, None, 1e300]}]}),
               dict(recorded_at="odd", active_streak=1e300, switched=False, unsettled=None),
               dict(recorded_at="odd", extra={"unrendered": [1, 2]}), {}]
        rows = [json.dumps(v) for v in odd]
        (acc / NAMES[4]).write_text("\n".join([*rows, "", "[]", "null", "true", "1", '"x"', '{"bad":', " "]) + "\n")
        r = invoke("odd-values-blank-nonobject-truncated")
        assert "ledger: readable; rows: 5; bad lines: 8" in r["output"], r
        # Missing fields render as em dash, and old rows outside last ten remain counted.
        (acc / NAMES[4]).write_text("\n".join(["{}"] * 15 + good))
        r = invoke("last-ten-file-order")
        assert "ledger: readable; rows: 40; bad lines: 0" in r["output"], r
        assert re.findall(r"^  recorded_at=(row-\d+);", r["output"], re.M) == [f"row-{i}" for i in range(15, 25)]
        (acc / NAMES[4]).write_text("\n".join(["{}"] * 3))
        assert "recorded_at=—; active_streak=—" in invoke("missing-keys")["output"]
        # Valid old/new schema and dated/legacy observations beside saver and ledger failures.
        manual = dict(schema_version="bts_contest_streak_manual_v3", effective_date="2027-07-15", active_streak=7,
                      best_streak=9, saver_state="not_earned", recorded_at="2027-07-15T16:00:00Z", source="manual")
        for sw, man, auto, saver_kind, ledger_kind in itertools.product(
            [doc(), doc(season=2026), doc(status="revoked", revoked_at="2027-07-16T16:00:00Z")],
            [GOOD[1], manual], [AUTO, AUTO | {"schema_version": "bts_contest_streak_auto_v1"},
                                   AUTO | {"schema_version": None}, AUTO | {"schema_version": [1, {}]}],
            ["good", "broken"], ["good", "broken"]):
            for name, value in zip(NAMES[:3], [sw, man, auto]):
                (acc / name).write_text(json.dumps(value))
            install(acc / NAMES[3], saver_kind, GOOD[3])
            install(acc / NAMES[4], ledger_kind, GOOD[4])
            invoke(["schemas", sw["status"], sw["season"], man.get("schema_version"), auto["schema_version"], saver_kind, ledger_kind])
        write_json(HERE / (label + "-boundaries.json"), bounds)
    elif family == "benchmark":
        path = HERE / "benchmark-ledger.jsonl"
        shutil.copyfile(path, acc / NAMES[4])
        started = time.perf_counter()
        r = invoke("5000x200", inspect=False)
        elapsed = time.perf_counter() - started
        assert "ledger: readable; rows: 5000; bad lines: 0" in r["output"], r
        r.update(seconds=elapsed, peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                 ledger_bytes=path.stat().st_size)
        print(label, family, {k: r[k] for k in ["seconds", "peak_rss_bytes", "ledger_bytes"]}, flush=True)
    write_json(HERE / (label + "-" + family + ".json"), records)
    print(label, family, "records", len(records), flush=True)


def main():
    if len(sys.argv) > 1:
        worker(sys.argv[1], Path(sys.argv[2]), sys.argv[3])
        return
    archive = subprocess.run(["git", "archive", "28ae3aa9", "src"], cwd=REPO, check=True, capture_output=True).stdout
    old = HERE / "old"
    old.mkdir()
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        tar.extractall(old, filter="data")
    # One million realistically shaped prediction objects; only ten summary lines display.
    preds = [dict(roundId=i, result="hit", playerId=100000 + i, team="NYM", name="Offline Player") for i in range(200)]
    row = dict(AUTO, predictions=preds, switched=True, settled_by_year={"2027": 200}, settled_unmapped=0, unsettled=0)
    with (HERE / "benchmark-ledger.jsonl").open("w") as f:
        for i in range(5000):
            f.write(json.dumps(row | {"recorded_at": f"row-{i}"}) + "\n")
    for family in ["matrix", "edges", "benchmark"]:
        for label, source in [("old", old / "src"), ("head", REPO / "src")]:
            subprocess.run([sys.executable, __file__, label, str(source), family], cwd=REPO, check=True,
                           env=os.environ | {"PYTHONDONTWRITEBYTECODE": "1", "TZ": "America/New_York"})
    summary = {}
    for family in ["matrix", "edges", "benchmark"]:
        old_rows = json.loads((HERE / ("old-" + family + ".json")).read_text())
        new_rows = json.loads((HERE / ("head-" + family + ".json")).read_text())
        assert len(old_rows) == len(new_rows)
        diffs = []
        for a, b in zip(old_rows, new_rows):
            assert a["key"] == b["key"], (a["key"], b["key"])
            if a["output"] != b["output"] or a["exit"] != b["exit"]:
                key = b["key"]
                category = "B2" if isinstance(key, list) and key[0] == "saver" and key[2] == "bool" else "B1" if isinstance(key, list) and key[0] == "depth" else "UNEXPECTED"
                if category == "B1":
                    assert a["output"].split("ledger:")[0] == b["output"].split("ledger:")[0]
                if category == "B2":
                    assert a["output"].replace("another season", "invalid") == b["output"]
                diffs.append(dict(key=key, category=category, old=a["output"][:2500], head=b["output"][:2500]))
        summary[family] = dict(cases_per_revision=len(new_rows), differences=len(diffs),
                               unexpected=sum(x["category"] == "UNEXPECTED" for x in diffs))
        write_json(HERE / ("differences-" + family + ".json"), diffs)
    write_json(HERE / "status-summary.json", summary)
    print(json.dumps(summary, indent=2))
    assert all(x["unexpected"] == 0 for x in summary.values())


if __name__ == "__main__":
    main()
