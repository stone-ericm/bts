"""Offline producer contract probe. Uses the checkout's refresh and a draft projection."""

import gzip
import inspect
import json
import os
import sys
import tempfile
import time
from datetime import date, datetime, time as clock_time, timedelta
from pathlib import Path
from urllib.parse import parse_qs, urlsplit


def deny_network(event, args):
    if event in {
        "socket.connect", "socket.getaddrinfo", "socket.gethostbyname",
        "socket.gethostbyaddr", "socket.getnameinfo", "socket.sendto", "socket.sendmsg",
    }:
        raise AssertionError(f"network forbidden: {event} {args!r}")


sys.addaudithook(deny_network)

import pandas as pd

from bts.features import park_drag, park_drag_producer as producer
from bts.health import park_drag_freshness as health


BASE = Path(tempfile.mkdtemp(prefix="producer-evidence-", dir=Path(__file__).parent))
STORE = Path("producer/pitches_current.csv.gz")
META = Path("producer/games_meta.csv")
STATIC = Path("producer/game_level_static.csv")
EXPORT = Path("park_drag_export.csv")
MANIFEST = Path("park_drag_manifest.json")
STATUS = Path("producer_status.json")
OPENER = date(2027, 3, 24)
S1 = OPENER + timedelta(days=1)
JULY = date(2027, 7, 15)
INSTANT = datetime.combine(S1, clock_time(7, 45))
SLEEPS = []
RESULTS = {}


class FrozenClock(datetime):
    @classmethod
    def now(cls, tz=None):
        return INSTANT.astimezone(tz) if tz else INSTANT


health.datetime = FrozenClock
producer._now_iso = lambda: INSTANT.isoformat()
time.time = lambda: INSTANT.timestamp()
time.sleep = SLEEPS.append


def draft_refresh():
    """Project the numbered rule at the empty branch, with real downstream calls.

    Step 1 gets the literal manifest predicate and lstat-style path presence.
    Step 4 directly assigns the fetch, preserving main's populated path.
    Exceptions and staged writes still follow the checkout's shared path.
    """
    source = inspect.getsource(producer.refresh)
    split = source.index('        store["game_date"] =')
    joined = source.index('        fetched = pd.DataFrame()')
    populated = source[split:joined]
    empty = '''        seeded_empty = store.empty
        if seeded_empty:
            manifest_path = root / "park_drag_manifest.json"
            export_path = root / "park_drag_export.csv"
            if os.path.lexists(export_path) or os.path.lexists(manifest_path):
                try:
                    prior_manifest = json.loads(manifest_path.read_text())
                    if not isinstance(prior_manifest, dict):
                        raise ValueError("manifest must be an object")
                    prior_date = date.fromisoformat(prior_manifest["max_source_game_date"])
                except (KeyError, TypeError, OSError, ValueError) as e:
                    raise ProducerError("cannot establish an old manifest") from e
                if prior_date.year >= today.year:
                    raise ProducerError("manifest has current or future source coverage")
            season = today.year
            yesterday = today - timedelta(days=1)
            window_start = date(season, 3, 1)
            if window_start > yesterday:
                raise ProducerError("nothing to fetch yet")
        else:
'''
    source = source[:split] + empty + "".join(
        "    " + line if line.strip() else line for line in populated.splitlines(keepends=True)
    ) + source[joined:]
    source = source.replace('        if len(fetched):\n', '''        if seeded_empty and fetched.empty:
            raise ProducerError("no pitches yet for the season; existing table kept")
        if len(fetched):
''', 1)
    old = '''            keep = store[(store["game_date"] < pd.Timestamp(window_start))]
            store = pd.concat([keep, fetched], ignore_index=True)  # window-replace
'''
    assert source.count(old) == 1
    source = source.replace(old, '''            if seeded_empty:
                store = fetched
            else:
                keep = store[(store["game_date"] < pd.Timestamp(window_start))]
                store = pd.concat([keep, fetched], ignore_index=True)  # window-replace
''')
    scope = {}
    exec(compile(source, "<draft projection>", "exec"), vars(producer), scope)
    return scope["refresh"]


DRAFT = draft_refresh()


class Reply:
    def __init__(self, text="", code=200, meta=None):
        self.text = text
        self.status_code = code
        self.meta = meta

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")

    def json(self):
        return self.meta


def pitches(day=OPENER, *, pk=900000, n=30):
    return pd.DataFrame([{
        "game_pk": pk, "game_date": str(day), "home_team": "NYY",
        "release_speed": 95.0, "vx0": 2.0, "vy0": -135.0, "vz0": -5.0,
        "ax": -2.0, "ay": 30.0, "az": -20.0,
    }] * n, columns=producer.REQUIRED_PITCH_COLS)


def feed(rows=None, *, code=200, text=None):
    csv = (pitches(n=0) if rows is None else rows).to_csv(index=False) if text is None else text
    return Reply(csv, code=code)


def prior_games(end=date(2026, 9, 27)):
    return pd.DataFrame([{
        "game_pk": i + 1, "season": end.year, "game_date": str(end - timedelta(days=11 - i)),
        "venue_id": 3313, "venue": "Yankee Stadium", "n_pitch": 90, "cd_trim": 0.32 - i / 1000,
    } for i in range(12)])


def prepare(name, *, store=None, static_empty=False, artifacts="old", gen=None):
    root = BASE / name
    (root / STORE.parent).mkdir(parents=True)
    (pitches(n=0) if store is None else store).to_csv(root / STORE, index=False, compression="gzip")
    pd.DataFrame(columns=["game_pk", "start_utc", "day_night", "temp_f", "condition", "wind",
                          "venue_id", "venue"]).to_csv(root / META, index=False)
    games = prior_games()
    (games.iloc[:0] if static_empty else games).to_csv(root / STATIC, index=False)
    if artifacts == "old":
        export, manifest = producer.build_export(games)
        manifest["generated_at"] = gen or "2026-09-28T07:45:00"
        export.to_csv(root / EXPORT, index=False)
        (root / MANIFEST).write_text(json.dumps(manifest))
    return root


def write_pair(root, source_day, *, manifest_day=None, gen=None):
    next_day = date.fromisoformat(source_day) + timedelta(days=1)
    (root / EXPORT).write_text(f"venue_id,date,park_drag_delta\n3313,{next_day},0.01\n")
    (root / MANIFEST).write_text(json.dumps({
        "max_source_game_date": manifest_day or source_day,
        "generated_at": gen or f"{next_day}T07:45:00",
    }))


def snapshot(root):
    contents = {}
    for path in root.rglob("*"):
        key = str(path.relative_to(root))
        if path.is_symlink():
            contents[key] = os.readlink(path)
        elif path.is_file():
            contents[key] = path.read_bytes()
    return contents


def alerts(root, today):
    try:
        return [[item.level, item.message] for item in health.check(root, today=today)]
    except Exception as exc:
        return {"raised": f"{type(exc).__name__}: {exc}"}


def run(name, root, today, reply, *, runner=DRAFT):
    global INSTANT
    INSTANT = datetime.combine(today, clock_time(7, 45))
    calls = []
    thrown = []
    SLEEPS.clear()

    def http_get(url, *, accept, timeout=180.0):
        parts = urlsplit(url)
        if parts.hostname == "baseballsavant.mlb.com":
            calls.append(parse_qs(parts.query))
            return reply
        if parts.hostname == "statsapi.mlb.com":
            calls.append(parts.path)
            return Reply(meta={"gameData": {
                "datetime": {"dateTime": f"{today}T01:00:00Z", "dayNight": "night"},
                "weather": {"temp": "75", "condition": "Clear", "wind": "Calm"},
                "venue": {"id": 3313, "name": "Yankee Stadium"},
            }})
        raise AssertionError(f"unexpected injected URL {url}")

    def trace(frame, event, arg):
        if frame.f_code is runner.__code__ and event == "exception":
            thrown.append([arg[0].__name__, str(arg[1])])
        return trace if frame.f_code is runner.__code__ else None

    before = snapshot(root)
    sys.settrace(trace)
    try:
        status = runner(root, today=today, http_get=http_get)
    finally:
        sys.settrace(None)
    assert status == json.loads((root / STATUS).read_text())
    after = snapshot(root)
    row = {
        "status": status, "thrown": thrown, "requests": calls, "sleep_seconds": list(SLEEPS),
        "changed": sorted(k for k in before.keys() | after.keys() if before.get(k) != after.get(k)),
        "health": alerts(root, today),
    }
    try:
        store = pd.read_csv(root / STORE)
    except (OSError, ValueError):
        pass
    else:
        row["store"] = {"rows": len(store), "columns": list(store.columns),
                        "dates": store.get("game_date", pd.Series()).unique().tolist()}
    RESULTS[name] = row
    return row


def main():
    root = prepare("main-empty")
    for day in (date(2027, 2, 28), date(2027, 3, 1), OPENER, S1, JULY):
        result = run(f"main-empty-{day}", root, day, feed(pitches()), runner=producer.refresh)
        assert result["thrown"] == [["ValueError", "cannot convert float NaN to integer"]]
        assert not result["requests"]

    for runner, suffix in ((producer.refresh, "main"), (DRAFT, "draft")):
        root = prepare(f"p1-{suffix}", static_empty=True, artifacts="none")
        write_pair(root, "2027-07-14")
        result = run(f"p1-{suffix}", root, JULY, feed(), runner=runner)
        assert not result["requests"] and not result["status"]["ok"]
        assert [a[0] for a in result["health"]] == ["WARN"]
        if suffix == "main":
            (root / STATUS).write_text(json.dumps({"ok": True, "current_season_games": 0}))
            RESULTS["p1-health-control-true"] = alerts(root, JULY)

    for defect in ("export-only", "manifest-only", "array", "bad-date", "missing-date",
                   "broken-json", "export-dir", "manifest-dir", "future", "dangling-export",
                   "mismatched-date", "covers-fetch-window"):
        root = prepare(f"p2-{defect}")
        if defect == "export-only":
            (root / MANIFEST).unlink()
        elif defect == "manifest-only":
            (root / EXPORT).unlink()
        elif defect in {"array", "bad-date", "missing-date", "broken-json"}:
            value = {"array": "[]", "bad-date": '{"max_source_game_date":"wrong"}',
                     "missing-date": "{}", "broken-json": "{"}[defect]
            (root / MANIFEST).write_text(value)
        elif defect in {"export-dir", "manifest-dir", "dangling-export"}:
            target = root / (MANIFEST if defect == "manifest-dir" else EXPORT)
            target.unlink()
            if defect == "dangling-export":
                target.symlink_to("missing.csv")
            else:
                target.mkdir()
        elif defect == "future":
            write_pair(root, "2028-07-14", gen="2027-07-15T07:45:00")
        elif defect == "mismatched-date":
            write_pair(root, "2027-07-14", manifest_day="2026-09-27")
        else:
            write_pair(root, "2027-07-14")
        run(f"p2-{defect}", root, JULY, feed(pitches(date(2027, 7, 14))))

    for day in (date(2027, 2, 28), date(2027, 3, 1), date(2027, 3, 2), OPENER,
                date(2027, 10, 1), date(2028, 1, 1), date(2028, 3, 2)):
        root = prepare(f"windows-{day}")
        run(f"windows-{day}", root, day, feed())
    root = prepare("p3-september", gen="2026-09-28T07:45:00")
    run("p3-after-march", root, date(2027, 9, 30), feed())
    for empty_static in (False, True):
        root = prepare(f"p4-{empty_static}", artifacts="none", static_empty=empty_static)
        run(f"p4-{empty_static}", root, OPENER, feed())

    for name, reply in {
        "403": feed(code=403), "429": feed(code=429), "html": feed(text="<html>error</html>"),
        "columns": feed(text="wrong\n1\n"), "guard": feed(pitches(n=23000)),
        "above-guard": feed(pitches(n=23001)), "bad-header-empty": feed(text="wrong\n"),
    }.items():
        root = prepare(f"p6-{name}")
        row = run(f"p6-{name}", root, S1, reply)
        if name in {"403", "429"}:
            assert len(row["requests"]) == 1 and row["status"]["rate_limited"] is True

    for empty_static in (False, True):
        root = prepare(f"p7-{empty_static}", static_empty=empty_static)
        run(f"p7-{empty_static}", root, S1, feed(pitches(n=10)))
    for name, rows in {
        "spring": pitches(date(2027, 3, 5)).assign(game_type="S"),
        "wrong-year": pitches(date(2026, 10, 1)), "before-march": pitches(date(2027, 2, 28)),
        "after-window": pitches(S1 + timedelta(days=1)),
        "short-valid-csv": pitches(n=26),
    }.items():
        root = prepare(f"feed-{name}")
        run(f"feed-{name}", root, S1, feed(rows))

    for day in (date(2027, 2, 28), date(2027, 3, 1), S1):
        root = prepare(f"p8-{day}", store=pitches(date(2026, 9, 27)))
        run(f"p8-{day}", root, day, feed())
    root = prepare("next-year", static_empty=False)
    run("next-year", root, date(2028, 3, 30), feed(pitches(date(2028, 3, 29))))
    for defect in ("missing-meta", "zero-byte", "gzip-no-header", "wrong-schema", "invalid-date",
                   "null-date", "corrupt-meta", "corrupt-static", "staged-write"):
        root = prepare(f"p9-{defect}")
        if defect == "missing-meta":
            (root / META).unlink()
        elif defect == "zero-byte":
            (root / STORE).write_bytes(b"")
        elif defect == "gzip-no-header":
            (root / STORE).write_bytes(gzip.compress(b""))
        elif defect == "wrong-schema":
            (root / STORE).write_bytes(gzip.compress(b"wrong,columns\n"))
        elif defect in {"invalid-date", "null-date"}:
            pitches().assign(game_date="bad" if defect == "invalid-date" else "").to_csv(
                root / STORE, index=False, compression="gzip")
        elif defect in {"corrupt-meta", "corrupt-static"}:
            (root / (META if defect == "corrupt-meta" else STATIC)).write_text("")
        else:
            (root / STORE.with_suffix(".gz.tmp")).write_bytes(gzip.compress(b"partial"))
        run(f"p9-{defect}", root, S1, feed(pitches()))

    for artifacts in ("old", "none"):
        root = prepare(f"opening-{artifacts}", artifacts=artifacts)
        opening = run(f"opening-{artifacts}", root, S1, feed(pitches()))
        assert opening["status"]["max_source_game_date"] == str(OPENER)
        assert opening["store"]["rows"] == 30
        table = park_drag.load_table(root / EXPORT)
        manifest = park_drag.load_manifest(root / EXPORT)
        RESULTS[f"serving-{artifacts}"] = park_drag.serving_value(table, manifest, 3313, S1)
        if artifacts == "old":
            opening_root = root
    seeded = prepare("manual-populated-control", store=pitches())
    run("manual-populated-control", seeded, S1, feed(pitches()), runner=producer.refresh)
    RESULTS["manual-control-artifact-bytes-match"] = {
        str(p): (opening_root / p).read_bytes() == (seeded / p).read_bytes()
        for p in (STORE, META, EXPORT, MANIFEST, STATUS)
    }
    day2 = S1 + timedelta(days=1)
    run("day2-complete", opening_root, day2, feed(pd.concat([pitches(), pitches(S1, pk=900001)])))
    other = BASE / "opening-none"
    run("day2-empty", other, day2, feed())
    run("day2-later-only", other, day2, feed(pitches(S1, pk=900001)))

    for name, today, reply in (("fetched", S1, feed(pitches())), ("empty", S1, feed()),
                               ("no-window", date(2027, 3, 1), feed())):
        compared = []
        prior = pitches()[list(reversed(producer.REQUIRED_PITCH_COLS))].assign(extra="retained")
        for runner, suffix in ((producer.refresh, "main"), (DRAFT, "draft")):
            root = prepare(f"p10-{name}-{suffix}", store=prior)
            run(f"p10-{name}-{suffix}", root, today, reply, runner=runner)
            compared.append(snapshot(root))
        RESULTS[f"p10-{name}-byte-match"] = compared[0] == compared[1]
        assert compared[0] == compared[1]

    evidence = BASE / "results.json"
    evidence.write_text(json.dumps(RESULTS, indent=2) + "\n")
    print(evidence)
    for name, result in RESULTS.items():
        if isinstance(result, dict) and "status" in result:
            status = result["status"]
            print(name, status.get("ok"), status.get("error", status.get("max_source_game_date")),
                  len(result["requests"]), result["changed"], result["health"])
        else:
            print(name, result)


if __name__ == "__main__":
    main()
