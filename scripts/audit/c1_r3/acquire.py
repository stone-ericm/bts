"""C1 rank 3: re-acquire the 2021–2025 regular-season game feeds (Eric 2026-10-04 evening, register row C1-r3-acquire).

Public, unauthenticated statsapi requests (`/api/v1.1/game/{pk}/feed/live`), paced politely with jitter between real
requests. Every attempt gets a durable intent receipt before the request and a completion receipt after it. **Any
403 or 429 stops at once:** no retry, no further request, a persistent `STOP_403_429.json` that refuses every later
run, and (through the C1 launcher) pauses all of C1 until Eric records a decision. Other errors retry three times
with backoff, then record a failure and move on. Already-stored feeds are verified and skipped, so a rerun resumes.

The acquisition reads no outcomes: it checks only that a body is JSON for the requested gamePk.

    python -m scripts.audit.c1_r3.acquire --seasons 2021 2022 2023 2024 2025
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import gzip
import hashlib
import json
import os
import random
import sys
import time
import urllib.error
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path

API = "https://statsapi.mlb.com"
STOP_NAME = "STOP_403_429.json"
DEFAULT_OUT = Path.home() / "projects" / "bts" / "data" / "hetzner_results" / "c1" / "r3"   # receipts, schedules, stop (restic-backed)
DEFAULT_FEEDS = Path.home() / "projects" / "bts" / "data" / "raw_c1"   # ~2 GB of feeds, re-acquirable: kept out of backups ($0 rule)
USER_AGENT = "bts-research/1.0 (C1 rank 3 historical feed re-acquisition)"
NOT_PLAYED = ("Postponed", "Cancelled")
RATE_LIMIT_CODES = (403, 429)
NO_RETRY_CODES = (400, 401, 404)


class RateLimited(Exception):
    def __init__(self, code: int):
        super().__init__(f"HTTP {code}")
        self.code = code


class Busy(Exception):
    """Another acquisition holds the writer lock."""


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def schedule_games(sched: dict) -> list[dict]:
    """Every regular-season gamePk with at least one listing that was played (not only postponed or cancelled),
    dated by its last played listing (a suspended game's completion date). `discover_games` in production keeps only
    status "Final", which drops games such as 2023-10-02's "Completed Early"."""
    by_pk: dict[int, dict] = {}
    for day in sched.get("dates", []):
        for g in day.get("games", []):
            pk, state = int(g["gamePk"]), g["status"]["detailedState"]
            date = g.get("officialDate") or day["date"]
            e = by_pk.setdefault(pk, {"gamePk": pk, "date": None, "statuses": set()})
            e["statuses"].add(state)
            if not state.startswith(NOT_PLAYED) and (e["date"] is None or date > e["date"]):
                e["date"] = date
    games = [{**e, "season": int(e["date"][:4]), "statuses": sorted(e["statuses"])}
             for e in by_pk.values() if e["date"] is not None]
    return sorted(games, key=lambda g: (g["date"], g["gamePk"]))


@contextlib.contextmanager
def writer_lock(out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    fh = open(out_dir / ".lock", "w")
    try:
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as e:
            raise Busy(str(out_dir)) from e
        yield
    finally:
        fh.close()


def _append(path: Path, rec: dict) -> None:
    """One durable JSON line (flushed and fsynced before returning)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())


def _write_durable(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)
        f.flush()
        os.fsync(f.fileno())


def _verified(path: Path, pk: int) -> bool:
    try:
        return _valid(gzip.decompress(path.read_bytes()), pk)
    except (OSError, EOFError, ValueError):
        return False


def _valid(body: bytes, pk: int) -> bool:
    try:
        d = json.loads(body)
    except ValueError:
        return False
    return isinstance(d, dict) and d.get("gamePk") == pk


def _get(url: str, fetch, sleep) -> bytes:
    """One feed with the C1 error policy: 403/429 raise RateLimited at once; 400/401/404 fail without retry; other
    HTTP and network errors retry up to three attempts with backoff."""
    for attempt in range(3):
        try:
            return fetch(url)
        except urllib.error.HTTPError as e:
            if e.code in RATE_LIMIT_CODES:
                raise RateLimited(e.code) from e
            if e.code in NO_RETRY_CODES or attempt == 2:
                raise
        except (urllib.error.URLError, TimeoutError, OSError):
            if attempt == 2:
                raise
        sleep(5.0 * (attempt + 1))
    raise AssertionError("unreachable")


def acquire(games: list[dict], *, out_dir: Path, feeds_dir: Path, fetch, sleep, jitter, now) -> int:
    """Download every game's feed. Returns 0 (all stored), 1 (finished with failures) or 3 (rate-limit stop)."""
    stop = out_dir / STOP_NAME
    with writer_lock(out_dir):
        if stop.exists():
            print(f"refusing: {stop} exists (403/429 stop; needs Eric's recorded decision)", file=sys.stderr)
            return 3
        failures, requested = 0, False
        for g in games:
            pk, season = g["gamePk"], g["season"]
            rel = Path(str(season)) / f"{pk}.json.gz"
            dest = feeds_dir / rel
            if dest.exists() and _verified(dest, pk):
                continue
            if requested:
                sleep(jitter())
            requested = True
            url = f"{API}/api/v1.1/game/{pk}/feed/live"
            receipts = out_dir / "receipts" / f"{now():%Y-%m-%d}.jsonl"
            attempt_id = uuid.uuid4().hex
            _append(receipts, {"kind": "intent", "attempt_id": attempt_id, "gamePk": pk, "url": url,
                               "started_utc": now().isoformat()})
            done = {"kind": "completion", "attempt_id": attempt_id, "gamePk": pk, "url": url}
            try:
                body = _get(url, fetch, sleep)
            except RateLimited as e:
                # The stop marker is written (durably) FIRST: a later receipt failure can never lose it.
                _write_durable(stop, json.dumps({"written_utc": now().isoformat(), "gamePk": pk, "url": url,
                                                 "http_status": e.code, "attempt_id": attempt_id}, indent=1) + "\n")
                try:
                    _append(receipts, {**done, "ended_utc": now().isoformat(), "outcome": "rate_limited",
                                       "http_status": e.code})
                except OSError as rec_err:
                    print(f"receipt write failed after the stop was recorded: {rec_err}", file=sys.stderr)
                print(f"STOP: HTTP {e.code} on {url}; wrote {stop}", file=sys.stderr)
                return 3
            except Exception as e:  # noqa: BLE001 - recorded, counted, and the run continues
                failures += 1
                _append(receipts, {**done, "ended_utc": now().isoformat(), "outcome": "failed",
                                   "error": f"{type(e).__name__}: {e}"[:300]})
                continue
            if not _valid(body, pk):
                failures += 1
                _append(receipts, {**done, "ended_utc": now().isoformat(), "outcome": "invalid",
                                   "bytes": len(body), "decoded_sha256": sha256(body)})
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            stored = gzip.compress(body, mtime=0)
            tmp = dest.with_suffix(".tmp")
            tmp.write_bytes(stored)
            os.replace(tmp, dest)
            _append(receipts, {**done, "ended_utc": now().isoformat(), "outcome": "stored", "bytes": len(body),
                               "decoded_sha256": sha256(body), "stored_path": str(rel),
                               "stored_sha256": sha256(stored), "statuses": g.get("statuses")})
        return 1 if failures else 0


def _fetch(url: str) -> bytes:
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": USER_AGENT}), timeout=30) as r:
        return r.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", type=int, nargs="+", required=True)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--feeds", type=Path, default=DEFAULT_FEEDS)
    ap.add_argument("--min-gap", type=float, default=1.0)
    ap.add_argument("--max-gap", type=float, default=2.5)
    args = ap.parse_args(argv)
    out = args.out.expanduser().resolve()
    now = lambda: datetime.now(timezone.utc)  # noqa: E731
    jitter = lambda: random.uniform(args.min_gap, args.max_gap)  # noqa: E731
    if (out / STOP_NAME).exists():
        print(f"refusing: {out / STOP_NAME} exists", file=sys.stderr)
        return 3
    games = []
    for season in args.seasons:
        sp = out / "schedules" / f"sched_{season}.json"
        if not sp.exists():
            url = (f"{API}/api/v1/schedule?sportId=1&season={season}&gameType=R"
                   "&fields=dates,date,games,gamePk,officialDate,status,detailedState")
            try:
                body = _get(url, _fetch, time.sleep)
            except RateLimited as e:
                out.mkdir(parents=True, exist_ok=True)
                (out / STOP_NAME).write_text(json.dumps({"written_utc": now().isoformat(), "url": url,
                                                         "http_status": e.code}, indent=1) + "\n")
                print(f"STOP: HTTP {e.code} on the schedule request", file=sys.stderr)
                return 3
            sp.parent.mkdir(parents=True, exist_ok=True)
            sp.write_bytes(body)
            time.sleep(jitter())
        season_games = schedule_games(json.loads(sp.read_bytes()))
        print(f"{season}: {len(season_games)} played games; schedule sha256 {sha256(sp.read_bytes())[:12]}",
              file=sys.stderr)
        games.extend(season_games)
    rc = acquire(games, out_dir=out, feeds_dir=args.feeds.expanduser().resolve(), fetch=_fetch, sleep=time.sleep,
                 jitter=jitter, now=now)
    print(f"done: rc={rc}, {len(games)} games listed", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
