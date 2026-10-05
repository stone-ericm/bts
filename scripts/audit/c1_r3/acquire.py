"""C1 rank 3: re-acquire the 2021–2025 regular-season game feeds (Eric 2026-10-04 evening, register row C1-r3-acquire).

Public, unauthenticated statsapi requests, paced politely with jitter between real requests.
- **Receipts per attempt:** every HTTP attempt (schedule, feed and each retry) has a durable intent receipt before
  the request and a completion receipt after it, sharing an attempt id. A request id groups retries.
- **Any 403 or 429 stops at once:** no retry, no further request. A durable `STOP_403_429.json` (file and directory
  fsynced) goes to the shared C1 pause location *and* the output directory, before any receipt. If the stop cannot be
  written the process fails, and its intent receipt stays unresolved, which the C1 launcher treats as a stop.
- **Other errors** retry three times with backoff, then record a failure and move on.
- **Durable bytes:** stored bytes are fsynced (file and directory) before their completion receipt is published.
- **Resume trusts receipts, not files:** a stored feed is skipped only when a `stored` completion receipt binds its
  decoded sha256. A file without a matching receipt is quarantined, recorded and fetched again.
- **No outcomes are read:** a body is accepted only as JSON whose top-level `gamePk` is the requested positive int.

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
C1_ROOT = Path.home() / "projects" / "bts" / "data" / "hetzner_results" / "c1"   # the shared C1 pause location
DEFAULT_OUT = C1_ROOT / "r3"                                                      # receipts, schedules, stop
DEFAULT_FEEDS = Path.home() / "projects" / "bts" / "data" / "raw_c1"              # ~2 GB, re-acquirable, unbacked ($0)
USER_AGENT = "bts-research/1.0 (C1 rank 3 historical feed re-acquisition)"
NOT_PLAYED = ("Postponed", "Cancelled")
PLAYED = ("Final", "Completed Early", "Game Over", "Suspended")
RATE_LIMIT_CODES = (403, 429)
NO_RETRY_CODES = (400, 401, 404)


class RateLimited(Exception):
    def __init__(self, code: int):
        super().__init__(f"HTTP {code}")
        self.code = code


class RequestFailed(Exception):
    """A request that failed after its retries (or a non-retryable HTTP error). The only failure the download loop
    counts and skips; receipt or stop write errors are not RequestFailed and propagate (fail closed)."""


class Busy(Exception):
    """Another acquisition holds the writer lock."""


class UnsupportedStatus(ValueError):
    pass


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def played_status(state: str) -> bool:
    """True for a played listing, False for postponed/cancelled; any other status is refused."""
    if state.startswith(NOT_PLAYED):
        return False
    if state.startswith(PLAYED):
        return True
    raise UnsupportedStatus(f"unsupported schedule status {state!r}")


def schedule_games(sched: dict) -> list[dict]:
    """The unique feed inventory: every gamePk with at least one played listing, dated by its last played listing's
    schedule date (a suspended game's completion day). This is not the opportunity calendar, which is built from the
    listings themselves (`c1_r4b.data.calendar_from_schedule`). Production's `discover_games` keeps only status
    "Final", which drops games such as 2023-10-02's "Completed Early"."""
    by_pk: dict[int, dict] = {}
    for day in sched.get("dates", []):
        for g in day.get("games", []):
            pk, state = g["gamePk"], g["status"]["detailedState"]
            if type(pk) is not int or pk <= 0:
                raise ValueError(f"invalid gamePk {pk!r}")
            e = by_pk.setdefault(pk, {"gamePk": pk, "date": None, "statuses": set()})
            e["statuses"].add(state)
            if played_status(state) and (e["date"] is None or day["date"] > e["date"]):
                e["date"] = day["date"]
    games = [{**e, "statuses": sorted(e["statuses"])} for e in by_pk.values() if e["date"] is not None]
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


def _fsync_dir(path: Path) -> None:
    fd = os.open(path, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _write_durable(path: Path, data: bytes) -> None:
    """Atomic durable write: temp file fsynced, renamed, directory fsynced."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)
    _fsync_dir(path.parent)


def _append(path: Path, rec: dict) -> None:
    """One durable JSON line (flushed and fsynced; a new file's directory fsynced)."""
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    if new:
        _fsync_dir(path.parent)


def read_receipts(out_dir: Path) -> list[dict]:
    recs = []
    for f in sorted((out_dir / "receipts").glob("*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def unresolved_intents(recs: list[dict]) -> list[str]:
    done = {r["attempt_id"] for r in recs if r.get("kind") == "completion"}
    return [r["attempt_id"] for r in recs if r.get("kind") == "intent" and r["attempt_id"] not in done]


def _valid(body: bytes, pk: int) -> bool:
    try:
        d = json.loads(body)
    except ValueError:
        return False
    gp = d.get("gamePk") if isinstance(d, dict) else None
    return type(gp) is int and gp > 0 and gp == pk


class Acquirer:
    """One run: receipts, stop protocol and request policy shared by schedule and feed requests."""

    def __init__(self, *, out_dir: Path, pause_root: Path, fetch, sleep, now):
        self.out, self.pause_root, self.fetch, self.sleep, self.now = out_dir, pause_root, fetch, sleep, now

    def receipts_path(self) -> Path:
        return self.out / "receipts" / f"{self.now():%Y-%m-%d}.jsonl"

    def stops(self) -> list[Path]:
        return [p for p in (self.out / STOP_NAME, self.pause_root / STOP_NAME) if p.exists()]

    def _stop(self, code: int, url: str, attempt_id: str) -> None:
        body = (json.dumps({"written_utc": self.now().isoformat(), "url": url, "http_status": code,
                            "attempt_id": attempt_id}, indent=1) + "\n").encode()
        for p in (self.pause_root / STOP_NAME, self.out / STOP_NAME):   # the shared pause location first
            _write_durable(p, body)

    def get(self, url: str, *, kind: str, game_pk: int | None = None) -> bytes:
        """One request with per-attempt receipts. Raises RateLimited after recording the stop (the stop is written
        before the completion receipt; a stop-write failure propagates with the intent left unresolved)."""
        request_id = uuid.uuid4().hex
        for attempt in range(3):
            attempt_id = uuid.uuid4().hex
            base = {"request_id": request_id, "attempt_id": attempt_id, "attempt": attempt + 1, "kind_of": kind,
                    "url": url, "gamePk": game_pk}
            _append(self.receipts_path(), {**base, "kind": "intent", "started_utc": self.now().isoformat()})
            t0 = time.monotonic()
            try:
                body = self.fetch(url)
            except urllib.error.HTTPError as e:
                if e.code in RATE_LIMIT_CODES:
                    self._stop(e.code, url, attempt_id)
                    self._complete(base, t0, outcome="rate_limited", http_status=e.code)
                    raise RateLimited(e.code) from e
                self._complete(base, t0, outcome="http_error", http_status=e.code)
                if e.code in NO_RETRY_CODES or attempt == 2:
                    raise RequestFailed(f"HTTP {e.code} on {url}") from e
            except (urllib.error.URLError, TimeoutError, OSError) as e:
                self._complete(base, t0, outcome="network_error", error=f"{type(e).__name__}: {e}"[:200])
                if attempt == 2:
                    raise RequestFailed(f"{type(e).__name__} on {url}") from e
            else:
                self._complete(base, t0, outcome="response", http_status=200, bytes=len(body),
                               decoded_sha256=sha256(body))
                return body
            self.sleep(5.0 * (attempt + 1))
        raise AssertionError("unreachable")

    def _complete(self, base: dict, t0: float, **fields) -> None:
        _append(self.receipts_path(), {**base, "kind": "completion", "ended_utc": self.now().isoformat(),
                                       "duration_s": round(time.monotonic() - t0, 3), **fields})


def acquire(games: list[dict], *, out_dir: Path, feeds_dir: Path, pause_root: Path, fetch, sleep, jitter, now) -> int:
    """Download every game's feed. Returns 0 (all stored), 1 (finished with failures) or 3 (stop in force)."""
    acq = Acquirer(out_dir=out_dir, pause_root=pause_root, fetch=fetch, sleep=sleep, now=now)
    with writer_lock(out_dir):
        if acq.stops():
            print(f"refusing: 403/429 stop in force {acq.stops()} (needs Eric's recorded decision)", file=sys.stderr)
            return 3
        recs = read_receipts(out_dir)
        if unresolved_intents(recs):
            print("refusing: unresolved request receipts (reconcile before any further request)", file=sys.stderr)
            return 3
        bound: dict[int, set] = {}
        for r in recs:
            if r.get("kind") == "completion" and r.get("outcome") == "stored":
                bound.setdefault(r["gamePk"], set()).add(r["decoded_sha256"])
        failures, requested = 0, False
        for g in games:
            pk = g["gamePk"]
            rel = Path(str(g["season"])) / f"{pk}.json.gz"
            dest = feeds_dir / rel
            if dest.exists():
                try:
                    have = sha256(gzip.decompress(dest.read_bytes()))
                except (OSError, EOFError):
                    have = None
                if have is not None and have in bound.get(pk, set()):
                    continue
                q = out_dir / "quarantine" / f"{pk}-{now():%Y%m%dT%H%M%S%f}.json.gz"
                q.parent.mkdir(parents=True, exist_ok=True)
                os.replace(dest, q)
                _append(acq.receipts_path(), {"kind": "reconciled_orphan", "gamePk": pk, "moved_to": q.name,
                                              "decoded_sha256": have, "at_utc": now().isoformat()})
            if requested:
                sleep(jitter())
            requested = True
            url = f"{API}/api/v1.1/game/{pk}/feed/live"
            try:
                body = acq.get(url, kind="feed", game_pk=pk)
            except RateLimited:
                print(f"STOP: rate limited on {url}", file=sys.stderr)
                return 3
            except RequestFailed as e:      # recorded per attempt, counted, and the run continues
                failures += 1
                print(f"failed {pk}: {e}", file=sys.stderr)
                continue
            if not _valid(body, pk):
                failures += 1
                _append(acq.receipts_path(), {"kind": "validation", "gamePk": pk, "outcome": "invalid",
                                              "decoded_sha256": sha256(body), "at_utc": now().isoformat()})
                continue
            stored = gzip.compress(body, mtime=0)
            _write_durable(dest, stored)
            _append(acq.receipts_path(), {"kind": "completion", "attempt_id": f"store-{uuid.uuid4().hex}",
                                          "gamePk": pk, "outcome": "stored", "decoded_sha256": sha256(body),
                                          "stored_path": str(rel), "stored_sha256": sha256(stored),
                                          "statuses": g.get("statuses"), "at_utc": now().isoformat()})
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
    out, pause_root = args.out.expanduser().resolve(), C1_ROOT.resolve()
    if pause_root not in out.parents and out != pause_root:
        print(f"refusing: --out must lie under the C1 tree {pause_root} (the launcher's stop scan)", file=sys.stderr)
        return 2
    now = lambda: datetime.now(timezone.utc)  # noqa: E731
    jitter = lambda: random.uniform(args.min_gap, args.max_gap)  # noqa: E731
    acq = Acquirer(out_dir=out, pause_root=pause_root, fetch=_fetch, sleep=time.sleep, now=now)
    games = []
    with writer_lock(out):
        if acq.stops() or unresolved_intents(read_receipts(out)):
            print("refusing: a 403/429 stop or an unresolved request receipt is in force", file=sys.stderr)
            return 3
        for season in args.seasons:
            sp = out / "schedules" / f"sched_{season}.json"
            if not sp.exists():
                url = (f"{API}/api/v1/schedule?sportId=1&season={season}&gameType=R"
                       "&fields=dates,date,games,gamePk,officialDate,status,detailedState")
                try:
                    body = acq.get(url, kind="schedule")
                except RateLimited:
                    return 3
                except RequestFailed as e:
                    print(f"schedule {season}: {e}", file=sys.stderr)
                    return 1
                _write_durable(sp, body)
                time.sleep(jitter())
            season_games = [{**g, "season": season} for g in schedule_games(json.loads(sp.read_bytes()))]
            print(f"{season}: {len(season_games)} played games; schedule sha256 {sha256(sp.read_bytes())[:12]}",
                  file=sys.stderr)
            games.extend(season_games)
    rc = acquire(games, out_dir=out, feeds_dir=args.feeds.expanduser().resolve(), pause_root=pause_root, fetch=_fetch,
                 sleep=time.sleep, jitter=jitter, now=now)
    print(f"done: rc={rc}, {len(games)} games listed", file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
