"""Capture BTS public static JSON files (pregame consensus + lookup archival).

MLB publishes the Beat the Streak app's data sheets as UNAUTHENTICATED static
JSON under mlb-play.mlbstatic.com/apps/beat-the-streak/game/json/. Two carry
forward-looking signal that exists nowhere else once the day passes:

  most_selected_players.json — per-round most-picked players (selection counts
      + MLB's own probabilityStarter model), populated for today AND tomorrow
  suggested_players.json     — MLB's own recommended picks, up to 2 rounds ahead

The rest are the lookup tables needed to interpret those rows later in the
season. units.json only carries current/upcoming games — without archival the
unitId -> game mapping for past rounds is unrecoverable (that gap is exactly
why historical PickRow rows have NaN opponent_team today).

Captures are content-deduped: a feed is stored only when its bytes' sha256
differs from the previous capture (marker file per feed), so a 30-min cron
cadence costs storage only when MLB actually changes a file.

Standalone-friendly ON PURPOSE: stdlib-only, no bts imports, runnable as
`python3 static_capture.py --out-dir ...` on the bare system interpreter
(bootstrap mode on the box before a deploy lands).
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

BASE_URL = "https://mlb-play.mlbstatic.com/apps/beat-the-streak/game/json"
# Browser-fidelity identity (see endpoints.browser_headers for the rationale).
# Duplicated here because this module is intentionally stdlib-only and
# standalone-runnable (bootstrap on the box before a deploy lands) — it must not
# import from bts.*. Keep the UA in sync with endpoints.BROWSER_UA.
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
STATIC_HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.mlb.com/apps/beat-the-streak/game",
    "Accept-Encoding": "gzip",
}
FETCH_TIMEOUT_S = 30.0

# feed name -> (url, required top-level key; None = any non-empty JSON object)
FEEDS: dict[str, tuple[str, str | None]] = {
    "most_selected_players": (f"{BASE_URL}/most_selected_players.json", "mostSelectedPlayers"),
    "suggested_players": (f"{BASE_URL}/suggested_players.json", "suggestedPlayers"),
    "rounds": (f"{BASE_URL}/rounds.json", "rounds"),
    "units": (f"{BASE_URL}/units.json", "units"),
    "players": (f"{BASE_URL}/players.json", "players"),
    "checksums": (f"{BASE_URL}/checksums.json", None),
}


def snapshot_filename(now: datetime) -> str:
    """UTC, second precision, no colons — lexicographic order == chronological."""
    return now.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + ".json"


def validate_payload(required_key: str | None, obj) -> bool:
    """Reject error pages / schema surprises before they reach storage."""
    if not isinstance(obj, dict) or not obj:
        return False
    if required_key is None:
        return True
    return required_key in obj and isinstance(obj[required_key], (list, dict))


def _maybe_gunzip(raw: bytes) -> bytes:
    """Fastly serves these files gzip'd even to clients that never sent
    Accept-Encoding, and urllib (unlike httpx) doesn't auto-decompress.
    Sniff the magic bytes rather than trusting the header."""
    if raw[:2] == b"\x1f\x8b":
        return gzip.decompress(raw)
    return raw


def _default_fetch(name: str, url: str) -> bytes:  # pragma: no cover - network
    req = urllib.request.Request(url, headers=STATIC_HEADERS)
    with urllib.request.urlopen(req, timeout=FETCH_TIMEOUT_S) as resp:
        return _maybe_gunzip(resp.read())


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    tmp.rename(path)


def _utc_iso(now: datetime) -> str:
    return now.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def capture_one(
    name: str,
    url: str,
    required_key: str | None,
    out_dir: Path,
    fetch: Callable[[str, str], bytes],
    now: datetime,
) -> dict:
    """Fetch one feed; store iff valid AND content changed. Never raises."""
    try:
        raw = fetch(name, url)
    except Exception as e:  # noqa: BLE001 - cron path: isolate feed failures
        return {"feed": name, "status": "fetch_error", "error": str(e), "path": None}
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, UnicodeDecodeError):
        obj = None
    if obj is None or not validate_payload(required_key, obj):
        return {"feed": name, "status": "invalid", "path": None}

    sha = hashlib.sha256(raw).hexdigest()
    marker = out_dir / name / ".last_sha256"
    prev = marker.read_text().strip() if marker.exists() else None
    if prev == sha:
        return {"feed": name, "status": "unchanged", "sha256": sha, "path": None}

    dest = out_dir / name / snapshot_filename(now)
    _atomic_write_bytes(dest, raw)
    _atomic_write_bytes(marker, (sha + "\n").encode())
    return {"feed": name, "status": "stored", "sha256": sha, "path": str(dest)}


def capture_all(
    out_dir: Path | str,
    fetch: Callable[[str, str], bytes] | None = None,
    now: datetime | None = None,
    feeds: dict[str, tuple[str, str | None]] = FEEDS,
) -> list[dict]:
    """Capture every feed; update capture_status.json; return per-feed results."""
    out_dir = Path(out_dir)
    fetch = fetch or _default_fetch
    now = now or datetime.now(timezone.utc)

    results = [capture_one(name, url, key, out_dir, fetch, now)
               for name, (url, key) in feeds.items()]

    status_path = out_dir / "capture_status.json"
    prior_feeds: dict = {}
    if status_path.exists():
        try:
            prior_feeds = json.loads(status_path.read_text()).get("feeds", {})
        except (json.JSONDecodeError, OSError):
            prior_feeds = {}
    feeds_status = {}
    for r in results:
        prev = prior_feeds.get(r["feed"], {})
        entry = {
            "status": r["status"],
            "sha256": r.get("sha256", prev.get("sha256")),
            "last_stored_utc": (_utc_iso(now) if r["status"] == "stored"
                                else prev.get("last_stored_utc")),
        }
        if r.get("error"):
            entry["last_error"] = r["error"]
        feeds_status[r["feed"]] = entry
    _atomic_write_bytes(status_path, json.dumps(
        {"last_run_utc": _utc_iso(now), "feeds": feeds_status}, indent=2).encode())
    return results


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", default="data/leaderboard/static_snapshots")
    args = ap.parse_args(argv)
    results = capture_all(Path(args.out_dir))
    for r in results:
        line = f"{r['feed']}: {r['status']}"
        if r.get("path"):
            line += f" -> {r['path']}"
        if r.get("error"):
            line += f" ({r['error']})"
        print(line)
    all_failed = all(r["status"] == "fetch_error" for r in results)
    return 1 if all_failed else 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
