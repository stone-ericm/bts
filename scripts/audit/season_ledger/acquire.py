"""Acquisition into a sealed evidence bundle (spec §3). Reads the frozen W0.7 snapshot; the only
network call is the MLB schedule fetch, whose failures become declared `missing` entries."""
from __future__ import annotations

import json
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

from . import BUILDER_VERSION
from .bundle import BundleEntry, write_manifest
from .ids import UTC_FORMAT, sha256_hex

SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule?sportId=1&date={date}&gameType=R&hydrate=team"
STATIC = Path("data/leaderboard/static_snapshots")
GRAB_STATIC = Path("data/leaderboard/final_grab_20260927/raw/static")
LOGS = ("cron.log", "journal_bts-scheduler_retained.txt")


def fetch_schedule(day: str, get: Callable[[str], bytes], *, attempts: int = 3,
                   sleep: Callable[[float], None] = time.sleep) -> bytes:
    """One MLB schedule response (final review #3). A failed request, or a body that is not a JSON object with a
    `dates` list, is retried with backoff (2 s, then 4 s); the last failure is raised, and `acquire` declares that
    date `missing` with the failure's class as its note. Only a schedule-shaped body is ever returned for sealing."""
    failure: Exception = ValueError("no fetch attempted")
    for attempt in range(attempts):
        if attempt:
            sleep(2 ** attempt)
        try:
            data = get(day)
            doc = json.loads(data)
        except Exception as exc:   # network and decode errors vary by stack; every one is retried
            failure = exc
            continue
        if isinstance(doc, dict) and isinstance(doc.get("dates"), list):
            return data
        failure = ValueError("schedule response without a dates list")
    raise failure


def _mtime_utc(path: Path) -> str:
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).strftime(UTC_FORMAT)


def _captures(directory: Path) -> list[Path]:
    """Files in a capture directory; dotfiles (.last_sha256 markers) are not inputs."""
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.is_file() and not p.name.startswith("."))


def acquire(*, snapshot_root, out_root, dates: list[str], fetch: Callable[[str], bytes],
            now_utc: Callable[[], str]) -> Path:
    snap, out_root = Path(snapshot_root), Path(out_root)
    if out_root.exists() and any(out_root.iterdir()):
        raise FileExistsError(f"bundle directory not empty: {out_root} (a new acquisition is a new version)")
    picks = snap / "data" / "picks"
    if not picks.is_dir():
        raise FileNotFoundError(f"no picks tree under {snap}")
    out_root.mkdir(parents=True, exist_ok=True)
    entries: list[BundleEntry] = []

    def copy(src: Path, rel: str) -> None:
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
        data = dest.read_bytes()
        entries.append(BundleEntry(rel_path=rel, status="present", sha256=sha256_hex(data), size=len(data),
                                   source_path=src.relative_to(snap).as_posix(), source_mtime_utc=_mtime_utc(src)))

    def missing(rel: str, note: str) -> None:
        entries.append(BundleEntry(rel_path=rel, status="missing", note=note))

    for src in sorted(p for p in picks.rglob("*") if p.is_file()):
        copy(src, "picks/" + src.relative_to(picks).as_posix())
    for feed in ("rounds", "units"):
        found = _captures(snap / STATIC / feed)
        for src in found:
            copy(src, f"static/{feed}/{src.name}")
        if not found:
            missing(f"static/{feed}/NO_CAPTURES", "no captures found")
    players = _captures(snap / STATIC / "players")
    for src in sorted({players[0], players[-1]}) if players else []:
        copy(src, f"static/players/{src.name}")
    if not players:
        missing("static/players/NO_CAPTURES", "no captures found")
    grab = _captures(snap / GRAB_STATIC)
    for src in grab:
        copy(src, f"static/grab_20260927/{src.name}")
    if not grab:
        missing("static/grab_20260927/NO_FILES", "grab static directory empty or absent")
    for name in LOGS:
        if (snap / name).is_file():
            copy(snap / name, f"logs/{name}")
        else:
            missing(f"logs/{name}", "not in snapshot")
    for day in dates:
        rel = f"schedules/{day}.json"
        try:
            data = fetch(day)
        except Exception as exc:   # recorded, never fatal: the bundle declares the gap
            missing(rel, f"fetch_failed:{exc.__class__.__name__}")
            continue
        dest = out_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        entries.append(BundleEntry(rel_path=rel, status="present", sha256=sha256_hex(data), size=len(data),
                                   source_path=SCHEDULE_URL.format(date=day), source_mtime_utc=now_utc()))
    return write_manifest(out_root, entries, acquired_at_utc=now_utc(), builder_version=BUILDER_VERSION,
                          source_root=str(snap))
