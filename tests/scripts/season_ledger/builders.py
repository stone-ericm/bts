"""Synthetic source builders for the season-ledger tests. They write the shapes production writes
(bts.picks.save_pick, bts.daily_decision.write_decision, scheduler save_state, the CLI contest-ledger
append); no test reads real data. Gzip output is pinned (mtime=0) so fixtures are byte-stable."""
from __future__ import annotations

import gzip
import json
from pathlib import Path

from scripts.audit.season_ledger.bundle import BundleEntry, write_manifest
from scripts.audit.season_ledger.ids import sha256_hex


def dumps(obj) -> bytes:
    return json.dumps(obj).encode()


def gz(data: bytes) -> bytes:
    return gzip.compress(data, mtime=0)


def seal_bundle(root, files: dict[str, bytes], missing=(), mtimes: dict[str, str] | None = None) -> None:
    root = Path(root)
    entries = []
    for rel, data in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        entries.append(BundleEntry(rel_path=rel, status="present", sha256=sha256_hex(data), size=len(data),
                                   source_mtime_utc=(mtimes or {}).get(rel)))
    entries += [BundleEntry(rel_path=rel, status="missing", note="not found at acquisition") for rel in missing]
    write_manifest(root, entries, acquired_at_utc="2026-09-28T16:00:00.000000Z", builder_version="test")


_PICK = {"batter_name": "Ada Batter", "batter_id": 101, "team": "TB", "lineup_position": 1,
         "pitcher_name": "P", "pitcher_id": 900, "p_game_hit": 0.78, "flags": [],
         "projected_lineup": False, "game_pk": 5001, "game_time": "2026-05-01T23:05:00Z", "pitcher_team": "BOS"}
_DD = {**_PICK, "batter_name": "Dee Leg", "batter_id": 202, "team": "NYY", "game_pk": 5002, "lineup_position": 2}


def pick_json(date: str, *, primary: dict | None = None, dd: dict | None = None, **file_fields) -> bytes:
    """A pick file as save_pick writes it (asdict(DailyPick)) with only the given file-level fields;
    pass dd={} for the default double-down."""
    doc = {"date": date, "run_time": f"{date}T15:00:00Z", "pick": {**_PICK, **(primary or {})},
           "double_down": None if dd is None else {**_DD, **dd}, "runner_up": None}
    doc.update(file_fields)
    return dumps(doc)
