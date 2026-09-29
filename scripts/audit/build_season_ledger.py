"""Season 2026 ledger, Phase 1 — `acquire` (box: reads the frozen snapshot, fetches MLB schedules)
and `compile` (anywhere, offline, from the sealed bundle).

  .venv/bin/python scripts/audit/build_season_ledger.py acquire \
      --snapshot data/hetzner_results/season_2026_snapshot/final-20260928 \
      --out data/hetzner_results/season_2026_ledger_evidence/v1
  .venv/bin/python scripts/audit/build_season_ledger.py compile \
      --bundle data/hetzner_results/season_2026_ledger_evidence/v1 \
      --out data/validation/season_2026_ledger/<sha>-<run-id> --code-sha <sha>
"""
from __future__ import annotations

import argparse
import hashlib
import sys
import time
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.audit.season_ledger.acquire import SCHEDULE_URL, acquire  # noqa: E402
from scripts.audit.season_ledger.compile import SEASON_DATES, compile_bundle  # noqa: E402
from scripts.audit.season_ledger.ids import UTC_FORMAT  # noqa: E402

USER_AGENT = "bts-season-ledger/1 (one-pass audit acquisition)"


def _fetch(day: str) -> bytes:
    req = urllib.request.Request(SCHEDULE_URL.format(date=day), headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read()
    time.sleep(0.5)   # courteous pacing to a public API
    return data


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Season 2026 ledger, Phase 1")
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("acquire")
    a.add_argument("--snapshot", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    c = sub.add_parser("compile")
    c.add_argument("--bundle", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    c.add_argument("--code-sha", default=None)
    c.add_argument("--uv-lock", type=Path, default=Path("uv.lock"))
    args = ap.parse_args(argv)
    if args.cmd == "acquire":
        path = acquire(snapshot_root=args.snapshot, out_root=args.out, dates=SEASON_DATES, fetch=_fetch,
                       now_utc=lambda: datetime.now(timezone.utc).strftime(UTC_FORMAT))
        print(f"sealed {path}")
        return 0
    lock_sha = hashlib.sha256(args.uv_lock.read_bytes()).hexdigest() if args.uv_lock.is_file() else None
    build = compile_bundle(args.bundle, args.out, uv_lock_sha256=lock_sha, code_sha=args.code_sha)
    print(f"row kinds {build['row_kinds']} | matches {build['match']} | recipes {build['recipe_labels']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
