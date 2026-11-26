#!/usr/bin/env python3
"""Backfill Statcast per-pitch swing data (bronze) — campaign Stage 0.

Resumable: each month-chunk is written to a scratch parquet on success and
skipped on re-run; season files are assembled from the chunks at the end.
Serial pulls with bounded retries (pybaseball's datasource has no timeout of
its own — we bound at the chunk level and keep politeness sleeps).

Usage:
    PYBASEBALL_CACHE=data/raw/pybaseball_cache uv run python \
        scripts/backfill_swing_data.py --start 2023-07-14 --end 2026-06-11 \
        --out data/processed --scratch data/raw/swing_chunks
    # incremental daily mode (re-pulls a rolling window; for cron later):
    ... --incremental-days 10
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bts.data.swing_pull import (  # noqa: E402
    month_chunks, normalize_bronze, write_bronze_season,
)

MAX_RETRIES = 3
SLEEP_BETWEEN_CHUNKS_S = 5


def pull_chunk(start: str, end: str) -> tuple[pd.DataFrame, list[str]]:
    from pybaseball import statcast
    last_err = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = statcast(start_dt=start, end_dt=end, verbose=False)
            return normalize_bronze(raw), list(raw.columns)
        except Exception as e:
            last_err = e
            print(f"  chunk {start}..{end} attempt {attempt} failed: {e}", file=sys.stderr)
            time.sleep(10 * attempt)
    raise RuntimeError(f"chunk {start}..{end} failed after {MAX_RETRIES} attempts: {last_err}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2023-07-14")
    ap.add_argument("--end", default=str(date.today() - timedelta(days=1)))
    ap.add_argument("--out", type=Path, default=Path("data/processed"))
    ap.add_argument("--scratch", type=Path, default=Path("data/raw/swing_chunks"))
    ap.add_argument("--incremental-days", type=int, default=None,
                    help="Re-pull only the last N days (rolling window vs stale cache)")
    args = ap.parse_args()

    if not os.environ.get("PYBASEBALL_CACHE"):
        os.environ["PYBASEBALL_CACHE"] = "data/raw/pybaseball_cache"
    args.scratch.mkdir(parents=True, exist_ok=True)

    if args.incremental_days:
        args.start = str(date.today() - timedelta(days=args.incremental_days))

    raw_cols_seen: list[str] = []
    for start, end in month_chunks(args.start, args.end):
        chunk_path = args.scratch / f"chunk_{start}_{end}.parquet"
        if chunk_path.exists() and not args.incremental_days:
            print(f"skip existing {chunk_path.name}", flush=True)
            continue
        print(f"pulling {start}..{end}", flush=True)
        df, raw_cols = pull_chunk(start, end)
        raw_cols_seen = sorted(set(raw_cols_seen) | set(raw_cols))
        df.to_parquet(chunk_path, index=False)
        print(f"  {len(df)} rows -> {chunk_path.name}", flush=True)
        time.sleep(SLEEP_BETWEEN_CHUNKS_S)

    # assemble season files from all chunks present
    all_chunks = sorted(args.scratch.glob("chunk_*.parquet"))
    if not all_chunks:
        print("no chunks; nothing to assemble", file=sys.stderr)
        return
    full = pd.concat([pd.read_parquet(p) for p in all_chunks], ignore_index=True)
    full["game_date"] = pd.to_datetime(full["game_date"])
    full = full.drop_duplicates(
        subset=["game_pk", "at_bat_number", "pitch_number"], keep="last"
    )
    for season, grp in full.groupby(full["game_date"].dt.year):
        path = write_bronze_season(
            grp.reset_index(drop=True), int(season), args.out, raw_columns=raw_cols_seen
        )
        print(f"season {season}: {len(grp)} rows -> {path}", flush=True)


if __name__ == "__main__":
    main()
