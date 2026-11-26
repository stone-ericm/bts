# Statcast Swing Campaign — Stage 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Land the swing-data foundation — bronze per-pitch parquets (mid-2023→present), leak-free `swing_daily` aggregate/rolling feature machinery, the paired-daily-NDCG@10 metric harness, and the pre-registered control builders — so Stage 1 screening can start on solid, audited ground.

**Architecture:** Per-pitch Savant data lands in wide bronze `swing_{season}.parquet` files (never merged into the PA frame — no stable pitch keys there). A `swing_daily` layer aggregates to (entity, date) keeping denominator rows, applies `shift(1).rolling()` exactly like `compute.py`'s date-level contract, and left-joins onto PA rows. A new `validate/slate_rank.py` implements the campaign's primary metric. Control builders (missingness placebo, permutation, leaky sentinel) live beside the features they police.

**Tech Stack:** pybaseball 2.2.7 (existing dep), pandas, stdlib. Spec: `docs/superpowers/specs/2026-06-12-statcast-swing-campaign-design.md`.

**File map:**
- Create: `src/bts/data/swing_pull.py` — date-chunked Savant pulls, bronze schema, manifest
- Create: `scripts/backfill_swing_data.py` — resumable backfill CLI
- Create: `src/bts/features/swing.py` — daily aggregates, rolling features, PA join, control builders
- Create: `src/bts/validate/slate_rank.py` — paired daily NDCG@10 + block bootstrap
- Create: `scripts/qa_swing_vs_leaderboard.py` — tolerance-protocol QA
- Tests: `tests/data/test_swing_pull.py`, `tests/test_swing_features.py`, `tests/test_slate_rank.py`

**Conventions that apply everywhere below:** all `uv` commands prefixed `UV_CACHE_DIR=/tmp/uv-cache`; ids are MLBAM (statcast `batter`/`pitcher` == PA frame `batter_id`/`pitcher_id`); dates normalized with `pd.to_datetime`.

**Scope note:** the spec's permutation control is eval-time (shuffle candidate
features within entity at model evaluation) and belongs to the Stage-1 screen
harness plan; Stage 0 builds the data-layer controls (missingness placebo,
leaky sentinel). Pseudocount shrinkage K is likewise tuned at screen; Stage 0
ships the min-sample reliability gating.

---

### Task 1: Bronze pull module (`swing_pull.py`)

**Files:**
- Create: `src/bts/data/swing_pull.py`
- Test: `tests/data/test_swing_pull.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/data/test_swing_pull.py`:

```python
"""Tests for the Savant swing-data bronze pull layer (Stage 0)."""
import json

import pandas as pd

from bts.data.swing_pull import (
    BRONZE_COLUMNS,
    normalize_bronze,
    write_bronze_season,
)


def _raw_statcast_frame():
    # A minimal frame shaped like pybaseball.statcast() output
    return pd.DataFrame({
        "game_date": ["2025-06-01", "2025-06-01"],
        "game_pk": [700001, 700001],
        "at_bat_number": [1, 1],
        "pitch_number": [1, 2],
        "batter": [665742, 665742],
        "pitcher": [594798, 594798],
        "events": [None, "strikeout"],
        "description": ["ball", "swinging_strike"],
        "type": ["B", "S"],
        "pitch_type": ["FF", "SL"],
        "game_type": ["R", "R"],
        "balls": [0, 1],
        "strikes": [0, 0],
        "stand": ["L", "L"],
        "p_throws": ["R", "R"],
        "zone": [13, 6],
        "plate_x": [0.9, 0.1],
        "plate_z": [1.1, 2.4],
        "sz_top": [3.4, 3.4],
        "sz_bot": [1.6, 1.6],
        "miss_distance": [None, 2.5],
        "swing_length": [None, 7.2],
        "attack_angle": [None, 11.0],
        "attack_direction": [None, -4.0],
        "swing_path_tilt": [None, 32.0],
        "intercept_ball_minus_batter_pos_x_inches": [None, 28.0],
        "intercept_ball_minus_batter_pos_y_inches": [None, 33.0],
        "unrelated_savant_column": ["x", "y"],
    })


def test_normalize_keeps_bronze_columns_and_drops_rest():
    out = normalize_bronze(_raw_statcast_frame())
    assert "unrelated_savant_column" not in out.columns
    assert set(out.columns) <= set(BRONZE_COLUMNS)
    # core ids always present
    for col in ["game_date", "game_pk", "batter", "pitcher", "description", "miss_distance"]:
        assert col in out.columns


def test_normalize_tolerates_missing_columns():
    raw = _raw_statcast_frame().drop(columns=["swing_path_tilt", "sz_top"])
    out = normalize_bronze(raw)
    # absent columns are created as NA so season files share one schema
    assert "swing_path_tilt" in out.columns
    assert out["swing_path_tilt"].isna().all()


def test_normalize_filters_to_regular_season():
    raw = _raw_statcast_frame()
    raw.loc[0, "game_type"] = "S"  # spring training
    out = normalize_bronze(raw)
    assert (out["game_type"] == "R").all()
    assert len(out) == 1


def test_write_bronze_season_writes_parquet_and_manifest(tmp_path):
    df = normalize_bronze(_raw_statcast_frame())
    path = write_bronze_season(df, 2025, tmp_path, raw_columns=list(_raw_statcast_frame().columns))

    assert path == tmp_path / "swing_2025.parquet"
    back = pd.read_parquet(path)
    assert len(back) == len(df)
    manifest = json.loads((tmp_path / "swing_2025.manifest.json").read_text())
    assert manifest["season"] == 2025
    assert manifest["n_rows"] == len(df)
    assert "unrelated_savant_column" in manifest["raw_columns"]
    assert "pulled_at" in manifest
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/data/test_swing_pull.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bts.data.swing_pull'`

- [ ] **Step 3: Implement**

Create `src/bts/data/swing_pull.py`:

```python
"""Bronze layer for Statcast per-pitch swing data (campaign Stage 0).

Pulls land WIDE (Codex round-2): ids, event context, count/handedness, zone
geometry, and ALL swing/bat-tracking columns — storage is cheap, re-pulls and
Savant schema drift are not. Per-pitch swing data is NEVER merged into the PA
frame (no stable pitch keys there); the swing_daily layer (bts.features.swing)
aggregates from these files.

pybaseball hygiene (its datasource has no request timeout): callers must set
a project-scoped cache (PYBASEBALL_CACHE), pull serially in date chunks with
bounded retries, and re-pull a rolling recent window for incremental updates
(stale current-season cache). See scripts/backfill_swing_data.py.

Spec: docs/superpowers/specs/2026-06-12-statcast-swing-campaign-design.md
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Wide bronze schema. Absent columns (older dates / Savant drift) are created
# as NA so every season file shares one schema; the manifest records what the
# raw pull actually contained.
BRONZE_COLUMNS = [
    # identity / join keys
    "game_date", "game_pk", "at_bat_number", "pitch_number", "sv_id",
    "batter", "pitcher",
    # event context
    "events", "description", "type", "pitch_type", "game_type",
    "balls", "strikes", "stand", "p_throws",
    # location / zone geometry
    "zone", "plate_x", "plate_z", "sz_top", "sz_bot",
    # swing / bat-tracking payload
    "miss_distance", "swing_length", "attack_angle", "attack_direction",
    "swing_path_tilt",
    "intercept_ball_minus_batter_pos_x_inches",
    "intercept_ball_minus_batter_pos_y_inches",
]


def normalize_bronze(raw: pd.DataFrame) -> pd.DataFrame:
    """Project a raw pybaseball.statcast() frame onto the bronze schema.

    Regular season only; missing bronze columns created as NA; extra raw
    columns dropped (but recorded by the caller in the manifest).
    """
    df = raw.copy()
    if "game_type" in df.columns:
        df = df[df["game_type"] == "R"]
    for col in BRONZE_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[BRONZE_COLUMNS].reset_index(drop=True)


def write_bronze_season(
    df: pd.DataFrame,
    season: int,
    out_dir: Path,
    raw_columns: list[str],
) -> Path:
    """Write swing_{season}.parquet + a manifest recording the raw pull."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"swing_{season}.parquet"
    df.to_parquet(path, index=False)
    manifest = {
        "season": season,
        "n_rows": int(len(df)),
        "pulled_at": datetime.now(timezone.utc).isoformat(),
        "bronze_columns": BRONZE_COLUMNS,
        "raw_columns": raw_columns,
        "date_min": str(df["game_date"].min()) if len(df) else None,
        "date_max": str(df["game_date"].max()) if len(df) else None,
    }
    (out_dir / f"swing_{season}.manifest.json").write_text(json.dumps(manifest, indent=2))
    return path
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/data/test_swing_pull.py -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/swing_pull.py tests/data/test_swing_pull.py
git commit -m "swing campaign S0: bronze pull layer (wide schema + manifest)"
```

---

### Task 2: Resumable backfill CLI

**Files:**
- Create: `scripts/backfill_swing_data.py`
- Test: `tests/data/test_swing_pull.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/data/test_swing_pull.py`:

```python
from bts.data.swing_pull import month_chunks


def test_month_chunks_cover_range_without_overlap():
    chunks = month_chunks("2023-07-14", "2023-09-02")
    assert chunks[0] == ("2023-07-14", "2023-07-31")
    assert chunks[1] == ("2023-08-01", "2023-08-31")
    assert chunks[-1] == ("2023-09-01", "2023-09-02")
    # contiguous, no overlap
    for (a_start, a_end), (b_start, b_end) in zip(chunks, chunks[1:]):
        assert pd.Timestamp(b_start) == pd.Timestamp(a_end) + pd.Timedelta(days=1)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/data/test_swing_pull.py::test_month_chunks_cover_range_without_overlap -q`
Expected: FAIL — `ImportError: cannot import name 'month_chunks'`

- [ ] **Step 3: Implement `month_chunks` in `swing_pull.py` and the CLI**

Append to `src/bts/data/swing_pull.py`:

```python
def month_chunks(start: str, end: str) -> list[tuple[str, str]]:
    """Split [start, end] into calendar-month-bounded chunks (inclusive)."""
    chunks = []
    cur = pd.Timestamp(start)
    final = pd.Timestamp(end)
    while cur <= final:
        month_end = (cur + pd.offsets.MonthEnd(0))
        chunk_end = min(month_end, final)
        chunks.append((str(cur.date()), str(chunk_end.date())))
        cur = chunk_end + pd.Timedelta(days=1)
    return chunks
```

Create `scripts/backfill_swing_data.py`:

```python
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
    BRONZE_COLUMNS, month_chunks, normalize_bronze, write_bronze_season,
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
```

- [ ] **Step 4: Run tests + syntax check**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/data/test_swing_pull.py -q && UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import ast; ast.parse(open('scripts/backfill_swing_data.py').read()); print('OK')"`
Expected: 5 passed; OK

- [ ] **Step 5: Commit**

```bash
git add src/bts/data/swing_pull.py scripts/backfill_swing_data.py tests/data/test_swing_pull.py
git commit -m "swing campaign S0: resumable month-chunked backfill CLI"
```

---

### Task 3: swing_daily aggregates (denominator-preserving)

**Files:**
- Create: `src/bts/features/swing.py`
- Test: `tests/test_swing_features.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_swing_features.py`:

```python
"""Tests for swing_daily aggregates + rolling features (campaign Stage 0)."""
import numpy as np
import pandas as pd

from bts.features.swing import (
    daily_swing_aggregates,
    rolling_swing_features,
    attach_swing_features,
    build_missingness_placebo,
    build_leaky_sentinel,
)

SWING_DESCRIPTIONS = ["swinging_strike", "foul", "hit_into_play"]


def _bronze(rows):
    base = {
        "game_date": "2025-06-01", "game_pk": 700001, "batter": 1, "pitcher": 9,
        "description": "swinging_strike", "miss_distance": 3.0,
        "swing_length": 7.0, "attack_angle": 10.0, "plate_z": 2.0,
        "sz_top": 3.4, "sz_bot": 1.6,
    }
    return pd.DataFrame([{**base, **r} for r in rows])


def test_daily_aggregates_keep_denominator_rows():
    # batter swings on a day but never whiffs -> row exists, whiff fields 0/NaN
    bronze = _bronze([
        {"description": "foul", "miss_distance": None},
        {"description": "hit_into_play", "miss_distance": None},
    ])
    daily = daily_swing_aggregates(bronze, entity="batter")
    assert len(daily) == 1
    row = daily.iloc[0]
    assert row["n_swings"] == 2
    assert row["n_whiffs"] == 0
    assert row["n_whiffs_tracked"] == 0
    assert pd.isna(row["miss_sum"]) or row["miss_sum"] == 0


def test_daily_aggregates_distinguish_untracked_whiffs():
    bronze = _bronze([
        {"description": "swinging_strike", "miss_distance": 2.0},
        {"description": "swinging_strike", "miss_distance": None},  # whiff, no tracking
    ])
    daily = daily_swing_aggregates(bronze, entity="batter")
    row = daily.iloc[0]
    assert row["n_whiffs"] == 2
    assert row["n_whiffs_tracked"] == 1
    assert row["miss_sum"] == 2.0


def test_daily_aggregates_vertical_attack_on_whiffs():
    # plate_z above zone midline -> "over" attack
    bronze = _bronze([
        {"miss_distance": 2.0, "plate_z": 3.2},   # high
        {"miss_distance": 1.0, "plate_z": 1.7},   # low
    ])
    daily = daily_swing_aggregates(bronze, entity="pitcher")
    row = daily.iloc[0]
    assert row["n_whiff_high"] == 1
    assert row["n_whiff_low"] == 1


def test_rolling_features_are_shift1_leak_free():
    daily = pd.DataFrame({
        "batter": [1, 1, 1],
        "date": pd.to_datetime(["2025-06-01", "2025-06-02", "2025-06-03"]),
        "n_swings": [10, 10, 10],
        "n_whiffs": [2, 4, 6],
        "n_whiffs_tracked": [2, 4, 6],
        "miss_sum": [4.0, 12.0, 24.0],
        "miss_sumsq": [10.0, 40.0, 100.0],
        "swing_len_sum": [70.0, 70.0, 70.0],
        "n_swings_tracked": [10, 10, 10],
        "attack_angle_sum": [100.0, 100.0, 100.0],
        "n_whiff_high": [1, 2, 3],
        "n_whiff_low": [1, 2, 3],
    })
    feats = rolling_swing_features(daily, entity="batter", windows=[2])
    # day 1: no prior data -> NaN
    assert pd.isna(feats.iloc[0]["batter_miss_dist_2g"])
    # day 2: only day 1 in window: 4.0/2 = 2.0
    assert feats.iloc[1]["batter_miss_dist_2g"] == 2.0
    # day 3: days 1+2: (4+12)/(2+4) = 16/6
    assert abs(feats.iloc[2]["batter_miss_dist_2g"] - 16 / 6) < 1e-9


def test_attach_joins_on_entity_and_date():
    pa = pd.DataFrame({
        "batter_id": [1], "pitcher_id": [9],
        "date": pd.to_datetime(["2025-06-03"]),
    })
    feats = pd.DataFrame({
        "batter": [1], "date": pd.to_datetime(["2025-06-03"]),
        "batter_miss_dist_2g": [2.5],
    })
    out = attach_swing_features(pa, batter_feats=feats, pitcher_feats=None)
    assert out.iloc[0]["batter_miss_dist_2g"] == 2.5


def test_missingness_placebo_is_boolean_flags_only():
    pa = pd.DataFrame({
        "batter_id": [1], "pitcher_id": [9],
        "date": pd.to_datetime(["2025-06-03"]),
        "batter_miss_dist_30g": [2.5],
        "pitcher_miss_dist_30g": [np.nan],
    })
    plc = build_missingness_placebo(pa, ["batter_miss_dist_30g", "pitcher_miss_dist_30g"])
    assert list(plc.columns) == ["has_batter_miss_dist_30g", "has_pitcher_miss_dist_30g"]
    assert plc.dtypes.map(lambda t: t == bool).all()
    assert plc.iloc[0]["has_batter_miss_dist_30g"] == True  # noqa: E712
    assert plc.iloc[0]["has_pitcher_miss_dist_30g"] == False  # noqa: E712


def test_leaky_sentinel_uses_same_day_data():
    daily = pd.DataFrame({
        "batter": [1, 1],
        "date": pd.to_datetime(["2025-06-01", "2025-06-02"]),
        "n_whiffs_tracked": [1, 2],
        "miss_sum": [2.0, 9.0],
    })
    pa = pd.DataFrame({
        "batter_id": [1], "pitcher_id": [9],
        "date": pd.to_datetime(["2025-06-02"]),
    })
    out = build_leaky_sentinel(pa, daily, entity="batter")
    # SAME-DAY mean miss = 9.0/2 — deliberately leaky, harness must flag it
    assert out.iloc[0]["LEAKY_same_day_miss"] == 4.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_swing_features.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bts.features.swing'`

- [ ] **Step 3: Implement**

Create `src/bts/features/swing.py`:

```python
"""swing_daily aggregates + leak-free rolling features (campaign Stage 0).

Per-pitch bronze (data/processed/swing_{season}.parquet) is aggregated to
(entity, date) KEEPING denominator rows — "no whiffs", "no swings", and "no
tracking" stay distinguishable — then rolled with the same date-level
shift(1) contract as compute.py. Features are left-joined onto PA rows;
the bronze frame is never merged into the PA frame.

Also home of the pre-registered control builders (missingness placebo,
leaky sentinel) so the controls evolve in lockstep with the features.

Spec: docs/superpowers/specs/2026-06-12-statcast-swing-campaign-design.md
"""

from __future__ import annotations

import numpy as np
import pandas as pd

WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
SWING_DESCRIPTIONS = WHIFF_DESCRIPTIONS | {
    "foul", "foul_tip", "hit_into_play", "foul_bunt", "bunt_foul_tip",
}


def daily_swing_aggregates(bronze: pd.DataFrame, entity: str) -> pd.DataFrame:
    """Aggregate bronze per-pitch rows to (entity, date) with denominators.

    entity: "batter" or "pitcher" (bronze column name).
    Output columns: n_swings, n_swings_tracked, n_whiffs, n_whiffs_tracked,
    miss_sum, miss_sumsq, swing_len_sum, attack_angle_sum,
    n_whiff_high, n_whiff_low.
    """
    df = bronze.copy()
    df["date"] = pd.to_datetime(df["game_date"])
    df["_is_swing"] = df["description"].isin(SWING_DESCRIPTIONS)
    df["_is_whiff"] = df["description"].isin(WHIFF_DESCRIPTIONS)
    df["_miss"] = pd.to_numeric(df["miss_distance"], errors="coerce")
    df["_swing_len"] = pd.to_numeric(df.get("swing_length"), errors="coerce")
    df["_attack"] = pd.to_numeric(df.get("attack_angle"), errors="coerce")
    sz_mid = (pd.to_numeric(df["sz_top"], errors="coerce")
              + pd.to_numeric(df["sz_bot"], errors="coerce")) / 2
    plate_z = pd.to_numeric(df["plate_z"], errors="coerce")
    df["_whiff_high"] = df["_is_whiff"] & df["_miss"].notna() & (plate_z > sz_mid)
    df["_whiff_low"] = df["_is_whiff"] & df["_miss"].notna() & (plate_z <= sz_mid)

    swings = df[df["_is_swing"]]
    agg = swings.groupby([entity, "date"]).agg(
        n_swings=("_is_swing", "sum"),
        n_swings_tracked=("_swing_len", "count"),
        n_whiffs=("_is_whiff", "sum"),
        n_whiffs_tracked=("_miss", "count"),
        miss_sum=("_miss", "sum"),
        miss_sumsq=("_miss", lambda s: float(np.nansum(np.square(s)))),
        swing_len_sum=("_swing_len", "sum"),
        attack_angle_sum=("_attack", "sum"),
        n_whiff_high=("_whiff_high", "sum"),
        n_whiff_low=("_whiff_low", "sum"),
    ).reset_index()
    return agg.sort_values([entity, "date"], kind="mergesort").reset_index(drop=True)


def rolling_swing_features(
    daily: pd.DataFrame,
    entity: str,
    windows: list[int] | None = None,
    min_whiffs: int = 8,
) -> pd.DataFrame:
    """shift(1).rolling(w) ratio features from daily sums (leak-free by construction).

    Ratio-of-rolling-sums (not mean-of-daily-means) so sparse days don't get
    equal weight. Values are NaN until min_whiffs tracked whiffs accumulate
    in the window (denominator reliability; spec 'whiff-denominator' control).
    Column naming: {entity}_{stat}_{w}g.
    """
    windows = windows or [7, 15, 30, 60]
    out = daily[[entity, "date"]].copy()
    g = daily.groupby(entity, sort=False)

    def _roll_sum(col: str, w: int) -> pd.Series:
        return g[col].transform(lambda s: s.shift(1).rolling(w, min_periods=1).sum())

    for w in windows:
        whiffs_tracked = _roll_sum("n_whiffs_tracked", w)
        miss_sum = _roll_sum("miss_sum", w)
        miss_sumsq = _roll_sum("miss_sumsq", w)
        swings = _roll_sum("n_swings", w)
        whiffs = _roll_sum("n_whiffs", w)
        swings_tracked = _roll_sum("n_swings_tracked", w)
        swing_len = _roll_sum("swing_len_sum", w)
        attack = _roll_sum("attack_angle_sum", w)
        hi = _roll_sum("n_whiff_high", w)
        lo = _roll_sum("n_whiff_low", w)

        enough = whiffs_tracked >= min_whiffs
        mean_miss = (miss_sum / whiffs_tracked).where(enough)
        var_miss = (miss_sumsq / whiffs_tracked - mean_miss**2).where(enough)

        out[f"{entity}_miss_dist_{w}g"] = mean_miss
        out[f"{entity}_miss_std_{w}g"] = np.sqrt(var_miss.clip(lower=0))
        out[f"{entity}_whiff_rate_{w}g"] = (whiffs / swings).where(swings >= min_whiffs)
        out[f"{entity}_whiff_high_share_{w}g"] = (hi / (hi + lo)).where((hi + lo) >= min_whiffs)
        out[f"{entity}_swing_len_{w}g"] = (swing_len / swings_tracked).where(swings_tracked >= min_whiffs)
        out[f"{entity}_attack_angle_{w}g"] = (attack / swings_tracked).where(swings_tracked >= min_whiffs)
    return out


def attach_swing_features(
    pa: pd.DataFrame,
    batter_feats: pd.DataFrame | None,
    pitcher_feats: pd.DataFrame | None,
) -> pd.DataFrame:
    """Left-join rolling swing features onto PA rows by (entity id, date)."""
    out = pa.copy()
    if batter_feats is not None:
        out = out.merge(
            batter_feats.rename(columns={"batter": "batter_id"}),
            on=["batter_id", "date"], how="left",
        )
    if pitcher_feats is not None:
        out = out.merge(
            pitcher_feats.rename(columns={"pitcher": "pitcher_id"}),
            on=["pitcher_id", "date"], how="left",
        )
    return out


def build_missingness_placebo(pa: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Boolean availability flags ONLY (no values, no counts — counts carry
    real playing-time signal). The placebo model must show ~nothing, else the
    eval is confounded by the post-2023 era marker."""
    out = pd.DataFrame(index=pa.index)
    for col in feature_cols:
        out[f"has_{col}"] = pa[col].notna()
    return out


def build_leaky_sentinel(pa: pd.DataFrame, daily: pd.DataFrame, entity: str) -> pd.DataFrame:
    """SAME-DAY (unshifted) mean miss distance — deliberately leaky.

    The known-strong sentinel the harness MUST flag as inflated; proves
    leakage detectability. Never a candidate feature.
    """
    d = daily.copy()
    d["LEAKY_same_day_miss"] = d["miss_sum"] / d["n_whiffs_tracked"]
    key = "batter_id" if entity == "batter" else "pitcher_id"
    out = pa.merge(
        d[[entity, "date", "LEAKY_same_day_miss"]].rename(columns={entity: key}),
        on=[key, "date"], how="left",
    )
    return out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_swing_features.py -q`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/features/swing.py tests/test_swing_features.py
git commit -m "swing campaign S0: swing_daily aggregates + shift(1) rolling features + controls"
```

---

### Task 4: NDCG metric harness (`slate_rank.py`)

**Files:**
- Create: `src/bts/validate/slate_rank.py`
- Test: `tests/test_slate_rank.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_slate_rank.py`:

```python
"""Tests for the paired daily NDCG@10 slate-ranking metric (campaign primary)."""
import numpy as np
import pandas as pd

from bts.validate.slate_rank import daily_ndcg, paired_daily_delta


def _slate(date, scores, hits):
    return pd.DataFrame({
        "date": pd.to_datetime([date] * len(scores)),
        "score": scores,
        "actual_hit": hits,
    })


def test_perfect_ranking_is_1():
    s = _slate("2025-06-01", [0.9, 0.8, 0.7, 0.6], [1, 1, 0, 0])
    assert daily_ndcg(s, "score", k=10) == 1.0


def test_worst_ranking_below_1():
    s = _slate("2025-06-01", [0.9, 0.8, 0.7, 0.6], [0, 0, 1, 1])
    assert daily_ndcg(s, "score", k=10) < 1.0


def test_k_truncation_ignores_tail():
    # 12 candidates; hits beyond rank 10 don't affect DCG but do affect ideal
    scores = list(np.linspace(0.9, 0.4, 12))
    hits = [1] + [0] * 10 + [1]
    s = _slate("2025-06-01", scores, hits)
    v = daily_ndcg(s, "score", k=10)
    assert 0 < v < 1.0


def test_paired_delta_zero_for_identical_scores():
    days = []
    rng = np.random.default_rng(7)
    for i in range(20):
        scores = rng.random(15)
        hits = (rng.random(15) < 0.65).astype(int)
        d = _slate(f"2025-06-{i+1:02d}", scores, hits)
        d["score_b"] = d["score"]
        d["season"] = 2025
        days.append(d)
    slate = pd.concat(days)
    res = paired_daily_delta(slate, "score", "score_b", k=10, n_boot=200, seed=1)
    assert res["delta"] == 0.0
    assert res["ci_low"] == 0.0 and res["ci_high"] == 0.0
    assert res["n_days"] == 20


def test_paired_delta_detects_better_ranker():
    days = []
    rng = np.random.default_rng(7)
    for i in range(60):
        n = 20
        hits = (rng.random(n) < 0.65).astype(int)
        good = hits * 1.0 + rng.normal(0, 0.3, n)   # correlated with outcome
        bad = rng.random(n)                          # noise
        d = pd.DataFrame({
            "date": pd.to_datetime([f"2025-06-01"]) + pd.Timedelta(days=i),
            "score": good, "score_b": bad, "actual_hit": hits, "season": 2025,
        })
        days.append(d)
    slate = pd.concat(days)
    res = paired_daily_delta(slate, "score", "score_b", k=10, n_boot=500, seed=1)
    assert res["delta"] > 0
    assert res["ci_low"] > 0  # clearly separated


def test_bootstrap_stratifies_by_season():
    days = []
    rng = np.random.default_rng(7)
    for i in range(10):
        for season in (2025, 2026):
            n = 12
            hits = (rng.random(n) < 0.65).astype(int)
            d = pd.DataFrame({
                "date": pd.to_datetime(f"{season}-06-01") + pd.Timedelta(days=i),
                "score": rng.random(n), "score_b": rng.random(n),
                "actual_hit": hits, "season": season,
            })
            days.append(d)
    slate = pd.concat(days)
    res = paired_daily_delta(slate, "score", "score_b", k=10, n_boot=100, seed=1)
    assert res["n_days"] == 20
    assert set(res["per_season_delta"]) == {2025, 2026}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_slate_rank.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'bts.validate.slate_rank'`

- [ ] **Step 3: Implement**

Create `src/bts/validate/slate_rank.py`:

```python
"""Paired per-day top-weighted slate ranking metric — the swing campaign's
PRIMARY confirmation metric (spec 2026-06-12, Codex-resolved hierarchy).

daily_ndcg: NDCG@k with the standard log2 discount over one day's ranked
slate, binary game-level got-a-hit labels. paired_daily_delta: candidate-vs-
baseline as paired per-day deltas with a season-stratified day-level block
bootstrap (days resampled within season; PA-level independence never assumed).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def daily_ndcg(day: pd.DataFrame, score_col: str, k: int = 10,
               label_col: str = "actual_hit") -> float:
    """NDCG@k for a single day's slate (binary labels, log2 discount)."""
    d = day.dropna(subset=[score_col])
    if d.empty or d[label_col].sum() == 0:
        return np.nan
    order = d.sort_values(score_col, ascending=False, kind="mergesort")
    labels = order[label_col].to_numpy(dtype=float)[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(labels) + 2))
    dcg = float((labels * discounts).sum())
    ideal = np.sort(d[label_col].to_numpy(dtype=float))[::-1][:k]
    idcg = float((ideal * discounts[: len(ideal)]).sum())
    return dcg / idcg if idcg > 0 else np.nan


def paired_daily_delta(
    slate: pd.DataFrame,
    score_a: str,
    score_b: str,
    k: int = 10,
    n_boot: int = 10_000,
    seed: int = 20260612,
    label_col: str = "actual_hit",
) -> dict:
    """Paired per-day NDCG@k delta (A − B) with season-stratified bootstrap.

    slate needs columns: date, season, {score_a}, {score_b}, {label_col}.
    Returns {delta, ci_low, ci_high, n_days, per_season_delta}.
    """
    per_day = []
    for (season, _date), day in slate.groupby(["season", "date"]):
        a = daily_ndcg(day, score_a, k=k, label_col=label_col)
        b = daily_ndcg(day, score_b, k=k, label_col=label_col)
        if not (np.isnan(a) or np.isnan(b)):
            per_day.append((season, a - b))
    if not per_day:
        return {"delta": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                "n_days": 0, "per_season_delta": {}}
    df = pd.DataFrame(per_day, columns=["season", "d"])
    rng = np.random.default_rng(seed)
    by_season = {s: g["d"].to_numpy() for s, g in df.groupby("season")}
    boots = np.empty(n_boot)
    for i in range(n_boot):
        parts = [g[rng.integers(0, len(g), len(g))] for g in by_season.values()]
        boots[i] = float(np.concatenate(parts).mean())
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {
        "delta": float(df["d"].mean()),
        "ci_low": float(lo),
        "ci_high": float(hi),
        "n_days": int(len(df)),
        "per_season_delta": {int(s): float(g.mean()) for s, g in df.groupby("season")["d"]},
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest tests/test_slate_rank.py -q`
Expected: 6 passed

- [ ] **Step 5: Commit**

```bash
git add src/bts/validate/slate_rank.py tests/test_slate_rank.py
git commit -m "swing campaign S0: paired daily NDCG@10 metric harness"
```

---

### Task 5: Leaderboard QA script

**Files:**
- Create: `scripts/qa_swing_vs_leaderboard.py`

- [ ] **Step 1: Write the QA script** (no unit tests — it IS a test, run against real data in Task 6)

Create `scripts/qa_swing_vs_leaderboard.py`:

```python
#!/usr/bin/env python3
"""QA: bronze per-pitch aggregates vs the Savant leaderboard (tolerance protocol).

Never exact equality (Savant applies its own tracking/competitive filters and
denominators). Pass criteria (spec 2026-06-12): after matching season +
player-type + qualifying minimum, Spearman >= 0.98 on n_swings and whiff
rate; median absolute percent error <= 2% (p95 <= 5%); mean miss-distance
median |diff| <= 0.5 inches.

Usage: uv run python scripts/qa_swing_vs_leaderboard.py --season 2025 \
           --bronze data/processed --player-type pitcher
"""
from __future__ import annotations

import argparse
import sys
import urllib.request
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from bts.features.swing import daily_swing_aggregates  # noqa: E402

LB_URL = ("https://baseballsavant.mlb.com/leaderboard/bat-tracking/"
          "swing-timing-miss-distance?season%5B%5D={season}&type={ptype}&csv=true")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--bronze", type=Path, default=Path("data/processed"))
    ap.add_argument("--player-type", choices=["batter", "pitcher"], default="pitcher")
    args = ap.parse_args()

    bronze = pd.read_parquet(args.bronze / f"swing_{args.season}.parquet")
    daily = daily_swing_aggregates(bronze, entity=args.player_type)
    ours = daily.groupby(args.player_type).agg(
        n_swings=("n_swings", "sum"),
        n_whiffs=("n_whiffs", "sum"),
        n_whiffs_tracked=("n_whiffs_tracked", "sum"),
        miss_sum=("miss_sum", "sum"),
    )
    ours["whiff_rate"] = ours["n_whiffs"] / ours["n_swings"]
    ours["mean_miss"] = ours["miss_sum"] / ours["n_whiffs_tracked"]

    url = LB_URL.format(season=args.season, ptype=args.player_type)
    with urllib.request.urlopen(url, timeout=30) as r:
        lb = pd.read_csv(StringIO(r.read().decode("utf-8-sig")))
    lb = lb.set_index("id")

    joined = ours.join(lb[["n_swings", "whiff_rate", "miss_distance"]],
                       how="inner", rsuffix="_lb")
    # qualifying minimum: leaderboard's displayed population only
    print(f"joined {len(joined)} {args.player_type}s (leaderboard population)")

    checks = []
    sp_swings = joined["n_swings"].corr(joined["n_swings_lb"], method="spearman")
    checks.append(("spearman n_swings >= 0.98", sp_swings, sp_swings >= 0.98))
    sp_wr = joined["whiff_rate"].corr(joined["whiff_rate_lb"], method="spearman")
    checks.append(("spearman whiff_rate >= 0.98", sp_wr, sp_wr >= 0.98))
    ape = (joined["n_swings"] - joined["n_swings_lb"]).abs() / joined["n_swings_lb"]
    checks.append(("median APE n_swings <= 2%", ape.median(), ape.median() <= 0.02))
    checks.append(("p95 APE n_swings <= 5%", ape.quantile(0.95), ape.quantile(0.95) <= 0.05))
    md = (joined["mean_miss"] - joined["miss_distance"]).abs()
    checks.append(("median |mean_miss diff| <= 0.5in", md.median(), md.median() <= 0.5))

    failed = 0
    for name, value, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}  (got {value:.4f})")
        failed += 0 if ok else 1
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Syntax check**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run python -c "import ast; ast.parse(open('scripts/qa_swing_vs_leaderboard.py').read()); print('OK')"`
Expected: OK

- [ ] **Step 3: Commit**

```bash
git add scripts/qa_swing_vs_leaderboard.py
git commit -m "swing campaign S0: leaderboard QA with tolerance protocol"
```

---

### Task 6: Run the backfill + QA (real data)

**Files:** none new (operational task)

- [ ] **Step 1: Run the full not-slow suite first**

Run: `UV_CACHE_DIR=/tmp/uv-cache uv run pytest -q -m "not slow"`
Expected: all pass (1900±; READ THE OUTPUT before proceeding — do not chain pushes)

- [ ] **Step 2: Deploy, then start the backfill on bts-hetzner** (data lives with prod; fleet relay pulls from there)

The box's working tree tracks the `deploy` branch — never `git checkout` individual
files from main onto it (dirties the tree and breaks the next deploy's
`--ff-only` pull). Ship via the normal deploy; a scheduler restart now is
harmless (it sleeps until the ~17:40 ET lineup check) and the canary covers it:

```bash
git push origin main
git push origin main:deploy
gh run watch --repo stone-ericm/bts $(gh run list --repo stone-ericm/bts --limit 1 --json databaseId -q '.[0].databaseId') || gh run list --repo stone-ericm/bts --limit 2
ssh bts-hetzner 'cd ~/projects/bts && git log --oneline -1'   # confirm new SHA before starting
ssh bts-hetzner 'cd ~/projects/bts && \
  PYBASEBALL_CACHE=data/raw/pybaseball_cache UV_CACHE_DIR=/tmp/uv-cache \
  nohup .venv/bin/python scripts/backfill_swing_data.py \
    --start 2023-07-14 --out data/processed --scratch data/raw/swing_chunks \
    > ~/logs/swing_backfill.log 2>&1 & echo started'
```

- [ ] **Step 3: Monitor until complete** (~35 month-chunks, politeness-throttled; expect hours)

```bash
ssh bts-hetzner 'tail -5 ~/logs/swing_backfill.log; ls ~/projects/bts/data/processed/swing_*.parquet 2>/dev/null'
```

Expected eventually: `swing_2023.parquet swing_2024.parquet swing_2025.parquet swing_2026.parquet` + manifests.

- [ ] **Step 4: Run the QA against 2025 (both player types)**

```bash
ssh bts-hetzner 'cd ~/projects/bts && UV_CACHE_DIR=/tmp/uv-cache .venv/bin/python \
  scripts/qa_swing_vs_leaderboard.py --season 2025 --player-type pitcher && \
  .venv/bin/python scripts/qa_swing_vs_leaderboard.py --season 2025 --player-type batter'
```

Expected: all PASS lines, exit 0. If FAIL: stop, diagnose (likely whiff-description set or game_type filtering), fix, re-run. Do not proceed to Stage 1 on failed QA.

- [ ] **Step 5: Spot-check feature build on real data**

```bash
ssh bts-hetzner 'cd ~/projects/bts && UV_CACHE_DIR=/tmp/uv-cache .venv/bin/python -c "
import pandas as pd
from bts.features.swing import daily_swing_aggregates, rolling_swing_features
b = pd.read_parquet(\"data/processed/swing_2025.parquet\")
d = daily_swing_aggregates(b, entity=\"pitcher\")
f = rolling_swing_features(d, entity=\"pitcher\", windows=[30])
print(f[\"pitcher_miss_dist_30g\"].describe())
print(\"coverage:\", f[\"pitcher_miss_dist_30g\"].notna().mean())
"'
```

Expected: mean ~2.5–3.5 inches, plausible spread, coverage >50% of pitcher-dates.

- [ ] **Step 6: Record Stage 0 completion in the spec + commit**

Update the spec's Status line to append: `Stage 0 COMPLETE <date> (bronze through <date_max>, QA passed, harness tested).` Then:

```bash
git add docs/superpowers/specs/2026-06-12-statcast-swing-campaign-design.md
git commit -m "swing campaign: Stage 0 complete (bronze + QA + harness)"
git push origin main
```

---

### Task 7: Memory + wrap

- [ ] **Step 1:** Update `~/projects/claude-shared/memory/bts_index.md` session-pickup with Stage 0 state (bronze paths, QA result, what Stage 1 needs: screen harness plan).
- [ ] **Step 2:** Log to conversation log. Stage 1 (screen harness + variant sweep) gets its own plan once the data is inspected.
