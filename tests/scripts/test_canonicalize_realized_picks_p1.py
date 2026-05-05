"""Tests for α P1 — batter_skill_quartile addition (Codex bus #215/#216).

Schema additions tested here:
- batter_skill_prior_pa (Int64, nullable)
- batter_skill_prior_hit_rate (Float64, nullable)
- batter_skill_quartile (Int64, nullable: {1, 2, 3, 4} or pd.NA)

Contract:
- As-of-pick-date computation: PA rows with date < pick.date strictly. No
  same-day leakage. No future PAs.
- League pool: ALL PA-frame batters with prior_pa >= MIN_PRIOR_PA as-of
  pick.date (including the pick batter himself if eligible). Quartile bounds
  computed from the eligible pool's prior_hit_rate distribution.
- Below-threshold (prior_pa < MIN_PRIOR_PA): quartile = NA, prior_pa and
  prior_hit_rate still populated for audit.
- Deterministic ties: <=q25 → 1, <=q50 → 2, <=q75 → 3, else 4 (ties go to
  the lower quartile).

This file is additive — does NOT re-run the P0 test suite. Run both files
together at verification time:
  uv run pytest tests/scripts/test_canonicalize_realized_picks.py \\
                tests/scripts/test_canonicalize_realized_picks_p1.py -q
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import pytest


SCRIPTS_DIR = Path(__file__).parent.parent.parent / "scripts"


def _load_canonicalize():
    spec = importlib.util.spec_from_file_location(
        "canonicalize_realized_picks",
        str(SCRIPTS_DIR / "canonicalize_realized_picks.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["canonicalize_realized_picks"] = mod
    spec.loader.exec_module(mod)
    return mod


_mod = _load_canonicalize()
canonicalize = _mod.canonicalize


RT_POST_BPM = "2026-05-01T15:00:00+00:00"

# 5-batter league pool with rates {0.10, 0.20, 0.30, 0.40, 0.50}.
# pandas linear-interp quantiles on n=5 sorted: q25=0.20, q50=0.30, q75=0.40.
# Used for end-to-end tests where the pick batter's rate is well clear of any
# boundary so adding them to the pool doesn't perturb the assignment.
POOL_BATTERS = [
    (200, 50, 5),   # 0.10
    (201, 50, 10),  # 0.20
    (202, 50, 15),  # 0.30
    (203, 50, 20),  # 0.40
    (204, 50, 25),  # 0.50
]


def _pool_rows(prior_date: str = "2026-04-15") -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bid, npa, nhit in POOL_BATTERS:
        for i in range(npa):
            rows.append({
                "batter_id": bid,
                "date": prior_date,
                "is_hit": bool(i < nhit),
                "game_pk": 60000 + bid,
                "venue_id": 99,
                "roof_type": "Open",
                "weather_temp": 70.0,
            })
    return rows


def _pick_batter_rows(
    *,
    batter_id: int,
    prior_pas: int,
    prior_hits: int,
    pick_date: str = "2026-05-01",
    pick_game_pk: int = 10001,
    venue_id: int = 5,
    roof_type: str = "Open",
    weather_temp: float = 70.0,
    prior_date: str = "2026-04-15",
    same_day_pas: int = 1,
    same_day_hit_count: int = 1,
) -> list[dict[str, Any]]:
    """Pick batter PAs: N prior on prior_date + M same-day on pick.date.

    same_day PAs are needed for env attribution (game_pk lookup) AND
    actual_hit attribution; they MUST be excluded from prior_pa per the
    no-same-day-leakage contract.
    """
    rows: list[dict[str, Any]] = []
    for i in range(prior_pas):
        rows.append({
            "batter_id": batter_id,
            "date": prior_date,
            "is_hit": bool(i < prior_hits),
            "game_pk": 50000 + batter_id,
            "venue_id": 99,
            "roof_type": "Open",
            "weather_temp": 70.0,
        })
    for j in range(same_day_pas):
        rows.append({
            "batter_id": batter_id,
            "date": pick_date,
            "is_hit": bool(j < same_day_hit_count),
            "game_pk": pick_game_pk,
            "venue_id": venue_id,
            "roof_type": roof_type,
            "weather_temp": weather_temp,
        })
    return rows


def _write_pa(pa_path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_parquet(pa_path, index=False)


def _write_pick(picks_dir: Path, name: str, body: dict[str, Any]) -> None:
    (picks_dir / name).write_text(json.dumps(body))


def _basic_pick(*, date, run_time, batter_id, game_pk, p=0.75, name="Test Batter"):
    return {
        "date": date,
        "run_time": run_time,
        "result": None,
        "pick": {
            "batter_id": batter_id,
            "batter_name": name,
            "pitcher_id": 99000 + batter_id,
            "game_pk": game_pk,
            "p_game_hit": p,
            "projected_lineup": False,
        },
    }


# ---- assign_quartile helper unit tests ----


def test_assign_quartile_below_q25() -> None:
    assign_quartile = _mod.assign_quartile
    assert assign_quartile(0.05, 0.20, 0.30, 0.40) == 1


def test_assign_quartile_tie_at_q25_low() -> None:
    assign_quartile = _mod.assign_quartile
    assert assign_quartile(0.20, 0.20, 0.30, 0.40) == 1, (
        "Tie at q25 must assign to quartile 1 (deterministic lower-bias)"
    )


def test_assign_quartile_between_q25_q50() -> None:
    assign_quartile = _mod.assign_quartile
    assert assign_quartile(0.25, 0.20, 0.30, 0.40) == 2


def test_assign_quartile_tie_at_q50_low() -> None:
    assign_quartile = _mod.assign_quartile
    assert assign_quartile(0.30, 0.20, 0.30, 0.40) == 2


def test_assign_quartile_between_q50_q75() -> None:
    assign_quartile = _mod.assign_quartile
    assert assign_quartile(0.35, 0.20, 0.30, 0.40) == 3


def test_assign_quartile_tie_at_q75_low() -> None:
    assign_quartile = _mod.assign_quartile
    assert assign_quartile(0.40, 0.20, 0.30, 0.40) == 3


def test_assign_quartile_above_q75() -> None:
    assign_quartile = _mod.assign_quartile
    assert assign_quartile(0.50, 0.20, 0.30, 0.40) == 4


def test_assign_quartile_none_inputs_return_none() -> None:
    assign_quartile = _mod.assign_quartile
    assert assign_quartile(None, 0.2, 0.3, 0.4) is None
    assert assign_quartile(0.3, None, None, None) is None
    assert assign_quartile(0.3, 0.2, None, 0.4) is None


# ---- end-to-end behavior ----


def test_no_same_day_pa_leakage(tmp_path: Path) -> None:
    """A PA on pick.date itself must NOT count toward prior_pa."""
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    rows = _pool_rows()  # full pool
    # Pick batter: 1 prior PA + 2 same-day PAs. prior_pa MUST be 1, not 3.
    rows += _pick_batter_rows(
        batter_id=100,
        prior_pas=1,
        prior_hits=1,
        same_day_pas=2,
        same_day_hit_count=1,
    )
    _write_pa(pa_path, rows)
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert int(row["batter_skill_prior_pa"]) == 1, (
        "same-day PAs must be excluded from prior_pa"
    )


def test_below_threshold_quartile_na(tmp_path: Path) -> None:
    """prior_pa = 49 → quartile NA; prior_pa + rate populated for audit."""
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    rows = _pool_rows()
    # Pick batter has 49 prior PAs (24 hits → 0.4898)
    rows += _pick_batter_rows(
        batter_id=100,
        prior_pas=49,
        prior_hits=24,
    )
    _write_pa(pa_path, rows)
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert int(row["batter_skill_prior_pa"]) == 49
    assert abs(float(row["batter_skill_prior_hit_rate"]) - (24 / 49)) < 1e-9
    assert pd.isna(row["batter_skill_quartile"]), (
        "prior_pa < MIN_PRIOR_PA must set quartile NA, not coerce to 1"
    )


def test_no_prior_pas_yields_na(tmp_path: Path) -> None:
    """Pick batter with no prior 2026 PAs → prior_pa=0, rate NA, quartile NA.

    Mimics 2026-03-29/30 picks in production where game_pk was None and the
    batter had no prior PAs in the season.
    """
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    rows = _pool_rows()
    # Pick batter has ONLY a same-day PA — no prior history
    rows += _pick_batter_rows(
        batter_id=100,
        prior_pas=0,
        prior_hits=0,
        same_day_pas=1,
    )
    _write_pa(pa_path, rows)
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert int(row["batter_skill_prior_pa"]) == 0
    assert pd.isna(row["batter_skill_prior_hit_rate"])
    assert pd.isna(row["batter_skill_quartile"])


def test_quartile_q1_low_skill(tmp_path: Path) -> None:
    """Pick batter rate well below q25 → quartile 1."""
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    rows = _pool_rows()
    # Pick batter: 50 PA, 2 hits → rate 0.04, well below q25=0.20
    rows += _pick_batter_rows(
        batter_id=100,
        prior_pas=50,
        prior_hits=2,
    )
    _write_pa(pa_path, rows)
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert int(row["batter_skill_prior_pa"]) == 50
    assert abs(float(row["batter_skill_prior_hit_rate"]) - 0.04) < 1e-9
    assert int(row["batter_skill_quartile"]) == 1


def test_quartile_q4_high_skill(tmp_path: Path) -> None:
    """Pick batter rate well above q75 → quartile 4."""
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    rows = _pool_rows()
    # Pick batter: 50 PA, 48 hits → rate 0.96, well above q75=0.40
    rows += _pick_batter_rows(
        batter_id=100,
        prior_pas=50,
        prior_hits=48,
    )
    _write_pa(pa_path, rows)
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert int(row["batter_skill_prior_pa"]) == 50
    assert int(row["batter_skill_quartile"]) == 4


# ---- dtype + schema preservation ----


def test_skill_dtypes_are_nullable(tmp_path: Path) -> None:
    """All 3 new skill columns must use pandas nullable extension dtypes so
    NA propagates explicitly (not silently as float-NaN-or-coerced-int).
    """
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    rows = _pool_rows()
    rows += _pick_batter_rows(batter_id=100, prior_pas=50, prior_hits=15)
    _write_pa(pa_path, rows)
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    assert df["batter_skill_prior_pa"].dtype == pd.Int64Dtype()
    assert df["batter_skill_prior_hit_rate"].dtype == pd.Float64Dtype()
    assert df["batter_skill_quartile"].dtype == pd.Int64Dtype()


def test_dtypes_round_trip_through_parquet(tmp_path: Path) -> None:
    """Write the canonicalize output, read it back, assert dtypes preserved."""
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    out_path = tmp_path / "out.parquet"

    rows = _pool_rows()
    # One eligible (50 PA) + one below-threshold (49 PA) → forces NA in mix
    rows += _pick_batter_rows(batter_id=100, prior_pas=50, prior_hits=15,
                              pick_date="2026-05-01", pick_game_pk=10001)
    rows += _pick_batter_rows(batter_id=101, prior_pas=49, prior_hits=10,
                              pick_date="2026-05-02", pick_game_pk=10002)
    _write_pa(pa_path, rows)
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))
    _write_pick(picks_dir, "2026-05-02.json", _basic_pick(
        date="2026-05-02", run_time=RT_POST_BPM,
        batter_id=101, game_pk=10002,
    ))

    df = canonicalize(picks_dir, pa_path)
    df.to_parquet(out_path, index=False)

    df_read = pd.read_parquet(out_path)
    assert df_read["batter_skill_prior_pa"].dtype == pd.Int64Dtype()
    assert df_read["batter_skill_prior_hit_rate"].dtype == pd.Float64Dtype()
    assert df_read["batter_skill_quartile"].dtype == pd.Int64Dtype()
    # NA preserved through round-trip on the below-threshold row
    na_row = df_read[df_read["batter_id"] == 101].iloc[0]
    assert pd.isna(na_row["batter_skill_quartile"])
    assert int(na_row["batter_skill_prior_pa"]) == 49


def test_p0_env_columns_still_present(tmp_path: Path) -> None:
    """Focused schema preservation: P1 must not drop or rename P0 env columns.

    Per Codex #215/#216, P1 tests should not re-run the entire P0 suite, but
    one targeted regression on the P0 schema is appropriate.
    """
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()

    rows = _pool_rows()
    rows += _pick_batter_rows(batter_id=100, prior_pas=50, prior_hits=15,
                              venue_id=19, roof_type="Open", weather_temp=60.0)
    _write_pa(pa_path, rows)
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    # P0 env columns + dtypes
    p0_env_cols = (
        "pick_venue_id",
        "pick_roof_type",
        "pick_weather_temp",
        "pick_is_indoor",
        "is_park_driven",
    )
    for col in p0_env_cols:
        assert col in df.columns, f"P0 env column missing: {col}"
    # P0 booleans still nullable
    assert df["pick_is_indoor"].dtype == pd.BooleanDtype()
    assert df["is_park_driven"].dtype == pd.BooleanDtype()
    # And the P0 rule still fires for Coors
    row = df.iloc[0]
    assert int(row["pick_venue_id"]) == 19
    assert bool(row["is_park_driven"]) is True
