"""Tests for scripts/canonicalize_realized_picks.py per-game env join (α P0).

Covers the schema additions per Codex bus #203:
- per-game environment table keyed by game_pk (NOT (batter_id, date)).
- 5 new columns: pick_venue_id, pick_roof_type, pick_weather_temp,
  pick_is_indoor, is_park_driven.
- Nullable pandas BooleanDtype for pick_is_indoor and is_park_driven; missing
  game env => NA, not False.
- Rule: is_park_driven = (pick_venue_id == COORS_VENUE_ID)
                        OR (pick_weather_temp >= 85.0 AND NOT pick_is_indoor).
- Doubleheader regression: same (batter_id, date) appearing at two different
  game_pks resolves env via pick.game_pk, not first-PA-for-batter-on-date.
- actual_hit attribution stays on (batter_id, date) per Codex #203 ("do not
  broaden scope").
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
    # The script defines a frozen dataclass; the dataclass decorator looks up
    # cls.__module__ in sys.modules during class creation, so the module must
    # be registered there before exec_module runs.
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


# Use a run_time after the post-bpm cutoff (2026-04-30T16:27Z) so regime is
# post_bpm; we are not exercising the regime plumbing in this file.
RT_POST_BPM = "2026-05-01T15:00:00+00:00"


def _write_pa(pa_path: Path, rows: list[dict[str, Any]]) -> None:
    pd.DataFrame(rows).to_parquet(pa_path, index=False)


def _write_pick(picks_dir: Path, name: str, body: dict[str, Any]) -> None:
    (picks_dir / name).write_text(json.dumps(body))


def _pa_row(
    *,
    batter_id: int,
    date: str,
    game_pk: int,
    venue_id: int,
    roof_type: str,
    weather_temp: float,
    is_hit: bool = True,
) -> dict[str, Any]:
    return {
        "batter_id": batter_id,
        "date": date,
        "is_hit": is_hit,
        "game_pk": game_pk,
        "venue_id": venue_id,
        "roof_type": roof_type,
        "weather_temp": weather_temp,
    }


def _basic_pick(
    *,
    date: str,
    run_time: str,
    batter_id: int,
    game_pk: int,
    p: float = 0.75,
    name: str = "Test Batter",
) -> dict[str, Any]:
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


# ---- schema ----


def test_schema_has_new_env_columns(tmp_path: Path) -> None:
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pa(pa_path, [_pa_row(
        batter_id=100, date="2026-05-01", game_pk=10001,
        venue_id=19, roof_type="Open", weather_temp=60.0,
    )])
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    assert not df.empty
    for col in (
        "pick_venue_id",
        "pick_roof_type",
        "pick_weather_temp",
        "pick_is_indoor",
        "is_park_driven",
    ):
        assert col in df.columns, f"missing column: {col}"
    assert df["pick_is_indoor"].dtype == pd.BooleanDtype()
    assert df["is_park_driven"].dtype == pd.BooleanDtype()


# ---- is_park_driven rule ----


def test_park_driven_coors_outdoor_cold(tmp_path: Path) -> None:
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pa(pa_path, [_pa_row(
        batter_id=100, date="2026-05-01", game_pk=10001,
        venue_id=19, roof_type="Open", weather_temp=60.0,
    )])
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert row["pick_venue_id"] == 19
    assert bool(row["pick_is_indoor"]) is False
    assert bool(row["is_park_driven"]) is True


def test_park_driven_hot_outdoor(tmp_path: Path) -> None:
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pa(pa_path, [_pa_row(
        batter_id=100, date="2026-05-01", game_pk=10001,
        venue_id=5, roof_type="Open", weather_temp=88.0,
    )])
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert row["pick_venue_id"] == 5
    assert bool(row["pick_is_indoor"]) is False
    assert bool(row["is_park_driven"]) is True


def test_park_driven_indoor_hot_excluded(tmp_path: Path) -> None:
    """Dome at 90F: weather branch excluded by indoor; venue is not Coors."""
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pa(pa_path, [_pa_row(
        batter_id=100, date="2026-05-01", game_pk=10001,
        venue_id=5, roof_type="Dome", weather_temp=90.0,
    )])
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert bool(row["pick_is_indoor"]) is True
    assert bool(row["is_park_driven"]) is False


def test_park_driven_outdoor_cold_neither(tmp_path: Path) -> None:
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pa(pa_path, [_pa_row(
        batter_id=100, date="2026-05-01", game_pk=10001,
        venue_id=5, roof_type="Open", weather_temp=70.0,
    )])
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert bool(row["is_park_driven"]) is False


def test_retractable_treated_as_indoor(tmp_path: Path) -> None:
    """Production convention from src/bts/features/compute.py:557-559:
    rt.lower() in {"dome","closed","retractable"} => indoor.
    """
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pa(pa_path, [_pa_row(
        batter_id=100, date="2026-05-01", game_pk=10001,
        venue_id=5, roof_type="Retractable", weather_temp=90.0,
    )])
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert bool(row["pick_is_indoor"]) is True
    assert bool(row["is_park_driven"]) is False


# ---- nullable behavior ----


def test_game_pk_miss_yields_na(tmp_path: Path) -> None:
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pa(pa_path, [_pa_row(
        batter_id=100, date="2026-05-01", game_pk=10001,
        venue_id=5, roof_type="Open", weather_temp=70.0,
    )])
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=99999,  # not in PA frame
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert pd.isna(row["pick_venue_id"])
    assert pd.isna(row["pick_weather_temp"])
    assert pd.isna(row["pick_is_indoor"])
    assert pd.isna(row["is_park_driven"])
    rt = row["pick_roof_type"]
    assert rt is None or pd.isna(rt) or rt == ""


def test_pick_game_pk_none_yields_na(tmp_path: Path) -> None:
    """pick.game_pk == None in source JSON (early-season picks before scheduler
    populated game_pk). Per Codex #205: should yield NA across all 5 derived
    columns, not silently coerce to a default.
    """
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pa(pa_path, [_pa_row(
        batter_id=100, date="2026-05-01", game_pk=10001,
        venue_id=5, roof_type="Open", weather_temp=70.0,
    )])
    pick_body = _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    )
    pick_body["pick"]["game_pk"] = None  # explicit None, not just missing
    _write_pick(picks_dir, "2026-05-01.json", pick_body)

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert pd.isna(row["pick_venue_id"])
    assert pd.isna(row["pick_weather_temp"])
    assert pd.isna(row["pick_is_indoor"])
    assert pd.isna(row["is_park_driven"])


def test_partial_env_missing_roof_yields_na(tmp_path: Path) -> None:
    """game_pk in PA frame but roof_type is missing. Hot non-Coors row: rule
    needs is_indoor to evaluate; with roof_type None we cannot determine
    indoor/outdoor, so is_park_driven is NA per Codex #205 partial-env contract.
    """
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pa(pa_path, [{
        "batter_id": 100,
        "date": "2026-05-01",
        "is_hit": True,
        "game_pk": 10001,
        "venue_id": 5,         # not Coors
        "roof_type": None,     # MISSING
        "weather_temp": 90.0,  # hot
    }])
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert row["pick_venue_id"] == 5
    assert row["pick_weather_temp"] == 90.0
    assert pd.isna(row["pick_is_indoor"]), "missing roof_type => is_indoor NA"
    assert pd.isna(row["is_park_driven"]), (
        "non-Coors hot with unknown indoor status => is_park_driven NA "
        "(cannot evaluate rule)"
    )


# ---- doubleheader regression (the test Codex #203 most wanted) ----


def test_doubleheader_env_selected_by_game_pk(tmp_path: Path) -> None:
    """Same (batter_id, date) at two game_pks; pick.game_pk drives the env join.

    Hot-trap version: game A is Coors (would yield is_park_driven=True via the
    Coors branch); game B is a non-Coors dome (yields False). Pick is for B.
    A first-PA-for-batter-on-date join could plausibly grab game A's env and
    flip the verdict; this test catches that regression.
    """
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pa(pa_path, [
        _pa_row(batter_id=100, date="2026-05-01", game_pk=20001,
                venue_id=19, roof_type="Open", weather_temp=60.0),  # Coors
        _pa_row(batter_id=100, date="2026-05-01", game_pk=20002,
                venue_id=5, roof_type="Dome", weather_temp=75.0),    # B
    ])
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=20002,  # pick is for game B
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    assert row["pick_venue_id"] == 5
    assert row["pick_venue_id"] != 19, "regression: pulled Coors env from sibling game"
    assert bool(row["pick_is_indoor"]) is True
    assert bool(row["is_park_driven"]) is False


# ---- actual_hit attribution unchanged ----


def test_actual_hit_attribution_unchanged(tmp_path: Path) -> None:
    """Codex #203: 'Existing actual_hit attribution can stay batter_id/date for
    this P0; do not broaden scope.'
    """
    pa_path = tmp_path / "pa.parquet"
    picks_dir = tmp_path / "picks"
    picks_dir.mkdir()
    _write_pa(pa_path, [_pa_row(
        batter_id=100, date="2026-05-01", game_pk=10001,
        venue_id=5, roof_type="Open", weather_temp=70.0, is_hit=True,
    )])
    _write_pick(picks_dir, "2026-05-01.json", _basic_pick(
        date="2026-05-01", run_time=RT_POST_BPM,
        batter_id=100, game_pk=10001,
    ))

    df = canonicalize(picks_dir, pa_path)
    row = df.iloc[0]
    # actual_hit comes through as a numpy bool, not Python bool — value, not identity
    assert bool(row["actual_hit"]) is True
    assert row["result_status"] == "resolved"
    assert row["attribution_source"] == "pa_frame"
