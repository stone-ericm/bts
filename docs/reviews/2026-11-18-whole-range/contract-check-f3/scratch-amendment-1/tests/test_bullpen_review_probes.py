"""Adversarial regression probes for opp_bullpen_hr_30g official-date attachment."""

from __future__ import annotations

from collections import defaultdict
import json
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from bts.features import compute, park_drag

FEATURE = "opp_bullpen_hr_30g"


def _pa_row(
    pa_id: int,
    date,
    game_pk: int,
    *,
    batter_id: int = 101,
    pitcher_id: int = 202,
    is_home: bool = True,
    is_hit=0,
    venue_id: int = 31,
    season: int = 2025,
) -> dict:
    return {
        "pa_id": pa_id,
        "date": date,
        "season": season,
        "game_pk": game_pk,
        "batter_id": batter_id,
        "pitcher_id": pitcher_id,
        "pitch_hand": "R",
        "venue_id": venue_id,
        "is_home": is_home,
        "is_hit": is_hit,
        "pitch_calls": ["X"],
        "pitch_types": ["FF"],
        "pitch_speeds": [93.0],
        "pitch_spin_rates": [2300.0],
        "pitch_extensions": [6.0],
        "pitch_break_vertical": [15.0],
        "pitch_break_horizontal": [5.0],
        "pitch_px": [0.0],
        "pitch_pz": [2.5],
        "sz_top": 3.5,
        "sz_bottom": 1.5,
        "final_count_balls": 1,
        "final_count_strikes": 1,
        "launch_speed": 91.0,
        "launch_angle": 12.0,
        "weather_temp": 72.0,
        "weather_wind_dir": "Out To CF",
        "weather_wind_speed": 8.0,
        "roof_type": "Open",
        "hp_umpire_id": 401,
        "hardness": "medium",
    }


def _run_compute(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    with (
        patch.object(compute, "_build_probable_pitcher_lookup", return_value=lookup),
        patch.object(park_drag, "get_table", return_value=None),
    ):
        return compute.compute_all_features(df)


def _plain_loop_reference(df: pd.DataFrame, lookup: dict) -> pd.Series:
    parsed_dates = pd.to_datetime(df["date"])
    buckets: dict[tuple[object, pd.Timestamp], list[float]] = defaultdict(list)
    targets: dict[int, tuple[object, pd.Timestamp]] = {}

    for idx in range(len(df)):
        row = df.iloc[idx]
        dt = parsed_dates.iloc[idx]
        info = lookup.get(row["game_pk"], {}) if lookup else {}
        if bool(row["is_home"]):
            prob = info.get("away")
            tid = info.get("away_tid")
        else:
            prob = info.get("home")
            tid = info.get("home_tid")
        targets[int(row["pa_id"])] = (tid, dt)
        if not lookup or pd.isna(tid) or pd.isna(dt):
            continue
        if pd.notna(prob) and row["pitcher_id"] != prob:
            b = buckets[(tid, dt)]
            if pd.notna(row["is_hit"]):
                b.append(float(row["is_hit"]))

    by_team: dict[object, list[tuple[pd.Timestamp, float]]] = defaultdict(list)
    for (tid, dt), hits in buckets.items():
        by_team[tid].append((dt, sum(hits) / len(hits) if hits else float("nan")))
    for tid in by_team:
        by_team[tid].sort(key=lambda pair: pair[0])

    expected: dict[int, float] = {}
    for pa_id, (tid, dt) in targets.items():
        if not lookup or pd.isna(tid) or pd.isna(dt):
            expected[pa_id] = float("nan")
            continue
        prior = [r for d, r in by_team.get(tid, ()) if d < dt]
        defined = [r for r in prior[-30:] if not np.isnan(r)]
        expected[pa_id] = sum(defined) / len(defined) if len(defined) >= 10 else float("nan")
    return pd.Series(expected, dtype="float64")


def _assert_matches_loop(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    out = _run_compute(df, lookup)
    assert not [c for c in out.columns if c.startswith("_")]
    actual = out.set_index("pa_id")[FEATURE]
    expected = _plain_loop_reference(df, lookup).reindex(actual.index)
    np.testing.assert_array_equal(actual.isna().to_numpy(), expected.isna().to_numpy())
    np.testing.assert_allclose(actual.dropna().to_numpy(), expected.dropna().to_numpy(), atol=1e-12, rtol=0.0)
    return out


def test_interleaved_multi_team_reliever_history_global_date_order():
    """Kills team-first sorting on right-side history when >=2 teams have interleaved reliever dates."""
    rows = []
    lookup = {}
    pid = 0
    for d in range(12):
        dt = f"2025-04-{d + 1:02d}"
        gpk = 1000 + d
        lookup[gpk] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
        # On day 10, team 51 uses starter only (gap) while team 52 uses reliever; flip on day 11.
        p_away = 21 if d == 10 else 22
        p_home = 31 if d == 11 else 32
        rows.append(_pa_row(pid, dt, gpk, pitcher_id=p_away, is_home=True, is_hit=int(d < 4))); pid += 1
        rows.append(_pa_row(pid, dt, gpk, pitcher_id=p_home, is_home=False, is_hit=int(d >= 3))); pid += 1

    df = pd.DataFrame(rows).sample(frac=1.0, random_state=19)
    out = _assert_matches_loop(df, lookup).set_index("pa_id")
    # Day 10 (pa_id=20 for team 51, starter-only gap): 10 prior buckets (4 hits) -> 0.4
    assert out.loc[20, FEATURE] == pytest.approx(0.4, abs=1e-12)
    # Day 11 (pa_id=23 for team 52, starter-only gap): 11 prior buckets (8 hits) -> 8/11
    assert out.loc[23, FEATURE] == pytest.approx(8 / 11, abs=1e-12)


def test_partial_lookup_float_keys_never_reliever_team_and_mixed_types():
    """Covers float64 PA keys from partial lookup, a team with zero reliever buckets, and mixed int/float/str keys."""
    rows = []
    lookup = {}
    pid = 0
    for d in range(10):
        dt = f"2025-05-{d + 1:02d}"
        gpk = 2000 + d
        # Team 51 always uses relievers; Team 53 NEVER uses relievers
        lookup[gpk] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 53}
        rows.append(_pa_row(pid, dt, gpk, pitcher_id=22, is_home=True, is_hit=int(d % 2 == 0))); pid += 1
        rows.append(_pa_row(pid, dt, gpk, pitcher_id=31, is_home=False, is_hit=1)); pid += 1

    # Day 10: Team 51 starter-only gap + Team 53 starter-only
    lookup[2010] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 53}
    rows.append(_pa_row(pid, "2025-05-11", 2010, pitcher_id=21, is_home=True, is_hit=0)); gap_pid = pid; pid += 1
    rows.append(_pa_row(pid, "2025-05-11", 2010, pitcher_id=31, is_home=False, is_hit=0)); norel_pid = pid; pid += 1

    # Day 11: Game 2011 completely absent from lookup -> makes away_tid & home_tid float64 with NaN!
    rows.append(_pa_row(pid, "2025-05-12", 2011, pitcher_id=22, is_home=True, is_hit=1)); unkn_pid = pid; pid += 1

    df = pd.DataFrame(rows)
    out = _assert_matches_loop(df, lookup).set_index("pa_id")
    assert out["opp_pitching_team_id"].dtype == np.float64
    assert out.loc[gap_pid, FEATURE] == pytest.approx(0.5, abs=1e-12)
    assert np.isnan(out.loc[norel_pid, FEATURE])
    assert np.isnan(out.loc[unkn_pid, FEATURE])

    # Also check object column with int(51), float(51.0), and str("ALT") on the same date (validate="m:1" safety)
    lookup_mixed = dict(lookup)
    lookup_mixed[2012] = {"away": 21, "home": 31, "away_tid": 51.0, "home_tid": "ALT"}
    rows_mixed = list(rows) + [
        _pa_row(pid, "2025-05-11", 2012, pitcher_id=21, is_home=True, is_hit=0),
        _pa_row(pid + 1, "2025-05-11", 2012, pitcher_id=31, is_home=False, is_hit=0),
    ]
    out_mixed = _run_compute(pd.DataFrame(rows_mixed), lookup_mixed).set_index("pa_id")
    assert out_mixed.loc[pid, FEATURE] == pytest.approx(0.5, abs=1e-12)
    assert np.isnan(out_mixed.loc[pid + 1, FEATURE])


def test_undefined_window_carries_nan_not_stale_value_across_gap():
    """When undefined daily rates push valid count in 30-bucket window below 10, gaps must get NaN."""
    rows = []
    lookup = {}
    pid = 0
    base = pd.Timestamp("2025-06-01")
    # 10 defined reliever buckets with rate 1.0
    for d in range(10):
        gpk = 3000 + d
        lookup[gpk] = {"away": 21, "away_tid": 51}
        rows.append(_pa_row(pid, base + pd.Timedelta(days=d), gpk, pitcher_id=22, is_hit=1)); pid += 1
    # 21 undefined reliever buckets (is_hit=NaN): now last 30 buckets have 9 defined + 21 NaN -> window mean is NaN!
    for d in range(10, 31):
        gpk = 3000 + d
        lookup[gpk] = {"away": 21, "away_tid": 51}
        rows.append(_pa_row(pid, base + pd.Timedelta(days=d), gpk, pitcher_id=22, is_hit=np.nan)); pid += 1
    # Starter-only gap on day 31 must match that most recent NaN bucket, NOT carry 1.0!
    lookup[3031] = {"away": 21, "away_tid": 51}
    rows.append(_pa_row(pid, base + pd.Timedelta(days=31), 3031, pitcher_id=21, is_hit=0)); gap_pid = pid

    out = _assert_matches_loop(pd.DataFrame(rows), lookup).set_index("pa_id")
    assert np.isnan(out.loc[gap_pid, FEATURE])


def test_timezone_aware_midnight_dates_preserve_dtype():
    """Timezone-aware midnight dates across DST transition work cleanly and preserve date dtype."""
    dates = pd.date_range("2025-10-28", periods=12, freq="D", tz="America/New_York")
    rows = []
    lookup = {}
    for i, dt in enumerate(dates):
        gpk = 4000 + i
        lookup[gpk] = {"away": 21, "away_tid": 51}
        rows.append(_pa_row(i, dt, gpk, pitcher_id=22 if i < 10 else 21, is_hit=int(i < 5)))
    out = _assert_matches_loop(pd.DataFrame(rows), lookup)
    assert getattr(out["date"].dt, "tz", None) is not None
    assert out.set_index("pa_id").loc[10, FEATURE] == pytest.approx(0.5, abs=1e-12)
    assert out.set_index("pa_id").loc[11, FEATURE] == pytest.approx(0.5, abs=1e-12)


def test_same_day_and_future_invariance_all_features():
    """Same-day perturbations leave bullpen unchanged on D; future perturbations leave ALL cols <= D exact."""
    rows = []
    lookup = {}
    pid = 0
    for d in range(15):
        dt = f"2025-07-{d + 1:02d}"
        gpk = 5000 + d
        lookup[gpk] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
        rows.append(_pa_row(pid, dt, gpk, batter_id=101, pitcher_id=22, is_home=True, is_hit=d % 2)); pid += 1
        rows.append(_pa_row(pid, dt, gpk, batter_id=102, pitcher_id=32, is_home=False, is_hit=(d + 1) % 2)); pid += 1

    df = pd.DataFrame(rows)
    target_d = "2025-07-12"
    ref = _run_compute(df, lookup)

    # Same-day perturbation: starter-only + flipped hits + added doubleheader game on target_d
    df_same = df.copy()
    on_d = df_same["date"] == target_d
    df_same.loc[on_d, "pitcher_id"] = np.where(df_same.loc[on_d, "is_home"], 21, 31)
    df_same.loc[on_d, "is_hit"] = 1 - df_same.loc[on_d, "is_hit"]
    lookup_same = dict(lookup)
    lookup_same[5999] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
    extra_game = pd.DataFrame([
        _pa_row(9001, target_d, 5999, batter_id=101, pitcher_id=22, is_home=True, is_hit=1),
        _pa_row(9002, target_d, 5999, batter_id=102, pitcher_id=32, is_home=False, is_hit=1),
    ])
    out_same = _run_compute(pd.concat([df_same, extra_game], ignore_index=True), lookup_same).set_index("pa_id")
    ref_idx = ref.set_index("pa_id")
    orig_d_ids = df.loc[df["date"] == target_d, "pa_id"]
    pd.testing.assert_series_equal(out_same.loc[orig_d_ids, FEATURE], ref_idx.loc[orig_d_ids, FEATURE], check_exact=True)

    # Future perturbation strictly after target_d: ALL columns on date <= target_d must match exactly
    df_fut = df.copy()
    after_d = df_fut["date"] > target_d
    df_fut.loc[after_d, "is_hit"] = 1 - df_fut.loc[after_d, "is_hit"]
    df_fut.loc[after_d, "pitcher_id"] = 999
    extra_fut = pd.DataFrame([_pa_row(9900, "2025-07-25", 5888, pitcher_id=22, is_home=True, is_hit=1)])
    lookup_fut = {**lookup, 5888: {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}}
    out_fut = _run_compute(pd.concat([df_fut, extra_fut], ignore_index=True), lookup_fut)
    ts_d = pd.Timestamp(target_d)
    pd.testing.assert_frame_equal(
        out_fut[out_fut["date"] <= ts_d].reset_index(drop=True),
        ref[ref["date"] <= ts_d].reset_index(drop=True),
        check_exact=True,
        check_index_type=True,
        check_column_type=True,
    )


def test_serving_bullpen_lookup_updates_after_terminal_no_relief_gap():
    """After a terminal starter-only date D, _build_feature_lookups serves D's fresh prior value."""
    try:
        import lightgbm  # noqa: F401
    except (ImportError, OSError):
        pytest.skip("lightgbm unavailable in this environment")
    from bts.model.predict import _build_feature_lookups

    rows = []
    lookup = {}
    # Days 0..9: 10 reliever buckets with hit=0 -> rolling mean after day 9 is 0.0
    for d in range(10):
        gpk = 6000 + d
        lookup[gpk] = {"away": 21, "away_tid": 51}
        rows.append(_pa_row(d, f"2025-08-{d + 1:02d}", gpk, pitcher_id=22, is_hit=0))
    # Day 10 (reliever with hit=1): stored value on Day 10 is 0.0, updated rolling mean is 1/11
    lookup[6010] = {"away": 21, "away_tid": 51}
    rows.append(_pa_row(10, "2025-08-11", 6010, pitcher_id=22, is_hit=1))
    # Day 11 (terminal starter-only complete game): stored value on Day 11 is now 1/11!
    lookup[6011] = {"away": 21, "away_tid": 51}
    rows.append(_pa_row(11, "2025-08-12", 6011, pitcher_id=21, is_hit=0))

    enriched = _run_compute(pd.DataFrame(rows), lookup)
    lk = _build_feature_lookups(enriched)
    assert lk["bullpen"][51] == pytest.approx(1 / 11, abs=1e-12)


def test_real_2025_two_team_interleaved_slice_matches_loop():
    """Multi-team 2025 slice (teams 113 and 135, both with gaps) against plain-loop reference."""
    data = Path(os.environ.get("BULLPEN_TEST_DATA", "data/processed")) / "pa_2025.parquet"
    lookup_path = Path(os.environ.get("BULLPEN_TEST_LOOKUP", "data/models/probable_pitcher_lookup.json"))
    if not data.is_file() or not lookup_path.is_file():
        pytest.skip("2025 PA data and probable lookup are required")
    lookup = {int(pk): row for pk, row in json.loads(lookup_path.read_text()).items()}
    frame = pd.read_parquet(data)
    dates = pd.to_datetime(frame["date"])
    team_away = frame["game_pk"].map(lambda pk: lookup.get(pk, {}).get("away_tid"))
    team_home = frame["game_pk"].map(lambda pk: lookup.get(pk, {}).get("home_tid"))
    pitching_team = team_away.where(frame["is_home"], team_home)
    sliced = frame.loc[(dates <= "2025-07-25") & pitching_team.isin([113, 135])].assign(
        pa_id=lambda r: np.arange(len(r))
    )
    assert len(sliced) > 500
    out = _assert_matches_loop(sliced, lookup).set_index("pa_id")
    for tid, gap_dt in ((135, "2025-05-10"), (113, "2025-07-23")):
        mask = (out["opp_pitching_team_id"] == tid) & (out["date"] == pd.Timestamp(gap_dt))
        assert mask.any() and out.loc[mask, FEATURE].notna().all()
