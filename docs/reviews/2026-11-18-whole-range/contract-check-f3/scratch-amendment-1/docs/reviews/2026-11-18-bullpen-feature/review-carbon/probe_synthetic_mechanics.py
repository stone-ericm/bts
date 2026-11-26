"""Adversarial synthetic probe for compute_all_features bullpen attachment (items 1-4)."""

from __future__ import annotations

from collections import defaultdict
import subprocess
import types
from unittest.mock import patch

import numpy as np
import pandas as pd

from bts.features import compute, park_drag

FEATURE = "opp_bullpen_hr_30g"


def load_base_compute_module():
    src = subprocess.check_output(
        ["git", "show", "43fc5628:src/bts/features/compute.py"], text=True
    )
    mod = types.ModuleType("bts_compute_base")
    mod.__file__ = "<43fc5628:src/bts/features/compute.py>"
    exec(compile(src, mod.__file__, "exec"), mod.__dict__)
    return mod


BASE_COMPUTE = load_base_compute_module()


def make_pa_row(
    pa_id: int,
    date,
    game_pk: int,
    batter_id: int = 101,
    pitcher_id: int = 202,
    is_home: bool = True,
    is_hit=0,
    venue_id: int = 1,
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


def run_branch(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    with (
        patch.object(compute, "_build_probable_pitcher_lookup", return_value=lookup),
        patch.object(park_drag, "get_table", return_value=None),
    ):
        return compute.compute_all_features(df)


def run_base(df: pd.DataFrame, lookup: dict) -> pd.DataFrame:
    with (
        patch.object(BASE_COMPUTE, "_build_probable_pitcher_lookup", return_value=lookup),
        patch.object(park_drag, "get_table", return_value=None),
    ):
        return BASE_COMPUTE.compute_all_features(df)


def scratch_reference(df: pd.DataFrame, lookup: dict) -> pd.Series:
    """Plain-loop reference from contract specification (no pandas rolling)."""
    parsed_dates = pd.to_datetime(df["date"])
    # Pool qualifying reliever PAs per (team_id, officialDate)
    reliever_buckets: dict[tuple[object, pd.Timestamp], list[float]] = defaultdict(list)
    row_keys: list[tuple[object, pd.Timestamp]] = []

    for idx in range(len(df)):
        row = df.iloc[idx]
        dt = parsed_dates.iloc[idx]
        gpk = row["game_pk"]
        is_home = bool(row["is_home"])
        info = lookup.get(gpk, {}) if lookup else {}
        if is_home:
            prob_pid = info.get("away")
            team_id = info.get("away_tid")
        else:
            prob_pid = info.get("home")
            team_id = info.get("home_tid")
        row_keys.append((team_id, dt))

        if not lookup:
            continue
        if pd.isna(team_id) or pd.isna(dt):
            continue
        if pd.notna(prob_pid) and row["pitcher_id"] != prob_pid:
            # Bucket exists even if is_hit is NaN!
            bucket = reliever_buckets[(team_id, dt)]
            if pd.notna(row["is_hit"]):
                bucket.append(float(row["is_hit"]))

    # Sort buckets chronologically per team
    team_history: dict[object, list[tuple[pd.Timestamp, float]]] = defaultdict(list)
    for (tid, dt), hits in reliever_buckets.items():
        rate = sum(hits) / len(hits) if len(hits) > 0 else float("nan")
        team_history[tid].append((dt, rate))
    for tid in team_history:
        team_history[tid].sort(key=lambda pair: pair[0])

    # Evaluate at each row's (team_id, dt)
    out = []
    for tid, dt in row_keys:
        if not lookup or pd.isna(tid) or pd.isna(dt):
            out.append(float("nan"))
            continue
        hist = team_history.get(tid, [])
        strictly_prior = [r for d, r in hist if d < dt]
        window = strictly_prior[-30:]
        defined = [r for r in window if not np.isnan(r)]
        if len(defined) >= 10:
            out.append(sum(defined) / len(defined))
        else:
            out.append(float("nan"))
    return pd.Series(out, index=df["pa_id"], dtype="float64")


def assert_matches_reference(df: pd.DataFrame, lookup: dict, label: str):
    res = run_branch(df, lookup)
    # Check no temporary columns leaked
    leaked = [c for c in res.columns if c.startswith("_")]
    assert not leaked, f"{label}: leaked temp cols {leaked}"
    actual = res.set_index("pa_id")[FEATURE]
    expected = scratch_reference(df, lookup).reindex(actual.index)
    np.testing.assert_array_equal(
        actual.isna().to_numpy(),
        expected.isna().to_numpy(),
        err_msg=f"{label}: NaN mask mismatch",
    )
    np.testing.assert_allclose(
        actual.dropna().to_numpy(),
        expected.dropna().to_numpy(),
        atol=1e-12,
        rtol=0.0,
        err_msg=f"{label}: finite value mismatch",
    )
    return res


def test_all():
    print(f"pandas={pd.__version__}, numpy={np.__version__}")

    # 1. Comprehensive window + gaps + doubleheader + season boundary + team ID change + unknown probable + null team/date + partial lookup + never-reliever team + undefined bucket
    rows = []
    lookup = {}
    pa_id = 0
    d0 = pd.Timestamp("2024-09-01")

    # Team 51: 40 dates across 2024 -> 2025 boundary, interleaved with Team 52 and Team 53 (never reliever)
    for step in range(45):
        dt = d0 + pd.Timedelta(days=step) if step < 20 else pd.Timestamp("2025-03-27") + pd.Timedelta(days=step - 20)
        season = dt.year
        gpk1 = 1000 + step * 10
        lookup[gpk1] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}

        if step in (9, 10, 22, 44):
            # step 9: only 9 prior buckets -> starter-only gap (<10 prior)
            # step 10: wait - make step 9 reliever so step 10 is first eligible no-relief date!
            pass

    # Build explicit scenario cleanly
    rows = []
    lookup = {}
    pa_id = 0

    # Team 51 (away) vs Team 52 (home), plus Team 53 (never has reliever bucket)
    # Days 0..8 (9 days): 2024-09-20..2024-09-28: Team 51 reliever (rate = step/10), Team 52 starter only
    for s in range(9):
        dt = pd.Timestamp("2024-09-20") + pd.Timedelta(days=s)
        gpk = 2000 + s
        lookup[gpk] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
        # 2 PAs for team 51 reliever to test hit pooling
        rows.append(make_pa_row(pa_id, str(dt.date()), gpk, pitcher_id=22, is_home=True, is_hit=1 if s % 2 == 0 else 0, season=2024)); pa_id += 1
        rows.append(make_pa_row(pa_id, str(dt.date()), gpk, pitcher_id=23, is_home=True, is_hit=0, season=2024)); pa_id += 1
        # Team 52 starter only (never reliever yet)
        rows.append(make_pa_row(pa_id, str(dt.date()), gpk, pitcher_id=31, is_home=False, is_hit=1, season=2024)); pa_id += 1

    # Day 9 (2024-09-29): Team 51 has 9 prior buckets -> starter-only date (must be NaN, <10)
    dt9 = "2024-09-29"
    lookup[2009] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
    rows.append(make_pa_row(pa_id, dt9, 2009, pitcher_id=21, is_home=True, is_hit=1, season=2024)); pa_id += 1

    # Day 10 (2024-09-30): Team 51 gets 10th reliever bucket (still 9 strictly prior -> NaN on Day 10!)
    dt10 = "2024-09-30"
    lookup[2010] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
    rows.append(make_pa_row(pa_id, dt10, 2010, pitcher_id=22, is_home=True, is_hit=1, season=2024)); pa_id += 1

    # Season boundary! 2025-03-27: FIRST ELIGIBLE NO-RELIEF DATE for Team 51 (exactly 10 prior buckets from 2024!)
    dt_open = "2025-03-27"
    lookup[3000] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 53}  # Team 53 only ever known, never reliever
    rows.append(make_pa_row(pa_id, dt_open, 3000, pitcher_id=21, is_home=True, is_hit=0, season=2025)); pa_id += 1
    rows.append(make_pa_row(pa_id, dt_open, 3000, pitcher_id=31, is_home=False, is_hit=1, season=2025)); pa_id += 1

    # Doubleheader on 2025-03-28 for Team 51: Game 1 uses reliever (all hits), Game 2 starter only
    # Both games must get the exact same value (10 prior buckets), Game 2 must NOT see Game 1!
    dt_dh = "2025-03-28"
    lookup[3001] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
    lookup[3002] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
    rows.append(make_pa_row(pa_id, dt_dh, 3002, pitcher_id=21, is_home=True, is_hit=0, season=2025)); pa_id += 1
    rows.append(make_pa_row(pa_id, dt_dh, 3001, pitcher_id=22, is_home=True, is_hit=1, season=2025)); pa_id += 1
    # Also Team 52 reliever bucket on 2025-03-28 (interleaved with Team 51!)
    rows.append(make_pa_row(pa_id, dt_dh, 3001, pitcher_id=32, is_home=False, is_hit=1, season=2025)); pa_id += 1

    # Gap on 2025-03-29 after doubleheader (now 11 prior buckets, mean changed from 2025-03-27!)
    dt_gap2 = "2025-03-29"
    lookup[3003] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 53}
    rows.append(make_pa_row(pa_id, dt_gap2, 3003, pitcher_id=21, is_home=True, is_hit=0, season=2025)); pa_id += 1
    rows.append(make_pa_row(pa_id, dt_gap2, 3003, pitcher_id=31, is_home=False, is_hit=0, season=2025)); pa_id += 1

    # Add 25 more reliever dates for Team 51 (total > 30 buckets) + interleaved Team 52 reliever dates
    # Include 2 undefined-rate buckets (is_hit = NaN)
    for k in range(25):
        dt_k = pd.Timestamp("2025-04-01") + pd.Timedelta(days=k)
        gpk = 3100 + k
        lookup[gpk] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
        hit_val = np.nan if k in (3, 10) else (1 if k % 3 == 0 else 0)
        rows.append(make_pa_row(pa_id, str(dt_k.date()), gpk, pitcher_id=22, is_home=True, is_hit=hit_val, season=2025)); pa_id += 1
        if k < 12:
            rows.append(make_pa_row(pa_id, str(dt_k.date()), gpk, pitcher_id=32, is_home=False, is_hit=0, season=2025)); pa_id += 1

    # Unknown probable with known team (2025-04-27)
    lookup[3200] = {"away": None, "home": 31, "away_tid": 51, "home_tid": 52}
    rows.append(make_pa_row(pa_id, "2025-04-27", 3200, pitcher_id=22, is_home=True, is_hit=1, season=2025)); pa_id += 1

    # Partial lookup: game 3201 absent from lookup completely (null team)
    rows.append(make_pa_row(pa_id, "2025-04-28", 3201, pitcher_id=22, is_home=True, is_hit=1, season=2025)); pa_id += 1

    # Team ID change: Team 51 becomes Team 951 on 2025-04-29 (starts fresh -> NaN)
    lookup[3202] = {"away": 21, "home": 31, "away_tid": 951, "home_tid": 52}
    rows.append(make_pa_row(pa_id, "2025-04-29", 3202, pitcher_id=22, is_home=True, is_hit=1, season=2025)); pa_id += 1

    # Null date row
    lookup[3203] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
    rows.append(make_pa_row(pa_id, None, 3203, pitcher_id=22, is_home=True, is_hit=1, season=2025)); pa_id += 1

    # Trailing gap for Team 51 and Team 52 on 2025-04-30
    lookup[3204] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
    rows.append(make_pa_row(pa_id, "2025-04-30", 3204, pitcher_id=21, is_home=True, is_hit=0, season=2025)); pa_id += 1
    rows.append(make_pa_row(pa_id, "2025-04-30", 3204, pitcher_id=31, is_home=False, is_hit=0, season=2025)); pa_id += 1

    df = pd.DataFrame(rows)
    res = assert_matches_reference(df, lookup, "comprehensive_ordered")
    print("PASS: comprehensive_ordered matches scratch reference")

    # Shuffled input + non-default index
    shuffled = df.sample(frac=1.0, random_state=123)
    shuffled.index = [f"row_{i}" for i in range(len(shuffled))]
    res_shuf = assert_matches_reference(shuffled, lookup, "comprehensive_shuffled_str_index")
    print(f"INFO: input index preserved? {res_shuf.index.equals(shuffled.index)} (out index={type(res_shuf.index).__name__})")

    # Check empty lookup schema vs base
    df_clean = df.dropna(subset=["date"]).copy()
    base_empty = run_base(df_clean, {})
    branch_empty = run_branch(df_clean, {})
    pd.testing.assert_frame_equal(
        base_empty, branch_empty, check_exact=True, check_index_type=True, check_column_type=True
    )
    assert "opp_pitching_team_id" not in branch_empty.columns
    print("PASS: empty probable lookup identical to base (opp_pitching_team_id absent)")

    # Check populated lookup with ZERO reliever PAs (daily_bullpen.empty == True) vs base
    df_no_rel = df_clean.copy()
    df_no_rel["pitcher_id"] = np.where(df_no_rel["is_home"], 21, 31)
    base_norel = run_base(df_no_rel, lookup)
    branch_norel = run_branch(df_no_rel, lookup)
    pd.testing.assert_frame_equal(
        base_norel, branch_norel, check_exact=True, check_index_type=True, check_column_type=True
    )
    print("PASS: zero-reliever-PA populated lookup identical to base")

    # Check dtype of opp_pitching_team_id on base vs branch across key-type cases:
    # Case K1: away_tid only (object team id in df, int64 in daily_bullpen!)
    df_k1 = pd.DataFrame([
        make_pa_row(i, f"2025-04-{i+1:02d}", 4000 + i, pitcher_id=22 if i < 10 else 21, is_home=True, is_hit=i % 2)
        for i in range(11)
    ])
    lookup_k1 = {4000 + i: {"away": 21, "away_tid": 51} for i in range(11)}
    try:
        b_k1 = run_base(df_k1, lookup_k1)
        base_k1_status = f"ok (dtype={b_k1['opp_pitching_team_id'].dtype})"
    except Exception as exc:
        base_k1_status = f"RAISED {type(exc).__name__}: {exc}"
    br_k1 = run_branch(df_k1, lookup_k1)
    print(f"Case K1 (away_tid int, home_tid missing -> object in df): base={base_k1_status}, branch=ok (dtype={br_k1['opp_pitching_team_id'].dtype})")

    # Case K2: partial lookup -> float64 in df
    lookup_k2 = {4000 + i: {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52} for i in range(10)}  # 4010 missing!
    b_k2 = run_base(df_k1, lookup_k2)
    br_k2 = run_branch(df_k1, lookup_k2)
    print(f"Case K2 (partial lookup -> float64 in df): base dtype={b_k2['opp_pitching_team_id'].dtype}, branch dtype={br_k2['opp_pitching_team_id'].dtype}")
    other_cols = [c for c in b_k2.columns if c != FEATURE]
    pd.testing.assert_frame_equal(b_k2[other_cols], br_k2[other_cols], check_exact=True, check_index_type=True, check_column_type=True)

    # Case K3: string team IDs in lookup, mixed int/float in lookup
    lookup_k3 = {4000 + i: {"away": 21, "away_tid": "TEAM_51"} for i in range(11)}
    br_k3 = assert_matches_reference(df_k1, lookup_k3, "string_team_ids")
    print(f"Case K3 (string team IDs): branch ok, last val={br_k3[FEATURE].iloc[-1]}")

    # Case K4: mixed int(51) and float(51.0) in lookup across dates & on SAME date!
    df_k4_rows = [
        make_pa_row(i, f"2025-04-{i+1:02d}", 5000 + i, pitcher_id=22, is_home=True, is_hit=1)
        for i in range(10)
    ]
    # On 2025-04-11, two games: one has int(51), one has float(51.0), plus a third with str("OTHER") so series stays object!
    df_k4_rows.append(make_pa_row(10, "2025-04-11", 5010, pitcher_id=21, is_home=True, is_hit=0))
    df_k4_rows.append(make_pa_row(11, "2025-04-11", 5011, pitcher_id=21, is_home=True, is_hit=0))
    df_k4_rows.append(make_pa_row(12, "2025-04-11", 5012, pitcher_id=21, is_home=True, is_hit=0))
    lookup_k4 = {5000 + i: {"away": 21, "away_tid": 51} for i in range(10)}
    lookup_k4[5010] = {"away": 21, "away_tid": 51}
    lookup_k4[5011] = {"away": 21, "away_tid": 51.0}
    lookup_k4[5012] = {"away": 21, "away_tid": "OTHER"}
    try:
        br_k4 = run_branch(pd.DataFrame(df_k4_rows), lookup_k4)
        print(f"Case K4 (mixed int(51), float(51.0), str in object col on same date): ok, vals={br_k4.set_index('pa_id').loc[[10, 11, 12], FEATURE].to_dict()}")
    except Exception as exc:
        print(f"Case K4 RAISED {type(exc).__name__}: {exc}")

    # Case TZ: timezone-aware midnight dates
    df_tz = df_k1.copy()
    df_tz["date"] = pd.to_datetime(df_tz["date"]).dt.tz_localize("America/New_York")
    try:
        br_tz = run_branch(df_tz, lookup_k1)
        b_tz = run_base(df_tz, lookup_k1)
        print(f"Case TZ (America/New_York midnight): branch ok={br_tz[FEATURE].iloc[-1]}, base={b_tz[FEATURE].iloc[-1]}, out date dtype={br_tz['date'].dtype}")
    except Exception as exc:
        print(f"Case TZ RAISED {type(exc).__name__}: {exc}")

    # Case TOD: time-of-day timestamps on same calendar date (doubleheader at 13:05 and 19:10)
    df_tod_rows = [
        make_pa_row(i, f"2025-04-{i+1:02d} 00:00:00", 6000 + i, pitcher_id=22, is_home=True, is_hit=0)
        for i in range(10)
    ]
    df_tod_rows.append(make_pa_row(10, "2025-04-11 13:05:00", 6010, pitcher_id=22, is_home=True, is_hit=1))
    df_tod_rows.append(make_pa_row(11, "2025-04-11 19:10:00", 6011, pitcher_id=21, is_home=True, is_hit=0))
    lookup_tod = {6000 + i: {"away": 21, "away_tid": 51} for i in range(12)}
    br_tod = run_branch(pd.DataFrame(df_tod_rows), lookup_tod).set_index("pa_id")
    b_tod = run_base(pd.DataFrame(df_tod_rows), lookup_tod).set_index("pa_id")
    print(
        f"Case TOD (13:05 reliever hit=1, 19:10 starter): "
        f"branch 13:05={br_tod.loc[10, FEATURE]}, branch 19:10={br_tod.loc[11, FEATURE]} (sees 13:05!), "
        f"branch batter_hr_7g 13:05={br_tod.loc[10, 'batter_hr_7g']}, 19:10={br_tod.loc[11, 'batter_hr_7g']}; "
        f"base batter_hr_7g 19:10={b_tod.loc[11, 'batter_hr_7g']}"
    )

    # Case DATE_RES: string dates vs datetime64[ns] vs datetime64[us]
    df_str = df_k1.copy()
    df_ns = df_k1.copy()
    df_ns["date"] = pd.to_datetime(df_ns["date"]).astype("datetime64[ns]")
    df_us = df_k1.copy()
    df_us["date"] = pd.to_datetime(df_us["date"]).astype("datetime64[us]")
    r_str = run_branch(df_str, lookup_k1)
    r_ns = run_branch(df_ns, lookup_k1)
    r_us = run_branch(df_us, lookup_k1)
    print(f"Case DATE_RES out date dtypes: str->{r_str['date'].dtype}, ns->{r_ns['date'].dtype}, us->{r_us['date'].dtype}")
    np.testing.assert_array_equal(r_str[FEATURE].to_numpy(), r_ns[FEATURE].to_numpy())
    np.testing.assert_array_equal(r_str[FEATURE].to_numpy(), r_us[FEATURE].to_numpy())


if __name__ == "__main__":
    test_all()
