"""Comprehensive real-data verification v3 for items 1, 2, 3, 4, and 5 (2017-2025 ONLY; no 2026)."""

from __future__ import annotations

from collections import defaultdict
import copy
import gc
import hashlib
import importlib.util
import json
from pathlib import Path
import resource
import sys
import time
import tracemalloc
from unittest.mock import patch

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO / "src") not in sys.path:
    sys.path.insert(0, str(REPO / "src"))

from bts.features import compute as branch_compute
from bts.features import park_drag

PA_DIR = Path("/Users/eric/projects/bts/data/processed")
LOOKUP_PATH = Path("/Users/eric/bts-runs/inputs/probable_pitcher_lookup.2017-2025.json")
BASE_COMPUTE_PATH = Path(__file__).resolve().parent / "base_tree" / "src" / "bts" / "features" / "compute.py"
FEATURE = "opp_bullpen_hr_30g"


def load_base_compute():
    spec = importlib.util.spec_from_file_location("bts_base_compute", BASE_COMPUTE_PATH)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


BASE_COMPUTE = load_base_compute()


def load_pa_year(year: int) -> pd.DataFrame:
    assert 2017 <= year <= 2025, f"FORBIDDEN year {year}: must be <= 2025"
    p = PA_DIR / f"pa_{year}.parquet"
    return pd.read_parquet(p)


def load_frozen_lookup() -> tuple[dict[int, dict], str]:
    raw_bytes = LOOKUP_PATH.read_bytes()
    sha = hashlib.sha256(raw_bytes).hexdigest()
    parsed = json.loads(raw_bytes)
    return {int(k): v for k, v in parsed.items()}, sha


def compute_with(mod, df: pd.DataFrame, lookup: dict[int, dict]) -> pd.DataFrame:
    with (
        patch.object(mod, "_build_probable_pitcher_lookup", return_value=lookup),
        patch.object(park_drag, "get_table", return_value=None),
    ):
        return mod.compute_all_features(df)


def get_maxrss_mb() -> float:
    # On macOS (Darwin), ru_maxrss is in bytes.
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)


def compare_base_vs_branch(
    label: str, df_in: pd.DataFrame, lookup: dict[int, dict]
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    t0 = time.perf_counter()
    base_out = compute_with(BASE_COMPUTE, df_in, lookup)
    t_base = time.perf_counter() - t0

    t1 = time.perf_counter()
    branch_out = compute_with(branch_compute, df_in, lookup)
    t_branch = time.perf_counter() - t1

    assert list(base_out.columns) == list(branch_out.columns), f"{label}: column order mismatch"
    assert len(base_out) == len(branch_out) == len(df_in), f"{label}: row count mismatch"
    leaked = [c for c in branch_out.columns if c.startswith("_")]
    assert not leaked, f"{label}: leaked temporary columns {leaked}"
    for col in base_out.columns:
        assert base_out[col].dtype == branch_out[col].dtype, (
            f"{label}: dtype mismatch on {col}: {base_out[col].dtype} vs {branch_out[col].dtype}"
        )

    other_cols = [c for c in base_out.columns if c != FEATURE]
    pd.testing.assert_frame_equal(
        base_out[other_cols],
        branch_out[other_cols],
        check_exact=True,
        check_index_type=True,
        check_column_type=True,
    )

    b_arr = base_out[FEATURE].to_numpy(dtype=np.float64)
    n_arr = branch_out[FEATURE].to_numpy(dtype=np.float64)
    old_na = np.isnan(b_arr)
    new_na = np.isnan(n_arr)
    filled = int((old_na & ~new_na).sum())
    lost = int((~old_na & new_na).sum())
    assert lost == 0, f"{label}: lost {lost} valued rows!"

    valued = ~old_na
    if valued.any():
        bit_mismatches = int((b_arr[valued].view(np.uint64) != n_arr[valued].view(np.uint64)).sum())
        max_abs_diff = float(np.max(np.abs(b_arr[valued] - n_arr[valued])))
    else:
        bit_mismatches = 0
        max_abs_diff = 0.0
    assert bit_mismatches == 0, f"{label}: {bit_mismatches} float-bit mismatches on valued rows!"

    both_na = old_na & new_na
    nan_bit_mismatches = (
        int((b_arr[both_na].view(np.uint64) != n_arr[both_na].view(np.uint64)).sum())
        if both_na.any()
        else 0
    )

    total_games = int(df_in["game_pk"].nunique())
    games_in_lookup = sum(1 for pk in df_in["game_pk"].unique() if pk in lookup)
    if "opp_pitching_team_id" in branch_out.columns:
        known_team_pas = int(branch_out["opp_pitching_team_id"].notna().sum())
    else:
        known_team_pas = 0
    away_p = df_in["game_pk"].map(lambda pk: lookup.get(pk, {}).get("away"))
    home_p = df_in["game_pk"].map(lambda pk: lookup.get(pk, {}).get("home"))
    known_prob_pas = int(pd.Series(np.where(df_in["is_home"], away_p, home_p)).notna().sum())

    fcols = branch_compute.FEATURE_COLS
    base_any_notna = int(base_out[fcols].notna().any(axis=1).sum())
    branch_any_notna = int(branch_out[fcols].notna().any(axis=1).sum())
    assert base_any_notna == branch_any_notna, f"{label}: train valid mask changed!"

    stats = {
        "label": label,
        "pas": len(df_in),
        "games_cov": f"{games_in_lookup}/{total_games}",
        "known_team_pas": known_team_pas,
        "known_prob_pas": known_prob_pas,
        "old_missing": int(old_na.sum()),
        "new_missing": int(new_na.sum()),
        "filled": filled,
        "lost": lost,
        "bit_mismatches": bit_mismatches,
        "nan_bit_mismatches": nan_bit_mismatches,
        "max_abs_diff": max_abs_diff,
        "t_base_s": round(t_base, 3),
        "t_branch_s": round(t_branch, 3),
    }
    print(f"[IDENTITY] {stats}", flush=True)
    return base_out, branch_out, stats


def plain_loop_reference(df_sorted: pd.DataFrame, lookup: dict[int, dict]) -> np.ndarray:
    """From-scratch plain Python loop over reliever-date buckets (zero pandas rolling/shift)."""
    dates = pd.to_datetime(df_sorted["date"]).tolist()
    gpks = df_sorted["game_pk"].tolist()
    is_homes = df_sorted["is_home"].tolist()
    pids = df_sorted["pitcher_id"].tolist()
    hits = df_sorted["is_hit"].tolist()

    buckets: dict[tuple[object, pd.Timestamp], list[float]] = defaultdict(list)
    row_keys: list[tuple[object, pd.Timestamp]] = []

    for dt, gpk, ih, pid, h in zip(dates, gpks, is_homes, pids, hits):
        info = lookup.get(gpk)
        if info is None:
            row_keys.append((None, dt))
            continue
        if ih:
            prob = info.get("away")
            tid = info.get("away_tid")
        else:
            prob = info.get("home")
            tid = info.get("home_tid")
        row_keys.append((tid, dt))
        if tid is None or pd.isna(tid) or pd.isna(dt):
            continue
        if prob is not None and not pd.isna(prob) and pid != prob:
            b = buckets[(tid, dt)]
            if h is not None and not pd.isna(h):
                b.append(float(h))

    by_team: dict[object, list[tuple[pd.Timestamp, float]]] = defaultdict(list)
    for (tid, dt), hlist in buckets.items():
        rate = sum(hlist) / len(hlist) if hlist else float("nan")
        by_team[tid].append((dt, rate))
    for tid in by_team:
        by_team[tid].sort(key=lambda x: x[0])

    cache: dict[tuple[object, pd.Timestamp], float] = {}
    out = np.empty(len(df_sorted), dtype=np.float64)
    for i, (tid, dt) in enumerate(row_keys):
        if not lookup or tid is None or pd.isna(tid) or pd.isna(dt):
            out[i] = np.nan
            continue
        key = (tid, dt)
        val = cache.get(key)
        if val is None:
            hist = by_team.get(tid, ())
            prior_rates = [r for d, r in hist if d < dt]
            win = prior_rates[-30:]
            defined = [r for r in win if not np.isnan(r)]
            val = sum(defined) / len(defined) if len(defined) >= 10 else float("nan")
            cache[key] = val
        out[i] = val
    return out


def assert_lookups_equal_except_bullpen(lk_a: dict, lk_b: dict, label: str):
    assert set(lk_a.keys()) == set(lk_b.keys()), f"{label}: lookup top keys differ"
    for k in lk_a:
        if k == "bullpen":
            continue
        va, vb = lk_a[k], lk_b[k]
        if isinstance(va, dict):
            assert set(va.keys()) == set(vb.keys()), f"{label}: keys differ in {k}"
            for subk in va:
                sva, svb = va[subk], vb[subk]
                if isinstance(sva, dict):
                    assert set(sva.keys()) == set(svb.keys()), f"{label}: subkeys differ in {k}.{subk}"
                    for k3 in sva:
                        x, y = sva[k3], svb[k3]
                        assert (pd.isna(x) and pd.isna(y)) or x == y, f"{label}: mismatch at {k}.{subk}.{k3}: {x} vs {y}"
                else:
                    assert (pd.isna(sva) and pd.isna(svb)) or sva == svb, f"{label}: mismatch at {k}.{subk}: {sva} vs {svb}"
        else:
            assert (pd.isna(va) and pd.isna(vb)) or va == vb, f"{label}: scalar mismatch at {k}: {va} vs {vb}"


def run_verification(
    pa_2024: pd.DataFrame,
    pa_2025: pd.DataFrame,
    lookup: dict[int, dict],
    load_all_fn,
    gap_sample_date: str = "2025-07-23",
    ord_sample_date: str = "2025-06-15",
    cut_dates: list[str] | None = None,
):
    print("\n=== SECTION 1: Identity on 2024, 2025, and 2024+2025 ===", flush=True)
    print(
        f"Native parquet date dtypes: 2024={pa_2024['date'].dtype}, 2025={pa_2025['date'].dtype}, "
        f"index={type(pa_2025.index).__name__}",
        flush=True,
    )

    base_2024, branch_2024, _ = compare_base_vs_branch("2024_alone", pa_2024, lookup)
    base_2025, branch_2025, _ = compare_base_vs_branch("2025_alone", pa_2025, lookup)
    pa_24_25 = pd.concat([pa_2024, pa_2025], ignore_index=True)
    base_24_25, branch_24_25, _ = compare_base_vs_branch("2024_2025_combined", pa_24_25, lookup)

    compare_base_vs_branch("2024_empty_lookup_vacuous_check", pa_2024, {})
    assert "opp_pitching_team_id" not in compute_with(branch_compute, pa_2024.iloc[:50], {}).columns

    print("\n=== SECTION 1b: Input order, index, duplicates, and date dtype perturbations (2025) ===", flush=True)
    pa_2025_uid = pa_2025.assign(probe_pa_uid=np.arange(len(pa_2025)))
    _, branch_uid_ref, _ = compare_base_vs_branch("2025_with_uid_baseline", pa_2025_uid, lookup)

    shuf_in = pa_2025_uid.sample(frac=1.0, random_state=20251118)
    base_shuf, branch_shuf, _ = compare_base_vs_branch("2025_shuffled", shuf_in, lookup)
    aligned_with_shuf_input_base = bool((base_shuf["probe_pa_uid"].to_numpy() == shuf_in["probe_pa_uid"].to_numpy()).all())
    aligned_with_shuf_input_branch = bool((branch_shuf["probe_pa_uid"].to_numpy() == shuf_in["probe_pa_uid"].to_numpy()).all())
    realigned_branch = branch_shuf.set_index("probe_pa_uid").loc[branch_uid_ref["probe_pa_uid"]].reset_index(drop=True)
    bullpen_shuf_bit_diff = int(
        (
            realigned_branch[FEATURE].dropna().to_numpy(np.float64).view(np.uint64)
            != branch_uid_ref[FEATURE].dropna().to_numpy(np.float64).view(np.uint64)
        ).sum()
    )
    assert realigned_branch[FEATURE].isna().equals(branch_uid_ref[FEATURE].isna())
    all_feat_cols = branch_compute.FEATURE_COLS + branch_compute.CONTEXT_COLS + branch_compute.STATCAST_COLS
    shuf_max_diffs = {}
    for fc in all_feat_cols:
        if fc not in realigned_branch.columns:
            continue
        s1, s2 = realigned_branch[fc], branch_uid_ref[fc]
        assert s1.isna().equals(s2.isna()), f"shuffle NaN mask changed on {fc}"
        if s1.notna().any():
            md = float((s1.dropna() - s2.dropna()).abs().max())
            if md > 0:
                shuf_max_diffs[fc] = md
    print(
        f"[SHUFFLE] output_matches_shuffled_input_order: base={aligned_with_shuf_input_base}, "
        f"branch={aligned_with_shuf_input_branch} (both sort by ['date','game_pk'].reset_index(drop=True)); "
        f"realigned opp_bullpen_hr_30g bit_mismatches={bullpen_shuf_bit_diff}; "
        f"other feature float diffs under within-game reorder={shuf_max_diffs}",
        flush=True,
    )

    idx_in = pa_2025.copy()
    idx_in.index = pd.Index(np.arange(9_000_000, 9_000_000 + len(pa_2025))[::-1], name="custom_rev_idx")
    base_idx, branch_idx, _ = compare_base_vs_branch("2025_nondefault_index", idx_in, lookup)
    pd.testing.assert_frame_equal(branch_idx, branch_2025, check_exact=True, check_index_type=True, check_column_type=True)
    print(
        f"[INDEX] base_index={type(base_idx.index).__name__}(name={base_idx.index.name}), "
        f"branch_index={type(branch_idx.index).__name__}(name={branch_idx.index.name})",
        flush=True,
    )

    step = max(1, len(pa_2025) // 900)
    dup_in = pd.concat([pa_2025, pa_2025.iloc[::step]], ignore_index=True)
    base_dup, branch_dup, _ = compare_base_vs_branch("2025_duplicated_rows", dup_in, lookup)
    print(f"[DUPLICATES] input_len={len(dup_in)}, base_len={len(base_dup)}, branch_len={len(branch_dup)}", flush=True)

    str_in = pa_2025.copy()
    str_in["date"] = pd.to_datetime(str_in["date"]).dt.strftime("%Y-%m-%d").astype(object)
    us_in = pa_2025.copy()
    us_in["date"] = pd.to_datetime(us_in["date"]).astype("datetime64[us]")
    ns_in = pa_2025.copy()
    ns_in["date"] = pd.to_datetime(ns_in["date"]).astype("datetime64[ns]")

    _, branch_str, _ = compare_base_vs_branch("2025_date_object_str", str_in, lookup)
    _, branch_us, _ = compare_base_vs_branch("2025_date_ts_us", us_in, lookup)
    _, branch_ns, _ = compare_base_vs_branch("2025_date_ts_ns", ns_in, lookup)
    non_date_cols = [c for c in branch_2025.columns if c != "date"]
    pd.testing.assert_frame_equal(branch_str[non_date_cols], branch_us[non_date_cols], check_exact=True)
    pd.testing.assert_frame_equal(branch_str[non_date_cols], branch_ns[non_date_cols], check_exact=True)
    print(
        f"[DATE_DTYPES] out date dtype: str_in->{branch_str['date'].dtype}, "
        f"us_in->{branch_us['date'].dtype}, ns_in->{branch_ns['date'].dtype}; all non-date cols exact match!",
        flush=True,
    )

    print("\n=== SECTION 2: Scratch Plain-Loop Reference on 2025 and 2024+2025 ===", flush=True)
    for ref_label, out_df in (("2025_alone", branch_2025), ("2024_2025_combined", branch_24_25)):
        expected = plain_loop_reference(out_df, lookup)
        actual = out_df[FEATURE].to_numpy(dtype=np.float64)
        np.testing.assert_array_equal(np.isnan(actual), np.isnan(expected), err_msg=f"{ref_label} NaN mask mismatch")
        fin = ~np.isnan(actual)
        max_diff = float(np.max(np.abs(actual[fin] - expected[fin]))) if fin.any() else 0.0
        bit_eq = int((actual[fin].view(np.uint64) != expected[fin].view(np.uint64)).sum()) if fin.any() else 0
        assert max_diff <= 1e-12, f"{ref_label}: max_diff {max_diff} > 1e-12"
        print(
            f"[PLAIN_LOOP] {ref_label}: rows={len(actual)}, finite={int(fin.sum())}, "
            f"missing={int((~fin).sum())}, max_abs_diff={max_diff:.3e}, bit_diffs_vs_python_sum={bit_eq}",
            flush=True,
        )

    diff_25 = branch_2025[base_2025[FEATURE].isna() & branch_2025[FEATURE].notna()]
    td_25 = diff_25.groupby(["opp_pitching_team_id", "date"]).agg(
        pas=("is_hit", "count"), hits=("is_hit", "sum"), gpks=("game_pk", "nunique")
    ).reset_index()
    hr_str = f"{td_25['hits'].sum()/td_25['pas'].sum():.6f}" if len(td_25) else "n/a"
    print(
        f"[2025_FILLED] team_dates={len(td_25)}, pas={int(td_25['pas'].sum())}, "
        f"hits={int(td_25['hits'].sum())}, hit_rate={hr_str}",
        flush=True,
    )
    mask_25_in_comb = branch_24_25["season"] == 2025
    diff_comb_25 = branch_24_25[mask_25_in_comb & base_24_25[FEATURE].isna() & branch_24_25[FEATURE].notna()]
    td_comb_25 = diff_comb_25.groupby(["opp_pitching_team_id", "date"]).size().reset_index(name="pas")
    extra_td = set(zip(td_comb_25["opp_pitching_team_id"], td_comb_25["date"])) - set(
        zip(td_25["opp_pitching_team_id"], td_25["date"])
    )
    print(f"[2024_2025_IN_2025_FILLED] team_dates={len(td_comb_25)}, extra_vs_2025_only={extra_td}", flush=True)

    tids_24 = set(branch_2024["opp_pitching_team_id"].dropna().unique())
    tids_25 = set(branch_2025["opp_pitching_team_id"].dropna().unique())
    print(f"[TEAM_IDS] 2024 count={len(tids_24)}, 2025 count={len(tids_25)}, sym_diff={tids_24 ^ tids_25}", flush=True)

    print("\n=== SECTION 3: Same-day and Future Invariance on Real 2025 Data ===", flush=True)
    team_day_games = branch_2025.groupby(["opp_pitching_team_id", "date"])["game_pk"].nunique()
    dh_candidates = team_day_games[team_day_games > 1].reset_index().sort_values("date")
    dh_warmed = dh_candidates[dh_candidates["date"] >= branch_2025["date"].quantile(0.3)]
    if dh_warmed.empty:
        dh_warmed = dh_candidates
    d_dh = dh_warmed.iloc[0]["date"]
    dh_tid = dh_warmed.iloc[0]["opp_pitching_team_id"]

    d_first = branch_2025["date"].min()
    d_last = branch_2025["date"].max()
    d_gap = pd.Timestamp(gap_sample_date)
    d_ord = pd.Timestamp(ord_sample_date)

    target_dates = [
        ("first_date", d_first),
        ("no_relief_date", d_gap),
        ("ordinary_date", d_ord),
        (f"doubleheader_date(team={dh_tid})", d_dh),
        ("last_date", d_last),
    ]

    base_frame = pa_2025_uid.copy()
    base_dates = pd.to_datetime(base_frame["date"])
    ref_uid_map = branch_uid_ref.set_index("probe_pa_uid")[FEATURE]
    ref_team_date_map = (
        branch_uid_ref.dropna(subset=["opp_pitching_team_id"])
        .drop_duplicates(["opp_pitching_team_id", "date"])
        .set_index(["opp_pitching_team_id", "date"])[FEATURE]
    )

    for d_label, d_val in target_dates:
        on_d = base_dates == d_val
        assert on_d.any(), f"No rows on {d_label} ({d_val})"
        uids_on_d = base_frame.loc[on_d, "probe_pa_uid"].to_numpy()
        gpks_on_d = base_frame.loc[on_d, "game_pk"].unique().tolist()

        lk_a = copy.deepcopy(lookup)
        for pk in gpks_on_d:
            entry = lk_a.setdefault(pk, {})
            if entry.get("away") is None:
                entry["away"] = 777001
            if entry.get("home") is None:
                entry["home"] = 777002
        df_a = base_frame.copy()
        df_a.loc[on_d, "pitcher_id"] = [
            lk_a[r.game_pk]["away"] if r.is_home else lk_a[r.game_pk]["home"]
            for r in df_a.loc[on_d].itertuples()
        ]
        df_a.loc[on_d, "is_hit"] = 1 - df_a.loc[on_d, "is_hit"]
        out_a = compute_with(branch_compute, df_a, lk_a).set_index("probe_pa_uid")
        pd.testing.assert_series_equal(out_a.loc[uids_on_d, FEATURE], ref_uid_map.loc[uids_on_d], check_exact=True)

        df_b = base_frame.copy()
        df_b.loc[on_d, "pitcher_id"] = 888999
        df_b.loc[on_d, "is_hit"] = 1
        out_b = compute_with(branch_compute, df_b, lk_a).set_index("probe_pa_uid")
        pd.testing.assert_series_equal(out_b.loc[uids_on_d, FEATURE], ref_uid_map.loc[uids_on_d], check_exact=True)

        keep_d = df_b.loc[on_d].iloc[::2].copy()
        dup_d = keep_d.copy()
        dup_d["probe_pa_uid"] = np.arange(10_000_000, 10_000_000 + len(dup_d))
        new_game_d = keep_d.iloc[: min(5, len(keep_d))].copy()
        new_game_pk = 99_999_901
        sample_pk = gpks_on_d[0]
        lk_c = copy.deepcopy(lookup)
        lk_c[new_game_pk] = dict(lookup[sample_pk])
        new_game_d["game_pk"] = new_game_pk
        new_game_d["pitcher_id"] = 888777
        new_game_d["is_hit"] = 1
        new_game_d["probe_pa_uid"] = np.arange(20_000_000, 20_000_000 + len(new_game_d))

        df_c = pd.concat([base_frame.loc[~on_d], keep_d, dup_d, new_game_d], ignore_index=True)
        out_c = compute_with(branch_compute, df_c, lk_c).set_index("probe_pa_uid")
        pd.testing.assert_series_equal(
            out_c.loc[keep_d["probe_pa_uid"], FEATURE],
            ref_uid_map.loc[keep_d["probe_pa_uid"]],
            check_exact=True,
        )
        for r in out_c.loc[new_game_d["probe_pa_uid"]].itertuples():
            exp_v = ref_team_date_map.loc[(r.opp_pitching_team_id, d_val)]
            assert (pd.isna(r.opp_bullpen_hr_30g) and pd.isna(exp_v)) or r.opp_bullpen_hr_30g == exp_v

        after_d = base_dates > d_val
        if after_d.any():
            df_d1 = base_frame.copy()
            df_d1.loc[after_d, "pitcher_id"] = 888999
            df_d1.loc[after_d, "is_hit"] = 1 - df_d1.loc[after_d, "is_hit"]
            after_idx = df_d1.index[after_d]
            df_d1 = df_d1.drop(index=after_idx[::3])
            out_d1 = compute_with(branch_compute, df_d1, lookup)
            out_d1_le = out_d1[out_d1["date"] <= d_val].reset_index(drop=True)
            ref_le = branch_uid_ref[branch_uid_ref["date"] <= d_val].reset_index(drop=True)
            pd.testing.assert_frame_equal(out_d1_le, ref_le, check_exact=True, check_index_type=True, check_column_type=True)

            out_trunc = compute_with(branch_compute, base_frame.loc[~after_d], lookup)
            pd.testing.assert_frame_equal(out_trunc, ref_le, check_exact=True, check_index_type=True, check_column_type=True)
            fut_msg = f"future_mutated_and_truncated_exact_on_all_{len(ref_le.columns)}_cols (rows<={d_val.date()}: {len(ref_le)})"
        else:
            fut_msg = "no_rows_after_last_date"

        print(f"[INVARIANCE] {d_label} ({d_val.date()}, {len(uids_on_d)} PAs): same-day A/B/C exact PASS; {fut_msg}", flush=True)

    del base_2024, branch_2024, pa_24_25, base_24_25, branch_24_25, base_shuf, branch_shuf
    del base_idx, branch_idx, base_dup, branch_dup, branch_str, branch_us, branch_ns
    gc.collect()

    print("\n=== SECTION 4: Full 2017-2025 Scale, Runtime, and Memory ===", flush=True)
    df_2017_2025 = load_all_fn()
    gc.collect()
    print(f"Loaded full pre-2026 history: {len(df_2017_2025):,} PAs, maxrss={get_maxrss_mb():.1f} MB", flush=True)

    tracemalloc.start()
    rss_b0 = get_maxrss_mb()
    t0 = time.perf_counter()
    base_all = compute_with(BASE_COMPUTE, df_2017_2025, lookup)
    t_base_all = time.perf_counter() - t0
    _, peak_base_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_b1 = get_maxrss_mb()

    gc.collect()
    tracemalloc.start()
    rss_n0 = get_maxrss_mb()
    t0 = time.perf_counter()
    branch_all = compute_with(branch_compute, df_2017_2025, lookup)
    t_branch_all = time.perf_counter() - t0
    _, peak_branch_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rss_n1 = get_maxrss_mb()

    other_all = [c for c in base_all.columns if c != FEATURE]
    pd.testing.assert_frame_equal(
        base_all[other_all], branch_all[other_all], check_exact=True, check_index_type=True, check_column_type=True
    )
    ba = base_all[FEATURE].to_numpy(np.float64)
    na = branch_all[FEATURE].to_numpy(np.float64)
    old_na_all = np.isnan(ba)
    new_na_all = np.isnan(na)
    filled_all = int((old_na_all & ~new_na_all).sum())
    lost_all = int((~old_na_all & new_na_all).sum())
    bit_mis_all = int((ba[~old_na_all].view(np.uint64) != na[~old_na_all].view(np.uint64)).sum())
    assert lost_all == 0 and bit_mis_all == 0

    in_2025_all = branch_all["season"] == 2025
    filled_2025_in_all = branch_all[in_2025_all & old_na_all & ~new_na_all]
    td_2025_in_all = filled_2025_in_all.groupby(["opp_pitching_team_id", "date"]).size().reset_index(name="pas")

    print(
        f"[SCALE_2017_2025] pas={len(df_2017_2025):,}, "
        f"base_time={t_base_all:.2f}s (tracemalloc_peak={peak_base_bytes/1e6:.1f}MB, maxrss={rss_b0:.0f}->{rss_b1:.0f}MB), "
        f"branch_time={t_branch_all:.2f}s (tracemalloc_peak={peak_branch_bytes/1e6:.1f}MB, maxrss={rss_n0:.0f}->{rss_n1:.0f}MB), "
        f"old_missing={int(old_na_all.sum()):,}, new_missing={int(new_na_all.sum()):,}, "
        f"filled_total={filled_all:,}, lost={lost_all}, bit_mismatches={bit_mis_all}, "
        f"filled_team_dates_in_2025={len(td_2025_in_all)} ({int(td_2025_in_all['pas'].sum())} PAs)",
        flush=True,
    )

    print("\n=== SECTION 5: Serving Lookups & Team-Date Replay ===", flush=True)
    from bts.model.predict import _build_feature_lookups

    lk_base_25 = _build_feature_lookups(base_2025)
    lk_branch_25 = _build_feature_lookups(branch_2025)
    assert_lookups_equal_except_bullpen(lk_base_25, lk_branch_25, "end_of_2025_2025only")
    bp_diffs_end25 = {
        int(t): (lk_base_25["bullpen"].get(t), lk_branch_25["bullpen"].get(t))
        for t in sorted(set(lk_base_25["bullpen"]) | set(lk_branch_25["bullpen"]))
        if lk_base_25["bullpen"].get(t) != lk_branch_25["bullpen"].get(t)
    }
    print(f"[SERVING_END_2025_ONLY] teams={len(lk_branch_25['bullpen'])}, diffs={bp_diffs_end25}", flush=True)

    lk_base_all = _build_feature_lookups(base_all)
    lk_branch_all = _build_feature_lookups(branch_all)
    assert_lookups_equal_except_bullpen(lk_base_all, lk_branch_all, "end_of_2025_2017_2025")
    bp_diffs_endall = {
        int(t): (lk_base_all["bullpen"].get(t), lk_branch_all["bullpen"].get(t))
        for t in sorted(set(lk_base_all["bullpen"]) | set(lk_branch_all["bullpen"]))
        if lk_base_all["bullpen"].get(t) != lk_branch_all["bullpen"].get(t)
    }
    print(f"[SERVING_END_2017_2025] teams={len(lk_branch_all['bullpen'])}, diffs={bp_diffs_endall}", flush=True)

    if cut_dates is None:
        cut_dates = ["2025-04-02", "2025-04-14", "2025-05-11", "2025-06-12", "2025-07-25", "2025-09-21"]
    for cd in cut_dates:
        ts = pd.Timestamp(cd)
        b_sub = base_2025[base_2025["date"] < ts]
        n_sub = branch_2025[branch_2025["date"] < ts]
        if len(b_sub) > 0:
            lb = _build_feature_lookups(b_sub)
            ln = _build_feature_lookups(n_sub)
            assert_lookups_equal_except_bullpen(lb, ln, f"cut_2025only_{cd}")
            diffs_25 = {
                int(t): (
                    round(lb["bullpen"].get(t), 10) if lb["bullpen"].get(t) is not None else None,
                    round(ln["bullpen"].get(t), 10) if ln["bullpen"].get(t) is not None else None,
                )
                for t in sorted(set(lb["bullpen"]) | set(ln["bullpen"]))
                if lb["bullpen"].get(t) != ln["bullpen"].get(t)
            }
        else:
            diffs_25 = {}
        ba_sub = base_all[base_all["date"] < ts]
        na_sub = branch_all[branch_all["date"] < ts]
        if len(ba_sub) > 0:
            lba = _build_feature_lookups(ba_sub)
            lna = _build_feature_lookups(na_sub)
            assert_lookups_equal_except_bullpen(lba, lna, f"cut_2017_2025_{cd}")
            diffs_all = {
                int(t): (
                    round(lba["bullpen"].get(t), 10) if lba["bullpen"].get(t) is not None else None,
                    round(lna["bullpen"].get(t), 10) if lna["bullpen"].get(t) is not None else None,
                )
                for t in sorted(set(lba["bullpen"]) | set(lna["bullpen"]))
                if lba["bullpen"].get(t) != lna["bullpen"].get(t)
            }
        else:
            diffs_all = {}
        print(f"[SERVING_CUT {cd}] 2025only_diffs={diffs_25}; 2017_2025_diffs={diffs_all}", flush=True)

    for nm, fr in (("base_2025", base_2025), ("branch_2025", branch_2025)):
        nunq = fr.groupby(["opp_pitching_team_id", "date"])[FEATURE].nunique(dropna=False)
        assert (nunq <= 1).all(), f"{nm} has multiple values within (team, date)!"

    td_old = (
        base_2025.dropna(subset=["opp_pitching_team_id"])
        .groupby(["opp_pitching_team_id", "date"], as_index=False)[FEATURE]
        .last()
        .sort_values(["opp_pitching_team_id", "date"])
        .rename(columns={FEATURE: "old_train"})
    )
    td_new = (
        branch_2025.dropna(subset=["opp_pitching_team_id"])
        .groupby(["opp_pitching_team_id", "date"], as_index=False)[FEATURE]
        .last()
        .sort_values(["opp_pitching_team_id", "date"])
        .rename(columns={FEATURE: "new_train"})
    )
    td = td_old.merge(td_new, on=["opp_pitching_team_id", "date"], validate="1:1")
    td["old_lookup"] = td.groupby("opp_pitching_team_id")["old_train"].transform(lambda s: s.ffill().shift(1))
    td["new_lookup"] = td.groupby("opp_pitching_team_id")["new_train"].transform(lambda s: s.ffill().shift(1))

    pairs = [
        ("Old training vs old lookup", "old_train", "old_lookup"),
        ("New training vs old lookup", "new_train", "old_lookup"),
        ("New training vs new lookup", "new_train", "new_lookup"),
        ("Old lookup vs new lookup", "old_lookup", "new_lookup"),
    ]
    print(f"\n[REPLAY_SUMMARY_2025_ONLY] total_team_dates={len(td)}", flush=True)
    for title, c1, c2 in pairs:
        s1, s2 = td[c1], td[c2]
        f1, f2 = s1.notna(), s2.notna()
        both_f = f1 & f2
        diff = (s1[both_f] - s2[both_f]).abs()
        unequal = int((diff > 0).sum())
        mean_abs = float(diff.mean()) if len(diff) else 0.0
        max_abs = float(diff.max()) if len(diff) else 0.0
        print(
            f"  {title:28s}: both_finite={int(both_f.sum())}, unequal={unequal}, "
            f"left_only={int((f1 & ~f2).sum())}, right_only={int((~f1 & f2).sum())}, "
            f"both_missing={int((~f1 & ~f2).sum())}, mean_abs={mean_abs:.8f}, max_abs={max_abs:.8f}",
            flush=True,
        )

    td["next_date"] = td.groupby("opp_pitching_team_id")["date"].shift(-1)
    td["next_old_lookup"] = td.groupby("opp_pitching_team_id")["old_lookup"].shift(-1)
    td["next_new_lookup"] = td.groupby("opp_pitching_team_id")["new_lookup"].shift(-1)
    td["next_new_train"] = td.groupby("opp_pitching_team_id")["new_train"].shift(-1)

    filled_rows = td[td["old_train"].isna() & td["new_train"].notna()].sort_values(["date", "opp_pitching_team_id"])
    print(f"\n[FOLLOWING_GAMES_30] count={len(filled_rows)}", flush=True)
    moved_count = 0
    deltas = []
    for r in filled_rows.itertuples():
        if pd.isna(r.next_date):
            print(f"  tid={int(r.opp_pitching_team_id):3d} | filled={r.date.strftime('%Y-%m-%d')} | next=NONE (terminal!)", flush=True)
            continue
        lk_b = _build_feature_lookups(base_2025[base_2025["date"] < r.next_date])["bullpen"].get(r.opp_pitching_team_id)
        lk_n = _build_feature_lookups(branch_2025[branch_2025["date"] < r.next_date])["bullpen"].get(r.opp_pitching_team_id)
        assert (pd.isna(lk_b) and pd.isna(r.next_old_lookup)) or lk_b == r.next_old_lookup
        assert lk_n == r.next_new_lookup
        assert np.isclose(r.next_new_lookup, r.next_new_train, atol=1e-15, rtol=0)
        delta = (r.next_new_lookup - r.next_old_lookup) if pd.notna(r.next_old_lookup) else float("nan")
        if pd.isna(delta) or delta != 0:
            moved_count += 1
        if pd.notna(delta):
            deltas.append(delta)
        old_s = f"{r.next_old_lookup:.10f}" if pd.notna(r.next_old_lookup) else "NaN"
        del_s = f"{delta:+.10f}" if pd.notna(delta) else "NaN"
        print(
            f"  tid={int(r.opp_pitching_team_id):3d} | filled={r.date.strftime('%Y-%m-%d')} | "
            f"next={r.next_date.strftime('%Y-%m-%d')} | old={old_s} | "
            f"new={r.next_new_lookup:.10f} | train={r.next_new_train:.10f} | delta={del_s}",
            flush=True,
        )
    if deltas:
        print(
            f"[FOLLOWING_SUMMARY] moved={moved_count}/{len(filled_rows)}, "
            f"min_delta={min(deltas):+.8f}, max_delta={max(deltas):+.8f}",
            flush=True,
        )

    td_all_old = (
        base_all.dropna(subset=["opp_pitching_team_id"])
        .groupby(["opp_pitching_team_id", "date"], as_index=False)[FEATURE]
        .last()
        .sort_values(["opp_pitching_team_id", "date"])
        .rename(columns={FEATURE: "old_train"})
    )
    td_all_new = (
        branch_all.dropna(subset=["opp_pitching_team_id"])
        .groupby(["opp_pitching_team_id", "date"], as_index=False)[FEATURE]
        .last()
        .sort_values(["opp_pitching_team_id", "date"])
        .rename(columns={FEATURE: "new_train"})
    )
    tda = td_all_old.merge(td_all_new, on=["opp_pitching_team_id", "date"], validate="1:1")
    tda["old_lookup"] = tda.groupby("opp_pitching_team_id")["old_train"].transform(lambda s: s.ffill().shift(1))
    tda["new_lookup"] = tda.groupby("opp_pitching_team_id")["new_train"].transform(lambda s: s.ffill().shift(1))
    tda25 = tda[tda["date"].dt.year == 2025]
    print(f"\n[REPLAY_SUMMARY_2017_2025_FOR_2025] total_team_dates={len(tda25)}", flush=True)
    for title, c1, c2 in pairs:
        s1, s2 = tda25[c1], tda25[c2]
        f1, f2 = s1.notna(), s2.notna()
        both_f = f1 & f2
        diff = (s1[both_f] - s2[both_f]).abs()
        mean_abs = float(diff.mean()) if len(diff) else 0.0
        max_abs = float(diff.max()) if len(diff) else 0.0
        print(
            f"  {title:28s}: both_finite={int(both_f.sum())}, unequal={int((diff > 0).sum())}, "
            f"left_only={int((f1 & ~f2).sum())}, right_only={int((~f1 & f2).sum())}, "
            f"both_missing={int((~f1 & ~f2).sum())}, mean_abs={mean_abs:.8f}, max_abs={max_abs:.8f}",
            flush=True,
        )

    print("\nALL REAL-DATA PROBES PASSED.", flush=True)


def run_selftest():
    """In-memory synthetic selftest so every line of run_verification is validated inside sandbox."""
    rows24, rows25, lk = [], [], {}
    for s, target_list, base_dt in ((2024, rows24, "2024-09-01"), (2025, rows25, "2025-04-01")):
        b = pd.Timestamp(base_dt)
        for d in range(15):
            dt = str((b + pd.Timedelta(days=d)).date())
            gpk = s * 100 + d
            lk[gpk] = {"away": 21, "home": 31, "away_tid": 113, "home_tid": 135}
            p_away = 21 if (s == 2025 and d == 11) else 22
            target_list.append({
                "date": dt, "season": s, "game_pk": gpk, "batter_id": 101, "pitcher_id": p_away,
                "pitch_hand": "R", "venue_id": 1, "is_home": True, "is_hit": d % 2,
                "pitch_calls": ["X"], "pitch_types": ["FF"], "pitch_speeds": [93.0],
                "pitch_spin_rates": [2300.0], "pitch_extensions": [6.0],
                "pitch_break_vertical": [15.0], "pitch_break_horizontal": [5.0],
                "pitch_px": [0.0], "pitch_pz": [2.5], "sz_top": 3.5, "sz_bottom": 1.5,
                "final_count_balls": 1, "final_count_strikes": 1, "launch_speed": 91.0,
                "launch_angle": 12.0, "weather_temp": 72.0, "weather_wind_dir": "Out To CF",
                "weather_wind_speed": 8.0, "roof_type": "Open", "hp_umpire_id": 401, "hardness": "hard",
            })
            target_list.append({**target_list[-1], "batter_id": 102, "pitcher_id": 32, "is_home": False, "is_hit": (d + 1) % 2})
            if s == 2025 and d == 8:
                gpk2 = 9999
                lk[gpk2] = {"away": 21, "home": 31, "away_tid": 113, "home_tid": 135}
                target_list.append({**target_list[-2], "game_pk": gpk2})
    d24, d25 = pd.DataFrame(rows24), pd.DataFrame(rows25)
    run_verification(
        d24, d25, lk,
        load_all_fn=lambda: pd.concat([d24, d25], ignore_index=True),
        gap_sample_date="2025-04-12",
        ord_sample_date="2025-04-06",
        cut_dates=["2025-04-13"],
    )


def main():
    if "--selftest" in sys.argv:
        run_selftest()
        return
    lookup, sha = load_frozen_lookup()
    print(f"Loaded frozen lookup: {len(lookup)} games, sha256={sha}", flush=True)
    pa_2024 = load_pa_year(2024)
    pa_2025 = load_pa_year(2025)
    run_verification(
        pa_2024,
        pa_2025,
        lookup,
        load_all_fn=lambda: pd.concat([load_pa_year(y) for y in range(2017, 2026)], ignore_index=True),
    )


if __name__ == "__main__":
    main()
