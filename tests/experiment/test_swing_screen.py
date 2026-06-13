"""Tests for the Stage-1 screen arm registry (swing campaign)."""
import numpy as np
import pandas as pd

from bts.experiment.swing_screen import (
    ARMS,
    FAMILY_OF,
    build_arm_frame,
)


def _pa_frame():
    # PA rows already carrying rolling swing features (as attach would produce)
    return pd.DataFrame({
        "batter_id": [1, 2], "pitcher_id": [9, 9],
        "date": pd.to_datetime(["2024-05-01", "2024-05-01"]),
        "season": [2024, 2024], "is_hit": [1, 0],
        "batter_miss_dist_7g": [2.0, 3.0],
        "batter_miss_dist_60g": [2.5, 2.5],
        "batter_intercept_y_7g": [31.0, 30.0],
        "batter_intercept_y_60g": [30.0, 30.0],
        "batter_swing_len_7g": [7.1, 7.0],
        "batter_swing_len_60g": [7.0, 7.0],
        "batter_whiff_high_share_30g": [0.6, 0.4],
        "pitcher_whiff_high_share_30g": [0.7, 0.7],
    })


def test_registry_arm_names_unique_and_families_mapped():
    assert len(ARMS) == len(set(ARMS))
    assert "baseline" in ARMS
    for arm in ARMS:
        assert arm in FAMILY_OF
    assert {"P", "B", "T", "S", "M", "omnibus", "control", "baseline"} >= set(FAMILY_OF.values())


def test_derived_drift_and_interaction_features():
    pa = _pa_frame()
    frame, cols = build_arm_frame("t_intercept_drift", pa)
    assert "t_intercept_drift" in cols  # plus availability flags (amendment #2)
    assert abs(frame["t_intercept_drift"].iloc[0] - 1.0) < 1e-9  # 31-30

    frame, cols = build_arm_frame("m_high_alignment", pa)
    assert abs(frame["m_high_alignment"].iloc[0] - 0.42) < 1e-9  # 0.6*0.7


def test_baseline_flags_match_candidate_flags():
    # the SAME flag set must appear in baseline and candidates (paired fairness)
    pa = _pa_frame()
    _, base_cols = build_arm_frame("baseline", pa)
    _, cand_cols = build_arm_frame("t_intercept_drift", pa)
    assert set(base_cols) == {c for c in cand_cols if c.startswith("has_")}


def test_permuted_control_preserves_values_but_breaks_dates():
    pa = pd.DataFrame({
        "batter_id": [1] * 6, "pitcher_id": [9] * 6,
        "date": pd.to_datetime([f"2024-05-{d:02d}" for d in range(1, 7)]),
        "season": 2024, "is_hit": 1,
        "batter_miss_dist_30g": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
    })
    # registry permutes within entity with a fixed seed
    frame, cols = build_arm_frame("ctl_permuted", pa, permute_seed=7)
    # permuted copies carry the VARIANT name (b_miss_30g <- batter_miss_dist_30g)
    col = [c for c in cols if c == "perm_b_miss_30g"]
    assert col, "permuted control must include permuted copies of omnibus features"
    vals = sorted(frame[col[0]].tolist())
    assert vals == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]      # marginal preserved
    assert frame[col[0]].tolist() != pa["batter_miss_dist_30g"].tolist()  # order broken


def test_placebo_arm_removed():
    # availability flags moved into the baseline (amendment #2); the old
    # ctl_placebo arm is gone
    import pytest
    with pytest.raises(KeyError):
        build_arm_frame("ctl_placebo", _pa_frame())


def test_run_screen_arm_end_to_end(tmp_path):
    from bts.experiment.swing_screen import run_screen_arm

    rng = np.random.default_rng(3)
    rows = []
    for season in (2023, 2024):
        for i in range(30):
            for b in range(18):
                hit_p = 0.55 + 0.2 * (b % 2)        # feature-correlated outcome
                rows.append({
                    "batter_id": b, "pitcher_id": 100 + (i % 3),
                    "game_pk": season * 1000 + i, "lineup_position": (b % 9) + 1,
                    "is_home": b < 9,
                    "date": pd.Timestamp(f"{season}-05-01") + pd.Timedelta(days=i),
                    "season": season,
                    "is_hit": int(rng.random() < hit_p),
                    "weather_temp": 70.0,
                    "batter_miss_dist_30g": 3.0 - 0.5 * (b % 2),
                })
    pa = pd.DataFrame(rows)

    res = run_screen_arm(
        arm="b_miss_30g", pa=pa, train_seasons=(2023,), screen_season=2024,
        seed=42, base_cols=["weather_temp"],
        lgb_overrides={"n_estimators": 10, "min_child_samples": 5},
        out_dir=tmp_path,
    )
    assert (tmp_path / "b_miss_30g_seed42.json").exists()
    assert res["arm"] == "b_miss_30g"
    assert res["n_days"] > 0
    assert "ndcg_mean" in res and "top1_hit" in res and "auc" in res


def test_run_screen_arm_date_split(tmp_path):
    from bts.experiment.swing_screen import run_screen_arm

    rng = np.random.default_rng(3)
    rows = []
    for season in (2023, 2024):
        for i in range(60):
            for b in range(18):
                rows.append({
                    "batter_id": b, "pitcher_id": 100 + (i % 3),
                    "game_pk": season * 1000 + i, "lineup_position": (b % 9) + 1,
                    "is_home": b < 9,
                    "date": pd.Timestamp(f"{season}-04-01") + pd.Timedelta(days=i),
                    "season": season,
                    "is_hit": int(rng.random() < 0.6),
                    "weather_temp": 70.0,
                    "batter_miss_dist_30g": 3.0 - 0.5 * (b % 2),
                })
    pa = pd.DataFrame(rows)

    res = run_screen_arm(
        arm="b_miss_30g", pa=pa, train_seasons=(2023,), screen_season=2024,
        seed=42, base_cols=["weather_temp"],
        lgb_overrides={"n_estimators": 10, "min_child_samples": 5},
        out_dir=tmp_path,
        train_extra_through="2024-04-30", screen_start="2024-05-01",
    )
    # screen days only from screen_start onward (Apr 2024 went to training)
    assert all(d["date"] >= "2024-05-01" for d in res["per_day"])
    assert res["n_days"] > 0
    assert res["train_extra_through"] == "2024-04-30"


def test_baseline_includes_availability_flags():
    # Amendment #2: availability flags live in the BASELINE so candidate
    # deltas measure value beyond coverage information.
    pa = _pa_frame()
    frame, cols = build_arm_frame("baseline", pa)
    assert cols, "baseline must carry availability flags"
    assert all(c.startswith("has_") for c in cols)


def test_candidate_arms_include_flags_plus_features():
    pa = _pa_frame()
    frame, cols = build_arm_frame("m_high_alignment", pa)
    assert "m_high_alignment" in cols
    assert any(c.startswith("has_") for c in cols)


def test_mask_only_control_preserves_nan_pattern_destroys_values():
    pa = _pa_frame()
    pa.loc[0, "batter_miss_dist_7g"] = np.nan
    frame, cols = build_arm_frame("ctl_mask_only", pa)
    mcol = [c for c in cols if c == "mask_b_miss_7g"]
    assert mcol
    assert pd.isna(frame[mcol[0]].iloc[0])          # NaN preserved
    assert frame[mcol[0]].iloc[1] == 1.0            # value destroyed -> constant


def test_sentinel_arms_require_attached_columns():
    pa = _pa_frame()
    import pytest
    with pytest.raises(ValueError):
        build_arm_frame("ctl_sentinel_gross", pa)
    with pytest.raises(ValueError):
        build_arm_frame("ctl_sentinel_m3", pa)
    pa["GROSS_same_day_whiffs"] = 1.0
    pa["M3LEAK_batter_miss_dist_30g"] = 2.5
    f1, c1 = build_arm_frame("ctl_sentinel_gross", pa)
    f2, c2 = build_arm_frame("ctl_sentinel_m3", pa)
    assert "GROSS_same_day_whiffs" in c1
    assert "M3LEAK_batter_miss_dist_30g" in c2


def _residual_synth(seed=0):
    """Synthetic covered-era frame: outcome depends on a hidden 'skill' that
    production features capture partially, plus a same-day leak column."""
    import numpy as np
    rng = np.random.default_rng(seed)
    rows = []
    # 2 training-relevant seasons of "full history" + covered era
    for season, d0, n_days in [(2022, "2022-04-01", 60),   # pre-coverage (prior only)
                               (2023, "2023-08-01", 50),   # covered warm
                               (2024, "2024-04-01", 170),  # covered: H1 train + H2 score
                               ]:
        for i in range(n_days):
            date = pd.Timestamp(d0) + pd.Timedelta(days=i)
            for b in range(16):
                skill = (b % 4) / 4.0                      # latent skill
                hit_p = 0.45 + 0.25 * skill
                hit = int(rng.random() < hit_p)
                covered = season >= 2023
                rows.append({
                    "batter_id": b, "pitcher_id": 100 + (i % 3),
                    "game_pk": season * 1000 + i, "lineup_position": (b % 9) + 1,
                    "is_home": b < 8,
                    "date": date, "season": season, "is_hit": hit,
                    "prod_feat": skill + rng.normal(0, 0.3),   # production feature (partial skill)
                    # candidate swing feature: extra skill info, covered only
                    "cand_feat": (skill + rng.normal(0, 0.2)) if covered else np.nan,
                    # same-day leak: equals the outcome-correlated signal of THIS day
                    "leak_same_day": float(hit) if covered else 0.0,
                    "swing_cov60": 0.95 if covered else 0.0,
                })
    return pd.DataFrame(rows)


def test_residual_prior_is_oof_on_covered_train(tmp_path):
    from bts.experiment.swing_screen import build_prod_prior_oof
    pa = _residual_synth()
    full = pa["season"].isin([2022, 2023, 2024]) & (pa["date"] < "2024-07-01")
    cov = (pa["season"] >= 2023) & (pa["date"] < "2024-07-01")
    ev = (pa["season"] == 2024) & (pa["date"] >= "2024-07-01")
    prior = build_prod_prior_oof(pa, ["prod_feat"], full, cov, ev, seed=42,
                                 lgb_overrides={"n_estimators": 15, "min_child_samples": 5})
    assert prior[cov].notna().all()    # OOF prior for every covered-train row
    assert prior[ev].notna().all()     # final-model prior for every eval row
    assert prior[~(cov | ev)].isna().all()


def test_residual_gross_sentinel_beats_baseline_noise_does_not(tmp_path):
    from bts.experiment.swing_screen import build_prod_prior_oof, run_residual_arm
    import numpy as np
    pa = _residual_synth()
    full = (pa["date"] < "2024-07-01")
    cov = (pa["season"] >= 2023) & (pa["date"] < "2024-07-01")
    ev = (pa["season"] == 2024) & (pa["date"] >= "2024-07-01")
    pa = pa.copy()
    pa["prod_prior"] = build_prod_prior_oof(pa, ["prod_feat"], full, cov, ev, seed=42,
                                            lgb_overrides={"n_estimators": 15, "min_child_samples": 5})
    ov = {"n_estimators": 20, "min_child_samples": 5}
    base = run_residual_arm(pa, [], cov, ev, seed=42, lgb_overrides=ov)
    leak = run_residual_arm(pa, ["leak_same_day"], cov, ev, seed=42, lgb_overrides=ov)
    noise = run_residual_arm(pa, ["noise_col"], cov, ev, seed=42, lgb_overrides=ov,
                             extra_cols={"noise_col": np.random.default_rng(1).random(len(pa))})
    # the same-day leak must lift rank-AUC over baseline, and must clearly
    # beat a pure-noise column (the harness-sanity claim; robust to the
    # single-seed jitter of this tiny synthetic — the real screen uses 30
    # seeds + week-block permutation)
    assert leak["auc_mean"] > base["auc_mean"] + 0.02
    assert leak["auc_mean"] > noise["auc_mean"] + 0.02
