"""Blend walk-forward backtest that saves daily prediction profiles.

Adapts the existing walk_forward_evaluate to use the 12-model blend
(same BLEND_CONFIGS from predict.py) and save top-10 ranked predictions
per day to parquet for strategy simulation.

Supports multiple model engines and objectives via 3-tuple blend configs:
  (name, cols)                              — default LightGBM classifier
  (name, cols, {"objective": "lambdarank"}) — LightGBM ranker
  (name, cols, {"objective": "rank_xendcg"}) — LightGBM XE-NDCG ranker
  (name, cols, {"engine": "catboost"})      — CatBoost classifier
  (name, cols, {"vrex_beta": 10.0})         — V-REx season reweighting
"""

import sys
from pathlib import Path

PROFILE_COLUMNS = ["date", "rank", "batter_id", "p_game_hit", "actual_hit", "n_pas"]

# Objectives that require batter-day grouping (ranker training)
RANKER_OBJECTIVES = {"lambdarank", "rank_xendcg"}

# Regression objectives (quantile, regression, etc.)
REGRESSION_OBJECTIVES = {"quantile", "regression", "regression_l1", "regression_l2"}


def _is_ranker(extra_params: dict) -> bool:
    return extra_params.get("objective") in RANKER_OBJECTIVES


def _is_regressor(extra_params: dict) -> bool:
    return extra_params.get("objective") in REGRESSION_OBJECTIVES


def _is_catboost(extra_params: dict) -> bool:
    return extra_params.get("engine") == "catboost"


def _aggregate_to_batter_day(
    df: "pd.DataFrame",
    feature_cols: list[str],
) -> "pd.DataFrame":
    """Aggregate PA-level data to batter-day level for ranker training.

    Features are taken from the first PA of each batter-day (date-level features
    are constant across PAs for the same batter-day). Label is hit (max over PAs).
    Returns DataFrame sorted by date with one row per (batter_id, date).
    """
    import pandas as pd

    agg_dict: dict = {col: "first" for col in feature_cols if col in df.columns}
    agg_dict["is_hit"] = "max"
    bd = df.groupby(["batter_id", "date"], as_index=False).agg(agg_dict)
    return bd.sort_values("date")


def _train_ranker(
    available: "pd.DataFrame",
    cols: list[str],
    merged_params: dict,
):
    """Train a LightGBM ranker on batter-day data with daily groups."""
    import lightgbm as lgb

    bd = _aggregate_to_batter_day(available, cols)
    train_X = bd[cols]
    train_y = bd["is_hit"]
    mask = train_X.notna().any(axis=1)
    bd_clean = bd[mask].sort_values("date")

    # Compute group sizes per game-day (must be in same order as data)
    groups = bd_clean.groupby("date").size().tolist()

    # LGBMRanker has different params from LGBMClassifier
    ranker_params = {k: v for k, v in merged_params.items() if k not in ("scale_pos_weight",)}
    model = lgb.LGBMRanker(**ranker_params, random_state=42)
    model.fit(bd_clean[cols], bd_clean["is_hit"], group=groups)
    return model


def _predict_ranker(
    model,
    day_data: "pd.DataFrame",
    cols: list[str],
):
    """Predict with a ranker. Returns per-PA scores (broadcast from batter-day)."""
    import numpy as np
    import pandas as pd

    # Aggregate test day to batter-day, predict, then broadcast back
    bd = _aggregate_to_batter_day(day_data, cols)
    pred_X = bd[cols]
    valid = pred_X.notna().any(axis=1)

    raw_scores = pd.Series(np.nan, index=bd.index)
    if valid.any():
        raw_scores[valid] = model.predict(pred_X[valid])

    # Min-max scale per day to (0, 1) so it can blend with probabilities
    if raw_scores.notna().any():
        s_min = raw_scores.min()
        s_max = raw_scores.max()
        if s_max > s_min:
            scaled = (raw_scores - s_min) / (s_max - s_min)
        else:
            scaled = raw_scores * 0 + 0.5
    else:
        scaled = raw_scores

    # Broadcast batter-day scores back to PA-level
    bd_scores = bd[["batter_id"]].copy()
    bd_scores["score"] = scaled.values

    pa_scores = day_data[["batter_id"]].merge(bd_scores, on="batter_id", how="left")
    pa_scores.index = day_data.index
    return pa_scores["score"]


def _train_catboost(
    available: "pd.DataFrame",
    cols: list[str],
    merged_params: dict,
):
    """Train a CatBoost classifier with has_time=True for temporal ordering."""
    try:
        from catboost import CatBoostClassifier
    except ImportError:
        raise RuntimeError(
            "CatBoost not installed. Run: uv pip install catboost"
        )

    train_X = available[cols].copy()
    train_y = available["is_hit"]
    mask = train_X.notna().any(axis=1)

    # Map LightGBM params to CatBoost equivalents
    cb_params = {
        "iterations": merged_params.get("n_estimators", 200),
        "depth": min(merged_params.get("max_depth", 6), 8),
        "learning_rate": merged_params.get("learning_rate", 0.05),
        "verbose": False,
        "random_seed": 42,
        "has_time": merged_params.get("has_time", True),
    }
    # Fill NaN since CatBoost needs explicit handling
    train_X_clean = train_X[mask].fillna(-999.0)
    model = CatBoostClassifier(**cb_params)
    model.fit(train_X_clean, train_y[mask])
    return model


def _predict_catboost(model, day_data, cols):
    """Predict with CatBoost classifier."""
    import numpy as np
    import pandas as pd

    pred_X = day_data[cols].fillna(-999.0)
    probs = pd.Series(np.nan, index=day_data.index)
    valid = day_data[cols].notna().any(axis=1)
    if valid.any():
        probs[valid] = model.predict_proba(pred_X[valid])[:, 1]
    return probs


def _train_vrex_lgbm(
    available: "pd.DataFrame",
    cols: list[str],
    merged_params: dict,
):
    """Train LightGBM with V-REx-style iterative season reweighting.

    Penalizes cross-season loss variance by upweighting samples from
    seasons with higher-than-average loss. Iterates n_rounds times.
    """
    import lightgbm as lgb
    import numpy as np

    beta = merged_params.pop("vrex_beta", 10.0)
    n_rounds = merged_params.pop("vrex_rounds", 3)

    train_X = available[cols]
    train_y = available["is_hit"]
    mask = train_X.notna().any(axis=1)
    X = train_X[mask]
    y = train_y[mask]
    seasons = available.loc[mask, "season"].values

    # Initialize uniform weights
    weights = np.ones(len(y)) / len(y) * len(y)  # mean 1

    # Strip vrex-specific params before passing to LightGBM
    lgb_only_params = {k: v for k, v in merged_params.items() if not k.startswith("vrex_")}

    model = None
    for r in range(n_rounds):
        model = lgb.LGBMClassifier(**lgb_only_params, random_state=42)
        model.fit(X, y, sample_weight=weights)

        # Compute per-season loss
        preds = model.predict_proba(X)[:, 1]
        eps = 1e-7
        per_sample_loss = -(y * np.log(preds + eps) + (1 - y) * np.log(1 - preds + eps))
        season_losses: dict = {}
        for s in np.unique(seasons):
            sel = seasons == s
            season_losses[s] = float(per_sample_loss[sel].mean())

        mean_loss = float(np.mean(list(season_losses.values())))

        # Reweight: upweight samples from seasons with above-average loss
        new_weights = np.ones(len(y))
        for s, sloss in season_losses.items():
            sel = seasons == s
            new_weights[sel] = max(0.1, 1.0 + beta * (sloss - mean_loss))
        weights = new_weights * (len(y) / new_weights.sum())  # renormalize to mean 1

    return model


def _train_lgbm_classifier(
    available: "pd.DataFrame",
    cols: list[str],
    merged_params: dict,
):
    """Default training path: LightGBM classifier."""
    import lightgbm as lgb

    train_X = available[cols]
    train_y = available["is_hit"]
    mask = train_X.notna().any(axis=1)

    # Strip non-LightGBM params
    lgb_only_params = {
        k: v for k, v in merged_params.items()
        if not k.startswith("vrex_") and k != "engine" and k != "has_time"
    }
    model = lgb.LGBMClassifier(**lgb_only_params, random_state=42)
    model.fit(train_X[mask], train_y[mask])
    return model


def _predict_lgbm_classifier(model, day_data, cols):
    """Default prediction path: LightGBM classifier predict_proba."""
    import numpy as np
    import pandas as pd

    pred_X = day_data[cols]
    valid = pred_X.notna().any(axis=1)
    probs = pd.Series(np.nan, index=day_data.index)
    if valid.any():
        probs[valid] = model.predict_proba(pred_X[valid])[:, 1]
    return probs


def _train_lgbm_regressor(
    available: "pd.DataFrame",
    cols: list[str],
    merged_params: dict,
):
    """Train LightGBM regressor (e.g., quantile regression for q10 skip signal)."""
    import lightgbm as lgb

    train_X = available[cols]
    train_y = available["is_hit"].astype(float)
    mask = train_X.notna().any(axis=1)

    # LGBMRegressor accepts objective="quantile", "regression", etc.
    # alpha controls the quantile level
    lgb_only_params = {
        k: v for k, v in merged_params.items()
        if not k.startswith("vrex_") and k != "engine" and k != "has_time"
    }
    model = lgb.LGBMRegressor(**lgb_only_params, random_state=42)
    model.fit(train_X[mask], train_y[mask])
    return model


def _predict_lgbm_regressor(model, day_data, cols):
    """Predict with LightGBM regressor (returns continuous values)."""
    import numpy as np
    import pandas as pd

    pred_X = day_data[cols]
    valid = pred_X.notna().any(axis=1)
    preds = pd.Series(np.nan, index=day_data.index)
    if valid.any():
        preds[valid] = model.predict(pred_X[valid])
    return preds


def blend_walk_forward(
    df: "pd.DataFrame",
    test_season: int,
    retrain_every: int = 7,
    top_n: int = 10,
    blend_configs: list | None = None,
    lgb_params: dict | None = None,
    capture_per_model: bool = False,
) -> "pd.DataFrame":
    """Run blend walk-forward evaluation and return daily profiles.

    For each game day in the test season:
    1. Train all N blend models on data before that day (retrained periodically)
    2. Predict P(hit|PA) with each model, average for blend ranking
    3. Aggregate to game-level P(>=1 hit) per batter
    4. Save top-N batters with blend p_game_hit and actual_hit

    Args:
        df: Feature-enriched PA DataFrame.
        test_season: Season to evaluate on.
        retrain_every: Retrain models every N days.
        top_n: Number of top-ranked batters to save per day.
        blend_configs: List of (name, cols) or (name, cols, extra_params) tuples.
            Defaults to BLEND_CONFIGS. 3-tuple allows per-model objective overrides.
        lgb_params: LightGBM training parameters. Defaults to LGB_PARAMS.
        capture_per_model: If True, also captures per-model PA-level predictions
            for use by FWLS / Hedge experiments. Adds columns like 'm_<name>'.

    Returns DataFrame with PROFILE_COLUMNS (plus per-model columns if requested).
    """
    import numpy as np
    import pandas as pd
    from bts.features.compute import FEATURE_COLS, STATCAST_COLS, TRAIN_START_YEAR
    from bts.model.predict import BLEND_CONFIGS, LGB_PARAMS

    if blend_configs is None:
        blend_configs = BLEND_CONFIGS
    if lgb_params is None:
        lgb_params = LGB_PARAMS

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    season_data = df[df["season"] == test_season]
    test_start = season_data["date"].min()
    train_pool = df[(df["date"] < test_start) & (df["season"] >= TRAIN_START_YEAR)].copy()
    test_data = season_data.copy()
    test_dates = sorted(test_data["date"].unique())

    print(f"Blend walk-forward: {len(test_dates)} test days, "
          f"train pool: {len(train_pool):,} PAs, "
          f"{len(blend_configs)} models", file=sys.stderr)

    all_profiles = []
    blend = None  # {name: (model, cols, predict_fn)}
    side_channel_names: set = set()  # models excluded from averaging (e.g., quantile)
    per_model_capture = []  # daily per-model predictions for capture mode

    for i, day in enumerate(test_dates):
        day_data = test_data[test_data["date"] == day].copy()

        # Retrain periodically
        if blend is None or (i % retrain_every == 0):
            available = pd.concat([train_pool, test_data[test_data["date"] < day]])

            blend = {}
            for config in blend_configs:
                if len(config) == 2:
                    name, cols = config
                    extra_params = {}
                else:
                    name, cols, extra_params = config

                merged_params = {**lgb_params, **extra_params}

                if _is_ranker(extra_params):
                    model = _train_ranker(available, cols, merged_params)
                    predict_fn = _predict_ranker
                elif _is_regressor(extra_params):
                    model = _train_lgbm_regressor(available, cols, merged_params)
                    predict_fn = _predict_lgbm_regressor
                    side_channel_names.add(name)  # exclude from averaging
                elif _is_catboost(extra_params):
                    model = _train_catboost(available, cols, merged_params)
                    predict_fn = _predict_catboost
                elif "vrex_beta" in extra_params:
                    model = _train_vrex_lgbm(available, cols, dict(merged_params))
                    predict_fn = _predict_lgbm_classifier
                else:
                    model = _train_lgbm_classifier(available, cols, merged_params)
                    predict_fn = _predict_lgbm_classifier

                blend[name] = (model, cols, predict_fn)

            if (i + 1) % 30 == 0 or i == 0:
                print(f"  Day {i+1}/{len(test_dates)} ({pd.Timestamp(day).date()}) "
                      f"— retrained on {len(available):,} PAs", file=sys.stderr)

        # Predict with all blend models
        blend_pa_scores = {}
        for name, (model, cols, predict_fn) in blend.items():
            try:
                blend_pa_scores[name] = predict_fn(model, day_data, cols)
            except Exception as e:
                print(f"  ! {name} predict failed on {day}: {e}", file=sys.stderr)
                blend_pa_scores[name] = pd.Series(np.nan, index=day_data.index)

        # Average PA-level predictions across models (excluding side-channel models)
        avg_scores = {
            name: scores for name, scores in blend_pa_scores.items()
            if name not in side_channel_names
        }
        pa_blend = pd.DataFrame(avg_scores).mean(axis=1)
        day_data["p_hit_blend"] = pa_blend

        # Optional: capture per-model scores
        if capture_per_model:
            for name, scores in blend_pa_scores.items():
                day_data[f"m_{name}"] = scores

        # Aggregate to game level: P(>=1 hit) = 1 - prod(1 - P(hit|PA))
        agg_dict = {
            "p_game_hit": ("p_hit_blend", lambda x: 1 - np.prod(1 - x.values)),
            "actual_hit": ("is_hit", "max"),
            "n_pas": ("is_hit", "count"),
        }
        if capture_per_model:
            for name in blend.keys():
                agg_dict[f"m_{name}"] = (f"m_{name}", "mean")

        game_preds = day_data.groupby(["batter_id", "game_pk"]).agg(**agg_dict).reset_index()

        # Rank and take top N
        game_preds = game_preds.nlargest(top_n, "p_game_hit").reset_index(drop=True)
        game_preds["rank"] = range(1, len(game_preds) + 1)
        game_preds["date"] = pd.Timestamp(day).date()

        cols_to_keep = list(PROFILE_COLUMNS)
        if capture_per_model:
            cols_to_keep += [f"m_{name}" for name in blend.keys()]
        all_profiles.append(game_preds[cols_to_keep])

    result = pd.concat(all_profiles, ignore_index=True)
    print(f"  Done: {len(result)} profile rows ({len(test_dates)} days × top-{top_n})", file=sys.stderr)
    return result


def save_profiles(df: "pd.DataFrame", season: int, output_dir: "Path") -> "Path":
    """Save daily profiles to parquet."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"backtest_{season}.parquet"
    df.to_parquet(path, index=False)
    print(f"  Saved {path} ({len(df)} rows)", file=sys.stderr)
    return path


def run_backtest(
    data_dir: str = "data/processed",
    output_dir: str = "data/simulation",
    seasons: list[int] | None = None,
    retrain_every: int = 7,
) -> None:
    """Run blend backtest for specified seasons and save profiles.

    Loads all PA parquets, computes features once, then runs blend
    walk-forward for each test season.
    """
    import pandas as pd
    from bts.features.compute import compute_all_features

    if seasons is None:
        seasons = [2021, 2022, 2023, 2024, 2025]

    proc = Path(data_dir)
    out = Path(output_dir)

    # Load all data and compute features once
    print("Loading PA data...", file=sys.stderr)
    dfs = []
    for parquet in sorted(proc.glob("pa_*.parquet")):
        dfs.append(pd.read_parquet(parquet))
    if not dfs:
        raise RuntimeError("No parquet files found. Run 'bts data build' first.")

    df = pd.concat(dfs, ignore_index=True)
    print(f"Computing features on {len(df):,} PAs...", file=sys.stderr)
    df = compute_all_features(df)

    for season in seasons:
        print(f"\n{'='*60}", file=sys.stderr)
        print(f"Season {season}", file=sys.stderr)
        print(f"{'='*60}", file=sys.stderr)
        profiles_df = blend_walk_forward(df, season, retrain_every=retrain_every)
        save_profiles(profiles_df, season, out)
