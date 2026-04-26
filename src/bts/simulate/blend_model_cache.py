"""Per-day blend model cache using native library formats.

LightGBM Booster: booster.save_model(path) / lgb.Booster(model_file=path)
CatBoost: cat_model.save_model(path, format='cbm') / CatBoostClassifier().load_model(path)

NO PICKLE — avoids arbitrary-code-execution risk and gives forward-compat
across library versions.

Cache key: (seed, date, blend_config_name, features_hash). features_hash is
SHA-1 of sorted feature columns to invalidate on schema change.

Cache layout:
  {cache_dir}/seed_{seed}/day_{YYYY-MM-DD}/
    meta.json        — feature cols signature + config list + predict_fn_kind
    {config_name}.{ext}  — one saved-model file per config (.lgb.txt or .cbm)

Note on predict_fn_kind: rather than guessing the prediction wrapper from the
serialized model, we record the kind explicitly when saving so loading is
unambiguous. The 12 baseline configs are all LightGBM classifiers, but storing
the kind defensively keeps the cache forward-compatible if a future blend config
uses a regressor / ranker / catboost model.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


# Adapter predict functions for *cached* (loaded-from-disk) models.
#
# The fresh-train predict_fns in backtest_blend assume the sklearn-wrapper API
# (LGBMClassifier.predict_proba, LGBMRegressor.predict). When we reload from
# native text format we get a raw lgb.Booster, which only exposes .predict()
# returning per-class-1 probability for binary classifiers (matching
# predict_proba(...)[:, 1] bit-exactly per the LightGBM source).
#
# These adapters mirror the fresh-train wrappers but call the Booster API
# instead. CatBoost reloads cleanly into the original CatBoostClassifier shape
# so we reuse _predict_catboost without an adapter. _predict_ranker and
# _predict_lgbm_regressor likewise call .predict() so they work on raw Boosters
# as-is (the LGBMRanker / LGBMRegressor wrappers happen to share the .predict
# entry point with raw Booster).


def _predict_lgbm_classifier_booster(model, day_data, cols):
    """Predict with a raw lgb.Booster reloaded from text format.

    Bit-exact with _predict_lgbm_classifier on the source classifier per
    LightGBM's predict_proba implementation (predict_proba returns
    [1 - p, p] columns; we slice column 1 which equals .predict()).
    """
    import numpy as np
    import pandas as pd

    pred_X = day_data[cols]
    valid = pred_X.notna().any(axis=1)
    probs = pd.Series(np.nan, index=day_data.index)
    if valid.any():
        probs[valid] = model.predict(pred_X[valid])
    return probs


# Mapping from predict_fn_kind string → import path. Keeps the cache file
# format independent of code-object identity (function references can't be
# JSON-serialized; their qualified names can).
#
# Note: the *cached* (loaded-from-disk) classifier uses the booster-aware
# adapter above, while regressor and ranker reuse the originals because the
# Booster.predict() entry point matches what those wrappers call.
_PREDICT_FNS: dict[str, str] = {
    "lgbm_classifier": "bts.simulate.blend_model_cache._predict_lgbm_classifier_booster",
    "lgbm_regressor": "bts.simulate.backtest_blend._predict_lgbm_regressor",
    "ranker": "bts.simulate.backtest_blend._predict_ranker",
    "catboost": "bts.simulate.backtest_blend._predict_catboost",
}


def _features_signature(feature_cols: list[str]) -> str:
    """SHA-1 prefix of the sorted feature column list — stable across reorderings."""
    canonical = ",".join(sorted(feature_cols))
    return hashlib.sha1(canonical.encode()).hexdigest()[:12]


def _resolve_predict_fn(kind: str):
    """Resolve a predict_fn_kind string to its callable."""
    qualified = _PREDICT_FNS.get(kind)
    if qualified is None:
        return None
    module_name, _, attr = qualified.rpartition(".")
    import importlib
    mod = importlib.import_module(module_name)
    return getattr(mod, attr)


def _classify_model(model, predict_fn) -> tuple[str, str, str] | None:
    """Determine (predict_fn_kind, file_extension, save_format) for a model.

    Returns None when the model type can't be cached (caller must retrain).
    """
    import lightgbm as lgb

    # LightGBM models — use the underlying Booster's save_model
    if isinstance(model, lgb.LGBMClassifier):
        return "lgbm_classifier", ".lgb.txt", "lgbm_text"
    if isinstance(model, lgb.LGBMRegressor):
        return "lgbm_regressor", ".lgb.txt", "lgbm_text"
    if isinstance(model, lgb.LGBMRanker):
        return "ranker", ".lgb.txt", "lgbm_text"
    if isinstance(model, lgb.Booster):
        # raw Booster — fall back to predict_fn-derived kind
        return "lgbm_classifier", ".lgb.txt", "lgbm_text"

    # CatBoost — duck-type via save_model(..., format='cbm')
    cls_name = type(model).__name__
    if cls_name in ("CatBoostClassifier", "CatBoostRegressor"):
        return "catboost", ".cbm", "catboost_cbm"

    return None


def cache_dir_for(cache_root: Path, seed: int, day: str) -> Path:
    """Return (and create) the per-(seed, day) cache directory."""
    p = Path(cache_root) / f"seed_{seed}" / f"day_{day}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_blend(
    blend: dict,
    cache_root: Path,
    seed: int,
    day: str,
    feature_cols_by_config: dict[str, list[str]],
) -> None:
    """Save each config's fitted model using its native format.

    Parameters
    ----------
    blend : dict
        {name: (model, cols, predict_fn)} as returned by ``_train_blend_for_day``.
    cache_root : Path
        Root cache directory; subdirs per (seed, day) are auto-created.
    seed : int
        Cache key dimension — typically ``BTS_LGBM_RANDOM_STATE``.
    day : str
        Cache key dimension — formatted as YYYY-MM-DD.
    feature_cols_by_config : dict
        {name: list[str]} of feature columns each config trained on, used to
        compute features_sig for cache-invalidation.
    """
    d = cache_dir_for(cache_root, seed, day)
    meta: dict = {"configs": {}}

    for name, (model, cols, _predict_fn) in blend.items():
        classification = _classify_model(model, _predict_fn)
        if classification is None:
            # Unsupported model type — skip caching (caller will retrain on miss).
            continue
        kind, ext, save_format = classification

        sig = _features_signature(feature_cols_by_config.get(name, cols))
        path = d / f"{name}{ext}"

        try:
            if save_format == "lgbm_text":
                # LGBMClassifier / LGBMRegressor / LGBMRanker expose .booster_;
                # raw Booster has save_model itself.
                booster = getattr(model, "booster_", model)
                booster.save_model(str(path))
            elif save_format == "catboost_cbm":
                model.save_model(str(path), format="cbm")
            else:
                continue
        except (TypeError, AttributeError, OSError):
            # Best-effort cache; fall back to retrain on miss.
            continue

        meta["configs"][name] = {
            "kind": kind,
            "path": path.name,
            "features_sig": sig,
        }

    (d / "meta.json").write_text(json.dumps(meta, indent=2))


def load_blend(
    cache_root: Path,
    seed: int,
    day: str,
    config_names: list[str],
    feature_cols_by_config: dict[str, list[str]],
) -> dict:
    """Load cached models for the given configs.

    Returns ``{name: (model, cols, predict_fn)}`` matching the blend tuple shape
    used elsewhere. Any config whose cached features_sig doesn't match the
    requested features_sig is SKIPPED (invalidated) — caller retrains those.
    """
    import lightgbm as lgb

    d = cache_dir_for(cache_root, seed, day)
    meta_path = d / "meta.json"
    if not meta_path.exists():
        return {}
    meta = json.loads(meta_path.read_text())
    out: dict = {}

    for name in config_names:
        entry = meta.get("configs", {}).get(name)
        if entry is None:
            continue

        cols = feature_cols_by_config.get(name, [])
        expected_sig = _features_signature(cols)
        if entry.get("features_sig") != expected_sig:
            continue  # invalidated, caller retrains

        path = d / entry["path"]
        if not path.exists():
            continue

        kind = entry.get("kind")
        predict_fn = _resolve_predict_fn(kind)
        if predict_fn is None:
            continue

        try:
            if kind in ("lgbm_classifier", "lgbm_regressor", "ranker"):
                model = lgb.Booster(model_file=str(path))
            elif kind == "catboost":
                from catboost import CatBoostClassifier
                model = CatBoostClassifier()
                model.load_model(str(path))
            else:
                continue
        except (OSError, RuntimeError, ImportError):
            continue

        out[name] = (model, cols, predict_fn)

    return out
