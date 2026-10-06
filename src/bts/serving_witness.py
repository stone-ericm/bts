"""Serving provenance for each persisted slate (C1 rank 4a prerequisite; the registration's X-E1 freeze rules,
`docs/sota_audit/2026-10-04-prereg-c1-calibration.md`).

A study that scores served forecasts must show which recipe, model artifact and inputs produced each capture, from
the capture itself rather than from a later reading of the configuration or the model directory. The local tier
builds this witness in the serving process, from what that process used, and `save_slate` stores it in the slate
envelope (`serving`):
- **recipe:** the sha256 of every `.py` file under `bts/model`, `bts/features` and `bts/data`, plus the source of the
  serving functions outside those packages (`RECIPE_FUNCTIONS`), and one fingerprint over both;
- **env:** the raw values of the recipe flags (`RECIPE_ENV`, an allowlist), and the names (never the values) of every
  other `BTS_*` variable;
- **packages:** the Python and numeric-stack versions;
- **model:** whether the blend came from the per-date cache or was trained in this run, its file name, and the
  sha256 of the bytes loaded (cache) or saved (trained);
- **inputs:** each PA parquet's name, size and sha256, from the bytes that were parsed;
- **calibration:** whether `BTS_USE_CALIBRATION` was on and whether a calibrator was applied.

**Limit:** the recipe hashes the files on disk when the witness is built. A deploy restarts the scheduler, so the
loaded code and the files differ only between a deploy's checkout and its restart.

This is observability. `build` never raises; a part that fails is recorded as null with its error.
"""
from __future__ import annotations

import hashlib
import importlib
import inspect
import json
import os
import platform
from datetime import datetime, timezone
from importlib.metadata import version
from pathlib import Path

SCHEMA = "bts_serving_witness_v1"
RECIPE_PACKAGES = ("model", "features", "data")
RECIPE_FUNCTIONS = ("bts.orchestrator:predict_local", "bts.picks:is_resume_date_game",
                    "bts.util:is_regular_season_game")
RECIPE_ENV = ("BTS_LGBM_RANDOM_STATE", "BTS_LGBM_DETERMINISTIC", "BTS_USE_CALIBRATION", "BTS_ROOKIE_GATE_K",
              "BTS_PITCHER_HR_30G_MIN_PERIODS", "BTS_REFRESH_ALWAYS")
PACKAGES = ("lightgbm", "numpy", "pandas", "pyarrow", "scikit-learn")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _function_source(spec: str) -> str:
    mod, name = spec.split(":")
    return inspect.getsource(getattr(importlib.import_module(mod), name))


def recipe(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    files = {}
    for pkg in RECIPE_PACKAGES:
        for p in sorted((root / pkg).rglob("*.py")):
            files[p.relative_to(root).as_posix()] = _sha(p.read_bytes())
    functions = {spec: _sha(_function_source(spec).encode()) for spec in RECIPE_FUNCTIONS}
    body = json.dumps({"files": files, "functions": functions}, sort_keys=True, separators=(",", ":"))
    return {"files": dict(sorted(files.items())), "functions": functions, "sha256": _sha(body.encode())}


def _packages() -> dict:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except Exception:                     # noqa: BLE001 - a missing package is recorded, never raised
            out[name] = None
    return out


def build(*, model: dict | None, inputs: list | None, calibration: dict | None) -> dict:
    errors = []
    try:
        rec = recipe()
    except Exception as exc:                  # noqa: BLE001
        rec = None
        errors.append(f"recipe: {type(exc).__name__}: {exc}"[:300])
    try:
        packages = _packages()
    except Exception as exc:                  # noqa: BLE001
        packages = None
        errors.append(f"packages: {type(exc).__name__}: {exc}"[:300])
    return {"schema": SCHEMA, "tier_type": "local", "recipe": rec,
            "env": {k: os.environ.get(k) for k in RECIPE_ENV},
            "env_names": sorted(k for k in os.environ if k.startswith("BTS_")),
            "packages": packages, "model": model, "inputs": inputs, "calibration": calibration,
            "built_at": datetime.now(timezone.utc).isoformat(), "errors": errors}
