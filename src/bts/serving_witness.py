"""Serving provenance for each persisted slate (C2 step 2a; design `docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md`).

C1 rank 4a's registration (`docs/sota_audit/2026-10-04-prereg-c1-calibration.md`, X-E1) scores served forecasts only
when each capture shows the recipe, model artifact and inputs that produced it, recorded in the serving process rather
than reconstructed later. The local tier builds this witness from what that process used, and `save_slate` stores it
in the slate envelope (`serving`):
- **recipe:** the sha256 of every `.py` file under `bts/model`, `bts/features` and `bts/data`, plus the source of the
  serving functions outside those packages (`RECIPE_FUNCTIONS`), and one fingerprint over both;
- **env:** the values of the recipe flags (`RECIPE_ENV`, an allowlist), and the names (never the values) of every
  other `BTS_*` variable;
- **packages:** the Python and numeric-stack versions;
- **model:** `{source: cache | trained | trained_unsaved, sha256}` from the branch `run_pipeline` actually took;
- **inputs:** each PA parquet's `{file, bytes, sha256}`, of the bytes that were parsed;
- **calibration:** the serving calibration record (design §3.2);
- **errors:** every provenance failure; any entry makes the witness incomplete provenance.

The amended witness is `bts_serving_witness_v2` (e66b440's v1 was never deployed; the model and calibration parts
changed shape).

**Limits:** the recipe hashes the files on disk when the witness is built, so the loaded code and the files can differ
only between a deploy's checkout and its restart. Env values are read when the witness is built; import-time
constants come from the same process environment. The witness does not retain the live schedule or lineup responses.

The containment helpers are shared by every capture point (design §3.0). Each contains its own failure: a provenance
problem is recorded in a local error list and never replaces, retries or alters the computation. `build` never
raises; a part that fails is null with its error.
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

SCHEMA = "bts_serving_witness_v2"
RECIPE_PACKAGES = ("model", "features", "data")
RECIPE_FUNCTIONS = ("bts.orchestrator:predict_local", "bts.picks:is_resume_date_game",
                    "bts.util:is_regular_season_game")
RECIPE_ENV = ("BTS_LGBM_RANDOM_STATE", "BTS_LGBM_DETERMINISTIC", "BTS_USE_CALIBRATION", "BTS_ROOKIE_GATE_K",
              "BTS_PITCHER_HR_30G_MIN_PERIODS", "BTS_REFRESH_ALWAYS", "BTS_PARK_DRAG_TABLE")
PACKAGES = ("lightgbm", "numpy", "pandas", "pyarrow", "scikit-learn")


def note(errors, msg: str) -> None:
    """Record a provenance error; recording itself never raises."""
    if errors is None:
        return
    try:
        errors.append(msg)
    except Exception:
        pass


def collect(items, item, errors, what: str) -> None:
    """Append to a provenance collector; a failed append is recorded, never raised."""
    if items is None:
        return
    try:
        items.append(item)
    except Exception as e:
        note(errors, f"{what}: collector append failed: {e!r}")


def sha256_or_none(raw, errors, what: str):
    """sha256 hex of the held bytes, or None with a recorded error."""
    try:
        return hashlib.sha256(raw).hexdigest()
    except Exception as e:
        note(errors, f"{what}: sha256 failed: {e!r}")
        return None


def canon_sha256(obj) -> str:
    """sha256 of canonical JSON (sorted keys, no whitespace, UTF-8); a non-finite float raises ValueError."""
    return hashlib.sha256(
        json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()


def _error(what: str, exc: BaseException) -> str:
    return f"{what}: {type(exc).__name__}: {exc}"[:300]


def _function_source(spec: str) -> str:
    mod, name = spec.split(":")
    return inspect.getsource(getattr(importlib.import_module(mod), name))


def recipe(root: Path | None = None) -> dict:
    root = Path(root) if root is not None else Path(__file__).resolve().parent
    files = {}
    for pkg in RECIPE_PACKAGES:
        for p in sorted((root / pkg).rglob("*.py")):
            files[p.relative_to(root).as_posix()] = hashlib.sha256(p.read_bytes()).hexdigest()
    files = dict(sorted(files.items()))
    functions = {spec: hashlib.sha256(_function_source(spec).encode()).hexdigest() for spec in RECIPE_FUNCTIONS}
    body = json.dumps({"files": files, "functions": functions}, sort_keys=True, separators=(",", ":"))
    return {"files": files, "functions": functions, "sha256": hashlib.sha256(body.encode()).hexdigest()}


def _packages() -> dict:
    out = {"python": platform.python_version()}
    for name in PACKAGES:
        try:
            out[name] = version(name)
        except Exception:                     # a missing package is recorded as null, never raised
            out[name] = None
    return out


def build(*, model, inputs, calibration, errors=None) -> dict:
    """The serving witness for one local-tier forecast. Never raises; a failed part is null with its error.

    `errors` are the upstream provenance errors (run_pipeline's and predict_local's); they are copied.
    """
    errs: list = []
    try:
        errs.extend(errors or [])
    except Exception as e:
        note(errs, _error("upstream errors", e))
    parts = {}
    for key, fn in (("recipe", recipe), ("packages", _packages),
                    ("env", lambda: {k: os.environ.get(k) for k in RECIPE_ENV}),
                    ("env_names", lambda: sorted(k for k in os.environ if k.startswith("BTS_")))):
        try:
            parts[key] = fn()
        except Exception as e:
            parts[key] = None
            note(errs, _error(key, e))
    try:
        built_at = datetime.now(timezone.utc).isoformat()
    except Exception as e:
        built_at = None
        note(errs, _error("built_at", e))
    return {"schema": SCHEMA, "tier_type": "local", "recipe": parts["recipe"], "env": parts["env"],
            "env_names": parts["env_names"], "packages": parts["packages"], "model": model, "inputs": inputs,
            "calibration": calibration, "built_at": built_at, "errors": errs}
