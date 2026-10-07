"""C2 step 2a, design §3: the serving witness builder `bts.serving_witness.build` (from e66b440, amended)."""
import hashlib
import json
from pathlib import Path

import pytest

import bts
from bts import serving_witness as W

SRC = Path(bts.__file__).resolve().parent
MODEL = {"source": "cache", "sha256": "a" * 64}
INPUTS = [{"file": "pa_2026.parquet", "bytes": 3, "sha256": "b" * 64}]
CAL = {"enabled": False, "applied": False, "status": "off"}


def _build(**k):
    return W.build(**{"model": MODEL, "inputs": INPUTS, "calibration": CAL, **k})


def test_the_witness_carries_every_part_under_schema_v2():
    w = _build()
    assert w["schema"] == "bts_serving_witness_v2" and w["tier_type"] == "local"
    assert (w["model"], w["inputs"], w["calibration"]) == (MODEL, INPUTS, CAL)
    assert set(w) == {"schema", "tier_type", "recipe", "env", "env_names", "packages", "model", "inputs",
                      "calibration", "built_at", "errors"}
    assert w["errors"] == []
    json.dumps(w, allow_nan=False)


def test_the_recipe_hashes_every_recipe_file_and_serving_function():
    rec = _build()["recipe"]
    expected = {p.relative_to(SRC).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for pkg in ("model", "features", "data") for p in (SRC / pkg).rglob("*.py")}
    assert rec["files"] == expected and list(rec["files"]) == sorted(expected)
    assert set(rec["functions"]) == {"bts.orchestrator:predict_local", "bts.picks:is_resume_date_game",
                                     "bts.util:is_regular_season_game"}
    body = json.dumps({"files": rec["files"], "functions": rec["functions"]}, sort_keys=True, separators=(",", ":"))
    assert rec["sha256"] == hashlib.sha256(body.encode()).hexdigest()


def test_env_records_allowlisted_values_and_only_the_names_of_other_bts_variables(monkeypatch):
    monkeypatch.setenv("BTS_USE_CALIBRATION", "1")
    monkeypatch.setenv("BTS_PARK_DRAG_TABLE", "/x/park_drag.parquet")
    monkeypatch.setenv("BTS_SOME_TOKEN", "s3cr3t-value")
    monkeypatch.delenv("BTS_LGBM_RANDOM_STATE", raising=False)
    w = _build()
    assert set(w["env"]) == set(W.RECIPE_ENV) and len(W.RECIPE_ENV) == 7
    assert w["env"]["BTS_USE_CALIBRATION"] == "1" and w["env"]["BTS_PARK_DRAG_TABLE"] == "/x/park_drag.parquet"
    assert w["env"]["BTS_LGBM_RANDOM_STATE"] is None
    assert "BTS_SOME_TOKEN" in w["env_names"] and w["env_names"] == sorted(w["env_names"])
    assert "s3cr3t-value" not in json.dumps(w)


def test_packages_record_python_and_the_numeric_stack():
    pk = _build()["packages"]
    assert set(pk) == {"python", "lightgbm", "numpy", "pandas", "pyarrow", "scikit-learn"} and pk["python"]


def test_upstream_errors_are_carried_without_aliasing():
    upstream = ["pa_2026.parquet: sha256 failed"]
    w = _build(errors=upstream)
    upstream.append("later")
    assert w["errors"] == ["pa_2026.parquet: sha256 failed"]


def test_a_recipe_failure_nulls_the_recipe_and_records_it(monkeypatch):
    monkeypatch.setattr(W, "recipe", lambda: (_ for _ in ()).throw(MemoryError("synthetic recipe failure")))
    w = _build()
    assert w["recipe"] is None and any("recipe" in e for e in w["errors"])


def test_a_packages_failure_nulls_packages_and_records_it(monkeypatch):
    monkeypatch.setattr(W, "_packages", lambda: (_ for _ in ()).throw(RuntimeError("synthetic packages failure")))
    w = _build()
    assert w["packages"] is None and any("packages" in e for e in w["errors"])


def test_an_env_failure_nulls_env_and_records_it(monkeypatch):
    class BadEnviron(dict):
        def get(self, *a):
            raise RuntimeError("synthetic environ failure")

        def __iter__(self):
            raise RuntimeError("synthetic environ failure")
    monkeypatch.setattr(W.os, "environ", BadEnviron())
    w = _build()
    assert w["env"] is None and w["env_names"] is None and len([e for e in w["errors"] if "env" in e]) == 2


def test_a_missing_package_is_recorded_as_null(monkeypatch):
    real = W.version
    monkeypatch.setattr(W, "version", lambda name: (_ for _ in ()).throw(LookupError(name)) if name == "pyarrow"
                        else real(name))
    assert _build()["packages"]["pyarrow"] is None


def test_unusable_upstream_errors_are_recorded_not_raised():
    class Unlistable:
        def __iter__(self):
            raise RuntimeError("synthetic")
    w = _build(errors=Unlistable())
    assert any("upstream errors" in e for e in w["errors"])


def test_the_containment_helpers_never_raise():
    class NoAppend(list):
        def append(self, x):
            raise RuntimeError("synthetic")
    W.note(NoAppend(), "x")
    W.note(None, "x")
    assert W.sha256_or_none(object(), [], "what") is None
