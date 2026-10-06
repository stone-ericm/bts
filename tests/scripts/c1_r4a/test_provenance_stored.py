"""C1 rank 4a: plain/gzip reads (review r1 R8) and the serving contract and witness checks (R2). Synthetic only."""
import gzip
import hashlib
import json
from datetime import date

import pytest

from scripts.audit.c1_r4a import provenance as PV
from scripts.audit.c1_r4a import stored as ST
from tests.scripts.c1_r4a.fixtures import RECIPE, contract, witness

D = date(2027, 4, 1)


# ---- stored -------------------------------------------------------------------------------------------------------

def test_absent_plain_and_gzip(tmp_path):
    assert ST.read(tmp_path / "x") is None
    (tmp_path / "p.json").write_bytes(b'{"a": 1}')
    s = ST.read(tmp_path / "p")
    assert (s.used, s.decoded, s.error, [f[:2] for f in s.files]) == ("json", b'{"a": 1}', None, [("json", "p.json")])
    gz = gzip.compress(b'{"a": 2}', mtime=0)
    (tmp_path / "g.json.gz").write_bytes(gz)
    s = ST.read(tmp_path / "g")
    assert (s.used, s.decoded, s.files) == ("json.gz", b'{"a": 2}', [("json.gz", "g.json.gz", gz)])
    assert s.manifest()["files"][0]["sha256"] == hashlib.sha256(gz).hexdigest()
    assert s.manifest()["decoded_sha256"] == hashlib.sha256(b'{"a": 2}').hexdigest()


def test_both_formats_identical_use_the_plain_file_and_keep_both(tmp_path):
    (tmp_path / "b.json").write_bytes(b'{"a": 1}')
    (tmp_path / "b.json.gz").write_bytes(gzip.compress(b'{"a": 1}'))
    s = ST.read(tmp_path / "b")
    assert s.used == "json" and s.error is None and len(s.files) == 2 and "identical" in s.note


def test_both_formats_different_is_a_conflict(tmp_path):
    (tmp_path / "b.json").write_bytes(b'{"a": 1}')
    (tmp_path / "b.json.gz").write_bytes(gzip.compress(b'{"a": 9}'))
    s = ST.read(tmp_path / "b")
    assert s.decoded is None and "different content" in s.error and len(s.files) == 2


def test_a_corrupt_gzip_or_unreadable_file_is_an_error_not_an_exception(tmp_path):
    (tmp_path / "c.json.gz").write_bytes(b"\x1f\x8b not gzip")
    s = ST.read(tmp_path / "c")
    assert s.decoded is None and "corrupt gzip" in s.error and s.files[0][1] == "c.json.gz"
    (tmp_path / "d.json").mkdir()                              # reading a directory raises an OSError
    s = ST.read(tmp_path / "d")
    assert "unreadable" in s.error and s.files == []


# ---- the serving contract -----------------------------------------------------------------------------------------

def load(raw):
    return PV.load(raw, hashlib.sha256(raw).hexdigest())


def test_the_contract_loads_only_its_pinned_well_formed_2027_self():
    c = load(contract())
    assert c.obj["recipe_sha256"] == RECIPE["sha256"]
    with pytest.raises(PV.ContractError, match="pin"):
        PV.load(contract(), "0" * 64)
    for bad in (contract(season=2026), contract(tier={"name": "local", "type": "ssh"}), contract(recipe_sha256="x"),
                contract(env={}), contract(calibration_enabled=0), contract(schedule=""), b"[]", b"nope"):
        with pytest.raises(PV.ContractError):
            load(bad)


def test_the_real_witness_builder_satisfies_the_reader():
    """The reader restates the recipe fingerprint; a witness from bts.serving_witness must recompute under it. The
    witness (e66b440) was reverted when 4a was deferred (row C1-r4a-deferral), so this skips until it returns. C2 step
    2a's amended witness is bts_serving_witness_v2; this reader pins v1 until step 3 aligns it (row C2-step2-split)."""
    W = pytest.importorskip("bts.serving_witness", reason="the serving witness was reverted with the 4a deferral")
    if getattr(W, "SCHEMA", None) != "bts_serving_witness_v1":
        pytest.skip("4a's reader pins witness v1; step 3 aligns it with the step-2a witness (row C2-step2-split)")
    w = W.build(model={"source": "cache", "file": "blend_2027-04-01.pkl", "sha256": "a" * 64},
                inputs=[{"file": "pa_2027.parquet", "bytes": 3, "sha256": "b" * 64}],
                calibration={"enabled": False, "applied": False})
    assert w["recipe"]["sha256"] == PV.recipe_fingerprint(w["recipe"]["files"], w["recipe"]["functions"])
    c = load(contract(recipe_sha256=w["recipe"]["sha256"], env=w["env"], packages=w["packages"]))
    assert PV.check({"tier": "local", "serving": w}, D, c) == ("ok", [])


def test_a_conforming_witness_is_ok():
    assert PV.check({"tier": "local", "serving": witness("2027-04-01")}, D, load(contract())) == ("ok", [])


@pytest.mark.parametrize("serving", [
    None, "x", witness("2027-04-01", schema="bts_serving_witness_v0"), witness("2027-04-01", recipe=None),
    witness("2027-04-01", recipe={**RECIPE, "files": {**RECIPE["files"], "model/predict.py": "0" * 64}}),
    witness("2027-04-01", recipe={**RECIPE, "functions": {}}),
    witness("2027-04-01", model={"source": "trained", "file": "blend_2027-04-01.pkl", "sha256": None}),
    witness("2027-04-01", model={"source": "elsewhere", "file": "blend_2027-04-01.pkl", "sha256": "e" * 64}),
    witness("2027-04-01", inputs=[]), witness("2027-04-01", inputs=None),
    witness("2027-04-01", inputs=[{"file": "pa_2027.parquet", "bytes": True, "sha256": "f" * 64}]),
    witness("2027-04-01", inputs=[{"file": "other.parquet", "bytes": 1, "sha256": "f" * 64}]),
    witness("2027-04-01", calibration=None), witness("2027-04-01", env=None), witness("2027-04-01", packages=[]),
])
def test_missing_provenance_is_missing(serving):
    status, reasons = PV.check({"tier": "local", "serving": serving}, D, load(contract()))
    assert status == "missing" and reasons


@pytest.mark.parametrize("envelope, why", [
    ({"tier": "mac", "serving": witness("2027-04-01")}, "tier"),
    ({"tier": "local", "serving": witness("2027-04-01", tier_type="ssh")}, "tier"),
    ({"tier": "local", "serving": witness("2027-04-01", recipe={"files": {"x.py": "1" * 64}, "functions": {"f": "2" * 64},
                                                                "sha256": PV.recipe_fingerprint({"x.py": "1" * 64},
                                                                                                {"f": "2" * 64})})},
     "recipe fingerprint"),
    ({"tier": "local", "serving": witness("2027-04-01", env={"BTS_LGBM_RANDOM_STATE": "7", "BTS_USE_CALIBRATION": None})},
     "flags"),
    ({"tier": "local", "serving": witness("2027-04-01", packages={"python": "3.12.13", "lightgbm": "9"})}, "package"),
    ({"tier": "local", "serving": witness("2027-04-01", calibration={"enabled": True, "applied": True})},
     "calibration"),
    ({"tier": "local", "serving": witness("2027-03-31")}, "daily blend"),
])
def test_an_evidenced_unregistered_change_is_changed(envelope, why):
    status, reasons = PV.check(envelope, D, load(contract()))
    assert status == "changed" and any(why in r for r in reasons)
