"""C2 step 2a §5 gate: the candidate reproduces the deployed baseline's golden outputs (design
`docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md` §5).

`tests/c2_2a/golden/data/` was written by `generate.py` at f882411 (`manifest.json` records the commit, the command,
the environment and the sha256 of every harness file). Each scenario is re-run here against the candidate with the same
harness. Everything observed must be equal — the predictions selection read, every SelectionResult, lock decision and
fallback plan, the transport log, every file written under data/picks, the cache bytes, any exception — except the
three intended slate differences: the schema tag (v2 -> v3), the envelope's `serving`, and posted rows' explicit
`projected: false`. The witness the candidate persisted is then checked per scenario.

The scenarios run the real pick path end to end (about 2 s of feature computation per cascade), so the module is
marked slow; the harness-integrity checks are not.
"""
import copy
import hashlib
import json
from pathlib import Path

import pytest

from tests.c2_2a.golden import generate as G
from tests.c2_2a.golden import scenarios as S

HERE = Path(__file__).resolve().parent / "golden"
DATA = HERE / "data"
REPO = Path(__file__).resolve().parents[2]
MANIFEST = json.loads((DATA / "manifest.json").read_text()) if (DATA / "manifest.json").exists() else None
BASELINE = "f882411"


def _canon(o):
    return hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"), allow_nan=False)
                          .encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- harness integrity (fast)

def test_the_goldens_come_from_the_deployed_baseline_with_this_exact_harness():
    assert MANIFEST is not None, "run generate.py at f882411 first"
    assert MANIFEST["schema"] == "c2_2a_golden_manifest_v1" and MANIFEST["partial"] is False
    assert MANIFEST["commit"].startswith(BASELINE) and MANIFEST["tracked_tree_clean"] is True
    assert MANIFEST["harness_sha256"] == G.harness_hashes(HERE)
    assert MANIFEST["scenarios"] == list(S.SCENARIOS)
    assert MANIFEST["environment"]["TZ"] == "America/New_York" and MANIFEST["environment"]["OMP_NUM_THREADS"] == "1"
    for name, sha in MANIFEST["files"].items():
        assert hashlib.sha256((DATA / name).read_bytes()).hexdigest() == sha, name


def test_the_environment_matches_the_goldens():
    env = G.environment()
    assert env["packages"] == MANIFEST["environment"]["packages"]
    assert env["python"] == MANIFEST["environment"]["python"]


# ---------------------------------------------------------------- the comparison

# A capture-preparation failure after the bytes were read falls back to the original read once (design §3.0), so the
# decoder fault reads each history pick exactly once more than the baseline; nowhere else may the reads differ.
FALLBACK_REREAD = {"fault_pick_decoder", "fault_pick_decoder_oserror"}


def _compare(golden: dict, cand: dict, name: str = "") -> None:
    g, c = copy.deepcopy(golden), copy.deepcopy(cand)
    if name in FALLBACK_REREAD:
        gr, cr = g.pop("pick_reads"), c.pop("pick_reads")
        assert gr and set(cr) == set(gr) and all(cr[f] == gr[f] + 1 for f in gr)
    gs, cs = g.pop("slate"), c.pop("slate")
    assert c == g                                          # the whole surface outside the slate: exact
    if gs is None:
        assert cs is None
        return
    assert cs is not None
    for key in ("tier", "date", "n_rows", "written_at", "envelope_keys", "rows"):
        assert cs[key] == gs[key], key                     # every persisted envelope value and row value but `projected`
    assert gs["schema_version"] == "bts_slate_v2" and cs["schema_version"] == "bts_slate_v3"
    assert gs["serving"] == S._MISSING_SERVING
    assert cs["projected"] == [v is True for v in gs["projected"]]
    assert all(type(v) is bool for v in cs["projected"])


@pytest.fixture(scope="module")
def observed(tmp_path_factory):
    out = {}
    root = tmp_path_factory.mktemp("golden-candidate")

    def get(name):
        if name not in out:
            out[name] = S.run(name, REPO, root / name)
        return out[name]
    return get


@pytest.mark.slow
@pytest.mark.parametrize("name", list(S.SCENARIOS))
def test_the_candidate_reproduces_the_baseline(name, observed):
    golden = json.loads((DATA / f"{name}.json").read_text())
    cand = observed(name)
    assert cand["unknown_urls"] == [] and golden["unknown_urls"] == []
    _compare(golden, cand, name)
    _check_witness(name, cand, observed)


# ---------------------------------------------------------------- the witness each scenario persisted

def _w(cand):
    sl = cand["slate"]
    return None if sl is None else sl["serving"]


def _assert_inputs_bound(w, cand):
    world = cand["world_inputs"]
    for i in w["inputs"] or []:
        if i["sha256"] is not None:
            assert world[f"data/processed/{i['file']}"] == i["sha256"] and i["bytes"] > 0


def _assert_calibration_bound(c, cand, check_map=True):
    # complete means the whole consumed inventory, in the resolver's order (r1 F3), not just the retained records
    assert [i["file"] for i in c["pick_inputs"]] == cand["resolver_inventory"][-1]
    world = dict(cand["world_inputs"])
    for rel, text in cand["files"].items():                # history files a scenario rewrote, as the day left them
        if rel.startswith("data/picks/2026-0") and not text.startswith("sha256:"):
            world[rel] = hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert c["n_fit"] == len(c["samples"])
    assert c["samples_sha256"] == _canon(c["samples"])
    files = [i["file"] for i in c["pick_inputs"]]
    assert files == sorted(files)
    for i in c["pick_inputs"]:
        rel = f"data/picks/{i['file']}"
        if rel in world and i["sha256"] is not None:
            assert world[rel] == i["sha256"]
    by_file = {i["file"]: i["sha256"] for i in c["pick_inputs"]}
    for b in c["samples"]:
        assert b["file_sha256"] == by_file[b["file"]] and b["slot"] in ("pick", "double_down")
    if c["map"] is not None and check_map:
        assert c["map_sha256"] == _canon(c["map"])
        assert (c["map"]["out_of_bounds"], c["map"]["y_min"], c["map"]["y_max"]) == ("clip", 0.0, 1.0)


EXPECT_CAL = {
    "calibration_on": ("applied", True), "calibration_off_explicit": ("off", False),
    "calibration_no_pa_file": ("no_pa_file", False), "calibration_insufficient_support": ("insufficient_support", False),
    "calibration_no_sklearn": ("no_sklearn", False), "calibration_changed_history": ("applied", True),
    "calibration_two_thresholds": ("applied", True), "calibration_decode_error": ("failed", False),
    "calibration_fit_failure": ("failed", False), "calibration_apply_failure": ("failed", False),
    "calibration_error_after_assignment": ("applied", True), "fault_calibration_pa_buffer": ("applied", True),
    "fault_pick_buffer": ("applied", True), "fault_pick_decoder": ("applied", True),
    "fault_collector_appends": ("applied", True), "fault_sample_canonicalisation": ("applied", True),
    "fault_map_extraction": ("applied", True), "fault_error_recording": ("applied", True),
    "fault_attrs_copy_calibration": ("applied", True), "fault_pick_decoder_oserror": ("applied", True),
    "fault_omitted_input_lost_error": ("applied", True), "fault_undescribable_pick": ("applied", True),
    "fault_map_hash": ("applied", True), "fault_provenance_allocation": ("applied", True),
    "fault_provenance_take": ("applied", True), "fault_pa_append_landed": ("applied", True),
}
PIPELINE_PROVENANCE_LOST = {"fault_provenance_allocation", "fault_provenance_take"}   # r2 R2-1
CACHE_SOURCE = {"model_cached", "fault_cache_buffer", "fault_cache_hash", "fault_undescribable_cache"}
ONE_CYCLE_DAYS = {"day_all_posted"}                   # locks at its first check: its only slate trained the model
NO_WITNESS = {"day_prediction_failure", "genuine_cache_unpickle", "genuine_cache_unpickle_stateful",
              "genuine_parquet_parse", "genuine_save_serialization", "genuine_partial_write",
              "fault_build", "fault_attrs_assignment"}


def _check_witness(name, cand, observed):
    w = _w(cand)
    if name in NO_WITNESS:
        assert w is None                                   # no slate, or the attachment boundary itself failed
        return
    assert isinstance(w, dict) and w["schema"] == "bts_serving_witness_v2" and w["tier_type"] == "local"
    assert w["recipe"] and w["packages"] and w["env"] is not None
    _assert_inputs_bound(w, cand)
    model, cache = w["model"], cand["cache_sha256"]
    c = w["calibration"]
    if name == "fault_calibration_record":
        assert c is None                                   # the record could not be built: null, never stale
    else:
        status, applied = EXPECT_CAL.get(name, ("off", False))
        assert (c["status"], c["applied"], c["enabled"]) == (status, applied, status != "off")
        if status in ("applied", "insufficient_support") and name not in (
                "fault_collector_appends", "fault_sample_canonicalisation", "fault_omitted_input_lost_error"):
            _assert_calibration_bound(c, cand, check_map=name != "fault_map_hash")
    clean = not name.startswith("fault_") and name not in (
        "calibration_decode_error", "calibration_fit_failure", "calibration_apply_failure",
        "calibration_error_after_assignment")
    if clean:
        assert w["errors"] == [] and c["errors"] == []
        assert all(i["sha256"] for i in w["inputs"]) and model["sha256"] == cache
    # A day's last slate comes from a later cycle, which loads the first cycle's cache; a single run trains unless
    # the scenario wrote a cache first.
    expected_source = ("cache" if name in CACHE_SOURCE or (name.startswith("day_") and name not in ONE_CYCLE_DAYS)
                       else "trained")
    if name in PIPELINE_PROVENANCE_LOST:                  # the pipeline's provenance is visibly unavailable (r2 R2-1)
        assert model is None and w["inputs"] is None and "run_pipeline provenance unavailable" in w["errors"]
    else:
        assert model["source"] == expected_source
    _SPECIFIC.get(name, lambda w, c, cand, observed: None)(w, c, cand, observed)


def _two_thresholds(w, c, cand, observed):
    assert c["n_fit"] == 30 and len(c["map"]["X_thresholds"]) == 2


def _changed_history(w, c, cand, observed):
    on = _w(observed("calibration_on"))["calibration"]
    assert c["samples_sha256"] != on["samples_sha256"]
    assert [i["sha256"] for i in c["pick_inputs"]] != [i["sha256"] for i in on["pick_inputs"]]


def _normal_reads(w, c, cand, observed):
    assert cand["pick_reads"] and set(cand["pick_reads"].values()) == {1}


def _null_model_hash(w, c, cand, observed):
    assert w["model"]["sha256"] is None and w["errors"]


def _null_input_hashes(w, c, cand, observed):
    assert all(i["sha256"] is None for i in w["inputs"]) and len(w["errors"]) >= 2


def _null_pa_input(w, c, cand, observed):
    assert c["pa_input"]["sha256"] is None and c["errors"]


def _null_pick_inputs(w, c, cand, observed):
    assert c["pick_inputs"] and all(i["sha256"] is None for i in c["pick_inputs"])
    assert all(b["file_sha256"] is None for b in c["samples"]) and c["errors"]


def _empty_collectors(w, c, cand, observed):
    # every append failed: each collection is withheld (null), never published partial (r1 F3)
    assert c["pick_inputs"] is None and c["samples"] is None and c["samples_sha256"] is None and c["n_fit"] > 0
    assert w["inputs"] is None and c["errors"] and w["errors"]


def _null_samples_hash(w, c, cand, observed):
    assert c["samples_sha256"] is None and c["map_sha256"] is None and c["errors"]


def _null_map(w, c, cand, observed):
    assert c["map"] is None and c["map_sha256"] is None and c["errors"]


def _one_null_input(w, c, cand, observed):
    nulls = [i for i in w["inputs"] if i["sha256"] is None]
    assert [i["file"] for i in nulls] == ["pa_2025.parquet"] and nulls[0]["bytes"] is None and w["errors"]


def _all_null_inputs(w, c, cand, observed):
    assert w["inputs"] and all(i["sha256"] is None and i["bytes"] is None for i in w["inputs"]) and w["errors"]


def _withheld_inputs(w, c, cand, observed):
    assert c["pick_inputs"] is None and c["n_fit"] and c["samples"] is not None


def _null_map_hash(w, c, cand, observed):
    assert c["map"] is not None and c["map_sha256"] is None and c["samples_sha256"] is not None and c["errors"]


def _package_error(w, c, cand, observed):
    assert w["packages"]["pyarrow"] is None and any("pyarrow" in e for e in w["errors"])


def _error_after_assignment(w, c, cand, observed):
    assert any("after assignment" in e for e in c["errors"])


def _failed_with_error(w, c, cand, observed):
    assert c["errors"]


def _landed_pa_withheld(w, c, cand, observed):
    # every PA append landed and raised, its error lost: withheld on both sides, whatever the lengths say (r2 R2-2)
    assert w["inputs"] is None and "PA inputs incomplete; withheld" in w["errors"]
    assert c["pa_input"] is None and "calibration PA input incomplete; withheld" in c["errors"]


_SPECIFIC = {
    "calibration_two_thresholds": _two_thresholds, "calibration_changed_history": _changed_history,
    "calibration_on": _normal_reads,
    "fault_cache_buffer": _null_model_hash, "fault_cache_hash": _null_model_hash,
    "fault_hashing_writer_construction": _null_model_hash, "fault_hashing_writer_hash": _null_model_hash,
    "fault_hashing_writer_finalisation": _null_model_hash,
    "fault_parquet_buffer": _null_input_hashes, "fault_parquet_hash": _null_input_hashes,
    "fault_calibration_pa_buffer": _null_pa_input,
    "fault_pick_buffer": _null_pick_inputs, "fault_pick_decoder": _null_pick_inputs,
    "fault_collector_appends": _empty_collectors, "fault_sample_canonicalisation": _null_samples_hash,
    "fault_map_extraction": _null_map, "calibration_error_after_assignment": _error_after_assignment,
    "calibration_decode_error": _failed_with_error, "calibration_fit_failure": _failed_with_error,
    "calibration_apply_failure": _failed_with_error,
    "fault_undescribable_parquet": _one_null_input, "fault_parquet_buffer_alloc": _all_null_inputs,
    "fault_undescribable_cache": _null_model_hash, "fault_short_write": _null_model_hash,
    "fault_pick_decoder_oserror": _null_pick_inputs, "fault_undescribable_pick": _null_pick_inputs,
    "fault_omitted_input_lost_error": _withheld_inputs, "fault_map_hash": _null_map_hash,
    "fault_package_query": _package_error, "fault_pa_append_landed": _landed_pa_withheld,
}


@pytest.mark.slow
def test_the_stateful_unpickler_ran_once_on_both_sides(observed):
    golden = json.loads((DATA / "genuine_cache_unpickle_stateful.json").read_text())
    assert golden["unpickle_calls"] == 1 and observed("genuine_cache_unpickle_stateful")["unpickle_calls"] == 1


@pytest.mark.slow
def test_delivery_at_the_cutoff_edges(observed):
    assert observed("cutoff_minus_one_second")["delivered"] is True
    assert observed("cutoff_exact")["delivered"] is False
    assert observed("cutoff_advancing_clock")["delivered"] is False
