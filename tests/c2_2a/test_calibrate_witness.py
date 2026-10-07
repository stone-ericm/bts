"""C2 step 2a, design §3.2 / §3.0: calibration's witness (`docs/superpowers/specs/2026-10-06-c2-2a-serving-record-design.md`).

The witness only observes: the resolver's samples, order, filters and return value are unchanged, and every witness
failure affects provenance only. Code review r4 (Eric's row C2-2a-review-r4): the resolver and the fit run their
deployed statements, guarded hooks beside them record into the run's witness (bts.serving_witness), and the pick
inputs and sample bindings stand only when earned: every pick file confirmed against an independent listing, and the
bindings' values exactly the samples'."""
import contextlib
import hashlib
import json
from datetime import date
from pathlib import Path

import pandas as pd
import pytest

from bts import serving_witness as W
from bts.model import calibrate as C
from bts.picks import _read_text_bytes


@contextlib.contextmanager
def witnessed():
    """An open witness for the code under test, as predict_local opens one."""
    w = W.Serving()
    token = W.begin(w)
    try:
        yield w
    finally:
        W.end(token)


def _picks(w):
    c = w.calibration
    return c.picks.records if c.pick_names is not None and c.picks.complete(c.pick_names) else None


TODAY = date(2026, 6, 30)


def _pick(d, *, p=0.8, bid=1, result="hit", dd=None, slots=None):
    rec = {"date": d, "result": result, "pick": {"batter_id": bid, "p_game_hit": p}}
    if dd is not None:
        rec["double_down"] = dd
    if slots is not None:
        rec["slot_results"] = slots
    return rec


def _write(dir_: Path, name: str, rec) -> Path:
    f = dir_ / name
    f.write_text(rec if isinstance(rec, str) else json.dumps(rec))
    return f


def _pa(rows):
    return pd.DataFrame([{"batter_id": b, "date": d, "is_hit": h, "is_resumed_portion": False} for b, d, h in rows])


@pytest.fixture
def world(tmp_path):
    picks = tmp_path / "picks"
    picks.mkdir()
    _write(picks, "2026-06-20.json", _pick("2026-06-20", p=0.81, bid=1,
                                            dd={"batter_id": 2, "p_game_hit": 0.77}))
    _write(picks, "2026-06-21.json", _pick("2026-06-21", p=0.79, bid=3, slots={"pick": "void"}))     # void: skipped
    _write(picks, "2026-06-22.json", "{not json")                                                      # malformed
    _write(picks, "2026-04-01.json", _pick("2026-04-01", p=0.7, bid=1))                                # out of window
    _write(picks, "2026-06-23.json", _pick("2026-06-23", p=0.75, bid=4, result="unresolved"))          # unresolved
    _write(picks, "2026-06-24.json", _pick("2026-06-24", p=0.83, bid=5))
    pa = _pa([(1, "2026-06-20", 1), (2, "2026-06-20", 0), (3, "2026-06-21", 1), (5, "2026-06-24", 0)])
    return picks, pa


def test_the_return_value_and_order_are_unchanged_by_the_witness(world):
    picks, pa = world
    plain = C._resolve_pick_outcomes(picks, pa, TODAY, 30)
    with witnessed() as w:
        observed = C._resolve_pick_outcomes(picks, pa, TODAY, 30)
    assert observed == plain == [(0.81, 1), (0.77, 0), (0.83, 0)]
    assert w.calibration.errors == []


def test_an_unwitnessed_resolver_reads_exactly_as_deployed(world, monkeypatch):
    picks, pa = world
    reads = []
    real_rb, real_rt = Path.read_bytes, Path.read_text
    monkeypatch.setattr(Path, "read_bytes", lambda self: (reads.append(("b", self.name)), real_rb(self))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(("t", self.name)),
                                                                   real_rt(self, *a, **k))[1])
    assert C._resolve_pick_outcomes(picks, pa, TODAY, 30) == [(0.81, 1), (0.77, 0), (0.83, 0)]
    assert sorted(reads) == sorted(("t", p.name) for p in picks.glob("2*.json"))


def test_bindings_follow_the_samples_exactly(world):
    picks, pa = world
    with witnessed() as w:
        samples = C._resolve_pick_outcomes(picks, pa, TODAY, 30)
    sha = {i["file"]: i["sha256"] for i in w.calibration.picks.records}
    assert w.calibration.bindings == [
        {"file": "2026-06-20.json", "file_sha256": sha["2026-06-20.json"], "date": "2026-06-20", "slot": "pick",
         "batter_id": 1, "p": 0.81, "y": 1},
        {"file": "2026-06-20.json", "file_sha256": sha["2026-06-20.json"], "date": "2026-06-20", "slot": "double_down",
         "batter_id": 2, "p": 0.77, "y": 0},
        {"file": "2026-06-24.json", "file_sha256": sha["2026-06-24.json"], "date": "2026-06-24", "slot": "pick",
         "batter_id": 5, "p": 0.83, "y": 0},
    ]
    assert [(b["p"], b["y"]) for b in w.calibration.bindings] == samples


def test_pick_inputs_record_every_held_buffer_in_read_order_including_skipped_files(world):
    picks, pa = world
    with witnessed() as w:
        C._resolve_pick_outcomes(picks, pa, TODAY, 30)
    inputs = _picks(w)
    names = sorted(p.name for p in picks.glob("2*.json"))
    assert [i["file"] for i in inputs] == names                    # malformed, out-of-window and unresolved included
    for i in inputs:
        raw = (picks / i["file"]).read_bytes()
        assert i == {"file": i["file"], "bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}


def test_normal_path_reads_each_pick_file_once(world, monkeypatch):
    picks, pa = world
    reads = []
    real_rb, real_rt = Path.read_bytes, Path.read_text
    monkeypatch.setattr(Path, "read_bytes", lambda self: (reads.append(("b", self.name)), real_rb(self))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(("t", self.name)),
                                                                   real_rt(self, *a, **k))[1])
    with witnessed():
        C._resolve_pick_outcomes(picks, pa, TODAY, 30)
    assert sorted(reads) == sorted(("b", p.name) for p in picks.glob("2*.json"))


def test_the_decoder_matches_read_text_and_the_production_helper(tmp_path):
    f = tmp_path / "2026-06-20.json"
    f.write_bytes('{"date": "2026-06-20", "name": "José Ramírez"}\r\n'.encode())
    raw = f.read_bytes()
    held = W._pick_text(f)(raw, None)
    assert held.read_text() == f.read_text() == _read_text_bytes(raw)


def test_a_hash_failure_keeps_the_sample_and_nulls_only_provenance(world, monkeypatch):
    picks, pa = world
    plain = C._resolve_pick_outcomes(picks, pa, TODAY, 30)

    class Boom:
        def __init__(self, *a, **k):
            raise MemoryError("synthetic hash failure")
    monkeypatch.setattr(hashlib, "sha256", Boom)
    with witnessed() as w:
        assert C._resolve_pick_outcomes(picks, pa, TODAY, 30) == plain
    c = w.calibration
    assert len(c.bindings) == 3 and all(b["file_sha256"] is None for b in c.bindings)
    assert all(i["sha256"] is None for i in _picks(w)) and c.errors


def test_a_capture_preparation_failure_falls_back_to_one_original_read(world, monkeypatch):
    picks, pa = world
    plain = C._resolve_pick_outcomes(picks, pa, TODAY, 30)
    reads = []
    real_rt = Path.read_text
    monkeypatch.setattr(Path, "read_bytes", lambda self: (_ for _ in ()).throw(MemoryError("synthetic buffer")))
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(self.name), real_rt(self, *a, **k))[1])
    with witnessed() as w:
        assert C._resolve_pick_outcomes(picks, pa, TODAY, 30) == plain
    c = w.calibration
    assert sorted(reads) == sorted(p.name for p in picks.glob("2*.json"))     # once each
    assert _picks(w) == [{"file": p.name, "bytes": None, "sha256": None} for p in sorted(picks.glob("2*.json"))]
    assert len(c.bindings) == 3 and all(b["file_sha256"] is None for b in c.bindings)
    assert len(c.errors) == len(_picks(w)) and all("read from the path" in e for e in c.errors)


def test_a_genuine_read_error_skips_the_file_without_an_added_retry(world, monkeypatch):
    """Deployed: the file's read raises, its handler skips it. Witnessed: the one read (into the held bytes) raises,
    and the computation statement then sees an OSError from a C-level stand-in, so the same handler skips it with no
    second read; the inventory is incomplete (withheld)."""
    picks, pa = world
    real_rt, real_rb = Path.read_text, Path.read_bytes
    with monkeypatch.context() as m:
        m.setattr(Path, "read_text", lambda self, *a, **k: (_ for _ in ()).throw(PermissionError("synthetic"))
                  if self.name == "2026-06-20.json" else real_rt(self, *a, **k))
        deployed = C._resolve_pick_outcomes(picks, pa, TODAY, 30)
    reads = []

    def flaky(self):
        reads.append(("b", self.name))
        if self.name == "2026-06-20.json":
            raise PermissionError("synthetic read error")
        return real_rb(self)
    monkeypatch.setattr(Path, "read_bytes", flaky)
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append(("t", self.name)),
                                                                   real_rt(self, *a, **k))[1])
    with witnessed() as w:
        assert C._resolve_pick_outcomes(picks, pa, TODAY, 30) == deployed == [(0.83, 0)]
    assert not [r for r in reads if r[0] == "t"] and reads.count(("b", "2026-06-20.json")) == 1
    assert _picks(w) is None and any("2026-06-20.json" in e for e in w.calibration.errors)


@pytest.mark.parametrize("witness", [False, True], ids=["unwitnessed", "witnessed"])
def test_a_genuine_decode_error_propagates_as_before(tmp_path, witness, monkeypatch):
    picks = tmp_path / "picks"
    picks.mkdir()
    (picks / "2026-06-20.json").write_bytes(b"\xff\xfe not utf-8 \xff")
    pa = _pa([(1, "2026-06-20", 1)])
    reads = []
    real_rb, real_rt = Path.read_bytes, Path.read_text
    monkeypatch.setattr(Path, "read_bytes", lambda self: (reads.append("b"), real_rb(self))[1])
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: (reads.append("t"), real_rt(self, *a, **k))[1])
    with (witnessed() if witness else contextlib.nullcontext()), pytest.raises(UnicodeDecodeError):
        C._resolve_pick_outcomes(picks, pa, TODAY, 30)
    assert len(reads) == 1                                          # one read either way


def test_a_collector_failure_never_drops_a_sample(world):
    picks, pa = world

    class NoAppend(list):
        def append(self, x):
            raise RuntimeError("synthetic collector failure")
    plain = C._resolve_pick_outcomes(picks, pa, TODAY, 30)
    with witnessed() as w:
        w.calibration.picks.records = NoAppend()
        w.calibration.bindings = NoAppend()
        assert C._resolve_pick_outcomes(picks, pa, TODAY, 30) == plain
    assert _picks(w) is None


def test_a_lost_confirmation_withholds_the_pick_inputs(world, monkeypatch):
    """Earned completeness: a record whose confirmation never ran does not count, so the part is withheld."""
    picks, pa = world
    real = W.Ledger.confirm
    monkeypatch.setattr(W.Ledger, "confirm", lambda self, used: None if getattr(used, "name", "") ==
                        "2026-06-24.json" else real(self, used))
    with witnessed() as w:
        C._resolve_pick_outcomes(picks, pa, TODAY, 30)
    assert len(w.calibration.picks.records) == 6 and _picks(w) is None


def test_a_changed_pick_listing_withholds_the_pick_inputs(world, monkeypatch):
    picks, pa = world
    real = W.names
    monkeypatch.setattr(W, "names", lambda d, pattern, count=None: real(d, pattern)[1:])
    with witnessed() as w:
        C._resolve_pick_outcomes(picks, pa, TODAY, 30)
    assert _picks(w) is None


def _fit_world(tmp_path, n=30, flip=False):
    picks = tmp_path / ("picks_flip" if flip else "picks")
    picks.mkdir()
    rows = []
    for i in range(n):
        d = f"2026-06-{1 + (i % 28):02d}"
        p = (0.6 if i % 2 else 0.9) if not flip else (0.9 if i % 2 else 0.6)
        _write(picks, f"{d}-{i:02d}.json".replace(f"-{i:02d}.json", "") + f"x{i:02d}.json",
               {"date": d, "result": "hit", "pick": {"batter_id": 100 + i, "p_game_hit": p}})
        rows.append((100 + i, d, i % 2))
    return picks, _pa(rows)


def _fit(picks, pa):
    with witnessed() as w:
        cal = C.fit_calibrator_from_picks(picks, pa, today=TODAY)
    return cal, w.calibration.fit


def test_fit_witness_records_n_fit_bindings_and_the_canonical_map(tmp_path):
    picks, pa = _fit_world(tmp_path)
    cal, w = _fit(picks, pa)
    plain = C.fit_calibrator_from_picks(picks, pa, today=TODAY)
    assert list(cal.X_thresholds_) == list(plain.X_thresholds_) and list(cal.y_thresholds_) == list(plain.y_thresholds_)
    assert w["status"] == "fitted" and w["n_fit"] == 30 == len(w["samples"])
    canon = lambda o: hashlib.sha256(json.dumps(o, sort_keys=True, separators=(",", ":"),   # noqa: E731
                                                allow_nan=False).encode("utf-8")).hexdigest()
    assert w["samples_sha256"] == canon(w["samples"])
    assert w["map"] == {"X_thresholds": cal.X_thresholds_.tolist(), "y_thresholds": cal.y_thresholds_.tolist(),
                        "increasing": bool(cal.increasing_), "out_of_bounds": "clip", "y_min": 0.0, "y_max": 1.0}
    assert w["map_sha256"] == canon(w["map"]) and len(w["map"]["X_thresholds"]) == 2       # 30 samples, 2 thresholds
    assert len(w["pick_inputs"]) == 30 and w["errors"] == []


def test_changing_historical_probabilities_changes_the_witness(tmp_path):
    a, pa = _fit_world(tmp_path)
    b, _ = _fit_world(tmp_path, flip=True)
    _, wa = _fit(a, pa)
    _, wb = _fit(b, pa)
    assert wa["samples_sha256"] != wb["samples_sha256"]
    assert [i["sha256"] for i in wa["pick_inputs"]] != [i["sha256"] for i in wb["pick_inputs"]]


def test_insufficient_support_is_witnessed_and_returns_none(tmp_path):
    picks, pa = _fit_world(tmp_path, n=5)
    cal, w = _fit(picks, pa)
    assert cal is None
    assert w["status"] == "insufficient_support" and w["n_fit"] == 5 and len(w["samples"]) == 5
    assert w["map"] is None


def test_no_sklearn_is_witnessed(tmp_path, monkeypatch):
    import builtins
    real = builtins.__import__

    def fake(name, *a, **k):
        if name.startswith("sklearn"):
            raise ImportError("no sklearn")
        return real(name, *a, **k)
    monkeypatch.setattr(builtins, "__import__", fake)
    cal, w = _fit(tmp_path, _pa([]))
    assert cal is None and w["status"] == "no_sklearn"


def test_a_map_hash_failure_returns_the_same_calibrator(tmp_path, monkeypatch):
    picks, pa = _fit_world(tmp_path)
    plain = C.fit_calibrator_from_picks(picks, pa, today=TODAY)

    def broken(*a, **k):
        raise ValueError("synthetic canonicalization failure")
    monkeypatch.setattr(W, "canon_sha256", broken)
    cal, w = _fit(picks, pa)
    assert list(cal.X_thresholds_) == list(plain.X_thresholds_)
    assert w["map_sha256"] is None and w["samples_sha256"] is None and w["errors"]


def test_mismatched_bindings_withhold_the_samples(tmp_path, monkeypatch):
    """Earned: the bindings stand only when their values are exactly the samples'."""
    picks, pa = _fit_world(tmp_path)
    real = W.pick_bind
    monkeypatch.setattr(W, "pick_bind", lambda f, d, slot, bid, sample: real(f, d, slot, bid, (sample[0], 1 - sample[1])))
    cal, w = _fit(picks, pa)
    assert cal is not None and w["samples"] is None and w["samples_sha256"] is None and w["errors"]


def test_canonical_hash_refuses_non_finite_values():
    with pytest.raises(ValueError):
        W.canon_sha256({"x": float("nan")})


def test_a_binding_carries_the_exact_appended_float(tmp_path):
    picks = tmp_path / "picks"
    picks.mkdir()
    _write(picks, "2026-06-20.json", _pick("2026-06-20", p=0.8123456789, bid=7))
    with witnessed() as w:
        samples = C._resolve_pick_outcomes(picks, _pa([(7, "2026-06-20", 1)]), TODAY, 30)
    b = w.calibration.bindings[0]
    assert samples == [(0.8123456789, 1)] and b["p"] == 0.8123456789 and b["y"] == 1


def test_n_fit_counts_samples_not_files(tmp_path):
    picks, pa = _fit_world(tmp_path, n=5)
    _write(picks, "2026-06-27x-malformed.json", "{not json")
    _write(picks, "2026-04-01x-old.json", _pick("2026-04-01", p=0.7, bid=1))
    cal, w = _fit(picks, pa)
    assert cal is None
    assert w["n_fit"] == 5 == len(w["samples"]) and len(w["pick_inputs"]) == 7
