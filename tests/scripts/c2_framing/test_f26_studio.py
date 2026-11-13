"""The studio runner's aggregate (X-37): it decides the test's disposition, so these pin what it refuses and the
direction of the comparison. The science it calls (f26.dispose, rank1) is tested in test_f26_rules.py."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from scripts.audit.c2_framing import f26 as F
from scripts.audit.c2_framing import f26_studio as ST

CAL = [f"2026-04-{d:02d}" for d in range(1, 21)]
PINS = {"pa_2026.parquet": "a" * 64}


def _seed_dir(out, seed, hits):
    """One finished seed: `hits` maps arm -> list of 0/1 over CAL."""
    d = out / f"seed_{seed}"
    d.mkdir(parents=True)
    for arm in F.ARMS:
        pd.DataFrame({"date": pd.to_datetime(CAL), "rank": 1, "actual_hit": hits[arm]}).to_parquet(
            d / f"profiles_{arm}_2026.parquet", index=False)
    units = [{"arm": a, "cpu_s": 3600.0} | ({} if a == "baseline" else {"catcher": {"counts": {"n": 1}}})
             for a in F.ARMS]
    (d / "manifest.json").write_text(json.dumps({"seed": seed, "input_pins": PINS, "calendar": CAL}))
    (d / "results.json").write_text(json.dumps({"seed": seed, "rank1": hits, "units": units,
                                                "arms": {a: {} for a in F.ARMS[1:]}}))
    return d


def _world(tmp_path, posted, n_seeds=10):
    inputs, out = tmp_path / "inputs", tmp_path / "out"
    inputs.mkdir()
    (inputs / ST.PREPARED).write_text(json.dumps({"pins": PINS}))
    base = [1, 0] * 10
    for seed in F.SEEDS[:n_seeds]:
        _seed_dir(out, seed, {"baseline": base, "A_posted": posted, "A_projected": base})
    return inputs, out


def test_a_better_posted_arm_reads_as_a_positive_delta(tmp_path):
    inputs, out = _world(tmp_path, posted=[1] * 20)            # A-posted hits every day; the baseline half of them
    agg = ST.aggregate(inputs, out)
    assert agg["A_posted"]["m"] == pytest.approx(0.5)
    assert agg["A_posted"]["disposition"] == "positive" and agg["A_posted"]["seeds_positive"] == 10
    assert agg["A_projected_descriptive"]["m"] == 0.0 and "disposition" not in agg["A_projected_descriptive"]
    assert agg["n_days"] == 20 and agg["total_cpu_h"] == pytest.approx(30.0)


def test_a_worse_posted_arm_reads_as_negative(tmp_path):
    inputs, out = _world(tmp_path, posted=[0] * 20)
    assert ST.aggregate(inputs, out)["A_posted"]["disposition"] == "negative"


def test_nine_seeds_refuse(tmp_path):
    inputs, out = _world(tmp_path, posted=[1] * 20, n_seeds=9)
    with pytest.raises(F.RunInvalid, match="stopped or unfinished"):
        ST.aggregate(inputs, out)


def test_a_stopped_seed_refuses(tmp_path):
    inputs, out = _world(tmp_path, posted=[1] * 20)
    (out / f"seed_{F.SEEDS[3]}" / "STOPPED.json").write_text("{}")
    with pytest.raises(F.RunInvalid, match="stopped or unfinished"):
        ST.aggregate(inputs, out)


def test_recorded_hits_that_the_profiles_do_not_reproduce_refuse(tmp_path):
    inputs, out = _world(tmp_path, posted=[1] * 20)
    p = out / f"seed_{F.SEEDS[0]}" / "results.json"
    res = json.loads(p.read_text())
    res["rank1"]["A_posted"][0] = 0
    p.write_text(json.dumps(res))
    with pytest.raises(F.RunInvalid, match="do not reproduce"):
        ST.aggregate(inputs, out)


def test_a_seed_run_on_other_inputs_refuses(tmp_path):
    inputs, out = _world(tmp_path, posted=[1] * 20)
    p = out / f"seed_{F.SEEDS[5]}" / "manifest.json"
    man = json.loads(p.read_text())
    man["input_pins"] = {"pa_2026.parquet": "b" * 64}
    p.write_text(json.dumps(man))
    with pytest.raises(F.RunInvalid, match="input pins"):
        ST.aggregate(inputs, out)


def test_a_seed_on_a_different_calendar_refuses(tmp_path):
    inputs, out = _world(tmp_path, posted=[1] * 20)
    d = out / f"seed_{F.SEEDS[2]}"
    short = CAL[:-1]
    for arm in F.ARMS:
        pd.DataFrame({"date": pd.to_datetime(short), "rank": 1, "actual_hit": 1}).to_parquet(
            d / f"profiles_{arm}_2026.parquet", index=False)
    (d / "manifest.json").write_text(json.dumps({"seed": F.SEEDS[2], "input_pins": PINS, "calendar": short}))
    res = json.loads((d / "results.json").read_text())
    res["rank1"] = {a: [1] * 19 for a in F.ARMS}
    (d / "results.json").write_text(json.dumps(res))
    with pytest.raises(F.RunInvalid, match="one nonempty evaluation calendar"):
        ST.aggregate(inputs, out)


def test_the_aggregate_is_written_once(tmp_path):
    inputs, out = _world(tmp_path, posted=[1] * 20)
    ST.aggregate(inputs, out)
    with pytest.raises(FileExistsError):
        ST.aggregate(inputs, out)


def test_a_seed_cannot_run_twice(tmp_path, monkeypatch):
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    inputs, out = _world(tmp_path, posted=[1] * 20)
    import bts.model.predict as P
    monkeypatch.setitem(P.LGB_PARAMS, "deterministic", True)
    monkeypatch.setitem(P.LGB_PARAMS, "force_row_wise", True)
    with pytest.raises(FileExistsError):
        ST.run(F.SEEDS[0], tmp_path, inputs, out)


def test_an_unregistered_seed_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    with pytest.raises(SystemExit, match="not a registered seed"):
        ST.run(7, tmp_path, tmp_path, tmp_path)


def test_the_calendar_comes_from_the_frozen_2026_file_alone(tmp_path):
    """The first real seed-1 start refused because the calendar was taken from the ten-season concat, where the
    resumed-portion flag is no longer boolean. The calendar must come from the pinned pa_2026 bytes."""
    import hashlib
    pa26 = pd.DataFrame({"date": pd.to_datetime(["2026-04-01", "2026-04-01", "2026-04-02", "2026-04-03"]),
                         "season": 2026, "is_resumed_portion": [False, False, False, True]})
    pa26.to_parquet(tmp_path / "pa_2026.parquet", index=False)
    pins = {"pa_2026.parquet": hashlib.sha256((tmp_path / "pa_2026.parquet").read_bytes()).hexdigest()}
    assert ST.calendar_2026(tmp_path, pins) == ["2026-04-01", "2026-04-02"]      # 04-03 has only resumed rows
    older = pd.DataFrame({"date": pd.to_datetime(["2025-04-01"]), "season": 2025})
    concat = pd.concat([older, pa26], ignore_index=True)
    with pytest.raises(F.InputRefused):                                         # why the concat cannot be used
        F.expected_calendar(concat[concat["season"] == 2026])
