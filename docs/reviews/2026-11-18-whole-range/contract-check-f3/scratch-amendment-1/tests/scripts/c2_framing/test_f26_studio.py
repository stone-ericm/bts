"""The studio runner (X-37) on a stubbed world: the feature computation and the walk-forward are stubbed (the same
stubs as test_f26_run.py); the runner's own files and the aggregate's acceptance checks are real.

The aggregate decides the test's disposition, so most of this file is what it must REFUSE. B1-B3 are the blocking
findings of its review (`docs/audit/2026-11-13-c2-framing-2026-studio-codex-r1.md`): each test was red before the fix.
"""
from __future__ import annotations

import hashlib
import json
import shutil

import numpy as np
import pandas as pd
import pytest

from scripts.audit.c2_framing import f26 as F
from scripts.audit.c2_framing import f26_studio as ST
from scripts.audit.c2_framing import screen as S
from tests.scripts.c2_framing import test_f26_run as T


def _patch(mp, df):
    import bts.features.compute as FC
    import bts.features.park_drag as PD
    import bts.model.predict as PR
    mp.setenv("BTS_LGBM_DETERMINISTIC", "1")
    mp.setitem(PR.LGB_PARAMS, "deterministic", True)
    mp.setitem(PR.LGB_PARAMS, "force_row_wise", True)
    mp.setattr(FC, "_build_probable_pitcher_lookup", FC._build_probable_pitcher_lookup)   # restored after each test
    mp.setattr(PD, "attach_park_drag", PD.attach_park_drag)
    mp.setattr(F, "load_inputs", lambda data_dir, inputs_dir, pins: df.copy())
    mp.setattr(FC, "compute_all_features", lambda d: d)
    mp.setattr(S, "SCORING", {"mc_trials": 200, "season_length": 180})


def _build(root, *, no_catchers=False):
    """Ten finished seeds over the stubbed world, built once per variant."""
    df, table = T._world()
    if no_catchers:
        for r in table:
            r["catcher_id"], r["reason"] = None, "no_candidate"
    inputs, out = root / "inputs", root / "out"
    inputs.mkdir(parents=True)
    games = {s: sorted(int(g) for g in df.loc[df["season"] == s, "game_pk"].unique()) for s in (2025, 2026)}
    files = {F.FROZEN_PA: T._parquet(df[df["season"] == 2026]), F.LOOKUP_NAME: b'{"1":{"away":5}}\n',
             F.TABLE_NAME: F.canonical(table), F.SOURCES_NAME: F.canonical(T._sources(games))}
    for name, b in files.items():
        (inputs / name).write_bytes(b)
    pins = {name: hashlib.sha256(b).hexdigest() for name, b in files.items()}
    (inputs / ST.PREPARED).write_bytes(F.canonical({"pins": pins}))
    with pytest.MonkeyPatch.context() as mp:
        _patch(mp, df)
        for seed in F.SEEDS:
            assert ST.run(seed, root, inputs, out, walk_forward=T._stub_walk_forward([])) == 0
    return df


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("f26_studio")
    return root, _build(root)


@pytest.fixture(scope="module")
def built_no_catchers(tmp_path_factory):
    root = tmp_path_factory.mktemp("f26_studio_nocatch")
    return root, _build(root, no_catchers=True)


def _copy(built, tmp_path, monkeypatch):
    root, df = built
    shutil.copytree(root / "inputs", tmp_path / "inputs")
    shutil.copytree(root / "out", tmp_path / "out")
    _patch(monkeypatch, df)
    return {"data": tmp_path, "inputs": tmp_path / "inputs", "out": tmp_path / "out", "df": df}


@pytest.fixture
def world(built, tmp_path, monkeypatch):
    return _copy(built, tmp_path, monkeypatch)


def _agg(w):
    return ST.aggregate(w["data"], w["inputs"], w["out"])


def _seed(w, k=0):
    return w["out"] / f"seed_{F.SEEDS[k]}"


def _edit_json(path, fn):
    obj = json.loads(path.read_text())
    fn(obj)
    path.write_text(json.dumps(obj))


def _edit_profile(d, arm, fn):
    p = d / f"profiles_{arm}_2026.parquet"
    frame = pd.read_parquet(p)
    frame = fn(frame)
    frame.to_parquet(p, index=False)
    return frame


# ---------------------------------------------------------------- the accepted case

def test_ten_valid_seeds_aggregate(world):
    agg = _agg(world)
    calendar = F.expected_calendar(world["df"][world["df"]["season"] == 2026])
    assert agg["n_days"] == len(calendar) and agg["calendar"] == [calendar[0], calendar[-1]]
    assert agg["A_posted"]["disposition"] in ("positive", "negative", "inconclusive")
    assert "disposition" not in agg["A_projected_descriptive"] and "m" in agg["A_projected_descriptive"]
    assert len(agg["rank_ties_by_seed"]) == 10 and set(agg["rank_ties_by_seed"][0]) == set(F.ARMS)
    assert agg["total_cpu_h"] >= agg["walk_forward_cpu_h"] >= 0
    assert (world["out"] / "aggregate.json").exists()


def test_the_delta_is_the_arm_minus_the_baseline():
    rank1s = [{"baseline": [1, 0, 0], "A_posted": [1, 1, 0]}, {"baseline": [0, 0, 1], "A_posted": [0, 0, 0]}]
    assert ST.deltas(rank1s, "A_posted").tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, -1.0]]


def test_the_aggregate_is_written_once(world):
    _agg(world)
    with pytest.raises(FileExistsError):
        _agg(world)


# ---------------------------------------------------------------- completeness: exactly the ten seeds, each finished

def test_nine_seeds_refuse(world):
    shutil.rmtree(_seed(world, 9))
    with pytest.raises(F.RunInvalid, match="stopped or unfinished"):
        _agg(world)


def test_a_stopped_seed_refuses(world):
    (_seed(world, 3) / "STOPPED.json").write_text("{}")
    with pytest.raises(F.RunInvalid, match="stopped or unfinished"):
        _agg(world)


def test_a_seed_run_on_other_inputs_refuses(world):
    _edit_json(_seed(world, 5) / "manifest.json", lambda m: m["input_pins"].update({F.FROZEN_PA: "b" * 64}))
    with pytest.raises(F.RunInvalid, match="input pins"):
        _agg(world)


def test_a_seed_under_other_settings_refuses(world):
    _edit_json(_seed(world, 2) / "manifest.json", lambda m: m.update({"retrain_every": 1}))
    with pytest.raises(F.RunInvalid, match="registered settings"):
        _agg(world)


def test_recorded_hits_that_the_profiles_do_not_reproduce_refuse(world):
    def flip(r):
        r["rank1"]["A_posted"][0] = 1 - r["rank1"]["A_posted"][0]
    _edit_json(_seed(world) / "results.json", flip)
    with pytest.raises(F.RunInvalid, match="do not reproduce"):
        _agg(world)


# ---------------------------------------------------------------- B1: an unavailable catcher contrast

def test_B1_no_finite_posted_catcher_value_refuses(built_no_catchers, tmp_path, monkeypatch):
    w = _copy(built_no_catchers, tmp_path, monkeypatch)
    with pytest.raises(F.RunInvalid, match="catcher contrast is unavailable"):
        _agg(w)


def test_B1_catcher_evidence_that_the_inputs_do_not_reproduce_refuses(world):
    d = _seed(world, 1)
    p = d / "catcher_A_posted_2026.parquet"
    ev = pd.read_parquet(p)
    ev.loc[ev.index[0], "catcher_id"] = 999999
    ev.to_parquet(p, index=False)
    with pytest.raises(F.RunInvalid, match="A_posted"):
        _agg(world)


# ---------------------------------------------------------------- B2: a date every seed omits

def test_B2_a_date_missing_from_every_seed_refuses(world):
    """All ten seeds consistently omit the last date: manifests, profiles and recorded hits agree with each other,
    and the calendar re-derived from the pinned pa_2026 still has it."""
    for k in range(10):
        d = _seed(world, k)
        man = json.loads((d / "manifest.json").read_text())
        last = man["calendar"][-1]
        _edit_json(d / "manifest.json", lambda m: m["calendar"].pop())
        for arm in F.ARMS:
            _edit_profile(d, arm, lambda f, last=last: f[pd.to_datetime(f["date"]).dt.date.astype(str) != last])
            _edit_json(d / "results.json", lambda r, arm=arm: r["rank1"][arm].pop())
    with pytest.raises(F.RunInvalid, match="calendar"):
        _agg(world)


# ---------------------------------------------------------------- B3: profiles that are not valid evidence

def _recohere(d, arm, frame):
    """Make the recorded rank-1 hits agree with an edited profile, so only the profile checks can refuse it."""
    top = frame[frame["rank"] == 1].sort_values("date")
    _edit_json(d / "results.json", lambda r: r["rank1"].update({arm: [int(h) for h in top["actual_hit"]]}))


def test_B3_nonbinary_hits_refuse(world):
    d = _seed(world, 4)

    def edit(f):
        f.loc[f["rank"] == 1, "actual_hit"] = 2
        return f
    _recohere(d, "A_posted", _edit_profile(d, "A_posted", edit))
    with pytest.raises(F.RunInvalid, match="hits are not 0/1"):
        _agg(world)


def test_B3_missing_probabilities_refuse(world):
    d = _seed(world, 4)

    def edit(f):
        f["p_game_hit"] = f["p_game_hit"].astype(float)
        f.loc[f.index[0], "p_game_hit"] = np.nan
        return f
    _edit_profile(d, "A_posted", edit)
    with pytest.raises(F.RunInvalid, match="missing values|probabilities"):
        _agg(world)


def test_B3_a_batter_game_absent_from_the_pinned_inputs_refuses(world):
    d = _seed(world, 4)

    def edit(f):
        f.loc[f.index[0], "batter_id"] = 987654
        return f
    _edit_profile(d, "baseline", edit)
    with pytest.raises(F.RunInvalid, match="not a scoreable batter-game"):
        _agg(world)


def test_B3_ranks_that_do_not_follow_the_probabilities_refuse(world):
    d = _seed(world, 6)

    def edit(f):
        day = f["date"].iloc[0]
        m = f["date"] == day
        f.loc[m, "p_game_hit"] = np.sort(f.loc[m, "p_game_hit"].to_numpy())      # ascending against rank
        return f
    _edit_profile(d, "A_projected", edit)
    with pytest.raises(F.RunInvalid, match="do not follow the retained probabilities"):
        _agg(world)


def test_B3_a_scorecard_the_profiles_do_not_reproduce_refuses(world):
    _edit_json(_seed(world, 7) / "scorecard_A_posted.json",
               lambda c: c["p_at_1_by_season"].update({k: 0.123 for k in c["p_at_1_by_season"]}))
    with pytest.raises(F.RunInvalid, match="scorecard"):
        _agg(world)


def test_B3_a_stored_summary_the_scorecards_do_not_reproduce_refuses(world):
    _edit_json(_seed(world, 8) / "results.json", lambda r: r["arms"]["A_posted"].update({"mean_max_streak": 99.0}))
    with pytest.raises(F.RunInvalid, match="summary"):
        _agg(world)


# ---------------------------------------------------------------- the run's own refusals

def test_a_seed_cannot_run_twice(world):
    with pytest.raises(FileExistsError):
        ST.run(F.SEEDS[0], world["data"], world["inputs"], world["out"])


def test_an_unregistered_seed_refuses(tmp_path, monkeypatch):
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    with pytest.raises(SystemExit, match="not a registered seed"):
        ST.run(7, tmp_path, tmp_path, tmp_path)


def test_the_calendar_comes_from_the_frozen_2026_file_alone(tmp_path):
    """The first real seed-1 start refused because the calendar was taken from the ten-season concat, where the
    resumed-portion flag is no longer boolean. The calendar must come from the pinned pa_2026 bytes."""
    pa26 = pd.DataFrame({"date": pd.to_datetime(["2026-04-01", "2026-04-01", "2026-04-02", "2026-04-03"]),
                         "season": 2026, "is_resumed_portion": [False, False, False, True]})
    pa26.to_parquet(tmp_path / "pa_2026.parquet", index=False)
    pins = {"pa_2026.parquet": hashlib.sha256((tmp_path / "pa_2026.parquet").read_bytes()).hexdigest()}
    assert ST.calendar_2026(tmp_path, pins) == ["2026-04-01", "2026-04-02"]      # 04-03 has only resumed rows
    older = pd.DataFrame({"date": pd.to_datetime(["2025-04-01"]), "season": 2025})
    concat = pd.concat([older, pa26], ignore_index=True)
    with pytest.raises(F.InputRefused):                                         # why the concat cannot be used
        F.expected_calendar(concat[concat["season"] == 2026])
