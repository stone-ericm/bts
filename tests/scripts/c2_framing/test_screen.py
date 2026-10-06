"""C2 side item (e): the catcher-grouped framing screen's research code (scripts/audit/c2_framing/screen.py)."""
import math

import numpy as np
import pandas as pd
import pytest

from scripts.audit.c2_framing import screen as S


def _frame(rows):
    return pd.DataFrame(rows, columns=["pitcher_id", "fielding_catcher_id", "date", "pa_borderline_csr", "season"])


def _reference(df, key):
    """An independent per-row implementation: for each row, the mean over the key's strictly earlier dates of that
    date's mean CSR, if at least 5 earlier dates have a non-null daily mean."""
    out = []
    for _, r in df.iterrows():
        k = r[key]
        if pd.isna(k):
            out.append(np.nan)
            continue
        same = df[df[key] == k]
        daily = same.groupby("date")["pa_borderline_csr"].mean().sort_index()
        earlier = daily[daily.index < r["date"]].dropna()
        out.append(float(earlier.mean()) if len(earlier) >= 5 else np.nan)
    return np.array(out, dtype=float)


@pytest.fixture
def pas():
    rng = np.random.default_rng(3)
    rows = []
    dates = [f"2019-04-{d:02d}" for d in range(1, 21)]
    for i, d in enumerate(dates):
        for pid, cid in ((10, 900), (11, 900), (12, 901), (13, np.nan)):
            for _ in range(3):
                csr = np.nan if rng.random() < 0.15 else round(float(rng.random()), 4)
                rows.append([pid, cid, d, csr, 2019])
    return _frame(rows)


def test_framing_by_any_key_matches_an_independent_reference(pas):
    for key in ("pitcher_id", "fielding_catcher_id"):
        out = S.framing_by(pas, key, "f")
        assert len(out) == len(pas) and list(out["pitcher_id"]) == list(pas["pitcher_id"])
        np.testing.assert_allclose(out["f"].to_numpy(float), _reference(pas, key), rtol=0, atol=1e-12, equal_nan=True)


def test_a_missing_catcher_gets_nan_and_never_joins_another(pas):
    out = S.framing_by(pas, "fielding_catcher_id", "f")
    assert out.loc[pas["fielding_catcher_id"].isna().to_numpy(), "f"].isna().all()
    assert out.loc[pas["fielding_catcher_id"].notna().to_numpy(), "f"].notna().any()


def test_the_feature_uses_only_strictly_earlier_dates(pas):
    base = S.framing_by(pas, "fielding_catcher_id", "f")
    later = pas.copy()
    later.loc[later["date"] >= "2019-04-15", "pa_borderline_csr"] = 0.999      # change the future only
    out = S.framing_by(later, "fielding_catcher_id", "f")
    upto = (pas["date"] <= "2019-04-15").to_numpy()
    np.testing.assert_array_equal(out.loc[upto, "f"].to_numpy(float), base.loc[upto, "f"].to_numpy(float))


def test_the_self_check_compares_against_the_production_pitcher_feature(pas):
    df = S.framing_by(pas, "pitcher_id", S.OLD_COL)
    assert S.self_check(df)["identical"] is True
    bad = df.copy()
    bad.loc[bad.index[-1], S.OLD_COL] = 0.5
    chk = S.self_check(bad)
    assert chk["identical"] is False


def test_framing_by_mirrors_the_production_pitcher_block_text():
    """framing_by is a parameterised copy of compute_all_features' pitcher block; if that block changes, this fails
    (the run's self-check then compares values on the real inputs)."""
    import inspect
    from bts.features import compute as C
    text = inspect.getsource(C.compute_all_features)
    for line in ('date_framing = df.groupby(["pitcher_id", "date"])["pa_borderline_csr"].mean().reset_index()',
                 'date_framing.columns = ["pitcher_id", "date", "date_csr"]',
                 'date_framing = date_framing.sort_values(["pitcher_id", "date"])',
                 'date_framing["pitcher_catcher_framing"] = date_framing.groupby("pitcher_id")["date_csr"].transform(',
                 "lambda x: x.shift(1).expanding(min_periods=5).mean()",
                 'date_framing[["pitcher_id", "date", "pitcher_catcher_framing"]]',
                 '.drop_duplicates(subset=["pitcher_id", "date"]),',
                 'on=["pitcher_id", "date"], how="left",'):
        assert line in text, line


def test_variant_bases():
    from bts.features.compute import FEATURE_COLS
    assert S.base_cols("baseline") == list(FEATURE_COLS)
    a = S.base_cols("A")
    assert len(a) == len(FEATURE_COLS) and S.NEW_COL in a and S.OLD_COL not in a
    assert a.index(S.NEW_COL) == FEATURE_COLS.index(S.OLD_COL)
    assert S.base_cols("B") == list(FEATURE_COLS) + [S.NEW_COL]
    with pytest.raises(ValueError):
        S.base_cols("C")


def test_blend_configs_rewrite_the_base_and_keep_each_configs_extras():
    from bts.features.compute import FEATURE_COLS
    from bts.model.predict import BLEND_CONFIGS
    for v in ("baseline", "A", "B"):
        cfgs = S.blend_configs(v)
        assert [c[0] for c in cfgs] == [c[0] for c in BLEND_CONFIGS]
        for new, old in zip(cfgs, BLEND_CONFIGS):
            extras = [c for c in old[1] if c not in FEATURE_COLS]
            assert new[1] == S.base_cols(v) + extras
    assert [c[1] for c in S.blend_configs("baseline")] == [c[1] for c in BLEND_CONFIGS]


def _seed(d24, d25, passed):
    return {"p_at_1_delta": {"2024": d24, "2025": d25}, "passed": passed, "reason": ""}


def test_disposition_positive_needs_both_seasons_the_practical_size_t_and_a_majority_of_passes():
    rows = [_seed(0.006, 0.005, True), _seed(0.005, 0.004, True), _seed(0.004, 0.006, False)]
    d = S.disposition(rows)
    assert d["disposition"] == "positive" and d["per_seed_passes"] == 2 and d["n_seeds"] == 3
    assert math.isclose(d["seed_level_mean"], (0.0055 + 0.0045 + 0.005) / 3)


@pytest.mark.parametrize("rows, why", [
    ([_seed(0.010, -0.001, True)] * 3, "one season's mean not above zero (every other rule met)"),
    ([_seed(0.002, 0.002, True)] * 3, "below the practical size"),
    ([_seed(0.012, 0.010, True), _seed(-0.002, -0.004, True), _seed(0.004, 0.006, True)], "t below 1.5"),
    ([_seed(0.006, 0.005, False), _seed(0.005, 0.004, False), _seed(0.004, 0.006, True)], "a minority of passes"),
])
def test_not_positive(rows, why):
    assert S.disposition(rows)["disposition"] == "inconclusive", why


def test_disposition_negative():
    rows = [_seed(-0.005, 0.0, False), _seed(0.0, -0.002, False), _seed(0.001, -0.001, True)]
    assert S.disposition(rows)["disposition"] == "negative"


def test_a_non_positive_mean_with_passing_seeds_is_inconclusive_not_negative():
    rows = [_seed(-0.005, 0.0, True), _seed(0.0, -0.002, True), _seed(0.001, -0.001, False)]
    assert S.disposition(rows)["disposition"] == "inconclusive"


def test_disposition_incomplete():
    assert S.disposition([])["disposition"] == "incomplete"
    assert S.disposition([{"p_at_1_delta": {"2024": 0.1}, "passed": True}])["disposition"] == "incomplete"


def test_t_edges():
    assert S._t([0.01, 0.01, 0.01]) == math.inf and S._t([-0.01] * 3) == -math.inf and S._t([0.0] * 3) == 0.0
    assert math.isclose(S._t([1.0, 2.0, 3.0]), 2.0 / (1.0 / math.sqrt(3)))


def test_first_unit_stop():
    assert not S.first_unit_stop(7.5 * 3600) and S.first_unit_stop(7.5 * 3600 + 1)


def test_pre_registered_constants():
    assert S.SEASONS_IN == tuple(range(2017, 2026)) and 2026 not in S.SEASONS_IN
    assert S.TEST_SEASONS == (2024, 2025) and S.BASIS == "estimated_pa" and S.RETRAIN_EVERY == 7
    import json
    from pathlib import Path
    seeds = json.loads((Path(__file__).resolve().parents[3] / "data/seed_sets/canonical-n10.json").read_text())
    assert S.STAGE_ONE_SEEDS == tuple(seeds["seeds"][:3])


def test_load_inputs_reads_only_the_pinned_files_from_their_hashed_bytes(tmp_path):
    import hashlib
    pins = {}
    for s in S.SEASONS_IN:
        p = tmp_path / f"pa_{s}.parquet"
        pd.DataFrame({"season": [s], "x": [s * 2]}).to_parquet(p)
        pins[p.name] = hashlib.sha256(p.read_bytes()).hexdigest()
    pd.DataFrame({"season": [2026], "x": [1]}).to_parquet(tmp_path / "pa_2026.parquet")
    df = S.load_inputs(tmp_path, pins)
    assert sorted(df["season"]) == list(S.SEASONS_IN)
    pd.DataFrame({"season": [2019], "x": [-1]}).to_parquet(tmp_path / "pa_2019.parquet")     # valid, but not pinned
    from scripts.audit.c1.admission import ProvenanceError
    with pytest.raises(ProvenanceError, match="not the pinned"):
        S.load_inputs(tmp_path, pins)


def test_run_refuses_without_the_deterministic_flag_or_a_stage_one_seed(monkeypatch, tmp_path):
    monkeypatch.delenv("BTS_LGBM_DETERMINISTIC", raising=False)
    with pytest.raises(SystemExit, match="DETERMINISTIC"):
        S.run(S.STAGE_ONE_SEEDS[0], tmp_path, tmp_path)
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    with pytest.raises(SystemExit, match="stage-one seed"):
        S.run(42, tmp_path, tmp_path)


def test_run_refuses_without_an_admission(monkeypatch, tmp_path):
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    with pytest.raises(SystemExit, match="refusing"):
        S.run(S.STAGE_ONE_SEEDS[0], tmp_path, tmp_path)


def test_aggregate(tmp_path):
    import json
    dirs = []
    for i, seed in enumerate(S.STAGE_ONE_SEEDS):
        d = tmp_path / f"r{i}"
        d.mkdir()
        (d / "results.json").write_text(json.dumps({"seed": seed, "total_cpu_s": 3600.0, "variants": {
            "A": _seed(0.006, 0.005, True), "B": _seed(-0.004, -0.003, False)}}))
        dirs.append(d)
    agg = S.aggregate(dirs)
    assert agg["seeds"] == list(S.STAGE_ONE_SEEDS) and math.isclose(agg["total_cpu_h"], 3.0)
    assert agg["variants"]["A"]["disposition"] == "positive" and agg["variants"]["B"]["disposition"] == "negative"


# ---------------------------------------------------------------- run() end to end (stubbed admission, inputs, features
# and walk-forward; the claim, manifest, self-check, stops, scoring and results are real)

def _stub_df():
    rng = np.random.default_rng(11)
    rows = []
    for season in (2023, 2024, 2025):
        for d in range(12):
            for pid, cid in ((10, 900), (11, 901), (12, np.nan)):
                rows.append({"season": season, "date": f"{season}-05-{d + 1:02d}", "pitcher_id": pid,
                             "fielding_catcher_id": cid, "pa_borderline_csr": float(rng.random())})
    df = pd.DataFrame(rows)
    return S.framing_by(df, "pitcher_id", S.OLD_COL)


def _stub_walk_forward(calls):
    def wf(df, season, retrain_every, blend_configs, game_probability_mode):
        calls.append((season, game_probability_mode, retrain_every, tuple(blend_configs[0][1])))
        is_a = S.NEW_COL in blend_configs[0][1] and S.OLD_COL not in blend_configs[0][1]
        rng = np.random.default_rng(season + (7 if is_a else 0))
        rows = []
        for d in range(40):
            for rank in range(1, 11):
                rows.append({"date": f"{season}-06-{(d % 28) + 1:02d}" if d < 28 else f"{season}-07-{d - 27:02d}",
                             "rank": rank, "batter_id": 1000 + rank, "p_game_hit": 0.8 - 0.01 * rank,
                             "actual_hit": int(rng.random() < (0.85 if is_a else 0.75)), "n_pas": 4,
                             "game_pk": 5000 + d})
        return pd.DataFrame(rows)
    return wf


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    import bts.features.compute as FC
    import bts.model.predict as PR
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    monkeypatch.setitem(PR.LGB_PARAMS, "deterministic", True)       # as built at import under the flag
    monkeypatch.setitem(PR.LGB_PARAMS, "force_row_wise", True)
    monkeypatch.setattr(S, "admission_gate", lambda: ("a" * 40, {"input_pins": {"pa_2019.parquet": "b" * 64}},
                                                       {"report_sha256": "c" * 64}))
    monkeypatch.setattr(S, "load_inputs", lambda data_dir, pins: _stub_df())
    monkeypatch.setattr(FC, "compute_all_features", lambda df: df)
    out = tmp_path / "out"
    out.mkdir()
    return out


def _run_dir(out, seed):
    dirs = [p for p in (out / f"seed_{seed}").iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def test_run_end_to_end(stubbed):
    import json
    calls = []
    seed = S.STAGE_ONE_SEEDS[0]
    assert S.run(seed, stubbed, stubbed, walk_forward=_stub_walk_forward(calls)) == 0
    d = _run_dir(stubbed, seed)
    assert (d / "CLAIM.json").exists()
    man = json.loads((d / "manifest.json").read_text())
    assert man["seed"] == seed and man["basis"] == "estimated_pa" and man["self_check"]["identical"] is True
    assert man["env"]["BTS_LGBM_RANDOM_STATE"] == str(seed) and man["env"]["BTS_LGBM_DETERMINISTIC"] == "1"
    assert man["coverage"][S.NEW_COL]["2024"]["rows"] == 36
    assert [c[0] for c in calls] == [2024, 2025] * 3 and {c[1] for c in calls} == {"estimated_pa"}
    assert {c[2] for c in calls} == {7}
    res = json.loads((d / "results.json").read_text())
    assert set(res["variants"]) == {"A", "B"} and len(res["units"]) == 6
    assert set(res["variants"]["A"]["p_at_1_delta"]) == {"2024", "2025"}
    assert res["variants"]["A"]["p_at_1_delta"]["2024"] > 0          # the stub makes A better
    assert res["variants"]["B"]["p_at_1_delta"] == {"2024": 0.0, "2025": 0.0}
    for v in ("baseline", "A", "B"):
        for s in (2024, 2025):
            assert (d / f"profiles_{v}_{s}.parquet").exists()


def test_a_second_run_of_the_same_seed_is_refused(stubbed):
    seed = S.STAGE_ONE_SEEDS[1]
    assert S.run(seed, stubbed, stubbed, walk_forward=_stub_walk_forward([])) == 0
    with pytest.raises(SystemExit, match="claimed run"):
        S.run(seed, stubbed, stubbed, walk_forward=_stub_walk_forward([]))


def test_the_self_check_stops_the_run_before_any_walk_forward(stubbed, monkeypatch):
    import json
    bad = _stub_df()
    bad.loc[bad.index[-1], S.OLD_COL] = 0.123
    monkeypatch.setattr(S, "load_inputs", lambda data_dir, pins: bad)
    calls = []
    assert S.run(S.STAGE_ONE_SEEDS[2], stubbed, stubbed, walk_forward=_stub_walk_forward(calls)) == 4
    d = _run_dir(stubbed, S.STAGE_ONE_SEEDS[2])
    assert calls == [] and json.loads((d / "STOPPED.json").read_text())["reason"] == "self_check"


def test_an_expensive_first_walk_forward_stops_the_run(stubbed, monkeypatch):
    import json
    ticks = iter([0.0, 0.0, 0.0, 8 * 3600.0] + [8 * 3600.0] * 50)
    monkeypatch.setattr(S, "cpu_seconds", lambda: next(ticks))
    calls = []
    assert S.run(S.STAGE_ONE_SEEDS[0], stubbed, stubbed, walk_forward=_stub_walk_forward(calls)) == 3
    d = _run_dir(stubbed, S.STAGE_ONE_SEEDS[0])
    stop = json.loads((d / "STOPPED.json").read_text())
    assert stop["reason"] == "first_unit_cpu" and len(calls) == 1 and not (d / "results.json").exists()


def test_run_refuses_when_lightgbm_params_lack_the_deterministic_flags(stubbed, monkeypatch):
    import bts.model.predict as PR
    monkeypatch.setitem(PR.LGB_PARAMS, "deterministic", False)
    with pytest.raises(SystemExit, match="deterministic flags"):
        S.run(S.STAGE_ONE_SEEDS[0], stubbed, stubbed, walk_forward=_stub_walk_forward([]))


def test_the_manifest_records_the_effective_lightgbm_params(stubbed):
    import json
    assert S.run(S.STAGE_ONE_SEEDS[0], stubbed, stubbed, walk_forward=_stub_walk_forward([])) == 0
    man = json.loads((_run_dir(stubbed, S.STAGE_ONE_SEEDS[0]) / "manifest.json").read_text())
    assert man["lgb_params"]["deterministic"] is True and man["lgb_params"]["force_row_wise"] is True
