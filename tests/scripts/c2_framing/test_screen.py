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
    good = _seed(0.006, 0.005, True)
    for n in (1, 2, 4):                                   # r1 B2: only exactly the three registered seeds
        assert S.disposition([good] * n)["disposition"] == "incomplete"


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


# ---------------------------------------------------------------- B6: the feature settings

def test_check_settings_requires_the_registered_values(monkeypatch):
    import bts.features.compute as C
    assert S.check_settings() == {"ROOKIE_GATE_K": 20, "PITCHER_HR_30G_MIN_PERIODS": 7}
    monkeypatch.setattr(C, "ROOKIE_GATE_K", 0)
    with pytest.raises(SystemExit, match="feature settings"):
        S.check_settings()


# ---------------------------------------------------------------- B1: closed inputs

def test_freeze_lookup_keeps_only_the_given_games_canonically():
    import json
    cache = json.dumps({"3": {"away": 1}, "1": {"away": 2}, "2026000001": {"away": 9}, "2": {"home": 3}}).encode()
    out = S.freeze_lookup(cache, [1, 2, 7])
    assert out == b'{"1":{"away":2},"2":{"home":3}}\n'
    assert S.frozen_lookup(out) == {1: {"away": 2}, 2: {"home": 3}}


def test_the_closed_inputs_replace_the_live_lookup_and_park_drag(tmp_path, monkeypatch):
    import json
    import bts.features.compute as C
    import bts.features.park_drag as PD
    monkeypatch.setattr(C, "_build_probable_pitcher_lookup", C._build_probable_pitcher_lookup)
    monkeypatch.setattr(PD, "attach_park_drag", PD.attach_park_drag)
    # canaries the live builder would read: a 2026 raw feed and a changed cache, in the working directory
    monkeypatch.chdir(tmp_path)
    (tmp_path / "data" / "raw" / "2026").mkdir(parents=True)
    (tmp_path / "data" / "raw" / "2026" / "999000.json").write_text(json.dumps({"gameData": {
        "probablePitchers": {"away": {"id": 7}}, "teams": {"away": {"id": 1}, "home": {"id": 2}}}}))
    (tmp_path / "data" / "models").mkdir(parents=True)
    (tmp_path / "data" / "models" / "probable_pitcher_lookup.json").write_text(json.dumps({"5": {"away": 99}}))
    reads = []
    real = __import__("pathlib").Path.read_text
    monkeypatch.setattr(__import__("pathlib").Path, "read_text",
                        lambda self, *a, **k: (reads.append(str(self)), real(self, *a, **k))[1])
    S.install_closed_inputs({1: {"away": 2}})
    assert C._build_probable_pitcher_lookup() == {1: {"away": 2}}
    assert reads == []                                     # neither the cache nor any raw feed was read
    assert not (tmp_path / "data" / "models" / "probable_pitcher_lookup.json").read_text().startswith('{"1"')
    table_reads = []
    monkeypatch.setattr(PD, "get_table", lambda *a, **k: table_reads.append(1))
    df = pd.DataFrame({"venue_id": [1], "date": ["2024-05-01"]})
    out = PD.attach_park_drag(df)
    assert out["park_drag_delta"].isna().all() and "park_drag_delta" not in df
    assert table_reads == []                                 # the external table is never consulted


def test_admission_requires_exactly_the_ten_pins(monkeypatch, tmp_path):
    import json
    from scripts.audit.c1 import admission as A
    adm = tmp_path / "admission.json"
    monkeypatch.setattr(A, "REPO", tmp_path)
    monkeypatch.setattr(S, "ADMISSION_REL", "admission.json")
    adm.write_text(json.dumps({"input_pins": {f"pa_{s}.parquet": "a" * 64 for s in S.SEASONS_IN}}))
    with pytest.raises(SystemExit, match="exactly"):
        S.admission_gate()                                 # the frozen lookup's pin is missing


# ---------------------------------------------------------------- B4: labels by the BTS scoring rule

def _pa_labels_frame():
    return pd.DataFrame({
        "batter_id": [1, 1, 2, 3, 3],
        "game_pk": [10, 10, 10, 11, 11],
        "is_hit": [0, 1, 1, 0, 0],
        "is_resumed_portion": [False, True, False, True, True],
    })


def test_original_portion_labels_exclude_the_resumed_portion():
    lab = S.original_portion_labels(_pa_labels_frame())
    assert lab.to_dict() == {(1, 10): 0, (2, 10): 1}       # batter 3 has only resumed PAs: no label


def test_relabel_rebuilds_labels_and_drops_void_rows():
    labels = S.original_portion_labels(_pa_labels_frame())
    prof = pd.DataFrame({"date": ["2024-05-01"] * 3, "rank": [1, 2, 3], "batter_id": [3, 1, 2],
                         "game_pk": [11, 10, 10], "p_game_hit": [0.9, 0.8, 0.7], "actual_hit": [0, 1, 1],
                         "n_pas": [2, 2, 1]})
    out, counts = S.relabel(prof, labels)
    assert counts == {"changed": 1, "void_dropped": 1}
    assert list(out["batter_id"]) == [1, 2] and list(out["rank"]) == [1, 2]
    assert list(out["actual_hit"]) == [0, 1]               # batter 1's resumed-only hit no longer counts


def test_resumed_counts():
    df = _pa_labels_frame().assign(season=[2024, 2024, 2024, 2025, 2025])
    assert S.resumed_counts(df) == {"2024": 1, "2025": 2}


# ---------------------------------------------------------------- B3: seed order, release, namespace

IDENT = {k: f"{k}-value" for k in ("review_report", "review_report_sha256", "reviewed_commit", "exposure_commit",
                                     "admission_sha256")}


def _release_row(run_name, source="Eric 2026-10-08, relayed", budget="30"):
    return (f"| C2-framing-release-seeds-2-3 | the seed-1 report | **RULED 2026-10-08 (Eric): RELEASE seeds 2–3 of the "
            f"framing screen after seed-1 run `{run_name}`; declared budget {budget} CPU-hours per seed** | {source} |")


def test_release_needs_the_exact_ruling_eric_and_a_positive_budget():
    run = "aaaaaaa-20261007T000000Z"
    assert S.release(_release_row(run)) == (run, 30.0)
    assert S.release(_release_row(run, source="Ericsson, manager; no owner ruling")) is None     # r2 R2-2
    assert S.release(_release_row(run, source="Manager 2026-10-08")) is None
    assert S.release(_release_row(run, budget="0")) is None
    assert S.release(_release_row(run).replace("RELEASE seeds", "DENY seeds")) is None
    assert S.release(_release_row(run).replace("per seed**", "per seed** and more")) is None
    assert S.release("no row here") is None


def test_the_command_line_has_no_output_root_option():
    with pytest.raises(SystemExit):
        S.main(["run", "--seed", str(S.STAGE_ONE_SEEDS[0]), "--data-dir", "x", "--out-root", "y"])


def test_run_refuses_without_the_deterministic_flag_or_a_stage_one_seed(monkeypatch, tmp_path):
    monkeypatch.delenv("BTS_LGBM_DETERMINISTIC", raising=False)
    with pytest.raises(SystemExit, match="DETERMINISTIC"):
        S.run(S.STAGE_ONE_SEEDS[0], tmp_path, tmp_path, _test_out_root=tmp_path)
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    with pytest.raises(SystemExit, match="stage-one seed"):
        S.run(42, tmp_path, tmp_path, _test_out_root=tmp_path)


def test_run_refuses_without_an_admission(monkeypatch, tmp_path):
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    with pytest.raises(SystemExit, match="refusing"):
        S.run(S.STAGE_ONE_SEEDS[0], tmp_path, tmp_path, _test_out_root=tmp_path)


# ---------------------------------------------------------------- run() end to end (stubbed admission, inputs, features
# and walk-forward; the claim, closed inputs, labels, manifest, stops, scoring and results are real)

def _stub_df():
    rng = np.random.default_rng(11)
    rows = []
    for season in (2023, 2024, 2025):
        for d in range(40):
            date = f"{season}-0{5 + d // 28}-{(d % 28) + 1:02d}"
            for b in range(1, 11):
                rows.append({"season": season, "date": date, "batter_id": b, "game_pk": season * 1000 + d,
                             "pitcher_id": 10 + b % 3, "fielding_catcher_id": (900, 901, np.nan)[b % 3],
                             "pa_borderline_csr": float(rng.random()),
                             "is_hit": int(rng.random() < (0.5 if b == 1 else 0.85)),
                             "is_resumed_portion": bool(d == 3 and b == 2)})
    df = pd.DataFrame(rows)
    return S.framing_by(df, "pitcher_id", S.OLD_COL)


def _stub_walk_forward(calls):
    def wf(df, season, retrain_every, blend_configs, game_probability_mode):
        calls.append((season, game_probability_mode, retrain_every, tuple(blend_configs[0][1])))
        is_a = S.NEW_COL in blend_configs[0][1] and S.OLD_COL not in blend_configs[0][1]
        part = df[df["season"] == season]
        rows = []
        for date, g in part.groupby("date"):
            order = sorted(g["batter_id"].unique(), reverse=is_a)        # A ranks the better batter 10 first
            for rank, b in enumerate(order, start=1):
                rows.append({"date": date, "rank": rank, "batter_id": int(b),
                             "game_pk": int(g.loc[g["batter_id"] == b, "game_pk"].iloc[0]),
                             "p_game_hit": 0.9 - 0.01 * rank, "actual_hit": 1, "n_pas": 4})
        return pd.DataFrame(rows)
    return wf


class _Admission:
    """The stubbed admission: a head that can move (a metadata-only descendant) under one accepted identity."""
    def __init__(self, pins):
        self.head = "a" * 40
        self.pins = pins

    def __call__(self):
        return self.head, {"input_pins": self.pins}, dict(IDENT)


@pytest.fixture
def stubbed(monkeypatch, tmp_path):
    import bts.features.compute as FC
    import bts.features.park_drag as PD
    import bts.model.predict as PR
    from scripts.audit.c1 import admission as A
    monkeypatch.setenv("BTS_LGBM_DETERMINISTIC", "1")
    monkeypatch.setitem(PR.LGB_PARAMS, "deterministic", True)       # as built at import under the flag
    monkeypatch.setitem(PR.LGB_PARAMS, "force_row_wise", True)
    monkeypatch.setattr(FC, "_build_probable_pitcher_lookup", FC._build_probable_pitcher_lookup)
    monkeypatch.setattr(PD, "attach_park_drag", PD.attach_park_drag)
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    lookup = b'{"1":{"away":5}}\n'
    (inputs / S.LOOKUP_NAME).write_bytes(lookup)
    pins = {n: "b" * 64 for n in S.INPUT_NAMES}
    pins[S.LOOKUP_NAME] = A.hashlib.sha256(lookup).hexdigest()
    adm = _Admission(pins)
    monkeypatch.setattr(S, "admission_gate", adm)
    monkeypatch.setattr(S, "load_inputs", lambda data_dir, pins: _stub_df())
    monkeypatch.setattr(FC, "compute_all_features", lambda df: df)
    import bts.validate.scorecard as SC
    real_card = SC.compute_full_scorecard
    monkeypatch.setattr(SC, "compute_full_scorecard", lambda p, **k: real_card(p, mc_trials=200))   # speed only
    out = tmp_path / "out"
    out.mkdir()
    return out, inputs, adm


def _run_dir(out, seed):
    dirs = [p for p in (out / f"seed_{seed}").iterdir() if p.is_dir()]
    assert len(dirs) == 1
    return dirs[0]


def _run(stubbed, seed, calls=None, **k):
    out, inputs, _ = stubbed
    return S.run(seed, out, inputs, walk_forward=_stub_walk_forward([] if calls is None else calls),
                 _test_out_root=out, **k)


def test_run_end_to_end(stubbed):
    import json
    import bts.features.compute as FC
    out, _, _ = stubbed
    calls = []
    seed = S.STAGE_ONE_SEEDS[0]
    assert _run(stubbed, seed, calls) == 0
    d = _run_dir(out, seed)
    assert (d / "CLAIM.json").exists()
    man = json.loads((d / "manifest.json").read_text())
    assert man["seed"] == seed and man["basis"] == "estimated_pa" and man["self_check"]["identical"] is True
    assert man["env"]["BTS_LGBM_RANDOM_STATE"] == str(seed) and man["env"]["BTS_LGBM_DETERMINISTIC"] == "1"
    assert man["feature_settings"] == S.SETTINGS and man["lookup_games"] == 1
    assert man["resumed_portion_rows"] == {"2023": 1, "2024": 1, "2025": 1}
    assert man["coverage"][S.NEW_COL]["2024"]["rows"] == 400
    assert FC._build_probable_pitcher_lookup() == {1: {"away": 5}}       # the frozen lookup is installed
    assert [c[0] for c in calls] == [2024, 2025] * 3 and {c[1] for c in calls} == {"estimated_pa"}
    assert {c[2] for c in calls} == {7}
    res = json.loads((d / "results.json").read_text())
    assert set(res["variants"]) == {"A", "B"} and len(res["units"]) == 6
    assert res["variants"]["A"]["p_at_1_delta"]["2024"] > 0             # A ranks the better batter first
    assert res["variants"]["B"]["p_at_1_delta"] == {"2024": 0.0, "2025": 0.0}
    assert res["units"][0]["labels"]["void_dropped"] == 1               # batter 2 on day 3: a resumed-only PA
    for v in ("baseline", "A", "B"):
        assert (d / f"scorecard_{v}.json").exists()
        for s in (2024, 2025):
            p = pd.read_parquet(d / f"profiles_{v}_{s}.parquet")
            assert p["actual_hit"].isin([0, 1]).all() and (p.groupby("date")["rank"].min() == 1).all()
    assert (d / "diff_A.json").exists() and (d / "diff_B.json").exists()
    S.validate_run(d, seed, out_root=out, identity=IDENT, pins=man["input_pins"])      # the run validates


def test_profiles_carry_the_rebuilt_labels_not_the_walk_forwards(stubbed):
    out, _, _ = stubbed
    seed = S.STAGE_ONE_SEEDS[0]
    assert _run(stubbed, seed) == 0
    p = pd.read_parquet(_run_dir(out, seed) / "profiles_baseline_2024.parquet")
    assert (p["actual_hit"] == 0).any()                    # the stub said 1 everywhere; the PA rows say otherwise


def test_a_second_run_of_the_same_seed_is_refused(stubbed):
    seed = S.STAGE_ONE_SEEDS[0]
    assert _run(stubbed, seed) == 0
    with pytest.raises(SystemExit, match="claimed run"):
        _run(stubbed, seed)


def test_the_self_check_stops_the_run_before_any_walk_forward(stubbed, monkeypatch):
    import json
    out, _, _ = stubbed
    bad = _stub_df()
    bad.loc[bad.index[-1], S.OLD_COL] = 0.123
    monkeypatch.setattr(S, "load_inputs", lambda data_dir, pins: bad)
    calls = []
    assert _run(stubbed, S.STAGE_ONE_SEEDS[0], calls) == 4
    d = _run_dir(out, S.STAGE_ONE_SEEDS[0])
    assert calls == [] and json.loads((d / "STOPPED.json").read_text())["reason"] == "self_check"


def test_an_expensive_first_walk_forward_stops_the_run(stubbed, monkeypatch):
    import json
    out, _, _ = stubbed
    ticks = iter([0.0, 0.0, 0.0, 8 * 3600.0] + [8 * 3600.0] * 50)
    monkeypatch.setattr(S, "cpu_seconds", lambda: next(ticks))
    calls = []
    assert _run(stubbed, S.STAGE_ONE_SEEDS[0], calls) == 3
    d = _run_dir(out, S.STAGE_ONE_SEEDS[0])
    stop = json.loads((d / "STOPPED.json").read_text())
    assert stop["reason"] == "first_unit_cpu" and len(calls) == 1 and not (d / "results.json").exists()


def test_run_refuses_when_lightgbm_params_lack_the_deterministic_flags(stubbed, monkeypatch):
    import bts.model.predict as PR
    monkeypatch.setitem(PR.LGB_PARAMS, "deterministic", False)
    with pytest.raises(SystemExit, match="deterministic flags"):
        _run(stubbed, S.STAGE_ONE_SEEDS[0])


def test_the_manifest_records_the_effective_lightgbm_params(stubbed):
    import json
    out, _, _ = stubbed
    assert _run(stubbed, S.STAGE_ONE_SEEDS[0]) == 0
    man = json.loads((_run_dir(out, S.STAGE_ONE_SEEDS[0]) / "manifest.json").read_text())
    assert man["lgb_params"]["deterministic"] is True and man["lgb_params"]["force_row_wise"] is True


def test_a_changed_frozen_lookup_is_refused(stubbed):
    from scripts.audit.c1.admission import ProvenanceError
    out, inputs, _ = stubbed
    (inputs / S.LOOKUP_NAME).write_bytes(b'{"1":{"away":6}}\n')
    with pytest.raises(ProvenanceError, match="not the pinned"):
        _run(stubbed, S.STAGE_ONE_SEEDS[0])


# ---------------------------------------------------------------- seed order and Eric's release (r2 R2-1, R2-2)

def _released(stubbed, monkeypatch, *, source="Eric 2026-10-08, relayed"):
    """Seed 1 runs, then Eric's release naming it is committed (a metadata descendant: the head moves)."""
    out, _, adm = stubbed
    assert _run(stubbed, S.STAGE_ONE_SEEDS[0]) == 0
    seed1 = _run_dir(out, S.STAGE_ONE_SEEDS[0])
    adm.head = "c" * 40
    from scripts.audit.c1 import admission as A
    text = (A.REPO / S.REGISTER_REL).read_text() + "\n" + _release_row(seed1.name, source=source) + "\n"
    real_read = Path.read_text

    def read_text(self, *a, **k):
        return text if self == A.REPO / S.REGISTER_REL else real_read(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", read_text)
    return seed1


from pathlib import Path  # noqa: E402


def test_seeds_two_and_three_are_refused_without_the_release(stubbed):
    with pytest.raises(SystemExit, match="release"):
        _run(stubbed, S.STAGE_ONE_SEEDS[1])


def test_the_seed1_release_lifecycle_aggregates_across_a_moved_head(stubbed, monkeypatch):
    out, _, _ = stubbed
    seed1 = _released(stubbed, monkeypatch)
    for seed in S.STAGE_ONE_SEEDS[1:]:
        assert _run(stubbed, seed) == 0
    dirs = [_run_dir(out, s) for s in S.STAGE_ONE_SEEDS]
    agg = S.aggregate(dirs, _test_out_root=out)
    assert agg["heads"] == ["a" * 40, "c" * 40, "c" * 40] and agg["identity"] == IDENT
    assert agg["variants"]["A"]["n_seeds"] == 3 and agg["variants"]["B"]["disposition"] == "negative"
    assert seed1.name in dirs[0].name


def test_an_ericsson_source_does_not_release(stubbed, monkeypatch):
    _released(stubbed, monkeypatch, source="Ericsson, manager; no owner ruling")
    with pytest.raises(SystemExit, match="release"):
        _run(stubbed, S.STAGE_ONE_SEEDS[1])


def test_an_empty_seed1_completion_does_not_release(stubbed, monkeypatch, tmp_path):
    import json
    out, inputs, adm = stubbed
    fake = out / f"seed_{S.STAGE_ONE_SEEDS[0]}" / "aaaaaaa-20261007T000000Z"
    fake.mkdir(parents=True)
    (fake / "results.json").write_text("{}")                             # r2 R2-2: a filename is not a completion
    from scripts.audit.c1 import admission as A
    text = (A.REPO / S.REGISTER_REL).read_text() + "\n" + _release_row(fake.name) + "\n"
    real_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: text if self == A.REPO / S.REGISTER_REL
                        else real_read(self, *a, **k))
    with pytest.raises(SystemExit, match="seed 1's run is not a complete admitted run"):
        _run(stubbed, S.STAGE_ONE_SEEDS[1])


def test_a_release_naming_another_run_does_not_release(stubbed, monkeypatch):
    out, _, adm = stubbed
    assert _run(stubbed, S.STAGE_ONE_SEEDS[0]) == 0
    from scripts.audit.c1 import admission as A
    text = (A.REPO / S.REGISTER_REL).read_text() + "\n" + _release_row("bbbbbbb-20261007T000000Z") + "\n"
    real_read = Path.read_text
    monkeypatch.setattr(Path, "read_text", lambda self, *a, **k: text if self == A.REPO / S.REGISTER_REL
                        else real_read(self, *a, **k))
    with pytest.raises(SystemExit, match="released run"):
        _run(stubbed, S.STAGE_ONE_SEEDS[1])


def test_launch_builds_the_launcher_command_with_the_seeds_budget(stubbed, monkeypatch):
    out, inputs, _ = stubbed
    calls = []

    class R:
        returncode = 0
    execute = lambda cmd, cwd: (calls.append(cmd), R())[1]      # noqa: E731
    assert S.launch(S.STAGE_ONE_SEEDS[0], out, inputs, _test_out_root=out, execute=execute) == 0
    cmd = calls[0]
    assert cmd[cmd.index("--cpu-hours") + 1] == "45" and cmd[cmd.index("--name") + 1] == "c2-framing-seed1"
    assert "BTS_LGBM_DETERMINISTIC=1" in cmd and cmd[cmd.index("--seed") + 1] == str(S.STAGE_ONE_SEEDS[0])
    with pytest.raises(SystemExit, match="release"):
        S.launch(S.STAGE_ONE_SEEDS[1], out, inputs, _test_out_root=out, execute=execute)
    assert len(calls) == 1
    _released(stubbed, monkeypatch)
    assert S.launch(S.STAGE_ONE_SEEDS[1], out, inputs, _test_out_root=out, execute=execute) == 0
    assert calls[1][calls[1].index("--cpu-hours") + 1] == "30" and "c2-framing-seed2" in calls[1]


# ---------------------------------------------------------------- the aggregate (r2 R2-1, R2-3)

@pytest.fixture
def three_runs(stubbed, monkeypatch):
    out, _, _ = stubbed
    _released(stubbed, monkeypatch)
    for seed in S.STAGE_ONE_SEEDS[1:]:
        assert _run(stubbed, seed) == 0
    return out, [_run_dir(out, s) for s in S.STAGE_ONE_SEEDS]


def test_aggregate_the_three_registered_seeds(three_runs):
    out, dirs = three_runs
    agg = S.aggregate(dirs, _test_out_root=out)
    assert agg["seeds"] == list(S.STAGE_ONE_SEEDS) and agg["variants"]["A"]["n_seeds"] == 3
    assert agg["variants"]["A"]["disposition"] in ("positive", "inconclusive")
    assert agg["variants"]["B"]["disposition"] == "negative"


@pytest.mark.parametrize("pick", [lambda d: d[:1], lambda d: d[:2], lambda d: [d[0], d[0], d[1]],
                                  lambda d: d + d[:1]])
def test_aggregate_refuses_partial_duplicate_or_extra_runs(three_runs, pick):
    out, dirs = three_runs
    with pytest.raises(S.RunInvalid, match="exactly 3 distinct run directories"):    # the count check itself
        S.aggregate(pick(dirs), _test_out_root=out)


def test_aggregate_refuses_a_foreign_seed(three_runs):
    import shutil
    out, dirs = three_runs
    foreign = out / "seed_42" / dirs[2].name
    shutil.copytree(dirs[2], foreign)
    with pytest.raises(S.RunInvalid, match="registered"):
        S.aggregate([dirs[0], dirs[1], foreign], _test_out_root=out)


def test_aggregate_refuses_runs_outside_the_canonical_namespace(three_runs, tmp_path):
    import shutil
    out, dirs = three_runs
    elsewhere = tmp_path / "elsewhere"
    copies = []
    for d in dirs:
        c = elsewhere / d.parent.name / d.name
        shutil.copytree(d, c)
        copies.append(c)
    with pytest.raises(S.RunInvalid, match="canonical claim namespace"):
        S.aggregate(copies, _test_out_root=out)


def _rewrite(path, fn):
    import json
    rec = json.loads(path.read_text())
    fn(rec)
    path.write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")


def _rebind_claim(d):
    """Make a damaged manifest's claim binding consistent again, so only the targeted check can refuse."""
    import hashlib, json
    man = json.loads((d / "manifest.json").read_text())
    man["claim_sha256"] = hashlib.sha256((d / "CLAIM.json").read_bytes()).hexdigest()
    (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")


@pytest.mark.parametrize("damage, match", [
    ("stopped", "stopped run"), ("opaque_claim", "CLAIM.json is not valid JSON"),
    ("claim_other_run", "claim does not name"), ("basis", "basis"), ("settings", "settings"),
    ("identity_missing", "identity"), ("pins_missing", "pins"), ("unit_missing", "six registered units"),
    ("summary", "stored summary"), ("missing_diff", "missing diff_B.json"),
    ("diff_and_summary", "diff does not match its retained scorecards"),
    ("profile", "does not reconcile"), ("deterministic", "lgb determinism"),
])
def test_aggregate_refuses_an_invalid_run(three_runs, damage, match):
    import json
    out, dirs = three_runs
    d = dirs[1]
    if damage == "stopped":
        (d / "STOPPED.json").write_text("{}")
    elif damage == "opaque_claim":
        (d / "CLAIM.json").write_bytes(b"not even JSON")
        _rebind_claim(d)
    elif damage == "claim_other_run":
        _rewrite(d / "CLAIM.json", lambda r: r.update(run="other"))
        _rebind_claim(d)
    elif damage == "basis":
        for x in dirs:                                    # consistently wrong across all three runs (r2 R2-3)
            _rewrite(x / "manifest.json", lambda r: r.update(basis="actual_pa"))
    elif damage == "settings":
        for x in dirs:
            _rewrite(x / "manifest.json", lambda r: r.update(feature_settings={"ROOKIE_GATE_K": 0,
                                                                               "PITCHER_HR_30G_MIN_PERIODS": 7}))
    elif damage == "identity_missing":
        for x in dirs:
            _rewrite(x / "manifest.json", lambda r: r.pop("identity"))
    elif damage == "pins_missing":
        for x in dirs:
            _rewrite(x / "manifest.json", lambda r: r.pop("input_pins"))
    elif damage == "unit_missing":
        _rewrite(d / "results.json", lambda r: r.update(units=r["units"][:5]))
    elif damage == "summary":
        _rewrite(d / "results.json", lambda r: r["variants"]["A"].update(passed=not r["variants"]["A"]["passed"]))
    elif damage == "missing_diff":
        (d / "diff_B.json").unlink()
    elif damage == "diff_and_summary":                   # r2 R2-3: coherent diff + summary, scorecards unchanged
        def bump(r):
            for k in r["p_at_1_by_season"]:
                r["p_at_1_by_season"][k]["delta"] = 0.02
        for x in dirs:
            _rewrite(x / "diff_B.json", bump)
            _rewrite(x / "results.json", lambda r: r["variants"]["B"].update(
                p_at_1_delta={"2024": 0.02, "2025": 0.02}, passed=True))
    elif damage == "profile":
        p = pd.read_parquet(d / "profiles_A_2024.parquet")
        p.loc[p["rank"] == 1, "actual_hit"] = 1 - p.loc[p["rank"] == 1, "actual_hit"]
        p.to_parquet(d / "profiles_A_2024.parquet", index=False)
    elif damage == "deterministic":
        for x in dirs:
            _rewrite(x / "manifest.json", lambda r: r["lgb_params"].update(deterministic=False))
    with pytest.raises(S.RunInvalid, match=match):
        S.aggregate(dirs, _test_out_root=out)


def test_aggregate_refuses_runs_of_different_identities(three_runs):
    out, dirs = three_runs
    _rewrite(dirs[2] / "manifest.json", lambda r: r["identity"].update(reviewed_commit="other"))
    with pytest.raises(S.RunInvalid, match="disagree on identity"):
        S.aggregate(dirs, _test_out_root=out)


# ---------------------------------------------------------------- the mutant runner's classification (r2 R2-4)

def _runner():
    import importlib.util
    path = Path(__file__).resolve().parents[3] / "docs/audit/2026-10-06-c2-framing-evidence/mutant_runner.py"
    spec = importlib.util.spec_from_file_location("framing_mutant_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.mark.parametrize("rc, out, verdict", [
    (1, "FAILED t.py::a - x\n1 failed, 3 passed in 0.1s", "RED"),
    (1, "FAILED t.py::a - x\nERROR t.py::b - setup\n1 failed, 1 error in 0.01s", "INCONCLUSIVE(exit 1, errors)"),
    (1, "FAILED t.py::a - x\n1 failed, 1 skipped in 0.1s", "INCONCLUSIVE(exit 1)"),
    (0, "4 passed in 0.1s", "SURVIVED"),
    (2, "1 error in 0.1s", "INCONCLUSIVE(exit 2)"),
    (5, "no tests ran in 0.01s", "INCONCLUSIVE(exit 5)"),
    (1, "FAILED t.py::a - x\n1 failed, 3 passed, 2 deselected in 0.1s", "RED"),
])
def test_the_runner_classifies_only_clean_failures_as_red(rc, out, verdict):
    assert _runner().classify(rc, out)[0] == verdict
