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


def test_the_registered_scoring_settings_are_the_scorers_defaults():
    import inspect
    from bts.validate.scorecard import compute_full_scorecard
    sig = inspect.signature(compute_full_scorecard).parameters
    assert S.SCORING == {"mc_trials": 10_000, "season_length": 180}
    assert {k: sig[k].default for k in S.SCORING} == S.SCORING


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


ADMITTED_HEADS = ("a" * 40, "c" * 40)     # the stub's reviewed head and its metadata-only descendant (Eric's release)


def _fake_head_admitted(repo, identity, head):
    """The stubbed repository: only the two stub heads are admitted descendants of IDENT (r3 R3-3)."""
    return [] if identity == IDENT and head in ADMITTED_HEADS else [f"run HEAD {str(head)[:7]} is not admitted"]


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
    monkeypatch.setattr(S, "SCORING", {"mc_trials": 200, "season_length": 180})      # speed only; same code path
    monkeypatch.setattr(S, "head_admitted", _fake_head_admitted)     # real git ancestry: test_head_admitted_*
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


def test_a_seed1_run_of_another_identity_does_not_release(stubbed, monkeypatch):
    """The release names seed 1's real run, but that run was made by other code than the admitted identity."""
    seed1 = _released(stubbed, monkeypatch)
    _rewrite(seed1 / "manifest.json", lambda r: r["identity"].update(reviewed_commit="other"))
    with pytest.raises(SystemExit, match=r"not a complete admitted run: .*\['admitted identity'\]"):
        _run(stubbed, S.STAGE_ONE_SEEDS[1])


def _launch_recording(stubbed, seed):
    out, inputs, _ = stubbed
    calls = []

    class R:
        returncode = 0
    S.launch(seed, out, inputs, _test_out_root=out, execute=lambda cmd, cwd: (calls.append(cmd), R())[1])
    return calls


def test_a_seed1_run_without_2025_evidence_does_not_release(stubbed, monkeypatch):
    """r3 R3-2: 2024's profiles standing in for 2025, everything downstream coherent — refused at run and launch."""
    seed1 = _released(stubbed, monkeypatch)
    _duplicate_2024_as_2025(seed1)
    with pytest.raises(SystemExit, match="not a complete admitted run: .*not complete 2025 evidence"):
        _run(stubbed, S.STAGE_ONE_SEEDS[1])
    with pytest.raises(SystemExit, match="not complete 2025 evidence"):
        _launch_recording(stubbed, S.STAGE_ONE_SEEDS[1])


def test_a_seed1_run_of_unadmitted_code_does_not_release(stubbed, monkeypatch):
    """r3 R3-3: a coherent seed-1 run whose HEAD is not an admitted descendant, under the same declared identity."""
    seed1 = _released(stubbed, monkeypatch)
    _move_head(seed1, "02516cfedc812f047295b6f2ab212fd72a944507")
    with pytest.raises(SystemExit, match="not a complete admitted run: .*is not admitted"):
        _run(stubbed, S.STAGE_ONE_SEEDS[1])
    with pytest.raises(SystemExit, match="is not admitted"):
        _launch_recording(stubbed, S.STAGE_ONE_SEEDS[1])


def _git(repo, *args):
    import subprocess
    return subprocess.run(["git", "-C", str(repo), "-c", "user.email=t@t", "-c", "user.name=t", *args],
                          capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo, files: dict, msg: str) -> str:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


def test_head_admitted_requires_a_metadata_descendant_of_the_exposure(tmp_path):
    """r3 R3-3, against a real disposable repository: a run HEAD qualifies only if the exposure commit is its ancestor
    and no executable-closure file other than admission.json differs from the reviewed commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    base = _commit(repo, {"scripts/audit/c2_framing/screen.py": "v0\n", S.REGISTER_REL: "r0\n"}, "base")
    reviewed = _commit(repo, {"scripts/audit/c2_framing/screen.py": "v1\n"}, "reviewed")
    exposure = _commit(repo, {S.REGISTER_REL: "r1 X-35\n"}, "exposure")
    admitted = _commit(repo, {S.ADMISSION_REL: "{}\n"}, "admission record")
    released = _commit(repo, {S.REGISTER_REL: "r1 X-35\nrelease\n"}, "Eric's release (metadata only)")
    changed = _commit(repo, {"scripts/audit/c2_framing/screen.py": "v2\n"}, "code change")
    _git(repo, "checkout", "-q", "-b", "side", base)
    side = _commit(repo, {"other.txt": "x\n"}, "not a descendant")
    ident = {**IDENT, "reviewed_commit": reviewed, "exposure_commit": exposure}
    assert S.head_admitted(repo, ident, admitted) == []
    assert S.head_admitted(repo, ident, released) == []
    assert S.head_admitted(repo, ident, exposure) == []
    assert any("executable" in x for x in S.head_admitted(repo, ident, changed))
    assert any("ancestor" in x for x in S.head_admitted(repo, ident, side))
    assert any("ancestor" in x for x in S.head_admitted(repo, ident, reviewed))      # before the exposure: not admitted
    assert S.head_admitted(repo, ident, "f" * 40)                                      # not a commit here
    assert S.head_admitted(repo, {**ident, "exposure_commit": "bogus"}, admitted)      # not a commit id


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


def _recohere(d):
    """Regenerate a run's diffs, results P@1 and stored summaries from its (edited) scorecards with the production
    helpers, so only the reconciliation against the retained profiles can refuse."""
    import json
    from bts.validate.scorecard import diff_scorecards
    cards = {v: json.loads((d / f"scorecard_{v}.json").read_text()) for v in ("baseline", "A", "B")}
    res = json.loads((d / "results.json").read_text())
    res["p_at_1_by_season"] = {v: c["p_at_1_by_season"] for v, c in cards.items()}
    for v in ("A", "B"):
        diff = json.loads(json.dumps(diff_scorecards(cards["baseline"], cards[v]), sort_keys=True))
        (d / f"diff_{v}.json").write_text(json.dumps(diff, indent=1, sort_keys=True) + "\n")
        res["variants"][v] = {**S.seed_summary(diff), "secondary": {
            "p_57_exact": diff.get("p_57_exact"), "mean_max_streak": diff.get("streak_metrics", {}).get("mean_max_streak")}}
    (d / "results.json").write_text(json.dumps(res, indent=1, sort_keys=True) + "\n")


def _edit_profile(d, v, season, fn):
    """Edit one retained profile, recompute that variant's scorecard from the retained profiles, and make the rest of
    the run coherent: only the season-evidence checks can refuse."""
    import json
    from bts.validate.scorecard import compute_full_scorecard
    path = d / f"profiles_{v}_{season}.parquet"
    fn(pd.read_parquet(path)).to_parquet(path, index=False)
    parts = [pd.read_parquet(d / f"profiles_{v}_{s}.parquet") for s in S.TEST_SEASONS]
    try:
        card = compute_full_scorecard(pd.concat(parts, ignore_index=True), **S.SCORING)
    except Exception:                                    # the scorer cannot even read it: the old card stays
        return
    (d / f"scorecard_{v}.json").write_text(json.dumps(card, indent=1, sort_keys=True) + "\n")
    _recohere(d)


def _duplicate_2024_as_2025(d):
    """r3 R3-2's case: each variant's 2025 profile replaced by its 2024 profile (rows still say 2024), the scorecards
    recomputed from what is retained, and everything downstream made coherent."""
    import json
    from bts.validate.scorecard import compute_full_scorecard
    for v in ("baseline", "A", "B"):
        p24 = pd.read_parquet(d / f"profiles_{v}_2024.parquet")
        p24.to_parquet(d / f"profiles_{v}_2025.parquet", index=False)
        card = compute_full_scorecard(pd.concat([p24, p24], ignore_index=True), **S.SCORING)
        (d / f"scorecard_{v}.json").write_text(json.dumps(card, indent=1, sort_keys=True) + "\n")
    _recohere(d)


def _move_head(d, head):
    """Coherently move a run's HEAD (claim, manifest, results) and rebind its claim hash (r3 R3-3)."""
    _rewrite(d / "CLAIM.json", lambda r: r.update(code=head))
    _rewrite(d / "manifest.json", lambda r: r.update(head=head))
    _rewrite(d / "results.json", lambda r: r.update(head=head))
    _rebind_claim(d)


def _rebind_claim(d):
    """Make a damaged manifest's claim binding consistent again, so only the targeted check can refuse."""
    import hashlib, json
    man = json.loads((d / "manifest.json").read_text())
    man["claim_sha256"] = hashlib.sha256((d / "CLAIM.json").read_bytes()).hexdigest()
    (d / "manifest.json").write_text(json.dumps(man, indent=1, sort_keys=True) + "\n")


DAMAGE = [
    ("stopped", "stopped run"), ("opaque_claim", "CLAIM.json is not valid JSON"),
    ("claim_other_run", "claim does not name"), ("basis", "basis"), ("settings", "settings"),
    ("identity_missing", "identity"), ("pins_missing", "pins"), ("unit_missing", "six registered units"),
    ("summary", "stored summary"), ("missing_diff", "missing diff_B.json"),
    ("diff_and_summary", "diff does not match its retained scorecards"),
    ("profile", "does not match a recomputation from its retained profiles"), ("deterministic", "lgb determinism"),
    ("claim_rewritten", r"\['claim binding'\]"), ("pins_names", r"\['pins', 'admitted pins'\]"),
    ("secondary_card", "does not match a recomputation from its retained profiles"),
    ("season_duplicate", "not complete 2025 evidence"), ("bogus_identity", r"\['admitted identity'\]"),
    ("foreign_head", "is not admitted"), ("scoring", r"\['scoring'\]"),
    ("results_p1", "does not reconcile across profiles, scorecard and results"),
    ("season_mixed", "not complete 2025 evidence"), ("dates_other_year", "not complete 2025 evidence"),
    ("pins_other", r"\['admitted pins'\]"), ("no_top_pick", "not complete 2025 evidence: ranks"),
    ("secondary_summary", "stored summary does not match its diff"),
    # r4 R4-1: rows present in the file but dropped by the scorer (a missing key) are never complete evidence
    ("season_missing", "not complete 2025 evidence: missing values"),
    ("rank_missing", "not complete 2025 evidence: missing values"),
    ("hit_missing", "not complete 2025 evidence: missing values"),
    ("ranks_not_from_one", "not complete 2025 evidence: ranks"),
    ("hit_not_binary", "not complete 2025 evidence: hits"),
    ("identity_extra_key", r"\['identity'\]"),         # r4 R4-3: exactly the five identity fields
    ("p_out_of_range", "not complete 2025 evidence: probabilities"),
    ("empty_unit", "not complete 2025 evidence: empty"),
    ("column_absent", "not complete 2025 evidence: columns"),
    ("p_strings", "not complete 2025 evidence: probabilities"),   # r5 R5-2: text that coerces to a skipped NA
]


@pytest.mark.parametrize("damage, match", DAMAGE, ids=[c[0] for c in DAMAGE])
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
    elif damage == "claim_rewritten":                    # still names this run and its code; only its bytes moved
        _rewrite(d / "CLAIM.json", lambda r: r.update(pid=r["pid"] + 1))
    elif damage == "pins_names":                         # nine pins, consistently, with a digest that matches them
        def drop(r):
            r["input_pins"].pop(S.LOOKUP_NAME)
            r["inputs_digest"] = S.pins_digest(r["input_pins"])
        for x in dirs:
            _rewrite(x / "manifest.json", drop)
    elif damage == "secondary_card":                     # r3 R3-1: profiles untouched, a decision-bearing card value moved
        for x in dirs[:2]:
            base = json.loads((x / "scorecard_baseline.json").read_text())
            _rewrite(x / "scorecard_B.json", lambda r: r.update(p_57_exact=(base["p_57_exact"] or 0.0) + 0.001))
            _recohere(x)
    elif damage == "season_duplicate":                   # r3 R3-2: 2024's profiles stand in for 2025, coherently
        _duplicate_2024_as_2025(d)
    elif damage == "bogus_identity":                     # r3 R3-3: declared identity strings, consistent across runs
        for x in dirs:
            _rewrite(x / "manifest.json", lambda r: r.update(identity={k: "bogus" for k in S.IDENTITY_KEYS}))
    elif damage == "foreign_head":                       # r3 R3-3: a coherent HEAD that is not an admitted descendant
        _move_head(d, "02516cfedc812f047295b6f2ab212fd72a944507")
    elif damage == "scoring":
        for x in dirs:
            _rewrite(x / "manifest.json", lambda r: r.update(scoring={"mc_trials": 10, "season_length": 180}))
    elif damage == "results_p1":                         # only the results' copy of P@1 moved
        _rewrite(d / "results.json", lambda r: r["p_at_1_by_season"]["B"].update({"2024": 0.5}))
    elif damage == "season_mixed":                       # some 2025 rows labelled 2024 (dates still 2025), coherent
        _edit_profile(d, "A", 2025, lambda p: p.assign(season=[2024 if i < 3 else 2025 for i in range(len(p))]))
    elif damage == "dates_other_year":                   # 2025's rows keep their season label but carry 2024 dates
        _edit_profile(d, "A", 2025, lambda p: p.assign(date=pd.to_datetime(p["date"]) - pd.DateOffset(years=1)))
    elif damage == "secondary_summary":                  # only the stored summary's secondary value moved
        _rewrite(d / "results.json", lambda r: r["variants"]["B"]["secondary"].update(p_57_exact={"delta": 0.5}))
    elif damage == "season_missing":                     # the reviewer's nullable season: NA rows pass `== s`.all()
        _edit_profile(d, "B", 2025, lambda p: p.assign(
            season=pd.array([pd.NA] * 3 + [2025] * (len(p) - 3), dtype="Float64")))
    elif damage == "rank_missing":
        _edit_profile(d, "B", 2025, lambda p: p.assign(rank=pd.array([pd.NA] + list(p["rank"].iloc[1:]), dtype="Int64")))
    elif damage == "hit_missing":
        _edit_profile(d, "B", 2025, lambda p: p.assign(
            actual_hit=pd.array([pd.NA] + list(p["actual_hit"].iloc[1:]), dtype="Int64")))
    elif damage == "ranks_not_from_one":                 # the first day's ranks shifted: that day has no rank-1 pick
        def shift(p):
            first = p["date"] == p["date"].iloc[0]
            return p.assign(rank=p["rank"] + first.astype(int))
        _edit_profile(d, "B", 2025, shift)
    elif damage == "hit_not_binary":
        _edit_profile(d, "B", 2025, lambda p: p.assign(actual_hit=[2] + list(p["actual_hit"].iloc[1:])))
    elif damage == "p_out_of_range":
        _edit_profile(d, "B", 2025, lambda p: p.assign(p_game_hit=[1.5] + list(p["p_game_hit"].iloc[1:])))
    elif damage == "empty_unit":
        _edit_profile(d, "B", 2025, lambda p: p.iloc[0:0])
    elif damage == "column_absent":
        _edit_profile(d, "B", 2025, lambda p: p.drop(columns=["p_game_hit"]))
    elif damage == "p_strings":
        _edit_profile(d, "B", 2025, lambda p: p.assign(p_game_hit=pd.array(
            ["bad"] + [str(x) for x in p["p_game_hit"].iloc[1:]], dtype="string")))
    elif damage == "identity_extra_key":
        _rewrite(dirs[2] / "manifest.json", lambda r: r["identity"].update(unrecognized_extra="probe"))
    elif damage == "no_top_pick":                        # 2025 rows remain, but no day has a rank-1 pick
        _edit_profile(d, "A", 2025, lambda p: p[p["rank"] != 1])
    elif damage == "pins_other":                         # ten well-formed pins, consistent, but not the admitted ones
        def other(r):
            r["input_pins"] = {k: "c" * 64 for k in r["input_pins"]}
            r["inputs_digest"] = S.pins_digest(r["input_pins"])
        for x in dirs:
            _rewrite(x / "manifest.json", other)
    with pytest.raises(S.RunInvalid, match=match):
        S.aggregate(dirs, _test_out_root=out)


def test_aggregate_refuses_runs_of_different_identities(three_runs):
    out, dirs = three_runs
    _rewrite(dirs[2] / "manifest.json", lambda r: r["identity"].update(reviewed_commit="other"))
    with pytest.raises(S.RunInvalid, match=r"\['admitted identity'\]"):     # r3 R3-3: against the gate's identity
        S.aggregate(dirs, _test_out_root=out)


def test_validate_refuses_an_incomplete_identity_even_when_admitted(three_runs):
    """The run's own completeness check, isolated: the trusted identity has the same empty field."""
    out, dirs = three_runs
    d = dirs[0]
    partial = {**IDENT, "review_report_sha256": ""}
    _rewrite(d / "manifest.json", lambda r: r.update(identity=dict(partial)))
    with pytest.raises(S.RunInvalid, match=r"\['identity'\]"):
        S.validate_run(d, S.STAGE_ONE_SEEDS[0], out_root=out, identity=partial,
                       pins=__import__("json").loads((d / "manifest.json").read_text())["input_pins"])


def test_a_rescoring_failure_is_a_refusal_not_a_crash(three_runs, monkeypatch):
    """r5 R5-2: anything the scorer cannot score is refused through RunInvalid, never an unexpected exception."""
    import bts.validate.scorecard as SC
    out, dirs = three_runs
    monkeypatch.setattr(SC, "compute_full_scorecard", lambda *a, **k: 1 + "x")
    with pytest.raises(S.RunInvalid, match="cannot be rescored"):
        S.validate_run(dirs[0], S.STAGE_ONE_SEEDS[0], out_root=out, identity=IDENT,
                       pins=__import__("json").loads((dirs[0] / "manifest.json").read_text())["input_pins"])


def test_validate_refuses_pins_other_than_the_ten_even_when_admitted(three_runs):
    """The run's own ten-name check, isolated: the trusted pins are the same nine, so only 'pins' can refuse."""
    out, dirs = three_runs
    d = dirs[0]

    def drop(r):
        r["input_pins"].pop(S.LOOKUP_NAME)
        r["inputs_digest"] = S.pins_digest(r["input_pins"])
    _rewrite(d / "manifest.json", drop)
    nine = __import__("json").loads((d / "manifest.json").read_text())["input_pins"]
    with pytest.raises(S.RunInvalid, match=r"\['pins'\]"):
        S.validate_run(d, S.STAGE_ONE_SEEDS[0], out_root=out, identity=IDENT, pins=nine)


# ---------------------------------------------------------------- the mutant runner's classification (r2 R2-4)

def _runner():
    import importlib.util
    path = Path(__file__).resolve().parents[3] / "docs/audit/2026-10-06-c2-framing-evidence/mutant_runner.py"
    spec = importlib.util.spec_from_file_location("framing_mutant_runner", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


INTENDED = ["t.py::a", "t.py::b", "t.py::c[x - y]"]


def _rec(node, outcome="passed", when="call", wasxfail=False):
    return {"node": node, "when": when, "outcome": outcome, "wasxfail": wasxfail}


def _phases(node, call="passed"):
    return [_rec(node, "passed", "setup"), _rec(node, call), _rec(node, "passed", "teardown")]


A, B, C = INTENDED


def _bundle(records, failed_nodes=(), rc=None, **over):
    """The driver's evidence: pytest's reports, the sentinel's own call observations (exception name or None), its
    session-finish record, plugins registered after it, and intended nodes carrying xfail/skip markers."""
    failed_nodes = set(failed_nodes)
    nodes = [r["node"] for r in records if "node" in r and r["when"] == "call"]
    b = {"rc": (1 if failed_nodes else 0) if rc is None else rc, "records": records,
         "calls": {n: ("AssertionError" if n in failed_nodes else None) for n in nodes},
         "finish": {"exitstatus": (1 if failed_nodes else 0) if rc is None else rc, "testsfailed": len(failed_nodes),
                    "testscollected": len(set(nodes))},
         "late_plugins": 0, "not_outermost": [], "marked": []}
    b.update(over)
    return b


ONE_FAIL = _phases(A, "failed") + _phases(B) + _phases(C)
CLASSIFY = [
    (1, _bundle(ONE_FAIL, [A]), "RED"),
    (1, _bundle(_phases(A, "failed") + _phases(B) + _phases(C, "failed"), [A, C]), "RED"),
    (0, _bundle(_phases(A) + _phases(B) + _phases(C)), "SURVIVED"),
    # r5 R5-1, r6 R6-1: no evidence at all (the driver never got control back), or a process code that is not pytest's
    (1, None, "INCONCLUSIVE(exit 1, no normal end)"),
    (2, _bundle(ONE_FAIL, [A]), "INCONCLUSIVE(exit 2, exit mismatch)"),
    (1, _bundle(ONE_FAIL + [{"interrupted": "Exit"}], [A]), "INCONCLUSIVE(exit 1, interrupted)"),
    (1, _bundle(ONE_FAIL + [{"internalerror": "RuntimeError"}], [A]), "INCONCLUSIVE(exit 1, internal error)"),
    (1, _bundle(ONE_FAIL + [{"collecterror": "t.py"}], [A]), "INCONCLUSIVE(exit 1, collection errors)"),
    # r6 R6-1: the outermost session-finish wrapper saw an inner wrapper abort, or never completed
    (1, _bundle(ONE_FAIL, [A], finish={"raised": "Exit"}), "INCONCLUSIVE(exit 1, session finish incomplete)"),
    (1, _bundle(ONE_FAIL, [A], finish=None), "INCONCLUSIVE(exit 1, session finish incomplete)"),
    (1, _bundle(ONE_FAIL, [A], late_plugins=1), "INCONCLUSIVE(exit 1, plugins registered late)"),
    # r7 R7-1: the sentinel found itself not outermost at a call or at session finish
    (1, _bundle(ONE_FAIL, [A], not_outermost=["sessionfinish"]), "INCONCLUSIVE(exit 1, sentinel not outermost)"),
    # r6 R6-2: expected-failure status is kept, and xfail/skip-marked nodes are never killers
    (1, _bundle(_phases(A, "failed") + [_rec(B, "passed", "setup"), _rec(B, "passed", wasxfail=True),
                                         _rec(B, "passed", "teardown")] + _phases(C), [A]),
     "INCONCLUSIVE(exit 1, xfail or skip)"),
    (1, _bundle(ONE_FAIL, [A], marked=[A]), "INCONCLUSIVE(exit 1, marked xfail or skip)"),
    (1, _bundle(_phases(A, "failed") + [_rec(B, "skipped", "setup")] + _phases(C), [A]),
     "INCONCLUSIVE(exit 1, xfail or skip)"),
    (1, _bundle(_phases(A, "failed") + [_rec(B, "failed", "setup")] + _phases(C), [A]), "INCONCLUSIVE(exit 1, errors)"),
    (1, _bundle(_phases(A, "failed"), [A]), "INCONCLUSIVE(exit 1, 2 not run)"),
    (1, _bundle(_phases(A, "failed") + _phases(B) + _phases(C, "failed")[:2], [A, C]), "INCONCLUSIVE(exit 1, 1 incomplete)"),
    (1, _bundle(ONE_FAIL + _phases("t.py::ab"), [A]), "INCONCLUSIVE(exit 1, 1 unintended)"),
    # a report whose outcome is not what the sentinel saw the test call do (a rewritten or fabricated report)
    (1, _bundle(ONE_FAIL, [A], calls={A: None, B: None, C: None}), "INCONCLUSIVE(exit 1, report mismatch)"),
    (1, _bundle(ONE_FAIL, [A], calls={A: "AssertionError", B: None}), "INCONCLUSIVE(exit 1, report mismatch)"),
    # pytest's own counters must agree with the reports
    (1, _bundle(ONE_FAIL, [A], finish={"exitstatus": 1, "testsfailed": 2, "testscollected": 3}),
     "INCONCLUSIVE(exit 1, session mismatch)"),
    (1, _bundle(ONE_FAIL, [A], finish={"exitstatus": 1, "testsfailed": 1, "testscollected": 4}),
     "INCONCLUSIVE(exit 1, session mismatch)"),
    (1, _bundle(ONE_FAIL, [A], finish={"exitstatus": 0, "testsfailed": 1, "testscollected": 3}),
     "INCONCLUSIVE(exit 1, session mismatch)"),
]
CLASSIFY_IDS = ["red", "red_two_failed", "survived", "no_evidence", "exit_mismatch", "interrupted", "internal_error",
                "collection_error", "finish_raised", "finish_missing", "late_plugin", "not_outermost", "xpass", "marked", "skipped",
                "errors", "fail_fast", "teardown_cut_short", "unintended_node", "report_flipped", "call_unobserved",
                "failed_count_mismatch", "collected_count_mismatch",
                "exit_status_mismatch"]


@pytest.mark.parametrize("rc, bundle, verdict", CLASSIFY, ids=CLASSIFY_IDS)
def test_the_runner_classifies_only_complete_clean_failures_as_red(rc, bundle, verdict):
    assert _runner().classify(rc, INTENDED, bundle)[0] == verdict


def _scratch_mutant(tmp_path, tests, body):
    """A scratch root of its own (the runner's rootdir and cwd, so collection never leaves it): a target, a test
    module, and a spec naming tests relative to that root (r4: the out-of-tree collection that failed in a sandbox)."""
    import json
    root = tmp_path / "scratch"
    root.mkdir()
    (root / "target.py").write_text("value = 1\n")
    (root / "test_scratch.py").write_text(body(root))
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([{"id": "X1", "rule": "scratch", "file": "target.py", "old": "value = 1",
                                 "new": "value = 2", "tests": tests}]))
    return spec, root


_VAL = ("import importlib.util, pathlib, pytest\n"
        "def val():\n"
        "    spec = importlib.util.spec_from_file_location('target', pathlib.Path(__file__).with_name('target.py'))\n"
        "    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.value\n")


def test_the_runner_clears_inherited_fail_fast_and_runs_every_named_test(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-x")
    canary = tmp_path / "canary"
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: _VAL + (
        "def test_first():\n    assert val() == 1\n"
        f"def test_second():\n    pathlib.Path({str(canary)!r}).write_text('ran')\n    assert val() == 1\n"))
    assert _runner().main(str(spec), root=root) == 0
    out = capsys.readouterr().out
    assert "X1 RED" in out and canary.read_text() == "ran"                 # the second body executed
    assert (root / "target.py").read_text() == "value = 1\n"               # restored


def test_the_runner_is_not_fooled_by_printed_outcomes(tmp_path, capsys):
    """r4 R4-2: the first test prints the second's PASSED line; the second's fixture stops pytest before its body."""
    canary = tmp_path / "canary"
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: _VAL + (
        "def test_first(request):\n"
        "    if val() != 1:\n"
        "        print('PASSED ' + request.node.nodeid.rsplit('::', 1)[0] + '::test_second')\n"
        "    assert val() == 1\n"
        "@pytest.fixture\n"
        "def gate():\n"
        "    if val() != 1:\n"
        "        pytest.exit('stop before the body', returncode=1)\n"
        f"def test_second(gate):\n    pathlib.Path({str(canary)!r}).write_text('ran')\n    assert val() == 1\n"))
    assert _runner().main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 INCONCLUSIVE" in out and "NOT RED: X1" in out and not canary.exists()
    assert (root / "target.py").read_text() == "value = 1\n"


def test_the_runner_is_not_fooled_by_an_aborted_teardown(tmp_path, capsys):
    """r5 R5-1: both bodies fail for real, but the second's teardown calls pytest.exit with a count-bearing message
    before its cleanup completes. pytest still unconfigures; the session did not end normally."""
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: _VAL + (
        "def test_first():\n    assert val() == 1\n"
        "@pytest.fixture\n"
        "def gate():\n"
        "    yield\n"
        "    if val() != 1:\n"
        "        pytest.exit('2 failed cleanup checks', returncode=1)\n"
        "    pathlib.Path(__file__).with_name('cleanup_completed').write_text('finished')\n"
        "def test_second(gate):\n    assert val() == 1\n"))
    assert _runner().main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 INCONCLUSIVE" in out and "NOT RED: X1" in out and not (root / "cleanup_completed").exists()
    assert (root / "target.py").read_text() == "value = 1\n"


def test_a_later_session_finish_abort_is_not_a_normal_end(tmp_path, capsys):
    """r5 R5-1: a conftest session-finish hook aborts after both tests failed for real. The plugin writes its session
    record last, so the abort prevents it; the session did not end normally."""
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: _VAL + (
        "def test_first():\n    assert val() == 1\n"
        "def test_second():\n    assert val() == 1\n"))
    (root / "conftest.py").write_text(_VAL + (
        "def pytest_sessionfinish(session, exitstatus):\n"
        "    if val() != 1:\n"
        "        pytest.exit('2 failed in a late hook', returncode=1)\n"))
    assert _runner().main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 INCONCLUSIVE(exit 1, session finish incomplete)" in out and "NOT RED: X1" in out
    assert (root / "target.py").read_text() == "value = 1\n"


def test_an_exit_after_the_last_report_is_still_an_interrupt(tmp_path, capsys):
    """r5 R5-1, isolated: pytest.exit after the last test's teardown has reported. Every phase report exists and the
    counters agree; only pytest's interrupt hook shows the session did not end normally."""
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: _VAL + (
        "def test_first():\n    assert val() == 1\n"
        "def test_second():\n    assert val() == 1\n"))
    (root / "conftest.py").write_text(_VAL + (
        "def pytest_runtest_logfinish(nodeid, location):\n"
        "    if nodeid.endswith('test_second') and val() != 1:\n"
        "        pytest.exit('2 failed after the last test', returncode=1)\n"))
    assert _runner().main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 INCONCLUSIVE(exit 1, interrupted)" in out and "NOT RED: X1" in out
    assert (root / "target.py").read_text() == "value = 1\n"


def _two_failing(extra_conftest="", second_marker=""):
    def body(r):
        return _VAL + ("def test_first():\n    assert val() == 1\n"
                       f"{second_marker}def test_second():\n    assert val() == 1\n")
    return body


def _inconclusive(tmp_path, capsys, conftest, body=None, expect=""):
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], body or _two_failing())
    if conftest:
        (root / "conftest.py").write_text(_VAL + conftest)
    assert _runner().main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert f"X1 INCONCLUSIVE(exit 1, {expect})" in out and "NOT RED: X1" in out, out
    assert (root / "target.py").read_text() == "value = 1\n"


def test_a_session_finish_wrapper_abort_is_not_a_normal_end(tmp_path, capsys):
    """r6 R6-1: a conftest hook WRAPPER aborts after the inner session-finish hooks (and the old record) ran."""
    _inconclusive(tmp_path, capsys, (
        "@pytest.hookimpl(wrapper=True)\n"
        "def pytest_sessionfinish(session, exitstatus):\n"
        "    result = yield\n"
        "    if val() != 1:\n"
        "        pytest.exit('2 failed in a wrapper', returncode=1)\n"
        "    return result\n"), expect="session finish incomplete")


def test_a_tryfirst_wrapper_abort_is_still_seen(tmp_path, capsys):
    """The sentinel is registered after collection, so even a conftest tryfirst wrapper runs inside it."""
    _inconclusive(tmp_path, capsys, (
        "@pytest.hookimpl(wrapper=True, tryfirst=True)\n"
        "def pytest_sessionfinish(session, exitstatus):\n"
        "    result = yield\n"
        "    if val() != 1:\n"
        "        pytest.exit('2 failed in a tryfirst wrapper', returncode=1)\n"
        "    return result\n"), expect="session finish incomplete")


def test_a_non_strict_xpass_is_not_a_kill(tmp_path, capsys):
    """r6 R6-2: under the mutant the first test fails and an xfail-marked second test unexpectedly passes."""
    body = lambda r: _VAL + ("def test_first():\n    assert val() == 1\n"
                             "@pytest.mark.xfail(reason='expected')\n"
                             "def test_second():\n    assert val() == 2\n")
    _inconclusive(tmp_path, capsys, "", body=body, expect="marked xfail or skip")


def test_a_runtime_xfail_is_not_a_kill(tmp_path, capsys):
    """A test that declares itself xfail at run time keeps that status in the evidence."""
    body = lambda r: _VAL + ("def test_first():\n    assert val() == 1\n"
                             "def test_second(request):\n"
                             "    request.applymarker(pytest.mark.xfail(reason='late'))\n"
                             "    assert val() == 2\n")
    _inconclusive(tmp_path, capsys, "", body=body, expect="xfail or skip")


def test_a_plugin_registered_during_the_run_is_refused(tmp_path, capsys):
    """A plugin registered after the sentinel could wrap it: the evidence is refused."""
    body = lambda r: _VAL + ("def test_first(request):\n"
                             "    request.config.pluginmanager.register(object(), 'late-object')\n"
                             "    assert val() == 1\n"
                             "def test_second():\n    assert val() == 1\n")
    _inconclusive(tmp_path, capsys, "", body=body, expect="plugins registered late")


def test_a_rewritten_report_is_refused(tmp_path, capsys):
    """A conftest makereport wrapper turns the second test's real pass into a reported failure."""
    body = lambda r: _VAL + ("def test_first():\n    assert val() == 1\n"
                             "def test_second():\n    pass\n")
    _inconclusive(tmp_path, capsys, (
        "@pytest.hookimpl(wrapper=True)\n"
        "def pytest_runtest_makereport(item, call):\n"
        "    rep = yield\n"
        "    if rep.when == 'call' and item.name == 'test_second' and val() != 1:\n"
        "        rep.outcome = 'failed'\n"
        "    return rep\n"), body=body, expect="report mismatch")


def test_an_unconfigure_abort_leaves_no_evidence(tmp_path, capsys):
    _inconclusive(tmp_path, capsys, (
        "def pytest_unconfigure(config):\n"
        "    if val() != 1:\n"
        "        pytest.exit('2 failed at unconfigure', returncode=1)\n"), expect="no normal end")


_LATE = ("class Late:\n"
         "    def __init__(self, pm, base):\n"
         "        self.pm, self.base = pm, base\n"
         "    @pytest.hookimpl(wrapper=True, tryfirst=True)\n"
         "    def pytest_sessionfinish(self, session, exitstatus):\n"
         "        result = yield\n"
         "        (pluggy.PluginManager.unregister if self.base else type(self.pm).unregister)(self.pm, self)\n"
         "        if val() != 1:\n"
         "            pytest.exit('2 failed in a late wrapper', returncode=1)\n"
         "        return result\n")


def test_a_self_unregistering_late_wrapper_is_refused(tmp_path, capsys):
    """r7 R7-1: a test registers a tryfirst session-finish wrapper (outside the sentinel), which aborts after the
    sentinel's record and unregisters itself before the final census. The registration event is retained."""
    body = lambda r: _VAL + "import pluggy\n" + _LATE + (
        "def test_first(request):\n"
        "    request.config.pluginmanager.register(Late(request.config.pluginmanager, False), 'late-wrapper')\n"
        "    assert val() == 1\n"
        "def test_second():\n    assert val() == 1\n")
    _inconclusive(tmp_path, capsys, "", body=body, expect="plugins registered late")


def test_the_late_registration_evidence_is_produced_and_retained(tmp_path):
    """r7 R7-1, the producer itself: a plugin registered during the run and unregistered at once still counts."""
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: _VAL + (
        "def test_first(request):\n"
        "    pm = request.config.pluginmanager\n"
        "    p = object()\n"
        "    pm.register(p, 'brief')\n"
        "    pm.unregister(p)\n"
        "    assert val() == 1\n"))
    rc, out, bundle = _runner().run_mutant(root, ["test_scratch.py"])
    assert bundle is not None and bundle["late_plugins"] >= 1 and bundle["not_outermost"] == []
    assert _runner().classify(rc, ["test_scratch.py::test_first"], bundle)[0] == "INCONCLUSIVE(exit 0, plugins registered late)"


def test_an_outer_wrapper_without_a_registration_event_is_still_seen(tmp_path, capsys):
    """Defence in depth: a wrapper registered through pluggy's base class (bypassing pytest's registration hook) and
    unregistered before the census is still caught, because the sentinel checks it is outermost when it runs."""
    body = lambda r: _VAL + "import pluggy\n" + _LATE + (
        "def test_first(request):\n"
        "    pm = request.config.pluginmanager\n"
        "    pluggy.PluginManager.register(pm, Late(pm, True), 'hidden-wrapper')\n"
        "    assert val() == 1\n"
        "def test_second():\n    assert val() == 1\n")
    _inconclusive(tmp_path, capsys, "", body=body, expect="sentinel not outermost")


def test_a_hidden_call_wrapper_is_seen(tmp_path, capsys):
    """Only the sentinel's per-call outermost check can see this: a call wrapper registered through pluggy's base class
    (no registration event) wraps the second test's call and unregisters itself before the final census."""
    body = lambda r: _VAL + "import pluggy\n" + (
        "class Hidden:\n"
        "    def __init__(self, pm):\n"
        "        self.pm = pm\n"
        "    @pytest.hookimpl(wrapper=True, tryfirst=True)\n"
        "    def pytest_runtest_call(self, item):\n"
        "        try:\n"
        "            return (yield)\n"
        "        finally:\n"
        "            pluggy.PluginManager.unregister(self.pm, self)\n"
        "def test_first(request):\n"
        "    pm = request.config.pluginmanager\n"
        "    pluggy.PluginManager.register(pm, Hidden(pm), 'hidden-call-wrapper')\n"
        "    assert val() == 1\n"
        "def test_second():\n    assert val() == 1\n")
    _inconclusive(tmp_path, capsys, "", body=body, expect="sentinel not outermost")


def test_the_runner_refuses_options_in_a_named_test_list(tmp_path, capsys):
    canary = tmp_path / "canary"
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py", "-x"], lambda r: _VAL + (
        "def test_first():\n    assert val() == 1\n"
        f"def test_second():\n    pathlib.Path({str(canary)!r}).write_text('ran')\n    assert val() == 1\n"))
    assert _runner().main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 INVALID" in out and "NOT RED: X1" in out and not canary.exists()
    assert (root / "target.py").read_text() == "value = 1\n"
