"""C2 side item (e): the catcher-grouped framing screen's research code (scripts/audit/c2_framing/screen.py)."""
import hashlib
import json
import math
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.audit.c2_framing import mutant_runner as R
from scripts.audit.c2_framing import screen as S

ROOT = Path(__file__).resolve().parents[3]


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
    source = (ROOT / "src/bts/features/compute.py").read_text()       # r9: read, not introspected
    lines = source[source.index("\ndef compute_all_features(") + 1:].split("\n")
    body = [lines[0]]
    for line in lines[1:]:
        if line and not line[0].isspace():                              # the next top-level statement
            break
        body.append(line)
    text = "\n".join(body)
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
    seeds = json.loads((ROOT / "data/seed_sets/canonical-n10.json").read_text())
    assert S.STAGE_ONE_SEEDS == tuple(seeds["seeds"][:3])


def test_the_registered_scoring_settings_are_the_scorers_defaults():
    import ast
    tree = ast.parse((ROOT / "src/bts/validate/scorecard.py").read_text())
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "compute_full_scorecard")
    args = fn.args.args[len(fn.args.args) - len(fn.args.defaults):] + fn.args.kwonlyargs
    defaults = list(fn.args.defaults) + list(fn.args.kw_defaults)
    sig = {a.arg: ast.literal_eval(d) for a, d in zip(args, defaults) if d is not None}
    assert S.SCORING == {"mc_trials": 10_000, "season_length": 180}
    assert {k: sig[k] for k in S.SCORING} == S.SCORING


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
    real = Path.read_text
    monkeypatch.setattr(Path, "read_text",
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


REAL_READ_TEXT = Path.read_text


def _register_plus(rows, *a):
    """The real register with these rows in place of any real row of the same id, and without the real rows named in a.
    The lookup takes a row id's first line, and since Eric's real release (11e0cfd) a row appended after the real one
    was never read."""
    from scripts.audit.c1 import admission as A
    ids = [r.split("|")[1].strip() for r in rows] + list(a)
    real = REAL_READ_TEXT(A.REPO / S.REGISTER_REL).split("\n")
    kept = [x for x in real if not any(x.startswith(f"| {i} |") for i in ids)]
    return "\n".join(kept + rows) + "\n"


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
    with pytest.raises(SystemExit, match="registered seed"):
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
    pins[S.LOOKUP_NAME] = hashlib.sha256(lookup).hexdigest()
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
    text = _register_plus([_release_row(seed1.name, source=source)])
    real_read = Path.read_text

    def read_text(self, *a, **k):
        return text if self == A.REPO / S.REGISTER_REL else real_read(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", read_text)
    return seed1


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
    text = _register_plus([_release_row(fake.name)])
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
    text = _register_plus([_release_row("bbbbbbb-20261007T000000Z")])
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
        card = compute_full_scorecard(pd.concat(parts, ignore_index=True), mc_trials=S.SCORING["mc_trials"], season_length=S.SCORING["season_length"])
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
        card = compute_full_scorecard(pd.concat([p24, p24], ignore_index=True), mc_trials=S.SCORING["mc_trials"], season_length=S.SCORING["season_length"])
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
                       pins=json.loads((d / "manifest.json").read_text())["input_pins"])


def test_a_rescoring_failure_is_a_refusal_not_a_crash(three_runs, monkeypatch):
    """r5 R5-2: anything the scorer cannot score is refused through RunInvalid, never an unexpected exception."""
    import bts.validate.scorecard as SC
    out, dirs = three_runs
    monkeypatch.setattr(SC, "compute_full_scorecard", lambda *a, **k: 1 + "x")
    with pytest.raises(S.RunInvalid, match="cannot be rescored"):
        S.validate_run(dirs[0], S.STAGE_ONE_SEEDS[0], out_root=out, identity=IDENT,
                       pins=json.loads((dirs[0] / "manifest.json").read_text())["input_pins"])


def test_validate_refuses_pins_other_than_the_ten_even_when_admitted(three_runs):
    """The run's own ten-name check, isolated: the trusted pins are the same nine, so only 'pins' can refuse."""
    out, dirs = three_runs
    d = dirs[0]

    def drop(r):
        r["input_pins"].pop(S.LOOKUP_NAME)
        r["inputs_digest"] = S.pins_digest(r["input_pins"])
    _rewrite(d / "manifest.json", drop)
    nine = json.loads((d / "manifest.json").read_text())["input_pins"]
    with pytest.raises(S.RunInvalid, match=r"\['pins'\]"):
        S.validate_run(d, S.STAGE_ONE_SEEDS[0], out_root=out, identity=IDENT, pins=nine)


# ---------------------------------------------------------------- stage two (addendum 2026-10-08-prereg-c2-framing-stage-two)

IDENT2 = {k: f"{k}-two" for k in S.IDENTITY_KEYS}
STAGE_TWO_HEAD = "d" * 40


def _cap_row(cap, budget, source="Eric 2026-10-08, relayed"):
    return (f"| C2-framing-stage-two-cap | the stage-two package | **RULED 2026-10-08 (Eric): RAISE the shared C1/C2 "
            f"compute cap from 100 to {cap} CPU-hours for the framing screen's stage two; declared budget {budget} "
            f"CPU-hours per seed for seeds 4–10** | {source} |")


def _with_register(monkeypatch, rows):
    """The real register, without Eric's real stage-two row, plus these rows, as every reader of the file sees it."""
    from scripts.audit.c1 import admission as A
    text = _register_plus(rows, "C2-framing-stage-two-cap")

    def read_text(self, *a, **k):
        return text if self == A.REPO / S.REGISTER_REL else REAL_READ_TEXT(self, *a, **k)
    monkeypatch.setattr(Path, "read_text", read_text)


def _mixed(calls):
    """A walk-forward stub under which A ties the baseline in 2024 and beats it in 2025 (B ties both): A is
    inconclusive over any number of seeds, B negative."""
    def wf(df, season, retrain_every, blend_configs, game_probability_mode):
        calls.append(season)
        is_a = S.NEW_COL in blend_configs[0][1] and S.OLD_COL not in blend_configs[0][1]
        part = df[df["season"] == season]
        rows = []
        for date, g in part.groupby("date"):
            order = sorted(g["batter_id"].unique(), reverse=is_a and season == 2025)
            for rank, b in enumerate(order, start=1):
                rows.append({"date": date, "rank": rank, "batter_id": int(b),
                             "game_pk": int(g.loc[g["batter_id"] == b, "game_pk"].iloc[0]),
                             "p_game_hit": 0.9 - 0.01 * rank, "actual_hit": 1, "n_pas": 4})
        return pd.DataFrame(rows)
    return wf


def _run_mixed(stubbed, seed):
    out, inputs, _ = stubbed
    return S.run(seed, out, inputs, walk_forward=_mixed([]), _test_out_root=out)


def _pin_stage_one(monkeypatch, out):
    """Stage one as accepted: its runs' names, and a hash list of their files pinned by its own sha256."""
    lines = []
    names = {}
    for seed in S.STAGE_ONE_SEEDS:
        d = _run_dir(out, seed)
        names[seed] = d.name
        for f in sorted(d.iterdir()):
            lines.append(f"{hashlib.sha256(f.read_bytes()).hexdigest()}  {f.relative_to(out).as_posix()}")
    listing = out.parent / "runs.sha256"
    listing.write_text("\n".join(lines) + "\n")
    monkeypatch.setattr(S, "STAGE_ONE_RUNS", names)
    monkeypatch.setattr(S, "STAGE_ONE_FILES", str(listing))
    monkeypatch.setattr(S, "STAGE_ONE_FILES_SHA256", hashlib.sha256(listing.read_bytes()).hexdigest())
    monkeypatch.setattr(S, "STAGE_ONE_IDENTITY", dict(IDENT))


def _to_stage_two(stubbed, monkeypatch, rows):
    """Stage two's admission (its own identity and head, the same pins) and the register with these rows."""
    out, _, adm = stubbed
    monkeypatch.setattr(S, "admission_gate", lambda: (STAGE_TWO_HEAD, {"input_pins": adm.pins}, dict(IDENT2)))
    monkeypatch.setattr(S.ledger, "CAP_H", 165.0)
    _with_register(monkeypatch, [_release_row(_run_dir(out, S.STAGE_ONE_SEEDS[0]).name)] + rows)


def _stage_one_by(stubbed, monkeypatch, fn):
    """Stage one's three runs under stage one's identity: seed 1, Eric's release, then seeds 2 and 3."""
    out, _, adm = stubbed
    assert fn(stubbed, S.STAGE_ONE_SEEDS[0]) == 0
    adm.head = "c" * 40
    _with_register(monkeypatch, [_release_row(_run_dir(out, S.STAGE_ONE_SEEDS[0]).name)])
    for seed in S.STAGE_ONE_SEEDS[1:]:
        assert fn(stubbed, seed) == 0
    _pin_stage_one(monkeypatch, out)


def _admitted_heads(repo, identity, head):
    """The stubbed repository with both stages: stage one's two heads under IDENT, stage two's under IDENT2."""
    if identity == IDENT2:
        return [] if head == STAGE_TWO_HEAD else [f"run HEAD {str(head)[:7]} is not admitted"]
    return _fake_head_admitted(repo, identity, head)


@pytest.fixture
def stage_one(stubbed, monkeypatch):
    monkeypatch.setattr(S, "ny_clock", lambda: (12, 0))              # outside the launch window, whatever the hour
    monkeypatch.setattr(S, "head_admitted", _admitted_heads)
    _stage_one_by(stubbed, monkeypatch, _run_mixed)
    _to_stage_two(stubbed, monkeypatch, [_cap_row("165", "20")])
    return stubbed


@pytest.fixture
def ten_runs(stage_one):
    out, _, _ = stage_one
    for seed in S.STAGE_TWO_SEEDS:
        assert _run_mixed(stage_one, seed) == 0
    return out, [_run_dir(out, s) for s in S.SEEDS]


def test_stage_two_seeds_are_the_rest_of_canonical_n10_in_order():
    seeds = json.loads((ROOT / "data/seed_sets/canonical-n10.json").read_text())["seeds"]
    assert S.STAGE_TWO_SEEDS == tuple(seeds[3:10]) and S.SEEDS == tuple(seeds[:10]) == S.STAGE_ONE_SEEDS + S.STAGE_TWO_SEEDS


def test_the_stage_two_row_needs_the_exact_ruling_eric_a_raise_and_a_positive_budget():
    assert S.stage_two_release(_cap_row("165", "20")) == (165.0, 20.0)
    assert S.stage_two_release(_cap_row("165", "20", source="Ericsson, manager; no owner ruling")) is None
    assert S.stage_two_release(_cap_row("165", "20", source="Manager 2026-10-08")) is None
    assert S.stage_two_release(_cap_row("165", "0")) is None
    assert S.stage_two_release(_cap_row("100", "20")) is None                         # not a raise
    assert S.stage_two_release(_cap_row("90", "20")) is None
    assert S.stage_two_release(_cap_row("165", "20").replace("RAISE the", "KEEP the")) is None
    assert S.stage_two_release(_cap_row("165", "20").replace("seeds 4–10**", "seeds 4–10** and more")) is None
    assert S.stage_two_release(_release_row("aaaaaaa-20261007T000000Z")) is None
    assert S.stage_two_release("no row here") is None


def test_the_launchers_cap_is_the_cap_in_erics_row():
    """The launcher's cap constant equals the cap Eric ruled (row C2-framing-stage-two-cap), read from the real register,
    and both are 165: a silent edit of either goes red."""
    assert S.stage_two_release((ROOT / S.REGISTER_REL).read_text()) == (S.ledger.CAP_H, 20.0) == (165.0, 20.0)


def test_disposition_over_ten_seeds_needs_six_passes():
    strong = _seed(0.02, 0.02, True)
    weak = _seed(0.02, 0.02, False)
    assert S.disposition([strong] * 6 + [weak] * 4, 10)["disposition"] == "positive"
    five = S.disposition([strong] * 5 + [weak] * 5, 10)
    assert five["disposition"] == "inconclusive" and five["per_seed_passes"] == 5 and five["n_seeds"] == 10
    assert S.disposition([_seed(-0.01, -0.01, False)] * 10, 10)["disposition"] == "negative"
    assert S.disposition([strong] * 9, 10)["disposition"] == "incomplete"
    assert S.disposition([strong] * 10, 3)["disposition"] == "incomplete"


def test_stage_two_seeds_run_one_at_a_time_in_order_under_their_own_identity(stage_one):
    out, _, _ = stage_one
    with pytest.raises(SystemExit, match="in order: earlier seed 2048"):
        _run_mixed(stage_one, S.STAGE_TWO_SEEDS[1])
    assert _run_mixed(stage_one, S.STAGE_TWO_SEEDS[0]) == 0
    assert _run_mixed(stage_one, S.STAGE_TWO_SEEDS[1]) == 0
    man = json.loads((_run_dir(out, S.STAGE_TWO_SEEDS[0]) / "manifest.json").read_text())
    assert man["identity"] == IDENT2 and man["head"] == STAGE_TWO_HEAD


@pytest.mark.parametrize("damage", ["two_runs", "stopped"])
def test_an_earlier_stage_two_seed_must_hold_one_complete_run(stage_one, damage):
    out, _, _ = stage_one
    assert _run_mixed(stage_one, S.STAGE_TWO_SEEDS[0]) == 0
    d = _run_dir(out, S.STAGE_TWO_SEEDS[0])
    if damage == "two_runs":
        (d.parent / "zzzzzzz-20261009T000000Z").mkdir()
        expect = "earlier seed 2048 has 2 runs, not one"
    else:
        (d / "STOPPED.json").write_text("{}")
        expect = "earlier seed 2048's run is not a complete admitted run"
    with pytest.raises(SystemExit, match=expect):
        _run_mixed(stage_one, S.STAGE_TWO_SEEDS[1])


def test_stage_two_needs_erics_cap_row(stage_one, monkeypatch):
    _to_stage_two(stage_one, monkeypatch, [])
    with pytest.raises(SystemExit, match="C2-framing-stage-two-cap"):
        _run_mixed(stage_one, S.STAGE_TWO_SEEDS[0])
    _to_stage_two(stage_one, monkeypatch, [_cap_row("165", "20", source="Manager 2026-10-08")])
    with pytest.raises(SystemExit, match="C2-framing-stage-two-cap"):
        _run_mixed(stage_one, S.STAGE_TWO_SEEDS[0])


def test_stage_two_needs_the_rows_cap_to_be_the_launchers(stage_one, monkeypatch):
    monkeypatch.setattr(S.ledger, "CAP_H", 150.0)
    with pytest.raises(SystemExit, match="is not the launcher's cap 150"):
        _run_mixed(stage_one, S.STAGE_TWO_SEEDS[0])


@pytest.mark.parametrize("damage", ["extra_file", "changed_byte", "second_run", "listing", "renamed"])
def test_stage_two_needs_stage_ones_accepted_runs_byte_for_byte(stage_one, monkeypatch, damage):
    """Each damage is one only its own check can refuse: a second run named to sort after the accepted one leaves the
    accepted run's bytes and validation intact."""
    out, _, _ = stage_one
    d = _run_dir(out, S.STAGE_ONE_SEEDS[1])
    if damage == "extra_file":                       # validate_run never reads it: only the hash list can refuse
        (d / "notes.txt").write_text("x\n")
        expect = "not the accepted bytes"
    elif damage == "changed_byte":
        (d / "units.json").write_bytes((d / "units.json").read_bytes() + b" ")
        expect = "not the accepted bytes"
    elif damage == "second_run":
        (d.parent / "zzzzzzz-20261009T000000Z").mkdir()
        expect = "not exactly its accepted run"
    elif damage == "listing":
        monkeypatch.setattr(S, "STAGE_ONE_FILES_SHA256", "0" * 64)
        expect = "does not have its pinned sha256"
    else:
        monkeypatch.setattr(S, "STAGE_ONE_RUNS", {**S.STAGE_ONE_RUNS, S.STAGE_ONE_SEEDS[1]: "zzzzzzz-20261009T000000Z"})
        expect = "not exactly its accepted run"
    with pytest.raises(SystemExit, match=f"stage one is not its accepted runs: .*{expect}"):
        _run_mixed(stage_one, S.STAGE_TWO_SEEDS[0])


def test_stage_ones_runs_are_validated_under_stage_ones_identity(stage_one, monkeypatch):
    monkeypatch.setattr(S, "STAGE_ONE_IDENTITY", dict(IDENT2))
    with pytest.raises(SystemExit, match=r"stage one is not its accepted runs: .*\['admitted identity'\]"):
        _run_mixed(stage_one, S.STAGE_TWO_SEEDS[0])


def test_stage_two_needs_an_inconclusive_stage_one_variant(stubbed, monkeypatch):
    """Pre-registration §5: only an inconclusive variant justifies stage two. Under the plain stub stage one gives A
    positive and B negative."""
    monkeypatch.setattr(S, "head_admitted", _admitted_heads)
    _stage_one_by(stubbed, monkeypatch, _run)
    _to_stage_two(stubbed, monkeypatch, [_cap_row("165", "20")])
    with pytest.raises(SystemExit, match=r"needs an inconclusive stage-one variant .*\['positive', 'negative'\]"):
        _run_mixed(stubbed, S.STAGE_TWO_SEEDS[0])


def test_the_launch_window_is_closed_from_0045_to_0310():
    assert S.launch_window_problem(0, 44) is None and S.launch_window_problem(3, 10) is None
    assert S.launch_window_problem(23, 59) is None and S.launch_window_problem(12, 0) is None
    for hour, minute in ((0, 45), (1, 30), (3, 0), (3, 9)):
        assert "00:45 to 03:10" in S.launch_window_problem(hour, minute)


def test_launch_runs_stage_two_with_erics_budget_outside_the_window(stage_one, monkeypatch):
    out, inputs, _ = stage_one
    calls = []

    class R:
        returncode = 0
    execute = lambda cmd, cwd: (calls.append(cmd), R())[1]      # noqa: E731
    monkeypatch.setattr(S, "ny_clock", lambda: (1, 0))
    with pytest.raises(SystemExit, match="00:45 to 03:10"):
        S.launch(S.STAGE_TWO_SEEDS[0], out, inputs, _test_out_root=out, execute=execute)
    assert calls == []
    monkeypatch.setattr(S, "ny_clock", lambda: (12, 0))
    assert S.launch(S.STAGE_TWO_SEEDS[0], out, inputs, _test_out_root=out, execute=execute) == 0
    cmd = calls[0]
    assert cmd[cmd.index("--cpu-hours") + 1] == "20" and cmd[cmd.index("--name") + 1] == "c2-framing-seed4"
    assert cmd[cmd.index("--seed") + 1] == str(S.STAGE_TWO_SEEDS[0]) and "BTS_LGBM_DETERMINISTIC=1" in cmd


def test_run_refuses_stage_two_seeds_inside_the_launch_window(stage_one, monkeypatch):
    """The job itself checks the window (the wrapper's own function), so a seed started by hand through the C1 launcher
    still refuses; nothing is claimed. Both edges: 00:45 refuses, 03:10 and 00:44 run."""
    out, _, _ = stage_one
    root = out / f"seed_{S.STAGE_TWO_SEEDS[0]}"
    for hour, minute in ((0, 45), (1, 30), (3, 9)):
        monkeypatch.setattr(S, "ny_clock", lambda: (hour, minute))
        with pytest.raises(SystemExit, match="00:45 to 03:10"):
            _run_mixed(stage_one, S.STAGE_TWO_SEEDS[0])
        assert not root.exists() or not any(x.is_dir() for x in root.iterdir())
    monkeypatch.setattr(S, "ny_clock", lambda: (3, 10))
    assert _run_mixed(stage_one, S.STAGE_TWO_SEEDS[0]) == 0
    monkeypatch.setattr(S, "ny_clock", lambda: (0, 44))
    assert _run_mixed(stage_one, S.STAGE_TWO_SEEDS[1]) == 0


def test_stage_one_seeds_have_no_launch_window(stubbed, monkeypatch):
    monkeypatch.setattr(S, "ny_clock", lambda: (1, 30))
    assert _run(stubbed, S.STAGE_ONE_SEEDS[0]) == 0


def _validation_passes_0045(monkeypatch):
    """A clock at 00:44 that the stage-two validation moves to 00:45 (a slow validation across the window's start)."""
    now = [(0, 44)]
    real = S.stage_two_allowed

    def slow(seed, register, out, **k):
        now[0] = (0, 45)
        return real(seed, register, out, **k)
    monkeypatch.setattr(S, "stage_two_allowed", slow)
    monkeypatch.setattr(S, "ny_clock", lambda: now[0])


def test_run_checks_the_window_after_its_validation(stage_one, monkeypatch):
    _validation_passes_0045(monkeypatch)
    with pytest.raises(SystemExit, match="00:45 to 03:10"):
        _run_mixed(stage_one, S.STAGE_TWO_SEEDS[0])


def test_launch_checks_the_window_after_its_validation(stage_one, monkeypatch):
    out, inputs, _ = stage_one
    calls = []

    class R:
        returncode = 0
    _validation_passes_0045(monkeypatch)
    with pytest.raises(SystemExit, match="00:45 to 03:10"):
        S.launch(S.STAGE_TWO_SEEDS[0], out, inputs, _test_out_root=out,
                 execute=lambda cmd, cwd: (calls.append(cmd), R())[1])
    assert calls == []


def test_the_clock_reads_new_york_time():
    """Against the system clock asked for America/New_York (a minute boundary between the two reads is allowed)."""
    want = subprocess.run(["env", "TZ=America/New_York", "date", "+%H %M"], capture_output=True, text=True,
                          check=True).stdout.split()
    hour, minute = S.ny_clock()
    assert (hour * 60 + minute - int(want[0]) * 60 - int(want[1])) % 1440 in (0, 1)


def test_stage_two_is_admitted_under_its_own_exposure_row_and_design():
    assert S.EXPOSURE_ROW == "X-36" and S.SCOPE == "catcher framing screen stage two"
    assert S.DESIGN in S.CLOSURE and S.DESIGN_TWO in S.CLOSURE and (ROOT / S.DESIGN_TWO).is_file()


def test_stage_ones_identity_and_runs_are_what_the_repository_holds():
    """The stage-one constants, checked against git: the admission record at stage one's admission commit, the review
    report at its exposure commit, and the committed hash list naming the three accepted runs."""
    def show(path):
        return subprocess.run(["git", "-C", str(ROOT), "show", path], capture_output=True, check=True).stdout
    ident = S.STAGE_ONE_IDENTITY
    record = show(f"22d31f23a0c2683a7dfe530ea3380861c090667f:{S.ADMISSION_REL}")
    assert hashlib.sha256(record).hexdigest() == ident["admission_sha256"]
    adm = json.loads(record)
    assert adm["reviewed_commit"] == ident["reviewed_commit"] and adm["exposure_commit"] == ident["exposure_commit"]
    assert adm["review_report"] == ident["review_report"]
    report = show(f"{ident['exposure_commit']}:{ident['review_report']}")
    assert hashlib.sha256(report).hexdigest() == ident["review_report_sha256"]
    listing = (ROOT / S.STAGE_ONE_FILES).read_bytes()
    assert hashlib.sha256(listing).hexdigest() == S.STAGE_ONE_FILES_SHA256
    names = {x.split()[1].split("/")[1] for x in (ROOT / S.STAGE_ONE_FILES).read_text().split("\n") if x.strip()}
    assert sorted(S.STAGE_ONE_RUNS) == sorted(S.STAGE_ONE_SEEDS)
    assert names == {S.STAGE_ONE_RUNS[s] for s in S.STAGE_ONE_SEEDS}


def test_aggregate_the_ten_registered_seeds_across_both_stages(ten_runs):
    out, d = ten_runs
    agg = S.aggregate_stage_two(d, _test_out_root=out)
    assert agg["seeds"] == list(S.SEEDS) and agg["identities"] == {"stage_one": IDENT, "stage_two": IDENT2}
    assert agg["heads"] == ["a" * 40] + ["c" * 40] * 2 + [STAGE_TWO_HEAD] * 7
    a, b = agg["variants"]["A"], agg["variants"]["B"]
    assert a["n_seeds"] == 10 and a["disposition"] == "inconclusive" and a["mean_p_at_1_delta"]["2024"] == 0.0
    assert a["mean_p_at_1_delta"]["2025"] > 0 and b["disposition"] == "negative"


@pytest.mark.parametrize("pick", [lambda d: d[:9], lambda d: d[:3], lambda d: d[:9] + d[:1], lambda d: d + d[:1]])
def test_aggregate_stage_two_refuses_partial_duplicate_or_extra_runs(ten_runs, pick):
    out, d = ten_runs
    with pytest.raises(S.RunInvalid, match="exactly 10 distinct run directories"):
        S.aggregate_stage_two(pick(d), _test_out_root=out)


def test_aggregate_stage_two_refuses_a_foreign_seed(ten_runs):
    import shutil
    out, d = ten_runs
    foreign = out / "seed_42" / d[9].name
    shutil.copytree(d[9], foreign)
    with pytest.raises(S.RunInvalid, match="registered"):
        S.aggregate_stage_two(d[:9] + [foreign], _test_out_root=out)


def test_aggregate_stage_two_validates_each_stage_under_its_own_identity(ten_runs):
    out, d = ten_runs
    _rewrite(d[3] / "manifest.json", lambda r: r.update(identity=dict(IDENT)))      # a stage-two run of stage one's identity
    with pytest.raises(S.RunInvalid, match=r"\['admitted identity'\]"):
        S.aggregate_stage_two(d, _test_out_root=out)


def test_aggregate_stage_two_takes_only_stage_ones_accepted_run_directories(ten_runs):
    out, d = ten_runs
    other = d[0].parent / "zzzzzzz-20261009T000000Z"           # a path in the namespace, never a run
    with pytest.raises(S.RunInvalid, match="not stage one's accepted run"):
        S.aggregate_stage_two([other] + d[1:], _test_out_root=out)


def test_aggregate_stage_two_needs_stage_ones_accepted_bytes(ten_runs):
    out, d = ten_runs
    (d[0] / "notes.txt").write_text("x\n")
    with pytest.raises(S.RunInvalid, match="accepted"):
        S.aggregate_stage_two(d, _test_out_root=out)


def test_aggregate_stage_two_refuses_stages_that_disagree(ten_runs):
    """A parameter validate_run leaves to the cross-run check (its deterministic flags kept), changed consistently in
    all seven stage-two runs: only the agreement across the stages can refuse."""
    out, d = ten_runs
    for x in d[3:]:
        _rewrite(x / "manifest.json", lambda r: r["lgb_params"].update({"learning_rate": 0.1}))
    with pytest.raises(S.RunInvalid, match="runs disagree on lgb_params"):
        S.aggregate_stage_two(d, _test_out_root=out)


# ---------------------------------------------------------------- the mutant runner's classification (r2 R2-4)

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
         "late_plugins": 0, "not_outermost": [], "marked": [], "plugin_changes": [], "foreign_plugins": [],
         "boundary": []}
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
    # r9: pluggy's hook-implementation add/remove code ran after the sentinel registered, by any API (r8 R8-1)
    (1, _bundle(ONE_FAIL, [A], plugin_changes=["HookCaller._add_hookimpl"]), "INCONCLUSIVE(exit 1, plugin system changed)"),
    (1, _bundle(ONE_FAIL, [A], plugin_changes=None), "INCONCLUSIVE(exit 1, plugin system changed)"),
    # r9: a plugin that is neither pytest's own nor the runner's was registered when the sentinel was
    (1, _bundle(ONE_FAIL, [A], foreign_plugins=["conftest"]), "INCONCLUSIVE(exit 1, foreign plugins)"),
    (1, _bundle(ONE_FAIL, [A], foreign_plugins=None), "INCONCLUSIVE(exit 1, foreign plugins)"),
    # r10: the run's audit hook refused code from outside the boundary (r9 R9-2, R9-3), even if the test caught it
    (1, _bundle(ONE_FAIL, [A], boundary=["not committed: /r/numpy.py"]),
     "INCONCLUSIVE(exit 1, code outside the boundary)"),
    (1, _bundle(ONE_FAIL, [A], boundary=None), "INCONCLUSIVE(exit 1, code outside the boundary)"),
]
CLASSIFY_IDS = ["red", "red_two_failed", "survived", "no_evidence", "exit_mismatch", "interrupted", "internal_error",
                "collection_error", "finish_raised", "finish_missing", "late_plugin", "not_outermost", "xpass", "marked", "skipped",
                "errors", "fail_fast", "teardown_cut_short", "unintended_node", "report_flipped", "call_unobserved",
                "failed_count_mismatch", "collected_count_mismatch",
                "exit_status_mismatch", "plugin_changed", "plugin_changes_missing", "foreign_plugin",
                "foreign_plugins_missing", "boundary", "boundary_missing"]


@pytest.mark.parametrize("rc, bundle, verdict", CLASSIFY, ids=CLASSIFY_IDS)
def test_the_runner_classifies_only_complete_clean_failures_as_red(rc, bundle, verdict):
    assert R.classify(rc, INTENDED, bundle)[0] == verdict


_GIT = ["git", "-c", "user.name=scratch", "-c", "user.email=scratch@example.invalid", "-c", "commit.gpgsign=false",
        "-c", "core.hooksPath=/dev/null"]


def _commit_all(repo):
    """Commit everything in a scratch root (r10: a run executes only installed and committed code)."""
    for args in (["init", "-q"], ["add", "-A"], ["commit", "-q", "--allow-empty", "-m", "scratch"]):
        subprocess.run([*_GIT, "-C", str(repo), *args], check=True, capture_output=True)


def _scratch_mutant(tmp_path, tests, body):
    """A scratch root of its own (the runner's rootdir and cwd, so collection never leaves it), committed: a target, a
    test module, and a spec naming tests relative to that root (r4: the out-of-tree collection that failed in a
    sandbox)."""
    root = tmp_path / "scratch"
    root.mkdir()
    (root / "target.py").write_text("value = 1\n")
    (root / "test_scratch.py").write_text(body(root))
    _commit_all(root)
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([{"id": "X1", "rule": "scratch", "file": "target.py", "old": "value = 1",
                                 "new": "value = 2", "tests": tests}]))
    return spec, root


# r9: an in-scope scratch suite reads the target as text, within the reviewed vocabulary
_READ = ("import pathlib\n"
         "def val():\n"
         "    return pathlib.Path(__file__).with_name('target.py').read_text()\n")
_ONE = "'value = 1\\n'"
# out of scope (refused by the gate): loading the target through importlib
_VAL = ("import importlib.util, pathlib, pytest\n"
        "def val():\n"
        "    spec = importlib.util.spec_from_file_location('target', pathlib.Path(__file__).with_name('target.py'))\n"
        "    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m.value\n")


def _two_read_failing(canary=None):
    second = f"    pathlib.Path({str(canary)!r}).write_text('ran')\n" if canary else ""
    return lambda r: _READ + (f"def test_first():\n    assert val() == {_ONE}\n"
                              f"def test_second():\n{second}    assert val() == {_ONE}\n")


def test_the_runner_clears_inherited_fail_fast_and_runs_every_named_test(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("PYTEST_ADDOPTS", "-x")
    canary = tmp_path / "canary"
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], _two_read_failing(canary))
    assert R.main(str(spec), root=root) == 0
    out = capsys.readouterr().out
    assert "X1 RED" in out and canary.read_text() == "ran"                 # the second body executed
    assert (root / "target.py").read_text() == "value = 1\n"               # restored


def test_the_runs_ignore_the_roots_ini_file(tmp_path, capsys):
    """r9: ini settings never apply (the run's configuration is an empty file): neither an ini fail-fast nor an ini
    fixture requirement (usefixtures) reaches the collection or the run."""
    canary = tmp_path / "canary"
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], _two_read_failing(canary))
    (root / "pytest.ini").write_text("[pytest]\naddopts = -x\nusefixtures = planted\n")
    assert R.main(str(spec), root=root) == 0
    assert "X1 RED" in capsys.readouterr().out and canary.read_text() == "ran"


def test_the_runs_load_no_conftest(tmp_path, capsys):
    """r9: conftests are never loaded (--noconftest), in the collection or the run, so their hooks never run."""
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], _two_read_failing())
    (root / "conftest.py").write_text("import pathlib\npathlib.Path(__file__).with_name('conftest_ran').write_text('y')\n")
    _commit_all(root)                                      # r10: committed, so only --noconftest keeps it from loading
    assert R.main(str(spec), root=root) == 0
    assert "X1 RED" in capsys.readouterr().out and not (root / "conftest_ran").exists()


def test_the_run_environment_is_scrubbed(monkeypatch):
    """r9: no inherited PYTHON* or PYTEST* variable reaches a run (PYTHONPATH, PYTEST_PLUGINS, PYTEST_ADDOPTS...);
    entry-point plugins are off, bytecode is neither written nor read, and the interpreter adds no script or user path.
    r10 (R9-1): collection is cut off at the root."""
    for k in ("PYTHONPATH", "PYTHONSTARTUP", "PYTEST_PLUGINS", "PYTEST_ADDOPTS", "PYTHONINSPECT"):
        monkeypatch.setenv(k, "planted")
    env = R._env()
    assert not [k for k in env if k.startswith(("PYTHON", "PYTEST")) and env[k] == "planted"]
    assert env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] == "1" and env["PYTHONDONTWRITEBYTECODE"] == "1"
    assert env["PYTHONPYCACHEPREFIX"].startswith("/dev/null/")                # no cache can exist there
    root = Path("/r")
    assert R._python()[1:] == ["-B", "-P", "-s"]
    args = R._args(root, "x")
    assert args[args.index("-c") + 1] == "/dev/null" and "--noconftest" in args and args[-1] == "x"
    assert "--rootdir=/r" in args and "--confcutdir=/r" in args


def test_the_runner_refuses_options_in_a_named_test_list(tmp_path, capsys):
    canary = tmp_path / "canary"
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py", "-x"], _two_read_failing(canary))
    assert R.main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 INVALID" in out and "NOT RED: X1" in out and not canary.exists()
    assert (root / "target.py").read_text() == "value = 1\n"


# ---------------------------------------------------------------- r9: the runner's scope, enforced by a static gate

def _gate(tmp_path, source, name="test_scratch.py", extra=None):
    root = tmp_path / "gate"
    (root / name).parent.mkdir(parents=True, exist_ok=True)
    (root / name).write_bytes(source if isinstance(source, bytes) else source.encode())
    for rel, text in (extra or {}).items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    return R.scope_problems(root, [name])


GATE_REFUSED = [
    ("import_unlisted", "import importlib\n", "import importlib"),
    ("from_import_unlisted", "from os import name\n", "import os"),       # `name` is an allowed attribute
    ("relative_import", "from . import x\n", "relative import"),
    ("star_import", "from pathlib import *\n", "import pathlib.*"),
    ("dotted_without_alias", "import scripts.audit.c2_framing.screen\n", "without an alias"),
    ("pytest_alias", "import pytest as p\n", "pytest imported as p"),
    ("import_rebinds_builtin", "import json as len\n", "rebinds builtin len"),
    ("from_import_rebinds_builtin", "from json import loads as len\n", "rebinds builtin len"),
    ("from_pytest_main", "from pytest import main\n", "import pytest.main"),
    ("pytest_exit", "import pytest\npytest.exit('x')\n", "pytest.exit"),
    ("pytest_main", "import pytest\npytest.main([])\n", "pytest.main"),
    ("pytest_bare", "import pytest\nx = [pytest]\n", "pytest used other than as pytest.<name>"),
    ("from_pytest", "from pytest import exit\n", "import pytest.exit"),
    ("builtin_unlisted", "getattr(1, 'real')\n", "builtin getattr"),
    ("dunder_import", "__import__('os')\n", "name __import__"),
    ("dunder_builtins", "__builtins__\n", "name __builtins__"),
    ("dunder_attribute", "x = (1).__class__\n", "attribute __class__"),
    ("attribute_unlisted", "import json\njson.JSONDecoder\n", "attribute JSONDecoder"),
    ("frame_attribute", "def f(e):\n    return e.tb_frame\n", "attribute tb_frame"),
    ("fstring_attribute", "x = f'{(1).__class__}'\n", "attribute __class__"),
    ("annotation_attribute", "x: (1).__class__ = 1\n", "attribute __class__"),
    ("decorator_attribute", "import json\n@json.JSONDecoder\ndef f():\n    pass\n", "attribute JSONDecoder"),
    ("hook_function", "def pytest_sessionfinish(session):\n    pass\n", "name pytest_sessionfinish"),
    ("hook_method", "class P:\n    def pytest_runtest_call(self):\n        pass\n", "name pytest_runtest_call"),
    ("pytest_plugins", "pytest_plugins = ['p']\n", "name pytest_plugins"),
    ("pytestmark", "pytestmark = []\n", "name pytestmark"),
    ("module_getattr", "def __getattr__(name):\n    return 1\n", "name __getattr__"),
    ("module_dunder_init", "def __init__():\n    pass\n", "name __init__"),
    ("dunder_loader", "x = __loader__\n", "name __loader__"),
    ("dunder_assign", "__test__ = False\n", "name __test__"),
    ("dunder_walrus", "y = (__x__ := 1)\n", "name __x__"),
    ("builtin_rebind", "len = 3\n", "rebinds builtin len"),
    ("builtin_rebind_except", "try:\n    pass\nexcept ValueError as len:\n    pass\n", "rebinds builtin len"),
    ("request_parameter", "def test_x(request):\n    pass\n", "parameter request"),
    ("pytestconfig_parameter", "def test_x(pytestconfig):\n    pass\n", "parameter pytestconfig"),
    ("lambda_parameter", "f = lambda request: 1\n", "parameter request"),
    ("keyword_unlisted", "import json\njson.dumps(1, cls=None)\n", "keyword cls"),
    ("keyword_splat", "import json\nd = {}\njson.dumps(1, **d)\n", "keyword splat"),
    ("keyword_splat_rebuilt", "import json\ndef f(**k):\n    k['cls'] = None\n    return json.dumps(1, **k)\n",
     "keyword splat"),
    ("metaclass", "class X(metaclass=type):\n    pass\n", "keyword metaclass"),
    ("setattr_two_arguments", "def test_x(monkeypatch):\n    monkeypatch.setattr('json.dumps', len)\n",
     "setattr needs (object, literal name, value)"),
    ("setattr_dynamic_name", "import json\ndef test_x(monkeypatch, name):\n    monkeypatch.setattr(json, name, 1)\n",
     "setattr needs (object, literal name, value)"),
    ("setattr_unlisted_name", "import json\ndef test_x(monkeypatch):\n    monkeypatch.setattr(json, 'JSONDecoder', 1)\n",
     "setattr name JSONDecoder"),
    ("match_statement", "match 1:\n    case _:\n        pass\n", "match statement"),
    ("syntax_error", "def (:\n", "does not parse"),
    ("utf7_hidden_import", b"# -*- coding: utf-7 -*-\nx = 1 #+AAo-import os\n", "import os"),
]


@pytest.mark.parametrize("source, expect", [g[1:] for g in GATE_REFUSED], ids=[g[0] for g in GATE_REFUSED])
def test_the_gate_refuses_code_outside_the_runners_scope(tmp_path, source, expect):
    problems = _gate(tmp_path, source)
    assert any(expect in p for p in problems), problems


def test_the_gate_scans_each_package_initializer_above_the_test(tmp_path):
    problems = _gate(tmp_path, "x = 1\n", name="pkg/test_scratch.py", extra={"pkg/__init__.py": "import importlib\n"})
    assert any(p.startswith("pkg/__init__.py:") and "import importlib" in p for p in problems), problems


def test_the_gate_refuses_a_named_test_that_is_not_a_file_in_the_root(tmp_path):
    root = tmp_path / "gate"
    (root / "sub").mkdir(parents=True)
    (tmp_path / "test_out.py").write_text("x = 1\n")
    problems = R.scope_problems(root, ["sub", "../test_out.py", "test_missing.py"])
    assert [p.split(":")[0] for p in problems] == ["sub", "../test_out.py", "test_missing.py"], problems


def test_the_gate_refuses_packages_that_reach_above_the_root(tmp_path):
    root = tmp_path / "gate"
    root.mkdir()
    for d in (tmp_path, root):
        (d / "__init__.py").write_text("")
    (root / "test_scratch.py").write_text("x = 1\n")
    assert R.scope_problems(root, ["test_scratch.py"]) == ["test_scratch.py: its packages reach above the root"]


def test_the_gate_scans_the_mutated_bytes_of_a_suite_file(tmp_path):
    root = tmp_path / "gate"
    root.mkdir()
    (root / "test_scratch.py").write_text("x = 1\n")
    assert R.scope_problems(root, ["test_scratch.py"]) == []
    over = {(root / "test_scratch.py").resolve(): b"import importlib\n"}
    assert any("import importlib" in p for p in R.scope_problems(root, ["test_scratch.py"], over))


def test_the_gate_admits_the_real_suite_and_a_plain_scratch_suite(tmp_path):
    files, _ = R.suite_files(ROOT, ["tests/scripts/c2_framing/test_screen.py::test_variant_bases"])
    assert [f.relative_to(ROOT).as_posix() for f in files] == [
        "tests/scripts/c2_framing/test_screen.py", "tests/scripts/c2_framing/__init__.py",
        "tests/scripts/__init__.py", "tests/__init__.py"]
    assert R.scope_problems(ROOT, ["tests/scripts/c2_framing/test_screen.py"]) == []
    assert _gate(tmp_path, _two_read_failing(tmp_path / "c")(tmp_path)) == []


DANGEROUS = {
    "builtins": {"getattr", "setattr", "delattr", "vars", "globals", "locals", "dir", "eval", "exec", "compile",
                 "__import__", "open", "breakpoint", "exit", "quit", "input", "help", "type", "object", "super",
                 "memoryview", "__build_class__", "print", "id", "hash", "callable", "classmethod", "staticmethod"},
    "modules": {"importlib", "importlib.util", "inspect", "sys", "os", "gc", "ctypes", "pickle", "marshal", "shelve",
                "pluggy", "_pytest", "runpy", "pkgutil", "builtins", "types", "traceback", "signal", "threading",
                "_thread", "atexit", "faulthandler", "operator", "string", "logging", "unittest", "code", "pdb"},
    "attributes": {"exit", "register", "unregister", "pluginmanager", "config", "hook", "ihook", "session", "node",
                   "request", "f_locals", "f_globals", "f_back", "f_builtins", "f_code", "tb_frame", "tb_next",
                   "gi_frame", "cr_frame", "ag_frame", "frame", "traceback", "tb", "modules", "sys", "os",
                   "importlib", "import_module", "util", "loader", "exec_module", "spec_from_file_location",
                   "module_from_spec", "load", "read_pickle", "to_pickle", "eval", "query", "attrgetter",
                   "methodcaller", "settrace", "setprofile", "monitoring", "_getframe", "kill", "killpg", "system",
                   "popen", "fork", "abort", "_exit", "interrupt_main", "raise_signal", "signal", "add_hookspecs",
                   "add_hookcall_monitoring", "subset_hook_caller", "get_plugins", "syspath_prepend", "delattr",
                   "context", "getfixturevalue", "applymarker", "add_marker", "addfinalizer", "xfail", "skip",
                   "skipif", "usefixtures", "importorskip", "pytester", "builtins", "ctypes", "ctypeslib", "gc",
                   "get_objects", "get_referrers", "inspect", "currentframe", "stack", "with_traceback", "pickle",
                   "agg", "aggregate_string", "apply", "transform", "pipe", "environ", "getattr", "open", "pytest",
                   "_pytest", "pluggy", "hookimpl", "hookspec", "Config", "Session", "PytestPluginManager"},
    "keywords": {"allow_pickle", "metaclass", "preexec_fn", "shell", "engine", "autouse", "indirect", "params"},
    "parameters": {"request", "pytestconfig", "pytester", "testdir", "tmpdir", "tmpdir_factory", "tmp_path_factory",
                   "cache", "caplog", "recwarn", "record_property", "record_xml_attribute", "doctest_namespace",
                   "record_testsuite_property", "subtests", "capsysbinary", "capfd", "capfdbinary", "capteesys"},
}


def test_the_reviewed_vocabulary_excludes_known_escape_routes():
    """The gate's allow-lists (the real suite's own vocabulary) share nothing with the names that reach pytest's plugin
    system, frames, dynamic import, deserialization or a process abort."""
    assert not R.ALLOWED_BUILTINS & DANGEROUS["builtins"]
    assert not R.ALLOWED_MODULES & DANGEROUS["modules"]
    assert not R.ALLOWED_ATTRIBUTES & DANGEROUS["attributes"]
    assert not R.ALLOWED_KEYWORDS & DANGEROUS["keywords"]
    assert not R.ALLOWED_PARAMETERS & DANGEROUS["parameters"]
    assert R.PYTEST_ATTRIBUTES == {"fixture", "mark", "raises"}
    assert "len" in R._BUILTIN_NAMES and "getattr" in R._BUILTIN_NAMES
    assert not [p for p in R.ALLOWED_PARAMETERS if p in R._BUILTIN_NAMES or p.startswith(("pytest", "__"))]
    assert not [a for a in R.ALLOWED_ATTRIBUTES if a.startswith("__")]


def test_the_runner_refuses_an_out_of_scope_suite_before_running_it(tmp_path, capsys):
    """Refused before collection: the module-level canary never runs, and the target is never mutated."""
    canary = tmp_path / "collected"
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: (
        f"import pathlib\nimport importlib\npathlib.Path({str(canary)!r}).write_text('y')\n"
        "def test_first():\n    assert 1\n"))
    assert R.main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 REFUSED" in out and "import importlib" in out and "NOT RED: X1" in out and not canary.exists()
    assert (root / "target.py").read_text() == "value = 1\n"


# ---------------------------------------------------------------- r9: nothing importable may change during the ledger

def test_the_change_scan_reports_changed_new_and_removed_entries(tmp_path):
    tree = tmp_path / "tree"
    (tree / "pkg").mkdir(parents=True)
    for n in ("kept.py", "edited.py", "gone.py", "exempt.py"):
        (tree / "pkg" / n).write_text("x = 1\n")
    marker = tmp_path / "marker"
    marker.write_text("t0")
    t0 = marker.stat().st_ctime_ns
    assert R.changed_since(t0, [tree]) == []
    (tree / "pkg" / "edited.py").write_text("x = 1\n")                     # same bytes, rewritten
    (tree / "pkg" / "exempt.py").write_text("x = 2\n")
    (tree / "pkg" / "gone.py").unlink()                                    # the directory entry changes
    changed = R.changed_since(t0, [tree], exempt=[tree / "pkg" / "exempt.py"])
    assert sorted(Path(c).name for c in changed) == ["edited.py", "pkg"], changed


def test_an_unlistable_directory_is_reported(tmp_path):
    tree = tmp_path / "tree"
    (tree / "locked").mkdir(parents=True)
    marker = tmp_path / "marker"
    marker.write_text("t0")
    t0 = marker.stat().st_ctime_ns
    (tree / "locked").chmod(0)                       # also a ctime change, so check the listing failure itself
    try:
        changed = R.changed_since(t0 + 10 ** 15, [tree])         # a t0 in the future: only the failure can report
    finally:
        (tree / "locked").chmod(0o755)
    assert [Path(c).name for c in changed] == ["locked"], changed


def test_the_scan_roots_cover_the_root_and_the_interpreters_trees(tmp_path):
    trees = R.interpreter_trees()
    assert any((Path(t) / "json" / "__init__.py").is_file() for t in trees)          # the standard library
    assert any((Path(t) / "pandas" / "__init__.py").is_file() for t in trees)        # site-packages
    roots = R.scan_roots(tmp_path)
    assert str(tmp_path.resolve()) in roots
    for t in trees:
        assert any(t == r or t.startswith(r + "/") for r in roots), t
    assert len(roots) == len(set(roots))
    assert not [r for r in roots for s in roots if r != s and r.startswith(s + "/")]   # no nested duplicates


def test_a_file_written_into_the_root_during_the_run_refuses_it(tmp_path, capsys):
    """r9: code written during the run and imported later would bypass the gate, so any change is refused."""
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: _READ + (
        f"def test_first():\n    pathlib.Path(__file__).with_name('planted.py').write_text('x = 1')\n"
        f"    assert val() == {_ONE}\n"))
    assert R.main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 INCONCLUSIVE(exit 1, files changed during the run)" in out and "NOT RED: X1" in out, out


def test_mutants_across_two_targets_are_all_red(tmp_path, capsys):
    """The runner's own restore of a target is not a change: the next mutant, of the other target or the same one, runs."""
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: _READ + (
        "def test_first():\n"
        f"    assert val() == {_ONE} and pathlib.Path(__file__).with_name('other.py').read_text() == {_ONE}\n"))
    (root / "other.py").write_text("value = 1\n")
    _commit_all(root)
    x1 = json.loads(spec.read_text())[0]
    spec.write_text(json.dumps([x1, dict(x1, id="X2", file="other.py"), dict(x1, id="X3", new="value = 3")]))
    assert R.main(str(spec), root=root) == 0
    out = capsys.readouterr().out
    assert "X1 RED" in out and "X2 RED" in out and "X3 RED" in out and "NOT RED: none" in out, out


def test_a_target_rewritten_during_the_run_refuses_it(tmp_path, capsys):
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: _READ + (
        "def test_first():\n    pathlib.Path(__file__).with_name('target.py').write_text('value = 2\\n')\n"
        f"    assert val() == {_ONE}\n"))
    assert R.main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 INCONCLUSIVE(exit 1, target changed during the run)" in out and "NOT RED: X1" in out, out
    assert (root / "target.py").read_text() == "value = 1\n"


# ---------------------------------------------------------------- the dynamic layers (defence in depth behind the gate)

def _dynamic(tmp_path, body, gate, expect, plugin=None):
    """An out-of-scope suite: the gate refuses it; run anyway through run_mutant (no gate), the evidence is refused."""
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], body)
    if plugin:
        (root / "scratch_plugin.py").write_text(_VAL + plugin)
        _commit_all(root)
    problems = R.scope_problems(root, ["test_scratch.py"])
    assert any(gate in p for p in problems), problems
    intended = R.collect(root, ["test_scratch.py"])
    (root / "target.py").write_text("value = 2\n")                         # the mutant, applied by hand
    rc, _, bundle = R.run_mutant(root, ["test_scratch.py"], {(root / "target.py").resolve(): b"value = 2\n"})
    assert bundle is None or bundle["boundary"] == [], bundle              # r10: these layers act inside the boundary
    assert R.classify(rc, intended, bundle)[0] == f"INCONCLUSIVE(exit {rc}, {expect})", bundle
    return bundle


_PLUGINS = "pytest_plugins = ['scratch_plugin']\n"


def test_a_clean_run_has_no_foreign_plugins_and_no_plugin_changes(tmp_path):
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], _two_read_failing())
    rc, _, bundle = R.run_mutant(root, ["test_scratch.py"])
    assert rc == 0 and bundle["foreign_plugins"] == [] and bundle["plugin_changes"] == [], bundle   # no anyio either
    assert bundle["boundary"] == [], bundle
    assert R.classify(rc, R.collect(root, ["test_scratch.py"]), bundle)[0] == "SURVIVED"


def test_the_runner_is_not_fooled_by_printed_outcomes(tmp_path):
    """r4 R4-2: the first test prints the second's PASSED line; the second's fixture stops pytest before its body."""
    _dynamic(tmp_path, lambda r: _VAL + (
        "def test_first(request):\n"
        "    if val() != 1:\n"
        "        print('PASSED ' + request.node.nodeid.rsplit('::', 1)[0] + '::test_second')\n"
        "    assert val() == 1\n"
        "@pytest.fixture\n"
        "def gate():\n"
        "    if val() != 1:\n"
        "        pytest.exit('stop before the body', returncode=1)\n"
        "def test_second(gate):\n    assert val() == 1\n"), "parameter request", "interrupted")


def test_the_runner_is_not_fooled_by_an_aborted_teardown(tmp_path):
    """r5 R5-1: both bodies fail for real, but the second's teardown calls pytest.exit with a count-bearing message."""
    _dynamic(tmp_path, lambda r: _VAL + (
        "def test_first():\n    assert val() == 1\n"
        "@pytest.fixture\n"
        "def gate():\n"
        "    yield\n"
        "    if val() != 1:\n"
        "        pytest.exit('2 failed cleanup checks', returncode=1)\n"
        "def test_second(gate):\n    assert val() == 1\n"), "pytest.exit", "interrupted")


def _two_val_failing(r):
    return _VAL + _PLUGINS + "def test_first():\n    assert val() == 1\ndef test_second():\n    assert val() == 1\n"


def test_a_later_session_finish_abort_is_not_a_normal_end(tmp_path):
    _dynamic(tmp_path, _two_val_failing, "name pytest_plugins", "session finish incomplete", plugin=(
        "def pytest_sessionfinish(session, exitstatus):\n"
        "    if val() != 1:\n"
        "        pytest.exit('2 failed in a late hook', returncode=1)\n"))


def test_an_exit_after_the_last_report_is_still_an_interrupt(tmp_path):
    _dynamic(tmp_path, _two_val_failing, "name pytest_plugins", "interrupted", plugin=(
        "def pytest_runtest_logfinish(nodeid, location):\n"
        "    if nodeid.endswith('test_second') and val() != 1:\n"
        "        pytest.exit('2 failed after the last test', returncode=1)\n"))


def test_a_session_finish_wrapper_abort_is_not_a_normal_end(tmp_path):
    _dynamic(tmp_path, _two_val_failing, "name pytest_plugins", "session finish incomplete", plugin=(
        "@pytest.hookimpl(wrapper=True)\n"
        "def pytest_sessionfinish(session, exitstatus):\n"
        "    result = yield\n"
        "    if val() != 1:\n"
        "        pytest.exit('2 failed in a wrapper', returncode=1)\n"
        "    return result\n"))


def test_a_tryfirst_wrapper_abort_is_still_seen(tmp_path):
    _dynamic(tmp_path, _two_val_failing, "name pytest_plugins", "session finish incomplete", plugin=(
        "@pytest.hookimpl(wrapper=True, tryfirst=True)\n"
        "def pytest_sessionfinish(session, exitstatus):\n"
        "    result = yield\n"
        "    if val() != 1:\n"
        "        pytest.exit('2 failed in a tryfirst wrapper', returncode=1)\n"
        "    return result\n"))


def test_a_rewritten_report_is_refused(tmp_path):
    body = lambda r: _VAL + _PLUGINS + "def test_first():\n    assert val() == 1\ndef test_second():\n    pass\n"
    _dynamic(tmp_path, body, "name pytest_plugins", "report mismatch", plugin=(
        "@pytest.hookimpl(wrapper=True)\n"
        "def pytest_runtest_makereport(item, call):\n"
        "    rep = yield\n"
        "    if rep.when == 'call' and item.name == 'test_second' and val() != 1:\n"
        "        rep.outcome = 'failed'\n"
        "    return rep\n"))


def test_an_unconfigure_abort_leaves_no_evidence(tmp_path):
    _dynamic(tmp_path, _two_val_failing, "name pytest_plugins", "no normal end", plugin=(
        "def pytest_unconfigure(config):\n"
        "    if val() != 1:\n"
        "        pytest.exit('2 failed at unconfigure', returncode=1)\n"))


def test_a_harmless_early_plugin_is_still_foreign(tmp_path):
    """r9: only pytest's own plugins and the runner's may be registered when the sentinel is."""
    bundle = _dynamic(tmp_path, _two_val_failing, "name pytest_plugins", "foreign plugins",
                      plugin="def pytest_report_header(config):\n    return 'scratch'\n")
    assert bundle["foreign_plugins"] == ["scratch_plugin"]


def test_an_early_plugin_unregistered_during_the_run_is_seen(tmp_path):
    """r9: removing a plugin registered before the sentinel adds nothing and leaves the sentinel outermost; only the
    watch on pluggy's remove code sees it."""
    body = lambda r: _VAL + _PLUGINS + ("def test_first(request):\n"
                                        "    request.config.pluginmanager.unregister(name='scratch_plugin')\n"
                                        "    assert val() == 1\n"
                                        "def test_second():\n    assert val() == 1\n")
    bundle = _dynamic(tmp_path, body, "parameter request", "plugin system changed",
                      plugin="def pytest_report_header(config):\n    return 'scratch'\n")
    assert bundle["late_plugins"] == 0 and bundle["not_outermost"] == []
    assert bundle["plugin_changes"] and all(c == "HookCaller._remove_plugin" for c in bundle["plugin_changes"])


def test_a_non_strict_xpass_is_not_a_kill(tmp_path):
    """r6 R6-2: under the mutant the first test fails and an xfail-marked second test unexpectedly passes."""
    _dynamic(tmp_path, lambda r: _VAL + ("def test_first():\n    assert val() == 1\n"
                                         "@pytest.mark.xfail(reason='expected')\n"
                                         "def test_second():\n    assert val() == 2\n"),
             "attribute xfail", "marked xfail or skip")


def test_a_runtime_xfail_is_not_a_kill(tmp_path):
    _dynamic(tmp_path, lambda r: _VAL + ("def test_first():\n    assert val() == 1\n"
                                         "def test_second(request):\n"
                                         "    request.applymarker(pytest.mark.xfail(reason='late'))\n"
                                         "    assert val() == 2\n"), "parameter request", "xfail or skip")


def test_a_plugin_registered_during_the_run_is_refused(tmp_path):
    _dynamic(tmp_path, lambda r: _VAL + ("def test_first(request):\n"
                                         "    request.config.pluginmanager.register(object(), 'late-object')\n"
                                         "    assert val() == 1\n"
                                         "def test_second():\n    assert val() == 1\n"),
             "parameter request", "plugins registered late")


_LATE = ("class Late:\n"
         "    def __init__(self, pm, base, before):\n"
         "        self.pm, self.base, self.before = pm, base, before\n"
         "    def _drop(self):\n"
         "        (pluggy.PluginManager.unregister if self.base else type(self.pm).unregister)(self.pm, self)\n"
         "    @pytest.hookimpl(wrapper=True, tryfirst=True)\n"
         "    def pytest_sessionfinish(self, session, exitstatus):\n"
         "        if self.before:\n"
         "            self._drop()\n"
         "        result = yield\n"
         "        if not self.before:\n"
         "            self._drop()\n"
         "        if val() != 1:\n"
         "            pytest.exit('2 failed in a late wrapper', returncode=1)\n"
         "        return result\n")


def _late(register):
    return lambda r: _VAL + "import pluggy\n" + _LATE + (
        "def test_first(request):\n"
        "    pm = request.config.pluginmanager\n"
        f"    {register}\n"
        "    assert val() == 1\n"
        "def test_second():\n    assert val() == 1\n")


def test_a_self_unregistering_late_wrapper_is_refused(tmp_path):
    """r7 R7-1: registered through pytest's API; it unregisters itself after the sentinel's record, then aborts."""
    _dynamic(tmp_path, _late("pm.register(Late(pm, False, False), 'late-wrapper')"), "parameter request",
             "plugins registered late")


def test_the_late_registration_evidence_is_produced_and_retained(tmp_path):
    """r7 R7-1, the producer itself: a plugin registered during the run and unregistered at once still counts."""
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: _VAL + (
        "def test_first(request):\n"
        "    pm = request.config.pluginmanager\n"
        "    p = object()\n"
        "    pm.register(p, 'brief')\n"
        "    pm.unregister(p)\n"
        "    assert val() == 1\n"))
    rc, out, bundle = R.run_mutant(root, ["test_scratch.py"])
    assert bundle is not None and bundle["late_plugins"] >= 1 and bundle["not_outermost"] == []
    assert R.classify(rc, ["test_scratch.py::test_first"], bundle)[0] == "INCONCLUSIVE(exit 0, plugins registered late)"


def test_an_outer_wrapper_without_a_registration_event_is_still_seen(tmp_path):
    """r7: registered through pluggy's base class (no registration event), it unregisters itself after yielding."""
    _dynamic(tmp_path, _late("pluggy.PluginManager.register(pm, Late(pm, True, False), 'hidden-wrapper')"),
             "parameter request", "sentinel not outermost")


def test_a_wrapper_gone_from_the_registry_before_yielding_is_still_seen(tmp_path):
    """r8 R8-1: the base-class wrapper unregisters itself BEFORE yielding, so the live registry the sentinel reads no
    longer shows it while it still wraps the sentinel, then aborts. The registration itself ran pluggy's add code."""
    bundle = _dynamic(tmp_path, _late("pluggy.PluginManager.register(pm, Late(pm, True, True), 'gone-wrapper')"),
                      "parameter request", "plugin system changed")
    assert bundle["late_plugins"] == 0 and bundle["not_outermost"] == []
    assert bundle["plugin_changes"] == ["HookCaller._add_hookimpl", "HookCaller._remove_plugin"], bundle  # both watched


def test_a_wrapper_gone_before_yielding_without_an_abort_is_still_seen(tmp_path):
    """r8 R8-1 control: the same lifecycle with no abort; the sessions ends normally but the plugin system changed."""
    body = lambda r: _VAL + "import pluggy\n" + _LATE.replace("        if val() != 1:\n"
                                                              "            pytest.exit('2 failed in a late wrapper', "
                                                              "returncode=1)\n", "") + (
        "def test_first(request):\n"
        "    pm = request.config.pluginmanager\n"
        "    pluggy.PluginManager.register(pm, Late(pm, True, True), 'gone-wrapper')\n"
        "    assert val() == 1\n"
        "def test_second():\n    assert val() == 1\n")
    _dynamic(tmp_path, body, "parameter request", "plugin system changed")


def test_a_hidden_call_wrapper_is_seen(tmp_path):
    """A call wrapper registered through pluggy's base class wraps the second test's call and unregisters itself."""
    _dynamic(tmp_path, lambda r: _VAL + "import pluggy\n" + (
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
        "def test_second():\n    assert val() == 1\n"), "parameter request", "sentinel not outermost")


def test_a_call_wrapper_gone_before_yielding_is_seen(tmp_path):
    """r8 R8-1, per call: the base-class call wrapper unregisters itself before yielding."""
    _dynamic(tmp_path, lambda r: _VAL + "import pluggy\n" + (
        "class Gone:\n"
        "    def __init__(self, pm):\n"
        "        self.pm = pm\n"
        "    @pytest.hookimpl(wrapper=True, tryfirst=True)\n"
        "    def pytest_runtest_call(self, item):\n"
        "        pluggy.PluginManager.unregister(self.pm, self)\n"
        "        return (yield)\n"
        "def test_first(request):\n"
        "    pm = request.config.pluginmanager\n"
        "    pluggy.PluginManager.register(pm, Gone(pm), 'gone-call-wrapper')\n"
        "    assert val() == 1\n"
        "def test_second():\n    assert val() == 1\n"), "parameter request", "plugin system changed")


# ---------------------------------------------------------------- r10: only installed and committed code runs

def test_collection_and_runs_stay_inside_the_root(tmp_path, capsys):
    """r9 R9-1: with an empty configuration file pytest walked the root's ancestors and ran a package initializer above
    the root. The collection cutoff is now the root itself."""
    canary = tmp_path / "ancestor_ran"
    (tmp_path / "__init__.py").write_text(f"import pathlib\npathlib.Path({str(canary)!r}).write_text('y')\n")
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], _two_read_failing())
    assert R.main(str(spec), root=root) == 0
    assert "X1 RED" in capsys.readouterr().out and not canary.exists()
    assert R.collect(root, ["test_scratch.py"]) == ["test_scratch.py::test_first", "test_scratch.py::test_second"]


def _numpy_suite(r):
    """An in-scope suite (the reviewed vocabulary) with an allowed import spelled numpy."""
    return _READ + f"import numpy as np\ndef test_first():\n    assert np.isclose(1, 1) and val() == {_ONE}\n"


def _look_alike(canary):
    return f"import pathlib\npathlib.Path({str(canary)!r}).write_text('ran')\ndef isclose(a, b):\n    return a == b\n"


LOOK_ALIKE = [("uncommitted", False, ["numpy.py: not committed", "numpy.py: shadows the installed module numpy"]),
              ("committed", True, ["numpy.py: shadows the installed module numpy"])]


@pytest.mark.parametrize("v, expect", [c[1:] for c in LOOK_ALIKE], ids=[c[0] for c in LOOK_ALIKE])
def test_a_look_alike_module_is_refused_before_the_run(tmp_path, capsys, v, expect):
    """r9 R9-2: an allowed import (numpy) resolved to a local module the gate never read. A run may execute only
    installed and committed code, and no module it can import from the root may take an installed module's name."""
    canary = tmp_path / "look_alike_ran"
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], _numpy_suite)
    (root / "numpy.py").write_text(_look_alike(canary))
    if v:
        _commit_all(root)
    assert R.main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 REFUSED" in out and "NOT RED: X1" in out and not canary.exists(), out
    assert R.boundary_problems(root, ["test_scratch.py"]) == expect
    assert (root / "target.py").read_text() == "value = 1\n"


def test_a_linked_module_is_refused_before_the_run(tmp_path, capsys):
    """r9 R9-3: a link named source outside every scanned tree, which a child could rewrite before the import. No
    symlink is allowed anywhere a run can import from."""
    canary = tmp_path / "linked_ran"
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "numpy.py").write_text(_look_alike(canary))
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], _numpy_suite)
    subprocess.run(["ln", "-s", str(tmp_path / "outside" / "numpy.py"), str(root / "numpy.py")], check=True)
    _commit_all(root)                                                 # git records the link, never what it names
    assert R.main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 REFUSED" in out and not canary.exists(), out
    assert R.boundary_problems(root, ["test_scratch.py"]) == ["numpy.py: a symlink",
                                                              "numpy.py: shadows the installed module numpy"]


def _committed_root(tmp_path, files):
    """A committed scratch root: a plain test module, then `files` (relative path -> text)."""
    root = tmp_path / "root"
    root.mkdir()
    (root / "test_scratch.py").write_text("def test_first():\n    assert 0\n")
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    _commit_all(root)
    return root


def _boundary(repo):
    return R.boundary_problems(repo, ["test_scratch.py"])


def test_the_boundary_admits_a_committed_root(tmp_path):
    root = _committed_root(tmp_path, {"helper.py": "x = 1\n", "pkg/__init__.py": "", "pkg/m.py": "y = 2\n",
                                      "pkg/json.py": "z = 3\n"})            # an installed name only below the top
    assert _boundary(root) == []


def test_the_boundary_admits_the_real_suite():
    assert R.boundary_problems(ROOT, ["tests/scripts/c2_framing/test_screen.py"]) == []


def test_the_boundary_refuses_uncommitted_modules(tmp_path):
    root = _committed_root(tmp_path, {"pkg/__init__.py": ""})
    for name in ("helper.py", "pkg/m.py", "ns/deep/m.py"):                 # a namespace package is importable too
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x = 1\n")
    assert _boundary(root) == ["helper.py: not committed", "ns/deep/m.py: not committed", "pkg/m.py: not committed"]


def test_the_boundary_refuses_a_module_that_differs_from_the_commit(tmp_path):
    root = _committed_root(tmp_path, {"helper.py": "x = 1\n"})
    (root / "helper.py").write_text("x = 2\n")
    assert _boundary(root) == ["helper.py: differs from the commit"]


def test_the_boundary_refuses_symlinks(tmp_path):
    root = _committed_root(tmp_path, {"helper.py": "x = 1\n"})
    (tmp_path / "outside").mkdir()
    for target, link in (("helper.py", "alias.py"), (str(tmp_path / "outside"), "pkg")):
        subprocess.run(["ln", "-s", target, str(root / link)], check=True)
    _commit_all(root)
    assert _boundary(root) == ["alias.py: a symlink", "pkg: a symlink"]


def test_the_boundary_refuses_compiled_modules(tmp_path):
    root = _committed_root(tmp_path, {})
    for name in ("cached.pyc", "native.so"):
        (root / name).write_bytes(b"\0")
    _commit_all(root)
    assert _boundary(root) == ["cached.pyc: a compiled module", "native.so: a compiled module"]


def test_the_boundary_ignores_what_cannot_be_imported(tmp_path):
    """Precision: data files, cached bytecode (never read: the runs' cache prefix is under /dev/null), and names that
    are not identifiers."""
    root = _committed_root(tmp_path, {})
    for name in ("notes.txt", "__pycache__/helper.cpython-312.pyc", "2026-run/evil.py", "my-mod.py", ".hidden/evil.py"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x = 1\n")
    assert _boundary(root) == []


def test_a_directory_holding_code_under_an_installed_name_is_a_look_alike(tmp_path):
    """A top-level directory provides code under its name when it holds anything importable (an installed namespace
    package would take it in); a bytecode cache holds nothing importable (test above)."""
    root = _committed_root(tmp_path, {"json/helper.py": "x = 1\n"})
    assert _boundary(root) == ["json: shadows the installed module json"]


def test_the_boundary_walks_the_directories_a_run_imports_from(tmp_path):
    """Inside the root, a run imports from the named test's package root (pytest puts it first on the path) and from
    any startup entry inside the root. A module no entry reaches cannot be imported, so it is not refused."""
    root = _committed_root(tmp_path, {"sub/test_inner.py": "def test_first():\n    assert 0\n"})
    for name in ("sub/numpy.py", "elsewhere/helper.py"):
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text("x = 1\n")
    assert R.boundary_problems(root, ["sub/test_inner.py"]) == ["sub/numpy.py: not committed",
                                                               "sub/numpy.py: shadows the installed module numpy"]


def test_the_boundary_refuses_a_root_that_is_not_a_committed_work_tree(tmp_path):
    root = tmp_path / "plain"
    root.mkdir()
    (root / "test_scratch.py").write_text("def test_first():\n    assert 0\n")
    assert _boundary(root) == [f"{root.resolve()}: not the top of a git work tree with a commit"]
    inner = _committed_root(tmp_path, {"sub/test_scratch.py": "def test_first():\n    assert 0\n"}) / "sub"
    assert _boundary(inner) == [f"{inner.resolve()}: not the top of a git work tree with a commit"]


def test_the_boundary_refuses_an_installed_tree_containing_the_root(tmp_path, monkeypatch):
    """A startup entry outside the root is an installed tree; one containing the root would make the root's own files
    look installed."""
    root = _committed_root(tmp_path, {})
    (root / "helper.py").write_text("x = 1\n")                             # still the root's own: not committed
    path, trees = R._startup()
    monkeypatch.setattr(R, "_startup", lambda: ([*path, str(tmp_path)], trees))
    assert _boundary(root) == [f"{tmp_path.resolve()}: an installed tree containing the root",
                               "helper.py: not committed"]


# The run's own audit hook, exercised past the gate and the boundary check (run_mutant alone, on the unmutated source).
# Each scratch test module catches the refusal and then fails, so without the hook's record the run would read RED.

def _in_run(repo, why):
    rc, _, bundle = R.run_mutant(repo, ["test_scratch.py"])
    assert bundle is not None and any(b.startswith(why + ": ") for b in bundle["boundary"]), bundle
    verdict = R.classify(rc, ["test_scratch.py::test_first"], bundle)[0]
    assert verdict == f"INCONCLUSIVE(exit {rc}, code outside the boundary)", bundle
    return bundle


def _catching(name):
    return f"try:\n    import {name}\nexcept ImportError:\n    pass\ndef test_first():\n    assert 0\n"


def _canary_module(canary):
    return f"import pathlib\npathlib.Path({str(canary)!r}).write_text('ran')\n"


def test_the_run_refuses_an_uncommitted_module(tmp_path):
    canary = tmp_path / "ran"
    root = _committed_root(tmp_path, {"test_scratch.py": _catching("helper")})
    (root / "helper.py").write_text(_canary_module(canary))
    _in_run(root, "not committed")
    assert not canary.exists()


def test_the_run_refuses_a_module_that_differs_from_the_commit(tmp_path):
    canary = tmp_path / "ran"
    root = _committed_root(tmp_path, {"test_scratch.py": _catching("helper"), "helper.py": "x = 1\n"})
    (root / "helper.py").write_text(_canary_module(canary))
    _in_run(root, "differs from the commit")
    assert not canary.exists()


def test_the_run_refuses_a_module_reached_through_a_symlink(tmp_path):
    canary = tmp_path / "ran"
    root = _committed_root(tmp_path, {"test_scratch.py": _catching("helper"), "real.py": _canary_module(canary)})
    subprocess.run(["ln", "-s", "real.py", str(root / "helper.py")], check=True)        # names committed bytes
    _in_run(root, "a symlink")
    assert not canary.exists()


def test_the_run_refuses_a_module_outside_the_root(tmp_path):
    """r9 R9-3, in the run: the link's physical source is outside the root and every installed tree."""
    canary = tmp_path / "ran"
    (tmp_path / "outside").mkdir()
    (tmp_path / "outside" / "helper.py").write_text(_canary_module(canary))
    root = _committed_root(tmp_path, {"test_scratch.py": _catching("helper")})
    subprocess.run(["ln", "-s", str(tmp_path / "outside" / "helper.py"), str(root / "helper.py")], check=True)
    _in_run(root, "outside the root and the installed trees")
    assert not canary.exists()


def test_the_run_refuses_a_committed_look_alike(tmp_path):
    """r9 R9-2, in the run: committed, but it takes an installed module's name."""
    canary = tmp_path / "ran"
    _in_run(_committed_root(tmp_path, {"test_scratch.py": _catching("numpy"), "numpy.py": _canary_module(canary)}),
            "shadows an installed module")
    assert not canary.exists()


def test_the_run_never_deserializes_bytecode(tmp_path):
    """Sourceless bytecode names any file it likes as its source: here the committed test module, whose bytes the run
    compiled. Only the refusal to deserialize bytecode stops it."""
    canary = tmp_path / "ran"
    root = _committed_root(tmp_path, {"test_scratch.py": _catching("cached")})
    (tmp_path / "cached_source.py").write_text(_canary_module(canary))
    named = str((root / "test_scratch.py").resolve())
    source = str(tmp_path / "cached_source.py")
    subprocess.run([*R._python(), "-c", f"import py_compile; py_compile.compile({source!r}, "
                    f"cfile={str(root / 'cached.pyc')!r}, dfile={named!r}, doraise=True)"], check=True)
    _in_run(root, "bytecode deserialized")
    assert not canary.exists()


def test_the_run_refuses_a_compiled_module_outside_the_installed_trees(tmp_path):
    found = subprocess.run([*R._python(), "-c", "import numpy.linalg._umath_linalg as m; print(m.__file__)"],
                           capture_output=True, text=True, check=True).stdout.strip()
    root = _committed_root(tmp_path, {"test_scratch.py": _catching("_umath_linalg")})
    (root / Path(found).name).write_bytes(Path(found).read_bytes())
    assert _boundary(root) == [f"{Path(found).name}: a compiled module"]           # its full platform suffix
    _in_run(root, "a compiled module outside the installed trees")


def test_the_run_refuses_a_native_library_outside_the_installed_trees(tmp_path):
    root = _committed_root(tmp_path, {"test_scratch.py": (
        "import ctypes, pathlib\n"
        "try:\n    ctypes.CDLL(str(pathlib.Path(__file__).with_name('native.so')))\nexcept (ImportError, OSError):\n"
        "    pass\ndef test_first():\n    assert 0\n")})
    (root / "native.so").write_bytes(b"\0")
    _in_run(root, "a native library outside the installed trees")


def test_collection_runs_under_the_boundary(tmp_path):
    """r10: collecting executes the test modules, so collection runs under the same hook; a refusal selects nothing."""
    canary = tmp_path / "ran"
    root = _committed_root(tmp_path, {"test_scratch.py": _catching("helper")})     # so collection itself completes
    (root / "helper.py").write_text(_canary_module(canary))
    assert R.collect(root, ["test_scratch.py"]) is None and not canary.exists()


def test_the_runner_refuses_a_boundary_problem_before_collecting(tmp_path, capsys):
    """Refused before collection: the committed test module's own top-level code never runs, and nothing is mutated."""
    canary = tmp_path / "collected"
    spec, root = _scratch_mutant(tmp_path, ["test_scratch.py"], lambda r: (
        f"import pathlib\npathlib.Path({str(canary)!r}).write_text('y')\ndef test_first():\n    assert 1\n"))
    (root / "helper.py").write_text("x = 1\n")                             # uncommitted, never imported
    assert R.main(str(spec), root=root) == 1
    out = capsys.readouterr().out
    assert "X1 REFUSED" in out and "helper.py: not committed" in out and not canary.exists(), out
    assert (root / "target.py").read_text() == "value = 1\n"


def test_the_boundary_walks_a_startup_entry_inside_the_root(tmp_path, monkeypatch):
    """A startup entry inside the root (the repository's own src, for the real root) is walked from its own top, which
    the named test's package root does not reach."""
    root = _committed_root(tmp_path, {"t/test_inner.py": "def test_first():\n    assert 0\n",
                                      "lib/numpy.py": "x = 1\n"})
    path, trees = R._startup()
    monkeypatch.setattr(R, "_startup", lambda: ([*path, str(root / "lib")], trees))
    assert R.boundary_problems(root, ["t/test_inner.py"]) == ["lib/numpy.py: shadows the installed module numpy"]


def test_the_boundary_walks_from_the_package_root(tmp_path):
    """A test inside packages runs with the directory above its topmost package first on the path."""
    root = _committed_root(tmp_path, {"pkg/__init__.py": "", "pkg/test_inner.py": "def test_first():\n    assert 0\n"})
    (root / "numpy.py").write_text("x = 1\n")
    assert R.boundary_problems(root, ["pkg/test_inner.py"]) == ["numpy.py: not committed",
                                                               "numpy.py: shadows the installed module numpy"]


def test_the_boundary_never_walks_above_the_root(tmp_path):
    """Packages reaching above the root are the gate's refusal; the boundary walks only inside the root."""
    (tmp_path / "__init__.py").write_text("")
    (tmp_path / "numpy.py").write_text("x = 1\n")
    assert _boundary(_committed_root(tmp_path, {"__init__.py": ""})) == []


def test_the_boundary_skips_installed_trees_inside_the_root(tmp_path, monkeypatch):
    """An installed tree inside the root (a virtual environment's site-packages) is installed code, not the root's."""
    root = _committed_root(tmp_path, {})
    (root / "site").mkdir()
    (root / "site" / "numpy.py").write_text("x = 1\n")
    path, trees = R._startup()
    monkeypatch.setattr(R, "_startup", lambda: (path, [*trees, str(root / "site")]))
    assert _boundary(root) == []


def test_a_file_in_place_of_a_committed_symlink_is_not_committed(tmp_path):
    """git stores a symlink as a blob of its target text; a regular file holding that text is not committed code."""
    root = _committed_root(tmp_path, {})
    subprocess.run(["ln", "-s", "x = 1", str(root / "alias.py")], check=True)
    _commit_all(root)
    (root / "alias.py").unlink()
    (root / "alias.py").write_text("x = 1")
    assert _boundary(root) == ["alias.py: not committed"]


def test_the_boundary_reads_a_sha256_repository(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    (root / "test_scratch.py").write_text("def test_first():\n    assert 0\n")
    subprocess.run([*_GIT, "-C", str(root), "init", "-q", "--object-format=sha256"], check=True, capture_output=True)
    _commit_all(root)
    assert _boundary(root) == []


def test_the_boundary_ignores_inherited_git_variables(tmp_path, monkeypatch):
    root = _committed_root(tmp_path, {})
    other = tmp_path / "other"
    other.mkdir()
    _commit_all(other)
    monkeypatch.setenv("GIT_DIR", str(other / ".git"))
    assert _boundary(root) == []


def test_an_editable_install_outside_the_root_is_installed(tmp_path):
    """For a scratch root, the repository's own src (an editable install) is outside the root: installed code."""
    root = _committed_root(tmp_path, {"test_scratch.py": "import bts\ndef test_first():\n    assert 0\n"})
    rc, _, bundle = R.run_mutant(root, ["test_scratch.py"])
    assert bundle is not None and bundle["boundary"] == [] and rc == 1, bundle


def test_the_mutated_target_is_the_one_change_a_run_executes(tmp_path, capsys):
    """The runner's own mutation is the run's one allowed difference from the commit, even for an imported target."""
    root = _committed_root(tmp_path, {"scripts/audit/c2_framing/screen.py": "SETTINGS = 1\n", "test_scratch.py": (
        "import scripts.audit.c2_framing.screen as S\ndef test_first():\n    assert S.SETTINGS == 1\n")})
    spec = tmp_path / "spec.json"
    spec.write_text(json.dumps([{"id": "X1", "rule": "imported", "file": "scripts/audit/c2_framing/screen.py",
                                 "old": "SETTINGS = 1", "new": "SETTINGS = 2", "tests": ["test_scratch.py"]}]))
    assert R.main(str(spec), root=root) == 0
    assert "X1 RED" in capsys.readouterr().out


def test_the_run_ignores_an_installed_tree_containing_the_root(tmp_path, monkeypatch):
    canary = tmp_path / "ran"
    root = _committed_root(tmp_path, {"test_scratch.py": _catching("helper")})
    (root / "helper.py").write_text(_canary_module(canary))
    path, trees = R._startup()
    monkeypatch.setattr(R, "_startup", lambda: (path, [*trees, str(tmp_path)]))
    _in_run(root, "not committed")
    assert not canary.exists()
