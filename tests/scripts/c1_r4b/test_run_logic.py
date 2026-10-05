"""T8 pure logic: disposition precedence (registration §6), Monte Carlo ambiguity (§4), aggregation, provenance."""
import hashlib
import json

import numpy as np
import pytest

from scripts.audit.c1_r4b import run as RUN


def summary(obj=0.6, sw=0.5, obj_s=(0.5, 0.6, 0.7, 0.4, 0.8), sw_s=(0.5, 0.4, 0.6, 0.3, 0.7),
            obj10=0.3, sw10=0.2, se10=0.01, r20=(0.30, 0.30, 0.30), r20_10=(0.25, 0.25, 0.25), se_r20=0.005,
            coverage=True, validation=True, projections=True):
    return {"coverage_complete": coverage, "validation_ok": validation, "projections_ok": projections,
            "contrast": {"objective": {"d0": {"mean": obj, "seasons": list(obj_s)}, "d10": {"mean": obj10, "mc_se": se10}},
                         "switch": {"d0": {"mean": sw, "seasons": list(sw_s)}, "d10": {"mean": sw10, "mc_se": se10}}},
            "reach20": {"A2": {"d0": r20[0], "d10": r20_10[0]}, "A1": {"d0": r20[1], "d10": r20_10[1]},
                        "A0": {"d0": r20[2], "d10": r20_10[2]}, "mc_se_d10": {"A1": se_r20, "A0": se_r20}}}


def test_a_clean_pass_is_positive():
    d = RUN.disposition(summary())
    assert d["disposition"] == "positive", d


@pytest.mark.parametrize("kw,want", [
    (dict(coverage=False), "inconclusive"),                          # precedence: incomplete coverage first
    (dict(validation=False), "inconclusive"),
    (dict(obj10=0.015, se10=0.01), "inconclusive"),                  # Δ=0.10 gate within 2 MC SE of 0
    (dict(obj10=0.0, se10=0.0), "inconclusive"),                     # sampled zero SE at exact equality
    (dict(obj=0.25, sw=0.25), "positive"),                           # Δ=0 is deterministic: inclusive >= 0.25
    (dict(obj=0.24), "inconclusive"),                                # below the bar, contrasts positive
    (dict(obj=-0.01), "negative"),                                   # mean contrast <= 0 at Δ=0
    (dict(sw=0.0), "negative"),
    (dict(obj_s=(0.5, 0.6, 0.7, -0.3, 0.8)), "negative"),            # a season below -0.25 (guardrail breach)
    (dict(obj_s=(0.5, 0.6, 0.0, -0.1, 0.8)), "inconclusive"),        # only 3 strictly positive seasons
    (dict(r20=(0.27, 0.30, 0.30)), "negative"),                      # reach-20 3 pp below at Δ=0
    (dict(r20_10=(0.22, 0.25, 0.25), se_r20=0.002), "negative"),     # 3 pp below at Δ=0.10, unambiguous
    (dict(r20_10=(0.229, 0.25, 0.25), se_r20=0.005), "inconclusive"),  # within the MC band of the 2 pp limit
    (dict(projections=False), "inconclusive"),
])
def test_disposition_rules(kw, want):
    assert RUN.disposition(summary(**kw))["disposition"] == want


def test_mc_se_is_the_replicate_sd_over_sqrt_reps():
    per_rep = np.array([0.1, 0.3, 0.2, 0.4])
    assert RUN.mc_se(per_rep) == pytest.approx(np.std(per_rep, ddof=1) / 2)


def test_equal_season_weighting_averages_seeds_first():
    # season A: 2 seeds; season B: 4 seeds -> equal-season mean, not pooled
    per_seed = {2021: [1.0, 3.0], 2022: [10.0, 10.0, 10.0, 10.0]}
    assert RUN.equal_season_mean(per_seed) == pytest.approx((2.0 + 10.0) / 2)


def test_profile_hashes_must_match_the_retained_w0_manifest(tmp_path):
    f = tmp_path / "mdp_estpa_run" / "box1" / "simulation_seed7" / "backtest_2021.parquet"
    f.parent.mkdir(parents=True)
    f.write_bytes(b"abc")
    line = f"{hashlib.sha256(b'abc').hexdigest()}  ./box1/simulation_seed7/backtest_2021.parquet\n"
    man = tmp_path / "mdp_estpa_run.sha256"
    man.write_text(line)
    assert RUN.verify_profiles(tmp_path / "mdp_estpa_run", man, [f]) == {"box1/simulation_seed7/backtest_2021.parquet":
                                                                         hashlib.sha256(b"abc").hexdigest()}
    f.write_bytes(b"abd")
    with pytest.raises(RUN.ProvenanceError):
        RUN.verify_profiles(tmp_path / "mdp_estpa_run", man, [f])


def test_the_checked_in_admission_is_unset_so_the_run_refuses():
    with pytest.raises(SystemExit, match="reviewed_commit"):
        RUN.admission_gate()


def _git(repo, *a):
    import subprocess
    return subprocess.run(["git", "-C", str(repo), *a], capture_output=True, text=True, check=True).stdout.strip()


def _commit(repo, files: dict, msg: str) -> str:
    for rel, text in files.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_text(text)
    _git(repo, "add", "-A")
    _git(repo, "-c", "user.name=t", "-c", "user.email=t@t", "commit", "-q", "-m", msg)
    return _git(repo, "rev-parse", "HEAD")


X31 = "| X-31 | the 4b screen | ... |\n"


@pytest.fixture
def repo(tmp_path):
    """reviewed commit R; X-31 published in X; admission pointing at both in Y (the only post-review edits)."""
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q")
    R_ = _commit(r, {"pkg/a.py": "x = 1\n", "pkg/admission.json": "{}", "reg.md": "| X-30 | y |\n"}, "reviewed")
    X_ = _commit(r, {"reg.md": "| X-30 | y |\n" + X31, "pkg/admission.json": json.dumps({"reviewed_commit": R_})}, "x31")
    adm = {"reviewed_commit": R_, "x31_commit": X_}
    _commit(r, {"pkg/admission.json": json.dumps(adm)}, "admission")
    return r, adm


def check(r, adm):
    return RUN.admission_check(r, adm, closure=("pkg",), admission_rel="pkg/admission.json", register_rel="reg.md")[1]


def test_admission_passes_with_only_the_registered_post_review_edits(repo):
    r, adm = repo
    assert check(r, adm) == []


def test_admission_refuses_unset_or_short_ids(repo):
    r, adm = repo
    assert check(r, {"reviewed_commit": None, "x31_commit": adm["x31_commit"]})
    assert check(r, {**adm, "x31_commit": adm["x31_commit"][:7]})


def test_admission_refuses_an_executable_change_after_review(repo):
    """r2 N4: a clean later commit changing a solver passed ancestry before; now it refuses."""
    r, adm = repo
    _commit(r, {"pkg/a.py": "x = 2\n"}, "later solver edit")
    assert any("changed since the reviewed commit" in x for x in check(r, adm))


def test_admission_refuses_an_untracked_import_source(repo):
    r, adm = repo
    (r / "pkg" / "shadow.py").write_text("y = 1\n")
    assert any("untracked" in x for x in check(r, adm))


def test_admission_refuses_an_x31_row_edited_after_publication(repo):
    r, adm = repo
    _commit(r, {"reg.md": "| X-30 | y |\n| X-31 | the 4b screen, edited | ... |\n"}, "edit x31")
    assert any("changed after its publication" in x for x in check(r, adm))


def test_admission_refuses_an_x31_commit_that_did_not_publish_the_row(repo):
    r, adm = repo
    later = _commit(r, {"notes.md": "n\n"}, "unrelated")
    assert any("not published in the named commit" in x for x in check(r, {**adm, "x31_commit": later}))


def test_admission_refuses_a_reviewed_commit_after_x31(repo):
    r, adm = repo
    head = _git(r, "rev-parse", "HEAD")
    assert any("not an ancestor of the X-31 commit" in x for x in check(r, {**adm, "reviewed_commit": head}))


def test_foreign_imports_lists_modules_from_outside_the_checkout(monkeypatch):
    import sys
    import types
    assert RUN.foreign_imports() == []
    m = types.ModuleType("bts.shadowed")
    m.__file__ = "/elsewhere/bts/shadowed.py"
    monkeypatch.setitem(sys.modules, "bts.shadowed", m)
    assert RUN.foreign_imports() == ["bts.shadowed: /elsewhere/bts/shadowed.py"]


def test_invalidation_records_are_validated_not_just_present():
    head = _git(RUN.REPO, "rev-parse", "HEAD")
    reg = "| C1-4b-review-r3 | x | **RULED 2026-10-05: A** | Eric |\n| C1-x | y | **PROPOSED** | — |\n"
    good = {"run": "r1", "reason": "killed", "correction_commit": head, "register_row": "C1-4b-review-r3",
            "approved_by": "Eric"}
    assert RUN.validate_invalidation(good, "r1", reg) == []
    assert RUN.validate_invalidation({}, "r1", reg)
    assert RUN.validate_invalidation([], "r1", reg)
    for k, bad in (("run", "r2"), ("reason", " "), ("correction_commit", "0" * 40), ("register_row", "C1-x"),
                   ("approved_by", "someone")):
        assert RUN.validate_invalidation({**good, k: bad}, "r1", reg), k


def test_coverage_is_checked_across_every_seed_not_just_the_first():
    from datetime import date
    days = {(2021, 1): {"unknown_dates": []}, (2021, 2): {"unknown_dates": [date(2021, 9, 1)]},
            (2022, 1): {"unknown_dates": []}, (2022, 2): {"unknown_dates": []}}
    cov = RUN.coverage(days, seasons=(2021, 2022), seeds=(1, 2))
    assert cov["complete"] is False
    assert cov["by_season"]["2021"] == {"2021-09-01": [2]}
    assert cov["by_season"]["2022"] == {}


NO_PLAY = "| C1-4b-2023-10-02 | gap | **RULED 2026-10-05: NO PLAY** | Eric |\n"
CONFIRMED = ("| C1-4b-generator-commit | x | **RULED:** (a) **Outcome 2026-10-05: CONFIRMED BY REPRODUCTION.** "
             "85224124f97a0d4ee8da3059ff54b1bad228d44e | Eric |\n")


def test_owner_gates_refuse_an_open_coverage_row_and_a_missing_generator_ruling():
    open_row = "| C1-4b-2023-10-02 | gap | **OPEN.** Nothing is recorded | — |\n"
    assert len(RUN.owner_gates(open_row)) == 2
    proposed = NO_PLAY + "| C1-4b-generator-commit | x | **PROPOSED, awaiting Eric** | — |\n"
    assert len(RUN.owner_gates(proposed)) == 1                      # a proposal is not a ruling
    pending = NO_PLAY + "| C1-4b-generator-commit | x | **RULED:** (a) **Outcome: PENDING** | Eric |\n"
    assert len(RUN.owner_gates(pending)) == 1                       # ruled, but its check has not run
    assert RUN.owner_gates(NO_PLAY + CONFIRMED) == []


def test_owner_gates_need_the_implemented_disposition_and_a_matching_outcome(monkeypatch):
    """Code review r2 N4: 'RULED: PLAY' plus 'RULED: UNKNOWN' passed before; text must match what the run implements."""
    play = "| C1-4b-2023-10-02 | gap | **RULED: PLAY** | Eric |\n"
    unknown = "| C1-4b-generator-commit | x | **RULED: UNKNOWN** | Eric |\n"
    assert len(RUN.owner_gates(play + unknown)) == 2
    no_outcome = "| C1-4b-generator-commit | x | **RULED:** y | Eric |\n"
    assert len(RUN.owner_gates(NO_PLAY + no_outcome)) == 1
    other_commit = CONFIRMED.replace("85224124f97a0d4ee8da3059ff54b1bad228d44e", "0" * 40)
    assert len(RUN.owner_gates(NO_PLAY + other_commit)) == 1
    recorded_unknown = "| C1-4b-generator-commit | x | **RULED:** (a) **Outcome 2026-10-05: UNKNOWN.** | Eric |\n"
    assert len(RUN.owner_gates(NO_PLAY + recorded_unknown)) == 1   # the recipe still names a commit
    monkeypatch.setitem(RUN.PROFILE_RECIPE, "generator_commit", None)
    assert RUN.owner_gates(NO_PLAY + recorded_unknown) == []
    assert len(RUN.owner_gates(NO_PLAY + CONFIRMED)) == 1


def test_the_live_register_passes_the_owner_gates():
    assert RUN.owner_gates((RUN.REPO / RUN.REGISTER_REL).read_text()) == []


def test_self_check_catches_a_broken_solver(monkeypatch):
    from scripts.audit.c1_r4b import solvers as S
    real = S.solve

    def broken(env, **kw):
        sol = real(env, **kw)
        sol.value[0, 0, -1, 1] += 0.01
        return sol
    monkeypatch.setattr(RUN.S, "solve", broken)
    assert RUN.self_check()["ok"] is False


def test_self_check_rejects_a_nan_value_array_with_the_real_policy(monkeypatch):
    """Code review r2 N6: max(worst, nan) could keep the finite worst, so a NaN value array passed."""
    from scripts.audit.c1_r4b import solvers as S
    real = S.solve

    def nan_values(env, **kw):
        sol = real(env, **kw)
        sol.value[...] = np.nan
        return sol
    monkeypatch.setattr(RUN.S, "solve", nan_values)
    out = RUN.self_check()
    assert out["ok"] is False and not np.isfinite(out["errors"]["solver_emax"])


def test_self_check_rejects_an_action_outside_the_domain(monkeypatch):
    from scripts.audit.c1_r4b import solvers as S
    real = S.solve

    def bad_action(env, **kw):
        sol = real(env, **kw)
        sol.policy[0, 0, 0, 0, 0] = 7
        return sol
    monkeypatch.setattr(RUN.S, "solve", bad_action)
    assert RUN.self_check()["ok"] is False


def test_self_check_rejects_a_non_finite_oracle_value(monkeypatch):
    monkeypatch.setattr(RUN.O, "optimal", lambda *a, **k: float("nan"))
    assert RUN.self_check()["ok"] is False


def test_r_bar_must_be_finite_and_positive():
    from scripts.audit.c1_r4b import fit as F
    d = {"opp": np.ones(4, bool), "known": np.ones(4, bool), "partner": np.ones(4, bool), "hit2": np.zeros(4, bool)}
    with pytest.raises(ValueError):
        F.r_bar([d])


def test_the_no_play_ruling_ends_the_2023_calendar_on_10_01():
    from datetime import date
    from scripts.audit.c1_r4b import data as D
    sched = {"dates": [{"date": d, "games": [{"gamePk": i + 1, "status": {"detailedState": "Final"}}]}
                       for i, d in enumerate(["2023-09-29", "2023-09-30", "2023-10-01", "2023-10-02"])]}
    cal = D.exclude_dates(D.calendar_from_schedule(sched, 2023), ["2023-10-02"])
    assert cal.final == date(2023, 10, 1) and cal.exclusive_end == date(2023, 10, 2) and not cal.no_opportunity


def test_achieved_haircut_reports_none_for_a_zero_primary_hit_denominator():
    """Code review r2 F6: an undefined conditional reduction is None (with its denominator), never 0."""
    sd = {"opp": np.ones(4, bool), "known": np.ones(4, bool), "partner": np.ones(4, bool),
          "hit1": np.zeros(4, bool), "hit2": np.array([True, True, False, True])}
    out = RUN._achieved(sd, np.array([[True, False, True, False]]))
    assert out["n_primary_hit"] == 0 and out["achieved_primary_hit_conditional_reduction"] is None
    sd["hit1"] = np.array([True, True, True, False])
    out = RUN._achieved(sd, np.array([[True, False, True, False]]))
    assert out["n_primary_hit"] == 3 and out["achieved_primary_hit_conditional_reduction"] == pytest.approx(1 / 3)


def test_seed_metrics_carry_the_play_total():
    res = {k: [1.0] for k in ("max", "resets", "reach20", "reach30", "reach40", "reach57")}
    res["actions"] = {"skip": [4.0], "single": [10.0], "double": [3.0]}
    res["skip_census"] = {}
    m = RUN._seed_metrics(res)
    assert float(m["act_play"][0]) == 13.0
