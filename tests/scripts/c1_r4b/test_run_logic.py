"""T8 pure logic: disposition precedence (registration §6), Monte Carlo ambiguity (§4), aggregation, provenance."""
import hashlib

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


def test_the_run_refuses_without_a_published_x31(tmp_path, monkeypatch):
    monkeypatch.setattr(RUN, "X31_COMMIT", None)
    with pytest.raises(SystemExit):
        RUN.x31_gate()


def test_coverage_is_checked_across_every_seed_not_just_the_first():
    from datetime import date
    days = {(2021, 1): {"unknown_dates": []}, (2021, 2): {"unknown_dates": [date(2021, 9, 1)]},
            (2022, 1): {"unknown_dates": []}, (2022, 2): {"unknown_dates": []}}
    cov = RUN.coverage(days, seasons=(2021, 2022), seeds=(1, 2))
    assert cov["complete"] is False
    assert cov["by_season"]["2021"] == {"2021-09-01": [2]}
    assert cov["by_season"]["2022"] == {}
