import pandas as pd

from bts.simulate import tail_policy as tail
from scripts.rebuild_tail_policy import bin_rates


def generated_policy(probabilities, outcomes, n_bins):
    days = pd.DataFrame({"p": probabilities, "hit": outcomes, "both": outcomes})
    boundaries, freq, hits, both, counts = bin_rates(days, n_bins)
    sol = tail.solve_emax_season_best(freq, hits, both)
    return tail.TailPolicy(
        objective=tail.OBJECTIVE_TAIL, policy_table=sol.policy, boundaries=boundaries,
        bin_freq=freq.tolist(), bin_p_hit=hits.tolist(), bin_p_both=both.tolist(),
        target=sol.target, max_days=sol.max_days, base_policy_sha256="a" * 64,
        manifest=tail.tail_manifest(n_bins=n_bins, hits=sum(outcomes), both=sum(outcomes),
                                    late_seed_days=len(days), bin_counts=counts.tolist()),
        built_at="2027-04-01T00:00:00+00:00", solver="generated",
    )


def test_tied_tail_quantiles_produce_saveable_policy(tmp_path):
    policy = generated_policy([0.1], [False], 3)
    tail.save_tail_policy(policy, tmp_path / "tail.npz")


def test_tail_optimizer_and_artifact_validator_agree_on_stop_rule(tmp_path):
    policy = generated_policy([0.1, 0.6], [False, True], 2)
    # One day follows today's guaranteed miss, with 1/2 chance of a sure double.
    # Holding at 1 on a skip thus yields 2 in expectation, vs. 1 after a reset.
    assert tail.lookup_tail_action(policy, 1, 1, 2, False, 0.1) == "skip"
    tail.save_tail_policy(policy, tmp_path / "tail.npz")
