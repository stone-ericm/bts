"""W2.3 MLB forecast benchmark: scoring on the frozen shared pool (design rev 3, gates 2, 5 and 6)."""
import math

import numpy as np
import pandas as pd

from scripts.audit.mlb_benchmark import metrics as m


def _pool():
    # two dates; date 1 has 3 rows, date 2 has 1 row (equal-date weighting must not be row weighting)
    return pd.DataFrame({
        "date": ["d1", "d1", "d1", "d2"], "row_order": [0, 1, 2, 0],
        "batter_id": [1, 2, 3, 4], "game_pk": [10, 20, 30, 40],
        "ours": [0.8, 0.7, 0.6, 0.9], "mlb": [0.6, 0.75, 0.6, 0.5],
        "outcome": ["no_hit", "hit", "no_pa", "hit"],
    })


def test_targets_t1_counts_no_pa_as_no_hit_and_t2_drops_it_and_unknown_is_never_known():
    df = pd.concat([_pool(), pd.DataFrame({"date": ["d2"], "row_order": [1], "batter_id": [5], "game_pk": [50],
                                           "ours": [0.5], "mlb": [0.5], "outcome": ["unknown"]})])
    t1, t2 = m.target(df, "T1"), m.target(df, "T2")
    assert list(t1["y"]) == [0, 1, 0, 1] and list(t2["y"]) == [0, 1, 1]
    assert "unknown" not in set(t1["outcome"]) | set(t2["outcome"])


def test_proper_scores_are_equal_date_means_of_within_date_means():
    t2 = m.target(_pool(), "T2")            # d1: rows (0.8,0) (0.7,1); d2: (0.9,1)
    d1 = ((0.8 - 0) ** 2 + (0.7 - 1) ** 2) / 2
    d2 = (0.9 - 1) ** 2
    assert math.isclose(m.equal_date_mean(t2, "ours", m.brier_rows), (d1 + d2) / 2)
    res_d1, res_d2 = (0.8 + 0.7) / 2 - 0.5, 0.9 - 1
    assert math.isclose(m.equal_date_mean(t2, "ours", m.residual_rows), (res_d1 + res_d2) / 2)


def test_top1_is_chosen_on_the_pool_before_labels_ties_by_row_order_and_never_replaced():
    df = _pool()
    df.loc[2, "mlb"] = 0.75                 # ties row 1 on d1; row 1 has the lower row_order
    w = m.top1(df, "mlb")
    assert list(zip(w["date"], w["batter_id"])) == [("d1", 2), ("d2", 4)]
    df2 = _pool(); df2.loc[0, "outcome"] = "unknown"
    w2 = m.top1(df2, "ours")                # d1 winner is batter 1 with an unknown label: kept, not replaced
    assert w2.loc[w2["date"] == "d1", "batter_id"].item() == 1


def test_paired_top1_uses_only_dates_where_both_winners_are_target_known():
    df = _pool()
    df.loc[0, "outcome"] = "no_pa"          # ours' d1 winner (batter 1) becomes no_pa
    pair = m.paired_top1(df, "ours", "mlb", "T2")
    assert pair["dates_common"] == ["d2"] and pair["ours_rate"] == 1.0 and pair["mlb_rate"] == 1.0
    assert pair["excluded"]["ours"] == {"no_pa": 1}
    dis = m.disagreement(df, "ours", "mlb", "T1")   # d1: ours picks 1, MLB picks 2; d2 has one row, both pick 4
    assert dis["dates"] == ["d1"] and dis["dates_common"] == ["d1"]
    assert dis["ours_rate"] == 0.0 and dis["mlb_rate"] == 1.0


def test_bootstrap_copies_of_a_date_are_separate_blocks_and_failures_are_counted():
    df = pd.DataFrame({"date": ["a", "a", "b"], "v": [1.0, 3.0, 10.0]})
    seen = []

    def stat(x):
        seen.append(x["_block"].nunique())
        return x.groupby("_block")["v"].mean().mean()
    out = m.block_bootstrap(df, stat, n_resamples=200, seed=1)
    assert set(seen) == {2}                 # always two blocks, even when both draws are the same date
    assert out["n_failed"] == 0 and out["lo"] <= out["hi"]
    point = stat(df.assign(_block=df["date"]))
    assert point == (2.0 + 10.0) / 2

    def flaky(x):
        return float("nan") if (x["date"] == "b").all() else 1.0
    out2 = m.block_bootstrap(df, flaky, n_resamples=400, seed=2)
    assert out2["n_failed"] > 0 and out2["n_ok"] + out2["n_failed"] == 400


def test_encompassing_fits_share_rows_and_equal_date_weights_and_report_unavailable_on_failure():
    rng = np.random.default_rng(0)
    n = 4000
    ours = rng.uniform(0.55, 0.9, n)
    mlb = np.clip(ours + rng.normal(0, 0.05, n), 0.01, 0.99)
    y = (rng.uniform(size=n) < ours).astype(int)
    df = pd.DataFrame({"date": np.repeat(np.arange(200), 20), "ours": ours, "mlb": mlb, "y": y})
    fit = m.encompassing(df.assign(_block=df["date"]))
    assert fit["available"] and abs(fit["slope_ours_only"] - 1.0) < 0.35
    assert fit["delta_log_loss"] <= 1e-12   # nested in-sample fit can only match or improve
    sep = pd.DataFrame({"date": [0, 0, 1, 1], "ours": [0.2, 0.3, 0.8, 0.9], "mlb": [0.2, 0.3, 0.8, 0.9],
                        "y": [0, 0, 1, 1]}).assign(_block=lambda x: x["date"])
    bad = m.encompassing(sep)
    assert bad["available"] is False and bad["reason"]
