"""W1.4b bin-collapse read: served p against the MDP policy's quality-bin boundaries, by epoch (no outcomes)."""
import pandas as pd

from scripts.audit import bin_collapse_read as b


def test_epoch_is_calendar_month_split_by_the_policy_regime():
    assert b.epoch("2026-04-15", "reach57") == "2026-04/reach57"
    assert b.epoch("2026-09-05", "emax_season_best") == "2026-09/emax_season_best"
    assert b.epoch("2026-09-01", None) == "2026-09/unknown"


def test_table_uses_the_health_checks_classifier_and_reports_below_lowest_and_dominance():
    df = pd.DataFrame({"date": ["2026-05-01"] * 4 + ["2026-06-01"] * 2,
                       "objective": ["reach57"] * 6, "slot": ["primary"] * 6,
                       "p": [0.70, 0.75, 0.80, 0.85, 0.79, 0.81]})
    out = b.bin_table(df, boundaries=[0.78, 0.80, 0.82])
    may = out["2026-05/reach57"]["primary"]
    assert may["n"] == 4 and may["counts"] == {0: 2, 1: 0, 2: 1, 3: 1} and may["below_lowest"] == 2
    assert may["dominant_share"] == 0.5
    assert out["2026-06/reach57"]["primary"]["counts"] == {0: 0, 1: 1, 2: 1, 3: 0}
