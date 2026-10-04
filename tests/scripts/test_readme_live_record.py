"""W1.6 README live record from the accepted W1.1 ledger (X-30)."""
import pandas as pd

from scripts.audit import readme_live_record as r


def test_counts_only_committed_contest_graded_selections_and_counts_every_exclusion():
    led = pd.DataFrame([
        {"row_kind": "selection", "date": "2026-04-01", "slot": "primary", "commit_status": "committed_evidenced",
         "bts_outcome_status": "graded", "bts_outcome": "hit"},
        {"row_kind": "selection", "date": "2026-05-01", "slot": "primary", "commit_status": "committed_evidenced",
         "bts_outcome_status": "graded", "bts_outcome": "not_hit"},
        {"row_kind": "selection", "date": "2026-05-02", "slot": "double_down", "commit_status": "committed_evidenced",
         "bts_outcome_status": "graded", "bts_outcome": "hit"},
        {"row_kind": "selection", "date": "2026-05-03", "slot": "primary", "commit_status": "unknown",
         "bts_outcome_status": "graded", "bts_outcome": "hit"},
        {"row_kind": "selection", "date": "2026-05-04", "slot": "primary", "commit_status": "committed_evidenced",
         "bts_outcome_status": "unknown", "bts_outcome": None},
        {"row_kind": "skip_day", "date": "2026-05-05", "slot": None, "commit_status": None,
         "bts_outcome_status": None, "bts_outcome": None},
    ])
    out = r.live_record(led, era_start="2026-04-30")
    assert out["season"]["primary"] == {"hit": 1, "graded": 2}
    assert out["season"]["double_down"] == {"hit": 1, "graded": 1}
    assert out["since_2026-04-30"]["primary"] == {"hit": 0, "graded": 1}
    assert out["exclusions"] == {"not_committed_evidenced": 1, "not_contest_graded": 1}
