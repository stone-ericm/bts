"""Add independently runnable F4 tests and test the scope of two oracles."""
import sys
from pathlib import Path

import mutations

TESTS = str(Path(__file__).with_name("test_offline_followup.py").resolve().relative_to(mutations.REPO))

base = mutations.CASES
mutations.CASES = [
    (TESTS + "::" + name, path, before, after)
    for name, (_, path, before, after) in zip(
        ("test_boolean_model_streak", "test_bounded_decoder_failure",
         "test_report_compacts_long_advisory", "test_dangling_streak_is_absent"),
        base[25:29], strict=True,
    )
] + [
    mutations.pair("wr_h_longest", "test_positive_day_rule_selects_a_play_at_exact_ties",
                   "simulate/tail_policy", 'return ("single", "double", "skip")[values.index(gain)]',
                   'return "single"'),
    mutations.pair("wr_h2_caps", "test_k6_dated_cli_rejects_before_writes", "cli",
                   'f"best streak must be at most {MAX_SEASON_STREAK}"', '"invalid best streak"'),
]

if __name__ == "__main__":
    mutations.main(sys.argv[1])
