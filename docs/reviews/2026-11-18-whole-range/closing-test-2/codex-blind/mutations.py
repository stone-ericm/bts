"""Run one isolated production mutation per selected test through pytest."""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SCRATCH_TEST = str((HERE / "test_offline_contracts.py").relative_to(REPO))


def pair(test_file, test, filename, before, after):
    return ("tests/test_" + test_file + ".py::" + test,
            "src/bts/" + filename + ".py", before, after)


CASES = [
    pair("wr_g_guard", "test_lookup_edge_shapes", "simulate/mdp",
         "range(q - 1, -1, -1)", "range(0, q)"),
    pair("wr_g_guard", "test_malformed_donor_cells", "simulate/mdp",
         "if donor_code in (1, 2):", "if donor_code != 0:"),
    pair("wr_g_guard", "test_malformed_queried_cell", "simulate/mdp",
         "    action = ACTIONS[policy_table[indices]]", "    action = 'skip'"),
    pair("wr_g_guard", "test_guard_journal", "scheduler",
         '    if guard is not None:\n        saver =', '    if False:\n        saver ='),
    pair("wr_g_guard", "test_decision_round_trip_and_skip_predicate", "daily_decision",
         'policy_guard if action != "skip" else None', 'policy_guard'),
    pair("wr_h_longest", "test_h1_h2_h3_resolver_grid", "simulate/tail_policy",
         "streak >= LONGEST_STREAK_FROM and days_remaining", "streak > LONGEST_STREAK_FROM and days_remaining"),
    pair("wr_h_longest", "test_h6_display_before_artifacts", "strategy",
         "        return 0.0", "        return 0.125"),
    pair("wr_h_longest", "test_h7_finalized_decision", "daily_decision",
         'OBJECTIVES = ("reach57", "emax_season_best", "longest_streak")',
         'OBJECTIVES = ("reach57", "emax_season_best")'),
    pair("wr_h_longest", "test_positive_day_rule_selects_a_play_at_exact_ties", "simulate/tail_policy",
         'return ("single", "double", "skip")[values.index(gain)]',
         'return ("single", "double", "skip")[2 - values[::-1].index(gain)]'),
    pair("wr_h_longest", "test_clamped_double_persists_both_actions", "strategy",
         'if action == "double" and not ctx.allow_double:', 'if False:'),
    pair("wr_h_longest", "test_r1_malformed_paired_base_forces_single", "strategy",
         "_validate_longest_base(policy)", "pass  # ignore invalid base"),
    pair("wr_h2_caps", "test_m2_switched_fetch_trusts_high_best", "contest_state",
         "best < contest.streak or best > MAX_SEASON_STREAK", "best < contest.streak or best > 57"),
    pair("wr_h2_caps", "test_m3_dated_high_best", "contest_state",
         "0 <= active <= MAX_SEASON_STREAK", "0 <= active <= 57"),
    pair("wr_h2_caps", "test_m4_required_pick_survives_reload", "picks",
         'kind in {"dated", "automatic"} and streak > MAX_SEASON_STREAK',
         'kind in {"dated", "automatic"} and streak > 57'),
    pair("wr_h2_caps", "test_k4_automatic_reads", "contest_state",
         "streak > MAX_SEASON_STREAK\n", "False\n"),
    pair("wr_h2_caps", "test_k5_fetch_validation", "contest_fetch",
         "if best_streak > MAX_SEASON_STREAK:", "if False:"),
    pair("wr_h2_caps", "test_k6_dated_cli_rejects_before_writes", "cli",
         "if best_streak > MAX_SEASON_STREAK:", "if False:"),
    pair("wr_f1_wrong_game", "test_m1_missing_then_wrong_game", "cli",
         'if result["state"] == "wrong_game" and type(slot["game_pk"]) is int',
         'if result["state"] == "confirmed" and type(slot["game_pk"]) is int'),
    pair("wr_f1_wrong_game", "test_m3_only_wrong_double_down_leg_is_named", "cli",
         'if ok:\n        slots = selection_slots(daily)', 'if False:\n        slots = selection_slots(daily)'),
    pair("wr_f1_wrong_game", "test_m10_raising_read_confirms_batter_only", "cli",
         'reason = "match_game_unchecked"', 'reason = "match"'),
    pair("wr_f1_wrong_game", "test_n10_full_proof_clears_warning", "cli",
         'was_alerted or "initial" in prior_escalations or "wrong_game" in prior_escalations',
         'False'),
    pair("wr_f2_saver_flag", "test_m1_m2_live_warning_and_delivery", "health/contest_state",
         'if flag.state in {"active", "not_earned"}:', 'if flag.state == "active":'),
    pair("wr_f2_saver_flag", "test_m3_after_midnight_uses_processed_season_and_current_files", "health/runner",
         'historical_day = today\n', 'historical_day = None\n'),
    pair("wr_f2_saver_flag", "test_existing_digest_sends_first_and_stays_below_chat_limit", "health/runner",
         '    if mismatch_attention:\n', '    if False:\n'),
    pair("wr_f2_saver_flag", "test_failed_send_and_interrupt_before_status_can_retry", "health/runner",
         '            require_message_id=True,', '            require_message_id=False,'),
    pair("wr_f4_streak_file", "test_m2_bad_contents", "day_outcome",
         "if type(streak) is not int or streak < 0:", "if not isinstance(streak, int) or streak < 0:"),
    pair("wr_f4_streak_file", "test_k4_decoder_limits_preserve_no_decision", "day_outcome",
         'return f"JSON exceeds decoder limits ({type(exc).__name__})"', 'return "not valid JSON"'),
    pair("wr_f4_streak_file", "test_k3_compact_dm", "day_outcome",
         'if len(body) > DM_MAX_CODE_POINTS:', 'if len(body) > 2000:'),
    pair("wr_f4_streak_file", "test_k1_dangling_symlink", "day_outcome",
         "except MissingFileTargetError:\n        return None", 'except MissingFileTargetError:\n        return "missing"'),
    pair("streak_season_cs2", "test_row3_tagless_three_starts_at_one_instead_of_carrying_over", "picks",
         'data.get("season")', 'data.get("season", season)'),
    (SCRATCH_TEST + "::test_skip_with_corrected_dated_read", "src/bts/day_outcome.py",
     'if dated_inputs is not None and dated_inputs != skip_inputs:', 'if False:'),
    (SCRATCH_TEST + "::test_early_window_stays_open_until_the_real_cutoff", "src/bts/entry_receipt.py",
     'SUBMISSION_CUTOFF_MIN < lead - minute <= window_min', 'SUBMISSION_CUTOFF_MIN <= lead - minute <= window_min'),
    (SCRATCH_TEST + "::test_early_run_defers_for_original_slot", "src/bts/cli.py",
     'if early and has_original_slot(now.date(), first_pitch, window_min):', 'if False:'),
]


PYTEST_CODE = """
import os
import sys
from pathlib import Path
import bts
assert Path(bts.__file__).is_relative_to(os.environ["WRBLIND_SRC"]), bts.__file__
def network(event, _):
    if event in ("socket.connect", "socket.getaddrinfo", "socket.bind", "socket.sendto"):
        raise AssertionError("offline mutation runner blocked " + event)
sys.addaudithook(network)
import pytest
raise SystemExit(pytest.main(sys.argv[1:]))
"""


def run(target, src, work):
    env = dict(os.environ, PYTHONPATH=f"{src}{os.pathsep}{REPO}",
               PYTHONDONTWRITEBYTECODE="1", WRBLIND_SRC=str(src),
               TZ="America/New_York", MPLCONFIGDIR=str(work / "mpl"))
    proc = subprocess.run([sys.executable, "-c", PYTEST_CODE, "-o", "addopts=",
                           "-p", "no:cacheprovider", "--tb=short", "-q", target],
                          env=env, cwd=REPO, capture_output=True, text=True, timeout=25)
    return {"code": proc.returncode, "output": proc.stdout + proc.stderr}


def verify(cases):
    for test, path, old, new in cases:
        original = (REPO / path).read_text()
        if original.count(old) != 1:
            print(f"BAD SPEC {test} {path} {old!r} appears {original.count(old)} times", flush=True)


def main(out):
    if out == "--verify":
        verify(CASES)
        return
    rows = []
    with TemporaryDirectory(prefix="wrblind-mutation-") as scratch:
        work = Path(scratch)
        src = work / "src"
        shutil.copytree(REPO / "src", src)
        for test, path, before, after in CASES:
            original = (REPO / path).read_text()
            rec = {"test": test, "path": path, "before": before, "after": after}
            if original.count(before) != 1:
                rec.update(verdict="invalid mutation", matches=original.count(before))
            else:
                base = run(test, REPO / "src", work)
                rec["base"] = base
                if base["code"] != 0 or " passed" not in base["output"]:
                    rec["verdict"] = "baseline did not pass"
                else:
                    target = work / path
                    try:
                        target.write_text(original.replace(before, after, 1))
                        mutant = run(test, src, work)
                    finally:
                        target.write_text(original)
                    rec["mutant"] = mutant
                    rec["verdict"] = {0: "survived", 1: "killed"}.get(mutant["code"], "invalid run")
            rows.append(rec)
            Path(out).write_text(json.dumps(rows, indent=2))
            print(f"{len(rows):02d}/{len(CASES)} {rec['verdict']}: {test}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
