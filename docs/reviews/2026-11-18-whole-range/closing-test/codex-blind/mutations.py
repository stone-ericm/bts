"""Run one baseline or one production-source mutation under the test-slot wrapper."""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory

REVIEW = Path(__file__).resolve().parent
ROOT = REVIEW.parents[1]

# Each case couples one real test to the production expression it is meant to detect.
CASES = [
    ("opener-boundary", "season_end_date", "2027_opening_night_has_a_full_season_ahead",
     "strategy", '2027: "2027-09-27"', '2027: "2026-09-28"'),
    ("finale-exclusive", "season_end_date", "2027_last_day_and_after",
     "strategy", '2027: "2027-09-27"', '2027: "2027-09-26"'),
    ("future-fallback", "season_end_date", "unconfigured_year_falls_back_to_late_september_and_keeps_picking",
     "strategy", '_DEFAULT_SEASON_END_MMDD = "09-28"', '_DEFAULT_SEASON_END_MMDD = "09-27"'),
    ("future-warn", "season_end_date", "health_warns_when_the_year_has_no_configured_season_end",
     "health/tail_policy", 'if today.year in SEASON_END_DATES:', 'if True:'),
    ("singleton-early", "thin_slate_checks", "a_one_game_slate_gets_three_checks",
     "scheduler", 'THIN_SLATE_EARLY_OFFSETS_MIN = (180, 120)', 'THIN_SLATE_EARLY_OFFSETS_MIN = (120,)'),
    ("moved-up", "thin_slate_checks", "a_two_hour_move_still_leaves_one_check_in_time",
     "scheduler", 'THIN_SLATE_EARLY_OFFSETS_MIN = (180, 120)', 'THIN_SLATE_EARLY_OFFSETS_MIN = (120,)'),
    ("two-game-early", "thin_slate_checks", "a_two_game_slate_gets_early_checks_for_both_games",
     "scheduler", 'THIN_SLATE_MAX_GAMES = 2', 'THIN_SLATE_MAX_GAMES = 1'),
    ("full-slate", "thin_slate_checks", "a_normal_slate_is_planned_exactly_as_before",
     "scheduler", 'THIN_SLATE_MAX_GAMES = 2', 'THIN_SLATE_MAX_GAMES = 3'),
    ("restart-past-early", "thin_slate_checks", "early_checks_already_in_the_past_are_dropped_but_the_regular_check_is_kept",
     "scheduler", 'if now is not None and t <= now:', 'if False:'),
    ("early-clusters", "thin_slate_checks", "early_checks_of_two_close_games_cluster_like_regular_checks",
     "scheduler", 'if clusters and (c["time_et"] - clusters[-1]["time_et"]) <= timedelta(minutes=cluster_min):',
     'if False:'),
    ("early-at-regular", "thin_slate_checks", "an_offset_no_earlier_than_the_regular_check_adds_nothing",
     "scheduler", 'if off <= offset_min:', 'if off < offset_min:'),
    ("outcome-clean-research", "day_outcome", "research_mode_expects_a_private_commit_and_no_entry",
     "day_outcome", 'if decision.get("delivery_status") == "private_locked":',
     'if decision.get("delivery_status") == "delivered":'),
    ("outcome-missing-decision", "day_outcome", "games_were_played_and_there_is_no_decision_record",
     "day_outcome", 'return out("no_decision", f"{games}', 'return out("no_games", f"{games}'),
    ("outcome-external-skip", "day_outcome", "a_skip_the_policy_did_not_decide_is_a_problem",
     "day_outcome", 'if decision.get("source") == "mdp":', 'if True:'),
    ("outcome-cutoff", "day_outcome", "a_confirmation_seen_only_after_the_cutoff_is_late",
     "day_outcome", 'in_time = [r for r in confirmed if (r.get("observation") or {}).get("before_cutoff") is True]',
     'in_time = confirmed'),
    ("outcome-entry-verification", "day_outcome", "a_confirmation_that_does_not_bind_to_this_day_does_not_count",
     "day_outcome", 'and decision_sha256 is not None and commit.get("decision_sha256") == decision_sha256',
     'and decision_sha256 is not None'),
    ("outcome-entry-game", "day_outcome", "batter_match_with_other_qualification_states_stays_unverifiable",
     "day_outcome", 'and all(s.get("state") in ("confirmed", "unit_unverified")',
     'and all(s.get("state") is not None'),
    ("outcome-all-void", "day_outcome", "all_void_games_are_no_games_even_with_a_decision",
     "day_outcome", 'if games > 0 and void_games == games:', 'if False:'),
    ("repeat-date", "streak_applied_dates", "the_same_date_is_applied_once",
     "picks", 'if date is not None and date in load_applied_dates(picks_dir, season=season):', 'if False:'),
    ("repeat-miss", "streak_applied_dates", "a_repeated_saver_miss_does_not_reset_the_streak",
     "picks", 'if date is not None and date in load_applied_dates(picks_dir, season=season):', 'if False:'),
    ("automatic-age", "automatic_reading_cs4c", "strict_elapsed_thresholds",
     "contest_state", 'elapsed > timedelta(hours=AUTOMATIC_READING_STALE_HOURS)', 'elapsed >= timedelta(hours=AUTOMATIC_READING_STALE_HOURS)'),
    ("automatic-kind", "automatic_reading_cs4c", "pure_kinds_and_no_io_or_clock",
     "contest_state", 'if s.observation_kind != "automatic":', 'if False:'),
    ("automatic-elapsed", "automatic_reading_cs4c", "dst_uses_utc_instants",
     "contest_state", 'elapsed = now.astimezone(timezone.utc) - recorded_utc',
     'elapsed = now.replace(tzinfo=None) - recorded_utc.astimezone(now.tzinfo).replace(tzinfo=None)'),
    ("live-startup", "live_gate_cs6a", "schedule_refuses_before_resource_access",
     "cli", 'if not dry_run:\n            click.echo(f"bts schedule: refusing', 'if False:\n            click.echo(f"bts schedule: refusing'),
    ("live-intent", "live_gate_cs6a", "entry_intent_can_exclude_the_live_gate",
     "entry_intent", 'if include_live_gate:', 'if True:'),
    ("dated-correction", "saved_pick_paths_cs5a", "scheduler_uses_same_cycle_object_and_policy_decision",
     "picks", 'live_kind == "dated" and record["dated"] == live_dated', 'live_kind == "dated"'),
    ("morning-lead", "morning_state_cs6b1", "margin_uses_earliest_planned_target",
     "scheduler", 'first_check_et - now < timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN)',
     'first_check_et - now <= timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN)'),
    ("morning-commits", "morning_state_cs6b1", "existing_commit_markers_suppress_startup_read",
     "scheduler", 'if is_scoreable_commit(date, picks_dir, None):', 'if False:'),
    ("morning-dedup-day", "morning_state_cs6b1_review", "the_dispatch_day_is_passed_explicitly",
     "scheduler", 'now_et_date=now.date(),\n            )\n        return', 'now_et_date=None,\n            )\n        return'),
]
HARNESS = """
import socket
import os
from pathlib import Path
from unittest.mock import patch
import pytest
import sys
from contextlib import ExitStack
import bts

assert Path(bts.__file__).resolve().is_relative_to(Path(os.environ["WRBLIND_MUTATED_SRC"]))

def forbidden(*args, **kwargs):
    raise AssertionError("unexpected network operation in mutation review")

with ExitStack() as stack:
    stack.enter_context(patch.object(socket, "getaddrinfo", forbidden))
    for method in ("connect", "connect_ex", "sendto", "bind"):
        stack.enter_context(patch.object(socket.socket, method, forbidden))
    sys.exit(pytest.main(sys.argv[1:]))
"""


def node(row):
    return f"tests/test_{row[1]}.py::test_{row[2]}"


def mutated(path, row):
    original = path.read_text()
    old, new = row[-2:]
    assert original.count(old) == 1, (row[0], row[3], original.count(old))
    result = original.replace(old, new)
    assert result != original, row[0]
    compile(result, str(path), "exec")
    return result


def validate():
    for row in CASES:
        name, function = node(row).split("::")
        assert f"def {function}(" in (ROOT / name).read_text(), row[0]
        mutated(ROOT / "src/bts" / f"{row[3]}.py", row)
    print(f"Validated {len(CASES)} source edits and test nodes")


def run(target):
    with TemporaryDirectory(dir=REVIEW) as scratch:
        scratch = Path(scratch)
        copy = scratch / "src"
        shutil.copytree(ROOT / "src", copy, ignore=shutil.ignore_patterns("__pycache__"))
        models = scratch / "data/models"
        models.mkdir(parents=True)
        for name in ("mdp_policy.npz", "mdp_tail_policy.npz"):
            shutil.copyfile(ROOT / "data/models" / name, models / name)
        nodes = list(dict.fromkeys(map(node, CASES)))
        if target != "baseline":
            case = next(row for row in CASES if row[0] == target)
            path = copy / "bts" / f"{case[3]}.py"
            path.write_text(mutated(path, case))
            nodes = [node(case)]
            print(f"{target}: {case[3]}.py :: {case[-2]} -> {case[-1]}", flush=True)
        env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(copy), str(ROOT))),
                   WRBLIND_MUTATED_SRC=str(copy), PYTHONDONTWRITEBYTECODE="1",
                   TZ="America/New_York", MPLCONFIGDIR=str(scratch), PYTEST_DISABLE_PLUGIN_AUTOLOAD="1")
        result = subprocess.run([sys.executable, "-c", HARNESS, "-p", "no:cacheprovider",
                                 "-c", str(ROOT / "pyproject.toml"), "--basetemp", str(scratch / "tmp"),
                                 "-q", "--tb=short", *(str(ROOT / item) for item in nodes)],
                                cwd=scratch, env=env, check=False)
        print(f"mutation-result {target} {result.returncode}", flush=True)
        return result.returncode if target == "baseline" else 0


if __name__ == "__main__":
    target = sys.argv[1]
    if target == "validate":
        validate()
    elif target == "list":
        print("baseline", *(row[0] for row in CASES), sep="\n")
    else:
        sys.exit(run(target))
