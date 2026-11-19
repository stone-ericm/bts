"""Run targeted single-line mutations against a disposable copy of the production package."""

import os
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PICK = 'test_pick_chosen_on_cs5a.py'
SAVED = 'test_saved_pick_paths_cs5a.py'
STALE = 'test_stale_delivery_cs4c.py'
AGE = 'test_automatic_reading_cs4c.py'
MORNING = 'test_morning_state_cs6b1.py'

# name, source module, original line fragment, mutant, test file, test id/filter
CASES = [
    ('explicit-required', 'picks', '"required": required,', '"required": False,', PICK, 'test_real_decision_state_identity_and_builder and automatic'),
    ('snapshot-copy', 'picks', 'dated.copy() if dated', 'dated if dated', PICK, 'test_real_decision_state_identity_and_builder and dated'),
    ('optional-refused', 'picks', 'elif not record["required"]:', 'elif False:', PICK, 'test_pure_matcher and optional'),
    ('dated-identity', 'picks', 'record["dated"] == live_dated', 'True', PICK, 'test_pure_matcher and best'),
    ('automatic-match', 'picks', 'live_kind == "automatic" or (live_kind', 'False or (live_kind', PICK, 'test_pure_matcher and automatic-automatic'),
    ('exact-int', 'picks', 'or type(streak) is not int or streak < 0', 'or not isinstance(streak, int) or streak < 0', PICK, 'test_invalid_record_normalized_in_all_loaders'),
    ('dated-canonical', 'picks', 'if effective != effective_date.isoformat():', 'if False:', PICK, 'test_invalid_record_normalized_in_all_loaders'),
    ('saver-consistency', 'picks', 'return value if value["saver_available"] == policy_saver else None', 'return value', PICK, 'test_invalid_record_normalized_in_all_loaders'),
    ('non-dated-identity', 'picks', 'return value if dated is None else None', 'return value', PICK, 'test_invalid_record_normalized_in_all_loaders'),
    ('fresh-identity', 'scheduler', 'and selection.pick_result is not None and selection.pick_result.daily is daily', 'and selection.pick_result is not None', SAVED, 'test_scheduler_uses_same_cycle_object_and_policy_decision and reloaded'),
    ('fresh-policy', 'scheduler', 'selection is not None and selection.decision is not None', 'selection is not None', SAVED, 'test_scheduler_uses_same_cycle_object_and_policy_decision and reused'),
    ('fallback-reload', 'scheduler', 'refusal = None if fresh_choice else saved_pick_refusal(daily, decision_state)', 'refusal = None', SAVED, 'test_overrun_reloads_the_same_cycle_choice_and_checks_it_as_saved'),
    ('fallback-fail', 'scheduler', 'refusal = None if fresh_choice else saved_pick_refusal(daily, decision_state)', 'refusal = None', SAVED, 'test_run_day_refusals_dedup_across_restart_and_missed_pick_has_no_candidate'),
    ('automatic-replacement', 'picks', 'live_kind == "automatic" or (live_kind', '(live_kind == "automatic" and record["streak"] == decision_state.streak) or (live_kind', SAVED, 'test_automatic_gap_then_real_fetch_and_restart_uses_cached_pick'),
    ('cli-warmup-refusal', 'cli', 'refusal = saved_pick_refusal(daily, decision_state)', 'refusal = None', SAVED, 'test_legacy_public_reuse and warmup and mismatch and run'),
    ('cli-catchup-refusal', 'cli', 'refusal = saved_pick_refusal(result.daily, decision_state)', 'refusal = None', SAVED, 'test_legacy_public_reuse and locked and missing and run'),
    ('orchestrate-catchup-refusal', 'orchestrator', 'refusal = saved_pick_refusal(result.daily, decision_state)', 'refusal = None', SAVED, 'test_legacy_public_reuse and locked and missing and orchestrate'),
    ('orchestrate-warmup-refusal', 'orchestrator', 'refusal = saved_pick_refusal(daily, decision_state)', 'refusal = None', SAVED, 'test_legacy_public_reuse and warmup and mismatch and orchestrate'),
    ('preview-stamp', 'cli', 'daily.chosen_on = chosen_on_record(decision_state, required=False)', 'daily.chosen_on = None', SAVED, 'test_fresh_choices_stamp_the_selection_read and preview'),
    ('run-stamp', 'cli', 'daily.chosen_on = chosen_on_record(decision_state, required=True)', 'daily.chosen_on = None', SAVED, 'test_fresh_choices_stamp_the_selection_read and run'),
    ('cascade-stamp', 'orchestrator', 'sel.pick_result.daily.chosen_on = chosen_on_record(decision_state, required=required)', 'sel.pick_result.daily.chosen_on = None', SAVED, 'test_real_policy_outputs_stamp_only_choices and double'),
    ('cascade-reuse-stamp', 'orchestrator', 'if sel.decision is not None and sel.pick_result is not None and sel.pick_result.daily is not None:', 'if sel.pick_result is not None and sel.pick_result.daily is not None:', SAVED, 'test_real_selection_reuse_preserves_record_or_absence and unavailable'),
    ('earliest-refusal-copy', 'scheduler', 'if _contest_state_required(config):\n        timing =', 'if False:\n        timing =', SAVED, 'test_missed_pick_alert_copy'),
    ('dated-clears-reading', 'scheduler', 'daily.automatic_reading = automatic_reading_record(decision_state, attempt_time)', 'daily.automatic_reading = automatic_reading_record(decision_state, attempt_time) or daily.automatic_reading', STALE, 'test_dated_attempt_clears_seeded_record'),
    ('dm-identifier', 'scheduler', 'if required and isinstance(msg_id, str) and msg_id:', 'if False:', STALE, 'test_run_day_chooses_delivers_and_warns_on_covered_old_file'),
    ('public-identifier', 'scheduler', 'if required and isinstance(uri, str) and uri:', 'if False:', STALE, 'test_attempt_snapshot_and_confirmed_warning and public'),
    ('age-14h', 'contest_state', 'elif elapsed > timedelta(hours=AUTOMATIC_READING_STALE_HOURS):', 'elif elapsed >= timedelta(hours=AUTOMATIC_READING_STALE_HOURS):', AGE, 'test_strict_elapsed_thresholds'),
    ('future-five-min', 'contest_state', 'if elapsed < timedelta(minutes=-5):', 'if elapsed <= timedelta(minutes=-5):', AGE, 'test_strict_elapsed_thresholds'),
    ('recorded-at-forwarded', 'contest_state', 'contest_recorded_at=contest.recorded_at,', 'contest_recorded_at=None,', AGE, 'test_real_loader_propagates_recorded_at'),
    ('startup-threshold', 'scheduler', 'elif first_check_et - now < timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN):', 'elif first_check_et - now <= timedelta(minutes=MORNING_CHECK_MIN_LEAD_MIN):', MORNING, 'test_margin_uses_earliest_planned_target'),
    ('startup-scoreable', 'scheduler', 'if is_scoreable_commit(date, picks_dir, None):', 'if False:', MORNING, 'test_existing_commit_markers_suppress_startup_read'),
    ('startup-delivered', 'scheduler', 'if daily is not None and (pick_was_delivered(daily) or daily.delivery_attempted):', 'if daily is not None and daily.delivery_attempted:', MORNING, 'test_existing_commit_markers_suppress_startup_read'),
    ('morning-dedup-date', 'scheduler', 'now_et_date=now.date(),', 'now_et_date=None,', MORNING, 'test_restarts_dedup_retry_and_roll_date'),
    ('morning-cross-date', 'scheduler', 'if date != now.date().isoformat():', 'if False:', MORNING, 'test_schedule_default_date_captured_before_midnight_load or (test_ineligible_start_has_no_effects and other_date)'),
    ('void-slate', 'day_outcome', 'if games > 0 and void_games == games:', 'if False:', 'test_day_outcome.py', 'test_command_all_postponed_or_cancelled_is_clean'),
    ('morning-dedup-date-control', 'scheduler',
     'status_path=health_dir / "contest_morning_check_dm_delivery_status.json",\n                now_et_date=now.date(),',
     'status_path=health_dir / "contest_morning_check_dm_delivery_status.json",\n                now_et_date=None,',
     'test_morning_state_cs6b1_carbon.py', 'test_morning_dispatches_pass_sampled_now_et_date_not_wall_clock and critical'),
    ('morning-warn-date-control', 'scheduler',
     'status_path=health_dir / "contest_stale_reading_dm_delivery_status.json",\n            now_et_date=now.date(),',
     'status_path=health_dir / "contest_stale_reading_dm_delivery_status.json",\n            now_et_date=None,',
     'test_morning_state_cs6b1_carbon.py', 'test_morning_dispatches_pass_sampled_now_et_date_not_wall_clock and warn'),
]


def run_test(src, filename, selector):
    env = os.environ | {
        'PYTHONPATH': os.pathsep.join(map(str, (src, ROOT))),
        'PYTHONDONTWRITEBYTECODE': '1', 'PYTEST_DISABLE_PLUGIN_AUTOLOAD': '1',
        'TZ': 'America/New_York',
    }
    command = [sys.executable, '-B', str(__file__), '--pytest', str(src),
               '-p', 'no:cacheprovider', '-q', '--tb=short', f'tests/{filename}', '-k', selector]
    return subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True, timeout=75)


def child():
    from unittest.mock import patch
    import bts
    import pytest

    assert Path(bts.__file__).is_relative_to(Path(sys.argv[2]))
    with patch('socket.socket.connect', side_effect=AssertionError('Sockets forbidden')), \
         patch('socket.socket.connect_ex', side_effect=AssertionError('Sockets forbidden')), \
         patch('socket.getaddrinfo', side_effect=AssertionError('DNS forbidden')), \
         patch('socket.socket.sendto', side_effect=AssertionError('Sockets forbidden')):
        raise SystemExit(pytest.main(sys.argv[3:]))


def audit(selected):
    with TemporaryDirectory(prefix='mutants-', dir=HERE) as path:
        src = Path(path) / 'src'
        shutil.copytree(ROOT / 'src' / 'bts', src / 'bts', ignore=shutil.ignore_patterns('__pycache__'))
        baseline = {}
        tally = {}
        for name, module, before, after, filename, selector in CASES:
            if selected and name not in selected:
                continue
            key = (filename, selector)
            if key not in baseline:
                baseline[key] = run_test(src, filename, selector)
            control = baseline[key]
            if control.returncode != 0:
                status = 'INVALID BASELINE'
                output = control.stdout + control.stderr
            else:
                target = src / 'bts' / f'{module}.py'
                original = target.read_text()
                count = original.count(before)
                if name == 'morning-dedup-date':
                    assert count == 2, (name, count)
                else:
                    assert count == 1, (name, count)
                try:
                    target.write_text(original.replace(before, after, 1))
                    mutant = run_test(src, filename, selector)
                finally:
                    target.write_text(original)
                status = {0: 'SURVIVED', 1: 'KILLED'}.get(mutant.returncode, 'INVALID MUTANT')
                output = mutant.stdout + mutant.stderr
            tally[status] = tally.get(status, 0) + 1
            print(f'{name}: {status} ({filename}: {selector})', flush=True)
            print(output, flush=True)
        print(tally)
        if any(status.startswith('INVALID') for status in tally):
            raise SystemExit(1)


if __name__ == '__main__':
    child() if '--pytest' in sys.argv else audit(set(sys.argv[1:]))
