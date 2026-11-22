"""Rerun the round-one set and mutate each round-two amendment."""

from pathlib import Path
import sys


HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

import run_mutants as runner
from run_mutants import Edit, Mutation

runner.HERE = HERE
runner.SCRATCH = HERE / "scratch"
runner.PATCHED = HERE.parents[2] / "src"

budget = runner.MUTATIONS["separate-status-budget"]
runner.MUTATIONS["separate-status-budget"] = Mutation(
    "pending_window_replans and (slow_status or partial or shared_budget)", budget.edits)

runner.MUTATIONS.update({
    "no-final-replan": Mutation("reuse_replans_gap_after_status_read", (
        Edit('                        if polled_for_defer and now < cutoff:\n'
             '                            plan = plan_fallback_action(now=now, **planner_inputs)\n', ''),
    )),
    "no-clear-retry": Mutation("clear_error and (reuse_refuses_when_attempt or reuse_checks_live_day)", (
        Edit('        try:\n            save_pick(daily, picks_dir)\n        except OSError:\n'
             '            save_pick(daily, picks_dir)\n        if now >= cutoff:',
             '        save_pick(daily, picks_dir)\n        if now >= cutoff:'),
    )),
    "no-locked-unconfirmed-alert": Mutation("pending_window_replans and attempt_scoreable", (
        Edit('        if record.get("delivery_status") == "locked_unconfirmed":\n'
             '            _alert_delivery_outcome_unknown(config, picks_dir, date)\n', ''),
    )),
    "late-before-void": Mutation("reused_postponed_pick_takes_void_refusal_at_cutoff", (
        Edit('                    if reused_check_cascade and plan.action == "defer":\n'
             '                        if (defer_void_games\n'
             '                                and len(defer_void_games) == len(_committed_pick_game_pks(daily))):\n'
             '                            _refuse_void_game_delivery(\n'
             '                                daily, config, picks_dir, state, date, "fallback", defer_void_games)\n'
             '                            continue\n'
             '                        now = _now_et()\n'
             '                        if now >= cutoff:\n'
             '                            _refuse_late_delivery(\n'
             '                                daily, config, picks_dir, state, date, "fallback", cutoff, now)\n'
             '                            continue\n',
             '                    if reused_check_cascade and plan.action == "defer":\n'
             '                        now = _now_et()\n'
             '                        if now >= cutoff:\n'
             '                            _refuse_late_delivery(\n'
             '                                daily, config, picks_dir, state, date, "fallback", cutoff, now)\n'
             '                            continue\n'
             '                        if (defer_void_games\n'
             '                                and len(defer_void_games) == len(_committed_pick_game_pks(daily))):\n'
             '                            _refuse_void_game_delivery(\n'
             '                                daily, config, picks_dir, state, date, "fallback", defer_void_games)\n'
             '                            continue\n'),
    )),
    "no-shadow-initial-commit": Mutation("fallback_reload and matching_scoreable", (
        Edit('                                preserve_reuse_commit(record)\n'
             '                                finish_fallback_lock(daily)\n'
             '                                break\n'
             '                            if pick_was_delivered(daily)',
             '                                preserve_reuse_commit(record)\n'
             '                                break\n'
             '                            if pick_was_delivered(daily)'),
    )),
    "reuse-log-before-rereads": Mutation("missed_pick_alert_survives_abandoned_reuse", (
        Edit('                        print("  FALLBACK: reusing this check\'s fresh cascade "\n'
             '                              "decision (no second refresh).", file=sys.stderr)\n', ''),
        Edit('                        before_refresh = set(confirmed_sides)\n'
             '                        refresh_started = _now_et()\n'
             '                        refresh_duration = 0.0',
             '                        print("  FALLBACK: reusing this check\'s fresh cascade "\n'
             '                              "decision (no second refresh).", file=sys.stderr)\n'
             '                        before_refresh = set(confirmed_sides)\n'
             '                        refresh_started = _now_et()\n'
             '                        refresh_duration = 0.0'),
    )),
})


if __name__ == "__main__":
    {"prepare": runner.prepare, "run": runner.run}[sys.argv[1]]()
