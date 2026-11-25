"""Try isolated behavior changes with no altered tracked sources or test assertions."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
MDP = "src/bts/simulate/mdp.py"
STRATEGY = "src/bts/strategy.py"
SCHED = "src/bts/scheduler.py"
RECORD = "src/bts/daily_decision.py"
CENSUS = "scripts/audit/boundary_shadow_census.py"
GUARD_TEST = "tests/test_wr_g_guard.py"
CENSUS_TEST = "tests/test_boundary_shadow_census_g.py"
NAMED = [GUARD_TEST, CENSUS_TEST, "tests/test_boundary_shadow_census.py",
         "tests/test_saver_cs3.py", "tests/test_wr_a_calendar_replay_carbon.py",
         "tests/test_tail_policy_r3_fixes.py", "tests/test_scheduler_tail_objective.py",
         "tests/test_daily_decision.py", "tests/test_strategy.py", "tests/test_decide_action.py",
         "tests/test_orchestrator.py", "tests/test_cli_integration.py", "tests/scripts/season_walk",
         "tests/test_lightgbm_optional.py", "tests/test_web_saver.py"]

MUTATIONS = {
    "lowest": (MDP, "range(q - 1, -1, -1)", "range(q)", "test_lookup_edge_shapes"),
    "tail": (STRATEGY, "source = \"mdp\"\n        tail =", """source = "mdp"
        if ctx.mdp and ctx.mdp.get("policy_table") is not None:
            reach_action, reach_guard = _mdp_action_from(ctx.mdp, ctx.primary_p, streak, ctx.date, saver)
            if reach_guard is not None:
                policy_action, guard = reach_action, reach_guard
        tail =""", "test_tail_ignores_reach57_donor"),
    "playing": (MDP, 'if action == "skip":\n        for j', "if action in ACTIONS:\n        for j",
                "test_raw_and_guarded_lookup_grid"),
    "clamp": (STRATEGY, "pick_bar=pick_bar, guard=guard,",
              "pick_bar=pick_bar, guard=guard if action == policy_action else None,",
              "test_affected_rows_and_independent_clamps"),
    "production_raw": (STRATEGY, """    from bts.simulate.mdp import lookup_action_guarded

    days_remaining = _days_remaining(date)
    return lookup_action_guarded(
        mdp["policy_table"], mdp["boundaries"],
        streak, days_remaining, saver, p_game_hit, mdp["season_length"],
    )""", """    from bts.simulate.mdp import lookup_action

    days_remaining = _days_remaining(date)
    return lookup_action(
        mdp["policy_table"], mdp["boundaries"],
        streak, days_remaining, saver, p_game_hit, mdp["season_length"],
    ), None""", "test_affected_rows_and_independent_clamps"),
    "offline_guarded": (MDP, 'return "skip" if indices is None else ACTIONS[policy_table[indices]]',
                        "return lookup_action_guarded(policy_table, boundaries, streak, days_remaining, saver, top_pick_confidence, season_length)[0]",
                        "test_raw_and_guarded_lookup_grid"),
    "skip_omit": (RECORD, "        path = decision_path(date, picks_dir)\n",
                  '        if action == "skip":\n            record.pop("policy_guard")\n        path = decision_path(date, picks_dir)\n',
                  "test_decision_round_trip_and_skip_predicate"),
    "skip_guard": (RECORD, 'policy_guard if action != "skip" else None', "policy_guard",
                   "test_decision_round_trip_and_skip_predicate"),
    "recovery": (SCHED, 'if saved.get("guard") is not None:', "if False:",
                 "test_pick_readers_and_fresh_recovered_commits"),
    "numpy": (MDP, '"bin": int(q)', '"bin": np.int64(q)', "test_affected_rows_and_independent_clamps"),
    "journal_extra": (SCHED, '    if dec.objective == "reach57" and not dec.degraded_reason:',
                      '    if guard is None:\n        print("  POLICY GUARD — unchanged", file=sys.stderr)\n    if dec.objective == "reach57" and not dec.degraded_reason:',
                      "test_guard_journal"),
    "early_return": (SCHED, "    from bts.daily_decision import write_decision\n    record = write_decision(",
                     """    from bts.daily_decision import load_decision, write_decision
    existing = load_decision(date, picks_dir)
    if existing is not None and existing.get("scoreable"):
        state.committed_pick_written = True
        save_state(state, picks_dir)
        return
    record = write_decision(""", "test_real_commit_replaces_prior_decision"),
    "selection_primary": (SCHED, "primary=(selection.primary_candidate if has_decision",
                          "primary=(selection.primary_candidate if selection is not None",
                          "test_guarded_delivery_records_selected_batter[warmup]"),
    "convert_bad_donor": (MDP, "if donor_code in (1, 2):", 'if ACTIONS[donor_code] in ("single", "double"):',
                          "test_malformed_donor_cells"),
    "journal_saver": (SCHED, 'saver = "available" if bool(int(sel.saver_available)) else "unavailable"',
                      'saver = "unavailable"', "test_guard_journal_saver_index"),
    "census_out_of_range": (CENSUS, ' < guard["bin"] < policy_table.shape[-1]', ' < guard["bin"]',
                            "test_invalid_guard_keeps_raw_parity_check"),
    "census_accept_play": (CENSUS, 'if guarded_play and raw_deployed == "skip":', "if guarded_play:",
                           "test_invalid_guard_keeps_raw_parity_check"),
    "census_unguarded": (CENSUS, 'if guarded_play and raw_deployed == "skip":',
                         'if raw_deployed == "skip" and (guarded_play or\n'
                         '                                        (guard is None and rec.get("action") in ("single", "double"))):',
                         None),
    "public_guard": (SCHED, '        _record_commit("delivered")\n        print(f"  LOCKED ({label}) — Posted',
                     """        from dataclasses import replace
        daily.policy_decision = (daily.policy_decision or {}) | {"guard": None}
        if selection is not None and selection.decision is not None:
            selection = replace(selection, decision=replace(selection.decision, guard=None))
        _record_commit("delivered")
        print(f"  LOCKED ({label}) — Posted""", "test_guarded_delivery_records_selected_batter[public]"),
    "allow_voided_clamp_slots": (SCHED, "else _row_from_daily(daily.double_down)),",
                                 "else _row_from_daily(daily.double_down or daily.pick)),", None),
}


def mutate(name):
    filename, before, after, test = MUTATIONS[name]
    target = HERE / "scratch" / name
    shutil.copytree(ROOT / "src", target / "src", dirs_exist_ok=True)
    original = ROOT / filename
    mutated = target / filename
    mutated.parent.mkdir(parents=True, exist_ok=True)
    source = original.read_text()
    assert source.count(before) == 1 and before != after, filename
    mutated.write_text(source.replace(before, after, 1))
    environment = os.environ | {"PYTHONPATH": f"{target / 'src'}:{ROOT}",
                                 "PYTHONDONTWRITEBYTECODE": "1"}
    loader = [
        "import pathlib, sys, importlib, pytest, bts",
        "assert pathlib.Path(bts.__file__).is_relative_to(pathlib.Path(sys.argv[1]) / 'src')",
        "print(bts.__file__, flush=True)",
        "if sys.argv[2] == 'census':",
        "    import scripts.audit",
        "    scripts.audit.__path__.insert(0, str(pathlib.Path(sys.argv[1]) / 'scripts/audit'))",
        "    module = importlib.import_module('scripts.audit.boundary_shadow_census')",
        "    assert pathlib.Path(module.__file__).is_relative_to(sys.argv[1])",
        "    print(module.__file__, flush=True)",
        "sys.exit(pytest.main(sys.argv[3:]))",
    ]
    args = [sys.executable, "-c", "\n".join(loader), str(target),
            "census" if filename == CENSUS else "bts", "-p", "no:cacheprovider",
            "-q", "--tb=short", "-x", "-k", "not test_do_post_saver_transition_socket_wiring"]
    tests = [f"{CENSUS_TEST if filename == CENSUS else GUARD_TEST}::{test}"] if test else NAMED
    with (HERE / f"mutation-{name}.log").open("w") as log:
        for suite in [tests, NAMED] if test else [NAMED]:
            result = subprocess.run(args + suite, cwd=ROOT, env=environment,
                                    stdout=log, stderr=subprocess.STDOUT, timeout=45)
            if result.returncode:
                break
    if result.returncode == 0:
        print(f"{name}: SURVIVED all named non-socket tests")
    elif result.returncode == 1 and "FAILED " in (HERE / f"mutation-{name}.log").read_text():
        failure = next(line for line in (HERE / f"mutation-{name}.log").read_text().splitlines()
                       if line.startswith("FAILED "))
        print(f"{name}: CAUGHT {failure}")
    else:
        raise RuntimeError(f"{name}: invalid pytest result {result.returncode}")


if __name__ == "__main__":
    for mutation in sys.argv[1:]:
        mutate(mutation)
