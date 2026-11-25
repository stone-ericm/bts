"""Scratch-copy mutation testing for Item 9: 18 brief-specified + 6 adversarial custom mutations."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SCRATCH = HERE / "scratch_mut"

MUTATIONS: list[tuple[str, str, str, str]] = [
    # 1. Lowest playing bin instead of highest
    (
        "m01_lowest_playing_bin",
        "src/bts/simulate/mdp.py",
        "for j in range(q - 1, -1, -1):",
        "for j in range(q):",
    ),
    # 2. Let guard apply in tail regime
    (
        "m02_guard_in_tail",
        "src/bts/strategy.py",
        '        if policy_action == "skip":\n            reason = (f"season-best {effective_m}',
        '        if policy_action == "skip":\n'
        '            _ga, _gg = _mdp_action_from(ctx.mdp, ctx.primary_p, streak, ctx.date, saver)\n'
        '            if _gg is not None:\n'
        '                policy_action, guard = _ga, _gg\n'
        '        if policy_action == "skip":\n            reason = (f"season-best {effective_m}',
    ),
    # 3. Let guard apply when raw action is not skip
    (
        "m03_guard_when_not_skip",
        "src/bts/simulate/mdp.py",
        '    if action == "skip":',
        '    if action != "double":',
    ),
    # 4. Drop guard on clamp
    (
        "m04_drop_guard_on_clamp",
        "src/bts/strategy.py",
        '    if action == "double" and not ctx.allow_double:\n        action = "single"',
        '    if action == "double" and not ctx.allow_double:\n        action = "single"\n        guard = None',
    ),
    # 5. Return raw action from _mdp_action_from
    (
        "m05_raw_from_mdp_action_from",
        "src/bts/strategy.py",
        "    from bts.simulate.mdp import lookup_action_guarded\n\n    days_remaining = _days_remaining(date)\n\n    return lookup_action_guarded(",
        "    from bts.simulate.mdp import lookup_action\n\n    days_remaining = _days_remaining(date)\n\n    return (lookup_action(",
    ),
    # 6. Guard lookup_action itself
    (
        "m06_guard_lookup_action_itself",
        "src/bts/simulate/mdp.py",
        '    return "skip" if indices is None else ACTIONS[policy_table[indices]]',
        "    return lookup_action_guarded(policy_table, boundaries, streak, days_remaining, saver, top_pick_confidence, season_length)[0]",
    ),
    # 7. Omit policy_guard key on a skip
    (
        "m07_omit_policy_guard_on_skip",
        "src/bts/daily_decision.py",
        '            "policy_guard": policy_guard if action != "skip" else None,\n            "finalized_at": _utc_iso(now),\n        }',
        '            "finalized_at": _utc_iso(now),\n        }\n        if action != "skip":\n            record["policy_guard"] = policy_guard',
    ),
    # 8. Write guard on a skip record
    (
        "m08_write_guard_on_skip_record",
        "src/bts/daily_decision.py",
        '"policy_guard": policy_guard if action != "skip" else None,',
        '"policy_guard": policy_guard,',
    ),
    # 9. Skip recovery from saved pick's policy_decision
    (
        "m09_skip_recovery_from_saved_pick",
        "src/bts/scheduler.py",
        '        if saved.get("guard") is not None:\n            meta["policy_guard"] = saved["guard"]',
        "        pass",
    ),
    # 10. Numpy integer in guard
    (
        "m10_numpy_int_in_guard",
        "src/bts/simulate/mdp.py",
        '"rule": "monotone_skip_v1", "bin": int(q),\n                    "from_bin": int(j),',
        '"rule": "monotone_skip_v1", "bin": np.int64(q),\n                    "from_bin": np.int64(j),',
    ),
    # 11. Print journal line for unguarded decision
    (
        "m11_print_journal_unguarded",
        "src/bts/scheduler.py",
        "    if guard is not None:",
        '    if True:\n        guard = guard or {"bin": 0, "from_bin": 0}',
    ),
    # 12. Restore round-1 early return in _write_commit_decision
    (
        "m12_restore_r1_early_return",
        "src/bts/scheduler.py",
        "    from bts.daily_decision import write_decision",
        "    from bts.daily_decision import load_decision, write_decision\n"
        "    existing = load_decision(date, picks_dir)\n"
        '    if existing is not None and existing.get("scoreable"):\n'
        "        state.committed_pick_written = True\n"
        "        save_state(state, picks_dir)\n"
        "        return",
    ),
    # 13. Take primary from selection when it has no decision
    (
        "m13_primary_from_selection_without_decision",
        "src/bts/scheduler.py",
        "        primary=(selection.primary_candidate if has_decision",
        "        primary=(selection.primary_candidate if selection is not None",
    ),
    # 14. Convert malformed donor cell instead of passing over it
    (
        "m14_convert_malformed_donor",
        "src/bts/simulate/mdp.py",
        "            donor_code = policy_table[s, d, v, j]\n            if donor_code in (1, 2):\n                return ACTIONS[int(donor_code)], {",
        '            lower_action = ACTIONS[policy_table[s, d, v, j]]\n            if lower_action in ("single", "double"):\n                return lower_action, {',
    ),
    # 15. Print saver as unavailable always
    (
        "m15_saver_always_unavailable",
        "src/bts/scheduler.py",
        '        saver = "available" if bool(int(sel.saver_available)) else "unavailable"',
        '        saver = "unavailable"',
    ),
    # 16. Accept census guard with bin beyond table
    (
        "m16_census_accept_oob_bin",
        "scripts/audit/boundary_shadow_census.py",
        '            and 0 <= guard["from_bin"] < guard["bin"] < policy_table.shape[-1]',
        '            and 0 <= guard["from_bin"] < guard["bin"]',
    ),
    # 17. Accept census guard when raw action is not skip
    (
        "m17_census_accept_raw_not_skip",
        "scripts/audit/boundary_shadow_census.py",
        '        if guarded_play and raw_deployed == "skip":',
        "        if guarded_play:",
    ),
    # 18. Treat unguarded play over raw skip as excluded
    (
        "m18_census_unguarded_play_excluded",
        "scripts/audit/boundary_shadow_census.py",
        '        if guarded_play and raw_deployed == "skip":',
        '        if (guarded_play or (guard is None and rec.get("scoreable") is True and rec.get("action") in ("single", "double"))) and raw_deployed == "skip":',
    ),
    # --- Custom adversarial mutations ---
    # 19. Take double_down from selection when it has no decision (Warmup double blind spot)
    (
        "m19_custom_double_down_from_selection_without_decision",
        "src/bts/scheduler.py",
        "        double_down=(selection.double_candidate if has_decision",
        "        double_down=(selection.double_candidate if selection is not None",
    ),
    # 20. Revert journal saver label to bool(sel.saver_available) (Round-1 E1)
    (
        "m20_custom_journal_bool_without_int",
        "src/bts/scheduler.py",
        "bool(int(sel.saver_available))",
        "bool(sel.saver_available)",
    ),
    # 21. Drop guard on public Bluesky commit
    (
        "m21_custom_drop_guard_on_public_commit",
        "src/bts/scheduler.py",
        '        _trigger_live_forward_capture_on_lock(config, date)\n        _record_commit("delivered")\n        print(f"  LOCKED ({label}) — Posted to Bluesky:',
        '        _trigger_live_forward_capture_on_lock(config, date)\n        selection = None\n        daily.policy_decision = None\n        _record_commit("delivered")\n        print(f"  LOCKED ({label}) — Posted to Bluesky:',
    ),
    # 22. Census skips from_bin < bin / exact key / rule checks (only checks bin < n_bins)
    (
        "m22_custom_census_weak_schema_check",
        "scripts/audit/boundary_shadow_census.py",
        '            and guard.keys() == {"rule", "bin", "from_bin", "table_action"}\n'
        '            and guard["rule"] == "monotone_skip_v1" and guard["table_action"] == "skip"\n'
        '            and type(guard["from_bin"]) is int and type(guard["bin"]) is int\n'
        '            and 0 <= guard["from_bin"] < guard["bin"] < policy_table.shape[-1]',
        '            and isinstance(guard.get("bin"), int) and guard["bin"] < policy_table.shape[-1]',
    ),
    # 23. Donor negative code (-1, -2) accepted in lookup_action_guarded
    (
        "m23_custom_negative_donor_code_accepted",
        "src/bts/simulate/mdp.py",
        "if donor_code in (1, 2):",
        "if donor_code in (1, 2, -1, -2):",
    ),
    # 24. Journal uses raw _days_remaining on real SelectionResult when pick_result exists (March 24-30 cap blind spot)
    (
        "m24_custom_journal_raw_days_on_real_selection",
        "src/bts/scheduler.py",
        '        print(f"  POLICY GUARD — At streak {dec.streak} with {dec.days_effective} policy days "',
        '        from bts.strategy import _days_remaining as _dr\n'
        '        _d_shown = _dr(sel.pick_result.daily.date) if getattr(sel, "pick_result", None) and sel.pick_result.daily else dec.days_effective\n'
        '        print(f"  POLICY GUARD — At streak {dec.streak} with {_d_shown} policy days "',
    ),
]


def _fix_m05(scratch_root: Path) -> None:
    p = scratch_root / "src/bts/strategy.py"
    txt = p.read_text()
    target = (
        '    return (lookup_action(\n'
        '        mdp["policy_table"], mdp["boundaries"],\n'
        '        streak, days_remaining, saver, p_game_hit, mdp["season_length"],\n'
        '    )'
    )
    rep = target + ", None)"
    assert target in txt
    p.write_text(txt.replace(target, rep, 1))


def _prepare_scratch() -> None:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH)
    SCRATCH.mkdir(parents=True)
    shutil.copytree(ROOT / "src", SCRATCH / "src")
    (SCRATCH / "scripts" / "audit").mkdir(parents=True)
    (SCRATCH / "scripts" / "__init__.py").write_text("")
    (SCRATCH / "scripts" / "audit" / "__init__.py").write_text("")
    shutil.copy2(ROOT / "scripts/audit/boundary_shadow_census.py", SCRATCH / "scripts/audit/boundary_shadow_census.py")
    shutil.copytree(ROOT / "scripts/audit/season_walk", SCRATCH / "scripts/audit/season_walk")


def _run_pytest(extra_args: list[str], env: dict[str, str]) -> tuple[int, str]:
    proc = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-q", "--tb=no", "--no-header",
            "tests/test_wr_g_guard.py",
            "tests/test_boundary_shadow_census_g.py",
            "tests/test_boundary_shadow_census.py",
            *extra_args,
        ],
        cwd=ROOT, capture_output=True, text=True, timeout=30, env=env,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def main() -> None:
    _prepare_scratch()
    env = os.environ | {"PYTHONPATH": f"{SCRATCH / 'src'}:{SCRATCH}:{ROOT}"}

    # Baseline sanity on untouched scratch copy
    rc0, out0 = _run_pytest(["-k", "not grid"], env)
    assert rc0 == 0, out0

    results = {}
    for name, rel_path, target, replacement in MUTATIONS:
        file_path = SCRATCH / rel_path
        original = file_path.read_text()
        assert target in original, (name, rel_path)
        try:
            file_path.write_text(original.replace(target, replacement, 1))
            if name == "m05_raw_from_mdp_action_from":
                _fix_m05(SCRATCH)
            # Fast pass first (skipping 300k grids), then full pass if fast pass stays green
            rc_fast, out_fast = _run_pytest(["-k", "not grid"], env)
            if rc_fast != 0:
                killed = True
                detail = out_fast.splitlines()[-1] if out_fast else "failed"
            else:
                rc_full, out_full = _run_pytest([], env)
                killed = rc_full != 0
                detail = out_full.splitlines()[-1] if out_full else "passed"
            results[name] = {"killed": killed, "detail": detail}
        finally:
            file_path.write_text(original)

    shutil.rmtree(SCRATCH, ignore_errors=True)

    brief_18 = {k: v for k, v in results.items() if not k.startswith(("m19", "m20", "m21", "m22", "m23", "m24"))}
    custom_6 = {k: v for k, v in results.items() if k.startswith(("m19", "m20", "m21", "m22", "m23", "m24"))}
    summary = {
        "brief_total": len(brief_18),
        "brief_killed": sum(v["killed"] for v in brief_18.values()),
        "brief_survivors": [k for k, v in brief_18.items() if not v["killed"]],
        "custom_total": len(custom_6),
        "custom_killed": sum(v["killed"] for v in custom_6.values()),
        "custom_survivors": [k for k, v in custom_6.items() if not v["killed"]],
        "results": results,
    }
    (HERE / "probe_09_mutations.json").write_text(json.dumps(summary, indent=2))
    print("PROBE_09_DONE", json.dumps({k: v for k, v in summary.items() if k != "results"}))


if __name__ == "__main__":
    main()
