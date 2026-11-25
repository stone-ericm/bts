"""Item 9 mutation testing probe: apply 16 single-change mutations on an isolated scratch copy of src/ and scripts/."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REVIEW_DIR = REPO / ".codex-review" / "wr-g"
SCRATCH_ROOT = REVIEW_DIR / "scratch_mut"

PRIMARY_TESTS = [
    "tests/test_wr_g_guard.py",
    "tests/test_boundary_shadow_census_g.py",
    "tests/test_boundary_shadow_census.py",
]
EXTENDED_TESTS = [
    "tests/test_daily_decision.py",
    "tests/test_scheduler_tail_objective.py",
    "tests/test_tail_policy_r3_fixes.py",
    "tests/test_decide_action.py",
    "tests/test_strategy.py",
]


MUTATIONS = [
    (
        "m01_lowest_playing_bin",
        "src/bts/simulate/mdp.py",
        "        for j in range(q - 1, -1, -1):",
        "        for j in range(0, q):",
    ),
    (
        "m02_guard_in_tail_regime",
        "src/bts/strategy.py",
        "        policy_action, degraded = _tail_action(\n            ctx.mdp, ctx.primary_p, streak, d_eff, saver, best_for_lookup)",
        "        policy_action, degraded = _tail_action(\n            ctx.mdp, ctx.primary_p, streak, d_eff, saver, best_for_lookup)\n        if ctx.mdp and ctx.mdp.get('policy_table') is not None:\n            policy_action, guard = _mdp_action_from(ctx.mdp, ctx.primary_p, streak, ctx.date, saver)",
    ),
    (
        "m03_guard_when_raw_not_skip",
        "src/bts/simulate/mdp.py",
        '    if action == "skip":',
        '    if action in ("skip", "single"):',
    ),
    (
        "m04_drop_guard_on_clamp",
        "src/bts/strategy.py",
        '    if action == "double" and not ctx.allow_double:\n        action = "single"\n    if action == "double" and not ctx.has_diff_game:\n        action = "single"',
        '    if action == "double" and not ctx.allow_double:\n        action = "single"\n        guard = None\n    if action == "double" and not ctx.has_diff_game:\n        action = "single"\n        guard = None',
    ),
    (
        "m05_raw_from_mdp_action_from",
        "src/bts/strategy.py",
        "    from bts.simulate.mdp import lookup_action_guarded\n\n    days_remaining = _days_remaining(date)\n\n    return lookup_action_guarded(\n        mdp[\"policy_table\"], mdp[\"boundaries\"],\n        streak, days_remaining, saver, p_game_hit, mdp[\"season_length\"],\n    )",
        "    from bts.simulate.mdp import lookup_action\n\n    days_remaining = _days_remaining(date)\n\n    return lookup_action(\n        mdp[\"policy_table\"], mdp[\"boundaries\"],\n        streak, days_remaining, saver, p_game_hit, mdp[\"season_length\"],\n    ), None",
    ),
    (
        "m06_guard_lookup_action_itself",
        "src/bts/simulate/mdp.py",
        '    indices = _lookup_indices(boundaries, streak, days_remaining, saver,\n                              top_pick_confidence, season_length)\n    return "skip" if indices is None else ACTIONS[policy_table[indices]]',
        "    return lookup_action_guarded(policy_table, boundaries, streak, days_remaining, saver, top_pick_confidence, season_length)[0]",
    ),
    (
        "m07_omit_policy_guard_on_skip",
        "src/bts/daily_decision.py",
        '            "policy_guard": policy_guard if action != "skip" else None,\n            "finalized_at": _utc_iso(now),\n        }',
        '            "policy_guard": policy_guard,\n            "finalized_at": _utc_iso(now),\n        }\n        if action == "skip":\n            del record["policy_guard"]',
    ),
    (
        "m08_write_guard_on_skip_record",
        "src/bts/daily_decision.py",
        '            "policy_guard": policy_guard if action != "skip" else None,',
        '            "policy_guard": policy_guard,',
    ),
    (
        "m09_skip_saved_pick_recovery",
        "src/bts/scheduler.py",
        '        if saved.get("guard") is not None:\n            meta["policy_guard"] = saved["guard"]',
        "        pass",
    ),
    (
        "m10_numpy_int_in_guard",
        "src/bts/simulate/mdp.py",
        '                    "rule": "monotone_skip_v1", "bin": int(q),\n                    "from_bin": int(j), "table_action": "skip",',
        '                    "rule": "monotone_skip_v1", "bin": np.int64(q),\n                    "from_bin": np.int64(j), "table_action": "skip",',
    ),
    (
        "m11_journal_on_unguarded_decision",
        "src/bts/scheduler.py",
        "    if guard is not None:",
        "    if True:\n        guard = guard or {'bin': 0, 'from_bin': 0}",
    ),
    (
        "m12_own_journal_hardcode_saver_unavailable",
        "src/bts/scheduler.py",
        '        saver = "available" if sel.saver_available else "unavailable"',
        '        saver = "unavailable"',
    ),
    (
        "m13_own_journal_raw_days_on_real_selection",
        "src/bts/scheduler.py",
        '        print(f"  POLICY GUARD — At streak {dec.streak} with {dec.days_effective} policy days "',
        '        from bts.strategy import _days_remaining\n        _d = _days_remaining(sel.pick_result.daily.date) if getattr(sel, "pick_result", None) and getattr(sel.pick_result, "daily", None) else dec.days_effective\n        print(f"  POLICY GUARD — At streak {dec.streak} with {_d} policy days "',
    ),
    (
        "m14_own_census_skip_guard_shape_validation",
        "scripts/audit/boundary_shadow_census.py",
        '        guarded_play = (\n            rec.get("scoreable") is True and rec.get("action") in ("single", "double")\n            and isinstance(guard, dict)\n            and guard.keys() == {"rule", "bin", "from_bin", "table_action"}\n            and guard["rule"] == "monotone_skip_v1" and guard["table_action"] == "skip"\n            and type(guard["from_bin"]) is int and type(guard["bin"]) is int\n            and 0 <= guard["from_bin"] < guard["bin"]\n        )',
        "        guarded_play = bool(guard)",
    ),
    (
        "m15_own_census_include_guarded_in_followup",
        "scripts/audit/boundary_shadow_census.py",
        '        if "primary" in row.get("diffs", {}) and row.get("parity") != "guarded_play_excluded"',
        '        if "primary" in row.get("diffs", {})',
    ),
    (
        "m16_own_census_accept_unguarded_raw_skip_play",
        "scripts/audit/boundary_shadow_census.py",
        '        if not guarded_play:\n            row["parity"] = attribution(raw_deployed, rec.get("action"))',
        '        if not guarded_play:\n            row["parity"] = ("guarded_play_excluded" if raw_deployed == "skip" and rec.get("action") in ("single", "double") else attribution(raw_deployed, rec.get("action")))',
    ),
]


def prepare_scratch() -> None:
    if SCRATCH_ROOT.exists():
        shutil.rmtree(SCRATCH_ROOT)
    SCRATCH_ROOT.mkdir(parents=True)
    shutil.copytree(REPO / "src", SCRATCH_ROOT / "src")
    shutil.copytree(REPO / "scripts", SCRATCH_ROOT / "scripts")
    for link_name in ("tests", "data", "docs", ".git", "pyproject.toml"):
        (SCRATCH_ROOT / link_name).symlink_to(REPO / link_name)


def run_pytest(test_paths: list[str], env: dict[str, str], timeout_s: int = 35) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-x", "-q", "--tb=line", *test_paths],
        cwd=SCRATCH_ROOT, env=env, capture_output=True, text=True, timeout=timeout_s,
    )
    return proc.returncode, (proc.stdout + proc.stderr).strip()


def main() -> None:
    prepare_scratch()
    env = dict(os.environ, PYTHONPATH=f"{SCRATCH_ROOT / 'src'}:{SCRATCH_ROOT}")

    chk = subprocess.run(
        [sys.executable, "-c", "import bts, scripts.audit.boundary_shadow_census as c; print(bts.__file__); print(c.__file__)"],
        cwd=SCRATCH_ROOT, env=env, check=True, capture_output=True, text=True, timeout=10,
    ).stdout.splitlines()
    assert str((SCRATCH_ROOT / "src").resolve()) in str(Path(chk[0]).resolve()), chk
    assert str((SCRATCH_ROOT / "scripts").resolve()) in str(Path(chk[1]).resolve()), chk

    results = []
    survived = []

    for name, rel_file, target, replacement in MUTATIONS:
        fpath = SCRATCH_ROOT / rel_file
        original = fpath.read_text()
        assert target in original, f"target not found for {name} in {rel_file}"
        assert original.count(target) == 1, f"ambiguous target for {name}"
        fpath.write_text(original.replace(target, replacement, 1))
        try:
            rc, out = run_pytest(PRIMARY_TESTS, env)
            caught_by = "primary" if rc != 0 else None
            if rc == 0:
                rc2, out2 = run_pytest(EXTENDED_TESTS, env)
                if rc2 != 0:
                    caught_by = "extended"
                    rc, out = rc2, out2
            last_line = out.splitlines()[-1] if out else ""
            entry = {
                "mutation": name,
                "file": rel_file,
                "caught": caught_by is not None,
                "caught_by": caught_by,
                "summary": last_line[:200],
            }
            results.append(entry)
            if not entry["caught"]:
                survived.append(name)
        finally:
            fpath.write_text(original)

    summary = {
        "scratch_bts_file": chk[0],
        "scratch_census_file": chk[1],
        "total_mutations": len(results),
        "caught_count": sum(1 for r in results if r["caught"]),
        "survived_count": len(survived),
        "survived": survived,
        "results": results,
    }
    (REVIEW_DIR / "probe_09_mutations_run.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
