"""Adversarial probe 7: scratch-copy mutation testing against the test suite (Item 7)."""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SCRATCH = OUT_DIR / "scratch"

MUTATIONS: list[tuple[str, str, str, str]] = [
    (
        "tie_order_skip_first",
        "src/bts/simulate/tail_policy.py",
        '    return ("single", "double", "skip")[values.index(gain)]',
        '    return ("skip", "single", "double")[(values[2], values[0], values[1]).index(gain)]',
    ),
    (
        "tie_order_double_first",
        "src/bts/simulate/tail_policy.py",
        '    return ("single", "double", "skip")[values.index(gain)]',
        '    return ("double", "single", "skip")[(values[1], values[0], values[2]).index(gain)]',
    ),
    (
        "swap_rule_operands",
        "src/bts/simulate/tail_policy.py",
        "        values = (p_hit * (1.0 + gain), p_both * (2.0 + gain), gain)",
        "        values = (p_both * (2.0 + gain), p_hit * (1.0 + gain), gain)",
    ),
    (
        "rule_off_by_one_days",
        "src/bts/simulate/tail_policy.py",
        "    for _ in range(days):",
        "    for _ in range(max(1, days - 1)):",
    ),
    (
        "drop_one_bin_check",
        "src/bts/strategy.py",
        '        if tail.n_bins != 1:\n            raise ValueError("tail policy must have one quality bin")',
        "        pass",
    ),
    (
        "drop_positive_horizon_check",
        "src/bts/strategy.py",
        '        if int(policy["season_length"]) <= 0:\n            raise ValueError("base policy season length must be positive")',
        "        pass",
    ),
    (
        "use_base_horizon_unclamped",
        "src/bts/strategy.py",
        "    if mdp_objective(streak, days) == OBJECTIVE_LONGEST:\n        return OBJECTIVE_LONGEST, min(days, LONGEST_STREAK_HORIZON)",
        '    if mdp_objective(streak, days) == OBJECTIVE_LONGEST:\n        return OBJECTIVE_LONGEST, effective_days(days, int((mdp or {}).get("season_length") or 180))',
    ),
    (
        "use_positive_base_horizon",
        "src/bts/strategy.py",
        "    if mdp_objective(streak, days) == OBJECTIVE_LONGEST:\n        return OBJECTIVE_LONGEST, min(days, LONGEST_STREAK_HORIZON)",
        '    if mdp_objective(streak, days) == OBJECTIVE_LONGEST:\n        _h = int((mdp or {}).get("season_length") or 180)\n        return OBJECTIVE_LONGEST, min(days, _h if _h > 0 else LONGEST_STREAK_HORIZON)',
    ),
    (
        "return_skip_when_degraded",
        "src/bts/strategy.py",
        '        return "single", reason, None',
        '        return "skip", reason, None',
    ),
    (
        "keep_tail_sha_when_degraded",
        "src/bts/strategy.py",
        '        return "single", reason, None',
        '        return "single", reason, "0" * 64',
    ),
    (
        "drop_new_enum_value",
        "src/bts/daily_decision.py",
        'OBJECTIVES = ("reach57", "emax_season_best", "longest_streak")',
        'OBJECTIVES = ("reach57", "emax_season_best")',
    ),
    (
        "decide_regime_after_loading_in_select",
        "src/bts/strategy.py",
        '    if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n        try:\n            mdp = _load_mdp()\n        except Exception as exc:\n            mdp = {"base_error": f"{type(exc).__name__}: {exc}"}\n    else:\n        mdp = _load_mdp()',
        "    mdp = _load_mdp()",
    ),
    (
        "decide_regime_after_loading_in_bar",
        "src/bts/strategy.py",
        "        if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n            return 0.0\n        if mdp is _UNSET:\n            mdp = _load_mdp()",
        "        if mdp is _UNSET:\n            mdp = _load_mdp()\n        if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n            return 0.0",
    ),
    (
        "make_bar_none",
        "src/bts/strategy.py",
        "        if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n            return 0.0",
        "        if _regime(None, streak, date)[0] == OBJECTIVE_LONGEST:\n            return None",
    ),
    (
        "boundary_strict_gt",
        "src/bts/simulate/tail_policy.py",
        "    if streak >= LONGEST_STREAK_FROM and days_remaining > 0:",
        "    if streak > LONGEST_STREAK_FROM and days_remaining > 0:",
    ),
    (
        "zero_days_included_in_longest",
        "src/bts/simulate/tail_policy.py",
        "    if streak >= LONGEST_STREAK_FROM and days_remaining > 0:",
        "    if streak >= LONGEST_STREAK_FROM and days_remaining >= 0:",
    ),
    (
        "catch_base_exception_in_rule",
        "src/bts/strategy.py",
        "    except Exception as exc:\n        reason = f\"longest-streak rule unavailable ({exc}); forced single\"",
        "    except BaseException as exc:\n        reason = f\"longest-streak rule unavailable ({exc}); forced single\"",
    ),
]


def evaluate_mutant(mut_root: Path, name: str, rel_file: str, target: str, replacement: str) -> dict:
    if mut_root.exists():
        shutil.rmtree(mut_root)
    shutil.copytree(REPO / "src", mut_root / "src")
    fpath = mut_root / rel_file
    src_text = fpath.read_text()
    assert target in src_text, f"Target not found for {name} in {rel_file}"
    fpath.write_text(src_text.replace(target, replacement, 1))

    env = os.environ | {"PYTHONPATH": os.pathsep.join((str(mut_root / "src"), str(REPO)))}

    # Fast stage (~1.5s): H tests (minus 20s parity subprocess), tail_policy tests, season_walk tests
    stage1 = subprocess.run(
        [
            sys.executable, "-m", "pytest", "-x", "-q", "--tb=no",
            "tests/test_wr_h_longest.py",
            "-k", "not test_legacy_parity_against_archived_base",
            "tests/simulate/test_tail_policy.py",
            "tests/scripts/season_walk/test_season_walk.py",
        ],
        cwd=REPO, env=env, capture_output=True, text=True, timeout=30,
    )
    if stage1.returncode != 0:
        last_line = (stage1.stdout.strip().splitlines() or ["failed"])[-1]
        return {"name": name, "status": "KILLED", "stage": "stage1_fast", "detail": last_line}

    # Stage 2: check every other focused test file individually (excluding test_legacy_parity_against_archived_base
    # only where it hardcodes REPO/src instead of PYTHONPATH, plus testing other files that use PYTHONPATH)
    extra_files = [
        ("tests/test_wr_g_guard.py", ["-k", "0-46"]),
        ("tests/test_strategy.py", []),
        ("tests/test_decide_action.py", []),
        ("tests/test_scheduler_tail_objective.py", []),
        ("tests/test_daily_decision_v3.py", []),
        ("tests/test_scheduler_decision_record.py", []),
    ]
    for tf, extra_args in extra_files:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-x", "-q", "--tb=no", tf, *extra_args],
            cwd=REPO, env=env, capture_output=True, text=True, timeout=35,
        )
        if proc.returncode != 0:
            last_line = (proc.stdout.strip().splitlines() or ["failed"])[-1]
            return {"name": name, "status": "KILLED", "stage": tf, "detail": last_line}

    return {"name": name, "status": "SURVIVED", "stage": "all_focused", "detail": "no test failed"}


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    mut_root = SCRATCH / "mut_work"
    results = []
    for name, rel_file, target, replacement in MUTATIONS:
        res = evaluate_mutant(mut_root, name, rel_file, target, replacement)
        results.append(res)
        print(f"{res['status']:8s} {name} ({res['stage']}: {res['detail']})", flush=True)
    if mut_root.exists():
        shutil.rmtree(mut_root)

    killed = [r["name"] for r in results if r["status"] == "KILLED"]
    survived = [r["name"] for r in results if r["status"] == "SURVIVED"]
    summary = {
        "total": len(results),
        "killed_count": len(killed),
        "survived_count": len(survived),
        "survived": survived,
        "results": results,
    }
    out_log = OUT_DIR / "probe_7_mutations.log"
    out_log.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"killed": len(killed), "survived": survived}, indent=2))


if __name__ == "__main__":
    main()
