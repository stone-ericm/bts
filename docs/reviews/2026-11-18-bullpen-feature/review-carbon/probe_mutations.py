"""Mutation probe against tests/test_bullpen_same_day.py synthetic tests and leakage_audit canary."""

from __future__ import annotations

import ast
from pathlib import Path
import subprocess
import types
from unittest.mock import patch

import numpy as np
import pandas as pd

from bts.features import compute as real_compute
import tests.test_bullpen_same_day as tmod

REPO = Path(__file__).resolve().parents[2]
ORIG_SRC = (REPO / "src/bts/features/compute.py").read_text()
BASE_SRC = subprocess.check_output(
    ["git", "show", "43fc5628:src/bts/features/compute.py"], text=True
)


def load_canary_fn():
    audit_src = (REPO / "scripts/leakage_audit.py").read_text()
    tree = ast.parse(audit_src, filename="scripts/leakage_audit.py")
    fn_nodes = [
        node for node in tree.body
        if isinstance(node, (ast.Import, ast.ImportFrom))
        or (isinstance(node, ast.FunctionDef) and node.name == "audit_bullpen_availability")
    ]
    mod_ast = ast.Module(body=fn_nodes, type_ignores=[])
    ns: dict = {}
    exec(compile(mod_ast, "scripts/leakage_audit.py", "exec"), ns)
    return ns["audit_bullpen_availability"]


CANARY_FN = load_canary_fn()


TARGET_BLOCK = """        team = "_pitcher_team_id"
        keys = [team, "date"]
        feature = "opp_bullpen_hr_30g"
        if daily_bullpen.empty:
            df[feature] = np.nan
        else:
            daily_bullpen[feature] = daily_bullpen.groupby(team)["bp_hr"].transform(
                lambda x: x.rolling(30, min_periods=10).mean()
            )
            # Attach the latest strictly earlier bucket, even when its mean is NaN.
            # The as-of join needs global date order and matching team key dtypes.
            match_team = "_bullpen_asof_team"
            known = df[keys].dropna().drop_duplicates().sort_values("date", kind="stable")
            history = daily_bullpen[keys + [feature]].sort_values("date", kind="stable")
            known[match_team] = known[team].astype(object)
            history[match_team] = history.pop(team).astype(object)
            prior = pd.merge_asof(
                known, history, on="date", by=match_team, direction="backward",
                allow_exact_matches=False,
            ).drop(columns=match_team)
            df = df.merge(prior, on=keys, how="left", validate="m:1")"""

assert TARGET_BLOCK in ORIG_SRC, "TARGET_BLOCK not found in compute.py"


def build_mutants() -> dict[str, str]:
    mutants: dict[str, str] = {}
    mutants["0_branch_control"] = ORIG_SRC
    mutants["1_allow_exact_matches_True"] = ORIG_SRC.replace(
        "allow_exact_matches=False,", "allow_exact_matches=True,"
    )
    # 2a: forward-fill shifted series (compute shift(1).rolling, attach with allow_exact_matches=True)
    mutants["2a_ffill_shifted_series"] = ORIG_SRC.replace(
        "lambda x: x.rolling(30, min_periods=10).mean()",
        "lambda x: x.shift(1).rolling(30, min_periods=10).mean()",
    ).replace(
        "allow_exact_matches=False,",
        "allow_exact_matches=True,",
    )
    # 2b: shift(1).rolling + allow_exact_matches=False
    mutants["2b_asof_shifted_strict"] = ORIG_SRC.replace(
        "lambda x: x.rolling(30, min_periods=10).mean()",
        "lambda x: x.shift(1).rolling(30, min_periods=10).mean()",
    )
    # 2c: dropna on rolling mean before merge_asof (carries stale finite across undefined windows)
    mutants["2c_ffill_across_nan_windows"] = ORIG_SRC.replace(
        'history = daily_bullpen[keys + [feature]].sort_values("date", kind="stable")',
        'history = daily_bullpen[keys + [feature]].dropna(subset=[feature]).sort_values("date", kind="stable")',
    )
    # 3: drop dtype alignment
    mutants["3_drop_dtype_alignment"] = ORIG_SRC.replace(
        """            match_team = "_bullpen_asof_team"
            known = df[keys].dropna().drop_duplicates().sort_values("date", kind="stable")
            history = daily_bullpen[keys + [feature]].sort_values("date", kind="stable")
            known[match_team] = known[team].astype(object)
            history[match_team] = history.pop(team).astype(object)
            prior = pd.merge_asof(
                known, history, on="date", by=match_team, direction="backward",
                allow_exact_matches=False,
            ).drop(columns=match_team)""",
        """            known = df[keys].dropna().drop_duplicates().sort_values("date", kind="stable")
            history = daily_bullpen[keys + [feature]].sort_values("date", kind="stable")
            prior = pd.merge_asof(
                known, history, on="date", by=team, direction="backward",
                allow_exact_matches=False,
            )""",
    )
    # 4a: sort both known and history by [team, "date"]
    mutants["4a_sort_both_by_team_first"] = ORIG_SRC.replace(
        'known = df[keys].dropna().drop_duplicates().sort_values("date", kind="stable")',
        'known = df[keys].dropna().drop_duplicates().sort_values([team, "date"], kind="stable")',
    ).replace(
        'history = daily_bullpen[keys + [feature]].sort_values("date", kind="stable")',
        'history = daily_bullpen[keys + [feature]].sort_values([team, "date"], kind="stable")',
    )
    # 4b: sort history by [team, "date"] (i.e. keep daily_bullpen's existing sort order!)
    mutants["4b_sort_history_by_team_first"] = ORIG_SRC.replace(
        'history = daily_bullpen[keys + [feature]].sort_values("date", kind="stable")',
        'history = daily_bullpen[keys + [feature]].sort_values([team, "date"], kind="stable")',
    )
    # 4c: sort known by [team, "date"] only
    mutants["4c_sort_known_by_team_first"] = ORIG_SRC.replace(
        'known = df[keys].dropna().drop_duplicates().sort_values("date", kind="stable")',
        'known = df[keys].dropna().drop_duplicates().sort_values([team, "date"], kind="stable")',
    )
    # 5: put shift back and merge on exact date (base!)
    mutants["5_base_shift_exact_merge"] = BASE_SRC
    # 6: min_periods=9
    mutants["6_min_periods_9"] = ORIG_SRC.replace("min_periods=10", "min_periods=9")
    # 7: window=29
    mutants["7_window_29"] = ORIG_SRC.replace("rolling(30,", "rolling(29,")
    # 8a: drop null-key exclusion completely
    mutants["8a_drop_dropna_all"] = ORIG_SRC.replace(
        "known = df[keys].dropna().drop_duplicates()",
        "known = df[keys].drop_duplicates()",
    )
    # 8b: drop null-team exclusion only (keep dropna on date)
    mutants["8b_drop_null_team_only"] = ORIG_SRC.replace(
        "known = df[keys].dropna().drop_duplicates()",
        'known = df[keys].dropna(subset=["date"]).drop_duplicates()',
    )
    return mutants


SYNTHETIC_TESTS = [
    ("shrunk_regression", tmod.test_bullpen_availability_does_not_depend_on_same_day_reliever),
    ("warmed_doubleheader", tmod.test_warmed_doubleheader_presence_and_outcomes),
    ("window_matches_loop", tmod.test_bullpen_window_matches_loop),
    ("empty_history_False", lambda: tmod.test_bullpen_empty_history(False)),
    ("empty_history_True", lambda: tmod.test_bullpen_empty_history(True)),
]


def run_mutant(name: str, src: str):
    mod = types.ModuleType("mutant_compute")
    exec(compile(src, f"<{name}>", "exec"), mod.__dict__)
    results = {}
    with patch.object(real_compute, "compute_all_features", mod.compute_all_features):
        for tname, tfn in SYNTHETIC_TESTS:
            try:
                tfn()
                results[tname] = "PASS"
            except Exception as exc:
                results[tname] = f"FAIL({type(exc).__name__})"
        # Canary
        try:
            ok = bool(CANARY_FN())
            results["canary"] = "PASS" if ok else "FAIL(False)"
        except Exception as exc:
            results["canary"] = f"FAIL({type(exc).__name__})"

        # Also check multi-team interleaved reliever dates!
        rows = []
        lookup = {}
        pid = 0
        for d in range(12):
            dt = f"2025-04-{d+1:02d}"
            gpk = 9000 + d
            lookup[gpk] = {"away": 21, "home": 31, "away_tid": 51, "home_tid": 52}
            rows.append({
                "pa_id": pid, "date": dt, "game_pk": gpk, "batter_id": 11,
                "pitcher_id": 22, "pitch_hand": "R", "venue_id": 31,
                "is_home": True, "is_hit": d % 2,
                "pitch_calls": ["X"], "pitch_types": ["FF"], "pitch_speeds": [93.0],
                "pitch_spin_rates": [2300.0], "pitch_extensions": [6.0],
                "pitch_break_vertical": [15.0], "pitch_break_horizontal": [5.0],
                "pitch_px": [0.0], "pitch_pz": [2.5], "sz_top": 3.5, "sz_bottom": 1.5,
                "final_count_balls": 0, "final_count_strikes": 0,
                "launch_speed": 91.0, "launch_angle": 10.0,
            }); pid += 1
            rows.append({
                **rows[-1], "pa_id": pid, "pitcher_id": 32, "is_home": False, "is_hit": 1 - (d % 2),
            }); pid += 1
        try:
            tmod._assert_reference(pd.DataFrame(rows), lookup)
            results["our_interleaved_2teams"] = "PASS"
        except Exception as exc:
            results["our_interleaved_2teams"] = f"FAIL({type(exc).__name__})"

    return results


if __name__ == "__main__":
    import io
    from contextlib import redirect_stdout

    mutants = build_mutants()
    for name, src in mutants.items():
        buf = io.StringIO()
        with redirect_stdout(buf):
            res = run_mutant(name, src)
        summary = ", ".join(f"{k}={v}" for k, v in res.items())
        print(f"{name:32s} -> {summary}")
