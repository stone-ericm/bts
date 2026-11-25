"""Adversarial probe 1b: rule arithmetic, regime dispatch, clamps, PolicyDecision fields, and constant patching."""
from __future__ import annotations

from dataclasses import asdict, fields, replace
from datetime import date, datetime, time, timedelta
from fractions import Fraction
import hashlib
from itertools import product
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = Path(__file__).resolve().parent
SCRATCH = OUT_DIR / "scratch"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from bts import strategy
from bts.simulate import tail_policy


def oracle_binary64(days: int, p_hit: float, p_both: float) -> tuple[str, float, tuple[float, float, float]]:
    """Independent IEEE-754 binary64 oracle for the one-bin uncapped expected-gain recurrence."""
    if days <= 0:
        return "skip", 0.0, (0.0, 0.0, 0.0)
    g = 0.0
    last = (0.0, 0.0, 0.0)
    for _ in range(days):
        v_single = float(p_hit) * (1.0 + g)
        v_double = float(p_both) * (2.0 + g)
        v_skip = g
        last = (v_single, v_double, v_skip)
        best = v_single
        if v_double > best:
            best = v_double
        if v_skip > best:
            best = v_skip
        g = best
    v_single, v_double, v_skip = last
    if v_single == g:
        return "single", g, last
    if v_double == g:
        return "double", g, last
    return "skip", g, last


def oracle_exact_fraction(days: int, p_hit: Fraction, p_both: Fraction) -> str:
    """Exact rational recurrence without floating-point rounding."""
    if days <= 0:
        return "skip"
    g = Fraction(0)
    v1 = v2 = v0 = Fraction(0)
    for _ in range(days):
        v1 = p_hit * (1 + g)
        v2 = p_both * (2 + g)
        v0 = g
        g = max(v1, v2, v0)
    if v1 == g:
        return "single"
    if v2 == g:
        return "double"
    return "skip"


def check_rule_arithmetic() -> dict:
    loaded = tail_policy.load_tail_policy(
        REPO / "data/models/mdp_tail_policy.npz",
        expected_base_sha=tail_policy.sha256_file(REPO / "data/models/mdp_policy.npz"),
    )
    p_ship = loaded.bin_p_hit[0]
    pb_ship = loaded.bin_p_both[0]
    assert p_ship == 0.7336111111111111
    assert pb_ship == 0.5511111111111111

    for non_pos in (-100, -1, 0, False):
        assert tail_policy.longest_streak_action(non_pos, p_ship, pb_ship) == "skip"

    shipped_ties = []
    for d in range(1, 401):
        actual = tail_policy.longest_streak_action(d, p_ship, pb_ship)
        want, gain, (v1, _v2, v0) = oracle_binary64(d, p_ship, pb_ship)
        assert actual == want, (d, actual, want)
        assert actual == ("double" if d <= 3 else "single"), (d, actual)
        if v1 == v0 == gain:
            shipped_ties.append(d)
    assert shipped_ties == list(range(116, 401)), (shipped_ties[:5], shipped_ties[-5:])

    for d in range(1, 61):
        exact = oracle_exact_fraction(d, Fraction(2641, 3600), Fraction(1984, 3600))
        assert exact == ("double" if d <= 3 else "single")

    special_rates = [
        (0.0, 0.0),
        (1e-15, 0.0),
        (0.25, 0.0),
        (0.75, 0.0),
        (1.0, 0.0),
        (1.0, 0.5),
        (1.0, 0.75),
        (1.0, 1.0),
        (0.5, 0.25),
        (0.8, 0.4),
        (0.2, 0.1),
        (0.5, math.nextafter(0.25, 0.0)),
        (0.5, math.nextafter(0.25, 1.0)),
        (0.01, 0.01),
        (0.5, 0.5),
        (0.99, 0.99),
        (p_ship, pb_ship),
    ]
    grid_ps = [i / 20.0 for i in range(21)]
    dense_rates = [(p, pb) for p in grid_ps for pb in grid_ps if pb <= p]
    all_rates = list(dict.fromkeys(special_rates + dense_rates))

    checked = 0
    for p, pb in all_rates:
        for d in range(1, 401):
            act = tail_policy.longest_streak_action(d, p, pb)
            want, _, _ = oracle_binary64(d, p, pb)
            assert act == want, (d, p, pb, act, want)
            assert act in ("single", "double"), (d, p, pb, act)
            checked += 1

    assert {tail_policy.longest_streak_action(d, 0.0, 0.0) for d in range(1, 401)} == {"single"}
    assert {tail_policy.longest_streak_action(d, 0.75, 0.0) for d in range(1, 401)} == {"single"}
    assert {tail_policy.longest_streak_action(d, 1.0, 1.0) for d in range(1, 401)} == {"double"}
    assert {tail_policy.longest_streak_action(d, 0.5, 0.25) for d in range(1, 401)} == {"single"}
    assert tail_policy.longest_streak_action(1, 0.5, math.nextafter(0.25, 1.0)) == "double"
    assert [tail_policy.longest_streak_action(d, 1.0, 0.75) for d in (1, 2, 3, 4)] == [
        "double", "double", "single", "single"
    ]
    assert tail_policy.longest_streak_action(1, 1.0, 0.5) == "single"

    return {
        "rate_pairs": len(all_rates),
        "rule_calls": checked,
        "shipped_tie_start": shipped_ties[0],
        "shipped_tie_count_to_400": len(shipped_ties),
    }


def _rewrite_base_horizon(tmp_dir: Path, artifact: dict, horizon: int) -> dict:
    base_path = tmp_dir / f"base_h{horizon}.npz"
    tail_path = tmp_dir / f"tail_h{horizon}.npz"
    with np.load(REPO / "data/models/mdp_policy.npz") as arrays:
        data = dict(arrays)
    data["season_length"] = np.array(horizon)
    np.savez_compressed(base_path, **data)
    sha = hashlib.sha256(base_path.read_bytes()).hexdigest()
    paired_tail = replace(artifact["tail"], base_policy_sha256=sha)
    tail_policy.save_tail_policy(paired_tail, tail_path)
    reloaded_tail = tail_policy.load_tail_policy(tail_path, expected_base_sha=sha)
    return {
        "policy_table": data["policy_table"],
        "boundaries": data["boundaries"].tolist(),
        "season_length": int(data["season_length"]),
        "base_sha256": sha,
        "tail": reloaded_tail,
        "tail_error": None,
        "base_error": None,
    }


def check_regime_and_decision_fields(tmp_dir: Path) -> dict:
    strategy._mdp_cache = None
    artifact = strategy._load_mdp()
    tail_sha = artifact["tail"].sha256

    streaks = [-100, -1, False, True, 0, 1, 28, 55, 56, 57, 58, 100, 10**12]
    raw_days_list = [0, 1, 2, 3, 4, 28, 29, 90, 115, 116, 180, 181, 269, 271]
    end_2028 = date(2028, 9, 28)

    regime_checks = 0
    for s in streaks:
        for raw in raw_days_list:
            day_str = (end_2028 - timedelta(days=raw)).isoformat()
            assert strategy._days_remaining(day_str) == raw
            obj_direct = tail_policy.mdp_objective(s, raw)
            obj_reg, d_eff = strategy._regime(artifact, s, day_str)
            assert obj_direct == obj_reg
            if s >= tail_policy.LONGEST_STREAK_FROM and raw > 0:
                assert obj_reg == tail_policy.OBJECTIVE_LONGEST
                assert d_eff == min(raw, tail_policy.LONGEST_STREAK_HORIZON)
            else:
                d_clamped = max(0, min(raw, 180))
                assert d_eff == d_clamped
                if d_clamped <= 0 or s >= 57:
                    assert obj_reg == tail_policy.OBJECTIVE_REACH57
                elif s + 2 * d_clamped >= 57:
                    assert obj_reg == tail_policy.OBJECTIVE_REACH57
                else:
                    assert obj_reg == tail_policy.OBJECTIVE_TAIL
            regime_checks += 1

    for s in (0, 56, 57, 100):
        assert tail_policy.mdp_objective(s, -5) == tail_policy.OBJECTIVE_REACH57

    # Alternate positive base horizon (season_length = 2 < 4!) paired with valid 1-bin tail:
    # must use LONGEST_STREAK_HORIZON (180), NOT base season_length (2), so raw=10 -> d_eff=10 -> single!
    short_base_mdp = _rewrite_base_horizon(tmp_dir, artifact, 2)
    ctx_short = strategy.DecisionContext(0.96, 0.95, True, "2027-09-17", True, short_base_mdp, 57, "trusted")
    dec_short = strategy.resolve_policy_decision(ctx_short, 57, False)
    assert dec_short.objective == "longest_streak"
    assert dec_short.days_effective == 10
    assert dec_short.policy_action == "single" and dec_short.action == "single"
    assert dec_short.degraded_reason is None and dec_short.tail_sha256 == short_base_mdp["tail"].sha256

    decision_checks = 0
    expected_fields = {
        "action", "policy_action", "source", "objective", "streak", "days_effective",
        "best_supplied", "best_status", "effective_best", "tail_sha256",
        "degraded_reason", "reason", "pick_bar", "guard",
    }
    assert {f.name for f in fields(strategy.PolicyDecision)} == expected_fields

    best_cases = [
        (None, None, "missing"),
        (None, "trusted", "missing"),
        (0, "trusted", "trusted"),
        (57, "trusted", "trusted"),
        (57, "untrusted", "untrusted"),
        (57, None, "untrusted"),
        (99, "bogus", "untrusted"),
    ]
    for s in (57, 58, 100, 10**12):
        for raw in (1, 2, 3, 4, 116, 180, 181, 271):
            day_str = (end_2028 - timedelta(days=raw)).isoformat()
            for saver, allow, has_diff, p in product(
                (False, True), (False, True), (False, True), (0.01, 0.80, 0.99)
            ):
                for best_val, best_in, best_norm in best_cases:
                    ctx = strategy.DecisionContext(
                        primary_p=p,
                        second_p=0.95,
                        has_diff_game=has_diff,
                        date=day_str,
                        allow_double=allow,
                        mdp=artifact,
                        best_streak=best_val,
                        best_status=best_in,
                    )
                    dec = strategy.resolve_policy_decision(ctx, s, saver)
                    act_tup, src_tup = strategy.decide_action(ctx, s, saver)
                    want_pol = "double" if raw <= 3 else "single"
                    want_act = "double" if (want_pol == "double" and allow and has_diff) else "single"
                    assert (act_tup, src_tup) == (want_act, "mdp")
                    assert asdict(dec) == {
                        "action": want_act,
                        "policy_action": want_pol,
                        "source": "mdp",
                        "objective": "longest_streak",
                        "streak": int(s),
                        "days_effective": min(raw, 180),
                        "best_supplied": best_val,
                        "best_status": best_norm,
                        "effective_best": None,
                        "tail_sha256": tail_sha,
                        "degraded_reason": None,
                        "reason": None,
                        "pick_bar": None,
                        "guard": None,
                    }
                    assert type(dec.streak) is int
                    decision_checks += 1

    return {"regime_checks": regime_checks, "decision_checks": decision_checks}


def check_literal_57_and_constant_patching(tmp_dir: Path) -> dict:
    diff = subprocess.run(
        ["git", "diff", "-U0", "0a2bec10", "813935bc", "--", "src", "scripts/audit/season_walk/walk.py"],
        cwd=REPO, capture_output=True, text=True, check=True, timeout=15,
    ).stdout
    added_code_57 = []
    for line in diff.splitlines():
        if line.startswith("+") and not line.startswith("+++"):
            body = line[1:].strip()
            cleaned = body.replace("OBJECTIVE_REACH57", "").replace("reach57", "")
            if "57" in cleaned and not cleaned.startswith("#") and not cleaned.startswith('"""'):
                added_code_57.append(body)
    assert added_code_57 == ["LONGEST_STREAK_FROM = 57"], f"Unexpected literal 57 in diff: {added_code_57}"

    patch_results = {}
    for target_val in (56, 58):
        copy_root = tmp_dir / f"const_{target_val}"
        if copy_root.exists():
            shutil.rmtree(copy_root)
        shutil.copytree(REPO / "src", copy_root / "src")
        tp_file = copy_root / "src/bts/simulate/tail_policy.py"
        text = tp_file.read_text()
        assert "LONGEST_STREAK_FROM = 57" in text
        tp_file.write_text(text.replace("LONGEST_STREAK_FROM = 57", f"LONGEST_STREAK_FROM = {target_val}", 1))

        probe_code = """
import json, sys
from datetime import date, datetime
from pathlib import Path
from bts import strategy
from bts.simulate import tail_policy
from scripts.audit.season_walk import walk

b = tail_policy.LONGEST_STREAK_FROM
assert b == int(sys.argv[1])
assert Path(strategy.__file__).resolve().is_relative_to(Path(sys.argv[2]).resolve())

art = strategy._load_mdp()
assert tail_policy.mdp_objective(b - 1, 10) == "reach57"
assert tail_policy.mdp_objective(b, 0) == "reach57"
assert tail_policy.mdp_objective(b, 10) == "longest_streak"
assert tail_policy.mdp_objective(b, 1) == "longest_streak"

ctx_d1 = strategy.DecisionContext(0.96, 0.95, True, "2027-09-26", True, art, b - 1, "trusted")
ctx_d10 = strategy.DecisionContext(0.96, 0.95, True, "2027-09-17", True, art, b - 1, "trusted")
below_d1 = strategy.resolve_policy_decision(ctx_d1, b - 1, False)
at_d1 = strategy.resolve_policy_decision(ctx_d1, b, False)
at_d10 = strategy.resolve_policy_decision(ctx_d10, b, False)
assert below_d1.objective == "reach57"
assert at_d1.objective == "longest_streak" and at_d1.action == "double"
assert at_d10.objective == "longest_streak" and at_d10.action == "single"
assert strategy.effective_pick_bar(b, "2027-07-15", False) == 0.0
assert strategy.effective_pick_bar(b - 1, "2027-07-15", False) != 0.0

frame = walk._predictions(date(2027, 9, 26), 0.96)
statuses = {int(pk): {"abstract": "P", "detailed": "Scheduled"} for pk in frame["game_pk"]}
tmp_picks = Path(sys.argv[2]) / "picks"
tmp_picks.mkdir(exist_ok=True)
with walk.offline(datetime(2027, 9, 26, 12, tzinfo=walk.ET)):
    sel = strategy.select_pick(frame, "2027-09-26", tmp_picks, streak=b, saver_available=False,
                               game_statuses_detailed=statuses)
assert sel.action == "double" and sel.decision.objective == "longest_streak"

assert walk.DEFAULT_STATES[-1].streak == b
with walk.offline(datetime(2027, 9, 26, 12, tzinfo=walk.ET)):
    install_problems = walk.install_state(tmp_picks / "cli", date(2027, 9, 26),
                                          walk.State("dated", b, b, "used", 0.96))
if b <= 57:
    assert install_problems == []
    rec = walk.walk_one(date(2027, 9, 26), walk.State("dated", b, b, "used", 0.96))
    assert rec["action"] == "double" and rec["objective"] == "longest_streak"
    assert walk.check([rec]) == []
else:
    assert below_d1.action == "skip"
    assert strategy.effective_pick_bar(57, "2027-07-15", False) is None
    assert len(install_problems) == 1

print(json.dumps({"boundary": b, "below_d1_action": below_d1.action, "install_problems": len(install_problems)}))
"""
        env = os.environ | {"PYTHONPATH": os.pathsep.join((str(copy_root / "src"), str(REPO)))}
        proc = subprocess.run(
            [sys.executable, "-c", probe_code, str(target_val), str(copy_root)],
            cwd=REPO, env=env, capture_output=True, text=True, check=True, timeout=30,
        )
        patch_results[str(target_val)] = json.loads(proc.stdout.strip().splitlines()[-1])

        pytest_proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "tests/simulate/test_tail_policy.py"],
            cwd=REPO, env=env, capture_output=True, text=True, check=True, timeout=45,
        )
        patch_results[f"pytest_tail_policy_{target_val}"] = pytest_proc.returncode

        if target_val == 56:
            # At 56 (<= state cap 57), test_wr_h_longest (excluding 0a2bec10 parity) and test_season_walk also pass!
            h_proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest", "-q",
                    "tests/test_wr_h_longest.py", "-k", "not test_legacy_parity_against_archived_base",
                ],
                cwd=REPO, env=env, capture_output=True, text=True, check=True, timeout=45,
            )
            sw_proc = subprocess.run(
                [sys.executable, "-m", "pytest", "-q", "tests/scripts/season_walk/test_season_walk.py"],
                cwd=REPO, env=env, capture_output=True, text=True, check=True, timeout=45,
            )
            patch_results["pytest_wr_h_56"] = h_proc.returncode
            patch_results["pytest_season_walk_56"] = sw_proc.returncode
        else:
            # At 58, pure resolver/display/decision tests in test_wr_h_longest pass, while live_case install_state hits the 57 cap (H2)
            h_pure_proc = subprocess.run(
                [
                    sys.executable, "-m", "pytest", "-q", "tests/test_wr_h_longest.py",
                    "-k", "not live_selection and not accepted_january and not h8_saved and not legacy_parity",
                ],
                cwd=REPO, env=env, capture_output=True, text=True, check=True, timeout=45,
            )
            patch_results["pytest_wr_h_pure_58"] = h_pure_proc.returncode

    return {"added_code_57": added_code_57, "patches": patch_results}


def test_probe_1b_all(tmp_path):
    r1 = check_rule_arithmetic()
    r2 = check_regime_and_decision_fields(tmp_path)
    r3 = check_literal_57_and_constant_patching(tmp_path)
    assert r1["rule_calls"] > 90000
    assert r2["decision_checks"] > 1500
    assert r3["patches"]["56"]["below_d1_action"] in ("single", "double")
    assert r3["patches"]["58"]["below_d1_action"] == "skip"


def main() -> None:
    SCRATCH.mkdir(parents=True, exist_ok=True)
    tmp_dir = SCRATCH / "p1b"
    if tmp_dir.exists():
        shutil.rmtree(tmp_dir)
    tmp_dir.mkdir(parents=True)
    summary = {
        "rule": check_rule_arithmetic(),
        "regime_and_fields": check_regime_and_decision_fields(tmp_dir),
        "constant_patching": check_literal_57_and_constant_patching(tmp_dir),
    }
    out_log = OUT_DIR / "probe_1_rule_and_regime.log"
    out_log.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
