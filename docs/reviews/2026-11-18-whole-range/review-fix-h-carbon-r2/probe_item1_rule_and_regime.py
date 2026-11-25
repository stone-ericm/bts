"""Adversarial probe for Item 1: longest_streak_action math, boundary/regime dispatch, PolicyDecision fields, literal 57 check, and patching LONGEST_STREAK_FROM to 56 and 58."""
from __future__ import annotations

from dataclasses import asdict
from datetime import date, timedelta
from fractions import Fraction
from itertools import product
from math import nextafter
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from bts import strategy
from bts.simulate import tail_policy as tp

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]


def independent_rule_binary64(days: int, p_hit: float, p_both: float) -> str:
    if days <= 0:
        return "skip"
    g = 0.0
    for _ in range(days):
        v_single = p_hit * (1.0 + g)
        v_double = p_both * (2.0 + g)
        v_skip = g
        best = v_single
        if v_double > best:
            best = v_double
        if v_skip > best:
            best = v_skip
        g = best
    if v_single == g:
        return "single"
    if v_double == g:
        return "double"
    return "skip"


def independent_exact_fraction(days: int, p_hit: float, p_both: float) -> str:
    if days <= 0:
        return "skip"
    ph = Fraction(p_hit)
    pb = Fraction(p_both)
    g = Fraction(0)
    for _ in range(days):
        vs = ph * (1 + g)
        vd = pb * (2 + g)
        vk = g
        g = max(vs, vd, vk)
    if vs == g:
        return "single"
    if vd == g:
        return "double"
    return "skip"


def check_rule_and_dispatch() -> None:
    shipped = strategy._load_mdp()
    sh_p = shipped["tail"].bin_p_hit[0]
    sh_pb = shipped["tail"].bin_p_both[0]
    assert (sh_p, sh_pb) == (0.7336111111111111, 0.5511111111111111)

    # Check shipped rates days 1..400 + verify exact float tie starts at day 116
    tie_days = []
    g = 0.0
    for d in range(1, 401):
        vs, vd, vk = sh_p * (1.0 + g), sh_pb * (2.0 + g), g
        g = max(vs, vd, vk)
        if vs == vk:
            tie_days.append(d)
        act = tp.longest_streak_action(d, sh_p, sh_pb)
        want = "double" if d <= 3 else "single"
        assert act == want, (d, act, want)
    assert tie_days[0] == 116, f"Expected float tie at 116, got {tie_days[0]}"
    assert tp.longest_streak_action(0, sh_p, sh_pb) == "skip"
    assert tp.longest_streak_action(-5, sh_p, sh_pb) == "skip"

    # Rate grid over days 1..400
    rate_pairs = [
        (sh_p, sh_pb),
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.5, 0.0),
        (0.5, 0.25),  # exact single==double tie at d=1: 0.5*1 == 0.25*2 == 0.5 -> single!
        (0.5, nextafter(0.25, 1.0)),
        (0.5, nextafter(0.25, 0.0)),
        (0.5, 0.5),   # pb == p > 0 -> double!
        (0.8, 0.8),
        (0.75, 0.50),
        (1e-300, 0.0),
        (1e-300, 1e-300),
        (nextafter(0.0, 1.0), 0.0),
        (nextafter(0.0, 1.0), nextafter(0.0, 1.0)),
        (nextafter(1.0, 0.0), nextafter(1.0, 0.0)),
    ]
    for p in (0.1, 0.25, 0.4, 0.6, 0.73, 0.85, 0.95):
        for mult in (0.0, 0.4, 0.5, 0.65, 0.75, 0.9, 1.0):
            rate_pairs.append((p, min(p, p * mult)))

    total_rule = 0
    for p, pb in rate_pairs:
        for d in range(1, 401):
            got = tp.longest_streak_action(d, p, pb)
            want = independent_rule_binary64(d, p, pb)
            assert got == want, (d, p, pb, got, want)
            assert got in {"single", "double"}, f"positive day returned {got}"
            total_rule += 1
    # Check exact single/double tie at (0.5, 0.25), d=1 resolves to single
    assert tp.longest_streak_action(1, 0.5, 0.25) == "single"
    # Check pb == p > 0 resolves to double (pb*(2+g) > p*(1+g) since p > 0)
    assert tp.longest_streak_action(1, 0.5, 0.5) == "double"
    assert tp.longest_streak_action(180, 0.5, 0.5) == "double"
    # Check p = 0, pb = 0 resolves to single (0.0 == 0.0 == 0.0 -> first exact max is single!)
    assert tp.longest_streak_action(1, 0.0, 0.0) == "single"
    assert tp.longest_streak_action(180, 0.0, 0.0) == "single"
    print(f"OK: longest_streak_action verified on {total_rule} (day, rate) pairs; tie start day={tie_days[0]}")

    # Dispatch around boundary (56, 57, 58, 100, 10**50, -1, -100, False, True) and raw days (-5, 0, 1, 2, 3, 4, 180, 181, 271)
    streaks = (55, 56, 57, 58, 100, 10**50, -1, -100, False, True)
    raw_days_list = (0, 1, 2, 3, 4, 28, 180, 181, 271)
    for s in streaks:
        for raw in raw_days_list:
            obj = tp.mdp_objective(s, raw)
            day_str = (date(2028, 9, 28) - timedelta(days=raw)).isoformat()
            reg_obj, reg_d = strategy._regime(shipped, s, day_str)
            assert obj == reg_obj
            if s >= 57 and raw > 0:
                assert obj == "longest_streak" and reg_d == min(raw, 180)
                ctx = strategy.DecisionContext(0.84, 0.83, True, day_str, True, shipped, 57, "trusted")
                dec = strategy.resolve_policy_decision(ctx, s, False)
                assert dec.objective == "longest_streak"
                assert dec.action == ("double" if raw <= 3 else "single")
                assert dec.streak == int(s)
            else:
                assert obj in {"reach57", "emax_season_best"}
                assert reg_d == min(raw, 180)
    # Also negative days_remaining directly on mdp_objective
    for s in streaks:
        assert tp.mdp_objective(s, -3) == "reach57"
    print("OK: mdp_objective, _regime, and resolve_policy_decision verified across boundary/extreme/boolean streaks")

    # Check diff for hardcoded 57 outside LONGEST_STREAK_FROM definition and comments
    diff_out = subprocess.run(
        ["git", "diff", "-U0", "0a2bec10", "84805852", "--", "src", "scripts", "tests"],
        cwd=REPO, capture_output=True, text=True, check=True, timeout=10,
    ).stdout
    added_57 = []
    cur_file = ""
    for line in diff_out.splitlines():
        if line.startswith("+++ b/"):
            cur_file = line[6:]
        elif line.startswith("+") and not line.startswith("+++"):
            code = line[1:].split("#")[0]
            if re.search(r"\b57\b", code):
                added_57.append((cur_file, line[1:].strip()))
    print("Added lines containing literal 57:", added_57)
    # Allowed: LONGEST_STREAK_FROM = 57, BOUNDARY = getattr(..., 57), and _validate_longest_base table.shape[0] < 57 (the reach-57 table size!)


def check_patched_boundary(new_val: int) -> None:
    with tempfile.TemporaryDirectory(dir=HERE, prefix=f"scratch-bound-{new_val}-") as tmp:
        s_src = Path(tmp) / "src"
        shutil.copytree(REPO / "src", s_src)
        tp_file = s_src / "bts/simulate/tail_policy.py"
        txt = tp_file.read_text()
        assert "LONGEST_STREAK_FROM = 57" in txt
        tp_file.write_text(txt.replace("LONGEST_STREAK_FROM = 57", f"LONGEST_STREAK_FROM = {new_val}", 1))

        code = f"""
import sys
from datetime import date
sys.path.insert(0, {str(s_src)!r})
sys.path.insert(1, {str(REPO)!r})
from bts import strategy
from bts.simulate import tail_policy as tp
from scripts.audit.season_walk import walk

assert tp.LONGEST_STREAK_FROM == {new_val}
shipped = strategy._load_mdp()

# Just below boundary stays legacy; at boundary switches to longest_streak!
assert tp.mdp_objective({new_val} - 1, 10) == "reach57"
assert tp.mdp_objective({new_val}, 10) == "longest_streak"
assert tp.mdp_objective({new_val}, 0) == "reach57"

assert strategy.effective_pick_bar({new_val}, "2027-07-15", False) == 0.0
bar_below = strategy.effective_pick_bar({new_val} - 1, "2027-07-15", False)
assert bar_below is None or bar_below > 0.0 or {new_val} - 1 < 56

for d_str, want in (("2027-07-15", "single"), ("2027-09-26", "double"), ("2027-09-27", "skip")):
    ctx = strategy.DecisionContext(0.84, 0.83, True, d_str, True, shipped, 56, "trusted")
    dec = strategy.resolve_policy_decision(ctx, {new_val}, False)
    if d_str == "2027-09-27":
        assert dec.objective == "reach57" and dec.action == "skip"
    else:
        assert dec.objective == "longest_streak" and dec.action == want

assert walk.DEFAULT_STATES[-1].streak == {new_val}
if {new_val} == 56:
    rec = walk.walk_one(date(2027, 9, 26), walk.DEFAULT_STATES[-1])
    assert rec["objective"] == "longest_streak" and rec["action"] == "double"
    assert walk.check([rec]) == []
"""
        env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
        proc = subprocess.run([sys.executable, "-c", code], cwd=REPO, env=env, capture_output=True, text=True, check=True, timeout=30)
        print(f"OK: Patched LONGEST_STREAK_FROM={new_val} verified")


def main() -> None:
    check_rule_and_dispatch()
    check_patched_boundary(56)
    check_patched_boundary(58)


if __name__ == "__main__":
    main()
