"""Adversarial probe for A1 (2027 season boundary, 2026 parity, shipped-policy replay)."""

from __future__ import annotations

from datetime import date, timedelta
from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
BASE_SRC = Path(__file__).resolve().parent / "base" / "src"


def _load_base_strategy():
    name = "_wr_a_base_strategy_mod"
    spec = spec_from_file_location(name, BASE_SRC / "bts" / "strategy.py")
    assert spec is not None and spec.loader is not None
    mod = module_from_spec(spec)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    finally:
        sys.modules.pop(name, None)
    return mod


def run_a1_checks() -> dict:
    from bts import strategy as head_strategy
    from bts.health import tail_policy as head_tail_health
    from bts.simulate import mdp as mdp_mod
    from bts.simulate import tail_policy as tail_sim

    base_strategy = _load_base_strategy()

    old_cache = head_strategy._mdp_cache
    try:
        head_strategy._mdp_cache = None
        mdp = head_strategy._load_mdp()
        assert mdp is not None
        assert mdp["policy_table"] is not None, f"base_error={mdp['base_error']}"
        assert mdp["tail"] is not None, f"tail_error={mdp['tail_error']}"
    finally:
        head_strategy._mdp_cache = old_cache

    boundaries = [float(x) for x in mdp["boundaries"]]
    tail_boundaries = [float(x) for x in mdp["tail"].boundaries]
    rep_probs = [round(boundaries[0] - 0.05, 4)] + [
        round(b + 0.001, 4) for b in boundaries
    ]

    # 1. All dates of 2026, 2027, 2028
    for year, n_days in ((2026, 365), (2027, 365), (2028, 366)):
        for offset in range(n_days):
            d = date(year, 1, 1) + timedelta(days=offset)
            iso = d.isoformat()
            head_d = head_strategy._days_remaining(iso)
            health_d = head_tail_health._days_remaining(d)
            assert head_d == health_d, (iso, head_d, health_d)
            if year == 2026:
                deployed = max(0, (date(2026, 9, 28) - d).days)
                base_d = base_strategy._days_remaining(iso)
                assert head_d == deployed == base_d, (iso, head_d, deployed, base_d)
            elif year == 2027:
                expected_2027 = max(0, (date(2027, 9, 27) - d).days)
                base_d = base_strategy._days_remaining(iso)
                assert head_d == expected_2027, (iso, head_d, expected_2027)
                if d <= date(2027, 9, 26):
                    assert head_d == base_d + 1, (iso, head_d, base_d)
                else:
                    assert head_d == base_d == 0, (iso, head_d, base_d)
            elif year == 2028:
                expected_2028 = max(0, (date(2028, 9, 28) - d).days)
                assert head_d == expected_2028, (iso, head_d, expected_2028)

    assert head_strategy._days_remaining("2027-03-24") == 187
    assert head_strategy._days_remaining("2027-09-26") == 1
    assert head_strategy._days_remaining("2027-09-27") == 0
    assert head_strategy._days_remaining("2027-10-15") == 0

    # 2. Every 2026 policy lookup unchanged across all 365 dates, all streaks 0..57, both savers, all bins
    checked_2026 = 0
    for offset in range(365):
        iso = (date(2026, 1, 1) + timedelta(days=offset)).isoformat()
        for streak in (0, 9, 10, 15, 16, 30, 43, 54, 55, 56, 57):
            for saver in (False, True):
                for p in rep_probs:
                    for best, bstatus in ((None, None), (streak, "trusted"), (56, "trusted")):
                        h_ctx = head_strategy.DecisionContext(
                            p, p, True, iso, True, mdp, best_streak=best, best_status=bstatus
                        )
                        b_ctx = base_strategy.DecisionContext(
                            p, p, True, iso, True, mdp, best_streak=best, best_status=bstatus
                        )
                        h_dec = head_strategy.resolve_policy_decision(h_ctx, streak, saver)
                        b_dec = base_strategy.resolve_policy_decision(b_ctx, streak, saver)
                        assert (h_dec.action, h_dec.objective, h_dec.days_effective) == (
                            b_dec.action, b_dec.objective, b_dec.days_effective
                        ), (iso, streak, saver, p, best, h_dec, b_dec)
                        checked_2026 += 1

    # 3. Unconfigured year fallback + WARN + CRITICAL window edges in 2027
    good_2028 = head_tail_health.check(
        base_path=ROOT / mdp_mod.DEFAULT_POLICY_PATH,
        tail_path=ROOT / tail_sim.DEFAULT_TAIL_POLICY_PATH,
        today=date(2028, 6, 1),
    )
    assert len(good_2028) == 1 and good_2028[0].level == "WARN"
    assert "2028" in good_2028[0].message and "2028-09-28" in good_2028[0].message

    missing_levels_2027 = {}
    for d_iso in ("2027-08-29", "2027-08-30", "2027-09-25", "2027-09-26", "2027-09-27"):
        alerts = head_tail_health.check(
            base_path=ROOT / "data/models/__nonexistent_base__.npz",
            tail_path=ROOT / "data/models/__nonexistent_tail__.npz",
            today=date.fromisoformat(d_iso),
        )
        missing_levels_2027[d_iso] = sorted({a.level for a in alerts})
    assert missing_levels_2027 == {
        "2027-08-29": ["WARN"],
        "2027-08-30": ["CRITICAL"],
        "2027-09-25": ["CRITICAL"],
        "2027-09-26": ["CRITICAL"],
        "2027-09-27": ["WARN"],
    }

    # 4. Replay last 10 days of 2027 (09-17 .. 09-26, plus 09-27) at streaks 40..56
    last_days = [
        (date(2027, 9, 17) + timedelta(days=i)).isoformat() for i in range(11)
    ]

    strong_p = 0.96  # highest quality bin
    strong_action_changes_best_eq_streak = []
    strong_action_changes_by_best = []
    non_strong_bin_action_changes = []

    for day_iso in last_days:
        base_d = base_strategy._days_remaining(day_iso)
        head_d = head_strategy._days_remaining(day_iso)
        for streak in range(40, 57):
            for saver in (False, True):
                # (a) Strong candidate, best = streak (active run IS season best, or untrusted/missing)
                h_ctx = head_strategy.DecisionContext(
                    strong_p, strong_p, True, day_iso, True, mdp,
                    best_streak=streak, best_status="trusted",
                )
                b_ctx = base_strategy.DecisionContext(
                    strong_p, strong_p, True, day_iso, True, mdp,
                    best_streak=streak, best_status="trusted",
                )
                h_dec = head_strategy.resolve_policy_decision(h_ctx, streak, saver)
                b_dec = base_strategy.resolve_policy_decision(b_ctx, streak, saver)
                if h_dec.action != b_dec.action:
                    strong_action_changes_best_eq_streak.append({
                        "date": day_iso,
                        "base_days": base_d,
                        "head_days": head_d,
                        "streak": streak,
                        "saver": saver,
                        "base": f"{b_dec.action} ({b_dec.objective})",
                        "head": f"{h_dec.action} ({h_dec.objective})",
                    })

                # (b) Strong candidate, ALL trusted best_streak in streak+1 .. 56
                for best_val in range(streak + 1, 57):
                    hc = head_strategy.DecisionContext(
                        strong_p, strong_p, True, day_iso, True, mdp,
                        best_streak=best_val, best_status="trusted",
                    )
                    bc = base_strategy.DecisionContext(
                        strong_p, strong_p, True, day_iso, True, mdp,
                        best_streak=best_val, best_status="trusted",
                    )
                    hd = head_strategy.resolve_policy_decision(hc, streak, saver)
                    bd = base_strategy.resolve_policy_decision(bc, streak, saver)
                    if hd.action != bd.action:
                        strong_action_changes_by_best.append({
                            "date": day_iso,
                            "base_days": base_d,
                            "head_days": head_d,
                            "streak": streak,
                            "best": best_val,
                            "saver": saver,
                            "base": f"{bd.action} ({bd.objective})",
                            "head": f"{hd.action} ({hd.objective})",
                        })

                # (c) Check every quality bin (with best = streak) for non-top-bin action changes
                for bin_idx, p in enumerate(rep_probs[:-1]):
                    hc = head_strategy.DecisionContext(
                        p, p, True, day_iso, True, mdp,
                        best_streak=streak, best_status="trusted",
                    )
                    bc = base_strategy.DecisionContext(
                        p, p, True, day_iso, True, mdp,
                        best_streak=streak, best_status="trusted",
                    )
                    hd = head_strategy.resolve_policy_decision(hc, streak, saver)
                    bd = base_strategy.resolve_policy_decision(bc, streak, saver)
                    if hd.action != bd.action:
                        non_strong_bin_action_changes.append({
                            "date": day_iso,
                            "base_days": base_d,
                            "head_days": head_d,
                            "streak": streak,
                            "saver": saver,
                            "bin": bin_idx,
                            "p": p,
                            "base": f"{bd.action} ({bd.objective})",
                            "head": f"{hd.action} ({hd.objective})",
                        })

    return {
        "boundaries": boundaries,
        "tail_boundaries": tail_boundaries,
        "rep_probs": rep_probs,
        "season_length": mdp["season_length"],
        "checked_2026_lookups": checked_2026,
        "missing_levels_2027": missing_levels_2027,
        "strong_action_changes_best_eq_streak": strong_action_changes_best_eq_streak,
        "strong_action_changes_by_best": strong_action_changes_by_best,
        "non_strong_bin_action_changes": non_strong_bin_action_changes,
    }


def test_probe_a1_replay():
    res = run_a1_checks()
    assert res["checked_2026_lookups"] > 200_000


if __name__ == "__main__":
    summary = run_a1_checks()
    print(json.dumps(summary, indent=2))
