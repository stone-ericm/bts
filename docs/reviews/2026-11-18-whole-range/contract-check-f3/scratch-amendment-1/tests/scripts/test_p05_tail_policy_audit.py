"""Rule tests for the P-05 tail-policy audit reader (design: docs/audit/2026-09-03-emax-tail-policy.md)."""
from __future__ import annotations

from scripts.audit.p05_tail_policy_audit import expected_objective, parse_policy_line, stop_rule_skip


def test_regime_is_reach57_iff_streak_plus_two_per_day_reaches_57():
    assert expected_objective(1, 28) == "reach57"            # 1 + 56 = 57: equality is reachable
    assert expected_objective(0, 28) == "emax_season_best"   # 56 < 57
    assert expected_objective(0, 25) == "emax_season_best"   # 9/03: 25 days left at streak 0


def test_stop_rule_skip_iff_best_cannot_be_beaten():
    assert stop_rule_skip(0, 9, 18) is True     # 9/19: max reachable 18 == best 18
    assert stop_rule_skip(0, 10, 18) is False   # 9/18: 20 > 18, keep playing
    assert stop_rule_skip(8, 16, 18) is False   # 9/12 at streak 8
    assert stop_rule_skip(50, 10, 57) is True   # capped at 57: nothing beats a 57 best


def test_parse_policy_line():
    line = ("2026-09-19T13:14:02-0400 bts-mlb uv[3642087]:   Policy: objective=emax_season_best action=skip "
            "streak=0 best=18 (trusted) effective_best=18 days=9 tail=dc5d0c992443")
    assert parse_policy_line(line) == {
        "date": "2026-09-19", "objective": "emax_season_best", "action": "skip", "streak": 0,
        "best": 18, "best_status": "trusted", "effective_best": 18, "days": 9, "tail": "dc5d0c992443",
    }
    assert parse_policy_line("2026-09-19T13:14:02-0400 bts-mlb uv[1]:   Pick: somebody") is None
