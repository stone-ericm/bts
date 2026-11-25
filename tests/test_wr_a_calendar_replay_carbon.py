"""Adversarial review A1 regressions: 2026/2027/2028 calendar parity and late-2027 shipped-policy replay."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import pytest

from bts import strategy
from bts.health import tail_policy as tail_health
from bts.simulate.tail_policy import OBJECTIVE_REACH57, OBJECTIVE_TAIL

ROOT = Path(__file__).resolve().parents[1]
if not (ROOT / "src" / "bts").exists():
    ROOT = Path(__file__).resolve().parents[3]


def _load_shipped_mdp(monkeypatch) -> dict:
    monkeypatch.chdir(ROOT)
    monkeypatch.setattr(strategy, "_mdp_cache", None)
    mdp = strategy._load_mdp()
    assert mdp is not None and mdp["policy_table"] is not None and mdp["tail"] is not None
    return mdp


def test_calendar_parity_across_2026_2027_and_leap_year_2028():
    for year, n_days in ((2026, 365), (2027, 365), (2028, 366)):
        for offset in range(n_days):
            day = date(year, 1, 1) + timedelta(days=offset)
            iso = day.isoformat()
            days = strategy._days_remaining(iso)
            assert tail_health._days_remaining(day) == days
            if year == 2026:
                assert days == max(0, (date(2026, 9, 28) - day).days)
            elif year == 2027:
                assert days == max(0, (date(2027, 9, 27) - day).days)
            else:
                assert days == max(0, (date(2028, 9, 28) - day).days)


@pytest.mark.parametrize(
    "day,streak,p,expected_days,expected_action,expected_obj",
    [
        ("2027-09-17", 55, 0.96, 10, "single", OBJECTIVE_REACH57),
        ("2027-09-17", 55, 0.83, 10, "single", OBJECTIVE_REACH57),
        ("2027-09-18", 40, 0.96, 9, "double", OBJECTIVE_REACH57),
        ("2027-09-21", 56, 0.96, 6, "single", OBJECTIVE_REACH57),
        ("2027-09-21", 56, 0.83, 6, "single", OBJECTIVE_REACH57),
        ("2027-09-22", 48, 0.96, 5, "double", OBJECTIVE_REACH57),
        ("2027-09-23", 45, 0.96, 4, "single", OBJECTIVE_TAIL),
        ("2027-09-25", 55, 0.60, 2, "skip", OBJECTIVE_REACH57),
        ("2027-09-25", 56, 0.60, 2, "skip", OBJECTIVE_REACH57),
        ("2027-09-26", 45, 0.96, 1, "double", OBJECTIVE_TAIL),
        ("2027-09-26", 55, 0.96, 1, "double", OBJECTIVE_REACH57),
        ("2027-09-26", 56, 0.96, 1, "single", OBJECTIVE_REACH57),
        ("2027-09-27", 56, 0.96, 0, "skip", OBJECTIVE_REACH57),
    ],
)
def test_late_2027_shipped_policy_actions(
    monkeypatch, day, streak, p, expected_days, expected_action, expected_obj
):
    mdp = _load_shipped_mdp(monkeypatch)
    assert strategy._days_remaining(day) == expected_days
    ctx = strategy.DecisionContext(
        p, p, True, day, True, mdp, best_streak=streak, best_status="trusted"
    )
    dec = strategy.resolve_policy_decision(ctx, streak, False)
    assert dec.action == expected_action
    assert dec.objective == expected_obj
