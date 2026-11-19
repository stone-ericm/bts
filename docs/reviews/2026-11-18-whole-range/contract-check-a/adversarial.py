"""Offline checks for underspecified contract edges and the key-only A3 edit."""

from copy import deepcopy
from dataclasses import replace
from datetime import date, timedelta
from io import BytesIO
import json
from pathlib import Path
import sys
from unittest.mock import patch

from bts import daily_decision, picks, scheduler, strategy
from bts.health import tail_policy

SCRATCH = Path(__file__).resolve().parent
REPO = SCRATCH.parents[1]
sys.path.insert(0, str(REPO / "docs/reviews/2026-11-18-whole-range/slice-3-codex"))
import offline

offline.ROOT = REPO
offline.HERE = SCRATCH
DAY = "2027-03-24"


def calendar_edges():
    start = date(2026, 1, 1)
    for offset in range(365):
        day = start + timedelta(days=offset)
        expected = max(0, (date(2026, 9, 28) - day).days)
        assert strategy._days_remaining(day.isoformat()) == tail_policy._days_remaining(day) == expected
    print("A1 deployed 2026 count and health parity: all 365 match")
    policy = strategy._load_mdp()
    scores = [0.6, *map(float, policy["boundaries"]), 0.96]
    for offset in range(181):
        day = (date(2027, 9, 26) - timedelta(days=offset)).isoformat()
        for streak in range(57):
            for saver in (False, True):
                for probability in scores:
                    ctx = strategy.DecisionContext(probability, probability, True, day, True,
                                                   policy, streak, "trusted")
                    before = strategy.resolve_policy_decision(ctx, streak, saver)
                    if before.action == "skip":
                        continue
                    with patch.dict(strategy.SEASON_END_DATES, {2027: "2027-09-27"}):
                        after = strategy.resolve_policy_decision(ctx, streak, saver)
                    if after.action == "skip":
                        print("A1 playable-to-skip:", day, streak, saver, probability,
                              before.action, before.objective, "->", after.action, after.objective)
                        return
    print("A1 no playable-to-skip state found in the sampled grid")


def overrun_replacement():
    with offline.offline(offline.instant(DAY, 18, 25)) as (w, mp):
        w.schedule[DAY] = w.schedule[DAY][:1]
        w.dated(0)
        w.predictions = lambda: w.frame((0.99,), (True,))
        check = scheduler.run_single_check
        lock = scheduler._lock_decision_from_predictions

        def injected_producer(*args, **kwargs):
            decisions = []

            def capture(*lock_args, **lock_kwargs):
                decision = lock(*lock_args, **lock_kwargs)
                decisions.append(decision)
                return decision

            with patch.object(scheduler, "_lock_decision_from_predictions", capture):
                result = check(*args, **kwargs)
            result["lock_decision"], = decisions
            original = result["pick_result"].daily
            replacement = deepcopy(original)
            replacement.pick = replace(original.pick, batter_name="Replacement Batter", batter_id=999)
            picks.save_pick(replacement, w.picks)
            w.advance(35 * 60)
            return result

        mp.setattr(scheduler, "run_single_check", injected_producer)
        scheduler.run_day(DAY, w.config)
        delivered, = w.sends("dm")
        commit = daily_decision.load_decision(DAY, w.picks)
        assert "Replacement Batter" in delivered
        assert commit["primary"]["batter_name"] == "Synthetic Batter 101"
        print("A3 key-only edit can send Replacement Batter but commit Synthetic Batter 101")


def missed_pick_lane():
    from bts import orchestrator

    with offline.offline(offline.instant(DAY, 18, 20)) as (w, _):
        w.schedule[DAY] = w.schedule[DAY][:1]
        w.dated(0)
        _, selection, _ = orchestrator.run_and_pick(w.config, DAY)
        path = picks.save_pick(selection.pick_result.daily, w.picks)
        w.dated(30)
        w.predictions = lambda: None
        scheduler.run_day(DAY, w.config)
        assert not w.sends("dm") and path.exists()
        assert daily_decision.load_decision(DAY, w.picks) is None
        assert any("[contest_state]" in body for body in w.sends("health-dm"))
        assert any("[missed_pick]" in body for body in w.sends("health-dm"))
        print("A4 one-game mismatch retains both contest-state and missed-pick DMs")


def schedule_edges():
    from bts.model import predict

    for sample in ({"message": "upstream error"}, {"dates": None}, {"dates": [{}]}):
        def response(*args, **kwargs):
            return BytesIO(json.dumps(sample).encode())

        with patch.object(scheduler, "retry_urlopen", response), patch.object(predict, "urlopen", response):
            for reader in (scheduler.fetch_schedule, predict._fetch_game_slots):
                try:
                    print("A5", sample, reader.__name__, reader(DAY))
                except Exception as exc:
                    print("A5", sample, reader.__name__, type(exc).__name__)


if __name__ == "__main__":
    calendar_edges()
    overrun_replacement()
    missed_pick_lane()
    schedule_edges()
