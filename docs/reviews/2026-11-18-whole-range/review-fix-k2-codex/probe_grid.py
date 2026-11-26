import hashlib
import json
from dataclasses import asdict
from datetime import date, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from offline import install

install()

from bts import orchestrator, picks, scheduler, strategy
from bts.health import tail_policy
from scripts.audit.season_walk import walk


STREAKS = (*range(61), 120)
POLICY = strategy._load_mdp()
HAS_FLOOR = "min_days" in strategy.DecisionContext.__dataclass_fields__


def grid():
    edges = POLICY["boundaries"]
    probabilities = sorted({0, .8, 1, *edges, *(edge - 1e-9 for edge in edges)})
    floor = {"min_days": 1} if HAS_FLOOR else {}
    digest = hashlib.sha256()
    rows = 0
    day = date(2027, 1, 1)
    finale = date(2027, 9, 26)
    while day <= finale:
        key = day.isoformat()
        raw_days = (finale - day).days + 1
        assert strategy._days_remaining(key) == raw_days
        for streak in STREAKS:
            for saver in (False, True):
                cases = [(p, streak, "trusted") for p in probabilities]
                if streak + 2 * raw_days < 57:
                    stop = streak + 2 * raw_days
                    cases.extend([(.8, stop - 1, "trusted"), (.8, stop, "trusted"),
                                  (.8, 57, "untrusted"), (.8, None, None)])
                bars = {}
                for p, best, status in cases:
                    if (best, status) not in bars:
                        bars[best, status] = strategy.effective_pick_bar(
                            streak, key, saver, POLICY, best, status, **floor)
                    context = strategy.DecisionContext(p, p - .01, True, key, True,
                                                       POLICY, best, status, **floor)
                    decision = strategy.resolve_policy_decision(context, streak, saver)
                    if HAS_FLOOR:
                        assert strategy._regime(None, streak, key, min_days=1) == strategy._regime(
                            None, streak, key)
                    record = (key, p, saver, asdict(decision), bars[best, status])
                    digest.update(repr(record).encode() + b"\n")
                    rows += 1
        day += timedelta(days=1)
    print(json.dumps(dict(rows=rows, sha256=digest.hexdigest(), probabilities=probabilities,
                          base=POLICY["base_sha256"], tail=POLICY["tail"].sha256)))


def raw_callers():
    checks = []
    for day in ("2027-01-01", "2027-03-24", "2027-09-26", "2027-09-27",
                "2027-12-31", "2028-01-01", "2028-09-28", "2028-09-29"):
        today = date.fromisoformat(day)
        health = [asdict(alert) for alert in tail_policy.check(
            today=today, base_path=Path("data/models/mdp_policy.npz"),
            tail_path=Path("data/models/mdp_tail_policy.npz"))]
        season = [asdict(alert) for alert in tail_policy.season_end_alerts(today=today)]
        for streak in (0, 20, 55, 56, 57, 120):
            ctx = strategy.DecisionContext(.8, .79, True, day, True, POLICY, streak, "trusted")
            checks.append((day, strategy._days_remaining(day),
                           asdict(strategy.resolve_policy_decision(ctx, streak, True)),
                           strategy.decide_action(ctx, streak, True),
                           strategy.effective_pick_bar(streak, day, True), health, season))
    print(json.dumps(checks, sort_keys=True))

    day = date(2027, 9, 27)
    key = day.isoformat()
    now = walk.REAL_DATETIME(2027, 9, 27, 16, 0, tzinfo=walk.ET)
    frame = walk._predictions(day, .96)
    unavailable = set(frame.game_pk)
    with TemporaryDirectory() as directory, walk.offline(now):
        root = Path(directory) / "picks"
        root.mkdir()
        assert not walk.install_state(root, day, walk.State("dated", 57, 57, "used", .96))
        walk._set_clock(now)
        config = walk._config(root)
        statuses = {pk: {"abstract": "P", "detailed": "Scheduled"} for pk in unavailable}
        with (patch.object(orchestrator, "run_cascade", return_value=(frame, "probe")),
              patch.object(picks, "get_game_statuses_detailed", return_value=statuses),
              patch.object(scheduler, "count_new_confirmations", return_value=0),
              patch.object(scheduler, "heartbeat_watchdog"),
              patch.object(strategy, "effective_pick_bar", wraps=strategy.effective_pick_bar) as bar):
            result = scheduler.run_single_check(key, sorted(unavailable), set(), config, .03, unavailable)
        assert result["selection"].decision is None
        assert len(bar.call_args_list) == 1
        assert bar.call_args.args[:2] == (57, key)
        assert not bar.call_args.kwargs
        print(json.dumps(result.get("skip_summary"), sort_keys=True))


if __name__ == "__main__":
    grid()
    raw_callers()
