import json
import runpy
import sys
from dataclasses import asdict
from pathlib import Path

from run_tests import offline

sys.addaudithook(offline)
import bts
from bts import strategy

ROOT = Path.cwd()
EXPECTED = Path(sys.argv[1]).resolve() / "bts" / "__init__.py"
print(bts.__file__, flush=True)
assert Path(bts.__file__).resolve() == EXPECTED

probe = runpy.run_path(str(ROOT / "docs/reviews/2026-11-18-whole-range/contract-check-k2b/probe_policy.py"))
if sys.argv[2] == "grid":
    mdp = probe["ARTIFACTS"]
    print("artifacts", mdp["base_sha256"], mdp["tail"].sha256, flush=True)
    probe["grid"]()
elif sys.argv[2] == "raw":
    cases = []
    for day in ("2027-09-26", "2027-09-27", "2027-12-31", "2028-09-28", "2028-09-29"):
        raw = strategy._days_remaining(day)
        rows = []
        for streak in (20, 55, 56, 57, 120):
            for saver in (False, True):
                ctx = strategy.DecisionContext(.8, .79, True, day, True, strategy._load_mdp(), streak, "trusted")
                decision = strategy.resolve_policy_decision(ctx, streak, saver)
                action = strategy.decide_action(ctx, streak, saver)
                bar = strategy.effective_pick_bar(streak, day, saver)
                if raw == 0:
                    assert (decision.days_effective, action, bar) == (0, ("skip", "mdp"), None)
                rows.append((streak, saver, asdict(decision), action, bar))
        health = probe["health"]
        today = probe["date"].fromisoformat(day)
        alerts = health.check(today=today, base_path=ROOT / "data/models/mdp_policy.npz",
                              tail_path=ROOT / "data/models/mdp_tail_policy.npz")
        cases.append((day, raw, rows, [asdict(alert) for alert in alerts],
                      [asdict(alert) for alert in health.season_end_alerts(today=today)]))
    print(json.dumps(cases, sort_keys=True))
else:
    raise ValueError(sys.argv[2])
