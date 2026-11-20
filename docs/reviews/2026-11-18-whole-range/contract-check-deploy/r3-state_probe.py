"""Run real base loaders on pinned local fixtures, plus an AST compatibility counterexample."""

import ast
from datetime import date, datetime, time, timedelta
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from bts import contest_state as cs, day_outcome, season_contract as sc
from bts.entry_intent import check_entry_intent, live_gate_problem
from bts.scheduler import _pick_delivery_mode


HERE = Path(__file__).resolve().parent
assert Path(cs.__file__).resolve().is_relative_to(HERE.parents[1] / "src")


def state_bytes(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


def midpoint(root, day):
    record = root / "account_state" / "contest_streak.manual.json"
    record.parent.mkdir(parents=True)
    midnight = datetime.combine(day + timedelta(days=1), time(), sc.ET)
    record.write_text(json.dumps({"schema_version": cs.DATED_MANUAL_SCHEMA, "effective_date": day.isoformat(),
                                 "season": day.year, "expires_at": (midnight - timedelta(microseconds=1)).isoformat(),
                                 "active_streak": 1, "best_streak": 2, "saver_state": "not_earned",
                                 "recorded_at": (midnight - timedelta(hours=8)).isoformat(), "source": "probe"}))
    before = state_bytes(root)
    results = {}
    for seconds in (-1, 1):
        now = midnight + timedelta(seconds=seconds)
        switch = sc.switch_state(root, now)
        status = next(line for line in sc.inspection_lines(root, now, switch, season=now.year)
                      if line.startswith("live decision now:"))
        try:
            verdict = cs.load_decision_streak_state(root, now=now, require_contest_state=True).observation_kind
        except cs.ContestStateError:
            verdict = "refused"
        results[now.isoformat()] = {"loader": verdict, "status": status.split(";", 1)[0]}
    assert [case["loader"] for case in results.values()] == ["dated", "refused"], results
    assert before == state_bytes(root)
    return results


def config_and_monitor():
    malformed = {"scheduler": {"pick_delivery": " private ", "entry_intent": "typo"}, "health_checks": []}
    dm_research = {"scheduler": {"posting_mode": "BLUESKY_DM",
                                 "entry_intent": "research", "contest_state_required": True}}
    assert _pick_delivery_mode(malformed) == "private" and live_gate_problem(malformed) is not None
    assert _pick_delivery_mode(dm_research) == "dm" and live_gate_problem(dm_research) is None
    assert not check_entry_intent(dm_research).ok
    outcome = day_outcome.classify_day(date="2027-07-06", games=1,
                                       decision={"action": "single", "scoreable": True,
                                                 "delivery_status": "private_locked"},
                                       decision_sha256=None, receipts=[], intent="research",
                                       expected_username=None)
    assert outcome.state == "research_commit"
    sample = "from collections.abc import Buffer\n"
    tree = ast.parse(sample, feature_version=(3, 11))
    assert tree.body[0].module.split(".")[0] in sys.stdlib_module_names
    # Buffer was added in Python 3.12. The two static checks accept this import.
    return {"malformed_private_problem": live_gate_problem(malformed),
            "dm_research_mismatch_slips_past_live_gate_only": True,
            "day_outcome_prior_research_decision": outcome.state, "3.11_static_checks_accept": sample.strip()}


with TemporaryDirectory(prefix="r3-state-", dir=HERE) as tmp:
    root = Path(tmp)
    results = {"ordinary_midnight": midpoint(root / "day", date(2027, 7, 6)),
               "year_midnight": midpoint(root / "year", date(2027, 12, 31)),
               "other": config_and_monitor()}
(HERE / "state-probe.log").write_text(json.dumps(results, indent=2) + "\n")
print("Passed 4 real loader/status reads and config/monitor compatibility checks")
