"""Check census parity for a missing guard and a structurally valid false guard."""
from dataclasses import asdict, replace
from datetime import datetime
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from bts import daily_decision
from bts.simulate.mdp import lookup_action
from scripts.audit.boundary_shadow_census import CensusHalt, run_census
from slots_and_journal import ARTIFACT, BOUNDS, TABLE, check_journal, selection

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]


def census_controls(root):
    day = "2026-03-25"
    ledger = root / "ledger.jsonl"
    ledger.write_text(json.dumps({"recorded_at": f"{day}T00:00:00Z", "active_streak": 50}))
    args = dict(picks_dir=root, ledger_path=ledger, policy_path=ROOT / "data/models/mdp_policy.npz",
                saver_active=False, start=day, end=day, as_of=day, output=root / "census.json")
    assert lookup_action(TABLE, BOUNDS, 50, 180, False, .96) == "skip"
    assert lookup_action(TABLE, BOUNDS, 50, 180, False, .60) == "skip"
    good = {"rule": "monotone_skip_v1", "bin": 4, "from_bin": 3, "table_action": "skip"}
    for label, guard in [("valid", good), ("in-range but raw donor skips", good | {"from_bin": 0}),
                         ("unguarded", None)]:
        daily_decision.write_decision(
            day, root, action="single", source="mdp", primary={"batter_id": 100, "p_game_hit": .96},
            streak=50, saver_available=False, delivery_status="delivered", scoreable=True,
            objective="reach57", policy_guard=guard, now=datetime.fromisoformat(day + "T22:00:00Z"))
        try:
            result = run_census(**args)
        except CensusHalt as exc:
            assert label == "unguarded" and "raw=skip recorded=single" in str(exc)
            print(label, "halts:", exc)
        else:
            parity = result["rows"][0]["parity"]
            print(label, result["rows"][0]["raw_deployed"], parity)
            assert label != "unguarded", "An unguarded play over a raw skip bypassed Gate B"
            assert parity == "guarded_play_excluded"


def saver_alias(root):
    sel = selection(root, 187, 50, -2, .96)
    ref = selection(root, 187, 50, False, .96)
    assert asdict(sel.decision) == asdict(ref.decision)
    assert lookup_action(TABLE, BOUNDS, 50, 180, -2, .96) == "skip"
    assert TABLE[50, 180, -2, 3] == TABLE[50, 180, 0, 3] == 1
    check_journal(replace(ref, saver_available=0), 50, 187, 0, .96, "single", "single")
    try:
        check_journal(sel, 50, 187, 0, .96, "single", "single")
    except AssertionError:
        from bts.scheduler import _log_policy_decision
        print("Input -2 indexes saver plane 0. Actual journal follows:", file=sys.stderr)
        _log_policy_decision(sel)
        raise


if __name__ == "__main__":
    with TemporaryDirectory() as temp:
        root = Path(temp)
        if len(sys.argv) > 1 and sys.argv[1] == "journal":
            saver_alias(root)
        else:
            census_controls(root)
