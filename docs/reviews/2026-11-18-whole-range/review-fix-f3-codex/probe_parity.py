"""Compare the actual morning command and message against an archived base package."""
import builtins
import io
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from bts import daily_decision, day_outcome, entry_receipt, receipt_io
from tests.test_day_outcome import _decision, _receipt

from review_support import (ARCHIVE, CURRENT, DATE, REQUESTS, SKIP, SWITCH, command, dated, identify_package,
                            put, setup, state, switch)


def run_case(root, label, decision, schedule=({},), intent="enter", receipts=(), transport="confirmed"):
    cfg, picks, account = setup(root, decision, intent=intent)
    put(account / CURRENT, dated())
    put(account / ARCHIVE, dated(streak=3))
    put(account / SWITCH, switch())
    put(account / REQUESTS, dated(streak=2))
    raw = daily_decision.load_decision_bytes(DATE, picks)[1]
    for i, receipt in enumerate(receipts):
        receipt["selection"]["commit"]["decision_sha256"] = entry_receipt._sha(raw)
        target = entry_receipt.receipts_root(picks) / DATE / f"{i}.json"
        receipt_io.publish(target, json.dumps(receipt).encode())

    attempts = []

    def watch(original):
        def observed(path=".", *args, **kwargs):
            if isinstance(path, (str, bytes, os.PathLike)):
                target = Path(os.fsdecode(path))
                if target == account or target.parent == account:
                    attempts.append(str(target.relative_to(root)))
            return original(path, *args, **kwargs)
        return observed

    def fetch(_):
        if isinstance(schedule, Exception):
            raise schedule
        return list(schedule)

    with patch.object(builtins, "open", watch(builtins.open)), patch.object(io, "open", watch(io.open)), \
         patch.object(os, "open", watch(os.open)), patch.object(os, "scandir", watch(os.scandir)):
        result = command(cfg, picks, schedule=schedule, transport=transport)
        outcome = day_outcome.gather(DATE, picks, {"scheduler": {
            "entry_intent": intent, "pick_delivery": "private" if intent == "research" else "dm",
        }}, expected_username="stonehengee", fetch_schedule=fetch)
    assert not attempts, (label, attempts)
    assert state(result) == outcome.state == label, (label, result, outcome)
    assert json.loads(result["stdout"]) == outcome.as_dict()
    print(json.dumps({"case": label, "message": outcome.message(), "evidence_reads": attempts, **result}))


def main():
    identify_package()
    batter = _receipt(confirmed=False)
    batter["qualification"]["slots"] = [{"state": "unit_unverified"}]
    games = ({"gamePk": 900},)
    void = ({"status": {"detailedState": "Postponed"}}, {"status": {"detailedState": "Cancelled"}})
    double = _decision("double", double={"batter_name": "Synthetic second batter"})
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        cases = [
            ("entered", _decision(), games, "enter", [_receipt()]),
            ("entered", double, (*games, *void), "enter", [_receipt(), _receipt(reason="no_pick", confirmed=False)]),
            ("entered_game_unverified", _decision(), games, "enter", [batter]),
            ("entered_late", _decision(), games, "enter", [_receipt(before_cutoff=False)]),
            ("not_entered", _decision(), games, "enter", [_receipt(confirmed=False, reason="no_pick")]),
            ("entry_unverifiable", _decision(), games, "enter", [_receipt(username="other")]),
            ("entry_unverifiable", double, games, "enter", [], "no-receipt"),
            ("no_decision", None, games, "enter", [], "error"),
            ("no_games", SKIP, (), "enter", []),
            ("no_games", SKIP, void, "enter", []),
            ("no_games", _decision(), void, "enter", [_receipt()]),
            ("commit_not_scoreable", _decision(scoreable=False), games, "enter", []),
            ("research_commit", _decision(delivery_status="private_locked"), games, "research", []),
            ("delivered_under_research", _decision(), games, "research", []),
            ("contradiction", _decision(), (), "enter", []),
            ("intent_unknown", SKIP, games, "invalid-intent", []),
            ("schedule_unverifiable", SKIP, OSError("offline schedule"), "enter", []),
            ("skip_not_deliberate", SKIP | {"source": "operator"}, games, "enter", []),
        ]
        for index, case in enumerate(cases):
            run_case(root / str(index), *case)


if __name__ == "__main__":
    main()
