"""Rebuild the closing sequence and test alert histories through the real Click commands."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from harness import RIGHT, WRONG, World, row


def cases(root):
    def scenario(name, steps):
        w = World(root / name)
        w.commit()
        w.capture([RIGHT, {**WRONG, "id": 78}, {**RIGHT, "id": 79, "feedId": None}])
        runs = []
        for at, status, dm, elapsed in steps:
            pending = {"no": [], "wrong": [row(78)], "right": [row()], "unbound": [row(79)],
                       "unresolved": [row(79, 99)]}[status]
            result = w.check(at, pending, dm=dm, elapsed=elapsed)
            runs.append({"at": at, "state": status, **result})
        return {"runs": runs, "morning": w.morning(), "health": w.health()}

    cases = {}
    histories = {
        "closing": [("18:30", "no"), ("18:45", "wrong"), ("19:00", "wrong")],
        "right_control": [("18:30", "no"), ("18:45", "right"), ("19:00", "right")],
        "unbound_control": [("18:30", "no"), ("18:45", "unbound"), ("19:00", "wrong")],
        "wrong_first": [("18:10", "wrong"), ("18:30", "wrong"), ("18:35", "wrong"),
                        ("18:49", "wrong"), ("18:50", "wrong"), ("19:05", "wrong")],
        "fixed_regressed": [("18:10", "wrong"), ("18:20", "right"), ("18:35", "wrong")],
        "removed": [("18:10", "wrong"), ("18:20", "no"), ("18:35", "no"), ("18:50", "no")],
        "unbound_after_wrong": [("18:10", "wrong"), ("18:20", "unbound"), ("18:35", "wrong")],
        "cutoff": [("19:04", "wrong", "success", 120), ("19:05", "wrong")],
        "failed_every_tier": [("18:10", "no", "failure"), ("18:11", "no"), ("18:15", "wrong", "failure"),
                              ("18:16", "wrong"), ("18:35", "wrong", "failure"), ("18:36", "wrong"),
                              ("18:50", "wrong", "failure"), ("18:51", "wrong"), ("18:52", "right", "failure"),
                              ("18:53", "right")],
        "failed_first": [("18:10", "wrong", "failure"), ("18:20", "wrong"), ("18:30", "right")],
        "only_failure": [("18:10", "wrong", "failure")],
        "present_then_failed_then_fixed": [("18:10", "no"), ("18:20", "unresolved"),
                                           ("18:25", "wrong", "failure"), ("18:30", "right")],
        "removed_after_failed_first": [("18:10", "wrong", "failure"), ("18:20", "no"), ("18:30", "wrong")],
        "wrong_without_recipient": [("18:10", "wrong", "none"), ("18:20", "wrong"), ("18:30", "right")],
        "late_right": [("19:04", "right", "success", 120)],
        "wrong_then_crosswalk_gap": [("18:10", "wrong"), ("18:20", "unresolved")],
    }
    for name, steps in histories.items():
        full = [(s[0], s[1], s[2] if len(s) >= 3 else "success", s[3] if len(s) >= 4 else 0) for s in steps]
        cases[name] = scenario(name, full)

    w = World(root / "rollback")
    w.commit()
    w.capture([WRONG])
    w.set_marker({"date": w.day, "status": "alerted", "reason": "wrong_game",
                  "escalations": ["initial", "wrong_game"], "receipt": "f1-attempt"})
    cases["rollback"] = {"runs": [w.check("18:30", [row()]), w.check("18:45", [row()])],
                         "morning": w.morning(), "health": w.health()}
    return cases


with TemporaryDirectory(dir=Path(__file__).parent) as scratch:
    print(json.dumps(cases(Path(scratch)), indent=2))
