"""Check qualification outcomes for every slot pair and less common marker/cutoff transitions."""
from itertools import product
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from bts import entry_receipt
from harness import RIGHT, WRONG, World, row


def all_qualifications(root):
    w = World(root / "injected-qualifications")
    w.capture([RIGHT])
    states = ("confirmed", "unit_unverified", "wrong_round", "ambiguous", "missing", "player_unverified",
              "unsupported_duplicate_selection", "wrong_game")
    pairs = [(s,) for s in states] + list(product(states, repeat=2))
    outputs = []
    for selected, verifier in product(pairs, ("match", "no_pick", "mismatch", "present_unverified")):
        dd = (1020, 102, f"{w.day}T20:10:00-04:00") if len(selected) == 2 else None
        w.commit(dd=dd)
        w.set_marker(None)
        ok = verifier in ("match", "present_unverified")
        pending = [row(), row(78, 2, 2)] if dd else [row()]

        def injected(*args):
            return {"slots": [{"state": s} for s in selected], "all_confirmed": all(s == "confirmed" for s in selected)}

        outcome = w.check("18:30", pending, more_patches=[("bts.entry_receipt.qualify", injected),
                          ("bts.contest_fetch.pick_entry_status", lambda *a: (ok, verifier))], publish=False)
        outputs.append({"qualify": selected, "verifier": verifier, "exit": outcome["exit"],
                        "marker": outcome["marker"]["status"], "reason": outcome["marker"]["reason"],
                        "dm": outcome["dm"]})
    return outputs


def sequences(root):
    outputs = {}
    for label, steps in {
        "failed_then_pass_cutoff": [("18:30", [row()], "failure", 0), ("19:04", [row()], "success", 120)],
        "alert_then_missing_binding": [("18:30", [row()], "success", 0), ("18:45", [row(79)], "success", 0)],
        "no_entry_then_wrong_when_tiers_spent": [("18:30", [], "success", 0), ("18:50", [], "success", 0),
                                                  ("18:51", [row()], "success", 0), ("19:00", [row()], "success", 0)],
    }.items():
        w = World(root / label)
        w.commit()
        w.capture([WRONG, {**RIGHT, "id": 79, "feedId": None}])
        outputs[label] = [w.check(at, data, dm=dm, elapsed=elapsed) for at, data, dm, elapsed in steps], w.health()

    for fault in ("profile", "pending", "rounds", "crosswalk"):
        w = World(root / fault)
        w.commit()
        w.capture([WRONG])
        outputs[fault] = w.check("18:30", [row()], fetch_failure=fault)

    w = World(root / "elapsed_check")
    w.commit(pitch=f"{w.day}T19:06:00-04:00")
    w.capture([WRONG])
    clock, original = [0], entry_receipt.load_units
    def load(*a):
        clock.append(120)
        return original(*a)
    outputs["cutoff_during_units_read"] = w.check("19:00", [row()], more_patches=[
        ("bts.entry_receipt.load_units", load), ("time.monotonic", lambda: 1000 + clock[-1])])

    for label, raw in (("json_null", b"null"), ("encoded_invalid", b"\xff"),
                       ("false_string_tiers", json.dumps({"date": w.day, "status": "alerted",
                                                          "escalations": "wrong_game"}).encode()),
                       ("invalid_tiers", json.dumps({"date": w.day, "status": "alerted",
                                                     "escalations": 17}).encode())):
        w = World(root / label)
        w.commit()
        w.capture([WRONG])
        w.marker.parent.mkdir()
        w.marker.write_bytes(raw)
        try:
            outputs[label] = w.check("18:30", [row()])
        except UnicodeError:
            outputs[label] = {"marker_unreadable_in_probe": "UnicodeDecodeError"}
    return outputs


with TemporaryDirectory(dir=Path(__file__).parent) as scratch:
    root = Path(scratch)
    print(json.dumps({"all_qualifications": all_qualifications(root), "sequences": sequences(root)}, indent=2))
