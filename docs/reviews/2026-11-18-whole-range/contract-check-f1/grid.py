"""Check every verifier/marker/transport/delivery cell and all qualification states."""
from collections import Counter
from datetime import date
from itertools import combinations, product
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory

from bts.contest_fetch import pick_entry_status
from bts.entry_receipt import load_units, qualify, target_rows
from harness import RIGHT, WRONG, World, XW, row


def classification(w, daily, pending, *, profile=None, units=None, crosswalk=XW):
    profile = profile or {"predictions": []}
    target = date.fromisoformat(w.day)
    rounds = {1: target}
    slots = [{"role": r, "batter_id": p.batter_id, "game_pk": p.game_pk}
             for r, p in (("pick", daily.pick), ("double_down", daily.double_down)) if p]
    required = {s["batter_id"] for s in slots if s["batter_id"] is not None}
    return {"verifier": pick_entry_status(profile, pending, rounds, target, required, crosswalk),
            "qualification": [s["state"] for s in qualify(slots, target_rows(profile, pending, rounds, target),
                                                           crosswalk, units)["slots"]]}


def qual_cases(root):
    results = {}

    def case(name, pending, bindings, *, game=101, dd=None, crosswalk=XW, profile=None, delivery="delivered",
             captures=None, pitch=None):
        w = World(root / name)
        second = (dd[0], dd[1], dd[2] if len(dd) > 2 else f"{w.day}T20:10:00-04:00") if dd else None
        daily = w.commit(game=game, dd=second, delivery=delivery, pitch=pitch)
        for idx, capture in enumerate(captures if captures is not None else [bindings]):
            if capture is not None:
                w.capture(capture, f"{w.day}T21{idx:02d}00Z")
        units, _ = load_units(w.picks, int(w.day[:4]))
        results[name] = {**classification(w, daily, pending, profile=profile, units=units, crosswalk=crosswalk),
                         **w.check("18:30", pending, profile=profile, crosswalk=crosswalk), "morning": w.morning()}

    case("no_pick", [], [RIGHT])
    case("mismatch", [row(player=2)], [RIGHT])
    case("unresolved", [row(player=99)], [RIGHT])
    case("single_right", [row()], [RIGHT])
    case("single_wrong", [row()], [WRONG])
    case("unit_no_capture", [row()], None)
    case("unit_unknown", [row(444)], [WRONG])
    case("unit_null", [row()], [{**WRONG, "feedId": None}])
    case("wrong_round", [row()], [{**WRONG, "roundId": 2}])
    case("wrong_round_numeric_string_masks_wrong_game", [row(rnd="1")], [WRONG])
    case("numeric_string_player", [row(player="1")], [WRONG])
    case("ambiguous_distinct_rows", [row(), row(78, number=2)], [RIGHT, {**WRONG, "id": 78}])
    case("identical_row_in_both_sources", [row()], [WRONG],
         profile={"predictions": [{"roundId": 1, "roundPredictions": [row()]}]})
    case("same_row_diff_number_in_both_sources", [row()], [WRONG],
         profile={"predictions": [{"roundId": 1, "roundPredictions": [row(number=2)]}]})
    case("no_saved_game", [row()], [RIGHT], game=None)
    case("stale_capture_after_rebind", [row()], [WRONG])
    case("both_captures_after_rebind", [row()], None, captures=[[WRONG], [RIGHT]])
    case("bound_and_null_conflicting_round", [row()], None, captures=[[WRONG], [{**RIGHT, "feedId": None, "roundId": 2}]])
    case("new_game_pk", [row()], [RIGHT], game=999)
    case("wrong_and_missing_slot_id", [row()], [WRONG], dd=(None, 102))
    second = {**WRONG, "id": 78}
    case("dh_right_distinct", [row(), row(78, 2, 2)], [RIGHT, second], dd=(1020, 102))
    case("dh_wrong_dd_later", [row(), row(77, 2, 2)], [RIGHT, second], dd=(1020, 102))
    case("dh_wrong_primary", [row(78), row(78, 2, 2)], [RIGHT, second], dd=(1020, 102))
    case("dh_swapped", [row(78), row(77, 2, 2)], [RIGHT, second], dd=(1020, 102))
    case("dh_same_batter_two_right", [row(), row(78, number=2)], [RIGHT, second], dd=(1010, 102))
    case("dh_same_batter_two_wrong", [row(78), row(78, number=2)], [RIGHT, second], dd=(1010, 102))
    case("dh_same_batter_only_primary", [row()], [RIGHT, second], dd=(1010, 102))
    case("dh_same_batter_only_dd", [row(78)], [RIGHT, second], dd=(1010, 102))
    case("duplicate_selection", [row()], [WRONG], dd=(1010, 101))
    case("unconfirmed_dd", [row(), row(77, 2, 2)], [RIGHT, second], dd=(1020, 102), delivery="locked_unconfirmed")
    case("bad_game_time", [row()], [WRONG], pitch="invalid")
    case("bad_dd_time", [row(), row(77, 2, 2)], [RIGHT, second], dd=(1020, 102, "invalid"))
    case("actual_entry_game_already_started", [row()], [{**WRONG, "startDate": f"{World(root / 'dates').day}T17:00:00-04:00"}])
    return results


def marker_options(day):
    def mark(status, tiers):
        return {"date": day, "status": status, "reason": "no_pick", "escalations": tiers}

    yield "none", None, None
    yield "unreadable_json", None, "{"
    yield "directory", None, None
    yield "json_list", None, "[]"
    yield "another_date", {**mark("confirmed", []), "date": "2027-01-01"}, None
    yield "confirmed", mark("confirmed", []), None
    yield "dm_failed", mark("dm_failed", []), None
    yield "present_unverified", mark("present_unverified", []), None
    yield "present_unverified_prior_initial", mark("present_unverified", ["initial"]), None
    yield "present_unverified_prior_wrong_game", mark("present_unverified", ["initial", "wrong_game", "t30"]), None
    yield "alerted_legacy", {"date": day, "status": "alerted"}, None
    tiers = ("initial", "wrong_game", "t30", "t15")
    for count in range(len(tiers) + 1):
        for subset in combinations(tiers, count):
            yield "alerted:" + "+".join(subset), mark("alerted", subset), None


def cells(root, group):
    w = World(root)
    w.capture([WRONG if group == "match_wrong" else RIGHT])
    pending = {"match_right": [row()], "match_wrong": [row()], "no_pick": [],
               "mismatch": [row(player=2)], "present_unverified": [row(player=99)]}[group]
    timings = {"early": ("18:20", 0), "t30": ("18:35", 0), "t15": ("18:50", 0), "late": ("19:04", 120)}
    for delivery in (None, "delivered", "private_locked", "locked_unconfirmed"):
        w.commit(delivery=delivery)
        for (name, marker, raw), dm, (band, (at, elapsed)) in product(
                marker_options(w.day), ("success", "failure", "none"), timings.items()):
            if name == "directory":
                w.set_marker(None)
                w.marker.parent.mkdir(exist_ok=True)
                w.marker.mkdir()
            else:
                w.set_marker(marker, raw)
            res = w.check(at, pending, dm=dm, elapsed=elapsed, publish=False)
            if w.marker.is_dir():
                w.marker.rmdir()
            receipt = res.pop("receipts")[0]
            res.update(category=group, delivery=delivery, prior=name, transport=dm, band=band,
                       receipt_outcome=receipt["outcome"], receipt_marker=receipt["marker_status"])
            print(json.dumps(res, sort_keys=True))


with TemporaryDirectory(dir=Path(__file__).parent) as scratch:
    root = Path(scratch)
    if sys.argv[1] == "qualify":
        print(json.dumps(qual_cases(root), indent=2))
    else:
        cells(root, sys.argv[1])
