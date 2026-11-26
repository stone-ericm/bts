"""Compare real CLI, capture bytes, morning and health with the archived base."""

from itertools import product
from pathlib import Path
import json
import os
import subprocess
import sys
import tempfile

import bts
from tests.test_wr_f1_wrong_game import DAY, MATCH, PITCH, RIGHT, WRONG, World, row


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
STATES = ("no-entry", "mismatch", "gap", "match", "fetch-error", "outside", "preview", "no-file")
CAPTURES = ("none", "right", "unbound", "conflicting", "wrong-round", "wrong-extra", "silent", "broken")


def simplify(result, root):
    ids = {}
    for receipt in result["receipts"]:
        ids[receipt["attempt_id"]] = f"receipt-{len(ids)}"

    def clean(value):
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        if isinstance(value, str):
            return ids.get(value, value.replace(str(root), "FIXTURE"))
        return value

    return clean(result)


def execute(destination, scratch):
    print(f"bts.__file__ = {bts.__file__}", flush=True)
    results = {}
    for state, capture, prior, delivery, dm in product(
            STATES, CAPTURES, ("none", "legacy", "current", "stale", "final"),
            ("delivered", "locked_unconfirmed"), ("success", "failure", "none")):
        key = "|".join((state, capture, prior, delivery, dm))
        with tempfile.TemporaryDirectory(dir=scratch) as tmp:
            world = World(Path(tmp))
            if state != "no-file":
                world.commit(pitch=PITCH, delivery=None if state == "preview" else delivery)
            if prior != "none":
                marker = {"date": DAY if prior != "stale" else "2027-03-23",
                          "status": "confirmed" if prior == "final" else "alerted", "reason": "no_pick"}
                if prior == "current":
                    marker["escalations"] = ["initial", "t30"]
                world.set_marker(marker)
            if capture == "right":
                world.capture([RIGHT])
            elif capture == "unbound":
                world.capture([{**RIGHT, "feedId": None}])
            elif capture == "conflicting":
                world.capture([RIGHT, WRONG])
            elif capture == "wrong-round":
                world.capture([{**WRONG, "roundId": 2}])
            elif capture == "wrong-extra":
                world.capture([RIGHT, {**WRONG, "id": 78}])
            elif capture == "silent":
                world.capture({"units-is-not-an-array": []})
            elif capture == "broken":
                world.capture([]).write_bytes(b"not gzip")
            pending = {"no-entry": [], "mismatch": [row(player=3)]}.get(state, MATCH)
            result = world.check("16:00" if state == "outside" else "19:07", pending, dm=dm,
                                 crosswalk={} if state == "gap" else None,
                                 fetch_failure="crosswalk" if state == "fetch-error" else None)
            result["morning"] = world.morning()
            result["health"] = world.health()
            result["marker_bytes"] = world.marker.read_text() if world.marker.exists() else None
            match = result["marker"]
            if match and result["marker_bytes"] and match.get("receipt"):
                result["marker_bytes"] = result["marker_bytes"].replace(match["receipt"], "RECEIPT")
            results[key] = simplify(result, world.root)
    destination.write_text(json.dumps(results, ensure_ascii=False))


if __name__ == "__main__":
    if len(sys.argv) > 1:
        execute(Path(sys.argv[1]), Path(sys.argv[2]))
    else:
        with tempfile.TemporaryDirectory(dir=HERE) as tmp:
            scratch = Path(tmp)
            archive = subprocess.check_output(["git", "archive", "1e69d0f0", "src"], cwd=ROOT, timeout=4)
            subprocess.run(["tar", "xf", "-", "-C", tmp], input=archive, check=True, timeout=4)
            results = {}
            for side, src in (("base", scratch / "src"), ("candidate", ROOT / "src")):
                path = scratch / f"{side}.json"
                subprocess.run([sys.executable, __file__, str(path), tmp], cwd=ROOT, check=True, timeout=38,
                               env={**os.environ, "PYTHONPATH": os.pathsep.join(map(str, (src, ROOT)))})
                results[side] = json.loads(path.read_text())
            allowed, failures = [], []
            for key, base in results["base"].items():
                candidate = results["candidate"][key]
                if base != candidate:
                    state, capture, prior, *_ = key.split("|")
                    if (state == "match" and capture == "broken" and prior != "final"
                            and candidate["marker"]["reason"] == "match_game_unchecked"
                            and base["marker"]["reason"] == "match"
                            and not (candidate["receipts"] or base["receipts"])
                            and candidate["morning"] == base["morning"]):
                        allowed.append(key)
                    else:
                        failures.append((key, {k: (base[k], candidate[k]) for k in base if base[k] != candidate[k]}))
            print(f"{len(results['base'])} pairs; {len(allowed)} Amendment 3 changes; {len(failures)} other differences")
            for item in failures[:8]:
                print(item)
            assert not failures
