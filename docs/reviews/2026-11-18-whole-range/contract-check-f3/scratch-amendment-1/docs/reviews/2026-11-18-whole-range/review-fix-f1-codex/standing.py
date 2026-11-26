"""Exercise malformed files, interrupted reads, time boundaries and marker I/O."""
from datetime import date
import errno
import gzip
import json
from pathlib import Path
import sys

from bts import entry_receipt
from support import Arena, DAY, MATCH, PITCH, RIGHT, WRONG, dump, normalized, print_source, row

ARENA = Arena("standing")


def packed(value):
    return gzip.compress(json.dumps(value).encode(), mtime=0)


def run(world, at="18:30", pending=MATCH, **kwargs):
    result = world.check(at, pending, **kwargs)
    dump(at, {**normalized(result, world.root), "receipts": [
        {key: receipt[key] for key in ("et_date", "season", "detail", "exit_code", "verifier", "qualification")}
        for receipt in result["receipts"]]})
    return result


def malformed(side):
    good = packed({"units": [WRONG]})
    cases = (
        ("plain", b"plain", "BadGzipFile"), ("truncated", good[:13], "EOFError"),
        ("bad JSON", gzip.compress(b"[bad"), "JSONDecodeError"),
        ("bad UTF-8", gzip.compress(b"\xff"), "UnicodeDecodeError"),
        ("object required", packed(["not object"]), "AttributeError"),
        ("units integer", packed({"units": 17}), "TypeError"),
        ("directory", None, "IsADirectoryError"), ("permission", good, "PermissionError"),
        ("units absent", packed({}), None), ("units null", packed({"units": None}), None),
        ("units false", packed({"units": False}), None), ("units zero", packed({"units": 0}), None),
        ("units object", packed({"units": {"id": 77}}), None),
        ("units string", packed({"units": "text"}), None),
        ("bad unit rows", packed({"units": [{"id": "77"}, {}, 19, None]}), None),
        ("missing unit", packed({"units": [RIGHT_DUMMY]}), None),
    )
    for label, payload, error in cases:
        print(label)
        world = ARENA.world()
        world.commit(pitch=PITCH)
        path = world.capture([])
        if payload is None:
            path.unlink()
            path.mkdir()
        else:
            path.write_bytes(payload)
        extra = []
        if label == "permission":
            original = Path.read_bytes

            def inaccessible(item):
                if item == path:
                    raise PermissionError(errno.EACCES, "fixture capture denied")
                return original(item)

            extra = [("pathlib.Path.read_bytes", inaccessible)]
        actual = run(world, more_patches=extra)
        reason = "match_game_unchecked" if error and side == "candidate" else "match"
        assert (actual["exit"], actual["marker"]["reason"]) == (0, reason)
        assert bool(actual["receipts"]) == (error is None)
        if error:
            expected = f"check-pick-entered: entry receipt unavailable ({error})\n"
            if side == "candidate":
                expected = f"check-pick-entered: game check unavailable ({error}); batter match only\n" + expected
            assert actual["stderr"] == expected
        else:
            assert actual["stderr"] == ""
            assert actual["receipts"][0]["qualification"]["slots"][0]["state"] == "unit_unverified"
        ARENA.clear(world)


RIGHT_DUMMY = {**RIGHT, "id": 88}


def interrupts(side):
    for exc in (KeyboardInterrupt, SystemExit):
        for publish_fails in (False, True):
            world = ARENA.world()
            world.commit(pitch=PITCH)
            world.capture([WRONG])
            original = entry_receipt.load_units
            visited = []

            def interrupt(picks, season):
                visited.append(season)
                if len(visited) == 1:
                    raise exc(17)
                if publish_fails:
                    raise OSError("fixture second read failed")
                return original(picks, season)

            actual = run(world, more_patches=[("bts.entry_receipt.load_units", interrupt)])
            if side == "candidate":
                assert not actual["dm"] and actual["marker"] is None and visited == [2027, 2027]
                if not publish_fails:
                    receipt, = actual["receipts"]
                    expected = (None, "raised KeyboardInterrupt") if exc is KeyboardInterrupt else (17, None)
                    assert (receipt["exit_code"], receipt["detail"]) == expected
                else:
                    assert not actual["receipts"] and "entry receipt unavailable" in actual["stderr"]
            else:
                assert actual["marker"]["reason"] == "match" and visited == [2027]
            ARENA.clear(world)


def boundaries(side):
    for day in ("2027-07-04", "2027-12-31"):
        first = ARENA.world(day)
        tomorrow = date.fromordinal(date.fromisoformat(day).toordinal() + 1).isoformat()
        second = type(first)(first.root, tomorrow)
        for world in (first, second):
            world.commit(pitch=f"{tomorrow}T00:50:00-{'05' if day.endswith('31') else '04'}:00")
        first.capture([WRONG], f"{day}T235900Z")
        second.capture([{**RIGHT, "roundId": 2}], f"{tomorrow}T045000Z")
        rounds = {1: date.fromisoformat(day), 2: date.fromisoformat(tomorrow)}
        initial = run(first, "23:59", rounds=rounds, elapsed=120)
        next_run = run(second, "00:05", [*MATCH, row(rnd=2)], rounds=rounds)
        assert initial["receipts"][0]["et_date"] == day
        assert next_run["receipts"][0]["et_date"] == tomorrow
        assert next_run["marker"]["date"] == tomorrow
        if day.endswith("31"):
            assert initial["receipts"][0]["season"] == 2027
            assert next_run["receipts"][0]["season"] == 2028
            assert initial["marker"]["reason"] == ("wrong_game" if side == "candidate" else "match")
        ARENA.clear(first)
    first = ARENA.world("2027-12-31")
    first.commit(pitch="2028-01-01T05:50:00")
    first.capture([WRONG], "20280101T045500Z")
    utc_new_year = run(first, "23:59")
    assert utc_new_year["marker"]["reason"] == "match"
    assert utc_new_year["receipts"][0]["qualification"]["slots"][0]["state"] == "unit_unverified"
    ARENA.clear(first)


def markers(side):
    for malformed in ("permission", "invalid JSON", "UTF-8", "null", "list", "noniterable tiers",
                      "directory"):
        print(malformed)
        world = ARENA.world()
        world.commit(pitch=PITCH)
        world.capture([RIGHT])
        world.marker.parent.mkdir()
        payloads = {"invalid JSON": b"{", "UTF-8": b"\xff", "null": b"null", "list": b"[]",
                    "noniterable tiers": packed({})}
        leaves = []
        if malformed == "permission":
            world.marker.write_text("{}")
            original = Path.read_text

            def inaccessible(path, *args, **kwargs):
                if path == world.marker:
                    raise PermissionError(errno.EACCES, "fixture marker denied")
                return original(path, *args, **kwargs)

            # Deny the first marker read only, then inspect the written file.
            reads = []

            def deny_once(path, *args, **kwargs):
                if path == world.marker:
                    reads.append(path)
                    if len(reads) == 1:
                        return inaccessible(path, *args, **kwargs)
                return original(path, *args, **kwargs)

            leaves = [("pathlib.Path.read_text", deny_once)]
        elif malformed == "directory":
            world.marker.mkdir()
        elif malformed == "noniterable tiers":
            world.marker.write_text(json.dumps({"date": DAY, "status": "alerted", "escalations": 9}))
        else:
            world.marker.write_bytes(payloads[malformed])
        actual = run(world, pending=() if malformed == "directory" else MATCH, more_patches=leaves)
        if malformed in ("permission", "invalid JSON"):
            assert actual["marker"]["reason"] == "match"
        else:
            assert actual["exit"] == 1
            if malformed == "directory":
                assert len(actual["dm"]) == 1
            else:
                assert actual["fetches"] == []
        ARENA.clear(world)


if __name__ == "__main__":
    print_source()
    ARENA = Arena(f"standing-{sys.argv[1]}-{sys.argv[2]}")
    for function in sys.argv[2:]:
        globals()[function](sys.argv[1])
