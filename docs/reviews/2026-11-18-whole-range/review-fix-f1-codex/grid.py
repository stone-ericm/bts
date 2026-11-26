"""Compare every observable from the unmodified base and F1."""
from datetime import date
from itertools import combinations, product
import json
import sys

from support import Arena, DAY, HERE, PITCH, normalized, print_source, read_marker_bytes, row

STATES = ("confirmed", "wrong_game", "unit_unverified", "wrong_round", "ambiguous",
          "missing", "player_unverified", "unsupported_duplicate_selection")
DELIVERIES = (None, "delivered", "private_locked", "locked_unconfirmed", "not_applicable")
NOWS = (("18:09", 0), ("18:10", 0), ("18:30", 0), ("19:04", 0), ("19:05", 0),
        ("19:19", 0), ("19:20", 0), ("19:34", 61), ("19:35", 0), ("19:36", 0))
TIERS = ("initial", "t30", "t15", "wrong_game")
EVERY_TIER_SET = [list(item) for size in range(5) for item in combinations(TIERS, size)]


def qualification(world, states):
    """Use numeric strings to attain missing/player-unverified on a real match."""
    pending, units = [], []
    for index, state in enumerate(states):
        unit, player, game = (77, 1, 101) if index == 0 else (78, 2, 201)
        if state == "unsupported_duplicate_selection":
            if index == 0:
                pending.append(row())
                units.append({"id": 77, "roundId": 1, "feedId": 102})
            continue
        pending.append(row(unit, str(player) if state in ("missing", "player_unverified") else player,
                           index + 1))
        if state == "ambiguous":
            pending.append(row(unit + 20, player, index + 1))
        if state != "missing":
            units.append({"id": unit, "roundId": 2 if state == "wrong_round" else 1,
                          "feedId": None if state == "unit_unverified" else
                          game + 1 if state == "wrong_game" else game})
    # A mistyped playerId makes the derivative state player_unverified for all
    # legs with the same numeric-string shape. Missing is separately reached
    # with genuine absent rows on the mismatch path.
    pending = [entry for entry in pending
               if not any(state == "missing" and str(entry["playerId"]) == str(index + 1)
                          for index, state in enumerate(states))]
    world.capture(units)
    return pending


def marker(world, kind):
    if kind == "none":
        return
    if kind == "unreadable":
        world.marker.parent.mkdir()
        world.marker.write_text("{not JSON")
        return
    if kind == "invalid-utf8":
        world.marker.parent.mkdir()
        world.marker.write_bytes(b"\xff")
        return
    status, reason, tiers, day = kind
    data = {"status": status, "reason": reason, "date": day}
    if tiers is not None:
        data["escalations"] = tiers
    world.set_marker(data)


def inputs(family):
    plain = (("alerted", "no_pick", None, DAY), ("dm_failed", "no_pick", None, DAY),
             ("present_unverified", "present_unverified", None, DAY),
             ("confirmed", "match", [], DAY), ("alerted", "wrong_game", ["wrong_game"], "2027-03-23"),
             "none", "unreadable", "invalid-utf8")
    if family == "cross":
        extras = (("alerted", "wrong_game", [], DAY), ("dm_failed", "wrong_game", [], DAY),
                  ("alerted", "wrong_game", ["wrong_game"], DAY))
        usable = STATES[:-1]
        selections = ([(state,) for state in usable] + list(product(usable, repeat=2)) +
                      [("unsupported_duplicate_selection",) * 2, ("wrong_game", "ambiguous")])
        for states, prior, delivery in product(selections, (*plain, *extras), DELIVERIES):
            yield (states, "from-rows", prior, delivery, "success", ("18:30", 0))
    elif family == "ladder":
        for i, (tiers, timing, dm, verdict) in enumerate(product(
                EVERY_TIER_SET, NOWS, ("success", "failure", "none"),
                ("no_pick", "mismatch", "present_unverified", "match"))):
            status = ("alerted", "dm_failed", "present_unverified")[(i + i // 4) % 3]
            reason = "wrong_game" if "wrong_game" in tiers and i % 2 == 0 else "no_pick"
            states = (("wrong_game", "confirmed") if i // 4 % 2 else ("confirmed",))
            yield (states, verdict, (status, reason, tiers, DAY), DELIVERIES[(i + i // 8) % 5], dm, timing)
    elif family == "fetch":
        for states, prior, verdict, delivery in product(
                (("confirmed",), ("wrong_game", "confirmed")), plain,
                ("fetch-profile", "fetch-pending", "fetch-rounds", "fetch-crosswalk", "identity"),
                DELIVERIES):
            yield (states, verdict, prior, delivery, "success", ("18:30", 0))


def observe(world, case):
    states, verdict, prior, delivery, dm, (at, elapsed) = case
    dd = None
    if len(states) == 2:
        dd = (1010, 101, PITCH) if states[0] == "unsupported_duplicate_selection" else (1020, 201, PITCH)
    world.commit(pitch=PITCH, dd=dd, delivery=delivery)
    if delivery == "not_applicable":
        from bts.daily_decision import write_decision
        write_decision(DAY, world.picks, action="skip", source="mdp", delivery_status=delivery,
                       scoreable=False)
    marker(world, prior)
    pending = qualification(world, states)
    options = {}
    if verdict == "no_pick":
        pending = []
    elif verdict == "mismatch":
        pending = [row(player=3)]
    elif verdict == "present_unverified":
        options["crosswalk"] = {}
    elif verdict.startswith("fetch-"):
        options["fetch_failure"] = verdict.removeprefix("fetch-")
    elif verdict == "identity":
        from types import SimpleNamespace
        options["more_patches"] = [("bts.leaderboard.auth.fetch_login_session",
                                    lambda *a, **k: SimpleNamespace(user_id="fixture", username="other"))]
    result = world.check(at, pending, dm=dm, elapsed=elapsed, **options)
    result["marker_bytes"] = read_marker_bytes(world)
    return normalized(result, world.root)


def receipt_of(observation):
    return observation["receipts"][0] if observation["receipts"] else {}


def classify(case, base, candidate):
    states, _, prior, _, dm, _ = case
    rc = receipt_of(base)
    match = (rc.get("verifier") or {}).get("reason") == "match"
    prior_wrong = (not isinstance(prior, str) and prior[3] == DAY and
                   (prior[1] == "wrong_game" or "wrong_game" in (prior[2] or ())))
    qualifications = (rc.get("qualification") or {}).get("slots", [])
    wrong = states[0] != "unsupported_duplicate_selection" and any(
        leg["state"] == "wrong_game" and type(leg["game_pk"]) is int for leg in qualifications)
    proven = states[0] != "unsupported_duplicate_selection" and bool(qualifications) and all(
        leg["state"] == "confirmed" for leg in qualifications)
    if match and (wrong or (prior_wrong and not proven)):
        if (candidate["exit"] != 1 or candidate["marker"] is None or
                candidate["marker"]["reason"] != "wrong_game"):
            return "wrong-game path missing"
        a, b = rc.copy(), receipt_of(candidate).copy()
        for key in ("exit_code", "marker_status"):
            a.pop(key, None)
            b.pop(key, None)
        return "wrong-game expected" if a == b else "wrong-game receipt drift"
    if match and prior_wrong and proven:
        return ("resolved expected" if candidate["exit"] == 0
                and candidate["marker"]["status"] == "confirmed" else "resolved path missing")
    if base != candidate:
        # The A2 initial exception is expected when the prior marker has only
        # observed a wrong game, with no delivery and no consumed initial tier.
        if not match and prior_wrong:
            if candidate["exit"] == 1 and (not isinstance(prior, str) and prior[1] == "wrong_game"):
                return "wrong-game history outside match"
        return "unexpected difference"
    return "identical"


def main():
    side, family, shard, shards = sys.argv[1:]
    shard, shards = int(shard), int(shards)
    print_source()
    arena = Arena(f"grid-{family}-{shard}")
    output = HERE / f"grid-{family}-{shard}.jsonl"
    cells = ((case, i) for i, case in enumerate(inputs(family)) if i % shards == shard)
    if side == "base":
        with output.open("w") as stream:
            for case, i in cells:
                world = arena.world()
                obs = observe(world, case)
                arena.clear(world)
                stream.write(json.dumps([i, obs]) + "\n")
        return
    from collections import Counter
    results, verifier, qualifications = Counter(), Counter(), Counter()
    shown = set()
    with output.open() as stream:
        for (case, i), line in zip(cells, stream, strict=True):
            baseline_i, base = json.loads(line)
            assert i == baseline_i
            world = arena.world()
            candidate = observe(world, case)
            arena.clear(world)
            rc = receipt_of(base)
            verifier[(rc.get("verifier") or {}).get("reason")] += 1
            qualifications.update(item["state"] for item in (rc.get("qualification") or {}).get("slots", []))
            outcome = classify(case, base, candidate)
            results[outcome] += 1
            if outcome not in ("identical", "wrong-game expected", "resolved expected") and outcome not in shown:
                shown.add(outcome)
                print(json.dumps({"outcome": outcome, "case": case, "base": base, "candidate": candidate}))
    print(json.dumps({"family": family, "shard": shard, "results": dict(results),
                      "verifier": dict(verifier), "qualification": dict(qualifications)}))


if __name__ == "__main__":
    main()
