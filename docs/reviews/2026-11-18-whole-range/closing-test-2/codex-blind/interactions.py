"""Exercise cross-component behavior through actual CLI and scheduler paths."""
import io
import json
import sys
from datetime import date
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from harness import Clock, PowerLoss, World, alert, dm, picks, scheduler, wall, write_json
from opening import D1, D2
from season import outcome, previous


def fallback(root, variant):
    world = World(root, D1, health=False)
    with world.switched_on():
        world.slate(D1, projected_until="19:00")
        world.dated(1, 1)
        if variant == "in_loop":
            world.at("18:12", lambda: world.dated())
            world.prediction_rule = lambda day: Clock.at < wall(day, "18:10")
        else:
            world.at("17:15", lambda: world.dated())
            world.prediction_rule = lambda day: Clock.at < wall(day, "16:10")
        world.run()
        world.set_day(D2, "08:30")
        outcome(world, D1)
    return {"case": variant, "events": world.events, "snapshot": world.snapshot()}


def auto(world, streak=20, best=20):
    year = str(date.fromisoformat(world.day).year)
    assert world.invoke("season-contract", "enable", "--season", year,
                        "--picks-dir", world.picks).exit_code == 0
    assert world.fetch(streak, best, previous(world.day)).exit_code == 0
    assert world.invoke("saver-state", "--init", "active", "--force", "--season", year,
                        "--picks-dir", world.picks).exit_code == 0


def stale(root, mode):
    from bts.model import predict
    world = World(root, D1, "00:00", health=False)
    with world.switched_on():
        world.slate(D1)
        auto(world)
        world.set_day(D1, "15:00")
        frame, _ = world.predictions([], D1)
        if mode == "dated":
            world.dated(20, 20, "active")
        if mode == "cli":
            world.set_day(D1, "17:00")
            with patch.object(predict, "run_pipeline", lambda *a, **kw: frame):
                world.invoke("run", "--date", D1, "--picks-dir", world.picks)
        else:
            world.run()
        world.set_day(D1, "17:00")
        world.run()
    return {"case": mode, "events": world.events, "snapshot": world.snapshot()}


def delivery_retry(root, real_send):
    world = World(root, D1, health=False)
    posted = []

    def urlopen(req, **kwargs):
        path = req.full_url.rsplit("/", 1)[-1].split("?")[0]
        if path.endswith("createSession"):
            body = {"accessJwt": "fixture"}
        elif path.endswith("resolveHandle"):
            body = {"did": "did:offline:user"}
        elif path.endswith("getConvoForMembers"):
            body = {"convo": {"id": "fixture"}}
        elif path.endswith("sendMessage"):
            assert kwargs["idempotent"] is False
            body = json.loads(req.data)
            posted.append(body)
            world.event("dm_endpoint_accepted", message=body["message"]["text"])
            if len(posted) == 1:
                raise TimeoutError("test: response lost after acceptance")
            body = {"id": "second-was-acknowledged"}
        else:
            raise AssertionError(path)
        return io.BytesIO(json.dumps(body).encode())

    with (
        world.switched_on(),
        patch.object(dm, "send_dm", real_send),
        patch.object(dm, "retry_urlopen", urlopen),
        patch.object(dm, "get_dm_password", lambda: "offline-only"),
    ):
        world.slate(D1)
        world.dated()
        world.run()
    assert len(posted) == 2, posted
    return {"case": "real_dm_timeout", "events": world.events, "snapshot": world.snapshot()}


def public_catchup(root, variant):
    from bts.model import predict
    world = World(root, D1, health=False)
    with world.switched_on():
        world.slate(D1)
        world.dated()
        if variant == "unconfirmed":
            world.send_failure = PowerLoss("accepted but no acknowledgement")
        try:
            world.run()
        except PowerLoss:
            world.event("power_loss")
        if variant == "delivered":
            world.set_day(D1, "21:00")
            world.run()
        for command in ("run", "orchestrate"):
            # Retain the same pre-command scheduler state for both catch-up entry points.
            path = world.picks / f"{D1}.json"
            original = path.read_bytes()
            args = ["--picks-dir", world.picks] if command == "run" else ["--config", world.config_path]
            frame, _ = world.predictions([], D1)
            with patch.object(predict, "run_pipeline", lambda *a, **kw: frame):
                world.invoke(command, "--date", D1, *args)
            path.write_bytes(original)
    return {"case": variant, "events": world.events, "snapshot": world.snapshot()}


def saver(root):
    world = World(root, D1, "12:00")
    with world.switched_on():
        world.slate(D1, ("22:10",))
        world.finish[D1, 700] = wall(D2, "00:15")
        auto(world)
        world.dated(20, 20, "used")
        world.run()
        world.invoke("season-contract", "status", "--picks-dir", world.picks)
        fixed = world.invoke("saver-state", "--use", "--picks-dir", world.picks)
        assert fixed.exit_code == 0
        world.invoke("season-contract", "status", "--picks-dir", world.picks)
        outcome(world, D1)
    return {"case": "saver_mismatch", "events": world.events, "snapshot": world.snapshot()}


def skip_review(root):
    day = "2027-07-15"
    world = World(root, day, health=False)
    with world.switched_on():
        world.slate(day, probabilities=(.65, .64, .63))
        world.dated(56, 56, "used")
        world.run()
        world.dated(55, 56, "active")
        world.set_day("2027-07-16", "08:30")
        outcome(world, day)
        (world.picks / "streak.json").write_bytes(b"\xff")
        outcome(world, day)
    return {"case": "review_prompts", "events": world.events, "snapshot": world.snapshot()}


def refusal_dm_limit(root):
    world = World(root, D1, "16:01")
    production_path = "/home/bts/projects/bts/data/picks"

    def capped(recipient, message):
        body = message.replace(str(world.picks), production_path)
        world.event("dm_length", codepoints=len(body), accepted=len(body) <= 1000,
                    body=body)
        if len(body) > 1000:
            raise ValueError("fixture chat rejects a message over 1000 characters")
        return world.send(recipient, message)

    with world.switched_on(), patch.object(alert, "send_dm", capped):
        world.slate(D1)
        world.invoke("season-contract", "revoke", "--season", 2027,
                     "--reason", "restored snapshot", "--picks-dir", world.picks)
        world.run()
        world.set_day(D2, "08:30")
        outcome(world, D1)
    return {"case": "refusal_dm_limit", "events": world.events, "snapshot": world.snapshot()}


def main():
    target = Path(sys.argv[1]).resolve()
    cases = [(fallback, "in_loop"), (fallback, "final"), (stale, "automatic"), (stale, "dated"),
             (stale, "cli"), (delivery_retry, dm.send_dm), (public_catchup, "unconfirmed"),
             (public_catchup, "delivered"), (saver,), (skip_review,), (refusal_dm_limit,)]
    rows = []
    for probe, *args in cases:
        with TemporaryDirectory(prefix="wrblind-interaction-") as temp:
            rows.append(probe(Path(temp), *args))
    target.write_text(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
