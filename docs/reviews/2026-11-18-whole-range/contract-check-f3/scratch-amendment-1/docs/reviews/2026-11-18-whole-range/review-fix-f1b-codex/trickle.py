"""Feed slow chunks into the real pending-response parser without a socket."""

from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
import json

import httpx

import bts
from bts.contest_fetch import fetch_pending_predictions
from tests.test_wr_f1_wrong_game import DAY, MATCH, WRONG, World


class Trickle(httpx.SyncByteStream):
    def __init__(self, payload, chunks, clock):
        self.body = json.dumps({"success": {"predictions": payload}}).encode()
        self.clock = clock
        self.chunks = chunks

    def __iter__(self):
        # Whitespace keeps the original JSON content while spreading it over the reads.
        for _ in range(self.chunks - 1):
            self.clock[0] += 28.5
            yield b" "
        self.clock[0] += 28.5
        yield self.body


if __name__ == "__main__":
    print(f"bts.__file__ = {bts.__file__}")
    with TemporaryDirectory(dir=Path(__file__).resolve().parent) as scratch:
        for label, payload, chunks in (("absent-before-cutoff", [], 30),
                                       ("absent-after-cutoff", [], 33),
                                       ("wrong-after-cutoff", MATCH, 33)):
            world = World(Path(scratch) / label)
            world.commit(pitch=f"{DAY}T19:35:00-04:00")
            world.capture([WRONG])
            clock = [1000.0]

            def respond(request):
                assert request.extensions["timeout"]["read"] == 30.0
                return httpx.Response(200, stream=Trickle(payload, chunks, clock))

            with httpx.Client(transport=httpx.MockTransport(respond)) as client:
                got = world.check("19:15", more_patches=[
                    ("bts.contest_fetch.fetch_pending_predictions", partial(fetch_pending_predictions, client=client)),
                    ("time.monotonic", lambda: clock[0]),
                ])
            elapsed = clock[0] - 1000
            assert elapsed > 840
            assert bool(got["dm"]) == (elapsed < 900)
            assert got["receipts"][-1]["observation"]["before_cutoff"] == (elapsed < 900)
            print(label, elapsed, got["exit"], got["marker"]["reason"], bool(got["dm"]), got["stdout"].strip())
